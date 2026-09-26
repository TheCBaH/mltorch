(* Builds a real [Node_executor.t] from [Loop_node_program] + [Loop_js_exec]
   (design §4.4, plan S4): a compile table keyed by output [Tensor_id.t],
   lazily filled on first evaluation (or eagerly by [precompile]), routing
   only the nodes the milestone predicate [m1_scope] currently covers.
   [Refused], a [Loop_js_exec.run] error, or an output count other than 1 all
   fall back to [direct ()] -- this executor's job is "run through generated
   JS when possible", the same convention [Loop_region_executor] already
   establishes for the Region-authored arm. *)

open Graph_ir

type entry = Compiled of Loop_js_exec.compiled | Refused of string

module Coverage = struct
  type t = {
    generated_js : (string, int) Hashtbl.t;
    fallback : (string, int) Hashtbl.t;
    pending : (string, int) Hashtbl.t;
  }

  let create () =
    {
      generated_js = Hashtbl.create 16;
      fallback = Hashtbl.create 16;
      pending = Hashtbl.create 16;
    }

  let bump tbl key =
    Hashtbl.replace tbl key
      (1 + Option.value ~default:0 (Hashtbl.find_opt tbl key))

  let total tbl = Hashtbl.fold (fun _ n acc -> acc + n) tbl 0

  let add_counts dst src =
    Hashtbl.iter
      (fun k n ->
        Hashtbl.replace dst k
          (n + Option.value ~default:0 (Hashtbl.find_opt dst k)))
      src

  (* Aggregates a per-graph executor's coverage into a running total. Needed
     because the compile table is keyed by bare [Tensor_id.t] (design §4.2's
     "per executor instance", not globally): an id is unique only within the
     graph an executor was built for, so summing many graphs' node/output
     counts requires a fresh executor (and table) per graph, with only the
     COVERAGE merged afterward -- never one executor shared across graphs,
     which silently reuses a compiled kernel for an unrelated node whose id
     happens to collide (found via T4.5's own walked-subject sweep: sharing
     one executor across all 51 walked ops produced real bitwise shadow
     disagreements, all traced to exactly this). *)
  let merge_into ~into t =
    add_counts into.generated_js t.generated_js;
    add_counts into.fallback t.fallback;
    add_counts into.pending t.pending

  (* Mirrors [Loop_region_executor.Coverage.check]: a whole-model run that
     silently took only the fallback/pending path is a worse defect than one
     that errors loudly. *)
  let check ~min_generated_js t =
    let generated_js = total t.generated_js in
    if generated_js >= min_generated_js then Ok ()
    else
      Error
        (Fmt.str
           "coverage: generated_js=%d fallback=%d pending=%d, expected \
            generated_js >= %d"
           generated_js (total t.fallback) (total t.pending) min_generated_js)

  (* Fails if any op kind OUTSIDE [allow] was ever [fallback] or [pending] --
     the allow-list shrinks with each milestone (design §4.4) and is empty at
     closure, so this is what "parity" means in code. *)
  let check_parity ~allow t =
    let offenders tbl =
      Hashtbl.fold
        (fun op_kind n acc ->
          if List.mem op_kind allow then acc else (op_kind, n) :: acc)
        tbl []
    in
    match offenders t.fallback @ offenders t.pending with
    | [] -> Ok ()
    | offenders ->
        Error
          (Fmt.str "check_parity: outside the allow-list: %a"
             Fmt.(list ~sep:comma (pair ~sep:(any "=") string int))
             (List.sort compare offenders))
end

let is_i64_fmt = function Payload.Fmt Payload.I64 -> true | _ -> false

let operand_fmt (g : graph) id =
  (Tensor_id.Map.find id g.Graph.tensors).Tensor_sig.fmt

(* M1 (design §3): the default float-pixel arm. Classified from the op and
   the requested output's own declared format, which already separates every
   M2 case from M1 EXCEPT two: [Mul_scalar]/[To_copy Float] keep an F32
   output even when reading an I64 operand (the promotion design §3 calls
   out by name), and the factories/[Unbind]/[Split_with_sizes] are excluded
   regardless of the format they happen to declare. Every other M2 arm
   (I64 Reshape/Permute/Add/Sub/Mul, To_copy Long, the Bool-storage arms, an
   index output) is already non-F32-declared, so the format check alone
   correctly excludes it -- including correctly ROUTING the *value* output
   of Max_dim/Max_pool2d_with_indices, which design §3 lists under M1, since
   only that node's INDEX output (ordinal one) is I64-declared. *)
let m1_scope (g : graph) (op : op) ~(out_fmt : Payload.packed_fmt) =
  match op with
  | Unbind _ | Split_with_sizes _ | Arange _ | Zeros _ | Eye _ -> false
  | Mul_scalar { Pointwise.Scalar_bin.x; _ } ->
      not (is_i64_fmt (operand_fmt g x))
  | To_copy { Pointwise.To_copy.target = Pointwise.To_copy.Float; x } ->
      not (is_i64_fmt (operand_fmt g x))
  | _ -> ( match out_fmt with Payload.Fmt Payload.F32 -> true | _ -> false)

type t = {
  limits : Kernel.Limits.t;
  table : (Tensor_id.t, entry) Hashtbl.t;
  coverage : Coverage.t;
  shadow : bool;
  on_fallback : string -> unit;
}

let create ?(limits = Kernel.Limits.default) ?(shadow = false)
    ?(on_fallback =
      fun reason ->
        Printf.eprintf "loop_node_executor: falling back to direct: %s\n%!"
          reason) () =
  {
    limits;
    table = Hashtbl.create 256;
    coverage = Coverage.create ();
    shadow;
    on_fallback;
  }

let oid_of (node : node) ~(output : Output_ordinal.t) =
  List.assoc output (Output_ordinal.indexed node.Node.outputs)

let entry_of t (g : graph) (node : node) ~output ~oid =
  match Hashtbl.find_opt t.table oid with
  | Some entry -> entry
  | None ->
      let entry =
        match Loop_node_program.lower ~limits:t.limits g node ~output with
        | Error e ->
            Refused (Fmt.str "%a" Loop_node_program.pp_error (Err.Error.kind e))
        | Ok program -> (
            match Loop_js_exec.compile program with
            | Error e ->
                Refused (Fmt.str "%a" Loop_js_exec.pp_error (Err.Error.kind e))
            | Ok compiled -> Compiled compiled)
      in
      Hashtbl.replace t.table oid entry;
      entry

(* Every routed node's own [oid], shape-only -- no tensor payload is needed
   (design §4.3), so this runs before any weight is loaded. In a warm
   process, later inferences pay no lowering/printing/[new Function] at all;
   [Loop_js_exec.compile]'s own memo by printed source still shares one
   function across structurally identical blocks underneath. *)
let precompile t (g : graph) =
  List.iter
    (fun (node : node) ->
      List.iter
        (fun (output, oid) ->
          let out_fmt = operand_fmt g oid in
          if m1_scope g node.Node.op ~out_fmt then
            ignore (entry_of t g node ~output ~oid))
        (Output_ordinal.indexed node.Node.outputs))
    g.Graph.nodes

exception Mismatch of Vec6.coord * float * float

let first_mismatch shape a b =
  try
    Vec6.iter shape (fun c ->
        let av = Tensor.read a c and bv = Tensor.read b c in
        if not (Float.equal av bv) then raise (Mismatch (c, av, bv)));
    None
  with Mismatch (c, av, bv) -> Some (c, av, bv)

let node_executor t : Node_executor.t =
  {
    run =
      (fun g node ~output ~out_shape ~operands ~direct ->
        let oid = oid_of node ~output in
        let op_name = Graph_ir.op_name node.Node.op in
        let out_fmt = operand_fmt g oid in
        let fallback reason =
          t.on_fallback (Fmt.str "%s: %s" op_name reason);
          Coverage.bump t.coverage.fallback op_name;
          direct ()
        in
        if not (m1_scope g node.Node.op ~out_fmt) then (
          Coverage.bump t.coverage.pending op_name;
          direct ())
        else
          match entry_of t g node ~output ~oid with
          | Refused reason -> fallback reason
          | Compiled compiled -> (
              let bind id = Tensor_id.Map.find_opt id operands in
              match Loop_js_exec.run compiled ~bind with
              | Error e ->
                  fallback
                    (Fmt.str "%a" Loop_js_exec.pp_error (Err.Error.kind e))
              | Ok result -> (
                  match Tensor_id.Map.bindings result with
                  | [ (_, generated) ] -> (
                      if not t.shadow then (
                        Coverage.bump t.coverage.generated_js op_name;
                        Ok generated)
                      else
                        match direct () with
                        | Error _ as e -> e
                        | Ok direct_tensor as ok -> (
                            match
                              first_mismatch out_shape generated direct_tensor
                            with
                            | None ->
                                Coverage.bump t.coverage.generated_js op_name;
                                ok
                            | Some (c, gv, dv) ->
                                Printf.eprintf
                                  "loop_node_executor: SHADOW DISAGREE %s \
                                   output=%d at %s: generated=%h direct=%h\n\
                                   %!"
                                  op_name
                                  (Output_ordinal.to_int output)
                                  (Fmt.str "%a" Vec6.pp_coord c)
                                  gv dv;
                                Coverage.bump t.coverage.fallback op_name;
                                ok))
                  | _ -> fallback "expected exactly one output buffer")));
  }
