(* Symbolic evaluation of a Native4D graph, into the SAME [Stage_program.t] a
   Native graph produces. The twin of [Eval_symbolic].

   That the result type is shared and unparameterised is what makes cross-dialect
   verification cheap: [Stage_program.t] is typed on [Tensor_id.t],
   [Tensor_sig.t] and [float Expr.Value.t], none of which are dialect-specific, so the whole
   grounding and comparison machinery below [Map_verify] works on a Native4D
   program with no change at all. Design §9.3 assumes this; it holds only because
   Native4D reuses [Tensor_sig.t] verbatim (correction C3). *)

let f32 = Payload.Fmt Payload.F32

type mixed_dtype = {
  mixed_op : string;
  a_fmt : Payload.packed_fmt;
  b_fmt : Payload.packed_fmt;
}

let pp_mixed_dtype fmt
    { mixed_op; a_fmt = Payload.Fmt a_fmt; b_fmt = Payload.Fmt b_fmt } =
  Format.fprintf fmt "%s: unsupported mixed dtype, a=%s b=%s" mixed_op
    (Payload.fmt_name a_fmt) (Payload.fmt_name b_fmt)

let is_i64 = function Payload.Fmt Payload.I64 -> true | _ -> false

(* The Native4D twin of [Eval_symbolic]'s own fix, same rationale: closes the
   mixed I64/F32 checked-admission gap for Native4D's Symbolic route (Direct
   already rejects this pair, commit [b96e3bf4]); Symbolic still has no
   I64,I64 [Compute_i64] dispatch of its own, so a matching pair is
   unaffected. *)
let check_mixed_dtype (g : Graph.graph) op =
  let fmt_of r = (Tensor_id.Map.find r g.Graph.Graph.tensors).Tensor_sig.fmt in
  let check_pair mixed_op a b =
    let a_fmt = fmt_of a and b_fmt = fmt_of b in
    if Bool.equal (is_i64 a_fmt) (is_i64 b_fmt) then ()
    else
      Err.or_raise ~pp_error:pp_mixed_dtype
        (Err.fail ~pos:__POS__ { mixed_op; a_fmt; b_fmt })
  in
  match op with
  | Op.Add { Pointwise.Bin.a; b } -> check_pair "add" a b
  | Op.Sub { Pointwise.Bin.a; b } -> check_pair "sub" a b
  | Op.Mul { Pointwise.Bin.a; b } -> check_pair "mul" a b
  | _ -> ()

let first_free_tid (g : Graph.graph) =
  Tensor_id.Map.fold
    (fun k _ acc -> max acc (Tensor_id.to_int k + 1))
    g.Graph.Graph.tensors 0

(* The Native4D twin of [Eval_symbolic.i64_arange_pixel] -- see its own
   comment for the full rationale (same [Expr.Value] typed constructors, same
   [length_exact]-backed overflow-safety argument, same "no [Eval_op.Make]"
   reason). [Symbolic] here is [Native.Symbolic], reused unqualified per this
   library's own `wrapped`/[include_subdirs] convention (see this file's dune
   comment), not a Native4D-specific redefinition. *)
let i64_arange_pixel
    ({ Factory.Arange.Exact.start; step; _ } : Factory.Arange.Exact.t) =
  Expr.Value.i64_add
    (Expr.Value.i64_const start)
    (Expr.Value.i64_mul
       (Expr.Value.i64_const step)
       (Expr.Value.i64_of_index (Symbolic.of_index Symbolic.out_vec.Vec6.c)))

let run (g : Graph.graph) : Stage_program.t =
  (* [Symbolic] is stateless, so there is no instance to create. Each stage body
     is a construction computation, run below from [Expr.Builder.initial]: stage
     expressions therefore reuse reducer ordinals, which is correct because a
     reducer identity means nothing outside the expression that binds it. *)
  let module E = Eval_op4.Make (Symbolic) in
  let consts = ref [] in
  let next_const = ref (first_free_tid g) in
  let fill v shape =
    let id = Tensor_id.of_int !next_const in
    incr next_const;
    let sg = Tensor_sig.create ~id ~name:"" ~shape ~fmt:f32 () in
    consts := (sg, v) :: !consts;
    sg
  in
  let process_node (env, stages, stages_i64) (node : Graph.node) =
    let op = node.Graph.Node.op in
    check_mixed_dtype g op;
    let operand r = Tensor_id.Map.find r env in
    let shape_of r = (Tensor_id.Map.find r env).Tensor_sig.shape in
    let outs = List.mapi (fun i oid -> (i, oid)) node.Graph.Node.outputs in
    match (op, outs) with
    (* Mirrors [Eval_symbolic]'s own exact-Arange special case. *)
    | ( Op.Arange4 { Ops4.Arange4.params = { fmt; exact = Some e; _ } },
        [ (_, oid) ] )
      when is_i64 fmt ->
        let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
        let st =
          {
            Stage_program.Stage_i64.id = oid;
            sg = out_sig;
            pixel = i64_arange_pixel e;
          }
        in
        (Tensor_id.Map.add oid out_sig env, stages, st :: stages_i64)
    (* Mirrors [Eval_symbolic]'s own dtype-preserving [Reshape] arm -- same
       rationale, same [Compute_i64 (Symbolic) (Symbolic)] instantiation
       ([Symbolic] here is [Native.Symbolic], reused unqualified per this
       library's own convention, not a Native4D-specific redefinition); only
       the [Shape4.to_vec6] conversion for [params.shape] differs, matching
       [Eval_direct4]'s own [Reshape4] arm. *)
    | Op.Reshape4 { Ops4.Reshape4.params; x }, [ (_, oid) ]
      when is_i64 (operand x).Tensor_sig.fmt ->
        let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
        let x_sig = operand x in
        let module C = Reshape.Reshape.Compute_i64 (Symbolic) (Symbolic) in
        let pixel =
          Expr.Builder.run
            (C.pixel
               { Reshape.Reshape.shape = Shape4.to_vec6 params.shape }
               ~x_shape:x_sig.Tensor_sig.shape ~x:x_sig Symbolic.out_vec)
        in
        let st = { Stage_program.Stage_i64.id = oid; sg = out_sig; pixel } in
        (Tensor_id.Map.add oid out_sig env, stages, st :: stages_i64)
    (* Mirrors [Eval_symbolic]'s own dtype-preserving [Permute] arm -- same
       rationale/instantiation as this file's own [Reshape4] arm just above;
       only the [Graph_shape4.perm6] conversion differs, matching
       [Eval_direct4]'s own [Permute4] arm. *)
    | Op.Permute4 { Ops4.Permute4.perm; x }, [ (_, oid) ]
      when is_i64 (operand x).Tensor_sig.fmt ->
        let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
        let x_sig = operand x in
        let module C = Permute.Permute.Compute_i64 (Symbolic) (Symbolic) in
        let pixel =
          Expr.Builder.run
            (C.pixel (Graph_shape4.perm6 perm) ~x:x_sig Symbolic.out_vec)
        in
        let st = { Stage_program.Stage_i64.id = oid; sg = out_sig; pixel } in
        (Tensor_id.Map.add oid out_sig env, stages, st :: stages_i64)
    (* Mirrors [Eval_symbolic]'s own multi-output group construction. *)
    | _, _
      when List.length outs > 1 && Region_computation4.is_region_authored op ->
        let group =
          match
            Region_computation4.group ~limits:Kernel.Limits.default ~op
              ~operand:(fun id -> Tensor_id.Map.find_opt id env)
          with
          | Ok group -> group
          | Error error ->
              Err.raise_error ~pp_error:Region_computation.pp_error error
        in
        let env, stages =
          List.fold_left
            (fun (env, stages) (output, oid) ->
              let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
              let st =
                {
                  Stage_program.Stage.id = oid;
                  sg = out_sig;
                  computation = Region_group.Ref.Grouped (group, output);
                }
              in
              (Tensor_id.Map.add oid out_sig env, st :: stages))
            (env, stages) outs
        in
        (env, stages, stages_i64)
    | _, _ ->
        let env, stages =
          List.fold_left
            (fun (env, stages) (output, oid) ->
              let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
              let regional =
                if Region_computation4.is_region_authored op then
                  Some
                    (Region_computation4.program ~limits:Kernel.Limits.default
                       ~op ~output ~output_shape:out_sig.shape
                       ~operand:(fun id -> Tensor_id.Map.find_opt id env)
                       ~fill:(fun _role value shape -> fill value shape))
                else None
              in
              let computation =
                match regional with
                | Some (Ok program) -> Region_group.Ref.Solo program
                | Some (Error error) ->
                    Err.raise_error ~pp_error:Region_computation.pp_error error
                | None ->
                    Region_group.Ref.Solo
                      (Region_program.pixel
                         (Expr.Builder.run
                            (E.pixel op ~output ~operand ~shape_of ~fill
                               Symbolic.out_vec)))
              in
              let st =
                { Stage_program.Stage.id = oid; sg = out_sig; computation }
              in
              (Tensor_id.Map.add oid out_sig env, st :: stages))
            (env, stages) outs
        in
        (env, stages, stages_i64)
  in
  let _env, rev_stages, rev_stages_i64 =
    List.fold_left process_node
      (g.Graph.Graph.tensors, [], [])
      g.Graph.Graph.nodes
  in
  {
    Stage_program.inputs =
      List.map
        (fun id -> (id, Tensor_id.Map.find id g.Graph.Graph.tensors))
        g.Graph.Graph.inputs;
    input_kinds = g.Graph.Graph.input_kinds;
    consts = List.rev !consts;
    stages = List.rev rev_stages;
    stages_i64 = List.rev rev_stages_i64;
    outputs = g.Graph.Graph.outputs;
  }
