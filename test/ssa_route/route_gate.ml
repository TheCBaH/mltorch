open Loop_ir
module S = Ssa_ir
module G = Graph_ir

(* The comparison behind the converter route. For one invocation, the placed
   kernel is lowered twice through [Ssa_backends.lowered]: the direct frontend
   and the structured one (Loop_lower.lower_structured then Ssa_of_loop). Under
   each configuration both programs are planned, executed over the same
   bindings and compared:

   - bitwise, direct against structured, and structured against the Loop
     oracle bundle's boundary values at the same precision;
   - the same contractions recognized and the same loops vectorized: a
     descriptor and operand multiset, and the vectorized decisions' trips, work
     and executions, never the allocated region ids.

   A refusal of the structured frontend is counted by its typed cause. *)

let alias = S.Ssa_effects.Distinct_buffers

module Config = struct
  type t = {
    name : string;
    numerics : S.Ssa_numerics.t;
    precision : Loop_numerics.Precision.t;
    target : S.Ssa_target.t option;
    contractions : bool;
  }

  let all =
    [
      {
        name = "f64";
        numerics = S.Ssa_numerics.Reference_f64;
        precision = Loop_numerics.Precision.F64;
        target = None;
        contractions = false;
      };
      {
        name = "neon";
        numerics = S.Ssa_numerics.Fp32_ordered;
        precision = Loop_numerics.Precision.F32;
        target = Some S.Ssa_target.neon128;
        contractions = false;
      };
      {
        name = "wasm";
        numerics = S.Ssa_numerics.Fp32_ordered;
        precision = Loop_numerics.Precision.F32;
        target = Some S.Ssa_target.wasm128;
        contractions = false;
      };
      {
        name = "coral";
        numerics = S.Ssa_numerics.Fp32_ordered;
        precision = Loop_numerics.Precision.F32;
        target = Some S.Ssa_target.zve32f128;
        contractions = true;
      };
    ]
end

module Verdict = struct
  type t = {
    bitwise_equal : bool;  (** structured outputs are the direct outputs *)
    oracle_equal : bool;  (** structured outputs are the oracle's *)
    same_contractions : bool;
    same_vectors : bool;
  }
end

let bname b = Printf.sprintf "b%d" (b : S.Ssa_id.Buffer.t :> int)

let rec instrs (r : S.Ssa_region.t) =
  List.concat_map
    (function
      | S.Ssa_stmt.Instr i -> [ i ]
      | S.Ssa_stmt.For { body; _ } | S.Ssa_stmt.Ordered_sum { body; _ } ->
          instrs body
      | S.Ssa_stmt.If { then_; else_; _ } -> instrs then_ @ instrs else_)
    r.S.Ssa_region.body

let contractions (p : S.Ssa_program.t) =
  instrs p.S.Ssa_program.entry
  |> List.filter_map (fun i ->
      match i.S.Ssa_instr.op with
      | S.Ssa_op.Contract { desc; a; b; _ } ->
          Some (Contraction.name desc ^ ":" ^ bname a ^ ":" ^ bname b)
      | _ -> None)
  |> List.sort String.compare

let vectors (plan : S.Ssa_plan.t) =
  List.filter_map
    (fun (d : S.Ssa_vectorize.Decision.t) ->
      match d.S.Ssa_vectorize.Decision.outcome with
      | S.Ssa_vectorize.Decision.Kept_scalar _ -> None
      | S.Ssa_vectorize.Decision.Vectorized ->
          Some
            (Fmt.str "trips=%Ld work=%Ld executions=%Ld"
               d.S.Ssa_vectorize.Decision.trips d.S.Ssa_vectorize.Decision.work
               d.S.Ssa_vectorize.Decision.executions))
    plan.S.Ssa_plan.vectorized
  |> List.sort String.compare

(* The first cell where [actual] is not [expected], as binary32 bits: every
   declared output is a binary32 buffer. *)
let mismatch expected actual =
  let found = ref None in
  Tensor_id.Map.iter
    (fun id (Tensor.Tensor a as x) ->
      match Tensor_id.Map.find_opt id actual with
      | None ->
          if !found = None then
            found := Some (Fmt.str "missing t%d" (Tensor_id.to_int id))
      | Some y ->
          Vec6.iter a.Tensor.shape (fun c ->
              if
                !found = None
                && Int32.bits_of_float (Tensor.read x c)
                   <> Int32.bits_of_float (Tensor.read y c)
              then
                found :=
                  Some
                    (Fmt.str "t%d at %d" (Tensor_id.to_int id)
                       (Vec6.offset a.Tensor.shape c :> int))))
    expected;
  if
    !found = None
    && Tensor_id.Map.cardinal expected <> Tensor_id.Map.cardinal actual
  then Some "output set differs"
  else !found

(* Equal when both ran and every output matches, or both failed with the same
   rendered row. *)
let same_run a b =
  match (a, b) with
  | Ok x, Ok y -> mismatch x y = None
  | Error x, Error y -> String.equal x y
  | Ok _, Error _ | Error _, Ok _ -> false

let execute (inv : Loop_bundle.invocation) ~bind program =
  match
    Err.payload (Ssa_lower.Ssa_exec.run inv.Loop_bundle.placed program ~bind)
  with
  | Ok outputs -> Ok outputs
  | Error e -> Error (Fmt.str "%a" Ssa_lower.Ssa_exec.pp_error e)

let plan (c : Config.t) program =
  Result.to_option
    (S.Ssa_plan.admitted
       (S.Ssa_plan.resolve ?target:c.Config.target ~alias
          ~contractions:c.Config.contractions ~numerics:c.Config.numerics
          program))

(* One invocation under one configuration. [values] are the oracle's boundary
   tensors at the configuration's precision. [None]: a plan was refused (a
   binary32 admission). *)
let verdict (inv : Loop_bundle.invocation) ~values ~direct ~structured
    (c : Config.t) =
  let binding =
    List.combine inv.Loop_bundle.program.Loop_program.buffers
      inv.Loop_bundle.edges
  in
  let bind id =
    List.find_map
      (fun ((b : Loop_buffer.t), edge) ->
        if Tensor_id.equal b.Loop_buffer.id id then
          match
            List.find_opt
              (fun (s : Loop_bundle.synthetic) ->
                Tensor_id.equal s.Loop_bundle.id edge)
              inv.Loop_bundle.synthetics
          with
          | Some s ->
              Some
                (Tensor.materialize s.Loop_bundle.shape (fun _ ->
                     s.Loop_bundle.value))
          | None -> G.Tensor_id.Map.find_opt edge values
        else None)
      binding
  in
  let local_of edge =
    List.find_map
      (fun ((b : Loop_buffer.t), e) ->
        if b.Loop_buffer.role = Loop_buffer.Output && Tensor_id.equal e edge
        then Some b.Loop_buffer.id
        else None)
      binding
    |> Option.get
  in
  let golden =
    List.fold_left
      (fun m (o : Loop_bundle.Output.t) ->
        G.Tensor_id.Map.add
          (local_of o.Loop_bundle.Output.oid)
          (G.Tensor_id.Map.find o.Loop_bundle.Output.oid values)
          m)
      Tensor_id.Map.empty inv.Loop_bundle.outputs
  in
  match (plan c direct, plan c structured) with
  | Some d, Some s ->
      let rd = execute inv ~bind d.S.Ssa_plan.program
      and rs = execute inv ~bind s.S.Ssa_plan.program in
      Some
        {
          Verdict.bitwise_equal = same_run rd rs;
          oracle_equal =
            (match rs with
            | Ok o -> mismatch golden o = None
            | Error _ -> false);
          same_contractions =
            contractions d.S.Ssa_plan.program
            = contractions s.S.Ssa_plan.program;
          same_vectors = vectors d = vectors s;
        }
  | _ -> None

(* The Loop oracle bundle's boundary values: every invocation's outputs, at one
   precision, over deterministic inputs and positive constants. *)
let oracle (b : Loop_bundle.t) ~constants ~inputs precision =
  let all_outputs =
    List.concat_map
      (fun (inv : Loop_bundle.invocation) ->
        List.map
          (fun (o : Loop_bundle.Output.t) -> o.Loop_bundle.Output.oid)
          inv.Loop_bundle.outputs)
      b.Loop_bundle.invocations
  in
  let outputs =
    Err.or_raise ~pp_error:Loop_bundle_oracle.pp_error
      (Loop_bundle_oracle.run ~precision
         { b with Loop_bundle.outputs = all_outputs }
         ~constants:(fun id -> List.assoc_opt id constants)
         ~inputs:(fun id -> List.assoc_opt id inputs))
  in
  List.fold_left
    (fun m (id, v) -> G.Tensor_id.Map.add id v m)
    G.Tensor_id.Map.empty
    (constants @ inputs @ List.combine all_outputs outputs)

let sig_of_edge (b : Loop_bundle.t) id =
  List.find_map
    (fun (inv : Loop_bundle.invocation) ->
      List.find_map
        (fun ((buf : Loop_buffer.t), edge) ->
          if Tensor_id.equal edge id then Some buf.Loop_buffer.sg else None)
        (List.combine inv.Loop_bundle.program.Loop_program.buffers
           inv.Loop_bundle.edges))
    b.Loop_bundle.invocations
  |> Option.get

let input_tensor (sg : Tensor_sig.t) =
  Tensor.materialize sg.Tensor_sig.shape (fun c ->
      float_of_int (((Vec6.offset sg.Tensor_sig.shape c :> int) mod 17) - 8)
      /. 8.)

let constant_tensor (sg : Tensor_sig.t) =
  Tensor.materialize sg.Tensor_sig.shape (fun c ->
      0.25
      +. 0.05
         *. float_of_int ((Vec6.offset sg.Tensor_sig.shape c :> int) mod 11))

module Tally = struct
  type counts = {
    compared : int;
    bitwise : int;
    oracle : int;
    contractions : int;
    vectors : int;
  }

  type t = {
    mutable invocations : int;
    mutable converted : int;
    mutable refusals : (string * int) list;
    mutable runs : (string * counts) list;
  }

  let zero =
    { compared = 0; bitwise = 0; oracle = 0; contractions = 0; vectors = 0 }

  let make () = { invocations = 0; converted = 0; refusals = []; runs = [] }

  let refuse t key =
    t.refusals <-
      (key, 1 + Option.value ~default:0 (List.assoc_opt key t.refusals))
      :: List.remove_assoc key t.refusals

  let run t name (v : Verdict.t option) =
    let c = Option.value ~default:zero (List.assoc_opt name t.runs) in
    let one x = if x then 1 else 0 in
    let next =
      match v with
      | None -> c
      | Some v ->
          {
            compared = c.compared + 1;
            bitwise = c.bitwise + one v.Verdict.bitwise_equal;
            oracle = c.oracle + one v.Verdict.oracle_equal;
            contractions = c.contractions + one v.Verdict.same_contractions;
            vectors = c.vectors + one v.Verdict.same_vectors;
          }
    in
    t.runs <- (name, next) :: List.remove_assoc name t.runs

  let pp ppf t =
    Fmt.pf ppf "invocations=%d converted=%d refused=%d@." t.invocations
      t.converted
      (t.invocations - t.converted);
    List.iter
      (fun (k, n) -> Fmt.pf ppf "  refused %d: %s@." n k)
      (List.sort compare t.refusals);
    List.iter
      (fun (c : Config.t) ->
        let r =
          Option.value ~default:zero (List.assoc_opt c.Config.name t.runs)
        in
        Fmt.pf ppf
          "  %-5s compared=%d bitwise=%d oracle=%d contractions=%d vectors=%d@."
          c.Config.name r.compared r.bitwise r.oracle r.contractions r.vectors)
      Config.all

  (* Every comparison that ran agreed. *)
  let clean t =
    List.for_all
      (fun (_, r) ->
        r.bitwise = r.compared && r.oracle = r.compared
        && r.contractions = r.compared
        && r.vectors = r.compared)
      t.runs
end

let refusal_key = function
  | Ssa_backends.Refusal.Converter u ->
      Fmt.str "converter: %a" Ssa_of_loop_unsupported.pp u
  | Ssa_backends.Refusal.Direct_lowering u ->
      Fmt.str "direct: %a" Ssa_lower.Ssa_unsupported.pp u
  | Ssa_backends.Refusal.Loop_lowering u ->
      Fmt.str "loop: %a" Loop_unsupported.pp u

(* [rewrite] perturbs the structured program before it is planned, to show the
   comparison can fail. *)
let bundle ?(rewrite = Fun.id) (g : G.graph) =
  let b = Err.or_raise ~pp_error:Loop_bundle.pp_error (Loop_bundle.build g) in
  let constants =
    List.map
      (fun id -> (id, constant_tensor (sig_of_edge b id)))
      b.Loop_bundle.constants
  and inputs =
    List.map
      (fun id -> (id, input_tensor (sig_of_edge b id)))
      b.Loop_bundle.inputs
  in
  let values =
    List.map
      (fun precision -> (precision, oracle b ~constants ~inputs precision))
      [ Loop_numerics.Precision.F64; Loop_numerics.Precision.F32 ]
  in
  let t = Tally.make () in
  List.iter
    (fun (inv : Loop_bundle.invocation) ->
      t.Tally.invocations <- t.Tally.invocations + 1;
      let direct =
        match
          Ssa_backends.lowered ~frontend:Ssa_backends.Frontend.Direct inv
        with
        | Ok p -> p
        | Error r -> Fmt.failwith "direct lowering refused: %s" (refusal_key r)
      in
      match
        Ssa_backends.lowered ~frontend:Ssa_backends.Frontend.Structured inv
      with
      | Error r -> Tally.refuse t (refusal_key r)
      | Ok structured ->
          t.Tally.converted <- t.Tally.converted + 1;
          let structured = rewrite structured in
          List.iter
            (fun (c : Config.t) ->
              Tally.run t c.Config.name
                (verdict inv
                   ~values:(List.assoc c.Config.precision values)
                   ~direct ~structured c))
            Config.all)
    b.Loop_bundle.invocations;
  t
