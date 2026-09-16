(* See eval_symbolic.mli. The env maps each edge id to its [Tensor_sig.t] (the
   [S.input] in symbolic mode); since the builder already created a signature for
   every edge, the env starts as [graph.tensors] and a node's operands resolve to
   their producers' signatures directly. A missing optional operand is filled with
   a fresh constant signature, recorded in [consts] for grounding. *)

open Graph_ir

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

(* [Eval_direct]'s own P5.4 fix (a mismatched I64/F32 pair fails at checked
   admission rather than silently computing through the default float
   path -- see its own comment) only reaches Direct-mode graphs: Symbolic's
   [process_node] dispatches every op through [E.pixel] uniformly, with no
   format check at all, so the identical mixed pair here still silently
   builds an [S.load]-based pixel that promotes the I64 operand via
   [Payload.get_float] in DOUBLE precision before a single F32 rounding --
   the exact defect class Direct's own fix closed. This closes the same gap
   for Symbolic; it does not add I64,I64 dispatch here (Symbolic still has no
   [Compute_i64] wiring for these ops — see the implementation tracker's
   P5.6 note), so a matching I64,I64 pair is unaffected and unchanged. *)
let check_mixed_dtype (gr : graph) op =
  let fmt_of r = (Tensor_id.Map.find r gr.Graph.tensors).Tensor_sig.fmt in
  let check_pair mixed_op a b =
    let a_fmt = fmt_of a and b_fmt = fmt_of b in
    if Bool.equal (is_i64 a_fmt) (is_i64 b_fmt) then ()
    else
      Err.or_raise ~pp_error:pp_mixed_dtype
        (Err.fail ~pos:__POS__ { mixed_op; a_fmt; b_fmt })
  in
  match op with
  | Add { Pointwise.Bin.a; b } -> check_pair "add" a b
  | Sub { Pointwise.Bin.a; b } -> check_pair "sub" a b
  | Mul { Pointwise.Bin.a; b } -> check_pair "mul" a b
  | _ -> ()

(* An exact int64 pixel for an I64-formatted [Factory.Arange] node:
   [start + i*step], mirroring [Factory.Arange.Compute(S).pixel]'s own float
   formula shape but built directly from [Expr.Value]'s typed int64
   constructors ([i64_const]/[i64_add]/[i64_mul]/[i64_of_index]) rather than
   through [Eval_op.Make (Symbolic)]/[Semantics.SEMANTICS] -- [Eval_op.Make]
   is parametrized over [SEMANTICS] alone, not [TYPED_SEMANTICS], so it
   cannot reach these (see the implementation tracker's D09/D10 notes). No
   [Expr.Builder] needed either: per [Expr.Value]'s own doc, "I64_const/
   I64_binary need no environment and cannot fail." Safe against overflow
   with NO checked-arithmetic node of its own: [Factory.Arange.length_exact]
   already proves [start + i*step < stop <= max_int] for every [i] below the
   count it admits (see the D02 evidence log entry), so a plain modular
   [i64_add]/[i64_mul] here computes the identical value
   [Factory.Arange.value_i64_exact]'s CHECKED version would, for every index
   this pixel is ever evaluated at. *)
let i64_arange_pixel
    ({ Factory.Arange.Exact.start; step; _ } : Factory.Arange.Exact.t) =
  Expr.Value.i64_add
    (Expr.Value.i64_const start)
    (Expr.Value.i64_mul
       (Expr.Value.i64_const step)
       (Expr.Value.i64_of_index (Symbolic.of_index Symbolic.out_vec.Vec6.c)))

let first_free_tid (g : graph) =
  Tensor_id.Map.fold
    (fun k _ acc -> max acc (Tensor_id.to_int k + 1))
    g.Graph.tensors 0

let stage_sources (g : graph) =
  ( List.map
      (fun id -> (id, Tensor_id.Map.find id g.Graph.tensors))
      g.Graph.inputs,
    g.Graph.input_kinds )

let run ?(limits = Kernel.Limits.default) (g : graph) : Stage_program.t =
  (* [Symbolic] is stateless, so there is no instance to create. Each stage body
     is a construction computation, run below from [Expr.Builder.initial]: stage
     expressions therefore REUSE reducer ordinals, which is correct because a
     reducer identity means nothing outside the expression that binds it. A
     consumer that composes two stages must freshen the inserted one. *)
  let module E = Eval_op.Make (Symbolic) in
  let consts = ref [] in
  let next_const = ref (first_free_tid g) in
  let fill v shape =
    let id = Tensor_id.of_int !next_const in
    incr next_const;
    let sg = Tensor_sig.create ~id ~name:"" ~shape ~fmt:f32 () in
    consts := (sg, v) :: !consts;
    sg
  in
  let process_node (gr : graph) (env, stages, stages_i64) (node : node) =
    let op = node.Node.op in
    check_mixed_dtype gr op;
    let operand r = Tensor_id.Map.find r env in
    let shape_of r = (Tensor_id.Map.find r env).Tensor_sig.shape in
    let outs = List.mapi (fun i oid -> (i, oid)) node.Node.outputs in
    match (op, outs) with
    (* An exact-Arange node with a real ATen-sourced int64 bound produces a
       [Stage_i64.t] instead of an ordinary float [Stage.t] -- the Symbolic
       twin of [eval_direct.ml]'s own [Some e -> value_i64_exact ...] branch.
       [exact = None] (a legacy float-scalar Arange, even when [fmt = I64])
       falls through to the unchanged generic float pixel below, exactly as
       before this session: there is no exact int64 view to build one from. *)
    | ( Arange { Factory.Arange.params = { fmt; exact = Some e; _ } },
        [ (_, oid) ] )
      when is_i64 fmt ->
        let out_sig = Tensor_id.Map.find oid gr.Graph.tensors in
        let st =
          {
            Stage_program.Stage_i64.id = oid;
            sg = out_sig;
            pixel = i64_arange_pixel e;
          }
        in
        (Tensor_id.Map.add oid out_sig env, stages, st :: stages_i64)
    (* The Symbolic twin of [Eval_direct]'s own dtype-preserving [Reshape]
       arm: the default arm below reaches [Reshape.Reshape.Compute(Symbolic)
       .pixel], whose final [Symbolic.load] builds an ordinary float [Load]
       expression, round-tripping every format through [Payload.get_float] at
       grounding time -- lossy above 2^53 for an I64 source. Branch on the
       OPERAND's declared signature format (paired with [Graph_builder.
       reshape]'s own I64-only fmt threading), routing an I64 source through
       [Compute_i64 (Symbolic) (Symbolic)] instead, which builds an
       [I64_load] expression via [Symbolic.i64_load] -- no [x_t]/materialized
       tensor needed at construction time, unlike Direct, since a symbolic
       pixel is a deferred expression, not an immediate value. This closes
       the "int64-to-int64 CONSUMER" half of the P4.1/P5.1 entry's own
       "still open" list: [Kernel.Value_i64.t]'s existing forward-reference
       machinery ([check_values_i64_order]/[materialize_values_i64], already
       built for exactly this) resolves the resulting [I64_load] of an
       earlier [values_i64] entry (e.g. an upstream exact Arange) without any
       further change. See the implementation tracker's P5.2/P4.1 note. *)
    | Reshape { Reshape.Reshape.params; x }, [ (_, oid) ]
      when is_i64 (operand x).Tensor_sig.fmt ->
        let out_sig = Tensor_id.Map.find oid gr.Graph.tensors in
        let x_sig = operand x in
        let module C = Reshape.Reshape.Compute_i64 (Symbolic) (Symbolic) in
        let pixel =
          Expr.Builder.run
            (C.pixel params ~x_shape:x_sig.Tensor_sig.shape ~x:x_sig
               Symbolic.out_vec)
        in
        let st = { Stage_program.Stage_i64.id = oid; sg = out_sig; pixel } in
        (Tensor_id.Map.add oid out_sig env, stages, st :: stages_i64)
    (* The Symbolic twin of [Eval_direct]'s own dtype-preserving [Permute]
       arm, the same shape as the [Reshape] arm just above (same rationale,
       same [Compute_i64 (Symbolic) (Symbolic)] instantiation, same "no [x_t]
       needed" reason -- see that arm's own comment). *)
    | Permute { Permute.Permute.perm; x }, [ (_, oid) ]
      when is_i64 (operand x).Tensor_sig.fmt ->
        let out_sig = Tensor_id.Map.find oid gr.Graph.tensors in
        let x_sig = operand x in
        let module C = Permute.Permute.Compute_i64 (Symbolic) (Symbolic) in
        let pixel = Expr.Builder.run (C.pixel perm ~x:x_sig Symbolic.out_vec) in
        let st = { Stage_program.Stage_i64.id = oid; sg = out_sig; pixel } in
        (Tensor_id.Map.add oid out_sig env, stages, st :: stages_i64)
    (* The Symbolic twin of [Eval_direct]'s own dtype-preserving tensor-tensor
       [Add]/[Sub]/[Mul] arms: [check_mixed_dtype] above already raises on a
       mismatched I64/F32 pair for these three ops, so by the time a node
       reaches this match, [is_i64] on ONE operand already implies the other
       agrees -- checking just [a]'s format (not both, unlike Direct's own
       arm, which has no preceding [check_mixed_dtype] of its own) is
       therefore sufficient, not merely convenient. Same [Compute_i64
       (Symbolic) (Symbolic)] shape as Reshape/Permute above: already
       carrier-generic, no change to [pointwise_binary.ml]. *)
    | Add { Pointwise.Bin.a; b }, [ (_, oid) ]
      when is_i64 (operand a).Tensor_sig.fmt ->
        let out_sig = Tensor_id.Map.find oid gr.Graph.tensors in
        let a_sig = operand a and b_sig = operand b in
        let module C = Pointwise.Add.Compute_i64 (Symbolic) (Symbolic) in
        let pixel =
          Expr.Builder.run
            (C.pixel ~a_shape:a_sig.Tensor_sig.shape
               ~b_shape:b_sig.Tensor_sig.shape a_sig b_sig Symbolic.out_vec)
        in
        let st = { Stage_program.Stage_i64.id = oid; sg = out_sig; pixel } in
        (Tensor_id.Map.add oid out_sig env, stages, st :: stages_i64)
    | Sub { Pointwise.Bin.a; b }, [ (_, oid) ]
      when is_i64 (operand a).Tensor_sig.fmt ->
        let out_sig = Tensor_id.Map.find oid gr.Graph.tensors in
        let a_sig = operand a and b_sig = operand b in
        let module C = Pointwise.Sub.Compute_i64 (Symbolic) (Symbolic) in
        let pixel =
          Expr.Builder.run
            (C.pixel ~a_shape:a_sig.Tensor_sig.shape
               ~b_shape:b_sig.Tensor_sig.shape a_sig b_sig Symbolic.out_vec)
        in
        let st = { Stage_program.Stage_i64.id = oid; sg = out_sig; pixel } in
        (Tensor_id.Map.add oid out_sig env, stages, st :: stages_i64)
    | Mul { Pointwise.Bin.a; b }, [ (_, oid) ]
      when is_i64 (operand a).Tensor_sig.fmt ->
        let out_sig = Tensor_id.Map.find oid gr.Graph.tensors in
        let a_sig = operand a and b_sig = operand b in
        let module C = Pointwise.Mul.Compute_i64 (Symbolic) (Symbolic) in
        let pixel =
          Expr.Builder.run
            (C.pixel ~a_shape:a_sig.Tensor_sig.shape
               ~b_shape:b_sig.Tensor_sig.shape a_sig b_sig Symbolic.out_vec)
        in
        let st = { Stage_program.Stage_i64.id = oid; sg = out_sig; pixel } in
        (Tensor_id.Map.add oid out_sig env, stages, st :: stages_i64)
    (* The Symbolic twin of [Eval_direct]'s own explicit int64-input
       promotion for [Mul_scalar]: unlike Reshape/Permute/Add/Sub/Mul, the
       OUTPUT here stays the ordinary float carrier (ATen promotes an
       integer tensor times a float scalar to a float result) -- so this
       produces an ordinary [Stage.t], pushed onto [stages], not a
       [Stage_i64.t]. Only the READ changes: [Pointwise.Mul_scalar.
       Compute_i64 (Symbolic) (Symbolic)] uses [Symbolic.i64_load]/
       [Symbolic.i64_to_float] (an explicit checked cast) rather than the
       default arm's [Symbolic.load], which round-trips through
       [Payload.get_float] and would perform the identical promotion only
       incidentally. [Symbolic.i64_to_float : int64 repr -> Symbolic.t]
       already matches [Compute_i64]'s own [T.i64_to_float] requirement, so
       no change to [pointwise_binary.ml] was needed here either. *)
    | Mul_scalar { Pointwise.Scalar_bin.x; scalar }, [ (_, oid) ]
      when is_i64 (operand x).Tensor_sig.fmt ->
        let out_sig = Tensor_id.Map.find oid gr.Graph.tensors in
        let x_sig = operand x in
        let module C = Pointwise.Mul_scalar.Compute_i64 (Symbolic) (Symbolic) in
        let pixel = Expr.Builder.run (C.pixel ~scalar x_sig Symbolic.out_vec) in
        let st =
          {
            Stage_program.Stage.id = oid;
            sg = out_sig;
            computation = Region_group.Ref.Solo (Region_program.pixel pixel);
          }
        in
        (Tensor_id.Map.add oid out_sig env, st :: stages, stages_i64)
    (* A multi-output Region-authored node (project step 19: today only
       Lstm) builds ONE shared group and hands every sibling stage a
       [Grouped] reference into it, rather than each independently building
       its own projected program -- see [Region_computation.group]. Every
       other case (single-output, or not Region-authored at all) keeps the
       existing one-stage-per-output-edge path unchanged, just wrapping its
       [Region_program.t] as [Solo]. *)
    | _, _ when List.length outs > 1 && Region_computation.is_region_authored op
      ->
        let group =
          match
            Region_computation.group ~limits ~op ~operand:(fun id ->
                Tensor_id.Map.find_opt id env)
          with
          | Ok group -> group
          | Error error ->
              Err.raise_error ~pp_error:Region_computation.pp_error error
        in
        let env, stages =
          List.fold_left
            (fun (env, stages) (output, oid) ->
              let out_sig = Tensor_id.Map.find oid gr.Graph.tensors in
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
              let out_sig = Tensor_id.Map.find oid gr.Graph.tensors in
              let regional =
                if Region_computation.is_region_authored op then
                  Some
                    (Region_computation.program ~limits ~op ~output
                       ~output_shape:out_sig.shape
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
    List.fold_left
      (fun acc node -> process_node g acc node)
      (g.Graph.tensors, [], []) g.Graph.nodes
  in
  let inputs, input_kinds = stage_sources g in
  {
    Stage_program.inputs;
    input_kinds;
    consts = List.rev !consts;
    stages = List.rev rev_stages;
    stages_i64 = List.rev rev_stages_i64;
    outputs = g.Graph.outputs;
  }
