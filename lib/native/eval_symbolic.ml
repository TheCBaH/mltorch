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
  let process_node (gr : graph) env stages (node : node) =
    let op = node.Node.op in
    check_mixed_dtype gr op;
    let operand r = Tensor_id.Map.find r env in
    let shape_of r = (Tensor_id.Map.find r env).Tensor_sig.shape in
    let outs = List.mapi (fun i oid -> (i, oid)) node.Node.outputs in
    (* A multi-output Region-authored node (project step 19: today only
       Lstm) builds ONE shared group and hands every sibling stage a
       [Grouped] reference into it, rather than each independently building
       its own projected program -- see [Region_computation.group]. Every
       other case (single-output, or not Region-authored at all) keeps the
       existing one-stage-per-output-edge path unchanged, just wrapping its
       [Region_program.t] as [Solo]. *)
    if List.length outs > 1 && Region_computation.is_region_authored op then
      let group =
        match
          Region_computation.group ~limits ~op ~operand:(fun id ->
              Tensor_id.Map.find_opt id env)
        with
        | Ok group -> group
        | Error error ->
            Err.raise_error ~pp_error:Region_computation.pp_error error
      in
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
    else
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
  let _env, rev_stages =
    List.fold_left
      (fun (env, stages) node -> process_node g env stages node)
      (g.Graph.tensors, []) g.Graph.nodes
  in
  let inputs, input_kinds = stage_sources g in
  {
    Stage_program.inputs;
    input_kinds;
    consts = List.rev !consts;
    stages = List.rev rev_stages;
    outputs = g.Graph.outputs;
  }
