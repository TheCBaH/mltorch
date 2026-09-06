(* See eval_symbolic.mli. The env maps each edge id to its [Tensor_sig.t] (the
   [S.input] in symbolic mode); since the builder already created a signature for
   every edge, the env starts as [graph.tensors] and a node's operands resolve to
   their producers' signatures directly. A missing optional operand is filled with
   a fresh constant signature, recorded in [consts] for grounding. *)

open Graph_ir

let f32 = Payload.Fmt Payload.F32

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
    let operand r = Tensor_id.Map.find r env in
    let shape_of r = (Tensor_id.Map.find r env).Tensor_sig.shape in
    (* One stage per output edge: single-output ops emit one; a [Discard]-style
           zero-output op emits none (the fold body, hence [E.pixel], never runs). *)
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
          | Some (Ok program) -> program
          | Some (Error error) ->
              Err.raise_error ~pp_error:Region_computation.pp_error error
          | None ->
              Region_program.pixel
                (Expr.Builder.run
                   (E.pixel op ~output ~operand ~shape_of ~fill Symbolic.out_vec))
        in
        let st = { Stage_program.Stage.id = oid; sg = out_sig; computation } in
        (Tensor_id.Map.add oid out_sig env, st :: stages))
      (env, stages)
      (List.mapi (fun i oid -> (i, oid)) node.Node.outputs)
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
