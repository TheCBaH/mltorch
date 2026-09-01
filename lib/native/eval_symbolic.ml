(* See eval_symbolic.mli. The env maps each edge id to its [Tensor_sig.t] (the
   [S.input] in symbolic mode); since the builder already created a signature for
   every edge, the env starts as [graph.tensors] and a node's operands resolve to
   their producers' signatures directly. A missing optional operand is filled with
   a fresh constant signature, recorded in [consts] for grounding. *)

open Graph_ir

type regionized = {
  program : Stage_program.t;
  candidates : (Region_program.t, Regionizer.error) Err.t Tensor_id.Map.t;
}

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

let build ?(regionizers = false) ?(limits = Kernel.Limits.default) (g : graph) =
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
  let process_node (gr : graph) env stages candidates (node : node) =
    let op = node.Node.op in
    let operand r = Tensor_id.Map.find r env in
    let shape_of r = (Tensor_id.Map.find r env).Tensor_sig.shape in
    (* One stage per output edge: single-output ops emit one; a [Discard]-style
           zero-output op emits none (the fold body, hence [E.pixel], never runs). *)
    List.fold_left
      (fun (env, stages, candidates) (output, oid) ->
        let out_sig = Tensor_id.Map.find oid gr.Graph.tensors in
        let regional =
          if regionizers && Regionizer.is_region_authored op then
            Some
              (Regionizer.program ~limits ~op ~output
                 ~output_shape:out_sig.shape
                 ~operand:(fun id -> Tensor_id.Map.find_opt id env)
                 ~fill:(fun _role value shape -> fill value shape))
          else None
        in
        let body, region =
          match regional with
          | Some (Ok program) ->
              ( Err.or_raise ~pp_error:Region_program.pp_error
                  (Region_program.specialize_pixel
                     ~max_size:limits.Kernel.Limits.max_size
                     ~max_depth:limits.Kernel.Limits.max_depth program),
                Some program )
          | None | Some (Error _) ->
              ( Expr.Builder.run
                  (E.pixel op ~output ~operand ~shape_of ~fill Symbolic.out_vec),
                None )
        in
        let st = { Stage_program.Stage.id = oid; sg = out_sig; body; region } in
        let candidates =
          match regional with
          | None -> candidates
          | Some program -> Tensor_id.Map.add oid program candidates
        in
        (Tensor_id.Map.add oid out_sig env, st :: stages, candidates))
      (env, stages, candidates)
      (List.mapi (fun i oid -> (i, oid)) node.Node.outputs)
  in
  let _env, rev_stages, candidates =
    List.fold_left
      (fun (env, stages, candidates) node ->
        process_node g env stages candidates node)
      (g.Graph.tensors, [], Tensor_id.Map.empty)
      g.Graph.nodes
  in
  let inputs, input_kinds = stage_sources g in
  let program =
    {
      Stage_program.inputs;
      input_kinds;
      consts = List.rev !consts;
      stages = List.rev rev_stages;
      outputs = g.Graph.outputs;
    }
  in
  { program; candidates }

let run g = (build g).program
let run_regionized ?limits g = build ~regionizers:true ?limits g
