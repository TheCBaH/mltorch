(* See eval_direct.mli. One [Eval_op.Make (Direct)] for the whole walk; the env maps
   each edge id to its concrete tensor. Subgraphs are evaluated by recursion: the
   sub's inputs are seeded from the call args, then its outputs are bound to the
   node's output edges (ids are unique tree-wide, so merging the sub's
   intermediates back cannot collide). *)

open Graph_ir
module E = Eval_op.Make (Direct)

let sig_shape (g : graph) r =
  (Tensor_id.Map.find r g.Graph.tensors).Tensor_sig.shape

let rec run_graph (g : graph) (env : Tensor.packed Tensor_id.Map.t) :
    Tensor.packed Tensor_id.Map.t =
  List.fold_left (fun env node -> eval_node g env node) env g.Graph.nodes

and eval_node (g : graph) (env : Tensor.packed Tensor_id.Map.t) (node : node) :
    Tensor.packed Tensor_id.Map.t =
  match node.Node.op with
  | Subgraph { graph = sub; args } ->
      let sub_env =
        List.fold_left2
          (fun e iid aid ->
            Tensor_id.Map.add iid (Tensor_id.Map.find aid env) e)
          Tensor_id.Map.empty sub.Graph.inputs args
      in
      let sub_result = run_graph sub sub_env in
      (* surface the sub's intermediates, then alias node outputs to sub outputs *)
      let env = Tensor_id.Map.union (fun _ _ b -> Some b) env sub_result in
      List.fold_left2
        (fun e oid sub_oid ->
          Tensor_id.Map.add oid (Tensor_id.Map.find sub_oid sub_result) e)
        env node.Node.outputs sub.Graph.outputs
  | op ->
      let operand r = Tensor_id.Map.find r env in
      let shape_of r = sig_shape g r in
      let fill v shape = Tensor.materialize shape (fun _ -> v) in
      let out_shape =
        match
          Graph_shape.output_shape op ~sig_of:(fun r ->
              Tensor_id.Map.find r g.Graph.tensors)
        with
        | [ sh ] -> sh
        | _ -> invalid_arg "Eval_direct: expected a single-output op"
      in
      let result =
        Schedule.evaluate out_shape (E.pixel op ~operand ~shape_of ~fill)
      in
      let oid =
        match node.Node.outputs with
        | [ oid ] -> oid
        | _ -> invalid_arg "Eval_direct: expected a single output id"
      in
      Tensor_id.Map.add oid result env

let run (g : graph) ~(inputs : (Tensor_id.t * Tensor.packed) list) =
  let env0 =
    List.fold_left
      (fun e (id, t) -> Tensor_id.Map.add id t e)
      Tensor_id.Map.empty inputs
  in
  run_graph g env0
