(* Materialise a node whose operands are all constants.
   See .ai/native_transform_design.md §12b.

   This is what lets every other pass stay simple: constant arithmetic is
   expressed as ordinary graph ops — a relayout emits a [Permute] of a weight,
   batch-norm folding emits the parameter arithmetic as nodes — and this one pass
   turns the resulting all-constant sub-DAG into data. Under [Pass.fixpoint] it
   collapses a whole sub-DAG, one layer per sweep, because a node only becomes
   foldable once the sweep before it bound its operands.

   Evaluation reuses [Region.extract] and [Eval_direct], so there is no second
   evaluation path that could disagree with the op it replaces. *)

open Graph_ir

(* Exactly one output and at least one operand. Zero outputs is [Discard], whose
   removal is dead-code elimination's business; several outputs would need every
   one of them bound in a single recipe, which [Recipe.fold_to_constant] does not
   express — left as future work rather than half-done.

   A constant with no bound payload is skipped rather than treated as an error:
   payloads live outside the IR (§2), so "declared constant, not yet loaded" is a
   legitimate state, and the node simply stays in the graph. *)
let foldable ~constants ~view (n : node) =
  match (n.Node.outputs, Graph_ir.operands n.Node.op) with
  | [ out ], (_ :: _ as operands)
    when List.for_all
           (fun id ->
             Graph_view.is_constant view id && Tensor_id.Map.mem id constants)
           operands ->
      Some (out, operands)
  | _ -> None

(* [None] is unreachable given what [foldable] checked and what the view already
   validated — the region is a known singleton, shapes and arities were verified
   by [Graph_view.of_graph], the extracted graph has no [Input]-kind inputs since
   every operand is constant, and every constant has a payload. Skipping is the
   safe response to a case that cannot arise: the graph keeps computing the node
   itself. *)
let evaluate ~constants ~view (n : node) out =
  match Region.of_nodes view (Node_id.Set.singleton n.Node.id) with
  | Error _ -> None
  | Ok region -> (
      let g = Region.extract view region in
      let payloads =
        List.filter_map
          (fun id ->
            Option.map (fun p -> (id, p)) (Tensor_id.Map.find_opt id constants))
          g.Graph.inputs
      in
      match Eval_direct.run g ~constants:payloads ~inputs:[] with
      | Error _ -> None
      (* [Eval_direct] returns every edge, so the value is there whether or not
         the folded node's output escaped the region. *)
      | Ok env -> Tensor_id.Map.find_opt out env)

let on_node ({ constants; view } : Pass.env) (n : node) =
  match foldable ~constants ~view n with
  | None -> None
  | Some (out, operands) ->
      Option.map
        (fun value ->
          Recipe.fold_to_constant ~node:n.Node.id ~output:out ~value
            ~sources:operands)
        (evaluate ~constants ~view n out)

let pass = Pass.per_node ~name:"fold_const" { Pass.on_node }
