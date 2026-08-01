(* Remove the nodes that do not contribute to a graph output.
   See .ai/native_transform_design.md §12g.

   The predicate is GLOBAL REACHABILITY, not local unusedness, and the pass runs
   WITHOUT [Pass.fixpoint]. Both are deliberate, and they are the same decision
   seen twice.

   "Every output unused" is the obvious local predicate and it is a trap this
   repository has already fallen into once. On a dead CHAIN only the terminal
   node satisfies it — each predecessor still has a use, namely the node about to
   be removed — so the chain peels one node per sweep and needs as many
   iterations as it has links. [Pass.fixpoint]'s default [max_iters = 16] then
   fails with [`Not_converged] on a graph that was converging perfectly well,
   which is exactly the failure §12d's review recorded for [Reuse_permute]:
   "sixteen sweeps needed, which exactly exhausted Pass.fixpoint's default
   max_iters = 16 fuel ... on a graph that was never actually stuck".

   Reachability avoids that by construction rather than by raising the bound. A
   node is live iff one of its outputs is reachable from [Graph.outputs] through
   the operand edges, which is a property of the WHOLE graph, so every dead node
   matches in the first sweep — [Pass.per_node] collects them all and merges
   their recipes before a single [Rewrite.apply]. Deleting the unreachable set
   cannot expose new unreachable nodes, so one sweep converges and no fuel bound
   enters the pipeline.

   It stays a [Pass.per_node] rather than a hand-built [Pass.t] because
   verification is not something a custom pass gets for free: [Pass.verified],
   which calls [Map_verify.step] and mints the [Audit], is reached only through
   the private [Pass.of_sweep], and [Pass.run_with] does not wrap what a pass
   returns. A [Pass.t] calling [Rewrite.apply] directly would skip verification
   silently even under [run_all ~verify] — success reported, nothing checked.

   [Discard] needs no special case: a sink produces nothing, so it is never
   reachable, so it is always removed. That is what makes this the pass
   .ai/native_multi_output_design.md defers the [Discard] sinks to. *)

open Graph_ir

(* Tensors that feed a graph output, walking backwards through each producer's
   operands. [Graph_view.def] answers [None] exactly for a graph input, which
   validation guarantees, so an absent producer terminates the walk rather than
   signalling a dangling edge. *)
let reachable_tensors view =
  let rec go seen = function
    | [] -> seen
    | id :: rest ->
        if Tensor_id.Set.mem id seen then go seen rest
        else
          let seen = Tensor_id.Set.add id seen in
          go seen
            (match Graph_view.def view id with
            | None -> rest
            | Some n -> Graph_ir.operands n.Node.op @ rest)
  in
  go Tensor_id.Set.empty (Graph_view.graph view).Graph.outputs

(* Recomputed per node rather than once per sweep: [Pass.per_node]'s callback is
   rank-2 in the version, so it cannot close over state created inside a sweep,
   and the only way to hoist this would be to export [Pass.of_sweep]. That trades
   public API for O(n) integer work on graphs of a few hundred nodes, which is
   the wrong way round. *)
let on_node : type v. Pass.env -> node -> (v, unit) Recipe.t option =
 fun { view; _ } n ->
  let live = reachable_tensors view in
  if List.exists (fun id -> Tensor_id.Set.mem id live) n.Node.outputs then None
  else Some (Recipe.replace ~remove:[ n.Node.id ] ~insert:[] ())

(* No [Pass.fixpoint] wrapper, per the header. A caller that wraps one anyway is
   not wrong, only wasteful: the second sweep finds nothing and converges. *)
let pass = Pass.per_node ~name:"dce" { Pass.on_node }
