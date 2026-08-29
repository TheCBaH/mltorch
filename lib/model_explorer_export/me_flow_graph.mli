(* Project a validated flow spine into the graph that renders it.

   A focused module rather than a case inside a dialect projector: this record
   shape has nothing in common with a [Graph_ir] walk, and [Me_build] is
   functorised over exactly that walk.

   See .ai/model_explorer_design.md. *)

type error =
  [ Me_flow.error
  | Me_limits.over_limit_error (* counted under [Me_limits.Scope.Flow] *) ]

val pp_error : Format.formatter -> [< error ] -> unit

val graph :
  limits:Me_limits.Limits.t ->
  Me_flow.t ->
  (Model_explorer.Graph.t, [> error ]) Err.t
(** The BIPARTITE graph: one node per state, one node per transition, and edges
    [before -> transition -> after]. Bipartite because Model Explorer has no
    edge-selection event and [subgraphIds] is a node field, so a transition has
    to BE a node for selecting it to offer its comparison.

    Every node carries exactly one output slot, leaves included.
    [Me_session.validate] counts a node's slots as the length of its
    [outputsMetadata] and rejects an edge into a slot that does not exist, so a
    leaf without one would make the whole session invalid rather than merely
    render oddly.

    Both ceilings are checked BEFORE anything is allocated, and that ordering is
    the bound. [Me_flow.validate] runs first because [max_states] and
    [max_transitions] live there; the projected [max_nodes_per_graph] and
    [max_edges_per_graph] are then checked here, because they are independently
    wire-selectable and nothing downstream enforces them — [Me_session.validate]
    counts graphs and the [int64] session totals, never a single graph's nodes
    or edges. Every other projector owns these two checks for the same reason.
*)
