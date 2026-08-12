(* PT2 → initial Native navigation.

   Built from [Pt2_native_graph.node_origins] ONLY. That sidecar is a bipartite
   relation between PT2 node identities and native node ids, and it is
   many-to-many in both directions: the lowerer may decompose one PT2 node into
   several native ones, and relayout may fold several into one.

   So the entries are CONNECTED COMPONENTS of that relation, not pairs. A
   pairwise encoding would put one node in several entries, which
   [sync_navigation.ts] requires never happens; components are disjoint by
   construction, which is the same requirement discharged rather than checked.

   Root-scoped: the left-pane universe is [Graph_path.root], because the
   lowerer is root-only and writes [graph_path = []]. Nested graphs stay source
   views and tensor origins stay metadata.

   See .ai/model_explorer_design.md. *)

type t = {
  entries : Me_session.Mapping_entry.t list;
      (** one per component, in canonical order *)
  created : string list;
      (** native node ids with no PT2 origin — an importer-introduced node *)
  deleted : string list;  (** PT2 node ids with no native counterpart *)
}

type error = Me_limits.over_limit_error
(** Counting only, under [Me_limits.Scope.Navigation]. *)

val pp_error : Format.formatter -> [< error ] -> unit

val of_origins :
  limits:Me_limits.Limits.t ->
  source_nodes:string list ->
  Pt2_native_graph.Node_origin.t list Graph_ir.Node_id.Map.t ->
  (t, [> error ]) Err.t
(** [source_nodes] is the PT2 root graph's own node ids, and it is a parameter
    rather than something derived from the sidecar because the sidecar is
    indexed BY NATIVE NODE: a PT2 node with no counterpart never appears in it
    at all, so [deleted] computed from it alone could only ever be empty. A
    field that cannot be populated is not a field.

    [created] and [deleted] are derived from the same relation the entries come
    from, so a node cannot be both in a component and reported as created or
    deleted.

    Deterministic: components are emitted in ascending order of their smallest
    native node id, and the ids within each side are sorted. Two runs over the
    same sidecar therefore produce byte-identical output, which the session's
    determinism claim quantifies over. *)

val sync : t -> Me_session.Sync_navigation.t
(** With [show_diff_highlights] set: {!t}'s [created] and [deleted] are what the
    highlight is derived from, so a caller that has one has the other. *)
