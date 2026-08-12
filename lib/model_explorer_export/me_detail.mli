(* Expression detail: one Kernel value's AST, and how it is merged into a
   session that has already been rendered.

   ON DEMAND, and that is the whole reason it is a delta rather than a stage. A
   kernel value's expression can be far larger than the graph node that produced
   it, and a session that carried every one of them up front would pay for every
   expression to show one.

   THE DELTA CARRIES NO EPOCH AND NO KEY. Three identities take part in a detail
   response — the pending request's, the metadata's, and the payload's — and
   comparing only the first two lets a shell announce key A with a payload for
   key B, which is then validated on its own terms and installed while the
   coordinator completes A. Removing the field beats comparing it: the validated
   key arrives as an ARGUMENT to {!apply}, so metadata and payload cannot name
   different values at all. The epoch belongs to the browser runtime and only
   the bridge holds it, so the staleness check lives there and this stays pure
   in the session and the delta.

   THE INITIAL SESSION STAYS REFERENTIALLY VALID. A Kernel value node carries no
   [subGraphIds] until its detail exists — only a [detail] attribute the shell
   reads — so nothing in a fresh session points at a graph that is not there.

   See .ai/model_explorer_design.md. *)

type error =
  [ `Key_disagrees_with_ids
    (** the delta's graph or view id is not [Detail_key.id key] *)
  | `Unsupported_detail_key
    (** well-formed, but names no value in that graph. Distinct from a malformed
        key: one is a bad request, the other a valid request about something
        absent *)
  | `Over_limit of string * int
  | `Document of Me_session.Session.error ]

val pp_error : Format.formatter -> [< error ] -> unit

(** {1 Building one} *)

val of_value :
  limits:Me_limits.Limits.t ->
  key:Me_request.Detail_key.t ->
  Kernel.Value.t ->
  (Model_explorer.Graph.t, [> `Over_limit of string * int ]) Err.t
(** The value's expression as a graph: one node per AST node, edges from child
    to parent, bounded by [max_detail_nodes] BEFORE the walk that would build
    them — an expression is exactly the shape whose size is not apparent from
    the thing that names it.

    The measure is [Expr.Fold.size], which counts INDEX TREES too and so exceeds
    the number of value nodes produced: a convolution is 88 by that measure and
    8 nodes here. The bound is conservative deliberately — the index trees are
    what the bounded per-node rendering pays for, so they belong in the figure
    the ceiling governs — and the rejection is named for what was measured
    rather than for what would have been built.

    A [Select]'s condition and a [Reduce]'s bounds are NOT walked: they are
    index-level terms, not value-level ones, and giving them nodes would put two
    languages in one graph. Every node instead carries the subtree rooted there,
    rendered and bounded — [Expr.Pp.index] takes a [names] function precisely so
    its output cannot depend on allocation history, and this walk holds no
    scoped naming environment to give it, while [Expr.Pp.value] builds its own.
*)

(** {1 The delta} *)

module Delta : sig
  type t = {
    schema_version : int;
    collection : string;
    graph : Model_explorer.Graph.t;  (** id must equal [Detail_key.id key] *)
    view : Me_session.View.t;  (** id must equal [Detail_key.id key] *)
    node_data : Me_session.Node_data_set.t list;
    diagnostics : Me_limits.Diagnostic.t list;
  }

  val jsont : t Jsont.t
end

val apply :
  key:Me_request.Detail_key.t ->
  limits:Me_limits.Limits.t ->
  Me_session.Session.t ->
  Delta.t ->
  (Me_session.Session.t, [> error ]) Err.t
(** In this order, and ALL of it before anything is mutated:

    + [graph.id = view.id = Detail_key.id key]. There is no separate parent-node
      check — [parent_node] is derived from the key's value, so it cannot
      disagree — and no payload-versus-metadata key check, since the payload
      carries no key.
    + the key names a node in [key.parent_graph], else
      [`Unsupported_detail_key].
    + the DELTA ALONE against [max_detail_nodes]. [max_detail_bytes] is NOT
      checked here — this API is value-level, and a byte ceiling has nothing to
      measure until the delta is encoded, which {!Me_export.encode_bounded} is
      what enforces it against.
    + the MERGED session against the aggregates, which count every already
      installed detail: [max_detail_graphs], [max_graphs], [max_total_nodes],
      [max_views].
    + the merged session re-validated.

    Re-requesting a detail REPLACES the existing one by [Detail_key.equal], so
    repeated requests cannot inflate the aggregates — only the committed result
    counts. Installing the graph, the view and the parent node's [subGraphIds]
    happens together, because a detail commits all of them or none. *)
