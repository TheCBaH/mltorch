(* Expression detail: one Kernel value's AST, and how it is merged into a
   session.

   INITIAL SESSIONS are referentially complete: they carry one expression graph
   per canonical operator and native links from that operator and its projected
   Stage/Kernel values. The delta remains the standalone CLI and protocol form,
   where it can merge one independently requested detail into a session.

   THE DELTA CARRIES NO EPOCH AND NO KEY. Three identities take part in a detail
   response — the pending request's, the metadata's, and the payload's — and
   comparing only the first two lets a shell announce key A with a payload for
   key B, which is then validated on its own terms and installed while the
   coordinator completes A. Removing the field beats comparing it: the validated
   key arrives as an ARGUMENT to {!apply}, so metadata and payload cannot name
   different values at all. The epoch belongs to the browser runtime and only
   the bridge holds it, so the staleness check lives there and this stays pure
   in the session and the delta.

   INITIAL LINKS STAY REFERENTIALLY VALID. The graph, view and [subGraphIds]
   links arrive together, so no document ever points at a graph it lacks.

   See .ai/model_explorer_design.md. *)

type error =
  [ `Document of Me_session.Session.error
  | `Key_disagrees_with_ids
    (** the delta's graph or view id is not [Detail_key.id key] *)
  | Me_limits.over_limit_error (* counted under [Me_limits.Scope.Detail] *)
  | `Unsupported_detail_key
    (** well-formed, but names no value in that graph. Distinct from a malformed
        key: one is a bad request, the other a valid request about something
        absent *) ]

val pp_error : Format.formatter -> [< error ] -> unit

(** {1 Building one} *)

val of_value :
  limits:Me_limits.Limits.t ->
  key:Me_request.Detail_key.t ->
  Kernel.Value.t ->
  (Model_explorer.Graph.t, [> Me_limits.over_limit_error ]) Err.t
(** One typed Region/Expr decomposition graph. Its edges go from a construct to
    its constituents and carry a [role] metadata field; value, Boolean, index,
    region, binding, and presentation nodes state their language and closed
    constructor as attributes. Bound variables retain their lexical binder id as
    data rather than becoming misleading dataflow edges.

    The exact number of emitted nodes, including presentation roots, binders,
    Boolean terms, index terms, coordinates, and Region locals, is measured
    before allocating graph nodes and checked against [max_detail_nodes]. *)

val display_of_i64_stage : Stage_program.Stage_i64.t -> Kernel.Value.t
(** A display-only [Kernel.Value.t] for an int64 stage (its pixel under
    [i64_to_float]), so an operator with an int64 output can be rendered. *)

val of_operator :
  limits:Me_limits.Limits.t ->
  key:Me_request.Detail_key.t ->
  outputs:Kernel.Value.t list ->
  (Model_explorer.Graph.t, [> Me_limits.over_limit_error ]) Err.t
(** One graph for every ordered output of a canonical Native operator. The
    caller derives [outputs] from the rebuilt node; this function only projects
    the already-authoritative Region computations. *)

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
    counts. Installing the graph, view and parent-node [subGraphIds] happens
    together. An operator detail also links every Stage or Kernel value with its
    exact canonical-origin attribute, so those values are alternate entry points
    to the same graph rather than eager copies of it. *)
