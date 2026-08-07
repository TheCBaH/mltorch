(* The flow spine: which graphs exist, and what turned each into the next.

   [Pass.Exec_id.t] names a TRANSITION, so it cannot name the [N + 1] states
   that [N] changed steps produce, and it does not exist at all for an import,
   a cross-dialect conversion, an adaptation, or a run in which no pass changed
   anything. So this module owns its own spine rather than deriving one from
   execution identities.

   The structure is a rooted DAG, not a chain: canonical Native branches to both
   Native4D and the symbolic stages, so "states = transitions + 1" describes
   nothing here.

   See .ai/model_explorer_design.md. *)

module Pass_execution : sig
  type t = { layer : Me_ids.Layer.t; exec : Pass.Exec_id.t }
  (** A pass execution QUALIFIED by its dialect. [Pass.Exec_id.t]'s ordinal is
      dense per specialization and pass names are arbitrary, so a Native and a
      Native4D execution can be equal; only this pair is unique session-wide.

      [Pass] does not depend on [Me_ids.Layer] — the exporter adds the layer
      while consuming each branch's trace, which is also the only place that
      knows which branch it is reading. *)

  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit

  module Map : Map.S with type key = t
end

module State : sig
  type t = {
    id : string;  (** [s/<layer>/<NNN>] *)
    graph : string;  (** [g/<layer>/<NNN>] *)
    layer : Me_ids.Layer.t;
    label : string;
    produced_by : string option;
        (** the [Transition.id] that produced it; [None] for a root *)
  }
end

module Transition : sig
  type kind = Import | Pass of Pass_execution.t | Pack | Cross_dialect | Adapt

  val kind_name : kind -> string

  type t = {
    id : string;  (** [t/<layer>/<NNN>] *)
    before : string;  (** a [State.id] *)
    after : string;  (** a [State.id] *)
    kind : kind;
    comparison : string option;
        (** the [Comparison.id] this transition opens, when it has one *)
  }
end

type t = {
  states : State.t list;
  transitions : Transition.t list;
  graph : string;
      (** the id of the BIPARTITE graph that renders this spine: one node per
          state and one node per transition, with edges
          [state -> transition -> state].

          Bipartite because Model Explorer has no edge event and [subgraphIds]
          is a node field, so an edge cannot carry navigation. A transition has
          to be a node for selecting it to open its comparison. *)
}

type error =
  [ `Duplicate_state of string
  | `Duplicate_transition of string
  | `Unknown_state of string  (** named by a transition endpoint *)
  | `No_root  (** no [Pt2] state without a producer *)
  | `Multiple_roots of int
  | `Unreachable_state of string
  | `Cycle of string
    (** A state on a cycle. Not reachable through the other checks — see
        {!validate} — and retained as a termination guard. *)
  | `Multiple_producers of string
  | `Producer_disagrees of string  (** [produced_by] is not the transition *)
  | `Illegal_transition of string  (** the transition id *)
  | `Pass_layer_disagrees of string
  | `Duplicate_pass_execution of string
  | `Over_limit of string * int  (** the aggregate, and the count offered *) ]

val pp_error : Format.formatter -> [< error ] -> unit

val validate : limits:Me_limits.Limits.t -> t -> (unit, [> error ]) Core.result
(** The rooted-DAG contract, in full: unique state and transition ids; exactly
    one [Pt2] root; every endpoint resolves; every state reachable from the
    root; acyclicity; at most one producer per state, agreeing with
    [produced_by]; only legal [(layer_before, kind, layer_after)] triples; for
    [Pass e], [e.layer] equal to the transition's own layer and to both
    endpoints'; and every [Pass_execution.t] occurring at most once
    session-wide.

    [max_states] and [max_transitions] are checked FIRST, before the walks that
    are linear in them: a bound checked after the work it bounds is not a bound.

    What it does NOT check is [Transition.comparison], which resolves against a
    table this module cannot see. [Me_session.validate] does that, and does it
    with the pane-graph equality that makes it meaningful. *)

val legal_triples : (Me_ids.Layer.t * string * Me_ids.Layer.t) list
(** Every admitted [(before, kind, after)] combination, by kind name — the table
    {!validate} checks against, exposed so a fixture can enumerate what is legal
    instead of restating it. *)

val jsont : t Jsont.t
(** The spine reaches the browser as data, not only as a rendered graph:
    [flow_nav.js] maps a selected node id to a [Transition.id] to its
    [Comparison], and [Transition.comparison] exists nowhere else. A session
    carrying the flow only as a bipartite graph would leave every transition
    node unclickable. *)
