(* [Visualization_session] v1 — the deterministic interchange document.

   DETERMINISTIC: no wall time, no nonce, no host state. Two loads of the same
   bytes with the same provenance and the same options must produce identical
   JSON, natively and under node. Provenance is part of that statement rather
   than noise — the digest rule below means identical bytes legitimately carry
   [Some d] when they arrived through the catalog and [None] when a local file
   supplied them.

   IMPORTED SESSION JSON IS OUT OF SCOPE FOR v1. There is no CLI or browser
   import mode; {!Session.jsont}'s decode half exists for controlled round-trip
   and conformance tests only. It is stated here because a decoder that exists
   reads as a decoder that is trusted, and this one is not.

   See .ai/model_explorer_design.md. *)

module Producer : sig
  type t = { tool : string; session_schema : int }
end

module Model_summary : sig
  type source_kind = Pt2 | Json

  type t = {
    name : string;
    source_kind : source_kind;
    source_bytes : int64;
    source_sha256 : string option;
        (** Populated ONLY when the caller supplied an expected digest — the
            catalog path, where it was computed at build time and the browser
            verified it. A CLI opening an arbitrary local file has nothing to
            verify against and records [None]. There is no SHA-256 in this
            repository or its switch and [Digest] is MD5; this rule is why none
            is needed, and both shells agreeing on [None] for local input is
            what keeps the parity gate unaffected. *)
    pt2_graph_count : int;
    op_targets : int;
  }
end

module Pane_state : sig
  type t = { collection : string; graph : string }
end

module Mapping_entry : sig
  type t = { left : string list; right : string list }
  (** One correspondence, as a pair of node-id sets — a connected component of
      the correspondence relation, so 1:N, N:1 and N:M are all one shape. *)
end

module Sync_navigation : sig
  type t = { entries : Mapping_entry.t list; show_diff_highlights : bool }
end

module Comparison : sig
  type t = {
    id : string;
    label : string;
    left : Pane_state.t;
    right : Pane_state.t;
    sync : Sync_navigation.t;
    overlays_left : Model_explorer.EdgeOverlay.t list;
    overlays_right : Model_explorer.EdgeOverlay.t list;
  }
end

module Node_data_set : sig
  type value = { value : float; label : string option }

  type t = {
    name : string;
    graph : string;  (** GRAPH-ADDRESSED: a set belongs to one graph *)
    results : (string * value) list;
        (** keyed by NODE ID. Upstream also permits an output tensor name;
            session v1 does not claim to mirror that. *)
  }

  val jsont : t Jsont.t
end

module Capability : sig
  type graph_stage =
    | Source
    | Initial_native
    | Canonical
    | Native4d
    | Stage_program
    | Kernel
    | Fusion
        (** Which exported GRAPH a capability is about. Not [Me_ids.Layer.t]:
            initial and canonical Native are two stages of one layer, and fusion
            is a stage of the Kernel layer. *)

  type feature =
    | Flow
    | Verification
    | Pass_audits
    | Fold
    | Expression_detail
    | Loop_ir
    | Codegen

  type key = Graph_stage of graph_stage | Feature of feature

  val key_name : key -> string
  val stage_name : graph_stage -> string

  val all_stages : graph_stage list
  (** Every stage, in constructor order. Built by walking a successor chain, so
      a stage cannot be added without reaching this list — and it is what a
      request's stage selection is NORMALISED against, which is how that list is
      bounded by construction rather than by a separate ceiling. *)

  val all_keys : key list
  (** Every key, in canonical order. Completeness is checked against this, so an
      absent key is a producer defect rather than an ambiguous silence. *)

  module Pass_audit_status : sig
    type t = {
      retained_reports : int64;
      omitted_reports : int64;
      omitted_counts : Pass.Outcome_counts.t;
    }
    (** A named record module rather than an inline variant record, per
        CLAUDE.md — which also keeps [omitted_reports] out of the same scope as
        [Pass.Audit_summary.t]'s field of that name. *)
  end

  type payload =
    | Graph of string
    | Verification_summary of Pass.Outcome_counts.t
        (** the COMPOSED whole-pipeline report *)
    | Pass_audit_status of Pass_audit_status.t  (** per-pass, may be partial *)
    | Present

  type reason =
    | Unsupported_operator
    | Unsupported_input
    | Unsupported_graph_shape
    | Outside_dialect_domain
    | Over_limit
    | Requires_payloads
    | Prerequisite_unavailable
    | Not_implemented

  type status =
    | Available of payload
    | Unavailable of { reason : reason; detail : string option }
    | Not_requested

  type t = { key : key; status : status }

  val compatible : t -> bool
  (** The key/payload/status table, which IS the specification:

      - any [Graph_stage], and [Feature Flow]: [Graph id]
      - [Feature Verification]: [Verification_summary]
      - [Feature Pass_audits]: [Pass_audit_status]
      - [Feature Fold], [Feature Expression_detail]: [Present]
      - [Feature Loop_ir], [Feature Codegen]: never [Available], and never
        [Not_requested] either — permanently [Unavailable Not_implemented]

      Verification splits from [Pass_audits] because
      [Native_interp.transformed.composed] survives per-pass audit truncation:
      mapping that truncation to one [Verification -> Unavailable Over_limit]
      hides a result that is still available, while [Available] would conceal
      that the detail is incomplete. Two keys say both things. *)

  val units : string
  (** Prose, in the signature on purpose: [Verification_summary] counts VERIFIER
      CLUSTERS in the composed report; [Pass_audit_status] counts AUDIT REPORTS
      in its two scalars and CLUSTERS in [omitted_counts]. *)
end

module View : sig
  type kind =
    | Stage of Capability.graph_stage
        (** the exported graph this view shows; "canonical" is a VIEW naming the
            final Native state's graph, not a separate projection, and when
            every pass and the pack are identities that graph is the initial one
        *)
    | Flow
    | Compare  (** names a {!Comparison} by the same id *)

  type t = {
    id : string;
    label : string;
    kind : kind;
    collection : string;
    graph : string;
  }

  val jsont : t Jsont.t
end

module Session : sig
  type t = {
    schema_version : int;
    producer : Producer.t;
    model : Model_summary.t;
    graph_collections : Model_explorer.GraphCollection.t list;
    views : View.t list;
    comparisons : Comparison.t list;
    node_data_sets : Node_data_set.t list;
    flow : Me_flow.t option;
    capabilities : Capability.t list;
    diagnostics : Me_limits.Diagnostic.t list;
    default_view : string;
  }

  type error =
    [ `Duplicate_graph of string
    | `Duplicate_node of string * string  (** graph, node *)
    | `Unknown_graph of string
    | `Unknown_node of string * string  (** graph, node *)
    | `Wrong_collection of string * string  (** graph, declared collection *)
    | `Dangling_edge of string * string  (** graph, source node *)
    | `Slot_mismatch of string * string
    | `Unknown_view of string
    | `Duplicate_view of string
    | `Duplicate_comparison of string
    | `Unknown_comparison of string
    | `Node_in_two_entries of string * string  (** comparison, node *)
    | `Comparison_panes_disagree of string  (** transition id *)
    | `Duplicate_capability of string
    | `Missing_capability of string
    | `Incompatible_capability of string
    | `Over_limit of string * int
    | `Over_limit_64 of string * int64
      (** the [int64]-ceiling aggregates: [max_total_nodes]/[max_total_edges]
          and the overlay-edges total, none of which a real model can approach
          but which an untrusted document is not trusted not to claim *)
    | Me_flow.error ]

  val pp_error : Format.formatter -> [< error ] -> unit

  val validate : limits:Me_limits.Limits.t -> t -> (unit, [> error ]) Err.t
  (** Unique graph ids, unique node ids within each graph; every [View.graph],
      [Pane_state.graph], node-data graph and node key, and [subGraphIds] entry
      resolves, and a [View]/[Pane_state] naming a graph in the WRONG collection
      is [`Wrong_collection] even though the graph id itself exists; every
      [IncomingEdge.sourceNodeId] exists in its graph with a matching
      [sourceNodeOutputId] — the output slot must actually exist, so a node with
      no [outputsMetadata] admits no incoming edge at all, not just none past a
      nonexistent slot; [default_view] resolves; within each [Comparison.sync],
      each [Mapping_entry] member resolves to a node in ITS SIDE's pane graph,
      not merely to a node id that exists somewhere, and each node id appears in
      at most one [Mapping_entry] on each pane's side — SCOPED TO THE
      COMPARISON, not globally, since canonical Native is the left pane of two
      comparisons and its nodes therefore legitimately appear on the left of two
      mapping sets; every capability key present exactly once with a compatible
      payload; the flow DAG contract; and every [Transition.comparison]
      resolving with its left/right pane graph ids equal to that transition's
      before/after state graph ids.

      Every ceiling in {!Me_limits.Limits.t} that names a document shape rather
      than a single field is enforced here too, not merely declared:
      [max_attrs_per_node], [max_metadata_items_per_node] (inputsMetadata),
      [max_outputs_metadata_per_node] and [max_namespace_depth] per node;
      [max_mapping_entries_per_comparison] per comparison;
      [max_total_nodes]/[max_total_edges] and the overlay-edges total, summed
      across every graph/comparison respectively, in checked [int64] so the
      running sum cannot itself narrow past the ceiling meant to catch it.

      Recoverable versus fatal: the eight [Capability.reason]s are capability
      outcomes. Invariant failures, a malformed generated graph and
      verification-policy rejection are session errors and are never downgraded
      to one. *)

  val jsont : t Jsont.t
  (** Encode is the production path. DECODE EXISTS FOR ROUND-TRIP AND
      CONFORMANCE TESTS ONLY — v1 has no import mode, and nothing in this
      library calls it on input it did not produce. *)
end

module Runtime : sig
  type t = { epoch : string; limits : Me_limits.Limits.t; session : Session.t }
  (** RUNTIME state, held by the browser and never serialized by
      {!Session.jsont}. The epoch is browser-request lifecycle, not a property
      of the model — which is exactly why it cannot live in the document: two
      independent loads under different epochs must produce byte-identical
      session JSON. *)
end
