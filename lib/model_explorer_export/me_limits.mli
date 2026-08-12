(* Ceilings and profiles for the Model Explorer export path, and they are
   different things.

   [Hard] holds NON-NEGOTIABLE ceilings, derived from the ZIP format, the
   js_of_ocaml 32-bit [int], the [Bytesrw.Bytes.Writer] limit type and
   [Kernel.Limits.Hard]'s precedent — never from measuring well-behaved models.
   A profile may tighten below them and never widen past.

   [Limits.t] is a profile: a tunable set of per-field ceilings plus one field
   ([response_live_bytes]) DERIVED from two of them, so the peak a profile
   implies is computed once, checked once, and cannot be recomputed differently
   at a call site.

   [Wire_limits.t] is the subset of profiles that may cross the worker
   boundary. It is [private Limits.t] behind a check against a ceiling, so
   "the worker and the page disagree about the profile" is unconstructable
   rather than merely rejected somewhere.

   See .ai/model_explorer_design.md. *)

module Invalid : sig
  type t = { name : string; value : int64 }
  (** [value] is the rejected figure, widened to [int64] whichever width the
      field has, so one error row serves both. [name] is the field, dotted for a
      nested one ("zip.max_entries"). *)
end

type live_error = [ `Live_overflow of string ]
(** The phase whose sum overflowed [int64] — [R_decode], [R_build], [R_commit]
    or [R_install]. Narrower than [error] because [response_live_bytes] is a
    pure calculator over two numbers and has no field to reject. *)

type error = [ `Invalid_limit of Invalid.t | live_error ]

val pp_error : Format.formatter -> [< error ] -> unit

(** {1 The over-limit domain}

    A count that crossed a ceiling, owned HERE rather than redeclared by each
    validator. Nine modules used to carry their own
    [`Over_limit of string * int] and print it their own way, which made the
    aggregate name a stringly-typed enum with nine independent spellings and no
    way for a caller to branch on which aggregate overran.

    Both halves of the payload are closed vocabularies: {!Scope} names the
    validator that rejected, {!Field} the aggregate it was counting. Neither is
    reconstructible from the other — three different scopes count [Nodes]. *)

module Scope : sig
  (** Which validator rejected. Carried in the payload rather than supplied by
      the printer: "which of the nine counted this" is a fact about the failure,
      and a caller that only reads the rendered string cannot act on it. *)
  type t =
    | Session
    | Graph
    | Value_graph
    | Source_graph
    | Detail
    | Navigation
    | Fusion
    | Flow
    | Verification

  val to_string : t -> string
  val all : t list
end

module Field : sig
  (** The aggregate that overran, in the renderer's own spelling.

      A CLOSED set: every ceiling in {!Limits} that something counts against
      appears exactly once, so adding a ceiling without giving it a name here is
      a compile error at the call site rather than a new string. *)
  type t =
    | Views
    | Comparisons
    | Node_data_sets
    | Diagnostics
    | Graphs
    | Nodes
    | Edges
    | Total_nodes
    | Total_edges
    | Attrs_per_node
    | Metadata_items_per_node
    | Outputs_metadata_per_node
    | Namespace_depth
    | Mapping_entries_per_comparison
    | Mapping_members
    | Mapping_members_per_entry
    | States
    | Transitions
    | Detail_graphs
    | Detail_nodes
    | Expression_nodes
    | Group_node_attributes
    | Node_data_results
    | Overlay_edges
    | Overlay_edges_total

  val to_string : t -> string
  (** The wire spelling — ["nodeDataResults"], not ["Node_data_results"]. *)

  val all : t list
  (** Built by walking a successor chain, as [Me_ids.Layer.all] is, so a
      constructor cannot be added without reaching this list. *)
end

module Over_limit : sig
  type t = { scope : Scope.t; field : Field.t; count : int64 }
  (** [count] is the figure OFFERED, not the ceiling, and is widened to [int64]
      whichever width the field has — the same decision, for the same reason, as
      {!Invalid.t}'s [value]. It replaces a separate [`Over_limit_64] tag that
      existed only to carry the two [int64] aggregates. *)

  val pp : Format.formatter -> t -> unit
end

type over_limit_error = [ `Over_limit of Over_limit.t ]
(** Narrower than {!error}, for the same reason {!live_error} is: a module that
    only counts has no limit field to reject. Validators FLAT-INCLUDE this — it
    is a shared base domain, not a crossed seam, so there is nothing to wrap and
    the origin of the count is already in [scope]. *)

val check :
  scope:Scope.t ->
  Field.t ->
  int ->
  ceiling:int ->
  (unit, [> over_limit_error ]) Err.t
(** [check ~scope field n ~ceiling] fails when [n > ceiling]. The one place the
    comparison is written; nine copies of it used to disagree about nothing but
    could. *)

val check64 :
  scope:Scope.t ->
  Field.t ->
  int64 ->
  ceiling:int64 ->
  (unit, [> over_limit_error ]) Err.t
(** {!check} for the [int64] aggregates. Separate rather than widening at the
    call site: a caller holding an [int] total under a 32-bit [int] would have
    to prove the conversion, and these two are summed as [int64] precisely so it
    never has to. *)

(** {1 Hard ceilings} *)

module Hard : sig
  (** Fixed before Stage 1 from reasoning, not measurement. Every value here is
      either a platform fact, a consequence of a grammar declared elsewhere, or
      a deliberately conservative expansion factor — never a figure that a later
      calibration may move, because the derived ceilings below are computed FROM
      them and a factor that changes after calibration changes the value it was
      used to derive. Measurement may only tighten a PROFILE. *)

  (** {2 The platform bound everything else answers to} *)

  val jsoo_safe_bytes : int64
  (** The aggregate live-heap ceiling for the browser shell. Under js_of_ocaml
      [Sys.int_size = 32] and [max_int = 2147483647], so a live total that is
      ever narrowed to [int] must stay inside that domain with room for the
      arithmetic that precedes the narrowing. 2^30 is half of [max_int]: the sum
      of any two totals bounded by it is still representable. *)

  (** {2 Request identity — protocol, not policy}

      Fixed here and nowhere else, so there is exactly one authority. These are
      CONSEQUENCES of the request-id grammar, which [Request_id.of_string]
      checks directly; they exist for the JS bridge and for readers, not as a
      second, weaker rule that could drift from it. *)

  val max_epoch_bytes : int
  (** 36 — a UUID's length. *)

  val max_request_id_bytes : int
  (** 47 = 36 + 1 + 10, the widest [<epoch>-<seq>] the grammar can produce. *)

  val max_seq_exclusive : int64
  (** [int64] and EXCLUSIVE. Valid sequences are [0 .. 4294967295]; the request
      after 4294967295 is exhaustion. 2^32 is not representable as a jsoo [int],
      so nothing narrows this on either backend. *)

  val max_restore_steps : int
  (** How many removal/insertion steps the mount restoration may take before the
      mount is declared inconsistent. Small. Every removal runs a
      [disconnectedCallback], which is user code and may append a child, so
      "remove every child that is not [current]" is a loop over
      attacker-influenced input and gets the same bound as every other one here.
      A repair that can be made to spin forever is not a repair. *)

  (** {2 Document ceilings} *)

  val max_request_json_bytes : int
  (** One encoded request. Requests carry an id, a name, two URLs, a digest, a
      detail key and a normalised stage list — all kilobyte-scale. *)

  val max_response_meta_bytes : int
  (** One response envelope's metadata, which is a small fixed record plus its
      diagnostics: [max_diagnostics * max_diagnostic_bytes] bounds the variable
      half. *)

  val max_response_document_bytes : int
  (** The ceiling a response document may never exceed, whatever profile its
      metadata claims — checked before [TextDecoder]. The per-kind PROFILE
      limits are tighter and are applied after the profile is validated, but
      that validation happens after the page has already sized an allocation, so
      without a hard value first a forged [max_session_bytes] would choose how
      much the main thread allocates. Exported through [mltorch.hard].

      Deliberately BELOW the [max_session_bytes]/[max_detail_bytes] field
      ceilings: the gap is what leaves {!Limits.within_hard_response} a check
      that can fail. *)

  (** {2 Expansion factors}

      Heap bytes per byte of document, for the terms the phase sums below need
      and that only measurement could establish exactly. They are frozen
      conservatively from reasoning because they are INPUTS to a hard
      derivation. *)

  val js_string_expansion : int
  (** A JS string is UTF-16: [n] UTF-8 bytes decode to at most [n] code units of
      2 bytes each. *)

  val js_value_expansion : int
  (** [JSON.parse] of [n] bytes. Every small number becomes a boxed value and
      every object a hidden-class instance. *)

  val session_expansion : int
  (** A decoded [Session.t] per byte of document. JSON arrays become OCaml lists
      — three words per element for as little as two serialized bytes — and
      strings gain a header and padding. *)

  val graph_expansion : int
  (** A Model Explorer processed graph per byte of document, INCLUDING the
      custom element that holds it: the element's own overhead is folded in here
      rather than invented as a separate constant. *)

  val render_state_expansion : int
  (** The encoded [Render_state] per byte of document. It is a projection of the
      session, so it cannot exceed it. *)

  val json_escape_expansion : int
  (** Escaping a sanitised string for JSON. [Diagnostic.create] replaces control
      bytes and invalid sequences with U+FFFD before the length rule applies, so
      the six-byte u-escape form is unreachable and only the quote and the
      backslash expand. *)

  (** {2 Inputs to the request peak}

      The retained set a request builder holds at once. Visible because
      {!max_request_live_bytes} is derived from them and a derivation whose
      inputs are private cannot be re-checked. *)

  val max_request_stages : int
  (** The cardinality of [Me_session.Capability.graph_stage], which bounds a
      request's normalised stage list. Stated here because the request peak is a
      hard constant and that module does not exist yet; when it does, its test
      asserts the two agree. *)

  val max_stage_bytes : int

  val max_digest_bytes : int
  (** A hex SHA-256. *)

  (** {2 Derived aggregates}

      Bounded separately from the per-field ceilings, because a sum of
      individually-in-range factors can still overflow. Four relations tie these
      to the ceilings above — the two peaks and one worst-case conversion
      against {!jsoo_safe_bytes}, and the diagnostics list against
      {!max_response_meta_bytes} — and all four are checked once, in [int64],
      when this module is initialised. A wrapped total passes a [<=] test. *)

  val max_request_live_bytes : int64
  (** The peak the request builder may hold live: the maximum of its three phase
      sums, over builder-owned allocations only. The raw JavaScript object the
      caller passed in is not ours to bound and is not counted. *)

  val max_response_live_bytes : int64
  (** The peak the response path may hold live: the phase sums of
      {!response_live_bytes} evaluated at hard scalars — at
      [max_response_document_bytes], not at any profile — so a [Hard] constant
      does not depend on the profile [Hard] constrains. *)

  (** {2 Per-field ceilings}

      One per tunable [Limits.t] field. [create] checks every field against its
      counterpart here, which is what "may tighten, never widen past" means. *)

  val max_json_bytes : int64
  val max_pt2_bytes : int64
  val max_nodes_per_graph : int
  val max_edges_per_graph : int
  val max_groups_per_graph : int
  val max_attrs_per_node : int
  val max_metadata_items_per_node : int
  val max_outputs_metadata_per_node : int
  val max_namespace_depth : int
  val max_namespace_component_bytes : int
  val max_label_bytes : int
  val max_id_bytes : int
  val max_attr_chars : int
  val max_url_bytes : int
  val max_graphs : int
  val max_total_nodes : int64
  val max_total_edges : int64
  val max_views : int
  val max_comparisons : int
  val max_node_data_sets : int
  val max_states : int
  val max_transitions : int
  val max_mapping_entries_per_comparison : int
  val max_mapping_members_per_entry : int
  val max_mapping_members_total : int
  val max_node_data_results_per_graph : int
  val max_overlay_edges_per_overlay : int
  val max_overlay_edges_total : int
  val max_diagnostics : int
  val max_diagnostic_bytes : int
  val max_session_bytes : int
  val max_trace_entries : int
  val max_audit_reports : int
  val max_detail_nodes : int
  val max_detail_graphs : int
  val max_detail_bytes : int
end

(** {1 The response peak calculator} *)

val response_live_bytes :
  max_session_bytes:int ->
  max_detail_bytes:int ->
  (int64, [> live_error ]) Err.t
(** The maximum of the four response phase sums, in checked [int64] — every
    product and every addition tested BEFORE it is performed, so an overflow is
    reported and never inspected after the fact.

    [Limits.create] calls this and rejects any profile whose peak exceeds
    [Hard.jsoo_safe_bytes]; the result is stored on the validated profile as
    [response_live_bytes], so nothing recomputes it and no caller can pass a
    different one. Enforcement is therefore BEFORE INPUT ACQUISITION: a profile
    that could not fit is refused when it is built, so no document is ever read
    under one.

    Monotone by construction — tightening either input can only lower the
    result. *)

(** {1 Profiles} *)

module Limits : sig
  type t = private {
    max_json_bytes : int64;
    max_pt2_bytes : int64;
    zip : Pt2_zip.Limits.t;
    max_nodes_per_graph : int;
    max_edges_per_graph : int;
    max_groups_per_graph : int;
    max_attrs_per_node : int;
    max_metadata_items_per_node : int;
    max_outputs_metadata_per_node : int;
    max_namespace_depth : int;
    max_namespace_component_bytes : int;
    max_label_bytes : int;
    max_id_bytes : int;
    max_attr_chars : int;
    max_url_bytes : int;
    max_graphs : int;
    max_total_nodes : int64;
    max_total_edges : int64;
    max_views : int;
    max_comparisons : int;
    max_node_data_sets : int;
    max_states : int;
    max_transitions : int;
    max_mapping_entries_per_comparison : int;
    max_mapping_members_per_entry : int;
    max_mapping_members_total : int;
    max_node_data_results_per_graph : int;
    max_overlay_edges_per_overlay : int;
    max_overlay_edges_total : int;
    max_diagnostics : int;
    max_diagnostic_bytes : int;
    max_session_bytes : int;
        (** [int], not [int64]: [Jsont_bytesrw] writes through a
            [Bytesrw.Bytes.Writer.limit], whose bound is an [int] and therefore
            32-bit under jsoo. [Hard] caps it to a jsoo-safe value and
            enforcement uses the writer's own limit/error boundary rather than
            measuring an allocated string. *)
    max_trace_entries : int;
    max_audit_reports : int;
    max_detail_nodes : int;
    max_detail_graphs : int;
    max_detail_bytes : int;
    response_live_bytes : int64;
        (** DERIVED by {!create} from [max_session_bytes] and
            [max_detail_bytes]. Not a parameter — the field exists so the
            derivation has somewhere to land. *)
  }

  val create :
    ?max_json_bytes:int64 ->
    ?max_pt2_bytes:int64 ->
    ?zip:Pt2_zip.Limits.t ->
    ?max_nodes_per_graph:int ->
    ?max_edges_per_graph:int ->
    ?max_groups_per_graph:int ->
    ?max_attrs_per_node:int ->
    ?max_metadata_items_per_node:int ->
    ?max_outputs_metadata_per_node:int ->
    ?max_namespace_depth:int ->
    ?max_namespace_component_bytes:int ->
    ?max_label_bytes:int ->
    ?max_id_bytes:int ->
    ?max_attr_chars:int ->
    ?max_url_bytes:int ->
    ?max_graphs:int ->
    ?max_total_nodes:int64 ->
    ?max_total_edges:int64 ->
    ?max_views:int ->
    ?max_comparisons:int ->
    ?max_node_data_sets:int ->
    ?max_states:int ->
    ?max_transitions:int ->
    ?max_mapping_entries_per_comparison:int ->
    ?max_mapping_members_per_entry:int ->
    ?max_mapping_members_total:int ->
    ?max_node_data_results_per_graph:int ->
    ?max_overlay_edges_per_overlay:int ->
    ?max_overlay_edges_total:int ->
    ?max_diagnostics:int ->
    ?max_diagnostic_bytes:int ->
    ?max_session_bytes:int ->
    ?max_trace_entries:int ->
    ?max_audit_reports:int ->
    ?max_detail_nodes:int ->
    ?max_detail_graphs:int ->
    ?max_detail_bytes:int ->
    t ->
    (t, [> error ]) Err.t
  (** Override fields of a base profile. Every field — including every nested
      [zip] field — is checked against its [Hard] ceiling, then
      [response_live_bytes] is derived and checked against
      [Hard.jsoo_safe_bytes]. A field may be any positive value at or below its
      ceiling: [create] does not require the result to be tighter than its BASE,
      because "tighter than [untrusted]" is a wire property and belongs to
      {!Wire_limits}, not to every programmatic caller. *)

  val within_hard_response : t -> (unit, [> error ]) Err.t
  (** The three relations a profile must satisfy before the browser may read a
      document produced under it: [max_session_bytes] and [max_detail_bytes] at
      or below [Hard.max_response_document_bytes], and the derived
      [response_live_bytes] at or below [Hard.max_response_live_bytes].
      Otherwise the browser would refuse a document its own checked producer is
      permitted to emit.

      The first two are load-bearing and both can fail. The third is implied by
      them — the peak is monotone in the document ceilings and
      [Hard.max_response_live_bytes] is that peak at
      [Hard.max_response_document_bytes] — and is kept as the direct statement
      of what the other two are a means to, becoming live only if a phase gains
      a term the document ceilings do not drive.

      Deliberately NOT part of {!create}: a profile is a valid profile without
      these — {!trusted} is one, and satisfies neither of the first two — and
      folding them in would remove the only profiles able to show the check is
      live. {!untrusted} and {!small} are checked against it where they are
      constructed, so enforcement happens before input acquisition. *)

  val untrusted : t
  (** The default for EVERY file-shaped input, CLI included, and the ceiling the
      wire is measured against. Provisional and conservative; the release
      profile is calibrated after Stages 2–4. *)

  val trusted : t
  (** The widest profile there is: every field at its [Hard] ceiling except the
      two document sizes, which sit at half of theirs because the phase sums at
      the full ceiling exceed [Hard.jsoo_safe_bytes] and an all-at-[Hard]
      profile is therefore not constructible. Internal/programmatic callers
      holding data they produced; never reachable from a file the user chose,
      and never one the browser may read a document under. *)

  val small : t
  (** Fieldwise no looser than {!untrusted}, and wire-selectable. *)

  val large : t
  (** Between {!untrusted} and [Hard]. Programmatic-only: {!Wire_limits} rejects
      it, which is what makes [--limits untrusted|small] a guarantee rather than
      a UI convention. *)
end

(** {1 The wire subset} *)

module Wire_limits : sig
  type t = private Limits.t
  (** A profile that may cross the worker boundary. [Limits.untrusted] is the
      ceiling — not [Hard]: a file input is untrusted and a caller may tighten
      [untrusted], never weaken it, and a value below [Hard] can be far looser
      than [untrusted]. Since the request IS the wire, this cannot be delegated
      to the page that built it, and a type nobody can inhabit except through
      the check makes every constructible request round-trip. *)

  val of_limits : ceiling:Limits.t -> Limits.t -> (t, [> error ]) Err.t
  (** Fieldwise against [ceiling] — every scalar AND every nested
      [Pt2_zip.Limits.t] field. [response_live_bytes] is not compared: it is
      derived, and a derived field that is already monotone in its two inputs is
      implied by them. *)

  val limits : t -> Limits.t

  val jsont : t Jsont.t
  (** The wire form: a FLAT object of the fields this profile TIGHTENS
      [Limits.untrusted] by, and nothing else. A wire profile is by construction
      no looser than [untrusted], so a field equal to it carries no information;
      [untrusted] therefore encodes as [{}], and two equal profiles encode to
      equal bytes — the determinism the session's claim quantifies over.

      Keys are the field's own DIAGNOSTIC name, dotted for a nested one
      ([zip.max_entries]), so the member a rejection names and the member that
      carried it are literally the same string rather than two tables that agree
      today.

      Values are JSON strings through [Jsont.int64_as_string]: two fields are
      [int64] with values past 2^31, and a JSON number would be read back
      through a 32-bit [int] under js_of_ocaml. Each is bounded against the
      ceiling BEFORE it is narrowed to the field's width, which is what makes
      the bound a bound (CLAUDE.md); decoding then finishes through
      {!of_limits}, so every decodable value is one the encoder could have
      produced. An unknown member is rejected by name rather than ignored — a
      profile the worker silently did not apply is the disagreement this type
      exists to prevent. *)
end

(** {1 Diagnostics} *)

module Diagnostic : sig
  (** The one type that crosses EVERY boundary in this design — session, worker
      response, delta — and therefore the one place an unbounded exception
      string could enter the wire. So the code vocabulary is closed (a JS
      [switch], not a string match) and the message is bounded by construction.
  *)

  module Code : sig
    type t =
      | Over_limit
      | Malformed_request
      | Invalid_limits
      | Invalid_source
      | Malformed_response
          (** DIRECTIONAL, and that is the point of having two:
              [Malformed_request] describes bytes the WORKER received,
              [Malformed_response] bytes the COORDINATOR received. Reporting
              both as one, inside a closed vocabulary meant to be switched on,
              turns a worker defect and a page defect into the same telemetry.
          *)
      | Request_in_flight
          (** Separated from [Malformed_request] for the same reason: a second
              concurrent submission is well-formed and refused by a coordinator
              invariant, not malformed. *)
      | Inconsistent_mount
          (** The mount could not be returned to a known state after a failed
              visible replacement. Distinct from [Internal] because it is the
              one condition in which the page and the bridge may disagree and
              neither can repair it: it requires an explicit disposal. A JS
              [switch] that could not name this case would report a
              recoverable-looking failure for the one that is not. *)
      | Settlement_mismatch
          (** The token names the most recent settlement and the caller asked
              for the other one. Not [Malformed_response]: the token is genuine
              and the caller is not confused about which transaction it means,
              only about how it ended. *)
      | Buffer_mismatch
      | Not_an_array_buffer
      | Stale_epoch
      | Unsupported_detail_key
      | Key_disagrees_with_ids
      | Unsupported_operator
      | Unsupported_input
      | Unsupported_graph_shape
      | Outside_dialect_domain
      | Requires_payloads
      | Prerequisite_unavailable
      | Not_implemented
      | Internal

    val to_string : t -> string
    (** The wire tag. *)

    val of_string : string -> t option
    (** Total in its failure, so an unknown tag from a future producer decodes
        as a named error rather than being silently dropped. *)

    val all : t list
    (** The decoder's vocabulary, and its own test. Built by walking a successor
        chain rather than written out beside the type, so a constructor cannot
        be added without reaching this list: a hand-written list would compile
        while {!of_string} quietly stopped recognising the new tag. *)

    val jsont : t Jsont.t
  end

  type t = private {
    code : Code.t;
    message : string;
    graph : string option;
    truncated : bool;  (** so "bounded" never reads as "complete" *)
  }

  val create : limits:Limits.t -> ?graph:string -> Code.t -> string -> t
  (** TOTAL — no result, so there is no path by which a diagnostic about a
      rejection becomes itself a rejection. Both variable-length fields are
      bounded here, which is what the response envelope's byte argument rests
      on:

      - [message] is sanitised then truncated to [max_diagnostic_bytes], setting
        [truncated]. Sanitisation is REPLACEMENT, not only truncation: {!of_exn}
        admits arbitrary OCaml bytes, which need not be valid UTF-8 at all, and
        [Jsont_bytesrw]'s encoder does not validate them — it emits them,
        producing a document that is not JSON. Nothing in OCaml then catches it,
        and the failure surfaces in the browser as a malformed response: the
        very condition this diagnostic existed to report. So invalid sequences
        and control bytes become U+FFFD before the length rule applies, and the
        length rule then cuts on a scalar-value boundary.
      - [graph] is checked against [max_id_bytes] and DROPPED — not truncated —
        when it does not fit, or when it is not clean UTF-8. Truncating an
        identifier yields a different identifier, which either resolves to the
        wrong graph or fails to resolve while looking authoritative; a missing
        graph is honestly missing. The drop sets [truncated] too, so it is
        visible. *)

  val of_exn : limits:Limits.t -> exn -> t
  (** The ONE bridge from an escaping exception, and deliberately lossy:
      [Code.Internal] plus [Printexc.to_string] through {!create}'s sanitiser.
      Nothing else may put an exception's text on the wire. *)

  val jsont : t Jsont.t
  (** Decodes through the same constructor as the producer, under HARD limits
      only: a [Jsont.t] carries no profile, so the profile-level bound is
      applied by whoever holds one. *)

  val pp : Format.formatter -> t -> unit
end
