(* The worker request: self-contained, epoch-bound, and closed by construction.

   Self-contained means the request carries everything the worker enforces —
   including the {b profile}. Leaving the worker to pick its own default would
   let a page build a session under one profile and a worker validate it under
   another, which is the disagreement [Me_limits.Wire_limits] exists to prevent;
   so [limits] is a field, and [handle] has no fallback.

   Closed by construction means every textual field goes through a checked
   constructor and the record types are [private]. A checked DECODER does not
   make a public encoder safe: with a public record the encoder can emit JSON
   its own decoder rejects, and the round-trip property then holds only of
   whichever values a fixture happened to pick.

   PROVENANCE ONLY — never the model bytes. Those travel beside this as a
   transferable [ArrayBuffer], because a pure OCaml type cannot hold a
   js_of_ocaml value and putting the bytes in the JSON would defeat the
   transfer. What is here is exactly the set the session's determinism claim
   quantifies over: two loads agreeing on [Source.t] and [Options.t] produce
   byte-identical session JSON.

   See .ai/model_explorer_design.md. *)

(** {1 Request identity} *)

module Request_id : sig
  type t = private string
  (** [<uuid>-<seq>]. *)

  val of_string : string -> (t, [> `Malformed_request_id ]) Core.result
  (** A GRAMMAR, not a length. 36 characters of canonical lowercase hyphenated
      UUID, a ['-'], then 1–10 decimal digits with no leading zero (bare ["0"]
      allowed) whose value is below [Me_limits.Hard.max_seq_exclusive].

      Exact rather than merely bounded, because the builder is the sole producer
      and an exact check costs one fixed-width parse — which makes [Hard]'s two
      byte constants consequences of this grammar rather than a second rule that
      can drift from it. A length ceiling alone admits [9999999999], a sequence
      no producer bounded by 2^32 can issue, so "consumer and producer accept
      the same strings" would be false in the direction that matters.

      The sequence is compared through [Int64.of_string] and NEVER through
      [int]: an [int] parse wraps under js_of_ocaml before the comparison, which
      is the defect the ceiling exists to prevent. *)

  val epoch : t -> string
  (** The UUID half. There is no [epoch] field anywhere in this module — it is a
      projection of the id, so a message cannot name one epoch in its id and
      another beside it. *)

  val to_string : t -> string
  val equal : t -> t -> bool
  val jsont : t Jsont.t
end

(** {1 What a detail request names} *)

module Detail_key : sig
  type t = private { parent_graph : string; value : Graph_ir.Tensor_id.t }
  (** ONE identity, not two. A Kernel value has exactly one tensor id and one
      body, so carrying [parent_node] alongside would admit a mismatched pair
      that validation would then have to reject; here it is DERIVED. *)

  type invalid = [ `Parent_too_long | `Derived_id_too_long ]

  val pp_invalid : Format.formatter -> [< invalid ] -> unit

  val create :
    limits:Me_limits.Limits.t ->
    parent_graph:string ->
    value:Graph_ir.Tensor_id.t ->
    (t, [> `Invalid_detail_key of invalid ]) Core.result
  (** [parent_graph] against [max_id_bytes] AND the derived {!id} against it,
      since a component can expand threefold under encoding — checking only the
      input would approve a key whose rendered id is over.

      Distinct from an unsupported key, which is a well-formed key naming no
      value: one is a malformed request and the other a valid request about
      something absent, and they carry different diagnostic codes. *)

  val validate :
    limits:Me_limits.Limits.t ->
    t ->
    (unit, [> `Invalid_detail_key of invalid ]) Core.result
  (** The profile-dependent half, separated from construction because a [t]
      carries no witness of the profile it was built under — a key valid under
      [untrusted] can be invalid in a [small] request, and only the enclosing
      request knows which profile applies. *)

  val parent_node : t -> string
  (** ["t<k>"], the Kernel value's node. *)

  val id : t -> string
  (** The derived detail graph/view id,
      [expr/<parent graph>/<parent node>/t<k>]. Its last two components are the
      SAME tensor id, and visibly so: that is what "one identity, not two" looks
      like once the parent stopped being a field. The parents are ids already
      produced elsewhere and are not re-encoded — encoding an encoded id is not
      idempotent. *)

  val equal : t -> t -> bool

  val jsont : t Jsont.t
  (** Enforces the FIXED ceilings only. A parameterless codec cannot silently
      pick a profile, and picking the wrong one is exactly the class of defect
      [Wire_limits] exists to prevent; the profile check is {!validate}, run by
      {!Request.jsont} once the profile has been decoded. *)
end

(** {1 Where the bytes came from} *)

module Source : sig
  module Origin : sig
    module Catalog : sig
      type t = { url : string; ref_ : string; verified_sha256 : string }
    end

    type t =
      | Local
      | Catalog of Catalog.t
          (** The digest lives INSIDE [Catalog]. A sibling
              [verified_sha256 option] would permit a [Local] source claiming a
              verified digest — a false verification claim entering the
              deterministic session — and a [Catalog] source with none. Neither
              is constructible, so the session's digest rule needs no validation
              at all. *)
  end

  type t = private {
    origin : Origin.t;
    name : string;
    bytes : int64;
    format : [ `Model_json | `Pt2 ];  (** declared; still content-validated *)
  }

  module Invalid : sig
    type kind =
      | Negative_bytes
      | Bad_digest
      | Name_too_long
      | Url_too_long
      | Ref_too_long

    type t = { kind : kind }
  end

  val pp_invalid : Format.formatter -> Invalid.t -> unit

  val create :
    limits:Me_limits.Limits.t ->
    origin:Origin.t ->
    name:string ->
    bytes:int64 ->
    format:[ `Model_json | `Pt2 ] ->
    (t, [> `Invalid_source of Invalid.t ]) Core.result
  (** [bytes >= 0]; the digest as exactly 64 lowercase hex bytes; [name] against
      [max_label_bytes] and [url]/[ref_] against [max_url_bytes].

      It does NOT check the buffer length: only the shell owns the buffer, so
      that equality stays there. *)

  val verified_sha256 : t -> string option
  (** [Some] exactly when the origin is [Catalog]. *)

  val jsont : t Jsont.t
end

(** {1 What changes the projection} *)

module Options : sig
  type namespace = Structural | Module

  type t = private {
    stages : Me_session.Capability.graph_stage list;
    fold : bool;
    verify_symbolic : Map_verify.Effort.t option;
    namespace : namespace;
  }

  val create :
    stages:Me_session.Capability.graph_stage list ->
    fold:bool ->
    verify_symbolic:Map_verify.Effort.t option ->
    namespace:namespace ->
    (t, [> `Invalid_options ]) Core.result
  (** [stages] is NORMALISED, not merely validated: duplicates removed and the
      result in [Capability.all_stages] order. So the list is bounded by that
      type's cardinality by construction — no separate ceiling — and two
      requests differing only in the order a user typed them are the SAME
      request, which is what makes "same options implies identical JSON" a
      statement about a canonical value rather than about a spelling.

      An empty list is [`Invalid_options]: a request asking for nothing is a
      caller defect, not an empty session. *)

  val jsont : t Jsont.t
end

(** {1 The request} *)

module Request : sig
  module Build_session : sig
    type t = private {
      id : Request_id.t;
      source : Source.t;
      options : Options.t;
      limits : Me_limits.Wire_limits.t;
    }
  end

  module Build_detail : sig
    type t = private {
      id : Request_id.t;
      source : Source.t;
      options : Options.t;
      limits : Me_limits.Wire_limits.t;
      key : Detail_key.t;
    }
  end

  type t = private
    | Build_session of Build_session.t
    | Build_detail of Build_detail.t

  type error =
    [ `Malformed_request_id
    | `Invalid_options
    | `Invalid_source of Source.Invalid.t
    | `Invalid_limits of Me_limits.error
    | `Invalid_detail_key of Detail_key.invalid ]

  val pp_error : Format.formatter -> [< error ] -> unit

  val build_session :
    id:Request_id.t ->
    source:Source.t ->
    options:Options.t ->
    limits:Me_limits.Wire_limits.t ->
    (t, [> error ]) Core.result

  val build_detail :
    id:Request_id.t ->
    source:Source.t ->
    options:Options.t ->
    limits:Me_limits.Wire_limits.t ->
    key:Detail_key.t ->
    (t, [> error ]) Core.result
  (** The ONLY producers, and what they do is REVALIDATE every profile-dependent
      component under this request's own profile — [Source]'s
      [name]/[url]/[ref_] and [Detail_key.validate] — not verify that the
      components were built under it. A [Source.t] keeps no witness of the
      profile it was checked against, so that property is unobtainable;
      revalidation is both obtainable and sufficient, since it is the request's
      profile the worker will enforce.

      So the [`Invalid_source] and [`Invalid_detail_key] rows are REACHABLE for
      a caller using the checked constructors, and that is the point: a source
      built under [untrusted] and combined with a [small] wire profile is valid
      alone and invalid here. *)

  val id : t -> Request_id.t
  val epoch : t -> string
  val key : t -> Detail_key.t option
  val limits : t -> Me_limits.Wire_limits.t
  val source : t -> Source.t
  val options : t -> Options.t

  val jsont : t Jsont.t
  (** Decoding is STAGED, not a flat object codec. Jsont decodes an object's
      fields independently, but [Source.create ~limits] needs the decoded limits
      — so this decodes an intermediate record of raw pieces, then validates in
      order: the id, the profile, the options, the source, the key. The last two
      take the profile the step before them produced, which is why the order is
      fixed and not an implementation detail.

      And it FINISHES THROUGH {!build_session}/{!build_detail} rather than
      assembling the record itself. That is what makes encoder and decoder one
      domain rather than two that happen to agree — the closure property has
      something to be true about instead of being tested on whichever values a
      fixture picked. *)
end
