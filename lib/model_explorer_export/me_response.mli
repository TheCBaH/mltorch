(* What the worker sends back, and the one seam it crosses on.

   THE DOCUMENT LIVES INSIDE THE CONSTRUCTOR THAT HAS ONE. Returning a
   [(t * string)] pair and letting the shell send any constructor makes every
   combination the envelope forbids constructible — a session with no document,
   a failure carrying one, a byte count disagreeing with the bytes beside it —
   and a JS test that decodes only the metadata cannot see any of it. There is
   no [bytes] field on a payload here at all: it is DERIVED in {!Wire} from the
   document it describes, so the two cannot disagree.

   THREE TYPES, NARROWING BY ONE CONSTRUCTOR EACH TIME, and the narrowing is the
   design. {!Handle_result} is what a function holding a validated request may
   return; {!Final} adds the pre-decode failure only the shell can produce; and
   {!Meta} is what actually crosses to JavaScript, because the final
   constructors carry a document the metadata must not contain and the metadata
   carries a byte count they do not have.

   See .ai/model_explorer_design.md. *)

module Phase : sig
  type t = Decode | Lower | Project | Encode

  val to_string : t -> string
  val of_string : string -> t option

  val all : t list
  (** In pipeline order, off a successor chain. *)
end

module Progress : sig
  type t = {
    id : Me_request.Request_id.t;
    phase : Phase.t;
    done_ : int64;
    total : int64 option;
  }
  (** STREAMED, and documentless by construction: {!Wire.of_progress} takes this
      type alone, so a progress message carrying a payload is unwritable. *)
end

module Session : sig
  type t = {
    id : Me_request.Request_id.t;
    limits : Me_limits.Wire_limits.t;
        (** the DECODED, validated profile — where the coordinator's
            authoritative one comes from *)
    json : string;
  }
end

module Delta : sig
  type t = {
    id : Me_request.Request_id.t;
    key : Me_request.Detail_key.t;
    json : string;
  }
end

module Failed : sig
  type t = {
    id : Me_request.Request_id.t;
    key : Me_request.Detail_key.t option;
    error : Me_limits.Diagnostic.t;
  }
  (** STRUCTURED: an over-limit detail has no delta to carry diagnostics, so the
      failure names its key and a typed diagnostic instead. Reached only once a
      VALID request exists, which is the condition under which naming an epoch
      and a key is meaningful at all. *)
end

module Protocol_failure : sig
  type t = { error : Me_limits.Diagnostic.t }
  (** The PRE-DECODE failure. Every check the shell performs before the request
      exists — request length, the request codec, the profile, the source, the
      buffer type and length — can fail with no validated id in hand, and a
      {!Failed} could then be built only by echoing untrusted fields. This
      constructor requires none, which is exactly why it exists. *)
end

module Handle_result : sig
  type t =
    | Session of Session.t
    | Delta of Delta.t
    | Failed of Failed.t
        (** What a handler may return: narrower than {!Final.t} by one
            constructor, and that constructor is the point. A protocol failure
            means the request was never understood, so a function that received
            a validated request cannot honestly produce one. Stating that as a
            type rather than as a rule in prose is what stops it being
            contradicted. *)
end

module Final : sig
  type t =
    | Session of Session.t
    | Delta of Delta.t
    | Failed of Failed.t
    | Protocol_failure of Protocol_failure.t
        (** Exactly one of these ends a request from the SHELL's point of view.
            {!Progress} is deliberately absent: it does not end anything. *)

  val of_handle_result : Handle_result.t -> t
end

module Meta : sig
  (** THE TYPE THAT CROSSES. Not {!Final.t}: its constructors carry a document
      the metadata must not contain, and the metadata carries a byte count they
      do not have. PRIVATE, whose only producer is {!Wire}, which is what keeps
      "a byte count that cannot disagree with a document" true now that the
      invariant lives on the type that is serialized. *)

  module Session : sig
    type t = private {
      id : Me_request.Request_id.t;
      limits : Me_limits.Wire_limits.t;
      bytes : int;
          (** [int], not [int64], and the width rule does not bite: it is the
              length of an OCaml string that already exists, bounded by
              [max_session_bytes] — itself an [int] because the writer limit is
              one. Every other integer here is [int64] through
              [Jsont.int64_as_string]. *)
    }
  end

  module Delta : sig
    type t = private {
      id : Me_request.Request_id.t;
      key : Me_request.Detail_key.t;
      bytes : int;
    }
  end

  type t = private
    | Progress of Progress.t
    | Session of Session.t
    | Delta of Delta.t
    | Failed of Failed.t
    | Protocol_failure of Protocol_failure.t
        (** {!Progress}, {!Failed} and {!Protocol_failure} have no document, so
            their final payloads ARE their metadata and are reused unchanged. *)

  val kind : t -> string
  val jsont : t Jsont.t

  val decode : string -> (t, string) result
  (** PRODUCTION, not a test helper: the coordinator routes on a small
      [JSON.parse] of ["kind"], and every authoritative read of metadata comes
      through here. *)
end

module Wire : sig
  type t = private { meta : string; payload : string option }
  (** The protocol seam, and the only one. [meta] and [payload] are produced
      together, so payload presence follows from the constructor rather than
      from a convention the shell is trusted to follow. *)

  type error = [ `Meta_too_large ]

  val pp_error : Format.formatter -> [< error ] -> unit
  val of_progress : Progress.t -> (t, [> error ]) Err.t

  val of_final : Final.t -> (t, [> error ]) Err.t
  (** Builds the corresponding {!Meta.t} — [payload = Some json] and
      [bytes = String.length json] for a session and a delta, [None] and no byte
      count for the other three — then encodes it through a WRITER-LIMITED
      encoder at [Hard.max_response_meta_bytes]. An aggregate bound, not an
      inference from individually-bounded fields: JSON escaping expands, and
      three bounded fields still sum.

      RESULT-VALUED because that writer can fail. A writer limit is a property
      of the writer, and a function whose implementation can fail should say so
      regardless of which inputs reach it today; the suite reaches the branch
      through an injectable ceiling. *)

  val constant_protocol_failure : t
  (** The terminal response, ALREADY ENCODED. The fallback path does not run the
      encoder at all, so "the fallback cannot itself fail" is a property of a
      value rather than an argument about a function — and the suite asserts the
      ceiling over THESE bytes, the ones that path actually posts. *)

  val of_final_bounded : max_meta_bytes:int -> Final.t -> (t, [> error ]) Err.t
  (** {!of_final} with the ceiling injected. Exposed only so the limit branch
      can be reached: a bound nobody can drive is a bound taken on trust. *)
end
