(* Model bytes in, session document out — and the ONE implementation of that.

   This is the body the CLI used to hold. It moved here because the browser
   compiles this library rather than a second implementation of it, so a
   divergence between what the CLI exports and what the page renders is not
   expressible; a session builder living in [bin/] would have made it
   expressible again the moment the worker gained one.

   IN MEMORY THROUGHOUT. The worker receives a transferred [ArrayBuffer], not a
   path, so detection, archive opening and decoding all take a string. The CLI
   still rejects an oversized file from [stat] before reading it — that is a
   thing only a filesystem caller can do, and it belongs there.

   See .ai/model_explorer_design.md. *)

type source_format = Model_json | Pt2_archive

val detect : bytes:string -> (source_format, [> `Unrecognised_format ]) Err.t
(** CONTENT, never a declared extension: a mislabelled file is the ordinary case
    rather than the adversarial one, and the two loaders fail very differently
    on each other's input. [PK\x03\x04] is an archive, ['{'] a [model.json]. *)

type error =
  [ `Unrecognised_format
  | `Declared_format_disagrees
    (** the request declared a format the bytes are not. DECLARED is a hint; the
        content decides, and a disagreement is a fact about the source *)
  | `Too_large of int64  (** the byte count offered *)
  | `Document_too_large
    (** the ENCODED document overran [max_session_bytes]/[max_detail_bytes]. No
        byte count: the writer aborts mid-stream once the limit is hit, so
        unlike [`Too_large] there is no final length to report. *)
  | `Model_json_decode of string
    (** the model.json decoder's own message — a THIRD-PARTY payload, named for
        its source so it is not mistaken for a case this module declined to
        classify *)
  | `Archive of Pt2_archive.error
  | `Lowering of Native_interp.error
    (** a FATAL row only. A row [Me_classify.lowering] calls recoverable never
        reaches here: it becomes a session whose capabilities say what is
        missing. *)
  | `Native4d of Native4d.Error.t
    (** a FATAL row only, for the same reason as [`Lowering]: a graph outside
        the dialect's domain is a capability carrying a reason, not an error *)
  | `Kernel of Kernel_adapt.error
    (** a FATAL row only, again: an over-limit or unsupported-shape kernel is a
        capability carrying the reason *)
  | `Unsupported_detail_key
    (** a detail request naming a value this model does not produce — or one it
        could not lower to a kernel at all *)
  | `Value_graph of Me_kernel.error
  | `Fusion of Me_fusion.error
  | `Source_view of Me_source.error
  | `View of Graph_view.error
  | `Project of Me_build.error
  | `Navigation of Me_pt2.error
  | `Verification of Me_verify.error
  | `Identifier of Me_ids.error
  | `Document of Me_session.Session.error ]

val pp_error : Format.formatter -> [< error ] -> unit

val diagnostic_code : [< error ] -> Me_limits.Diagnostic.Code.t
(** Which closed code a failure crosses the wire as. Total, and it draws the
    same line [Me_classify] draws: a model that is not what it claimed is
    [Invalid_source], a ceiling is [Over_limit], and an invariant of ours is
    [Internal]. *)

(** {1 The session} *)

module Options : sig
  type t = {
    stages : Me_session.Capability.graph_stage list;
        (** {!session} only -- {!detail} reaches the kernel through its own
            smaller pipeline and does not read this field. A stage not listed
            here is [Not_requested] in the result, and its own computation (and
            that of anything ONLY it needs, e.g. the symbolic evaluation
            [Stage_program]/[Kernel] share) does not run -- unlike
            [Initial_native]/[Canonical], which are the backbone every
            comparison and every later stage builds on, cheap to project, and
            always present regardless of this list. *)
    fold : bool;
    verify_symbolic : Map_verify.Effort.t option;
    name : string;  (** the model's own name, which becomes the collection *)
    source_bytes : int64;
    source_sha256 : string option;
        (** Populated ONLY when a caller supplied an expected digest. A local
            file has nothing to verify against, and recording one we computed
            ourselves would assert a provenance nobody checked. *)
  }
end

val session :
  limits:Me_limits.Limits.t ->
  options:Options.t ->
  bytes:string ->
  (Me_session.Session.t, [> error ]) Err.t
(** Detect, decode, lower, transform, project, validate.

    A model this repository cannot LOWER is not an error here: the exported
    program decoded, which is the whole of what [Graph_stage Source] claims, so
    the result is a successful session whose capabilities say what is missing
    and why. Only a [Me_classify.Fatal] row fails. *)

val detail :
  limits:Me_limits.Limits.t ->
  options:Options.t ->
  key:Me_request.Detail_key.t ->
  bytes:string ->
  (Me_detail.Delta.t, [> error ]) Err.t
(** One value's expression, as a delta.

    A SMALLER pipeline than {!session}'s: a detail needs the kernel and nothing
    else — no source view, no comparisons, no flow — so projecting all of that
    to reach one value would make the on-demand path pay for the eager one it
    exists to avoid.

    It re-lowers rather than caching, which is what a self-contained request
    means: the worker holds no session between requests, so there is nothing
    that could be stale. *)

val encode_bounded :
  max_bytes:int -> 'a Jsont.t -> 'a -> (string, [> error ]) Err.t
(** Encode through a writer that refuses to grow past [max_bytes], failing with
    [`Document_too_large] rather than returning a string already over the
    ceiling. {!session} and {!detail} return VALUES rather than JSON, on
    purpose, so this is the one place a caller can encode either without
    re-deriving the bound: {!handle} uses it for both, and so should any other
    caller building session or detail JSON, CLI included. *)

(** {1 The worker entry point} *)

val handle :
  emit:(Me_response.Progress.t -> unit) ->
  Me_request.Request.t ->
  bytes:string ->
  Me_response.Handle_result.t
(** TOTAL. Every failure a valid request can produce is already a constructor of
    the return type, so there is no error row left to widen and no second place
    where a failure could be turned into a message.

    It holds a validated request, so it always knows which request it is failing
    about — which is exactly the condition under which the answer is [Failed]
    rather than a protocol failure. {!Me_response.Handle_result} makes the other
    one unwritable here.

    The request carries no separate identity: [id] is a field of the request, so
    a response cannot name one request in its id and another beside it. *)
