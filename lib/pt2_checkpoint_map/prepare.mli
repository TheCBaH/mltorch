(** Turn a validated map and the bytes of its sources into the final value of
    every capture, each proved to be the one the map pinned.

    The host supplies sources as bigstrings it owns -- a memory map, a
    downloaded buffer, an in-memory copy; this module reads no file. Order of
    work: every source is checked against its pin (size, then SHA-256, then a
    safetensors header), and only then is any capture produced. A capture is
    produced from its origin, checked for dtype, shape and byte length, and
    hashed; the {!t} exists only if {e every} capture passed, unused ones
    included.

    Checkpoint values that need no conversion are views into the source, so the
    source bytes must stay alive and unmodified for as long as the result is in
    use; the views hold the bigstring, so dropping the host's own reference is
    safe. Converted and generated values are owned copies. *)

type source = { bytes : Safetensors.Bigstring.t; name : string }

type verified
(** Sources whose size and digest match their pins and whose headers are valid.
    Only {!verify_sources} makes one. *)

type hasher = Safetensors.Bigstring.t -> Pt2_sha256.Digest.t

val verify_sources :
  ?limits:Limits.t ->
  ?hash:hasher ->
  Document.t ->
  source list ->
  (verified, [> Fault.error ]) Err.t
(** The supplied names must be exactly the document's declared files (checkpoint
    files and the graph-owned file, if any). [hash] defaults to
    {!Pt2_sha256.bigstring}; a host may pass a faster one -- it is trusted to
    hash the bytes it is given, since the digest it returns is only compared. *)

type t
(** Every capture's final value. *)

val capture_set :
  ?limits:Limits.t ->
  ?hash:hasher ->
  Document.t ->
  verified ->
  (t, [> Fault.error ]) Err.t
(** Produce and check every capture, in target order; the first failure is
    reported. The sum of the sources and the buffers allocated stays within
    [limits.max_prepared_bytes], checked before each allocation. *)

val find : t -> string -> Pt2_storage.t option
(** The same storage on every call: nothing is fetched, converted, generated or
    hashed again. *)

val load : t -> string -> (Pt2_storage.t, string) result
(** {!find} in the shape [Pt2_archive.of_parts] takes for its [load]. *)

val targets : t -> string list
(** Sorted. *)

val owned_bytes : t -> int64
(** Bytes this module allocated (conversions, generated and inline values). *)

val source_bytes : t -> int64
(** Bytes of all sources, mapped or not. *)
