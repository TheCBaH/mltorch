(** A content-addressed store of verified files. A blob lives at a path derived
    from its SHA-256 alone -- never from a name a producer chose -- and is
    promoted into place by an atomic rename only after its size and digest have
    been checked, so a reader never sees a partial or unverified file. *)

module Pin = Pt2_checkpoint_map.Document.Pin

type t

type file_hasher = string -> (Pt2_sha256.Digest.t, string) result
(** Hash a file by path. The default maps it and uses {!Pt2_sha256}; a host may
    supply a faster one (a subprocess), whose answer is only compared. *)

val default_hasher : file_hasher

val create : string -> (t, [> Fault.error ]) Err.t
(** The cache root, created with [blobs/], [bundles/] and [tmp/]. *)

val mkdir_p : string -> unit
(** [mkdir -p]. Raises [Unix.Unix_error] on failure. *)

val root : t -> string
val blob_path : t -> Pt2_sha256.Digest.t -> string

val bundle_path : t -> Pt2_sha256.Digest.t -> string
(** Where the extracted archive with that digest lives. *)

val temp_dir : t -> string

type lookup = Absent | Present of string

val lookup :
  ?hash:file_hasher ->
  layer:Pt2_fixture.Fault.layer ->
  t ->
  Pin.t ->
  (lookup, [> Fault.error ]) Err.t
(** [Present path] only after the file's size and digest match the pin. A file
    that is there and wrong is an error, not [Absent]: the caller decides
    whether to replace it. *)

val promote :
  ?hash:file_hasher ->
  layer:Pt2_fixture.Fault.layer ->
  t ->
  tmp:string ->
  Pin.t ->
  (string, [> Fault.error ]) Err.t
(** Verify [tmp] against the pin and rename it to its blob path, replacing a bad
    blob if there is one. [tmp] is removed on failure. *)
