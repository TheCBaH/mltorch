(** An extracted slim bundle on disk, and the chain that gets one: publication
    index, then manifest, then archive, each fetched into the cache by its pin
    and checked against the layer above. An archive is extracted only after its
    exact member set and every member's size and digest match the manifest, into
    a temporary directory that is verified once more and then renamed into
    [bundles/<archive digest>]; a bundle directory therefore either exists whole
    and verified or does not exist. *)

type config = {
  cache : Cache.t;
  hash : Cache.file_hasher option;
  limits : Targz.limits;
  transport : Transport.t option;  (** [None]: offline. *)
}

val config :
  ?hash:Cache.file_hasher ->
  ?limits:Targz.limits ->
  ?transport:Transport.t ->
  Cache.t ->
  config

type t = {
  dir : string;
  entry : Pt2_fixture.Cohort.entry;
  manifest : Pt2_fixture.Manifest.t;
}

val ensure :
  config ->
  Pt2_fixture.Cohort.t ->
  Pt2_fixture.Cohort.entry ->
  (t, [> Fault.error ]) Err.t

val verify_dir :
  ?hash:Cache.file_hasher ->
  dir:string ->
  Pt2_fixture.Manifest.t ->
  (unit, [> Fault.error ]) Err.t
(** The directory holds exactly the manifest's members -- no missing file, no
    extra file, no link or other non-regular entry -- each with the pinned size
    and digest. *)

val read_member :
  ?max_bytes:int -> t -> string -> (string, [> Fault.error ]) Err.t

val verify_inventory :
  ?hash:Cache.file_hasher ->
  dir:string ->
  Pt2_fixture.Manifest.member Schema_runtime.String_map.t ->
  (unit, [> Fault.error ]) Err.t
(** The same exact regular-file/digest checks without PT2-specific metadata. The
    root itself must be a directory, never a symbolic link. *)

val ensure_inventory :
  config ->
  Cache.Pin.t ->
  Pt2_fixture.Manifest.member Schema_runtime.String_map.t ->
  (string, [> Fault.error ]) Err.t
(** Acquire a pinned archive and atomically extract its independently trusted
    inventory, or reverify an existing extraction. Task fixtures share the
    bounded gzip/tar and cache lifecycle with checkpoint fixtures. *)
