(** Obtain a verified file: from the cache if it is there and right, else
    through the transport into a temporary file that is verified and promoted.
    Offline (no transport) never touches the network. *)

val ensure :
  ?hash:Cache.file_hasher ->
  ?transport:Transport.t ->
  layer:Pt2_fixture.Fault.layer ->
  Cache.t ->
  Cache.Pin.t ->
  (string, [> Fault.error ]) Err.t
(** The path of the verified blob. A cached file that fails its pin is replaced
    when a transport is available and is an error offline. *)

val read : ?max_bytes:int -> string -> (string, [> Fault.error ]) Err.t
(** A whole small file (a manifest, an index), refusing one over [max_bytes]
    (default 64 MiB) before reading it. *)
