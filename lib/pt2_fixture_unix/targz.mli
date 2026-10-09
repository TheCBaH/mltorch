(** A bounded reader for the producer's [.tar.gz] archives: gzip with a CRC and
    length trailer, then ustar members. It accepts regular files only and
    refuses, rather than interprets, anything else -- links, absolute or
    escaping names, duplicate names, extended headers. All limits are checked
    from headers before the data they describe is copied. *)

type limits = {
  max_archive_bytes : int;  (** Decompressed tar stream. *)
  max_member_bytes : int;
  max_members : int;
}

val default_limits : limits

type member = { data : string; name : string }

val members : ?limits:limits -> string -> (member list, [> Fault.error ]) Err.t
(** The regular files of a gzip-compressed tar, in archive order. *)

val safe_name : string -> bool
(** A member name that stays inside a directory: non-empty, at most 255 bytes,
    relative, with no empty, [.] or [..] segment, backslash or NUL. *)
