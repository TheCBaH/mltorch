(** Host-side failures: files, transport, archive framing. The layered pin
    faults are {!Pt2_fixture.Fault}. *)

type tar =
  | Absolute_name
  | Bad_checksum
  | Bad_size
  | Duplicate_name
  | Link_member
  | Name_too_long
  | Truncated
  | Unsafe_name
  | Unsupported_member
      (** Alphabetical. What is wrong with one member header. *)

type limit =
  | Archive_bytes
  | Member_bytes
  | Member_count
      (** A ceiling the reader enforces before extracting or allocating. *)

module Tar_fault : sig
  type t = { kind : tar; name : string }
end

module Limit_fault : sig
  type t = { actual : int64; limit : int64; what : limit }
end

type error =
  [ `Gzip of string
  | `File_io of string * string
  | `Limit of Limit_fault.t
  | `Offline of string
  | `Tar of Tar_fault.t
  | `Transport of string * string
  | Pt2_fixture.Fault.error ]

val pp_error : error Fmt.t
