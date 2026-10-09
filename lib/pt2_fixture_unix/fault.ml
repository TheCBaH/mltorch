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

type limit = Archive_bytes | Member_bytes | Member_count

module Tar_fault = struct
  type t = { kind : tar; name : string }
end

module Limit_fault = struct
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

let pp_tar ppf k =
  Fmt.string ppf
    (match k with
    | Absolute_name -> "absolute member name"
    | Bad_checksum -> "header checksum mismatch"
    | Bad_size -> "unparsable member size"
    | Duplicate_name -> "duplicate member"
    | Link_member -> "link member"
    | Name_too_long -> "member name too long"
    | Truncated -> "truncated archive"
    | Unsafe_name -> "unsafe member name"
    | Unsupported_member -> "unsupported member type")

let pp_limit ppf l =
  Fmt.string ppf
    (match l with
    | Archive_bytes -> "decompressed archive size"
    | Member_bytes -> "member size"
    | Member_count -> "member count")

let pp_error ppf : error -> unit = function
  | `Gzip m -> Fmt.pf ppf "archive is not valid gzip: %s" m
  | `File_io (path, m) -> Fmt.pf ppf "%S: %s" path m
  | `Limit { Limit_fault.what; limit; actual } ->
      Fmt.pf ppf "%a is %Ld, over the limit %Ld" pp_limit what actual limit
  | `Offline name ->
      Fmt.pf ppf "%S is not in the cache and no transport is configured" name
  | `Tar { Tar_fault.kind; name } -> Fmt.pf ppf "%a: %S" pp_tar kind name
  | `Transport (url, m) -> Fmt.pf ppf "download of %s failed: %s" url m
  | #Pt2_fixture.Fault.error as e -> Pt2_fixture.Fault.pp_error ppf e
