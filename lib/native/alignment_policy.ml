(* See alignment_policy.mli. *)

open Core.Storage_units

(* The policy's only literals, both powers of two: a refusal is a broken
   build, raised once at initialization. *)
let literal v = Err.or_raise ~pp_error (Byte_alignment.of_int64 v)
let cache_line_alignment = literal 64L
let page_alignment = literal 4096L
let page_size = Byte_alignment.to_size page_alignment

let default size ~payload_min =
  Byte_alignment.max payload_min
    (if Byte_size.compare size page_size <= 0 then cache_line_alignment
     else page_alignment)

type t = { host : Byte_alignment.t option }

let standard = { host = None }
let with_host a = { host = Some a }
let host t = t.host

let alignment t size ~payload_min =
  let d = default size ~payload_min in
  Option.fold ~none:d ~some:(Byte_alignment.max d) t.host

let equal a b = Option.equal Byte_alignment.equal a.host b.host

let pp ppf t =
  match t.host with
  | None -> Format.pp_print_string ppf "standard"
  | Some a -> Format.fprintf ppf "standard, host %a" Byte_alignment.pp a
