(* Typed byte quantities from test literals: a literal a test means to be
   valid, so a refusal is a broken test, raised. *)

open Core.Storage_units

let ok r = Err.or_raise ~pp_error r
let size v = ok (Byte_size.of_int64 v)
let offset v = ok (Byte_offset.of_int64 v)
let alignment v = ok (Byte_alignment.of_int64 v)
let one = alignment 1L
