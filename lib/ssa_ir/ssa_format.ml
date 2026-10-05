(* A buffer's storage format and how a load decodes it to a working type.
   Decoding and encoding are explicit in the access, never implied by the
   buffer: a load names the working type it produces, a store the encode it
   applies. *)
type t = Bool | F32 | I64

let name = function Bool -> "bool" | F32 -> "f32" | I64 -> "i64"
