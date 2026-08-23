(* See float_bits.mli. *)

let exact = Int64.bits_of_float
let portable x = if Float.is_nan x then 0x7ff8000000000000L else exact x
let compare_exact a b = Int64.compare (exact a) (exact b)
let equal_exact a b = Int64.equal (exact a) (exact b)
let compare_portable a b = Int64.compare (portable a) (portable b)
let equal_portable a b = Int64.equal (portable a) (portable b)
