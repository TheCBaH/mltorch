(* The four scalar value types of the core language. Closed and alphabetical;
   the binary encoding is the spec's, not the order. *)
type t = F32 | F64 | I32 | I64 | V128

let to_byte = function
  | F32 -> 0x7D
  | F64 -> 0x7C
  | I32 -> 0x7F
  | I64 -> 0x7E
  | V128 -> 0x7B

let name = function
  | F32 -> "f32"
  | F64 -> "f64"
  | I32 -> "i32"
  | I64 -> "i64"
  | V128 -> "v128"

let equal (a : t) b = a = b
let pp ppf t = Fmt.string ppf (name t)
