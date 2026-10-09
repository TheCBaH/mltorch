type t = BF16 | BOOL | F16 | F32 | F64 | I16 | I32 | I64 | I8 | U8

let equal (a : t) (b : t) = a = b
let all = [ BF16; BOOL; F16; F32; F64; I16; I32; I64; I8; U8 ]

let to_code = function
  | BF16 -> "BF16"
  | BOOL -> "BOOL"
  | F16 -> "F16"
  | F32 -> "F32"
  | F64 -> "F64"
  | I16 -> "I16"
  | I32 -> "I32"
  | I64 -> "I64"
  | I8 -> "I8"
  | U8 -> "U8"

let of_code code = List.find_opt (fun d -> String.equal (to_code d) code) all

let of_torch_name = function
  | "bfloat16" -> Some BF16
  | "bool" -> Some BOOL
  | "float16" -> Some F16
  | "float32" -> Some F32
  | "float64" -> Some F64
  | "int16" -> Some I16
  | "int32" -> Some I32
  | "int64" -> Some I64
  | "int8" -> Some I8
  | "uint8" -> Some U8
  | _ -> None

let of_pt2 : Pt2_dtype.t -> t = function
  | Bool -> BOOL
  | Float32 -> F32
  | Float64 -> F64
  | Int16 -> I16
  | Int32 -> I32
  | Int64 -> I64
  | Int8 -> I8
  | UInt8 -> U8

let byte_width = function
  | BOOL | I8 | U8 -> 1
  | BF16 | F16 | I16 -> 2
  | F32 | I32 -> 4
  | F64 | I64 -> 8

let pp ppf d = Fmt.string ppf (to_code d)
