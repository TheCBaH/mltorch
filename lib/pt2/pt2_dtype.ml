(* Element dtype of a stored tensor, the subset that appears in the torchvision
   `.pt2` weights and `.pt` sample inputs. Standalone (no ATen dependency); the
   bridge in lib/pt2_aten maps these to [Aten_scalar_type.t] when handing a
   tensor to the op layer. *)

type t = Bool | Float32 | Float64 | Int16 | Int32 | Int64 | Int8 | UInt8

type error =
  [ `Unknown_storage_class of string
  | `Unsupported_scalar_type of Pytorch_types.ScalarType.t ]

let element_size = function
  | Bool -> 1
  | Float32 -> 4
  | Float64 -> 8
  | Int16 -> 2
  | Int32 -> 4
  | Int64 -> 8
  | Int8 -> 1
  | UInt8 -> 1

(* From the schema's [ScalarType] enum (as decoded from model.json / weight
   configs). Raises on a dtype this reader does not yet handle. *)
let scalar_type_name : Pytorch_types.ScalarType.t -> string = function
  | BFLOAT16 -> "BFLOAT16"
  | BOOL -> "BOOL"
  | BYTE -> "BYTE"
  | CHAR -> "CHAR"
  | COMPLEXDOUBLE -> "COMPLEXDOUBLE"
  | COMPLEXFLOAT -> "COMPLEXFLOAT"
  | COMPLEXHALF -> "COMPLEXHALF"
  | DOUBLE -> "DOUBLE"
  | FLOAT -> "FLOAT"
  | FLOAT8E4M3FN -> "FLOAT8E4M3FN"
  | FLOAT8E4M3FNUZ -> "FLOAT8E4M3FNUZ"
  | FLOAT8E5M2 -> "FLOAT8E5M2"
  | FLOAT8E5M2FNUZ -> "FLOAT8E5M2FNUZ"
  | FLOAT8E8M0FNU -> "FLOAT8E8M0FNU"
  | HALF -> "HALF"
  | INT -> "INT"
  | LONG -> "LONG"
  | SHORT -> "SHORT"
  | UINT16 -> "UINT16"
  | UINT32 -> "UINT32"
  | UINT64 -> "UINT64"
  | Pytorch_types.ScalarType.UNKNOWN -> "UNKNOWN"

let pp_error ppf : error -> unit = function
  | `Unknown_storage_class name -> Fmt.pf ppf "unknown storage class %S" name
  | `Unsupported_scalar_type st ->
      Fmt.pf ppf "unsupported ScalarType %s" (scalar_type_name st)

let of_scalar_type (st : Pytorch_types.ScalarType.t) =
  match st with
  | BOOL -> Err.return Bool
  | BYTE -> Err.return UInt8
  | CHAR -> Err.return Int8
  | DOUBLE -> Err.return Float64
  | FLOAT -> Err.return Float32
  | INT -> Err.return Int32
  | LONG -> Err.return Int64
  | SHORT -> Err.return Int16
  | _ -> Err.fail (`Unsupported_scalar_type st)

(* From the storage class name used in a pickled `.pt` (e.g. "FloatStorage"). *)
let of_storage_name = function
  | "BoolStorage" -> Err.return Bool
  | "ByteStorage" -> Err.return UInt8
  | "CharStorage" -> Err.return Int8
  | "DoubleStorage" -> Err.return Float64
  | "FloatStorage" -> Err.return Float32
  | "IntStorage" -> Err.return Int32
  | "LongStorage" -> Err.return Int64
  | "ShortStorage" -> Err.return Int16
  | s -> Err.fail (`Unknown_storage_class s)

let to_string = function
  | Bool -> "bool"
  | Float32 -> "float32"
  | Float64 -> "float64"
  | Int16 -> "int16"
  | Int32 -> "int32"
  | Int64 -> "int64"
  | Int8 -> "int8"
  | UInt8 -> "uint8"
