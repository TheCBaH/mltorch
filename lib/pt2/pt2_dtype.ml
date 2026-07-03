(* Element dtype of a stored tensor, the subset that appears in the torchvision
   `.pt2` weights and `.pt` sample inputs. Standalone (no ATen dependency); the
   bridge in lib/pt2_aten maps these to [Aten_scalar_type.t] when handing a
   tensor to the op layer. *)

type t = Float32 | Float64 | Int64 | Int32 | Int16 | Int8 | UInt8 | Bool

type error =
  [ `Unsupported_scalar_type of Pytorch_types.ScalarType.t
  | `Unknown_storage_class of string ]

let element_size = function
  | Float64 | Int64 -> 8
  | Float32 | Int32 -> 4
  | Int16 -> 2
  | Int8 | UInt8 | Bool -> 1

(* From the schema's [ScalarType] enum (as decoded from model.json / weight
   configs). Raises on a dtype this reader does not yet handle. *)
let scalar_type_name = function
  | Pytorch_types.ScalarType.UNKNOWN -> "UNKNOWN"
  | BYTE -> "BYTE"
  | CHAR -> "CHAR"
  | SHORT -> "SHORT"
  | INT -> "INT"
  | LONG -> "LONG"
  | HALF -> "HALF"
  | FLOAT -> "FLOAT"
  | DOUBLE -> "DOUBLE"
  | COMPLEXHALF -> "COMPLEXHALF"
  | COMPLEXFLOAT -> "COMPLEXFLOAT"
  | COMPLEXDOUBLE -> "COMPLEXDOUBLE"
  | BOOL -> "BOOL"
  | BFLOAT16 -> "BFLOAT16"
  | UINT16 -> "UINT16"
  | FLOAT8E4M3FN -> "FLOAT8E4M3FN"
  | FLOAT8E5M2 -> "FLOAT8E5M2"
  | FLOAT8E4M3FNUZ -> "FLOAT8E4M3FNUZ"
  | FLOAT8E5M2FNUZ -> "FLOAT8E5M2FNUZ"
  | FLOAT8E8M0FNU -> "FLOAT8E8M0FNU"
  | UINT32 -> "UINT32"
  | UINT64 -> "UINT64"

let pp_error ppf : error -> unit = function
  | `Unsupported_scalar_type st ->
      Format.fprintf ppf "unsupported ScalarType %s" (scalar_type_name st)
  | `Unknown_storage_class name ->
      Format.fprintf ppf "unknown storage class %S" name

let of_scalar_type (st : Pytorch_types.ScalarType.t) =
  match st with
  | FLOAT -> Core.return Float32
  | DOUBLE -> Core.return Float64
  | LONG -> Core.return Int64
  | INT -> Core.return Int32
  | SHORT -> Core.return Int16
  | CHAR -> Core.return Int8
  | BYTE -> Core.return UInt8
  | BOOL -> Core.return Bool
  | _ -> Core.fail (`Unsupported_scalar_type st)

(* From the storage class name used in a pickled `.pt` (e.g. "FloatStorage"). *)
let of_storage_name = function
  | "FloatStorage" -> Core.return Float32
  | "DoubleStorage" -> Core.return Float64
  | "LongStorage" -> Core.return Int64
  | "IntStorage" -> Core.return Int32
  | "ShortStorage" -> Core.return Int16
  | "CharStorage" -> Core.return Int8
  | "ByteStorage" -> Core.return UInt8
  | "BoolStorage" -> Core.return Bool
  | s -> Core.fail (`Unknown_storage_class s)

let to_string = function
  | Float32 -> "float32"
  | Float64 -> "float64"
  | Int64 -> "int64"
  | Int32 -> "int32"
  | Int16 -> "int16"
  | Int8 -> "int8"
  | UInt8 -> "uint8"
  | Bool -> "bool"
