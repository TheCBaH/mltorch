(* Element dtype of a stored tensor, the subset that appears in the torchvision
   `.pt2` weights and `.pt` sample inputs. Standalone (no ATen dependency); the
   bridge in lib/pt2_aten maps these to [Aten_scalar_type.t] when handing a
   tensor to the op layer. *)

type t = Float32 | Float64 | Int64 | Int32 | Int16 | Int8 | UInt8 | Bool

let element_size = function
  | Float64 | Int64 -> 8
  | Float32 | Int32 -> 4
  | Int16 -> 2
  | Int8 | UInt8 | Bool -> 1

(* From the schema's [ScalarType] enum (as decoded from model.json / weight
   configs). Raises on a dtype this reader does not yet handle. *)
let of_scalar_type (st : Pytorch_types.ScalarType.t) =
  match st with
  | FLOAT -> Float32
  | DOUBLE -> Float64
  | LONG -> Int64
  | INT -> Int32
  | SHORT -> Int16
  | CHAR -> Int8
  | BYTE -> UInt8
  | BOOL -> Bool
  | _ -> failwith "Pt2_dtype: unsupported ScalarType"

(* From the storage class name used in a pickled `.pt` (e.g. "FloatStorage"). *)
let of_storage_name = function
  | "FloatStorage" -> Float32
  | "DoubleStorage" -> Float64
  | "LongStorage" -> Int64
  | "IntStorage" -> Int32
  | "ShortStorage" -> Int16
  | "CharStorage" -> Int8
  | "ByteStorage" -> UInt8
  | "BoolStorage" -> Bool
  | s -> failwith ("Pt2_dtype: unknown storage class " ^ s)

let to_string = function
  | Float32 -> "float32"
  | Float64 -> "float64"
  | Int64 -> "int64"
  | Int32 -> "int32"
  | Int16 -> "int16"
  | Int8 -> "int8"
  | UInt8 -> "uint8"
  | Bool -> "bool"
