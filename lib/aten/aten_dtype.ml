(* A dtype tag that ties an OCaml element type ['a] and a Bigarray kind ['b] to
   a c10::ScalarType, so typed tensor data access is checked by the compiler:
   reading an int64 tensor as float32 is a type error, not a runtime surprise.

   Only dtypes with a clean ctypes <-> Bigarray element correspondence are
   listed (the element OCaml type is identical on both sides), plus [Bool],
   whose one-byte cell is exposed as an OCaml [int] through a ctypes view over
   [uint8_t] (a nonzero byte is true; ATen writes 0 or 1). Half has no Bigarray
   kind; the 8/16-bit ints map a C 1/2-byte cell to OCaml [int] the same way and
   are added when a model needs them. *)

type ('a, 'b) t =
  | Bool : (int, Bigarray.int8_unsigned_elt) t
  | Float32 : (float, Bigarray.float32_elt) t
  | Float64 : (float, Bigarray.float64_elt) t
  | Int32 : (int32, Bigarray.int32_elt) t
  | Int64 : (int64, Bigarray.int64_elt) t

let bool = Bool
let float32 = Float32
let float64 = Float64
let int32 = Int32
let int64 = Int64

let scalar_type : type a b. (a, b) t -> Aten_scalar_type.t = function
  | Bool -> Aten_scalar_type.Bool
  | Float32 -> Aten_scalar_type.Float
  | Float64 -> Aten_scalar_type.Double
  | Int32 -> Aten_scalar_type.Int
  | Int64 -> Aten_scalar_type.Long

(* The c10::ScalarType integer code. *)
let to_int d = Aten_scalar_type.to_int (scalar_type d)

(* The matching Bigarray kind (its element type is ['a]). *)
let kind : type a b. (a, b) t -> (a, b) Bigarray.kind = function
  | Bool -> Bigarray.int8_unsigned
  | Float32 -> Bigarray.float32
  | Float64 -> Bigarray.float64
  | Int32 -> Bigarray.int32
  | Int64 -> Bigarray.int64

(* The matching ctypes element type (also ['a]), for pointer reinterpretation. *)
let typ : type a b. (a, b) t -> a Ctypes.typ = function
  | Bool ->
      Ctypes.view ~read:Unsigned.UInt8.to_int ~write:Unsigned.UInt8.of_int
        Ctypes.uint8_t
  | Float32 -> Ctypes.float
  | Float64 -> Ctypes.double
  | Int32 -> Ctypes.int32_t
  | Int64 -> Ctypes.int64_t
