type t =
  | Big_int
  | Big_int64_array
  | Float32_array
  | Float64_array
  | Int16_array
  | Int32_array
  | Int8_array
  | Math
  | Number
  | Uint16_array
  | Uint32_array
  | Uint8_array

let all =
  [
    Big_int;
    Big_int64_array;
    Float32_array;
    Float64_array;
    Int16_array;
    Int32_array;
    Int8_array;
    Math;
    Number;
    Uint16_array;
    Uint32_array;
    Uint8_array;
  ]

let name = function
  | Big_int -> "BigInt"
  | Big_int64_array -> "BigInt64Array"
  | Float32_array -> "Float32Array"
  | Float64_array -> "Float64Array"
  | Int16_array -> "Int16Array"
  | Int32_array -> "Int32Array"
  | Int8_array -> "Int8Array"
  | Math -> "Math"
  | Number -> "Number"
  | Uint16_array -> "Uint16Array"
  | Uint32_array -> "Uint32Array"
  | Uint8_array -> "Uint8Array"
