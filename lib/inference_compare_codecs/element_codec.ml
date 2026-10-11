open Inference_compare_schema

type t = { id : Id.Codec.t; element : Capability.element; byte_width : int }

let id name =
  match Id.Codec.of_string name with
  | Ok c -> c
  | Error _ -> invalid_arg ("Element_codec.id: " ^ name)

let bool = id "bool"
let f32 = id "f32"
let f64 = id "f64"
let i32 = id "i32"
let i64 = id "i64"
let u8 = id "u8"

module Registry = struct
  type codec = t
  type t = codec list

  let builtin : t =
    [
      { id = bool; element = Boolean; byte_width = 1 };
      { id = f32; element = Float_binary32; byte_width = 4 };
      { id = f64; element = Float_binary64; byte_width = 8 };
      { id = i32; element = Signed_integer; byte_width = 4 };
      { id = i64; element = Signed_integer; byte_width = 8 };
      { id = u8; element = Unsigned_integer; byte_width = 1 };
    ]

  let find (t : t) id = List.find_opt (fun c -> Id.Codec.equal c.id id) t

  let add (t : t) (c : codec) =
    match find t c.id with
    | Some _ -> Err.fail (`Duplicate_codec c.id)
    | None -> Ok (t @ [ c ])

  let capabilities (t : t) =
    List.map
      (fun c ->
        {
          Capability.Codec.codec = c.id;
          element = c.element;
          byte_width = c.byte_width;
          element_inspection = Supported;
          exact_values = Supported;
        })
      t
end

let is_float c =
  match c.element with
  | Float_binary32 | Float_binary64 -> true
  | Boolean | Signed_integer | Unsigned_integer -> false

let is_integer c = not (is_float c)

let bits c s i =
  let off = i * c.byte_width in
  match c.byte_width with
  | 1 -> Int64.of_int (Char.code (String.unsafe_get s off))
  | 2 -> Int64.of_int (String.get_uint16_le s off)
  | 4 -> Int64.logand (Int64.of_int32 (String.get_int32_le s off)) 0xFFFF_FFFFL
  | 8 -> String.get_int64_le s off
  | w -> invalid_arg (Printf.sprintf "Element_codec.bits: width %d" w)

let set_bits c b i v =
  let off = i * c.byte_width in
  match c.byte_width with
  | 1 -> Bytes.set b off (Char.chr (Int64.to_int (Int64.logand v 0xFFL)))
  | 2 -> Bytes.set_uint16_le b off (Int64.to_int (Int64.logand v 0xFFFFL))
  | 4 -> Bytes.set_int32_le b off (Int64.to_int32 v)
  | 8 -> Bytes.set_int64_le b off v
  | w -> invalid_arg (Printf.sprintf "Element_codec.set_bits: width %d" w)

let integer_value c s i =
  let raw = bits c s i in
  match c.element with
  | Signed_integer -> (
      match c.byte_width with
      | 1 -> Int64.shift_right (Int64.shift_left raw 56) 56
      | 2 -> Int64.shift_right (Int64.shift_left raw 48) 48
      | 4 -> Int64.shift_right (Int64.shift_left raw 32) 32
      | _ -> raw)
  | Boolean | Unsigned_integer -> raw
  | Float_binary32 | Float_binary64 ->
      invalid_arg "Element_codec.integer_value: float codec"

let float_value c s i =
  match c.element with
  | Float_binary32 -> Int32.float_of_bits (Int64.to_int32 (bits c s i))
  | Float_binary64 -> Int64.float_of_bits (bits c s i)
  | Boolean | Signed_integer | Unsigned_integer ->
      invalid_arg "Element_codec.float_value: integer codec"
