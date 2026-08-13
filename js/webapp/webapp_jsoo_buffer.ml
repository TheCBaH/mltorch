open Js_of_ocaml

let array_buffer_length value =
  if not (Js.instanceof (Js.Unsafe.coerce value) Typed_array.arrayBuffer) then
    Err.fail `Not_an_array_buffer
  else
    Ok
      (Js.float_of_number (Js.Unsafe.coerce (Js.Unsafe.get value "byteLength")))

let to_string buffer ~len =
  let bytes =
    Js.Unsafe.new_obj Typed_array.uint8Array_fromBuffer
      [| Js.Unsafe.inject buffer |]
  in
  String.init len (fun i -> Char.chr (Js.Unsafe.get bytes i))

let fresh_array_buffer text =
  let buffer =
    Js.Unsafe.new_obj Typed_array.arrayBuffer
      [| Js.Unsafe.inject (String.length text) |]
  in
  let bytes =
    Js.Unsafe.new_obj Typed_array.uint8Array_fromBuffer
      [| Js.Unsafe.inject buffer |]
  in
  String.iteri (fun i c -> Js.Unsafe.set bytes i (Char.code c)) text;
  buffer
