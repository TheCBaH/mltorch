open Js_of_ocaml

val array_buffer_length :
  Js.Unsafe.any -> (float, [> `Not_an_array_buffer ]) Err.t

val to_string : Typed_array.arrayBuffer Js.t -> len:int -> string
val fresh_array_buffer : string -> Typed_array.arrayBuffer Js.t
