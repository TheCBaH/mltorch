(** The host objects generated code may name. Closed, so a global cannot be
    misspelled into a free variable, and no {!Js_ident.t} can shadow one. *)

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

val all : t list

val name : t -> string
(** The identifier JavaScript knows it by, e.g. [Big_int] is ["BigInt"]. *)
