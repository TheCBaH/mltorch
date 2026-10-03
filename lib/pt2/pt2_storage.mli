(** The bytes of a tensor storage: a bigstring, the payload kind that
    [Unix.map_file], js_of_ocaml's [Typed_array.Bigstring] and
    ocaml-safetensors' mapped views all share, so a weight can come from a zip
    member or straight from a memory-mapped checkpoint with no copy. Reads are
    little-endian and bounds-checked; an out-of-range read raises
    [Invalid_argument]. *)

type t = (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

val empty : t
val length : t -> int

val of_string : string -> t
(** One copy of [s]. *)

val get_uint8 : t -> int -> int
val get_int32_le : t -> int -> int32
val get_int64_le : t -> int -> int64
