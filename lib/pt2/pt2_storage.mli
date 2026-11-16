(** The bytes of a tensor storage: a bigstring, the payload kind that
    [Unix.map_file], js_of_ocaml's [Typed_array.Bigstring] and
    ocaml-safetensors' mapped views all share, so a weight can come from a zip
    member or straight from a memory-mapped checkpoint with no copy. Reads are
    little-endian and bounds-checked; an out-of-range read raises
    [Invalid_argument]. *)

type t = (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

val empty : t
val length : t -> Core.Storage_units.Byte_size.t

val of_string : string -> t
(** One copy of [s]. *)

val to_string : t -> string
(** One copy of the bytes. *)

val get_uint8 : t -> Core.Storage_units.Byte_offset.t -> int
val get_int32_le : t -> Core.Storage_units.Byte_offset.t -> int32
val get_int64_le : t -> Core.Storage_units.Byte_offset.t -> int64

(** {2 Element cells}

    A cell is the index of an element of a given width: the form a loop over a
    layout already proven inside the storage ([Pt2_tensor.Cells]) has at hand,
    so it need not build an offset per element. The byte offset [cell * width]
    is checked not to wrap before it is read. *)

val cell_count : t -> Core.Storage_units.Element_bytes.t -> int
(** How many whole elements of that width the storage holds. *)

val cell_uint8 : t -> cell:int -> int
val cell_int32_le : t -> cell:int -> int32
val cell_int64_le : t -> cell:int -> int64
