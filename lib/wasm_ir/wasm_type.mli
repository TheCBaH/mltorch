type t = F32 | F64 | I32 | I64 | V128

val to_byte : t -> int
(** The spec's value-type byte (e.g. [I32] is [0x7F]). *)

val name : t -> string
val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit
