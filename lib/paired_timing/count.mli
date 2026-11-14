(** A number of elements, never negative. *)

type t = private int

val of_int : int -> t
(** Raises [Invalid_argument] on a negative count. *)

val to_int : t -> int
val pp : Format.formatter -> t -> unit
