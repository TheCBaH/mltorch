(** A branch target or export. Two labels with the same name are the same label
    within one kernel, wherever the values were made. *)

type t

val compare : t -> t -> int
val equal : t -> t -> bool
val name : t -> string
val named : string -> t
val pp : t Fmt.t
