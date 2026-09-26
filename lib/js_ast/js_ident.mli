type t = private string

val v : string -> t
(** [[A-Za-z_$][A-Za-z0-9_$]*], not a reserved word, not a {!Js_global.t}.
    Raises [Invalid_argument] otherwise: a name is a programming constant, never
    data, so a bad one is a defect and not an error value. *)

val to_string : t -> string
val compare : t -> t -> int
val equal : t -> t -> bool
val pp : t Fmt.t

module Set : Set.S with type elt = t
