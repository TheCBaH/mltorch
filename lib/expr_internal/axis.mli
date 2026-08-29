(* The fixed ordered frame the language indexes over. Deliberately no [jsont]
   and no [of_string]: serialisation belongs to the consumer that owns a wire
   format, and pulling it in here would cost this library a [jsont] dependency
   it has no other use for. [lib/native]'s [Axis] keeps those and aliases this
   type manifestly, so the two can never drift. *)

type t = N | T | D | H | W | C

val all : t list
val to_int : t -> int
val compare : t -> t -> int
val equal : t -> t -> bool
val to_string : t -> string
val pp : Format.formatter -> t -> unit
