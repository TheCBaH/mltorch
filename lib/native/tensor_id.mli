type t = private int

val of_int : int -> t (* builder-internal allocation *)
val to_int : t -> int
val equal : t -> t -> bool
val compare : t -> t -> int
val pp : Format.formatter -> t -> unit
val jsont : t Jsont.t

module Map : Map.S with type key = t
module Set : Set.S with type elt = t
