(** A zero-based operand or result position. A positional bound keeps a position
    representable on every backend; arity is checked against the node the
    position is applied to, not here. *)

module type S = sig
  type t

  val of_index : int -> t option
  val to_index : t -> int
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
  val decode : t Wire.decoder
  val encode : t Wire.encoder
end

val max_index : int

module Make () : S
