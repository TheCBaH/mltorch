(** Validated text for one semantic role. Every module a functor here returns
    has its own abstract [t], so an [Id.Unit.t] can never stand for an
    [Id.Case.t] although both serialize as strings. *)

module type S = sig
  type t

  val of_string : string -> (t, Text_error.t) Err.t
  val to_string : t -> string
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

module type ROLE = sig
  val role : string
end

(** Identifiers and names: nonempty, bounded, no control characters and no
    leading or trailing whitespace. *)
module Make_identifier (_ : ROLE) () : S

(** Explanatory prose: bounded and free of control characters other than line
    breaks. May be empty. *)
module Make_prose (_ : ROLE) () : S
