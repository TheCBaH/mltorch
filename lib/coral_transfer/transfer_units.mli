(** The two quantities a transfer plan trades in: rows of A and bytes of DTCM.
    Each is a fresh type with the arithmetic the plan does, so a row count
    cannot be passed where a byte count is wanted. *)

module Rows : sig
  type t = private int

  val of_int : int -> t
  (** Raises [Invalid_argument] on a negative count. *)

  val to_int : t -> int
  val one : t
  val is_positive : t -> bool
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val min : t -> t -> t

  val lcm : t -> t -> t
  (** Both positive. *)

  val is_multiple : of_:t -> t -> bool
  (** [is_multiple ~of_:u r]: [r] is a whole number of [u]s; [u] positive. *)

  val round_down : to_:t -> t -> t
  val round_up : to_:t -> t -> t
  val pp : Format.formatter -> t -> unit
end

module Bytes : sig
  type t = private int

  val of_int : int -> t
  (** Raises [Invalid_argument] on a negative size. *)

  val to_int : t -> int
  val zero : t
  val add : t -> t -> t

  val sub_floor : t -> t -> t
  (** [sub_floor a b] is [a - b], or zero when [b] exceeds [a]. *)

  val times : Rows.t -> t -> t
  (** [times rows per_row]. *)

  val double : t -> t
  val max : t -> t -> t
  val compare : t -> t -> int
  val equal : t -> t -> bool

  val rows_fitting : t -> per_row:t -> Rows.t
  (** The most whole rows of [per_row] bytes (at least one byte) in the size. *)

  val is_whole : of_:t -> t -> bool
  (** The size is a multiple of [of_]. *)

  val pp : Format.formatter -> t -> unit
end
