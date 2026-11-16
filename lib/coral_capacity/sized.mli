(** A size in bytes as [int64], one fresh type per [Make ()] application so two
    domains of bytes cannot be compared or added by mistake. *)

module type S = sig
  type t = private int64

  val of_int64 : int64 -> t
  (** Raises [Invalid_argument] on a negative size. *)

  val to_int64 : t -> int64
  val zero : t

  val add : t -> t -> t
  (** Raises [Invalid_argument] when the sum overflows. *)

  val compare : t -> t -> int
  val equal : t -> t -> bool
  val pp : t Fmt.t
end

module Make () : S
