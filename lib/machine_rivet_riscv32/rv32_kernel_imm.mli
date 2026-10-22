(** Immediate domains. Each [t] is an unchecked request for a value in its
    domain: [of_int] wraps the integer without looking at it, and the type only
    keeps one domain from being passed for another. [Rv32_kernel.finalize] is
    the range boundary, so a literal stays free to write and no type claims a
    value was validated. *)

module Domain : sig
  (** Alphabetical. *)
  type t = Imm12 | Li | Shamt | Simm5 | U20

  val pp : t Fmt.t
end

module Imm_range : sig
  type t = { domain : Domain.t; value : int }
  (** A value outside its domain, as written. *)

  val pp : t Fmt.t
end

module type S = sig
  type t

  val domain : Domain.t
  val equal : t -> t -> bool
  val in_range : t -> bool
  val of_int : int -> t
  val pp : t Fmt.t
  val range : t -> Imm_range.t
  val to_int : t -> int
end

module Imm12 : sig
  include S

  (** A signed 12-bit offset or addend, -2048 to 2047. *)

  val minus_one : t
  val zero : t
end

module Li : sig
  include S

  (** What Rivet's [li] encodes as one instruction: a signed 12-bit value, or a
      multiple of 4096 within the signed 32-bit range (a [lui] alone). Any other
      value is refused when the module is planned. *)
end

module Shamt : S
(** A shift amount, 0 to 31. *)

module Simm5 : sig
  include S

  (** A vector immediate, -16 to 15. *)

  val zero : t
end

module U20 : S
(** The upper 20 bits of a [lui], 0 to 0xFFFFF. *)
