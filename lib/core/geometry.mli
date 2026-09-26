(* The guarded scalar types of a windowed op's hyper-parameters — a stride, a
   padding, a kernel — and the H/W pair they come in. They live below the
   expression language so that [Expr]'s pooling intrinsic can take the very
   types [native]'s op configuration holds, with no conversion at the seam.
   [native]'s [Op_config] re-exports them and adds the wire codecs
   ([Jsont] is not a dependency of this library). *)

module Nonneg : sig
  (* A non-negative int (>= 0), distinct from a bare int so a stray negative
     value (e.g. a typo'd pad) is caught at construction. Zero is a normal
     value here (e.g. "no padding") — contrast [Pos], where zero is
     rejected. *)
  type t = private int

  val of_int : int -> t (* raises [Invalid_argument] if < 0 *)
  val to_int : t -> int

  val to_int64 : t -> int64
  (** The named exit for arithmetic done in [Int64] on purpose. *)
  (* also available as the free coercion [(x :> int)] *)

  val pp : Format.formatter -> t -> unit
end

module Pos : sig
  (* A strictly positive int (>= 1) — for quantities where zero is
     meaningless (a stride of 0 would mean the output coordinate never
     advances), not merely non-negative. Contrast [Nonneg]. *)
  type t = private int

  val of_int : int -> t (* raises [Invalid_argument] if < 1 *)
  val to_int : t -> int

  val to_int64 : t -> int64
  (** The named exit for arithmetic done in [Int64] on purpose. *)
  (* also available as the free coercion [(x :> int)] *)

  val pp : Format.formatter -> t -> unit
end

module Hw : sig
  (* The H/W pair shape shared by conv/pool op config (kernel size, stride,
     padding). Scoped to H and W only — not a general N-axis vector (that's
     [Vec6], a different role: tensor shapes/coords over all six axes, not a
     pair of op hyperparameters). *)
  type 'a t = { h : 'a; w : 'a }

  val pp : (Format.formatter -> 'a -> unit) -> Format.formatter -> 'a t -> unit
end
