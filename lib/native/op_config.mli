(* Small guarded/shaped types shared by op [params] records (kernel size,
   stride, padding, …) — too trivial individually to earn their own files, so
   grouped here as submodules. See .ai/native_op_config.md. *)

module Nonneg : sig
  (* A non-negative int (>= 0), distinct from a bare int so a stray negative
     value (e.g. a typo'd pad) is caught at construction. Zero is a normal
     value here (e.g. "no padding") — contrast [Pos], where zero is
     rejected. *)
  type t = private int

  val of_int : int -> t (* raises [Invalid_argument] if < 0 *)
  val to_int : t -> int
  (* also available as the free coercion [(x :> int)] *)

  val jsont : t Jsont.t
end

module Pos : sig
  (* A strictly positive int (>= 1) — for quantities where zero is
     meaningless (a stride of 0 would mean the output coordinate never
     advances), not merely non-negative. Contrast [Nonneg]. *)
  type t = private int

  val of_int : int -> t (* raises [Invalid_argument] if < 1 *)
  val to_int : t -> int
  (* also available as the free coercion [(x :> int)] *)

  val jsont : t Jsont.t
end

module Hw : sig
  (* The H/W pair shape shared by conv/pool op config (kernel size, stride,
     padding). Scoped to H and W only — not a general N-axis vector (that's
     [Vec6], a different role: tensor shapes/coords over all six axes, not a
     pair of op hyperparameters). *)
  type 'a t = { h : 'a; w : 'a }

  val jsont : 'a Jsont.t -> 'a t Jsont.t
end
