(* The per-coordinate ground form of a stage body: an [Expr.t] with every index
   expression evaluated at one concrete output coordinate, every [Reduce]
   unrolled at its now-concrete bounds, and every [Max_pool] window expanded.
   No binders survive, so two ground expressions denote the same value exactly
   when they are structurally equal — which is what makes [Identical] (bit-for-
   bit) checkable by [compare] alone, with no algebra.

   Grounding the INDICES rather than canonicalising them is not an optimisation,
   it is forced. [Reshape.Compute.pixel] reaches its input through
   [flat_offset] then [delinearize], emitting [Index_floor_div_pos] and a
   mod-idiom per axis, while [Permute.Compute.pixel] emits a bare [Index_var].
   Those agree only under [0 <= coord_a < extent_a] — not an affine identity, so
   no canonical "sum of k_i * var_i + c" form could ever discharge
   [reshape_to_permute]. Evaluated at a coordinate they are simply the same six
   integers.

   See .ai/native_transform_verify.md. *)

module Cell : sig
  (* One scalar element of one edge: the leaf that expansion stops at, and the
     free variable a payload-independent proof quantifies over. *)
  type t = { coord : Vec6.coord; id : Tensor_id.t }

  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit

  module Set : Set.S with type elt = t
end

module Valuation : sig
  (* A concrete assignment to free cells. Carried by a refutation, so a
     counterexample can be replayed through [eval] rather than asserted. *)
  type t

  val empty : t
  val find : t -> Cell.t -> float
  val of_list : (Cell.t * float) list -> t
  val pp : Format.formatter -> t -> unit
end

type guard = Lt of t * t | Pool_better of { best : t; value : t }
(* [Max_op.pool_better], for max-pool's INDEX fold. A guard limited to
         [Lt] could not express the NaN branch at all, so this is a
         constructor rather than an encoding as a comparison. *)

and t =
  | Binary of Expr.binary_op * t * t
  | Cell of Cell.t
  | Const of float
  | Max of Max_op.t * t * t
  | Round of t (* A stage boundary: this value is stored, and storage is f32. *)
  | Select of guard * t * t
  | Unary of Expr.unary_op * t

val cells : t -> Cell.Set.t

(* [Const] is compared by [Int64.bits_of_float], never by [( = )] or
   [Float.compare]: an [Identical] claim is about bits, and the ordinary
   comparisons equate -0. with +0. and every NaN with every other. *)
val compare : t -> t -> int
val equal : t -> t -> bool

(* Drop every [Round]. This is exactly the [Equivalent] reading of a term —
   "equal in exact arithmetic; rounding may differ" — and it must NOT be applied
   when checking [Identical]. *)
val erase_rounds : t -> t

(* [Round] applies the f32 round-trip, so [eval] reproduces what the engine
   computes rather than an idealised real-arithmetic value. Raises
   [Not_found] if the valuation omits a cell the term reads. *)
val eval : t -> Valuation.t -> float
val pp : Format.formatter -> t -> unit
val size : t -> int

(* The result of normalising. [blocked] names the cells whose [Round] could not
   be collapsed, so the driver can tell "these terms differ" from "these terms
   differ, and possibly only because a collapse was unavailable". *)
type normalised = { blocked : Cell.Set.t; expr : t }

(* Collapse rounding where that is sound, given [stored_f32 cell] = "reading
   this cell yields a value already representable in f32":

     Round (Cell c) -> Cell c      only when [stored_f32 c]
     Round (Round e) -> Round e    always (f32 rounding is idempotent)
     Round (Const v) -> Const (f32 v)

   The first rule's side condition is load-bearing. [Payload.get_float] decodes
   I32/I64 through [Int32.to_float]/[Int64.to_float] — values above 2^24 are not
   f32-representable — and I8/I16 through [Quant.dequantize], a scale multiply
   whose product need not be either. F32, F16 and BF16 are safe. Dropping the
   condition would let a permute stage's materialization vanish for a quantized
   input, where it is observable. *)
val normalise : stored_f32:(Cell.t -> bool) -> t -> normalised
