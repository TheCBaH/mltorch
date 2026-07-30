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

(* Which graph a cell's raw id belongs to — the two graphs share one numeric
   namespace, so a bare [Tensor_id.t] cannot say whether [t2] is the source's or
   the destination's, and the verifier's answer turns on exactly that.

   Grounding emits [Src] and [Dst] only. [Boundary] is introduced by [project]
   and nowhere else: it is a comparison-time claim that two side-qualified cells
   name ONE value, which is a hypothesis about the map and not a fact about
   either graph. Keeping it out of grounding is what stops that claim leaking
   into [stored_f32], [expand] or a stage lookup, all of which need the real
   storage edge. See .ai/native_transform_verify.md §7. *)
module Origin : sig
  type t = Boundary of Cluster_var.t | Dst of Tensor_id.t | Src of Tensor_id.t

  val compare : t -> t -> int
  val edge : t -> Tensor_id.t option
  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit

  module Map : Map.S with type key = t
end

module Cell : sig
  (* One scalar element of one edge: the leaf that expansion stops at, and the
     free variable a payload-independent proof quantifies over. *)
  type t = { coord : Vec6.coord; origin : Origin.t }

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

  (* Draw [n] over the given cells. Deterministic in [(cell, n)] alone — no RNG
     state, so the same draw is reproducible from a printed verdict and does not
     depend on the order cells are visited.

     Draw 0 is a coordinate ramp rather than noise, because the errors this is
     looking for are permutation and indexing mistakes, and a ramp separates
     every cell of a tensor where a constant would not. Later draws are
     pseudo-random and non-zero, so a [Div] does not degenerate. *)
  val draw : int -> Cell.Set.t -> t
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

(* Rewrite every cell [boundary] gives a variable for into a [Boundary] cell at
   the same coordinate, leaving the rest side-qualified. This is what makes two
   graphs' differently-numbered edges compare equal, and it is a claim about the
   MAP rather than about either graph — so it is applied to a copy, at
   comparison time, and never fed back into grounding.

   NEVER normalise a projected term: projection discards the storage edge that
   [Ground_eval.Env.stored_f32] needs to decide whether a [Round] may collapse,
   and an unknown cell answers [false], so normalising afterwards silently
   blocks collapses that are sound. Normalise the raw term, then project. *)
val project : boundary:(Origin.t -> Cluster_var.t option) -> t -> t

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
