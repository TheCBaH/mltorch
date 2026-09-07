(* The per-coordinate ground form of a stage body: an [Expr.Value.t] with every index
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

   REPRESENTATION: [t]/[guard] are handles into an [Arena.t] — a hash-consed
   DAG, not a tree. Every smart constructor interns its argument shape, so two
   equal-shaped calls against the SAME arena return the identical node; sharing
   introduced this way (a repeated max-pool accumulator, a repeated-squaring
   chain, a recurrence's [prev] reference) is then automatically visible to
   every consumer below — [size]/[cells] count a shared node once, [compare]/
   [hash] agree for it in O(1) once memoized, and [pp] renders it with one
   [let] binding rather than unfolding it at every occurrence. Construction is
   otherwise exactly as inert as the old plain variant type: no algebraic
   simplification, reordering, round elimination, or NaN canonicalisation
   happens in a constructor. See .ai/native_transform_verify.md. *)

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
  type t =
    | Boundary of Cluster_var.t
    | Capture of Const_ssa.Capture.t
    | Dst of Tensor_id.t
    | Src of Tensor_id.t

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

module Arena : sig
  (* One hash-consing table pair (value nodes, guard nodes). Construction is
     append-only and acyclic: a node's children always have a strictly smaller
     id than the node itself, which is what lets every traversal below stay an
     explicit, stack-safe walk rather than recursion through the OCaml call
     stack — a node's dependencies are always already built by the time it is
     visited in increasing-id order.

     Owned by one [Ground_eval] attempt/root lineage; see that module's
     [Meter]/[Term] for the lifetime contract. Two different arenas never share
     a node, by construction — a cross-arena [compare]/[equal]/[hash] still
     works (see below), but a low-level constructor called with a child from a
     foreign arena is an [Invalid_argument], a real caller bug rather than a
     recoverable [Err.t] case, matching this repository's [invalid_arg]
     convention for a broken module invariant. *)
  type t

  val create : unit -> t
end

type guard
(* [Expr.Max_op.pool_better], for max-pool's INDEX fold. A guard limited to
       [Lt] could not express the NaN branch at all, so this is a
       constructor rather than an encoding as a comparison. *)

type t

type view =
  | Binary of Expr.Value.binary_op * t * t
  | Cell of Cell.t
  | Const of float
  | Max of Expr.Max_op.t * t * t
  | Round of t (* A stage boundary: this value is stored, and storage is f32. *)
  | Select of guard * t * t
  | Unary of Expr.Value.unary_op * t

type guard_view = Lt of t * t | Pool_better of { best : t; value : t }

val out : t -> view
val guard_out : guard -> guard_view
val arena : t -> Arena.t

(* ---- arena-bound smart constructors --------------------------------------

   Every constructor interns: a call whose (tag, operator, child ids) already
   exists in [Arena.t] returns the existing node rather than allocating one.
   No simplification beyond that — [binary a Add (const a 0.) x] is a genuine
   two-node [Binary], not folded to [x]. Raises [Invalid_argument] if any child
   was built in a different arena than the one passed in. *)

val binary : Arena.t -> Expr.Value.binary_op -> t -> t -> t
val cell : Arena.t -> Cell.t -> t
val const : Arena.t -> float -> t
val max : Arena.t -> Expr.Max_op.t -> t -> t -> t
val round : Arena.t -> t -> t
val select : Arena.t -> guard -> t -> t -> t
val unary : Arena.t -> Expr.Value.unary_op -> t -> t
val lt : Arena.t -> t -> t -> guard
val pool_better : Arena.t -> best:t -> value:t -> guard

(* [Const] is compared by [Core.Float_bits.exact], never by [( = )] or
   [Float.compare]: an [Identical] claim is about bits, and the ordinary
   comparisons equate -0. with +0. and every NaN with every other.

   Cross-arena: two nodes built independently, in different arenas, that
   happen to be structurally identical compare/hash/equal as such — nothing
   below assumes same-arena inputs. Same-identity (same arena, same id) is
   recognised in O(1); otherwise this recurses structurally with per-call pair
   memoisation, so a diamond or a repeated accumulator does not cost
   exponential comparison work. [hash] is O(1): every node caches a bottom-up
   structural digest at construction, so equal terms hash equal (including
   across arenas) but hash equality never itself proves [equal]. *)
val compare : t -> t -> int
val equal : t -> t -> bool
val hash : t -> int
val compare_guard : guard -> guard -> int
val equal_guard : guard -> guard -> bool
val hash_guard : guard -> int

(* The distinct reachable value-node count from this root — a hash-consed
   DIAMOND counts its shared node once, unlike the size of the tree that would
   unfold it. A guard costs nothing on its own; the value nodes it reads still
   count once each. Computed by an explicit, iterative reachability walk (no
   recursion, so no depth hazard) — cheap enough to call at every root
   registration/replacement, per the design record; callers that need it
   repeatedly should still cache it rather than recompute. *)
val size : t -> int
val cells : t -> Cell.Set.t

(* Rewrite every cell [boundary] gives a variable for into a [Boundary] cell at
   the same coordinate, leaving the rest side-qualified, into [into]. This is
   what makes two graphs' differently-numbered edges compare equal, and it is a
   claim about the MAP rather than about either graph — so it targets a
   separate, short-lived comparison arena and is never fed back into grounding.

   NEVER normalise a projected term: projection discards the storage edge that
   [Ground_eval.Env.stored_f32] needs to decide whether a [Round] may collapse,
   and an unknown cell answers [false], so normalising afterwards silently
   blocks collapses that are sound. Normalise the raw term, then project. *)
val project :
  into:Arena.t -> boundary:(Origin.t -> Cluster_var.t option) -> t -> t

(* Drop every [Round], into [into]. This is exactly the [Equivalent] reading of
   a term — "equal in exact arithmetic; rounding may differ" — and it must NOT
   be applied when checking [Identical]. *)
val erase_rounds : into:Arena.t -> t -> t

(* [Round] applies the f32 round-trip, so [eval] reproduces what the engine
   computes rather than an idealised real-arithmetic value. Raises
   [Not_found] if the valuation omits a cell the term reads. Memoised per node
   for one call (so a shared subterm is evaluated once) and deliberately lazy
   in [Select]: only the branch the guard selects is ever evaluated, so an
   unselected branch may read a cell absent from [v] with no error. *)
val eval : t -> Valuation.t -> float

(* Deterministic. A node reachable from more than one place is printed once, as
   a numbered [let], and referenced by name everywhere else — never by a raw
   allocation id, and never by unfolding it again. A term with no sharing
   prints exactly as the old unshared syntax did. *)
val pp : Format.formatter -> t -> unit

(* The result of normalising. [blocked] names the cells whose [Round] could not
   be collapsed, so the driver can tell "these terms differ" from "these terms
   differ, and possibly only because a collapse was unavailable". *)
type normalised = { blocked : Cell.Set.t; expr : t }

(* Collapse rounding where that is sound, given [stored_f32 cell] = "reading
   this cell yields a value already representable in f32", into [into]:

     Round (Cell c) -> Cell c      only when [stored_f32 c]
     Round (Round e) -> Round e    always (f32 rounding is idempotent)
     Round (Const v) -> Const (f32 v)

   The first rule's side condition is load-bearing. [Payload.get_float] decodes
   I32/I64 through [Int32.to_float]/[Int64.to_float] — values above 2^24 are not
   f32-representable — and I8/I16 through [Quant.dequantize], a scale multiply
   whose product need not be either. F32, F16 and BF16 are safe. Dropping the
   condition would let a permute stage's materialization vanish for a quantized
   input, where it is observable. *)
val normalise : into:Arena.t -> stored_f32:(Cell.t -> bool) -> t -> normalised
