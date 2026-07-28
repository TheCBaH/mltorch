(* Comparing two ground terms as polynomials in their free cells, which is what
   distribution and re-association look like: `(Σ xₖ·Wₖ)·s` against
   `Σ xₖ·(Wₖ·s)`. That is exactly the batch-norm fold, and no amount of
   structural comparison reaches it.

   This NEVER yields a proof. Coefficients `1` and `1 + ε` pass a tolerance
   while being neither exactly equal nor boundedly close for an unbounded free
   cell, so agreement here is evidence — reported as [Tested (Agrees _)] — and
   the caller decides what to do with it. Proving [Equivalent] outright would
   need exact rational coefficients, and no current pass would benefit:
   [fold_batch_norm] re-derives its constants numerically (eps arrives as a
   constant EDGE with a payload against a source-side [Const]), so exact
   equality fails whatever the arithmetic.

   [Round] is erased before comparison. That is the [Equivalent] reading —
   "equal in exact arithmetic; rounding may differ" — and it must not be used
   to decide [Identical].

   See .ai/native_transform_verify.md §15. *)

module Generator : sig
  (* What a monomial is built from: a free cell, or a subterm the polynomial
     view cannot see inside ([Select], [Max], [Unary], division by a
     non-constant). Opaque generators are keyed EXACTLY, which is where the
     incompleteness lives — never unsoundness. *)
  type t = Cell of Ground_expr.Cell.t | Opaque of Ground_expr.t

  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit
end

type t

val of_ground : Ground_expr.t -> t
val pp : Format.formatter -> t -> unit

(* Same monomials, and coefficients within [|a - b| <= tolerance * max 1 |a| |b|]
   — scaled-relative with an absolute floor, so a near-zero coefficient is held
   to [tolerance] absolutely rather than being asked for exact agreement.

   Deliberately NOT the same acceptance bar as an output-level check like
   fold_batch_norm_test.ml's [max_abs_diff <= 1e-5 * max 1 (max_abs before)]:
   per-coefficient and whole-tensor agreement are incomparable in general, since
   many small coefficient errors can sum while one large error on a near-zero
   activation may never surface. That is why the verifier is added ALONGSIDE
   that check rather than replacing it. *)
val agree_within : tolerance:float -> t -> t -> bool

(* The relation the driver actually uses: [agree_within] on maximal arithmetic
   regions, recursing through matching non-arithmetic heads. Without the
   recursion a relu wrapping the fold — [select(E < 0, 0, E)] on both sides with
   E differing only by rounding — would compare two unequal opaque generators
   and report disagreement.

   The gap it leaves: an arithmetic COMBINATION of structurally-different
   non-polynomial subterms (say `relu a + 1` against `relu a' + 1` with
   `a ≈ a'`) still compares those subterms exactly. Closing it needs matching
   atoms up to this same relation, which is a matching problem no current pass
   poses. *)
val agree : tolerance:float -> Ground_expr.t -> Ground_expr.t -> bool
