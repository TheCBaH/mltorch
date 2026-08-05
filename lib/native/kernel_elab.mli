(* Capture-safe elaboration of ONE producer edge into its consumer.

   Takes a [Kernel.Use.t], not a [Fusion_plan.t]: the planner has to elaborate a
   candidate to enforce its budget, so the reverse dependency would be a compile
   cycle. It is also why a set is the wrong argument. A set forces this API to
   answer a placement question it has no store set for — given ordinary
   dependencies X->A, X->Y, A->B and virtual uses {X->Y, A->B}, the endpoints are
   disjoint, yet inlining A into B leaves a load of X whose store nothing here
   guarantees, and no value is shared between the two use records for a conflict
   error to even name. One edge removes the question: the inlined producer's own
   loads are by construction not virtual.

   This is deliberately NOT the planner's legality rule. [Fusion_plan] adds
   global unique-use, no-intrinsic-use and consumer-class restrictions, which
   exist for PLACEMENT. A producer with one site in this consumer and a second
   consumer elsewhere elaborates perfectly well; it simply cannot be
   virtualized. Keeping the two apart is what lets the planner widen later
   without touching this. *)

type error =
  [ `Body of Kernel.Body_error.t
  | `Unknown_use of Kernel.Use.t
  | `Not_a_dependency of Kernel.Use.t
  | `Unsupported_use of Kernel.Use.t ]

val pp_error : Format.formatter -> [< error ] -> unit

module Site : sig
  type t
  (** A VALIDATED edge: both endpoints resolved and the single admissible load
      coordinate found. Opaque, so holding one means having passed [site], and
      [elaborate_site] can rewrite without resolving or folding again.

      That matters for the planner, which checks a candidate and then elaborates
      it: with only a coordinate-returning [site] and a self-contained
      [elaborate], a legal edge re-folded its consumer's whole body and repeated
      both endpoint lookups. *)

  val use : t -> Kernel.Use.t
  val coord : t -> Expr.Role.Position.t Expr.Index.t Expr.Coord.t
end

val site : Kernel.t -> Kernel.Use.t -> (Site.t, error) Core.result
(** The single admissible load site for this edge, or why there is none.
    SITE-LOCAL: resolves the two endpoints and folds only the named consumer.
    Routing it through [Analysis.of_kernel] would make one-edge elaboration
    proportional to the whole kernel AST, which is the planner's cost — incurred
    because it checks many edges — and not something a standalone call should
    pay. Use [Analysis] plus [site_in] to check many edges.

    The supported class, decided BEFORE any rewriting:

    - exactly one occurrence of the named producer inside the named consumer —
      one occurrence of THAT edge, not one load in the consumer, which an [Add]
      never has since it also loads its residual operand;
    - per axis, the coordinate is the plain output variable for that axis with
      EQUAL producer and consumer extents, or [zero] where the producer extent
      is 1.

    Deciding first is not tidiness. [Kernel.load_uses] is a set of pairs and so
    loses site multiplicity: a legitimate pair may occur many times in the
    consumer, each occurrence taking its own freshened producer copy, and
    [substitute_output] additionally installs the consumer's index tree at every
    producer output-variable occurrence. Two in-budget bodies can therefore
    build — or walk — a result quadratic in the body budget before a check on
    the finished tree could refuse it.

    The extent equality is not decoration either. [Kernel.create] deliberately
    does not prove symbolic load upper bounds, so a producer of H extent 1 under
    a consumer of extent 2, loading at plain [Output H], satisfies single-site
    syntax. Both execution paths reject consumer coordinate H=1 — stored reads
    past the materialized producer, [value_at] bounds-checks it — but
    substituting the producer BODY carries no such check, so a constant producer
    quietly succeeds and the elaborated expression stops meaning what the edge
    meant.

    Shared with [Fusion_plan] rather than restated there, so the two cannot
    drift. *)

module Analysis : sig
  type t
  (** Load-site evidence extracted from a kernel, and OWNED: reachable only
      through [of_kernel], so a caller cannot fabricate it or mix data belonging
      to another consumer.

      An earlier shape took the extracted load list as an argument and treated
      it as proof, which proved nothing. One fabricated pointwise entry yielded
      a [Site.t] for a pair that is not a dependency at all, and
      [elaborate_site] then rewrote the consumer's REAL body — substituting
      nothing and reporting the unchanged body as a successful elaboration, or
      substituting every real occurrence of a multi-site edge the checks existed
      to refuse. A private type whose constructor accepts unchecked data is not
      private.

      It also indexes occurrences per (consumer, producer), so admission reads a
      summary rather than filtering the consumer's whole load list — which cost
      O(M^2) for a consumer with M distinct producers, paid in full by every
      candidate even when the next check would reject it. *)

  val of_kernel : Kernel.t -> t
  val kernel : t -> Kernel.t

  val candidates : t -> Tensor_id.t -> Kernel.Use.t list
  (** A consumer's ordinary-load edges, in first-occurrence order. *)

  val load_count : t -> Tensor_id.t -> int
  (** Ordinary loads of a producer across the kernel, SATURATED at 2: the total
      is an [int] and the per-value limits permit an aggregate past 2^31, which
      wraps negative under js_of_ocaml. Legality asks one versus more than one.
  *)
end

val site_in : Analysis.t -> Kernel.Use.t -> (Site.t, error) Core.result
(** [site] against an analysis already built. Same rule and same code — a caller
    checking many edges builds one [Analysis.t] instead of paying a pass per
    call. *)

val elaborate_site : Site.t -> (Expr.Value.t, error) Core.result
(** Rewrite an already-validated site. Performs no lookup and no fold. *)

val elaborate : Kernel.t -> Kernel.Use.t -> (Expr.Value.t, error) Core.result
(** The consumer's body with that one producer edge inlined, WITHOUT the
    consumer's own [Result_conversion] — the same contract as
    [Kernel.Value.body], so a caller materializing the result applies it exactly
    as the evaluator would. Derived: for inspection, budget validation and
    future lowering, never stored back into [Kernel.t]. *)
