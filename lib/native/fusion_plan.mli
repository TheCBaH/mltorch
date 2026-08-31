(* Placement beside an unchanged kernel. [Kernel.t] is never rewritten: fusion
   changes where a value is computed, not what it means.

   Placement is TWO facts, not one enum — which dependency EDGES are virtual,
   and which VALUES need stores. An externally live producer is both: virtual
   for its consumer and still stored for whoever else needs it. A single
   [Materialized | Virtual] per value cannot say that. *)

module Rejection : sig
  type t =
    | Budget_exceeded of {
        use : Kernel.Use.t;
        error : Kernel_elab.error Err.Error.t;
      }
    | Intrinsic_use of Kernel.Use.t
    | Multiple_uses of { producer : Tensor_id.t; at_least : int }
        (** A LOWER BOUND, not a count. The planner's counter saturates at two,
            because the cross-body total is an [int] and the per-value limits
            permit a mathematical aggregate past 2^31 — which wraps negative
            under js_of_ocaml and would read as unique-use. Legality only asks
            one versus more than one, so nothing needs the exact figure, and the
            field is named and printed so the diagnostic does not claim it. *)
    | Non_pointwise_use of {
        use : Kernel.Use.t;
        reason : Kernel_elab.error Err.Error.t;
            (** The [Err.Error.t] is RETAINED, not unwrapped to its [.kind]:
                stripping it at the call site would discard the detection
                backtrace, and CLAUDE.md requires a named conversion wherever a
                wrapper is dropped deliberately. Nothing here needs it dropped.
            *)
      }
    | Regional_computation of Kernel.Use.t
    | Overlaps_selected of { rejected : Kernel.Use.t; selected : Kernel.Use.t }
    | Reducing_consumer of Kernel.Use.t

  val pp : Format.formatter -> t -> unit
end

module Decision : sig
  type t = private
    | Reject of { producer : Tensor_id.t; reason : Rejection.t }
    | Virtualize of { use : Kernel.Use.t; also_stored : bool }

  val pp : Format.formatter -> t -> unit
end

type t = private {
  kernel : Kernel.t;
      (** The plan is valid ONLY for this kernel. [Tensor_id.t] is a small
          integer local to a graph, so a plan built for kernel A is otherwise
          silently applicable to kernel B: with overlapping ids the evaluator
          would skip stores or virtualize unrelated dependencies, and without
          them it would return an incomplete result map. Carrying the immutable
          kernel makes the mismatch unconstructable, which beats a fingerprint
          that has to be remembered at every boundary. *)
  virtual_uses : Kernel.Use.Set.t;
  stores : Tensor_id.Set.t;
}

val default : Kernel.t -> t
(** Every value stored, no use virtual. *)

val plan : Kernel.t -> t * Decision.t list
(** The only other way to build a [t]. [Decision.t] is private and there is no
    [of_decisions], so an unchecked report cannot be turned back into a plan — a
    private record buys nothing when its public constructor takes public data.

    Legality for the first class:

    - the edge satisfies [Kernel_elab.site] — one occurrence, shape-compatible
      pointwise coordinate. Called, not restated, so the two cannot drift;
    - the producer has exactly one ordinary load across all values and no
      intrinsic reference — [substitute_loads] cannot touch a descriptor;
    - the consumer has no reduction binders and no intrinsics. Without this a
      consumer containing a reduction whose producer load happened to use only
      output coordinates would pass the coordinate test and be quietly admitted
      to a class named "pointwise";
    - the elaborated body is within budget.

    Selection is deterministic in both dimensions, since candidates can overlap
    across consumers (a chain A->B->C) and within one (C = add A B): consumers
    are visited in [Kernel.values] topological order, and a consumer's incoming
    candidates in first-occurrence order in [Expr.Fold.loads] — lexical in the
    body, not set or id order, which would make the plan depend on a container
    rather than the expression. The first legal candidate wins; a later one
    touching an already-selected value is [Overlaps_selected].

    Reporting is equally fixed: one decision per produced value with at least
    one downstream dependency, ordered by producer topology, and the first
    failing check in the planner's explicit precedence order is the reason.

    An externally live producer is NOT a rejection — it is virtualized for its
    consumer and keeps its store, with [also_stored = true]. *)
