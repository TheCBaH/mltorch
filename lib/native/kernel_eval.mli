(* The Kernel reference interpreter. See .ai/native_kernel_dsl_design.md.

   One rule governs where the result conversion is applied, and it is why the
   two entry points below are named apart internally:

     a LOGICAL VALUE includes its result conversion; a STORED BODY does not.

   [Kernel.Value.body] is unconverted; anything observing the value — a store, a
   load of it, [value_at] — observes [Result_conversion.apply] around it,
   exactly once. Applying it zero times or twice is the mistake this split
   exists to make impossible; [Round_f32] happens to be idempotent today, and
   the representation is meant to grow past that. *)

module Binding_mismatch : sig
  type kind =
    | Format
    | Quant
    | Shape
    | Storage_length of { expected : int; actual : int }

  type t = { id : Tensor_id.t; kind : kind }

  val pp : Format.formatter -> t -> unit
end

type error =
  [ Expr.Eval.error
  | `Binding_mismatch of Binding_mismatch.t
  | `Recursion_too_deep of int
  | `Unbound_input of Tensor_id.t
  | `Unknown_value of Tensor_id.t ]

val pp_error : Format.formatter -> [< error ] -> unit
(** [`Recursion_too_deep] carries [Kernel.Limits.Hard.eval_recursion]. It is a
    RUNTIME guard because that is where the recursion is: a producer transition
    costs far more stack than an expression level, so no static count of levels
    bounds it, and the buffer-based [run] never recurses at all — rejecting a
    deep DAG at construction would refuse kernels that execute perfectly well.
*)

val value_at :
  Kernel.t ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  Tensor_id.t ->
  int Expr.Coord.t ->
  (float, error) Err.t
(** RECURSIVE: an internal load is resolved by evaluating its producer at the
    load coordinate. The result INCLUDES the value's result conversion.
    Bounds-checks its own coordinate against the value's shape first, since
    [Expr.Eval] treats the output coordinate as given and checks only what a
    load reads. Memoised per call. *)

val run :
  Kernel.t ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed Tensor_id.Map.t, error) Err.t
(** BUFFER-BASED: materialises every value in topological order, and an internal
    load READS the already-materialised producer. Returns every materialised
    value keyed by id, mirroring [Stage_program.ground] so tests can inspect
    intermediates. Equivalent to [run_plan (Fusion_plan.default k)].

    That this reads buffers is the point. An evaluator which always recursed
    into producer bodies would already BE virtual execution, so comparing it
    against a fused run would compare two identical paths and establish nothing
    about buffer elimination. *)

val run_plan :
  Fusion_plan.t ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed Tensor_id.Map.t, error) Err.t
(** Execution under a placement. There is no second kernel argument: the plan
    carries the one it was built for, so a plan cannot be applied to a kernel
    whose ids merely happen to overlap.

    1. every value in [stores] is materialised, in topological order; 2. a load
    on an edge in [virtual_uses] evaluates the producer at the concrete load
    coordinate, conversion included exactly once — the [value_at] path; 3. every
    other internal load reads the materialised producer; 4. the returned map
    contains exactly [stores].

    A producer that is both virtual and stored is materialised AND recursed into
    by its nominated consumer, which is why virtualization is keyed by edge. *)
