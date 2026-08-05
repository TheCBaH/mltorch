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
    | Shape
    | Format
    | Quant
    | Storage_length of { expected : int; actual : int }

  type t = { id : Tensor_id.t; kind : kind }

  val pp : Format.formatter -> t -> unit
end

type error =
  [ Expr.Eval.error
  | `Unbound_input of Tensor_id.t
  | `Binding_mismatch of Binding_mismatch.t
  | `Unknown_value of Tensor_id.t ]

val pp_error : Format.formatter -> [< error ] -> unit

val value_at :
  Kernel.t ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  Tensor_id.t ->
  int Expr.Coord.t ->
  (float, error) Core.result
(** RECURSIVE: an internal load is resolved by evaluating its producer at the
    load coordinate. The result INCLUDES the value's result conversion.
    Bounds-checks its own coordinate against the value's shape first, since
    [Expr.Eval] treats the output coordinate as given and checks only what a
    load reads. Memoised per call. *)

val run :
  Kernel.t ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed Tensor_id.Map.t, error) Core.result
(** BUFFER-BASED: materialises every value in topological order, and an internal
    load READS the already-materialised producer. Returns every materialised
    value keyed by id, mirroring [Stage_program.ground] so tests can inspect
    intermediates.

    That this reads buffers is the point. An evaluator which always recursed
    into producer bodies would already BE virtual execution, so comparing it
    against a fused run would compare two identical paths and establish nothing
    about buffer elimination. *)
