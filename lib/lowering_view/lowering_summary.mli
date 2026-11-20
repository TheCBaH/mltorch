(** One node's overlay facts, cheap enough to take of every invocation of a
    model: what the default SSA route (planned for [simd_fp32_relaxed], 128-bit
    vectors, as {!Lowering_view} defaults it) made of the node, and the Coral
    estimate of its cycles. Each field is data; rendering is the edge's. *)

open Graph_ir
module Metric = Lowering_metric

type t = {
  node : Node_id.t;
  precision : Loop_ir.Loop_numerics.Precision.t option;
      (** the plan's working precision; [None] when there is no plan *)
  vectorized : bool;  (** at least one loop is a vector loop *)
  vector_loops : int;
  scalar_loops : int;
  ssa_refusal : Lowering_artifact.Refusal.t option;
      (** why the invocation has no SSA plan *)
  coral_cycles : float option;  (** [None] when the Coral route refuses it *)
}

val metric : t -> Metric.t -> float option
(** The metric's value; [None] when the node has none. *)

val of_invocation :
  granted:(Node_id.t -> bool) -> Loop_ir.Loop_bundle.invocation -> t
(** Total: a refusal is a field of the result. A granted node (see
    {!Lowering_view.of_invocation}) is [Granted] with nothing else. *)

val of_bundle : granted:(Node_id.t -> bool) -> Loop_ir.Loop_bundle.t -> t list
(** One per invocation, in the bundle's order. *)
