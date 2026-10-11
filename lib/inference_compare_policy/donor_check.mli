(** Resolution of an experiment's donor choices against the structural plan:
    every override names an output its unit actually writes, appears once, and
    selects a value that something downstream or a final output uses. An
    override on an unread value would silently do nothing, so it is refused. *)

open Inference_compare_schema

val validate :
  Inference_compare_plan.Structural.t ->
  Experiment.t ->
  (unit, Policy_error.t) Err.t
(** [Unknown_override] for an unknown unit/output, a duplicate, or a value with
    no downstream or final use. *)
