(** The differential harness for direct SSA lowering: [Kernel_eval.run_plan] is
    the oracle, compared bitwise on values and by kind and payload on failures.
    A refusal of the request is a verdict of its own, never a pass, and a
    program the verifier rejects is a defect of the lowering, never a failure
    kind. Primitive counters are diagnostics, not part of the verdict; logical
    work marks are compared by {!marks}. *)

module Disagreement : sig
  type t =
    | Error_kind of { reference : string; ssa : string }
    | Error_payload of string
        (** Same kind, different payload; the kind is named. *)
    | Invalid_program of Ssa_ir.Ssa_verify.diagnostic
    | Missing_output of Tensor_id.t
    | Reference_only_failed of string
    | Ssa_only_failed of string
    | Unexpected_output of Tensor_id.t
    | Value_mismatch of Tensor_id.t

  val pp : Format.formatter -> t -> unit
end

type verdict =
  | Agree
  | Agree_on_failure of string
      (** Both executors failed, with the same kind and payload. *)
  | Disagree of Disagreement.t
  | Refused of Ssa_lower.Ssa_unsupported.t

val pp_verdict : Format.formatter -> verdict -> unit

val compare :
  reference:(Tensor.packed Tensor_id.Map.t, Kernel_eval.error) Err.t ->
  ssa:(Tensor.packed Tensor_id.Map.t, Ssa_lower.Ssa_exec.error) Err.t ->
  verdict
(** The verdict alone, so a test can perturb one side and watch it flip. *)

val run : Fusion_plan.t -> bind:(Tensor_id.t -> Tensor.packed option) -> verdict
(** [Kernel_eval.run_plan], and, if lowering accepts the plan, the SSA
    interpreter over the lowered program. *)

type marks = {
  emitters : int;
  keys : int;
  locals : int;
  reductions : int;
  scan_updates : int;
  scans : int;
}
(** The logical work of one run, per {!Ssa_ir.Ssa_mark}. *)

val marks :
  Fusion_plan.t ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  (marks * marks, [ `Refused of Ssa_lower.Ssa_unsupported.t | `Failed ]) result
(** The marks of the SSA run and of the Loop interpreter's run, in that order,
    for a plan both lower and run. *)
