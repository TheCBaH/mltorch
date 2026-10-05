(** The differential harness for direct SSA lowering: [Kernel_eval.run_plan] is
    the oracle, compared bitwise on values and by kind and payload on failures.
    A refusal of the request is a verdict of its own, never a pass, and a
    program the verifier rejects is a defect of the lowering, never a failure
    kind. Primitive counters are diagnostics, not part of the verdict; logical
    work marks are compared by {!marks}. *)

module Disagreement : sig
  type t =
    | Cfg_lowering of Ssa_ir.Ssa_cfg_lower.error
        (** The program has no graph form: a loop whose last increment may leave
            the index domain. *)
    | Error_kind of { reference : string; ssa : string }
    | Error_payload of string
        (** Same kind, different payload; the kind is named. *)
    | Invalid_cfg of Ssa_ir.Ssa_cfg_verify.diagnostic
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
  | Not_admitted of Ssa_ir.Ssa_numerics.Refusal.t
      (** binary32 execution was asked for and the plan's payloads have no
          binary32 decode *)
  | Refused of Ssa_lower.Ssa_unsupported.t

val pp_verdict : Format.formatter -> verdict -> unit

val compare :
  reference:(Tensor.packed Tensor_id.Map.t, Kernel_eval.error) Err.t ->
  ssa:(Tensor.packed Tensor_id.Map.t, Ssa_lower.Ssa_exec.error) Err.t ->
  verdict
(** The verdict alone, so a test can perturb one side and watch it flip. *)

val compare_kernel :
  reference:(Tensor.packed Tensor_id.Map.t, Kernel_eval.error) Err.t ->
  actual:(Tensor.packed Tensor_id.Map.t, Kernel_eval.error) Err.t ->
  verdict
(** {!compare} for an executor that reports the reference's own error rows (a
    generated-code host): bitwise values, kind and payload on failures. *)

val run :
  ?prepare:(Ssa_ir.Ssa_program.t -> Ssa_ir.Ssa_program.t) ->
  ?engine:Ssa_lower.Ssa_exec.engine ->
  Fusion_plan.t ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  verdict
(** [Kernel_eval.run_plan], and, if lowering accepts the plan, the SSA
    interpreter over the lowered program after [prepare] (the identity by
    default): a pass or a pipeline under test is checked against the reference
    exactly as the lowering is. *)

val run_f32 :
  ?prepare:(Ssa_ir.Ssa_program.t -> Ssa_ir.Ssa_program.t) ->
  Fusion_plan.t ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  verdict
(** The binary32 reading of a plan: the lowered program after
    {!Ssa_ir.Ssa_precision.to_f32} and [prepare], against the Loop interpreter
    run at binary32, which is the binary32 oracle of a scalar kernel. A plan
    {!Ssa_ir.Ssa_numerics.admit} refuses is {!Not_admitted}, never a
    disagreement. *)

val run_planned :
  ?alias:Ssa_ir.Ssa_effects.policy ->
  numerics:Ssa_ir.Ssa_numerics.t ->
  target:Ssa_ir.Ssa_target.t ->
  Fusion_plan.t ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  verdict * Ssa_ir.Ssa_plan.t option
(** The plan {!Ssa_ir.Ssa_plan.resolve} makes for the lowered kernel, run and
    judged twice. Against its own oracle (its vector program spelled out lane by
    lane), bit for bit, always. Against the reference it is defined to match: a
    binary64 plan the evaluator's bits; a binary32 plan under the ordered policy
    the Loop interpreter at binary32, bit for bit; a binary32 plan under the
    relaxed policy the binary64 evaluator within [1e-4 + 1e-4 |x|] (a reordered
    sum is not bitwise anything else). The resolved plan is returned so a caller
    can say what was planned; [None] for a kernel lowering refuses. *)

module Plan_comparison : sig
  type t = {
    ssa_f32 : bool;  (** the SSA plan runs in binary32 *)
    loop_f32 : bool;  (** the Loop plan does *)
    ssa_loops : int;  (** vector loops the SSA plan holds *)
    loop_loops : int;
    ssa_blocked : int;  (** row loops blocked *)
    loop_blocked : int;
  }
end

val compare_plans :
  ?alias:Ssa_ir.Ssa_effects.policy ->
  numerics:Ssa_ir.Ssa_numerics.t ->
  target:Ssa_ir.Ssa_target.t ->
  Fusion_plan.t ->
  Plan_comparison.t option
(** The decisions the SSA planner and the Loop planner make for the same kernel,
    target and policy, side by side. They are two implementations of one
    planning contract, so a disagreement is either a legitimate difference in
    what each analysis can prove or a defect in one of them; the comparison does
    not say which. [None] if either refuses to lower the kernel or no Loop
    target has the SSA target's name. *)

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
