(** The differential harness: [Kernel_eval.run_plan] is the oracle for the Loop
    IR, compared bitwise on values and by kind AND payload on failures. A
    refusal is a verdict of its own, never a pass. *)

module Disagreement : sig
  type t =
    | Error_kind of { reference : string; loop : string }
    | Error_payload of string
        (** Same kind, different payload; the kind is named. *)
    | Executor of t
        (** A further executor ({!Executor}) disagreed with the reference. *)
    | Host_failure of string
        (** A further executor's host refused or threw: its emitter's contract
            says this never happens, so it is a defect, not a failure kind. *)
    | Loop_only_failed of string
    | Missing_output of Tensor_id.t
    | Reference_only_failed of string
    | Unexpected_output of Tensor_id.t
    | Value_mismatch of Tensor_id.t

  val pp : Format.formatter -> t -> unit
end

type verdict =
  | Agree
  | Agree_on_failure of string
      (** Both executors failed, with the same kind and payload. *)
  | Disagree of Disagreement.t
  | Refused of Loop_unsupported.t

val pp_verdict : Format.formatter -> verdict -> unit

val kind : Kernel_eval.error -> string
(** The constructor of an error, by name; message text is never compared. Rows
    the Loop IR has no vocabulary for are ["other"]. *)

val tensors_equal : Tensor.packed -> Tensor.packed -> bool
(** Bitwise, and exact for int64 (a decoded-float comparison would not be past
    [2^53]). Every non-NaN float is compared by its bits, so signed zeros
    differ; a NaN equals any NaN, since a JavaScript engine does not portably
    preserve a payload or a sign. *)

val compare :
  reference:(Tensor.packed Tensor_id.Map.t, Kernel_eval.error) Err.t ->
  loop:(Tensor.packed Tensor_id.Map.t, Loop_interp.error) Err.t ->
  verdict
(** The verdict alone, so a test can perturb one side and watch it flip. *)

(** A further executor of a lowered program: the generated JavaScript run
    in-process ([Loop_js_exec], js_of_ocaml only). Native code has none. *)
module Executor : sig
  type error =
    [ Loop_interp.error | `Js_compile of string | `Js_exception of string ]

  type t =
    Loop_program.t ->
    bind:(Tensor_id.t -> Tensor.packed option) ->
    (Tensor.packed Tensor_id.Map.t, error) Err.t
end

val install : Executor.t option -> unit
(** The executor [run] uses when none is passed. A suite that can run generated
    code installs it once, so every existing fixture is then checked by it too.
*)

val run :
  ?exec:Executor.t ->
  Fusion_plan.t ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  verdict
(** Runs [Kernel_eval.run_plan] and, if lowering accepts the plan, the Loop
    interpreter, and then [exec] (else the installed executor) if there is one.
    Every executor must agree with the reference; the first disagreement is the
    verdict. *)
