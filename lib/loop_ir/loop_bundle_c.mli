(** The whole-model C translation unit for a {!Loop_bundle.t} (the
    [model_infer.c] of the C backend design): every distinct kernel, then
    [model_run], the fixed schedule that binds each invocation's buffers to
    payload and workspace offsets. It performs no I/O and no allocation and
    keeps no mutable global state.

    Admitted: separate layout, borrowed constants and inputs, output-only
    retention, no quantized format. Everything else is a typed refusal before
    any text is made; there is no fallback. *)

open Graph_ir

type stats = { invocations : int; distinct_kernels : int; source_bytes : int }

type t = {
  source : string;  (** [model_infer.c] *)
  weights : C_payload_layout.t;
  inputs : C_payload_layout.t;
  outputs : C_payload_layout.t;
  identity : string;  (** 16 bytes, the artifact identity in the headers *)
  workspace : C_workspace_plan.t;
  stats : stats;
}

type error =
  [ C_payload_layout.error
  | C_workspace_plan.error
  | Loop_c.error
  | `Missing_signature of Tensor_id.t
  | `Output_outside_workspace of Tensor_id.t ]

val pp_error : Format.formatter -> [< error ] -> unit

val build : ?vector:Loop_target.t -> Loop_bundle.t -> (t, error) Err.t
(** [vector] vectorizes each kernel's independent loops for the target under the
    strict contract ({!Loop_c.kernel}); the unit then uses GCC/Clang generic
    vectors. *)

val default_config : Storage_script.Config.t
(** The configuration the C backend admits: separate layout, borrowed constants
    and inputs. Pass it to {!Loop_bundle.build}. *)
