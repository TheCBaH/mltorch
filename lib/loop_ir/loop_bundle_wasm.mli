(** The whole-model WebAssembly module for a {!Loop_bundle.t}: every distinct
    kernel as a function, then [model_run], the fixed schedule that binds each
    invocation's buffers to payload and workspace offsets. It performs no I/O,
    allocates nothing and keeps no mutable state. It is the Wasm counterpart of
    {!Loop_bundle_c}, and shares its payload layouts and workspace plan, so the
    weights, inputs and outputs files are byte-identical to the C backend's
    (only the identity in their headers differs).

    [model_run(weights, inputs, workspace, outputs) -> i32] takes absolute
    addresses of the four regions in the module's one linear memory: the first
    three hold the payload files (header included) and the workspace, the last
    receives the outputs file's tensors. It returns [0], or [1] after writing
    the failure record at address [0] with the failing invocation's position.

    Admitted: separate layout, borrowed constants and inputs. Everything else is
    a typed refusal before any byte is made; there is no fallback. *)

open Graph_ir

(** Where the four regions go by default; the module's memory is sized for it. A
    host may place them elsewhere inside memory it provides. *)
module Placement : sig
  type t = {
    weights : int;
    inputs : int;
    workspace : int;
    outputs : int;
    total : int;  (** bytes of memory the placement needs *)
  }
end

type stats = {
  invocations : int;
  distinct_kernels : int;
  module_bytes : int;
  numerics : Loop_numerics.t;  (** the policy the module was built under *)
  f32_invocations : int;  (** invocations of a binary32 kernel *)
  f32_kernels : int;  (** distinct binary32 kernels *)
  fp32_refusals : (Loop_numerics.Refusal.t * int) list;
      (** invocations a [Simd_fp32_*] policy left binary64 because admission
          refused them, by reason: coverage is never silently partial *)
}

type t = {
  module_ : Wasm.Module.t;
  bytes : string;  (** the encoded module *)
  weights : C_payload_layout.t;
  inputs : C_payload_layout.t;
  outputs : C_payload_layout.t;
  identity : string;  (** 16 bytes: the digest of [bytes] *)
  workspace : C_workspace_plan.t;
  placement : Placement.t;
  stats : stats;
}

type error =
  [ C_payload_layout.error
  | C_workspace_plan.error
  | Loop_wasm.error
  | Wasm_check.error
  | `Memory_over_policy of int64
    (** the placement or an offset exceeds the 2 GiB memory policy *)
  | `Missing_signature of Tensor_id.t
  | `Output_outside_workspace of Tensor_id.t ]

val pp_error : Format.formatter -> [< error ] -> unit

val build :
  ?vector:Loop_target.t ->
  ?numerics:Loop_numerics.t ->
  Loop_bundle.t ->
  (t, error) Err.t
(** [vector] lowers the independent loops the strict vectorizer finds for the
    target to 128-bit SIMD ({!Loop_wasm.lower}); the module then needs the
    [simd128] feature. *)

val default_config : Storage_script.Config.t
(** The configuration admitted: separate layout, borrowed constants and inputs.
    Pass it to {!Loop_bundle.build}. *)
