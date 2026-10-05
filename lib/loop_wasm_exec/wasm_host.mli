(** The native host of the whole-model Wasm backend, over node: generate the
    module and the weights payload, then per input pack the inputs, run
    [model_run] in a node subprocess and decode its outputs. The Wasm
    counterpart of [Loop_c_exec.Host], with the same payload files; a missing
    [node] is an error, never a skip.

    Preparation and execution are separate: a {!prepared} artifact is built once
    and run for any number of inputs. Failure artifacts stay in the artifact
    directory. *)

open Graph_ir
open Loop_ir

type prepared

type error =
  [ `Generate of Loop_bundle_wasm.error
  | `Missing_constant of Tensor_id.t
  | `Missing_input of Tensor_id.t
  | `Binding_mismatch of Kernel_eval.Binding_mismatch.t
  | `Io of string
  | `Node_unavailable of string
  | `Run_failed of Loop_c_exec.Proc.status * string
    (** node exited abnormally: it could not compile or instantiate the module,
        or [model_run] trapped *)
  | `Inference_failed of int * Loop_interp.error
    (** the failing invocation's position in the schedule, and the reference
        interpreter's row for the same failure *)
  | `Bad_failure_record of string
  | `Bad_output of string ]

val pp_error : Format.formatter -> [< error ] -> unit

val node : string list ref
(** The command that runs a script: [["node"]]. *)

val prepare :
  ?vector:Loop_target.t ->
  ?numerics:Loop_numerics.t ->
  ?kernel:
    (table_alloc:(bytes:int -> int) ->
    Loop_bundle.invocation ->
    (Loop_wasm.kernel, string) result) ->
  dir:string ->
  Loop_bundle.t ->
  constants:(Tensor_id.t -> Tensor.packed option) ->
  (prepared, error) Err.t
(** Generates [model.wasm] and [weights.bin] into [dir] (created if absent).
    [constants] supplies every tensor of the bundle's [constants], read once
    here. [dir] is kept. *)

val run :
  ?poison:bool ->
  ?repeat:int ->
  prepared ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed list, error) Err.t
(** Packs the bound inputs, runs [model_run] and returns the graph outputs in
    graph output order (a repeated output repeated). [poison] fills the
    workspace and the outputs region with a pattern first, to expose a read of
    stale memory. [repeat] runs the schedule that many further times on the same
    instance, dirty workspace included, and checks each run's outputs equal the
    first's. *)

val timings : prepared -> (string * float) list
(** The phases of the last {!run}, in milliseconds: node-side [compile],
    [instantiate], [copy_in], [first_run], [warm_run] (median of [repeat], when
    asked) and [copy_out]. *)

val bundle_wasm : prepared -> Loop_bundle_wasm.t
val directory : prepared -> string
val module_path : prepared -> string
val weights_path : prepared -> string
