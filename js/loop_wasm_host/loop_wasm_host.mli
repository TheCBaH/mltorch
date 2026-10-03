(** Runs a [Loop_bundle.t] as WebAssembly in this process: generate the module
    ({!Loop_ir.Loop_bundle_wasm}), compile and instantiate it through the host's
    [WebAssembly], place the weights once, then per input place the inputs, call
    [model_run] and copy the outputs out. js_of_ocaml only.

    [prepare] is synchronous ([new WebAssembly.Module]), which node and workers
    allow; a browser's main thread caps synchronous compilation of a large
    module, so {!prepare_async} compiles through the promise API instead.
    Inference itself is synchronous either way. Linear memory is allocated once
    and never grown, so views taken for a run stay valid for that run; they are
    retaken every run and released by {!dispose}. *)

open Graph_ir
open Loop_ir

type t

type error =
  [ `Generate of Loop_bundle_wasm.error
  | `Wasm_unavailable  (** the host has no [WebAssembly] object *)
  | `Wasm_compile of string
    (** the engine refused the module: a defect of the emitter, or a host that
        forbids Wasm compilation (a CSP without [wasm-unsafe-eval]) *)
  | `Wasm_instantiate of string
    (** the imports or memory were refused: for instance, memory beyond what the
        host can allocate *)
  | `Missing_constant of Tensor_id.t
  | `Missing_input of Tensor_id.t
  | `Binding_mismatch of Kernel_eval.Binding_mismatch.t
  | `Disposed
  | `Inference_failed of int * Loop_interp.error
    (** the failing invocation's position in the schedule, and the reference
        interpreter's row for the same failure *)
  | `Bad_failure_record of string
  | `Js_exception of string  (** [model_run] threw: a trap, a defect *) ]

val pp_error : Format.formatter -> [< error ] -> unit

val prepare :
  Loop_bundle.t ->
  constants:(Tensor_id.t -> Tensor.packed option) ->
  (t, error) Err.t

val prepare_async :
  Loop_bundle.t ->
  constants:(Tensor_id.t -> Tensor.packed option) ->
  ((t, error) Err.t -> unit) ->
  unit
(** Compiles with [WebAssembly.instantiate] and calls the continuation exactly
    once, from a later turn of the event loop. *)

val run :
  t ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed list, error) Err.t
(** The graph outputs in graph output order (a repeated output repeated), each a
    fresh tensor. The workspace is not cleared between runs: the schedule
    initialises what it reads. *)

val dispose : t -> unit
(** Drops the instance and its memory. Idempotent; {!run} then is [`Disposed].
*)

val bundle_wasm : t -> Loop_bundle_wasm.t
(** The generated module and layouts, for inspection and measurement. *)

val memory_bytes : t -> int
(** The byte length of the instance's linear memory. *)
