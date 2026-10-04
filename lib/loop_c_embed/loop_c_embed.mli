(** The whole-model C backend run in this process. The model is generated in the
    {!Loop_c_dialect.Compcert_scalar} dialect, with a one-argument [entry] that
    calls [model_run] on a single caller-owned region, and handed to a
    {!LOADER}: an embedded CompCert plus rivet's [Native_exec], or anything else
    that can run the text. No file is written for the model and no process is
    started.

    A {!prepared} model is the loaded code: immutable, shared by any number of
    {!context}s. A context owns its memory, one region holding the weights, the
    inputs, the outputs, the workspace and the failure record; it is used by one
    caller at a time. The weights are copied into each context. *)

open Graph_ir
open Loop_ir

module type LOADER = sig
  type loaded
  type error

  val pp_error : Format.formatter -> error -> unit

  val load : host_symbols:string list -> string -> (loaded, error) result
  (** Loads a unit of preprocessor-free C whose [long entry(void *io)] is the
      entry. [host_symbols] are the libc and libm functions it declares
      [extern], to bind to the host. *)

  val call :
    loaded ->
    io:(char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t ->
    (int64, error) result
  (** Calls [entry] with the address of [io]'s data. *)

  val close : loaded -> unit
end

type error =
  [ `Generate of Loop_bundle_c.error
  | `Load of string  (** the loader refused the unit *)
  | `Call of string  (** the loader could not run [entry] *)
  | `Missing_constant of Tensor_id.t
  | `Missing_input of Tensor_id.t
  | `Binding_mismatch of Kernel_eval.Binding_mismatch.t
  | `Io of string
  | `Run_failed of int  (** [model_run] returned something but 0 or 1 *)
  | `Inference_failed of int * Loop_interp.error
    (** the failing invocation's position, and the reference interpreter's row
        for the same failure *)
  | `Bad_failure_record of string
  | `Bad_output of string
  | `Closed ]

val pp_error : Format.formatter -> [< error ] -> unit

module Make (_ : LOADER) : sig
  type prepared
  type context

  val prepare :
    ?numerics:Loop_numerics.t -> Loop_bundle.t -> (prepared, error) Err.t
  (** Generates the unit (scalar only: no [vector]) and loads it. *)

  val context :
    prepared ->
    constants:(Tensor_id.t -> Tensor.packed option) ->
    (context, error) Err.t
  (** A fresh region with the weights written into it, from [constants]. *)

  val run :
    ?poison:bool ->
    context ->
    bind:(Tensor_id.t -> Tensor.packed option) ->
    (Tensor.packed list, error) Err.t
  (** Packs the inputs, calls the model and returns the graph outputs in graph
      output order. [poison] fills the workspace with a pattern first. *)

  val close_context : context -> unit
  (** Releases the region; idempotent. A later {!run} is [`Closed]. *)

  val close : prepared -> unit
  (** Releases the loaded code, after the last call in flight returns; contexts
      are untouched. *)

  val source : prepared -> string
  (** The unit that was loaded, adapter included. *)

  val bundle_c : prepared -> Loop_bundle_c.t
end
