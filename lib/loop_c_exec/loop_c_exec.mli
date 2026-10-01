(** Runs a [Loop_program.t] as the C {!Loop_c} emits for it, compiled with the
    host compiler and executed as a subprocess over a blob holding every buffer.
    Test support for the emitter: the production lifecycle (mapped payloads, a
    whole-model unit) is separate.

    The result has [Loop_interp.run]'s shape, so {!Loop_check}'s verdict
    (bitwise values, kind and payload on failures) applies unchanged. *)

open Loop_ir

type error =
  [ Loop_interp.error
  | `C_compile of string  (** the compiler rejected the emitted source *)
  | `C_host of string
    (** the compiler could not start, the child died, or its result could not be
        decoded: the emitter's contract is that this never happens *)
  | `C_unsupported of Loop_c.error ]

val pp_error : Format.formatter -> [< error ] -> unit

val compiler : string list ref
(** The compiler command and the flags every compilation uses, before the source
    and output arguments. *)

module Proc = C_proc
module Host = C_host

val compile :
  string -> (string, [> `C_compile of string | `C_host of string ]) result
(** Compiles one translation unit with {!compiler}, memoised by source and
    flags, and returns the executable's path (removed at exit). *)

val source :
  Loop_program.t ->
  (string * string list, [> `C_unsupported of Loop_c.error ]) Err.t
(** The complete translation unit (a [main] driving one kernel call) and the
    list of buffer offsets in the blob, for inspection. *)

val exec :
  ?outputs:(Tensor_id.t -> Tensor.packed option) ->
  Loop_program.t ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed Tensor_id.Map.t, error) Err.t
(** Compiles (memoised by source), runs, and returns exactly the Output buffers.
    [outputs] behaves as {!Loop_interp.run}'s. *)

val executor : Loop_check.Executor.t
(** [exec] as a differential-harness executor, to {!Loop_check.install}. *)
