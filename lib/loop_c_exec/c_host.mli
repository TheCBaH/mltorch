(** The native host of the whole-model C backend: generate the two C files and
    the weights payload, compile them with the host compiler, then per input
    pack the inputs, launch the standalone binary and decode its outputs.

    Preparation and execution are separate: a {!prepared} artifact is compiled
    once and run for any number of inputs, and its binary is an ordinary
    executable that runs without this process (see {!command}). The compiler is
    an explicit argument vector, never a shell string; no shell is involved
    anywhere. Failure artifacts (sources, compiler log) stay in the artifact
    directory. *)

open Graph_ir
open Loop_ir

type prepared

type error =
  [ `Generate of Loop_bundle_c.error
  | `Compiler_unavailable of string
  | `Compile_failed of C_proc.status * string
    (** the compiler's exit status and its diagnostics; the sources remain in
        the artifact directory *)
  | `Missing_constant of Tensor_id.t
  | `Missing_input of Tensor_id.t
  | `Binding_mismatch of Kernel_eval.Binding_mismatch.t
  | `Quant_missing of Tensor_id.t
  | `Io of string
  | `Run_failed of C_proc.status * string
    (** the binary exited with a payload, allocation or usage status, or died on
        a signal *)
  | `Inference_failed of int * Loop_interp.error
    (** the failing invocation's position in the schedule, and the reference
        interpreter's row for the same failure *)
  | `Bad_failure_record of string
  | `Bad_output of string ]

val pp_error : Format.formatter -> [< error ] -> unit

val default_compiler : string list
(** [gcc -std=c11 -O2 -ffp-contract=off -fno-strict-aliasing -Wall -Wextra
     -Werror]: the argument vector before the sources. [LOOP_C_CFLAGS], if set,
    replaces [-O2] (for [-O0] and sanitizer runs). *)

val prepare :
  ?vector:Loop_target.t ->
  ?numerics:Loop_numerics.t ->
  ?compiler:string list ->
  dir:string ->
  Loop_bundle.t ->
  constants:(Tensor_id.t -> Tensor.packed option) ->
  (prepared, error) Err.t
(** Generates [model_infer.c] and [model_main.c] and [weights.bin] into [dir]
    (created if absent), and compiles them to [dir/model]. [constants] supplies
    every tensor of the bundle's [constants], read once here. [dir] is kept. *)

val run :
  ?poison:bool ->
  prepared ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed list, error) Err.t
(** Packs the bound inputs, runs the binary and returns the graph outputs in
    graph output order (one per entry, a repeated output repeated). [poison]
    fills the workspace with a pattern first, to expose a read of stale memory.
    The output file is a fresh path per run and is removed after decoding. *)

val command : prepared -> inputs:string -> outputs:string -> string list
(** The argument vector that runs the binary on already-packed files. *)

val write_inputs :
  prepared ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  path:string ->
  (unit, error) Err.t
(** Packs an inputs file, as {!run} does. *)

val read_outputs : prepared -> path:string -> (Tensor.packed list, error) Err.t
(** Validates and decodes an outputs file, as {!run} does. *)

val directory : prepared -> string
val executable : prepared -> string
val weights_path : prepared -> string
val bundle_c : prepared -> Loop_bundle_c.t

val compiler_identity : prepared -> string
(** The compiler's own version banner, captured at preparation. *)

val compiler : prepared -> string list
(** The compiler argument vector used, before the sources. *)
