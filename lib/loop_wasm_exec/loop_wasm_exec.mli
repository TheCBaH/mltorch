(** Runs a [Loop_program.t] as the WebAssembly {!Loop_ir.Loop_wasm} emits for
    it, under node as a subprocess over a blob holding every buffer. Test
    support for the emitter: the production lifecycle is separate.

    The result has [Loop_interp.run]'s shape, so {!Loop_ir.Loop_check}'s verdict
    (bitwise values, kind and payload on failures) applies unchanged. A missing
    [node] is a failure, never a skip. *)

open Loop_ir

type error =
  [ Loop_interp.error
  | `Wasm_host of string
    (** node could not start, rejected the module, or died: the emitter's
        contract is that this never happens *)
  | `Wasm_invalid of Wasm_check.Invalid.t
  | `Wasm_unsupported of Loop_wasm.error ]

val pp_error : Format.formatter -> [< error ] -> unit

module Host = Wasm_host
module Via_c = Wasm_c_host
module Node = Wasm_node

val node : string list ref
(** The command that runs a script: [["node"]]. *)

val exec :
  ?simd:bool ->
  ?outputs:(Tensor_id.t -> Tensor.packed option) ->
  Loop_program.t ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed Tensor_id.Map.t, error) Err.t
(** Lowers, runs and returns exactly the Output buffers. [outputs] behaves as
    {!Loop_interp.run}'s. *)

val exec_counted :
  ?outputs:(Tensor_id.t -> Tensor.packed option) ->
  Loop_program.t ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed Tensor_id.Map.t * (Loop_mark.t * int) list, error) Err.t
(** [exec] on a counting build ({!Loop_wasm.lower}'s [count_marks]): also the
    number of times each [Mark] statement ran, in [Loop_mark.all] order. *)

val executor : Loop_check.Executor.t
(** [exec] as a differential-harness executor, to {!Loop_check.install}. *)

val executor_simd : Loop_check.Executor.t
(** [executor] over modules lowered with [simd]: the strict 128-bit vector
    loops, which must produce the same bits as the scalar executor. *)
