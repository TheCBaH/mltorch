(** Runs a [Loop_program.t] as the JavaScript [Loop_js] emits for it, in this
    process, over the very storage of the bound tensors: an input is passed as
    is and an output is allocated here and written in place, so nothing is
    copied. js_of_ocaml only.

    The result has [Loop_interp.run]'s shape, so the differential harness's
    verdict (bitwise values, kind and payload on failures) applies unchanged. *)

open Loop_ir

type compiled

type error =
  [ Loop_interp.error
  | `Js_compile of string
    (** the engine refused the printed source: a [SyntaxError], which means the
        printer is wrong, or a host that forbids [new Function] (a browser CSP
        without [unsafe-eval]) *)
  | `Js_exception of string
    (** a host exception escaped the kernel, or a failure record could not be
        decoded: the emitter's contract is that this never happens *) ]

val pp_error : Format.formatter -> [< error ] -> unit

val compile_source :
  string -> (Js_of_ocaml.Js.Unsafe.any, [> `Js_compile of string ]) Err.t
(** [new Function(source)()] for a factory body, memoised by the source. Exposed
    so the refusal of a source the engine cannot parse can be tested. *)

val compile_as :
  Loop_program.t -> string -> (compiled, [> `Js_compile of string ]) Err.t
(** [compile] with the factory body supplied instead of printed: for a test that
    needs a kernel the emitter would never write, such as one that throws. *)

val compile : Loop_program.t -> (compiled, [> `Js_compile of string ]) Err.t
(** [new Function(factory_body)()]. Memoised by the printed source, which is
    deterministic, so identical programs share one function. *)

val run :
  compiled ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed Tensor_id.Map.t, error) Err.t
(** Input buffers are validated by [Kernel_eval.check_binding] and against the
    typed array the emitter declares ([`Binding_mismatch] otherwise); Output
    buffers are allocated fresh; the result holds exactly the Output buffers. A
    non-[null] return is decoded with [Loop_js_failure] into the row
    [Loop_interp] would report. *)

val exec :
  Loop_program.t ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed Tensor_id.Map.t, error) Err.t
(** [compile] then [run]. *)

val int64_view :
  (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  Js_of_ocaml.Js.Unsafe.any
(** A [BigInt64Array] over an int64 Bigarray's own bytes (jsoo stores it as an
    [Int32Array] of [lo, hi] pairs). Exposed so the alias can be tested against
    the Bigarray directly. *)

val little_endian : bool
(** The alias of an int64 buffer relies on the host byte order; a big-endian
    host is refused ([`Js_exception]), not byte-swapped. *)
