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

val compile_kernel :
  sites:Loop_failure.t array ->
  Loop_program.t ->
  string ->
  (compiled, [> `Js_compile of string ]) Err.t
(** [compile_as] for a kernel some other emitter made for the program's buffers:
    [sites] is the failure-site table its records are decoded against, and the
    program supplies the argument arrays and their validation. *)

val compile_as :
  Loop_program.t -> string -> (compiled, [> `Js_compile of string ]) Err.t
(** [compile] with the factory body supplied instead of printed: for a test that
    needs a kernel the emitter would never write, such as one that throws. *)

val compile : Loop_program.t -> (compiled, [> `Js_compile of string ]) Err.t
(** [new Function(factory_body)()]. Memoised by the printed source, which is
    deterministic, so identical programs share one function. *)

val run :
  ?outputs:(Tensor_id.t -> Tensor.packed option) ->
  compiled ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed Tensor_id.Map.t, error) Err.t
(** Input buffers are validated by [Kernel_eval.check_binding] and against the
    typed array the emitter declares ([`Binding_mismatch] otherwise). An Output
    buffer is allocated fresh, unless [outputs] binds it, in which case it is
    validated the same way and zero-filled first (matching [Loop_interp.run]'s
    own [?outputs] contract). The result holds exactly the Output buffers. A
    non-[null] return is decoded with [Loop_js_failure] into the row
    [Loop_interp] would report. *)

val exec :
  ?outputs:(Tensor_id.t -> Tensor.packed option) ->
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

val failure : Loop_program.t -> Js_of_ocaml.Js.Unsafe.any -> error
(** Decodes a failure record against [program]'s own failure sites, as [run]
    does for a single kernel: for a caller that composes several programs. *)

val storage : Tensor.packed -> Js_of_ocaml.Js.Unsafe.any
(** The typed array holding a tensor's own storage (an int64 tensor's is the
    [Int32Array] of [lo, hi] pairs; see [int64_view]). *)

val argument :
  Loop_buffer.t ->
  Tensor.packed ->
  ( Js_of_ocaml.Js.Unsafe.any,
    [> `Binding_mismatch of Kernel_eval.Binding_mismatch.t ] )
  result
(** The typed array [buffer]'s kernel takes for [tensor]'s storage: the
    [BigInt64Array] alias for int64, the storage itself otherwise. *)
