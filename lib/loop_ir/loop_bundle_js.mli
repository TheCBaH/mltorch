(* The scoped JavaScript composition of a [Loop_bundle.t] (plan W3, design §3
   "JavaScript emission"): one wrapper entry point plus each distinct kernel
   in its own top-level declaration, calling into role-aware arena pools
   rather than allocating fresh buffers per call.

   Kernels are interned by complete canonical printed source
   ([Js_print.factory_body]), so two invocations of the same op shape (same
   ops/shapes, different global tensor ids -- kernel bodies are printed with
   POSITIONAL buffer names, never global ids) share one top-level function.
   Distinct kernels get a bundle-unique name (renaming [Loop_js.function_name],
   which every compiled kernel shares); their runtime PRELUDE helpers do not
   need the same isolation, since [Loop_js_runtime.helpers] is a fixed,
   name-keyed table -- two kernels needing the same helper always get the
   textually identical declaration, so the bundle's own prelude is their
   union, deduped by printed text.

   Every [Alloc_script.Kind.t] is a pool ([typed_array_name]); the runtime
   aliases an int64 tensor's storage as a [BigInt64Array], as
   [Loop_js_exec.int64_view] does. A borrowed/quantized/outside edge (no arena)
   is `Unbound_arena, except a [Borrowed] graph input or constant, which is an
   entry parameter ([externals]) used in place and never written.

   Verified (see the implementation tracker's W3 evidence log) against both
   the [chain] fixture and the real [mobilenetv2_050] archive: bit-exact
   (at F32 precision) against the native reference, run under node with
   manually populated pool typed arrays. Not yet wired into a committed test
   or CI target -- W4 provides the real [Arena.t]-backed runtime a permanent
   test should drive instead of hand-populated pools. *)

open Graph_ir

type error =
  [ `Kernel_refused of string
    (** the [kernel] hook declined a program: its own words *)
  | `Local_too_large of Tensor_id.t
  | `Pool_index_overflow of Tensor_id.t * int64
  | `Storage_units of Core.Storage_units.error
  | `Unbound_arena of Tensor_id.t ]

val pp_error : Format.formatter -> [< error ] -> unit

type t = {
  program : Js_ast.Program.t;
  distinct_kernels : int;
  pools : (Storage_script.Arena_id.t * Alloc_script.Kind.t) list;
      (** the wrapper entry's parameters, in order: one typed array per logical
          arena/kind pool the bundle touches *)
  externals : Tensor_id.t list;
      (** borrowed graph inputs/constants: the entry parameters after [pools],
          in order, each the caller's own typed array *)
  entry_name : Js_ident.t;
}

val build :
  ?kernel:(Loop_bundle.invocation -> (Js_ast.Program.t, string) result) ->
  Loop_bundle.t ->
  (t, error) Err.t
(** The entry returns [null] on success, or [[position, record]] for the first
    failing invocation: its index in [Loop_bundle.t.invocations] and the
    kernel's own failure record, decoded against that invocation's program.
    [kernel] makes each invocation's function instead of {!Loop_js.to_ast}: it
    must take the invocation program's buffers positionally, all of them, in
    order, and return failure records of the same shape. *)

val locate :
  Loop_bundle.t ->
  Tensor_id.t ->
  ( Storage_script.Arena_id.t * Alloc_script.Kind.t * int * int,
    [> error ] )
  Err.t
(** An arena-backed edge's arena, pool kind, element offset and element count --
    the same bounded figures [build]'s [subarray] views use. *)

val typed_array_name : Alloc_script.Kind.t -> string
(** The JavaScript typed array a pool of this kind is: [BigInt64Array] for
    [Int64], [Uint8Array] for [Int8_unsigned], and so on. *)

val pool_numel :
  Loop_bundle.t ->
  Storage_script.Arena_id.t ->
  Alloc_script.Kind.t ->
  (int, [> error ]) Err.t
(** A pool's element count, bounded to a 32-bit index. *)
