(** WebAssembly emission for a [Loop_program.t]: one exported function
    [loop_kernel] over linear memory, structured [block]/[loop] control, no host
    call except the [Math] imports a program reaches, and no trap on a value.
    Output is deterministic: lowering twice is byte-identical, and names depend
    on position of first appearance, never on allocation ids.

    The function takes an [i32] [local] base pointer (the kernel's local arrays
    live there, zeroed by the program's own [Alloc]s) and then one [i32] pointer
    per program buffer, in program order, and returns [0] on success or [1]
    after writing the failure record at {!error_address}, whose layout is
    [Loop_wasm_failure]'s (the C backend's [struct model_error]). The module
    defines and exports its [memory]; the first [heap_base] bytes are its own
    (error record, per-channel quantization tables, local arrays), and a host
    places buffers at or above it. See the Wasm backend design doc in [.ai/] for
    the ABI, numeric and size policy. *)

type error =
  [ `Index_constant_out_of_range of int | `Local_arrays_too_large of int64 ]

val pp_error : Format.formatter -> [< error ] -> unit

type t = {
  module_ : Wasm.Module.t;
      (** defines the smallest memory that covers the static region *)
  local_base : int;
      (** the [local] pointer to pass: where the kernel's local arrays live *)
  mark_base : int option;
      (** where a counting build keeps one [i32] per [Loop_mark.all], in that
          order; [None] for the default build, which emits nothing for a mark *)
  heap_base : int;
      (** 16-aligned end of the module's own bytes, local arrays included: a
          host places buffers at or above it *)
  sites : Loop_failure.t array;
      (** the program's [Fail_if] failures in emission order, as
          [Loop_js_failure.sites]: a record that names a site indexes it *)
}

val function_name : string
(** ["loop_kernel"]. *)

val error_address : int
(** The byte address of the failure record: [0]. *)

val lower : ?count_marks:bool -> Loop_program.t -> (t, error) Err.t
(** [count_marks] (default [false]) makes each [Mark] statement bump its word at
    [mark_base], so a test can compare the counts with the interpreter's. The
    default build is unchanged byte for byte and no inference path calls the
    host per mark. *)

(** {1 Kernels for composition}

    A kernel is a function over symbolic callees, not yet a module: the whole
    model composes many of them and links once. *)

type kernel = {
  func : Wasm.Func.t;
      (** [(local, b0, ..) -> i32]; its calls are symbolic, to be renumbered by
          {!Loop_wasm_link.link} *)
  callees : Loop_wasm_runtime.Callee.t list;
  local_bytes : int64;  (** the local region it needs at [local] *)
  data : Wasm.Data.t list;
      (** constant tables, at the absolute addresses [table_alloc] returned *)
  sites : Loop_failure.t array;
}

val kernel :
  ?mark_base:int ->
  table_alloc:(bytes:int -> int) ->
  Loop_program.t ->
  (kernel, [> error ]) Err.t

val with_pages : t -> pages:int -> Wasm.Module.t
(** The module with [pages] initial memory pages; [Invalid_argument] below the
    static region's need. *)

val encode : t -> (string, Wasm_check.error) Err.t
(** [Wasm_encode.module_] of [module_]. *)
