(** The module layout every Wasm emitter shares: the error record, constant
    tables, local region and host buffers, in that order, and the linked module
    a kernel becomes. *)

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
  precision : Loop_numerics.Precision.t;
      (** the working precision the kernel was lowered in *)
}

type kernel = {
  func : Wasm.Func.t;
      (** [(local, b0, ..) -> i32]; its calls are symbolic, to be renumbered by
          {!Loop_wasm_link.link} *)
  callees : Loop_wasm_runtime.Callee.t list;
  local_bytes : int64;  (** the local region it needs at [local] *)
  data : Wasm.Data.t list;
      (** constant tables, at the absolute addresses [table_alloc] returned *)
  sites : Loop_failure.t array;
  precision : Loop_numerics.Precision.t;  (** the working precision it uses *)
  refusal : Loop_numerics.Refusal.t option;
      (** why an fp32 policy left it binary64, when admission was the reason *)
}

val lower_with :
  ?numerics:Loop_numerics.t ->
  ?count_marks:bool ->
  (mark_base:int option -> table_alloc:(bytes:int -> int) -> (kernel, 'e) Err.t) ->
  (t, 'e) Err.t
(** {!lower} for a kernel some other emitter makes: [produce] gets the mark base
    and the constant-table allocator the module's layout fixes, and its kernel
    is linked, given a memory and exported exactly as {!lower} does. *)
