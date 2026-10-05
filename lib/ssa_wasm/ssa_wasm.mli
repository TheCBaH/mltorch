(** Direct WebAssembly emission from a structured SSA program: one exported
    function [loop_kernel] over linear memory, structured [block]/[loop]
    control, no host call except the [Math] imports a program reaches, and no
    trap on a value, with the argument convention, memory layout, helpers and
    failure-record ABI of {!Loop_ir.Loop_wasm}, so the same host runs it.

    Values live in locals typed by the program's own types: an index is an
    [i32], an int64 an [i64], a binary32 value an [f32], a predicate an [i32],
    and a vector or mask a group of [v128] registers (two [f64] lanes or four
    [f32] lanes each). A loop's carried values transfer through the operand
    stack, so the rebinding is simultaneous without a temporary. A fused
    operation is the one relaxed-SIMD [f32x4.relaxed_madd], only where the plan
    asks for it; a scalar fused operation has no WebAssembly form and is
    refused. *)

type error = Ssa_wasm_ctx.error

val pp_error : Format.formatter -> [< error ] -> unit

val arguments : Ssa_ir.Ssa_program.t -> Ssa_ir.Ssa_buffer.t list
(** The buffers the function takes (after the [local] pointer), in declaration
    order: those an operation names. *)

val kernel :
  relaxed_madd:bool ->
  table_alloc:(bytes:int -> int) ->
  Ssa_ir.Ssa_program.t ->
  (Loop_ir.Loop_wasm.kernel, error) result
(** A kernel for composition into a module. *)

val lower :
  ?relaxed_madd:bool ->
  ?numerics:Loop_ir.Loop_numerics.t ->
  Ssa_ir.Ssa_program.t ->
  (Loop_ir.Loop_wasm.t, error) result
(** A whole module with the layout {!Loop_ir.Loop_wasm.lower} makes.
    [relaxed_madd] (default [false]) says the plan was made for relaxed SIMD:
    [numerics] defaults to the reference policy. *)
