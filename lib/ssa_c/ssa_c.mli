(** Direct C emission from a structured SSA program: one [static int] function
    over typed pointers with the same shape, failure-record ABI and runtime
    helpers as {!Loop_ir.Loop_c}, so every existing host runs it, and no Loop IR
    in between. Output is deterministic: values are named by definition order,
    never by allocation id.

    Types are the program's own: a binary32 value is a [float], a fused
    operation is [fma]/[fmaf], a vector is a GCC/Clang generic vector of its
    logical width and a mask a vector of 64-bit lanes. Nothing decides precision
    or contraction here; that was settled in the program. Only formats with a C
    implementation are admitted: a quantized buffer is a typed refusal. *)

type error = Ssa_c_ctx.error

val pp_error : Format.formatter -> [< error ] -> unit

val arguments : Ssa_ir.Ssa_program.t -> Ssa_ir.Ssa_buffer.t list
(** The buffers the kernel takes, in declaration order: those an operation
    names. A declared buffer nothing touches is no argument; validating its
    binding is the caller's. *)

val kernel :
  name:string ->
  Ssa_ir.Ssa_program.t ->
  (Loop_ir.Loop_c.t * Loop_ir.Loop_failure.t array, error) result
(** The kernel in the shape the C hosts take, and the failure sites its records
    are decoded against. *)
