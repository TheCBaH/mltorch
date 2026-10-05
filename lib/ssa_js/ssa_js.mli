(** Direct JavaScript emission from a structured SSA program: one named function
    over typed arrays, ordinary [for] loops, no [eval] and no closure per
    element, with the argument convention and failure-record shape of
    {!Loop_ir.Loop_js} so the same host runs it. A failure is a returned record,
    never a host exception.

    Value representations: an index is a [Number] (exact below 2^53, and the
    domain is 32 bits), an int64 a [BigInt] wrapped with [asIntN], a float a
    [Number] (a binary32 value rounded with [Math.fround] after each operation),
    a predicate a boolean. Output is deterministic: values are named by
    definition order, never by allocation id.

    Refused with a typed error, never emitted: vectors and masks, fused
    multiply-add, binary32 [erf], and an int64 to binary32 conversion (each has
    no JavaScript form that keeps its single rounding). *)

type error =
  [ `Unsupported_format of Ssa_ir.Ssa_id.Buffer.t * string
  | `Unsupported_operation of string ]

val pp_error : Format.formatter -> [< error ] -> unit
val function_name : string

val typed_array : Ssa_ir.Ssa_buffer.t -> string
(** The typed-array constructor a buffer's argument must be: the storage cell,
    so F16 and BF16 are [Uint16Array] of raw bits and a quantized format its
    integer array, decoded on load. *)

val arguments : Ssa_ir.Ssa_program.t -> Ssa_ir.Ssa_buffer.t list
(** The buffers the function takes, in declaration order: those an operation
    names. A declared buffer nothing touches is no argument. *)

val program :
  ?buffers:Ssa_ir.Ssa_buffer.t list ->
  ?sites:Loop_ir.Loop_failure.t array ->
  Ssa_ir.Ssa_program.t ->
  (Js_ast.Program.t * Loop_ir.Loop_failure.t array, error) result
(** The program and the failure sites its records are decoded against. Every
    identifier is bound ([Js_check.closed] runs before it returns). [buffers]
    (default {!arguments}) are the arguments taken, in order. [sites] is a
    failure-site table to decode against, as for {!Ssa_c.kernel}. *)
