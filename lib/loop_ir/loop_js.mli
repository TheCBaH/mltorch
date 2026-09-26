(** JavaScript emission for a [Loop_program.t]: one named function over typed
    arrays, ordinary [for] loops, no [eval] and no closure per element. Output
    is deterministic: emitting twice is byte-identical, and names depend on
    position of first appearance, never on allocation ids.

    The function takes one typed array per program buffer, in program order
    ([b0], [b1], ...), and returns [null] on success or a failure record
    [{ kind: "<name>", ... }]: a runtime failure is a value, never a host
    exception. *)

val typed_array : Loop_buffer.t -> string
(** The JavaScript typed-array constructor a buffer's argument must be: the
    storage cell, so F16 and BF16 are [Uint16Array] of raw bits and a quantized
    format is its integer array, decoded on load. *)

val function_name : string
(** The name of the emitted function. *)

val emit : Loop_program.t -> string
