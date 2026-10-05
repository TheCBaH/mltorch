(** Structured SSA to the Loop IR: a migration instrument that lets the existing
    emitters and interpreter run an SSA program, not the final consumer
    interface. Every SSA value becomes a Loop temporary, so the result has more
    locals and copies than a direct emitter would produce.

    A loop's carried values are copied as one parallel transfer (every yield is
    snapshotted before any parameter is overwritten), a checked index operation
    becomes the [Fail_if] of its own overflow at the same site, and a coordinate
    load is guarded by the bounds check the SSA load performs. *)

val loop_buffer : Ssa_ir.Ssa_buffer.t -> Loop_ir.Loop_buffer.t
(** A declared buffer as the Loop IR names it: its signature, role and id. *)

val convert : Ssa_ir.Ssa_program.t -> Loop_ir.Loop_program.t
(** The program must verify; [Invalid_argument] otherwise, and for a program
    with vector operations, which the Loop IR names no form for: expand them to
    scalar lanes first. *)
