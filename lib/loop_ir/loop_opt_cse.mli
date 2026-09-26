(** Stage 8 of the optimization pipeline ([.ai/loop_ir_optimization_design.md]
    pass 5, "load CSE"): within one straight-line [Store]/[Assign]'s float value
    expression, an identical [Load (b, coord)] of a buffer no [Store] anywhere
    in the program targets is shared through one float temp, computed once
    immediately before the statement instead of twice inside it. [relu]'s double
    load (`b0[k] < 0 ? 0 : b0[k]`) is the target.

    Scoped to a single statement's own expression tree, never across statements
    or loop iterations: unlike a general common-subexpression pass, this shares
    nothing that could be stale by the time a second occurrence reads it, so it
    carries none of the "is this temp reused as a mutable accumulator" hazard a
    cross-statement rewrite would (see the hoisting pass this design attempted
    and reverted). *)

val run : Loop_program.t -> Loop_program.t
