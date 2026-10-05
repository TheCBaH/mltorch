(** Multiply-add contraction: under a policy that permits it and on a target
    with a fused operation, [a + x * y] (either operand order) becomes one
    {!Ssa_op.Float_fma} with a single rounding. As the Loop planner's
    contraction does, only [Add] contracts ([a * b - c] would need a negation),
    a multiplication on the right is taken first, and a multiplication another
    use still needs stays.

    Vector adds contract the same way: a lane-wise add of a lane-wise
    multiplication becomes a lane-wise fused operation. The scalar statements
    around a vector loop contract too, unless [scalar] is [false], for a target
    whose fused operation exists only on vectors. *)

val pass : scalar:bool -> Ssa_program.t -> Ssa_program.t * bool
(** Whether anything was contracted. *)

val contracted : Ssa_program.t -> int
(** How many fused operations the program holds, scalar and lane-wise. *)
