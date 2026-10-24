(** The dot fusion of the [Fp32_contracted] policy over the loops the planner
    leaves: a reduction loop (one whose body marks a reduction) that carries an
    accumulator [acc] used once, in [acc + x * y] (either operand order, scalar
    or lane-wise) with the product made in the same body, yields
    [fma (x, y, acc)] instead: the left fold of an ordered sum, each step
    rounded once, which blocking and vectorization have rewritten into a [for]
    carrying several accumulators. An [Ordered_sum] itself is fused when it is
    lowered ({!Ssa_cfg_lower.Accumulate}). The product stays, for its other
    uses. *)

val program : Ssa_program.t -> Ssa_program.t
