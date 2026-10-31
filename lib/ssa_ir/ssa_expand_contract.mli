(** Each [Contract] with [Ordered] numerics over binary32 or bfloat16 operands
    as the loops of {!Contraction.eval}: an ordered sum per output cell. A
    fused, paired or integer contraction is left as it is. For a backend, or a
    Machine IR route, that realizes no [Contract] of its own. *)

val program : Ssa_program.t -> Ssa_program.t
