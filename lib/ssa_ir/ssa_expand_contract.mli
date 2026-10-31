(** Each [Contract] with [Ordered] numerics over binary32 or bfloat16 operands,
    or [Fused_sequential] over binary32, as the loops of {!Contraction.eval}: an
    ordered sum per output cell. The fused form is the same loop; it computes
    the fused sums only under a plan that contracts dots, which the caller must
    ensure. A paired or integer contraction is left as it is. For a backend, or
    a Machine IR route, that realizes no [Contract] of its own. *)

val program : Ssa_program.t -> Ssa_program.t
