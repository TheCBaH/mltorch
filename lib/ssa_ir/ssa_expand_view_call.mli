(** Each [View_call] as the loops of {!Ssa_expand_contract}, every cell offset
    by its operand's base value: the same cells {!Ssa_view_call.eval} reads and
    writes, for a backend with no view-call of its own. The verifier proved the
    footprint, so the expanded accesses never fail. A contraction
    {!Ssa_expand_contract.expandable} refuses (paired, integer) stays a
    [View_call]. *)

val program : Ssa_program.t -> Ssa_program.t
