(** Stage 2 of the optimization pipeline ([.ai/loop_ir_optimization_design.md]
    pass 2): fold pure [Const] arithmetic and the algebraic identities that
    never drop a non-constant subexpression ([Add (Const 0, a)], [Scale (1, a)],
    nested [Scale]). [Min]/[Max]/[Clamp_low] on proven ranges are deliberately
    out of scope -- they need an environment of enclosing loop variables' ranges
    this context-free rewrite doesn't have, and the one place they would matter
    most (window-loop bounds) is handled precisely by stage 4's relational
    scheme instead of a general interval fold. Runs after stage 1, so a
    unit-loop's substituted constants collapse into their surrounding index
    arithmetic. *)

val fold_index : Loop_index.t -> Loop_index.t
(** One node's fold, given already-folded children (used directly by the unit
    test on domain-edge behavior; the pipeline reaches it through {!run} via
    {!Loop_index_map}, which applies it bottom-up everywhere an index appears).
*)

val run : Loop_program.t -> Loop_program.t
