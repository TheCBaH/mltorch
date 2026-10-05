(** Direct lowering of a validated [Fusion_plan.t] to a verified SSA program.
    Anything outside the supported slice is a typed [`Unsupported], never a
    partial program. *)

type error = [ `Unsupported of Ssa_unsupported.t ]

val pp_error : Format.formatter -> [< error ] -> unit

val lower : Fusion_plan.t -> (Ssa_ir.Ssa_program.t, error) Err.t
(** One six-axis loop nest per stored value, in [Kernel.values] order: N
    outermost and C innermost, as [Vec6.iter] visits a tensor. The body is the
    value's pixel expression with its virtual producer, if any, inlined by the
    capture-safe elaborator and its result conversion applied, then one store.
    Reductions are ordered sums seeded at [+0.]. Every index operation is
    checked and every load checks its coordinate, at the source's evaluation
    point. *)
