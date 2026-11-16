(** Loop IR to structured SSA: a temporary comparison bridge, not the permanent
    frontend. Mutable temporaries become SSA values; a temporary assigned in a
    loop body and live before it becomes an iteration argument, and one assigned
    in both arms of an [if] becomes its result.

    Covered: loops, ordered sums, float and index temporaries, loads, stores,
    marks. Anything else is a typed refusal ({!Ssa_of_loop_unsupported}): in
    particular a [Fail_if], because the SSA form must reproduce the guard's
    evaluation site and no recipe for it exists yet. *)

type error = [ `Unsupported of Ssa_of_loop_unsupported.t ]

val pp_error : Format.formatter -> [< error ] -> unit
val convert : Loop_ir.Loop_program.t -> (Ssa_ir.Ssa_program.t, error) Err.t
