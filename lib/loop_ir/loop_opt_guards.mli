(** Stage 3 of the optimization pipeline ([.ai/loop_ir_optimization_design.md]
    pass 3): re-run [Loop_range] over the program stages 1 and 2 already
    simplified, and drop a [Fail_if (Index_overflows _ | Out_of_range _)] -- or
    an [Or] of only those two shapes, which is what a multi-axis bounds check
    ([Loop_lower_index.load_guard]) actually emits -- that [Loop_range] proves
    can never fire. Unlike stage 2's context-free [Loop_index_map], a guard's
    provability depends on where in the program it sits, so this walks top-down,
    rebuilding the [Loop_range.Env.t] of enclosing loop variables' ranges the
    original lowering had while it built the program, but discarded once
    lowering finished.

    A bounds check a clamped window loop's own bounds imply
    ([for k in [max (0, -base), min (K, n - base)) ... base + k in [0, n)]) is
    proven relationally, from facts [d * k >= a] / [d * k <= a] its bounds
    give, and an unflattened coordinate's [x - d * floor (x / d)] through
    [Loop_linear.range]. *)

val run : Loop_program.t -> Loop_program.t

val with_upper_slack : int -> Loop_program.t -> Loop_program.t
(** [run] with the [1] of a loop's [k <= hi - 1] fact replaced, for the mutation
    test: [with_upper_slack 1] is [run], and [2] claims a bound one tighter than
    the loop gives. *)
