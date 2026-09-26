(** Loop-invariant hoisting of a statement: a closed-valued [Assign] of a float
    temporary or [Array_set] of a constant cell moves out of a loop that runs at
    least once, when no other statement in the loop writes its target and none
    before it reads it. *)

val run : Loop_program.t -> Loop_program.t

val with_write_check : bool -> Loop_program.t -> Loop_program.t
(** [with_write_check true] is [run]; [false] skips the "no other write"
    condition, for the mutation test (it hoists an accumulator's reset). *)
