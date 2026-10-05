(** The deterministic text of a program: definitions are numbered by their
    position in the structure, so the order in which a builder happened to
    allocate ids, and the ids themselves, never appear. A diagnostic format
    only: nothing parses it. *)

val to_string : Ssa_program.t -> string
val pp : Format.formatter -> Ssa_program.t -> unit
