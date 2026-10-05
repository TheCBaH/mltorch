(** The deterministic text of a graph: blocks in reverse postorder named by
    position, values by definition order, as {!Ssa_pp} prints a program. A
    diagnostic format only. *)

val to_string : Ssa_cfg.t -> string
val pp : Format.formatter -> Ssa_cfg.t -> unit
