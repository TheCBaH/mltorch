(** The CFG executor: it verifies the graph first, then runs blocks by following
    terminators and binding each edge's arguments to the target's parameters all
    at once. It owns only control flow: every operation runs through
    {!Ssa_interp.Machine.exec}, so a disagreement with the structured
    interpreter on any program is a disagreement about control flow. *)

type error = [ Ssa_interp.failure | Ssa_cfg_verify.error ]

val pp_error : Format.formatter -> [< error ] -> unit

val run :
  ?counters:Ssa_interp.Counters.t ->
  ?fused:bool ->
  Ssa_cfg.t ->
  memory:Ssa_memory.t ->
  (unit, error) Err.t
