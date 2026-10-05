(* A transfer of control to [target]. The arguments bind the target's
   parameters, all at once: every argument is read before any parameter is
   rebound, so an edge that swaps two values needs no temporary. *)
type t = { target : Ssa_id.Block.t; args : Ssa_value.t list }
