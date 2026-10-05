(* A basic block: parameters (the only phi convention), straight-line
   operations, and one terminator. An effectful operation names the effect it
   consumes exactly as in a region. *)
type t = {
  id : Ssa_id.Block.t;
  params : Ssa_value.t list;
  body : Ssa_instr.t list;
  terminator : Ssa_cfg_terminator.t;
}
