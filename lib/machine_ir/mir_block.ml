(* A virtual basic block: typed parameters (the only phi convention), an order
   parameter, straight-line instructions and one terminator. ['op] and ['term]
   are the stage's own opcode and terminator families, so a generic and a
   selected instruction cannot share a block by accident. *)
type ('op, 'term) t = {
  id : Mir_id.Block.t;
  params : Mir_value.t list;
  order : Mir_value.t;
  body : 'op Mir_instr.t list;
  terminator : 'term;
}
