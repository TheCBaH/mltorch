(* One instruction: an opcode (generic, or a target's selected form), its value
   results and, when the opcode is ordered, the order state it threads. *)
type 'op t = {
  id : Mir_id.Instr.t;
  results : Mir_value.t list;
  op : 'op;
  order : Mir_order.t option;
  origin : Mir_origin.t;
}
