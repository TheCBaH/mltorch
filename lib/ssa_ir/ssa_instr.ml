(* One operation with its results. An effectful operation names the effect it
   consumes in [token] and returns the next effect as its last result; a pure
   one has [token = None]. *)
type t = {
  results : Ssa_value.t list;
  op : Ssa_op.t;
  token : Ssa_value.t option;
  origin : Ssa_origin.t;
}

let operands i = Ssa_op.operands i.op @ Option.to_list i.token
