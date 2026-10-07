(* A transfer to [target]: the arguments bind its parameters and [order] its
   order parameter, all at once — every argument is read before any parameter
   is rebound. *)
type t = {
  target : Mir_id.Block.t;
  args : Mir_value.t list;
  order : Mir_value.t;
}

let operands e = e.args @ [ e.order ]
