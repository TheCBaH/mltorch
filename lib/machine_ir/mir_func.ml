(* A function: its entry block's parameters are its parameters, and every
   return supplies [results]. [blocks] lists the entry first; the order after it
   carries no meaning. *)
type ('op, 'term) t = {
  id : Mir_id.Func.t;
  name : string;
  entry : Mir_id.Block.t;
  results : Mir_type.t list;
  blocks : ('op, 'term) Mir_block.t list;
}

let find_block t id =
  List.find_opt
    (fun (b : (_, _) Mir_block.t) -> Mir_id.Block.equal b.Mir_block.id id)
    t.blocks

let params t =
  match find_block t t.entry with Some b -> b.Mir_block.params | None -> []
