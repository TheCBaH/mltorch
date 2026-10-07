(* A virtual SSA definition: an id and its type. Equality is by id; the type
   travels so a verifier can check every use against the definition. Not a
   register, stack slot or source variable. *)
type t = { id : Mir_id.Value.t; ty : Mir_type.t }

let equal a b = Mir_id.Value.equal a.id b.id
let compare a b = Mir_id.Value.compare a.id b.id
let pp fmt v = Mir_id.Value.pp fmt v.id
