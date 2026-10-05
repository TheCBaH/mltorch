(* A program-local SSA definition: an id and the runtime witness of its type.
   Equality is by id; the type travels so a verifier can check a use against
   the definition it names. Not a source variable, tensor id or register. *)
type t = { id : Ssa_id.Value.t; ty : Ssa_type.t }

let equal a b = Ssa_id.Value.equal a.id b.id
let compare a b = Ssa_id.Value.compare a.id b.id
