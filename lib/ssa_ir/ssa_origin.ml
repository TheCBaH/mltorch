(* Where an operation came from. A pass keeps a stable site when it moves an
   operation and gives a clone its own instance; neither is a name for a value. *)
type t = Output of Ssa_id.Buffer.t | Unknown

let equal a b =
  match (a, b) with
  | Output a, Output b -> Ssa_id.Buffer.equal a b
  | Unknown, Unknown -> true
  | (Output _ | Unknown), _ -> false

let pp fmt = function
  | Output b -> Fmt.pf fmt "out %a" Ssa_id.Buffer.pp b
  | Unknown -> Fmt.string fmt "?"
