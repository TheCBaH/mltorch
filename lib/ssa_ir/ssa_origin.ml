(* Where an operation came from. A pass keeps a stable site when it moves an
   operation and gives a clone its own instance. [Output] names the buffer an
   operation stores to; [Value] the logical value of the placed kernel whose
   lowering made it, which a pass keeps for what it moves or rewrites and a
   clone also gets. *)
type t = Output of Ssa_id.Buffer.t | Unknown | Value of Ssa_id.Buffer.t

let equal a b =
  match (a, b) with
  | Output a, Output b -> Ssa_id.Buffer.equal a b
  | Unknown, Unknown -> true
  | Value a, Value b -> Ssa_id.Buffer.equal a b
  | (Output _ | Unknown | Value _), _ -> false

let pp fmt = function
  | Output b -> Fmt.pf fmt "out %a" Ssa_id.Buffer.pp b
  | Unknown -> Fmt.string fmt "?"
  | Value b -> Fmt.pf fmt "value %a" Ssa_id.Buffer.pp b
