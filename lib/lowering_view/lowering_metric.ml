(* @alphabetical *)
type t = Coral_cycles | Scalar_loops | Vector_loops

let all = [ Coral_cycles; Scalar_loops; Vector_loops ]

let name = function
  | Coral_cycles -> "coral_cycles"
  | Scalar_loops -> "scalar_loops"
  | Vector_loops -> "vector_loops"

let of_name s = List.find_opt (fun m -> String.equal (name m) s) all
