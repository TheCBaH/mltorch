type role = Input | Output | Scratch
type t = { id : Tensor_id.t; sg : Tensor_sig.t; role : role }

let role_name = function
  | Input -> "in"
  | Output -> "out"
  | Scratch -> "scratch"
