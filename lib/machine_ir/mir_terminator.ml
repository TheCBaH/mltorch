(* How a generic block ends. *)
type t =
  | Branch of Mir_branch.t
  | Fail of Mir_fail.t
  | Jump of Mir_edge.t
  | Return of Mir_return.t

let edges = function
  | Branch { Mir_branch.then_; else_; _ } -> [ then_; else_ ]
  | Jump e -> [ e ]
  | Fail _ | Return _ -> []

let successors t =
  List.map (fun (e : Mir_edge.t) -> e.Mir_edge.target) (edges t)

(* Value operands, edge arguments and order states included. *)
let operands = function
  | Branch { Mir_branch.cond; then_; else_ } ->
      (cond :: Mir_edge.operands then_) @ Mir_edge.operands else_
  | Fail { Mir_fail.payload; order; _ } -> payload @ [ order ]
  | Jump e -> Mir_edge.operands e
  | Return { Mir_return.values; order } -> values @ [ order ]
