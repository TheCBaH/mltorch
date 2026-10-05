(* How a block ends. [Return] gives back the invocation's effect: the whole
   program's result, since the buffers are the outputs. *)
type t =
  | Branch of {
      cond : Ssa_value.t;
      else_ : Ssa_cfg_edge.t;
      then_ : Ssa_cfg_edge.t;
    }
  | Jump of Ssa_cfg_edge.t
  | Return of Ssa_value.t

let edges = function
  | Branch { then_; else_; _ } -> [ then_; else_ ]
  | Jump e -> [ e ]
  | Return _ -> []

let successors t = List.map (fun (e : Ssa_cfg_edge.t) -> e.target) (edges t)

let operands = function
  | Branch { cond; then_; else_ } -> (cond :: then_.args) @ else_.args
  | Jump e -> e.args
  | Return v -> [ v ]
