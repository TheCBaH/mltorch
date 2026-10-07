(* A two-way branch on a predicate. *)
type t = { cond : Mir_value.t; then_ : Mir_edge.t; else_ : Mir_edge.t }
