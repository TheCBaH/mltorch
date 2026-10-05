(* Where an access lands in its object: a six-axis coordinate (checked against
   the extents), or a dense row-major element offset of type [index]. A flat
   offset is an element index, never a byte offset, and is never formed on a
   per-channel quantized buffer. *)
type t = Coord of Ssa_value.t Expr.Coord.t | Flat of Ssa_value.t

let operands = function Coord c -> Expr.Coord.to_list c | Flat v -> [ v ]

let map f = function
  | Coord c -> Coord (Expr.Coord.map f c)
  | Flat v -> Flat (f v)
