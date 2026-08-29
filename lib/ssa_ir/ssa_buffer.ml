(* A declared object: identity, six-axis extents and storage format. Distinct
   ids do not prove disjoint memory. *)
type role = Input | Output | Scratch

type t = {
  id : Ssa_id.Buffer.t;
  extents : int64 Expr.Coord.t;
  format : Ssa_format.t;
  role : role;
}

let role_name = function
  | Input -> "in"
  | Output -> "out"
  | Scratch -> "scratch"

(* The element-index domain: a flat access carries a row-major element offset
   of type [index], so no buffer may have more elements than an index holds. *)
let max_elements = 0x7FFF_FFFFL

(* The product of the six extents, or [None] once it would pass
   [max_elements]. Each factor is divided into the ceiling before it is
   multiplied, so no intermediate product wraps. *)
let elements (e : int64 Expr.Coord.t) =
  Expr.Coord.fold
    (fun acc x ->
      match acc with
      | None -> None
      | Some n ->
          if Int64.compare x 1L < 0 then None
          else if Int64.compare n (Int64.div max_elements x) > 0 then None
          else Some (Int64.mul n x))
    (Some 1L) e

let source t = Expr.Source.create (t.id :> int)
