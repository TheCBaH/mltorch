(* The interpreter's storage: each buffer one flat row-major array of cells. A
   [Floats] cell of an [F32] buffer already holds a binary32 value, and a
   [Bool] one is 0 or 1. Independent of Bigarray and of any tensor type, so the
   interpreter owns no storage layout but its own. *)
type cells = Floats of float array | Int64s of int64 array
type t = cells Ssa_id.Buffer.Map.t

let zeroed (b : Ssa_buffer.t) =
  match Ssa_buffer.elements b.Ssa_buffer.extents with
  | None -> invalid_arg "Ssa_memory.zeroed: more elements than an index holds"
  | Some n -> (
      let n = Int64.to_int n in
      match b.Ssa_buffer.format with
      | Ssa_format.Bool | Ssa_format.F32 -> Floats (Array.make n 0.)
      | Ssa_format.I64 -> Int64s (Array.make n 0L))

(* Zeroed storage for the buffers [keep] selects, in a program's order. *)
let allocate ~keep (buffers : Ssa_buffer.t list) =
  List.fold_left
    (fun m (b : Ssa_buffer.t) ->
      if keep b then Ssa_id.Buffer.Map.add b.Ssa_buffer.id (zeroed b) m else m)
    Ssa_id.Buffer.Map.empty buffers

let find (m : t) id = Ssa_id.Buffer.Map.find_opt id m

(* The row-major element offset of an in-range coordinate, in [int64]. *)
let offset (extents : int64 Expr.Coord.t) (c : int64 Expr.Coord.t) =
  List.fold_left2
    (fun acc e i -> Int64.add (Int64.mul acc e) i)
    0L
    (Expr.Coord.to_list extents)
    (Expr.Coord.to_list c)

(* The first axis, in [Expr.Axis.all] order, whose component is outside. *)
let first_outside (extents : int64 Expr.Coord.t) (c : int64 Expr.Coord.t) =
  List.find_opt
    (fun a ->
      let i = Expr.Coord.get c a in
      Int64.compare i 0L < 0 || Int64.compare i (Expr.Coord.get extents a) >= 0)
    Expr.Axis.all
