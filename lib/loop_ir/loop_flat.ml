(* The per-axis coordinates of a Loop IR access as one dense row-major offset,
   folded as it is built so a unit extent or a constant component prints
   nothing: the linearisation [Vec6.offset] defines, as an index expression.
   Only constants small enough that the product cannot leave the index domain
   are folded. *)
let small k = k > -0x4000_0000 && k < 0x4000_0000

let scale k (a : Loop_index.t) : Loop_index.t =
  match a with
  | _ when k = 1 -> a
  | Loop_index.Const m when small m && small k && small (k * m) ->
      Loop_index.Const (k * m)
  | _ -> Loop_index.Scale (k, a)

let add (a : Loop_index.t) (b : Loop_index.t) : Loop_index.t =
  match (a, b) with
  | Loop_index.Const 0, x | x, Loop_index.Const 0 -> x
  | Loop_index.Const x, Loop_index.Const y when small x && small y ->
      Loop_index.Const (x + y)
  | _ -> Loop_index.Add (a, b)

let offset (b : Loop_buffer.t) (c : Loop_index.coord) : Loop_index.t =
  let shape = b.Loop_buffer.sg.Tensor_sig.shape in
  List.fold_left
    (fun acc a ->
      let extent = Dim.to_int (Vec6.get shape a) in
      let i = Expr.Coord.get c a in
      match acc with
      | None -> Some i
      | Some acc -> Some (add (scale extent acc) i))
    None Expr.Axis.all
  |> Option.get
