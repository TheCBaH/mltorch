(* See expr_bridge.mli. *)

let source_of_id id = Expr.Source.create (Tensor_id.to_int id)
let id_of_source s = Tensor_id.of_int (Expr.Source.to_int s)

let coord_of_vec6 (v : 'a Vec6.t) : 'a Expr.Coord.t =
  Expr.Coord.make ~n:v.Vec6.n ~t:v.Vec6.t ~d:v.Vec6.d ~h:v.Vec6.h ~w:v.Vec6.w
    ~c:v.Vec6.c

let vec6_of_coord (c : 'a Expr.Coord.t) : 'a Vec6.t =
  Vec6.make ~n:c.Expr.Coord.n ~t:c.Expr.Coord.t ~d:c.Expr.Coord.d
    ~h:c.Expr.Coord.h ~w:c.Expr.Coord.w ~c:c.Expr.Coord.c

let env ~binding =
  let load s (c : int Expr.Coord.t) =
    (* [Core.of_option], not a hand-rolled match: it is the repository's named
       bridge from an option into the result framework, and it captures the
       detection backtrace the way [Core.fail] does. The payload is built
       eagerly, which is fine here -- it is a pure, cheap constructor. *)
    let open Core.Syntax in
    let* (Tensor.Tensor t) =
      Core.of_option (`Unknown_source s) (binding (id_of_source s))
    in
    (* Bounds-checked BEFORE a [Vec6.coord] exists. Building one first would
       raise out of [Dim.index] on a negative component, and reading would raise
       [Invalid_argument] above an extent -- either way an exception would escape
       a [Core.result] API. *)
    let out_of_range =
      List.find_opt
        (fun a ->
          let i = Expr.Coord.get c a in
          i < 0 || i >= Dim.to_int (Vec6.get t.Tensor.shape a))
        Expr.Axis.all
    in
    match out_of_range with
    | Some a -> Core.fail (`Coord_out_of_range (s, a, Expr.Coord.get c a, c))
    | None ->
        Core.return
          (Tensor.read_at_raw (Tensor.Tensor t) (fun a -> Expr.Coord.get c a))
  in
  { Expr.Eval.Env.load }
