(* See expr_bridge.mli. *)

let source_of_id id = Expr.Source.create (Tensor_id.to_int id)
let id_of_source s = Tensor_id.of_int (Expr.Source.to_int s)

let coord_of_vec6 (v : 'a Vec6.t) : 'a Expr.Coord.t =
  Expr.Coord.make ~n:v.Vec6.n ~t:v.Vec6.t ~d:v.Vec6.d ~h:v.Vec6.h ~w:v.Vec6.w
    ~c:v.Vec6.c

let vec6_of_coord (c : 'a Expr.Coord.t) : 'a Vec6.t =
  Vec6.make ~n:c.Expr.Coord.n ~t:c.Expr.Coord.t ~d:c.Expr.Coord.d
    ~h:c.Expr.Coord.h ~w:c.Expr.Coord.w ~c:c.Expr.Coord.c

(* Shared by [load]/[load_index]: resolves [s] to its bound tensor and checks
   [c] against its shape, bounds-checked BEFORE a [Vec6.coord] exists.
   Building one first would raise out of [Dim.index] on a negative component,
   and reading would raise [Invalid_argument] above an extent -- either way an
   exception would escape a [Err.t] API. *)
let bound_in_range ~binding s (c : int Expr.Coord.t) =
  let open Err.Syntax in
  (* [Err.of_option], not a hand-rolled match: it is the repository's named
     bridge from an option into the result framework, and it captures the
     detection backtrace the way [Err.fail] does. The payload is built
     eagerly, which is fine here -- it is a pure, cheap constructor. *)
  let* (Tensor.Tensor t) =
    Err.of_option (`Unknown_source s) (binding (id_of_source s))
  in
  let out_of_range =
    List.find_opt
      (fun a ->
        let i = Expr.Coord.get c a in
        i < 0 || i >= Dim.to_int (Vec6.get t.Tensor.shape a))
      Expr.Axis.all
  in
  match out_of_range with
  | Some a -> Err.fail (`Coord_out_of_range (s, a, Expr.Coord.get c a, c))
  | None -> Err.return (Tensor.Tensor t)

let env ~binding =
  let load s (c : int Expr.Coord.t) =
    let open Err.Syntax in
    let+ t = bound_in_range ~binding s c in
    Tensor.read_at_raw t (fun a -> Expr.Coord.get c a)
  in
  let load_index s (c : int Expr.Coord.t) =
    let open Err.Syntax in
    let* t = bound_in_range ~binding s c in
    Tensor.read_i64_at6 t (fun a -> Expr.Coord.get c a)
    |> Err.map_error (fun (`Wrong_format (Payload.Fmt f)) ->
        `Data_source_wrong_format (Payload.fmt_name f))
  in
  { Expr.Eval.Env.load; load_index }
