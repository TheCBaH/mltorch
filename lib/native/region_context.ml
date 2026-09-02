(* Small, operation-neutral helpers for constructing a Region program.  The
   graph dispatcher owns provenance and role resolution; operation modules own
   their formula.  Keeping this module free of either prevents the construction
   helpers from becoming a second operation API. *)

type error = Invalid_partition | Invalid_program

let pp_error fmt = function
  | Invalid_partition -> Fmt.string fmt "invalid region partition"
  | Invalid_program -> Fmt.string fmt "invalid region program"

let same_shape left right =
  List.for_all
    (fun axis -> Dim.equal (Vec6.get left axis) (Vec6.get right axis))
    Expr.Axis.all

let source (sg : Tensor_sig.t) = Expr_bridge.source_of_id sg.Tensor_sig.id

let output_coord =
  Expr.Coord.make
    ~n:(Expr.Index.output Expr.Axis.N)
    ~t:(Expr.Index.output Expr.Axis.T)
    ~d:(Expr.Index.output Expr.Axis.D)
    ~h:(Expr.Index.output Expr.Axis.H)
    ~w:(Expr.Index.output Expr.Axis.W)
    ~c:(Expr.Index.output Expr.Axis.C)

let load sg coord = Expr.Value.load (source sg) coord
let load_output sg = load sg output_coord

let reduce_dims ~kind ~dims ~shape ~leaf =
  let rec go dims overrides =
    match dims with
    | [] -> Expr.Builder.return (leaf overrides)
    | axis :: rest ->
        Expr.Builder.reduction ~kind ~lo:Expr.Index.zero
          ~hi:(Expr.Index.const (Dim.to_int (Vec6.get shape axis)))
          (fun index -> go rest ((axis, index) :: overrides))
  in
  Expr.Builder.run (go dims [])

let reduced_coord overrides =
  List.fold_left
    (fun coord (axis, index) -> Expr.Coord.set coord axis index)
    output_coord overrides

let partition dims =
  Err.map_error
    (fun _ -> Invalid_partition)
    (Region_partition.of_whole_axes dims)

let program result = Err.map_error (fun _ -> Invalid_program) result
