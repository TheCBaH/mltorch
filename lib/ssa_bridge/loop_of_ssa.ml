open Ssa_ir
open Loop_ir

let temp (v : Ssa_value.t) = Loop_temp.of_int (v.Ssa_value.id :> int)
let var (v : Ssa_value.t) = Loop_var.of_int (v.Ssa_value.id :> int)
let ix v = Loop_index.Temp (temp v)
let fx v = Loop_expr.Temp (Loop_carrier.Float, temp v)
let ex v = Loop_expr.Temp (Loop_carrier.Int64, temp v)
let is_type ty (v : Ssa_value.t) = Ssa_type.equal v.Ssa_value.ty ty

let float_type (v : Ssa_value.t) =
  match v.Ssa_value.ty with
  | Ssa_type.Scalar Ssa_type.F32 -> `F32
  | Ssa_type.Scalar Ssa_type.F64 -> `F64
  | _ -> invalid_arg "Loop_of_ssa: not a float"

(* Reading an SSA value as the Loop expression that holds it. *)
let float_expr v = fx v
let int_of (n : int64) = Int64.to_int n

(* [v := e] for a float, index or int64 value. A predicate has no temporary: it
   exists only as the condition of an [if], and an effect has no machine form. *)
let assign (v : Ssa_value.t) ~float ~index ~i64 =
  match v.Ssa_value.ty with
  | Ssa_type.Scalar (Ssa_type.F32 | Ssa_type.F64) ->
      [ Loop_stmt.Assign (Loop_carrier.Float, temp v, float ()) ]
  | Ssa_type.Scalar Ssa_type.Index ->
      [ Loop_stmt.Assign_index (temp v, index ()) ]
  | Ssa_type.Scalar Ssa_type.I64 ->
      [ Loop_stmt.Assign (Loop_carrier.Int64, temp v, i64 ()) ]
  | Ssa_type.Effect -> []
  | Ssa_type.Mask _
  | Ssa_type.Scalar (Ssa_type.Offset | Ssa_type.Pred)
  | Ssa_type.Vec _ ->
      invalid_arg "Loop_of_ssa: a value with no Loop carrier"

(* [dst := src], the copy of a carried value. *)
let copy dst src =
  assign dst
    ~float:(fun () -> fx src)
    ~index:(fun () -> ix src)
    ~i64:(fun () -> ex src)

let buffer_of (p : Ssa_program.t) =
  let table =
    List.map
      (fun (b : Ssa_buffer.t) ->
        let id = Tensor_id.of_int (b.Ssa_buffer.id :> int) in
        let e a = Int64.to_int (Expr.Coord.get b.Ssa_buffer.extents a) in
        let shape =
          Vec6.shape ~n:(e Expr.Axis.N) ~t:(e Expr.Axis.T) ~d:(e Expr.Axis.D)
            ~h:(e Expr.Axis.H) ~w:(e Expr.Axis.W) ~c:(e Expr.Axis.C)
        in
        let fmt =
          match b.Ssa_buffer.format with
          | Ssa_format.Bool -> Payload.Fmt Payload.Bool
          | Ssa_format.F32 -> Payload.Fmt Payload.F32
          | Ssa_format.I64 -> Payload.Fmt Payload.I64
        in
        let role =
          match b.Ssa_buffer.role with
          | Ssa_buffer.Input -> Loop_buffer.Input
          | Ssa_buffer.Output -> Loop_buffer.Output
          | Ssa_buffer.Scratch -> Loop_buffer.Scratch
        in
        ( b.Ssa_buffer.id,
          {
            Loop_buffer.id;
            sg = Tensor_sig.create ~id ~name:"" ~shape ~fmt ();
            role;
          } ))
      p.Ssa_program.buffers
  in
  fun id ->
    match List.find_opt (fun (i, _) -> Ssa_id.Buffer.equal i id) table with
    | Some (_, b) -> b
    | None -> invalid_arg "Loop_of_ssa: undeclared buffer"

(* The bounds check a coordinate load performs, as the Loop IR spells it. *)
let load_guard (b : Loop_buffer.t) (c : Loop_index.coord) =
  let extent a = Dim.to_int (Vec6.get b.Loop_buffer.sg.Tensor_sig.shape a) in
  let checks =
    List.map
      (fun a -> Loop_bool.Out_of_range (Expr.Coord.get c a, extent a))
      Expr.Axis.all
  in
  let cond =
    match checks with
    | first :: rest ->
        List.fold_left (fun acc p -> Loop_bool.Or (acc, p)) first rest
    | [] -> invalid_arg "Loop_of_ssa: no axes"
  in
  Loop_stmt.Fail_if
    (cond, Loop_failure.Load_out_of_range { buffer = b; coord = c })

let convert (p : Ssa_program.t) =
  (match Err.payload (Ssa_verify.check p) with
  | Ok () -> ()
  | Error e -> invalid_arg (Fmt.str "Loop_of_ssa: %a" Ssa_verify.pp_error e));
  let buffer = buffer_of p in
  (* fresh temporaries for the snapshots of parallel transfers *)
  let fresh = ref (p.Ssa_program.next_value :> int) in
  let snapshot_of (v : Ssa_value.t) =
    let id = !fresh in
    incr fresh;
    { v with Ssa_value.id = Ssa_id.Value.of_int id }
  in
  let coord_of (c : Ssa_value.t Expr.Coord.t) = Expr.Coord.map ix c in
  let access = function
    | Ssa_access.Coord c -> `Coord (coord_of c)
    | Ssa_access.Flat o -> `Flat (ix o)
  in
  let op (i : Ssa_instr.t) =
    let result () =
      match i.Ssa_instr.results with
      | r :: _ -> r
      | [] -> invalid_arg "Loop_of_ssa: no result"
    in
    match i.Ssa_instr.op with
    | Ssa_op.Const c -> (
        let r = result () in
        match c with
        | Ssa_const.F32 x | Ssa_const.F64 x ->
            assign r
              ~float:(fun () -> Loop_expr.Const x)
              ~index:(fun () -> assert false)
              ~i64:(fun () -> assert false)
        | Ssa_const.I64 x ->
            assign r
              ~float:(fun () -> assert false)
              ~index:(fun () -> assert false)
              ~i64:(fun () -> Loop_expr.I64_const x)
        | Ssa_const.Index x ->
            assign r
              ~float:(fun () -> assert false)
              ~index:(fun () -> Loop_index.Const (int_of x))
              ~i64:(fun () -> assert false)
        | Ssa_const.Pred _ -> [])
    | Ssa_op.Convert (c, a) -> (
        let r = result () in
        match c with
        | Ssa_op.Convert.F32_to_f64 ->
            [ Loop_stmt.Assign (Loop_carrier.Float, temp r, fx a) ]
        | Ssa_op.Convert.F64_to_f32 ->
            [
              Loop_stmt.Assign
                (Loop_carrier.Float, temp r, Loop_expr.Round_f32 (fx a));
            ]
        | Ssa_op.Convert.Index_to_f64 ->
            [
              Loop_stmt.Assign
                (Loop_carrier.Float, temp r, Loop_expr.Value_of_index (ix a));
            ]
        | Ssa_op.Convert.Index_to_i64 ->
            [
              Loop_stmt.Assign
                (Loop_carrier.Int64, temp r, Loop_expr.I64_of_index (ix a));
            ])
    | Ssa_op.Float_binary (o, a, b) ->
        let r = result () in
        let sum = Loop_expr.Binary (o, float_expr a, float_expr b) in
        let sum =
          match float_type r with
          | `F32 -> Loop_expr.Round_f32 sum
          | `F64 -> sum
        in
        [ Loop_stmt.Assign (Loop_carrier.Float, temp r, sum) ]
    | Ssa_op.Index_add (a, b) ->
        let tree = Loop_index.Add (ix a, ix b) in
        [
          Loop_stmt.Fail_if
            ( Loop_bool.Index_overflows tree,
              Loop_failure.Index_overflow { index = tree } );
          Loop_stmt.Assign_index (temp (result ()), tree);
        ]
    | Ssa_op.Index_scale (k, a) ->
        let tree = Loop_index.Scale (int_of k, ix a) in
        [
          Loop_stmt.Fail_if
            ( Loop_bool.Index_overflows tree,
              Loop_failure.Index_overflow { index = tree } );
          Loop_stmt.Assign_index (temp (result ()), tree);
        ]
    | Ssa_op.Load { buffer = id; at; decode } -> (
        let b = buffer id in
        let r = result () in
        match (access at, decode) with
        | `Coord c, (Ssa_op.Decode.Bool_to_f64 | Ssa_op.Decode.F32_to_f64) ->
            [
              load_guard b c;
              Loop_stmt.Assign
                (Loop_carrier.Float, temp r, Loop_expr.Load (b, c));
            ]
        | `Flat o, (Ssa_op.Decode.Bool_to_f64 | Ssa_op.Decode.F32_to_f64) ->
            [
              Loop_stmt.Assign
                (Loop_carrier.Float, temp r, Loop_expr.Load_flat (b, o));
            ]
        | `Coord c, Ssa_op.Decode.I64 ->
            [
              load_guard b c;
              Loop_stmt.Assign
                (Loop_carrier.Int64, temp r, Loop_expr.Load_i64 (b, c));
            ]
        | `Flat o, Ssa_op.Decode.I64 ->
            [
              Loop_stmt.Assign
                (Loop_carrier.Int64, temp r, Loop_expr.Load_i64_flat (b, o));
            ])
    | Ssa_op.Mark m ->
        [
          Loop_stmt.Mark
            (match m with
            | Ssa_mark.Emitter -> Loop_mark.Emitter
            | Ssa_mark.Key -> Loop_mark.Key
            | Ssa_mark.Local -> Loop_mark.Local
            | Ssa_mark.Reduction -> Loop_mark.Reduction
            | Ssa_mark.Scan -> Loop_mark.Scan
            | Ssa_mark.Scan_update -> Loop_mark.Scan_update);
        ]
    | Ssa_op.Store { buffer = id; at; encode; value } -> (
        let b = buffer id in
        let stored =
          match encode with
          | Ssa_op.Encode.Bool_nonzero -> Loop_stored.Bool (fx value)
          | Ssa_op.Encode.F32_round -> Loop_stored.F32 (fx value)
          | Ssa_op.Encode.I64 -> Loop_stored.I64 (ex value)
        in
        match access at with
        | `Coord coord ->
            [ Loop_stmt.Store { buffer = b; coord; value = stored } ]
        | `Flat offset ->
            [ Loop_stmt.Store_flat { buffer = b; offset; value = stored } ])
  in
  let drop_effect vs =
    List.filter (fun v -> not (is_type Ssa_type.Effect v)) vs
  in
  let rec region (r : Ssa_region.t) = List.concat_map stmt r.Ssa_region.body
  and stmt : Ssa_region.t Ssa_stmt.t -> Loop_stmt.t list = function
    | Ssa_stmt.Instr i -> op i
    | Ssa_stmt.For { lo; hi; step = _; inits; results; body } ->
        let iv, params =
          match body.Ssa_region.params with
          | iv :: params -> (iv, drop_effect params)
          | [] -> invalid_arg "Loop_of_ssa: loop without induction value"
        in
        let inits = drop_effect inits in
        let results = drop_effect results in
        let yields = drop_effect body.Ssa_region.yields in
        (* every yield is read before any parameter is overwritten *)
        let snapshots = List.map snapshot_of yields in
        let update =
          List.concat (List.map2 copy snapshots yields)
          @ List.concat (List.map2 copy params snapshots)
        in
        List.concat (List.map2 copy params inits)
        @ [
            Loop_stmt.For
              {
                var = var iv;
                lo = ix lo;
                hi = ix hi;
                body =
                  Loop_stmt.Assign_index (temp iv, Loop_index.Var (var iv))
                  :: (region body @ update);
              };
          ]
        @ List.concat (List.map2 copy results params)
    | Ssa_stmt.If { cond; results; then_; else_ } ->
        let results = drop_effect results in
        let branch (r : Ssa_region.t) =
          region r
          @ List.concat
              (List.map2 copy results (drop_effect r.Ssa_region.yields))
        in
        ignore cond;
        [ Loop_stmt.If (predicate cond, branch then_, branch else_) ]
    | Ssa_stmt.Ordered_sum { lo; hi; seed; token = _; results; body } ->
        let iv =
          match body.Ssa_region.params with
          | iv :: _ -> iv
          | [] -> invalid_arg "Loop_of_ssa: sum without induction value"
        in
        let sum, _ =
          match results with
          | [ sum; e ] -> (sum, e)
          | _ -> invalid_arg "Loop_of_ssa: sum results"
        in
        let term =
          match body.Ssa_region.yields with
          | term :: _ -> term
          | [] -> invalid_arg "Loop_of_ssa: sum without a term"
        in
        let acc = snapshot_of sum in
        let next = Loop_expr.Binary (Expr.Value.Add, fx acc, fx term) in
        let next =
          match float_type sum with
          | `F32 -> Loop_expr.Round_f32 next
          | `F64 -> next
        in
        [ Loop_stmt.Assign (Loop_carrier.Float, temp acc, fx seed) ]
        @ [
            Loop_stmt.For
              {
                var = var iv;
                lo = ix lo;
                hi = ix hi;
                body =
                  Loop_stmt.Assign_index (temp iv, Loop_index.Var (var iv))
                  :: region body
                  @ [ Loop_stmt.Assign (Loop_carrier.Float, temp acc, next) ];
              };
          ]
        @ [ Loop_stmt.Assign (Loop_carrier.Float, temp sum, fx acc) ]
  and predicate (c : Ssa_value.t) =
    (* A predicate exists only as a constant for now; find its definition. *)
    match find_const c with
    | Some true -> Loop_bool.Index_eq (Loop_index.Const 0, Loop_index.Const 0)
    | Some false ->
        Loop_bool.Not
          (Loop_bool.Index_eq (Loop_index.Const 0, Loop_index.Const 0))
    | None -> invalid_arg "Loop_of_ssa: a predicate that is not a constant"
  and find_const (c : Ssa_value.t) =
    let rec in_region (r : Ssa_region.t) =
      List.find_map in_stmt r.Ssa_region.body
    and in_stmt : Ssa_region.t Ssa_stmt.t -> bool option = function
      | Ssa_stmt.Instr
          { Ssa_instr.results = [ r ]; op = Ssa_op.Const (Ssa_const.Pred b); _ }
        when Ssa_value.equal r c ->
          Some b
      | Ssa_stmt.Instr _ -> None
      | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
          in_region body
      | Ssa_stmt.If { then_; else_; _ } -> (
          match in_region then_ with
          | Some b -> Some b
          | None -> in_region else_)
    in
    in_region p.Ssa_program.entry
  in
  {
    Loop_program.buffers =
      List.map
        (fun (b : Ssa_buffer.t) -> buffer b.Ssa_buffer.id)
        p.Ssa_program.buffers;
    body = region p.Ssa_program.entry;
    scan_limits = Expr.Scan_limits.default;
    max_depth = 64;
  }
