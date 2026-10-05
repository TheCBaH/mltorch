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
  | Ssa_type.Effect | Ssa_type.Local -> []
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
        let sg = Ssa_lower.Ssa_sig.signature b in
        let role =
          match b.Ssa_buffer.role with
          | Ssa_buffer.Input -> Loop_buffer.Input
          | Ssa_buffer.Output -> Loop_buffer.Output
          | Ssa_buffer.Scratch -> Loop_buffer.Scratch
        in
        (b.Ssa_buffer.id, { Loop_buffer.id = sg.Tensor_sig.id; sg; role }))
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
  (* A scratch object is a Loop array named by its handle's id, with the slots
     and variable its checked reads need. *)
  let locals : (int, int64 * Expr.Local_var.t option) Hashtbl.t =
    Hashtbl.create 8
  in
  let array (v : Ssa_value.t) = Loop_array.of_int (v.Ssa_value.id :> int) in
  let local_guard ~var ~at ~extent =
    Loop_stmt.Fail_if
      ( Loop_bool.Out_of_range (at, extent),
        Loop_failure.Local_out_of_range { local = var; index = at; extent } )
  in
  (* A predicate has no temporary: it is the Loop expression that computes it
     from the temporaries of its operands, kept until a consumer reads it. *)
  let preds : (int, Loop_expr.pred) Hashtbl.t = Hashtbl.create 16 in
  let pred_of (v : Ssa_value.t) =
    match Hashtbl.find_opt preds (v.Ssa_value.id :> int) with
    | Some p -> p
    | None -> invalid_arg "Loop_of_ssa: a predicate with no definition"
  in
  let define_pred (v : Ssa_value.t) p =
    Hashtbl.replace preds (v.Ssa_value.id :> int) p;
    []
  in
  let always = Loop_bool.Index_eq (Loop_index.Const 0, Loop_index.Const 0) in
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
    | Ssa_op.Check_access { buffer = id; at } -> (
        let b = buffer id in
        match access at with `Coord c -> [ load_guard b c ] | `Flat _ -> [])
    | Ssa_op.Check_local { var; at; extent } ->
        [ local_guard ~var ~at:(ix at) ~extent:(int_of extent) ]
    | Ssa_op.Check_scan { var; row; lane; row_extent; lane_extent } ->
        [
          Loop_stmt.Fail_if
            ( Loop_bool.Out_of_range (ix row, int_of row_extent),
              Loop_failure.Scan_row_out_of_range
                {
                  local = var;
                  row = ix row;
                  lane = ix lane;
                  extent = int_of row_extent;
                } );
          Loop_stmt.Fail_if
            ( Loop_bool.Out_of_range (ix lane, int_of lane_extent),
              Loop_failure.Scan_lane_out_of_range
                {
                  local = var;
                  row = ix row;
                  lane = ix lane;
                  extent = int_of lane_extent;
                } );
        ]
    | Ssa_op.Local_alloc { slots; var } ->
        let r = result () in
        Hashtbl.replace locals (r.Ssa_value.id :> int) (slots, var);
        [
          Loop_stmt.Alloc
            (array r, Slot.count_of_extent (Slot.extent (int_of slots)));
        ]
    | Ssa_op.Local_read { local; at } ->
        let r = result () in
        let guard =
          match Hashtbl.find_opt locals (local.Ssa_value.id :> int) with
          | Some (slots, Some var) ->
              [ local_guard ~var ~at:(ix at) ~extent:(int_of slots) ]
          | Some (_, None) | None -> []
        in
        guard
        @ [
            Loop_stmt.Assign
              ( Loop_carrier.Float,
                temp r,
                Loop_expr.Array_get (array local, ix at) );
          ]
    | Ssa_op.Local_write { local; at; value } ->
        [ Loop_stmt.Array_set (array local, ix at, fx value) ]
    | Ssa_op.Meter_charge -> [ Loop_stmt.Charge_scan_update ]
    | Ssa_op.Meter_release w -> [ Loop_stmt.Release_scan_state (int_of w) ]
    | Ssa_op.Meter_reserve w -> [ Loop_stmt.Reserve_scan_state (int_of w) ]
    | Ssa_op.Meter_reset -> [ Loop_stmt.Reset_meter ]
    | Ssa_op.Check_gather { raw; extent } ->
        let bound n = Loop_expr.I64_const n in
        [
          Loop_stmt.Fail_if
            ( Loop_bool.Or
                ( Loop_bool.I64_lt (ex raw, bound (Int64.neg extent)),
                  Loop_bool.Not (Loop_bool.I64_lt (ex raw, bound extent)) ),
              Loop_failure.Gather_out_of_range
                { raw = ex raw; extent = int_of extent } );
        ]
    | Ssa_op.Float_to_i64 a ->
        (* NaN, an infinity, or outside [-2^63, 2^63): the upper bound is the
           exact power of two and exclusive, not [Int64.max_int]'s float, which
           rounds up to that same power *)
        let two63 = Float.pow 2. 63. in
        let v = fx a in
        [
          Loop_stmt.Fail_if
            ( Loop_bool.Or
                ( Loop_bool.Not (Loop_bool.Value_eq (v, v)),
                  Loop_bool.Or
                    ( Loop_bool.Value_lt (v, Loop_expr.Const (-.two63)),
                      Loop_bool.Not
                        (Loop_bool.Value_lt (v, Loop_expr.Const two63)) ) ),
              Loop_failure.I64_from_float { value = v } );
          Loop_stmt.Assign
            (Loop_carrier.Int64, temp (result ()), Loop_expr.Float_to_i64 v);
        ]
    | Ssa_op.I64_arith (o, a, b) ->
        let op =
          match o with
          | Ssa_op.I64_op.Add -> Expr.Value.I64_add
          | Ssa_op.I64_op.Mul -> Expr.Value.I64_mul
          | Ssa_op.I64_op.Sub -> Expr.Value.I64_sub
        in
        [
          Loop_stmt.Assign
            ( Loop_carrier.Int64,
              temp (result ()),
              Loop_expr.I64_binary (op, ex a, ex b) );
        ]
    | Ssa_op.I64_compare (c, a, b) ->
        define_pred (result ())
          (match c with
          | Ssa_op.Compare.Eq -> Loop_bool.I64_eq (ex a, ex b)
          | Ssa_op.Compare.Lt -> Loop_bool.I64_lt (ex a, ex b))
    | Ssa_op.I64_div (a, b) ->
        (* a zero divisor first, then [min_int / -1], as the reference checks *)
        [
          Loop_stmt.Fail_if
            ( Loop_bool.I64_eq (ex b, Loop_expr.I64_const 0L),
              Loop_failure.I64_division_by_zero );
          Loop_stmt.If
            ( Loop_bool.I64_eq (ex a, Loop_expr.I64_const Int64.min_int),
              [
                Loop_stmt.Fail_if
                  ( Loop_bool.I64_eq (ex b, Loop_expr.I64_const (-1L)),
                    Loop_failure.I64_division_overflow );
              ],
              [] );
          Loop_stmt.Assign
            ( Loop_carrier.Int64,
              temp (result ()),
              Loop_expr.I64_binary (Expr.Value.I64_div, ex a, ex b) );
        ]
    | Ssa_op.Index_of_i64 a ->
        [ Loop_stmt.Assign_index_of_i64 (temp (result ()), ex a) ]
    | Ssa_op.Float_compare (c, a, b) ->
        define_pred (result ())
          (match c with
          | Ssa_op.Compare.Eq -> Loop_bool.Value_eq (fx a, fx b)
          | Ssa_op.Compare.Lt -> Loop_bool.Value_lt (fx a, fx b))
    | Ssa_op.Float_max (a, b) ->
        [
          Loop_stmt.Assign
            ( Loop_carrier.Float,
              temp (result ()),
              Loop_expr.Float_max (fx a, fx b) );
        ]
    | Ssa_op.Float_unary (u, a) ->
        [
          Loop_stmt.Assign
            (Loop_carrier.Float, temp (result ()), Loop_expr.Unary (u, fx a));
        ]
    | Ssa_op.Index_ceil_div (k, a) ->
        [
          Loop_stmt.Assign_index
            (temp (result ()), Loop_index.Ceil_div_pos (ix a, int_of k));
        ]
    | Ssa_op.Index_clamp_low a ->
        [
          Loop_stmt.Assign_index (temp (result ()), Loop_index.Clamp_low (ix a));
        ]
    | Ssa_op.Index_compare (c, a, b) ->
        define_pred (result ())
          (match c with
          | Ssa_op.Compare.Eq -> Loop_bool.Index_eq (ix a, ix b)
          | Ssa_op.Compare.Lt -> Loop_bool.Index_lt (ix a, ix b))
    | Ssa_op.Index_floor_div (k, a) ->
        [
          Loop_stmt.Assign_index
            (temp (result ()), Loop_index.Floor_div_pos (ix a, int_of k));
        ]
    | Ssa_op.Index_max (a, b) ->
        [
          Loop_stmt.Assign_index (temp (result ()), Loop_index.Max (ix a, ix b));
        ]
    | Ssa_op.Index_min (a, b) ->
        [
          Loop_stmt.Assign_index (temp (result ()), Loop_index.Min (ix a, ix b));
        ]
    | Ssa_op.Pool_better (a, b) ->
        define_pred (result ()) (Loop_bool.Pool_better (fx a, fx b))
    | Ssa_op.Pred_not a -> define_pred (result ()) (Loop_bool.Not (pred_of a))
    | Ssa_op.Pred_or (a, b) ->
        define_pred (result ()) (Loop_bool.Or (pred_of a, pred_of b))
    | Ssa_op.Select (p, a, b) -> (
        let r = result () in
        match r.Ssa_value.ty with
        | Ssa_type.Scalar (Ssa_type.F32 | Ssa_type.F64) ->
            [
              Loop_stmt.Assign
                ( Loop_carrier.Float,
                  temp r,
                  Loop_expr.Select (pred_of p, fx a, fx b) );
            ]
        | Ssa_type.Scalar Ssa_type.I64 ->
            [
              Loop_stmt.Assign
                ( Loop_carrier.Int64,
                  temp r,
                  Loop_expr.Select (pred_of p, ex a, ex b) );
            ]
        | _ ->
            (* the Loop index language has no conditional: a statement *)
            [
              Loop_stmt.If
                ( pred_of p,
                  [ Loop_stmt.Assign_index (temp r, ix a) ],
                  [ Loop_stmt.Assign_index (temp r, ix b) ] );
            ])
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
        | Ssa_const.Pred b ->
            define_pred r (if b then always else Loop_bool.Not always))
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
        | Ssa_op.Convert.I64_to_f32 ->
            (* the Loop IR rounds an int64 once only at the working precision
               of the whole kernel, never to binary32 on its own *)
            invalid_arg "Loop_of_ssa: i64_to_f32 has no Loop form"
        | Ssa_op.Convert.I64_to_f64 ->
            [
              Loop_stmt.Assign
                (Loop_carrier.Float, temp r, Loop_expr.I64_to_float (ex a));
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
    | Ssa_op.Index_add_in_domain (a, b) ->
        [
          Loop_stmt.Assign_index (temp (result ()), Loop_index.Add (ix a, ix b));
        ]
    | Ssa_op.Index_scale_in_domain (k, a) ->
        [
          Loop_stmt.Assign_index
            (temp (result ()), Loop_index.Scale (int_of k, ix a));
        ]
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
    | Ssa_op.Load { buffer = id; at; decode }
    | Ssa_op.Load_in_bounds { buffer = id; at; decode } -> (
        let b = buffer id in
        let r = result () in
        (* a load proved in bounds needs no guard of its own *)
        let guard c =
          match i.Ssa_instr.op with
          | Ssa_op.Load_in_bounds _ -> []
          | _ -> [ load_guard b c ]
        in
        match (access at, decode = Ssa_op.Decode.I64) with
        | `Coord c, false ->
            guard c
            @ [
                Loop_stmt.Assign
                  (Loop_carrier.Float, temp r, Loop_expr.Load (b, c));
              ]
        | `Flat o, false ->
            [
              Loop_stmt.Assign
                (Loop_carrier.Float, temp r, Loop_expr.Load_flat (b, o));
            ]
        | `Coord c, true ->
            guard c
            @ [
                Loop_stmt.Assign
                  (Loop_carrier.Int64, temp r, Loop_expr.Load_i64 (b, c));
              ]
        | `Flat o, true ->
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
    | Ssa_op.Lanewise _ | Ssa_op.Mark_lanes _ | Ssa_op.Vec_extract _
    | Ssa_op.Vec_insert _ | Ssa_op.Vec_iota _ | Ssa_op.Vec_load _
    | Ssa_op.Vec_splat _ | Ssa_op.Vec_store _ ->
        invalid_arg
          "Loop_of_ssa: a vector operation has no Loop form; expand it to \
           scalar lanes first"
  in
  let drop_effect vs =
    List.filter (fun v -> not (is_type Ssa_type.Effect v)) vs
  in
  let rec region (r : Ssa_region.t) = List.concat_map stmt r.Ssa_region.body
  and stmt : Ssa_region.t Ssa_stmt.t -> Loop_stmt.t list = function
    | Ssa_stmt.Instr i -> op i
    | Ssa_stmt.For { lo; hi; step; inits; results; body } ->
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
        (* the Loop IR counts in unit steps: a stride is a count of trips and
           the induction value is reconstructed from the trip number *)
        let counter, count, induction =
          if Int64.equal step 1L then (var iv, ix hi, Loop_index.Var (var iv))
          else
            let k =
              Loop_var.of_int
                (let n = !fresh in
                 incr fresh;
                 n)
            in
            ( k,
              Loop_index.Ceil_div_pos
                ( Loop_index.Add (ix hi, Loop_index.Scale (-1, ix lo)),
                  int_of step ),
              Loop_index.Add
                (ix lo, Loop_index.Scale (int_of step, Loop_index.Var k)) )
        in
        let first = if Int64.equal step 1L then ix lo else Loop_index.Const 0 in
        List.concat (List.map2 copy params inits)
        @ [
            Loop_stmt.For
              {
                var = counter;
                lo = first;
                hi = count;
                body =
                  Loop_stmt.Assign_index (temp iv, induction)
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
        [ Loop_stmt.If (pred_of cond, branch then_, branch else_) ]
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
