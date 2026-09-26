let rec index ~f (idx : Loop_index.t) : Loop_index.t =
  f
    (match idx with
    | Loop_index.Add (a, b) -> Loop_index.Add (index ~f a, index ~f b)
    | Loop_index.Ceil_div_pos (a, k) -> Loop_index.Ceil_div_pos (index ~f a, k)
    | Loop_index.Clamp_low a -> Loop_index.Clamp_low (index ~f a)
    | Loop_index.Const _ -> idx
    | Loop_index.Floor_div_pos (a, k) -> Loop_index.Floor_div_pos (index ~f a, k)
    | Loop_index.Max (a, b) -> Loop_index.Max (index ~f a, index ~f b)
    | Loop_index.Min (a, b) -> Loop_index.Min (index ~f a, index ~f b)
    | Loop_index.Scale (k, a) -> Loop_index.Scale (k, index ~f a)
    | Loop_index.Temp _ -> idx
    | Loop_index.Var _ -> idx)

let coord ~f (c : Loop_index.coord) = Expr.Coord.map (index ~f) c

let rec expr : type a.
    f:(Loop_index.t -> Loop_index.t) -> a Loop_expr.t -> a Loop_expr.t =
 fun ~f e ->
  match e with
  | Loop_expr.Array_get (arr, idx) -> Loop_expr.Array_get (arr, index ~f idx)
  | Loop_expr.Binary (op, a, b) -> Loop_expr.Binary (op, expr ~f a, expr ~f b)
  | Loop_expr.Const _ -> e
  | Loop_expr.Float_max (a, b) -> Loop_expr.Float_max (expr ~f a, expr ~f b)
  | Loop_expr.Float_to_i64 a -> Loop_expr.Float_to_i64 (expr ~f a)
  | Loop_expr.I64_binary (op, a, b) ->
      Loop_expr.I64_binary (op, expr ~f a, expr ~f b)
  | Loop_expr.I64_const _ -> e
  | Loop_expr.I64_of_index idx -> Loop_expr.I64_of_index (index ~f idx)
  | Loop_expr.I64_to_float a -> Loop_expr.I64_to_float (expr ~f a)
  | Loop_expr.Load (buf, c) -> Loop_expr.Load (buf, coord ~f c)
  | Loop_expr.Load_flat (buf, i) -> Loop_expr.Load_flat (buf, index ~f i)
  | Loop_expr.Load_i64 (buf, c) -> Loop_expr.Load_i64 (buf, coord ~f c)
  | Loop_expr.Load_i64_flat (buf, i) -> Loop_expr.Load_i64_flat (buf, index ~f i)
  | Loop_expr.Round_f32 a -> Loop_expr.Round_f32 (expr ~f a)
  | Loop_expr.Select (p, a, b) ->
      Loop_expr.Select (pred ~f p, expr ~f a, expr ~f b)
  | Loop_expr.Temp _ -> e
  | Loop_expr.Unary (op, a) -> Loop_expr.Unary (op, expr ~f a)
  | Loop_expr.Value_of_index idx -> Loop_expr.Value_of_index (index ~f idx)

and pred ~f (p : Loop_expr.pred) : Loop_expr.pred =
  match p with
  | Loop_bool.I64_eq (a, b) -> Loop_bool.I64_eq (expr ~f a, expr ~f b)
  | Loop_bool.I64_lt (a, b) -> Loop_bool.I64_lt (expr ~f a, expr ~f b)
  | Loop_bool.Index_eq (a, b) -> Loop_bool.Index_eq (index ~f a, index ~f b)
  | Loop_bool.Index_lt (a, b) -> Loop_bool.Index_lt (index ~f a, index ~f b)
  | Loop_bool.Index_overflows a -> Loop_bool.Index_overflows (index ~f a)
  | Loop_bool.Not p -> Loop_bool.Not (pred ~f p)
  | Loop_bool.Or (a, b) -> Loop_bool.Or (pred ~f a, pred ~f b)
  | Loop_bool.Out_of_range (a, k) -> Loop_bool.Out_of_range (index ~f a, k)
  | Loop_bool.Pool_better (a, b) -> Loop_bool.Pool_better (expr ~f a, expr ~f b)
  | Loop_bool.Value_eq (a, b) -> Loop_bool.Value_eq (expr ~f a, expr ~f b)
  | Loop_bool.Value_lt (a, b) -> Loop_bool.Value_lt (expr ~f a, expr ~f b)

let stored ~f : Loop_stored.t -> Loop_stored.t = function
  | Loop_stored.Bool e -> Loop_stored.Bool (expr ~f e)
  | Loop_stored.F32 e -> Loop_stored.F32 (expr ~f e)
  | Loop_stored.I64 e -> Loop_stored.I64 (expr ~f e)

let failure ~f : Loop_failure.t -> Loop_failure.t = function
  | Loop_failure.Gather_out_of_range { raw; extent } ->
      Loop_failure.Gather_out_of_range { raw = expr ~f raw; extent }
  | Loop_failure.I64_division_by_zero as fl -> fl
  | Loop_failure.I64_division_overflow as fl -> fl
  | Loop_failure.I64_from_float { value } ->
      Loop_failure.I64_from_float { value = expr ~f value }
  | Loop_failure.Index_overflow { index = i } ->
      Loop_failure.Index_overflow { index = index ~f i }
  | Loop_failure.Load_out_of_range { buffer; coord = c } ->
      Loop_failure.Load_out_of_range { buffer; coord = coord ~f c }
  | Loop_failure.Local_out_of_range { local; index = i; extent } ->
      Loop_failure.Local_out_of_range { local; index = index ~f i; extent }
  | Loop_failure.Scan_lane_out_of_range { local; row; lane; extent } ->
      Loop_failure.Scan_lane_out_of_range
        { local; row = index ~f row; lane = index ~f lane; extent }
  | Loop_failure.Scan_row_out_of_range { local; row; lane; extent } ->
      Loop_failure.Scan_row_out_of_range
        { local; row = index ~f row; lane = index ~f lane; extent }

let rec stmts ~f (ss : Loop_stmt.t list) = List.map (stmt ~f) ss

and stmt ~f (s : Loop_stmt.t) : Loop_stmt.t =
  match s with
  | Loop_stmt.Alloc _ -> s
  | Loop_stmt.Array_set (arr, idx, e) ->
      Loop_stmt.Array_set (arr, index ~f idx, expr ~f e)
  | Loop_stmt.Assign (c, t, e) -> Loop_stmt.Assign (c, t, expr ~f e)
  | Loop_stmt.Assign_index (t, idx) -> Loop_stmt.Assign_index (t, index ~f idx)
  | Loop_stmt.Assign_index_of_i64 (t, e) ->
      Loop_stmt.Assign_index_of_i64 (t, expr ~f e)
  | Loop_stmt.Charge_scan_update -> s
  | Loop_stmt.Fail_if (p, fl) -> Loop_stmt.Fail_if (pred ~f p, failure ~f fl)
  | Loop_stmt.For { var; lo; hi; body } ->
      Loop_stmt.For
        { var; lo = index ~f lo; hi = index ~f hi; body = stmts ~f body }
  | Loop_stmt.If (p, a, b) -> Loop_stmt.If (pred ~f p, stmts ~f a, stmts ~f b)
  | Loop_stmt.Mark _ -> s
  | Loop_stmt.Release_scan_state _ -> s
  | Loop_stmt.Reserve_scan_state _ -> s
  | Loop_stmt.Reset_meter -> s
  | Loop_stmt.Store { buffer; coord = c; value } ->
      Loop_stmt.Store { buffer; coord = coord ~f c; value = stored ~f value }
  | Loop_stmt.Store_flat { buffer; offset; value } ->
      Loop_stmt.Store_flat
        { buffer; offset = index ~f offset; value = stored ~f value }
