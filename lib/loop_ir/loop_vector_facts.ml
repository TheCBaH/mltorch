(* Facts about indices, expressions and statements that both the vectorizer and
   the verifier read, so what one proves the other checks the same way. *)

let rec mentions v : Loop_index.t -> bool = function
  | Loop_index.Var v' -> Loop_var.equal v v'
  | Loop_index.Add (a, b) | Loop_index.Max (a, b) | Loop_index.Min (a, b) ->
      mentions v a || mentions v b
  | Loop_index.Ceil_div_pos (a, _)
  | Loop_index.Clamp_low a
  | Loop_index.Floor_div_pos (a, _)
  | Loop_index.Scale (_, a) ->
      mentions v a
  | Loop_index.Const _ | Loop_index.Temp _ -> false

(* The coefficient of the loop variable in an offset, provided the offset is an
   exact linear form whose only dependence on the variable is through [Var v]
   itself. *)
let coefficient var offset =
  match Loop_linear.of_index offset with
  | None -> Error `Not_affine
  | Some lin ->
      let opaque =
        List.exists
          (fun (atom, _) ->
            (not (atom = Loop_index.Var var)) && mentions var atom)
          lin.Loop_linear.terms
      in
      if opaque then Error `Not_affine
      else Ok (Loop_linear.coefficient lin (Loop_index.Var var))

let format_ok (b : Loop_buffer.t) =
  let (Payload.Fmt f) = b.Loop_buffer.sg.Tensor_sig.fmt in
  match Payload.fmt_name f with
  | "bool" | "f32" | "f64" | "i32" -> true
  | _ -> false

(* The float and int temporaries the scalar loop assigns: a splat may read none
   of them, since each varies with the iteration. *)
let rec assigned_temps acc (s : Loop_stmt.t) =
  match s with
  | Loop_stmt.Assign (_, t, _)
  | Loop_stmt.Assign_index (t, _)
  | Loop_stmt.Assign_index_of_i64 (t, _) ->
      Loop_temp.Set.add t acc
  | Loop_stmt.For { body; _ } -> List.fold_left assigned_temps acc body
  | Loop_stmt.If (_, a, b) ->
      List.fold_left assigned_temps (List.fold_left assigned_temps acc a) b
  | Loop_stmt.Alloc _ | Loop_stmt.Array_set _ | Loop_stmt.Charge_scan_update
  | Loop_stmt.Fail_if _ | Loop_stmt.Mark _ | Loop_stmt.Release_scan_state _
  | Loop_stmt.Reserve_scan_state _ | Loop_stmt.Reset_meter | Loop_stmt.Store _
  | Loop_stmt.Store_flat _ ->
      acc

(* Whether a scalar expression reads the loop variable or one of [temps]. *)
let rec expr_depends : type a.
    Loop_var.t -> Loop_temp.Set.t -> a Loop_expr.t -> bool =
 fun v temps e ->
  let idx = mentions v in
  let coord (c : Loop_index.coord) =
    List.exists (fun a -> idx (Expr.Coord.get c a)) Expr.Axis.all
  in
  match e with
  | Loop_expr.Array_get (_, i) -> idx i
  | Loop_expr.Binary (_, a, b) | Loop_expr.Float_max (a, b) ->
      expr_depends v temps a || expr_depends v temps b
  | Loop_expr.Const _ | Loop_expr.I64_const _ -> false
  | Loop_expr.Float_to_i64 a -> expr_depends v temps a
  | Loop_expr.I64_to_float a -> expr_depends v temps a
  | Loop_expr.Round_f32 a -> expr_depends v temps a
  | Loop_expr.Unary (_, a) -> expr_depends v temps a
  | Loop_expr.I64_binary (_, a, b) ->
      expr_depends v temps a || expr_depends v temps b
  | Loop_expr.I64_of_index i | Loop_expr.Value_of_index i -> idx i
  | Loop_expr.Load (_, c) | Loop_expr.Load_i64 (_, c) -> coord c
  | Loop_expr.Load_flat (_, i) | Loop_expr.Load_i64_flat (_, i) -> idx i
  | Loop_expr.Select (p, a, b) ->
      pred_depends v temps p || expr_depends v temps a || expr_depends v temps b
  | Loop_expr.Temp (_, t) -> Loop_temp.Set.mem t temps

and pred_depends v temps (p : Loop_expr.pred) =
  match p with
  | Loop_bool.I64_eq (a, b) | Loop_bool.I64_lt (a, b) ->
      expr_depends v temps a || expr_depends v temps b
  | Loop_bool.Index_eq (a, b) | Loop_bool.Index_lt (a, b) ->
      mentions v a || mentions v b
  | Loop_bool.Index_overflows i | Loop_bool.Out_of_range (i, _) -> mentions v i
  | Loop_bool.Not p -> pred_depends v temps p
  | Loop_bool.Or (p, q) -> pred_depends v temps p || pred_depends v temps q
  | Loop_bool.Pool_better (a, b)
  | Loop_bool.Value_eq (a, b)
  | Loop_bool.Value_lt (a, b) ->
      expr_depends v temps a || expr_depends v temps b

let rec expr_buffers : type a. a Loop_expr.t -> Loop_buffer.t list =
 fun e ->
  match e with
  | Loop_expr.Load (b, _)
  | Loop_expr.Load_i64 (b, _)
  | Loop_expr.Load_flat (b, _)
  | Loop_expr.Load_i64_flat (b, _) ->
      [ b ]
  | Loop_expr.Binary (_, a, b) | Loop_expr.Float_max (a, b) ->
      expr_buffers a @ expr_buffers b
  | Loop_expr.I64_binary (_, a, b) -> expr_buffers a @ expr_buffers b
  | Loop_expr.Float_to_i64 a -> expr_buffers a
  | Loop_expr.I64_to_float a -> expr_buffers a
  | Loop_expr.Round_f32 a -> expr_buffers a
  | Loop_expr.Unary (_, a) -> expr_buffers a
  | Loop_expr.Select (p, a, b) ->
      pred_buffers p @ expr_buffers a @ expr_buffers b
  | Loop_expr.Array_get _ | Loop_expr.Const _ | Loop_expr.I64_const _
  | Loop_expr.I64_of_index _ | Loop_expr.Temp _ | Loop_expr.Value_of_index _ ->
      []

and pred_buffers (p : Loop_expr.pred) =
  match p with
  | Loop_bool.I64_eq (a, b) | Loop_bool.I64_lt (a, b) ->
      expr_buffers a @ expr_buffers b
  | Loop_bool.Pool_better (a, b)
  | Loop_bool.Value_eq (a, b)
  | Loop_bool.Value_lt (a, b) ->
      expr_buffers a @ expr_buffers b
  | Loop_bool.Not p -> pred_buffers p
  | Loop_bool.Or (p, q) -> pred_buffers p @ pred_buffers q
  | Loop_bool.Index_eq _ | Loop_bool.Index_lt _ | Loop_bool.Index_overflows _
  | Loop_bool.Out_of_range _ ->
      []

(* Every float or int64 temporary an expression reads. *)
let rec expr_temps : type a. a Loop_expr.t -> Loop_temp.t list =
 fun e ->
  match e with
  | Loop_expr.Temp (_, t) -> [ t ]
  | Loop_expr.Binary (_, a, b) | Loop_expr.Float_max (a, b) ->
      expr_temps a @ expr_temps b
  | Loop_expr.I64_binary (_, a, b) -> expr_temps a @ expr_temps b
  | Loop_expr.Float_to_i64 a -> expr_temps a
  | Loop_expr.I64_to_float a -> expr_temps a
  | Loop_expr.Round_f32 a -> expr_temps a
  | Loop_expr.Unary (_, a) -> expr_temps a
  | Loop_expr.Select (p, a, b) -> pred_temps p @ expr_temps a @ expr_temps b
  | Loop_expr.Array_get _ | Loop_expr.Const _ | Loop_expr.I64_const _
  | Loop_expr.I64_of_index _ | Loop_expr.Load _ | Loop_expr.Load_flat _
  | Loop_expr.Load_i64 _ | Loop_expr.Load_i64_flat _
  | Loop_expr.Value_of_index _ ->
      []

and pred_temps (p : Loop_expr.pred) =
  match p with
  | Loop_bool.I64_eq (a, b) | Loop_bool.I64_lt (a, b) ->
      expr_temps a @ expr_temps b
  | Loop_bool.Pool_better (a, b)
  | Loop_bool.Value_eq (a, b)
  | Loop_bool.Value_lt (a, b) ->
      expr_temps a @ expr_temps b
  | Loop_bool.Not p -> pred_temps p
  | Loop_bool.Or (p, q) -> pred_temps p @ pred_temps q
  | Loop_bool.Index_eq _ | Loop_bool.Index_lt _ | Loop_bool.Index_overflows _
  | Loop_bool.Out_of_range _ ->
      []

(* The temporaries a statement list reads, with multiplicity. *)
let rec stmt_temp_reads acc (s : Loop_stmt.t) =
  match s with
  | Loop_stmt.Assign (_, _, e) -> expr_temps e @ acc
  | Loop_stmt.Assign_index_of_i64 (_, e) -> expr_temps e @ acc
  | Loop_stmt.Array_set (_, _, e) -> expr_temps e @ acc
  | Loop_stmt.Store { value; _ } | Loop_stmt.Store_flat { value; _ } -> (
      match value with
      | Loop_stored.Bool e | Loop_stored.F32 e -> expr_temps e @ acc
      | Loop_stored.I64 e -> expr_temps e @ acc)
  | Loop_stmt.Fail_if (p, _) -> pred_temps p @ acc
  | Loop_stmt.If (p, a, b) ->
      List.fold_left stmt_temp_reads
        (List.fold_left stmt_temp_reads (pred_temps p @ acc) a)
        b
  | Loop_stmt.For { body; _ } -> List.fold_left stmt_temp_reads acc body
  | Loop_stmt.Alloc _ | Loop_stmt.Assign_index _ | Loop_stmt.Charge_scan_update
  | Loop_stmt.Mark _ | Loop_stmt.Release_scan_state _
  | Loop_stmt.Reserve_scan_state _ | Loop_stmt.Reset_meter ->
      acc

let rec has_loop (s : Loop_stmt.t) =
  match s with
  | Loop_stmt.For _ -> true
  | Loop_stmt.If (_, a, b) -> List.exists has_loop a || List.exists has_loop b
  | _ -> false
