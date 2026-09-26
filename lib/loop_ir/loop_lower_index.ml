(* Index lowering, and the two checks an emitted access needs.

   [Expr.Eval.index] checks every [Add] and [Scale] and reports overflow, then a
   load checks its coordinate against the source's shape. Both become explicit
   [Fail_if]s, and each is emitted only where the interval proof over the
   concrete loop extents cannot discharge it, so a well-behaved kernel carries
   none. *)

open Loop_lower_ctx

let rec index : type role. Loop_lower_ctx.t -> role Expr.Index.t -> Loop_index.t
    =
 fun ctx -> function
  | Expr.Index.Add (a, b) ->
      let a = index ctx a in
      let b = index ctx b in
      Loop_index.Add (a, b)
  | Expr.Index.Assume_position a -> index ctx a
  | Expr.Index.Ceil_div_pos (a, d) -> Loop_index.Ceil_div_pos (index ctx a, d)
  | Expr.Index.Clamp_low a -> Loop_index.Clamp_low (index ctx a)
  | Expr.Index.Const n -> Loop_index.Const n
  | Expr.Index.Data (src, coord, extent) -> gather ctx src coord ~extent
  | Expr.Index.Floor_div_pos (a, d) -> Loop_index.Floor_div_pos (index ctx a, d)
  | Expr.Index.Max (a, b) ->
      let a = index ctx a in
      let b = index ctx b in
      Loop_index.Max (a, b)
  | Expr.Index.Min (a, b) ->
      let a = index ctx a in
      let b = index ctx b in
      Loop_index.Min (a, b)
  | Expr.Index.Of_position a -> index ctx a
  | Expr.Index.Output axis -> Expr.Coord.get ctx.axes axis
  | Expr.Index.Reduce v -> (
      match Expr.Reduce_var.Map.find_opt v ctx.reducers with
      | Some i -> i
      | None -> invalid_arg "Loop_lower: reducer is not in scope")
  | Expr.Index.Scale (k, a) -> Loop_index.Scale (k, index ctx a)
  | Expr.Index.Zero -> Loop_index.Const 0

(* The overflow guard for an index about to be used. [Index_overflows] on an
   index the proof already covers would be dead weight in every emitted loop. *)
and guarded ctx i =
  if not (Loop_range.proven ctx.ranges i) then
    emit ctx
      (Loop_stmt.Fail_if
         (Loop_bool.Index_overflows i, Loop_failure.Index_overflow { index = i }));
  i

and checked : type role. Loop_lower_ctx.t -> role Expr.Index.t -> Loop_index.t =
 fun ctx i -> guarded ctx (index ctx i)

(* A load's coordinate, already lowered: every component is guarded against
   overflow, in the order [Expr.Eval] evaluates them and before the bounds check,
   and the bounds check names the buffer and covers only the axes the proof
   leaves open. The failure row reports the first failing axis whichever subset
   of axes the check tests, because a proven axis cannot fail. *)
and load_guard ctx (b : Loop_buffer.t) (c : Loop_index.coord) : Loop_index.coord
    =
  let components =
    List.map (fun a -> (a, guarded ctx (Expr.Coord.get c a))) Expr.Axis.all
  in
  let coord = Expr.Coord.of_fn (fun a -> List.assoc a components) in
  let extent a = Dim.to_int (Vec6.get b.Loop_buffer.sg.Tensor_sig.shape a) in
  let open_axes =
    List.filter
      (fun (a, i) ->
        not
          (Loop_range.within
             ~inner:(Loop_range.of_index ctx.ranges i)
             ~outer:{ Loop_range.lo = 0L; hi = Int64.of_int (extent a - 1) }))
      components
  in
  (match open_axes with
  | [] -> ()
  | (a, i) :: rest ->
      let cond =
        List.fold_left
          (fun acc (a, i) ->
            Loop_bool.Or (acc, Loop_bool.Out_of_range (i, extent a)))
          (Loop_bool.Out_of_range (i, extent a))
          rest
      in
      emit ctx
        (Loop_stmt.Fail_if
           (cond, Loop_failure.Load_out_of_range { buffer = b; coord })));
  coord

and load_coord : type role.
    Loop_lower_ctx.t ->
    Loop_buffer.t ->
    role Expr.Index.t Expr.Coord.t ->
    Loop_index.coord =
 fun ctx b c ->
  load_guard ctx b (Expr.Coord.of_fn (fun a -> index ctx (Expr.Coord.get c a)))

(* [Index.Data]: a runtime gather. The source's raw stored int64 is read (its
   coordinate guarded and bounds-checked like any load), checked against ATen's
   valid range [-extent, extent - 1] IN THE INT64 DOMAIN, normalized (a negative
   value gains [extent]), and only then narrowed to an index. Narrowing first
   would let a value near [Int64.min_int] wrap into a spuriously in-range int,
   which is the defect the check before the narrowing exists to prevent. *)
and gather ctx src coord ~extent =
  let id = Expr_bridge.id_of_source src in
  let raw_of b =
    let c = load_coord ctx b coord in
    Loop_expr.Load_i64 (b, c)
  in
  let raw =
    match Tensor_id.Map.find_opt id ctx.sources with
    | None -> refuse ctx Loop_unsupported.Unmaterialized_source
    | Some (Buffer b) when Loop_lower_ctx.format_name b = "i64" -> raw_of b
    | Some (Fill_i64 (b, v)) ->
        ignore (load_coord ctx b coord);
        Loop_expr.I64_const v
    | Some (Buffer b | Fill (b, _)) ->
        refuse ctx (Loop_unsupported.Load_format (Loop_lower_ctx.format_name b))
  in
  let t = fresh_temp ctx in
  emit ctx (Loop_stmt.Assign (Loop_carrier.Int64, t, raw));
  let raw = Loop_expr.Temp (Loop_carrier.Int64, t) in
  let bound n = Loop_expr.I64_const (Int64.of_int n) in
  emit ctx
    (Loop_stmt.Fail_if
       ( Loop_bool.Or
           ( Loop_bool.I64_lt (raw, bound (-extent)),
             Loop_bool.Not (Loop_bool.I64_lt (raw, bound extent)) ),
         Loop_failure.Gather_out_of_range { raw; extent } ));
  let normalized = fresh_temp ctx in
  emit ctx
    (Loop_stmt.Assign_index_of_i64
       ( normalized,
         Loop_expr.Select
           ( Loop_bool.I64_lt (raw, Loop_expr.I64_const 0L),
             Loop_expr.I64_binary (Expr.Value.I64_add, raw, bound extent),
             raw ) ));
  Loop_range.Env.set_temp normalized
    { Loop_range.lo = 0L; hi = Int64.of_int (extent - 1) }
    ctx.ranges;
  Loop_index.Temp normalized
