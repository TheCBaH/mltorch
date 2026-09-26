(* Lowering a float [Expr.Value.t] body to statements plus one pure expression.

   The body is walked in evaluation order and every effect it needs (a bounds
   check, a reduction loop) is appended to the current block BEFORE the
   expression that consumes it, so what is left is a pure tree over temporaries
   and loads. A [Select] whose branches need no statements stays a [Select]; one
   whose branch needs a check or a loop becomes an [If], because only the
   selected branch may fail or run. *)

open Loop_lower_ctx

let round_f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let temp_float t = Loop_expr.Temp (Loop_carrier.Float, t)
let temp_i64 t = Loop_expr.Temp (Loop_carrier.Int64, t)

let assign_i64 ctx e =
  let t = fresh_temp ctx in
  emit ctx (Loop_stmt.Assign (Loop_carrier.Int64, t, e));
  t

(* [v] is NaN, an infinity, or outside [-2^63, 2^63): the values a float cannot
   become an int64 from. The upper bound is the exact power of two and exclusive,
   not [Int64.max_int]'s float, which rounds up to that same power and would admit
   it. *)
let out_of_i64_range v =
  let two63 = Float.pow 2. 63. in
  Loop_bool.Or
    ( Loop_bool.Not (Loop_bool.Value_eq (v, v)),
      Loop_bool.Or
        ( Loop_bool.Value_lt (v, Loop_expr.Const (-.two63)),
          Loop_bool.Not (Loop_bool.Value_lt (v, Loop_expr.Const two63)) ) )

let assign_float ctx e =
  let t = fresh_temp ctx in
  emit ctx (Loop_stmt.Assign (Loop_carrier.Float, t, e));
  t

(* A fresh loop variable over [lo, hi) and the ranges with it bound: the
   interval it takes is what the proof of every index inside the loop rests on. *)
let bind_loop ctx ~lo ~hi =
  let var = fresh_var ctx in
  let lo_range = Loop_range.of_index ctx.ranges lo
  and hi_range = Loop_range.of_index ctx.ranges hi in
  let range =
    {
      Loop_range.lo = lo_range.Loop_range.lo;
      hi = Stdlib.max lo_range.Loop_range.lo (Int64.pred hi_range.Loop_range.hi);
    }
  in
  (var, Loop_range.Env.add_var var range ctx.ranges)

let rec value ctx : float Expr.Value.t -> float Loop_expr.t = function
  | Expr.Value.Binary (op, a, b) ->
      let a = value ctx a in
      let b = value ctx b in
      Loop_expr.Binary (op, a, b)
  | Expr.Value.Const x -> Loop_expr.Const x
  | Expr.Value.I64_to_float a -> Loop_expr.I64_to_float (value_i64 ctx a)
  | Expr.Value.Intrinsic (Expr.Intrinsic.Max_pool d) -> max_pool ctx d
  | Expr.Value.Load (src, coord) -> load ctx src coord
  | Expr.Value.Local v -> local ctx v
  | Expr.Value.Local_at (v, i) -> local_at ctx v i
  | Expr.Value.Local_scan_at (v, row, lane) -> scan_cached ctx v row lane
  | Expr.Value.Scan_at (s, row, lane) -> scan_inline ctx s row lane
  | Expr.Value.Reduce r -> reduce ctx r
  | Expr.Value.Round_f32 a -> Loop_expr.Round_f32 (value ctx a)
  | Expr.Value.Select (p, a, b) -> select ctx Loop_carrier.Float value p a b
  | Expr.Value.Unary (op, a) -> Loop_expr.Unary (op, value ctx a)
  | Expr.Value.Value_of_index i ->
      Loop_expr.Value_of_index (Loop_lower_index.checked ctx i)

(* A scalar local's single slot. [Region_program.check] proved the read matches
   the declared shape and comes after the write, so neither is re-checked. *)
and local ctx v =
  match local_of ctx v with
  | Slots { array; range; _ } ->
      Loop_expr.Array_get
        (array, Loop_index.Const (range.Slot.Range.offset :> int))
  | Prev_row _ -> invalid_arg "Loop_lower: prev is read only through local_at"

(* A vector local, or a scan's [prev], at a computed position. The read is
   bounds-checked against the declared extent unless the interval proof covers
   it, and fails as the reference's unbound local does: the reader answers [None]
   outside the range. *)
and local_at ctx v i =
  let array, base, extent =
    match local_of ctx v with
    | Slots { array; range; _ } ->
        let offset = (range.Slot.Range.offset :> int) in
        ( array,
          (if offset = 0 then None else Some (Loop_index.Const offset)),
          (range.Slot.Range.count :> int) )
    | Prev_row { array; base; width } -> (array, Some base, width)
  in
  let i = Loop_lower_index.checked ctx i in
  if not (within ctx i ~extent) then
    emit ctx
      (Loop_stmt.Fail_if
         ( Loop_bool.Out_of_range (i, extent),
           Loop_failure.Local_out_of_range { local = v; index = i; extent } ));
  let at = match base with None -> i | Some b -> Loop_index.Add (b, i) in
  Loop_expr.Array_get (array, Loop_lower_index.guarded ctx at)

and within ctx i ~extent =
  Loop_range.within
    ~inner:(Loop_range.of_index ctx.ranges i)
    ~outer:{ Loop_range.lo = 0L; hi = Int64.of_int (extent - 1) }

and local_of ctx v =
  match Expr.Local_var.Map.find_opt v ctx.locals with
  | Some l -> l
  | None -> invalid_arg "Loop_lower: local is not written yet"

(* The projection's bounds, row first: a simultaneous failure reports the row.
   Both indices are evaluated (and guarded) before either check, as the
   reference evaluates them. *)
and guard_projection ctx ~local ~row ~lane ~steps ~width =
  if not (within ctx row ~extent:(steps + 1)) then
    emit ctx
      (Loop_stmt.Fail_if
         ( Loop_bool.Out_of_range (row, steps + 1),
           Loop_failure.Scan_row_out_of_range
             { local; row; lane; extent = steps + 1 } ));
  if not (within ctx lane ~extent:width) then
    emit ctx
      (Loop_stmt.Fail_if
         ( Loop_bool.Out_of_range (lane, width),
           Loop_failure.Scan_lane_out_of_range
             { local; row; lane; extent = width } ))

(* A cached read of a trace local: row-major, [offset + row * width + lane]. *)
and scan_cached ctx v row lane =
  match local_of ctx v with
  | Slots { array; range; shape = Region_local.Shape.Scan { width; steps } } ->
      let width = (width :> int) and steps = (steps :> int) in
      let row = Loop_lower_index.checked ctx row in
      let lane = Loop_lower_index.checked ctx lane in
      guard_projection ctx ~local:(Some v) ~row ~lane ~steps ~width;
      let offset = (range.Slot.Range.offset :> int) in
      let at =
        Loop_index.Add
          ( Loop_index.Const offset,
            Loop_index.Add (Loop_index.Scale (width, row), lane) )
      in
      Loop_expr.Array_get (array, Loop_lower_index.guarded ctx at)
  | Slots _ | Prev_row _ ->
      invalid_arg "Loop_lower: a cached scan read of a local that is no trace"

(* An inline [Scan_at]: the recurrence re-executed, with no sharing across
   evaluations, on two rolling rows of [width] cells. The bounds are checked
   first, then the live state is reserved (the meter must exist, and here it
   always does), then the initial row is filled, then [row] update rows each
   charged once per lane BEFORE its body runs. The next row is written to a
   second array and copied back, so [prev] reads the previous row throughout the
   update, as the reference's [Array.init] over a fixed [prev_row] does. *)
and scan_inline ctx (s : Expr.Scan.t) row lane =
  let width = s.Expr.Scan.width and steps = s.Expr.Scan.steps in
  let row = Loop_lower_index.checked ctx row in
  let lane = Loop_lower_index.checked ctx lane in
  guard_projection ctx ~local:None ~row ~lane ~steps ~width;
  ctx.meter := true;
  emit ctx (Loop_stmt.Reserve_scan_state width);
  let prev = fresh_array ctx and cur = fresh_array ctx in
  let size = Slot.(count_of_extent (extent width)) in
  ctx.hoisted :=
    Loop_stmt.Alloc (cur, size)
    :: Loop_stmt.Alloc (prev, size)
    :: !(ctx.hoisted);
  let lane_range = Loop_range.span ~lo:0 ~hi:width in
  let each_lane ctx ~bind body =
    let l = fresh_var ctx in
    let inner =
      {
        ctx with
        reducers = bind (Loop_index.Var l) ctx.reducers;
        ranges = Loop_range.Env.add_var l lane_range ctx.ranges;
      }
    in
    let (), stmts =
      in_block inner (fun inner -> body inner (Loop_index.Var l))
    in
    Loop_stmt.For
      {
        var = l;
        lo = Loop_index.Const 0;
        hi = Loop_index.Const width;
        body = stmts;
      }
  in
  emit ctx
    (each_lane ctx ~bind:(Expr.Reduce_var.Map.add s.Expr.Scan.lane)
       (fun inner l ->
         let e = value inner s.Expr.Scan.init in
         emit inner (Loop_stmt.Array_set (prev, l, e))));
  let step, ranges = bind_loop ctx ~lo:(Loop_index.Const 0) ~hi:row in
  let stepper =
    {
      ctx with
      reducers =
        Expr.Reduce_var.Map.add s.Expr.Scan.step (Loop_index.Var step)
          ctx.reducers;
      ranges;
      locals =
        Expr.Local_var.Map.add s.Expr.Scan.prev
          (Prev_row { array = prev; base = Loop_index.Const 0; width })
          ctx.locals;
    }
  in
  let (), rows =
    in_block stepper (fun stepper ->
        emit stepper
          (each_lane stepper ~bind:(Expr.Reduce_var.Map.add s.Expr.Scan.lane)
             (fun inner l ->
               emit inner Loop_stmt.Charge_scan_update;
               let e = value inner s.Expr.Scan.update in
               emit inner (Loop_stmt.Array_set (cur, l, e))));
        emit stepper
          (each_lane stepper
             ~bind:(fun _ reducers -> reducers)
             (fun inner l ->
               emit inner
                 (Loop_stmt.Array_set (prev, l, Loop_expr.Array_get (cur, l))))))
  in
  emit ctx
    (Loop_stmt.For
       { var = step; lo = Loop_index.Const 0; hi = row; body = rows });
  let result = assign_float ctx (Loop_expr.Array_get (prev, lane)) in
  emit ctx (Loop_stmt.Release_scan_state width);
  temp_float result

and pred ctx : Expr.Bool.t -> Loop_expr.pred = function
  | Expr.Bool.I64_eq (a, b) ->
      let a = value_i64 ctx a in
      let b = value_i64 ctx b in
      Loop_bool.I64_eq (a, b)
  | Expr.Bool.I64_lt (a, b) ->
      let a = value_i64 ctx a in
      let b = value_i64 ctx b in
      Loop_bool.I64_lt (a, b)
  | Expr.Bool.Index_eq (a, b) ->
      let a = Loop_lower_index.checked ctx a in
      let b = Loop_lower_index.checked ctx b in
      Loop_bool.Index_eq (a, b)
  | Expr.Bool.Value_eq (a, b) ->
      let a = value ctx a in
      let b = value ctx b in
      Loop_bool.Value_eq (a, b)
  | Expr.Bool.Value_lt (a, b) ->
      let a = value ctx a in
      let b = value ctx b in
      Loop_bool.Value_lt (a, b)

and load ctx src coord =
  source_load ctx src ~coord:(fun ctx b ->
      Loop_lower_index.load_coord ctx b coord)

(* Resolve a source and read it at a coordinate the caller lowers against the
   resolved buffer. A [Filled] input still bounds-checks: only the read is
   folded, and to what a materialized fill would decode to, not to [v]. *)
and source_load ctx src ~coord =
  let id = Expr_bridge.id_of_source src in
  match Tensor_id.Map.find_opt id ctx.sources with
  | None -> refuse ctx Loop_unsupported.Unmaterialized_source
  | Some (Buffer b) -> Loop_expr.Load (b, coord ctx b)
  | Some (Fill (b, v)) ->
      ignore (coord ctx b);
      Loop_expr.Const
        (match format_name b with
        | "bool" -> if v <> 0. then 1. else 0.
        | _ -> round_f32 v)
  | Some (Fill_i64 (b, v)) ->
      ignore (coord ctx b);
      Loop_expr.Const (Int64.to_float v)

(* Only the selected branch may fail or run: a branch that needs no statements
   keeps the [Select] a pure tree, and one that needs a check or a loop becomes an
   [If] assigning a temporary. Both carriers go through here. *)
and select : type a.
    Loop_lower_ctx.t ->
    a Loop_carrier.t ->
    (Loop_lower_ctx.t -> a Expr.Value.t -> a Loop_expr.t) ->
    Expr.Bool.t ->
    a Expr.Value.t ->
    a Expr.Value.t ->
    a Loop_expr.t =
 fun ctx carrier lower p a b ->
  let p = pred ctx p in
  let a, stmts_a = in_block ctx (fun ctx -> lower ctx a) in
  let b, stmts_b = in_block ctx (fun ctx -> lower ctx b) in
  if stmts_a = [] && stmts_b = [] then Loop_expr.Select (p, a, b)
  else
    let t = fresh_temp ctx in
    let assign e = Loop_stmt.Assign (carrier, t, e) in
    emit ctx (Loop_stmt.If (p, stmts_a @ [ assign a ], stmts_b @ [ assign b ]));
    Loop_expr.Temp (carrier, t)

(* An exact int64 body: modular [+ - *], checked [/], a checked float-to-int64,
   and the ordered reductions. Failures are explicit checks emitted in evaluation
   order, before the expression that would fail. *)
and value_i64 ctx : int64 Expr.Value.t -> int64 Loop_expr.t = function
  | Expr.Value.Float_to_i64 a ->
      let v = temp_float (assign_float ctx (value ctx a)) in
      emit ctx
        (Loop_stmt.Fail_if
           (out_of_i64_range v, Loop_failure.I64_from_float { value = v }));
      Loop_expr.Float_to_i64 v
  | Expr.Value.I64_binary (Expr.Value.I64_div, a, b) ->
      let a = temp_i64 (assign_i64 ctx (value_i64 ctx a)) in
      let b = temp_i64 (assign_i64 ctx (value_i64 ctx b)) in
      (* A zero divisor first, then [min_int / -1], as [apply_i64_binary] checks. *)
      emit ctx
        (Loop_stmt.Fail_if
           ( Loop_bool.I64_eq (b, Loop_expr.I64_const 0L),
             Loop_failure.I64_division_by_zero ));
      emit ctx
        (Loop_stmt.If
           ( Loop_bool.I64_eq (a, Loop_expr.I64_const Int64.min_int),
             [
               Loop_stmt.Fail_if
                 ( Loop_bool.I64_eq (b, Loop_expr.I64_const (-1L)),
                   Loop_failure.I64_division_overflow );
             ],
             [] ));
      Loop_expr.I64_binary (Expr.Value.I64_div, a, b)
  | Expr.Value.I64_binary (op, a, b) ->
      let a = value_i64 ctx a in
      let b = value_i64 ctx b in
      Loop_expr.I64_binary (op, a, b)
  | Expr.Value.I64_const n -> Loop_expr.I64_const n
  | Expr.Value.I64_load (src, coord) -> load_i64 ctx src coord
  | Expr.Value.I64_local _ | Expr.Value.I64_local_at _ ->
      refuse ctx Loop_unsupported.Local_read
  | Expr.Value.I64_of_index i ->
      Loop_expr.I64_of_index (Loop_lower_index.checked ctx i)
  | Expr.Value.I64_sum r -> reduce_i64 ctx r
  | Expr.Value.Select (p, a, b) -> select ctx Loop_carrier.Int64 value_i64 p a b

and load_i64 ctx src coord =
  let id = Expr_bridge.id_of_source src in
  match Tensor_id.Map.find_opt id ctx.sources with
  | None -> refuse ctx Loop_unsupported.Unmaterialized_source
  | Some (Buffer b) when format_name b = "i64" ->
      Loop_expr.Load_i64 (b, Loop_lower_index.load_coord ctx b coord)
  | Some (Fill_i64 (b, v)) ->
      ignore (Loop_lower_index.load_coord ctx b coord);
      Loop_expr.I64_const v
  | Some (Buffer b | Fill (b, _)) ->
      refuse ctx (Loop_unsupported.Load_format (format_name b))

(* The int64 reductions, with their own seeds: [Sum] is modular from [0L], the
   maxima keep the signed maximum from [Int64.min_int], and [Argmax_index] reports
   the position of the FIRST maximum (a later value must be strictly greater),
   [lo] for an empty range. There is no NaN to consider. *)
and reduce_i64 ctx (r : Expr.Reduction.i64) =
  let lo = Loop_lower_index.checked ctx r.Expr.Reduction.i64_lo in
  let hi = Loop_lower_index.checked ctx r.Expr.Reduction.i64_hi in
  let var, ranges = bind_loop ctx ~lo ~hi in
  let inner =
    {
      ctx with
      reducers =
        Expr.Reduce_var.Map.add r.Expr.Reduction.i64_var (Loop_index.Var var)
          ctx.reducers;
      ranges;
    }
  in
  let loop body = emit ctx (Loop_stmt.For { var; lo; hi; body }) in
  let best = fresh_temp ctx in
  let seed =
    match r.Expr.Reduction.i64_kind with
    | Expr.Reduction.Sum -> 0L
    | _ -> Int64.min_int
  in
  emit ctx
    (Loop_stmt.Assign (Loop_carrier.Int64, best, Loop_expr.I64_const seed));
  match r.Expr.Reduction.i64_kind with
  | Expr.Reduction.Argmax_index ->
      let best_i = fresh_temp ctx in
      emit ctx (Loop_stmt.Assign_index (best_i, lo));
      let (), body =
        in_block inner (fun inner ->
            emit inner (Loop_stmt.Mark Loop_mark.Reduction);
            let x =
              temp_i64
                (assign_i64 inner (value_i64 inner r.Expr.Reduction.i64_body))
            in
            emit inner
              (Loop_stmt.If
                 ( Loop_bool.I64_lt (temp_i64 best, x),
                   [
                     Loop_stmt.Assign (Loop_carrier.Int64, best, x);
                     Loop_stmt.Assign_index (best_i, Loop_index.Var var);
                   ],
                   [] )))
      in
      loop body;
      Loop_expr.I64_of_index (Loop_index.Temp best_i)
  | (Expr.Reduction.Sum | Expr.Reduction.Max | Expr.Reduction.Argmax_value) as
    kind ->
      let (), body =
        in_block inner (fun inner ->
            emit inner (Loop_stmt.Mark Loop_mark.Reduction);
            let x =
              temp_i64
                (assign_i64 inner (value_i64 inner r.Expr.Reduction.i64_body))
            in
            let acc = temp_i64 best in
            emit inner
              (Loop_stmt.Assign
                 ( Loop_carrier.Int64,
                   best,
                   match kind with
                   | Expr.Reduction.Sum ->
                       Loop_expr.I64_binary (Expr.Value.I64_add, acc, x)
                   | _ -> Loop_expr.Select (Loop_bool.I64_lt (acc, x), x, acc)
                 )))
      in
      loop body;
      temp_i64 best

(* The ordered half-open left fold [Expr.Eval] specifies, with its seeds:
   [Sum] from [+0.] (never the first element, which would make an all-[-0.] sum
   [-0.]), [Max] from [-inf], and the argmaxes advancing value and index
   together under the one predicate [pool_better]. *)
and reduce ctx (r : Expr.Reduction.t) =
  let lo = Loop_lower_index.checked ctx r.Expr.Reduction.lo in
  let hi = Loop_lower_index.checked ctx r.Expr.Reduction.hi in
  let var, ranges = bind_loop ctx ~lo ~hi in
  let inner =
    {
      ctx with
      reducers =
        Expr.Reduce_var.Map.add r.Expr.Reduction.var (Loop_index.Var var)
          ctx.reducers;
      ranges;
    }
  in
  let loop body = emit ctx (Loop_stmt.For { var; lo; hi; body }) in
  match r.Expr.Reduction.kind with
  | Expr.Reduction.Sum | Expr.Reduction.Max ->
      let acc = fresh_temp ctx in
      let seed, combine =
        match r.Expr.Reduction.kind with
        | Expr.Reduction.Sum ->
            (0., fun a b -> Loop_expr.Binary (Expr.Value.Add, a, b))
        | _ -> (Float.neg_infinity, fun a b -> Loop_expr.Float_max (a, b))
      in
      emit ctx
        (Loop_stmt.Assign (Loop_carrier.Float, acc, Loop_expr.Const seed));
      let (), body =
        in_block inner (fun inner ->
            emit inner (Loop_stmt.Mark Loop_mark.Reduction);
            let x = value inner r.Expr.Reduction.body in
            emit inner
              (Loop_stmt.Assign
                 (Loop_carrier.Float, acc, combine (temp_float acc) x)))
      in
      loop body;
      temp_float acc
  | Expr.Reduction.Argmax_index | Expr.Reduction.Argmax_value -> (
      let best = fresh_temp ctx and best_i = fresh_temp ctx in
      emit ctx
        (Loop_stmt.Assign
           (Loop_carrier.Float, best, Loop_expr.Const Float.neg_infinity));
      emit ctx (Loop_stmt.Assign_index (best_i, lo));
      let (), body =
        in_block inner (fun inner ->
            emit inner (Loop_stmt.Mark Loop_mark.Reduction);
            let x = assign_float inner (value inner r.Expr.Reduction.body) in
            emit inner
              (Loop_stmt.If
                 ( Loop_bool.Pool_better (temp_float best, temp_float x),
                   [
                     Loop_stmt.Assign (Loop_carrier.Float, best, temp_float x);
                     Loop_stmt.Assign_index (best_i, Loop_index.Var var);
                   ],
                   [] )))
      in
      loop body;
      match r.Expr.Reduction.kind with
      | Expr.Reduction.Argmax_value -> temp_float best
      | _ -> Loop_expr.Value_of_index (Loop_index.Temp best_i))

(* [Intrinsic.Max_pool] as the rows-then-columns double loop over the window
   [Intrinsic.window] defines, clipped to the input extents. The window and the
   flat index are the same arithmetic as the intrinsic's own, and the interval
   proof bounds both at the extreme output positions, since the window is
   monotone in the output position. Value and index advance together under the
   one predicate [pool_better]; an ordinary tie keeps the incumbent, and a NaN
   re-triggers, so the LAST NaN wins. *)
and max_pool ctx (d : Expr.Intrinsic.Max_pool.t) =
  let open Expr.Intrinsic.Max_pool in
  let output a = Loop_lower_index.checked ctx (Expr.Coord.get d.out a) in
  let out_h = output Expr.Axis.H in
  let out_w = output Expr.Axis.W in
  let window out ~stride ~pad ~kernel ~extent =
    let base =
      Loop_index.Add (Loop_index.Scale (stride, out), Loop_index.Const (-pad))
    in
    let lo =
      Loop_lower_index.guarded ctx (Loop_index.Max (Loop_index.Const 0, base))
    in
    let hi =
      Loop_lower_index.guarded ctx
        (Loop_index.Min
           ( Loop_index.Const extent,
             Loop_index.Add (base, Loop_index.Const kernel) ))
    in
    (lo, hi)
  in
  let hlo, hhi =
    window out_h
      ~stride:(d.stride.h :> int)
      ~pad:(d.pad.h :> int)
      ~kernel:(d.kernel.h :> int)
      ~extent:(d.input.h :> int)
  in
  let wlo, whi =
    window out_w
      ~stride:(d.stride.w :> int)
      ~pad:(d.pad.w :> int)
      ~kernel:(d.kernel.w :> int)
      ~extent:(d.input.w :> int)
  in
  let best = fresh_temp ctx and best_ix = fresh_temp ctx in
  emit ctx
    (Loop_stmt.Assign
       (Loop_carrier.Float, best, Loop_expr.Const Float.neg_infinity));
  emit ctx (Loop_stmt.Assign_index (best_ix, Loop_index.Const 0));
  let ih, ranges = bind_loop ctx ~lo:hlo ~hi:hhi in
  let iw, ranges = bind_loop { ctx with ranges } ~lo:wlo ~hi:whi in
  let inner = { ctx with ranges } in
  let (), columns =
    in_block inner (fun inner ->
        let coord inner b =
          Loop_lower_index.load_guard inner b
            (Expr.Coord.of_fn (fun a ->
                 if a = Expr.Axis.H then Loop_index.Var ih
                 else if a = Expr.Axis.W then Loop_index.Var iw
                 else Loop_lower_index.index inner (Expr.Coord.get d.out a)))
        in
        let read = source_load inner d.source ~coord in
        let x = assign_float inner read in
        let flat =
          Loop_index.Add
            ( Loop_index.Scale ((d.input.w :> int), Loop_index.Var ih),
              Loop_index.Var iw )
        in
        let (), update =
          in_block inner (fun inner ->
              let flat = Loop_lower_index.guarded inner flat in
              emit inner
                (Loop_stmt.Assign (Loop_carrier.Float, best, temp_float x));
              emit inner (Loop_stmt.Assign_index (best_ix, flat)))
        in
        emit inner
          (Loop_stmt.If
             (Loop_bool.Pool_better (temp_float best, temp_float x), update, [])))
  in
  emit ctx
    (Loop_stmt.For
       {
         var = ih;
         lo = hlo;
         hi = hhi;
         body =
           [ Loop_stmt.For { var = iw; lo = wlo; hi = whi; body = columns } ];
       });
  match d.result with
  | Value -> temp_float best
  | Index -> Loop_expr.Value_of_index (Loop_index.Temp best_ix)
