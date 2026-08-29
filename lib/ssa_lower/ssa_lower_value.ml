open Ssa_ir
open Ssa_lower_ctx
module B = Ssa_builder

(* Lowering an [Expr.Value.t] to SSA values, in evaluation order: every check,
   load and reduction is built before the operation that consumes it. *)

(* Only the selected arm may load, fail or loop: an [if], never an eager
   [select], until a pass has proved an arm total. One definition for both
   carriers. *)
let lazy_select (ctx : Ssa_lower_ctx.t) p (yes : Ssa_lower_ctx.t -> 'a B.value)
    (no : Ssa_lower_ctx.t -> 'a B.value) : 'a B.value =
  let (B.Cons (x, B.Nil)) =
    B.if_ ctx.b p
      ~then_:(fun b -> B.Cons (yes { ctx with b }, B.Nil))
      ~else_:(fun b -> B.Cons (no { ctx with b }, B.Nil))
  in
  x

let rec value ctx : float Expr.Value.t -> Ssa_type.f64 B.value = function
  | Expr.Value.Binary (op, a, b) ->
      let a = value ctx a in
      let b = value ctx b in
      B.f64_binary ctx.b op a b
  | Expr.Value.Const x -> B.f64 ctx.b x
  | Expr.Value.I64_to_float a -> B.i64_to_f64 ctx.b (value_i64 ctx a)
  | Expr.Value.Intrinsic (Expr.Intrinsic.Max_pool d) -> max_pool ctx d
  | Expr.Value.Load (src, coord) ->
      load ctx src (fun ctx -> Ssa_lower_index.coord ctx coord)
  | Expr.Value.Local v -> local ctx v
  | Expr.Value.Local_at (v, i) -> local_at ctx v i
  | Expr.Value.Local_scan_at (v, row, lane) -> scan_cached ctx v row lane
  | Expr.Value.Scan_at (s, row, lane) -> scan_inline ctx s row lane
  | Expr.Value.Reduce r -> reduce ctx r
  | Expr.Value.Round_f32 a ->
      let a = value ctx a in
      B.f32_to_f64 ctx.b (B.f64_to_f32 ctx.b a)
  | Expr.Value.Select (p, yes, no) ->
      let p = pred ctx p in
      lazy_select ctx p (fun ctx -> value ctx yes) (fun ctx -> value ctx no)
  | Expr.Value.Unary (op, a) -> B.f64_unary ctx.b op (value ctx a)
  | Expr.Value.Value_of_index i ->
      B.index_to_f64 ctx.b (Ssa_lower_index.index ctx i)

(* An exact int64 body: modular [+ - *], checked [/], a checked float-to-int64,
   and the ordered reductions. A failing operation is the checked operation
   itself, at the point the source evaluates it. *)
and value_i64 ctx : int64 Expr.Value.t -> Ssa_type.i64 B.value = function
  | Expr.Value.Float_to_i64 a -> B.float_to_i64 ctx.b (value ctx a)
  | Expr.Value.I64_binary (op, a, b) -> (
      let a = value_i64 ctx a in
      let b = value_i64 ctx b in
      match op with
      | Expr.Value.I64_add -> B.i64_arith ctx.b Ssa_op.I64_op.Add a b
      | Expr.Value.I64_div -> B.i64_div ctx.b a b
      | Expr.Value.I64_mul -> B.i64_arith ctx.b Ssa_op.I64_op.Mul a b
      | Expr.Value.I64_sub -> B.i64_arith ctx.b Ssa_op.I64_op.Sub a b)
  | Expr.Value.I64_const n -> B.i64 ctx.b n
  | Expr.Value.I64_load (src, coord) -> load_i64 ctx src coord
  | Expr.Value.I64_local _ | Expr.Value.I64_local_at _ ->
      refuse ctx Ssa_unsupported.Local_read
  | Expr.Value.I64_of_index i ->
      B.index_to_i64 ctx.b (Ssa_lower_index.index ctx i)
  | Expr.Value.I64_sum r -> reduce_i64 ctx r
  | Expr.Value.Select (p, yes, no) ->
      let p = pred ctx p in
      lazy_select ctx p
        (fun ctx -> value_i64 ctx yes)
        (fun ctx -> value_i64 ctx no)

and pred ctx : Expr.Bool.t -> Ssa_type.pred B.value = function
  | Expr.Bool.I64_eq (a, b) ->
      let a = value_i64 ctx a in
      let b = value_i64 ctx b in
      B.i64_compare ctx.b Ssa_op.Compare.Eq a b
  | Expr.Bool.I64_lt (a, b) ->
      let a = value_i64 ctx a in
      let b = value_i64 ctx b in
      B.i64_compare ctx.b Ssa_op.Compare.Lt a b
  | Expr.Bool.Index_eq (a, b) ->
      let a = Ssa_lower_index.index ctx a in
      let b = Ssa_lower_index.index ctx b in
      B.index_compare ctx.b Ssa_op.Compare.Eq a b
  | Expr.Bool.Value_eq (a, b) ->
      let a = value ctx a in
      let b = value ctx b in
      B.float_compare ctx.b Ssa_op.Compare.Eq a b
  | Expr.Bool.Value_lt (a, b) ->
      let a = value ctx a in
      let b = value ctx b in
      B.float_compare ctx.b Ssa_op.Compare.Lt a b

(* A scalar local's single slot. [Region_program.check] proved the read matches
   the declared shape and comes after the write, so neither is re-checked. *)
and local ctx v =
  match local_of ctx v with
  | Slots { handle; _ } -> B.local_read ctx.b handle (B.index ctx.b 0L)
  | Prev_row _ -> invalid_arg "Ssa_lower: prev is read only through local_at"

(* A vector local, or a scan's [prev], at a computed position. The read is
   checked against the variable's declared extent and fails as the reference's
   unbound local does: the reader answers [None] outside the range. *)
and local_at ctx v i =
  let handle, base, extent =
    match local_of ctx v with
    | Slots { handle; count; _ } -> (handle, None, count)
    | Prev_row { handle; base; width } -> (handle, Some base, width)
  in
  let i = Ssa_lower_index.index ctx i in
  B.check_local ctx.b ~var:v ~extent:(Int64.of_int extent) i;
  let at =
    match base with None -> i | Some base -> B.index_add ctx.b base i
  in
  B.local_read ctx.b handle at

and local_of ctx v =
  match Expr.Local_var.Map.find_opt v ctx.locals with
  | Some l -> l
  | None -> invalid_arg "Ssa_lower: local is not written yet"

(* A cached read of a trace local: row-major, [row * width + lane]. The row is
   checked first, then the lane, before either is used. *)
and scan_cached ctx v row lane =
  match local_of ctx v with
  | Slots { handle; shape = Region_local.Shape.Scan { width; steps }; _ } ->
      let width = (width :> int) and steps = (steps :> int) in
      let row = Ssa_lower_index.index ctx row in
      let lane = Ssa_lower_index.index ctx lane in
      B.check_scan ctx.b ~var:(Some v) ~row ~lane
        ~row_extent:(Int64.of_int (steps + 1))
        ~lane_extent:(Int64.of_int width);
      let at =
        B.index_add ctx.b (B.index_scale ctx.b (Int64.of_int width) row) lane
      in
      B.local_read ctx.b handle at
  | Slots _ | Prev_row _ ->
      invalid_arg "Ssa_lower: a cached scan read of a local that is no trace"

(* An inline [Scan_at]: the recurrence re-executed, with no sharing across
   evaluations, on two rolling rows of [width] cells. The bounds are checked
   first, then the live state is reserved, then the initial row is filled, then
   [row] update rows each charged once per lane BEFORE its body runs. The next
   row goes to a second object and is copied back, so [prev] reads the previous
   row throughout the update, as the reference's [Array.init] over a fixed
   [prev_row] does. *)
and scan_inline ctx (s : Expr.Scan.t) row lane =
  let width = s.Expr.Scan.width and steps = s.Expr.Scan.steps in
  ctx.meter := true;
  let row = Ssa_lower_index.index ctx row in
  let lane = Ssa_lower_index.index ctx lane in
  B.check_scan ctx.b ~var:None ~row ~lane
    ~row_extent:(Int64.of_int (steps + 1))
    ~lane_extent:(Int64.of_int width);
  B.meter_reserve ctx.b ~width:(Int64.of_int width);
  let prev = B.local_alloc ctx.b ~slots:(Int64.of_int width) in
  let cur = B.local_alloc ctx.b ~slots:(Int64.of_int width) in
  let zero b = B.index b 0L in
  let each_lane b ~bind body =
    let B.Nil =
      B.for_ b ~lo:(zero b)
        ~hi:(B.index b (Int64.of_int width))
        ~init:B.Nil
        (fun b l B.Nil ->
          body { ctx with b; reducers = bind l ctx.reducers } l;
          B.Nil)
    in
    ()
  in
  each_lane ctx.b ~bind:(Expr.Reduce_var.Map.add s.Expr.Scan.lane)
    (fun inner l -> B.local_write inner.b prev l (value inner s.Expr.Scan.init));
  let B.Nil =
    B.for_ ctx.b ~lo:(zero ctx.b) ~hi:row ~init:B.Nil (fun b step B.Nil ->
        let stepper =
          {
            ctx with
            b;
            reducers =
              Expr.Reduce_var.Map.add s.Expr.Scan.step step ctx.reducers;
            locals =
              Expr.Local_var.Map.add s.Expr.Scan.prev
                (Prev_row { handle = prev; base = zero b; width })
                ctx.locals;
          }
        in
        each_lane b
          ~bind:(fun l -> Expr.Reduce_var.Map.add s.Expr.Scan.lane l)
          (fun inner l ->
            ignore stepper;
            B.meter_charge inner.b;
            let inner =
              {
                inner with
                reducers =
                  Expr.Reduce_var.Map.add s.Expr.Scan.step step inner.reducers;
                locals = stepper.locals;
              }
            in
            B.local_write inner.b cur l (value inner s.Expr.Scan.update));
        each_lane b
          ~bind:(fun _ reducers -> reducers)
          (fun inner l ->
            B.local_write inner.b prev l (B.local_read inner.b cur l));
        B.Nil)
  in
  let result = B.local_read ctx.b prev lane in
  B.meter_release ctx.b ~width:(Int64.of_int width);
  result

(* A source read as a float, at a coordinate lowered by [coord] once the source
   is known. A [Filled] input still bounds-checks: only the read is folded, and
   to what a materialized fill would decode to, not to the value it was given. *)
and load ctx src coord_of =
  match Tensor_id.Map.find_opt (Expr_bridge.id_of_source src) ctx.sources with
  | None -> refuse ctx Ssa_unsupported.Unmaterialized_source
  | Some (Unsupported_format name) ->
      refuse ctx (Ssa_unsupported.Load_format name)
  | Some (Buffer b) ->
      let decode = Ssa_op.Decode.to_float b.Ssa_buffer.format in
      let coord = coord_of ctx in
      B.load_f64 ctx.b b.Ssa_buffer.id ~decode (B.Coord coord)
  | Some (Fill (b, v)) ->
      B.check_access ctx.b b.Ssa_buffer.id (B.Coord (coord_of ctx));
      B.f64 ctx.b
        (match b.Ssa_buffer.format with
        | Ssa_format.Bool -> if v <> 0. then 1. else 0.
        | _ -> Ssa_const.round_f32 v)
  | Some (Fill_i64 (b, v)) ->
      B.check_access ctx.b b.Ssa_buffer.id (B.Coord (coord_of ctx));
      B.f64 ctx.b (Int64.to_float v)

and load_i64 ctx src coord =
  match Tensor_id.Map.find_opt (Expr_bridge.id_of_source src) ctx.sources with
  | None -> refuse ctx Ssa_unsupported.Unmaterialized_source
  | Some (Unsupported_format name) ->
      refuse ctx (Ssa_unsupported.Load_format name)
  | Some (Buffer b)
    when Ssa_format.family b.Ssa_buffer.format = Ssa_format.Family.I64 ->
      B.load_i64 ctx.b b.Ssa_buffer.id
        (B.Coord (Ssa_lower_index.coord ctx coord))
  | Some (Fill_i64 (b, v)) ->
      B.check_access ctx.b b.Ssa_buffer.id
        (B.Coord (Ssa_lower_index.coord ctx coord));
      B.i64 ctx.b v
  | Some (Buffer b | Fill (b, _)) ->
      refuse ctx
        (Ssa_unsupported.Load_format (Ssa_format.name b.Ssa_buffer.format))

(* The ordered half-open folds [Expr.Eval] specifies, with their seeds: [Sum]
   from [+0.] (never the first element, which would make an all-[-0.] sum
   [-0.]), [Max] from [-inf], and the argmaxes advancing value and index
   together under the one predicate [pool_better]. One reduction mark per term,
   before the term is evaluated. *)
and reduce ctx (r : Expr.Reduction.t) =
  let lo = Ssa_lower_index.index ctx r.Expr.Reduction.lo in
  let hi = Ssa_lower_index.index ctx r.Expr.Reduction.hi in
  let term b i =
    B.mark b Ssa_mark.Reduction;
    let ctx =
      {
        ctx with
        b;
        reducers = Expr.Reduce_var.Map.add r.Expr.Reduction.var i ctx.reducers;
      }
    in
    value ctx r.Expr.Reduction.body
  in
  match r.Expr.Reduction.kind with
  | Expr.Reduction.Sum ->
      let seed = B.f64 ctx.b 0. in
      B.ordered_sum ctx.b ~lo ~hi ~seed term
  | Expr.Reduction.Max ->
      let seed = B.f64 ctx.b Float.neg_infinity in
      let (B.Cons (best, B.Nil)) =
        B.for_ ctx.b ~lo ~hi
          ~init:(B.Cons (seed, B.Nil))
          (fun b i (B.Cons (best, B.Nil)) ->
            B.Cons (B.f64_max b best (term b i), B.Nil))
      in
      best
  | (Expr.Reduction.Argmax_index | Expr.Reduction.Argmax_value) as kind ->
      let seed = B.f64 ctx.b Float.neg_infinity in
      let (B.Cons (best, B.Cons (best_i, B.Nil))) =
        B.for_ ctx.b ~lo ~hi
          ~init:(B.Cons (seed, B.Cons (lo, B.Nil)))
          (fun b i (B.Cons (best, B.Cons (best_i, B.Nil))) ->
            let x = term b i in
            let wins = B.pool_better b best x in
            B.Cons
              (B.select b wins x best, B.Cons (B.select b wins i best_i, B.Nil)))
      in
      if kind = Expr.Reduction.Argmax_value then best
      else B.index_to_f64 ctx.b best_i

(* The int64 reductions, with their own seeds: [Sum] is modular from [0L], the
   maxima keep the signed maximum from [Int64.min_int], and [Argmax_index]
   reports the position of the FIRST maximum (a later value must be strictly
   greater), [lo] for an empty range. There is no NaN to consider. *)
and reduce_i64 ctx (r : Expr.Reduction.i64) =
  let lo = Ssa_lower_index.index ctx r.Expr.Reduction.i64_lo in
  let hi = Ssa_lower_index.index ctx r.Expr.Reduction.i64_hi in
  let term b i =
    B.mark b Ssa_mark.Reduction;
    let ctx =
      {
        ctx with
        b;
        reducers =
          Expr.Reduce_var.Map.add r.Expr.Reduction.i64_var i ctx.reducers;
      }
    in
    value_i64 ctx r.Expr.Reduction.i64_body
  in
  match r.Expr.Reduction.i64_kind with
  | Expr.Reduction.Sum ->
      let seed = B.i64 ctx.b 0L in
      let (B.Cons (sum, B.Nil)) =
        B.for_ ctx.b ~lo ~hi
          ~init:(B.Cons (seed, B.Nil))
          (fun b i (B.Cons (acc, B.Nil)) ->
            B.Cons (B.i64_arith b Ssa_op.I64_op.Add acc (term b i), B.Nil))
      in
      sum
  | Expr.Reduction.Max | Expr.Reduction.Argmax_value ->
      let seed = B.i64 ctx.b Int64.min_int in
      let (B.Cons (best, B.Nil)) =
        B.for_ ctx.b ~lo ~hi
          ~init:(B.Cons (seed, B.Nil))
          (fun b i (B.Cons (best, B.Nil)) ->
            let x = term b i in
            let wins = B.i64_compare b Ssa_op.Compare.Lt best x in
            B.Cons (B.select b wins x best, B.Nil))
      in
      best
  | Expr.Reduction.Argmax_index ->
      let seed = B.i64 ctx.b Int64.min_int in
      let (B.Cons (_, B.Cons (best_i, B.Nil))) =
        B.for_ ctx.b ~lo ~hi
          ~init:(B.Cons (seed, B.Cons (lo, B.Nil)))
          (fun b i (B.Cons (best, B.Cons (best_i, B.Nil))) ->
            let x = term b i in
            let wins = B.i64_compare b Ssa_op.Compare.Lt best x in
            B.Cons
              (B.select b wins x best, B.Cons (B.select b wins i best_i, B.Nil)))
      in
      B.index_to_i64 ctx.b best_i

(* [Intrinsic.Max_pool] as the rows-then-columns double loop over the window the
   intrinsic defines, clipped to the input extents. Value and index advance
   together under the one predicate [pool_better]: an ordinary tie keeps the
   incumbent, and a NaN re-triggers, so the LAST NaN wins. The window and the
   flat index are the intrinsic's own arithmetic, every [Add] and [Scale] of it
   checked. *)
and max_pool ctx (d : Expr.Intrinsic.Max_pool.t) =
  let open Expr.Intrinsic.Max_pool in
  let output a = Ssa_lower_index.index ctx (Expr.Coord.get d.out a) in
  let out_h = output Expr.Axis.H in
  let out_w = output Expr.Axis.W in
  let window out ~stride ~pad ~kernel ~extent =
    let scaled = B.index_scale ctx.b (Int64.of_int stride) out in
    let base = B.index_add ctx.b scaled (B.index ctx.b (Int64.of_int (-pad))) in
    let lo = B.index_max ctx.b (B.index ctx.b 0L) base in
    let hi =
      B.index_min ctx.b
        (B.index ctx.b (Int64.of_int extent))
        (B.index_add ctx.b base (B.index ctx.b (Int64.of_int kernel)))
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
  let seed = B.f64 ctx.b Float.neg_infinity in
  let first = B.index ctx.b 0L in
  let (B.Cons (best, B.Cons (best_ix, B.Nil))) =
    B.for_ ctx.b ~lo:hlo ~hi:hhi
      ~init:(B.Cons (seed, B.Cons (first, B.Nil)))
      (fun b ih (B.Cons (best, B.Cons (best_ix, B.Nil))) ->
        B.for_ b ~lo:wlo ~hi:whi
          ~init:(B.Cons (best, B.Cons (best_ix, B.Nil)))
          (fun b iw (B.Cons (best, B.Cons (best_ix, B.Nil))) ->
            let inner = { ctx with b } in
            let coord inner =
              let components =
                List.map
                  (fun a ->
                    ( a,
                      if a = Expr.Axis.H then ih
                      else if a = Expr.Axis.W then iw
                      else Ssa_lower_index.index inner (Expr.Coord.get d.out a)
                    ))
                  Expr.Axis.all
              in
              Expr.Coord.of_fn (fun a -> List.assoc a components)
            in
            let x = load inner d.source coord in
            let flat =
              B.index_add b
                (B.index_scale b (Int64.of_int (d.input.w :> int)) ih)
                iw
            in
            let wins = B.pool_better b best x in
            B.Cons
              ( B.select b wins x best,
                B.Cons (B.select b wins flat best_ix, B.Nil) )))
  in
  match d.result with Value -> best | Index -> B.index_to_f64 ctx.b best_ix
