open Ssa_ir
open Loop_ir
module B = Ssa_builder

type error = [ `Unsupported of Ssa_of_loop_unsupported.t ]

let pp_error fmt : [< error ] -> unit = function
  | `Unsupported u -> Ssa_of_loop_unsupported.pp fmt u

module Int_map = Map.Make (Int)

(* What each Loop name denotes at a point of the walk. Temporaries are keyed by
   their id, loop variables by theirs; float and index temporaries share the id
   space of [Loop_temp], so one map holds both. *)
type env = { temps : Ssa_value.t Int_map.t; vars : Ssa_value.t Int_map.t }

type ctx = {
  esc : error Err.Escape.t;
  b : B.t;
  buffers : (Ssa_buffer.t * Loop_buffer.t) Tensor_id.Map.t;
}

let refuse ctx construct =
  Err.Escape.throw ctx.esc
    (`Unsupported { Ssa_of_loop_unsupported.construct } : error)

let temp_key t = Loop_temp.to_int t
let var_key v = Loop_var.to_int v

let lookup ctx map key =
  match Int_map.find_opt key map with
  | Some v -> v
  | None -> refuse ctx Ssa_of_loop_unsupported.Unassigned_temp

let literal ctx n =
  let n = Int64.of_int n in
  if Ssa_const.in_index_domain n then n
  else refuse ctx Ssa_of_loop_unsupported.Index_literal

let divisor ctx d =
  let d' = Int64.of_int d in
  if d > 0 && Ssa_const.in_index_domain d' then d'
  else refuse ctx Ssa_of_loop_unsupported.Index_divisor

let rec index ctx env : Loop_index.t -> Ssa_type.index B.value = function
  | Loop_index.Add (a, b) ->
      let a = index ctx env a in
      let b = index ctx env b in
      B.index_add ctx.b a b
  | Loop_index.Const n -> B.index ctx.b (literal ctx n)
  | Loop_index.Scale (k, a) ->
      let a = index ctx env a in
      B.index_scale ctx.b (literal ctx k) a
  | Loop_index.Temp t -> B.as_index (lookup ctx env.temps (temp_key t))
  | Loop_index.Var v -> B.as_index (lookup ctx env.vars (var_key v))
  | Loop_index.Ceil_div_pos (a, d) ->
      B.index_ceil_div ctx.b (divisor ctx d) (index ctx env a)
  | Loop_index.Clamp_low a -> B.index_clamp_low ctx.b (index ctx env a)
  | Loop_index.Floor_div_pos (a, d) ->
      B.index_floor_div ctx.b (divisor ctx d) (index ctx env a)
  | Loop_index.Max (a, b) ->
      let a = index ctx env a in
      let b = index ctx env b in
      B.index_max ctx.b a b
  | Loop_index.Min (a, b) ->
      let a = index ctx env a in
      let b = index ctx env b in
      B.index_min ctx.b a b

(* The axes in [Expr.Axis.all] order, as an explicit list: the evaluation order
   of a record's fields is not specified. *)
let coord ctx env (c : Loop_index.coord) =
  let components =
    List.map (fun a -> (a, index ctx env (Expr.Coord.get c a))) Expr.Axis.all
  in
  Expr.Coord.of_fn (fun a -> List.assoc a components)

let buffer ctx (b : Loop_buffer.t) =
  match Tensor_id.Map.find_opt b.Loop_buffer.id ctx.buffers with
  | Some (sb, _) -> sb
  | None ->
      let (Payload.Fmt f) = b.Loop_buffer.sg.Tensor_sig.fmt in
      refuse ctx (Ssa_of_loop_unsupported.Load_format (Payload.fmt_name f))

let rec expr ctx env : float Loop_expr.t -> Ssa_type.f64 B.value = function
  | Loop_expr.Binary (op, a, b) ->
      let a = expr ctx env a in
      let b = expr ctx env b in
      B.f64_binary ctx.b op a b
  | Loop_expr.Const x -> B.f64 ctx.b x
  | Loop_expr.Load (b, c) ->
      let sb = buffer ctx b in
      let decode = Ssa_op.Decode.to_float sb.Ssa_buffer.format in
      B.load_f64 ctx.b sb.Ssa_buffer.id ~decode (B.Coord (coord ctx env c))
  | Loop_expr.Load_flat (b, o) ->
      let sb = buffer ctx b in
      let decode = Ssa_op.Decode.to_float sb.Ssa_buffer.format in
      B.load_f64 ctx.b sb.Ssa_buffer.id ~decode (B.Flat (index ctx env o))
  | Loop_expr.Round_f32 a ->
      let a = expr ctx env a in
      B.f32_to_f64 ctx.b (B.f64_to_f32 ctx.b a)
  | Loop_expr.Temp (Loop_carrier.Float, t) ->
      B.as_f64 (lookup ctx env.temps (temp_key t))
  | Loop_expr.Value_of_index i -> B.index_to_f64 ctx.b (index ctx env i)
  | Loop_expr.Array_get _ -> refuse ctx Ssa_of_loop_unsupported.Array
  | Loop_expr.Float_max (a, b) ->
      let a = expr ctx env a in
      let b = expr ctx env b in
      B.f64_max ctx.b a b
  | Loop_expr.Fma _ -> refuse ctx Ssa_of_loop_unsupported.Fma
  | Loop_expr.I64_to_float a -> B.i64_to_f64 ctx.b (expr_i64 ctx env a)
  | Loop_expr.Select (p, a, b) ->
      (* a [Select] the Loop lowering kept needed no statement in either arm, so
         both arms are total and may be evaluated before the choice *)
      let p = pred ctx env p in
      let a = expr ctx env a in
      let b = expr ctx env b in
      B.select ctx.b p a b
  | Loop_expr.Unary (op, a) -> B.f64_unary ctx.b op (expr ctx env a)

(* An int64 expression is exact. A division and a float-to-int64 are the
   checked SSA operations: the Loop lowering writes their guards as statements of
   their own, which {!guard} converts to the same checks, so a program that
   passed its guards cannot fail again here. *)
and expr_i64 ctx env : int64 Loop_expr.t -> Ssa_type.i64 B.value = function
  | Loop_expr.Float_to_i64 a -> B.float_to_i64 ctx.b (expr ctx env a)
  | Loop_expr.I64_binary (op, a, b) -> (
      let a = expr_i64 ctx env a in
      let b = expr_i64 ctx env b in
      match op with
      | Expr.Value.I64_add -> B.i64_arith ctx.b Ssa_op.I64_op.Add a b
      | Expr.Value.I64_div -> B.i64_div ctx.b a b
      | Expr.Value.I64_mul -> B.i64_arith ctx.b Ssa_op.I64_op.Mul a b
      | Expr.Value.I64_sub -> B.i64_arith ctx.b Ssa_op.I64_op.Sub a b)
  | Loop_expr.I64_const n -> B.i64 ctx.b n
  | Loop_expr.I64_of_index i -> B.index_to_i64 ctx.b (index ctx env i)
  | Loop_expr.Load_i64 (b, c) ->
      let sb = buffer ctx b in
      B.load_i64 ctx.b sb.Ssa_buffer.id (B.Coord (coord ctx env c))
  | Loop_expr.Load_i64_flat (b, o) ->
      let sb = buffer ctx b in
      B.load_i64 ctx.b sb.Ssa_buffer.id (B.Flat (index ctx env o))
  | Loop_expr.Select (p, a, b) ->
      let p = pred ctx env p in
      let a = expr_i64 ctx env a in
      let b = expr_i64 ctx env b in
      B.select ctx.b p a b
  | Loop_expr.Temp (Loop_carrier.Int64, t) ->
      B.as_i64 (lookup ctx env.temps (temp_key t))

and pred ctx env : Loop_expr.pred -> Ssa_type.pred B.value = function
  | Loop_bool.Index_eq (a, b) ->
      let a = index ctx env a in
      let b = index ctx env b in
      B.index_compare ctx.b Ssa_op.Compare.Eq a b
  | Loop_bool.Index_lt (a, b) ->
      let a = index ctx env a in
      let b = index ctx env b in
      B.index_compare ctx.b Ssa_op.Compare.Lt a b
  | Loop_bool.Not p -> B.pred_not ctx.b (pred ctx env p)
  | Loop_bool.Or (p, q) ->
      let p = pred ctx env p in
      let q = pred ctx env q in
      B.pred_or ctx.b p q
  | Loop_bool.Out_of_range (i, extent) ->
      (* [i < 0 || i >= extent] *)
      let i = index ctx env i in
      let zero = B.index ctx.b 0L in
      let bound = B.index ctx.b (Int64.of_int extent) in
      let below = B.index_compare ctx.b Ssa_op.Compare.Lt i zero in
      let within = B.index_compare ctx.b Ssa_op.Compare.Lt i bound in
      B.pred_or ctx.b below (B.pred_not ctx.b within)
  | Loop_bool.Pool_better (best, value) ->
      let best = expr ctx env best in
      let value = expr ctx env value in
      B.pool_better ctx.b best value
  | Loop_bool.Value_eq (a, b) ->
      let a = expr ctx env a in
      let b = expr ctx env b in
      B.float_compare ctx.b Ssa_op.Compare.Eq a b
  | Loop_bool.Value_lt (a, b) ->
      let a = expr ctx env a in
      let b = expr ctx env b in
      B.float_compare ctx.b Ssa_op.Compare.Lt a b
  | Loop_bool.I64_eq (a, b) ->
      let a = expr_i64 ctx env a in
      let b = expr_i64 ctx env b in
      B.i64_compare ctx.b Ssa_op.Compare.Eq a b
  | Loop_bool.I64_lt (a, b) ->
      let a = expr_i64 ctx env a in
      let b = expr_i64 ctx env b in
      B.i64_compare ctx.b Ssa_op.Compare.Lt a b
  | Loop_bool.Index_overflows _ -> refuse ctx Ssa_of_loop_unsupported.Guard

(* Temporaries a statement list assigns, in id order, with nesting included. *)
let rec assigned acc : Loop_stmt.t list -> int list =
 fun stmts ->
  List.fold_left
    (fun acc -> function
      | Loop_stmt.Assign (_, t, _)
      | Loop_stmt.Assign_index (t, _)
      | Loop_stmt.Assign_index_of_i64 (t, _) ->
          temp_key t :: acc
      | Loop_stmt.For { body; _ } -> assigned acc body
      | Loop_stmt.If (_, yes, no) -> assigned (assigned acc yes) no
      | Loop_stmt.Reduce_sum { acc = a; body; _ } ->
          assigned (temp_key a :: acc) body
      | Loop_stmt.Alloc _ | Loop_stmt.Array_set _ | Loop_stmt.Charge_scan_update
      | Loop_stmt.Fail_if _ | Loop_stmt.Mark _ | Loop_stmt.Release_scan_state _
      | Loop_stmt.Reserve_scan_state _ | Loop_stmt.Reset_meter
      | Loop_stmt.Store _ | Loop_stmt.Store_flat _ ->
          acc)
    acc stmts

let assigned_set stmts = List.sort_uniq Int.compare (assigned [] stmts)

(* A store's value, evaluated before its coordinate as the reference does, and
   written through the encode its buffer takes. *)
let store ctx env sb access (value : Loop_stored.t) =
  match value with
  | Loop_stored.Bool e ->
      let x = expr ctx env e in
      B.store_f64 ctx.b sb.Ssa_buffer.id ~encode:Ssa_op.Encode.Bool_nonzero
        (access ()) x
  | Loop_stored.F32 e ->
      let x = expr ctx env e in
      B.store_f64 ctx.b sb.Ssa_buffer.id ~encode:Ssa_op.Encode.F32_round
        (access ()) x
  | Loop_stored.I64 e ->
      let x = expr_i64 ctx env e in
      B.store_i64 ctx.b sb.Ssa_buffer.id (access ()) x

(* The bounds a float must lie in to become an int64: not NaN, not an infinity,
   in [-2^63, 2^63). The upper bound is the exact power of two and exclusive,
   not [Int64.max_int]'s float, which rounds up to that same power. *)
let out_of_i64_range v =
  let two63 = Float.pow 2. 63. in
  Loop_bool.Or
    ( Loop_bool.Not (Loop_bool.Value_eq (v, v)),
      Loop_bool.Or
        ( Loop_bool.Value_lt (v, Loop_expr.Const (-.two63)),
          Loop_bool.Not (Loop_bool.Value_lt (v, Loop_expr.Const two63)) ) )

(* The leaves of a left-nested [Or] chain. *)
let rec disjuncts = function
  | Loop_bool.Or (p, q) -> disjuncts p @ disjuncts q
  | p -> [ p ]

(* A [Fail_if] the SSA form reproduces at the same site: the pair the Loop
   lowering writes for an index that may leave the domain, and for a load whose
   coordinate may leave its buffer. Anything else is refused, so a guard whose
   meaning this converter does not know is never silently dropped. *)
let guard ctx env cond (failure : Loop_failure.t) =
  match failure with
  | Loop_failure.Index_overflow { index = tree } -> (
      match cond with
      | Loop_bool.Index_overflows t when t = tree ->
          (* the checked operations of the tree, at this site *)
          ignore (index ctx env tree)
      | _ -> refuse ctx Ssa_of_loop_unsupported.Guard)
  | Loop_failure.Load_out_of_range { buffer = b; coord = c } ->
      let shape = b.Loop_buffer.sg.Tensor_sig.shape in
      let checks_the_coordinate = function
        | Loop_bool.Out_of_range (i, extent) ->
            List.exists
              (fun a ->
                Expr.Coord.get c a = i && Dim.to_int (Vec6.get shape a) = extent)
              Expr.Axis.all
        | _ -> false
      in
      if not (List.for_all checks_the_coordinate (disjuncts cond)) then
        refuse ctx Ssa_of_loop_unsupported.Guard;
      let sb = buffer ctx b in
      B.check_access ctx.b sb.Ssa_buffer.id (B.Coord (coord ctx env c))
  | Loop_failure.Gather_out_of_range { raw; extent } ->
      let bound n = Loop_expr.I64_const (Int64.of_int n) in
      let expected =
        Loop_bool.Or
          ( Loop_bool.I64_lt (raw, bound (-extent)),
            Loop_bool.Not (Loop_bool.I64_lt (raw, bound extent)) )
      in
      if Stdlib.compare cond expected <> 0 then
        refuse ctx Ssa_of_loop_unsupported.Guard;
      B.check_gather ctx.b (expr_i64 ctx env raw) ~extent:(Int64.of_int extent)
  | Loop_failure.I64_division_by_zero -> (
      match cond with
      | Loop_bool.I64_eq (b, Loop_expr.I64_const 0L) ->
          (* [1 / b] fails exactly when [b] is zero, and in no other way *)
          ignore (B.i64_div ctx.b (B.i64 ctx.b 1L) (expr_i64 ctx env b))
      | _ -> refuse ctx Ssa_of_loop_unsupported.Guard)
  | Loop_failure.I64_division_overflow -> (
      match cond with
      | Loop_bool.I64_eq (b, Loop_expr.I64_const -1L) ->
          (* [min_int / b] fails exactly when [b] is -1, and only under that
             condition is it evaluated *)
          let b = expr_i64 ctx env b in
          let minus_one = B.i64 ctx.b (-1L) in
          let hit = B.i64_compare ctx.b Ssa_op.Compare.Eq b minus_one in
          ignore
            (B.if_dyn ctx.b hit
               ~then_:(fun inner ->
                 ignore (B.i64_div inner (B.i64 inner Int64.min_int) b);
                 [])
               ~else_:(fun _ -> []))
      | _ -> refuse ctx Ssa_of_loop_unsupported.Guard)
  | Loop_failure.I64_from_float { value } ->
      if Stdlib.compare cond (out_of_i64_range value) <> 0 then
        refuse ctx Ssa_of_loop_unsupported.Guard;
      (* the conversion checks exactly NaN, an infinity and out of range *)
      ignore (B.float_to_i64 ctx.b (expr ctx env value))
  | Loop_failure.Local_out_of_range _ | Loop_failure.Scan_lane_out_of_range _
  | Loop_failure.Scan_row_out_of_range _ ->
      refuse ctx Ssa_of_loop_unsupported.Guard

let rec stmts ctx env = List.fold_left (stmt ctx) env

and stmt ctx env : Loop_stmt.t -> env = function
  | Loop_stmt.Assign (Loop_carrier.Float, t, e) ->
      let v = expr ctx env e in
      { env with temps = Int_map.add (temp_key t) (v :> Ssa_value.t) env.temps }
  | Loop_stmt.Assign (Loop_carrier.Int64, t, e) ->
      let v = expr_i64 ctx env e in
      { env with temps = Int_map.add (temp_key t) (v :> Ssa_value.t) env.temps }
  | Loop_stmt.Assign_index_of_i64 (t, e) ->
      let v = B.index_of_i64 ctx.b (expr_i64 ctx env e) in
      { env with temps = Int_map.add (temp_key t) (v :> Ssa_value.t) env.temps }
  | Loop_stmt.Assign_index (t, i) ->
      let v = index ctx env i in
      { env with temps = Int_map.add (temp_key t) (v :> Ssa_value.t) env.temps }
  | Loop_stmt.Fail_if (cond, failure) ->
      guard ctx env cond failure;
      env
  | Loop_stmt.For { var; lo; hi; body } ->
      let lo = index ctx env lo in
      let hi = index ctx env hi in
      let carried =
        List.filter (fun t -> Int_map.mem t env.temps) (assigned_set body)
      in
      let init = List.map (fun t -> Int_map.find t env.temps) carried in
      let results =
        B.for_dyn ctx.b ~lo ~hi ~init (fun b iv params ->
            let inner =
              {
                temps =
                  List.fold_left2
                    (fun m t p -> Int_map.add t p m)
                    env.temps carried params;
                vars = Int_map.add (var_key var) (iv :> Ssa_value.t) env.vars;
              }
            in
            let after = stmts { ctx with b } inner body in
            List.map (fun t -> Int_map.find t after.temps) carried)
      in
      {
        env with
        temps =
          List.fold_left2
            (fun m t r -> Int_map.add t r m)
            env.temps carried results;
      }
  | Loop_stmt.If (p, yes, no) ->
      let cond = pred ctx env p in
      let in_yes = assigned_set yes and in_no = assigned_set no in
      (* a temporary is merged when it was defined before the branch, or both
         arms define it; one defined in a single arm and not before is local *)
      let merged =
        List.filter
          (fun t ->
            Int_map.mem t env.temps || (List.mem t in_yes && List.mem t in_no))
          (List.sort_uniq Int.compare (in_yes @ in_no))
      in
      let branch arm b =
        let after = stmts { ctx with b } env arm in
        List.map (fun t -> lookup ctx after.temps t) merged
      in
      let results =
        B.if_dyn ctx.b cond ~then_:(branch yes) ~else_:(branch no)
      in
      {
        env with
        temps =
          List.fold_left2
            (fun m t r -> Int_map.add t r m)
            env.temps merged results;
      }
  | Loop_stmt.Mark m ->
      B.mark ctx.b
        (match m with
        | Loop_mark.Emitter -> Ssa_mark.Emitter
        | Loop_mark.Key -> Ssa_mark.Key
        | Loop_mark.Local -> Ssa_mark.Local
        | Loop_mark.Reduction -> Ssa_mark.Reduction
        | Loop_mark.Scan -> Ssa_mark.Scan
        | Loop_mark.Scan_update -> Ssa_mark.Scan_update);
      env
  | Loop_stmt.Reduce_sum { var; lo; hi; acc; seed; body; term; at = _ } ->
      let lo = index ctx env lo in
      let hi = index ctx env hi in
      let seed = B.f64 ctx.b seed in
      let sum =
        B.ordered_sum ctx.b ~lo ~hi ~seed (fun b iv ->
            B.mark b Ssa_mark.Reduction;
            let ctx = { ctx with b } in
            let inner =
              {
                env with
                vars = Int_map.add (var_key var) (iv :> Ssa_value.t) env.vars;
              }
            in
            let after = stmts ctx inner body in
            expr ctx after term)
      in
      {
        env with
        temps = Int_map.add (temp_key acc) (sum :> Ssa_value.t) env.temps;
      }
  | Loop_stmt.Store { buffer = b; coord = c; value } ->
      let sb = buffer ctx b in
      store ctx env sb (fun () -> B.Coord (coord ctx env c)) value;
      env
  | Loop_stmt.Store_flat { buffer = b; offset; value } ->
      let sb = buffer ctx b in
      store ctx env sb (fun () -> B.Flat (index ctx env offset)) value;
      env
  | Loop_stmt.Alloc _ | Loop_stmt.Array_set _ ->
      refuse ctx Ssa_of_loop_unsupported.Alloc
  | Loop_stmt.Charge_scan_update ->
      refuse ctx Ssa_of_loop_unsupported.Charge_scan
  | Loop_stmt.Release_scan_state _ | Loop_stmt.Reserve_scan_state _ ->
      refuse ctx Ssa_of_loop_unsupported.Scan_state
  | Loop_stmt.Reset_meter -> refuse ctx Ssa_of_loop_unsupported.Meter

let role : Loop_buffer.role -> Ssa_buffer.role = function
  | Loop_buffer.Input -> Ssa_buffer.Input
  | Loop_buffer.Output -> Ssa_buffer.Output
  | Loop_buffer.Scratch -> Ssa_buffer.Scratch

let convert (p : Loop_program.t) =
  Err.Escape.with_escape @@ fun esc ->
  let declared =
    List.filter_map
      (fun (lb : Loop_buffer.t) ->
        Option.map
          (fun sb -> (sb, lb))
          (Ssa_lower.Ssa_sig.buffer lb.Loop_buffer.sg (role lb.Loop_buffer.role)))
      p.Loop_program.buffers
  in
  let buffers =
    List.fold_left
      (fun m ((_, lb) as entry) -> Tensor_id.Map.add lb.Loop_buffer.id entry m)
      Tensor_id.Map.empty declared
  in
  Err.or_raise ~pp_error:Ssa_verify.pp_error
    (B.program ~buffers:(List.map fst declared) (fun b ->
         let ctx = { esc; b; buffers } in
         ignore
           (stmts ctx
              { temps = Int_map.empty; vars = Int_map.empty }
              p.Loop_program.body)))
