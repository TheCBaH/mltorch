module V = Loop_vector

let rec expr : type a. a Loop_expr.t -> a Loop_expr.t = function
  | Loop_expr.Binary (Expr.Value.Add, a, Loop_expr.Binary (Expr.Value.Mul, x, y))
    ->
      Loop_expr.Fma (expr x, expr y, expr a)
  | Loop_expr.Binary (Expr.Value.Add, Loop_expr.Binary (Expr.Value.Mul, x, y), a)
    ->
      Loop_expr.Fma (expr x, expr y, expr a)
  | Loop_expr.Binary (op, a, b) -> Loop_expr.Binary (op, expr a, expr b)
  | Loop_expr.Float_max (a, b) -> Loop_expr.Float_max (expr a, expr b)
  | Loop_expr.Float_to_i64 a -> Loop_expr.Float_to_i64 (expr a)
  | Loop_expr.Fma (a, b, c) -> Loop_expr.Fma (expr a, expr b, expr c)
  | Loop_expr.I64_binary (op, a, b) -> Loop_expr.I64_binary (op, expr a, expr b)
  | Loop_expr.I64_to_float a -> Loop_expr.I64_to_float (expr a)
  | Loop_expr.Round_f32 a -> Loop_expr.Round_f32 (expr a)
  | Loop_expr.Select (p, a, b) -> Loop_expr.Select (pred p, expr a, expr b)
  | Loop_expr.Unary (op, a) -> Loop_expr.Unary (op, expr a)
  | Loop_expr.Array_get _ as e -> e
  | Loop_expr.Const _ as e -> e
  | Loop_expr.I64_const _ as e -> e
  | Loop_expr.I64_of_index _ as e -> e
  | Loop_expr.Load _ as e -> e
  | Loop_expr.Load_flat _ as e -> e
  | Loop_expr.Load_i64 _ as e -> e
  | Loop_expr.Load_i64_flat _ as e -> e
  | Loop_expr.Temp _ as e -> e
  | Loop_expr.Value_of_index _ as e -> e

and pred : Loop_expr.pred -> Loop_expr.pred = function
  | Loop_bool.I64_eq (a, b) -> Loop_bool.I64_eq (expr a, expr b)
  | Loop_bool.I64_lt (a, b) -> Loop_bool.I64_lt (expr a, expr b)
  | Loop_bool.Not p -> Loop_bool.Not (pred p)
  | Loop_bool.Or (p, q) -> Loop_bool.Or (pred p, pred q)
  | Loop_bool.Pool_better (a, b) -> Loop_bool.Pool_better (expr a, expr b)
  | Loop_bool.Value_eq (a, b) -> Loop_bool.Value_eq (expr a, expr b)
  | Loop_bool.Value_lt (a, b) -> Loop_bool.Value_lt (expr a, expr b)
  | ( Loop_bool.Index_eq _ | Loop_bool.Index_lt _ | Loop_bool.Index_overflows _
    | Loop_bool.Out_of_range _ ) as p ->
      p

let stored : Loop_stored.t -> Loop_stored.t = function
  | Loop_stored.Bool e -> Loop_stored.Bool (expr e)
  | Loop_stored.F32 e -> Loop_stored.F32 (expr e)
  | Loop_stored.I64 e -> Loop_stored.I64 (expr e)

let rec stmt (s : Loop_stmt.t) : Loop_stmt.t =
  match s with
  | Loop_stmt.Array_set (a, i, e) -> Loop_stmt.Array_set (a, i, expr e)
  | Loop_stmt.Assign (c, t, e) -> Loop_stmt.Assign (c, t, expr e)
  | Loop_stmt.Assign_index_of_i64 (t, e) ->
      Loop_stmt.Assign_index_of_i64 (t, expr e)
  | Loop_stmt.Fail_if (p, f) -> Loop_stmt.Fail_if (pred p, f)
  | Loop_stmt.For f -> Loop_stmt.For { f with body = List.map stmt f.body }
  | Loop_stmt.If (p, yes, no) ->
      Loop_stmt.If (pred p, List.map stmt yes, List.map stmt no)
  | Loop_stmt.Reduce_sum r ->
      Loop_stmt.Reduce_sum
        { r with body = List.map stmt r.body; term = expr r.term }
  | Loop_stmt.Store r -> Loop_stmt.Store { r with value = stored r.value }
  | Loop_stmt.Store_flat r ->
      Loop_stmt.Store_flat { r with value = stored r.value }
  | ( Loop_stmt.Alloc _ | Loop_stmt.Assign_index _
    | Loop_stmt.Charge_scan_update | Loop_stmt.Mark _
    | Loop_stmt.Release_scan_state _ | Loop_stmt.Reserve_scan_state _
    | Loop_stmt.Reset_meter ) as s ->
      s

let rec vexpr (e : V.t) : V.t =
  match e with
  | V.Binary (Expr.Value.Add, a, V.Binary (Expr.Value.Mul, x, y))
  | V.Binary (Expr.Value.Add, V.Binary (Expr.Value.Mul, x, y), a) ->
      V.Fma (vexpr x, vexpr y, vexpr a)
  | V.Binary (op, a, b) -> V.Binary (op, vexpr a, vexpr b)
  | V.Float_max (a, b) -> V.Float_max (vexpr a, vexpr b)
  | V.Fma (a, b, c) -> V.Fma (vexpr a, vexpr b, vexpr c)
  | V.Round_f32 a -> V.Round_f32 (vexpr a)
  | V.Select (m, a, b) -> V.Select (vmask m, vexpr a, vexpr b)
  | V.Unary (op, a) -> V.Unary (op, vexpr a)
  | V.Const _ | V.Index_value _ | V.Load _ | V.Splat _ | V.Temp _ -> e

and vmask (m : V.mask) : V.mask =
  match m with
  | V.Not m -> V.Not (vmask m)
  | V.Or (a, b) -> V.Or (vmask a, vmask b)
  | V.Pool_better (a, b) -> V.Pool_better (vexpr a, vexpr b)
  | V.Value_eq (a, b) -> V.Value_eq (vexpr a, vexpr b)
  | V.Value_lt (a, b) -> V.Value_lt (vexpr a, vexpr b)

let vstored : V.stored -> V.stored = function
  | V.Bool e -> V.Bool (vexpr e)
  | V.F32 e -> V.F32 (vexpr e)

let rec vstmt (s : V.stmt) : V.stmt =
  match s with
  | V.Assign (t, e) -> V.Assign (t, vexpr e)
  | V.Inner r -> V.Inner { r with body = List.map vstmt r.body }
  | V.Store r -> V.Store { r with value = vstored r.value }
  | (V.Index_assign _ | V.Mark _) as s -> s

let reduction ~fuse_reductions (r : V.Reduction.t) : V.Reduction.t =
  let term = vexpr r.V.Reduction.term in
  {
    r with
    V.Reduction.term;
    fused =
      (fuse_reductions
      && match term with V.Binary (Expr.Value.Mul, _, _) -> true | _ -> false);
  }

let rec node ~fuse_reductions ~scalar (n : V.node) : V.node =
  let again = node ~fuse_reductions ~scalar in
  match n with
  | V.If (p, yes, no) ->
      V.If
        ((if scalar then pred p else p), List.map again yes, List.map again no)
  | V.Loop r -> V.Loop { r with body = List.map again r.body }
  | V.Reduction r -> V.Reduction (reduction ~fuse_reductions r)
  | V.Scalar s -> V.Scalar (if scalar then stmt s else s)
  | V.Vector l ->
      V.Vector
        {
          l with
          V.body = List.map vstmt l.V.body;
          scalar = (if scalar then stmt l.V.scalar else l.V.scalar);
        }

let program ?(fuse_reductions = false) ?(scalar = true) (p : V.program) :
    V.program =
  { p with V.body = List.map (node ~fuse_reductions ~scalar) p.V.body }
