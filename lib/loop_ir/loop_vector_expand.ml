module V = Loop_vector

(* The ids already used by the program, so the expansion mints past them. *)
type ids = { mutable temp : int; mutable var : int }

let rec scan_index ids : Loop_index.t -> unit = function
  | Loop_index.Var v -> ids.var <- max ids.var (Loop_var.to_int v)
  | Loop_index.Temp t -> ids.temp <- max ids.temp (Loop_temp.to_int t)
  | Loop_index.Add (a, b) | Loop_index.Max (a, b) | Loop_index.Min (a, b) ->
      scan_index ids a;
      scan_index ids b
  | Loop_index.Ceil_div_pos (a, _)
  | Loop_index.Clamp_low a
  | Loop_index.Floor_div_pos (a, _)
  | Loop_index.Scale (_, a) ->
      scan_index ids a
  | Loop_index.Const _ -> ()

let scan_coord ids (c : Loop_index.coord) =
  List.iter (fun a -> scan_index ids (Expr.Coord.get c a)) Expr.Axis.all

let rec scan_expr : type a. ids -> a Loop_expr.t -> unit =
 fun ids e ->
  match e with
  | Loop_expr.Array_get (_, i) -> scan_index ids i
  | Loop_expr.Binary (_, a, b) | Loop_expr.Float_max (a, b) ->
      scan_expr ids a;
      scan_expr ids b
  | Loop_expr.Const _ | Loop_expr.I64_const _ -> ()
  | Loop_expr.Float_to_i64 a -> scan_expr ids a
  | Loop_expr.I64_to_float a -> scan_expr ids a
  | Loop_expr.Round_f32 a -> scan_expr ids a
  | Loop_expr.Unary (_, a) -> scan_expr ids a
  | Loop_expr.I64_binary (_, a, b) ->
      scan_expr ids a;
      scan_expr ids b
  | Loop_expr.I64_of_index i | Loop_expr.Value_of_index i -> scan_index ids i
  | Loop_expr.Load (_, c) | Loop_expr.Load_i64 (_, c) -> scan_coord ids c
  | Loop_expr.Load_flat (_, i) | Loop_expr.Load_i64_flat (_, i) ->
      scan_index ids i
  | Loop_expr.Select (p, a, b) ->
      scan_pred ids p;
      scan_expr ids a;
      scan_expr ids b
  | Loop_expr.Temp (_, t) -> ids.temp <- max ids.temp (Loop_temp.to_int t)

and scan_pred ids (p : Loop_expr.pred) =
  match p with
  | Loop_bool.I64_eq (a, b) | Loop_bool.I64_lt (a, b) ->
      scan_expr ids a;
      scan_expr ids b
  | Loop_bool.Index_eq (a, b) | Loop_bool.Index_lt (a, b) ->
      scan_index ids a;
      scan_index ids b
  | Loop_bool.Index_overflows i | Loop_bool.Out_of_range (i, _) ->
      scan_index ids i
  | Loop_bool.Not p -> scan_pred ids p
  | Loop_bool.Or (p, q) ->
      scan_pred ids p;
      scan_pred ids q
  | Loop_bool.Pool_better (a, b)
  | Loop_bool.Value_eq (a, b)
  | Loop_bool.Value_lt (a, b) ->
      scan_expr ids a;
      scan_expr ids b

let rec scan_stmt ids (s : Loop_stmt.t) =
  match s with
  | Loop_stmt.Alloc _ | Loop_stmt.Charge_scan_update | Loop_stmt.Mark _
  | Loop_stmt.Release_scan_state _ | Loop_stmt.Reserve_scan_state _
  | Loop_stmt.Reset_meter ->
      ()
  | Loop_stmt.Array_set (_, i, e) ->
      scan_index ids i;
      scan_expr ids e
  | Loop_stmt.Assign (_, t, e) ->
      ids.temp <- max ids.temp (Loop_temp.to_int t);
      scan_expr ids e
  | Loop_stmt.Assign_index (t, i) ->
      ids.temp <- max ids.temp (Loop_temp.to_int t);
      scan_index ids i
  | Loop_stmt.Assign_index_of_i64 (t, e) ->
      ids.temp <- max ids.temp (Loop_temp.to_int t);
      scan_expr ids e
  | Loop_stmt.Fail_if (p, _) -> scan_pred ids p
  | Loop_stmt.For { var; lo; hi; body } ->
      ids.var <- max ids.var (Loop_var.to_int var);
      scan_index ids lo;
      scan_index ids hi;
      List.iter (scan_stmt ids) body
  | Loop_stmt.If (p, a, b) ->
      scan_pred ids p;
      List.iter (scan_stmt ids) a;
      List.iter (scan_stmt ids) b
  | Loop_stmt.Store { coord; value; _ } ->
      scan_coord ids coord;
      scan_stored ids value
  | Loop_stmt.Store_flat { offset; value; _ } ->
      scan_index ids offset;
      scan_stored ids value

and scan_stored ids (v : Loop_stored.t) =
  match v with
  | Loop_stored.Bool e | Loop_stored.F32 e -> scan_expr ids e
  | Loop_stored.I64 e -> scan_expr ids e

(* [subst var by index]: every [Var var] replaced by [by]. *)
let subst var by idx =
  Loop_index_map.index
    ~f:(function Loop_index.Var v when Loop_var.equal v var -> by | i -> i)
    idx

let add_const idx k =
  if k = 0 then idx else Loop_index.Add (idx, Loop_index.Const k)

let expand (p : V.program) : Loop_program.t =
  let ids = { temp = -1; var = -1 } in
  List.iter (scan_stmt ids) p.V.scalar.Loop_program.body;
  let next_temp = ref (ids.temp + 1) and next_var = ref (ids.var + 1) in
  let fresh_temp () =
    let t = Loop_temp.of_int !next_temp in
    incr next_temp;
    t
  in
  let fresh_var () =
    let v = Loop_var.of_int !next_var in
    incr next_var;
    v
  in
  let const_of = function
    | Loop_index.Const n -> n
    | _ ->
        invalid_arg "Loop_vector_expand: a vector loop without constant bounds"
  in
  let vector_loop (l : V.loop) : Loop_stmt.t list =
    let lo = const_of l.V.lo and hi = const_of l.V.hi in
    let lanes = l.V.lanes in
    let trips = max 0 (hi - lo) / lanes in
    let j = fresh_var () in
    (* Lane [k] of iteration [j] runs the body at [var = lo + lanes * j + k]. *)
    let lane_var k =
      add_const
        (Loop_index.Add
           (Loop_index.Const lo, Loop_index.Scale (lanes, Loop_index.Var j)))
        k
    in
    let temps = Hashtbl.create 8 in
    let temp_of t k =
      match Hashtbl.find_opt temps (V.Temp.to_int t, k) with
      | Some s -> s
      | None ->
          let s = fresh_temp () in
          Hashtbl.add temps (V.Temp.to_int t, k) s;
          s
    in
    let at (a : V.Access.t) k =
      Loop_index.Add
        ( subst l.V.var (lane_var 0) a.V.Access.offset,
          Loop_index.Const (k * a.V.Access.stride) )
    in
    let rec expr (e : V.t) k : float Loop_expr.t =
      match e with
      | V.Binary (op, a, b) -> Loop_expr.Binary (op, expr a k, expr b k)
      | V.Const x -> Loop_expr.Const x
      | V.Float_max (a, b) -> Loop_expr.Float_max (expr a k, expr b k)
      | V.Index_value { base; step } ->
          Loop_expr.Value_of_index
            (Loop_index.Add
               (subst l.V.var (lane_var 0) base, Loop_index.Const (k * step)))
      | V.Load a -> Loop_expr.Load_flat (a.V.Access.buffer, at a k)
      | V.Round_f32 a -> Loop_expr.Round_f32 (expr a k)
      | V.Select (m, a, b) -> Loop_expr.Select (mask m k, expr a k, expr b k)
      | V.Splat s -> s
      | V.Temp t -> Loop_expr.Temp (Loop_carrier.Float, temp_of t k)
      | V.Unary (op, a) -> Loop_expr.Unary (op, expr a k)
    and mask (m : V.mask) k : Loop_expr.pred =
      match m with
      | V.Not m -> Loop_bool.Not (mask m k)
      | V.Or (a, b) -> Loop_bool.Or (mask a k, mask b k)
      | V.Pool_better (a, b) -> Loop_bool.Pool_better (expr a k, expr b k)
      | V.Value_eq (a, b) -> Loop_bool.Value_eq (expr a k, expr b k)
      | V.Value_lt (a, b) -> Loop_bool.Value_lt (expr a k, expr b k)
    in
    let lanes_of f = List.init lanes f in
    let rec stmts ss = List.concat_map stmt ss
    and stmt (s : V.stmt) =
      match s with
      | V.Assign (t, e) ->
          lanes_of (fun k ->
              Loop_stmt.Assign (Loop_carrier.Float, temp_of t k, expr e k))
      | V.Store { access; value } ->
          lanes_of (fun k ->
              Loop_stmt.Store_flat
                {
                  buffer = access.V.Access.buffer;
                  offset = at access k;
                  value =
                    (match value with
                    | V.F32 e -> Loop_stored.F32 (expr e k)
                    | V.Bool e -> Loop_stored.Bool (expr e k));
                })
      | V.Mark m -> lanes_of (fun _ -> Loop_stmt.Mark m)
      | V.Index_assign (t, i) ->
          (* The same for every lane: assigned once. *)
          [ Loop_stmt.Assign_index (t, subst l.V.var (lane_var 0) i) ]
      | V.Inner { var; lo; hi; body } ->
          [ Loop_stmt.For { var; lo; hi; body = stmts body } ]
    in
    let body = stmts l.V.body in
    let remainder =
      match l.V.scalar with
      | Loop_stmt.For f ->
          [
            Loop_stmt.For
              {
                f with
                lo = Loop_index.Const (lo + (trips * lanes));
                hi = l.V.hi;
              };
          ]
      | s -> [ s ]
    in
    (if trips = 0 then []
     else
       [
         Loop_stmt.For
           {
             var = j;
             lo = Loop_index.Const 0;
             hi = Loop_index.Const trips;
             body;
           };
       ])
    @ remainder
  in
  let rec node (n : V.node) : Loop_stmt.t list =
    match n with
    | V.Scalar s -> [ s ]
    | V.Vector l -> vector_loop l
    | V.If (c, a, b) -> [ Loop_stmt.If (c, nodes a, nodes b) ]
    | V.Loop { var; lo; hi; body } ->
        [ Loop_stmt.For { var; lo; hi; body = nodes body } ]
  and nodes ns = List.concat_map node ns in
  { p.V.scalar with Loop_program.body = nodes p.V.body }
