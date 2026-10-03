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
  | Loop_expr.Fma (a, b, c) ->
      scan_expr ids a;
      scan_expr ids b;
      scan_expr ids c
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
  | Loop_stmt.Reduce_sum { var; lo; hi; acc; body; term; _ } ->
      ids.var <- max ids.var (Loop_var.to_int var);
      ids.temp <- max ids.temp (Loop_temp.to_int acc);
      scan_index ids lo;
      scan_index ids hi;
      List.iter (scan_stmt ids) body;
      scan_expr ids term
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

(* Lane [k] of a vector expression at [var = base], as a scalar expression: the
   scalar program's own loads and arithmetic, one lane at a time. [temp] names the
   scalar temporary that holds a vector temporary's lane. *)
let rec lane_expr ~var ~base ~temp (e : V.t) k : float Loop_expr.t =
  let expr e k = lane_expr ~var ~base ~temp e k in
  match e with
  | V.Binary (op, a, b) -> Loop_expr.Binary (op, expr a k, expr b k)
  | V.Const x -> Loop_expr.Const x
  | V.Float_max (a, b) -> Loop_expr.Float_max (expr a k, expr b k)
  | V.Fma (a, b, c) -> Loop_expr.Fma (expr a k, expr b k, expr c k)
  | V.Index_value { base = b; step } ->
      Loop_expr.Value_of_index
        (Loop_index.Add (subst var base b, Loop_index.Const (k * step)))
  | V.Load a ->
      Loop_expr.Load_flat
        ( a.V.Access.buffer,
          Loop_index.Add
            ( subst var base a.V.Access.offset,
              Loop_index.Const (k * a.V.Access.stride) ) )
  | V.Round_f32 a -> Loop_expr.Round_f32 (expr a k)
  | V.Select (m, a, b) ->
      Loop_expr.Select (lane_mask ~var ~base ~temp m k, expr a k, expr b k)
  | V.Splat s -> s
  | V.Temp t -> Loop_expr.Temp (Loop_carrier.Float, temp t k)
  | V.Unary (op, a) -> Loop_expr.Unary (op, expr a k)

and lane_mask ~var ~base ~temp (m : V.mask) k : Loop_expr.pred =
  let expr e k = lane_expr ~var ~base ~temp e k in
  match m with
  | V.Not m -> Loop_bool.Not (lane_mask ~var ~base ~temp m k)
  | V.Or (a, b) ->
      Loop_bool.Or
        (lane_mask ~var ~base ~temp a k, lane_mask ~var ~base ~temp b k)
  | V.Pool_better (a, b) -> Loop_bool.Pool_better (expr a k, expr b k)
  | V.Value_eq (a, b) -> Loop_bool.Value_eq (expr a k, expr b k)
  | V.Value_lt (a, b) -> Loop_bool.Value_lt (expr a k, expr b k)

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
    let expr e k =
      lane_expr ~var:l.V.var ~base:(lane_var 0) ~temp:temp_of e k
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
  (* A scheduled sum as the scalar statements its definition spells out
     ({!Loop_vector.Reduction}): lane accumulators, the main rounds as a loop and
     the leftover vectors straight-line, the adjacent-pair trees, the sequential
     tail and the seed. *)
  let reduction (r : V.Reduction.t) : Loop_stmt.t list =
    let lanes = r.V.Reduction.lanes and parts = r.V.Reduction.parts in
    let lo = r.V.Reduction.lo in
    let n = r.V.Reduction.hi - lo in
    let full = n / lanes in
    let main = full / parts and extra = full mod parts in
    let tail = n - (full * lanes) in
    let float t = Loop_expr.Temp (Loop_carrier.Float, t) in
    let add a b = Loop_expr.Binary (Expr.Value.Add, a, b) in
    let assign t e = Loop_stmt.Assign (Loop_carrier.Float, t, e) in
    let mark = Loop_stmt.Mark Loop_mark.Reduction in
    let lane_temps =
      Array.init parts (fun _ -> Array.init lanes (fun _ -> fresh_temp ()))
    in
    let no_temps _ _ =
      invalid_arg "Loop_vector_expand: a reduction term has no temporaries"
    in
    let term_at base k =
      lane_expr ~var:r.V.Reduction.var ~base ~temp:no_temps r.V.Reduction.term k
    in
    (* One accumulate: [acc + term], or [fma(a, b, acc)] when the sum is fused. *)
    let accumulate acc_expr base k =
      match (r.V.Reduction.fused, r.V.Reduction.term) with
      | true, V.Binary (Expr.Value.Mul, a, b) ->
          Loop_expr.Fma
            ( lane_expr ~var:r.V.Reduction.var ~base ~temp:no_temps a k,
              lane_expr ~var:r.V.Reduction.var ~base ~temp:no_temps b k,
              acc_expr )
      | true, _ ->
          invalid_arg "Loop_vector_expand: a fused reduction without a product"
      | false, _ -> add acc_expr (term_at base k)
    in
    let each_lane f = List.concat (List.init lanes f) in
    let init =
      List.concat
        (List.init parts (fun j ->
             List.init lanes (fun k ->
                 assign lane_temps.(j).(k) (Loop_expr.Const 0.))))
    in
    let i = fresh_var () in
    let main_stmts =
      if main = 0 then []
      else
        [
          Loop_stmt.For
            {
              var = i;
              lo = Loop_index.Const 0;
              hi = Loop_index.Const main;
              body =
                List.concat
                  (List.init parts (fun j ->
                       let base =
                         Loop_index.Add
                           ( Loop_index.Const (lo + (j * lanes)),
                             Loop_index.Scale (parts * lanes, Loop_index.Var i)
                           )
                       in
                       each_lane (fun k ->
                           [
                             mark;
                             assign
                               lane_temps.(j).(k)
                               (accumulate (float lane_temps.(j).(k)) base k);
                           ])));
            };
        ]
    in
    let extra_stmts =
      List.concat
        (List.init extra (fun e ->
             let base =
               Loop_index.Const (lo + (((main * parts) + e) * lanes))
             in
             each_lane (fun k ->
                 [
                   mark;
                   assign
                     lane_temps.(e).(k)
                     (accumulate (float lane_temps.(e).(k)) base k);
                 ])))
    in
    (* The adjacent-pair tree: neighbours are added, an odd one out carried on. *)
    let rec tree = function
      | [] -> invalid_arg "Loop_vector_expand.tree"
      | [ x ] -> x
      | xs ->
          let rec pairs = function
            | a :: b :: rest -> add a b :: pairs rest
            | rest -> rest
          in
          tree (pairs xs)
    in
    let combined = Array.init lanes (fun _ -> fresh_temp ()) in
    let combine =
      List.init lanes (fun k ->
          assign combined.(k)
            (tree (List.init parts (fun j -> float lane_temps.(j).(k)))))
    in
    let horizontal = fresh_temp () in
    let horizontal_stmt =
      assign horizontal (tree (List.init lanes (fun k -> float combined.(k))))
    in
    let tail_temp = fresh_temp () in
    let tail_stmts =
      assign tail_temp (Loop_expr.Const 0.)
      :: List.concat
           (List.init tail (fun u ->
                let base = Loop_index.Const (lo + (full * lanes) + u) in
                [ mark; assign tail_temp (accumulate (float tail_temp) base 0) ]))
    in
    init @ main_stmts @ extra_stmts @ combine @ [ horizontal_stmt ] @ tail_stmts
    @ [
        assign r.V.Reduction.acc
          (add (Loop_expr.Const r.V.Reduction.seed)
             (add (float horizontal) (float tail_temp)));
      ]
  in
  let rec node (n : V.node) : Loop_stmt.t list =
    match n with
    | V.Scalar s -> [ s ]
    | V.Reduction r -> reduction r
    | V.Vector l -> vector_loop l
    | V.If (c, a, b) -> [ Loop_stmt.If (c, nodes a, nodes b) ]
    | V.Loop { var; lo; hi; body } ->
        [ Loop_stmt.For { var; lo; hi; body = nodes body } ]
  and nodes ns = List.concat_map node ns in
  { p.V.scalar with Loop_program.body = nodes p.V.body }
