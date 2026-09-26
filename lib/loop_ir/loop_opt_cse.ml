(* Every [Load] of an expression, each tagged with whether evaluating the
   expression always reaches it. A [Select] branch and the right of an [Or]
   are evaluated only when their condition holds, and a read there may be in
   range only because of that condition. *)
let rec collect_loads : type a.
    cond:bool -> a Loop_expr.t -> (Loop_buffer.t * Loop_index.coord * bool) list
    =
 fun ~cond e ->
  match e with
  | Loop_expr.Array_get _ -> []
  | Loop_expr.Binary (_, a, b) -> collect_loads ~cond a @ collect_loads ~cond b
  | Loop_expr.Const _ -> []
  | Loop_expr.Float_max (a, b) -> collect_loads ~cond a @ collect_loads ~cond b
  | Loop_expr.Float_to_i64 a -> collect_loads ~cond a
  | Loop_expr.I64_binary (_, a, b) ->
      collect_loads ~cond a @ collect_loads ~cond b
  | Loop_expr.I64_const _ -> []
  | Loop_expr.I64_of_index _ -> []
  | Loop_expr.I64_to_float a -> collect_loads ~cond a
  | Loop_expr.Load (buf, coord) -> [ (buf, coord, cond) ]
  | Loop_expr.Load_flat _ -> []
  | Loop_expr.Load_i64 _ | Loop_expr.Load_i64_flat _ -> []
  | Loop_expr.Round_f32 a -> collect_loads ~cond a
  | Loop_expr.Select (p, a, b) ->
      collect_loads_pred ~cond p @ collect_loads ~cond:true a
      @ collect_loads ~cond:true b
  | Loop_expr.Temp _ -> []
  | Loop_expr.Unary (_, a) -> collect_loads ~cond a
  | Loop_expr.Value_of_index _ -> []

and collect_loads_pred ~cond (p : Loop_expr.pred) =
  match p with
  | Loop_bool.I64_eq (a, b) | Loop_bool.I64_lt (a, b) ->
      collect_loads ~cond a @ collect_loads ~cond b
  | Loop_bool.Index_eq _ | Loop_bool.Index_lt _ | Loop_bool.Index_overflows _
  | Loop_bool.Out_of_range _ ->
      []
  | Loop_bool.Not p -> collect_loads_pred ~cond p
  | Loop_bool.Or (a, b) ->
      collect_loads_pred ~cond a @ collect_loads_pred ~cond:true b
  | Loop_bool.Pool_better (a, b)
  | Loop_bool.Value_eq (a, b)
  | Loop_bool.Value_lt (a, b) ->
      collect_loads ~cond a @ collect_loads ~cond b

let same_load (b1, c1) (b2, c2) =
  Tensor_id.equal b1.Loop_buffer.id b2.Loop_buffer.id && Stdlib.( = ) c1 c2

(* A load read at least twice, at least once unconditionally: computing it
   once before the statement then reads nothing the statement would not. *)
let distinct_duplicate_keys loads =
  let keys = List.map (fun (b, c, _) -> (b, c)) loads in
  List.fold_left
    (fun acc (b, c, cond) ->
      let key = (b, c) in
      if
        (not cond)
        && (not (List.exists (same_load key) acc))
        && List.length (List.filter (same_load key) keys) >= 2
      then key :: acc
      else acc)
    [] loads

let rec rewrite_expr : type a.
    (Loop_buffer.t * Loop_index.coord * Loop_temp.t) list ->
    a Loop_expr.t ->
    a Loop_expr.t =
 fun subs e ->
  match e with
  | Loop_expr.Array_get _ -> e
  | Loop_expr.Binary (op, a, b) ->
      Loop_expr.Binary (op, rewrite_expr subs a, rewrite_expr subs b)
  | Loop_expr.Const _ -> e
  | Loop_expr.Float_max (a, b) ->
      Loop_expr.Float_max (rewrite_expr subs a, rewrite_expr subs b)
  | Loop_expr.Float_to_i64 a -> Loop_expr.Float_to_i64 (rewrite_expr subs a)
  | Loop_expr.I64_binary (op, a, b) ->
      Loop_expr.I64_binary (op, rewrite_expr subs a, rewrite_expr subs b)
  | Loop_expr.I64_const _ -> e
  | Loop_expr.I64_of_index _ -> e
  | Loop_expr.I64_to_float a -> Loop_expr.I64_to_float (rewrite_expr subs a)
  | Loop_expr.Load (buf, coord) -> (
      match
        List.find_opt (fun (b, c, _) -> same_load (buf, coord) (b, c)) subs
      with
      | Some (_, _, t) -> Loop_expr.Temp (Loop_carrier.Float, t)
      | None -> e)
  | Loop_expr.Load_flat _ -> e
  | Loop_expr.Load_i64 _ | Loop_expr.Load_i64_flat _ -> e
  | Loop_expr.Round_f32 a -> Loop_expr.Round_f32 (rewrite_expr subs a)
  | Loop_expr.Select (p, a, b) ->
      Loop_expr.Select
        (rewrite_pred subs p, rewrite_expr subs a, rewrite_expr subs b)
  | Loop_expr.Temp _ -> e
  | Loop_expr.Unary (op, a) -> Loop_expr.Unary (op, rewrite_expr subs a)
  | Loop_expr.Value_of_index _ -> e

and rewrite_pred subs (p : Loop_expr.pred) : Loop_expr.pred =
  match p with
  | Loop_bool.I64_eq (a, b) ->
      Loop_bool.I64_eq (rewrite_expr subs a, rewrite_expr subs b)
  | Loop_bool.I64_lt (a, b) ->
      Loop_bool.I64_lt (rewrite_expr subs a, rewrite_expr subs b)
  | Loop_bool.Index_eq _ | Loop_bool.Index_lt _ | Loop_bool.Index_overflows _
  | Loop_bool.Out_of_range _ ->
      p
  | Loop_bool.Not p -> Loop_bool.Not (rewrite_pred subs p)
  | Loop_bool.Or (a, b) ->
      Loop_bool.Or (rewrite_pred subs a, rewrite_pred subs b)
  | Loop_bool.Pool_better (a, b) ->
      Loop_bool.Pool_better (rewrite_expr subs a, rewrite_expr subs b)
  | Loop_bool.Value_eq (a, b) ->
      Loop_bool.Value_eq (rewrite_expr subs a, rewrite_expr subs b)
  | Loop_bool.Value_lt (a, b) ->
      Loop_bool.Value_lt (rewrite_expr subs a, rewrite_expr subs b)

let rec stored_buffer_ids (stmts : Loop_stmt.t list) acc =
  List.fold_left
    (fun acc (stmt : Loop_stmt.t) ->
      match stmt with
      | Loop_stmt.Store { buffer; _ } | Loop_stmt.Store_flat { buffer; _ } ->
          Tensor_id.Set.add buffer.Loop_buffer.id acc
      | Loop_stmt.For { body; _ } -> stored_buffer_ids body acc
      | Loop_stmt.If (_, a, b) -> stored_buffer_ids b (stored_buffer_ids a acc)
      | _ -> acc)
    acc stmts

let rec max_temp (stmts : Loop_stmt.t list) (n : Loop_temp.Next.t) =
  List.fold_left
    (fun n (stmt : Loop_stmt.t) ->
      match stmt with
      | Loop_stmt.Assign (_, t, _)
      | Loop_stmt.Assign_index (t, _)
      | Loop_stmt.Assign_index_of_i64 (t, _) ->
          Loop_temp.Next.after t n
      | Loop_stmt.For { body; _ } -> max_temp body n
      | Loop_stmt.If (_, a, b) -> max_temp b (max_temp a n)
      | _ -> n)
    n stmts

let cse_value ~stored ~fresh (rebuild : float Loop_expr.t -> Loop_stmt.t)
    (e : float Loop_expr.t) : Loop_stmt.t list =
  let candidates =
    List.filter
      (fun (b, _, _) -> not (Tensor_id.Set.mem b.Loop_buffer.id stored))
      (collect_loads ~cond:false e)
  in
  match distinct_duplicate_keys candidates with
  | [] -> [ rebuild e ]
  | keys ->
      let subs = List.map (fun (buf, coord) -> (buf, coord, fresh ())) keys in
      let inits =
        List.map
          (fun (buf, coord, t) ->
            Loop_stmt.Assign (Loop_carrier.Float, t, Loop_expr.Load (buf, coord)))
          subs
      in
      inits @ [ rebuild (rewrite_expr subs e) ]

let rec rewrite ~stored ~fresh (stmts : Loop_stmt.t list) : Loop_stmt.t list =
  List.concat_map (rewrite_stmt ~stored ~fresh) stmts

and rewrite_stmt ~stored ~fresh (stmt : Loop_stmt.t) : Loop_stmt.t list =
  match stmt with
  | Loop_stmt.Store { buffer; coord; value = Loop_stored.F32 e } ->
      cse_value ~stored ~fresh
        (fun e -> Loop_stmt.Store { buffer; coord; value = Loop_stored.F32 e })
        e
  | Loop_stmt.Store { buffer; coord; value = Loop_stored.Bool e } ->
      cse_value ~stored ~fresh
        (fun e -> Loop_stmt.Store { buffer; coord; value = Loop_stored.Bool e })
        e
  | Loop_stmt.Assign (Loop_carrier.Float, t, e) ->
      cse_value ~stored ~fresh
        (fun e -> Loop_stmt.Assign (Loop_carrier.Float, t, e))
        e
  | Loop_stmt.For { var; lo; hi; body } ->
      [ Loop_stmt.For { var; lo; hi; body = rewrite ~stored ~fresh body } ]
  | Loop_stmt.If (p, a, b) ->
      [ Loop_stmt.If (p, rewrite ~stored ~fresh a, rewrite ~stored ~fresh b) ]
  | s -> [ s ]

let run program =
  let stored =
    stored_buffer_ids program.Loop_program.body Tensor_id.Set.empty
  in
  let supply = ref (max_temp program.Loop_program.body Loop_temp.Next.first) in
  let fresh () =
    let id, next = Loop_temp.Next.alloc !supply in
    supply := next;
    id
  in
  {
    program with
    Loop_program.body = rewrite ~stored ~fresh program.Loop_program.body;
  }
