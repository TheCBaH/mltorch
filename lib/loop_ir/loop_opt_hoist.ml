(* Loop-invariant hoisting of a statement: an [Assign] of a float temporary or
   an [Array_set] of a constant cell, at the top level of a loop's body, whose
   value is closed (built from constants alone), moves to just before the
   loop. It then runs once instead of once per iteration: [sdpa]'s
   [a0[0] = Math.sqrt(1 / Math.sqrt(5))] per query row.

   The first attempt at this pass checked that the EXPRESSION was invariant
   and hoisted an accumulator's reset ([x0 = 0] before [x0 = x0 + ...]) out of
   the loop that re-initializes it per iteration. What makes the STATEMENT
   invariant is where its target is read and written, so a statement moves
   only when all of these hold:

   - The loop runs at least once: both bounds are constants, [lo < hi]. A loop
     that may not run never executes the statement at all (invariant 4's
     zero-trip rule).
   - The value is closed: no load, temporary, array read, loop variable or
     index, and nothing that can fail. Every iteration computes the same
     value, and computing it earlier reads nothing it would not have.
   - No other statement in the loop writes the target: not another
     assignment of the temporary (the accumulator), and not an [Array_set] of
     the same array whose index can reach the cell, nor an [Alloc] of it. So
     the target holds this value from the statement onward, every iteration.
   - No statement before it in the body reads the target (a read of the
     cell's array at an index that can reach the cell). On the first
     iteration such a read would see the value from before the loop, which
     hoisting would replace.

   Then every read inside the loop sees what it saw before, and after the loop
   the target holds the value the last iteration left. [Array_set] and
   [Assign] are not among invariant 5's ordered effects. The pass runs bottom
   up, so a statement hoisted out of an inner loop is a candidate for the
   enclosing one. *)

let rec closed : type a. a Loop_expr.t -> bool = function
  | Loop_expr.Const _ -> true
  | Loop_expr.Binary (_, a, b) | Loop_expr.Float_max (a, b) ->
      closed a && closed b
  | Loop_expr.Round_f32 a | Loop_expr.Unary (_, a) -> closed a
  | Loop_expr.Select (p, a, b) -> closed_pred p && closed a && closed b
  | Loop_expr.Array_get _ | Loop_expr.Float_to_i64 _ | Loop_expr.I64_binary _
  | Loop_expr.I64_const _ | Loop_expr.I64_of_index _ | Loop_expr.I64_to_float _
  | Loop_expr.Load _ | Loop_expr.Load_flat _ | Loop_expr.Load_i64 _
  | Loop_expr.Load_i64_flat _ | Loop_expr.Temp _ | Loop_expr.Value_of_index _ ->
      false

and closed_pred (p : Loop_expr.pred) =
  match p with
  | Loop_bool.Not p -> closed_pred p
  | Loop_bool.Or (a, b) -> closed_pred a && closed_pred b
  | Loop_bool.Pool_better (a, b)
  | Loop_bool.Value_eq (a, b)
  | Loop_bool.Value_lt (a, b) ->
      closed a && closed b
  | Loop_bool.I64_eq _ | Loop_bool.I64_lt _ | Loop_bool.Index_eq _
  | Loop_bool.Index_lt _ | Loop_bool.Index_overflows _
  | Loop_bool.Out_of_range _ ->
      false

type target = Temp of Loop_temp.t | Cell of Loop_array.t * int

let target_of (s : Loop_stmt.t) =
  match s with
  | Loop_stmt.Assign (Loop_carrier.Float, t, e) when closed e -> Some (Temp t)
  | Loop_stmt.Array_set (a, Loop_index.Const c, e) when closed e ->
      Some (Cell (a, c))
  | _ -> None

(* Whether an index evaluated in [env] can equal [c]. *)
let may_hit env (i : Loop_index.t) c =
  let r = Loop_range.of_index env i in
  let c = Int64.of_int c in
  Int64.compare r.Loop_range.lo c <= 0 && Int64.compare c r.Loop_range.hi <= 0

(* Every read of [target] in an expression, as a predicate. *)
let rec reads : type a. Loop_range.Env.t -> target -> a Loop_expr.t -> bool =
 fun env target e ->
  let go e = reads env target e in
  match e with
  | Loop_expr.Temp (_, t) -> (
      match target with Temp t' -> Loop_temp.equal t t' | Cell _ -> false)
  | Loop_expr.Array_get (a, i) -> (
      match target with
      | Cell (a', c) -> Loop_array.equal a a' && may_hit env i c
      | Temp _ -> false)
  | Loop_expr.Binary (_, a, b) | Loop_expr.Float_max (a, b) -> go a || go b
  | Loop_expr.I64_binary (_, a, b) -> go a || go b
  | Loop_expr.Float_to_i64 a -> go a
  | Loop_expr.I64_to_float a -> go a
  | Loop_expr.Round_f32 a | Loop_expr.Unary (_, a) -> go a
  | Loop_expr.Select (p, a, b) -> reads_pred env target p || go a || go b
  | Loop_expr.Const _ | Loop_expr.I64_const _ | Loop_expr.I64_of_index _
  | Loop_expr.Load _ | Loop_expr.Load_flat _ | Loop_expr.Load_i64 _
  | Loop_expr.Load_i64_flat _ | Loop_expr.Value_of_index _ ->
      false

and reads_pred env target (p : Loop_expr.pred) =
  match p with
  | Loop_bool.I64_eq (a, b) | Loop_bool.I64_lt (a, b) ->
      reads env target a || reads env target b
  | Loop_bool.Not p -> reads_pred env target p
  | Loop_bool.Or (a, b) -> reads_pred env target a || reads_pred env target b
  | Loop_bool.Pool_better (a, b)
  | Loop_bool.Value_eq (a, b)
  | Loop_bool.Value_lt (a, b) ->
      reads env target a || reads env target b
  | Loop_bool.Index_eq _ | Loop_bool.Index_lt _ | Loop_bool.Index_overflows _
  | Loop_bool.Out_of_range _ ->
      false

let reads_stored env target : Loop_stored.t -> bool = function
  | Loop_stored.Bool e | Loop_stored.F32 e -> reads env target e
  | Loop_stored.I64 e -> reads env target e

let reads_failure env target : Loop_failure.t -> bool = function
  | Loop_failure.Gather_out_of_range { raw; _ } -> reads env target raw
  | Loop_failure.I64_from_float { value } -> reads env target value
  | Loop_failure.I64_division_by_zero | Loop_failure.I64_division_overflow
  | Loop_failure.Index_overflow _ | Loop_failure.Load_out_of_range _
  | Loop_failure.Local_out_of_range _ | Loop_failure.Scan_lane_out_of_range _
  | Loop_failure.Scan_row_out_of_range _ ->
      false

(* Folds [f] over every statement of a block, nested ones included, with the
   scope each is evaluated in. *)
let rec fold_scoped f env acc (stmts : Loop_stmt.t list) =
  List.fold_left
    (fun acc (s : Loop_stmt.t) ->
      let acc = f env acc s in
      match s with
      | Loop_stmt.For { var; lo; hi; body } ->
          let env' =
            Loop_range.Env.add_var var (Loop_opt_scope.var_range env lo hi) env
          in
          fold_scoped f env' acc body
      | Loop_stmt.If (_, a, b) -> fold_scoped f env (fold_scoped f env acc a) b
      | _ -> acc)
    acc stmts

let stmt_reads env target (s : Loop_stmt.t) =
  match s with
  | Loop_stmt.Array_set (_, _, e) -> reads env target e
  | Loop_stmt.Assign (_, _, e) -> reads env target e
  | Loop_stmt.Assign_index_of_i64 (_, e) -> reads env target e
  | Loop_stmt.Fail_if (p, f) ->
      reads_pred env target p || reads_failure env target f
  | Loop_stmt.For _ -> false
  | Loop_stmt.If (p, _, _) -> reads_pred env target p
  | Loop_stmt.Store { value; _ } | Loop_stmt.Store_flat { value; _ } ->
      reads_stored env target value
  | Loop_stmt.Alloc _ | Loop_stmt.Assign_index _ | Loop_stmt.Charge_scan_update
  | Loop_stmt.Mark _ | Loop_stmt.Release_scan_state _
  | Loop_stmt.Reserve_scan_state _ | Loop_stmt.Reset_meter ->
      false

let stmt_writes env target (s : Loop_stmt.t) =
  match (target, s) with
  | Temp t, Loop_stmt.Assign (_, t', _) -> Loop_temp.equal t t'
  | Cell (a, c), Loop_stmt.Array_set (a', i, _) ->
      Loop_array.equal a a' && may_hit env i c
  | Cell (a, _), Loop_stmt.Alloc (a', _) -> Loop_array.equal a a'
  | _ -> false

let any env pred stmts =
  fold_scoped (fun env acc s -> acc || pred env s) env false stmts

(* [check_writes] is the mutation hook: the stage-7 mutation proof turns it off
   and hoists the accumulator reset the first attempt did. *)
let hoistable ~check_writes env body (i, s) =
  match target_of s with
  | None -> false
  | Some target ->
      let before = List.filteri (fun j _ -> j < i) body in
      let others = List.filteri (fun j _ -> j <> i) body in
      (not (any env (fun env s -> stmt_reads env target s) before))
      && ((not check_writes)
         || not (any env (fun env s -> stmt_writes env target s) others))

let runs_once (lo : Loop_index.t) (hi : Loop_index.t) =
  match (lo, hi) with
  | Loop_index.Const l, Loop_index.Const h -> l < h
  | _ -> false

let rec hoist ~check_writes ~single env (stmts : Loop_stmt.t list) =
  List.concat_map
    (fun (s : Loop_stmt.t) ->
      match s with
      | Loop_stmt.For { var; lo; hi; body } ->
          let env' =
            Loop_range.Env.add_var var (Loop_opt_scope.var_range env lo hi) env
          in
          let body = hoist ~check_writes ~single env' body in
          if not (runs_once lo hi) then [ Loop_stmt.For { var; lo; hi; body } ]
          else
            let indexed = List.mapi (fun i s -> (i, s)) body in
            let out, stay =
              List.partition (hoistable ~check_writes env' body) indexed
            in
            List.map snd out
            @ [ Loop_stmt.For { var; lo; hi; body = List.map snd stay } ]
      | Loop_stmt.If (p, a, b) ->
          [
            Loop_stmt.If
              ( p,
                hoist ~check_writes ~single env a,
                hoist ~check_writes ~single env b );
          ]
      | Loop_stmt.Assign_index (t, i) ->
          if single t then
            Loop_range.Env.set_temp t (Loop_range.of_index env i) env;
          [ s ]
      | s -> [ s ])
    stmts

let with_write_check check_writes program =
  let single = Loop_opt_scope.single_assignment program.Loop_program.body in
  {
    program with
    Loop_program.body =
      hoist ~check_writes ~single (Loop_range.Env.create ())
        program.Loop_program.body;
  }

let run = with_write_check true
