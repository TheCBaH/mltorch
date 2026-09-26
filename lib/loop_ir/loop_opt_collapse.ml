(* Loop collapsing: [for v1 in [0, n1): for v2 in [0, n2): B] becomes
   [for g in [0, n1 * n2): B'] when [v1] and [v2] only ever reach [B] through
   the dense offsets of its accesses, and in each of them [v1] steps [n2]
   times as far as [v2] ([c1 = n2 * c2]): then [c1 * v1 + c2 * v2] is
   [c2 * g] with [g = n2 * v1 + v2], the iteration order is unchanged, and
   every access of [B'] that named them becomes a flat access
   ([Loop_expr.Load_flat], [Loop_stmt.Store_flat]) at the rewritten offset.
   A broadcast axis ([c1 = 0] beside [c2 <> 0]) and a transposed one fail the
   ratio, so they stay nested.

   Pairs are collapsed bottom-up, so a chain of dense loops collapses into one:
   the merged variable is the next pair's inner one, and its coefficient
   [c2] is the ratio's base again.

   What keeps each rewrite exact:
   - The iteration count and order are the nest's, and [B] runs once per
     iteration, so every statement's effects (a [Mark], a [Store], a failure)
     happen the same number of times in the same order (invariant 5).
   - A mention of [v1]/[v2] anywhere but an access offset (a guard, a failure
     payload, an index temporary, [Value_of_index], an inner loop's bound)
     refuses the pair, so no failure payload changes (invariant 2).
   - An access's offset value is unchanged, and the rewritten offset is proven
     overflow-free over [g]'s range and the enclosing scope, or the pair is
     refused (invariant 3). *)

let ( let* ) = Option.bind

type pair = {
  outer : Loop_var.t;
  inner : Loop_var.t;
  n_inner : int;
  merged : Loop_var.t;
}

let rec mentions (p : pair) : Loop_index.t -> bool = function
  | Loop_index.Var v -> Loop_var.equal v p.outer || Loop_var.equal v p.inner
  | Loop_index.Temp _ | Loop_index.Const _ -> false
  | Loop_index.Add (a, b) | Loop_index.Max (a, b) | Loop_index.Min (a, b) ->
      mentions p a || mentions p b
  | Loop_index.Ceil_div_pos (a, _)
  | Loop_index.Clamp_low a
  | Loop_index.Floor_div_pos (a, _)
  | Loop_index.Scale (_, a) ->
      mentions p a

let coord_mentions p c =
  List.exists (fun a -> mentions p (Expr.Coord.get c a)) Expr.Axis.all

(* [Vec6.offset]'s row-major linearisation, as a linear form. *)
let dense_form (b : Loop_buffer.t) (c : Loop_index.coord) =
  let shape = b.Loop_buffer.sg.Tensor_sig.shape in
  List.fold_left
    (fun acc a ->
      let* acc = acc in
      let* comp = Loop_linear.of_index (Expr.Coord.get c a) in
      let* scaled = Loop_linear.scale (Dim.to_int (Vec6.get shape a)) acc in
      Loop_linear.add scaled comp)
    (Some (Loop_linear.const 0))
    Expr.Axis.all

let per_channel (b : Loop_buffer.t) =
  match b.Loop_buffer.sg.Tensor_sig.quant with
  | Some q -> Quant.channel_count q <> None
  | None -> false

(* The flat offset of one access after the merge, or [None] when the access
   refuses it. *)
let merged_offset ~ratio p env (b : Loop_buffer.t) (form : Loop_linear.t option)
    =
  let* form = form in
  let direct (a, _) =
    match a with
    | Loop_index.Var v -> Loop_var.equal v p.outer || Loop_var.equal v p.inner
    | _ -> false
  in
  if per_channel b then None
  else if
    List.exists (fun t -> (not (direct t)) && mentions p (fst t)) form.terms
  then None
  else
    let c1 = Loop_linear.coefficient form (Loop_index.Var p.outer)
    and c2 = Loop_linear.coefficient form (Loop_index.Var p.inner) in
    if not (ratio ~c1 ~c2 ~n_inner:p.n_inner) then None
    else
      let* without =
        Loop_linear.sub form
          {
            Loop_linear.terms =
              [ (Loop_index.Var p.outer, c1); (Loop_index.Var p.inner, c2) ]
              |> List.filter (fun (_, c) -> c <> 0);
            const = 0;
          }
      in
      let* with_g =
        Loop_linear.add
          {
            Loop_linear.terms =
              (if c2 = 0 then [] else [ (Loop_index.Var p.merged, c2) ]);
            const = 0;
          }
          without
      in
      let idx = Loop_linear.to_index with_g in
      if Loop_range.proven env idx then Some idx else None

(* Every access of the body that names [v1]/[v2], rewritten flat; [None] as
   soon as one refuses. The scope is tracked through inner loops and index
   temporaries, since an offset is proven where it is evaluated. *)
let rewrite_body ~ratio ~single p env0 body =
  let refused = ref false in
  let offset env b form =
    match merged_offset ~ratio p env b form with
    | Some i -> i
    | None ->
        refused := true;
        Loop_index.Const 0
  in
  let rec expr : type a. Loop_range.Env.t -> a Loop_expr.t -> a Loop_expr.t =
   fun env e ->
    let ex e = expr env e in
    match e with
    | Loop_expr.Load (b, c) when coord_mentions p c ->
        Loop_expr.Load_flat (b, offset env b (dense_form b c))
    | Loop_expr.Load_i64 (b, c) when coord_mentions p c ->
        Loop_expr.Load_i64_flat (b, offset env b (dense_form b c))
    | Loop_expr.Load_flat (b, i) when mentions p i ->
        Loop_expr.Load_flat (b, offset env b (Loop_linear.of_index i))
    | Loop_expr.Load_i64_flat (b, i) when mentions p i ->
        Loop_expr.Load_i64_flat (b, offset env b (Loop_linear.of_index i))
    | Loop_expr.Array_get _ | Loop_expr.Const _ | Loop_expr.I64_const _
    | Loop_expr.I64_of_index _ | Loop_expr.Load _ | Loop_expr.Load_flat _
    | Loop_expr.Load_i64 _ | Loop_expr.Load_i64_flat _ | Loop_expr.Temp _
    | Loop_expr.Value_of_index _ ->
        e
    | Loop_expr.Binary (op, a, b) -> Loop_expr.Binary (op, ex a, ex b)
    | Loop_expr.Float_max (a, b) -> Loop_expr.Float_max (ex a, ex b)
    | Loop_expr.Float_to_i64 a -> Loop_expr.Float_to_i64 (ex a)
    | Loop_expr.I64_binary (op, a, b) -> Loop_expr.I64_binary (op, ex a, ex b)
    | Loop_expr.I64_to_float a -> Loop_expr.I64_to_float (ex a)
    | Loop_expr.Round_f32 a -> Loop_expr.Round_f32 (ex a)
    | Loop_expr.Select (c, a, b) -> Loop_expr.Select (pred env c, ex a, ex b)
    | Loop_expr.Unary (op, a) -> Loop_expr.Unary (op, ex a)
  and pred env (q : Loop_expr.pred) : Loop_expr.pred =
    match q with
    | Loop_bool.I64_eq (a, b) -> Loop_bool.I64_eq (expr env a, expr env b)
    | Loop_bool.I64_lt (a, b) -> Loop_bool.I64_lt (expr env a, expr env b)
    | Loop_bool.Not q -> Loop_bool.Not (pred env q)
    | Loop_bool.Or (a, b) -> Loop_bool.Or (pred env a, pred env b)
    | Loop_bool.Pool_better (a, b) ->
        Loop_bool.Pool_better (expr env a, expr env b)
    | Loop_bool.Value_eq (a, b) -> Loop_bool.Value_eq (expr env a, expr env b)
    | Loop_bool.Value_lt (a, b) -> Loop_bool.Value_lt (expr env a, expr env b)
    | Loop_bool.Index_eq _ | Loop_bool.Index_lt _ | Loop_bool.Index_overflows _
    | Loop_bool.Out_of_range _ ->
        q
  in
  let stored env : Loop_stored.t -> Loop_stored.t = function
    | Loop_stored.Bool e -> Loop_stored.Bool (expr env e)
    | Loop_stored.F32 e -> Loop_stored.F32 (expr env e)
    | Loop_stored.I64 e -> Loop_stored.I64 (expr env e)
  in
  let rec stmts env ss = List.map (stmt env) ss
  and stmt env (s : Loop_stmt.t) : Loop_stmt.t =
    match s with
    | Loop_stmt.Store { buffer; coord; value } when coord_mentions p coord ->
        let value = stored env value in
        Loop_stmt.Store_flat
          {
            buffer;
            offset = offset env buffer (dense_form buffer coord);
            value;
          }
    | Loop_stmt.Store { buffer; coord; value } ->
        Loop_stmt.Store { buffer; coord; value = stored env value }
    | Loop_stmt.Store_flat { buffer; offset = i; value } ->
        let value = stored env value in
        let i =
          if mentions p i then offset env buffer (Loop_linear.of_index i) else i
        in
        Loop_stmt.Store_flat { buffer; offset = i; value }
    | Loop_stmt.For { var; lo; hi; body } ->
        let env' =
          Loop_range.Env.add_var var (Loop_opt_scope.var_range env lo hi) env
        in
        Loop_stmt.For { var; lo; hi; body = stmts env' body }
    | Loop_stmt.If (q, a, b) ->
        Loop_stmt.If (pred env q, stmts env a, stmts env b)
    | Loop_stmt.Assign (c, t, e) -> Loop_stmt.Assign (c, t, expr env e)
    | Loop_stmt.Array_set (a, i, e) -> Loop_stmt.Array_set (a, i, expr env e)
    | Loop_stmt.Assign_index (t, i) ->
        if single t then
          Loop_range.Env.set_temp t (Loop_range.of_index env i) env;
        s
    | Loop_stmt.Fail_if (q, f) -> Loop_stmt.Fail_if (pred env q, f)
    | Loop_stmt.Assign_index_of_i64 (t, e) ->
        Loop_stmt.Assign_index_of_i64 (t, expr env e)
    | Loop_stmt.Alloc _ | Loop_stmt.Charge_scan_update | Loop_stmt.Mark _
    | Loop_stmt.Release_scan_state _ | Loop_stmt.Reserve_scan_state _
    | Loop_stmt.Reset_meter ->
        s
  in
  let body = stmts env0 body in
  (* Whatever mention is left is one no access offset carries. *)
  let left = ref false in
  ignore
    (Loop_index_map.stmts
       ~f:(fun i ->
         (match i with
         | Loop_index.Var v
           when Loop_var.equal v p.outer || Loop_var.equal v p.inner ->
             left := true
         | _ -> ());
         i)
       body);
  if !refused || !left then None else Some body

let extent (lo : Loop_index.t) (hi : Loop_index.t) =
  match (lo, hi) with
  | Loop_index.Const 0, Loop_index.Const n when n >= 1 -> Some n
  | _ -> None

let rec collapse ~ratio ~single ~fresh env (stmts : Loop_stmt.t list) =
  List.map
    (fun (s : Loop_stmt.t) ->
      match s with
      | Loop_stmt.For { var; lo; hi; body } -> (
          let env1 =
            Loop_range.Env.add_var var (Loop_opt_scope.var_range env lo hi) env
          in
          let body = collapse ~ratio ~single ~fresh env1 body in
          let unchanged = Loop_stmt.For { var; lo; hi; body } in
          match (extent lo hi, body) with
          | ( Some n1,
              [
                Loop_stmt.For { var = inner; lo = lo2; hi = hi2; body = body2 };
              ] ) -> (
              match extent lo2 hi2 with
              | None -> unchanged
              | Some n2 -> (
                  let total = Loop_range.saturating_mul n1 (Int64.of_int n2) in
                  if
                    not
                      (Loop_range.within ~inner:(Loop_range.point total)
                         ~outer:Loop_range.domain)
                  then unchanged
                  else
                    let merged = fresh () in
                    let p = { outer = var; inner; n_inner = n2; merged } in
                    let total = Int64.to_int total in
                    let env' =
                      Loop_range.Env.add_var merged
                        (Loop_range.span ~lo:0 ~hi:total)
                        env
                    in
                    match rewrite_body ~ratio ~single p env' body2 with
                    | Some body ->
                        Loop_stmt.For
                          {
                            var = merged;
                            lo = Loop_index.Const 0;
                            hi = Loop_index.Const total;
                            body;
                          }
                    | None -> unchanged))
          | _ -> unchanged)
      | Loop_stmt.If (q, a, b) ->
          Loop_stmt.If
            ( q,
              collapse ~ratio ~single ~fresh env a,
              collapse ~ratio ~single ~fresh env b )
      | Loop_stmt.Assign_index (t, i) ->
          if single t then
            Loop_range.Env.set_temp t (Loop_range.of_index env i) env;
          s
      | s -> s)
    stmts

let rec max_var (stmts : Loop_stmt.t list) n =
  List.fold_left
    (fun n (s : Loop_stmt.t) ->
      match s with
      | Loop_stmt.For { var; body; _ } ->
          max_var body (Loop_var.Next.after var n)
      | Loop_stmt.If (_, a, b) -> max_var b (max_var a n)
      | _ -> n)
    n stmts

let with_ratio ratio program =
  let body = program.Loop_program.body in
  let single = Loop_opt_scope.single_assignment body in
  let supply = ref (max_var body Loop_var.Next.first) in
  let fresh () =
    let v, next = Loop_var.Next.alloc !supply in
    supply := next;
    v
  in
  {
    program with
    Loop_program.body =
      collapse ~ratio ~single ~fresh (Loop_range.Env.create ()) body;
  }

(* [c1 = n2 * c2], in [int64]: the product of two in-domain ints can leave a
   32-bit one. *)
let contiguous ~c1 ~c2 ~n_inner =
  Int64.equal (Int64.of_int c1)
    (Loop_range.saturating_mul n_inner (Int64.of_int c2))

let run = with_ratio contiguous
