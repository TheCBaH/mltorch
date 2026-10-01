module V = Loop_vector
module F = Loop_vector_facts
module T = Loop_target

module Reason = struct
  type t =
    | Body_statement of string
    | Expression of string
    | Live_out_temp
    | Loop_carried
    | Non_affine_access
    | Non_constant_bounds
    | Store_through_broadcast
    | Too_short of { trips : int; lanes : int }
    | Unprofitable

  let name = function
    | Body_statement k -> "statement:" ^ k
    | Expression k -> "expression:" ^ k
    | Live_out_temp -> "live_out_temp"
    | Loop_carried -> "loop_carried"
    | Non_affine_access -> "non_affine_access"
    | Non_constant_bounds -> "non_constant_bounds"
    | Store_through_broadcast -> "store_through_broadcast"
    | Too_short _ -> "too_short"
    | Unprofitable -> "unprofitable"

  let equal (a : t) b = a = b
  let pp ppf r = Fmt.string ppf (name r)
end

module Decision = struct
  type outcome = Vectorized | Kept_scalar of Reason.t

  type t = {
    trips : int;
    work : int64;
    executions : int64;
    ops : int;
    outcome : outcome;
  }
end

type report = Decision.t list

let ( let* ) = Result.bind

type ctx = {
  var : Loop_var.t;
  loop_temps : Loop_temp.Set.t;
  mutable assigned : V.Temp.t Loop_temp.Map.t;
      (** the vector temporary each scalar temporary currently names: only what
          is definitely assigned at this point *)
  mutable next_temp : int;
  mutable ops : (T.Op.t * int) list;
  inner_ok : bool;
  mutable weight : int;
      (** how many times the statement being converted runs per iteration of the
          loop: the product of the trip counts of the inner loops around it, so
          the cost model weighs an inner loop's body by its length *)
}

let count ctx op =
  let w = ctx.weight in
  ctx.ops <-
    (match List.assoc_opt op ctx.ops with
    | Some n -> (op, n + w) :: List.remove_assoc op ctx.ops
    | None -> (op, w) :: ctx.ops)

let invariant ctx e = not (F.expr_depends ctx.var ctx.loop_temps e)

let access ctx (b : Loop_buffer.t) (offset : Loop_index.t) =
  if not (F.format_ok b) then Error (Reason.Expression "load or store format")
  else
    match F.coefficient ctx.var offset with
    | Error `Not_affine -> Error Reason.Non_affine_access
    | Ok stride -> Ok { V.Access.buffer = b; offset; stride }

let fmt_name (b : Loop_buffer.t) =
  let (Payload.Fmt f) = b.Loop_buffer.sg.Tensor_sig.fmt in
  Payload.fmt_name f

let load_op (a : V.Access.t) =
  if a.V.Access.stride = 0 then T.Op.Broadcast_load
  else if a.V.Access.stride <> 1 then T.Op.Strided_load
  else if fmt_name a.V.Access.buffer = "i32" then T.Op.Convert_i32_load
  else T.Op.Contiguous_load

let rec conv ctx (e : float Loop_expr.t) : (V.t, Reason.t) result =
  match e with
  | Loop_expr.Const x ->
      count ctx T.Op.Const;
      Ok (V.Const x)
  | _ when invariant ctx e ->
      count ctx T.Op.Splat;
      Ok (V.Splat e)
  | Loop_expr.Binary (op, a, b) ->
      let* a = conv ctx a in
      let* b = conv ctx b in
      count ctx
        (match op with
        | Expr.Value.Add -> T.Op.Add
        | Expr.Value.Div -> T.Op.Div
        | Expr.Value.Mul -> T.Op.Mul
        | Expr.Value.Sub -> T.Op.Sub);
      Ok (V.Binary (op, a, b))
  | Loop_expr.Float_max (a, b) ->
      let* a = conv ctx a in
      let* b = conv ctx b in
      count ctx T.Op.Float_max;
      Ok (V.Float_max (a, b))
  | Loop_expr.Round_f32 a ->
      let* a = conv ctx a in
      count ctx T.Op.Round_f32;
      Ok (V.Round_f32 a)
  | Loop_expr.Unary (op, a) ->
      let* a = conv ctx a in
      count ctx
        (match op with
        | Expr.Value.Sqrt | Expr.Value.Trunc -> T.Op.Sqrt_trunc
        | Expr.Value.Cos | Expr.Value.Erf | Expr.Value.Exp | Expr.Value.Log
        | Expr.Value.Sin ->
            T.Op.Transcendental);
      Ok (V.Unary (op, a))
  | Loop_expr.Load_flat (b, i) ->
      let* a = access ctx b i in
      count ctx (load_op a);
      Ok (V.Load a)
  | Loop_expr.Load (b, c) ->
      let* a = access ctx b (Loop_flat.offset b c) in
      count ctx (load_op a);
      Ok (V.Load a)
  | Loop_expr.Select (p, a, b) ->
      let* m = conv_pred ctx p in
      let* a = conv ctx a in
      let* b = conv ctx b in
      count ctx T.Op.Select;
      Ok (V.Select (m, a, b))
  | Loop_expr.Value_of_index i -> (
      match F.coefficient ctx.var i with
      | Error `Not_affine -> Error Reason.Non_affine_access
      | Ok step ->
          count ctx T.Op.Index_value;
          Ok (V.Index_value { base = i; step }))
  | Loop_expr.Temp (Loop_carrier.Float, t) -> (
      match Loop_temp.Map.find_opt t ctx.assigned with
      | Some vt -> Ok (V.Temp vt)
      | None -> Error Reason.Loop_carried)
  | Loop_expr.Array_get _ -> Error (Reason.Expression "local array read")
  | Loop_expr.I64_to_float _ -> Error (Reason.Expression "int64 value")

and conv_pred ctx (p : Loop_expr.pred) : (V.mask, Reason.t) result =
  match p with
  | Loop_bool.Value_eq (a, b) ->
      let* a = conv ctx a in
      let* b = conv ctx b in
      count ctx T.Op.Value_compare;
      Ok (V.Value_eq (a, b))
  | Loop_bool.Value_lt (a, b) ->
      let* a = conv ctx a in
      let* b = conv ctx b in
      count ctx T.Op.Value_compare;
      Ok (V.Value_lt (a, b))
  | Loop_bool.Pool_better (a, b) ->
      let* a = conv ctx a in
      let* b = conv ctx b in
      count ctx T.Op.Value_compare;
      Ok (V.Pool_better (a, b))
  | Loop_bool.Not p ->
      let* m = conv_pred ctx p in
      count ctx T.Op.Logic;
      Ok (V.Not m)
  | Loop_bool.Or (p, q) ->
      let* a = conv_pred ctx p in
      let* b = conv_pred ctx q in
      count ctx T.Op.Logic;
      Ok (V.Or (a, b))
  | Loop_bool.I64_eq _ | Loop_bool.I64_lt _ | Loop_bool.Index_eq _
  | Loop_bool.Index_lt _ | Loop_bool.Index_overflows _
  | Loop_bool.Out_of_range _ ->
      Error (Reason.Expression "index or int64 predicate")

let store ctx buffer offset (value : Loop_stored.t) =
  let* a = access ctx buffer offset in
  if a.V.Access.stride = 0 then Error Reason.Store_through_broadcast
  else
    let* v =
      match value with
      | Loop_stored.F32 e ->
          let* e = conv ctx e in
          Ok (V.F32 e)
      | Loop_stored.Bool e ->
          let* e = conv ctx e in
          Ok (V.Bool e)
      | Loop_stored.I64 _ -> Error (Reason.Expression "int64 store")
    in
    count ctx
      (match value with
      | Loop_stored.Bool _ -> T.Op.Bool_store
      | _ ->
          if a.V.Access.stride = 1 then T.Op.Contiguous_store
          else T.Op.Strided_store);
    Ok (V.Store { access = a; value = v })

let rec stmt ctx (s : Loop_stmt.t) : (V.stmt list, Reason.t) result =
  match s with
  | Loop_stmt.Assign (Loop_carrier.Float, t, e) ->
      let* e = conv ctx e in
      (* A temporary assigned again names the same vector temporary: an
         accumulator updated by an inner loop. *)
      let vt =
        match Loop_temp.Map.find_opt t ctx.assigned with
        | Some vt -> vt
        | None ->
            let vt = V.Temp.of_int ctx.next_temp in
            ctx.next_temp <- ctx.next_temp + 1;
            vt
      in
      ctx.assigned <- Loop_temp.Map.add t vt ctx.assigned;
      Ok [ V.Assign (vt, e) ]
  | Loop_stmt.Mark m -> Ok [ V.Mark m ]
  | Loop_stmt.Assign_index (t, i) ->
      if F.mentions ctx.var i then
        Error (Reason.Body_statement "index assignment from the loop variable")
      else Ok [ V.Index_assign (t, i) ]
  | Loop_stmt.Store_flat { buffer; offset; value } ->
      let* st = store ctx buffer offset value in
      Ok [ st ]
  | Loop_stmt.Store { buffer; coord; value } ->
      let* st = store ctx buffer (Loop_flat.offset buffer coord) value in
      Ok [ st ]
  | Loop_stmt.Assign (Loop_carrier.Int64, _, _) ->
      Error (Reason.Body_statement "int64 assignment")
  | Loop_stmt.Assign_index_of_i64 _ ->
      Error (Reason.Body_statement "index assignment")
  | Loop_stmt.Fail_if _ -> Error (Reason.Body_statement "failure check")
  | Loop_stmt.Alloc _ | Loop_stmt.Array_set _ ->
      Error (Reason.Body_statement "local array")
  | Loop_stmt.If _ -> Error (Reason.Body_statement "branch")
  | Loop_stmt.Charge_scan_update | Loop_stmt.Release_scan_state _
  | Loop_stmt.Reserve_scan_state _ | Loop_stmt.Reset_meter ->
      Error (Reason.Body_statement "scan meter")
  | Loop_stmt.For _ when not ctx.inner_ok ->
      Error (Reason.Body_statement "inner loop (target declines)")
  | Loop_stmt.For { var = iv; lo; hi; body } ->
      if F.mentions ctx.var lo || F.mentions ctx.var hi then
        Error (Reason.Body_statement "inner loop bounds from the loop variable")
      else
        let trips =
          match (lo, hi) with
          | Loop_index.Const a, Loop_index.Const b -> max 1 (b - a)
          | _ -> 16
        in
        let saved = ctx.assigned and saved_weight = ctx.weight in
        ctx.weight <- saved_weight * trips;
        let converted =
          List.fold_left
            (fun acc s ->
              let* done_ = acc in
              let* v = stmt ctx s in
              Ok (done_ @ v))
            (Ok []) body
        in
        (* What the inner loop assigned for the first time is not definitely
           assigned after it, which may run no iterations. *)
        ctx.assigned <-
          Loop_temp.Map.filter
            (fun t _ -> Loop_temp.Map.mem t saved)
            ctx.assigned;
        ctx.weight <- saved_weight;
        let* body = converted in
        Ok [ V.Inner { var = iv; lo; hi; body } ]

let const_bounds = function
  | Loop_index.Const lo, Loop_index.Const hi -> Some (lo, hi)
  | _ -> None

(* Attempts one leaf loop. [reads_total] counts every read of each temporary in
   the whole program, so a temporary the loop assigns that is read anywhere but
   inside it is live out. *)
let attempt ~(target : T.t) ~reads_total (loop : Loop_stmt.t) =
  match loop with
  | Loop_stmt.For { var; lo; hi; body } -> (
      match const_bounds (lo, hi) with
      | None -> (0, [], Error Reason.Non_constant_bounds)
      | Some (lo_n, hi_n) -> (
          let trips = max 0 (hi_n - lo_n) in
          let lanes = target.T.lanes in
          if trips < lanes then
            (trips, [], Error (Reason.Too_short { trips; lanes }))
          else
            let loop_temps =
              List.fold_left F.assigned_temps Loop_temp.Set.empty body
            in
            let ctx =
              {
                var;
                loop_temps;
                assigned = Loop_temp.Map.empty;
                next_temp = 0;
                ops = [];
                weight = 1;
                inner_ok = target.T.inner_loops;
              }
            in
            let converted =
              List.fold_left
                (fun acc s ->
                  let* done_ = acc in
                  let* v = stmt ctx s in
                  Ok (done_ @ v))
                (Ok []) body
            in
            let ops = List.fold_left (fun n (_, k) -> n + k) 0 ctx.ops in
            match converted with
            | Error r -> (trips, ctx.ops, Error r)
            | Ok vbody ->
                let inside =
                  List.fold_left F.stmt_temp_reads [] body
                  |> List.fold_left
                       (fun m t ->
                         Loop_temp.Map.update t
                           (function None -> Some 1 | Some n -> Some (n + 1))
                           m)
                       Loop_temp.Map.empty
                in
                let live_out =
                  Loop_temp.Set.exists
                    (fun t ->
                      let total =
                        Option.value ~default:0
                          (Loop_temp.Map.find_opt t reads_total)
                      in
                      let here =
                        Option.value ~default:0
                          (Loop_temp.Map.find_opt t inside)
                      in
                      total > here)
                    loop_temps
                in
                if live_out then (trips, ctx.ops, Error Reason.Live_out_temp)
                else if not (T.profitable target ctx.ops) then
                  (trips, ctx.ops, Error Reason.Unprofitable)
                else
                  let vloop =
                    { V.var; lo; hi; lanes; body = vbody; scalar = loop }
                  in
                  ignore ops;
                  (trips, ctx.ops, Ok vloop)))
  | _ -> invalid_arg "Loop_vectorize.attempt: not a loop"

let reads_of_program (p : Loop_program.t) =
  List.fold_left F.stmt_temp_reads [] p.Loop_program.body
  |> List.fold_left
       (fun m t ->
         Loop_temp.Map.update t
           (function None -> Some 1 | Some n -> Some (n + 1))
           m)
       Loop_temp.Map.empty

(* The innermost-loop iterations one execution of a loop covers. *)
let rec work (s : Loop_stmt.t) =
  match s with
  | Loop_stmt.For { lo; hi; body; _ } ->
      let trips =
        match (lo, hi) with
        | Loop_index.Const a, Loop_index.Const b -> Int64.of_int (max 0 (b - a))
        | _ -> 1L
      in
      let inner =
        List.fold_left
          (fun acc s -> if F.has_loop s then Int64.add acc (work s) else acc)
          0L body
      in
      Int64.mul trips (if Int64.equal inner 0L then 1L else inner)
  | Loop_stmt.If (_, a, b) ->
      List.fold_left (fun acc s -> Int64.add acc (work s)) 0L (a @ b)
  | _ -> 0L

let program ?(target = T.wasm128) (p : Loop_program.t) =
  let reads_total = reads_of_program p in
  (* Loops are tried innermost first: a loop whose body holds a vector loop is
     left as a scalar loop around it, and a loop whose own iterations are
     independent takes its inner loops with it (an inner reduction runs once, for
     all lanes in lockstep, in each lane's own order). A decision for a loop an
     outer vector loop absorbed is dropped: the outer decision covers it. *)
  let rec nodes ~enclosing stmts =
    List.fold_left
      (fun (acc, decs) (s : Loop_stmt.t) ->
        match s with
        | Loop_stmt.For { var; lo; hi; body } -> (
            let factor =
              match (lo, hi) with
              | Loop_index.Const a, Loop_index.Const b ->
                  Int64.of_int (max 1 (b - a))
              | _ -> 1L
            in
            let children, child_decs =
              nodes ~enclosing:(Int64.mul enclosing factor) body
            in
            if V.count_vector_loops children > 0 then
              ( acc @ [ V.Loop { var; lo; hi; body = children } ],
                decs @ child_decs )
            else
              let trips, ops, outcome = attempt ~target ~reads_total s in
              let n_ops = List.fold_left (fun n (_, k) -> n + k) 0 ops in
              let record outcome =
                {
                  Decision.trips;
                  executions = enclosing;
                  work = work s;
                  ops = n_ops;
                  outcome;
                }
              in
              match outcome with
              | Ok vloop -> (
                  match
                    Loop_vector_check.program
                      { V.scalar = p; body = [ V.Vector vloop ] }
                  with
                  | Ok () ->
                      ( acc @ [ V.Vector vloop ],
                        decs @ [ record Decision.Vectorized ] )
                  | Error _ ->
                      ( acc @ [ V.Scalar s ],
                        decs @ child_decs
                        @ [ record (Decision.Kept_scalar Reason.Loop_carried) ]
                      ))
              | Error r ->
                  ( acc @ [ V.Scalar s ],
                    decs @ child_decs @ [ record (Decision.Kept_scalar r) ] ))
        | Loop_stmt.If (c, a, b) ->
            let na, da = nodes ~enclosing a in
            let nb, db = nodes ~enclosing b in
            (acc @ [ V.If (c, na, nb) ], decs @ da @ db)
        | s -> (acc @ [ V.Scalar s ], decs))
      ([], []) stmts
  in
  let body, decisions = nodes ~enclosing:1L p.Loop_program.body in
  ({ V.scalar = p; body }, decisions)

let tally (r : report) =
  let add acc name n w =
    let c, x = Option.value ~default:(0, 0L) (List.assoc_opt name acc) in
    (name, (c + n, Int64.add x w)) :: List.remove_assoc name acc
  in
  List.fold_left
    (fun acc (d : Decision.t) ->
      let weight =
        Int64.mul (Int64.of_int d.Decision.trips) d.Decision.executions
      in
      let name =
        match d.Decision.outcome with
        | Decision.Vectorized -> "vectorized"
        | Decision.Kept_scalar r -> Reason.name r
      in
      add acc name 1 weight)
    [] r
  |> List.sort (fun (a, _) (b, _) -> compare a b)
