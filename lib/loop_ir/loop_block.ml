module V = Loop_vector

(* Row [r] of a block of [rows]: the loop variable [var] becomes
   [lo + rows * var + r], so a blocked iteration covers [rows] consecutive
   iterations of the original loop. *)
let leaf ~var ~rows ~lo ~row = function
  | Loop_index.Var v when Loop_var.equal v var ->
      Loop_index.Add
        (Loop_index.Scale (rows, Loop_index.Var v), Loop_index.Const (lo + row))
  | i -> i

let rec max_temp acc (s : V.stmt) =
  match s with
  | V.Assign (t, _) -> max acc (V.Temp.to_int t)
  | V.Inner { body; _ } -> List.fold_left max_temp acc body
  | V.Index_assign _ | V.Mark _ | V.Store _ -> acc

let rec has_index_assign (s : V.stmt) =
  match s with
  | V.Index_assign _ -> true
  | V.Inner { body; _ } -> List.exists has_index_assign body
  | V.Assign _ | V.Mark _ | V.Store _ -> false

(* The accumulators a vector body carries: temporaries an inner loop assigns
   from their own previous value. A body with none has no latency chain for
   other rows to overlap with. *)
let carried body =
  let rec reads t (e : V.t) =
    match e with
    | V.Temp u -> V.Temp.equal t u
    | V.Binary (_, a, b) | V.Float_max (a, b) -> reads t a || reads t b
    | V.Fma (a, b, c) -> reads t a || reads t b || reads t c
    | V.Round_f32 a | V.Unary (_, a) -> reads t a
    | V.Select (m, a, b) -> mask t m || reads t a || reads t b
    | V.Const _ | V.Index_value _ | V.Load _ | V.Splat _ -> false
  and mask t (m : V.mask) =
    match m with
    | V.Not m -> mask t m
    | V.Or (a, b) -> mask t a || mask t b
    | V.Pool_better (a, b) | V.Value_eq (a, b) | V.Value_lt (a, b) ->
        reads t a || reads t b
  in
  let rec go inside acc (s : V.stmt) =
    match s with
    | V.Assign (t, e) when inside && reads t e && not (List.mem t acc) ->
        t :: acc
    | V.Inner { body; _ } -> List.fold_left (go true) acc body
    | V.Assign _ | V.Index_assign _ | V.Mark _ | V.Store _ -> acc
  in
  List.length (List.fold_left (go false) [] body)

(* One row's copy of a vector loop's body: its variable shifted, its vector
   temporaries renamed apart from the other rows'. *)
let copy_stmt ~sub ~expr ~temp (s : V.stmt) =
  let rec vexpr (e : V.t) : V.t =
    match e with
    | V.Binary (op, a, b) -> V.Binary (op, vexpr a, vexpr b)
    | V.Const _ -> e
    | V.Float_max (a, b) -> V.Float_max (vexpr a, vexpr b)
    | V.Fma (a, b, c) -> V.Fma (vexpr a, vexpr b, vexpr c)
    | V.Index_value { base; step } -> V.Index_value { base = sub base; step }
    | V.Load a -> V.Load (access a)
    | V.Round_f32 a -> V.Round_f32 (vexpr a)
    | V.Select (m, a, b) -> V.Select (mask m, vexpr a, vexpr b)
    | V.Splat s -> V.Splat (expr s)
    | V.Temp t -> V.Temp (temp t)
    | V.Unary (op, a) -> V.Unary (op, vexpr a)
  and mask (m : V.mask) : V.mask =
    match m with
    | V.Not m -> V.Not (mask m)
    | V.Or (a, b) -> V.Or (mask a, mask b)
    | V.Pool_better (a, b) -> V.Pool_better (vexpr a, vexpr b)
    | V.Value_eq (a, b) -> V.Value_eq (vexpr a, vexpr b)
    | V.Value_lt (a, b) -> V.Value_lt (vexpr a, vexpr b)
  and access (a : V.Access.t) =
    { a with V.Access.offset = sub a.V.Access.offset }
  and stmt (s : V.stmt) : V.stmt =
    match s with
    | V.Assign (t, e) -> V.Assign (temp t, vexpr e)
    | V.Index_assign _ -> invalid_arg "Loop_block: an index assignment"
    | V.Inner { var; lo; hi; body } ->
        V.Inner { var; lo = sub lo; hi = sub hi; body = List.map stmt body }
    | V.Mark _ -> s
    | V.Store { access = a; value } ->
        V.Store
          {
            access = access a;
            value =
              (match value with
              | V.Bool e -> V.Bool (vexpr e)
              | V.F32 e -> V.F32 (vexpr e));
          }
  in
  stmt s

(* The rows' copies run in lockstep: an inner loop every row has, with the same
   bounds, becomes one loop whose body holds each row's body in turn, so a load
   the rows share is issued once per iteration and the rows' independent
   accumulators overlap each other's latency. Rows are independent (the
   verifier proves it), so each row's own operation order is unchanged. *)
let rec jam (copies : V.stmt list list) : V.stmt list option =
  match copies with
  | [] -> Some []
  | first :: _ ->
      let ( let* ) = Option.bind in
      let rec position j acc =
        if j = List.length first then Some (List.rev acc)
        else
          let column = List.map (fun c -> List.nth c j) copies in
          let* stmts =
            match column with
            | V.Inner { var; lo; hi; _ } :: _ ->
                let same = function
                  | V.Inner i -> i.var = var && i.lo = lo && i.hi = hi
                  | V.Assign _ | V.Index_assign _ | V.Mark _ | V.Store _ ->
                      false
                in
                if List.for_all same column then
                  let* body =
                    jam
                      (List.map
                         (function
                           | V.Inner i -> i.body
                           | V.Assign _ | V.Index_assign _ | V.Mark _
                           | V.Store _ ->
                               [])
                         column)
                  in
                  Some [ V.Inner { var; lo; hi; body } ]
                else None
            | _ -> Some column
          in
          position (j + 1) (List.rev_append stmts acc)
      in
      if List.for_all (fun c -> List.length c = List.length first) copies then
        position 0 []
      else None

let block_loop ~target (p : V.program) ~var ~lo ~hi (l : V.loop) =
  let ( let* ) = Option.bind in
  let rows = hi - lo in
  let accs = carried l.V.body in
  let factor = min rows (max 2 (target.Loop_target.row_block / max 1 accs)) in
  if
    target.Loop_target.row_block < 2
    || factor < 2 || accs = 0
    || List.exists has_index_assign l.V.body
  then None
  else
    let groups = rows / factor in
    let stride = 1 + List.fold_left max_temp 0 l.V.body in
    let row_fns r =
      let sub = Loop_index_map.index ~f:(leaf ~var ~rows:factor ~lo ~row:r) in
      let expr e =
        Loop_index_map.expr ~f:(leaf ~var ~rows:factor ~lo ~row:r) e
      in
      let temp t = V.Temp.of_int (V.Temp.to_int t + (r * stride)) in
      (sub, expr, temp)
    in
    let copies =
      List.init factor (fun r ->
          let sub, expr, temp = row_fns r in
          List.map (copy_stmt ~sub ~expr ~temp) l.V.body)
    in
    let* body = jam copies in
    let* scalar =
      match l.V.scalar with
      | Loop_stmt.For f ->
          Some
            (Loop_stmt.For
               {
                 f with
                 body =
                   List.concat
                     (List.init factor (fun r ->
                          Loop_index_map.stmts
                            ~f:(leaf ~var ~rows:factor ~lo ~row:r)
                            f.body));
               })
      | _ -> None
    in
    let blocked = { l with V.body; scalar } in
    match
      Loop_vector_check.program { p with V.body = [ V.Vector blocked ] }
    with
    | Error _ -> None
    | Ok () ->
        let main =
          V.Loop
            {
              var;
              lo = Loop_index.Const 0;
              hi = Loop_index.Const groups;
              body = [ V.Vector blocked ];
            }
        in
        let rest = rows - (groups * factor) in
        Some
          (if rest = 0 then [ main ]
           else
             [
               main;
               V.Loop
                 {
                   var;
                   lo = Loop_index.Const (lo + (groups * factor));
                   hi = Loop_index.Const hi;
                   body = [ V.Vector l ];
                 };
             ])

let program ~target (p : V.program) =
  let count = ref 0 in
  let rec node (n : V.node) : V.node list =
    match n with
    | V.If (c, a, b) ->
        [ V.If (c, List.concat_map node a, List.concat_map node b) ]
    | V.Loop
        {
          var;
          lo = Loop_index.Const lo;
          hi = Loop_index.Const hi;
          body = [ V.Vector l ];
        } as n -> (
        match block_loop ~target p ~var ~lo ~hi l with
        | Some nodes ->
            incr count;
            nodes
        | None -> [ n ])
    | V.Loop r -> [ V.Loop { r with body = List.concat_map node r.body } ]
    | V.Reduction _ | V.Scalar _ | V.Vector _ -> [ n ]
  in
  let body = List.concat_map node p.V.body in
  ({ p with V.body }, !count)
