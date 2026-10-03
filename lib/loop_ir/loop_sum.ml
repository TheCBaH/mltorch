let rec expand (s : Loop_stmt.t) : Loop_stmt.t list =
  match s with
  | Loop_stmt.Reduce_sum { var; lo; hi; acc; seed; body; term; at = _ } ->
      [
        Loop_stmt.Assign (Loop_carrier.Float, acc, Loop_expr.Const seed);
        Loop_stmt.For
          {
            var;
            lo;
            hi;
            body =
              (Loop_stmt.Mark Loop_mark.Reduction :: block body)
              @ [
                  Loop_stmt.Assign
                    ( Loop_carrier.Float,
                      acc,
                      Loop_expr.Binary
                        ( Expr.Value.Add,
                          Loop_expr.Temp (Loop_carrier.Float, acc),
                          term ) );
                ];
          };
      ]
  | Loop_stmt.For f -> [ Loop_stmt.For { f with body = block f.body } ]
  | Loop_stmt.If (p, yes, no) -> [ Loop_stmt.If (p, block yes, block no) ]
  | s -> [ s ]

and block l = List.concat_map expand l

let program (p : Loop_program.t) = { p with Loop_program.body = block p.body }

let rec count_stmt acc (s : Loop_stmt.t) =
  match s with
  | Loop_stmt.Reduce_sum { body; _ } -> List.fold_left count_stmt (acc + 1) body
  | Loop_stmt.For { body; _ } -> List.fold_left count_stmt acc body
  | Loop_stmt.If (_, yes, no) ->
      List.fold_left count_stmt (List.fold_left count_stmt acc yes) no
  | _ -> acc

let count (p : Loop_program.t) = List.fold_left count_stmt 0 p.Loop_program.body

type error =
  [ `Accumulator_assigned_in_body of Loop_temp.t
  | `Accumulator_read_by_term of Loop_temp.t
  | `Variable_reused of Loop_var.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Accumulator_assigned_in_body t ->
      Format.fprintf ppf "a sum's body assigns its accumulator %d"
        (Loop_temp.to_int t)
  | `Accumulator_read_by_term t ->
      Format.fprintf ppf "a sum's term or body reads its accumulator %d"
        (Loop_temp.to_int t)
  | `Variable_reused v ->
      Format.fprintf ppf "a sum reuses the variable of an enclosing loop %d"
        (Loop_var.to_int v)

module F = Loop_vector_facts

(* ---- recovery ---------------------------------------------------------------

   The optimizer reshapes a program after lowering (guards proved away, indices
   collapsed, invariant code hoisted), and its passes read expanded sums. The
   planner wants the optimized bodies, so it recovers each sum from the shape the
   expansion defines: a [For] whose body opens with a [Reduction] mark and ends
   with [acc <- acc + e], where [acc] is mentioned nowhere else in the loop and
   was seeded by a constant assignment right before it. The check is on the
   statements themselves, so it is
   sound whatever the program's history, and {!expand} of the result is the input
   program exactly. *)

let mentions_temp acc (s : Loop_stmt.t) =
  Loop_temp.Set.mem acc (F.assigned_temps Loop_temp.Set.empty s)
  || List.exists (Loop_temp.equal acc) (F.stmt_temp_reads [] s)

(* [Some (body, term)] when the loop body is a sum of [acc]. *)
let accumulate acc body =
  match body with
  | Loop_stmt.Mark Loop_mark.Reduction :: rest -> (
      match List.rev rest with
      | Loop_stmt.Assign
          ( Loop_carrier.Float,
            acc',
            Loop_expr.Binary
              (Expr.Value.Add, Loop_expr.Temp (Loop_carrier.Float, acc''), term)
          )
        :: before_rev
        when Loop_temp.equal acc acc' && Loop_temp.equal acc acc'' ->
          let before = List.rev before_rev in
          if
            (not (List.exists (mentions_temp acc) before))
            && not (List.exists (Loop_temp.equal acc) (F.expr_temps term))
          then Some (before, term)
          else None
      | _ -> None)
  | _ -> None

let accumulator_of body =
  match body with
  | Loop_stmt.Mark Loop_mark.Reduction :: rest -> (
      match List.rev rest with
      | Loop_stmt.Assign (Loop_carrier.Float, acc, _) :: _ -> Some acc
      | _ -> None)
  | _ -> None

let rec recover_block (block : Loop_stmt.t list) : Loop_stmt.t list =
  (* Walk left to right keeping the statements already passed, most recent
     first, so a seed can be found behind the loop it feeds. *)
  let rec go passed = function
    | [] -> List.rev passed
    | Loop_stmt.For { var; lo; hi; body } :: rest -> (
        let body = recover_block body in
        let plain = Loop_stmt.For { var; lo; hi; body } in
        match accumulator_of body with
        | None -> go (plain :: passed) rest
        | Some acc -> (
            match accumulate acc body with
            | None -> go (plain :: passed) rest
            | Some (before, term) -> (
                (* the seed is the statement just before the loop, so expanding
                   the node gives back this very program *)
                match passed with
                | Loop_stmt.Assign (Loop_carrier.Float, a, Loop_expr.Const seed)
                  :: passed'
                  when Loop_temp.equal a acc ->
                    go
                      (Loop_stmt.Reduce_sum
                         {
                           var;
                           lo;
                           hi;
                           acc;
                           seed;
                           body = before;
                           term;
                           at = None;
                         }
                      :: passed')
                      rest
                | _ -> go (plain :: passed) rest)))
    | Loop_stmt.If (p, yes, no) :: rest ->
        go
          (Loop_stmt.If (p, recover_block yes, recover_block no) :: passed)
          rest
    | s :: rest -> go (s :: passed) rest
  in
  go [] block

let recover (p : Loop_program.t) =
  { p with Loop_program.body = recover_block p.Loop_program.body }

let check (p : Loop_program.t) =
  Err.Escape.with_escape (fun esc ->
      let rec stmt ~enclosing (s : Loop_stmt.t) =
        match s with
        | Loop_stmt.Reduce_sum { var; body; term; acc; _ } ->
            if List.exists (Loop_var.equal var) enclosing then
              Err.Escape.throw esc (`Variable_reused var);
            if
              Loop_temp.Set.mem acc
                (List.fold_left F.assigned_temps Loop_temp.Set.empty body)
            then Err.Escape.throw esc (`Accumulator_assigned_in_body acc);
            let reads =
              F.expr_temps term @ List.fold_left F.stmt_temp_reads [] body
            in
            if List.exists (Loop_temp.equal acc) reads then
              Err.Escape.throw esc (`Accumulator_read_by_term acc);
            List.iter (stmt ~enclosing:(var :: enclosing)) body
        | Loop_stmt.For { var; body; _ } ->
            List.iter (stmt ~enclosing:(var :: enclosing)) body
        | Loop_stmt.If (_, yes, no) ->
            List.iter (stmt ~enclosing) yes;
            List.iter (stmt ~enclosing) no
        | _ -> ()
      in
      List.iter (stmt ~enclosing:[]) p.Loop_program.body)
