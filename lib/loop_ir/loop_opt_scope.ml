(* What a pass that re-analyses an already-lowered program rebuilds of the
   [Loop_range.Env.t] the lowering had while it built it: loop variables'
   ranges from their bounds, and index temporaries' ranges from their one
   assignment. *)

(* The range a loop variable can take across the loop's iterations, as a
   conservative enclosure: [lo]'s smallest possible value to [hi]'s largest
   possible value minus one. An empty loop (no iteration ever runs) is folded
   into the single point [lo] here, the same convention [Loop_range.span]
   uses for a compile-time-empty loop -- safe, since nothing inside a loop
   that never runs is ever evaluated, whatever this range says. *)
let var_range env (lo : Loop_index.t) (hi : Loop_index.t) : Loop_range.t =
  let lo_r = Loop_range.of_index env lo and hi_r = Loop_range.of_index env hi in
  let hi_incl = Int64.sub hi_r.Loop_range.hi 1L in
  {
    Loop_range.lo = lo_r.Loop_range.lo;
    hi = Stdlib.max lo_r.Loop_range.lo hi_incl;
  }

(* How many statements assign each index temporary. An argmax's [best_i] and a
   max-pool's [best_ix] are seeded before their loop and reassigned inside it,
   so the range at one assignment says nothing about a read that may see
   another: only a temporary assigned exactly once may be recorded, and every
   other one stays [unbounded]. *)
let rec count_index_assigns counts (stmts : Loop_stmt.t list) =
  List.fold_left
    (fun counts (stmt : Loop_stmt.t) ->
      match stmt with
      | Loop_stmt.Assign_index (t, _) | Loop_stmt.Assign_index_of_i64 (t, _) ->
          Loop_temp.Map.update t
            (fun n -> Some (1 + Option.value ~default:0 n))
            counts
      | Loop_stmt.For { body; _ } -> count_index_assigns counts body
      | Loop_stmt.If (_, a, b) ->
          count_index_assigns (count_index_assigns counts a) b
      | _ -> counts)
    counts stmts

let single_assignment (body : Loop_stmt.t list) : Loop_temp.t -> bool =
  let counts = count_index_assigns Loop_temp.Map.empty body in
  fun t -> Loop_temp.Map.find_opt t counts = Some 1
