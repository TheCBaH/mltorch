open Ssa_bridge
open Ssa_ir

(* Every walked plan resolved under a numerical policy for a target and judged as
   {!Ssa_check.run_planned} says: against its own oracle bit for bit, then
   against the reference that policy is defined to match. The plans' own facts
   are tallied, so the report says how many kernels went binary32, how many
   loops were vectorized, how many sums scheduled and how many multiply-adds
   fused: a sweep that never left binary64 would pass for the wrong reason. *)

type facts = {
  mutable plans : int;
  mutable f32 : int;
  mutable vector_loops : int;
  mutable sums : int;
  mutable fused : int;
  mutable same_precision : int;
  mutable same_loops : int;
  mutable more_loops : int;
  mutable fewer_loops : int;
  mutable same_blocked : int;
}

let facts : (string, facts) Hashtbl.t = Hashtbl.create 8

let facts_of config =
  match Hashtbl.find_opt facts config with
  | Some f -> f
  | None ->
      let f =
        {
          plans = 0;
          f32 = 0;
          vector_loops = 0;
          sums = 0;
          fused = 0;
          same_precision = 0;
          same_loops = 0;
          more_loops = 0;
          fewer_loops = 0;
          same_blocked = 0;
        }
      in
      Hashtbl.add facts config f;
      f

let check ~config ~numerics ~target plan ~bind =
  let verdict, resolved = Ssa_check.run_planned ~numerics ~target plan ~bind in
  (match resolved with
  | None -> ()
  | Some (r : Ssa_plan.t) ->
      let f = facts_of config in
      f.plans <- f.plans + 1;
      if r.Ssa_plan.precision = Ssa_numerics.Precision.F32 then
        f.f32 <- f.f32 + 1;
      f.vector_loops <-
        f.vector_loops
        + List.length
            (List.filter
               (fun (d : Ssa_vectorize.Decision.t) ->
                 d.Ssa_vectorize.Decision.outcome
                 = Ssa_vectorize.Decision.Vectorized)
               r.Ssa_plan.vectorized);
      f.sums <-
        f.sums
        + List.length
            (List.filter
               (fun (d : Ssa_vector_sum.Decision.t) ->
                 match d.Ssa_vector_sum.Decision.outcome with
                 | Ssa_vector_sum.Decision.Scheduled _ -> true
                 | Ssa_vector_sum.Decision.Kept_sequential _ -> false)
               r.Ssa_plan.sums);
      f.fused <- f.fused + r.Ssa_plan.contracted);
  (match Ssa_check.compare_plans ~numerics ~target plan with
  | None -> ()
  | Some c ->
      let f = facts_of config in
      let open Ssa_check.Plan_comparison in
      if c.ssa_f32 = c.loop_f32 then f.same_precision <- f.same_precision + 1;
      if c.ssa_loops = c.loop_loops then f.same_loops <- f.same_loops + 1
      else if c.ssa_loops > c.loop_loops then f.more_loops <- f.more_loops + 1
      else f.fewer_loops <- f.fewer_loops + 1;
      if c.ssa_blocked = c.loop_blocked then
        f.same_blocked <- f.same_blocked + 1);
  verdict

let sweep ~config ~numerics ~target ~shard =
  Ssa_config_sweep.sweep ~config
    ~check:(check ~config ~numerics ~target)
    ~shard ~shards:4

let report ~config =
  Ssa_config_sweep.report ~config;
  let f = facts_of config in
  Fmt.pr
    "  plans=%d binary32=%d vector-loops=%d scheduled-sums=%d \
     fused-multiply-adds=%d@."
    f.plans f.f32 f.vector_loops f.sums f.fused;
  Fmt.pr
    "  against the Loop plan: same-precision=%d same-vector-loops=%d \
     more-vector-loops=%d fewer-vector-loops=%d same-blocked-rows=%d@."
    f.same_precision f.same_loops f.more_loops f.fewer_loops f.same_blocked
