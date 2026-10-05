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
}

let facts : (string, facts) Hashtbl.t = Hashtbl.create 8

let facts_of config =
  match Hashtbl.find_opt facts config with
  | Some f -> f
  | None ->
      let f = { plans = 0; f32 = 0; vector_loops = 0; sums = 0; fused = 0 } in
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
    f.plans f.f32 f.vector_loops f.sums f.fused
