open Ssa_bridge
open Ssa_ir

(* Every walked plan through the binary32 vector plan ({!Ssa_plan.fp32_vector})
   for neon128 and wasm128, each priced and forced (every legal loop taken),
   judged three ways: against its own oracle bit for bit and against the Loop
   interpreter at binary32 (as {!Ssa_check.run_planned} judges an
   [Fp32_ordered] plan), and against the scalar [Fp32_ordered] plan of the same
   kernel bit for bit, outputs and failures alike. A plan agrees only when all
   four targets do. Where the scalar plan itself disagrees with the Loop
   interpreter and the vector plan equals the scalar plan, the vector plan is
   not the cause: that is tallied as inherited, not as a disagreement.
   Tallied too: vector loops, and plans that hold a binary64 vector (none
   may). The references are run once per kernel plan; a vector plan that is
   the agreeing scalar plan unchanged takes its verdict, and where both plans
   of one precision agree with their reference the scalar comparison follows,
   so neither is run again. *)

type facts = {
  mutable plans : int;
  mutable vector_loops : int;
  mutable forced_loops : int;
  mutable wide_vectors : int;
  mutable inherited : int;
}

let facts =
  {
    plans = 0;
    vector_loops = 0;
    forced_loops = 0;
    wide_vectors = 0;
    inherited = 0;
  }

let targets =
  [
    (false, Ssa_target.neon128);
    (false, Ssa_target.wasm128);
    (true, Ssa_target.forced Ssa_target.neon128);
    (true, Ssa_target.forced Ssa_target.wasm128);
  ]

let resolve ~target ~alias p =
  match Ssa_plan.fp32_vector ~alias ~target p with
  | Ok t -> t
  | Error r -> Fmt.invalid_arg "not admitted: %a" Ssa_numerics.Refusal.pp r

let vector_loops (r : Ssa_plan.t) =
  List.length
    (List.filter
       (fun (d : Ssa_vectorize.Decision.t) ->
         match d.Ssa_vectorize.Decision.outcome with
         | Ssa_vectorize.Decision.Vectorized -> true
         | Ssa_vectorize.Decision.Kept_scalar _ -> false)
       r.Ssa_plan.vectorized)

let agrees = function
  | Ssa_check.Agree | Ssa_check.Agree_on_failure _ -> true
  | Ssa_check.Disagree _ | Ssa_check.Not_admitted _ | Ssa_check.Refused _ ->
      false

let check plan ~bind =
  let reference = Ssa_check.Reference.make plan ~bind in
  let scalar_verdict, scalar =
    Ssa_check.run_planned ~reference ~numerics:Ssa_numerics.Fp32_ordered
      ~target:Ssa_target.scalar plan ~bind
  in
  let exec (p : Ssa_plan.t) =
    Err.map_error
      (fun e ->
        match e with
        | #Ssa_interp.failure as f -> (f :> Kernel_eval.error)
        | `Binding_mismatch _ | `Unbound_input _ | `Cfg_lowering _
        | `Invalid_cfg _ | `Invalid_program _ ->
            invalid_arg "the plan does not run")
      (Ssa_lower.Ssa_exec.run plan p.Ssa_plan.program ~bind)
  in
  let scalar = Option.map (fun s -> (s, lazy (exec s))) scalar in
  let lowered = Err.payload (Ssa_lower.Ssa_lower_plan.lower plan) in
  List.fold_left
    (fun acc (forced, target) ->
      let run resolve =
        Ssa_check.run_planned ~resolve ~reference
          ~numerics:Ssa_numerics.Fp32_ordered ~target plan ~bind
      in
      let verdict, resolved =
        match (lowered, scalar) with
        | Ok program, Some (s, _) when agrees scalar_verdict ->
            let r =
              resolve ~target ~alias:Ssa_effects.Distinct_buffers program
            in
            (* the scalar plan itself, so its verdict *)
            if
              Stdlib.( = ) r.Ssa_plan.program s.Ssa_plan.program
              && r.Ssa_plan.precision = s.Ssa_plan.precision
            then (scalar_verdict, Some r)
            else run (fun ~alias:_ _ -> r)
        | _ -> run (resolve ~target)
      in
      let verdict =
        match (resolved, scalar) with
        | Some r, Some (s, scalar_run) -> (
            facts.plans <- facts.plans + 1;
            if forced then
              facts.forced_loops <- facts.forced_loops + vector_loops r
            else facts.vector_loops <- facts.vector_loops + vector_loops r;
            if
              List.exists
                (fun ((v : Ssa_value.t), _) ->
                  match v.Ssa_value.ty with
                  | Ssa_type.Vec _ -> true
                  | Ssa_type.Effect | Ssa_type.Local | Ssa_type.Mask _
                  | Ssa_type.Scalar _ ->
                      false)
                (Ssa_numerics.wide_values r.Ssa_plan.program)
            then facts.wide_vectors <- facts.wide_vectors + 1;
            if
              agrees verdict && agrees scalar_verdict
              && r.Ssa_plan.precision = s.Ssa_plan.precision
            then verdict
            else
              match
                Ssa_check.compare_kernel ~reference:(Lazy.force scalar_run)
                  ~actual:(exec r)
              with
              | (Ssa_check.Agree | Ssa_check.Agree_on_failure _) as same ->
                  if agrees verdict || agrees scalar_verdict then verdict
                  else (
                    facts.inherited <- facts.inherited + 1;
                    same)
              | ( Ssa_check.Disagree _ | Ssa_check.Not_admitted _
                | Ssa_check.Refused _ ) as v ->
                  v)
        | _ -> verdict
      in
      if agrees acc then verdict else acc)
    Ssa_check.Agree targets

let sweep ~shard =
  Ssa_config_sweep.sweep ~config:"fp32-vector" ~check ~shard ~shards:4

let report () =
  Ssa_config_sweep.report ~config:"fp32-vector";
  Fmt.pr
    "  plans=%d vector-loops=%d forced-vector-loops=%d wide-vectors=%d \
     inherited-from-scalar=%d@."
    facts.plans facts.vector_loops facts.forced_loops facts.wide_vectors
    facts.inherited
