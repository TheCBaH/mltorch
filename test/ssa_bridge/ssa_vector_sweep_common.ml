open Ssa_bridge
open Ssa_ir

(* Vectorized execution of every walked plan against the reference evaluator:
   the plan is lowered, optimized as the pipeline does, vectorized for a target,
   and run. Two targets: the cost model's own choice, and one that takes every
   legal loop (the paths the cost model would decline). The decisions of every
   plan are tallied, so the report says how much of the sweep was vectorized and
   what kept the rest scalar. *)

let alias = Ssa_effects.Distinct_buffers

let before =
  [ Ssa_opt.simplify; Ssa_opt.guards; Ssa_opt.simplify; Ssa_opt.hoist ~alias ]

let after =
  [
    Ssa_opt.block ~alias ~group:Ssa_opt_block.Auto;
    Ssa_opt.hoist ~alias;
    Ssa_opt.share ~alias;
    Ssa_opt.simplify;
  ]

let reports : (string, Ssa_vectorize.report) Hashtbl.t = Hashtbl.create 4

let prepare ~config ~target ?(expand = false) program =
  let p = fst (Ssa_opt.run ~alias ~passes:before program) in
  let q, report = Ssa_vectorize.program ~alias ~target p in
  Hashtbl.replace reports config
    (report @ Option.value ~default:[] (Hashtbl.find_opt reports config));
  let q = fst (Ssa_opt.run ~alias ~passes:after q) in
  if expand then Ssa_vec_expand.program q else q

let sweep ~config ~target ?expand ~shard () =
  Ssa_config_sweep.sweep ~config
    ~check:(fun plan ~bind ->
      Ssa_check.run ~prepare:(prepare ~config ~target ?expand) plan ~bind)
    ~shard ~shards:4

let report ~config =
  Ssa_config_sweep.report ~config;
  let tally =
    Ssa_vectorize.tally
      (Option.value ~default:[] (Hashtbl.find_opt reports config))
  in
  List.iter
    (fun (name, (loops, work)) ->
      Fmt.pr "  %-34s loops=%d work=%Ld@." name loops work)
    tally
