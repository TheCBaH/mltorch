open Ssa_bridge
open Ssa_ir

(* The graph interpreter against the reference on every walked plan: the plan
   is lowered to the structured program, prepared (as lowered, optimized, or
   optimized and vectorized for a target), converted to a control-flow graph
   and run there. The graph executor shares every operation with the structured
   interpreter, so a disagreement is a disagreement about control flow:
   loops, zero-trip exits, branches, carried values, ordered sums and the
   effect chain. *)

let alias = Ssa_effects.Distinct_buffers
let optimize p = fst (Ssa_opt.run ~alias p)

let sweep ~config ~prepare ~shard =
  Ssa_config_sweep.sweep ~config
    ~check:(fun plan ~bind ->
      Ssa_check.run ~prepare ~engine:Ssa_lower.Ssa_exec.Cfg plan ~bind)
    ~shard ~shards:4

let vectorize ~config p =
  Ssa_vector_sweep_common.prepare ~config
    ~target:(Ssa_target.forced Ssa_target.neon128)
    p

let run ~shard =
  sweep ~config:"cfg-lowered" ~prepare:Fun.id ~shard;
  sweep ~config:"cfg-optimized" ~prepare:optimize ~shard;
  sweep ~config:"cfg-vectorized"
    ~prepare:(vectorize ~config:"cfg-vectorized")
    ~shard

let report () =
  List.iter
    (fun config ->
      Fmt.pr "%s@." config;
      Ssa_config_sweep.report ~config)
    [ "cfg-lowered"; "cfg-optimized"; "cfg-vectorized" ]
