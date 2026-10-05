open Ssa_bridge

(* Binary32 execution of every walked plan against the Loop interpreter at
   binary32. *)
let config = "f32"
let check plan ~bind = Ssa_check.run_f32 plan ~bind
let sweep ~shard = Ssa_config_sweep.sweep ~config ~check ~shard ~shards:4
let report () = Ssa_config_sweep.report ~config
