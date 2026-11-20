(* The runtime mode, shared with the other native routes. *)

include Machine_rivet_common.Rivet_runtime

let admit mode artifact =
  Result.map_error
    (fun h -> Rivet_x64_refusal.Helper h)
    (Machine_rivet_common.Rivet_runtime.admit mode artifact)
