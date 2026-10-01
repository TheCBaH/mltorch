(* See region_executor.mli. *)

type t =
  ?counters:Region_execution.counters ->
  dst:Tensor.packed ->
  Region_execution.lowered ->
  env:Expr.Eval.Env.t ->
  bindings:Tensor.packed Tensor_id.Map.t ->
  (Tensor.packed, Region_eval.error) Err.t

type group =
  ?counters:Region_execution.counters ->
  dsts:(Region_group.Ordinal.t * Tensor.packed) list ->
  Region_execution.lowered_group ->
  env:Expr.Eval.Env.t ->
  bindings:Tensor.packed Tensor_id.Map.t ->
  ((Region_group.Ordinal.t * Tensor.packed) list, Region_eval.error) Err.t

let default : t =
 fun ?counters ~dst lowered ~env ~bindings:_ ->
  Result.map
    (fun () -> dst)
    (Region_execution.materialize_into ?counters ~dst lowered ~env)

let default_group : group =
 fun ?counters ~dsts lowered_group ~env ~bindings:_ ->
  Result.map
    (fun () -> dsts)
    (Region_execution.materialize_group_into ?counters ~dsts lowered_group ~env)
