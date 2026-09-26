type t =
  ?counters:Region_execution.counters ->
  Region_execution.lowered ->
  env:Expr.Eval.Env.t ->
  bindings:Tensor.packed Tensor_id.Map.t ->
  (Tensor.packed, Region_eval.error) Err.t

type group =
  ?counters:Region_execution.counters ->
  Region_execution.lowered_group ->
  env:Expr.Eval.Env.t ->
  bindings:Tensor.packed Tensor_id.Map.t ->
  selected:Region_group.Ordinal.t list ->
  ((Region_group.Ordinal.t * Tensor.packed) list, Region_eval.error) Err.t

let default : t =
 fun ?counters lowered ~env ~bindings:_ ->
  Region_execution.materialize ?counters lowered ~env

let default_group : group =
 fun ?counters lowered_group ~env ~bindings:_ ~selected ->
  Region_execution.materialize_group ?counters lowered_group ~env ~selected
