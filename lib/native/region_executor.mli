(* The pluggable seam between [Eval_direct]'s region-authored ops (RmsNorm,
   LayerNorm, Softmax, Sdpa, the Lstm group) and however a lowered Region
   program actually runs. [default]/[default_group] wrap
   [Region_execution.materialize]/[materialize_group] (behaviorally, not
   syntactically, identical -- [~bindings] is the one parameter they ignore),
   so an [Eval_direct.run] caller that never passes [?region_executor] gets
   the same result as before this seam existed.

   [~bindings] carries every source [lowered]'s program reads as a real,
   already-materialized [Tensor.packed] -- [region_result]'s own
   [operand_env] merged with its synthetic-default bindings, the same map
   [env]'s scalar loader is built over. [env] alone cannot serve an
   alternate executor that needs a source's SHAPE (e.g. to build a
   [Kernel.Input.t] for [Loop_lower]): [Expr.Eval.Env.t] is a deliberately
   narrow per-coordinate scalar loader ([lib/expr] does not depend on
   [native]'s [Tensor]/[Tensor_sig] at all), so it exposes no shape/format
   for anything it can load. *)

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

val default : t
val default_group : group
