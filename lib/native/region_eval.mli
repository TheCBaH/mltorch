type error = [ Expr.Eval.error | Region_partition.error ]

val pp_error : Format.formatter -> [< error ] -> unit

val value_at :
  ?scan_limits:Expr.Scan_limits.t ->
  Region_program.t ->
  output_shape:Vec6.shape ->
  env:Expr.Eval.Env.t ->
  output:Vec6.coord ->
  (float, error) Err.t
(** Fresh concrete scalar projection. It reconstructs the output's key and
    evaluates all locals (including a scan's own trace) for this one
    observation, sharing one fresh [Expr.Scan_meter.t] across them; it is
    intentionally not the materialization traversal. [scan_limits] defaults to
    [Expr.Scan_limits.default] -- this is the reference/test oracle, not a
    production entry point, so an explicit default here is not the "silent
    substitution" production callers must avoid. *)

val materialize :
  ?scan_limits:Expr.Scan_limits.t ->
  Region_program.t ->
  output_shape:Vec6.shape ->
  env:Expr.Eval.Env.t ->
  (Tensor.packed, error) Err.t
(** Reference materialization with a key-indexed local cache. A fresh
    [Expr.Scan_meter.t] is created per Region key, shared by every local
    (scalar, vector, and a scan's own trace fill) and the emitter for that key.
*)

val materialize_i64 :
  output_shape:Vec6.shape ->
  env:Expr.Eval.Env.t ->
  int64 Expr.Value.t ->
  (Tensor.packed, error) Err.t
(** Exact int64 counterpart of [materialize]'s pixel-degenerate case: no locals,
    one evaluation of [output] per output coordinate via [Expr.Eval.value_i64]
    (never [Expr.Eval.value], which would force a precision-losing
    [i64_to_float] wrap), reusing [Tensor.materialize_i64] rather than a per-key
    loop since there is no shared local state to amortize. A
    [Region_program.t]-shaped int64 program (partition, typed locals, admission
    checks) is deliberately not built here. *)
