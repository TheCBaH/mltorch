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
