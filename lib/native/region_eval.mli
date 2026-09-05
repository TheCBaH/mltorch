type error =
  [ Expr.Eval.error
  | Region_partition.error
  | `Scan_execution_not_implemented of Expr.Local_var.t ]
(** [Scan_execution_not_implemented] fires for a scan-shaped local reaching
    either evaluator: [Region_program] can construct, check and render a trace
    local (see the scan design record's Stage 1), but running one -- charging a
    shared meter across its lanes and steps -- is a later, dedicated step. This
    case is a temporary boundary, not a permanent limitation, and disappears
    once that lands. *)

val pp_error : Format.formatter -> [< error ] -> unit

val value_at :
  Region_program.t ->
  output_shape:Vec6.shape ->
  env:Expr.Eval.Env.t ->
  output:Vec6.coord ->
  (float, error) Err.t
(** Fresh concrete scalar projection. It reconstructs the output's key and
    evaluates all scalar locals for this one observation; it is intentionally
    not the materialization traversal. *)

val materialize :
  Region_program.t ->
  output_shape:Vec6.shape ->
  env:Expr.Eval.Env.t ->
  (Tensor.packed, error) Err.t
(** Reference materialization with a key-indexed local cache. *)
