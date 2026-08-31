type error = [ Expr.Eval.error | Region_partition.error ]

val pp_error : Format.formatter -> [< error ] -> unit

val value_at :
  Region_program.t ->
  output_shape:Vec6.shape ->
  env:Expr.Eval.Env.t ->
  output:Vec6.coord ->
  (float, error) Err.t

val materialize :
  Region_program.t ->
  output_shape:Vec6.shape ->
  env:Expr.Eval.Env.t ->
  (Tensor.packed, error) Err.t
