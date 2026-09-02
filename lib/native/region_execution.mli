type counters = {
  mutable keys : int;
  mutable locals : int;
  mutable emitters : int;
  mutable loads : int;
  mutable reductions : int;
}

type lowered
type t = Pixel_loop of Expr.Value.t | Region_loop of lowered

val counters : unit -> counters
val lower : Region_program.t -> t

val lower_region : Region_program.t -> lowered
(** As [lower], for a caller that already knows -- structurally, e.g. from
    [Region_program.pixel_expression] returning [None] -- that [program] is not
    a plain pixel expression, so it need not re-derive that by matching on [t].
*)

val materialize :
  ?counters:counters ->
  lowered ->
  output_shape:Vec6.shape ->
  env:Expr.Eval.Env.t ->
  (Tensor.packed, Region_eval.error) Err.t

val value_at :
  lowered ->
  output_shape:Vec6.shape ->
  env:Expr.Eval.Env.t ->
  output:Vec6.coord ->
  (float, Region_eval.error) Err.t
(** Fresh scalar projection of an already-validated lowered Region program. The
    output coordinate is bounds-checked by [Region_partition.key_of_output].
    Production tensor execution uses [materialize] once, never this function per
    output. *)
