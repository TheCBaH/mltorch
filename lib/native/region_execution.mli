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

val lower :
  max_size:int ->
  max_depth:int ->
  max_local_slots:int ->
  max_scan_state:int ->
  max_scan_updates:int64 ->
  output_shape:Vec6.shape ->
  Region_program.t ->
  (t, Region_program.error) Err.t
(** Re-validates [program] -- [Region_program.check] against [max_size]/
    [max_depth], then [Region_program.preflight] against the three scan resource
    dimensions and [output_shape] -- before lowering it, so a caller holding a
    [t] never has to trust that an intervening rewrite (e.g.
    [Kernel.Result_conversion.apply] via [Region_program.with_output]) left it
    well-formed. Applies to a Pixel program too: [pixel_expression = Some] used
    to reach [Pixel_loop] with no validation at all. *)

val lower_region :
  max_size:int ->
  max_depth:int ->
  max_local_slots:int ->
  max_scan_state:int ->
  max_scan_updates:int64 ->
  output_shape:Vec6.shape ->
  Region_program.t ->
  (lowered, Region_program.error) Err.t
(** As [lower], for a caller that already knows -- structurally, e.g. from
    [Region_program.pixel_expression] returning [None] -- that [program] is not
    a plain pixel expression, so it need not re-derive that by matching on [t].
*)

val materialize :
  ?counters:counters ->
  lowered ->
  env:Expr.Eval.Env.t ->
  (Tensor.packed, Region_eval.error) Err.t
(** [lowered] retains the [output_shape] validated at [lower_region] time, so
    this can no longer be called with a shape that disagrees with it. *)

val value_at :
  lowered ->
  env:Expr.Eval.Env.t ->
  output:Vec6.coord ->
  (float, Region_eval.error) Err.t
(** Fresh scalar projection of an already-validated lowered Region program. The
    output coordinate is bounds-checked by [Region_partition.key_of_output].
    Production tensor execution uses [materialize] once, never this function per
    output. *)
