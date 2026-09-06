type counters = {
  mutable keys : int;
  mutable locals : int;
  mutable emitters : int;
  mutable loads : int;
  mutable reductions : int;
  mutable scans : int;
  mutable scan_updates : int;
}

type lowered
type t = Pixel_loop of Expr.Value.t | Region_loop of lowered

val counters : unit -> counters

val lower :
  max_size:int ->
  max_depth:int ->
  max_local_slots:int ->
  scan_limits:Expr.Scan_limits.t ->
  output_shape:Vec6.shape ->
  Region_program.t ->
  (t, Region_program.error) Err.t
(** Re-validates [program] -- [Region_program.check] against [max_size]/
    [max_depth], then [Region_program.preflight] against [scan_limits]'s two
    scan resource dimensions plus [max_local_slots] and [output_shape] -- before
    lowering it, so a caller holding a [t] never has to trust that an
    intervening rewrite (e.g. [Kernel.Result_conversion.apply] via
    [Region_program.with_output]) left it well-formed. Applies to a Pixel
    program too: [pixel_expression = Some] used to reach [Pixel_loop] with no
    validation at all. [scan_limits] is typically
    [Kernel.Limits.scan_limits limits] for a caller already holding a
    [Kernel.Limits.t]. *)

val lower_region :
  max_size:int ->
  max_depth:int ->
  max_local_slots:int ->
  scan_limits:Expr.Scan_limits.t ->
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
(** [lowered] retains the [output_shape] and [scan_limits] validated at
    [lower_region] time, so this can no longer be called with a shape that
    disagrees with it, and needs no meter of its own: a fresh
    [Expr.Scan_meter.t] is created per Region key, shared by every local
    (including a scan's own trace fill) and the emitter for that key. *)

val value_at :
  lowered ->
  env:Expr.Eval.Env.t ->
  output:Vec6.coord ->
  (float, Region_eval.error) Err.t
(** Fresh scalar projection of an already-validated lowered Region program, with
    its own fresh [Expr.Scan_meter.t] (built from [lowered]'s [scan_limits]).
    The output coordinate is bounds-checked by [Region_partition.key_of_output].
    Production tensor execution uses [materialize] once, never this function per
    output. *)
