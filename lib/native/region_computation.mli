type error =
  | Invalid_group of Region_group.error
  | Invalid_partition
  | Invalid_program of Region_program.error
  | Invalid_shape of Shape_error.t
  | Missing_operand of Tensor_id.t
  | Output_ordinal of int
  | Output_shape

type synthetic_role = Layer_bias | Layer_weight | Rms_weight | Sdpa_mask

val is_region_authored : Graph_ir.op -> bool

val program :
  limits:Kernel.Limits.t ->
  op:Graph_ir.op ->
  output:int ->
  output_shape:Vec6.shape ->
  operand:(Tensor_id.t -> Tensor_sig.t option) ->
  fill:(synthetic_role -> float -> Vec6.shape -> Tensor_sig.t) ->
  (Region_program.t, error) Err.t

val group :
  limits:Kernel.Limits.t ->
  op:Graph_ir.op ->
  operand:(Tensor_id.t -> Tensor_sig.t option) ->
  (Region_group.t, error) Err.t
(** Builds every output ordinal's Region computation from ONE shared recurrence,
    for the operations {!is_region_authored} that support it (currently only
    [Lstm] -- see [Region_group]). Unlike {!program}, there is no single
    [output]/[output_shape] to check against: the per-ordinal shape checks
    already happened while resolving the group's own [Region_group.Emitter.t]
    list. *)

val pp_error : Format.formatter -> error -> unit
