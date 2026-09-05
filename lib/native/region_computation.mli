type error =
  | Invalid_partition
  | Invalid_program of Region_program.error
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

val pp_error : Format.formatter -> error -> unit
