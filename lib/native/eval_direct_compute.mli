(* The non-Region "compute" half of [Eval_direct.eval_node]'s per-output
   dispatch (design §4.2, plan T2.1): every computing arm from [Unbind] down
   to the default float-pixel arm, moved here verbatim and in the same order
   to bring [eval_direct.ml] back under the file-size cap. The Bool and
   mixed-dtype checked-admission arms stay in [Eval_direct.admit], which runs
   BEFORE this is ever called, so [compute] never sees an input those arms
   would have rejected — this is why removing them here does not change any
   op's decidable domain (see [Eval_direct]'s own comment on that argument).
   Region-authored ops are excluded too: [Eval_direct] dispatches those to
   [Region_computation]/[Region_executor.t] itself, never through here. *)

open Graph_ir

type error =
  [ `Arange_i64_overflow of Factory.Arange.Overflow.t
  | `Unsupported_to_copy_bool_source of Payload.packed_fmt
  | `Unsupported_to_copy_long_source of Payload.packed_fmt ]

val pp_error : Format.formatter -> [< error ] -> unit

val compute :
  graph ->
  op ->
  output:Output_ordinal.t ->
  out_shape:Vec6.shape ->
  operand_env:Tensor.packed Tensor_id.Map.t ->
  shape_env:Vec6.shape Tensor_id.Map.t ->
  fill:(float -> Vec6.shape -> Tensor.packed) ->
  (Tensor.packed, [> error ]) Err.t
