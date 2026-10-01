(* Direct (concrete) evaluation of a native graph. Walks the topo-ordered nodes
   threading an immutable env, runs each op through [Eval_op.Make (Direct)] and
   [Schedule.evaluate]. Structural groups do not affect evaluation. See
   .ai/native_graph_design.md.

   [?retain] decides what the result holds, keyed by edge id:
   - [Release_schedule.Retain.All], the default: EVERY edge's tensor (inputs,
     intermediates, outputs), so callers can print any intermediate.
   - [Only s]: exactly [g.outputs] and those of [s] that were bound. Every other
     edge leaves the env right after its last reader runs, so its payload is
     garbage from then on rather than at the end of the run. Outputs and errors
     are identical to an [All] run.
   Neither setting ever holds an index output nothing reads: it is not
   allocated. See .ai/ (tensor release). *)

open Graph_ir

type context = Operand | Sig_shape
type missing_tensor = { context : context; id : Tensor_id.t }
type arity_mismatch = { expected : int; actual : int }

type mixed_dtype = {
  mixed_op : string;
  a_fmt : Payload.packed_fmt;
  b_fmt : Payload.packed_fmt;
}

type scalar_op = { scalar_op : string; fmt : Payload.packed_fmt }

type error =
  [ Eval_direct_compute.error
  | Graph_shape.error
  | `Missing_constant of Tensor_id.t
  | `Missing_input of Tensor_id.t
  | `Missing_tensor of missing_tensor
  | `Output_arity_mismatch of arity_mismatch
  | `Region_construction of Region_computation.error
  | `Region_execution of Region_eval.error
  | `Unsupported_bool_arithmetic of mixed_dtype
  | `Unsupported_bool_scalar_arithmetic of scalar_op
  | `Unsupported_mixed_dtype of mixed_dtype ]

type hooks =
  | Hooks : { on_start : node -> 'a; on_end : node -> 'a -> unit } -> hooks

val pp_error : Format.formatter -> [< error ] -> unit

val run :
  ?hooks:hooks ->
  ?region_counters:Region_execution.counters Tensor_id.Map.t ->
  ?region_executor:Region_executor.t ->
  ?region_group_executor:Region_executor.group ->
  ?node_executor:Node_executor.t ->
  ?limits:Kernel.Limits.t ->
  ?retain:Release_schedule.Retain.t ->
  ?constants:(Tensor_id.t * Tensor.packed) list ->
  graph ->
  inputs:(Tensor_id.t * Tensor.packed) list ->
  (Tensor.packed Tensor_id.Map.t, error) Err.t
