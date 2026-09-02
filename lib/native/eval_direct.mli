(* Direct (concrete) evaluation of a native graph. Walks the topo-ordered nodes
   threading an immutable env, runs each op through [Eval_op.Make (Direct)] and
   [Schedule.evaluate]. Structural groups do not affect evaluation. Returns EVERY edge's
   tensor (inputs, intermediates, outputs), keyed by edge id — so callers can print
   any intermediate, not just the graph outputs. See .ai/native_graph_design.md. *)

open Graph_ir

type context = Operand | Sig_shape
type missing_tensor = { context : context; id : Tensor_id.t }
type arity_mismatch = { expected : int; actual : int }

type error =
  [ Graph_shape.error
  | `Missing_constant of Tensor_id.t
  | `Missing_input of Tensor_id.t
  | `Missing_tensor of missing_tensor
  | `Output_arity_mismatch of arity_mismatch
  | `Region_construction of Region_computation.error
  | `Region_execution of Region_eval.error ]

type hooks =
  | Hooks : { on_start : node -> 'a; on_end : node -> 'a -> unit } -> hooks

val pp_error : Format.formatter -> [< error ] -> unit

val run :
  ?hooks:hooks ->
  ?region_counters:Region_execution.counters Tensor_id.Map.t ->
  ?limits:Kernel.Limits.t ->
  ?constants:(Tensor_id.t * Tensor.packed) list ->
  graph ->
  inputs:(Tensor_id.t * Tensor.packed) list ->
  (Tensor.packed Tensor_id.Map.t, error) Err.t
