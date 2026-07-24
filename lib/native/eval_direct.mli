(* Direct (concrete) evaluation of a native graph. Walks the topo-ordered nodes
   threading an immutable env, runs each op through [Eval_op.Make (Direct)] and
   [Schedule.evaluate], and recurses into embedded subgraphs. Returns EVERY edge's
   tensor (inputs, intermediates, outputs), keyed by edge id — so callers can print
   any intermediate, not just the graph outputs. See .ai/native_graph_design.md. *)

open Graph_ir

type context = Operand | Sig_shape | Subgraph_input | Subgraph_output
type missing_tensor = { context : context; id : Tensor_id.t }
type arity_mismatch = { expected : int; actual : int }

type error =
  [ Graph_shape.error
  | `Missing_tensor of missing_tensor
  | `Subgraph_input_arity_mismatch of arity_mismatch
  | `Subgraph_output_arity_mismatch of arity_mismatch
  | `Output_arity_mismatch of arity_mismatch ]

val pp_error : Format.formatter -> [< error ] -> unit

val run :
  graph ->
  inputs:(Tensor_id.t * Tensor.packed) list ->
  (Tensor.packed Tensor_id.Map.t, error) Core.result
