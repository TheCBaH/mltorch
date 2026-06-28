(* Direct (concrete) evaluation of a native graph. Walks the topo-ordered nodes
   threading an immutable env, runs each op through [Eval_op.Make (Direct)] and
   [Schedule.evaluate], and recurses into embedded subgraphs. Returns EVERY edge's
   tensor (inputs, intermediates, outputs), keyed by edge id — so callers can print
   any intermediate, not just the graph outputs. See .ai/native_graph_design.md. *)

open Graph_ir

val run :
  graph ->
  inputs:(Tensor_id.t * Tensor.packed) list ->
  Tensor.packed Tensor_id.Map.t
