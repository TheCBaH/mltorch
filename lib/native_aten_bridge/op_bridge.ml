(* Native-side dispatch: given a graph node and the ATen environment (inputs),
   build a native Graph_ir.graph encoding the equivalent computation.

   Returns [None] if no native implementation exists for this op.  Returns
   [Some (Error e)] if the op is mapped but argument conversion or param
   validation fails.  Returns [Some (Ok (g, bindings))] where [g] is the native
   graph and [bindings] maps each graph input id to its converted native tensor.

   Ops requiring NCHW<->NHWC relayout (conv2d, max_pool2d, linear/addmm) produce
   a graph named "<op>_relayout" that wraps the core op in permute nodes.  All
   other ops produce a flat single-op graph.  Unmapped ops return None.

   This file is now a thin facade: error payload types and printing live in
   op_bridge_error.ml, argument-decoding/permutation/param helpers in
   op_bridge_decode.ml, and the per-node dispatch arms are grouped by
   operation family in op_bridge_pointwise.ml, op_bridge_linalg.ml,
   op_bridge_conv.ml, op_bridge_pool.ml, op_bridge_recurrent.ml,
   op_bridge_reduce.ml, op_bridge_norm.ml, op_bridge_attention.ml,
   op_bridge_shape.ml, and op_bridge_factory.ml. Every
   family's [dispatch] has the same signature and disjoint [node.target]
   cases (no op name is handled by more than one family), so trying each in
   turn and taking the first non-[None] result is equivalent to the original
   single match. *)

type error = Op_bridge_error.error

let pp_error = Op_bridge_error.pp_error

let dispatchers =
  [
    Op_bridge_attention.dispatch;
    Op_bridge_conv.dispatch;
    Op_bridge_factory.dispatch;
    Op_bridge_linalg.dispatch;
    Op_bridge_norm.dispatch;
    Op_bridge_pointwise.dispatch;
    Op_bridge_pool.dispatch;
    Op_bridge_recurrent.dispatch;
    Op_bridge_reduce.dispatch;
    Op_bridge_shape.dispatch;
  ]

let dispatch ~(aten_env : Op_bridge_error.aten_env)
    (node : Pytorch_types.Node.t) :
    (Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list, error) Err.t
    option =
  List.find_map (fun f -> f ~aten_env node) dispatchers
