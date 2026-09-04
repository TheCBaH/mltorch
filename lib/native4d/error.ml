(* See error.mli. *)

open Graph_ir

type t =
  [ `Axis_outside_dialect of Node_id.t * Axis.t
  | `Bad_constant_payload of Tensor_id.t
  | `Batch_norm_extent of
    Node_id.t * Tensor_id.t * Dim.extent Dim.t * Dim.extent Dim.t
  | `Batched_matmul_batch_axis of Node_id.t
  | `Constant_store of Constant_store.error
  | `Dynamic_batch_norm of Node_id.t
  | `Live_max_pool_indices of Node_id.t * Tensor_id.t
  | `Map of Graph_map.error
  | `Missing_constant_payload of Node_id.t * Tensor_id.t
  | `Non_four_dimensional_tensor of Tensor_id.t * Vec6.shape
  | `Sdpa_batch_axis of Node_id.t
  | `Unsupported_grouped_transposed_conv of Node_id.t * int
  | `Unsupported_op of Node_id.t * op
  | `View of Framework.View4.error ]

let pp fmt : [< t ] -> unit = function
  | `Axis_outside_dialect (node, axis) ->
      Fmt.pf fmt "@[node %a: axis %a is outside the N/H/W/C dialect@]"
        Node_id.pp node Axis.pp axis
  | `Bad_constant_payload id ->
      Fmt.pf fmt
        "@[constant %a's payload does not match its recorded signature@]"
        Tensor_id.pp id
  | `Batch_norm_extent (node, id, extent, channels) ->
      Fmt.pf fmt
        "@[node %a: batch norm parameter %a has extent %a on C, but the \
         normalized axis has %a@]"
        Node_id.pp node Tensor_id.pp id Dim.pp extent Dim.pp channels
  | `Batched_matmul_batch_axis node ->
      Fmt.pf fmt
        "@[node %a: batched matmul has extent > 1 on D, which the N/H/W/C \
         dialect has no name for; N and H already carry the op's real batch \
         axes@]"
        Node_id.pp node
  | `Constant_store e -> Constant_store.pp_error fmt e
  | `Dynamic_batch_norm node ->
      Fmt.pf fmt
        "@[node %a: batch norm parameters are not all constant, so no \
         per-channel scale can be precomputed@]"
        Node_id.pp node
  | `Live_max_pool_indices (node, id) ->
      Fmt.pf fmt
        "@[node %a: max-pool index output %a is live; the dialect has no \
         argmax-pool operation@]"
        Node_id.pp node Tensor_id.pp id
  | `Map e -> Fmt.pf fmt "@[map: %a@]" Graph_map.pp_error e
  | `Missing_constant_payload (node, id) ->
      Fmt.pf fmt
        "@[node %a needs constant %a's payload at conversion time, and none \
         was supplied@]"
        Node_id.pp node Tensor_id.pp id
  | `Non_four_dimensional_tensor (id, shape) ->
      Fmt.pf fmt "@[tensor %a has extent on T or D: %a@]" Tensor_id.pp id
        Vec6.pp_shape shape
  | `Sdpa_batch_axis node ->
      Fmt.pf fmt
        "@[node %a: scaled-dot-product attention has extent > 1 on D, its \
         batch axis, which the N/H/W/C dialect has no name for@]"
        Node_id.pp node
  | `Unsupported_grouped_transposed_conv (node, groups) ->
      Fmt.pf fmt
        "@[node %a: transposed convolution has %d groups; only 1 legalizes@]"
        Node_id.pp node groups
  | `Unsupported_op (node, op) ->
      Fmt.pf fmt "@[<hv 2>node %a: no legalization for@ %a@]" Node_id.pp node
        (Graph_ir.pp_op_with ~pp_ref:Tensor_id.pp)
        op
  | `View e ->
      Fmt.pf fmt "@[destination graph is invalid: %a@]" Framework.View4.pp_error
        e
