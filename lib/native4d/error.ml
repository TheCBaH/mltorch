(* See error.mli. *)

open Graph_ir

type t =
  [ `Axis_outside_dialect of Node_id.t * Axis.t
  | `Dynamic_batch_norm of Node_id.t
  | `Live_max_pool_indices of Node_id.t * Tensor_id.t
  | `Non_four_dimensional_tensor of Tensor_id.t * Vec6.shape
  | `Unsupported_bmm_batch of Node_id.t * Dim.extent Dim.t
  | `Unsupported_grouped_conv of Node_id.t * int
  | `Unsupported_grouped_transposed_conv of Node_id.t * int
  | `Unsupported_op of Node_id.t * op ]

let pp fmt : [< t ] -> unit = function
  | `Axis_outside_dialect (node, axis) ->
      Fmt.pf fmt "@[node %a: axis %a is outside the N/H/W/C dialect@]"
        Node_id.pp node Axis.pp axis
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
  | `Non_four_dimensional_tensor (id, shape) ->
      Fmt.pf fmt "@[tensor %a has extent on T or D: %a@]" Tensor_id.pp id
        Vec6.pp_shape shape
  | `Unsupported_bmm_batch (node, batch) ->
      Fmt.pf fmt
        "@[node %a: bmm batch extent is %a; only a single batch legalizes to a \
         1x1 convolution@]"
        Node_id.pp node Dim.pp batch
  | `Unsupported_grouped_conv (node, groups) ->
      Fmt.pf fmt
        "@[node %a: convolution has %d groups, which is neither 1 nor \
         depthwise@]"
        Node_id.pp node groups
  | `Unsupported_grouped_transposed_conv (node, groups) ->
      Fmt.pf fmt
        "@[node %a: transposed convolution has %d groups; only 1 legalizes@]"
        Node_id.pp node groups
  | `Unsupported_op (node, op) ->
      Fmt.pf fmt "@[<hv 2>node %a: no legalization for@ %a@]" Node_id.pp node
        (Graph_ir.pp_op_with ~pp_ref:Tensor_id.pp)
        op
