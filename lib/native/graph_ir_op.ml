(* [type op] and its per-op module signature, split out of graph_ir.ml (which
   crossed the tracked 1000-line ceiling, scripts/check-file-size.sh) rather
   than folded into it. Neither is part of graph_ir.mli's public surface
   ([op] itself is, via graph_ir.ml's `open Graph_ir_op`, but [OP] and
   [op_registry] are internal wiring) -- this file exists purely so
   [op_registry]'s second half (graph_ir_registry_ext.ml) has something to
   depend on without creating a cycle back through graph_ir.ml. *)

type tensor_ref = Tensor_id.t

type op =
  (* Constructors kept in global alphabetical order (see graph_ir.mli). Each
     each op carries its own payload record (params + operand refs),
     defined in that op's module; the shared serialise / dataflow / pp logic is
     driven from [op_registry] below, not a per-constructor match. *)
  | Add of Pointwise.Add.t
  | Addcmul of Pointwise.Addcmul.t
  | Add_scalar of Pointwise.Add_scalar.t
  | Adaptive_avg_pool2d of Pool.AdaptiveAvgPool2d.t
  | Adaptive_max_pool2d of Pool.AdaptiveMaxPool2d.t
  | Adaptive_max_pool2d_with_indices of Pool.AdaptiveMaxPool2dWithIndices.t
  | Amax of Reduce.Amax.t
  | Avg_pool2d of Pool.AvgPool2d.t
  | Batch_norm of Norm.BatchNorm.t
  | Batch_norm_no_stats of Norm.BatchNormNoStats.t
  | Batched_matmul of Matmul.Batched_matmul.t
  | Bitwise_not of Pointwise.Bitwise_not.t
  | Bmm of Matmul.Bmm.t
  | Clamp of Pointwise.Clamp.t
  | Clone of Pointwise.Clone.t
  | Col2im of Im2col.Col2im.t
  | Concat of Concat.Concat.t
  | Conv1d of Conv.Conv1d.t
  | Conv2d of Conv.Conv2d.t
  | Conv2d_padding of Conv.Conv2d_padding.t
  | Conv3d of Conv.Conv3d.t
  | Convolution of Conv.Convolution.t
  | Cos of Pointwise.Cos.t
  | Cumsum of Reduce.Cumsum.t
  | Div of Pointwise.Div.t
  | Div_scalar of Pointwise.Div_scalar.t
  | Discard of { x : tensor_ref }
  | Expand of Pointwise.Expand.t
  | Eye of Factory.Eye.t
  | Floor_div_scalar of Pointwise.Floor_div_scalar.t
  | Gelu of Pointwise.Gelu.t
  | Group_norm of Norm.GroupNorm.t
  | Hardsigmoid of Pointwise.Hardsigmoid.t
  | Hardswish of Pointwise.Hardswish.t
  | Hardtanh of Pointwise.Hardtanh.t
  | Index_tensor of Index_tensor.Index_tensor.t
  | Im2col of Im2col.Im2col.t
  | Layer_norm of Norm.LayerNorm.t
  | Leaky_relu of Pointwise.Leaky_relu.t
  | Linear of Linear.Linear.t
  | Lstm of Lstm.Lstm.t
  | Max_dim of Reduce.MaxDim.t
  | Max_pool2d of Pool.MaxPool2d.t
  | Max_pool2d_with_indices of Pool.MaxPool2dWithIndices.t
  | Mean of Reduce.Mean.t
  | Meshgrid of Meshgrid.Meshgrid.t
  | Mul of Pointwise.Mul.t
  | Mul_scalar of Pointwise.Mul_scalar.t
  | Pad of Pad.Pad.t
  | Permute of Permute.Permute.t
  | Pow of Pointwise.Pow.t
  | Relu of Pointwise.Relu.t
  | Repeat of Repeat.Repeat.t
  | RepeatInterleave of Repeat.RepeatInterleave.t
  | Reshape of Reshape.Reshape.t
  | Rms_norm of Norm.RmsNorm.t
  | Rpow_scalar of Pointwise.Rpow_scalar.t
  | Rsub_scalar of Pointwise.Rsub_scalar.t
  | Sdpa of Attention.Sdpa.t
  | Select of Split.Select.t
  | Select_scatter of Split.Select_scatter.t
  | Sigmoid of Pointwise.Sigmoid.t
  | Silu of Pointwise.Silu.t
  | Sin of Pointwise.Sin.t
  | Softmax of Reduce.Softmax.t
  | Slice of Split.Slice.t
  | Split_with_sizes of Split.Split_with_sizes.t
  | Sqrt of Pointwise.Sqrt.t
  | Stack of Concat.Stack.t
  | Sub of Pointwise.Sub.t
  | Sum of Reduce.Sum.t
  | To_copy of Pointwise.To_copy.t
  | Unbind of Split.Unbind.t
  | Unfold of Unfold.Unfold.t
  | Upsample_bicubic2d of Resize.Bicubic2d.t
  | Upsample_bilinear2d of Resize.Bilinear2d.t
  | Upsample_nearest2d of Resize.Nearest2d.t
  | Vector_norm of Reduce.Vector_norm.t
  | Arange of Factory.Arange.t
  | Zeros of Factory.Zeros.t

(* Per-op interface. Each op module supplies the name, codec, dataflow accessors
   and printer for its OWN payload; [inject]/[project] (added by the wrappers in
   [op_registry], since they name the variant) splice that payload in and out of
   [op]. The common code below folds the registry instead of matching every
   constructor, so adding an op needs only its module plus one registry entry. *)
module type OP = sig
  type t

  val name : string
  val jsont : t Jsont.t
  val operands : t -> tensor_ref list
  val map_operands : (tensor_ref -> tensor_ref) -> t -> t
  val pp : tensor_ref Fmt.t -> Format.formatter -> t -> unit
  val inject : t -> op
  val project : op -> t option
end
