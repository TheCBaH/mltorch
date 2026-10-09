(* The op registry, split out of graph_ir.ml when it crossed the tracked
   1000-line ceiling (scripts/check-file-size.sh): one entry per [op]
   constructor, binding the op's module to [inject]/[project]. *)

open Graph_ir_op

(* In global alphabetical order, mirroring the [op] constructors. *)
let op_registry : (module OP) list =
  [
    (module struct
      include Pointwise.Abs

      let inject t = Abs t
      let project = function Abs t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Add

      let inject t = Add t
      let project = function Add t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Addcmul

      let inject t = Addcmul t
      let project = function Addcmul t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Add_scalar

      let inject t = Add_scalar t
      let project = function Add_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pool.AdaptiveAvgPool2d

      let inject t = Adaptive_avg_pool2d t
      let project = function Adaptive_avg_pool2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pool.AdaptiveMaxPool2d

      let inject t = Adaptive_max_pool2d t
      let project = function Adaptive_max_pool2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pool.AdaptiveMaxPool2dWithIndices

      let inject t = Adaptive_max_pool2d_with_indices t

      let project = function
        | Adaptive_max_pool2d_with_indices t -> Some t
        | _ -> None
    end : OP);
    (module struct
      include Reduce.Amax

      let inject t = Amax t
      let project = function Amax t -> Some t | _ -> None
    end : OP);
    (module struct
      include Reduce.Argmax

      let inject t = Argmax t
      let project = function Argmax t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pool.AvgPool2d

      let inject t = Avg_pool2d t
      let project = function Avg_pool2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Norm.BatchNorm

      let inject t = Batch_norm t
      let project = function Batch_norm t -> Some t | _ -> None
    end : OP);
    (module struct
      include Norm.BatchNormNoStats

      let inject t = Batch_norm_no_stats t
      let project = function Batch_norm_no_stats t -> Some t | _ -> None
    end : OP);
    (module struct
      include Matmul.Batched_matmul

      let inject t = Batched_matmul t
      let project = function Batched_matmul t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Bitwise_and

      let inject t = Bitwise_and t
      let project = function Bitwise_and t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Bitwise_not

      let inject t = Bitwise_not t
      let project = function Bitwise_not t -> Some t | _ -> None
    end : OP);
    (module struct
      include Matmul.Bmm

      let inject t = Bmm t
      let project = function Bmm t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Clamp

      let inject t = Clamp t
      let project = function Clamp t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Clone

      let inject t = Clone t
      let project = function Clone t -> Some t | _ -> None
    end : OP);
    (module struct
      include Im2col.Col2im

      let inject t = Col2im t
      let project = function Col2im t -> Some t | _ -> None
    end : OP);
    (module struct
      include Concat.Concat

      let inject t = Concat t
      let project = function Concat t -> Some t | _ -> None
    end : OP);
    (module struct
      include Conv.Conv1d

      let inject t = Conv1d t
      let project = function Conv1d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Conv.Conv2d

      let inject t = Conv2d t
      let project = function Conv2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Conv.Conv2d_padding

      let inject t = Conv2d_padding t
      let project = function Conv2d_padding t -> Some t | _ -> None
    end : OP);
    (module struct
      include Conv.Conv3d

      let inject t = Conv3d t
      let project = function Conv3d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Conv.Convolution

      let inject t = Convolution t
      let project = function Convolution t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Cos

      let inject t = Cos t
      let project = function Cos t -> Some t | _ -> None
    end : OP);
    (module struct
      include Reduce.Cumsum

      let inject t = Cumsum t
      let project = function Cumsum t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Div

      let inject t = Div t
      let project = function Div t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Div_scalar

      let inject t = Div_scalar t
      let project = function Div_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Embedding.Embedding

      let inject t = Embedding t
      let project = function Embedding t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Eq_scalar

      let inject t = Eq_scalar t
      let project = function Eq_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Eq_tensor

      let inject t = Eq_tensor t
      let project = function Eq_tensor t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Expand

      let inject t = Expand t
      let project = function Expand t -> Some t | _ -> None
    end : OP);
    (module struct
      include Factory.Eye

      let inject t = Eye t
      let project = function Eye t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Floor_div_scalar

      let inject t = Floor_div_scalar t
      let project = function Floor_div_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Ge_scalar

      let inject t = Ge_scalar t
      let project = function Ge_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Gelu

      let inject t = Gelu t
      let project = function Gelu t -> Some t | _ -> None
    end : OP);
    (module struct
      include Norm.GroupNorm

      let inject t = Group_norm t
      let project = function Group_norm t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Gt_scalar

      let inject t = Gt_scalar t
      let project = function Gt_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Hardsigmoid

      let inject t = Hardsigmoid t
      let project = function Hardsigmoid t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Hardswish

      let inject t = Hardswish t
      let project = function Hardswish t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Hardtanh

      let inject t = Hardtanh t
      let project = function Hardtanh t -> Some t | _ -> None
    end : OP);
    (module struct
      include Index_tensor.Index_pair

      let inject t = Index_pair t
      let project = function Index_pair t -> Some t | _ -> None
    end : OP);
    (module struct
      include Index_tensor.Index_tensor

      let inject t = Index_tensor t
      let project = function Index_tensor t -> Some t | _ -> None
    end : OP);
    (module struct
      include Im2col.Im2col

      let inject t = Im2col t
      let project = function Im2col t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Le_tensor

      let inject t = Le_tensor t
      let project = function Le_tensor t -> Some t | _ -> None
    end : OP);
    (module struct
      include Norm.LayerNorm

      let inject t = Layer_norm t
      let project = function Layer_norm t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Leaky_relu

      let inject t = Leaky_relu t
      let project = function Leaky_relu t -> Some t | _ -> None
    end : OP);
    (module struct
      include Linear.Linear

      let inject t = Linear t
      let project = function Linear t -> Some t | _ -> None
    end : OP);
    (module struct
      include Lstm.Lstm

      let inject t = Lstm t
      let project = function Lstm t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Lt_scalar

      let inject t = Lt_scalar t
      let project = function Lt_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Reduce.MaxDim

      let inject t = Max_dim t
      let project = function Max_dim t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pool.MaxPool2d

      let inject t = Max_pool2d t
      let project = function Max_pool2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pool.MaxPool2dWithIndices

      let inject t = Max_pool2d_with_indices t
      let project = function Max_pool2d_with_indices t -> Some t | _ -> None
    end : OP);
    (module struct
      include Reduce.Mean

      let inject t = Mean t
      let project = function Mean t -> Some t | _ -> None
    end : OP);
    (module struct
      include Meshgrid.Meshgrid

      let inject t = Meshgrid t
      let project = function Meshgrid t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Mul

      let inject t = Mul t
      let project = function Mul t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Mul_scalar

      let inject t = Mul_scalar t
      let project = function Mul_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Ne_scalar

      let inject t = Ne_scalar t
      let project = function Ne_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Ne_tensor

      let inject t = Ne_tensor t
      let project = function Ne_tensor t -> Some t | _ -> None
    end : OP);
    (module struct
      include Factory.New_ones

      let inject t = New_ones t
      let project = function New_ones t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pad.Pad

      let inject t = Pad t
      let project = function Pad t -> Some t | _ -> None
    end : OP);
    (module struct
      include Permute.Permute

      let inject t = Permute t
      let project = function Permute t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Pow

      let inject t = Pow t
      let project = function Pow t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Relu

      let inject t = Relu t
      let project = function Relu t -> Some t | _ -> None
    end : OP);
    (module struct
      include Repeat.Repeat

      let inject t = Repeat t
      let project = function Repeat t -> Some t | _ -> None
    end : OP);
    (module struct
      include Repeat.RepeatInterleave

      let inject t = RepeatInterleave t
      let project = function RepeatInterleave t -> Some t | _ -> None
    end : OP);
    (module struct
      include Reshape.Reshape

      let inject t = Reshape t
      let project = function Reshape t -> Some t | _ -> None
    end : OP);
    (module struct
      include Norm.RmsNorm

      let inject t = Rms_norm t
      let project = function Rms_norm t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Rpow_scalar

      let inject t = Rpow_scalar t
      let project = function Rpow_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Rsub_scalar

      let inject t = Rsub_scalar t
      let project = function Rsub_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Attention.Sdpa

      let inject t = Sdpa t
      let project = function Sdpa t -> Some t | _ -> None
    end : OP);
    (module struct
      include Split.Select

      let inject t = Select t
      let project = function Select t -> Some t | _ -> None
    end : OP);
    (module struct
      include Split.Select_scatter

      let inject t = Select_scatter t
      let project = function Select_scatter t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Sigmoid

      let inject t = Sigmoid t
      let project = function Sigmoid t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Silu

      let inject t = Silu t
      let project = function Silu t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Sin

      let inject t = Sin t
      let project = function Sin t -> Some t | _ -> None
    end : OP);
    (module struct
      include Reduce.Softmax

      let inject t = Softmax t
      let project = function Softmax t -> Some t | _ -> None
    end : OP);
    (module struct
      include Split.Slice

      let inject t = Slice t
      let project = function Slice t -> Some t | _ -> None
    end : OP);
    (module struct
      include Split.Split_with_sizes

      let inject t = Split_with_sizes t
      let project = function Split_with_sizes t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Sqrt

      let inject t = Sqrt t
      let project = function Sqrt t -> Some t | _ -> None
    end : OP);
    (module struct
      include Concat.Stack

      let inject t = Stack t
      let project = function Stack t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Sub

      let inject t = Sub t
      let project = function Sub t -> Some t | _ -> None
    end : OP);
    (module struct
      include Reduce.Sum

      let inject t = Sum t
      let project = function Sum t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Tanh

      let inject t = Tanh t
      let project = function Tanh t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.To_copy

      let inject t = To_copy t
      let project = function To_copy t -> Some t | _ -> None
    end : OP);
    (module struct
      include Split.Unbind

      let inject t = Unbind t
      let project = function Unbind t -> Some t | _ -> None
    end : OP);
    (module struct
      include Unfold.Unfold

      let inject t = Unfold t
      let project = function Unfold t -> Some t | _ -> None
    end : OP);
    (module struct
      include Resize.Bicubic2d

      let inject t = Upsample_bicubic2d t
      let project = function Upsample_bicubic2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Resize.Bilinear2d

      let inject t = Upsample_bilinear2d t
      let project = function Upsample_bilinear2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Resize.Nearest2d

      let inject t = Upsample_nearest2d t
      let project = function Upsample_nearest2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Reduce.Vector_norm

      let inject t = Vector_norm t
      let project = function Vector_norm t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Where_scalar_other

      let inject t = Where_scalar_other t
      let project = function Where_scalar_other t -> Some t | _ -> None
    end : OP);
    (module struct
      include Factory.Arange

      let inject t = Arange t
      let project = function Arange t -> Some t | _ -> None
    end : OP);
    (module struct
      include Factory.Zeros

      let inject t = Zeros t
      let project = function Zeros t -> Some t | _ -> None
    end : OP);
  ]
