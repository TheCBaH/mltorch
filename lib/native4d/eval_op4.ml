(* The single Native4D op-dispatch point, functorised over the semantics [S] —
   the Native4D twin of [Eval_op], applied once at [Direct] and once at
   [Symbolic] so the two execution modes cannot drift.

   Every Pixel-authored arm calls Native's [Compute (S)].  RMSNorm, LayerNorm,
   Softmax4 and Sdpa are Region-authored instead: Direct and Symbolic route
   them through [Region_computation4.program], so they deliberately have no
   scalar arm here.
   .ai/native4d_design.md §2 lists reimplementing numeric kernels already
   expressed by the Native operation as a non-goal.  This file therefore has no
   arithmetic of its own — only parameter translation for Pixel operations.

   And no tensor is copied to cross the boundary. The weight layouts are
   Native's unchanged (§6.1): forward [Cout,1,1,Kh,Kw,Cin/groups], transposed
   [Cin,1,1,Kh,Kw,Cout/groups]. Choosing a different public layout would mean a
   permutation before every shared compute call, which is exactly the cost the
   design says a different backend's layout should pay in ITS lowering rather
   than here. That no permutation appears below is the first of the three
   architectural claims §14 asks the milestone to prove.

   Parameter translation lives in [Graph_shape4] — [conv2d_params],
   [transposed_params], [perm6], [mean_params], [rms_params] — because shape
   inference needs exactly the same translations, and one definition cannot
   disagree with itself. *)

open Op

(* Conv/Linear bias is laid out [1,1,1,1,1,Cout], Cout being the weight's N
   extent — Native's [Eval_op.bias_shape], restated here rather than exported
   from there because it is two lines and the dependency would be the wrong way
   round. *)
let bias_shape ~weight_shape =
  Vec6.set
    (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
    Axis.C
    (Vec6.get weight_shape Axis.N)

module Make (S : Semantics.SEMANTICS) = struct
  (* [output] is the ordinal in [Node.outputs] order. Only [Unbind] reads it —
     every other op in the dialect has exactly one output — but the drivers have
     always threaded a real value, so nothing outside this file changed when it
     stopped being ignored. *)
  let pixel (op : Op.t) ~output ~(operand : Tensor_ref.t -> S.input)
      ~(shape_of : Tensor_ref.t -> Vec6.shape)
      ~(fill : float -> Vec6.shape -> S.input)
      (out : Semantics.position S.index Vec6.t) : S.t =
    (* An absent bias is the identity shift, materialised the same way Native
       does it — [fill] is the one S-specific capability the caller supplies. *)
    let conv_bias weight = function
      | None -> fill 0. (bias_shape ~weight_shape:(shape_of weight))
      | Some b -> operand b
    in
    match op with
    | Add { Pointwise.Bin.a; b } ->
        let module C = Pointwise.Add.Compute (S) in
        C.pixel ~a_shape:(shape_of a) ~b_shape:(shape_of b) (operand a)
          (operand b) out
    | Add_scalar { Pointwise.Scalar_bin.x; scalar } ->
        let module C = Pointwise.Add_scalar.Compute (S) in
        C.pixel ~scalar (operand x) out
    | Adaptive_avg_pool2d { Pool.AdaptiveAvgPool2d.params; x } ->
        let module C = Pool.AdaptiveAvgPool2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Adaptive_max_pool2d { Pool.AdaptiveMaxPool2d.params; x } ->
        let module C = Pool.AdaptiveMaxPool2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Avg_pool2d { Pool.AvgPool2d.params; x } ->
        let module C = Pool.AvgPool2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Batch_norm_no_stats { Ops4.Batch_norm_no_stats.params; x; weight; bias }
      ->
        let module C = Norm.BatchNormNoStats.Compute (S) in
        let fill_or v = function
          | None -> fill v (shape_of x)
          | Some r -> operand r
        in
        C.pixel ~output
          (Graph_shape4.batch_norm_no_stats_params params)
          ~x_shape:(shape_of x) ~x:(operand x) ~weight:(fill_or 1. weight)
          ~bias:(fill_or 0. bias) out
    | Batched_matmul { Matmul.Batched_matmul.input; mat2 } ->
        let module C = Matmul.Batched_matmul.Compute (S) in
        C.pixel ~input_shape:(shape_of input) ~mat2_shape:(shape_of mat2)
          ~input:(operand input) ~mat2:(operand mat2) out
    | Clamp { Pointwise.Clamp.params; x } ->
        let module C = Pointwise.Clamp.Compute (S) in
        C.pixel params (operand x) out
    (* Through the same [concat_params] adapter [Graph_shape4] uses, so the
       axis the shape rule joins along is the axis the compute reads. Every
       operand's own shape is paired with its value, exactly the pairing
       Native's [Concat.Compute.pixel] wants. *)
    | Concat4 { Ops4.Concat4.params; xs } ->
        let module C = Concat.Concat.Compute (S) in
        C.pixel
          (Graph_shape4.concat_params params)
          ~xs:(List.map (fun r -> (shape_of r, operand r)) xs)
          out
    (* The two forward convolutions differ ONLY in the group count they hand
       shared compute — which is the whole content of the dialect's claim that
       grouping belongs in the constructor rather than in the data. *)
    | Conv2d { Ops4.Conv_payload.params; x; weight; bias } ->
        let module C = Conv.Conv2d.Compute (S) in
        C.pixel
          (Graph_shape4.conv2d_params params ~groups:1)
          ~x_shape:(shape_of x) ~weight_shape:(shape_of weight) ~x:(operand x)
          ~weight:(operand weight) ~bias:(conv_bias weight bias) out
    | Depthwise_conv2d { Ops4.Conv_payload.params; x; weight; bias } ->
        let module C = Conv.Conv2d.Compute (S) in
        let groups = Dim.to_int params.Ops4.Conv_params.in_channels in
        C.pixel
          (Graph_shape4.conv2d_params params ~groups)
          ~x_shape:(shape_of x) ~weight_shape:(shape_of weight) ~x:(operand x)
          ~weight:(operand weight) ~bias:(conv_bias weight bias) out
    (* Through the same adapter [Graph_shape4] uses, so the axis the shape
       rule preserves is the axis the compute walks, by construction. *)
    | Cumsum4 { Ops4_cumsum.Cumsum4.params; x } ->
        let module C = Reduce.Cumsum.Compute (S) in
        C.pixel (Graph_shape4.cumsum_params params) ~x:(operand x) out
    | Div { Pointwise.Bin.a; b } ->
        let module C = Pointwise.Div.Compute (S) in
        C.pixel ~a_shape:(shape_of a) ~b_shape:(shape_of b) (operand a)
          (operand b) out
    | Div_scalar { Pointwise.Scalar_bin.x; scalar } ->
        let module C = Pointwise.Div_scalar.Compute (S) in
        C.pixel ~scalar (operand x) out
    | Expand4 { Ops4.Expand4.x; _ } ->
        let module C = Pointwise.Expand.Compute (S) in
        C.pixel ~x_shape:(shape_of x) (operand x) out
    | Gelu { Pointwise.Gelu.x; approximate } ->
        let module C = Pointwise.Gelu.Compute (S) in
        C.pixel approximate (operand x) out
    (* Through [Graph_shape4]'s adapter, so the axis and group count the
       shape rule validated are what the reduction runs over -- the same
       discipline the [Layer_norm] arm above follows, absent operands filled
       the same way. *)
    | Group_norm4 { Ops4.Group_norm4.params; x; weight; bias } ->
        let module C = Norm.GroupNorm.Compute (S) in
        let fill_or v = function
          | None -> fill v (shape_of x)
          | Some r -> operand r
        in
        C.pixel
          (Graph_shape4.group_norm_params params)
          ~x_shape:(shape_of x) ~x:(operand x) ~weight:(fill_or 1. weight)
          ~bias:(fill_or 0. bias) out
    | Grouped_conv2d { Ops4.Grouped_conv_payload.params; x; weight; bias } ->
        let module C = Conv.Conv2d.Compute (S) in
        C.pixel
          (Graph_shape4.grouped_conv2d_params params)
          ~x_shape:(shape_of x) ~weight_shape:(shape_of weight) ~x:(operand x)
          ~weight:(operand weight) ~bias:(conv_bias weight bias) out
    | Hardsigmoid { Pointwise.Hardsigmoid.x } ->
        let module C = Pointwise.Hardsigmoid.Compute (S) in
        C.pixel (operand x) out
    | Hardswish { Pointwise.Hardswish.x } ->
        let module C = Pointwise.Hardswish.Compute (S) in
        C.pixel (operand x) out
    | Hardtanh { Pointwise.Hardtanh.params; x } ->
        let module C = Pointwise.Hardtanh.Compute (S) in
        C.pixel params (operand x) out
    (* Through the same adapter [Graph_shape4] uses, so the axis the shape
       rule overwrites is the axis the compute gathers along, by
       construction. *)
    | IndexTensor4 { Ops4.IndexTensor4.params; self; index } ->
        let module C = Index_tensor.Index_tensor.Compute (S) in
        C.pixel
          (Graph_shape4.index_tensor_params params)
          ~self_shape:(shape_of self) ~self:(operand self)
          ~index:(operand index) out
    | Layer_norm _ -> invalid_arg "Eval_op4.pixel: LayerNorm is Region-authored"
    | Leaky_relu { Pointwise.Leaky_relu.params; x } ->
        let module C = Pointwise.Leaky_relu.Compute (S) in
        C.pixel params (operand x) out
    | Max_keepdims { Ops4.Max_keepdims.params; x } ->
        let module C = Reduce.Amax.Compute (S) in
        C.pixel
          (Graph_shape4.max_params params)
          ~x_shape:(shape_of x) ~x:(operand x) out
    | Max_pool2d { Pool.MaxPool2d.params; x } ->
        let module C = Pool.MaxPool2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Mean_keepdims { Ops4.Mean_keepdims.params; x } ->
        let module C = Reduce.Mean.Compute (S) in
        C.pixel
          (Graph_shape4.mean_params params)
          ~x_shape:(shape_of x) ~x:(operand x) out
    | Mul { Pointwise.Bin.a; b } ->
        let module C = Pointwise.Mul.Compute (S) in
        C.pixel ~a_shape:(shape_of a) ~b_shape:(shape_of b) (operand a)
          (operand b) out
    | Mul_scalar { Pointwise.Scalar_bin.x; scalar } ->
        let module C = Pointwise.Mul_scalar.Compute (S) in
        C.pixel ~scalar (operand x) out
    | Pad4 { Ops4.Pad4.params; x } ->
        let module C = Pad.Pad.Compute (S) in
        C.pixel
          (Graph_shape4.pad_params params)
          ~x_shape:(shape_of x) ~x:(operand x) out
    | Permute4 { Ops4.Permute4.perm; x } ->
        let module C = Permute.Permute.Compute (S) in
        C.pixel (Graph_shape4.perm6 perm) ~x:(operand x) out
    | Pow { Pointwise.Scalar_bin.x; scalar } ->
        let module C = Pointwise.Pow.Compute (S) in
        C.pixel ~scalar (operand x) out
    | Relu { Pointwise.Relu.x } ->
        let module C = Pointwise.Relu.Compute (S) in
        C.pixel (operand x) out
    | Repeat4 { Ops4.Repeat4.x; _ } ->
        let module C = Repeat.Repeat.Compute (S) in
        C.pixel ~x_shape:(shape_of x) (operand x) out
    | RepeatInterleave4 { Ops4.RepeatInterleave4.params; x } ->
        let module C = Repeat.RepeatInterleave.Compute (S) in
        C.pixel (Graph_shape4.repeat_interleave_params params) (operand x) out
    | Reshape4 { Ops4.Reshape4.params; x } ->
        let module C = Reshape.Reshape.Compute (S) in
        C.pixel
          { Reshape.Reshape.shape = Shape4.to_vec6 params.Ops4.Reshape4.shape }
          ~x_shape:(shape_of x) ~x:(operand x) out
    | Rms_norm _ -> invalid_arg "Eval_op4.pixel: RMSNorm is Region-authored"
    | Rsub_scalar { Pointwise.Rsub_scalar.params; x } ->
        let module C = Pointwise.Rsub_scalar.Compute (S) in
        C.pixel params (operand x) out
    | Sdpa _ -> invalid_arg "Eval_op4.pixel: Sdpa is Region-authored"
    (* Through the same adapter [Graph_shape4] uses, so the axis the shape
       rule drops is the axis the compute reads along, by construction --
       [Split.Select.Compute] itself delegates to [Split.Slice.Compute]. *)
    | Select4 { Ops4.Select4.params; x } ->
        let module C = Split.Select.Compute (S) in
        C.pixel (Graph_shape4.select_params params) ~x:(operand x) out
    (* Through the same adapter [Graph_shape4] uses, so the axis the shape
       rule checks [src] against is the axis the compute writes at, by
       construction. *)
    | Select_scatter4 { Ops4.Select_scatter4.params; self; src } ->
        let module C = Split.Select_scatter.Compute (S) in
        C.pixel
          (Graph_shape4.select_scatter_params params)
          ~self:(operand self) ~src:(operand src) out
    | Sigmoid { Pointwise.Sigmoid.x } ->
        let module C = Pointwise.Sigmoid.Compute (S) in
        C.pixel (operand x) out
    | Silu { Pointwise.Silu.x } ->
        let module C = Pointwise.Silu.Compute (S) in
        C.pixel (operand x) out
    | Slice4 { Ops4.Slice4.params; x } ->
        let module C = Split.Slice.Compute (S) in
        C.pixel (Graph_shape4.slice_params params) ~x:(operand x) out
    | Softmax4 _ -> invalid_arg "Eval_op4.pixel: Softmax is Region-authored"
    (* Through the same adapter [Graph_shape4] uses, so the axis and sizes the
       shape rule reads are the ones the compute reads along -- the same
       discipline the [Unbind] arm above follows. [offset] is the sum of every
       EARLIER size, one definition ([Split.Split_with_sizes.offset_of])
       shared with Native's own arm so the two cannot compute it differently. *)
    | Split_with_sizes4 { Ops4.Split_with_sizes4.params; x } ->
        let module C = Split.Split_with_sizes.Compute (S) in
        let native_params = Graph_shape4.split_with_sizes_params params in
        let offset =
          Split.Split_with_sizes.offset_of ~output native_params.sizes
        in
        C.pixel ~offset native_params ~x:(operand x) out
    | Sqrt { Pointwise.Sqrt.x } ->
        let module C = Pointwise.Sqrt.Compute (S) in
        C.pixel (operand x) out
    (* Through the same [stack_params] adapter [Graph_shape4] uses, so the
       axis the shape rule inserts along is the axis the compute reads. No
       per-operand SHAPE pairing, unlike [Concat4]'s arm above -- [Stack]
       selects an operand BY INDEX, not by a within-segment offset, so
       [Concat.Stack.Compute.pixel] wants only the raw operand values. *)
    | Stack4 { Ops4.Stack4.params; xs } ->
        let module C = Concat.Stack.Compute (S) in
        C.pixel (Graph_shape4.stack_params params) ~xs:(List.map operand xs) out
    | Sub { Pointwise.Bin.a; b } ->
        let module C = Pointwise.Sub.Compute (S) in
        C.pixel ~a_shape:(shape_of a) ~b_shape:(shape_of b) (operand a)
          (operand b) out
    | Sum_keepdims { Ops4.Sum_keepdims.params; x } ->
        let module C = Reduce.Sum.Compute (S) in
        C.pixel
          (Graph_shape4.sum_params params)
          ~x_shape:(shape_of x) ~x:(operand x) out
    | To_copy { Pointwise.To_copy.target; x } ->
        let module C = Pointwise.To_copy.Compute (S) in
        C.pixel target (operand x) out
    | Transposed_conv2d { Ops4.Transposed_conv2d.params; x; weight; bias } ->
        let module C = Conv.Convolution.Compute (S) in
        let params = Graph_shape4.transposed_params params in
        let bias =
          match bias with
          | None ->
              (* Transposed output channels come from the weight's C extent, not
                 its N, so [Convolution] has its own bias-shape rule. *)
              fill 0.
                (Conv.Convolution.bias_shape ~weight_shape:(shape_of weight)
                   params)
          | Some b -> operand b
        in
        C.pixel params ~x_shape:(shape_of x) ~weight_shape:(shape_of weight)
          ~x:(operand x) ~weight:(operand weight) ~bias out
    (* Through the SAME adapter [Graph_shape4] uses, so the axis the shape rule
       drops is the axis the compute reads along, by construction. *)
    | Unbind { Ops4.Unbind.params; x } ->
        let module C = Split.Unbind.Compute (S) in
        C.pixel (Graph_shape4.unbind_params params) ~output ~x:(operand x) out
    | Upsample_bilinear2d { Resize.Bilinear2d.params; x } ->
        let module C = Resize.Bilinear2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Upsample_nearest2d { Resize.Nearest2d.params; x } ->
        let module C = Resize.Nearest2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Vector_norm_keepdims { Ops4.Vector_norm_keepdims.params; x } ->
        let module C = Reduce.Vector_norm.Compute (S) in
        C.pixel
          (Graph_shape4.vector_norm_params params)
          ~x_shape:(shape_of x) ~x:(operand x) out
    | Arange4 { Ops4.Arange4.params } ->
        let module C = Factory.Arange.Compute (S) in
        C.pixel
          Factory.Arange.
            {
              start = params.start;
              stop = params.stop;
              step = params.step;
              fmt = params.fmt;
            }
          out
    | Zeros4 { Ops4.Zeros4.params } ->
        let module C = Factory.Zeros.Compute (S) in
        C.pixel
          {
            Factory.Zeros.shape = Shape4.to_vec6 params.shape;
            fmt = params.fmt;
          }
    | Eye4 { Ops4.Eye4.params } ->
        let module C = Factory.Eye.Compute (S) in
        C.pixel
          { Factory.Eye.shape = Shape4.to_vec6 params.shape; fmt = params.fmt }
          out
end
