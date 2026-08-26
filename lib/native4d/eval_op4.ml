(* The single Native4D op-dispatch point, functorised over the semantics [S] —
   the Native4D twin of [Eval_op], applied once at [Direct] and once at
   [Symbolic] so the two execution modes cannot drift.

   EVERY ARM CALLS NATIVE'S [Compute (S)]. .ai/native4d_design.md §2 lists
   "reimplementing numeric kernels already expressed by the Native [Compute (S)]
   functors" as a non-goal, and §6 as the reuse this dialect exists to
   demonstrate: an operation defines a scalar algorithm over
   [Semantics.SEMANTICS], and a second dialect should adapt its parameters and
   delegate rather than duplicate. So there is no arithmetic in this file at
   all — only parameter translation.

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
      | Some b -> operand b
      | None -> fill 0. (bias_shape ~weight_shape:(shape_of weight))
    in
    match op with
    | Add { Pointwise.Bin.a; b } ->
        let module C = Pointwise.Add.Compute (S) in
        C.pixel ~a_shape:(shape_of a) ~b_shape:(shape_of b) (operand a)
          (operand b) out
    | Div { Pointwise.Bin.a; b } ->
        let module C = Pointwise.Div.Compute (S) in
        C.pixel ~a_shape:(shape_of a) ~b_shape:(shape_of b) (operand a)
          (operand b) out
    | Mul { Pointwise.Bin.a; b } ->
        let module C = Pointwise.Mul.Compute (S) in
        C.pixel ~a_shape:(shape_of a) ~b_shape:(shape_of b) (operand a)
          (operand b) out
    | Sub { Pointwise.Bin.a; b } ->
        let module C = Pointwise.Sub.Compute (S) in
        C.pixel ~a_shape:(shape_of a) ~b_shape:(shape_of b) (operand a)
          (operand b) out
    | Add_scalar { Pointwise.Scalar_bin.x; scalar } ->
        let module C = Pointwise.Add_scalar.Compute (S) in
        C.pixel ~scalar (operand x) out
    | Div_scalar { Pointwise.Scalar_bin.x; scalar } ->
        let module C = Pointwise.Div_scalar.Compute (S) in
        C.pixel ~scalar (operand x) out
    | Mul_scalar { Pointwise.Scalar_bin.x; scalar } ->
        let module C = Pointwise.Mul_scalar.Compute (S) in
        C.pixel ~scalar (operand x) out
    | Pow { Pointwise.Scalar_bin.x; scalar } ->
        let module C = Pointwise.Pow.Compute (S) in
        C.pixel ~scalar (operand x) out
    | Clamp { Pointwise.Clamp.params; x } ->
        let module C = Pointwise.Clamp.Compute (S) in
        C.pixel params (operand x) out
    | Gelu { Pointwise.Gelu.x; approximate } ->
        let module C = Pointwise.Gelu.Compute (S) in
        C.pixel approximate (operand x) out
    | Hardsigmoid { Pointwise.Hardsigmoid.x } ->
        let module C = Pointwise.Hardsigmoid.Compute (S) in
        C.pixel (operand x) out
    | Hardswish { Pointwise.Hardswish.x } ->
        let module C = Pointwise.Hardswish.Compute (S) in
        C.pixel (operand x) out
    | Hardtanh { Pointwise.Hardtanh.params; x } ->
        let module C = Pointwise.Hardtanh.Compute (S) in
        C.pixel params (operand x) out
    | Relu { Pointwise.Relu.x } ->
        let module C = Pointwise.Relu.Compute (S) in
        C.pixel (operand x) out
    | Sigmoid { Pointwise.Sigmoid.x } ->
        let module C = Pointwise.Sigmoid.Compute (S) in
        C.pixel (operand x) out
    | Silu { Pointwise.Silu.x } ->
        let module C = Pointwise.Silu.Compute (S) in
        C.pixel (operand x) out
    | Sqrt { Pointwise.Sqrt.x } ->
        let module C = Pointwise.Sqrt.Compute (S) in
        C.pixel (operand x) out
    | Avg_pool2d { Pool.AvgPool2d.params; x } ->
        let module C = Pool.AvgPool2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Adaptive_avg_pool2d { Pool.AdaptiveAvgPool2d.params; x } ->
        let module C = Pool.AdaptiveAvgPool2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Max_pool2d { Pool.MaxPool2d.params; x } ->
        let module C = Pool.MaxPool2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
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
    | Transposed_conv2d { Ops4.Transposed_conv2d.params; x; weight; bias } ->
        let module C = Conv.Convolution.Compute (S) in
        let params = Graph_shape4.transposed_params params in
        let bias =
          match bias with
          | Some b -> operand b
          | None ->
              (* Transposed output channels come from the weight's C extent, not
                 its N, so [Convolution] has its own bias-shape rule. *)
              fill 0.
                (Conv.Convolution.bias_shape ~weight_shape:(shape_of weight)
                   params)
        in
        C.pixel params ~x_shape:(shape_of x) ~weight_shape:(shape_of weight)
          ~x:(operand x) ~weight:(operand weight) ~bias out
    | Mean_keepdims { Ops4.Mean_keepdims.params; x } ->
        let module C = Reduce.Mean.Compute (S) in
        C.pixel
          (Graph_shape4.mean_params params)
          ~x_shape:(shape_of x) ~x:(operand x) out
    | Max_keepdims { Ops4.Max_keepdims.params; x } ->
        let module C = Reduce.Amax.Compute (S) in
        C.pixel
          (Graph_shape4.max_params params)
          ~x_shape:(shape_of x) ~x:(operand x) out
    (* Through the SAME adapter [Graph_shape4] uses, so the axis the shape rule
       drops is the axis the compute reads along, by construction. *)
    | Unbind { Ops4.Unbind.params; x } ->
        let module C = Split.Unbind.Compute (S) in
        C.pixel (Graph_shape4.unbind_params params) ~output ~x:(operand x) out
    (* Through [Graph_shape4]'s adapter, so the axes the shape rule validated
       are the axes the reduction runs over. The absent operands are filled
       exactly as [Eval_op] fills them -- 1 for the scale, 0 for the offset --
       so an absent operand means the same thing in both dialects. *)
    | Layer_norm { Ops4.Layer_norm.params; x; weight; bias } ->
        let module C = Norm.LayerNorm.Compute (S) in
        let fill_or v = function
          | Some r -> operand r
          | None -> fill v (shape_of x)
        in
        C.pixel
          (Graph_shape4.layer_norm_params params)
          ~x_shape:(shape_of x) ~x:(operand x) ~weight:(fill_or 1. weight)
          ~bias:(fill_or 0. bias) out
    | Rms_norm { Ops4.Rms_norm.params; x; weight } ->
        let module C = Norm.RmsNorm.Compute (S) in
        let weight =
          match weight with Some w -> operand w | None -> fill 1. (shape_of x)
          (* absent weight = identity scale *)
        in
        C.pixel
          (Graph_shape4.rms_params params)
          ~x_shape:(shape_of x) ~x:(operand x) ~weight out
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
    | Pad4 { Ops4.Pad4.params; x } ->
        let module C = Pad.Pad.Compute (S) in
        C.pixel
          (Graph_shape4.pad_params params)
          ~x_shape:(shape_of x) ~x:(operand x) out
    | Slice4 { Ops4.Slice4.params; x } ->
        let module C = Split.Slice.Compute (S) in
        C.pixel (Graph_shape4.slice_params params) ~x:(operand x) out
    | Permute4 { Ops4.Permute4.perm; x } ->
        let module C = Permute.Permute.Compute (S) in
        C.pixel (Graph_shape4.perm6 perm) ~x:(operand x) out
    | Reshape4 { Ops4.Reshape4.params; x } ->
        let module C = Reshape.Reshape.Compute (S) in
        C.pixel
          { Reshape.Reshape.shape = Shape4.to_vec6 params.Ops4.Reshape4.shape }
          ~x_shape:(shape_of x) ~x:(operand x) out
    | Upsample_bilinear2d { Resize.Bilinear2d.params; x } ->
        let module C = Resize.Bilinear2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Vector_norm_keepdims { Ops4.Vector_norm_keepdims.params; x } ->
        let module C = Reduce.Vector_norm.Compute (S) in
        C.pixel
          (Graph_shape4.vector_norm_params params)
          ~x_shape:(shape_of x) ~x:(operand x) out
end
