(* The single op-dispatch point, functorised over the semantics [S]. Applied once
   at [Direct] and once at [Symbolic], so the per-op wiring lives in exactly one
   place and cannot drift between the two execution modes.

   Each arm reads its operands off the op's TYPED FIELDS and resolves each
   [tensor_ref] to a data handle ([operand]) and shape ([shape_of]); an absent
   optional operand becomes a constant-filled handle via [fill] (zeros for a
   missing bias, ones for a missing rms-norm weight). [fill] is the only
   S-specific extra capability the caller supplies — Direct fills a real tensor,
   Symbolic mints a signature it will bind to a constant at ground time. See
   .ai/native_graph_design.md. *)

open Graph_ir

(* Conv/Linear bias is laid out [1,1,1,1,1,Cout]; Cout is the weight's N extent.
   [Affine_bias] owns that definition, so the zero this synthesizes for an
   absent operand and the shape [Graph_shape] rejects a present one against
   cannot come to disagree. *)
let bias_shape ~weight_shape = Affine_bias.shape ~weight_shape

module Make (S : Semantics.SEMANTICS) = struct
  (* [output] selects which output edge's pixel to build, for multi-output ops
     (0-based, in [Node.outputs] order). Single-output ops ignore it. *)
  let pixel (op : op) ~output ~(operand : tensor_ref -> S.input)
      ~(shape_of : tensor_ref -> Vec6.shape)
      ~(fill : float -> Vec6.shape -> S.input)
      (out : Semantics.position S.index Vec6.t) : S.t =
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
    | Amax { Reduce.Amax.params; x } ->
        let module C = Reduce.Amax.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Avg_pool2d { Pool.AvgPool2d.params; x } ->
        let module C = Pool.AvgPool2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Batch_norm
        { Norm.BatchNorm.params; x; weight; bias; running_mean; running_var } ->
        let module C = Norm.BatchNorm.Compute (S) in
        let weight =
          match weight with Some w -> operand w | None -> fill 1. (shape_of x)
          (* absent weight = identity scale *)
        in
        let bias =
          match bias with Some b -> operand b | None -> fill 0. (shape_of x)
          (* absent bias = identity shift *)
        in
        C.pixel params ~x:(operand x) ~weight ~bias
          ~running_mean:(operand running_mean)
          ~running_var:(operand running_var) out
    | Bmm { Matmul.Bmm.input; mat2 } ->
        let module C = Matmul.Bmm.Compute (S) in
        C.pixel ~input_shape:(shape_of input) ~input:(operand input)
          ~mat2:(operand mat2) out
    | Clamp { Pointwise.Clamp.params; x } ->
        let module C = Pointwise.Clamp.Compute (S) in
        C.pixel params (operand x) out
    | Clone { Pointwise.Clone.x } ->
        let module C = Pointwise.Clone.Compute (S) in
        C.pixel (operand x) out
    | Concat { Concat.Concat.params; xs } ->
        let module C = Concat.Concat.Compute (S) in
        C.pixel params ~xs:(List.map (fun r -> (shape_of r, operand r)) xs) out
    | Conv2d { Conv.Conv2d.params; x; weight; bias } ->
        let module C = Conv.Conv2d.Compute (S) in
        let bias =
          match bias with
          | None -> fill 0. (bias_shape ~weight_shape:(shape_of weight))
          | Some b -> operand b
        in
        C.pixel params ~x_shape:(shape_of x) ~weight_shape:(shape_of weight)
          ~x:(operand x) ~weight:(operand weight) ~bias out
    | Conv2d_padding { Conv.Conv2d_padding.params; x; weight; bias } ->
        let module C = Conv.Conv2d_padding.Compute (S) in
        let bias =
          match bias with
          | None -> fill 0. (bias_shape ~weight_shape:(shape_of weight))
          | Some b -> operand b
        in
        C.pixel params ~x_shape:(shape_of x) ~weight_shape:(shape_of weight)
          ~x:(operand x) ~weight:(operand weight) ~bias out
    | Convolution { Conv.Convolution.params; x; weight; bias } ->
        let module C = Conv.Convolution.Compute (S) in
        let bias =
          match bias with
          | None ->
              fill 0.
                (Conv.Convolution.bias_shape ~weight_shape:(shape_of weight)
                   params)
          | Some b -> operand b
        in
        C.pixel params ~x_shape:(shape_of x) ~weight_shape:(shape_of weight)
          ~x:(operand x) ~weight:(operand weight) ~bias out
    | Expand { Pointwise.Expand.params = _; x } ->
        let module C = Pointwise.Expand.Compute (S) in
        C.pixel ~x_shape:(shape_of x) (operand x) out
    | Gelu { Pointwise.Gelu.x; approximate } ->
        let module C = Pointwise.Gelu.Compute (S) in
        C.pixel approximate (operand x) out
    | Group_norm { Norm.GroupNorm.params; x; weight; bias } ->
        let module C = Norm.GroupNorm.Compute (S) in
        (* Absent weight = identity scale, absent bias = identity shift,
           filled HERE for the reason the [Layer_norm] arm's comment gives. *)
        let weight =
          match weight with None -> fill 1. (shape_of x) | Some w -> operand w
        in
        let bias =
          match bias with None -> fill 0. (shape_of x) | Some b -> operand b
        in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) ~weight ~bias out
    | Hardsigmoid { Pointwise.Hardsigmoid.x } ->
        let module C = Pointwise.Hardsigmoid.Compute (S) in
        C.pixel (operand x) out
    | Hardswish { Pointwise.Hardswish.x } ->
        let module C = Pointwise.Hardswish.Compute (S) in
        C.pixel (operand x) out
    | Hardtanh { Pointwise.Hardtanh.params; x } ->
        let module C = Pointwise.Hardtanh.Compute (S) in
        C.pixel params (operand x) out
    | Layer_norm { Norm.LayerNorm.params; x; weight; bias } ->
        let module C = Norm.LayerNorm.Compute (S) in
        (* Absent weight = identity scale, absent bias = identity shift. Filled
           HERE, never in an importer: [Graph_ir] carries both as options and
           Native4D reads the options, so materialising a constant upstream
           would make the two import paths build structurally different graphs
           for the same node. *)
        let weight =
          match weight with None -> fill 1. (shape_of x) | Some w -> operand w
        in
        let bias =
          match bias with None -> fill 0. (shape_of x) | Some b -> operand b
        in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) ~weight ~bias out
    | Linear { Linear.Linear.params; x; weight; bias } ->
        let module C = Linear.Linear.Compute (S) in
        let bias =
          match bias with
          | None -> fill 0. (bias_shape ~weight_shape:(shape_of weight))
          | Some b -> operand b
        in
        C.pixel params ~x:(operand x) ~weight:(operand weight) ~bias out
    | Max_pool2d { Pool.MaxPool2d.params; x } ->
        let module C = Pool.MaxPool2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Max_pool2d_with_indices { Pool.MaxPool2dWithIndices.params; x } ->
        let module C = Pool.MaxPool2dWithIndices.Compute (S) in
        let pix = if output = 0 then C.value_pixel else C.index_pixel in
        pix params ~x_shape:(shape_of x) ~x:(operand x) out
    | Mean { Reduce.Mean.params; x } ->
        let module C = Reduce.Mean.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Div { Pointwise.Bin.a; b } ->
        let module C = Pointwise.Div.Compute (S) in
        C.pixel ~a_shape:(shape_of a) ~b_shape:(shape_of b) (operand a)
          (operand b) out
    | Div_scalar { Pointwise.Scalar_bin.x; scalar } ->
        let module C = Pointwise.Div_scalar.Compute (S) in
        C.pixel ~scalar (operand x) out
    | Mul { Pointwise.Bin.a; b } ->
        let module C = Pointwise.Mul.Compute (S) in
        C.pixel ~a_shape:(shape_of a) ~b_shape:(shape_of b) (operand a)
          (operand b) out
    | Mul_scalar { Pointwise.Scalar_bin.x; scalar } ->
        let module C = Pointwise.Mul_scalar.Compute (S) in
        C.pixel ~scalar (operand x) out
    | Pad { Pad.Pad.params; x } ->
        let module C = Pad.Pad.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Permute { Permute.Permute.perm; x } ->
        let module C = Permute.Permute.Compute (S) in
        C.pixel perm ~x:(operand x) out
    | Pow { Pointwise.Scalar_bin.x; scalar } ->
        let module C = Pointwise.Pow.Compute (S) in
        C.pixel ~scalar (operand x) out
    | Relu { Pointwise.Relu.x } ->
        let module C = Pointwise.Relu.Compute (S) in
        C.pixel (operand x) out
    | Reshape { Reshape.Reshape.params; x } ->
        let module C = Reshape.Reshape.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Rms_norm { Norm.RmsNorm.params; x; weight } ->
        let module C = Norm.RmsNorm.Compute (S) in
        let weight =
          match weight with None -> fill 1. (shape_of x) | Some w -> operand w
          (* absent weight = identity scale *)
        in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) ~weight out
    | Sdpa { Attention.Sdpa.params; query; key; value; mask } ->
        let module C = Attention.Sdpa.Compute (S) in
        (* Absent mask fills a single element, shaped [1,1,1,1,1,1] (every
           axis extent-1, so numel = 1), not the score shape (op8-impl.md
           F11): [fill] allocates a real tensor under [Direct], and
           [broadcast_coord] maps every axis of a ones-SHAPED tensor to index
           0, so this one element serves every read regardless of the score's
           real extent -- a score-shaped fill would cost D*H*Wq*Wk floats to
           represent nothing. The fill VALUE is 0.0, the additive identity
           (add nothing when no mask was given), not 1. *)
        let mask_shape, mask =
          match mask with
          | None ->
              let ones = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
              (ones, fill 0. ones)
          | Some m -> (shape_of m, operand m)
        in
        C.pixel params ~query_shape:(shape_of query) ~key_shape:(shape_of key)
          ~mask_shape ~query:(operand query) ~key:(operand key)
          ~value:(operand value) ~mask out
    | Select { Split.Select.params; x } ->
        let module C = Split.Select.Compute (S) in
        C.pixel params ~x:(operand x) out
    | Sigmoid { Pointwise.Sigmoid.x } ->
        let module C = Pointwise.Sigmoid.Compute (S) in
        C.pixel (operand x) out
    | Silu { Pointwise.Silu.x } ->
        let module C = Pointwise.Silu.Compute (S) in
        C.pixel (operand x) out
    | Softmax { Reduce.Softmax.params; x } ->
        let module C = Reduce.Softmax.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Sqrt { Pointwise.Sqrt.x } ->
        let module C = Pointwise.Sqrt.Compute (S) in
        C.pixel (operand x) out
    | Stack { Concat.Stack.params; xs } ->
        let module C = Concat.Stack.Compute (S) in
        C.pixel params ~xs:(List.map operand xs) out
    | Sub { Pointwise.Bin.a; b } ->
        let module C = Pointwise.Sub.Compute (S) in
        C.pixel ~a_shape:(shape_of a) ~b_shape:(shape_of b) (operand a)
          (operand b) out
    | Sum { Reduce.Sum.params; x } ->
        let module C = Reduce.Sum.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Slice { Split.Slice.params; x } ->
        let module C = Split.Slice.Compute (S) in
        C.pixel params ~x:(operand x) out
    (* Homogeneous multi-output: every ordinal runs the same algorithm and
       differs only in the coordinate it reads, so [~output] goes in as data
       rather than selecting between named pixel functions the way
       [Max_pool2d_with_indices] does above. *)
    | Unbind { Split.Unbind.params; x } ->
        let module C = Split.Unbind.Compute (S) in
        C.pixel params ~output ~x:(operand x) out
    (* [offset] is the sum of every EARLIER piece's size -- [output]'s ordinal
       picks the prefix, [Compute.pixel] wants only the sum. *)
    | Split_with_sizes { Split.Split_with_sizes.params; x } ->
        let module C = Split.Split_with_sizes.Compute (S) in
        let offset =
          Split.Split_with_sizes.offset_of ~output
            params.Split.Split_with_sizes.sizes
        in
        C.pixel ~offset params ~x:(operand x) out
    | Upsample_bilinear2d { Resize.Bilinear2d.params; x } ->
        let module C = Resize.Bilinear2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Upsample_nearest2d { Resize.Nearest2d.params; x } ->
        let module C = Resize.Nearest2d.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Vector_norm { Reduce.Vector_norm.params; x } ->
        let module C = Reduce.Vector_norm.Compute (S) in
        C.pixel params ~x_shape:(shape_of x) ~x:(operand x) out
    | Discard _ ->
        invalid_arg
          "Eval_op.pixel: Discard produces no output, so it has no pixel"
end
