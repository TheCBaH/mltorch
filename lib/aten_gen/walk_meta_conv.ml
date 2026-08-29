(* Split out of walk_meta.ml see walk_meta_entry.ml for
   the [t] record these build. *)

open Walk_meta_entry

let conv2d =
  {
    module_name = "Conv2d_walk";
    target = "torch.ops.aten.conv2d.default";
    recipe = "Recipe_conv";
    initial =
      "Aten_walk_recipes.Recipe_conv.{ kernel_h = 3; kernel_w = 3; stride_h = \
       1; stride_w = 1; pad_h = 1; pad_w = 1; dilation_h = 1; dilation_w = 1; \
       groups = 1; in_channels = 4; out_channels = 8; n = 1; input_h = 8; \
       input_w = 8 }";
    axes =
      "Aten_walk_recipes.Recipe_conv.axes ~kernel_h:[ 1; 2; 3; 4; 5 ] \
       ~kernel_w:[ 1; 2; 3; 4; 5 ] ~stride_h:[ 1; 2; 3 ] ~stride_w:[ 1; 2; 3 ] \
       ~pad_h:[ 0; 1; 2 ] ~pad_w:[ 0; 1; 2 ] ~dilation_h:[ 1; 2; 3 ] \
       ~dilation_w:[ 1; 2; 3 ] ~groups:[ 1; 2; 4 ] ~in_channels:[ 4; 8; 12; 16 \
       ] ~out_channels:[ 4; 8; 12; 16 ] ~n:[ 1; 2 ] ~input_h:[ 8; 10; 12; 14; \
       16 ] ~input_w:[ 8; 10; 12; 14; 16 ]";
    build =
      {|let input, pcg = Walk.tensor_spec pcg (Recipe_conv.input_shape c) in
    let weight, pcg = Walk.tensor_spec pcg (Recipe_conv.weight_shape c) in
    let bias, pcg = Walk.tensor_spec pcg (Recipe_conv.bias_shape c) in
    ( Aten_op_spec.Op_conv2d.(
        spec
          {
            input;
            weight;
            bias = Some bias;
            stride = Recipe_conv.strides c;
            padding = Recipe_conv.pads c;
            dilation = Recipe_conv.dilations c;
            groups = Recipe_conv.groups c;
          }),
      pcg )|};
  }

(* convolution.default: a non-transposed grouped conv -- identical shape
   semantics to conv2d, so it reuses Recipe_conv and pins transposed=false /
   output_padding=[0;0] (PyTorch requires output_padding=0 when not transposed). *)
let convolution =
  {
    module_name = "Convolution_walk";
    target = "torch.ops.aten.convolution.default";
    recipe = "Recipe_conv";
    initial =
      "Aten_walk_recipes.Recipe_conv.{ kernel_h = 3; kernel_w = 3; stride_h = \
       1; stride_w = 1; pad_h = 1; pad_w = 1; dilation_h = 1; dilation_w = 1; \
       groups = 1; in_channels = 4; out_channels = 8; n = 1; input_h = 8; \
       input_w = 8 }";
    axes =
      "Aten_walk_recipes.Recipe_conv.axes ~kernel_h:[ 1; 2; 3; 4; 5 ] \
       ~kernel_w:[ 1; 2; 3; 4; 5 ] ~stride_h:[ 1; 2; 3 ] ~stride_w:[ 1; 2; 3 ] \
       ~pad_h:[ 0; 1; 2 ] ~pad_w:[ 0; 1; 2 ] ~dilation_h:[ 1; 2; 3 ] \
       ~dilation_w:[ 1; 2; 3 ] ~groups:[ 1; 2; 4 ] ~in_channels:[ 4; 8; 12; 16 \
       ] ~out_channels:[ 4; 8; 12; 16 ] ~n:[ 1; 2 ] ~input_h:[ 8; 10; 12; 14; \
       16 ] ~input_w:[ 8; 10; 12; 14; 16 ]";
    build =
      {|let input, pcg = Walk.tensor_spec pcg (Recipe_conv.input_shape c) in
    let weight, pcg = Walk.tensor_spec pcg (Recipe_conv.weight_shape c) in
    let bias, pcg = Walk.tensor_spec pcg (Recipe_conv.bias_shape c) in
    ( Aten_op_spec.Op_convolution.(
        spec
          {
            input;
            weight;
            bias = Some bias;
            stride = Recipe_conv.strides c;
            padding = Recipe_conv.pads c;
            dilation = Recipe_conv.dilations c;
            transposed = false;
            output_padding = [ 0; 0 ];
            groups = Recipe_conv.groups c;
          }),
      pcg )|};
  }

(* conv2d.padding: string padding mode. Odd kernels keep "same" well-behaved. *)
let conv2d_padding =
  {
    module_name = "Conv2d_padding_walk";
    target = "torch.ops.aten.conv2d.padding";
    recipe = "Recipe_conv_padding";
    initial =
      "Aten_walk_recipes.Recipe_conv_padding.{ kernel_h = 3; kernel_w = 3; \
       stride_h = 1; stride_w = 1; dilation_h = 1; dilation_w = 1; groups = 1; \
       in_channels = 4; out_channels = 8; n = 1; input_h = 8; input_w = 8; \
       padding = \"same\" }";
    axes =
      "Aten_walk_recipes.Recipe_conv_padding.axes ~kernel_h:[ 1; 3; 5 ] \
       ~kernel_w:[ 1; 3; 5 ] ~stride_h:[ 1; 2 ] ~stride_w:[ 1; 2 ] \
       ~dilation_h:[ 1; 2 ] ~dilation_w:[ 1; 2 ] ~groups:[ 1; 2; 4 ] \
       ~in_channels:[ 4; 8; 16 ] ~out_channels:[ 4; 8; 16 ] ~n:[ 1; 2 ] \
       ~input_h:[ 8; 10; 12 ] ~input_w:[ 8; 10; 12 ] ~padding:[ \"valid\"; \
       \"same\" ]";
    build =
      {|let input, pcg = Walk.tensor_spec pcg (Recipe_conv_padding.input_shape c) in
    let weight, pcg = Walk.tensor_spec pcg (Recipe_conv_padding.weight_shape c) in
    let bias, pcg = Walk.tensor_spec pcg (Recipe_conv_padding.bias_shape c) in
    ( Aten_op_spec.Op_conv2d_padding.(
        spec
          {
            input;
            weight;
            bias = Some bias;
            stride = Recipe_conv_padding.strides c;
            padding = Recipe_conv_padding.padding c;
            dilation = Recipe_conv_padding.dilations c;
            groups = Recipe_conv_padding.groups c;
          }),
      pcg )|};
  }
