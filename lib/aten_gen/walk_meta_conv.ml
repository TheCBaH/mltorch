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

let conv1d =
  {
    module_name = "Conv1d_walk";
    target = "torch.ops.aten.conv1d.default";
    recipe = "Recipe_conv1d";
    initial =
      "Aten_walk_recipes.Recipe_conv1d.{ \
       k=3;s=1;p=1;d=1;g=1;inc=4;outc=8;n=1;len=8 }";
    axes =
      "Aten_walk_recipes.Recipe_conv1d.axes ~k:[1;2;3;5] ~s:[1;2;3] ~p:[0;1;2] \
       ~d:[1;2] ~g:[1;2;4] ~inc:[4;8;12] ~outc:[4;8;12] ~n:[1;2] \
       ~len:[8;10;12]";
    build =
      {|let input, pcg = Walk.tensor_spec pcg (Recipe_conv1d.input_shape c) in
    let weight, pcg = Walk.tensor_spec pcg (Recipe_conv1d.weight_shape c) in
    let bias, pcg = Walk.tensor_spec pcg (Recipe_conv1d.bias_shape c) in
    (Aten_op_spec.Op_conv1d.(spec {input; weight; bias=Some bias; stride=[c.s]; padding=[c.p]; dilation=[c.d]; groups=c.g}),pcg)|};
  }

(* Conv3d is bridge-supported but intentionally Native4D-bounded.  Keep channels
   fixed and vary batch plus D/H/W through Recipe_unbind's valid shape axes. *)
let conv3d =
  {
    module_name = "Conv3d_walk";
    target = "torch.ops.aten.conv3d.default";
    recipe = "Recipe_unbind";
    initial =
      "Aten_walk_recipes.Recipe_unbind.{ n = 1; c = 4; h = 4; w = 4; dim = 0 }";
    axes =
      "Aten_walk_recipes.Recipe_unbind.axes ~n:[ 1; 2 ] ~c:[ 3; 4; 6 ] ~h:[ 3; \
       4; 6 ] ~w:[ 3; 4; 6 ] ~dim:[ 0 ]";
    build =
      {|let n, d, h, w = match Recipe_unbind.self_shape c with [ n; d; h; w ] -> (n, d, h, w) | _ -> assert false in
    let input, pcg = Walk.tensor_spec pcg [ n; 4; d; h; w ] in
    let weight, pcg = Walk.tensor_spec pcg [ 8; 4; 3; 3; 3 ] in
    let bias, pcg = Walk.tensor_spec pcg [ 8 ] in
    (Aten_op_spec.Op_conv3d.(spec { input; weight; bias = Some bias;
       stride = [1;1;1]; padding = [1;1;1]; dilation = [1;1;1]; groups = 1 }), pcg)|};
  }
