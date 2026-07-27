(* Hand-crafted meta augmenting native_functions.yaml for the random-walk
   generator (Aten_walk_gen). For ops whose argument shapes are correlated in
   ways the yaml can't express (conv's weight shape, a pool window, a reduction
   over dims), this names the family recipe and supplies the initial config, the
   per-axis candidate sets, and the build body that fills the generated
   Op_<name> record. The generator wraps each entry in the [Walk.Op] skeleton
   (cfg/cascade/pp come from the recipe) and checks [target] resolves to a real
   generated spec, so the meta can't drift from the bindings. *)

type t = {
  module_name : string;  (** emitted walk module name, e.g. "Conv2d_walk" *)
  target : string;  (** op target; must match a generated Op_*.spec *)
  recipe : string;
      (** recipe module under Aten_walk_recipes, e.g. "Recipe_conv" *)
  initial : string;  (** OCaml expr: the initial cfg *)
  axes : string;  (** OCaml expr: the axes list *)
  build : string;
      (** OCaml expr: build body, with [pcg], [c] and an open Aten_walk_recipes
          in scope, returning [Op_spec.t * Pcg.t] *)
}

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

let max_pool2d =
  {
    module_name = "Max_pool2d_walk";
    target = "torch.ops.aten.max_pool2d.default";
    recipe = "Recipe_pool";
    initial =
      "Aten_walk_recipes.Recipe_pool.{ kernel_h = 2; kernel_w = 2; stride_h = \
       2; stride_w = 2; pad_h = 0; pad_w = 0; n = 1; c = 4; input_h = 8; \
       input_w = 8 }";
    axes =
      "Aten_walk_recipes.Recipe_pool.axes ~kernel_h:[ 2; 3; 4 ] ~kernel_w:[ 2; \
       3; 4 ] ~stride_h:[ 1; 2; 3 ] ~stride_w:[ 1; 2; 3 ] ~pad_h:[ 0; 1 ] \
       ~pad_w:[ 0; 1 ] ~n:[ 1; 2 ] ~c:[ 4; 8; 16 ] ~input_h:[ 8; 10; 12; 16 ] \
       ~input_w:[ 8; 10; 12; 16 ]";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_pool.self_shape c) in
    ( Aten_op_spec.Op_max_pool2d.(
        spec
          {
            self;
            kernel_size = Recipe_pool.kernel_size c;
            stride = Recipe_pool.strides c;
            padding = Recipe_pool.pads c;
            dilation = Recipe_pool.dilation c;
            ceil_mode = Recipe_pool.ceil_mode c;
          }),
      pcg )|};
  }

let mean_dim =
  {
    module_name = "Mean_dim_walk";
    target = "torch.ops.aten.mean.dim";
    recipe = "Recipe_reduce";
    initial =
      "Aten_walk_recipes.Recipe_reduce.{ n = 2; c = 4; h = 8; w = 8; dims = [ \
       2; 3 ]; keepdim = false }";
    axes =
      "Aten_walk_recipes.Recipe_reduce.axes ~n:[ 1; 2; 4 ] ~c:[ 4; 8; 16 ] \
       ~h:[ 4; 8; 16 ] ~w:[ 4; 8; 16 ] \
       ~dims:Aten_walk_recipes.Recipe_reduce.all_dim_subsets ~keepdim:[ true; \
       false ]";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_reduce.self_shape c) in
    ( Aten_op_spec.Op_mean_dim.(
        spec
          {
            self;
            dim = Some (Recipe_reduce.dims c);
            keepdim = Recipe_reduce.keepdim c;
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

let max_pool2d_with_indices =
  {
    module_name = "Max_pool2d_with_indices_walk";
    target = "torch.ops.aten.max_pool2d_with_indices.default";
    recipe = "Recipe_pool";
    initial =
      "Aten_walk_recipes.Recipe_pool.{ kernel_h = 2; kernel_w = 2; stride_h = \
       2; stride_w = 2; pad_h = 0; pad_w = 0; n = 1; c = 4; input_h = 8; \
       input_w = 8 }";
    axes =
      "Aten_walk_recipes.Recipe_pool.axes ~kernel_h:[ 2; 3; 4 ] ~kernel_w:[ 2; \
       3; 4 ] ~stride_h:[ 1; 2; 3 ] ~stride_w:[ 1; 2; 3 ] ~pad_h:[ 0; 1 ] \
       ~pad_w:[ 0; 1 ] ~n:[ 1; 2 ] ~c:[ 4; 8; 16 ] ~input_h:[ 8; 10; 12; 16 ] \
       ~input_w:[ 8; 10; 12; 16 ]";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_pool.self_shape c) in
    ( Aten_op_spec.Op_max_pool2d_with_indices.(
        spec
          {
            self;
            kernel_size = Recipe_pool.kernel_size c;
            stride = Recipe_pool.strides c;
            padding = Recipe_pool.pads c;
            dilation = Recipe_pool.dilation c;
            ceil_mode = Recipe_pool.ceil_mode c;
          }),
      pcg )|};
  }

let avg_pool2d =
  {
    module_name = "Avg_pool2d_walk";
    target = "torch.ops.aten.avg_pool2d.default";
    recipe = "Recipe_pool";
    initial =
      "Aten_walk_recipes.Recipe_pool.{ kernel_h = 2; kernel_w = 2; stride_h = \
       2; stride_w = 2; pad_h = 0; pad_w = 0; n = 1; c = 4; input_h = 8; \
       input_w = 8 }";
    axes =
      "Aten_walk_recipes.Recipe_pool.axes ~kernel_h:[ 2; 3; 4 ] ~kernel_w:[ 2; \
       3; 4 ] ~stride_h:[ 1; 2; 3 ] ~stride_w:[ 1; 2; 3 ] ~pad_h:[ 0; 1 ] \
       ~pad_w:[ 0; 1 ] ~n:[ 1; 2 ] ~c:[ 4; 8; 16 ] ~input_h:[ 8; 10; 12; 16 ] \
       ~input_w:[ 8; 10; 12; 16 ]";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_pool.self_shape c) in
    ( Aten_op_spec.Op_avg_pool2d.(
        spec
          {
            self;
            kernel_size = Recipe_pool.kernel_size c;
            stride = Recipe_pool.strides c;
            padding = Recipe_pool.pads c;
            ceil_mode = Recipe_pool.ceil_mode c;
            count_include_pad = true;
            divisor_override = None;
          }),
      pcg )|};
  }

let adaptive_avg_pool2d =
  {
    module_name = "Adaptive_avg_pool2d_walk";
    target = "torch.ops.aten.adaptive_avg_pool2d.default";
    recipe = "Recipe_adaptive";
    initial =
      "Aten_walk_recipes.Recipe_adaptive.{ n = 1; c = 4; input_h = 8; input_w \
       = 8; out_h = 4; out_w = 4 }";
    axes =
      "Aten_walk_recipes.Recipe_adaptive.axes ~n:[ 1; 2 ] ~c:[ 4; 8; 16 ] \
       ~input_h:[ 6; 8; 10; 12 ] ~input_w:[ 6; 8; 10; 12 ] ~out_h:[ 1; 2; 4; 7 \
       ] ~out_w:[ 1; 2; 4; 7 ]";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_adaptive.self_shape c) in
    ( Aten_op_spec.Op_adaptive_avg_pool2d.(
        spec { self; output_size = Recipe_adaptive.output_size c }),
      pcg )|};
  }

(* clamp.default has no generated default walk: both its bounds default to None,
   and that is precisely the pair ATen rejects, so the generator has nothing
   valid to synthesise and leaves it in [needs_meta]. This entry supplies whole
   valid bound pairs, which is what gets clamp compared against real ATen. *)
let clamp =
  {
    module_name = "Clamp_walk";
    target = "torch.ops.aten.clamp.default";
    recipe = "Recipe_bounds";
    initial =
      "Aten_walk_recipes.Recipe_bounds.{ n = 1; c = 4; h = 8; w = 8; bounds = \
       { min = Some (Aten_spec.Scalar_value.Int 0); max = Some \
       (Aten_spec.Scalar_value.Int 6) } }";
    axes =
      "Aten_walk_recipes.Recipe_bounds.axes ~n:[ 1; 2 ] ~c:[ 3; 4; 8 ] ~h:[ 4; \
       8 ] ~w:[ 4; 8 ] ~bounds:Aten_walk_recipes.Recipe_bounds.optional_pairs";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_bounds.self_shape c) in
    ( Aten_op_spec.Op_clamp.(
        spec
          {
            self;
            min = Recipe_bounds.min c;
            max = Recipe_bounds.max c;
          }),
      pcg )|};
  }

(* hardtanh.default DOES get a generated default walk, but that only ever
   exercises the schema default (-1, 1). MobileNet-v2 uses (0, 6) exclusively,
   so this override widens the bound pair to the configurations the importer
   actually produces, plus a reversed pair. *)
let hardtanh =
  {
    module_name = "Hardtanh_walk";
    target = "torch.ops.aten.hardtanh.default";
    recipe = "Recipe_bounds";
    initial =
      "Aten_walk_recipes.Recipe_bounds.{ n = 1; c = 4; h = 8; w = 8; bounds = \
       { min = Some (Aten_spec.Scalar_value.Int 0); max = Some \
       (Aten_spec.Scalar_value.Int 6) } }";
    axes =
      "Aten_walk_recipes.Recipe_bounds.axes ~n:[ 1; 2 ] ~c:[ 3; 4; 8 ] ~h:[ 4; \
       8 ] ~w:[ 4; 8 ] ~bounds:Aten_walk_recipes.Recipe_bounds.required_pairs";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_bounds.self_shape c) in
    let min_val, max_val = Recipe_bounds.required c in
    ( Aten_op_spec.Op_hardtanh.(spec { self; min_val; max_val }), pcg )|};
  }

let entries =
  [
    conv2d;
    convolution;
    conv2d_padding;
    max_pool2d;
    max_pool2d_with_indices;
    avg_pool2d;
    adaptive_avg_pool2d;
    mean_dim;
    clamp;
    hardtanh;
  ]

let find target = List.find_opt (fun e -> e.target = target) entries
