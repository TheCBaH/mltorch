(* Split out of walk_meta.ml see walk_meta_entry.ml for
   the [t] record these build. *)

open Walk_meta_entry

let max_pool2d =
  {
    module_name = "Max_pool2d_walk";
    target = "torch.ops.aten.max_pool2d.default";
    recipe = "Recipe_pool";
    initial =
      "Aten_walk_recipes.Recipe_pool.{ kernel_h = 2; kernel_w = 2; stride_h = \
       2; stride_w = 2; pad_h = 0; pad_w = 0; n = 1; c = 4; input_h = 8; \
       input_w = 8; ceil_mode = false }";
    axes =
      "Aten_walk_recipes.Recipe_pool.axes ~kernel_h:[ 2; 3; 4 ] ~kernel_w:[ 2; \
       3; 4 ] ~stride_h:[ 1; 2; 3 ] ~stride_w:[ 1; 2; 3 ] ~pad_h:[ 0; 1 ] \
       ~pad_w:[ 0; 1 ] ~n:[ 1; 2 ] ~c:[ 4; 8; 16 ] ~input_h:[ 8; 10; 12; 16 ] \
       ~input_w:[ 8; 10; 12; 16 ] ~ceil_mode:[ true; false ]";
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

let max_pool2d_with_indices =
  {
    module_name = "Max_pool2d_with_indices_walk";
    target = "torch.ops.aten.max_pool2d_with_indices.default";
    recipe = "Recipe_pool";
    initial =
      "Aten_walk_recipes.Recipe_pool.{ kernel_h = 2; kernel_w = 2; stride_h = \
       2; stride_w = 2; pad_h = 0; pad_w = 0; n = 1; c = 4; input_h = 8; \
       input_w = 8; ceil_mode = false }";
    axes =
      "Aten_walk_recipes.Recipe_pool.axes ~kernel_h:[ 2; 3; 4 ] ~kernel_w:[ 2; \
       3; 4 ] ~stride_h:[ 1; 2; 3 ] ~stride_w:[ 1; 2; 3 ] ~pad_h:[ 0; 1 ] \
       ~pad_w:[ 0; 1 ] ~n:[ 1; 2 ] ~c:[ 4; 8; 16 ] ~input_h:[ 8; 10; 12; 16 ] \
       ~input_w:[ 8; 10; 12; 16 ] ~ceil_mode:[ true; false ]";
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
    recipe = "Recipe_avg_pool";
    initial =
      "Aten_walk_recipes.Recipe_avg_pool.{ window = \
       Aten_walk_recipes.Recipe_pool.{ kernel_h = 2; kernel_w = 2; stride_h = \
       2; stride_w = 2; pad_h = 0; pad_w = 0; n = 1; c = 4; input_h = 8; \
       input_w = 8; ceil_mode = false }; count_include_pad = true }";
    axes =
      "Aten_walk_recipes.Recipe_avg_pool.axes ~kernel_h:[ 2; 3; 4 ] \
       ~kernel_w:[ 2; 3; 4 ] ~stride_h:[ 1; 2; 3 ] ~stride_w:[ 1; 2; 3 ] \
       ~pad_h:[ 0; 1 ] ~pad_w:[ 0; 1 ] ~n:[ 1; 2 ] ~c:[ 4; 8; 16 ] ~input_h:[ \
       8; 10; 12; 16 ] ~input_w:[ 8; 10; 12; 16 ] ~ceil_mode:[ true; false ] \
       ~count_include_pad:[ true; false ]";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_avg_pool.self_shape c) in
    ( Aten_op_spec.Op_avg_pool2d.(
        spec
          {
            self;
            kernel_size = Recipe_avg_pool.kernel_size c;
            stride = Recipe_avg_pool.strides c;
            padding = Recipe_avg_pool.pads c;
            ceil_mode = Recipe_avg_pool.ceil_mode c;
            count_include_pad = Recipe_avg_pool.count_include_pad c;
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

(* Same config space as [adaptive_avg_pool2d] above -- [Recipe_adaptive]'s own
   shape/output_size axes have no notion of which reduction runs, so the
   walk recipe is shared unchanged; only the [build]'s target op differs. *)
let adaptive_max_pool2d =
  {
    module_name = "Adaptive_max_pool2d_walk";
    target = "torch.ops.aten.adaptive_max_pool2d.default";
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
    ( Aten_op_spec.Op_adaptive_max_pool2d.(
        spec { self; output_size = Recipe_adaptive.output_size c }),
      pcg )|};
  }
