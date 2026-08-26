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

(* amax.default: same reduction-family recipe as [mean_dim] ("mean.dim and
   friends" -- Recipe_reduce's own doc comment). Unlike [mean_dim]'s [dim],
   ATen's `amax` declares [dim] as a required (defaulted, not optional)
   int list, so the generated [Op_amax.t] field is a plain [int list], not an
   [int list option] -- no [Some] wrapper here. *)
let amax =
  {
    module_name = "Amax_walk";
    target = "torch.ops.aten.amax.default";
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
    ( Aten_op_spec.Op_amax.(
        spec
          {
            self;
            dim = Recipe_reduce.dims c;
            keepdim = Recipe_reduce.keepdim c;
          }),
      pcg )|};
  }

(* pow.Tensor_Scalar: general support, one Native [Pow] node whose [Compute]
   special-cases exactly the exponents ATen's own [PowKernel.cpp] does
   (sqrt/rsqrt/reciprocal/square/cube/reciprocal-of-square) and falls back to
   [exp(exponent * log x)] otherwise -- so [value]'s candidates deliberately
   cover that whole special-cased set PLUS one generic exponent (1.5) that
   exercises the fallback. [self] is drawn POSITIVE
   ([Walk.tensor_spec_positive], not [tensor_spec]): several candidates
   (0.5, -0.5, -1, -2) are undefined or singular at a non-positive base, and
   a NaN/NaN or inf/inf pair is not something a tolerance-based comparator
   can call "matched" -- the same reasoning [Recipe_reduce]'s siblings and
   [Native_op_walk.Sqrt]'s own walk already follow. *)
let pow_tensor_scalar =
  {
    module_name = "Pow_tensor_scalar_walk";
    target = "torch.ops.aten.pow.Tensor_Scalar";
    recipe = "Recipe_scalar_value";
    initial =
      "Aten_walk_recipes.Recipe_scalar_value.{ n = 1; c = 4; h = 8; w = 8; \
       value = Aten_spec.Scalar_value.Float 0.5 }";
    axes =
      "Aten_walk_recipes.Recipe_scalar_value.axes ~n:[ 1; 2 ] ~c:[ 3; 4; 8 ] \
       ~h:[ 4; 8 ] ~w:[ 4; 8 ] ~value:[ Aten_spec.Scalar_value.Int 2; Int 3; \
       Int (-1); Int (-2); Float 0.5; Float (-0.5); Float 1.5 ]";
    build =
      {|let self, pcg = Walk.tensor_spec_positive pcg (Recipe_scalar_value.self_shape c) in
    ( Aten_op_spec.Op_pow_Tensor_Scalar.(spec { self; exponent = Recipe_scalar_value.value c }), pcg )|};
  }

(* linalg_vector_norm.default: same reduction-family recipe as [mean_dim]/
   [amax] ("mean.dim and friends"). The bridge accepts only the schema
   default `ord=2`, so [ord] is pinned rather than drawn. *)
let linalg_vector_norm =
  {
    module_name = "Linalg_vector_norm_walk";
    target = "torch.ops.aten.linalg_vector_norm.default";
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
    ( Aten_op_spec.Op_linalg_vector_norm.(
        spec
          {
            self;
            ord = Aten_spec.Scalar_value.Int 2;
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

(* unbind.int already gets a generated DEFAULT walk (one tensor arg, every other
   arg fillable), so unlike the entries above this is not filling a gap — it is
   an override, for the reason the design record gives for hardtanh: the default
   only ever exercises the schema defaults, and here that means dim=0 forever.
   The whole negative- and nonzero-dim space is unreachable without this. *)
let unbind_int =
  {
    module_name = "Unbind_int_walk";
    target = "torch.ops.aten.unbind.int";
    recipe = "Recipe_unbind";
    initial =
      "Aten_walk_recipes.Recipe_unbind.{ n = 2; c = 3; h = 4; w = 4; dim = 0 }";
    axes =
      "Aten_walk_recipes.Recipe_unbind.axes ~n:[ 1; 2; 3 ] ~c:[ 2; 3; 4 ] ~h:[ \
       2; 4; 6 ] ~w:[ 2; 4; 6 ] ~dim:Aten_walk_recipes.Recipe_unbind.all_dims";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_unbind.self_shape c) in
    ( Aten_op_spec.Op_unbind_int.(
        spec { self; dim = Recipe_unbind.dim c }),
      pcg )|};
  }

(* pad.default: SymInt[] pad has no schema default, so like view/_unsafe_view it
   never reaches the generated Default tier and this entry is the only walk it
   can have. The whole configuration -- pad list, mode and value together -- is
   ONE axis, because mode and value are not independent (ATen refuses a value on
   a non-constant mode) and reflect's amounts are bounded by the extent they
   mirror. [Recipe_pad] derives those amounts from the shape currently drawn, so
   the pattern axis and the shape axes stay consistent without a cascade. *)
let pad =
  {
    module_name = "Pad_walk";
    target = "torch.ops.aten.pad.default";
    recipe = "Recipe_pad";
    initial =
      "Aten_walk_recipes.Recipe_pad.{ n = 1; c = 3; h = 4; w = 4; pattern = \
       Aten_walk_recipes.Recipe_pad.Const_w }";
    axes =
      "Aten_walk_recipes.Recipe_pad.axes ~n:[ 1; 2 ] ~c:[ 2; 3; 4 ] ~h:[ 3; 4; \
       6 ] ~w:[ 3; 4; 6 ] ~pattern:Aten_walk_recipes.Recipe_pad.all_patterns";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_pad.self_shape c) in
    ( Aten_op_spec.Op_pad.(
        spec
          {
            self;
            pad = Recipe_pad.pads c;
            mode = Recipe_pad.mode c;
            value = Recipe_pad.value c;
          }),
      pcg )|};
  }

(* slice.Tensor: both bounds are [SymInt?], and [fill_non_tensor] declines an
   optional TYPE before it looks at the default, so this op has no Default tier
   either. That is the conservatism doing real work here -- slice.Tensor's
   schema defaults are jointly the IDENTITY slice, so a generated default walk
   would hold every parameter at it and report [matched] for any implementation
   that returned its input.

   Rank and dim travel as ONE candidate (a dim is only valid for a rank) and the
   bounds are derived from the drawn extent rather than stored, so [cascade] is
   the identity. See .ai/native_walk_design.md. *)
let slice_tensor =
  {
    module_name = "Slice_tensor_walk";
    target = "torch.ops.aten.slice.Tensor";
    recipe = "Recipe_slice";
    initial =
      "Aten_walk_recipes.Recipe_slice.{ n = 2; c = 3; h = 4; w = 5; config = { \
       rank = 4; dim = 0; pattern = Aten_walk_recipes.Recipe_slice.Head } }";
    axes =
      "Aten_walk_recipes.Recipe_slice.axes ~n:[ 2; 3; 4 ] ~c:[ 2; 3; 5 ] ~h:[ \
       2; 4; 6 ] ~w:[ 2; 5; 7 ] \
       ~config:Aten_walk_recipes.Recipe_slice.all_configs";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_slice.self_shape c) in
    ( Aten_op_spec.Op_slice_Tensor.(
        spec
          {
            self;
            dim = Recipe_slice.dim c;
            start = Recipe_slice.start c;
            end_ = Recipe_slice.stop c;
            step = Recipe_slice.step c;
          }),
      pcg )|};
  }

(* view.default / _unsafe_view.default: SymInt[] size has no default (unlike a
   scalar or a bool), so neither reaches the generated Default tier
   (op3-impl.md F5) -- the target is a factorization of the source's element
   count, which [Recipe_view] draws as one axis correlated to the current
   shape via [cascade], never independently. Two entries, same recipe/initial/
   axes: view.default is not strictly required by this group (row 3.2 is
   _unsafe_view), but it is the strongest available evidence that commit 1's
   shared numel resolver still accepts what it used to -- an independent ATen
   oracle exercising the same code path _unsafe_view's evidence already
   covers by construction (both share one dispatch arm). *)
let view_default =
  {
    module_name = "View_walk";
    target = "torch.ops.aten.view.default";
    recipe = "Recipe_view";
    initial =
      "Aten_walk_recipes.Recipe_view.{ n = 1; c = 4; h = 4; w = 4; pattern = \
       Aten_walk_recipes.Recipe_view.Flatten; target = [] }";
    axes =
      "Aten_walk_recipes.Recipe_view.axes ~n:[ 1; 2 ] ~c:[ 2; 4; 6 ] ~h:[ 2; \
       3; 4 ] ~w:[ 2; 3; 4 ] \
       ~pattern:Aten_walk_recipes.Recipe_view.all_patterns";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_view.self_shape c) in
    ( Aten_op_spec.Op_view.(spec { self; size = Recipe_view.target c }), pcg )|};
  }

let unsafe_view =
  {
    module_name = "Unsafe_view_walk";
    target = "torch.ops.aten._unsafe_view.default";
    recipe = "Recipe_view";
    initial =
      "Aten_walk_recipes.Recipe_view.{ n = 1; c = 4; h = 4; w = 4; pattern = \
       Aten_walk_recipes.Recipe_view.Flatten; target = [] }";
    axes =
      "Aten_walk_recipes.Recipe_view.axes ~n:[ 1; 2 ] ~c:[ 2; 4; 6 ] ~h:[ 2; \
       3; 4 ] ~w:[ 2; 3; 4 ] \
       ~pattern:Aten_walk_recipes.Recipe_view.all_patterns";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_view.self_shape c) in
    ( Aten_op_spec.Op__unsafe_view.(spec { self; size = Recipe_view.target c }),
      pcg )|};
  }

(* transpose.int: [int dim0]/[int dim1] have no default (op3-impl.md F5), and
   a dim pair is only valid for a specific rank, so [Recipe_transpose] draws
   rank and the pair together as one correlated axis. *)
let transpose_int =
  {
    module_name = "Transpose_int_walk";
    target = "torch.ops.aten.transpose.int";
    recipe = "Recipe_transpose";
    initial =
      "Aten_walk_recipes.Recipe_transpose.{ n = 2; c = 3; h = 4; w = 5; config \
       = List.hd Aten_walk_recipes.Recipe_transpose.all_configs }";
    axes =
      "Aten_walk_recipes.Recipe_transpose.axes ~n:[ 1; 2; 3 ] ~c:[ 2; 3; 4 ] \
       ~h:[ 2; 3; 4 ] ~w:[ 2; 3; 4 ] \
       ~config:Aten_walk_recipes.Recipe_transpose.all_configs";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_transpose.self_shape c) in
    ( Aten_op_spec.Op_transpose_int.(
        spec
          {
            self;
            dim0 = Recipe_transpose.dim0 c;
            dim1 = Recipe_transpose.dim1 c;
          }),
      pcg )|};
  }

(* sub.Tensor is a two-tensor op: the generated DEFAULT walk admits an op only
   when it has exactly one tensor argument (Aten_walk_gen.default_tensor), so
   there is no generic walk to override here -- this is filling a gap, the
   first of its kind (op3-impl.md F5). [alpha] is held at its schema default
   (1); the accept/reject boundary at a non-default alpha is pinned by hand
   fixtures instead (test/native_bridge_test.ml), not walked. *)
let sub_tensor =
  {
    module_name = "Sub_tensor_walk";
    target = "torch.ops.aten.sub.Tensor";
    recipe = "Recipe_binary";
    initial =
      "Aten_walk_recipes.Recipe_binary.{ n = 2; c = 4; h = 8; w = 8; pattern = \
       Aten_walk_recipes.Recipe_binary.Equal }";
    axes =
      "Aten_walk_recipes.Recipe_binary.axes ~n:[ 1; 2; 4 ] ~c:[ 4; 8; 16 ] \
       ~h:[ 4; 8; 16 ] ~w:[ 4; 8; 16 ] \
       ~pattern:Aten_walk_recipes.Recipe_binary.all_patterns";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_binary.lhs_shape c) in
    let other, pcg = Walk.tensor_spec pcg (Recipe_binary.rhs_shape c) in
    ( Aten_op_spec.Op_sub_Tensor.(
        spec { self; other; alpha = Aten_spec.Scalar_value.Int 1 }),
      pcg )|};
  }

(* add.Tensor: identical shape to [sub_tensor], giving the tensor-tensor form
   real broadcast-shape ATen-walk coverage it currently lacks. [alpha] is held
   at its schema default (1) for the same reason [sub_tensor]'s is: the
   accept/reject boundary at a non-default alpha is pinned by hand fixtures
   (test/native_bridge_test.ml), not walked. Does not and cannot cover the
   scalar-in-tensor-slot configuration (`add.Tensor(x, 3)`) -- no walk over
   real ATen calls can produce that, since the walk generator synthesises real
   tensor arguments; that stays on native_bridge_test.ml's verify_print. *)
let add_tensor =
  {
    module_name = "Add_tensor_walk";
    target = "torch.ops.aten.add.Tensor";
    recipe = "Recipe_binary";
    initial =
      "Aten_walk_recipes.Recipe_binary.{ n = 2; c = 4; h = 8; w = 8; pattern = \
       Aten_walk_recipes.Recipe_binary.Equal }";
    axes =
      "Aten_walk_recipes.Recipe_binary.axes ~n:[ 1; 2; 4 ] ~c:[ 4; 8; 16 ] \
       ~h:[ 4; 8; 16 ] ~w:[ 4; 8; 16 ] \
       ~pattern:Aten_walk_recipes.Recipe_binary.all_patterns";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_binary.lhs_shape c) in
    let other, pcg = Walk.tensor_spec pcg (Recipe_binary.rhs_shape c) in
    ( Aten_op_spec.Op_add_Tensor.(
        spec { self; other; alpha = Aten_spec.Scalar_value.Int 1 }),
      pcg )|};
  }

(* add_.Tensor: the in-place overload, same shape and [alpha] handling as
   [add_tensor] (a required schema field held at its default, same reason).
   It has its own generated [Op_add__Tensor] spec (target =
   "torch.ops.aten.add_.Tensor"), so it needs its own [walk_meta] entry --
   the generator keys off the exact target string, and doesn't infer this
   one from [add_tensor]'s. The bridge lowers both through the same dispatch
   arm (op_bridge.ml), and other in-place variants (relu_, hardsigmoid_,
   silu_, …) already walk cleanly at the Default tier, so in-place dispatch
   through real ATen is already exercised -- this was simply never given a
   recipe. *)
let add_inplace_tensor =
  {
    module_name = "Add_inplace_tensor_walk";
    target = "torch.ops.aten.add_.Tensor";
    recipe = "Recipe_binary";
    initial =
      "Aten_walk_recipes.Recipe_binary.{ n = 2; c = 4; h = 8; w = 8; pattern = \
       Aten_walk_recipes.Recipe_binary.Equal }";
    axes =
      "Aten_walk_recipes.Recipe_binary.axes ~n:[ 1; 2; 4 ] ~c:[ 4; 8; 16 ] \
       ~h:[ 4; 8; 16 ] ~w:[ 4; 8; 16 ] \
       ~pattern:Aten_walk_recipes.Recipe_binary.all_patterns";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_binary.lhs_shape c) in
    let other, pcg = Walk.tensor_spec pcg (Recipe_binary.rhs_shape c) in
    ( Aten_op_spec.Op_add__Tensor.(
        spec { self; other; alpha = Aten_spec.Scalar_value.Int 1 }),
      pcg )|};
  }

(* mul.Tensor: identical shape to [sub_tensor]/[add_tensor] minus the [alpha]
   field -- mul.Tensor's schema has none. Covers the genuine tensor-tensor
   form; the scalar-in-a-tensor-slot configuration
   (`mul.Tensor(x, 3)`/`mul.Tensor(x, 0.1)`) is, like add's, out of reach for
   a walk over real ATen calls and stays on native_bridge_test.ml's
   verify_print. *)
let mul_tensor =
  {
    module_name = "Mul_tensor_walk";
    target = "torch.ops.aten.mul.Tensor";
    recipe = "Recipe_binary";
    initial =
      "Aten_walk_recipes.Recipe_binary.{ n = 2; c = 4; h = 8; w = 8; pattern = \
       Aten_walk_recipes.Recipe_binary.Equal }";
    axes =
      "Aten_walk_recipes.Recipe_binary.axes ~n:[ 1; 2; 4 ] ~c:[ 4; 8; 16 ] \
       ~h:[ 4; 8; 16 ] ~w:[ 4; 8; 16 ] \
       ~pattern:Aten_walk_recipes.Recipe_binary.all_patterns";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_binary.lhs_shape c) in
    let other, pcg = Walk.tensor_spec pcg (Recipe_binary.rhs_shape c) in
    ( Aten_op_spec.Op_mul_Tensor.(spec { self; other }), pcg )|};
  }

(* mul.Scalar: `mul.Scalar(Tensor self, Scalar other)`'s [other] is a required
   schema Scalar, a shape no existing recipe carries alone -- [Recipe_bounds]
   carries an optional PAIR, for clamp/hardtanh. See
   [Aten_walk_recipes.Recipe_scalar_value]. *)
let mul_scalar =
  {
    module_name = "Mul_scalar_walk";
    target = "torch.ops.aten.mul.Scalar";
    recipe = "Recipe_scalar_value";
    initial =
      "Aten_walk_recipes.Recipe_scalar_value.{ n = 1; c = 4; h = 8; w = 8; \
       value = Aten_spec.Scalar_value.Float 0.1 }";
    axes =
      "Aten_walk_recipes.Recipe_scalar_value.axes ~n:[ 1; 2 ] ~c:[ 3; 4; 8 ] \
       ~h:[ 4; 8 ] ~w:[ 4; 8 ] \
       ~value:Aten_walk_recipes.Recipe_scalar_value.candidates";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_scalar_value.self_shape c) in
    ( Aten_op_spec.Op_mul_Scalar.(spec { self; other = Recipe_scalar_value.value c }), pcg )|};
  }

(* linear.default: the trailing axis is the feature axis and every leading axis
   passes through, so the [leading] candidates vary the input's RANK as well as
   its extents. A rank-2-only walk would never exercise the pass-through, and a
   bias-always walk would never exercise [Graph_ir]'s optional-bias path. *)
let linear =
  {
    module_name = "Linear_walk";
    target = "torch.ops.aten.linear.default";
    recipe = "Recipe_linear";
    initial =
      "Aten_walk_recipes.Recipe_linear.{ leading = [ 4 ]; in_features = 8; \
       out_features = 6; bias = true }";
    axes =
      "Aten_walk_recipes.Recipe_linear.axes ~leading:[ [ 4 ]; [ 2; 3 ]; [ 2; \
       3; 4 ] ] ~in_features:[ 3; 8; 16 ] ~out_features:[ 1; 6; 12 ] ~bias:[ \
       true; false ]";
    build =
      {|let input, pcg = Walk.tensor_spec pcg (Recipe_linear.input_shape c) in
    let weight, pcg = Walk.tensor_spec pcg (Recipe_linear.weight_shape c) in
    let bias, pcg =
      if Recipe_linear.has_bias c then
        let b, pcg = Walk.tensor_spec pcg (Recipe_linear.bias_shape c) in
        (Some b, pcg)
      else (None, pcg)
    in
    ( Aten_op_spec.Op_linear.(
        spec { input; weight; bias }),
      pcg )|};
  }

(* rms_norm.default: the normalized extents appear in three places -- the input's
   trailing axes, [normalized_shape], and the weight -- and one list supplies all
   three, so they cannot disagree. Both the single- and multi-axis cases and both
   weight states are walked; each divides by a different count or reads the
   weight on a different axis, and every one of them preserves the output
   shape. *)
let rms_norm =
  {
    module_name = "Rms_norm_walk";
    target = "torch.ops.aten.rms_norm.default";
    recipe = "Recipe_norm";
    initial =
      "Aten_walk_recipes.Recipe_norm.{ leading = [ 2; 3 ]; normalized = [ 4 ]; \
       eps = None; weight = true; bias = false; cudnn = false }";
    axes =
      "Aten_walk_recipes.Recipe_norm.axes ~leading:[ [ 2 ]; [ 2; 3 ]; [ 2; 3; \
       4 ] ] ~normalized:[ [ 4 ]; [ 5 ]; [ 3; 4 ]; [ 2; 3; 4 ] ] ~eps:[ None; \
       Some 1e-5; Some 0. ] ~weight:[ true; false ] ()";
    build =
      {|let input, pcg = Walk.tensor_spec pcg (Recipe_norm.input_shape c) in
    let weight, pcg =
      if Recipe_norm.has_weight c then
        let w, pcg = Walk.tensor_spec pcg (Recipe_norm.weight_shape c) in
        (Some w, pcg)
      else (None, pcg)
    in
    ( Aten_op_spec.Op_rms_norm.(
        spec
          {
            input;
            normalized_shape = Recipe_norm.normalized_shape c;
            weight;
            eps = Recipe_norm.eps c;
          }),
      pcg )|};
  }

(* layer_norm.default: rms_norm's recipe with the second affine operand and the
   cudnn_enable axis. The four affine STATES are what this walk buys over the
   hand-built fixtures -- each is a structurally different graph, and the
   one-sided ones are what a paired encoding gets wrong.

   [eps] never draws [None]: the schema default is 1e-05 and the spec always
   emits the argument, so a [None] step would build the same spec as [Some 1e-5]
   and spend a walk step proving nothing. The three drawn values are the two the
   corpus contains plus zero, which is the value that makes the reciprocal
   sqrt's argument the variance alone.

   [cudnn_enable] is walked at both values against the same expected result --
   ATen's composite names it [bool /* cudnn_enable, deprecated */] and drops it,
   so what this pins is that the CPU path really does ignore it. *)
let layer_norm =
  {
    module_name = "Layer_norm_walk";
    target = "torch.ops.aten.layer_norm.default";
    recipe = "Recipe_norm";
    initial =
      "Aten_walk_recipes.Recipe_norm.{ leading = [ 2; 3 ]; normalized = [ 4 ]; \
       eps = Some 1e-5; weight = true; bias = true; cudnn = true }";
    axes =
      "Aten_walk_recipes.Recipe_norm.axes ~leading:[ [ 2 ]; [ 2; 3 ]; [ 2; 3; \
       4 ] ] ~normalized:[ [ 4 ]; [ 5 ]; [ 3; 4 ]; [ 2; 3; 4 ] ] ~eps:[ Some \
       1e-5; Some 1e-6; Some 0. ] ~weight:[ true; false ] ~bias:[ true; false \
       ] ~cudnn:[ true; false ] ()";
    build =
      {|let input, pcg = Walk.tensor_spec pcg (Recipe_norm.input_shape c) in
    let affine present shape pcg =
      if present then
        let t, pcg = Walk.tensor_spec pcg shape in
        (Some t, pcg)
      else (None, pcg)
    in
    let weight, pcg =
      affine (Recipe_norm.has_weight c) (Recipe_norm.weight_shape c) pcg
    in
    let bias, pcg =
      affine (Recipe_norm.has_bias c) (Recipe_norm.bias_shape c) pcg
    in
    ( Aten_op_spec.Op_layer_norm.(
        spec
          {
            input;
            normalized_shape = Recipe_norm.normalized_shape c;
            weight;
            bias;
            eps = Recipe_norm.eps_or_default c;
            cudnn_enable = Recipe_norm.cudnn c;
          }),
      pcg )|};
  }

(* native_layer_norm.default: the same recipe and the same four affine states,
   without the cudnn_enable axis (no such argument survives export) and with a
   REQUIRED eps. This is the target 148 corpus nodes actually carry, so its
   drawn epsilons lead with 1e-06, the value all 148 spell. *)
let native_layer_norm =
  {
    module_name = "Native_layer_norm_walk";
    target = "torch.ops.aten.native_layer_norm.default";
    recipe = "Recipe_norm";
    initial =
      "Aten_walk_recipes.Recipe_norm.{ leading = [ 2; 3 ]; normalized = [ 4 ]; \
       eps = Some 1e-6; weight = true; bias = true; cudnn = false }";
    axes =
      "Aten_walk_recipes.Recipe_norm.axes ~leading:[ [ 2 ]; [ 2; 3 ]; [ 2; 3; \
       4 ] ] ~normalized:[ [ 4 ]; [ 5 ]; [ 3; 4 ]; [ 2; 3; 4 ] ] ~eps:[ Some \
       1e-6; Some 1e-5; Some 0. ] ~weight:[ true; false ] ~bias:[ true; false \
       ] ()";
    build =
      {|let input, pcg = Walk.tensor_spec pcg (Recipe_norm.input_shape c) in
    let affine present shape pcg =
      if present then
        let t, pcg = Walk.tensor_spec pcg shape in
        (Some t, pcg)
      else (None, pcg)
    in
    let weight, pcg =
      affine (Recipe_norm.has_weight c) (Recipe_norm.weight_shape c) pcg
    in
    let bias, pcg =
      affine (Recipe_norm.has_bias c) (Recipe_norm.bias_shape c) pcg
    in
    ( Aten_op_spec.Op_native_layer_norm.(
        spec
          {
            input;
            normalized_shape = Recipe_norm.normalized_shape c;
            weight;
            bias;
            eps = Recipe_norm.eps_or_default c;
          }),
      pcg )|};
  }

(* _native_batch_norm_legit_no_training.default: rank-4 input, [c]-shaped
   running stats and independently optional affine operands, see
   [Aten_walk_recipes.Recipe_batch_norm]. [running_var] is drawn positive
   ([Walk.tensor_spec_positive]) for the reason given there. *)
let native_batch_norm =
  {
    module_name = "Native_batch_norm_walk";
    target = "torch.ops.aten._native_batch_norm_legit_no_training.default";
    recipe = "Recipe_batch_norm";
    initial =
      "Aten_walk_recipes.Recipe_batch_norm.{ n = 2; c = 4; h = 8; w = 8; eps = \
       1e-5; weight = true; bias = true }";
    axes =
      "Aten_walk_recipes.Recipe_batch_norm.axes ~n:[ 1; 2; 4 ] ~c:[ 3; 4; 8 ] \
       ~h:[ 4; 8; 16 ] ~w:[ 4; 8; 16 ] ~eps:[ 1e-5; 1e-3; 0. ] ~weight:[ true; \
       false ] ~bias:[ true; false ]";
    build =
      {|let input, pcg = Walk.tensor_spec pcg (Recipe_batch_norm.input_shape c) in
    let running_mean, pcg =
      Walk.tensor_spec pcg (Recipe_batch_norm.channel_shape c)
    in
    let running_var, pcg =
      Walk.tensor_spec_positive pcg (Recipe_batch_norm.channel_shape c)
    in
    let weight, pcg =
      if Recipe_batch_norm.has_weight c then
        let w, pcg = Walk.tensor_spec pcg (Recipe_batch_norm.channel_shape c) in
        (Some w, pcg)
      else (None, pcg)
    in
    let bias, pcg =
      if Recipe_batch_norm.has_bias c then
        let b, pcg = Walk.tensor_spec pcg (Recipe_batch_norm.channel_shape c) in
        (Some b, pcg)
      else (None, pcg)
    in
    ( Aten_op_spec.Op__native_batch_norm_legit_no_training.(
        spec
          {
            input;
            weight;
            bias;
            running_mean;
            running_var;
            momentum = Aten_spec.Float32.to_f32 Recipe_batch_norm.momentum;
            eps = Aten_spec.Float32.to_f32 (Recipe_batch_norm.eps c);
          }),
      pcg )|};
  }

(* scaled_dot_product_attention.default (op8-impl.md commit 4, F9). Three
   tensor arguments plus an optional mask put this straight into
   [needs_meta] the moment the op is registered -- [fill_non_tensor] refuses
   an optional type before it reads the default, and [default_tensor] admits
   only a single tensor argument. [Recipe_sdpa] correlates every shape so an
   ATen-rejected config is unrepresentable rather than merely unlikely (F4):
   [check_attn_mask_shape]'s failures are silent (a fallback to the `math`
   backend, not an error), so an uncorrelated recipe would not even show up
   as a `skipped` line -- it would quietly report against the wrong oracle.
   [dropout_p]/[is_causal]/[enable_gqa] are fixed at the only values the
   scope admits; they are typed rejections at the importer boundary
   (op8-impl.md commit 3), not axes a correctness walk should vary. *)
let sdpa =
  {
    module_name = "Sdpa_walk";
    target = "torch.ops.aten.scaled_dot_product_attention.default";
    recipe = "Recipe_sdpa";
    initial =
      "Aten_walk_recipes.Recipe_sdpa.{ batch = 1; heads = 2; sq = 3; sk = 4; e \
       = 5; mask = Aten_walk_recipes.Recipe_sdpa.No_mask; scale = None }";
    axes =
      "Aten_walk_recipes.Recipe_sdpa.axes ~batch:[ 1; 2 ] ~heads:[ 1; 2; 3 ] \
       ~sq:[ 1; 3; 5 ] ~sk:[ 1; 4; 6 ] ~e:[ 1; 3; 8 ] \
       ~mask:Aten_walk_recipes.Recipe_sdpa.all_mask_kinds ~scale:[ None; Some \
       0.1; Some 1.0; Some 2.0 ] ()";
    build =
      {|let query, pcg = Walk.tensor_spec pcg (Recipe_sdpa.query_shape c) in
    let key, pcg = Walk.tensor_spec pcg (Recipe_sdpa.key_shape c) in
    let value, pcg = Walk.tensor_spec pcg (Recipe_sdpa.value_shape c) in
    let attn_mask, pcg =
      match Recipe_sdpa.mask_shape c with
      | None -> (None, pcg)
      | Some shape ->
          let m, pcg = Walk.tensor_spec pcg shape in
          (Some m, pcg)
    in
    ( Aten_op_spec.Op_scaled_dot_product_attention.(
        spec
          {
            query;
            key;
            value;
            attn_mask;
            dropout_p = 0.0;
            is_causal = false;
            scale = Recipe_sdpa.scale c;
            enable_gqa = false;
          }),
      pcg )|};
  }

(* div.Tensor: identical shape to [sub_tensor]/[mul_tensor] -- no [alpha]
   field, same as mul.Tensor. Covers the genuine tensor-tensor form; the
   scalar-in-a-tensor-slot configuration is out of reach for the same reason
   given at [mul_tensor], and stays on native_bridge_test.ml's verify_print.
   Unlike those, the rhs (divisor) is drawn via [Walk.tensor_spec_nonzero],
   not [Walk.tensor_spec]: a plain uniform(-1,1) divisor is very likely to
   land close to zero somewhere across a few hundred elements, and a
   near-zero divisor amplifies an ordinary last-bit rounding difference
   between ATen's kernel and native's into a reported mismatch. *)
let div_tensor =
  {
    module_name = "Div_tensor_walk";
    target = "torch.ops.aten.div.Tensor";
    recipe = "Recipe_binary";
    initial =
      "Aten_walk_recipes.Recipe_binary.{ n = 2; c = 4; h = 8; w = 8; pattern = \
       Aten_walk_recipes.Recipe_binary.Equal }";
    axes =
      "Aten_walk_recipes.Recipe_binary.axes ~n:[ 1; 2; 4 ] ~c:[ 4; 8; 16 ] \
       ~h:[ 4; 8; 16 ] ~w:[ 4; 8; 16 ] \
       ~pattern:Aten_walk_recipes.Recipe_binary.all_patterns";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_binary.lhs_shape c) in
    let other, pcg = Walk.tensor_spec_nonzero pcg (Recipe_binary.rhs_shape c) in
    ( Aten_op_spec.Op_div_Tensor.(spec { self; other }), pcg )|};
  }

(* permute.default: rank-correlated permutation, same shape as
   [transpose_int] but exercising the bridge's variadic
   [native_perm_of_aten] path (a full permutation) rather than a single swap
   pair. *)
let permute =
  {
    module_name = "Permute_walk";
    target = "torch.ops.aten.permute.default";
    recipe = "Recipe_permute";
    initial =
      "Aten_walk_recipes.Recipe_permute.{ n = 2; c = 3; h = 4; w = 5; config = \
       List.hd Aten_walk_recipes.Recipe_permute.all_configs }";
    axes =
      "Aten_walk_recipes.Recipe_permute.axes ~n:[ 1; 2; 3 ] ~c:[ 2; 3; 4 ] \
       ~h:[ 2; 3; 4 ] ~w:[ 2; 3; 4 ] \
       ~config:Aten_walk_recipes.Recipe_permute.all_configs";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_permute.self_shape c) in
    ( Aten_op_spec.Op_permute.(spec { self; dims = Recipe_permute.dims c }), pcg )|};
  }

(* addmm.default: see [Aten_walk_recipes.Recipe_addmm] for the shape
   constraint the bridge's Linear-based lowering imposes on [self]. *)
let addmm =
  {
    module_name = "Addmm_walk";
    target = "torch.ops.aten.addmm.default";
    recipe = "Recipe_addmm";
    initial =
      "Aten_walk_recipes.Recipe_addmm.{ n = 4; in_features = 8; out_features = \
       6 }";
    axes =
      "Aten_walk_recipes.Recipe_addmm.axes ~n:[ 1; 2; 4 ] ~in_features:[ 3; 8; \
       16 ] ~out_features:[ 1; 6; 12 ]";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_addmm.self_shape c) in
    let mat1, pcg = Walk.tensor_spec pcg (Recipe_addmm.mat1_shape c) in
    let mat2, pcg = Walk.tensor_spec pcg (Recipe_addmm.mat2_shape c) in
    ( Aten_op_spec.Op_addmm.(
        spec
          {
            self;
            mat1;
            mat2;
            beta = Aten_spec.Scalar_value.Int 1;
            alpha = Aten_spec.Scalar_value.Int 1;
          }),
      pcg )|};
  }

(* bmm.default: independent batch/row/contract/col axes, see
   [Aten_walk_recipes.Recipe_bmm]. *)
let bmm =
  {
    module_name = "Bmm_walk";
    target = "torch.ops.aten.bmm.default";
    recipe = "Recipe_bmm";
    initial = "Aten_walk_recipes.Recipe_bmm.{ batch = 2; n = 3; m = 4; p = 5 }";
    axes =
      "Aten_walk_recipes.Recipe_bmm.axes ~batch:[ 1; 2; 3 ] ~n:[ 2; 3; 5 ] \
       ~m:[ 2; 4; 6 ] ~p:[ 2; 5; 7 ]";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_bmm.self_shape c) in
    let mat2, pcg = Walk.tensor_spec pcg (Recipe_bmm.mat2_shape c) in
    (Aten_op_spec.Op_bmm.(spec { self; mat2 }), pcg)|};
  }

let entries =
  [
    conv2d;
    convolution;
    conv2d_padding;
    linear;
    rms_norm;
    layer_norm;
    native_layer_norm;
    max_pool2d;
    max_pool2d_with_indices;
    avg_pool2d;
    adaptive_avg_pool2d;
    mean_dim;
    amax;
    pow_tensor_scalar;
    linalg_vector_norm;
    clamp;
    hardtanh;
    unbind_int;
    pad;
    slice_tensor;
    sub_tensor;
    add_tensor;
    add_inplace_tensor;
    mul_tensor;
    div_tensor;
    mul_scalar;
    view_default;
    unsafe_view;
    transpose_int;
    permute;
    addmm;
    bmm;
    native_batch_norm;
    sdpa;
  ]

let find target = List.find_opt (fun e -> e.target = target) entries
