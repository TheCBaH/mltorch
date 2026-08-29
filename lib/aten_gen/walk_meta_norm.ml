(* Split out of walk_meta.ml see walk_meta_entry.ml for
   the [t] record these build. *)

open Walk_meta_entry

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
