(* Split out of walk_meta.ml see walk_meta_entry.ml for
   the [t] record these build. *)

open Walk_meta_entry

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

(* matmul.default: both shape families the bridge accepts -- see
   [Aten_walk_recipes.Recipe_matmul] and .ai/matmul_softmax_design.md §4-5.
   [d]=[h]=1 exercises the batch-less family (binds to [Bmm]); [d]>1 or
   [h]>1 exercises the batched/multi-head family (binds to the new
   [Batched_matmul]) -- both operands always share [d]/[h], so there is
   nothing to "cascade" out of the config space; every combination is
   admissible. *)
let matmul =
  {
    module_name = "Matmul_walk";
    target = "torch.ops.aten.matmul.default";
    recipe = "Recipe_matmul";
    initial =
      "Aten_walk_recipes.Recipe_matmul.{ d = 1; h = 1; n = 3; m = 4; p = 5 }";
    axes =
      "Aten_walk_recipes.Recipe_matmul.axes ~d:[ 1; 2; 3 ] ~h:[ 1; 2; 4 ] ~n:[ \
       2; 3; 5 ] ~m:[ 2; 4; 6 ] ~p:[ 2; 5; 7 ]";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_matmul.self_shape c) in
    let other, pcg = Walk.tensor_spec pcg (Recipe_matmul.other_shape c) in
    (Aten_op_spec.Op_matmul.(spec { self; other }), pcg)|};
  }
