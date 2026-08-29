(* Split out of walk_meta.ml see walk_meta_entry.ml for
   the [t] record these build. *)

open Walk_meta_entry

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
