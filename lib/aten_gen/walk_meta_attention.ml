(* Split out of walk_meta.ml see walk_meta_entry.ml for
   the [t] record these build. *)

open Walk_meta_entry

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
