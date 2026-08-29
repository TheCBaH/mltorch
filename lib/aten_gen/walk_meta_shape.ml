(* Split out of walk_meta.ml see walk_meta_entry.ml for
   the [t] record these build. *)

open Walk_meta_entry

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
