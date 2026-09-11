(* Split out of walk_meta.ml see walk_meta_entry.ml for
   the [t] record these build. *)

open Walk_meta_entry

(* `_to_copy.default`'s default dtype is the Native float identity and its
   non-blocking flag has no value effect.  Vary the tensor shape while holding
   unsupported layout/device/memory-format options absent. *)
let to_copy =
  {
    module_name = "To_copy_walk";
    target = "torch.ops.aten._to_copy.default";
    recipe = "Recipe_default";
    initial = "Aten_walk_recipes.Recipe_default.{ shape = [ 2; 3; 4; 4 ] }";
    axes = "Aten_walk_recipes.Recipe_default.axes";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_default.shape c) in
    ( Aten_op_spec.Op__to_copy.(
        spec { self; pin_memory = None; non_blocking = false }),
      pcg )|};
  }

(* select.int has two correlated integral arguments: a rank-valid dimension and
   an in-range index for that dimension.  [Recipe_select] carries the scenario
   together and derives last indices from the current shape. *)
let select_int =
  {
    module_name = "Select_int_walk";
    target = "torch.ops.aten.select.int";
    recipe = "Recipe_select";
    initial =
      "Aten_walk_recipes.Recipe_select.{ n = 2; c = 3; h = 4; w = 4; config = \
       { rank = 4; dim = 2; index = First } }";
    axes =
      "Aten_walk_recipes.Recipe_select.axes ~n:[ 2; 3; 4 ] ~c:[ 2; 3; 5 ] ~h:[ \
       2; 4; 6 ] ~w:[ 2; 5; 7 ] \
       ~config:Aten_walk_recipes.Recipe_select.all_configs";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_select.self_shape c) in
    ( Aten_op_spec.Op_select_int.(
        spec { self; dim = Recipe_select.dim c; index = Recipe_select.index c }),
      pcg )|};
  }

(* unsqueeze.default uses the same rank-4 shape/dimension space as unbind;
   each listed dimension is valid for insertion and includes both spellings.
   A dedicated recipe is unnecessary because no bound depends on the extent. *)
let unsqueeze =
  {
    module_name = "Unsqueeze_walk";
    target = "torch.ops.aten.unsqueeze.default";
    recipe = "Recipe_unbind";
    initial =
      "Aten_walk_recipes.Recipe_unbind.{ n = 2; c = 3; h = 4; w = 4; dim = 0 }";
    axes =
      "Aten_walk_recipes.Recipe_unbind.axes ~n:[ 1; 2; 3 ] ~c:[ 2; 3; 4 ] ~h:[ \
       2; 4; 6 ] ~w:[ 2; 4; 6 ] ~dim:Aten_walk_recipes.Recipe_unbind.all_dims";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_unbind.self_shape c) in
    ( Aten_op_spec.Op_unsqueeze.(spec { self; dim = Recipe_unbind.dim c }),
      pcg )|};
  }

(* unfold.default needs a positive window and stride valid for the chosen
   dimension. [Recipe_unfold] derives both from that axis's current extent. *)
let unfold =
  {
    module_name = "Unfold_walk";
    target = "torch.ops.aten.unfold.default";
    recipe = "Recipe_unfold";
    initial =
      "Aten_walk_recipes.Recipe_unfold.{ n = 2; c = 3; h = 4; w = 4; config = \
       { dimension = 2; mode = Overlap } }";
    axes =
      "Aten_walk_recipes.Recipe_unfold.axes ~n:[ 2; 3; 4 ] ~c:[ 2; 3; 5 ] ~h:[ \
       2; 4; 6 ] ~w:[ 2; 5; 7 ] \
       ~config:Aten_walk_recipes.Recipe_unfold.all_configs";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_unfold.self_shape c) in
    ( Aten_op_spec.Op_unfold.(
        spec { self; dimension = Recipe_unfold.dimension c;
               size = Recipe_unfold.size c; step = Recipe_unfold.step c }),
      pcg )|};
  }

(* zeros.default is a shape-only factory.  Its output shape is its required
   [size] argument, so Recipe_default's shape mutations are directly meaningful. *)
let zeros =
  {
    module_name = "Zeros_walk";
    target = "torch.ops.aten.zeros.default";
    recipe = "Recipe_default";
    initial = "Aten_walk_recipes.Recipe_default.{ shape = [ 2; 3; 4; 4 ] }";
    axes = "Aten_walk_recipes.Recipe_default.axes";
    build =
      {|let pcg = pcg in
    ( Aten_op_spec.Op_zeros.(
        spec { size = Recipe_default.shape c; pin_memory = None }),
      pcg )|};
  }

(* eye.m has no tensor operand: independently vary its positive row and column
   counts, which are precisely its materialized output dimensions. *)
let eye_m =
  {
    module_name = "Eye_m_walk";
    target = "torch.ops.aten.eye.m";
    recipe = "Recipe_eye";
    initial = "Aten_walk_recipes.Recipe_eye.{ n = 3; m = 4 }";
    axes =
      "Aten_walk_recipes.Recipe_eye.axes ~n:[ 1; 2; 3; 5 ] ~m:[ 1; 2; 4; 6 ]";
    build =
      {|let pcg = pcg in
    ( Aten_op_spec.Op_eye_m.(
        spec { n = Recipe_eye.n c; m = Recipe_eye.m c; pin_memory = None }),
      pcg )|};
  }

(* The arange overloads share monotonic, like-typed scalar endpoints.  The
   recipe's whole-range candidates prevent ATen-invalid descending ranges. *)
let arange_default =
  {
    module_name = "Arange_default_walk";
    target = "torch.ops.aten.arange.default";
    recipe = "Recipe_arange";
    initial =
      "Aten_walk_recipes.Recipe_arange.{ start = Aten_spec.Scalar_value.Int 0; \
       end_ = Aten_spec.Scalar_value.Int 5 }";
    axes =
      "Aten_walk_recipes.Recipe_arange.axes \
       ~range:Aten_walk_recipes.Recipe_arange.candidates";
    build =
      {|let pcg = pcg in
    ( Aten_op_spec.Op_arange.(
        spec { end_ = Recipe_arange.end_ c; pin_memory = None }), pcg )|};
  }

let arange_start =
  {
    module_name = "Arange_start_walk";
    target = "torch.ops.aten.arange.start";
    recipe = "Recipe_arange";
    initial =
      "Aten_walk_recipes.Recipe_arange.{ start = Aten_spec.Scalar_value.Int 0; \
       end_ = Aten_spec.Scalar_value.Int 5 }";
    axes =
      "Aten_walk_recipes.Recipe_arange.axes \
       ~range:Aten_walk_recipes.Recipe_arange.candidates";
    build =
      {|let pcg = pcg in
    ( Aten_op_spec.Op_arange_start.(
        spec { start = Recipe_arange.start c; end_ = Recipe_arange.end_ c;
               pin_memory = None }), pcg )|};
  }

(* select_scatter is select's write-back form.  Its source shape is the self
   shape with the selected dimension removed; derive it from Recipe_select so
   dimensions, signed indices and source rank cannot drift apart. *)
let select_scatter =
  {
    module_name = "Select_scatter_walk";
    target = "torch.ops.aten.select_scatter.default";
    recipe = "Recipe_select";
    initial =
      "Aten_walk_recipes.Recipe_select.{ n = 2; c = 3; h = 4; w = 4; config = \
       { rank = 4; dim = 2; index = First } }";
    axes =
      "Aten_walk_recipes.Recipe_select.axes ~n:[ 2; 3; 4 ] ~c:[ 2; 3; 5 ] ~h:[ \
       2; 4; 6 ] ~w:[ 2; 5; 7 ] \
       ~config:Aten_walk_recipes.Recipe_select.all_configs";
    build =
      {|let self_shape = Recipe_select.self_shape c in
    let self, pcg = Walk.tensor_spec pcg self_shape in
    let dim = Recipe_select.dim c in
    let normalized_dim = if dim < 0 then dim + List.length self_shape else dim in
    let src_shape = List.filteri (fun i _ -> i <> normalized_dim) self_shape in
    let src, pcg = Walk.tensor_spec pcg src_shape in
    ( Aten_op_spec.Op_select_scatter.(
        spec { self; src; dim; index = Recipe_select.index c }), pcg )|};
  }

let split_tensor =
  {
    module_name = "Split_tensor_walk";
    target = "torch.ops.aten.split.Tensor";
    recipe = "Recipe_split";
    initial =
      "Aten_walk_recipes.Recipe_split.{ n = 2; c = 3; h = 4; w = 4; dim = 0 }";
    axes =
      "Aten_walk_recipes.Recipe_split.axes ~n:[ 2; 3; 4 ] ~c:[ 2; 3; 5 ] ~h:[ \
       2; 4; 6 ] ~w:[ 2; 5; 7 ] ~dim:[ 0; 1; 2; 3; -1; -2; -3; -4 ]";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_split.self_shape c) in
    (Aten_op_spec.Op_split_Tensor.(spec { self; split_size = Recipe_split.split_size c; dim = Recipe_split.dim c }), pcg)|};
  }

let split_with_sizes =
  {
    module_name = "Split_with_sizes_walk";
    target = "torch.ops.aten.split_with_sizes.default";
    recipe = "Recipe_split";
    initial =
      "Aten_walk_recipes.Recipe_split.{ n = 2; c = 3; h = 4; w = 4; dim = 0 }";
    axes =
      "Aten_walk_recipes.Recipe_split.axes ~n:[ 2; 3; 4 ] ~c:[ 2; 3; 5 ] ~h:[ \
       2; 4; 6 ] ~w:[ 2; 5; 7 ] ~dim:[ 0; 1; 2; 3; -1; -2; -3; -4 ]";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_split.self_shape c) in
    (Aten_op_spec.Op_split_with_sizes.(spec { self; split_sizes = Recipe_split.split_sizes c; dim = Recipe_split.dim c }), pcg)|};
  }

(* stack needs a same-shaped Tensor[] input. Recipe_unbind provides a rank-4
   shape plus valid axis spellings; construct two independently valued tensors
   of that shared shape. *)
let stack =
  {
    module_name = "Stack_walk";
    target = "torch.ops.aten.stack.default";
    recipe = "Recipe_unbind";
    initial =
      "Aten_walk_recipes.Recipe_unbind.{ n = 2; c = 3; h = 4; w = 4; dim = 0 }";
    axes =
      "Aten_walk_recipes.Recipe_unbind.axes ~n:[ 1; 2; 3 ] ~c:[ 2; 3; 4 ] ~h:[ \
       2; 4; 6 ] ~w:[ 2; 4; 6 ] ~dim:Aten_walk_recipes.Recipe_unbind.all_dims";
    build =
      {|let shape = Recipe_unbind.self_shape c in
    let x, pcg = Walk.tensor_spec pcg shape in
    let y, pcg = Walk.tensor_spec pcg shape in
    (Aten_op_spec.Op_stack.(spec { tensors = [ x; y ]; dim = Recipe_unbind.dim c }), pcg)|};
  }

let cat =
  {
    module_name = "Cat_walk";
    target = "torch.ops.aten.cat.default";
    recipe = "Recipe_unbind";
    initial =
      "Aten_walk_recipes.Recipe_unbind.{ n = 2; c = 3; h = 4; w = 4; dim = 0 }";
    axes =
      "Aten_walk_recipes.Recipe_unbind.axes ~n:[ 1; 2; 3 ] ~c:[ 2; 3; 4 ] ~h:[ \
       2; 4; 6 ] ~w:[ 2; 4; 6 ] ~dim:Aten_walk_recipes.Recipe_unbind.all_dims";
    build =
      {|let shape = Recipe_unbind.self_shape c in
    let x, pcg = Walk.tensor_spec pcg shape in
    let y, pcg = Walk.tensor_spec pcg shape in
    (Aten_op_spec.Op_cat.(spec { tensors = [ x; y ]; dim = Recipe_unbind.dim c }), pcg)|};
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

(* expand.default: [SymInt[] size] has no schema default, so like view/
   _unsafe_view it never reaches the generated Default tier -- and unlike
   them, an independently-drawn [size] would almost always be
   broadcast-incompatible with [self] rather than merely a different valid
   factorization, so this needs its own recipe rather than sharing
   [Recipe_view]'s. [Recipe_expand] draws which axis (if any) [self] holds at
   1 as one correlated [pattern] axis, the same shape [Recipe_view]'s own
   [pattern] takes for its target factorization. *)
let expand =
  {
    module_name = "Expand_walk";
    target = "torch.ops.aten.expand.default";
    recipe = "Recipe_expand";
    initial =
      "Aten_walk_recipes.Recipe_expand.{ n = 2; c = 3; h = 4; w = 4; pattern = \
       Aten_walk_recipes.Recipe_expand.No_broadcast; self = []; size = [] }";
    axes =
      "Aten_walk_recipes.Recipe_expand.axes ~n:[ 1; 2; 3 ] ~c:[ 1; 2; 4 ] ~h:[ \
       1; 3; 4 ] ~w:[ 1; 3; 4 ] \
       ~pattern:Aten_walk_recipes.Recipe_expand.all_patterns";
    build =
      {|let self, pcg = Walk.tensor_spec pcg (Recipe_expand.self_shape c) in
    ( Aten_op_spec.Op_expand.(
        spec { self; size = Recipe_expand.size c; implicit = false }),
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
