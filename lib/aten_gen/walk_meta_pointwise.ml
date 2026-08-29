(* Split out of walk_meta.ml see walk_meta_entry.ml for
   the [t] record these build. *)

open Walk_meta_entry

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
