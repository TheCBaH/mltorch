(* Random-walk equivalence tests over the generated walk modules
   (lib/aten_op_walk): each walk starts from a fixed initial config, then walks a
   seeded PCG over the op's axes, re-cascading to keep the config valid and
   validating native vs ATen at every step, stopping on the first mismatch.

   Per-op blocks pin a stable seed for the family-recipe ops; [bridge coverage]
   sweeps every generated walk and prints the [needs_meta] backlog. *)

module Pcg = Aten_spec.Pcg

(* Buffer formatter output so ppx_expect can capture it. *)
let capture f = print_string (Core.Pretty.capture_to_string f)

let%expect_test "conv2d.default walk 5 steps" =
  (* [assert]: a mismatch truncates the printed walk, so the golden catches it
     too -- but as a diff that reads like a rewritten expectation rather than a
     failure. Asserting distinguishes the two. *)
  capture (fun ppf ->
      assert (
        Op_walk.run
          (module Aten_op_walk.Conv2d_walk)
          ~ppf
          ~pcg:(Pcg.seed ~seed:42L ~seq:1L)
          ~steps:5));
  [%expect
    {|
    step 0: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.conv2d.default: matched
    step 1 [n]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.conv2d.default: matched
    step 2 [pad_w]: {kernel=3x3 stride=1x1 pad=1x0 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.conv2d.default: matched
    step 3 [pad_h]: {kernel=3x3 stride=1x1 pad=1x0 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.conv2d.default: matched
    step 4 [pad_w]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.conv2d.default: matched
    step 5 [input_w]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=12}
    [spec] torch.ops.aten.conv2d.default: matched |}]

(* Eight steps rather than five: [leading] is one axis of three, and the
   pass-through of the input's LEADING axes -- the property a rank-2-only walk
   cannot see -- is only exercised when that axis is the one drawn. A shorter
   run on this seed never varies it. *)
let%expect_test "linear.default walk 8 steps" =
  capture (fun ppf ->
      assert (
        Op_walk.run
          (module Aten_op_walk.Linear_walk)
          ~ppf
          ~pcg:(Pcg.seed ~seed:7L ~seq:1L)
          ~steps:8));
  [%expect
    {|
    step 0: {input=[4,8] out_features=6 bias=true}
    [spec] torch.ops.aten.linear.default: matched
    step 1 [in_features]: {input=[4,8] out_features=6 bias=true}
    [spec] torch.ops.aten.linear.default: matched
    step 2 [leading]: {input=[2,3,4,8] out_features=6 bias=true}
    [spec] torch.ops.aten.linear.default: matched
    step 3 [bias]: {input=[2,3,4,8] out_features=6 bias=false}
    [spec] torch.ops.aten.linear.default: matched
    step 4 [out_features]: {input=[2,3,4,8] out_features=1 bias=false}
    [spec] torch.ops.aten.linear.default: matched
    step 5 [leading]: {input=[4,8] out_features=1 bias=false}
    [spec] torch.ops.aten.linear.default: matched
    step 6 [bias]: {input=[4,8] out_features=1 bias=true}
    [spec] torch.ops.aten.linear.default: matched
    step 7 [bias]: {input=[4,8] out_features=1 bias=false}
    [spec] torch.ops.aten.linear.default: matched
    step 8 [in_features]: {input=[4,8] out_features=1 bias=false}
    [spec] torch.ops.aten.linear.default: matched |}]

(* Ten steps, and the point is what varies: [normalized] changes the axis COUNT,
   [weight] changes the graph's shape, [eps] changes the constant. Each of those
   preserves the output shape, so a shorter run that happened to vary none of
   them would be a suite of identical configurations reporting coverage.

   Every line must read "matched". A "skipped" here would mean the recipe
   generated a spec ATen rejects -- which is the failure mode an uncorrelated
   normalized_shape produces, and it does not look like a failure. *)
let%expect_test "rms_norm.default walk 10 steps" =
  capture (fun ppf ->
      assert (
        Op_walk.run
          (module Aten_op_walk.Rms_norm_walk)
          ~ppf
          ~pcg:(Pcg.seed ~seed:11L ~seq:1L)
          ~steps:10));
  [%expect
    {|
    step 0: {input=[2,3,4] normalized=[4] eps=default weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 1 [eps]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 2 [eps]: {input=[2,3,4] normalized=[4] eps=default weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 3 [leading]: {input=[2,3,4] normalized=[4] eps=default weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 4 [normalized]: {input=[2,3,5] normalized=[5] eps=default weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 5 [weight]: {input=[2,3,5] normalized=[5] eps=default weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 6 [weight]: {input=[2,3,5] normalized=[5] eps=default weight=false bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 7 [leading]: {input=[2,5] normalized=[5] eps=default weight=false bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 8 [normalized]: {input=[2,2,3,4] normalized=[2,3,4] eps=default weight=false bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 9 [normalized]: {input=[2,2,3,4] normalized=[2,3,4] eps=default weight=false bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 10 [weight]: {input=[2,2,3,4] normalized=[2,3,4] eps=default weight=false bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched |}]

(* The same recipe with the second affine operand and the cudnn_enable axis. The
   four affine STATES are what this buys over rms_norm's walk: each is a
   structurally different graph, and the one-sided ones ("bias but no weight")
   are the states no exported model produces and the ones a paired encoding gets
   wrong.

   20 steps rather than rms_norm's 10, and the seed is the one that reaches all
   four affine states inside them -- a run that never reached a state is a state
   nobody walked, and picking the step count without checking which states it
   covers is how a walk comes to look like coverage. Every line must read
   "matched"; a "skipped" would mean the recipe generated a spec ATen rejects.

   This is also where ATen's vectorized Welford kernel meets the native scalar
   two-pass form. Both agree at Verify.default_atol, so atol_for_target is
   untouched. *)
let%expect_test "layer_norm.default walk 20 steps" =
  capture (fun ppf ->
      assert (
        Op_walk.run
          (module Aten_op_walk.Layer_norm_walk)
          ~ppf
          ~pcg:(Pcg.seed ~seed:13L ~seq:1L)
          ~steps:20));
  [%expect
    {|
    step 0: {input=[2,3,4] normalized=[4] eps=1e-05 weight=true bias=true cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 1 [bias]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=true bias=false cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 2 [leading]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=true bias=false cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 3 [weight]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=true bias=false cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 4 [weight]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=false bias=false cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 5 [weight]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=false bias=false cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 6 [bias]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=false bias=false cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 7 [leading]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=false bias=false cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 8 [eps]: {input=[2,3,4] normalized=[4] eps=1e-06 weight=false bias=false cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 9 [eps]: {input=[2,3,4] normalized=[4] eps=1e-06 weight=false bias=false cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 10 [bias]: {input=[2,3,4] normalized=[4] eps=1e-06 weight=false bias=true cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 11 [leading]: {input=[2,4] normalized=[4] eps=1e-06 weight=false bias=true cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 12 [weight]: {input=[2,4] normalized=[4] eps=1e-06 weight=true bias=true cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 13 [weight]: {input=[2,4] normalized=[4] eps=1e-06 weight=true bias=true cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 14 [cudnn_enable]: {input=[2,4] normalized=[4] eps=1e-06 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 15 [normalized]: {input=[2,4] normalized=[4] eps=1e-06 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 16 [normalized]: {input=[2,2,3,4] normalized=[2,3,4] eps=1e-06 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 17 [bias]: {input=[2,2,3,4] normalized=[2,3,4] eps=1e-06 weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 18 [leading]: {input=[2,2,3,4] normalized=[2,3,4] eps=1e-06 weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 19 [leading]: {input=[2,2,3,4] normalized=[2,3,4] eps=1e-06 weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 20 [weight]: {input=[2,2,3,4] normalized=[2,3,4] eps=1e-06 weight=false bias=false cudnn=false}
    [spec] torch.ops.aten.layer_norm.default: matched |}]

(* The decomposed target, which the corpus actually carries. Same recipe, no
   cudnn axis, and a required eps whose drawn values lead with 1e-06 -- the
   value all 148 corpus nodes spell.

   The bridge exposes ONE output against ATen's three; that is the
   leading-outputs rule, and a "matched" line here is what says the rule applies
   to a fixed tuple. An "output count" line would mean it did not.

   26 steps, again chosen so all four affine states are reached; this is also
   the walk that draws eps=0, which the layer_norm run above does not. *)
let%expect_test "native_layer_norm.default walk 26 steps" =
  capture (fun ppf ->
      assert (
        Op_walk.run
          (module Aten_op_walk.Native_layer_norm_walk)
          ~ppf
          ~pcg:(Pcg.seed ~seed:23L ~seq:2L)
          ~steps:26));
  [%expect
    {|
    step 0: {input=[2,3,4] normalized=[4] eps=1e-06 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 1 [weight]: {input=[2,3,4] normalized=[4] eps=1e-06 weight=false bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 2 [weight]: {input=[2,3,4] normalized=[4] eps=1e-06 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 3 [weight]: {input=[2,3,4] normalized=[4] eps=1e-06 weight=false bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 4 [bias]: {input=[2,3,4] normalized=[4] eps=1e-06 weight=false bias=false cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 5 [eps]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=false bias=false cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 6 [weight]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=false bias=false cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 7 [bias]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=false bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 8 [weight]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=false bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 9 [eps]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=false bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 10 [normalized]: {input=[2,3,2,3,4] normalized=[2,3,4] eps=1e-05 weight=false bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 11 [normalized]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=false bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 12 [bias]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=false bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 13 [normalized]: {input=[2,3,5] normalized=[5] eps=1e-05 weight=false bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 14 [leading]: {input=[2,5] normalized=[5] eps=1e-05 weight=false bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 15 [eps]: {input=[2,5] normalized=[5] eps=0 weight=false bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 16 [bias]: {input=[2,5] normalized=[5] eps=0 weight=false bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 17 [eps]: {input=[2,5] normalized=[5] eps=0 weight=false bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 18 [leading]: {input=[2,3,4,5] normalized=[5] eps=0 weight=false bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 19 [normalized]: {input=[2,3,4,2,3,4] normalized=[2,3,4] eps=0 weight=false bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 20 [bias]: {input=[2,3,4,2,3,4] normalized=[2,3,4] eps=0 weight=false bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 21 [weight]: {input=[2,3,4,2,3,4] normalized=[2,3,4] eps=0 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 22 [bias]: {input=[2,3,4,2,3,4] normalized=[2,3,4] eps=0 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 23 [eps]: {input=[2,3,4,2,3,4] normalized=[2,3,4] eps=1e-06 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 24 [bias]: {input=[2,3,4,2,3,4] normalized=[2,3,4] eps=1e-06 weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 25 [normalized]: {input=[2,3,4,3,4] normalized=[3,4] eps=1e-06 weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 26 [weight]: {input=[2,3,4,3,4] normalized=[3,4] eps=1e-06 weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched |}]

let%expect_test "max_pool2d.default walk 5 steps" =
  capture (fun ppf ->
      assert (
        Op_walk.run
          (module Aten_op_walk.Max_pool2d_walk)
          ~ppf
          ~pcg:(Pcg.seed ~seed:17L ~seq:3L)
          ~steps:5));
  [%expect
    {|
    step 0: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 1 [stride_h]: {kernel=2x2 stride=1x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 2 [stride_w]: {kernel=2x2 stride=1x1 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 3 [stride_h]: {kernel=2x2 stride=2x1 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 4 [input_h]: {kernel=2x2 stride=2x1 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 5 [input_h]: {kernel=2x2 stride=2x1 pad=0x0 n=1 c=4 H=12 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d.default: matched |}]

let%expect_test "mean.dim walk 5 steps" =
  capture (fun ppf ->
      assert (
        Op_walk.run
          (module Aten_op_walk.Mean_dim_walk)
          ~ppf
          ~pcg:(Pcg.seed ~seed:99L ~seq:4L)
          ~steps:5));
  [%expect
    {|
    step 0: {shape=[2,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.mean.dim: matched
    step 1 [n]: {shape=[1,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.mean.dim: matched
    step 2 [n]: {shape=[1,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.mean.dim: matched
    step 3 [h]: {shape=[1,4,16,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.mean.dim: matched
    step 4 [c]: {shape=[1,8,16,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.mean.dim: matched
    step 5 [w]: {shape=[1,8,16,4] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.mean.dim: matched |}]

(* Twenty steps, not five, and what the extra fifteen buy is the pattern column.
   The configuration is ONE axis of five, and [field_axis] redraws uniformly
   including the value already held, so a five-step walk here sees a single
   configuration and varies only the shape under it.

   Even twenty steps do not sweep all seven: this seed reaches six, missing
   [reflect_hw]. That is a property of a random walk and not a gap in the
   evidence -- the CONFIGURATION space is covered exhaustively and against the
   same ATen oracle by test/native_bridge_test.ml, which runs fourteen pad
   configurations including reflect on every boundary. What the walk adds, and
   the bridge fixtures cannot, is shape and tensor-value variation UNDER each
   configuration, which is why the pad amounts here are derived from the drawn
   extents rather than fixed. *)
let%expect_test "pad.default walk 20 steps" =
  capture (fun ppf ->
      assert (
        Op_walk.run
          (module Aten_op_walk.Pad_walk)
          ~ppf
          ~pcg:(Pcg.seed ~seed:23L ~seq:6L)
          ~steps:20));
  [%expect
    {|
    step 0: {shape=[1,3,4,4] pattern=const_w pad=[1,2] mode=constant value=0.1}
    [spec] torch.ops.aten.pad.default: matched
    step 1 [h]: {shape=[1,3,6,4] pattern=const_w pad=[1,2] mode=constant value=0.1}
    [spec] torch.ops.aten.pad.default: matched
    step 2 [c]: {shape=[1,3,6,4] pattern=const_w pad=[1,2] mode=constant value=0.1}
    [spec] torch.ops.aten.pad.default: matched
    step 3 [h]: {shape=[1,3,6,4] pattern=const_w pad=[1,2] mode=constant value=0.1}
    [spec] torch.ops.aten.pad.default: matched
    step 4 [pattern]: {shape=[1,3,6,4] pattern=const_w_no_value pad=[2,1] mode=constant value=none}
    [spec] torch.ops.aten.pad.default: matched
    step 5 [pattern]: {shape=[1,3,6,4] pattern=crop_and_pad pad=[-1,0,2,1] mode=constant value=0.1}
    [spec] torch.ops.aten.pad.default: matched
    step 6 [pattern]: {shape=[1,3,6,4] pattern=reflect_asym_w pad=[2,1,0,0] mode=reflect value=none}
    [spec] torch.ops.aten.pad.default: matched
    step 7 [c]: {shape=[1,2,6,4] pattern=reflect_asym_w pad=[2,1,0,0] mode=reflect value=none}
    [spec] torch.ops.aten.pad.default: matched
    step 8 [h]: {shape=[1,2,6,4] pattern=reflect_asym_w pad=[2,1,0,0] mode=reflect value=none}
    [spec] torch.ops.aten.pad.default: matched
    step 9 [h]: {shape=[1,2,4,4] pattern=reflect_asym_w pad=[2,1,0,0] mode=reflect value=none}
    [spec] torch.ops.aten.pad.default: matched
    step 10 [n]: {shape=[1,2,4,4] pattern=reflect_asym_w pad=[2,1,0,0] mode=reflect value=none}
    [spec] torch.ops.aten.pad.default: matched
    step 11 [h]: {shape=[1,2,3,4] pattern=reflect_asym_w pad=[2,1,0,0] mode=reflect value=none}
    [spec] torch.ops.aten.pad.default: matched
    step 12 [w]: {shape=[1,2,3,3] pattern=reflect_asym_w pad=[2,1,0,0] mode=reflect value=none}
    [spec] torch.ops.aten.pad.default: matched
    step 13 [h]: {shape=[1,2,4,3] pattern=reflect_asym_w pad=[2,1,0,0] mode=reflect value=none}
    [spec] torch.ops.aten.pad.default: matched
    step 14 [pattern]: {shape=[1,2,4,3] pattern=const_w pad=[1,2] mode=constant value=0.1}
    [spec] torch.ops.aten.pad.default: matched
    step 15 [h]: {shape=[1,2,4,3] pattern=const_w pad=[1,2] mode=constant value=0.1}
    [spec] torch.ops.aten.pad.default: matched
    step 16 [pattern]: {shape=[1,2,4,3] pattern=const_hw pad=[1,2,3,1] mode=constant value=-2.5}
    [spec] torch.ops.aten.pad.default: matched
    step 17 [n]: {shape=[1,2,4,3] pattern=const_hw pad=[1,2,3,1] mode=constant value=-2.5}
    [spec] torch.ops.aten.pad.default: matched
    step 18 [c]: {shape=[1,3,4,3] pattern=const_hw pad=[1,2,3,1] mode=constant value=-2.5}
    [spec] torch.ops.aten.pad.default: matched
    step 19 [h]: {shape=[1,3,3,3] pattern=const_hw pad=[1,2,3,1] mode=constant value=-2.5}
    [spec] torch.ops.aten.pad.default: matched
    step 20 [pattern]: {shape=[1,3,3,3] pattern=crop_h pad=[0,0,-1,-1] mode=constant value=7}
    [spec] torch.ops.aten.pad.default: matched |}]

(* Twenty steps, for [pad.default]'s reason: the configuration is one axis of
   five, so a five-step run holds it fixed and varies only the shape. This seed
   reaches four of the ten patterns and, more usefully, four different RANKS
   (4, 1, 3, 4) with both dim spellings -- which is the correlation the recipe
   exists for, since a dim is only valid for a rank and a bound only for an
   extent. Watch step 7: the rank drops to 1 and [-3,-1) re-resolves against the
   new extent without a cascade, because [Recipe_slice.bounds] derives it rather
   than storing it.

   The CONFIGURATION space is covered exhaustively, and against the same ATen
   oracle, by test/native_bridge_test.ml. What the walk adds is shape and
   tensor-value variation under each configuration. *)
let%expect_test "slice.Tensor walk 20 steps" =
  capture (fun ppf ->
      assert (
        Op_walk.run
          (module Aten_op_walk.Slice_tensor_walk)
          ~ppf
          ~pcg:(Pcg.seed ~seed:1L ~seq:7L)
          ~steps:20));
  [%expect
    {|
    step 0: {shape=[2,3,4,5] dim=0 pattern=head [0,1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 1 [config]: {shape=[2,3,4,5] dim=3 pattern=both_neg [-3,-1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 2 [n]: {shape=[3,3,4,5] dim=3 pattern=both_neg [-3,-1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 3 [w]: {shape=[3,3,4,5] dim=3 pattern=both_neg [-3,-1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 4 [w]: {shape=[3,3,4,7] dim=3 pattern=both_neg [-3,-1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 5 [c]: {shape=[3,3,4,7] dim=3 pattern=both_neg [-3,-1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 6 [config]: {shape=[3,3,4,7] dim=-1 pattern=stride2 [none,none) step=2}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 7 [config]: {shape=[7] dim=0 pattern=both_neg [-3,-1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 8 [w]: {shape=[7] dim=0 pattern=both_neg [-3,-1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 9 [h]: {shape=[7] dim=0 pattern=both_neg [-3,-1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 10 [h]: {shape=[7] dim=0 pattern=both_neg [-3,-1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 11 [h]: {shape=[7] dim=0 pattern=both_neg [-3,-1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 12 [w]: {shape=[7] dim=0 pattern=both_neg [-3,-1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 13 [c]: {shape=[7] dim=0 pattern=both_neg [-3,-1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 14 [n]: {shape=[7] dim=0 pattern=both_neg [-3,-1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 15 [n]: {shape=[7] dim=0 pattern=both_neg [-3,-1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 16 [w]: {shape=[7] dim=0 pattern=both_neg [-3,-1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 17 [config]: {shape=[3,2,7] dim=-3 pattern=both_neg [-3,-1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 18 [config]: {shape=[4,3,2,7] dim=1 pattern=stride3_offset [1,none) step=3}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 19 [c]: {shape=[4,3,2,7] dim=1 pattern=stride3_offset [1,none) step=3}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 20 [n]: {shape=[3,3,2,7] dim=1 pattern=stride3_offset [1,none) step=3}
    [spec] torch.ops.aten.slice.Tensor: matched |}]

(* Every one of [Recipe_sdpa.all_mask_kinds], enumerated directly rather than
   hoped for from a seed (op8-impl-review.md P2, round 2). The shared random
   walk draws one axis uniformly from all seven (batch/heads/sq/sk/e/mask/
   scale) each step, so a given mask kind is only drawn when BOTH the mask
   axis is picked AND its value lands on that kind -- "bridge coverage"'s
   three-step budget below never draws a present mask at all on sdpa's
   per-index seed (every step there reads mask=none), and a step count large
   enough to guarantee all 21 kinds by chance runs into the hundreds (coupon
   collector over ~7x21 draws, not the handful of steps every other dedicated
   walk in this file uses). Walking [all_mask_kinds] directly is exhaustive by
   construction and immune to any future change to [Pcg]'s draw sequence --
   the actual gap P2 reported: the fixup that introduced [all_mask_kinds] made
   every combination REPRESENTABLE, but nothing had made every combination
   REACHED.

   Every line must read "matched": a mismatch here would mean a combined mask
   broadcast (two or more of D/H/Wq/Wk collapsed to extent 1 together)
   disagrees with ATen even though each axis broadcast alone is covered by
   the other tests and by test/data/sdpa's fixtures. *)
let%expect_test "scaled_dot_product_attention.default: every mask kind" =
  capture (fun ppf ->
      List.iter
        (fun mask ->
          let cfg =
            {
              Aten_op_walk.Sdpa_walk.initial with
              Aten_walk_recipes.Recipe_sdpa.mask;
            }
          in
          Format.fprintf ppf "%a@." Aten_op_walk.Sdpa_walk.pp cfg;
          let subject, _pcg =
            Aten_op_walk.Sdpa_walk.build (Pcg.seed ~seed:1L ~seq:1L) cfg
          in
          assert (Aten_spec_run.run ~ppf subject))
        Aten_walk_recipes.Recipe_sdpa.all_mask_kinds);
  [%expect
    {|
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=none scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=2d(q=false,k=false) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=2d(q=false,k=true) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=2d(q=true,k=false) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=2d(q=true,k=true) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=4d(d=false,h=false,q=false,k=false) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=4d(d=false,h=false,q=false,k=true) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=4d(d=false,h=false,q=true,k=false) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=4d(d=false,h=false,q=true,k=true) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=4d(d=false,h=true,q=false,k=false) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=4d(d=false,h=true,q=false,k=true) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=4d(d=false,h=true,q=true,k=false) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=4d(d=false,h=true,q=true,k=true) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=4d(d=true,h=false,q=false,k=false) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=4d(d=true,h=false,q=false,k=true) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=4d(d=true,h=false,q=true,k=false) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=4d(d=true,h=false,q=true,k=true) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=4d(d=true,h=true,q=false,k=false) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=4d(d=true,h=true,q=false,k=true) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=4d(d=true,h=true,q=true,k=false) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    {batch=1 heads=2 sq=3 sk=4 e=5 mask=4d(d=true,h=true,q=true,k=true) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched |}]

(* Every target in the golden's `needs_meta:` list below has no Op_bridge
   dispatch arm at all (confirmed by grepping op_bridge.ml for each target
   string) -- there is nothing to walk yet, not a walk recipe someone forgot.
   select.int/unsqueeze.default/cat.default/squeeze.dims/expand.default are
   tracked as future structural-op work; the rest
   (eq.Scalar/any.dim/where.self/full_like.default/reshape.default/
   _softmax.default/batch_norm.default/dropout(_).default/argmax.default/
   topk.default) are outside this repo's current op coverage entirely. A
   target that DOES have a dispatch arm but still lands here is the real
   bug this comment exists to keep visible -- check with grep before adding
   a new entry to this list without a walk_meta recipe alongside it. *)
let%expect_test "bridge coverage" =
  capture (fun ppf ->
      List.iteri
        (fun i m ->
          assert (
            Op_walk.run m ~ppf
              ~pcg:(Pcg.seed ~seed:(Int64.of_int i) ~seq:1L)
              ~steps:3))
        Aten_op_walk.all_walks;
      Format.fprintf ppf "needs_meta:@.";
      List.iter (fun t -> Format.fprintf ppf "  %s@." t) Aten_op_walk.needs_meta);
  [%expect
    {|
    step 0: {shape=[2,4,8,8] pattern=equal}
    [spec] torch.ops.aten.add.Tensor: matched
    step 1 [w]: {shape=[2,4,8,16] pattern=equal}
    [spec] torch.ops.aten.add.Tensor: matched
    step 2 [h]: {shape=[2,4,8,16] pattern=equal}
    [spec] torch.ops.aten.add.Tensor: matched
    step 3 [w]: {shape=[2,4,8,8] pattern=equal}
    [spec] torch.ops.aten.add.Tensor: matched
    step 0: {shape=[2,4,8,8] pattern=equal}
    [spec] torch.ops.aten.add_.Tensor: matched
    step 1 [h]: {shape=[2,4,16,8] pattern=equal}
    [spec] torch.ops.aten.add_.Tensor: matched
    step 2 [c]: {shape=[2,8,16,8] pattern=equal}
    [spec] torch.ops.aten.add_.Tensor: matched
    step 3 [w]: {shape=[2,8,16,4] pattern=equal}
    [spec] torch.ops.aten.add_.Tensor: matched
    step 0: {shape=[2,4,8,8] pattern=equal}
    [spec] torch.ops.aten.sub.Tensor: matched
    step 1 [pattern]: {shape=[2,4,8,8] pattern=rhs[0]=1}
    [spec] torch.ops.aten.sub.Tensor: matched
    step 2 [pattern]: {shape=[2,4,8,8] pattern=lhs[3]=1}
    [spec] torch.ops.aten.sub.Tensor: matched
    step 3 [pattern]: {shape=[2,4,8,8] pattern=rhs[2]=1}
    [spec] torch.ops.aten.sub.Tensor: matched
    step 0: {shape=[2,4,8,8] pattern=equal}
    [spec] torch.ops.aten.mul.Tensor: matched
    step 1 [w]: {shape=[2,4,8,4] pattern=equal}
    [spec] torch.ops.aten.mul.Tensor: matched
    step 2 [h]: {shape=[2,4,4,4] pattern=equal}
    [spec] torch.ops.aten.mul.Tensor: matched
    step 3 [n]: {shape=[2,4,4,4] pattern=equal}
    [spec] torch.ops.aten.mul.Tensor: matched
    step 0: {shape=[1,4,8,8] value=float:0.1}
    [spec] torch.ops.aten.mul.Scalar: matched
    step 1 [n]: {shape=[2,4,8,8] value=float:0.1}
    [spec] torch.ops.aten.mul.Scalar: matched
    step 2 [w]: {shape=[2,4,8,8] value=float:0.1}
    [spec] torch.ops.aten.mul.Scalar: matched
    step 3 [value]: {shape=[2,4,8,8] value=int:-3}
    [spec] torch.ops.aten.mul.Scalar: matched
    step 0: {shape=[2,4,8,8] pattern=equal}
    [spec] torch.ops.aten.div.Tensor: matched
    step 1 [c]: {shape=[2,4,8,8] pattern=equal}
    [spec] torch.ops.aten.div.Tensor: matched
    step 2 [n]: {shape=[4,4,8,8] pattern=equal}
    [spec] torch.ops.aten.div.Tensor: matched
    step 3 [pattern]: {shape=[4,4,8,8] pattern=lhs[3]=1}
    [spec] torch.ops.aten.div.Tensor: matched
    step 0: {shape=[1,4,8,8] min=0 max=6}
    [spec] torch.ops.aten.clamp.default: matched
    step 1 [n]: {shape=[1,4,8,8] min=0 max=6}
    [spec] torch.ops.aten.clamp.default: matched
    step 2 [c]: {shape=[1,3,8,8] min=0 max=6}
    [spec] torch.ops.aten.clamp.default: matched
    step 3 [h]: {shape=[1,3,8,8] min=0 max=6}
    [spec] torch.ops.aten.clamp.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.logical_not.default: skipped (no native impl)
    step 1 [shape]: {shape=[8,3,4,4]}
    [spec] torch.ops.aten.logical_not.default: skipped (no native impl)
    step 2 [shape]: {shape=[8,3,2,4]}
    [spec] torch.ops.aten.logical_not.default: skipped (no native impl)
    step 3 [shape]: {shape=[6,3,2,4]}
    [spec] torch.ops.aten.logical_not.default: skipped (no native impl)
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.relu.default: matched
    step 1 [shape]: {shape=[8,3,4,4]}
    [spec] torch.ops.aten.relu.default: matched
    step 2 [shape]: {shape=[8,3,4,4]}
    [spec] torch.ops.aten.relu.default: matched
    step 3 [shape]: {shape=[8,4,4,4]}
    [spec] torch.ops.aten.relu.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.relu_.default: matched
    step 1 [shape]: {shape=[2,3,2,4]}
    [spec] torch.ops.aten.relu_.default: matched
    step 2 [shape]: {shape=[3,3,2,4]}
    [spec] torch.ops.aten.relu_.default: matched
    step 3 [shape]: {shape=[3,3,8,4]}
    [spec] torch.ops.aten.relu_.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.sigmoid.default: matched
    step 1 [shape]: {shape=[4,3,4,4]}
    [spec] torch.ops.aten.sigmoid.default: matched
    step 2 [shape]: {shape=[4,3,3,4]}
    [spec] torch.ops.aten.sigmoid.default: matched
    step 3 [shape]: {shape=[3,3,3,4]}
    [spec] torch.ops.aten.sigmoid.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.hardsigmoid.default: matched
    step 1 [shape]: {shape=[8,3,4,4]}
    [spec] torch.ops.aten.hardsigmoid.default: matched
    step 2 [shape]: {shape=[8,3,4,4]}
    [spec] torch.ops.aten.hardsigmoid.default: matched
    step 3 [shape]: {shape=[4,3,4,4]}
    [spec] torch.ops.aten.hardsigmoid.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.hardsigmoid_.default: matched
    step 1 [shape]: {shape=[2,3,2,4]}
    [spec] torch.ops.aten.hardsigmoid_.default: matched
    step 2 [shape]: {shape=[2,6,2,4]}
    [spec] torch.ops.aten.hardsigmoid_.default: matched
    step 3 [shape]: {shape=[2,2,2,4]}
    [spec] torch.ops.aten.hardsigmoid_.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.hardswish.default: matched
    step 1 [shape]: {shape=[2,3,4,2]}
    [spec] torch.ops.aten.hardswish.default: matched
    step 2 [shape]: {shape=[2,3,4,3]}
    [spec] torch.ops.aten.hardswish.default: matched
    step 3 [shape]: {shape=[2,3,4,3]}
    [spec] torch.ops.aten.hardswish.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.hardswish_.default: matched
    step 1 [shape]: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.hardswish_.default: matched
    step 2 [shape]: {shape=[2,3,6,4]}
    [spec] torch.ops.aten.hardswish_.default: matched
    step 3 [shape]: {shape=[2,3,6,4]}
    [spec] torch.ops.aten.hardswish_.default: matched
    step 0: {shape=[1,4,8,8] min=0 max=6}
    [spec] torch.ops.aten.hardtanh.default: matched
    step 1 [n]: {shape=[2,4,8,8] min=0 max=6}
    [spec] torch.ops.aten.hardtanh.default: matched
    step 2 [c]: {shape=[2,4,8,8] min=0 max=6}
    [spec] torch.ops.aten.hardtanh.default: matched
    step 3 [n]: {shape=[2,4,8,8] min=0 max=6}
    [spec] torch.ops.aten.hardtanh.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.hardtanh_.default: matched
    step 1 [shape]: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.hardtanh_.default: matched
    step 2 [shape]: {shape=[2,2,4,4]}
    [spec] torch.ops.aten.hardtanh_.default: matched
    step 3 [shape]: {shape=[2,2,6,4]}
    [spec] torch.ops.aten.hardtanh_.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.silu.default: matched
    step 1 [shape]: {shape=[2,3,6,4]}
    [spec] torch.ops.aten.silu.default: matched
    step 2 [shape]: {shape=[2,3,6,4]}
    [spec] torch.ops.aten.silu.default: matched
    step 3 [shape]: {shape=[2,3,6,4]}
    [spec] torch.ops.aten.silu.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.silu_.default: matched
    step 1 [shape]: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.silu_.default: matched
    step 2 [shape]: {shape=[2,3,4,6]}
    [spec] torch.ops.aten.silu_.default: matched
    step 3 [shape]: {shape=[2,3,3,6]}
    [spec] torch.ops.aten.silu_.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.flatten.using_ints: skipped (no native impl)
    step 1 [shape]: {shape=[6,3,4,4]}
    [spec] torch.ops.aten.flatten.using_ints: skipped (no native impl)
    step 2 [shape]: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.flatten.using_ints: skipped (no native impl)
    step 3 [shape]: {shape=[2,3,3,4]}
    [spec] torch.ops.aten.flatten.using_ints: skipped (no native impl)
    step 0: {shape=[4,5] rank=2 dims=[0,1]}
    [spec] torch.ops.aten.permute.default: matched
    step 1 [c]: {shape=[4,5] rank=2 dims=[0,1]}
    [spec] torch.ops.aten.permute.default: matched
    step 2 [h]: {shape=[3,5] rank=2 dims=[0,1]}
    [spec] torch.ops.aten.permute.default: matched
    step 3 [h]: {shape=[2,5] rank=2 dims=[0,1]}
    [spec] torch.ops.aten.permute.default: matched
    step 0: {shape=[4,5] rank=2 dims=(0,1)}
    [spec] torch.ops.aten.transpose.int: matched
    step 1 [n]: {shape=[4,5] rank=2 dims=(0,1)}
    [spec] torch.ops.aten.transpose.int: matched
    step 2 [w]: {shape=[4,3] rank=2 dims=(0,1)}
    [spec] torch.ops.aten.transpose.int: matched
    step 3 [config]: {shape=[3,4,3] rank=3 dims=(1,2)}
    [spec] torch.ops.aten.transpose.int: matched
    step 0: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.convolution.default: matched
    step 1 [out_channels]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=4 n=1 H=8 W=8}
    [spec] torch.ops.aten.convolution.default: matched
    step 2 [in_channels]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=4 n=1 H=8 W=8}
    [spec] torch.ops.aten.convolution.default: matched
    step 3 [input_h]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=4 n=1 H=14 W=8}
    [spec] torch.ops.aten.convolution.default: matched
    step 0: {mat1=[4,8] mat2=[8,6] self=[6]}
    [spec] torch.ops.aten.addmm.default: matched
    step 1 [out_features]: {mat1=[4,8] mat2=[8,1] self=[1]}
    [spec] torch.ops.aten.addmm.default: matched
    step 2 [n]: {mat1=[1,8] mat2=[8,1] self=[1]}
    [spec] torch.ops.aten.addmm.default: matched
    step 3 [in_features]: {mat1=[1,8] mat2=[8,1] self=[1]}
    [spec] torch.ops.aten.addmm.default: matched
    step 0: {self=[2,3,4] mat2=[2,4,5]}
    [spec] torch.ops.aten.bmm.default: matched
    step 1 [n]: {self=[2,3,4] mat2=[2,4,5]}
    [spec] torch.ops.aten.bmm.default: matched
    step 2 [n]: {self=[2,5,4] mat2=[2,4,5]}
    [spec] torch.ops.aten.bmm.default: matched
    step 3 [p]: {self=[2,5,4] mat2=[2,4,2]}
    [spec] torch.ops.aten.bmm.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.gelu.default: matched
    step 1 [shape]: {shape=[6,3,4,4]}
    [spec] torch.ops.aten.gelu.default: matched
    step 2 [shape]: {shape=[6,3,4,3]}
    [spec] torch.ops.aten.gelu.default: matched
    step 3 [shape]: {shape=[8,3,4,3]}
    [spec] torch.ops.aten.gelu.default: matched
    step 0: {input=[2,3,4] normalized=[4] eps=1e-05 weight=true bias=true cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 1 [cudnn_enable]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=true bias=true cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 2 [weight]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=false bias=true cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 3 [cudnn_enable]: {input=[2,3,4] normalized=[4] eps=1e-05 weight=false bias=true cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 0: {input=[2,4,8,8] eps=1e-05 weight=true bias=true}
    [spec] torch.ops.aten._native_batch_norm_legit_no_training.default: matched
    step 1 [weight]: {input=[2,4,8,8] eps=1e-05 weight=false bias=true}
    [spec] torch.ops.aten._native_batch_norm_legit_no_training.default: matched
    step 2 [bias]: {input=[2,4,8,8] eps=1e-05 weight=false bias=true}
    [spec] torch.ops.aten._native_batch_norm_legit_no_training.default: matched
    step 3 [w]: {input=[2,4,8,8] eps=1e-05 weight=false bias=true}
    [spec] torch.ops.aten._native_batch_norm_legit_no_training.default: matched
    step 0: {input=[2,3,4] normalized=[4] eps=1e-06 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 1 [leading]: {input=[2,3,4,4] normalized=[4] eps=1e-06 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 2 [eps]: {input=[2,3,4,4] normalized=[4] eps=1e-06 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 3 [weight]: {input=[2,3,4,4] normalized=[4] eps=1e-06 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 0: {input=[2,3,4] normalized=[4] eps=default weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 1 [weight]: {input=[2,3,4] normalized=[4] eps=default weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 2 [weight]: {input=[2,3,4] normalized=[4] eps=default weight=false bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 3 [weight]: {input=[2,3,4] normalized=[4] eps=default weight=false bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 0: {batch=1 heads=2 sq=3 sk=4 e=5 mask=none scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    step 1 [sk]: {batch=1 heads=2 sq=3 sk=1 e=5 mask=none scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    step 2 [mask]: {batch=1 heads=2 sq=3 sk=1 e=5 mask=4d(d=false,h=true,q=true,k=false) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    step 3 [sq]: {batch=1 heads=2 sq=5 sk=1 e=5 mask=4d(d=false,h=true,q=true,k=false) scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    step 0: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d_with_indices.default: matched
    step 1 [c]: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d_with_indices.default: matched
    step 2 [stride_h]: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d_with_indices.default: matched
    step 3 [stride_h]: {kernel=2x2 stride=3x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d_with_indices.default: matched
    step 0: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 1 [kernel_w]: {kernel=2x3 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 2 [kernel_h]: {kernel=3x3 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 3 [n]: {kernel=3x3 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 0: {shape=[1,4,8,8] output_size=[4,4]}
    [spec] torch.ops.aten.adaptive_avg_pool2d.default: matched
    step 1 [input_h]: {shape=[1,4,12,8] output_size=[4,4]}
    [spec] torch.ops.aten.adaptive_avg_pool2d.default: matched
    step 2 [out_h]: {shape=[1,4,12,8] output_size=[7,4]}
    [spec] torch.ops.aten.adaptive_avg_pool2d.default: matched
    step 3 [n]: {shape=[1,4,12,8] output_size=[7,4]}
    [spec] torch.ops.aten.adaptive_avg_pool2d.default: matched
    step 0: {input=[4,8] out_features=6 bias=true}
    [spec] torch.ops.aten.linear.default: matched
    step 1 [in_features]: {input=[4,8] out_features=6 bias=true}
    [spec] torch.ops.aten.linear.default: matched
    step 2 [leading]: {input=[2,3,8] out_features=6 bias=true}
    [spec] torch.ops.aten.linear.default: matched
    step 3 [out_features]: {input=[2,3,8] out_features=1 bias=true}
    [spec] torch.ops.aten.linear.default: matched
    step 0: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.conv2d.default: matched
    step 1 [pad_h]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.conv2d.default: matched
    step 2 [groups]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=4 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.conv2d.default: matched
    step 3 [pad_w]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=4 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.conv2d.default: matched
    step 0: {kernel=3x3 stride=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8 padding=same}
    [spec] torch.ops.aten.conv2d.padding: matched
    step 1 [dilation_h]: {kernel=3x3 stride=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8 padding=same}
    [spec] torch.ops.aten.conv2d.padding: matched
    step 2 [dilation_h]: {kernel=3x3 stride=1x1 dilation=2x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8 padding=same}
    [spec] torch.ops.aten.conv2d.padding: matched
    step 3 [input_w]: {kernel=3x3 stride=1x1 dilation=2x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=12 padding=same}
    [spec] torch.ops.aten.conv2d.padding: matched
    step 0: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.avg_pool2d.default: skipped (no native impl)
    step 1 [input_h]: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=10 W=8 ceil_mode=false}
    [spec] torch.ops.aten.avg_pool2d.default: skipped (no native impl)
    step 2 [ceil_mode]: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=10 W=8 ceil_mode=true}
    [spec] torch.ops.aten.avg_pool2d.default: skipped (no native impl)
    step 3 [pad_w]: {kernel=2x2 stride=2x2 pad=0x1 n=1 c=4 H=10 W=8 ceil_mode=true}
    [spec] torch.ops.aten.avg_pool2d.default: skipped (no native impl)
    step 0: {shape=[1,4,4,4] pattern=flatten target=[64]}
    [spec] torch.ops.aten.view.default: matched
    step 1 [w]: {shape=[1,4,4,3] pattern=flatten target=[48]}
    [spec] torch.ops.aten.view.default: matched
    step 2 [c]: {shape=[1,4,4,3] pattern=flatten target=[48]}
    [spec] torch.ops.aten.view.default: matched
    step 3 [w]: {shape=[1,4,4,4] pattern=flatten target=[64]}
    [spec] torch.ops.aten.view.default: matched
    step 0: {shape=[1,4,4,4] pattern=flatten target=[64]}
    [spec] torch.ops.aten._unsafe_view.default: matched
    step 1 [h]: {shape=[1,4,4,4] pattern=flatten target=[64]}
    [spec] torch.ops.aten._unsafe_view.default: matched
    step 2 [n]: {shape=[2,4,4,4] pattern=flatten target=[128]}
    [spec] torch.ops.aten._unsafe_view.default: matched
    step 3 [h]: {shape=[2,4,2,4] pattern=flatten target=[64]}
    [spec] torch.ops.aten._unsafe_view.default: matched
    step 0: {shape=[1,3,4,4] pattern=const_w pad=[1,2] mode=constant value=0.1}
    [spec] torch.ops.aten.pad.default: matched
    step 1 [pattern]: {shape=[1,3,4,4] pattern=crop_and_pad pad=[-1,0,2,1] mode=constant value=0.1}
    [spec] torch.ops.aten.pad.default: matched
    step 2 [w]: {shape=[1,3,4,4] pattern=crop_and_pad pad=[-1,0,2,1] mode=constant value=0.1}
    [spec] torch.ops.aten.pad.default: matched
    step 3 [h]: {shape=[1,3,4,4] pattern=crop_and_pad pad=[-1,0,2,1] mode=constant value=0.1}
    [spec] torch.ops.aten.pad.default: matched
    step 0: {shape=[2,3,4,5] dim=0 pattern=head [0,1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 1 [h]: {shape=[2,3,4,5] dim=0 pattern=head [0,1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 2 [n]: {shape=[2,3,4,5] dim=0 pattern=head [0,1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 3 [h]: {shape=[2,3,4,5] dim=0 pattern=head [0,1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 0: {shape=[2,3,4,4] dim=0}
    [spec] torch.ops.aten.unbind.int: matched
    step 1 [h]: {shape=[2,3,6,4] dim=0}
    [spec] torch.ops.aten.unbind.int: matched
    step 2 [n]: {shape=[2,3,6,4] dim=0}
    [spec] torch.ops.aten.unbind.int: matched
    step 3 [c]: {shape=[2,3,6,4] dim=0}
    [spec] torch.ops.aten.unbind.int: matched
    step 0: {shape=[2,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.mean.dim: matched
    step 1 [n]: {shape=[1,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.mean.dim: matched
    step 2 [keepdim]: {shape=[1,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.mean.dim: matched
    step 3 [keepdim]: {shape=[1,4,8,8] dims=[2,3] keepdim=true}
    [spec] torch.ops.aten.mean.dim: matched
    step 0: {shape=[2,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.amax.default: matched
    step 1 [w]: {shape=[2,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.amax.default: matched
    step 2 [dims]: {shape=[2,4,8,8] dims=[2] keepdim=false}
    [spec] torch.ops.aten.amax.default: matched
    step 3 [h]: {shape=[2,4,4,8] dims=[2] keepdim=false}
    [spec] torch.ops.aten.amax.default: matched
    step 0: {shape=[1,4,8,8] value=float:0.5}
    [spec] torch.ops.aten.pow.Tensor_Scalar: matched
    step 1 [value]: {shape=[1,4,8,8] value=float:1.5}
    [spec] torch.ops.aten.pow.Tensor_Scalar: matched
    step 2 [n]: {shape=[2,4,8,8] value=float:1.5}
    [spec] torch.ops.aten.pow.Tensor_Scalar: matched
    step 3 [w]: {shape=[2,4,8,4] value=float:1.5}
    [spec] torch.ops.aten.pow.Tensor_Scalar: matched
    step 0: {shape=[2,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.linalg_vector_norm.default: matched
    step 1 [w]: {shape=[2,4,8,4] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.linalg_vector_norm.default: matched
    step 2 [n]: {shape=[2,4,8,4] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.linalg_vector_norm.default: matched
    step 3 [h]: {shape=[2,4,4,4] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.linalg_vector_norm.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.clone.default: matched
    step 1 [shape]: {shape=[8,3,4,4]}
    [spec] torch.ops.aten.clone.default: matched
    step 2 [shape]: {shape=[8,2,4,4]}
    [spec] torch.ops.aten.clone.default: matched
    step 3 [shape]: {shape=[8,3,4,4]}
    [spec] torch.ops.aten.clone.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.cpu.default: skipped (no native impl)
    step 1 [shape]: {shape=[2,3,8,4]}
    [spec] torch.ops.aten.cpu.default: skipped (no native impl)
    step 2 [shape]: {shape=[2,3,8,6]}
    [spec] torch.ops.aten.cpu.default: skipped (no native impl)
    step 3 [shape]: {shape=[2,3,4,6]}
    [spec] torch.ops.aten.cpu.default: skipped (no native impl)
    needs_meta:
      torch.ops.aten.eq.Scalar
      torch.ops.aten.any.dim
      torch.ops.aten.where.self
      torch.ops.aten.full_like.default
      torch.ops.aten.reshape.default
      torch.ops.aten._softmax.default
      torch.ops.aten.cat.default
      torch.ops.aten.stack.default
      torch.ops.aten.batch_norm.default
      torch.ops.aten.dropout.default
      torch.ops.aten.dropout_.default
      torch.ops.aten.expand.default
      torch.ops.aten.select.int
      torch.ops.aten.split_with_sizes.default
      torch.ops.aten.squeeze.dims
      torch.ops.aten.unsqueeze.default
      torch.ops.aten.argmax.default
      torch.ops.aten.topk.default |}]
