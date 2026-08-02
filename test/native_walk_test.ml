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
    step 0: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 1 [stride_h]: {kernel=2x2 stride=1x2 pad=0x0 n=1 c=4 H=8 W=8}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 2 [stride_w]: {kernel=2x2 stride=1x1 pad=0x0 n=1 c=4 H=8 W=8}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 3 [stride_w]: {kernel=2x2 stride=1x2 pad=0x0 n=1 c=4 H=8 W=8}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 4 [n]: {kernel=2x2 stride=1x2 pad=0x0 n=1 c=4 H=8 W=8}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 5 [pad_w]: {kernel=2x2 stride=1x2 pad=0x0 n=1 c=4 H=8 W=8}
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
    step 0: {shape=[1,4,8,8] min=0 max=6}
    [spec] torch.ops.aten.clamp.default: matched
    step 1 [bounds]: {shape=[1,4,8,8] min=none max=6}
    [spec] torch.ops.aten.clamp.default: matched
    step 2 [h]: {shape=[1,4,8,8] min=none max=6}
    [spec] torch.ops.aten.clamp.default: matched
    step 3 [bounds]: {shape=[1,4,8,8] min=0 max=nan}
    [spec] torch.ops.aten.clamp.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.logical_not.default: skipped (no native impl)
    step 1 [shape]: {shape=[2,3,4,2]}
    [spec] torch.ops.aten.logical_not.default: skipped (no native impl)
    step 2 [shape]: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.logical_not.default: skipped (no native impl)
    step 3 [shape]: {shape=[2,3,3,4]}
    [spec] torch.ops.aten.logical_not.default: skipped (no native impl)
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.relu.default: matched
    step 1 [shape]: {shape=[2,3,4,3]}
    [spec] torch.ops.aten.relu.default: matched
    step 2 [shape]: {shape=[3,3,4,3]}
    [spec] torch.ops.aten.relu.default: matched
    step 3 [shape]: {shape=[3,3,4,3]}
    [spec] torch.ops.aten.relu.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.relu_.default: matched
    step 1 [shape]: {shape=[2,3,8,4]}
    [spec] torch.ops.aten.relu_.default: matched
    step 2 [shape]: {shape=[2,3,2,4]}
    [spec] torch.ops.aten.relu_.default: matched
    step 3 [shape]: {shape=[2,6,2,4]}
    [spec] torch.ops.aten.relu_.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.sigmoid.default: skipped (no native impl)
    step 1 [shape]: {shape=[2,8,4,4]}
    [spec] torch.ops.aten.sigmoid.default: skipped (no native impl)
    step 2 [shape]: {shape=[2,4,4,4]}
    [spec] torch.ops.aten.sigmoid.default: skipped (no native impl)
    step 3 [shape]: {shape=[2,4,4,8]}
    [spec] torch.ops.aten.sigmoid.default: skipped (no native impl)
    step 0: {shape=[1,4,8,8] min=0 max=6}
    [spec] torch.ops.aten.hardtanh.default: matched
    step 1 [n]: {shape=[1,4,8,8] min=0 max=6}
    [spec] torch.ops.aten.hardtanh.default: matched
    step 2 [h]: {shape=[1,4,4,8] min=0 max=6}
    [spec] torch.ops.aten.hardtanh.default: matched
    step 3 [w]: {shape=[1,4,4,4] min=0 max=6}
    [spec] torch.ops.aten.hardtanh.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.hardtanh_.default: matched
    step 1 [shape]: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.hardtanh_.default: matched
    step 2 [shape]: {shape=[2,6,4,4]}
    [spec] torch.ops.aten.hardtanh_.default: matched
    step 3 [shape]: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.hardtanh_.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.silu_.default: skipped (no native impl)
    step 1 [shape]: {shape=[8,3,4,4]}
    [spec] torch.ops.aten.silu_.default: skipped (no native impl)
    step 2 [shape]: {shape=[8,3,2,4]}
    [spec] torch.ops.aten.silu_.default: skipped (no native impl)
    step 3 [shape]: {shape=[6,3,2,4]}
    [spec] torch.ops.aten.silu_.default: skipped (no native impl)
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.flatten.using_ints: skipped (no native impl)
    step 1 [shape]: {shape=[8,3,4,4]}
    [spec] torch.ops.aten.flatten.using_ints: skipped (no native impl)
    step 2 [shape]: {shape=[8,3,4,4]}
    [spec] torch.ops.aten.flatten.using_ints: skipped (no native impl)
    step 3 [shape]: {shape=[8,4,4,4]}
    [spec] torch.ops.aten.flatten.using_ints: skipped (no native impl)
    step 0: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.convolution.default: matched
    step 1 [groups]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=2 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.convolution.default: matched
    step 2 [dilation_w]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=2 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.convolution.default: matched
    step 3 [pad_h]: {kernel=3x3 stride=1x1 pad=0x1 dilation=1x1 groups=2 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.convolution.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.gelu.default: skipped (no native impl)
    step 1 [shape]: {shape=[4,3,4,4]}
    [spec] torch.ops.aten.gelu.default: skipped (no native impl)
    step 2 [shape]: {shape=[4,3,3,4]}
    [spec] torch.ops.aten.gelu.default: skipped (no native impl)
    step 3 [shape]: {shape=[3,3,3,4]}
    [spec] torch.ops.aten.gelu.default: skipped (no native impl)
    step 0: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8}
    [spec] torch.ops.aten.max_pool2d_with_indices.default: matched
    step 1 [c]: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=8 H=8 W=8}
    [spec] torch.ops.aten.max_pool2d_with_indices.default: matched
    step 2 [stride_w]: {kernel=2x2 stride=2x1 pad=0x0 n=1 c=8 H=8 W=8}
    [spec] torch.ops.aten.max_pool2d_with_indices.default: matched
    step 3 [kernel_w]: {kernel=2x4 stride=2x1 pad=0x0 n=1 c=8 H=8 W=8}
    [spec] torch.ops.aten.max_pool2d_with_indices.default: matched
    step 0: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 1 [kernel_w]: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 2 [n]: {kernel=2x2 stride=2x2 pad=0x0 n=2 c=4 H=8 W=8}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 3 [stride_w]: {kernel=2x2 stride=2x2 pad=0x0 n=2 c=4 H=8 W=8}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 0: {shape=[1,4,8,8] output_size=[4,4]}
    [spec] torch.ops.aten.adaptive_avg_pool2d.default: skipped (no native impl)
    step 1 [c]: {shape=[1,4,8,8] output_size=[4,4]}
    [spec] torch.ops.aten.adaptive_avg_pool2d.default: skipped (no native impl)
    step 2 [c]: {shape=[1,4,8,8] output_size=[4,4]}
    [spec] torch.ops.aten.adaptive_avg_pool2d.default: skipped (no native impl)
    step 3 [out_w]: {shape=[1,4,8,8] output_size=[4,7]}
    [spec] torch.ops.aten.adaptive_avg_pool2d.default: skipped (no native impl)
    step 0: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.conv2d.default: matched
    step 1 [n]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=2 H=8 W=8}
    [spec] torch.ops.aten.conv2d.default: matched
    step 2 [pad_h]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=2 H=8 W=8}
    [spec] torch.ops.aten.conv2d.default: matched
    step 3 [groups]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=4 in_c=4 out_c=8 n=2 H=8 W=8}
    [spec] torch.ops.aten.conv2d.default: matched
    step 0: {kernel=3x3 stride=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8 padding=same}
    [spec] torch.ops.aten.conv2d.padding: matched
    step 1 [input_h]: {kernel=3x3 stride=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=10 W=8 padding=same}
    [spec] torch.ops.aten.conv2d.padding: matched
    step 2 [kernel_w]: {kernel=3x1 stride=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=10 W=8 padding=same}
    [spec] torch.ops.aten.conv2d.padding: matched
    step 3 [dilation_h]: {kernel=3x1 stride=1x1 dilation=2x1 groups=1 in_c=4 out_c=8 n=1 H=10 W=8 padding=same}
    [spec] torch.ops.aten.conv2d.padding: matched
    step 0: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8}
    [spec] torch.ops.aten.avg_pool2d.default: skipped (aten interp: unhandled op)
    step 1 [kernel_w]: {kernel=2x3 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8}
    [spec] torch.ops.aten.avg_pool2d.default: skipped (aten interp: unhandled op)
    step 2 [input_h]: {kernel=2x3 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8}
    [spec] torch.ops.aten.avg_pool2d.default: skipped (aten interp: unhandled op)
    step 3 [n]: {kernel=2x3 stride=2x2 pad=0x0 n=2 c=4 H=8 W=8}
    [spec] torch.ops.aten.avg_pool2d.default: skipped (aten interp: unhandled op)
    step 0: {shape=[2,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.mean.dim: matched
    step 1 [n]: {shape=[1,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.mean.dim: matched
    step 2 [c]: {shape=[1,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.mean.dim: matched
    step 3 [c]: {shape=[1,16,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.mean.dim: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.clone.default: matched
    step 1 [shape]: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.clone.default: matched
    step 2 [shape]: {shape=[2,3,4,6]}
    [spec] torch.ops.aten.clone.default: matched
    step 3 [shape]: {shape=[2,3,3,6]}
    [spec] torch.ops.aten.clone.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.cpu.default: skipped (no native impl)
    step 1 [shape]: {shape=[6,3,4,4]}
    [spec] torch.ops.aten.cpu.default: skipped (no native impl)
    step 2 [shape]: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.cpu.default: skipped (no native impl)
    step 3 [shape]: {shape=[2,3,3,4]}
    [spec] torch.ops.aten.cpu.default: skipped (no native impl)
    needs_meta:
      torch.ops.aten.add.Tensor
      torch.ops.aten.add_.Tensor
      torch.ops.aten.mul.Tensor
      torch.ops.aten.mul.Scalar
      torch.ops.aten.div.Tensor
      torch.ops.aten.eq.Scalar
      torch.ops.aten.any.dim
      torch.ops.aten.where.self
      torch.ops.aten.full_like.default
      torch.ops.aten.reshape.default
      torch.ops.aten.permute.default
      torch.ops.aten.addmm.default
      torch.ops.aten.bmm.default
      torch.ops.aten._softmax.default
      torch.ops.aten.cat.default
      torch.ops.aten._native_batch_norm_legit_no_training.default
      torch.ops.aten.native_layer_norm.default
      torch.ops.aten.rms_norm.default
      torch.ops.aten.linear.default
      torch.ops.aten.batch_norm.default
      torch.ops.aten.dropout.default
      torch.ops.aten.dropout_.default
      torch.ops.aten.view.default
      torch.ops.aten.expand.default
      torch.ops.aten.select.int
      torch.ops.aten.squeeze.dims
      torch.ops.aten.unsqueeze.default
      torch.ops.aten.argmax.default
      torch.ops.aten.topk.default |}]
