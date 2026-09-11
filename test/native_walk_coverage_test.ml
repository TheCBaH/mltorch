(* Split out of native_walk_test.ml (the per-op family-recipe walks) once that
   file crossed the tracked 1000-line ceiling (scripts/check-file-size.sh):
   this half sweeps every generated walk module in one pass and prints the
   [needs_meta] backlog, rather than pinning any one op's own trace. *)

module Pcg = Aten_spec.Pcg

(* Buffer formatter output so ppx_expect can capture it. *)
let capture f = print_string (Core.Pretty.capture_to_string f)

(* Every target in the golden's `needs_meta:` list below has no Op_bridge
   dispatch arm with a walk recipe: some genuinely have no dispatch arm yet
   (untracked structural-op work); others have an arm but no [Walk_meta]
   entry, because a required non-tensor argument (no schema default) keeps
   them out of the generator's automatic Default tier -- see
   `.ai/native_walk_design.md`. A target that DOES have both an arm and a
   recipe but still lands here is the real bug this comment exists to keep
   visible -- check with grep before adding a new entry to this list without
   a walk_meta recipe alongside it. *)
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
    step 0: {input=[2,4,8,8] eps=1e-05 weight=true bias=true}
    [spec] torch.ops.aten._native_batch_norm_legit_no_training.default: matched
    step 1 [w]: {input=[2,4,8,8] eps=1e-05 weight=true bias=true}
    [spec] torch.ops.aten._native_batch_norm_legit_no_training.default: matched
    step 2 [bias]: {input=[2,4,8,8] eps=1e-05 weight=true bias=false}
    [spec] torch.ops.aten._native_batch_norm_legit_no_training.default: matched
    step 3 [weight]: {input=[2,4,8,8] eps=1e-05 weight=false bias=false}
    [spec] torch.ops.aten._native_batch_norm_legit_no_training.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten._to_copy.default: matched
    step 1 [shape]: {shape=[2,3,4,2]}
    [spec] torch.ops.aten._to_copy.default: matched
    step 2 [shape]: {shape=[2,3,4,4]}
    [spec] torch.ops.aten._to_copy.default: matched
    step 3 [shape]: {shape=[2,3,3,4]}
    [spec] torch.ops.aten._to_copy.default: matched
    step 0: {shape=[1,4,4,4] pattern=flatten target=[64]}
    [spec] torch.ops.aten._unsafe_view.default: matched
    step 1 [w]: {shape=[1,4,4,4] pattern=flatten target=[64]}
    [spec] torch.ops.aten._unsafe_view.default: matched
    step 2 [h]: {shape=[1,4,2,4] pattern=flatten target=[32]}
    [spec] torch.ops.aten._unsafe_view.default: matched
    step 3 [w]: {shape=[1,4,2,2] pattern=flatten target=[16]}
    [spec] torch.ops.aten._unsafe_view.default: matched
    step 0: {shape=[1,4,8,8] output_size=[4,4]}
    [spec] torch.ops.aten.adaptive_avg_pool2d.default: matched
    step 1 [input_w]: {shape=[1,4,8,6] output_size=[4,4]}
    [spec] torch.ops.aten.adaptive_avg_pool2d.default: matched
    step 2 [input_h]: {shape=[1,4,8,6] output_size=[4,4]}
    [spec] torch.ops.aten.adaptive_avg_pool2d.default: matched
    step 3 [out_h]: {shape=[1,4,8,6] output_size=[2,4]}
    [spec] torch.ops.aten.adaptive_avg_pool2d.default: matched
    step 0: {shape=[1,4,8,8] output_size=[4,4]}
    [spec] torch.ops.aten.adaptive_max_pool2d.default: matched
    step 1 [input_h]: {shape=[1,4,8,8] output_size=[4,4]}
    [spec] torch.ops.aten.adaptive_max_pool2d.default: matched
    step 2 [n]: {shape=[2,4,8,8] output_size=[4,4]}
    [spec] torch.ops.aten.adaptive_max_pool2d.default: matched
    step 3 [input_h]: {shape=[2,4,6,8] output_size=[4,4]}
    [spec] torch.ops.aten.adaptive_max_pool2d.default: matched
    step 0: {shape=[2,4,8,8] pattern=equal}
    [spec] torch.ops.aten.add.Tensor: matched
    step 1 [w]: {shape=[2,4,8,8] pattern=equal}
    [spec] torch.ops.aten.add.Tensor: matched
    step 2 [c]: {shape=[2,8,8,8] pattern=equal}
    [spec] torch.ops.aten.add.Tensor: matched
    step 3 [pattern]: {shape=[2,8,8,8] pattern=rhs[3]=1}
    [spec] torch.ops.aten.add.Tensor: matched
    step 0: {shape=[2,4,8,8] pattern=equal}
    [spec] torch.ops.aten.add_.Tensor: matched
    step 1 [n]: {shape=[2,4,8,8] pattern=equal}
    [spec] torch.ops.aten.add_.Tensor: matched
    step 2 [h]: {shape=[2,4,4,8] pattern=equal}
    [spec] torch.ops.aten.add_.Tensor: matched
    step 3 [c]: {shape=[2,16,4,8] pattern=equal}
    [spec] torch.ops.aten.add_.Tensor: matched
    step 0: {mat1=[4,8] mat2=[8,6] self=[6]}
    [spec] torch.ops.aten.addmm.default: matched
    step 1 [n]: {mat1=[2,8] mat2=[8,6] self=[6]}
    [spec] torch.ops.aten.addmm.default: matched
    step 2 [in_features]: {mat1=[2,8] mat2=[8,6] self=[6]}
    [spec] torch.ops.aten.addmm.default: matched
    step 3 [in_features]: {mat1=[2,16] mat2=[16,6] self=[6]}
    [spec] torch.ops.aten.addmm.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.alias.default: matched
    step 1 [shape]: {shape=[8,3,4,4]}
    [spec] torch.ops.aten.alias.default: matched
    step 2 [shape]: {shape=[8,3,4,4]}
    [spec] torch.ops.aten.alias.default: matched
    step 3 [shape]: {shape=[8,4,4,4]}
    [spec] torch.ops.aten.alias.default: matched
    step 0: {shape=[2,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.amax.default: matched
    step 1 [c]: {shape=[2,8,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.amax.default: matched
    step 2 [n]: {shape=[2,8,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.amax.default: matched
    step 3 [keepdim]: {shape=[2,8,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.amax.default: matched
    step 0: {start=int:0 end=int:5}
    [spec] torch.ops.aten.arange.default: matched
    step 1 [range]: {start=int:-3 end=int:2}
    [spec] torch.ops.aten.arange.default: matched
    step 2 [range]: {start=int:0 end=int:5}
    [spec] torch.ops.aten.arange.default: matched
    step 3 [range]: {start=float:-1.25 end=float:1.75}
    [spec] torch.ops.aten.arange.default: matched
    step 0: {start=int:0 end=int:5}
    [spec] torch.ops.aten.arange.start: matched
    step 1 [range]: {start=int:0 end=int:5}
    [spec] torch.ops.aten.arange.start: matched
    step 2 [range]: {start=int:0 end=int:5}
    [spec] torch.ops.aten.arange.start: matched
    step 3 [range]: {start=int:-3 end=int:2}
    [spec] torch.ops.aten.arange.start: matched
    step 0: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false} count_include_pad=true
    [spec] torch.ops.aten.avg_pool2d.default: matched
    step 1 [kernel_h]: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false} count_include_pad=true
    [spec] torch.ops.aten.avg_pool2d.default: matched
    step 2 [input_w]: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=16 ceil_mode=false} count_include_pad=true
    [spec] torch.ops.aten.avg_pool2d.default: matched
    step 3 [stride_h]: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=16 ceil_mode=false} count_include_pad=true
    [spec] torch.ops.aten.avg_pool2d.default: matched
    step 0: {self=[2,3,4] mat2=[2,4,5]}
    [spec] torch.ops.aten.bmm.default: matched
    step 1 [batch]: {self=[1,3,4] mat2=[1,4,5]}
    [spec] torch.ops.aten.bmm.default: matched
    step 2 [batch]: {self=[3,3,4] mat2=[3,4,5]}
    [spec] torch.ops.aten.bmm.default: matched
    step 3 [n]: {self=[3,5,4] mat2=[3,4,5]}
    [spec] torch.ops.aten.bmm.default: matched
    step 0: {shape=[2,3,4,4] dim=0}
    [spec] torch.ops.aten.cat.default: matched
    step 1 [c]: {shape=[2,3,4,4] dim=0}
    [spec] torch.ops.aten.cat.default: matched
    step 2 [n]: {shape=[1,3,4,4] dim=0}
    [spec] torch.ops.aten.cat.default: matched
    step 3 [h]: {shape=[1,3,4,4] dim=0}
    [spec] torch.ops.aten.cat.default: matched
    step 0: {shape=[1,4,8,8] min=0 max=6}
    [spec] torch.ops.aten.clamp.default: matched
    step 1 [n]: {shape=[2,4,8,8] min=0 max=6}
    [spec] torch.ops.aten.clamp.default: matched
    step 2 [c]: {shape=[2,4,8,8] min=0 max=6}
    [spec] torch.ops.aten.clamp.default: matched
    step 3 [n]: {shape=[2,4,8,8] min=0 max=6}
    [spec] torch.ops.aten.clamp.default: matched
    step 0: {shape=[1,4,8,8] value=float:0.1}
    [spec] torch.ops.aten.clamp_min.default: matched
    step 1 [c]: {shape=[1,4,8,8] value=float:0.1}
    [spec] torch.ops.aten.clamp_min.default: matched
    step 2 [w]: {shape=[1,4,8,4] value=float:0.1}
    [spec] torch.ops.aten.clamp_min.default: matched
    step 3 [c]: {shape=[1,3,8,4] value=float:0.1}
    [spec] torch.ops.aten.clamp_min.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.clone.default: matched
    step 1 [shape]: {shape=[2,3,6,4]}
    [spec] torch.ops.aten.clone.default: matched
    step 2 [shape]: {shape=[2,3,6,4]}
    [spec] torch.ops.aten.clone.default: matched
    step 3 [shape]: {shape=[2,3,6,4]}
    [spec] torch.ops.aten.clone.default: matched
    step 0: {k=3 s=1 p=1 d=1 g=1 in=4 out=8 n=1 len=8}
    [spec] torch.ops.aten.conv1d.default: matched
    step 1 [inc]: {k=3 s=1 p=1 d=1 g=1 in=4 out=8 n=1 len=8}
    [spec] torch.ops.aten.conv1d.default: matched
    step 2 [outc]: {k=3 s=1 p=1 d=1 g=1 in=4 out=8 n=1 len=8}
    [spec] torch.ops.aten.conv1d.default: matched
    step 3 [inc]: {k=3 s=1 p=1 d=1 g=1 in=4 out=8 n=1 len=8}
    [spec] torch.ops.aten.conv1d.default: matched
    step 0: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.conv2d.default: matched
    step 1 [stride_h]: {kernel=3x3 stride=2x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.conv2d.default: matched
    step 2 [pad_w]: {kernel=3x3 stride=2x1 pad=1x0 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.conv2d.default: matched
    step 3 [pad_w]: {kernel=3x3 stride=2x1 pad=1x2 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.conv2d.default: matched
    step 0: {kernel=3x3 stride=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8 padding=same}
    [spec] torch.ops.aten.conv2d.padding: matched
    step 1 [groups]: {kernel=3x3 stride=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8 padding=same}
    [spec] torch.ops.aten.conv2d.padding: matched
    step 2 [in_channels]: {kernel=3x3 stride=1x1 dilation=1x1 groups=1 in_c=16 out_c=8 n=1 H=8 W=8 padding=same}
    [spec] torch.ops.aten.conv2d.padding: matched
    step 3 [groups]: {kernel=3x3 stride=1x1 dilation=1x1 groups=2 in_c=16 out_c=8 n=1 H=8 W=8 padding=same}
    [spec] torch.ops.aten.conv2d.padding: matched
    step 0: {shape=[1,4,4,4] dim=0}
    [spec] torch.ops.aten.conv3d.default: matched
    step 1 [c]: {shape=[1,6,4,4] dim=0}
    [spec] torch.ops.aten.conv3d.default: matched
    step 2 [w]: {shape=[1,6,4,6] dim=0}
    [spec] torch.ops.aten.conv3d.default: matched
    step 3 [c]: {shape=[1,4,4,6] dim=0}
    [spec] torch.ops.aten.conv3d.default: matched
    step 0: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
    [spec] torch.ops.aten.convolution.default: matched
    step 1 [out_channels]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=4 n=1 H=8 W=8}
    [spec] torch.ops.aten.convolution.default: matched
    step 2 [in_channels]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=4 n=1 H=8 W=8}
    [spec] torch.ops.aten.convolution.default: matched
    step 3 [input_h]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=4 n=1 H=14 W=8}
    [spec] torch.ops.aten.convolution.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.cpu.default: skipped (no native impl)
    step 1 [shape]: {shape=[2,3,3,4]}
    [spec] torch.ops.aten.cpu.default: skipped (no native impl)
    step 2 [shape]: {shape=[2,3,6,4]}
    [spec] torch.ops.aten.cpu.default: skipped (no native impl)
    step 3 [shape]: {shape=[2,3,3,4]}
    [spec] torch.ops.aten.cpu.default: skipped (no native impl)
    step 0: {shape=[2,4,8,8] dims=[3] keepdim=false}
    [spec] torch.ops.aten.cumsum.default: matched
    step 1 [h]: {shape=[2,4,16,8] dims=[3] keepdim=false}
    [spec] torch.ops.aten.cumsum.default: matched
    step 2 [dims]: {shape=[2,4,16,8] dims=[0] keepdim=false}
    [spec] torch.ops.aten.cumsum.default: matched
    step 3 [c]: {shape=[2,16,16,8] dims=[0] keepdim=false}
    [spec] torch.ops.aten.cumsum.default: matched
    step 0: {shape=[2,4,8,8] pattern=equal}
    [spec] torch.ops.aten.div.Tensor: matched
    step 1 [w]: {shape=[2,4,8,4] pattern=equal}
    [spec] torch.ops.aten.div.Tensor: matched
    step 2 [pattern]: {shape=[2,4,8,4] pattern=rhs[3]=1}
    [spec] torch.ops.aten.div.Tensor: matched
    step 3 [n]: {shape=[2,4,8,4] pattern=rhs[3]=1}
    [spec] torch.ops.aten.div.Tensor: matched
    step 0: {self=[2,3,4,4] broadcast=none size=[2,3,4,4]}
    [spec] torch.ops.aten.expand.default: matched
    step 1 [pattern]: {self=[2,1,4,4] broadcast=c size=[2,3,4,4]}
    [spec] torch.ops.aten.expand.default: matched
    step 2 [pattern]: {self=[2,3,4,4] broadcast=none size=[2,3,4,4]}
    [spec] torch.ops.aten.expand.default: matched
    step 3 [h]: {self=[2,3,1,4] broadcast=none size=[2,3,1,4]}
    [spec] torch.ops.aten.expand.default: matched
    step 0: {n=3 m=4}
    [spec] torch.ops.aten.eye.m: matched
    step 1 [n]: {n=2 m=4}
    [spec] torch.ops.aten.eye.m: matched
    step 2 [n]: {n=2 m=4}
    [spec] torch.ops.aten.eye.m: matched
    step 3 [n]: {n=3 m=4}
    [spec] torch.ops.aten.eye.m: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.flatten.using_ints: skipped (no native impl)
    step 1 [shape]: {shape=[2,4,4,4]}
    [spec] torch.ops.aten.flatten.using_ints: skipped (no native impl)
    step 2 [shape]: {shape=[2,2,4,4]}
    [spec] torch.ops.aten.flatten.using_ints: skipped (no native impl)
    step 3 [shape]: {shape=[2,2,4,2]}
    [spec] torch.ops.aten.flatten.using_ints: skipped (no native impl)
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.gelu.default: matched
    step 1 [shape]: {shape=[2,3,4,6]}
    [spec] torch.ops.aten.gelu.default: matched
    step 2 [shape]: {shape=[2,6,4,6]}
    [spec] torch.ops.aten.gelu.default: matched
    step 3 [shape]: {shape=[2,2,4,6]}
    [spec] torch.ops.aten.gelu.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.hardsigmoid.default: matched
    step 1 [shape]: {shape=[2,3,8,4]}
    [spec] torch.ops.aten.hardsigmoid.default: matched
    step 2 [shape]: {shape=[2,6,8,4]}
    [spec] torch.ops.aten.hardsigmoid.default: matched
    step 3 [shape]: {shape=[8,6,8,4]}
    [spec] torch.ops.aten.hardsigmoid.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.hardsigmoid_.default: matched
    step 1 [shape]: {shape=[4,3,4,4]}
    [spec] torch.ops.aten.hardsigmoid_.default: matched
    step 2 [shape]: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.hardsigmoid_.default: matched
    step 3 [shape]: {shape=[3,3,4,4]}
    [spec] torch.ops.aten.hardsigmoid_.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.hardswish.default: matched
    step 1 [shape]: {shape=[2,3,3,4]}
    [spec] torch.ops.aten.hardswish.default: matched
    step 2 [shape]: {shape=[2,3,3,4]}
    [spec] torch.ops.aten.hardswish.default: matched
    step 3 [shape]: {shape=[2,3,3,4]}
    [spec] torch.ops.aten.hardswish.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.hardswish_.default: matched
    step 1 [shape]: {shape=[2,3,3,4]}
    [spec] torch.ops.aten.hardswish_.default: matched
    step 2 [shape]: {shape=[2,3,3,4]}
    [spec] torch.ops.aten.hardswish_.default: matched
    step 3 [shape]: {shape=[8,3,3,4]}
    [spec] torch.ops.aten.hardswish_.default: matched
    step 0: {shape=[1,4,8,8] min=0 max=6}
    [spec] torch.ops.aten.hardtanh.default: matched
    step 1 [n]: {shape=[2,4,8,8] min=0 max=6}
    [spec] torch.ops.aten.hardtanh.default: matched
    step 2 [bounds]: {shape=[2,4,8,8] min=-1 max=1}
    [spec] torch.ops.aten.hardtanh.default: matched
    step 3 [bounds]: {shape=[2,4,8,8] min=0 max=6}
    [spec] torch.ops.aten.hardtanh.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.hardtanh_.default: matched
    step 1 [shape]: {shape=[2,2,4,4]}
    [spec] torch.ops.aten.hardtanh_.default: matched
    step 2 [shape]: {shape=[2,8,4,4]}
    [spec] torch.ops.aten.hardtanh_.default: matched
    step 3 [shape]: {shape=[2,8,4,4]}
    [spec] torch.ops.aten.hardtanh_.default: matched
    step 0: {input=[2,3,4] normalized=[4] eps=1e-05 weight=true bias=true cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 1 [leading]: {input=[2,3,4,4] normalized=[4] eps=1e-05 weight=true bias=true cudnn=true}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 2 [cudnn_enable]: {input=[2,3,4,4] normalized=[4] eps=1e-05 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 3 [weight]: {input=[2,3,4,4] normalized=[4] eps=1e-05 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.layer_norm.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.leaky_relu.default: matched
    step 1 [shape]: {shape=[2,3,2,4]}
    [spec] torch.ops.aten.leaky_relu.default: matched
    step 2 [shape]: {shape=[2,3,2,4]}
    [spec] torch.ops.aten.leaky_relu.default: matched
    step 3 [shape]: {shape=[8,3,2,4]}
    [spec] torch.ops.aten.leaky_relu.default: matched
    step 0: {shape=[2,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.linalg_vector_norm.default: matched
    step 1 [w]: {shape=[2,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.linalg_vector_norm.default: matched
    step 2 [c]: {shape=[2,8,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.linalg_vector_norm.default: matched
    step 3 [w]: {shape=[2,8,8,16] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.linalg_vector_norm.default: matched
    step 0: {input=[4,8] out_features=6 bias=true}
    [spec] torch.ops.aten.linear.default: matched
    step 1 [bias]: {input=[4,8] out_features=6 bias=true}
    [spec] torch.ops.aten.linear.default: matched
    step 2 [leading]: {input=[4,8] out_features=6 bias=true}
    [spec] torch.ops.aten.linear.default: matched
    step 3 [bias]: {input=[4,8] out_features=6 bias=false}
    [spec] torch.ops.aten.linear.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.logical_not.default: skipped (no native impl)
    step 1 [shape]: {shape=[6,3,4,4]}
    [spec] torch.ops.aten.logical_not.default: skipped (no native impl)
    step 2 [shape]: {shape=[6,3,4,8]}
    [spec] torch.ops.aten.logical_not.default: skipped (no native impl)
    step 3 [shape]: {shape=[8,3,4,8]}
    [spec] torch.ops.aten.logical_not.default: skipped (no native impl)
    step 0: {batch=3 seq=4 input=5 hidden=6 layers=1 bidirectional=false has_biases=true batch_first=false dropout=0}
    [spec] torch.ops.aten.lstm.input: matched
    step 1 [dropout]: {batch=3 seq=4 input=5 hidden=6 layers=1 bidirectional=false has_biases=true batch_first=false dropout=0}
    [spec] torch.ops.aten.lstm.input: matched
    step 2 [batch]: {batch=5 seq=4 input=5 hidden=6 layers=1 bidirectional=false has_biases=true batch_first=false dropout=0}
    [spec] torch.ops.aten.lstm.input: matched
    step 3 [batch]: {batch=5 seq=4 input=5 hidden=6 layers=1 bidirectional=false has_biases=true batch_first=false dropout=0}
    [spec] torch.ops.aten.lstm.input: matched
    step 0: {self=[1,1,3,4] other=[1,1,4,5] d_bc=neither h_bc=neither}
    [spec] torch.ops.aten.matmul.default: matched
    step 1 [d_bc]: {self=[1,1,3,4] other=[1,1,4,5] d_bc=neither h_bc=neither}
    [spec] torch.ops.aten.matmul.default: matched
    step 2 [other_rank]: {self=[1,1,3,4] other=[1,1,4,5] d_bc=neither h_bc=neither}
    [spec] torch.ops.aten.matmul.default: matched
    step 3 [h]: {self=[1,4,3,4] other=[1,4,4,5] d_bc=neither h_bc=neither}
    [spec] torch.ops.aten.matmul.default: matched
    step 0: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 1 [ceil_mode]: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 2 [pad_w]: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 3 [c]: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=8 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d.default: matched
    step 0: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d_with_indices.default: matched
    step 1 [pad_h]: {kernel=2x2 stride=2x2 pad=1x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d_with_indices.default: matched
    step 2 [stride_w]: {kernel=2x2 stride=2x2 pad=1x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d_with_indices.default: matched
    step 3 [input_h]: {kernel=2x2 stride=2x2 pad=1x0 n=1 c=4 H=8 W=8 ceil_mode=false}
    [spec] torch.ops.aten.max_pool2d_with_indices.default: matched
    step 0: {shape=[2,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.mean.dim: matched
    step 1 [dims]: {shape=[2,4,8,8] dims=[0,1,2] keepdim=false}
    [spec] torch.ops.aten.mean.dim: matched
    step 2 [dims]: {shape=[2,4,8,8] dims=[1,2] keepdim=false}
    [spec] torch.ops.aten.mean.dim: matched
    step 3 [w]: {shape=[2,4,8,8] dims=[1,2] keepdim=false}
    [spec] torch.ops.aten.mean.dim: matched
    step 0: {shape=[1,4,8,8] value=float:0.1}
    [spec] torch.ops.aten.mul.Scalar: matched
    step 1 [c]: {shape=[1,3,8,8] value=float:0.1}
    [spec] torch.ops.aten.mul.Scalar: matched
    step 2 [h]: {shape=[1,3,8,8] value=float:0.1}
    [spec] torch.ops.aten.mul.Scalar: matched
    step 3 [h]: {shape=[1,3,4,8] value=float:0.1}
    [spec] torch.ops.aten.mul.Scalar: matched
    step 0: {shape=[2,4,8,8] pattern=equal}
    [spec] torch.ops.aten.mul.Tensor: matched
    step 1 [n]: {shape=[1,4,8,8] pattern=equal}
    [spec] torch.ops.aten.mul.Tensor: matched
    step 2 [c]: {shape=[1,4,8,8] pattern=equal}
    [spec] torch.ops.aten.mul.Tensor: matched
    step 3 [pattern]: {shape=[1,4,8,8] pattern=rhs[1]=1}
    [spec] torch.ops.aten.mul.Tensor: matched
    step 0: {input=[2,3,4] normalized=[4] eps=1e-06 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 1 [leading]: {input=[2,3,4] normalized=[4] eps=1e-06 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 2 [weight]: {input=[2,3,4] normalized=[4] eps=1e-06 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 3 [leading]: {input=[2,3,4] normalized=[4] eps=1e-06 weight=true bias=true cudnn=false}
    [spec] torch.ops.aten.native_layer_norm.default: matched
    step 0: {shape=[1,3,4,4] pattern=const_w pad=[1,2] mode=constant value=0.1}
    [spec] torch.ops.aten.pad.default: matched
    step 1 [h]: {shape=[1,3,4,4] pattern=const_w pad=[1,2] mode=constant value=0.1}
    [spec] torch.ops.aten.pad.default: matched
    step 2 [h]: {shape=[1,3,6,4] pattern=const_w pad=[1,2] mode=constant value=0.1}
    [spec] torch.ops.aten.pad.default: matched
    step 3 [pattern]: {shape=[1,3,6,4] pattern=const_hw pad=[1,2,3,1] mode=constant value=-2.5}
    [spec] torch.ops.aten.pad.default: matched
    step 0: {shape=[4,5] rank=2 dims=[0,1]}
    [spec] torch.ops.aten.permute.default: matched
    step 1 [config]: {shape=[3,4,5] rank=3 dims=[0,1,2]}
    [spec] torch.ops.aten.permute.default: matched
    step 2 [n]: {shape=[3,4,5] rank=3 dims=[0,1,2]}
    [spec] torch.ops.aten.permute.default: matched
    step 3 [w]: {shape=[3,4,4] rank=3 dims=[0,1,2]}
    [spec] torch.ops.aten.permute.default: matched
    step 0: {shape=[1,4,8,8] value=float:0.5}
    [spec] torch.ops.aten.pow.Tensor_Scalar: matched
    step 1 [w]: {shape=[1,4,8,8] value=float:0.5}
    [spec] torch.ops.aten.pow.Tensor_Scalar: matched
    step 2 [h]: {shape=[1,4,8,8] value=float:0.5}
    [spec] torch.ops.aten.pow.Tensor_Scalar: matched
    step 3 [c]: {shape=[1,8,8,8] value=float:0.5}
    [spec] torch.ops.aten.pow.Tensor_Scalar: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.relu.default: matched
    step 1 [shape]: {shape=[6,3,4,4]}
    [spec] torch.ops.aten.relu.default: matched
    step 2 [shape]: {shape=[6,4,4,4]}
    [spec] torch.ops.aten.relu.default: matched
    step 3 [shape]: {shape=[6,4,6,4]}
    [spec] torch.ops.aten.relu.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.relu_.default: matched
    step 1 [shape]: {shape=[2,6,4,4]}
    [spec] torch.ops.aten.relu_.default: matched
    step 2 [shape]: {shape=[2,6,8,4]}
    [spec] torch.ops.aten.relu_.default: matched
    step 3 [shape]: {shape=[2,6,2,4]}
    [spec] torch.ops.aten.relu_.default: matched
    step 0: {input=[2,3,4] normalized=[4] eps=default weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 1 [weight]: {input=[2,3,4] normalized=[4] eps=default weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 2 [leading]: {input=[2,4] normalized=[4] eps=default weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 3 [weight]: {input=[2,4] normalized=[4] eps=default weight=true bias=false cudnn=false}
    [spec] torch.ops.aten.rms_norm.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.rsqrt.default: matched
    step 1 [shape]: {shape=[2,3,4,2]}
    [spec] torch.ops.aten.rsqrt.default: matched
    step 2 [shape]: {shape=[2,2,4,2]}
    [spec] torch.ops.aten.rsqrt.default: matched
    step 3 [shape]: {shape=[2,3,4,2]}
    [spec] torch.ops.aten.rsqrt.default: matched
    step 0: {shape=[1,4,8,8] value=float:0.1}
    [spec] torch.ops.aten.rsub.Scalar: matched
    step 1 [h]: {shape=[1,4,8,8] value=float:0.1}
    [spec] torch.ops.aten.rsub.Scalar: matched
    step 2 [w]: {shape=[1,4,8,4] value=float:0.1}
    [spec] torch.ops.aten.rsub.Scalar: matched
    step 3 [h]: {shape=[1,4,8,4] value=float:0.1}
    [spec] torch.ops.aten.rsub.Scalar: matched
    step 0: {batch=1 heads=2 sq=3 sk=4 e=5 batch_bc=real heads_bc=real mask=none scale=default}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    step 1 [scale]: {batch=1 heads=2 sq=3 sk=4 e=5 batch_bc=real heads_bc=real mask=none scale=0.1}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    step 2 [batch]: {batch=1 heads=2 sq=3 sk=4 e=5 batch_bc=real heads_bc=real mask=none scale=0.1}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    step 3 [sq]: {batch=1 heads=2 sq=1 sk=4 e=5 batch_bc=real heads_bc=real mask=none scale=0.1}
    [spec] torch.ops.aten.scaled_dot_product_attention.default: matched
    step 0: {shape=[2,3,4,4] rank=4 dim=2 index=0 (first)}
    [spec] torch.ops.aten.select.int: matched
    step 1 [w]: {shape=[2,3,4,7] rank=4 dim=2 index=0 (first)}
    [spec] torch.ops.aten.select.int: matched
    step 2 [h]: {shape=[2,3,4,7] rank=4 dim=2 index=0 (first)}
    [spec] torch.ops.aten.select.int: matched
    step 3 [config]: {shape=[2,3,4,7] rank=4 dim=0 index=1 (last)}
    [spec] torch.ops.aten.select.int: matched
    step 0: {shape=[2,3,4,4] rank=4 dim=2 index=0 (first)}
    [spec] torch.ops.aten.select_scatter.default: matched
    step 1 [config]: {shape=[2,3,4,4] rank=4 dim=-1 index=-1 (neg_last)}
    [spec] torch.ops.aten.select_scatter.default: matched
    step 2 [n]: {shape=[3,3,4,4] rank=4 dim=-1 index=-1 (neg_last)}
    [spec] torch.ops.aten.select_scatter.default: matched
    step 3 [config]: {shape=[4] rank=1 dim=0 index=0 (first)}
    [spec] torch.ops.aten.select_scatter.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.sigmoid.default: matched
    step 1 [shape]: {shape=[2,3,6,4]}
    [spec] torch.ops.aten.sigmoid.default: matched
    step 2 [shape]: {shape=[6,3,6,4]}
    [spec] torch.ops.aten.sigmoid.default: matched
    step 3 [shape]: {shape=[6,3,6,3]}
    [spec] torch.ops.aten.sigmoid.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.silu.default: matched
    step 1 [shape]: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.silu.default: matched
    step 2 [shape]: {shape=[2,8,4,4]}
    [spec] torch.ops.aten.silu.default: matched
    step 3 [shape]: {shape=[2,2,4,4]}
    [spec] torch.ops.aten.silu.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.silu_.default: matched
    step 1 [shape]: {shape=[2,3,8,4]}
    [spec] torch.ops.aten.silu_.default: matched
    step 2 [shape]: {shape=[2,8,8,4]}
    [spec] torch.ops.aten.silu_.default: matched
    step 3 [shape]: {shape=[2,8,8,4]}
    [spec] torch.ops.aten.silu_.default: matched
    step 0: {shape=[2,3,4,5] dim=0 pattern=head [0,1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 1 [w]: {shape=[2,3,4,2] dim=0 pattern=head [0,1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 2 [config]: {shape=[4,2] dim=1 pattern=head [0,1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 3 [h]: {shape=[2,2] dim=1 pattern=head [0,1) step=1}
    [spec] torch.ops.aten.slice.Tensor: matched
    step 0: {shape=[2,4,8,8] dims=[3] keepdim=false}
    [spec] torch.ops.aten.softmax.int: matched
    step 1 [h]: {shape=[2,4,4,8] dims=[3] keepdim=false}
    [spec] torch.ops.aten.softmax.int: matched
    step 2 [dims]: {shape=[2,4,4,8] dims=[1] keepdim=false}
    [spec] torch.ops.aten.softmax.int: matched
    step 3 [w]: {shape=[2,4,4,16] dims=[1] keepdim=false}
    [spec] torch.ops.aten.softmax.int: matched
    step 0: {shape=[2,3,4,4] dim=0 split_size=2}
    [spec] torch.ops.aten.split.Tensor: matched
    step 1 [h]: {shape=[2,3,4,4] dim=0 split_size=2}
    [spec] torch.ops.aten.split.Tensor: matched
    step 2 [n]: {shape=[2,3,4,4] dim=0 split_size=2}
    [spec] torch.ops.aten.split.Tensor: matched
    step 3 [n]: {shape=[4,3,4,4] dim=0 split_size=3}
    [spec] torch.ops.aten.split.Tensor: matched
    step 0: {shape=[2,3,4,4] dim=0 split_size=2}
    [spec] torch.ops.aten.split_with_sizes.default: matched
    step 1 [c]: {shape=[2,3,4,4] dim=0 split_size=2}
    [spec] torch.ops.aten.split_with_sizes.default: matched
    step 2 [w]: {shape=[2,3,4,5] dim=0 split_size=2}
    [spec] torch.ops.aten.split_with_sizes.default: matched
    step 3 [c]: {shape=[2,3,4,5] dim=0 split_size=2}
    [spec] torch.ops.aten.split_with_sizes.default: matched
    step 0: {shape=[2,3,4,4] dim=0}
    [spec] torch.ops.aten.stack.default: matched
    step 1 [n]: {shape=[1,3,4,4] dim=0}
    [spec] torch.ops.aten.stack.default: matched
    step 2 [h]: {shape=[1,3,2,4] dim=0}
    [spec] torch.ops.aten.stack.default: matched
    step 3 [dim]: {shape=[1,3,2,4] dim=-2}
    [spec] torch.ops.aten.stack.default: matched
    step 0: {shape=[2,4,8,8] pattern=equal}
    [spec] torch.ops.aten.sub.Tensor: matched
    step 1 [pattern]: {shape=[2,4,8,8] pattern=lhs[3]=1}
    [spec] torch.ops.aten.sub.Tensor: matched
    step 2 [h]: {shape=[2,4,16,8] pattern=lhs[3]=1}
    [spec] torch.ops.aten.sub.Tensor: matched
    step 3 [pattern]: {shape=[2,4,16,8] pattern=rhs[1]=1}
    [spec] torch.ops.aten.sub.Tensor: matched
    step 0: {shape=[2,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.sum.dim_IntList: matched
    step 1 [c]: {shape=[2,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.sum.dim_IntList: matched
    step 2 [w]: {shape=[2,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.sum.dim_IntList: matched
    step 3 [w]: {shape=[2,4,8,8] dims=[2,3] keepdim=false}
    [spec] torch.ops.aten.sum.dim_IntList: matched
    step 0: {shape=[4,5] rank=2 dims=(0,1)}
    [spec] torch.ops.aten.transpose.int: matched
    step 1 [config]: {shape=[2,3,4,5] rank=4 dims=(2,2)}
    [spec] torch.ops.aten.transpose.int: matched
    step 2 [w]: {shape=[2,3,4,4] rank=4 dims=(2,2)}
    [spec] torch.ops.aten.transpose.int: matched
    step 3 [config]: {shape=[2,3,4,4] rank=4 dims=(-2,-3)}
    [spec] torch.ops.aten.transpose.int: matched
    step 0: {shape=[2,3,4,4] dim=0}
    [spec] torch.ops.aten.unbind.int: matched
    step 1 [w]: {shape=[2,3,4,4] dim=0}
    [spec] torch.ops.aten.unbind.int: matched
    step 2 [dim]: {shape=[2,3,4,4] dim=3}
    [spec] torch.ops.aten.unbind.int: matched
    step 3 [n]: {shape=[2,3,4,4] dim=3}
    [spec] torch.ops.aten.unbind.int: matched
    step 0: {shape=[2,3,4,4] dim=2 size=3 step=1 overlap}
    [spec] torch.ops.aten.unfold.default: matched
    step 1 [w]: {shape=[2,3,4,5] dim=2 size=3 step=1 overlap}
    [spec] torch.ops.aten.unfold.default: matched
    step 2 [config]: {shape=[2,3,4,5] dim=1 size=3 step=2 stepped}
    [spec] torch.ops.aten.unfold.default: matched
    step 3 [config]: {shape=[2,3,4,5] dim=-2 size=3 step=2 stepped}
    [spec] torch.ops.aten.unfold.default: matched
    step 0: {shape=[2,3,4,4] dim=0}
    [spec] torch.ops.aten.unsqueeze.default: matched
    step 1 [n]: {shape=[1,3,4,4] dim=0}
    [spec] torch.ops.aten.unsqueeze.default: matched
    step 2 [dim]: {shape=[1,3,4,4] dim=0}
    [spec] torch.ops.aten.unsqueeze.default: matched
    step 3 [dim]: {shape=[1,3,4,4] dim=0}
    [spec] torch.ops.aten.unsqueeze.default: matched
    step 0: {shape=[1,4,4,4] pattern=flatten target=[64]}
    [spec] torch.ops.aten.view.default: matched
    step 1 [w]: {shape=[1,4,4,2] pattern=flatten target=[32]}
    [spec] torch.ops.aten.view.default: matched
    step 2 [n]: {shape=[1,4,4,2] pattern=flatten target=[32]}
    [spec] torch.ops.aten.view.default: matched
    step 3 [pattern]: {shape=[1,4,4,2] pattern=minus_one_c target=[-1,4,2]}
    [spec] torch.ops.aten.view.default: matched
    step 0: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.zeros.default: matched
    step 1 [shape]: {shape=[2,3,4,4]}
    [spec] torch.ops.aten.zeros.default: matched
    step 2 [shape]: {shape=[3,3,4,4]}
    [spec] torch.ops.aten.zeros.default: matched
    step 3 [shape]: {shape=[3,3,8,4]}
    [spec] torch.ops.aten.zeros.default: matched
    needs_meta:
      torch.ops.aten._softmax.default
      torch.ops.aten.any.dim
      torch.ops.aten.argmax.default
      torch.ops.aten.batch_norm.default
      torch.ops.aten.dropout.default
      torch.ops.aten.dropout_.default
      torch.ops.aten.eq.Scalar
      torch.ops.aten.full_like.default
      torch.ops.aten.reshape.default
      torch.ops.aten.squeeze.dims
      torch.ops.aten.topk.default
      torch.ops.aten.where.self |}]
