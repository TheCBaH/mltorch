10-step walks with ATen-vs-native comparison, over the 4 of resnet18's 9 real
op targets (see .ai/pt2_node_spec_design.md) that actually have a walk recipe:
convolution, relu, max_pool2d_with_indices, mean.dim. The other 5
(_native_batch_norm_legit_no_training, add.Tensor, view, permute, addmm) have
no axis recipe yet (Aten_op_walk.needs_meta) and are deliberately not in this
list — nothing to walk for them here, so they're left out rather than
included just to print "no walk recipe".

Reuses the existing generated walk modules (lib/aten_op_walk, see
.ai/native_walk_design.md) rather than the pt2-derived JSON fixtures directly:
each walk starts from its own small synthetic config (not resnet18's real
shapes — those are exercised by pt2_node_spec_cram.t/pt2_node_walk_cram.t
instead) and mutates one *structural* axis per step (kernel/stride/dilation/
groups/channel-count/shape — not just the tensor's random fill, see
bin/native_walk_run.ml), comparing ATen against the native engine at every
step.

conv2d is deliberately excluded from the pt2-fixture-based comparison cram
tests (its ATen-vs-native comparison takes ~25s even on resnet18's smallest
conv) - the toy-scale config here is unaffected by that, since it's a few
elements, not a real 224x224 image.

  $ for t in \
  >   torch.ops.aten.convolution.default \
  >   torch.ops.aten.relu.default \
  >   torch.ops.aten.max_pool2d_with_indices.default \
  >   torch.ops.aten.mean.dim; do
  >   echo "=== $t ==="
  >   ../bin/native_walk_run.exe "$t" 10
  > done
  === torch.ops.aten.convolution.default ===
  step 0: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
  torch.ops.aten.convolution.default(input=f32[1,4,8,8]~values, weight=f32[8,4,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,1], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,8,8,8] min=-6.014 max=4.557 mean=-0.2906
    status: matched
  step 1 [n]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
  torch.ops.aten.convolution.default(input=f32[1,4,8,8]~values, weight=f32[8,4,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,1], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,8,8,8] min=-6.941 max=6.8 mean=0.2413
    status: matched
  step 2 [pad_w]: {kernel=3x3 stride=1x1 pad=1x0 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
  torch.ops.aten.convolution.default(input=f32[1,4,8,8]~values, weight=f32[8,4,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,0], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,8,8,6] min=-5.976 max=6.637 mean=-0.133
    status: matched
  step 3 [pad_h]: {kernel=3x3 stride=1x1 pad=1x0 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
  torch.ops.aten.convolution.default(input=f32[1,4,8,8]~values, weight=f32[8,4,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,0], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,8,8,6] min=-5.926 max=7.4 mean=0.1433
    status: matched
  step 4 [pad_w]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
  torch.ops.aten.convolution.default(input=f32[1,4,8,8]~values, weight=f32[8,4,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,1], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,8,8,8] min=-6.251 max=5.717 mean=-0.03416
    status: matched
  step 5 [input_w]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=12}
  torch.ops.aten.convolution.default(input=f32[1,4,8,12]~values, weight=f32[8,4,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,1], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,8,8,12] min=-6.056 max=5.874 mean=0.1114
    status: matched
  step 6 [input_h]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=12 W=12}
  torch.ops.aten.convolution.default(input=f32[1,4,12,12]~values, weight=f32[8,4,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,1], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,8,12,12] min=-6.347 max=7.189 mean=0.04896
    status: matched
  step 7 [groups]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=4 in_c=4 out_c=8 n=1 H=12 W=12}
  torch.ops.aten.convolution.default(input=f32[1,4,12,12]~values, weight=f32[8,1,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,1], dilation=[1,1], transposed=false, output_padding=[0,0], groups=4)
    -> out0: [1,8,12,12] min=-3.963 max=3.53 mean=0.1434
    status: matched
  step 8 [dilation_w]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x2 groups=4 in_c=4 out_c=8 n=1 H=12 W=12}
  torch.ops.aten.convolution.default(input=f32[1,4,12,12]~values, weight=f32[8,1,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,1], dilation=[1,2], transposed=false, output_padding=[0,0], groups=4)
    -> out0: [1,8,12,10] min=-3.537 max=3.296 mean=0.2432
    status: matched
  step 9 [in_channels]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x2 groups=4 in_c=16 out_c=8 n=1 H=12 W=12}
  torch.ops.aten.convolution.default(input=f32[1,16,12,12]~values, weight=f32[8,4,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,1], dilation=[1,2], transposed=false, output_padding=[0,0], groups=4)
    -> out0: [1,8,12,10] min=-7.664 max=6.362 mean=-0.04285
    status: matched
  step 10 [pad_h]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x2 groups=4 in_c=16 out_c=8 n=1 H=12 W=12}
  torch.ops.aten.convolution.default(input=f32[1,16,12,12]~values, weight=f32[8,4,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,1], dilation=[1,2], transposed=false, output_padding=[0,0], groups=4)
    -> out0: [1,8,12,10] min=-6.253 max=6.409 mean=0.2715
    status: matched
  === torch.ops.aten.relu.default ===
  step 0: {shape=[2,3,4,4]}
  torch.ops.aten.relu.default(self=f32[2,3,4,4]~values)
    -> out0: [2,3,4,4] min=0 max=0.9684 mean=0.2463
    status: matched
  step 1 [shape]: {shape=[2,3,4,8]}
  torch.ops.aten.relu.default(self=f32[2,3,4,8]~values)
    -> out0: [2,3,4,8] min=0 max=0.983 mean=0.2658
    status: matched
  step 2 [shape]: {shape=[2,3,4,6]}
  torch.ops.aten.relu.default(self=f32[2,3,4,6]~values)
    -> out0: [2,3,4,6] min=0 max=0.9908 mean=0.2143
    status: matched
  step 3 [shape]: {shape=[2,2,4,6]}
  torch.ops.aten.relu.default(self=f32[2,2,4,6]~values)
    -> out0: [2,2,4,6] min=0 max=0.9897 mean=0.2558
    status: matched
  step 4 [shape]: {shape=[2,2,4,6]}
  torch.ops.aten.relu.default(self=f32[2,2,4,6]~values)
    -> out0: [2,2,4,6] min=0 max=0.9899 mean=0.2599
    status: matched
  step 5 [shape]: {shape=[2,2,4,6]}
  torch.ops.aten.relu.default(self=f32[2,2,4,6]~values)
    -> out0: [2,2,4,6] min=0 max=0.9818 mean=0.2989
    status: matched
  step 6 [shape]: {shape=[2,2,4,6]}
  torch.ops.aten.relu.default(self=f32[2,2,4,6]~values)
    -> out0: [2,2,4,6] min=0 max=0.9986 mean=0.2395
    status: matched
  step 7 [shape]: {shape=[2,2,3,6]}
  torch.ops.aten.relu.default(self=f32[2,2,3,6]~values)
    -> out0: [2,2,3,6] min=0 max=0.8925 mean=0.2168
    status: matched
  step 8 [shape]: {shape=[2,2,6,6]}
  torch.ops.aten.relu.default(self=f32[2,2,6,6]~values)
    -> out0: [2,2,6,6] min=0 max=0.996 mean=0.2506
    status: matched
  step 9 [shape]: {shape=[2,2,6,6]}
  torch.ops.aten.relu.default(self=f32[2,2,6,6]~values)
    -> out0: [2,2,6,6] min=0 max=0.9858 mean=0.2346
    status: matched
  step 10 [shape]: {shape=[2,2,2,6]}
  torch.ops.aten.relu.default(self=f32[2,2,2,6]~values)
    -> out0: [2,2,2,6] min=0 max=0.9443 mean=0.216
    status: matched
  === torch.ops.aten.max_pool2d_with_indices.default ===
  step 0: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,4,8,8]~values, kernel_size=[2,2], stride=[2,2], padding=[0,0], dilation=[1,1], ceil_mode=false)
    -> out0: [1,4,4,4] min=-0.4467 max=0.983 mean=0.616
    -> out1: [1,4,4,4] min=0 max=63 mean=31.73
    status: matched
  step 1 [stride_w]: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=false}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,4,8,8]~values, kernel_size=[2,2], stride=[2,2], padding=[0,0], dilation=[1,1], ceil_mode=false)
    -> out0: [1,4,4,4] min=-0.3017 max=0.9908 mean=0.6012
    -> out1: [1,4,4,4] min=0 max=63 mean=31.38
    status: matched
  step 2 [ceil_mode]: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=true}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,4,8,8]~values, kernel_size=[2,2], stride=[2,2], padding=[0,0], dilation=[1,1], ceil_mode=true)
    -> out0: [1,4,4,4] min=-0.07556 max=0.9899 mean=0.6308
    -> out1: [1,4,4,4] min=0 max=63 mean=31.7
    status: matched
  step 3 [stride_w]: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=true}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,4,8,8]~values, kernel_size=[2,2], stride=[2,2], padding=[0,0], dilation=[1,1], ceil_mode=true)
    -> out0: [1,4,4,4] min=-0.5801 max=0.9986 mean=0.5482
    -> out1: [1,4,4,4] min=0 max=63 mean=31.67
    status: matched
  step 4 [pad_w]: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=true}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,4,8,8]~values, kernel_size=[2,2], stride=[2,2], padding=[0,0], dilation=[1,1], ceil_mode=true)
    -> out0: [1,4,4,4] min=-0.2323 max=0.9858 mean=0.5984
    -> out1: [1,4,4,4] min=1 max=63 mean=32.59
    status: matched
  step 5 [kernel_w]: {kernel=2x3 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=true}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,4,8,8]~values, kernel_size=[2,3], stride=[2,2], padding=[0,0], dilation=[1,1], ceil_mode=true)
    -> out0: [1,4,4,4] min=-0.4286 max=0.9806 mean=0.6591
    -> out1: [1,4,4,4] min=0 max=63 mean=31.55
    status: matched
  step 6 [n]: {kernel=2x3 stride=2x2 pad=0x0 n=2 c=4 H=8 W=8 ceil_mode=true}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[2,4,8,8]~values, kernel_size=[2,3], stride=[2,2], padding=[0,0], dilation=[1,1], ceil_mode=true)
    -> out0: [2,4,4,4] min=-0.1861 max=0.9937 mean=0.7066
    -> out1: [2,4,4,4] min=0 max=63 mean=32.09
    status: matched
  step 7 [input_h]: {kernel=2x3 stride=2x2 pad=0x0 n=2 c=4 H=8 W=8 ceil_mode=true}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[2,4,8,8]~values, kernel_size=[2,3], stride=[2,2], padding=[0,0], dilation=[1,1], ceil_mode=true)
    -> out0: [2,4,4,4] min=-0.7068 max=0.9964 mean=0.6834
    -> out1: [2,4,4,4] min=1 max=63 mean=32.57
    status: matched
  step 8 [pad_w]: {kernel=2x3 stride=2x2 pad=0x1 n=2 c=4 H=8 W=8 ceil_mode=true}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[2,4,8,8]~values, kernel_size=[2,3], stride=[2,2], padding=[0,1], dilation=[1,1], ceil_mode=true)
    -> out0: [2,4,4,5] min=-0.6059 max=0.9965 mean=0.6344
    -> out1: [2,4,4,5] min=0 max=63 mean=31.73
    status: matched
  step 9 [pad_w]: {kernel=2x3 stride=2x2 pad=0x0 n=2 c=4 H=8 W=8 ceil_mode=true}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[2,4,8,8]~values, kernel_size=[2,3], stride=[2,2], padding=[0,0], dilation=[1,1], ceil_mode=true)
    -> out0: [2,4,4,4] min=-0.6785 max=0.9944 mean=0.6523
    -> out1: [2,4,4,4] min=0 max=63 mean=31.8
    status: matched
  step 10 [n]: {kernel=2x3 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8 ceil_mode=true}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,4,8,8]~values, kernel_size=[2,3], stride=[2,2], padding=[0,0], dilation=[1,1], ceil_mode=true)
    -> out0: [1,4,4,4] min=-0.3443 max=0.9955 mean=0.6974
    -> out1: [1,4,4,4] min=2 max=63 mean=31.45
    status: matched
  === torch.ops.aten.mean.dim ===
  step 0: {shape=[2,4,8,8] dims=[2,3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[2,4,8,8]~values, dim=[2,3], keepdim=false)
    -> out0: [2,4] min=-0.1234 max=0.1967 mean=-0.004231
    status: matched
  step 1 [h]: {shape=[2,4,4,8] dims=[2,3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[2,4,4,8]~values, dim=[2,3], keepdim=false)
    -> out0: [2,4] min=-0.1334 max=0.1789 mean=0.008679
    status: matched
  step 2 [c]: {shape=[2,16,4,8] dims=[2,3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[2,16,4,8]~values, dim=[2,3], keepdim=false)
    -> out0: [2,16] min=-0.1746 max=0.2484 mean=-0.02071
    status: matched
  step 3 [n]: {shape=[4,16,4,8] dims=[2,3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[4,16,4,8]~values, dim=[2,3], keepdim=false)
    -> out0: [4,16] min=-0.1924 max=0.2454 mean=0.004063
    status: matched
  step 4 [h]: {shape=[4,16,4,8] dims=[2,3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[4,16,4,8]~values, dim=[2,3], keepdim=false)
    -> out0: [4,16] min=-0.2343 max=0.2614 mean=-0.002311
    status: matched
  step 5 [dims]: {shape=[4,16,4,8] dims=[3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[4,16,4,8]~values, dim=[3], keepdim=false)
    -> out0: [4,16,4] min=-0.5966 max=0.4525 mean=-0.04213
    status: matched
  step 6 [n]: {shape=[1,16,4,8] dims=[3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[1,16,4,8]~values, dim=[3], keepdim=false)
    -> out0: [1,16,4] min=-0.4192 max=0.5305 mean=0.01959
    status: matched
  step 7 [w]: {shape=[1,16,4,16] dims=[3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[1,16,4,16]~values, dim=[3], keepdim=false)
    -> out0: [1,16,4] min=-0.3849 max=0.3673 mean=0.002148
    status: matched
  step 8 [w]: {shape=[1,16,4,8] dims=[3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[1,16,4,8]~values, dim=[3], keepdim=false)
    -> out0: [1,16,4] min=-0.4864 max=0.4261 mean=0.04278
    status: matched
  step 9 [n]: {shape=[4,16,4,8] dims=[3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[4,16,4,8]~values, dim=[3], keepdim=false)
    -> out0: [4,16,4] min=-0.5142 max=0.5029 mean=0.004892
    status: matched
  step 10 [h]: {shape=[4,16,16,8] dims=[3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[4,16,16,8]~values, dim=[3], keepdim=false)
    -> out0: [4,16,16] min=-0.6271 max=0.6917 mean=-0.01048
    status: matched
