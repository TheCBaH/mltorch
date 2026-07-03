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
    -> out0: [1,8,8,8] min=-6.01405 max=4.55686 mean=-0.29064
    status: matched
  step 1 [n]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
  torch.ops.aten.convolution.default(input=f32[1,4,8,8]~values, weight=f32[8,4,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,1], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,8,8,8] min=-6.94096 max=6.7996 mean=0.241328
    status: matched
  step 2 [pad_w]: {kernel=3x3 stride=1x1 pad=1x0 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
  torch.ops.aten.convolution.default(input=f32[1,4,8,8]~values, weight=f32[8,4,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,0], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,8,8,6] min=-5.97562 max=6.63734 mean=-0.133047
    status: matched
  step 3 [pad_h]: {kernel=3x3 stride=1x1 pad=1x0 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
  torch.ops.aten.convolution.default(input=f32[1,4,8,8]~values, weight=f32[8,4,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,0], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,8,8,6] min=-5.92639 max=7.40028 mean=0.143297
    status: matched
  step 4 [pad_w]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=8}
  torch.ops.aten.convolution.default(input=f32[1,4,8,8]~values, weight=f32[8,4,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,1], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,8,8,8] min=-6.25144 max=5.71717 mean=-0.0341615
    status: matched
  step 5 [input_w]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=8 W=12}
  torch.ops.aten.convolution.default(input=f32[1,4,8,12]~values, weight=f32[8,4,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,1], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,8,8,12] min=-6.05644 max=5.87415 mean=0.111367
    status: matched
  step 6 [input_h]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 in_c=4 out_c=8 n=1 H=12 W=12}
  torch.ops.aten.convolution.default(input=f32[1,4,12,12]~values, weight=f32[8,4,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,1], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,8,12,12] min=-6.34679 max=7.18934 mean=0.0489586
    status: matched
  step 7 [groups]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=4 in_c=4 out_c=8 n=1 H=12 W=12}
  torch.ops.aten.convolution.default(input=f32[1,4,12,12]~values, weight=f32[8,1,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,1], dilation=[1,1], transposed=false, output_padding=[0,0], groups=4)
    -> out0: [1,8,12,12] min=-3.96333 max=3.5304 mean=0.143408
    status: matched
  step 8 [dilation_w]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x2 groups=4 in_c=4 out_c=8 n=1 H=12 W=12}
  torch.ops.aten.convolution.default(input=f32[1,4,12,12]~values, weight=f32[8,1,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,1], dilation=[1,2], transposed=false, output_padding=[0,0], groups=4)
    -> out0: [1,8,12,10] min=-3.5366 max=3.29633 mean=0.243246
    status: matched
  step 9 [in_channels]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x2 groups=4 in_c=16 out_c=8 n=1 H=12 W=12}
  torch.ops.aten.convolution.default(input=f32[1,16,12,12]~values, weight=f32[8,4,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,1], dilation=[1,2], transposed=false, output_padding=[0,0], groups=4)
    -> out0: [1,8,12,10] min=-7.66387 max=6.36203 mean=-0.0428473
    status: matched
  step 10 [pad_h]: {kernel=3x3 stride=1x1 pad=1x1 dilation=1x2 groups=4 in_c=16 out_c=8 n=1 H=12 W=12}
  torch.ops.aten.convolution.default(input=f32[1,16,12,12]~values, weight=f32[8,4,3,3]~values, bias=f32[8]~values, stride=[1,1], padding=[1,1], dilation=[1,2], transposed=false, output_padding=[0,0], groups=4)
    -> out0: [1,8,12,10] min=-6.25334 max=6.40886 mean=0.271495
    status: matched
  === torch.ops.aten.relu.default ===
  step 0: {shape=[2,3,4,4]}
  torch.ops.aten.relu.default(self=f32[2,3,4,4]~values)
    -> out0: [2,3,4,4] min=0 max=0.968412 mean=0.246306
    status: matched
  step 1 [shape]: {shape=[2,3,4,8]}
  torch.ops.aten.relu.default(self=f32[2,3,4,8]~values)
    -> out0: [2,3,4,8] min=0 max=0.982979 mean=0.265762
    status: matched
  step 2 [shape]: {shape=[2,3,4,6]}
  torch.ops.aten.relu.default(self=f32[2,3,4,6]~values)
    -> out0: [2,3,4,6] min=0 max=0.990828 mean=0.214318
    status: matched
  step 3 [shape]: {shape=[2,2,4,6]}
  torch.ops.aten.relu.default(self=f32[2,2,4,6]~values)
    -> out0: [2,2,4,6] min=0 max=0.989652 mean=0.255756
    status: matched
  step 4 [shape]: {shape=[2,2,4,6]}
  torch.ops.aten.relu.default(self=f32[2,2,4,6]~values)
    -> out0: [2,2,4,6] min=0 max=0.989859 mean=0.259856
    status: matched
  step 5 [shape]: {shape=[2,2,4,6]}
  torch.ops.aten.relu.default(self=f32[2,2,4,6]~values)
    -> out0: [2,2,4,6] min=0 max=0.981829 mean=0.298877
    status: matched
  step 6 [shape]: {shape=[2,2,4,6]}
  torch.ops.aten.relu.default(self=f32[2,2,4,6]~values)
    -> out0: [2,2,4,6] min=0 max=0.998559 mean=0.239489
    status: matched
  step 7 [shape]: {shape=[2,2,3,6]}
  torch.ops.aten.relu.default(self=f32[2,2,3,6]~values)
    -> out0: [2,2,3,6] min=0 max=0.892483 mean=0.216828
    status: matched
  step 8 [shape]: {shape=[2,2,6,6]}
  torch.ops.aten.relu.default(self=f32[2,2,6,6]~values)
    -> out0: [2,2,6,6] min=0 max=0.996044 mean=0.250636
    status: matched
  step 9 [shape]: {shape=[2,2,6,6]}
  torch.ops.aten.relu.default(self=f32[2,2,6,6]~values)
    -> out0: [2,2,6,6] min=0 max=0.985757 mean=0.234579
    status: matched
  step 10 [shape]: {shape=[2,2,2,6]}
  torch.ops.aten.relu.default(self=f32[2,2,2,6]~values)
    -> out0: [2,2,2,6] min=0 max=0.944347 mean=0.21595
    status: matched
  === torch.ops.aten.max_pool2d_with_indices.default ===
  step 0: {kernel=2x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,4,8,8]~values, kernel_size=[2,2], stride=[2,2], padding=[0,0], dilation=[1,1], ceil_mode=false)
    -> out0: [1,4,4,4] min=-0.44667 max=0.982979 mean=0.616017
    -> out1: [1,4,4,4] <non-f32>
    status: skipped (no native impl)
  step 1 [kernel_h]: {kernel=3x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,4,8,8]~values, kernel_size=[3,2], stride=[2,2], padding=[0,0], dilation=[1,1], ceil_mode=false)
    -> out0: [1,4,3,4] min=0.0443707 max=0.990828 mean=0.689316
    -> out1: [1,4,3,4] <non-f32>
    status: skipped (no native impl)
  step 2 [input_w]: {kernel=3x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,4,8,8]~values, kernel_size=[3,2], stride=[2,2], padding=[0,0], dilation=[1,1], ceil_mode=false)
    -> out0: [1,4,3,4] min=0.0538757 max=0.989859 mean=0.753792
    -> out1: [1,4,3,4] <non-f32>
    status: skipped (no native impl)
  step 3 [stride_h]: {kernel=3x2 stride=2x2 pad=0x0 n=1 c=4 H=8 W=8}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,4,8,8]~values, kernel_size=[3,2], stride=[2,2], padding=[0,0], dilation=[1,1], ceil_mode=false)
    -> out0: [1,4,3,4] min=-0.310519 max=0.998559 mean=0.672365
    -> out1: [1,4,3,4] <non-f32>
    status: skipped (no native impl)
  step 4 [stride_h]: {kernel=3x2 stride=3x2 pad=0x0 n=1 c=4 H=8 W=8}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,4,8,8]~values, kernel_size=[3,2], stride=[3,2], padding=[0,0], dilation=[1,1], ceil_mode=false)
    -> out0: [1,4,2,4] min=0.0130346 max=0.985757 mean=0.751554
    -> out1: [1,4,2,4] <non-f32>
    status: skipped (no native impl)
  step 5 [input_h]: {kernel=3x2 stride=3x2 pad=0x0 n=1 c=4 H=12 W=8}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,4,12,8]~values, kernel_size=[3,2], stride=[3,2], padding=[0,0], dilation=[1,1], ceil_mode=false)
    -> out0: [1,4,4,4] min=-0.063115 max=0.985821 mean=0.707871
    -> out1: [1,4,4,4] <non-f32>
    status: skipped (no native impl)
  step 6 [c]: {kernel=3x2 stride=3x2 pad=0x0 n=1 c=16 H=12 W=8}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,16,12,8]~values, kernel_size=[3,2], stride=[3,2], padding=[0,0], dilation=[1,1], ceil_mode=false)
    -> out0: [1,16,4,4] min=-0.242112 max=0.996465 mean=0.727567
    -> out1: [1,16,4,4] <non-f32>
    status: skipped (no native impl)
  step 7 [pad_h]: {kernel=3x2 stride=3x2 pad=0x0 n=1 c=16 H=12 W=8}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,16,12,8]~values, kernel_size=[3,2], stride=[3,2], padding=[0,0], dilation=[1,1], ceil_mode=false)
    -> out0: [1,16,4,4] min=-0.575668 max=0.996187 mean=0.682924
    -> out1: [1,16,4,4] <non-f32>
    status: skipped (no native impl)
  step 8 [n]: {kernel=3x2 stride=3x2 pad=0x0 n=1 c=16 H=12 W=8}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,16,12,8]~values, kernel_size=[3,2], stride=[3,2], padding=[0,0], dilation=[1,1], ceil_mode=false)
    -> out0: [1,16,4,4] min=-0.104805 max=0.998836 mean=0.727042
    -> out1: [1,16,4,4] <non-f32>
    status: skipped (no native impl)
  step 9 [input_w]: {kernel=3x2 stride=3x2 pad=0x0 n=1 c=16 H=12 W=12}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,16,12,12]~values, kernel_size=[3,2], stride=[3,2], padding=[0,0], dilation=[1,1], ceil_mode=false)
    -> out0: [1,16,4,6] min=-0.188989 max=0.999435 mean=0.696247
    -> out1: [1,16,4,6] <non-f32>
    status: skipped (no native impl)
  step 10 [stride_w]: {kernel=3x2 stride=3x1 pad=0x0 n=1 c=16 H=12 W=12}
  torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,16,12,12]~values, kernel_size=[3,2], stride=[3,1], padding=[0,0], dilation=[1,1], ceil_mode=false)
    -> out0: [1,16,4,11] min=-0.294304 max=0.999098 mean=0.732048
    -> out1: [1,16,4,11] <non-f32>
    status: skipped (no native impl)
  === torch.ops.aten.mean.dim ===
  step 0: {shape=[2,4,8,8] dims=[2,3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[2,4,8,8]~values, dim=[2,3], keepdim=false)
    -> out0: [2,4] min=-0.123431 max=0.196727 mean=-0.00423134
    status: matched
  step 1 [h]: {shape=[2,4,4,8] dims=[2,3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[2,4,4,8]~values, dim=[2,3], keepdim=false)
    -> out0: [2,4] min=-0.133423 max=0.178904 mean=0.00867901
    status: matched
  step 2 [c]: {shape=[2,16,4,8] dims=[2,3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[2,16,4,8]~values, dim=[2,3], keepdim=false)
    -> out0: [2,16] min=-0.174574 max=0.248416 mean=-0.0207106
    status: matched
  step 3 [n]: {shape=[4,16,4,8] dims=[2,3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[4,16,4,8]~values, dim=[2,3], keepdim=false)
    -> out0: [4,16] min=-0.192426 max=0.245427 mean=0.00406327
    status: matched
  step 4 [h]: {shape=[4,16,4,8] dims=[2,3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[4,16,4,8]~values, dim=[2,3], keepdim=false)
    -> out0: [4,16] min=-0.234256 max=0.261361 mean=-0.00231074
    status: matched
  step 5 [dims]: {shape=[4,16,4,8] dims=[3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[4,16,4,8]~values, dim=[3], keepdim=false)
    -> out0: [4,16,4] min=-0.596596 max=0.452454 mean=-0.0421324
    status: matched
  step 6 [n]: {shape=[1,16,4,8] dims=[3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[1,16,4,8]~values, dim=[3], keepdim=false)
    -> out0: [1,16,4] min=-0.419154 max=0.53054 mean=0.0195922
    status: matched
  step 7 [w]: {shape=[1,16,4,16] dims=[3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[1,16,4,16]~values, dim=[3], keepdim=false)
    -> out0: [1,16,4] min=-0.384851 max=0.367302 mean=0.00214752
    status: matched
  step 8 [w]: {shape=[1,16,4,8] dims=[3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[1,16,4,8]~values, dim=[3], keepdim=false)
    -> out0: [1,16,4] min=-0.486426 max=0.426056 mean=0.0427801
    status: matched
  step 9 [n]: {shape=[4,16,4,8] dims=[3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[4,16,4,8]~values, dim=[3], keepdim=false)
    -> out0: [4,16,4] min=-0.514233 max=0.502941 mean=0.00489192
    status: matched
  step 10 [h]: {shape=[4,16,16,8] dims=[3] keepdim=false}
  torch.ops.aten.mean.dim(self=f32[4,16,16,8]~values, dim=[3], keepdim=false)
    -> out0: [4,16,16] min=-0.627089 max=0.691665 mean=-0.0104756
    status: matched
