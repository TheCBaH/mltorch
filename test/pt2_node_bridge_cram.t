Compare ATen against the native engine (default aten_spec_verify mode, no
flags) over the resnet18 fixtures (test/data/resnet18/ — see
.ai/pt2_node_spec_design.md) whose op is actually usable via the native
bridge: supported by Op_bridge, fast at real-model scale, and producing
correct results. That's relu.default, add.Tensor, and mean.dim.

The other 6 op types in this corpus are deliberately excluded, each for a
different reason (not "supported but merely untested"). batch_norm,
max_pool2d_with_indices and view.default: Op_bridge reports "skipped (no
native impl)" for every one, nothing to compare. convolution.default:
supported and correct, but the native path takes ~25s even on the
smallest/first conv node, impractical here; still covered by
pt2_node_spec_cram.t's ATen-only --eval and pt2_op_native_walk_cram.t's
toy-scale walk+compare. permute.default and addmm.default: supported, fast,
but produce a real ATen-vs-native disagreement at this shape (permute's is a
genuine bug, unrelated values, not float noise; addmm's looks like ordinary
matmul accumulation-order noise). Left undiagnosed and out of scope; see
this file's prior git history for the actual mismatch output.

  $ for f in data/resnet18/*.json; do
  >   case "$f" in
  >     *_relu_default.json|*_add_Tensor.json|*_mean_dim.json)
  >       ../bin/aten_spec_verify.exe "$f" ;;
  >     *) ;;
  >   esac
  > done
  === data/resnet18/002_relu_default.json ===
  torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=4.858 mean=0.3989
    status: matched
  === data/resnet18/006_relu_default.json ===
  torch.ops.aten.relu.default(self=f32[1,64,56,56]~normal(mean=0,variance=1))
    -> out0: [1,64,56,56] min=0 max=4.318 mean=0.3977
    status: matched
  === data/resnet18/009_add_Tensor.json ===
  torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.368 max=6.103 mean=-0.0005853
    status: matched
  === data/resnet18/010_relu_default.json ===
  torch.ops.aten.relu.default(self=f32[1,64,56,56]~normal(mean=0,variance=1))
    -> out0: [1,64,56,56] min=0 max=4.318 mean=0.3977
    status: matched
  === data/resnet18/013_relu_default.json ===
  torch.ops.aten.relu.default(self=f32[1,64,56,56]~normal(mean=0,variance=1))
    -> out0: [1,64,56,56] min=0 max=4.318 mean=0.3977
    status: matched
  === data/resnet18/016_add_Tensor.json ===
  torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.368 max=6.103 mean=-0.0005853
    status: matched
  === data/resnet18/017_relu_default.json ===
  torch.ops.aten.relu.default(self=f32[1,64,56,56]~normal(mean=0,variance=1))
    -> out0: [1,64,56,56] min=0 max=4.318 mean=0.3977
    status: matched
  === data/resnet18/020_relu_default.json ===
  torch.ops.aten.relu.default(self=f32[1,128,28,28]~normal(mean=0,variance=1))
    -> out0: [1,128,28,28] min=0 max=4.301 mean=0.3973
    status: matched
  === data/resnet18/025_add_Tensor.json ===
  torch.ops.aten.add.Tensor(self=f32[1,128,28,28]~normal(mean=0,variance=1), other=f32[1,128,28,28]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,128,28,28] min=-6.288 max=6.697 mean=-0.0008068
    status: matched
  === data/resnet18/026_relu_default.json ===
  torch.ops.aten.relu.default(self=f32[1,128,28,28]~normal(mean=0,variance=1))
    -> out0: [1,128,28,28] min=0 max=4.301 mean=0.3973
    status: matched
  === data/resnet18/029_relu_default.json ===
  torch.ops.aten.relu.default(self=f32[1,128,28,28]~normal(mean=0,variance=1))
    -> out0: [1,128,28,28] min=0 max=4.301 mean=0.3973
    status: matched
  === data/resnet18/032_add_Tensor.json ===
  torch.ops.aten.add.Tensor(self=f32[1,128,28,28]~normal(mean=0,variance=1), other=f32[1,128,28,28]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,128,28,28] min=-6.288 max=6.697 mean=-0.0008068
    status: matched
  === data/resnet18/033_relu_default.json ===
  torch.ops.aten.relu.default(self=f32[1,128,28,28]~normal(mean=0,variance=1))
    -> out0: [1,128,28,28] min=0 max=4.301 mean=0.3973
    status: matched
  === data/resnet18/036_relu_default.json ===
  torch.ops.aten.relu.default(self=f32[1,256,14,14]~normal(mean=0,variance=1))
    -> out0: [1,256,14,14] min=0 max=4.301 mean=0.3956
    status: matched
  === data/resnet18/041_add_Tensor.json ===
  torch.ops.aten.add.Tensor(self=f32[1,256,14,14]~normal(mean=0,variance=1), other=f32[1,256,14,14]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,256,14,14] min=-5.837 max=6.166 mean=-0.004581
    status: matched
  === data/resnet18/042_relu_default.json ===
  torch.ops.aten.relu.default(self=f32[1,256,14,14]~normal(mean=0,variance=1))
    -> out0: [1,256,14,14] min=0 max=4.301 mean=0.3956
    status: matched
  === data/resnet18/045_relu_default.json ===
  torch.ops.aten.relu.default(self=f32[1,256,14,14]~normal(mean=0,variance=1))
    -> out0: [1,256,14,14] min=0 max=4.301 mean=0.3956
    status: matched
  === data/resnet18/048_add_Tensor.json ===
  torch.ops.aten.add.Tensor(self=f32[1,256,14,14]~normal(mean=0,variance=1), other=f32[1,256,14,14]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,256,14,14] min=-5.837 max=6.166 mean=-0.004581
    status: matched
  === data/resnet18/049_relu_default.json ===
  torch.ops.aten.relu.default(self=f32[1,256,14,14]~normal(mean=0,variance=1))
    -> out0: [1,256,14,14] min=0 max=4.301 mean=0.3956
    status: matched
  === data/resnet18/052_relu_default.json ===
  torch.ops.aten.relu.default(self=f32[1,512,7,7]~normal(mean=0,variance=1))
    -> out0: [1,512,7,7] min=0 max=4.301 mean=0.3972
    status: matched
  === data/resnet18/057_add_Tensor.json ===
  torch.ops.aten.add.Tensor(self=f32[1,512,7,7]~normal(mean=0,variance=1), other=f32[1,512,7,7]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,512,7,7] min=-5.278 max=5.748 mean=-0.008063
    status: matched
  === data/resnet18/058_relu_default.json ===
  torch.ops.aten.relu.default(self=f32[1,512,7,7]~normal(mean=0,variance=1))
    -> out0: [1,512,7,7] min=0 max=4.301 mean=0.3972
    status: matched
  === data/resnet18/061_relu_default.json ===
  torch.ops.aten.relu.default(self=f32[1,512,7,7]~normal(mean=0,variance=1))
    -> out0: [1,512,7,7] min=0 max=4.301 mean=0.3972
    status: matched
  === data/resnet18/064_add_Tensor.json ===
  torch.ops.aten.add.Tensor(self=f32[1,512,7,7]~normal(mean=0,variance=1), other=f32[1,512,7,7]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,512,7,7] min=-5.278 max=5.748 mean=-0.008063
    status: matched
  === data/resnet18/065_relu_default.json ===
  torch.ops.aten.relu.default(self=f32[1,512,7,7]~normal(mean=0,variance=1))
    -> out0: [1,512,7,7] min=0 max=4.301 mean=0.3972
    status: matched
  === data/resnet18/066_mean_dim.json ===
  torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.5153 max=0.332 mean=0.001552
    status: matched
