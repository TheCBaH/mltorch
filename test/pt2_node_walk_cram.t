Walk one representative fixture per distinct op appearing in resnet18's real
exported graph (test/data/resnet18/ — see .ai/pt2_node_spec_design.md), 10
steps each: every step re-synthesizes the same op invocation from an
independent seed (same target/shapes/hyperparameters — the "configuration" —
every step, only the random tensor fill varies) and evaluates it through the
real ATen kernel only (--eval, no native comparison — see
Aten_spec_run.walk_eval). Each step prints its configuration and a compact
summary of the output tensor (shape + min/max/mean, not the raw values —
some of these tensors, e.g. the 224x224x3 input image, are far too large to
dump element-by-element).

  $ for f in \
  >   data/resnet18/000_convolution_default.json \
  >   data/resnet18/001__native_batch_norm_legit_no_training_default.json \
  >   data/resnet18/002_relu_default.json \
  >   data/resnet18/003_max_pool2d_with_indices_default.json \
  >   data/resnet18/009_add_Tensor.json \
  >   data/resnet18/066_mean_dim.json \
  >   data/resnet18/067_view_default.json \
  >   data/resnet18/068_permute_default.json \
  >   data/resnet18/069_addmm_default.json; do
  >   ../bin/aten_spec_verify.exe --eval --walk 10 "$f"
  > done
  === data/resnet18/000_convolution_default.json ===
  [step 1/10] modified axis: input
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-7.315 max=7.467 mean=-0.00106
    status: ok
  [step 2/10] modified axis: weight
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-7.498 max=9.098 mean=0.00146
    status: ok
  [step 3/10] modified axis: input
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-8.224 max=8.42 mean=-0.001249
    status: ok
  [step 4/10] modified axis: weight
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-7.997 max=8.281 mean=-0.0004593
    status: ok
  [step 5/10] modified axis: input
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-7.662 max=7.25 mean=0.0003026
    status: ok
  [step 6/10] modified axis: weight
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-7.818 max=7.321 mean=0.002931
    status: ok
  [step 7/10] modified axis: input
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-7.245 max=8.733 mean=0.001524
    status: ok
  [step 8/10] modified axis: weight
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-7.62 max=7.685 mean=-0.005635
    status: ok
  [step 9/10] modified axis: input
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-7.708 max=7.585 mean=-0.0007614
    status: ok
  [step 10/10] modified axis: weight
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-7.798 max=7.787 mean=0.001386
    status: ok
  === data/resnet18/001__native_batch_norm_legit_no_training_default.json ===
  [step 1/10] modified axis: input
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-1.607 max=1.572 mean=0.1903
    -> out1: [0] <unsupported dtype>
    -> out2: [0] <unsupported dtype>
    status: ok
  [step 2/10] modified axis: weight
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-1.484 max=1.564 mean=0.1903
    -> out1: [0] <unsupported dtype>
    -> out2: [0] <unsupported dtype>
    status: ok
  [step 3/10] modified axis: bias
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-1.223 max=1.512 mean=0.1782
    -> out1: [0] <unsupported dtype>
    -> out2: [0] <unsupported dtype>
    status: ok
  [step 4/10] modified axis: running_mean
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-1.24 max=1.528 mean=0.1781
    -> out1: [0] <unsupported dtype>
    -> out2: [0] <unsupported dtype>
    status: ok
  [step 5/10] modified axis: running_var
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-2.083 max=3.416 mean=0.1785
    -> out1: [0] <unsupported dtype>
    -> out2: [0] <unsupported dtype>
    status: ok
  [step 6/10] modified axis: input
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-2.191 max=3.297 mean=0.1789
    -> out1: [0] <unsupported dtype>
    -> out2: [0] <unsupported dtype>
    status: ok
  [step 7/10] modified axis: weight
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-2.197 max=2.97 mean=0.1805
    -> out1: [0] <unsupported dtype>
    -> out2: [0] <unsupported dtype>
    status: ok
  [step 8/10] modified axis: bias
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-2.453 max=2.715 mean=0.1719
    -> out1: [0] <unsupported dtype>
    -> out2: [0] <unsupported dtype>
    status: ok
  [step 9/10] modified axis: running_mean
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-2.539 max=2.628 mean=0.1693
    -> out1: [0] <unsupported dtype>
    -> out2: [0] <unsupported dtype>
    status: ok
  [step 10/10] modified axis: running_var
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-3.15 max=3.781 mean=0.1692
    -> out1: [0] <unsupported dtype>
    -> out2: [0] <unsupported dtype>
    status: ok
  === data/resnet18/002_relu_default.json ===
  [step 1/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=4.834 mean=0.3985
    status: ok
  [step 2/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=5.096 mean=0.3977
    status: ok
  [step 3/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=4.861 mean=0.399
    status: ok
  [step 4/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=5.05 mean=0.3986
    status: ok
  [step 5/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=4.693 mean=0.3984
    status: ok
  [step 6/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=4.816 mean=0.4004
    status: ok
  [step 7/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=4.593 mean=0.4
    status: ok
  [step 8/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=4.809 mean=0.3981
    status: ok
  [step 9/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=4.935 mean=0.399
    status: ok
  [step 10/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=4.594 mean=0.3992
    status: ok
  === data/resnet18/003_max_pool2d_with_indices_default.json ===
  [step 1/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.956 max=4.834 mean=1.475
    -> out1: [1,64,56,56] min=0 max=12543 mean=6216
    status: ok
  [step 2/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-1.011 max=5.096 mean=1.474
    -> out1: [1,64,56,56] min=0 max=12543 mean=6216
    status: ok
  [step 3/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.8576 max=4.861 mean=1.48
    -> out1: [1,64,56,56] min=0 max=12543 mean=6216
    status: ok
  [step 4/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.912 max=5.05 mean=1.478
    -> out1: [1,64,56,56] min=0 max=12543 mean=6216
    status: ok
  [step 5/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.7999 max=4.693 mean=1.477
    -> out1: [1,64,56,56] min=0 max=12543 mean=6216
    status: ok
  [step 6/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.867 max=4.816 mean=1.48
    -> out1: [1,64,56,56] min=0 max=12543 mean=6216
    status: ok
  [step 7/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.6902 max=4.593 mean=1.478
    -> out1: [1,64,56,56] min=0 max=12543 mean=6216
    status: ok
  [step 8/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.7334 max=4.809 mean=1.475
    -> out1: [1,64,56,56] min=0 max=12543 mean=6216
    status: ok
  [step 9/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.6322 max=4.935 mean=1.476
    -> out1: [1,64,56,56] min=0 max=12543 mean=6216
    status: ok
  [step 10/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.8236 max=4.594 mean=1.477
    -> out1: [1,64,56,56] min=0 max=12543 mean=6216
    status: ok
  === data/resnet18/009_add_Tensor.json ===
  [step 1/10] modified axis: self
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.137 max=6.763 mean=4.94e-05
    status: ok
  [step 2/10] modified axis: other
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-5.921 max=6.688 mean=-0.001036
    status: ok
  [step 3/10] modified axis: self
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.181 max=6.166 mean=-0.001457
    status: ok
  [step 4/10] modified axis: other
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.275 max=6.426 mean=-0.002411
    status: ok
  [step 5/10] modified axis: self
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.582 max=6.335 mean=-0.003609
    status: ok
  [step 6/10] modified axis: other
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.033 max=6.14 mean=-0.003824
    status: ok
  [step 7/10] modified axis: self
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.6 max=6.291 mean=0.001639
    status: ok
  [step 8/10] modified axis: other
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.244 max=6.363 mean=0.004063
    status: ok
  [step 9/10] modified axis: self
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.967 max=6.642 mean=0.0001112
    status: ok
  [step 10/10] modified axis: other
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.635 max=6.343 mean=0.00282
    status: ok
  === data/resnet18/066_mean_dim.json ===
  [step 1/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.4983 max=0.4158 mean=0.007447
    status: ok
  [step 2/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.4344 max=0.5371 mean=-0.004595
    status: ok
  [step 3/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.4048 max=0.4338 mean=-0.005417
    status: ok
  [step 4/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.4378 max=0.4245 mean=-0.008076
    status: ok
  [step 5/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.4585 max=0.5104 mean=-0.002418
    status: ok
  [step 6/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.3542 max=0.4484 mean=0.001342
    status: ok
  [step 7/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.4061 max=0.5148 mean=0.01475
    status: ok
  [step 8/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.3891 max=0.3957 mean=-0.002827
    status: ok
  [step 9/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.4468 max=0.4159 mean=-0.004182
    status: ok
  [step 10/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.3682 max=0.461 mean=0.009752
    status: ok
  === data/resnet18/067_view_default.json ===
  [step 1/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-3.497 max=2.642 mean=-0.06785
    status: ok
  [step 2/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-3.063 max=2.499 mean=-0.06922
    status: ok
  [step 3/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-2.213 max=2.79 mean=0.02585
    status: ok
  [step 4/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-3.09 max=3.081 mean=0.05811
    status: ok
  [step 5/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-3.265 max=2.965 mean=-0.02475
    status: ok
  [step 6/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-4.507 max=2.873 mean=0.008338
    status: ok
  [step 7/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-2.615 max=2.729 mean=-0.02816
    status: ok
  [step 8/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-2.968 max=2.604 mean=-0.05164
    status: ok
  [step 9/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-2.786 max=2.558 mean=-0.02224
    status: ok
  [step 10/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-3.824 max=3.517 mean=-0.02853
    status: ok
  === data/resnet18/068_permute_default.json ===
  [step 1/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.305 max=0.3357 mean=8.412e-05
    status: ok
  [step 2/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.3322 max=0.354 mean=-0.000111
    status: ok
  [step 3/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.3407 max=0.3376 mean=3.883e-05
    status: ok
  [step 4/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.3216 max=0.3508 mean=-4.204e-05
    status: ok
  [step 5/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.322 max=0.326 mean=-8.521e-05
    status: ok
  [step 6/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.3199 max=0.3345 mean=0.0001055
    status: ok
  [step 7/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.3439 max=0.319 mean=0.0002267
    status: ok
  [step 8/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.3023 max=0.3341 mean=-9.737e-06
    status: ok
  [step 9/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.3258 max=0.3428 mean=0.000125
    status: ok
  [step 10/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.3265 max=0.3191 mean=5.631e-05
    status: ok
  === data/resnet18/069_addmm_default.json ===
  [step 1/10] modified axis: self
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-66.96 max=68.38 mean=0.5997
    status: ok
  [step 2/10] modified axis: mat1
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-71.62 max=66.34 mean=-0.673
    status: ok
  [step 3/10] modified axis: mat2
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-84.28 max=75.27 mean=-0.372
    status: ok
  [step 4/10] modified axis: self
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-84.31 max=75.26 mean=-0.3718
    status: ok
  [step 5/10] modified axis: mat1
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-81.78 max=75.52 mean=0.3474
    status: ok
  [step 6/10] modified axis: mat2
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-79.62 max=64.56 mean=-1.009
    status: ok
  [step 7/10] modified axis: self
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-79.6 max=64.57 mean=-1.009
    status: ok
  [step 8/10] modified axis: mat1
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-68.57 max=74.54 mean=2.078
    status: ok
  [step 9/10] modified axis: mat2
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-73 max=66.79 mean=-0.4854
    status: ok
  [step 10/10] modified axis: self
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-73 max=66.78 mean=-0.4854
    status: ok
