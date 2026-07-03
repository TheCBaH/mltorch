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
    -> out0: [1,64,112,112] min=-7.31472 max=7.46716 mean=-0.00105998
    status: ok
  [step 2/10] modified axis: weight
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-7.49806 max=9.09768 mean=0.00146035
    status: ok
  [step 3/10] modified axis: input
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-8.22359 max=8.42037 mean=-0.0012493
    status: ok
  [step 4/10] modified axis: weight
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-7.99694 max=8.28138 mean=-0.000459295
    status: ok
  [step 5/10] modified axis: input
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-7.66219 max=7.24977 mean=0.000302599
    status: ok
  [step 6/10] modified axis: weight
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-7.81759 max=7.32069 mean=0.00293121
    status: ok
  [step 7/10] modified axis: input
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-7.24543 max=8.73273 mean=0.00152425
    status: ok
  [step 8/10] modified axis: weight
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-7.61955 max=7.6851 mean=-0.00563462
    status: ok
  [step 9/10] modified axis: input
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-7.70756 max=7.58472 mean=-0.000761373
    status: ok
  [step 10/10] modified axis: weight
    torch.ops.aten.convolution.default(input=f32[1,3,224,224]~normal(mean=0,variance=1), weight=f32[64,3,7,7]~normal(mean=2.94198e-05,variance=0.0168198), bias=none, stride=[2,2], padding=[3,3], dilation=[1,1], transposed=false, output_padding=[0,0], groups=1)
    -> out0: [1,64,112,112] min=-7.7977 max=7.78653 mean=0.00138555
    status: ok
  === data/resnet18/001__native_batch_norm_legit_no_training_default.json ===
  [step 1/10] modified axis: input
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-1.60666 max=1.57169 mean=0.190348
    -> out1: [0] <non-f32>
    -> out2: [0] <non-f32>
    status: ok
  [step 2/10] modified axis: weight
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-1.48381 max=1.56395 mean=0.190277
    -> out1: [0] <non-f32>
    -> out2: [0] <non-f32>
    status: ok
  [step 3/10] modified axis: bias
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-1.22336 max=1.512 mean=0.178212
    -> out1: [0] <non-f32>
    -> out2: [0] <non-f32>
    status: ok
  [step 4/10] modified axis: running_mean
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-1.24001 max=1.52798 mean=0.178069
    -> out1: [0] <non-f32>
    -> out2: [0] <non-f32>
    status: ok
  [step 5/10] modified axis: running_var
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-2.08278 max=3.41598 mean=0.178521
    -> out1: [0] <non-f32>
    -> out2: [0] <non-f32>
    status: ok
  [step 6/10] modified axis: input
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-2.19112 max=3.29717 mean=0.178871
    -> out1: [0] <non-f32>
    -> out2: [0] <non-f32>
    status: ok
  [step 7/10] modified axis: weight
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-2.19724 max=2.97032 mean=0.180514
    -> out1: [0] <non-f32>
    -> out2: [0] <non-f32>
    status: ok
  [step 8/10] modified axis: bias
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-2.4528 max=2.71476 mean=0.17187
    -> out1: [0] <non-f32>
    -> out2: [0] <non-f32>
    status: ok
  [step 9/10] modified axis: running_mean
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-2.53926 max=2.6283 mean=0.169321
    -> out1: [0] <non-f32>
    -> out2: [0] <non-f32>
    status: ok
  [step 10/10] modified axis: running_var
    torch.ops.aten._native_batch_norm_legit_no_training.default(input=f32[1,64,112,112]~normal(mean=0,variance=1), weight=f32[64]~normal(mean=0.257577,variance=0.0149257), bias=f32[64]~normal(mean=0.18112,variance=0.0879732), running_mean=f32[64]~normal(mean=0.000847435,variance=0.00235273), running_var=f32[64]~uniform(low=1.0352e-13,high=13.7179), momentum=0.10000000149, eps=9.99999974738e-06)
    -> out0: [1,64,112,112] min=-3.15018 max=3.78133 mean=0.169153
    -> out1: [0] <non-f32>
    -> out2: [0] <non-f32>
    status: ok
  === data/resnet18/002_relu_default.json ===
  [step 1/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=4.83361 mean=0.398548
    status: ok
  [step 2/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=5.09626 mean=0.397702
    status: ok
  [step 3/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=4.86058 mean=0.399029
    status: ok
  [step 4/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=5.0504 mean=0.398571
    status: ok
  [step 5/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=4.69283 mean=0.39836
    status: ok
  [step 6/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=4.81576 mean=0.400353
    status: ok
  [step 7/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=4.59281 mean=0.399986
    status: ok
  [step 8/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=4.80944 mean=0.398105
    status: ok
  [step 9/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=4.93489 mean=0.399018
    status: ok
  [step 10/10] modified axis: self
    torch.ops.aten.relu.default(self=f32[1,64,112,112]~normal(mean=0,variance=1))
    -> out0: [1,64,112,112] min=0 max=4.59415 mean=0.399176
    status: ok
  === data/resnet18/003_max_pool2d_with_indices_default.json ===
  [step 1/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.956038 max=4.83361 mean=1.47478
    -> out1: [1,64,56,56] <non-f32>
    status: ok
  [step 2/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-1.01082 max=5.09626 mean=1.47433
    -> out1: [1,64,56,56] <non-f32>
    status: ok
  [step 3/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.85756 max=4.86058 mean=1.47952
    -> out1: [1,64,56,56] <non-f32>
    status: ok
  [step 4/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.911976 max=5.0504 mean=1.47839
    -> out1: [1,64,56,56] <non-f32>
    status: ok
  [step 5/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.799903 max=4.69283 mean=1.47697
    -> out1: [1,64,56,56] <non-f32>
    status: ok
  [step 6/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.866971 max=4.81576 mean=1.48038
    -> out1: [1,64,56,56] <non-f32>
    status: ok
  [step 7/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.690198 max=4.59281 mean=1.47819
    -> out1: [1,64,56,56] <non-f32>
    status: ok
  [step 8/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.733448 max=4.80944 mean=1.47519
    -> out1: [1,64,56,56] <non-f32>
    status: ok
  [step 9/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.632212 max=4.93489 mean=1.47633
    -> out1: [1,64,56,56] <non-f32>
    status: ok
  [step 10/10] modified axis: self
    torch.ops.aten.max_pool2d_with_indices.default(self=f32[1,64,112,112]~normal(mean=0,variance=1), kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1], ceil_mode=false)
    -> out0: [1,64,56,56] min=-0.823552 max=4.59415 mean=1.47679
    -> out1: [1,64,56,56] <non-f32>
    status: ok
  === data/resnet18/009_add_Tensor.json ===
  [step 1/10] modified axis: self
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.13687 max=6.76256 mean=4.93954e-05
    status: ok
  [step 2/10] modified axis: other
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-5.9212 max=6.68837 mean=-0.00103551
    status: ok
  [step 3/10] modified axis: self
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.18059 max=6.16629 mean=-0.00145653
    status: ok
  [step 4/10] modified axis: other
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.27517 max=6.42632 mean=-0.00241106
    status: ok
  [step 5/10] modified axis: self
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.58218 max=6.33458 mean=-0.00360936
    status: ok
  [step 6/10] modified axis: other
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.03311 max=6.1404 mean=-0.003824
    status: ok
  [step 7/10] modified axis: self
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.59983 max=6.29059 mean=0.00163941
    status: ok
  [step 8/10] modified axis: other
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.24431 max=6.36312 mean=0.00406253
    status: ok
  [step 9/10] modified axis: self
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.96742 max=6.64169 mean=0.000111209
    status: ok
  [step 10/10] modified axis: other
    torch.ops.aten.add.Tensor(self=f32[1,64,56,56]~normal(mean=0,variance=1), other=f32[1,64,56,56]~normal(mean=0,variance=1), alpha=1)
    -> out0: [1,64,56,56] min=-6.6352 max=6.34329 mean=0.00282023
    status: ok
  === data/resnet18/066_mean_dim.json ===
  [step 1/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.498337 max=0.415793 mean=0.00744704
    status: ok
  [step 2/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.434375 max=0.537106 mean=-0.00459466
    status: ok
  [step 3/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.404836 max=0.433808 mean=-0.00541683
    status: ok
  [step 4/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.437813 max=0.424532 mean=-0.00807612
    status: ok
  [step 5/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.458543 max=0.510414 mean=-0.00241805
    status: ok
  [step 6/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.354235 max=0.448363 mean=0.00134194
    status: ok
  [step 7/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.406065 max=0.514848 mean=0.0147488
    status: ok
  [step 8/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.389128 max=0.395681 mean=-0.00282746
    status: ok
  [step 9/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.446782 max=0.415941 mean=-0.00418214
    status: ok
  [step 10/10] modified axis: self
    torch.ops.aten.mean.dim(self=f32[1,512,7,7]~normal(mean=0,variance=1), dim=[-1,-2], keepdim=true)
    -> out0: [1,512,1,1] min=-0.368217 max=0.461047 mean=0.00975154
    status: ok
  === data/resnet18/067_view_default.json ===
  [step 1/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-3.49665 max=2.64208 mean=-0.0678476
    status: ok
  [step 2/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-3.06262 max=2.49866 mean=-0.0692244
    status: ok
  [step 3/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-2.21297 max=2.7898 mean=0.0258471
    status: ok
  [step 4/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-3.09024 max=3.08106 mean=0.0581061
    status: ok
  [step 5/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-3.26466 max=2.96531 mean=-0.0247528
    status: ok
  [step 6/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-4.50685 max=2.87254 mean=0.00833787
    status: ok
  [step 7/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-2.61547 max=2.72913 mean=-0.0281566
    status: ok
  [step 8/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-2.96841 max=2.6036 mean=-0.0516362
    status: ok
  [step 9/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-2.78557 max=2.55781 mean=-0.0222442
    status: ok
  [step 10/10] modified axis: self
    torch.ops.aten.view.default(self=f32[1,512,1,1]~normal(mean=0,variance=1), size=[1,512])
    -> out0: [1,512] min=-3.8241 max=3.51684 mean=-0.028527
    status: ok
  === data/resnet18/068_permute_default.json ===
  [step 1/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.304981 max=0.335738 mean=8.41158e-05
    status: ok
  [step 2/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.332233 max=0.353981 mean=-0.000110954
    status: ok
  [step 3/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.340738 max=0.337611 mean=3.88256e-05
    status: ok
  [step 4/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.321626 max=0.350795 mean=-4.2036e-05
    status: ok
  [step 5/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.321951 max=0.325959 mean=-8.52068e-05
    status: ok
  [step 6/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.319855 max=0.334498 mean=0.000105478
    status: ok
  [step 7/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.343946 max=0.319012 mean=0.000226686
    status: ok
  [step 8/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.302265 max=0.334059 mean=-9.73667e-06
    status: ok
  [step 9/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.325772 max=0.342772 mean=0.000124959
    status: ok
  [step 10/10] modified axis: self
    torch.ops.aten.permute.default(self=f32[1000,512]~normal(mean=5.85066e-08,variance=0.00482454), dims=[1,0])
    -> out0: [512,1000] min=-0.326515 max=0.319105 mean=5.63062e-05
    status: ok
  === data/resnet18/069_addmm_default.json ===
  [step 1/10] modified axis: self
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-66.9596 max=68.3841 mean=0.599699
    status: ok
  [step 2/10] modified axis: mat1
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-71.6173 max=66.3433 mean=-0.673
    status: ok
  [step 3/10] modified axis: mat2
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-84.2841 max=75.2728 mean=-0.371964
    status: ok
  [step 4/10] modified axis: self
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-84.3098 max=75.2576 mean=-0.371772
    status: ok
  [step 5/10] modified axis: mat1
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-81.7759 max=75.516 mean=0.347395
    status: ok
  [step 6/10] modified axis: mat2
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-79.6218 max=64.5575 mean=-1.00878
    status: ok
  [step 7/10] modified axis: self
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-79.5954 max=64.5669 mean=-1.00864
    status: ok
  [step 8/10] modified axis: mat1
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-68.5695 max=74.5369 mean=2.07834
    status: ok
  [step 9/10] modified axis: mat2
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-73.0044 max=66.7928 mean=-0.485443
    status: ok
  [step 10/10] modified axis: self
    torch.ops.aten.addmm.default(self=f32[1000]~normal(mean=-5.97377e-08,variance=0.000253469), mat1=f32[1,512]~normal(mean=0,variance=1), mat2=f32[512,1000]~normal(mean=0,variance=1), beta=1, alpha=1)
    -> out0: [1,1000] min=-73.0018 max=66.7824 mean=-0.485418
    status: ok
