Convert resnet18's real exported graph to the native engine's Graph_ir node by
node (`native_graph print`, see .ai/native_aten_bridge_layout.md). Gated on
PT2_DATA so the default suite needs no download; run via `make pt2.runtest`
after `make pt2.download`.

`--verbose` prints each source ATen node (target and name) before its
translated native graph, pinning the exact per-op relayout output alongside
which ATen node produced it. Every resnet18 node now converts to the native
Graph_ir end to end — the whole graph, ending at the fc `addmm`, with no
unsupported op or conversion failure.

  $ ../bin/native_graph.exe print --verbose --pt2 "$PT2_DATA/resnet18/resnet18.pt2"
  === torch.ops.aten.convolution.default (convolution) ===
  graph
  inputs: [t0 f32 [H=3 W=224 C=224], t1 f32 [D=64 H=3 W=7 C=7]]
  nodes:
    n0: [t2 f32 [H=224 W=224 C=3]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=64 T=1 D=1 H=7 W=7 C=3]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=112 W=112 C=64]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=2; w=2};
               padding={h=3; w=3};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=64 W=112 C=112]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=64 W=112 C=112]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training) ===
  graph
  inputs:
    [t0 f32 [H=64 W=112 C=112], t1 f32 [C=64], t2 f32 [C=64], t3 f32 [C=64],
     t4 f32 [C=64]]
  nodes:
    n0: [t5 f32 [H=112 W=112 C=64]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=112 W=112 C=64]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=64 W=112 C=112]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=64 W=112 C=112]]
  === torch.ops.aten.relu.default (relu) ===
  graph
  inputs: [t0 f32 [H=64 W=112 C=112]]
  nodes:
    n0: [t1 f32 [H=64 W=112 C=112]] = relu x=t0
  outputs: [t1 f32 [H=64 W=112 C=112]]
  === torch.ops.aten.max_pool2d_with_indices.default (max_pool2d_with_indices) ===
  graph
  inputs: [t0 f32 [H=64 W=112 C=112]]
  nodes:
    n0: [t1 f32 [H=112 W=112 C=64]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t2 f32 [H=56 W=56 C=64], t3 f32 [H=56 W=56 C=64]] =
      max_pool2d_with_indices
        x=t1
        params={kernel={h=3; w=3}; stride={h=2; w=2}; pad={h=1; w=1}}
    n2: [] = discard x=t3
    n3: [t4 f32 [H=64 W=56 C=56]] = permute x=t2 perm=[H<-C, W<-H, C<-W]
  outputs: [t4 f32 [H=64 W=56 C=56]]
  === torch.ops.aten.convolution.default (convolution_1) ===
  graph
  inputs: [t0 f32 [H=64 W=56 C=56], t1 f32 [D=64 H=64 W=3 C=3]]
  nodes:
    n0: [t2 f32 [H=56 W=56 C=64]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=64 T=1 D=1 H=3 W=3 C=64]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=56 W=56 C=64]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=64 W=56 C=56]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=64 W=56 C=56]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_1) ===
  graph
  inputs:
    [t0 f32 [H=64 W=56 C=56], t1 f32 [C=64], t2 f32 [C=64], t3 f32 [C=64],
     t4 f32 [C=64]]
  nodes:
    n0: [t5 f32 [H=56 W=56 C=64]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=56 W=56 C=64]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=64 W=56 C=56]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=64 W=56 C=56]]
  === torch.ops.aten.relu.default (relu_1) ===
  graph
  inputs: [t0 f32 [H=64 W=56 C=56]]
  nodes:
    n0: [t1 f32 [H=64 W=56 C=56]] = relu x=t0
  outputs: [t1 f32 [H=64 W=56 C=56]]
  === torch.ops.aten.convolution.default (convolution_2) ===
  graph
  inputs: [t0 f32 [H=64 W=56 C=56], t1 f32 [D=64 H=64 W=3 C=3]]
  nodes:
    n0: [t2 f32 [H=56 W=56 C=64]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=64 T=1 D=1 H=3 W=3 C=64]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=56 W=56 C=64]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=64 W=56 C=56]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=64 W=56 C=56]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_2) ===
  graph
  inputs:
    [t0 f32 [H=64 W=56 C=56], t1 f32 [C=64], t2 f32 [C=64], t3 f32 [C=64],
     t4 f32 [C=64]]
  nodes:
    n0: [t5 f32 [H=56 W=56 C=64]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=56 W=56 C=64]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=64 W=56 C=56]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=64 W=56 C=56]]
  === torch.ops.aten.add.Tensor (add) ===
  graph
  inputs: [t0 f32 [H=64 W=56 C=56], t1 f32 [H=64 W=56 C=56]]
  nodes:
    n0: [t2 f32 [H=64 W=56 C=56]] = add a=t0 b=t1
  outputs: [t2 f32 [H=64 W=56 C=56]]
  === torch.ops.aten.relu.default (relu_2) ===
  graph
  inputs: [t0 f32 [H=64 W=56 C=56]]
  nodes:
    n0: [t1 f32 [H=64 W=56 C=56]] = relu x=t0
  outputs: [t1 f32 [H=64 W=56 C=56]]
  === torch.ops.aten.convolution.default (convolution_3) ===
  graph
  inputs: [t0 f32 [H=64 W=56 C=56], t1 f32 [D=64 H=64 W=3 C=3]]
  nodes:
    n0: [t2 f32 [H=56 W=56 C=64]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=64 T=1 D=1 H=3 W=3 C=64]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=56 W=56 C=64]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=64 W=56 C=56]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=64 W=56 C=56]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_3) ===
  graph
  inputs:
    [t0 f32 [H=64 W=56 C=56], t1 f32 [C=64], t2 f32 [C=64], t3 f32 [C=64],
     t4 f32 [C=64]]
  nodes:
    n0: [t5 f32 [H=56 W=56 C=64]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=56 W=56 C=64]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=64 W=56 C=56]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=64 W=56 C=56]]
  === torch.ops.aten.relu.default (relu_3) ===
  graph
  inputs: [t0 f32 [H=64 W=56 C=56]]
  nodes:
    n0: [t1 f32 [H=64 W=56 C=56]] = relu x=t0
  outputs: [t1 f32 [H=64 W=56 C=56]]
  === torch.ops.aten.convolution.default (convolution_4) ===
  graph
  inputs: [t0 f32 [H=64 W=56 C=56], t1 f32 [D=64 H=64 W=3 C=3]]
  nodes:
    n0: [t2 f32 [H=56 W=56 C=64]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=64 T=1 D=1 H=3 W=3 C=64]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=56 W=56 C=64]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=64 W=56 C=56]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=64 W=56 C=56]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_4) ===
  graph
  inputs:
    [t0 f32 [H=64 W=56 C=56], t1 f32 [C=64], t2 f32 [C=64], t3 f32 [C=64],
     t4 f32 [C=64]]
  nodes:
    n0: [t5 f32 [H=56 W=56 C=64]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=56 W=56 C=64]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=64 W=56 C=56]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=64 W=56 C=56]]
  === torch.ops.aten.add.Tensor (add_1) ===
  graph
  inputs: [t0 f32 [H=64 W=56 C=56], t1 f32 [H=64 W=56 C=56]]
  nodes:
    n0: [t2 f32 [H=64 W=56 C=56]] = add a=t0 b=t1
  outputs: [t2 f32 [H=64 W=56 C=56]]
  === torch.ops.aten.relu.default (relu_4) ===
  graph
  inputs: [t0 f32 [H=64 W=56 C=56]]
  nodes:
    n0: [t1 f32 [H=64 W=56 C=56]] = relu x=t0
  outputs: [t1 f32 [H=64 W=56 C=56]]
  === torch.ops.aten.convolution.default (convolution_5) ===
  graph
  inputs: [t0 f32 [H=64 W=56 C=56], t1 f32 [D=128 H=64 W=3 C=3]]
  nodes:
    n0: [t2 f32 [H=56 W=56 C=64]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=128 T=1 D=1 H=3 W=3 C=64]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=28 W=28 C=128]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=128 W=28 C=28]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=128 W=28 C=28]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_5) ===
  graph
  inputs:
    [t0 f32 [H=128 W=28 C=28], t1 f32 [C=128], t2 f32 [C=128], t3 f32 [C=128],
     t4 f32 [C=128]]
  nodes:
    n0: [t5 f32 [H=28 W=28 C=128]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=28 W=28 C=128]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=128 W=28 C=28]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=128 W=28 C=28]]
  === torch.ops.aten.relu.default (relu_5) ===
  graph
  inputs: [t0 f32 [H=128 W=28 C=28]]
  nodes:
    n0: [t1 f32 [H=128 W=28 C=28]] = relu x=t0
  outputs: [t1 f32 [H=128 W=28 C=28]]
  === torch.ops.aten.convolution.default (convolution_6) ===
  graph
  inputs: [t0 f32 [H=128 W=28 C=28], t1 f32 [D=128 H=128 W=3 C=3]]
  nodes:
    n0: [t2 f32 [H=28 W=28 C=128]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=128 T=1 D=1 H=3 W=3 C=128]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=28 W=28 C=128]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=128 W=28 C=28]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=128 W=28 C=28]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_6) ===
  graph
  inputs:
    [t0 f32 [H=128 W=28 C=28], t1 f32 [C=128], t2 f32 [C=128], t3 f32 [C=128],
     t4 f32 [C=128]]
  nodes:
    n0: [t5 f32 [H=28 W=28 C=128]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=28 W=28 C=128]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=128 W=28 C=28]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=128 W=28 C=28]]
  === torch.ops.aten.convolution.default (convolution_7) ===
  graph
  inputs: [t0 f32 [H=64 W=56 C=56], t1 f32 [D=128 H=64 W=1 C=1]]
  nodes:
    n0: [t2 f32 [H=56 W=56 C=64]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=128 T=1 D=1 H=1 W=1 C=64]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=28 W=28 C=128]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=2; w=2};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=128 W=28 C=28]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=128 W=28 C=28]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_7) ===
  graph
  inputs:
    [t0 f32 [H=128 W=28 C=28], t1 f32 [C=128], t2 f32 [C=128], t3 f32 [C=128],
     t4 f32 [C=128]]
  nodes:
    n0: [t5 f32 [H=28 W=28 C=128]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=28 W=28 C=128]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=128 W=28 C=28]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=128 W=28 C=28]]
  === torch.ops.aten.add.Tensor (add_2) ===
  graph
  inputs: [t0 f32 [H=128 W=28 C=28], t1 f32 [H=128 W=28 C=28]]
  nodes:
    n0: [t2 f32 [H=128 W=28 C=28]] = add a=t0 b=t1
  outputs: [t2 f32 [H=128 W=28 C=28]]
  === torch.ops.aten.relu.default (relu_6) ===
  graph
  inputs: [t0 f32 [H=128 W=28 C=28]]
  nodes:
    n0: [t1 f32 [H=128 W=28 C=28]] = relu x=t0
  outputs: [t1 f32 [H=128 W=28 C=28]]
  === torch.ops.aten.convolution.default (convolution_8) ===
  graph
  inputs: [t0 f32 [H=128 W=28 C=28], t1 f32 [D=128 H=128 W=3 C=3]]
  nodes:
    n0: [t2 f32 [H=28 W=28 C=128]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=128 T=1 D=1 H=3 W=3 C=128]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=28 W=28 C=128]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=128 W=28 C=28]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=128 W=28 C=28]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_8) ===
  graph
  inputs:
    [t0 f32 [H=128 W=28 C=28], t1 f32 [C=128], t2 f32 [C=128], t3 f32 [C=128],
     t4 f32 [C=128]]
  nodes:
    n0: [t5 f32 [H=28 W=28 C=128]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=28 W=28 C=128]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=128 W=28 C=28]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=128 W=28 C=28]]
  === torch.ops.aten.relu.default (relu_7) ===
  graph
  inputs: [t0 f32 [H=128 W=28 C=28]]
  nodes:
    n0: [t1 f32 [H=128 W=28 C=28]] = relu x=t0
  outputs: [t1 f32 [H=128 W=28 C=28]]
  === torch.ops.aten.convolution.default (convolution_9) ===
  graph
  inputs: [t0 f32 [H=128 W=28 C=28], t1 f32 [D=128 H=128 W=3 C=3]]
  nodes:
    n0: [t2 f32 [H=28 W=28 C=128]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=128 T=1 D=1 H=3 W=3 C=128]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=28 W=28 C=128]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=128 W=28 C=28]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=128 W=28 C=28]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_9) ===
  graph
  inputs:
    [t0 f32 [H=128 W=28 C=28], t1 f32 [C=128], t2 f32 [C=128], t3 f32 [C=128],
     t4 f32 [C=128]]
  nodes:
    n0: [t5 f32 [H=28 W=28 C=128]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=28 W=28 C=128]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=128 W=28 C=28]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=128 W=28 C=28]]
  === torch.ops.aten.add.Tensor (add_3) ===
  graph
  inputs: [t0 f32 [H=128 W=28 C=28], t1 f32 [H=128 W=28 C=28]]
  nodes:
    n0: [t2 f32 [H=128 W=28 C=28]] = add a=t0 b=t1
  outputs: [t2 f32 [H=128 W=28 C=28]]
  === torch.ops.aten.relu.default (relu_8) ===
  graph
  inputs: [t0 f32 [H=128 W=28 C=28]]
  nodes:
    n0: [t1 f32 [H=128 W=28 C=28]] = relu x=t0
  outputs: [t1 f32 [H=128 W=28 C=28]]
  === torch.ops.aten.convolution.default (convolution_10) ===
  graph
  inputs: [t0 f32 [H=128 W=28 C=28], t1 f32 [D=256 H=128 W=3 C=3]]
  nodes:
    n0: [t2 f32 [H=28 W=28 C=128]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=256 T=1 D=1 H=3 W=3 C=128]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=14 W=14 C=256]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=256 W=14 C=14]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=256 W=14 C=14]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_10) ===
  graph
  inputs:
    [t0 f32 [H=256 W=14 C=14], t1 f32 [C=256], t2 f32 [C=256], t3 f32 [C=256],
     t4 f32 [C=256]]
  nodes:
    n0: [t5 f32 [H=14 W=14 C=256]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=14 W=14 C=256]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=256 W=14 C=14]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=256 W=14 C=14]]
  === torch.ops.aten.relu.default (relu_9) ===
  graph
  inputs: [t0 f32 [H=256 W=14 C=14]]
  nodes:
    n0: [t1 f32 [H=256 W=14 C=14]] = relu x=t0
  outputs: [t1 f32 [H=256 W=14 C=14]]
  === torch.ops.aten.convolution.default (convolution_11) ===
  graph
  inputs: [t0 f32 [H=256 W=14 C=14], t1 f32 [D=256 H=256 W=3 C=3]]
  nodes:
    n0: [t2 f32 [H=14 W=14 C=256]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=256 T=1 D=1 H=3 W=3 C=256]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=14 W=14 C=256]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=256 W=14 C=14]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=256 W=14 C=14]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_11) ===
  graph
  inputs:
    [t0 f32 [H=256 W=14 C=14], t1 f32 [C=256], t2 f32 [C=256], t3 f32 [C=256],
     t4 f32 [C=256]]
  nodes:
    n0: [t5 f32 [H=14 W=14 C=256]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=14 W=14 C=256]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=256 W=14 C=14]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=256 W=14 C=14]]
  === torch.ops.aten.convolution.default (convolution_12) ===
  graph
  inputs: [t0 f32 [H=128 W=28 C=28], t1 f32 [D=256 H=128 W=1 C=1]]
  nodes:
    n0: [t2 f32 [H=28 W=28 C=128]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=256 T=1 D=1 H=1 W=1 C=128]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=14 W=14 C=256]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=2; w=2};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=256 W=14 C=14]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=256 W=14 C=14]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_12) ===
  graph
  inputs:
    [t0 f32 [H=256 W=14 C=14], t1 f32 [C=256], t2 f32 [C=256], t3 f32 [C=256],
     t4 f32 [C=256]]
  nodes:
    n0: [t5 f32 [H=14 W=14 C=256]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=14 W=14 C=256]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=256 W=14 C=14]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=256 W=14 C=14]]
  === torch.ops.aten.add.Tensor (add_4) ===
  graph
  inputs: [t0 f32 [H=256 W=14 C=14], t1 f32 [H=256 W=14 C=14]]
  nodes:
    n0: [t2 f32 [H=256 W=14 C=14]] = add a=t0 b=t1
  outputs: [t2 f32 [H=256 W=14 C=14]]
  === torch.ops.aten.relu.default (relu_10) ===
  graph
  inputs: [t0 f32 [H=256 W=14 C=14]]
  nodes:
    n0: [t1 f32 [H=256 W=14 C=14]] = relu x=t0
  outputs: [t1 f32 [H=256 W=14 C=14]]
  === torch.ops.aten.convolution.default (convolution_13) ===
  graph
  inputs: [t0 f32 [H=256 W=14 C=14], t1 f32 [D=256 H=256 W=3 C=3]]
  nodes:
    n0: [t2 f32 [H=14 W=14 C=256]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=256 T=1 D=1 H=3 W=3 C=256]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=14 W=14 C=256]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=256 W=14 C=14]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=256 W=14 C=14]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_13) ===
  graph
  inputs:
    [t0 f32 [H=256 W=14 C=14], t1 f32 [C=256], t2 f32 [C=256], t3 f32 [C=256],
     t4 f32 [C=256]]
  nodes:
    n0: [t5 f32 [H=14 W=14 C=256]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=14 W=14 C=256]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=256 W=14 C=14]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=256 W=14 C=14]]
  === torch.ops.aten.relu.default (relu_11) ===
  graph
  inputs: [t0 f32 [H=256 W=14 C=14]]
  nodes:
    n0: [t1 f32 [H=256 W=14 C=14]] = relu x=t0
  outputs: [t1 f32 [H=256 W=14 C=14]]
  === torch.ops.aten.convolution.default (convolution_14) ===
  graph
  inputs: [t0 f32 [H=256 W=14 C=14], t1 f32 [D=256 H=256 W=3 C=3]]
  nodes:
    n0: [t2 f32 [H=14 W=14 C=256]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=256 T=1 D=1 H=3 W=3 C=256]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=14 W=14 C=256]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=256 W=14 C=14]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=256 W=14 C=14]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_14) ===
  graph
  inputs:
    [t0 f32 [H=256 W=14 C=14], t1 f32 [C=256], t2 f32 [C=256], t3 f32 [C=256],
     t4 f32 [C=256]]
  nodes:
    n0: [t5 f32 [H=14 W=14 C=256]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=14 W=14 C=256]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=256 W=14 C=14]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=256 W=14 C=14]]
  === torch.ops.aten.add.Tensor (add_5) ===
  graph
  inputs: [t0 f32 [H=256 W=14 C=14], t1 f32 [H=256 W=14 C=14]]
  nodes:
    n0: [t2 f32 [H=256 W=14 C=14]] = add a=t0 b=t1
  outputs: [t2 f32 [H=256 W=14 C=14]]
  === torch.ops.aten.relu.default (relu_12) ===
  graph
  inputs: [t0 f32 [H=256 W=14 C=14]]
  nodes:
    n0: [t1 f32 [H=256 W=14 C=14]] = relu x=t0
  outputs: [t1 f32 [H=256 W=14 C=14]]
  === torch.ops.aten.convolution.default (convolution_15) ===
  graph
  inputs: [t0 f32 [H=256 W=14 C=14], t1 f32 [D=512 H=256 W=3 C=3]]
  nodes:
    n0: [t2 f32 [H=14 W=14 C=256]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=512 T=1 D=1 H=3 W=3 C=256]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=7 W=7 C=512]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=512 W=7 C=7]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=512 W=7 C=7]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_15) ===
  graph
  inputs:
    [t0 f32 [H=512 W=7 C=7], t1 f32 [C=512], t2 f32 [C=512], t3 f32 [C=512],
     t4 f32 [C=512]]
  nodes:
    n0: [t5 f32 [H=7 W=7 C=512]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=7 W=7 C=512]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=512 W=7 C=7]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=512 W=7 C=7]]
  === torch.ops.aten.relu.default (relu_13) ===
  graph
  inputs: [t0 f32 [H=512 W=7 C=7]]
  nodes:
    n0: [t1 f32 [H=512 W=7 C=7]] = relu x=t0
  outputs: [t1 f32 [H=512 W=7 C=7]]
  === torch.ops.aten.convolution.default (convolution_16) ===
  graph
  inputs: [t0 f32 [H=512 W=7 C=7], t1 f32 [D=512 H=512 W=3 C=3]]
  nodes:
    n0: [t2 f32 [H=7 W=7 C=512]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=512 T=1 D=1 H=3 W=3 C=512]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=7 W=7 C=512]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=512 W=7 C=7]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=512 W=7 C=7]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_16) ===
  graph
  inputs:
    [t0 f32 [H=512 W=7 C=7], t1 f32 [C=512], t2 f32 [C=512], t3 f32 [C=512],
     t4 f32 [C=512]]
  nodes:
    n0: [t5 f32 [H=7 W=7 C=512]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=7 W=7 C=512]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=512 W=7 C=7]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=512 W=7 C=7]]
  === torch.ops.aten.convolution.default (convolution_17) ===
  graph
  inputs: [t0 f32 [H=256 W=14 C=14], t1 f32 [D=512 H=256 W=1 C=1]]
  nodes:
    n0: [t2 f32 [H=14 W=14 C=256]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=512 T=1 D=1 H=1 W=1 C=256]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=7 W=7 C=512]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=2; w=2};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=512 W=7 C=7]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=512 W=7 C=7]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_17) ===
  graph
  inputs:
    [t0 f32 [H=512 W=7 C=7], t1 f32 [C=512], t2 f32 [C=512], t3 f32 [C=512],
     t4 f32 [C=512]]
  nodes:
    n0: [t5 f32 [H=7 W=7 C=512]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=7 W=7 C=512]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=512 W=7 C=7]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=512 W=7 C=7]]
  === torch.ops.aten.add.Tensor (add_6) ===
  graph
  inputs: [t0 f32 [H=512 W=7 C=7], t1 f32 [H=512 W=7 C=7]]
  nodes:
    n0: [t2 f32 [H=512 W=7 C=7]] = add a=t0 b=t1
  outputs: [t2 f32 [H=512 W=7 C=7]]
  === torch.ops.aten.relu.default (relu_14) ===
  graph
  inputs: [t0 f32 [H=512 W=7 C=7]]
  nodes:
    n0: [t1 f32 [H=512 W=7 C=7]] = relu x=t0
  outputs: [t1 f32 [H=512 W=7 C=7]]
  === torch.ops.aten.convolution.default (convolution_18) ===
  graph
  inputs: [t0 f32 [H=512 W=7 C=7], t1 f32 [D=512 H=512 W=3 C=3]]
  nodes:
    n0: [t2 f32 [H=7 W=7 C=512]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=512 T=1 D=1 H=3 W=3 C=512]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=7 W=7 C=512]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=512 W=7 C=7]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=512 W=7 C=7]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_18) ===
  graph
  inputs:
    [t0 f32 [H=512 W=7 C=7], t1 f32 [C=512], t2 f32 [C=512], t3 f32 [C=512],
     t4 f32 [C=512]]
  nodes:
    n0: [t5 f32 [H=7 W=7 C=512]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=7 W=7 C=512]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=512 W=7 C=7]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=512 W=7 C=7]]
  === torch.ops.aten.relu.default (relu_15) ===
  graph
  inputs: [t0 f32 [H=512 W=7 C=7]]
  nodes:
    n0: [t1 f32 [H=512 W=7 C=7]] = relu x=t0
  outputs: [t1 f32 [H=512 W=7 C=7]]
  === torch.ops.aten.convolution.default (convolution_19) ===
  graph
  inputs: [t0 f32 [H=512 W=7 C=7], t1 f32 [D=512 H=512 W=3 C=3]]
  nodes:
    n0: [t2 f32 [H=7 W=7 C=512]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t3 f32 [N=512 T=1 D=1 H=3 W=3 C=512]] =
      permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
    n2: [t4 f32 [H=7 W=7 C=512]] =
      convolution
        x=t2
        weight=t3
        bias=none
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n3: [t5 f32 [H=512 W=7 C=7]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
  outputs: [t5 f32 [H=512 W=7 C=7]]
  === torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_19) ===
  graph
  inputs:
    [t0 f32 [H=512 W=7 C=7], t1 f32 [C=512], t2 f32 [C=512], t3 f32 [C=512],
     t4 f32 [C=512]]
  nodes:
    n0: [t5 f32 [H=7 W=7 C=512]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    n1: [t6 f32 [H=7 W=7 C=512]] =
      batch_norm
        x=t5
        weight=t1
        bias=t2
        running_mean=t3
        running_var=t4
        params={channel=C; eps=1e-05}
    n2: [t7 f32 [H=512 W=7 C=7]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
  outputs: [t7 f32 [H=512 W=7 C=7]]
  === torch.ops.aten.add.Tensor (add_7) ===
  graph
  inputs: [t0 f32 [H=512 W=7 C=7], t1 f32 [H=512 W=7 C=7]]
  nodes:
    n0: [t2 f32 [H=512 W=7 C=7]] = add a=t0 b=t1
  outputs: [t2 f32 [H=512 W=7 C=7]]
  === torch.ops.aten.relu.default (relu_16) ===
  graph
  inputs: [t0 f32 [H=512 W=7 C=7]]
  nodes:
    n0: [t1 f32 [H=512 W=7 C=7]] = relu x=t0
  outputs: [t1 f32 [H=512 W=7 C=7]]
  === torch.ops.aten.mean.dim (mean) ===
  graph
  inputs: [t0 f32 [H=512 W=7 C=7]]
  nodes:
    n0: [t1 f32 [H=512 W=1 C=1]] = mean x=t0 params={dims=[C, W]; keepdim=true}
  outputs: [t1 f32 [H=512 W=1 C=1]]
  === torch.ops.aten.view.default (view) ===
  graph
  inputs: [t0 f32 [H=512 W=1 C=1]]
  nodes:
    n0: [t1 f32 [C=512]] = reshape x=t0 params={shape=[C=512]}
  outputs: [t1 f32 [C=512]]
  === torch.ops.aten.permute.default (permute) ===
  graph
  inputs: [t0 f32 [W=1000 C=512]]
  nodes:
    n0: [t1 f32 [W=512 C=1000]] = permute x=t0 perm=[W<-C, C<-W]
  outputs: [t1 f32 [W=512 C=1000]]
  === torch.ops.aten.addmm.default (addmm) ===
  graph
  inputs: [t0 f32 [C=1000], t1 f32 [C=512], t2 f32 [W=512 C=1000]]
  nodes:
    n0: [t3 f32 [N=1000 T=1 D=1 H=1 W=1 C=512]] =
      permute x=t2 perm=[N<-C, W<-N, C<-W]
    n1: [t4 f32 [C=1000]] =
      linear x=t1 weight=t3 bias=t0 params={in_features=512}
  outputs: [t4 f32 [C=1000]]
