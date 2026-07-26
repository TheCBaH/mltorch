Rewrite ResNet-18's imported graph with the transformation framework, this
time with `--fold` so every optimization in the pipeline runs: permute
cancellation, constant folding of the weight-permute chain, batch-norm
folding into the preceding convolution, and a second constant fold to
collapse the parameter arithmetic the bn fold emits. See
.ai/native_transform_design.md §12b-§12c. Gated on PT2_DATA; run with
`make pt2.runtest` after `make pt2.download-cram`.

Structure only, as in native_transform_cram.t: executing the result is
`make native-transform-verify`, because a full inference is slow and the
residual it reports is floating point, neither of which belongs in a
golden. `--fold` loads the whole archive so folding has real weight data
to compute over, so the node count and constant provenance differ from
the unfolded structural test.

  $ ../bin/native_graph.exe transform --fold --pt2 "$PT2_DATA/resnet18/resnet18.pt2"
  nodes: 174 -> 92
  constants: 42, of which 41 folded
  graph
  inputs:
    [t61 f32 [C=1000] {pt2=root:p_fc_bias target=fc.bias} constant,
     t122 f32 [H=3 W=224 C=224] {pt2=root:x},
     t295 f32 [N=1000 T=1 D=1 H=1 W=1 C=512] {folded from=[p_fc_weight]} constant,
     t297 f32 [N=64 T=1 D=1 H=7 W=7 C=3] {folded from=[p_conv1_weight,
                                                       p_bn1_weight,
                                                       b_bn1_running_var]} constant,
     t298 f32 [C=64] {folded from=[p_bn1_weight, p_bn1_bias,
                                   b_bn1_running_mean, b_bn1_running_var]} constant,
     t299 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_0_conv1_weight,
                                                        p_layer1_0_bn1_weight,
                                                        b_layer1_0_bn1_running_var]} constant,
     t300 f32 [C=64] {folded from=[p_layer1_0_bn1_weight, p_layer1_0_bn1_bias,
                                   b_layer1_0_bn1_running_mean,
                                   b_layer1_0_bn1_running_var]} constant,
     t301 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_0_conv2_weight,
                                                        p_layer1_0_bn2_weight,
                                                        b_layer1_0_bn2_running_var]} constant,
     t302 f32 [C=64] {folded from=[p_layer1_0_bn2_weight, p_layer1_0_bn2_bias,
                                   b_layer1_0_bn2_running_mean,
                                   b_layer1_0_bn2_running_var]} constant,
     t303 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_1_conv1_weight,
                                                        p_layer1_1_bn1_weight,
                                                        b_layer1_1_bn1_running_var]} constant,
     t304 f32 [C=64] {folded from=[p_layer1_1_bn1_weight, p_layer1_1_bn1_bias,
                                   b_layer1_1_bn1_running_mean,
                                   b_layer1_1_bn1_running_var]} constant,
     t305 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_1_conv2_weight,
                                                        p_layer1_1_bn2_weight,
                                                        b_layer1_1_bn2_running_var]} constant,
     t306 f32 [C=64] {folded from=[p_layer1_1_bn2_weight, p_layer1_1_bn2_bias,
                                   b_layer1_1_bn2_running_mean,
                                   b_layer1_1_bn2_running_var]} constant,
     t307 f32 [N=128 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer2_0_conv1_weight,
                                                         p_layer2_0_bn1_weight,
                                                         b_layer2_0_bn1_running_var]} constant,
     t308 f32 [C=128] {folded from=[p_layer2_0_bn1_weight, p_layer2_0_bn1_bias,
                                    b_layer2_0_bn1_running_mean,
                                    b_layer2_0_bn1_running_var]} constant,
     t309 f32 [N=128 T=1 D=1 H=1 W=1 C=64] {folded from=[p_layer2_0_downsample_0_weight,
                                                         p_layer2_0_downsample_1_weight,
                                                         b_layer2_0_downsample_1_running_var]} constant,
     t310 f32 [C=128] {folded from=[p_layer2_0_downsample_1_weight,
                                    p_layer2_0_downsample_1_bias,
                                    b_layer2_0_downsample_1_running_mean,
                                    b_layer2_0_downsample_1_running_var]} constant,
     t311 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer2_0_conv2_weight,
                                                          p_layer2_0_bn2_weight,
                                                          b_layer2_0_bn2_running_var]} constant,
     t312 f32 [C=128] {folded from=[p_layer2_0_bn2_weight, p_layer2_0_bn2_bias,
                                    b_layer2_0_bn2_running_mean,
                                    b_layer2_0_bn2_running_var]} constant,
     t313 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer2_1_conv1_weight,
                                                          p_layer2_1_bn1_weight,
                                                          b_layer2_1_bn1_running_var]} constant,
     t314 f32 [C=128] {folded from=[p_layer2_1_bn1_weight, p_layer2_1_bn1_bias,
                                    b_layer2_1_bn1_running_mean,
                                    b_layer2_1_bn1_running_var]} constant,
     t315 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer2_1_conv2_weight,
                                                          p_layer2_1_bn2_weight,
                                                          b_layer2_1_bn2_running_var]} constant,
     t316 f32 [C=128] {folded from=[p_layer2_1_bn2_weight, p_layer2_1_bn2_bias,
                                    b_layer2_1_bn2_running_mean,
                                    b_layer2_1_bn2_running_var]} constant,
     t317 f32 [N=256 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer3_0_conv1_weight,
                                                          p_layer3_0_bn1_weight,
                                                          b_layer3_0_bn1_running_var]} constant,
     t318 f32 [C=256] {folded from=[p_layer3_0_bn1_weight, p_layer3_0_bn1_bias,
                                    b_layer3_0_bn1_running_mean,
                                    b_layer3_0_bn1_running_var]} constant,
     t319 f32 [N=256 T=1 D=1 H=1 W=1 C=128] {folded from=[p_layer3_0_downsample_0_weight,
                                                          p_layer3_0_downsample_1_weight,
                                                          b_layer3_0_downsample_1_running_var]} constant,
     t320 f32 [C=256] {folded from=[p_layer3_0_downsample_1_weight,
                                    p_layer3_0_downsample_1_bias,
                                    b_layer3_0_downsample_1_running_mean,
                                    b_layer3_0_downsample_1_running_var]} constant,
     t321 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer3_0_conv2_weight,
                                                          p_layer3_0_bn2_weight,
                                                          b_layer3_0_bn2_running_var]} constant,
     t322 f32 [C=256] {folded from=[p_layer3_0_bn2_weight, p_layer3_0_bn2_bias,
                                    b_layer3_0_bn2_running_mean,
                                    b_layer3_0_bn2_running_var]} constant,
     t323 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer3_1_conv1_weight,
                                                          p_layer3_1_bn1_weight,
                                                          b_layer3_1_bn1_running_var]} constant,
     t324 f32 [C=256] {folded from=[p_layer3_1_bn1_weight, p_layer3_1_bn1_bias,
                                    b_layer3_1_bn1_running_mean,
                                    b_layer3_1_bn1_running_var]} constant,
     t325 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer3_1_conv2_weight,
                                                          p_layer3_1_bn2_weight,
                                                          b_layer3_1_bn2_running_var]} constant,
     t326 f32 [C=256] {folded from=[p_layer3_1_bn2_weight, p_layer3_1_bn2_bias,
                                    b_layer3_1_bn2_running_mean,
                                    b_layer3_1_bn2_running_var]} constant,
     t327 f32 [N=512 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer4_0_conv1_weight,
                                                          p_layer4_0_bn1_weight,
                                                          b_layer4_0_bn1_running_var]} constant,
     t328 f32 [C=512] {folded from=[p_layer4_0_bn1_weight, p_layer4_0_bn1_bias,
                                    b_layer4_0_bn1_running_mean,
                                    b_layer4_0_bn1_running_var]} constant,
     t329 f32 [N=512 T=1 D=1 H=1 W=1 C=256] {folded from=[p_layer4_0_downsample_0_weight,
                                                          p_layer4_0_downsample_1_weight,
                                                          b_layer4_0_downsample_1_running_var]} constant,
     t330 f32 [C=512] {folded from=[p_layer4_0_downsample_1_weight,
                                    p_layer4_0_downsample_1_bias,
                                    b_layer4_0_downsample_1_running_mean,
                                    b_layer4_0_downsample_1_running_var]} constant,
     t331 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {folded from=[p_layer4_0_conv2_weight,
                                                          p_layer4_0_bn2_weight,
                                                          b_layer4_0_bn2_running_var]} constant,
     t332 f32 [C=512] {folded from=[p_layer4_0_bn2_weight, p_layer4_0_bn2_bias,
                                    b_layer4_0_bn2_running_mean,
                                    b_layer4_0_bn2_running_var]} constant,
     t333 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {folded from=[p_layer4_1_conv1_weight,
                                                          p_layer4_1_bn1_weight,
                                                          b_layer4_1_bn1_running_var]} constant,
     t334 f32 [C=512] {folded from=[p_layer4_1_bn1_weight, p_layer4_1_bn1_bias,
                                    b_layer4_1_bn1_running_mean,
                                    b_layer4_1_bn1_running_var]} constant,
     t335 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {folded from=[p_layer4_1_conv2_weight,
                                                          p_layer4_1_bn2_weight,
                                                          b_layer4_1_bn2_running_var]} constant,
     t336 f32 [C=512] {folded from=[p_layer4_1_bn2_weight, p_layer4_1_bn2_bias,
                                    b_layer4_1_bn2_running_mean,
                                    b_layer4_1_bn2_running_var]} constant]
  nodes:
    group g1 torch.ops.aten.convolution.default:
      n0 {derived}: [t123 f32 [H=224 W=224 C=3] {derived}] =
        permute x=t122 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
    n174 {derived}: [t337 f32 [H=112 W=112 C=64] {derived}] =
      convolution
        x=t123 {derived}
        weight=t297 {folded from=[p_conv1_weight, p_bn1_weight,
                                  b_bn1_running_var]}
        bias=t298 {folded from=[p_bn1_weight, p_bn1_bias, b_bn1_running_mean,
                                b_bn1_running_var]}
        params={stride={h=2; w=2};
               padding={h=3; w=3};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g2 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n6 {pt2=root[1] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t129 f32 [H=64
                                                                      W=112
                                                                      C=112] {pt2=root:getitem}] =
        permute x=t337 {derived} perm=[H<-C, W<-H, C<-W]
    n7 {pt2=root[2] torch.ops.aten.relu.default}: [t130 f32 [H=64 W=112 C=112] {pt2=root:relu}] =
      relu x=t129 {pt2=root:getitem}
    group g3 torch.ops.aten.max_pool2d_with_indices.default:
      n8 {derived}: [t131 f32 [H=112 W=112 C=64] {derived}] =
        permute x=t130 {pt2=root:relu} perm=[H<-W, W<-C, C<-H]
      n9 {derived}: [t132 f32 [H=56 W=56 C=64] {derived},
                     t133 f32 [H=56 W=56 C=64] {derived}] =
        max_pool2d_with_indices
          x=t131 {derived}
          params={kernel={h=3; w=3}; stride={h=2; w=2}; pad={h=1; w=1}}
      n10 {derived}: [] = discard x=t133 {derived}
      n11 {pt2=root[3] torch.ops.aten.max_pool2d_with_indices.default}: [t134 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_3}] =
        permute x=t132 {derived} perm=[H<-C, W<-H, C<-W]
    group g4 torch.ops.aten.convolution.default:
      n12 {derived}: [t135 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t134 {pt2=root:getitem_3} perm=[H<-W, W<-C, C<-H]
    n175 {derived}: [t338 f32 [H=56 W=56 C=64] {derived}] =
      convolution
        x=t135 {derived}
        weight=t299 {folded from=[p_layer1_0_conv1_weight,
                                  p_layer1_0_bn1_weight,
                                  b_layer1_0_bn1_running_var]}
        bias=t300 {folded from=[p_layer1_0_bn1_weight, p_layer1_0_bn1_bias,
                                b_layer1_0_bn1_running_mean,
                                b_layer1_0_bn1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g5 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n18 {pt2=root[5] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t141 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_5}] =
        permute x=t338 {derived} perm=[H<-C, W<-H, C<-W]
    n19 {pt2=root[6] torch.ops.aten.relu.default}: [t142 f32 [H=64 W=56 C=56] {pt2=root:relu_1}] =
      relu x=t141 {pt2=root:getitem_5}
    group g6 torch.ops.aten.convolution.default:
      n20 {derived}: [t143 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t142 {pt2=root:relu_1} perm=[H<-W, W<-C, C<-H]
    n176 {derived}: [t339 f32 [H=56 W=56 C=64] {derived}] =
      convolution
        x=t143 {derived}
        weight=t301 {folded from=[p_layer1_0_conv2_weight,
                                  p_layer1_0_bn2_weight,
                                  b_layer1_0_bn2_running_var]}
        bias=t302 {folded from=[p_layer1_0_bn2_weight, p_layer1_0_bn2_bias,
                                b_layer1_0_bn2_running_mean,
                                b_layer1_0_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g7 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n26 {pt2=root[8] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t149 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_8}] =
        permute x=t339 {derived} perm=[H<-C, W<-H, C<-W]
    n27 {pt2=root[9] torch.ops.aten.add.Tensor}: [t150 f32 [H=64 W=56 C=56] {pt2=root:add}] =
      add a=t149 {pt2=root:getitem_8} b=t134 {pt2=root:getitem_3}
    n28 {pt2=root[10] torch.ops.aten.relu.default}: [t151 f32 [H=64 W=56 C=56] {pt2=root:relu_2}] =
      relu x=t150 {pt2=root:add}
    group g8 torch.ops.aten.convolution.default:
      n29 {derived}: [t152 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t151 {pt2=root:relu_2} perm=[H<-W, W<-C, C<-H]
    n177 {derived}: [t340 f32 [H=56 W=56 C=64] {derived}] =
      convolution
        x=t152 {derived}
        weight=t303 {folded from=[p_layer1_1_conv1_weight,
                                  p_layer1_1_bn1_weight,
                                  b_layer1_1_bn1_running_var]}
        bias=t304 {folded from=[p_layer1_1_bn1_weight, p_layer1_1_bn1_bias,
                                b_layer1_1_bn1_running_mean,
                                b_layer1_1_bn1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g9 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n35 {pt2=root[12] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t158 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_11}] =
        permute x=t340 {derived} perm=[H<-C, W<-H, C<-W]
    n36 {pt2=root[13] torch.ops.aten.relu.default}: [t159 f32 [H=64 W=56 C=56] {pt2=root:relu_3}] =
      relu x=t158 {pt2=root:getitem_11}
    group g10 torch.ops.aten.convolution.default:
      n37 {derived}: [t160 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t159 {pt2=root:relu_3} perm=[H<-W, W<-C, C<-H]
    n178 {derived}: [t341 f32 [H=56 W=56 C=64] {derived}] =
      convolution
        x=t160 {derived}
        weight=t305 {folded from=[p_layer1_1_conv2_weight,
                                  p_layer1_1_bn2_weight,
                                  b_layer1_1_bn2_running_var]}
        bias=t306 {folded from=[p_layer1_1_bn2_weight, p_layer1_1_bn2_bias,
                                b_layer1_1_bn2_running_mean,
                                b_layer1_1_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g11 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n43 {pt2=root[15] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t166 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_14}] =
        permute x=t341 {derived} perm=[H<-C, W<-H, C<-W]
    n44 {pt2=root[16] torch.ops.aten.add.Tensor}: [t167 f32 [H=64 W=56 C=56] {pt2=root:add_1}] =
      add a=t166 {pt2=root:getitem_14} b=t151 {pt2=root:relu_2}
    n45 {pt2=root[17] torch.ops.aten.relu.default}: [t168 f32 [H=64 W=56 C=56] {pt2=root:relu_4}] =
      relu x=t167 {pt2=root:add_1}
    group g12 torch.ops.aten.convolution.default:
      n46 {derived}: [t169 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t168 {pt2=root:relu_4} perm=[H<-W, W<-C, C<-H]
    group g16 torch.ops.aten.convolution.default:
      n61 {derived}: [t184 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t168 {pt2=root:relu_4} perm=[H<-W, W<-C, C<-H]
    n179 {derived}: [t342 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t169 {derived}
        weight=t307 {folded from=[p_layer2_0_conv1_weight,
                                  p_layer2_0_bn1_weight,
                                  b_layer2_0_bn1_running_var]}
        bias=t308 {folded from=[p_layer2_0_bn1_weight, p_layer2_0_bn1_bias,
                                b_layer2_0_bn1_running_mean,
                                b_layer2_0_bn1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n180 {derived}: [t343 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t184 {derived}
        weight=t309 {folded from=[p_layer2_0_downsample_0_weight,
                                  p_layer2_0_downsample_1_weight,
                                  b_layer2_0_downsample_1_running_var]}
        bias=t310 {folded from=[p_layer2_0_downsample_1_weight,
                                p_layer2_0_downsample_1_bias,
                                b_layer2_0_downsample_1_running_mean,
                                b_layer2_0_downsample_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g13 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n52 {pt2=root[19] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t175 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_17}] =
        permute x=t342 {derived} perm=[H<-C, W<-H, C<-W]
    group g17 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n67 {pt2=root[24] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t190 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_23}] =
        permute x=t343 {derived} perm=[H<-C, W<-H, C<-W]
    n53 {pt2=root[20] torch.ops.aten.relu.default}: [t176 f32 [H=128 W=28 C=28] {pt2=root:relu_5}] =
      relu x=t175 {pt2=root:getitem_17}
    group g14 torch.ops.aten.convolution.default:
      n54 {derived}: [t177 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t176 {pt2=root:relu_5} perm=[H<-W, W<-C, C<-H]
    n181 {derived}: [t344 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t177 {derived}
        weight=t311 {folded from=[p_layer2_0_conv2_weight,
                                  p_layer2_0_bn2_weight,
                                  b_layer2_0_bn2_running_var]}
        bias=t312 {folded from=[p_layer2_0_bn2_weight, p_layer2_0_bn2_bias,
                                b_layer2_0_bn2_running_mean,
                                b_layer2_0_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g15 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n60 {pt2=root[22] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t183 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_20}] =
        permute x=t344 {derived} perm=[H<-C, W<-H, C<-W]
    n68 {pt2=root[25] torch.ops.aten.add.Tensor}: [t191 f32 [H=128 W=28 C=28] {pt2=root:add_2}] =
      add a=t183 {pt2=root:getitem_20} b=t190 {pt2=root:getitem_23}
    n69 {pt2=root[26] torch.ops.aten.relu.default}: [t192 f32 [H=128 W=28 C=28] {pt2=root:relu_6}] =
      relu x=t191 {pt2=root:add_2}
    group g18 torch.ops.aten.convolution.default:
      n70 {derived}: [t193 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t192 {pt2=root:relu_6} perm=[H<-W, W<-C, C<-H]
    n182 {derived}: [t345 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t193 {derived}
        weight=t313 {folded from=[p_layer2_1_conv1_weight,
                                  p_layer2_1_bn1_weight,
                                  b_layer2_1_bn1_running_var]}
        bias=t314 {folded from=[p_layer2_1_bn1_weight, p_layer2_1_bn1_bias,
                                b_layer2_1_bn1_running_mean,
                                b_layer2_1_bn1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g19 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n76 {pt2=root[28] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t199 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_26}] =
        permute x=t345 {derived} perm=[H<-C, W<-H, C<-W]
    n77 {pt2=root[29] torch.ops.aten.relu.default}: [t200 f32 [H=128 W=28 C=28] {pt2=root:relu_7}] =
      relu x=t199 {pt2=root:getitem_26}
    group g20 torch.ops.aten.convolution.default:
      n78 {derived}: [t201 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t200 {pt2=root:relu_7} perm=[H<-W, W<-C, C<-H]
    n183 {derived}: [t346 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t201 {derived}
        weight=t315 {folded from=[p_layer2_1_conv2_weight,
                                  p_layer2_1_bn2_weight,
                                  b_layer2_1_bn2_running_var]}
        bias=t316 {folded from=[p_layer2_1_bn2_weight, p_layer2_1_bn2_bias,
                                b_layer2_1_bn2_running_mean,
                                b_layer2_1_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g21 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n84 {pt2=root[31] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t207 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_29}] =
        permute x=t346 {derived} perm=[H<-C, W<-H, C<-W]
    n85 {pt2=root[32] torch.ops.aten.add.Tensor}: [t208 f32 [H=128 W=28 C=28] {pt2=root:add_3}] =
      add a=t207 {pt2=root:getitem_29} b=t192 {pt2=root:relu_6}
    n86 {pt2=root[33] torch.ops.aten.relu.default}: [t209 f32 [H=128 W=28 C=28] {pt2=root:relu_8}] =
      relu x=t208 {pt2=root:add_3}
    group g22 torch.ops.aten.convolution.default:
      n87 {derived}: [t210 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t209 {pt2=root:relu_8} perm=[H<-W, W<-C, C<-H]
    group g26 torch.ops.aten.convolution.default:
      n102 {derived}: [t225 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t209 {pt2=root:relu_8} perm=[H<-W, W<-C, C<-H]
    n184 {derived}: [t347 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t210 {derived}
        weight=t317 {folded from=[p_layer3_0_conv1_weight,
                                  p_layer3_0_bn1_weight,
                                  b_layer3_0_bn1_running_var]}
        bias=t318 {folded from=[p_layer3_0_bn1_weight, p_layer3_0_bn1_bias,
                                b_layer3_0_bn1_running_mean,
                                b_layer3_0_bn1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n185 {derived}: [t348 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t225 {derived}
        weight=t319 {folded from=[p_layer3_0_downsample_0_weight,
                                  p_layer3_0_downsample_1_weight,
                                  b_layer3_0_downsample_1_running_var]}
        bias=t320 {folded from=[p_layer3_0_downsample_1_weight,
                                p_layer3_0_downsample_1_bias,
                                b_layer3_0_downsample_1_running_mean,
                                b_layer3_0_downsample_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g23 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n93 {pt2=root[35] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t216 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_32}] =
        permute x=t347 {derived} perm=[H<-C, W<-H, C<-W]
    group g27 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n108 {pt2=root[40] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t231 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_38}] =
        permute x=t348 {derived} perm=[H<-C, W<-H, C<-W]
    n94 {pt2=root[36] torch.ops.aten.relu.default}: [t217 f32 [H=256 W=14 C=14] {pt2=root:relu_9}] =
      relu x=t216 {pt2=root:getitem_32}
    group g24 torch.ops.aten.convolution.default:
      n95 {derived}: [t218 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t217 {pt2=root:relu_9} perm=[H<-W, W<-C, C<-H]
    n186 {derived}: [t349 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t218 {derived}
        weight=t321 {folded from=[p_layer3_0_conv2_weight,
                                  p_layer3_0_bn2_weight,
                                  b_layer3_0_bn2_running_var]}
        bias=t322 {folded from=[p_layer3_0_bn2_weight, p_layer3_0_bn2_bias,
                                b_layer3_0_bn2_running_mean,
                                b_layer3_0_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g25 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n101 {pt2=root[38] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t224 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_35}] =
        permute x=t349 {derived} perm=[H<-C, W<-H, C<-W]
    n109 {pt2=root[41] torch.ops.aten.add.Tensor}: [t232 f32 [H=256 W=14 C=14] {pt2=root:add_4}] =
      add a=t224 {pt2=root:getitem_35} b=t231 {pt2=root:getitem_38}
    n110 {pt2=root[42] torch.ops.aten.relu.default}: [t233 f32 [H=256 W=14
                                                                C=14] {pt2=root:relu_10}] =
      relu x=t232 {pt2=root:add_4}
    group g28 torch.ops.aten.convolution.default:
      n111 {derived}: [t234 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t233 {pt2=root:relu_10} perm=[H<-W, W<-C, C<-H]
    n187 {derived}: [t350 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t234 {derived}
        weight=t323 {folded from=[p_layer3_1_conv1_weight,
                                  p_layer3_1_bn1_weight,
                                  b_layer3_1_bn1_running_var]}
        bias=t324 {folded from=[p_layer3_1_bn1_weight, p_layer3_1_bn1_bias,
                                b_layer3_1_bn1_running_mean,
                                b_layer3_1_bn1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g29 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n117 {pt2=root[44] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t240 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_41}] =
        permute x=t350 {derived} perm=[H<-C, W<-H, C<-W]
    n118 {pt2=root[45] torch.ops.aten.relu.default}: [t241 f32 [H=256 W=14
                                                                C=14] {pt2=root:relu_11}] =
      relu x=t240 {pt2=root:getitem_41}
    group g30 torch.ops.aten.convolution.default:
      n119 {derived}: [t242 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t241 {pt2=root:relu_11} perm=[H<-W, W<-C, C<-H]
    n188 {derived}: [t351 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t242 {derived}
        weight=t325 {folded from=[p_layer3_1_conv2_weight,
                                  p_layer3_1_bn2_weight,
                                  b_layer3_1_bn2_running_var]}
        bias=t326 {folded from=[p_layer3_1_bn2_weight, p_layer3_1_bn2_bias,
                                b_layer3_1_bn2_running_mean,
                                b_layer3_1_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g31 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n125 {pt2=root[47] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t248 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_44}] =
        permute x=t351 {derived} perm=[H<-C, W<-H, C<-W]
    n126 {pt2=root[48] torch.ops.aten.add.Tensor}: [t249 f32 [H=256 W=14 C=14] {pt2=root:add_5}] =
      add a=t248 {pt2=root:getitem_44} b=t233 {pt2=root:relu_10}
    n127 {pt2=root[49] torch.ops.aten.relu.default}: [t250 f32 [H=256 W=14
                                                                C=14] {pt2=root:relu_12}] =
      relu x=t249 {pt2=root:add_5}
    group g32 torch.ops.aten.convolution.default:
      n128 {derived}: [t251 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t250 {pt2=root:relu_12} perm=[H<-W, W<-C, C<-H]
    group g36 torch.ops.aten.convolution.default:
      n143 {derived}: [t266 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t250 {pt2=root:relu_12} perm=[H<-W, W<-C, C<-H]
    n189 {derived}: [t352 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t251 {derived}
        weight=t327 {folded from=[p_layer4_0_conv1_weight,
                                  p_layer4_0_bn1_weight,
                                  b_layer4_0_bn1_running_var]}
        bias=t328 {folded from=[p_layer4_0_bn1_weight, p_layer4_0_bn1_bias,
                                b_layer4_0_bn1_running_mean,
                                b_layer4_0_bn1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n190 {derived}: [t353 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t266 {derived}
        weight=t329 {folded from=[p_layer4_0_downsample_0_weight,
                                  p_layer4_0_downsample_1_weight,
                                  b_layer4_0_downsample_1_running_var]}
        bias=t330 {folded from=[p_layer4_0_downsample_1_weight,
                                p_layer4_0_downsample_1_bias,
                                b_layer4_0_downsample_1_running_mean,
                                b_layer4_0_downsample_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g33 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n134 {pt2=root[51] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t257 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_47}] =
        permute x=t352 {derived} perm=[H<-C, W<-H, C<-W]
    group g37 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n149 {pt2=root[56] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t272 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_53}] =
        permute x=t353 {derived} perm=[H<-C, W<-H, C<-W]
    n135 {pt2=root[52] torch.ops.aten.relu.default}: [t258 f32 [H=512 W=7 C=7] {pt2=root:relu_13}] =
      relu x=t257 {pt2=root:getitem_47}
    group g34 torch.ops.aten.convolution.default:
      n136 {derived}: [t259 f32 [H=7 W=7 C=512] {derived}] =
        permute x=t258 {pt2=root:relu_13} perm=[H<-W, W<-C, C<-H]
    n191 {derived}: [t354 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t259 {derived}
        weight=t331 {folded from=[p_layer4_0_conv2_weight,
                                  p_layer4_0_bn2_weight,
                                  b_layer4_0_bn2_running_var]}
        bias=t332 {folded from=[p_layer4_0_bn2_weight, p_layer4_0_bn2_bias,
                                b_layer4_0_bn2_running_mean,
                                b_layer4_0_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g35 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n142 {pt2=root[54] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t265 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_50}] =
        permute x=t354 {derived} perm=[H<-C, W<-H, C<-W]
    n150 {pt2=root[57] torch.ops.aten.add.Tensor}: [t273 f32 [H=512 W=7 C=7] {pt2=root:add_6}] =
      add a=t265 {pt2=root:getitem_50} b=t272 {pt2=root:getitem_53}
    n151 {pt2=root[58] torch.ops.aten.relu.default}: [t274 f32 [H=512 W=7 C=7] {pt2=root:relu_14}] =
      relu x=t273 {pt2=root:add_6}
    group g38 torch.ops.aten.convolution.default:
      n152 {derived}: [t275 f32 [H=7 W=7 C=512] {derived}] =
        permute x=t274 {pt2=root:relu_14} perm=[H<-W, W<-C, C<-H]
    n192 {derived}: [t355 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t275 {derived}
        weight=t333 {folded from=[p_layer4_1_conv1_weight,
                                  p_layer4_1_bn1_weight,
                                  b_layer4_1_bn1_running_var]}
        bias=t334 {folded from=[p_layer4_1_bn1_weight, p_layer4_1_bn1_bias,
                                b_layer4_1_bn1_running_mean,
                                b_layer4_1_bn1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g39 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n158 {pt2=root[60] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t281 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_56}] =
        permute x=t355 {derived} perm=[H<-C, W<-H, C<-W]
    n159 {pt2=root[61] torch.ops.aten.relu.default}: [t282 f32 [H=512 W=7 C=7] {pt2=root:relu_15}] =
      relu x=t281 {pt2=root:getitem_56}
    group g40 torch.ops.aten.convolution.default:
      n160 {derived}: [t283 f32 [H=7 W=7 C=512] {derived}] =
        permute x=t282 {pt2=root:relu_15} perm=[H<-W, W<-C, C<-H]
    n193 {derived}: [t356 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t283 {derived}
        weight=t335 {folded from=[p_layer4_1_conv2_weight,
                                  p_layer4_1_bn2_weight,
                                  b_layer4_1_bn2_running_var]}
        bias=t336 {folded from=[p_layer4_1_bn2_weight, p_layer4_1_bn2_bias,
                                b_layer4_1_bn2_running_mean,
                                b_layer4_1_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g41 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n166 {pt2=root[63] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t289 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_59}] =
        permute x=t356 {derived} perm=[H<-C, W<-H, C<-W]
    n167 {pt2=root[64] torch.ops.aten.add.Tensor}: [t290 f32 [H=512 W=7 C=7] {pt2=root:add_7}] =
      add a=t289 {pt2=root:getitem_59} b=t274 {pt2=root:relu_14}
    n168 {pt2=root[65] torch.ops.aten.relu.default}: [t291 f32 [H=512 W=7 C=7] {pt2=root:relu_16}] =
      relu x=t290 {pt2=root:add_7}
    n169 {pt2=root[66] torch.ops.aten.mean.dim}: [t292 f32 [H=512 W=1 C=1] {pt2=root:mean}] =
      mean x=t291 {pt2=root:relu_16} params={dims=[C, W]; keepdim=true}
    n194 {pt2=root[67] torch.ops.aten.view.default}: [t293 f32 [C=512] {pt2=root:view}] =
      permute x=t292 {pt2=root:mean} perm=[H<-W, W<-C, C<-H]
    group g42 torch.ops.aten.addmm.default:
      n173 {pt2=root[69] torch.ops.aten.addmm.default}: [t296 f32 [C=1000] {pt2=root:addmm}] =
        linear
          x=t293 {pt2=root:view}
          weight=t295 {folded from=[p_fc_weight]}
          bias=t61 {pt2=root:p_fc_bias target=fc.bias}
          params={in_features=512}
  outputs: [t296 f32 [C=1000] {pt2=root:addmm}]
