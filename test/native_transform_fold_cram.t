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
  nodes: 174 -> 50
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
    n175 {pt2=root[2] torch.ops.aten.relu.default}: [t338 f32 [H=112 W=112
                                                               C=64] {derived}] =
      relu x=t337 {derived}
    group g3 torch.ops.aten.max_pool2d_with_indices.default:
      n9 {derived}: [t132 f32 [H=56 W=56 C=64] {derived},
                     t133 f32 [H=56 W=56 C=64] {derived}] =
        max_pool2d_with_indices
          x=t338 {derived}
          params={kernel={h=3; w=3}; stride={h=2; w=2}; pad={h=1; w=1}}
      n10 {derived}: [] = discard x=t133 {derived}
    n176 {derived}: [t339 f32 [H=56 W=56 C=64] {derived}] =
      convolution
        x=t132 {derived}
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
    n177 {pt2=root[6] torch.ops.aten.relu.default}: [t340 f32 [H=56 W=56 C=64] {derived}] =
      relu x=t339 {derived}
    n178 {derived}: [t341 f32 [H=56 W=56 C=64] {derived}] =
      convolution
        x=t340 {derived}
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
    n179 {pt2=root[9] torch.ops.aten.add.Tensor}: [t342 f32 [H=56 W=56 C=64] {derived}] =
      add a=t341 {derived} b=t132 {derived}
    n180 {pt2=root[10] torch.ops.aten.relu.default}: [t343 f32 [H=56 W=56 C=64] {derived}] =
      relu x=t342 {derived}
    n181 {derived}: [t344 f32 [H=56 W=56 C=64] {derived}] =
      convolution
        x=t343 {derived}
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
    n182 {pt2=root[13] torch.ops.aten.relu.default}: [t345 f32 [H=56 W=56 C=64] {derived}] =
      relu x=t344 {derived}
    n183 {derived}: [t346 f32 [H=56 W=56 C=64] {derived}] =
      convolution
        x=t345 {derived}
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
    n184 {pt2=root[16] torch.ops.aten.add.Tensor}: [t347 f32 [H=56 W=56 C=64] {derived}] =
      add a=t346 {derived} b=t343 {derived}
    n185 {pt2=root[17] torch.ops.aten.relu.default}: [t348 f32 [H=56 W=56 C=64] {derived}] =
      relu x=t347 {derived}
    n186 {derived}: [t349 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t348 {derived}
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
    n187 {derived}: [t350 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t348 {derived}
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
    n188 {pt2=root[20] torch.ops.aten.relu.default}: [t351 f32 [H=28 W=28
                                                                C=128] {derived}] =
      relu x=t349 {derived}
    n189 {derived}: [t352 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t351 {derived}
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
    n190 {pt2=root[25] torch.ops.aten.add.Tensor}: [t353 f32 [H=28 W=28 C=128] {derived}] =
      add a=t352 {derived} b=t350 {derived}
    n191 {pt2=root[26] torch.ops.aten.relu.default}: [t354 f32 [H=28 W=28
                                                                C=128] {derived}] =
      relu x=t353 {derived}
    n192 {derived}: [t355 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t354 {derived}
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
    n193 {pt2=root[29] torch.ops.aten.relu.default}: [t356 f32 [H=28 W=28
                                                                C=128] {derived}] =
      relu x=t355 {derived}
    n194 {derived}: [t357 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t356 {derived}
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
    n195 {pt2=root[32] torch.ops.aten.add.Tensor}: [t358 f32 [H=28 W=28 C=128] {derived}] =
      add a=t357 {derived} b=t354 {derived}
    n196 {pt2=root[33] torch.ops.aten.relu.default}: [t359 f32 [H=28 W=28
                                                                C=128] {derived}] =
      relu x=t358 {derived}
    n197 {derived}: [t360 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t359 {derived}
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
    n198 {derived}: [t361 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t359 {derived}
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
    n199 {pt2=root[36] torch.ops.aten.relu.default}: [t362 f32 [H=14 W=14
                                                                C=256] {derived}] =
      relu x=t360 {derived}
    n200 {derived}: [t363 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t362 {derived}
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
    n201 {pt2=root[41] torch.ops.aten.add.Tensor}: [t364 f32 [H=14 W=14 C=256] {derived}] =
      add a=t363 {derived} b=t361 {derived}
    n202 {pt2=root[42] torch.ops.aten.relu.default}: [t365 f32 [H=14 W=14
                                                                C=256] {derived}] =
      relu x=t364 {derived}
    n203 {derived}: [t366 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t365 {derived}
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
    n204 {pt2=root[45] torch.ops.aten.relu.default}: [t367 f32 [H=14 W=14
                                                                C=256] {derived}] =
      relu x=t366 {derived}
    n205 {derived}: [t368 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t367 {derived}
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
    n206 {pt2=root[48] torch.ops.aten.add.Tensor}: [t369 f32 [H=14 W=14 C=256] {derived}] =
      add a=t368 {derived} b=t365 {derived}
    n207 {pt2=root[49] torch.ops.aten.relu.default}: [t370 f32 [H=14 W=14
                                                                C=256] {derived}] =
      relu x=t369 {derived}
    n208 {derived}: [t371 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t370 {derived}
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
    n209 {derived}: [t372 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t370 {derived}
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
    n210 {pt2=root[52] torch.ops.aten.relu.default}: [t373 f32 [H=7 W=7 C=512] {derived}] =
      relu x=t371 {derived}
    n211 {derived}: [t374 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t373 {derived}
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
    n212 {pt2=root[57] torch.ops.aten.add.Tensor}: [t375 f32 [H=7 W=7 C=512] {derived}] =
      add a=t374 {derived} b=t372 {derived}
    n213 {pt2=root[58] torch.ops.aten.relu.default}: [t376 f32 [H=7 W=7 C=512] {derived}] =
      relu x=t375 {derived}
    n214 {derived}: [t377 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t376 {derived}
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
    n215 {pt2=root[61] torch.ops.aten.relu.default}: [t378 f32 [H=7 W=7 C=512] {derived}] =
      relu x=t377 {derived}
    n216 {derived}: [t379 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t378 {derived}
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
    n217 {pt2=root[64] torch.ops.aten.add.Tensor}: [t380 f32 [H=7 W=7 C=512] {derived}] =
      add a=t379 {derived} b=t376 {derived}
    n218 {pt2=root[65] torch.ops.aten.relu.default}: [t381 f32 [H=7 W=7 C=512] {derived}] =
      relu x=t380 {derived}
    n219 {pt2=root[66] torch.ops.aten.mean.dim}: [t382 f32 [C=512] {pt2=root:view}] =
      mean x=t381 {derived} params={dims=[W, H]; keepdim=true}
    group g42 torch.ops.aten.addmm.default:
      n173 {pt2=root[69] torch.ops.aten.addmm.default}: [t296 f32 [C=1000] {pt2=root:addmm}] =
        linear
          x=t382 {pt2=root:view}
          weight=t295 {folded from=[p_fc_weight]}
          bias=t61 {pt2=root:p_fc_bias target=fc.bias}
          params={in_features=512}
  outputs: [t296 f32 [C=1000] {pt2=root:addmm}]
