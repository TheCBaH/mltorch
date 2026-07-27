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
     t297 f32 [N=64 T=1 D=1 H=7 W=7 C=3] {folded from=[p_conv1_weight,p_bn1_weight,b_bn1_running_var]} constant,
     t298 f32 [C=64] {folded from=[p_bn1_weight,p_bn1_bias,b_bn1_running_mean,b_bn1_running_var]} constant,
     t299 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_0_conv1_weight,p_layer1_0_bn1_weight,b_layer1_0_bn1_running_var]} constant,
     t300 f32 [C=64] {folded from=[p_layer1_0_bn1_weight,p_layer1_0_bn1_bias,b_layer1_0_bn1_running_mean,b_layer1_0_bn1_running_var]} constant,
     t301 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_0_conv2_weight,p_layer1_0_bn2_weight,b_layer1_0_bn2_running_var]} constant,
     t302 f32 [C=64] {folded from=[p_layer1_0_bn2_weight,p_layer1_0_bn2_bias,b_layer1_0_bn2_running_mean,b_layer1_0_bn2_running_var]} constant,
     t303 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_1_conv1_weight,p_layer1_1_bn1_weight,b_layer1_1_bn1_running_var]} constant,
     t304 f32 [C=64] {folded from=[p_layer1_1_bn1_weight,p_layer1_1_bn1_bias,b_layer1_1_bn1_running_mean,b_layer1_1_bn1_running_var]} constant,
     t305 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_1_conv2_weight,p_layer1_1_bn2_weight,b_layer1_1_bn2_running_var]} constant,
     t306 f32 [C=64] {folded from=[p_layer1_1_bn2_weight,p_layer1_1_bn2_bias,b_layer1_1_bn2_running_mean,b_layer1_1_bn2_running_var]} constant,
     t307 f32 [N=128 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer2_0_conv1_weight,p_layer2_0_bn1_weight,b_layer2_0_bn1_running_var]} constant,
     t308 f32 [C=128] {folded from=[p_layer2_0_bn1_weight,p_layer2_0_bn1_bias,b_layer2_0_bn1_running_mean,b_layer2_0_bn1_running_var]} constant,
     t309 f32 [N=128 T=1 D=1 H=1 W=1 C=64] {folded from=[p_layer2_0_downsample_0_weight,p_layer2_0_downsample_1_weight,b_layer2_0_downsample_1_running_var]} constant,
     t310 f32 [C=128] {folded from=[p_layer2_0_downsample_1_weight,p_layer2_0_downsample_1_bias,b_layer2_0_downsample_1_running_mean,b_layer2_0_downsample_1_running_var]} constant,
     t311 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer2_0_conv2_weight,p_layer2_0_bn2_weight,b_layer2_0_bn2_running_var]} constant,
     t312 f32 [C=128] {folded from=[p_layer2_0_bn2_weight,p_layer2_0_bn2_bias,b_layer2_0_bn2_running_mean,b_layer2_0_bn2_running_var]} constant,
     t313 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer2_1_conv1_weight,p_layer2_1_bn1_weight,b_layer2_1_bn1_running_var]} constant,
     t314 f32 [C=128] {folded from=[p_layer2_1_bn1_weight,p_layer2_1_bn1_bias,b_layer2_1_bn1_running_mean,b_layer2_1_bn1_running_var]} constant,
     t315 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer2_1_conv2_weight,p_layer2_1_bn2_weight,b_layer2_1_bn2_running_var]} constant,
     t316 f32 [C=128] {folded from=[p_layer2_1_bn2_weight,p_layer2_1_bn2_bias,b_layer2_1_bn2_running_mean,b_layer2_1_bn2_running_var]} constant,
     t317 f32 [N=256 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer3_0_conv1_weight,p_layer3_0_bn1_weight,b_layer3_0_bn1_running_var]} constant,
     t318 f32 [C=256] {folded from=[p_layer3_0_bn1_weight,p_layer3_0_bn1_bias,b_layer3_0_bn1_running_mean,b_layer3_0_bn1_running_var]} constant,
     t319 f32 [N=256 T=1 D=1 H=1 W=1 C=128] {folded from=[p_layer3_0_downsample_0_weight,p_layer3_0_downsample_1_weight,b_layer3_0_downsample_1_running_var]} constant,
     t320 f32 [C=256] {folded from=[p_layer3_0_downsample_1_weight,p_layer3_0_downsample_1_bias,b_layer3_0_downsample_1_running_mean,b_layer3_0_downsample_1_running_var]} constant,
     t321 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer3_0_conv2_weight,p_layer3_0_bn2_weight,b_layer3_0_bn2_running_var]} constant,
     t322 f32 [C=256] {folded from=[p_layer3_0_bn2_weight,p_layer3_0_bn2_bias,b_layer3_0_bn2_running_mean,b_layer3_0_bn2_running_var]} constant,
     t323 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer3_1_conv1_weight,p_layer3_1_bn1_weight,b_layer3_1_bn1_running_var]} constant,
     t324 f32 [C=256] {folded from=[p_layer3_1_bn1_weight,p_layer3_1_bn1_bias,b_layer3_1_bn1_running_mean,b_layer3_1_bn1_running_var]} constant,
     t325 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer3_1_conv2_weight,p_layer3_1_bn2_weight,b_layer3_1_bn2_running_var]} constant,
     t326 f32 [C=256] {folded from=[p_layer3_1_bn2_weight,p_layer3_1_bn2_bias,b_layer3_1_bn2_running_mean,b_layer3_1_bn2_running_var]} constant,
     t327 f32 [N=512 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer4_0_conv1_weight,p_layer4_0_bn1_weight,b_layer4_0_bn1_running_var]} constant,
     t328 f32 [C=512] {folded from=[p_layer4_0_bn1_weight,p_layer4_0_bn1_bias,b_layer4_0_bn1_running_mean,b_layer4_0_bn1_running_var]} constant,
     t329 f32 [N=512 T=1 D=1 H=1 W=1 C=256] {folded from=[p_layer4_0_downsample_0_weight,p_layer4_0_downsample_1_weight,b_layer4_0_downsample_1_running_var]} constant,
     t330 f32 [C=512] {folded from=[p_layer4_0_downsample_1_weight,p_layer4_0_downsample_1_bias,b_layer4_0_downsample_1_running_mean,b_layer4_0_downsample_1_running_var]} constant,
     t331 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {folded from=[p_layer4_0_conv2_weight,p_layer4_0_bn2_weight,b_layer4_0_bn2_running_var]} constant,
     t332 f32 [C=512] {folded from=[p_layer4_0_bn2_weight,p_layer4_0_bn2_bias,b_layer4_0_bn2_running_mean,b_layer4_0_bn2_running_var]} constant,
     t333 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {folded from=[p_layer4_1_conv1_weight,p_layer4_1_bn1_weight,b_layer4_1_bn1_running_var]} constant,
     t334 f32 [C=512] {folded from=[p_layer4_1_bn1_weight,p_layer4_1_bn1_bias,b_layer4_1_bn1_running_mean,b_layer4_1_bn1_running_var]} constant,
     t335 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {folded from=[p_layer4_1_conv2_weight,p_layer4_1_bn2_weight,b_layer4_1_bn2_running_var]} constant,
     t336 f32 [C=512] {folded from=[p_layer4_1_bn2_weight,p_layer4_1_bn2_bias,b_layer4_1_bn2_running_mean,b_layer4_1_bn2_running_var]} constant]
  nodes:
    group g1 torch.ops.aten.convolution.default:
      n0 {derived}: [t123 f32 [H=224 W=224 C=3] {derived}] =
        permute x=t122 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
    n174 {derived}: [t337 f32 [H=112 W=112 C=64] {derived}] =
      convolution
        x=t123 {derived}
        weight=t297 {folded from=[p_conv1_weight,p_bn1_weight,b_bn1_running_var]}
        bias=t298 {folded from=[p_bn1_weight,p_bn1_bias,b_bn1_running_mean,b_bn1_running_var]}
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
        weight=t299 {folded from=[p_layer1_0_conv1_weight,p_layer1_0_bn1_weight,b_layer1_0_bn1_running_var]}
        bias=t300 {folded from=[p_layer1_0_bn1_weight,p_layer1_0_bn1_bias,b_layer1_0_bn1_running_mean,b_layer1_0_bn1_running_var]}
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
        weight=t301 {folded from=[p_layer1_0_conv2_weight,p_layer1_0_bn2_weight,b_layer1_0_bn2_running_var]}
        bias=t302 {folded from=[p_layer1_0_bn2_weight,p_layer1_0_bn2_bias,b_layer1_0_bn2_running_mean,b_layer1_0_bn2_running_var]}
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
        weight=t303 {folded from=[p_layer1_1_conv1_weight,p_layer1_1_bn1_weight,b_layer1_1_bn1_running_var]}
        bias=t304 {folded from=[p_layer1_1_bn1_weight,p_layer1_1_bn1_bias,b_layer1_1_bn1_running_mean,b_layer1_1_bn1_running_var]}
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
        weight=t305 {folded from=[p_layer1_1_conv2_weight,p_layer1_1_bn2_weight,b_layer1_1_bn2_running_var]}
        bias=t306 {folded from=[p_layer1_1_bn2_weight,p_layer1_1_bn2_bias,b_layer1_1_bn2_running_mean,b_layer1_1_bn2_running_var]}
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
        weight=t307 {folded from=[p_layer2_0_conv1_weight,p_layer2_0_bn1_weight,b_layer2_0_bn1_running_var]}
        bias=t308 {folded from=[p_layer2_0_bn1_weight,p_layer2_0_bn1_bias,b_layer2_0_bn1_running_mean,b_layer2_0_bn1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n187 {derived}: [t350 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t348 {derived}
        weight=t309 {folded from=[p_layer2_0_downsample_0_weight,p_layer2_0_downsample_1_weight,b_layer2_0_downsample_1_running_var]}
        bias=t310 {folded from=[p_layer2_0_downsample_1_weight,p_layer2_0_downsample_1_bias,b_layer2_0_downsample_1_running_mean,b_layer2_0_downsample_1_running_var]}
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
        weight=t311 {folded from=[p_layer2_0_conv2_weight,p_layer2_0_bn2_weight,b_layer2_0_bn2_running_var]}
        bias=t312 {folded from=[p_layer2_0_bn2_weight,p_layer2_0_bn2_bias,b_layer2_0_bn2_running_mean,b_layer2_0_bn2_running_var]}
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
        weight=t313 {folded from=[p_layer2_1_conv1_weight,p_layer2_1_bn1_weight,b_layer2_1_bn1_running_var]}
        bias=t314 {folded from=[p_layer2_1_bn1_weight,p_layer2_1_bn1_bias,b_layer2_1_bn1_running_mean,b_layer2_1_bn1_running_var]}
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
        weight=t315 {folded from=[p_layer2_1_conv2_weight,p_layer2_1_bn2_weight,b_layer2_1_bn2_running_var]}
        bias=t316 {folded from=[p_layer2_1_bn2_weight,p_layer2_1_bn2_bias,b_layer2_1_bn2_running_mean,b_layer2_1_bn2_running_var]}
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
        weight=t317 {folded from=[p_layer3_0_conv1_weight,p_layer3_0_bn1_weight,b_layer3_0_bn1_running_var]}
        bias=t318 {folded from=[p_layer3_0_bn1_weight,p_layer3_0_bn1_bias,b_layer3_0_bn1_running_mean,b_layer3_0_bn1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n198 {derived}: [t361 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t359 {derived}
        weight=t319 {folded from=[p_layer3_0_downsample_0_weight,p_layer3_0_downsample_1_weight,b_layer3_0_downsample_1_running_var]}
        bias=t320 {folded from=[p_layer3_0_downsample_1_weight,p_layer3_0_downsample_1_bias,b_layer3_0_downsample_1_running_mean,b_layer3_0_downsample_1_running_var]}
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
        weight=t321 {folded from=[p_layer3_0_conv2_weight,p_layer3_0_bn2_weight,b_layer3_0_bn2_running_var]}
        bias=t322 {folded from=[p_layer3_0_bn2_weight,p_layer3_0_bn2_bias,b_layer3_0_bn2_running_mean,b_layer3_0_bn2_running_var]}
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
        weight=t323 {folded from=[p_layer3_1_conv1_weight,p_layer3_1_bn1_weight,b_layer3_1_bn1_running_var]}
        bias=t324 {folded from=[p_layer3_1_bn1_weight,p_layer3_1_bn1_bias,b_layer3_1_bn1_running_mean,b_layer3_1_bn1_running_var]}
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
        weight=t325 {folded from=[p_layer3_1_conv2_weight,p_layer3_1_bn2_weight,b_layer3_1_bn2_running_var]}
        bias=t326 {folded from=[p_layer3_1_bn2_weight,p_layer3_1_bn2_bias,b_layer3_1_bn2_running_mean,b_layer3_1_bn2_running_var]}
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
        weight=t327 {folded from=[p_layer4_0_conv1_weight,p_layer4_0_bn1_weight,b_layer4_0_bn1_running_var]}
        bias=t328 {folded from=[p_layer4_0_bn1_weight,p_layer4_0_bn1_bias,b_layer4_0_bn1_running_mean,b_layer4_0_bn1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n209 {derived}: [t372 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t370 {derived}
        weight=t329 {folded from=[p_layer4_0_downsample_0_weight,p_layer4_0_downsample_1_weight,b_layer4_0_downsample_1_running_var]}
        bias=t330 {folded from=[p_layer4_0_downsample_1_weight,p_layer4_0_downsample_1_bias,b_layer4_0_downsample_1_running_mean,b_layer4_0_downsample_1_running_var]}
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
        weight=t331 {folded from=[p_layer4_0_conv2_weight,p_layer4_0_bn2_weight,b_layer4_0_bn2_running_var]}
        bias=t332 {folded from=[p_layer4_0_bn2_weight,p_layer4_0_bn2_bias,b_layer4_0_bn2_running_mean,b_layer4_0_bn2_running_var]}
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
        weight=t333 {folded from=[p_layer4_1_conv1_weight,p_layer4_1_bn1_weight,b_layer4_1_bn1_running_var]}
        bias=t334 {folded from=[p_layer4_1_bn1_weight,p_layer4_1_bn1_bias,b_layer4_1_bn1_running_mean,b_layer4_1_bn1_running_var]}
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
        weight=t335 {folded from=[p_layer4_1_conv2_weight,p_layer4_1_bn2_weight,b_layer4_1_bn2_running_var]}
        bias=t336 {folded from=[p_layer4_1_bn2_weight,p_layer4_1_bn2_bias,b_layer4_1_bn2_running_mean,b_layer4_1_bn2_running_var]}
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

MobileNet through the full folding pipeline, printed in full for the same reason
as in native_transform_cram.t. What this adds over the structural run is visible
in the layout: every batch_norm has folded into its convolution and the weight
relayouts have become load-time constants, so the permutes that remain are the
ones folding could not hoist.

MobileNet-v3-small: the hardswish ops are untouched by folding — they sit on
activations, not on constants.

  $ ../bin/native_graph.exe transform --fold --pt2 "$PT2_DATA/mobilenet_v3_small/mobilenet_v3_small.pt2"
  nodes: 488 -> 282
  constants: 108, of which 88 folded
  graph
  inputs:
    [t7 f32 [C=8] {pt2=root:p_features_1_block_1_fc1_bias target=features.1.block.1.fc1.bias} constant,
     t9 f32 [C=16] {pt2=root:p_features_1_block_1_fc2_bias target=features.1.block.1.fc2.bias} constant,
     t38 f32 [C=24] {pt2=root:p_features_4_block_2_fc1_bias target=features.4.block.2.fc1.bias} constant,
     t40 f32 [C=96] {pt2=root:p_features_4_block_2_fc2_bias target=features.4.block.2.fc2.bias} constant,
     t51 f32 [C=64] {pt2=root:p_features_5_block_2_fc1_bias target=features.5.block.2.fc1.bias} constant,
     t53 f32 [C=240] {pt2=root:p_features_5_block_2_fc2_bias target=features.5.block.2.fc2.bias} constant,
     t64 f32 [C=64] {pt2=root:p_features_6_block_2_fc1_bias target=features.6.block.2.fc1.bias} constant,
     t66 f32 [C=240] {pt2=root:p_features_6_block_2_fc2_bias target=features.6.block.2.fc2.bias} constant,
     t77 f32 [C=32] {pt2=root:p_features_7_block_2_fc1_bias target=features.7.block.2.fc1.bias} constant,
     t79 f32 [C=120] {pt2=root:p_features_7_block_2_fc2_bias target=features.7.block.2.fc2.bias} constant,
     t90 f32 [C=40] {pt2=root:p_features_8_block_2_fc1_bias target=features.8.block.2.fc1.bias} constant,
     t92 f32 [C=144] {pt2=root:p_features_8_block_2_fc2_bias target=features.8.block.2.fc2.bias} constant,
     t103 f32 [C=72] {pt2=root:p_features_9_block_2_fc1_bias target=features.9.block.2.fc1.bias} constant,
     t105 f32 [C=288] {pt2=root:p_features_9_block_2_fc2_bias target=features.9.block.2.fc2.bias} constant,
     t116 f32 [C=144] {pt2=root:p_features_10_block_2_fc1_bias target=features.10.block.2.fc1.bias} constant,
     t118 f32 [C=576] {pt2=root:p_features_10_block_2_fc2_bias target=features.10.block.2.fc2.bias} constant,
     t129 f32 [C=144] {pt2=root:p_features_11_block_2_fc1_bias target=features.11.block.2.fc1.bias} constant,
     t131 f32 [C=576] {pt2=root:p_features_11_block_2_fc2_bias target=features.11.block.2.fc2.bias} constant,
     t139 f32 [C=1024] {pt2=root:p_classifier_0_bias target=classifier.0.bias} constant,
     t141 f32 [C=1000] {pt2=root:p_classifier_3_bias target=classifier.3.bias} constant,
     t244 f32 [H=3 W=224 C=224] {pt2=root:x},
     t267 f32 [N=8 T=1 D=1 H=1 W=1 C=16] {folded from=[p_features_1_block_1_fc1_weight]} constant,
     t272 f32 [N=16 T=1 D=1 H=1 W=1 C=8] {folded from=[p_features_1_block_1_fc2_weight]} constant,
     t360 f32 [N=24 T=1 D=1 H=1 W=1 C=96] {folded from=[p_features_4_block_2_fc1_weight]} constant,
     t365 f32 [N=96 T=1 D=1 H=1 W=1 C=24] {folded from=[p_features_4_block_2_fc2_weight]} constant,
     t406 f32 [N=64 T=1 D=1 H=1 W=1 C=240] {folded from=[p_features_5_block_2_fc1_weight]} constant,
     t411 f32 [N=240 T=1 D=1 H=1 W=1 C=64] {folded from=[p_features_5_block_2_fc2_weight]} constant,
     t453 f32 [N=64 T=1 D=1 H=1 W=1 C=240] {folded from=[p_features_6_block_2_fc1_weight]} constant,
     t458 f32 [N=240 T=1 D=1 H=1 W=1 C=64] {folded from=[p_features_6_block_2_fc2_weight]} constant,
     t500 f32 [N=32 T=1 D=1 H=1 W=1 C=120] {folded from=[p_features_7_block_2_fc1_weight]} constant,
     t505 f32 [N=120 T=1 D=1 H=1 W=1 C=32] {folded from=[p_features_7_block_2_fc2_weight]} constant,
     t546 f32 [N=40 T=1 D=1 H=1 W=1 C=144] {folded from=[p_features_8_block_2_fc1_weight]} constant,
     t551 f32 [N=144 T=1 D=1 H=1 W=1 C=40] {folded from=[p_features_8_block_2_fc2_weight]} constant,
     t593 f32 [N=72 T=1 D=1 H=1 W=1 C=288] {folded from=[p_features_9_block_2_fc1_weight]} constant,
     t598 f32 [N=288 T=1 D=1 H=1 W=1 C=72] {folded from=[p_features_9_block_2_fc2_weight]} constant,
     t639 f32 [N=144 T=1 D=1 H=1 W=1 C=576] {folded from=[p_features_10_block_2_fc1_weight]} constant,
     t644 f32 [N=576 T=1 D=1 H=1 W=1 C=144] {folded from=[p_features_10_block_2_fc2_weight]} constant,
     t686 f32 [N=144 T=1 D=1 H=1 W=1 C=576] {folded from=[p_features_11_block_2_fc1_weight]} constant,
     t691 f32 [N=576 T=1 D=1 H=1 W=1 C=144] {folded from=[p_features_11_block_2_fc2_weight]} constant,
     t722 f32 [N=1024 T=1 D=1 H=1 W=1 C=576] {folded from=[p_classifier_0_weight]} constant,
     t731 f32 [N=1000 T=1 D=1 H=1 W=1 C=1024] {folded from=[p_classifier_3_weight]} constant,
     t733 f32 [N=16 T=1 D=1 H=3 W=3 C=3] {folded from=[p_features_0_0_weight,p_features_0_1_weight,b_features_0_1_running_var]} constant,
     t734 f32 [C=16] {folded from=[p_features_0_1_weight,p_features_0_1_bias,b_features_0_1_running_mean,b_features_0_1_running_var]} constant,
     t735 f32 [N=16 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_1_block_0_0_weight,p_features_1_block_0_1_weight,b_features_1_block_0_1_running_var]} constant,
     t736 f32 [C=16] {folded from=[p_features_1_block_0_1_weight,p_features_1_block_0_1_bias,b_features_1_block_0_1_running_mean,b_features_1_block_0_1_running_var]} constant,
     t737 f32 [N=16 T=1 D=1 H=1 W=1 C=16] {folded from=[p_features_1_block_2_0_weight,p_features_1_block_2_1_weight,b_features_1_block_2_1_running_var]} constant,
     t738 f32 [C=16] {folded from=[p_features_1_block_2_1_weight,p_features_1_block_2_1_bias,b_features_1_block_2_1_running_mean,b_features_1_block_2_1_running_var]} constant,
     t739 f32 [N=72 T=1 D=1 H=1 W=1 C=16] {folded from=[p_features_2_block_0_0_weight,p_features_2_block_0_1_weight,b_features_2_block_0_1_running_var]} constant,
     t740 f32 [C=72] {folded from=[p_features_2_block_0_1_weight,p_features_2_block_0_1_bias,b_features_2_block_0_1_running_mean,b_features_2_block_0_1_running_var]} constant,
     t741 f32 [N=72 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_2_block_1_0_weight,p_features_2_block_1_1_weight,b_features_2_block_1_1_running_var]} constant,
     t742 f32 [C=72] {folded from=[p_features_2_block_1_1_weight,p_features_2_block_1_1_bias,b_features_2_block_1_1_running_mean,b_features_2_block_1_1_running_var]} constant,
     t743 f32 [N=24 T=1 D=1 H=1 W=1 C=72] {folded from=[p_features_2_block_2_0_weight,p_features_2_block_2_1_weight,b_features_2_block_2_1_running_var]} constant,
     t744 f32 [C=24] {folded from=[p_features_2_block_2_1_weight,p_features_2_block_2_1_bias,b_features_2_block_2_1_running_mean,b_features_2_block_2_1_running_var]} constant,
     t745 f32 [N=88 T=1 D=1 H=1 W=1 C=24] {folded from=[p_features_3_block_0_0_weight,p_features_3_block_0_1_weight,b_features_3_block_0_1_running_var]} constant,
     t746 f32 [C=88] {folded from=[p_features_3_block_0_1_weight,p_features_3_block_0_1_bias,b_features_3_block_0_1_running_mean,b_features_3_block_0_1_running_var]} constant,
     t747 f32 [N=88 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_3_block_1_0_weight,p_features_3_block_1_1_weight,b_features_3_block_1_1_running_var]} constant,
     t748 f32 [C=88] {folded from=[p_features_3_block_1_1_weight,p_features_3_block_1_1_bias,b_features_3_block_1_1_running_mean,b_features_3_block_1_1_running_var]} constant,
     t749 f32 [N=24 T=1 D=1 H=1 W=1 C=88] {folded from=[p_features_3_block_2_0_weight,p_features_3_block_2_1_weight,b_features_3_block_2_1_running_var]} constant,
     t750 f32 [C=24] {folded from=[p_features_3_block_2_1_weight,p_features_3_block_2_1_bias,b_features_3_block_2_1_running_mean,b_features_3_block_2_1_running_var]} constant,
     t751 f32 [N=96 T=1 D=1 H=1 W=1 C=24] {folded from=[p_features_4_block_0_0_weight,p_features_4_block_0_1_weight,b_features_4_block_0_1_running_var]} constant,
     t752 f32 [C=96] {folded from=[p_features_4_block_0_1_weight,p_features_4_block_0_1_bias,b_features_4_block_0_1_running_mean,b_features_4_block_0_1_running_var]} constant,
     t753 f32 [N=96 T=1 D=1 H=5 W=5 C=1] {folded from=[p_features_4_block_1_0_weight,p_features_4_block_1_1_weight,b_features_4_block_1_1_running_var]} constant,
     t754 f32 [C=96] {folded from=[p_features_4_block_1_1_weight,p_features_4_block_1_1_bias,b_features_4_block_1_1_running_mean,b_features_4_block_1_1_running_var]} constant,
     t755 f32 [N=40 T=1 D=1 H=1 W=1 C=96] {folded from=[p_features_4_block_3_0_weight,p_features_4_block_3_1_weight,b_features_4_block_3_1_running_var]} constant,
     t756 f32 [C=40] {folded from=[p_features_4_block_3_1_weight,p_features_4_block_3_1_bias,b_features_4_block_3_1_running_mean,b_features_4_block_3_1_running_var]} constant,
     t757 f32 [N=240 T=1 D=1 H=1 W=1 C=40] {folded from=[p_features_5_block_0_0_weight,p_features_5_block_0_1_weight,b_features_5_block_0_1_running_var]} constant,
     t758 f32 [C=240] {folded from=[p_features_5_block_0_1_weight,p_features_5_block_0_1_bias,b_features_5_block_0_1_running_mean,b_features_5_block_0_1_running_var]} constant,
     t759 f32 [N=240 T=1 D=1 H=5 W=5 C=1] {folded from=[p_features_5_block_1_0_weight,p_features_5_block_1_1_weight,b_features_5_block_1_1_running_var]} constant,
     t760 f32 [C=240] {folded from=[p_features_5_block_1_1_weight,p_features_5_block_1_1_bias,b_features_5_block_1_1_running_mean,b_features_5_block_1_1_running_var]} constant,
     t761 f32 [N=40 T=1 D=1 H=1 W=1 C=240] {folded from=[p_features_5_block_3_0_weight,p_features_5_block_3_1_weight,b_features_5_block_3_1_running_var]} constant,
     t762 f32 [C=40] {folded from=[p_features_5_block_3_1_weight,p_features_5_block_3_1_bias,b_features_5_block_3_1_running_mean,b_features_5_block_3_1_running_var]} constant,
     t763 f32 [N=240 T=1 D=1 H=1 W=1 C=40] {folded from=[p_features_6_block_0_0_weight,p_features_6_block_0_1_weight,b_features_6_block_0_1_running_var]} constant,
     t764 f32 [C=240] {folded from=[p_features_6_block_0_1_weight,p_features_6_block_0_1_bias,b_features_6_block_0_1_running_mean,b_features_6_block_0_1_running_var]} constant,
     t765 f32 [N=240 T=1 D=1 H=5 W=5 C=1] {folded from=[p_features_6_block_1_0_weight,p_features_6_block_1_1_weight,b_features_6_block_1_1_running_var]} constant,
     t766 f32 [C=240] {folded from=[p_features_6_block_1_1_weight,p_features_6_block_1_1_bias,b_features_6_block_1_1_running_mean,b_features_6_block_1_1_running_var]} constant,
     t767 f32 [N=40 T=1 D=1 H=1 W=1 C=240] {folded from=[p_features_6_block_3_0_weight,p_features_6_block_3_1_weight,b_features_6_block_3_1_running_var]} constant,
     t768 f32 [C=40] {folded from=[p_features_6_block_3_1_weight,p_features_6_block_3_1_bias,b_features_6_block_3_1_running_mean,b_features_6_block_3_1_running_var]} constant,
     t769 f32 [N=120 T=1 D=1 H=1 W=1 C=40] {folded from=[p_features_7_block_0_0_weight,p_features_7_block_0_1_weight,b_features_7_block_0_1_running_var]} constant,
     t770 f32 [C=120] {folded from=[p_features_7_block_0_1_weight,p_features_7_block_0_1_bias,b_features_7_block_0_1_running_mean,b_features_7_block_0_1_running_var]} constant,
     t771 f32 [N=120 T=1 D=1 H=5 W=5 C=1] {folded from=[p_features_7_block_1_0_weight,p_features_7_block_1_1_weight,b_features_7_block_1_1_running_var]} constant,
     t772 f32 [C=120] {folded from=[p_features_7_block_1_1_weight,p_features_7_block_1_1_bias,b_features_7_block_1_1_running_mean,b_features_7_block_1_1_running_var]} constant,
     t773 f32 [N=48 T=1 D=1 H=1 W=1 C=120] {folded from=[p_features_7_block_3_0_weight,p_features_7_block_3_1_weight,b_features_7_block_3_1_running_var]} constant,
     t774 f32 [C=48] {folded from=[p_features_7_block_3_1_weight,p_features_7_block_3_1_bias,b_features_7_block_3_1_running_mean,b_features_7_block_3_1_running_var]} constant,
     t775 f32 [N=144 T=1 D=1 H=1 W=1 C=48] {folded from=[p_features_8_block_0_0_weight,p_features_8_block_0_1_weight,b_features_8_block_0_1_running_var]} constant,
     t776 f32 [C=144] {folded from=[p_features_8_block_0_1_weight,p_features_8_block_0_1_bias,b_features_8_block_0_1_running_mean,b_features_8_block_0_1_running_var]} constant,
     t777 f32 [N=144 T=1 D=1 H=5 W=5 C=1] {folded from=[p_features_8_block_1_0_weight,p_features_8_block_1_1_weight,b_features_8_block_1_1_running_var]} constant,
     t778 f32 [C=144] {folded from=[p_features_8_block_1_1_weight,p_features_8_block_1_1_bias,b_features_8_block_1_1_running_mean,b_features_8_block_1_1_running_var]} constant,
     t779 f32 [N=48 T=1 D=1 H=1 W=1 C=144] {folded from=[p_features_8_block_3_0_weight,p_features_8_block_3_1_weight,b_features_8_block_3_1_running_var]} constant,
     t780 f32 [C=48] {folded from=[p_features_8_block_3_1_weight,p_features_8_block_3_1_bias,b_features_8_block_3_1_running_mean,b_features_8_block_3_1_running_var]} constant,
     t781 f32 [N=288 T=1 D=1 H=1 W=1 C=48] {folded from=[p_features_9_block_0_0_weight,p_features_9_block_0_1_weight,b_features_9_block_0_1_running_var]} constant,
     t782 f32 [C=288] {folded from=[p_features_9_block_0_1_weight,p_features_9_block_0_1_bias,b_features_9_block_0_1_running_mean,b_features_9_block_0_1_running_var]} constant,
     t783 f32 [N=288 T=1 D=1 H=5 W=5 C=1] {folded from=[p_features_9_block_1_0_weight,p_features_9_block_1_1_weight,b_features_9_block_1_1_running_var]} constant,
     t784 f32 [C=288] {folded from=[p_features_9_block_1_1_weight,p_features_9_block_1_1_bias,b_features_9_block_1_1_running_mean,b_features_9_block_1_1_running_var]} constant,
     t785 f32 [N=96 T=1 D=1 H=1 W=1 C=288] {folded from=[p_features_9_block_3_0_weight,p_features_9_block_3_1_weight,b_features_9_block_3_1_running_var]} constant,
     t786 f32 [C=96] {folded from=[p_features_9_block_3_1_weight,p_features_9_block_3_1_bias,b_features_9_block_3_1_running_mean,b_features_9_block_3_1_running_var]} constant,
     t787 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {folded from=[p_features_10_block_0_0_weight,p_features_10_block_0_1_weight,b_features_10_block_0_1_running_var]} constant,
     t788 f32 [C=576] {folded from=[p_features_10_block_0_1_weight,p_features_10_block_0_1_bias,b_features_10_block_0_1_running_mean,b_features_10_block_0_1_running_var]} constant,
     t789 f32 [N=576 T=1 D=1 H=5 W=5 C=1] {folded from=[p_features_10_block_1_0_weight,p_features_10_block_1_1_weight,b_features_10_block_1_1_running_var]} constant,
     t790 f32 [C=576] {folded from=[p_features_10_block_1_1_weight,p_features_10_block_1_1_bias,b_features_10_block_1_1_running_mean,b_features_10_block_1_1_running_var]} constant,
     t791 f32 [N=96 T=1 D=1 H=1 W=1 C=576] {folded from=[p_features_10_block_3_0_weight,p_features_10_block_3_1_weight,b_features_10_block_3_1_running_var]} constant,
     t792 f32 [C=96] {folded from=[p_features_10_block_3_1_weight,p_features_10_block_3_1_bias,b_features_10_block_3_1_running_mean,b_features_10_block_3_1_running_var]} constant,
     t793 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {folded from=[p_features_11_block_0_0_weight,p_features_11_block_0_1_weight,b_features_11_block_0_1_running_var]} constant,
     t794 f32 [C=576] {folded from=[p_features_11_block_0_1_weight,p_features_11_block_0_1_bias,b_features_11_block_0_1_running_mean,b_features_11_block_0_1_running_var]} constant,
     t795 f32 [N=576 T=1 D=1 H=5 W=5 C=1] {folded from=[p_features_11_block_1_0_weight,p_features_11_block_1_1_weight,b_features_11_block_1_1_running_var]} constant,
     t796 f32 [C=576] {folded from=[p_features_11_block_1_1_weight,p_features_11_block_1_1_bias,b_features_11_block_1_1_running_mean,b_features_11_block_1_1_running_var]} constant,
     t797 f32 [N=96 T=1 D=1 H=1 W=1 C=576] {folded from=[p_features_11_block_3_0_weight,p_features_11_block_3_1_weight,b_features_11_block_3_1_running_var]} constant,
     t798 f32 [C=96] {folded from=[p_features_11_block_3_1_weight,p_features_11_block_3_1_bias,b_features_11_block_3_1_running_mean,b_features_11_block_3_1_running_var]} constant,
     t799 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {folded from=[p_features_12_0_weight,p_features_12_1_weight,b_features_12_1_running_var]} constant,
     t800 f32 [C=576] {folded from=[p_features_12_1_weight,p_features_12_1_bias,b_features_12_1_running_mean,b_features_12_1_running_var]} constant]
  nodes:
    group g1 torch.ops.aten.convolution.default:
      n0 {derived}: [t245 f32 [H=224 W=224 C=3] {derived}] =
        permute x=t244 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
    n488 {derived}: [t801 f32 [H=112 W=112 C=16] {derived}] =
      convolution
        x=t245 {derived}
        weight=t733 {folded from=[p_features_0_0_weight,p_features_0_1_weight,b_features_0_1_running_var]}
        bias=t734 {folded from=[p_features_0_1_weight,p_features_0_1_bias,b_features_0_1_running_mean,b_features_0_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g2 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n6 {pt2=root[1] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t251 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:getitem}] =
        permute x=t801 {derived} perm=[H<-C, W<-H, C<-W]
    n7 {pt2=root[2] torch.ops.aten.add.Tensor}: [t252 f32 [H=16 W=112 C=112] {pt2=root:add}] =
      add_scalar x=t251 {pt2=root:getitem} scalar=3
    n8 {pt2=root[3] torch.ops.aten.clamp.default}: [t253 f32 [H=16 W=112 C=112] {pt2=root:clamp}] =
      clamp x=t252 {pt2=root:add} params={min=0; max=none}
    n9 {pt2=root[4] torch.ops.aten.clamp.default}: [t254 f32 [H=16 W=112 C=112] {pt2=root:clamp_1}] =
      clamp x=t253 {pt2=root:clamp} params={min=none; max=6}
    n10 {pt2=root[5] torch.ops.aten.mul.Tensor}: [t255 f32 [H=16 W=112 C=112] {pt2=root:mul}] =
      mul a=t251 {pt2=root:getitem} b=t254 {pt2=root:clamp_1}
    n11 {pt2=root[6] torch.ops.aten.div.Tensor}: [t256 f32 [H=16 W=112 C=112] {pt2=root:div}] =
      div_scalar x=t255 {pt2=root:mul} scalar=6
    group g3 torch.ops.aten.convolution.default:
      n12 {derived}: [t257 f32 [H=112 W=112 C=16] {derived}] =
        permute x=t256 {pt2=root:div} perm=[H<-W, W<-C, C<-H]
    n489 {derived}: [t802 f32 [H=56 W=56 C=16] {derived}] =
      convolution
        x=t257 {derived}
        weight=t735 {folded from=[p_features_1_block_0_0_weight,p_features_1_block_0_1_weight,b_features_1_block_0_1_running_var]}
        bias=t736 {folded from=[p_features_1_block_0_1_weight,p_features_1_block_0_1_bias,b_features_1_block_0_1_running_mean,b_features_1_block_0_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=16}
    n490 {pt2=root[9] torch.ops.aten.relu.default}: [t803 f32 [H=56 W=56 C=16] {derived}] =
      relu x=t802 {derived}
    n491 {pt2=root[8] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t264 f32 [H=16
                                                                      W=56
                                                                      C=56] {pt2=root:relu}] =
      permute x=t803 {derived} perm=[H<-C, W<-H, C<-W]
    n20 {pt2=root[10] torch.ops.aten.mean.dim}: [t265 f32 [H=16 W=1 C=1] {pt2=root:mean}] =
      mean x=t264 {pt2=root:relu} params={dims=[C, W]; keepdim=true}
    group g5 torch.ops.aten.convolution.default:
      n21 {derived}: [t266 f32 [C=16] {derived}] =
        permute x=t265 {pt2=root:mean} perm=[H<-W, W<-C, C<-H]
      n23 {derived}: [t268 f32 [C=8] {derived}] =
        convolution
          x=t266 {derived}
          weight=t267 {folded from=[p_features_1_block_1_fc1_weight]}
          bias=t7 {pt2=root:p_features_1_block_1_fc1_bias target=features.1.block.1.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n492 {pt2=root[12] torch.ops.aten.relu.default}: [t804 f32 [C=8] {derived}] =
      relu x=t268 {derived}
    group g6 torch.ops.aten.convolution.default:
      n28 {derived}: [t273 f32 [C=16] {derived}] =
        convolution
          x=t804 {derived}
          weight=t272 {folded from=[p_features_1_block_1_fc2_weight]}
          bias=t9 {pt2=root:p_features_1_block_1_fc2_bias target=features.1.block.1.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n493 {pt2=root[14] torch.ops.aten.add.Tensor}: [t805 f32 [C=16] {derived}] =
      add_scalar x=t273 {derived} scalar=3
    n494 {pt2=root[15] torch.ops.aten.clamp.default}: [t806 f32 [C=16] {derived}] =
      clamp x=t805 {derived} params={min=0; max=none}
    n495 {pt2=root[16] torch.ops.aten.clamp.default}: [t807 f32 [C=16] {derived}] =
      clamp x=t806 {derived} params={min=none; max=6}
    n496 {pt2=root[17] torch.ops.aten.div.Tensor}: [t808 f32 [C=16] {derived}] =
      div_scalar x=t807 {derived} scalar=6
    n497 {pt2=root[13] torch.ops.aten.convolution.default}: [t278 f32 [H=16 W=1
                                                                      C=1] {pt2=root:div_1}] =
      permute x=t808 {derived} perm=[H<-C, W<-H, C<-W]
    n34 {pt2=root[18] torch.ops.aten.mul.Tensor}: [t279 f32 [H=16 W=56 C=56] {pt2=root:mul_1}] =
      mul a=t278 {pt2=root:div_1} b=t264 {pt2=root:relu}
    group g7 torch.ops.aten.convolution.default:
      n35 {derived}: [t280 f32 [H=56 W=56 C=16] {derived}] =
        permute x=t279 {pt2=root:mul_1} perm=[H<-W, W<-C, C<-H]
    n498 {derived}: [t809 f32 [H=56 W=56 C=16] {derived}] =
      convolution
        x=t280 {derived}
        weight=t737 {folded from=[p_features_1_block_2_0_weight,p_features_1_block_2_1_weight,b_features_1_block_2_1_running_var]}
        bias=t738 {folded from=[p_features_1_block_2_1_weight,p_features_1_block_2_1_bias,b_features_1_block_2_1_running_mean,b_features_1_block_2_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n499 {derived}: [t810 f32 [H=56 W=56 C=72] {derived}] =
      convolution
        x=t809 {derived}
        weight=t739 {folded from=[p_features_2_block_0_0_weight,p_features_2_block_0_1_weight,b_features_2_block_0_1_running_var]}
        bias=t740 {folded from=[p_features_2_block_0_1_weight,p_features_2_block_0_1_bias,b_features_2_block_0_1_running_mean,b_features_2_block_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n500 {pt2=root[23] torch.ops.aten.relu.default}: [t811 f32 [H=56 W=56 C=72] {derived}] =
      relu x=t810 {derived}
    n501 {derived}: [t812 f32 [H=28 W=28 C=72] {derived}] =
      convolution
        x=t811 {derived}
        weight=t741 {folded from=[p_features_2_block_1_0_weight,p_features_2_block_1_1_weight,b_features_2_block_1_1_running_var]}
        bias=t742 {folded from=[p_features_2_block_1_1_weight,p_features_2_block_1_1_bias,b_features_2_block_1_1_running_mean,b_features_2_block_1_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=72}
    n502 {pt2=root[26] torch.ops.aten.relu.default}: [t813 f32 [H=28 W=28 C=72] {derived}] =
      relu x=t812 {derived}
    n503 {derived}: [t814 f32 [H=28 W=28 C=24] {derived}] =
      convolution
        x=t813 {derived}
        weight=t743 {folded from=[p_features_2_block_2_0_weight,p_features_2_block_2_1_weight,b_features_2_block_2_1_running_var]}
        bias=t744 {folded from=[p_features_2_block_2_1_weight,p_features_2_block_2_1_bias,b_features_2_block_2_1_running_mean,b_features_2_block_2_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n504 {derived}: [t815 f32 [H=28 W=28 C=88] {derived}] =
      convolution
        x=t814 {derived}
        weight=t745 {folded from=[p_features_3_block_0_0_weight,p_features_3_block_0_1_weight,b_features_3_block_0_1_running_var]}
        bias=t746 {folded from=[p_features_3_block_0_1_weight,p_features_3_block_0_1_bias,b_features_3_block_0_1_running_mean,b_features_3_block_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n505 {pt2=root[31] torch.ops.aten.relu.default}: [t816 f32 [H=28 W=28 C=88] {derived}] =
      relu x=t815 {derived}
    n506 {derived}: [t817 f32 [H=28 W=28 C=88] {derived}] =
      convolution
        x=t816 {derived}
        weight=t747 {folded from=[p_features_3_block_1_0_weight,p_features_3_block_1_1_weight,b_features_3_block_1_1_running_var]}
        bias=t748 {folded from=[p_features_3_block_1_1_weight,p_features_3_block_1_1_bias,b_features_3_block_1_1_running_mean,b_features_3_block_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=88}
    n507 {pt2=root[34] torch.ops.aten.relu.default}: [t818 f32 [H=28 W=28 C=88] {derived}] =
      relu x=t817 {derived}
    n508 {derived}: [t819 f32 [H=28 W=28 C=24] {derived}] =
      convolution
        x=t818 {derived}
        weight=t749 {folded from=[p_features_3_block_2_0_weight,p_features_3_block_2_1_weight,b_features_3_block_2_1_running_var]}
        bias=t750 {folded from=[p_features_3_block_2_1_weight,p_features_3_block_2_1_bias,b_features_3_block_2_1_running_mean,b_features_3_block_2_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n509 {pt2=root[37] torch.ops.aten.add.Tensor}: [t820 f32 [H=28 W=28 C=24] {derived}] =
      add a=t819 {derived} b=t814 {derived}
    n510 {derived}: [t821 f32 [H=28 W=28 C=96] {derived}] =
      convolution
        x=t820 {derived}
        weight=t751 {folded from=[p_features_4_block_0_0_weight,p_features_4_block_0_1_weight,b_features_4_block_0_1_running_var]}
        bias=t752 {folded from=[p_features_4_block_0_1_weight,p_features_4_block_0_1_bias,b_features_4_block_0_1_running_mean,b_features_4_block_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g22 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n95 {pt2=root[39] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t340 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_27}] =
        permute x=t821 {derived} perm=[H<-C, W<-H, C<-W]
    n96 {pt2=root[40] torch.ops.aten.add.Tensor}: [t341 f32 [H=96 W=28 C=28] {pt2=root:add_3}] =
      add_scalar x=t340 {pt2=root:getitem_27} scalar=3
    n97 {pt2=root[41] torch.ops.aten.clamp.default}: [t342 f32 [H=96 W=28 C=28] {pt2=root:clamp_4}] =
      clamp x=t341 {pt2=root:add_3} params={min=0; max=none}
    n98 {pt2=root[42] torch.ops.aten.clamp.default}: [t343 f32 [H=96 W=28 C=28] {pt2=root:clamp_5}] =
      clamp x=t342 {pt2=root:clamp_4} params={min=none; max=6}
    n99 {pt2=root[43] torch.ops.aten.mul.Tensor}: [t344 f32 [H=96 W=28 C=28] {pt2=root:mul_2}] =
      mul a=t340 {pt2=root:getitem_27} b=t343 {pt2=root:clamp_5}
    n100 {pt2=root[44] torch.ops.aten.div.Tensor}: [t345 f32 [H=96 W=28 C=28] {pt2=root:div_2}] =
      div_scalar x=t344 {pt2=root:mul_2} scalar=6
    group g23 torch.ops.aten.convolution.default:
      n101 {derived}: [t346 f32 [H=28 W=28 C=96] {derived}] =
        permute x=t345 {pt2=root:div_2} perm=[H<-W, W<-C, C<-H]
    n511 {derived}: [t822 f32 [H=14 W=14 C=96] {derived}] =
      convolution
        x=t346 {derived}
        weight=t753 {folded from=[p_features_4_block_1_0_weight,p_features_4_block_1_1_weight,b_features_4_block_1_1_running_var]}
        bias=t754 {folded from=[p_features_4_block_1_1_weight,p_features_4_block_1_1_bias,b_features_4_block_1_1_running_mean,b_features_4_block_1_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=2; w=2};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=96}
    group g24 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n107 {pt2=root[46] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t352 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_30}] =
        permute x=t822 {derived} perm=[H<-C, W<-H, C<-W]
    n108 {pt2=root[47] torch.ops.aten.add.Tensor}: [t353 f32 [H=96 W=14 C=14] {pt2=root:add_4}] =
      add_scalar x=t352 {pt2=root:getitem_30} scalar=3
    n109 {pt2=root[48] torch.ops.aten.clamp.default}: [t354 f32 [H=96 W=14
                                                                 C=14] {pt2=root:clamp_6}] =
      clamp x=t353 {pt2=root:add_4} params={min=0; max=none}
    n110 {pt2=root[49] torch.ops.aten.clamp.default}: [t355 f32 [H=96 W=14
                                                                 C=14] {pt2=root:clamp_7}] =
      clamp x=t354 {pt2=root:clamp_6} params={min=none; max=6}
    n111 {pt2=root[50] torch.ops.aten.mul.Tensor}: [t356 f32 [H=96 W=14 C=14] {pt2=root:mul_3}] =
      mul a=t352 {pt2=root:getitem_30} b=t355 {pt2=root:clamp_7}
    n112 {pt2=root[51] torch.ops.aten.div.Tensor}: [t357 f32 [H=96 W=14 C=14] {pt2=root:div_3}] =
      div_scalar x=t356 {pt2=root:mul_3} scalar=6
    n113 {pt2=root[52] torch.ops.aten.mean.dim}: [t358 f32 [H=96 W=1 C=1] {pt2=root:mean_1}] =
      mean x=t357 {pt2=root:div_3} params={dims=[C, W]; keepdim=true}
    group g25 torch.ops.aten.convolution.default:
      n114 {derived}: [t359 f32 [C=96] {derived}] =
        permute x=t358 {pt2=root:mean_1} perm=[H<-W, W<-C, C<-H]
      n116 {derived}: [t361 f32 [C=24] {derived}] =
        convolution
          x=t359 {derived}
          weight=t360 {folded from=[p_features_4_block_2_fc1_weight]}
          bias=t38 {pt2=root:p_features_4_block_2_fc1_bias target=features.4.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n512 {pt2=root[54] torch.ops.aten.relu.default}: [t823 f32 [C=24] {derived}] =
      relu x=t361 {derived}
    group g26 torch.ops.aten.convolution.default:
      n121 {derived}: [t366 f32 [C=96] {derived}] =
        convolution
          x=t823 {derived}
          weight=t365 {folded from=[p_features_4_block_2_fc2_weight]}
          bias=t40 {pt2=root:p_features_4_block_2_fc2_bias target=features.4.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n513 {pt2=root[56] torch.ops.aten.add.Tensor}: [t824 f32 [C=96] {derived}] =
      add_scalar x=t366 {derived} scalar=3
    n514 {pt2=root[57] torch.ops.aten.clamp.default}: [t825 f32 [C=96] {derived}] =
      clamp x=t824 {derived} params={min=0; max=none}
    n515 {pt2=root[58] torch.ops.aten.clamp.default}: [t826 f32 [C=96] {derived}] =
      clamp x=t825 {derived} params={min=none; max=6}
    n516 {pt2=root[59] torch.ops.aten.div.Tensor}: [t827 f32 [C=96] {derived}] =
      div_scalar x=t826 {derived} scalar=6
    n517 {pt2=root[55] torch.ops.aten.convolution.default}: [t371 f32 [H=96 W=1
                                                                      C=1] {pt2=root:div_4}] =
      permute x=t827 {derived} perm=[H<-C, W<-H, C<-W]
    n127 {pt2=root[60] torch.ops.aten.mul.Tensor}: [t372 f32 [H=96 W=14 C=14] {pt2=root:mul_4}] =
      mul a=t371 {pt2=root:div_4} b=t357 {pt2=root:div_3}
    group g27 torch.ops.aten.convolution.default:
      n128 {derived}: [t373 f32 [H=14 W=14 C=96] {derived}] =
        permute x=t372 {pt2=root:mul_4} perm=[H<-W, W<-C, C<-H]
    n518 {derived}: [t828 f32 [H=14 W=14 C=40] {derived}] =
      convolution
        x=t373 {derived}
        weight=t755 {folded from=[p_features_4_block_3_0_weight,p_features_4_block_3_1_weight,b_features_4_block_3_1_running_var]}
        bias=t756 {folded from=[p_features_4_block_3_1_weight,p_features_4_block_3_1_bias,b_features_4_block_3_1_running_mean,b_features_4_block_3_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n519 {derived}: [t829 f32 [H=14 W=14 C=240] {derived}] =
      convolution
        x=t828 {derived}
        weight=t757 {folded from=[p_features_5_block_0_0_weight,p_features_5_block_0_1_weight,b_features_5_block_0_1_running_var]}
        bias=t758 {folded from=[p_features_5_block_0_1_weight,p_features_5_block_0_1_bias,b_features_5_block_0_1_running_mean,b_features_5_block_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g30 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n141 {pt2=root[64] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t386 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_36}] =
        permute x=t829 {derived} perm=[H<-C, W<-H, C<-W]
    n142 {pt2=root[65] torch.ops.aten.add.Tensor}: [t387 f32 [H=240 W=14 C=14] {pt2=root:add_6}] =
      add_scalar x=t386 {pt2=root:getitem_36} scalar=3
    n143 {pt2=root[66] torch.ops.aten.clamp.default}: [t388 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_10}] =
      clamp x=t387 {pt2=root:add_6} params={min=0; max=none}
    n144 {pt2=root[67] torch.ops.aten.clamp.default}: [t389 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_11}] =
      clamp x=t388 {pt2=root:clamp_10} params={min=none; max=6}
    n145 {pt2=root[68] torch.ops.aten.mul.Tensor}: [t390 f32 [H=240 W=14 C=14] {pt2=root:mul_5}] =
      mul a=t386 {pt2=root:getitem_36} b=t389 {pt2=root:clamp_11}
    n146 {pt2=root[69] torch.ops.aten.div.Tensor}: [t391 f32 [H=240 W=14 C=14] {pt2=root:div_5}] =
      div_scalar x=t390 {pt2=root:mul_5} scalar=6
    group g31 torch.ops.aten.convolution.default:
      n147 {derived}: [t392 f32 [H=14 W=14 C=240] {derived}] =
        permute x=t391 {pt2=root:div_5} perm=[H<-W, W<-C, C<-H]
    n520 {derived}: [t830 f32 [H=14 W=14 C=240] {derived}] =
      convolution
        x=t392 {derived}
        weight=t759 {folded from=[p_features_5_block_1_0_weight,p_features_5_block_1_1_weight,b_features_5_block_1_1_running_var]}
        bias=t760 {folded from=[p_features_5_block_1_1_weight,p_features_5_block_1_1_bias,b_features_5_block_1_1_running_mean,b_features_5_block_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=2; w=2};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=240}
    group g32 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n153 {pt2=root[71] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t398 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_39}] =
        permute x=t830 {derived} perm=[H<-C, W<-H, C<-W]
    n154 {pt2=root[72] torch.ops.aten.add.Tensor}: [t399 f32 [H=240 W=14 C=14] {pt2=root:add_7}] =
      add_scalar x=t398 {pt2=root:getitem_39} scalar=3
    n155 {pt2=root[73] torch.ops.aten.clamp.default}: [t400 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_12}] =
      clamp x=t399 {pt2=root:add_7} params={min=0; max=none}
    n156 {pt2=root[74] torch.ops.aten.clamp.default}: [t401 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_13}] =
      clamp x=t400 {pt2=root:clamp_12} params={min=none; max=6}
    n157 {pt2=root[75] torch.ops.aten.mul.Tensor}: [t402 f32 [H=240 W=14 C=14] {pt2=root:mul_6}] =
      mul a=t398 {pt2=root:getitem_39} b=t401 {pt2=root:clamp_13}
    n158 {pt2=root[76] torch.ops.aten.div.Tensor}: [t403 f32 [H=240 W=14 C=14] {pt2=root:div_6}] =
      div_scalar x=t402 {pt2=root:mul_6} scalar=6
    n159 {pt2=root[77] torch.ops.aten.mean.dim}: [t404 f32 [H=240 W=1 C=1] {pt2=root:mean_2}] =
      mean x=t403 {pt2=root:div_6} params={dims=[C, W]; keepdim=true}
    group g33 torch.ops.aten.convolution.default:
      n160 {derived}: [t405 f32 [C=240] {derived}] =
        permute x=t404 {pt2=root:mean_2} perm=[H<-W, W<-C, C<-H]
      n162 {derived}: [t407 f32 [C=64] {derived}] =
        convolution
          x=t405 {derived}
          weight=t406 {folded from=[p_features_5_block_2_fc1_weight]}
          bias=t51 {pt2=root:p_features_5_block_2_fc1_bias target=features.5.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n521 {pt2=root[79] torch.ops.aten.relu.default}: [t831 f32 [C=64] {derived}] =
      relu x=t407 {derived}
    group g34 torch.ops.aten.convolution.default:
      n167 {derived}: [t412 f32 [C=240] {derived}] =
        convolution
          x=t831 {derived}
          weight=t411 {folded from=[p_features_5_block_2_fc2_weight]}
          bias=t53 {pt2=root:p_features_5_block_2_fc2_bias target=features.5.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n522 {pt2=root[81] torch.ops.aten.add.Tensor}: [t832 f32 [C=240] {derived}] =
      add_scalar x=t412 {derived} scalar=3
    n523 {pt2=root[82] torch.ops.aten.clamp.default}: [t833 f32 [C=240] {derived}] =
      clamp x=t832 {derived} params={min=0; max=none}
    n524 {pt2=root[83] torch.ops.aten.clamp.default}: [t834 f32 [C=240] {derived}] =
      clamp x=t833 {derived} params={min=none; max=6}
    n525 {pt2=root[84] torch.ops.aten.div.Tensor}: [t835 f32 [C=240] {derived}] =
      div_scalar x=t834 {derived} scalar=6
    n526 {pt2=root[80] torch.ops.aten.convolution.default}: [t417 f32 [H=240
                                                                      W=1 C=1] {pt2=root:div_7}] =
      permute x=t835 {derived} perm=[H<-C, W<-H, C<-W]
    n173 {pt2=root[85] torch.ops.aten.mul.Tensor}: [t418 f32 [H=240 W=14 C=14] {pt2=root:mul_7}] =
      mul a=t417 {pt2=root:div_7} b=t403 {pt2=root:div_6}
    group g35 torch.ops.aten.convolution.default:
      n174 {derived}: [t419 f32 [H=14 W=14 C=240] {derived}] =
        permute x=t418 {pt2=root:mul_7} perm=[H<-W, W<-C, C<-H]
    n527 {derived}: [t836 f32 [H=14 W=14 C=40] {derived}] =
      convolution
        x=t419 {derived}
        weight=t761 {folded from=[p_features_5_block_3_0_weight,p_features_5_block_3_1_weight,b_features_5_block_3_1_running_var]}
        bias=t762 {folded from=[p_features_5_block_3_1_weight,p_features_5_block_3_1_bias,b_features_5_block_3_1_running_mean,b_features_5_block_3_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n528 {pt2=root[88] torch.ops.aten.add.Tensor}: [t837 f32 [H=14 W=14 C=40] {derived}] =
      add a=t836 {derived} b=t828 {derived}
    n529 {derived}: [t838 f32 [H=14 W=14 C=240] {derived}] =
      convolution
        x=t837 {derived}
        weight=t763 {folded from=[p_features_6_block_0_0_weight,p_features_6_block_0_1_weight,b_features_6_block_0_1_running_var]}
        bias=t764 {folded from=[p_features_6_block_0_1_weight,p_features_6_block_0_1_bias,b_features_6_block_0_1_running_mean,b_features_6_block_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g38 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n188 {pt2=root[90] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t433 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_45}] =
        permute x=t838 {derived} perm=[H<-C, W<-H, C<-W]
    n189 {pt2=root[91] torch.ops.aten.add.Tensor}: [t434 f32 [H=240 W=14 C=14] {pt2=root:add_10}] =
      add_scalar x=t433 {pt2=root:getitem_45} scalar=3
    n190 {pt2=root[92] torch.ops.aten.clamp.default}: [t435 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_16}] =
      clamp x=t434 {pt2=root:add_10} params={min=0; max=none}
    n191 {pt2=root[93] torch.ops.aten.clamp.default}: [t436 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_17}] =
      clamp x=t435 {pt2=root:clamp_16} params={min=none; max=6}
    n192 {pt2=root[94] torch.ops.aten.mul.Tensor}: [t437 f32 [H=240 W=14 C=14] {pt2=root:mul_8}] =
      mul a=t433 {pt2=root:getitem_45} b=t436 {pt2=root:clamp_17}
    n193 {pt2=root[95] torch.ops.aten.div.Tensor}: [t438 f32 [H=240 W=14 C=14] {pt2=root:div_8}] =
      div_scalar x=t437 {pt2=root:mul_8} scalar=6
    group g39 torch.ops.aten.convolution.default:
      n194 {derived}: [t439 f32 [H=14 W=14 C=240] {derived}] =
        permute x=t438 {pt2=root:div_8} perm=[H<-W, W<-C, C<-H]
    n530 {derived}: [t839 f32 [H=14 W=14 C=240] {derived}] =
      convolution
        x=t439 {derived}
        weight=t765 {folded from=[p_features_6_block_1_0_weight,p_features_6_block_1_1_weight,b_features_6_block_1_1_running_var]}
        bias=t766 {folded from=[p_features_6_block_1_1_weight,p_features_6_block_1_1_bias,b_features_6_block_1_1_running_mean,b_features_6_block_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=2; w=2};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=240}
    group g40 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n200 {pt2=root[97] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t445 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_48}] =
        permute x=t839 {derived} perm=[H<-C, W<-H, C<-W]
    n201 {pt2=root[98] torch.ops.aten.add.Tensor}: [t446 f32 [H=240 W=14 C=14] {pt2=root:add_11}] =
      add_scalar x=t445 {pt2=root:getitem_48} scalar=3
    n202 {pt2=root[99] torch.ops.aten.clamp.default}: [t447 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_18}] =
      clamp x=t446 {pt2=root:add_11} params={min=0; max=none}
    n203 {pt2=root[100] torch.ops.aten.clamp.default}: [t448 f32 [H=240 W=14
                                                                  C=14] {pt2=root:clamp_19}] =
      clamp x=t447 {pt2=root:clamp_18} params={min=none; max=6}
    n204 {pt2=root[101] torch.ops.aten.mul.Tensor}: [t449 f32 [H=240 W=14 C=14] {pt2=root:mul_9}] =
      mul a=t445 {pt2=root:getitem_48} b=t448 {pt2=root:clamp_19}
    n205 {pt2=root[102] torch.ops.aten.div.Tensor}: [t450 f32 [H=240 W=14 C=14] {pt2=root:div_9}] =
      div_scalar x=t449 {pt2=root:mul_9} scalar=6
    n206 {pt2=root[103] torch.ops.aten.mean.dim}: [t451 f32 [H=240 W=1 C=1] {pt2=root:mean_3}] =
      mean x=t450 {pt2=root:div_9} params={dims=[C, W]; keepdim=true}
    group g41 torch.ops.aten.convolution.default:
      n207 {derived}: [t452 f32 [C=240] {derived}] =
        permute x=t451 {pt2=root:mean_3} perm=[H<-W, W<-C, C<-H]
      n209 {derived}: [t454 f32 [C=64] {derived}] =
        convolution
          x=t452 {derived}
          weight=t453 {folded from=[p_features_6_block_2_fc1_weight]}
          bias=t64 {pt2=root:p_features_6_block_2_fc1_bias target=features.6.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n531 {pt2=root[105] torch.ops.aten.relu.default}: [t840 f32 [C=64] {derived}] =
      relu x=t454 {derived}
    group g42 torch.ops.aten.convolution.default:
      n214 {derived}: [t459 f32 [C=240] {derived}] =
        convolution
          x=t840 {derived}
          weight=t458 {folded from=[p_features_6_block_2_fc2_weight]}
          bias=t66 {pt2=root:p_features_6_block_2_fc2_bias target=features.6.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n532 {pt2=root[107] torch.ops.aten.add.Tensor}: [t841 f32 [C=240] {derived}] =
      add_scalar x=t459 {derived} scalar=3
    n533 {pt2=root[108] torch.ops.aten.clamp.default}: [t842 f32 [C=240] {derived}] =
      clamp x=t841 {derived} params={min=0; max=none}
    n534 {pt2=root[109] torch.ops.aten.clamp.default}: [t843 f32 [C=240] {derived}] =
      clamp x=t842 {derived} params={min=none; max=6}
    n535 {pt2=root[110] torch.ops.aten.div.Tensor}: [t844 f32 [C=240] {derived}] =
      div_scalar x=t843 {derived} scalar=6
    n536 {pt2=root[106] torch.ops.aten.convolution.default}: [t464 f32 [H=240
                                                                      W=1 C=1] {pt2=root:div_10}] =
      permute x=t844 {derived} perm=[H<-C, W<-H, C<-W]
    n220 {pt2=root[111] torch.ops.aten.mul.Tensor}: [t465 f32 [H=240 W=14 C=14] {pt2=root:mul_10}] =
      mul a=t464 {pt2=root:div_10} b=t450 {pt2=root:div_9}
    group g43 torch.ops.aten.convolution.default:
      n221 {derived}: [t466 f32 [H=14 W=14 C=240] {derived}] =
        permute x=t465 {pt2=root:mul_10} perm=[H<-W, W<-C, C<-H]
    n537 {derived}: [t845 f32 [H=14 W=14 C=40] {derived}] =
      convolution
        x=t466 {derived}
        weight=t767 {folded from=[p_features_6_block_3_0_weight,p_features_6_block_3_1_weight,b_features_6_block_3_1_running_var]}
        bias=t768 {folded from=[p_features_6_block_3_1_weight,p_features_6_block_3_1_bias,b_features_6_block_3_1_running_mean,b_features_6_block_3_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n538 {pt2=root[114] torch.ops.aten.add.Tensor}: [t846 f32 [H=14 W=14 C=40] {derived}] =
      add a=t845 {derived} b=t837 {derived}
    n539 {derived}: [t847 f32 [H=14 W=14 C=120] {derived}] =
      convolution
        x=t846 {derived}
        weight=t769 {folded from=[p_features_7_block_0_0_weight,p_features_7_block_0_1_weight,b_features_7_block_0_1_running_var]}
        bias=t770 {folded from=[p_features_7_block_0_1_weight,p_features_7_block_0_1_bias,b_features_7_block_0_1_running_mean,b_features_7_block_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g46 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n235 {pt2=root[116] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t480 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_54}] =
        permute x=t847 {derived} perm=[H<-C, W<-H, C<-W]
    n236 {pt2=root[117] torch.ops.aten.add.Tensor}: [t481 f32 [H=120 W=14 C=14] {pt2=root:add_14}] =
      add_scalar x=t480 {pt2=root:getitem_54} scalar=3
    n237 {pt2=root[118] torch.ops.aten.clamp.default}: [t482 f32 [H=120 W=14
                                                                  C=14] {pt2=root:clamp_22}] =
      clamp x=t481 {pt2=root:add_14} params={min=0; max=none}
    n238 {pt2=root[119] torch.ops.aten.clamp.default}: [t483 f32 [H=120 W=14
                                                                  C=14] {pt2=root:clamp_23}] =
      clamp x=t482 {pt2=root:clamp_22} params={min=none; max=6}
    n239 {pt2=root[120] torch.ops.aten.mul.Tensor}: [t484 f32 [H=120 W=14 C=14] {pt2=root:mul_11}] =
      mul a=t480 {pt2=root:getitem_54} b=t483 {pt2=root:clamp_23}
    n240 {pt2=root[121] torch.ops.aten.div.Tensor}: [t485 f32 [H=120 W=14 C=14] {pt2=root:div_11}] =
      div_scalar x=t484 {pt2=root:mul_11} scalar=6
    group g47 torch.ops.aten.convolution.default:
      n241 {derived}: [t486 f32 [H=14 W=14 C=120] {derived}] =
        permute x=t485 {pt2=root:div_11} perm=[H<-W, W<-C, C<-H]
    n540 {derived}: [t848 f32 [H=14 W=14 C=120] {derived}] =
      convolution
        x=t486 {derived}
        weight=t771 {folded from=[p_features_7_block_1_0_weight,p_features_7_block_1_1_weight,b_features_7_block_1_1_running_var]}
        bias=t772 {folded from=[p_features_7_block_1_1_weight,p_features_7_block_1_1_bias,b_features_7_block_1_1_running_mean,b_features_7_block_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=2; w=2};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=120}
    group g48 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n247 {pt2=root[123] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t492 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_57}] =
        permute x=t848 {derived} perm=[H<-C, W<-H, C<-W]
    n248 {pt2=root[124] torch.ops.aten.add.Tensor}: [t493 f32 [H=120 W=14 C=14] {pt2=root:add_15}] =
      add_scalar x=t492 {pt2=root:getitem_57} scalar=3
    n249 {pt2=root[125] torch.ops.aten.clamp.default}: [t494 f32 [H=120 W=14
                                                                  C=14] {pt2=root:clamp_24}] =
      clamp x=t493 {pt2=root:add_15} params={min=0; max=none}
    n250 {pt2=root[126] torch.ops.aten.clamp.default}: [t495 f32 [H=120 W=14
                                                                  C=14] {pt2=root:clamp_25}] =
      clamp x=t494 {pt2=root:clamp_24} params={min=none; max=6}
    n251 {pt2=root[127] torch.ops.aten.mul.Tensor}: [t496 f32 [H=120 W=14 C=14] {pt2=root:mul_12}] =
      mul a=t492 {pt2=root:getitem_57} b=t495 {pt2=root:clamp_25}
    n252 {pt2=root[128] torch.ops.aten.div.Tensor}: [t497 f32 [H=120 W=14 C=14] {pt2=root:div_12}] =
      div_scalar x=t496 {pt2=root:mul_12} scalar=6
    n253 {pt2=root[129] torch.ops.aten.mean.dim}: [t498 f32 [H=120 W=1 C=1] {pt2=root:mean_4}] =
      mean x=t497 {pt2=root:div_12} params={dims=[C, W]; keepdim=true}
    group g49 torch.ops.aten.convolution.default:
      n254 {derived}: [t499 f32 [C=120] {derived}] =
        permute x=t498 {pt2=root:mean_4} perm=[H<-W, W<-C, C<-H]
      n256 {derived}: [t501 f32 [C=32] {derived}] =
        convolution
          x=t499 {derived}
          weight=t500 {folded from=[p_features_7_block_2_fc1_weight]}
          bias=t77 {pt2=root:p_features_7_block_2_fc1_bias target=features.7.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n541 {pt2=root[131] torch.ops.aten.relu.default}: [t849 f32 [C=32] {derived}] =
      relu x=t501 {derived}
    group g50 torch.ops.aten.convolution.default:
      n261 {derived}: [t506 f32 [C=120] {derived}] =
        convolution
          x=t849 {derived}
          weight=t505 {folded from=[p_features_7_block_2_fc2_weight]}
          bias=t79 {pt2=root:p_features_7_block_2_fc2_bias target=features.7.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n542 {pt2=root[133] torch.ops.aten.add.Tensor}: [t850 f32 [C=120] {derived}] =
      add_scalar x=t506 {derived} scalar=3
    n543 {pt2=root[134] torch.ops.aten.clamp.default}: [t851 f32 [C=120] {derived}] =
      clamp x=t850 {derived} params={min=0; max=none}
    n544 {pt2=root[135] torch.ops.aten.clamp.default}: [t852 f32 [C=120] {derived}] =
      clamp x=t851 {derived} params={min=none; max=6}
    n545 {pt2=root[136] torch.ops.aten.div.Tensor}: [t853 f32 [C=120] {derived}] =
      div_scalar x=t852 {derived} scalar=6
    n546 {pt2=root[132] torch.ops.aten.convolution.default}: [t511 f32 [H=120
                                                                      W=1 C=1] {pt2=root:div_13}] =
      permute x=t853 {derived} perm=[H<-C, W<-H, C<-W]
    n267 {pt2=root[137] torch.ops.aten.mul.Tensor}: [t512 f32 [H=120 W=14 C=14] {pt2=root:mul_13}] =
      mul a=t511 {pt2=root:div_13} b=t497 {pt2=root:div_12}
    group g51 torch.ops.aten.convolution.default:
      n268 {derived}: [t513 f32 [H=14 W=14 C=120] {derived}] =
        permute x=t512 {pt2=root:mul_13} perm=[H<-W, W<-C, C<-H]
    n547 {derived}: [t854 f32 [H=14 W=14 C=48] {derived}] =
      convolution
        x=t513 {derived}
        weight=t773 {folded from=[p_features_7_block_3_0_weight,p_features_7_block_3_1_weight,b_features_7_block_3_1_running_var]}
        bias=t774 {folded from=[p_features_7_block_3_1_weight,p_features_7_block_3_1_bias,b_features_7_block_3_1_running_mean,b_features_7_block_3_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n548 {derived}: [t855 f32 [H=14 W=14 C=144] {derived}] =
      convolution
        x=t854 {derived}
        weight=t775 {folded from=[p_features_8_block_0_0_weight,p_features_8_block_0_1_weight,b_features_8_block_0_1_running_var]}
        bias=t776 {folded from=[p_features_8_block_0_1_weight,p_features_8_block_0_1_bias,b_features_8_block_0_1_running_mean,b_features_8_block_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g54 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n281 {pt2=root[141] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t526 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_63}] =
        permute x=t855 {derived} perm=[H<-C, W<-H, C<-W]
    n282 {pt2=root[142] torch.ops.aten.add.Tensor}: [t527 f32 [H=144 W=14 C=14] {pt2=root:add_17}] =
      add_scalar x=t526 {pt2=root:getitem_63} scalar=3
    n283 {pt2=root[143] torch.ops.aten.clamp.default}: [t528 f32 [H=144 W=14
                                                                  C=14] {pt2=root:clamp_28}] =
      clamp x=t527 {pt2=root:add_17} params={min=0; max=none}
    n284 {pt2=root[144] torch.ops.aten.clamp.default}: [t529 f32 [H=144 W=14
                                                                  C=14] {pt2=root:clamp_29}] =
      clamp x=t528 {pt2=root:clamp_28} params={min=none; max=6}
    n285 {pt2=root[145] torch.ops.aten.mul.Tensor}: [t530 f32 [H=144 W=14 C=14] {pt2=root:mul_14}] =
      mul a=t526 {pt2=root:getitem_63} b=t529 {pt2=root:clamp_29}
    n286 {pt2=root[146] torch.ops.aten.div.Tensor}: [t531 f32 [H=144 W=14 C=14] {pt2=root:div_14}] =
      div_scalar x=t530 {pt2=root:mul_14} scalar=6
    group g55 torch.ops.aten.convolution.default:
      n287 {derived}: [t532 f32 [H=14 W=14 C=144] {derived}] =
        permute x=t531 {pt2=root:div_14} perm=[H<-W, W<-C, C<-H]
    n549 {derived}: [t856 f32 [H=14 W=14 C=144] {derived}] =
      convolution
        x=t532 {derived}
        weight=t777 {folded from=[p_features_8_block_1_0_weight,p_features_8_block_1_1_weight,b_features_8_block_1_1_running_var]}
        bias=t778 {folded from=[p_features_8_block_1_1_weight,p_features_8_block_1_1_bias,b_features_8_block_1_1_running_mean,b_features_8_block_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=2; w=2};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=144}
    group g56 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n293 {pt2=root[148] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t538 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_66}] =
        permute x=t856 {derived} perm=[H<-C, W<-H, C<-W]
    n294 {pt2=root[149] torch.ops.aten.add.Tensor}: [t539 f32 [H=144 W=14 C=14] {pt2=root:add_18}] =
      add_scalar x=t538 {pt2=root:getitem_66} scalar=3
    n295 {pt2=root[150] torch.ops.aten.clamp.default}: [t540 f32 [H=144 W=14
                                                                  C=14] {pt2=root:clamp_30}] =
      clamp x=t539 {pt2=root:add_18} params={min=0; max=none}
    n296 {pt2=root[151] torch.ops.aten.clamp.default}: [t541 f32 [H=144 W=14
                                                                  C=14] {pt2=root:clamp_31}] =
      clamp x=t540 {pt2=root:clamp_30} params={min=none; max=6}
    n297 {pt2=root[152] torch.ops.aten.mul.Tensor}: [t542 f32 [H=144 W=14 C=14] {pt2=root:mul_15}] =
      mul a=t538 {pt2=root:getitem_66} b=t541 {pt2=root:clamp_31}
    n298 {pt2=root[153] torch.ops.aten.div.Tensor}: [t543 f32 [H=144 W=14 C=14] {pt2=root:div_15}] =
      div_scalar x=t542 {pt2=root:mul_15} scalar=6
    n299 {pt2=root[154] torch.ops.aten.mean.dim}: [t544 f32 [H=144 W=1 C=1] {pt2=root:mean_5}] =
      mean x=t543 {pt2=root:div_15} params={dims=[C, W]; keepdim=true}
    group g57 torch.ops.aten.convolution.default:
      n300 {derived}: [t545 f32 [C=144] {derived}] =
        permute x=t544 {pt2=root:mean_5} perm=[H<-W, W<-C, C<-H]
      n302 {derived}: [t547 f32 [C=40] {derived}] =
        convolution
          x=t545 {derived}
          weight=t546 {folded from=[p_features_8_block_2_fc1_weight]}
          bias=t90 {pt2=root:p_features_8_block_2_fc1_bias target=features.8.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n550 {pt2=root[156] torch.ops.aten.relu.default}: [t857 f32 [C=40] {derived}] =
      relu x=t547 {derived}
    group g58 torch.ops.aten.convolution.default:
      n307 {derived}: [t552 f32 [C=144] {derived}] =
        convolution
          x=t857 {derived}
          weight=t551 {folded from=[p_features_8_block_2_fc2_weight]}
          bias=t92 {pt2=root:p_features_8_block_2_fc2_bias target=features.8.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n551 {pt2=root[158] torch.ops.aten.add.Tensor}: [t858 f32 [C=144] {derived}] =
      add_scalar x=t552 {derived} scalar=3
    n552 {pt2=root[159] torch.ops.aten.clamp.default}: [t859 f32 [C=144] {derived}] =
      clamp x=t858 {derived} params={min=0; max=none}
    n553 {pt2=root[160] torch.ops.aten.clamp.default}: [t860 f32 [C=144] {derived}] =
      clamp x=t859 {derived} params={min=none; max=6}
    n554 {pt2=root[161] torch.ops.aten.div.Tensor}: [t861 f32 [C=144] {derived}] =
      div_scalar x=t860 {derived} scalar=6
    n555 {pt2=root[157] torch.ops.aten.convolution.default}: [t557 f32 [H=144
                                                                      W=1 C=1] {pt2=root:div_16}] =
      permute x=t861 {derived} perm=[H<-C, W<-H, C<-W]
    n313 {pt2=root[162] torch.ops.aten.mul.Tensor}: [t558 f32 [H=144 W=14 C=14] {pt2=root:mul_16}] =
      mul a=t557 {pt2=root:div_16} b=t543 {pt2=root:div_15}
    group g59 torch.ops.aten.convolution.default:
      n314 {derived}: [t559 f32 [H=14 W=14 C=144] {derived}] =
        permute x=t558 {pt2=root:mul_16} perm=[H<-W, W<-C, C<-H]
    n556 {derived}: [t862 f32 [H=14 W=14 C=48] {derived}] =
      convolution
        x=t559 {derived}
        weight=t779 {folded from=[p_features_8_block_3_0_weight,p_features_8_block_3_1_weight,b_features_8_block_3_1_running_var]}
        bias=t780 {folded from=[p_features_8_block_3_1_weight,p_features_8_block_3_1_bias,b_features_8_block_3_1_running_mean,b_features_8_block_3_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n557 {pt2=root[165] torch.ops.aten.add.Tensor}: [t863 f32 [H=14 W=14 C=48] {derived}] =
      add a=t862 {derived} b=t854 {derived}
    n558 {derived}: [t864 f32 [H=14 W=14 C=288] {derived}] =
      convolution
        x=t863 {derived}
        weight=t781 {folded from=[p_features_9_block_0_0_weight,p_features_9_block_0_1_weight,b_features_9_block_0_1_running_var]}
        bias=t782 {folded from=[p_features_9_block_0_1_weight,p_features_9_block_0_1_bias,b_features_9_block_0_1_running_mean,b_features_9_block_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g62 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n328 {pt2=root[167] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t573 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_72}] =
        permute x=t864 {derived} perm=[H<-C, W<-H, C<-W]
    n329 {pt2=root[168] torch.ops.aten.add.Tensor}: [t574 f32 [H=288 W=14 C=14] {pt2=root:add_21}] =
      add_scalar x=t573 {pt2=root:getitem_72} scalar=3
    n330 {pt2=root[169] torch.ops.aten.clamp.default}: [t575 f32 [H=288 W=14
                                                                  C=14] {pt2=root:clamp_34}] =
      clamp x=t574 {pt2=root:add_21} params={min=0; max=none}
    n331 {pt2=root[170] torch.ops.aten.clamp.default}: [t576 f32 [H=288 W=14
                                                                  C=14] {pt2=root:clamp_35}] =
      clamp x=t575 {pt2=root:clamp_34} params={min=none; max=6}
    n332 {pt2=root[171] torch.ops.aten.mul.Tensor}: [t577 f32 [H=288 W=14 C=14] {pt2=root:mul_17}] =
      mul a=t573 {pt2=root:getitem_72} b=t576 {pt2=root:clamp_35}
    n333 {pt2=root[172] torch.ops.aten.div.Tensor}: [t578 f32 [H=288 W=14 C=14] {pt2=root:div_17}] =
      div_scalar x=t577 {pt2=root:mul_17} scalar=6
    group g63 torch.ops.aten.convolution.default:
      n334 {derived}: [t579 f32 [H=14 W=14 C=288] {derived}] =
        permute x=t578 {pt2=root:div_17} perm=[H<-W, W<-C, C<-H]
    n559 {derived}: [t865 f32 [H=7 W=7 C=288] {derived}] =
      convolution
        x=t579 {derived}
        weight=t783 {folded from=[p_features_9_block_1_0_weight,p_features_9_block_1_1_weight,b_features_9_block_1_1_running_var]}
        bias=t784 {folded from=[p_features_9_block_1_1_weight,p_features_9_block_1_1_bias,b_features_9_block_1_1_running_mean,b_features_9_block_1_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=2; w=2};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=288}
    group g64 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n340 {pt2=root[174] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t585 f32 [H=288
                                                                      W=7 C=7] {pt2=root:getitem_75}] =
        permute x=t865 {derived} perm=[H<-C, W<-H, C<-W]
    n341 {pt2=root[175] torch.ops.aten.add.Tensor}: [t586 f32 [H=288 W=7 C=7] {pt2=root:add_22}] =
      add_scalar x=t585 {pt2=root:getitem_75} scalar=3
    n342 {pt2=root[176] torch.ops.aten.clamp.default}: [t587 f32 [H=288 W=7
                                                                  C=7] {pt2=root:clamp_36}] =
      clamp x=t586 {pt2=root:add_22} params={min=0; max=none}
    n343 {pt2=root[177] torch.ops.aten.clamp.default}: [t588 f32 [H=288 W=7
                                                                  C=7] {pt2=root:clamp_37}] =
      clamp x=t587 {pt2=root:clamp_36} params={min=none; max=6}
    n344 {pt2=root[178] torch.ops.aten.mul.Tensor}: [t589 f32 [H=288 W=7 C=7] {pt2=root:mul_18}] =
      mul a=t585 {pt2=root:getitem_75} b=t588 {pt2=root:clamp_37}
    n345 {pt2=root[179] torch.ops.aten.div.Tensor}: [t590 f32 [H=288 W=7 C=7] {pt2=root:div_18}] =
      div_scalar x=t589 {pt2=root:mul_18} scalar=6
    n346 {pt2=root[180] torch.ops.aten.mean.dim}: [t591 f32 [H=288 W=1 C=1] {pt2=root:mean_6}] =
      mean x=t590 {pt2=root:div_18} params={dims=[C, W]; keepdim=true}
    group g65 torch.ops.aten.convolution.default:
      n347 {derived}: [t592 f32 [C=288] {derived}] =
        permute x=t591 {pt2=root:mean_6} perm=[H<-W, W<-C, C<-H]
      n349 {derived}: [t594 f32 [C=72] {derived}] =
        convolution
          x=t592 {derived}
          weight=t593 {folded from=[p_features_9_block_2_fc1_weight]}
          bias=t103 {pt2=root:p_features_9_block_2_fc1_bias target=features.9.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n560 {pt2=root[182] torch.ops.aten.relu.default}: [t866 f32 [C=72] {derived}] =
      relu x=t594 {derived}
    group g66 torch.ops.aten.convolution.default:
      n354 {derived}: [t599 f32 [C=288] {derived}] =
        convolution
          x=t866 {derived}
          weight=t598 {folded from=[p_features_9_block_2_fc2_weight]}
          bias=t105 {pt2=root:p_features_9_block_2_fc2_bias target=features.9.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n561 {pt2=root[184] torch.ops.aten.add.Tensor}: [t867 f32 [C=288] {derived}] =
      add_scalar x=t599 {derived} scalar=3
    n562 {pt2=root[185] torch.ops.aten.clamp.default}: [t868 f32 [C=288] {derived}] =
      clamp x=t867 {derived} params={min=0; max=none}
    n563 {pt2=root[186] torch.ops.aten.clamp.default}: [t869 f32 [C=288] {derived}] =
      clamp x=t868 {derived} params={min=none; max=6}
    n564 {pt2=root[187] torch.ops.aten.div.Tensor}: [t870 f32 [C=288] {derived}] =
      div_scalar x=t869 {derived} scalar=6
    n565 {pt2=root[183] torch.ops.aten.convolution.default}: [t604 f32 [H=288
                                                                      W=1 C=1] {pt2=root:div_19}] =
      permute x=t870 {derived} perm=[H<-C, W<-H, C<-W]
    n360 {pt2=root[188] torch.ops.aten.mul.Tensor}: [t605 f32 [H=288 W=7 C=7] {pt2=root:mul_19}] =
      mul a=t604 {pt2=root:div_19} b=t590 {pt2=root:div_18}
    group g67 torch.ops.aten.convolution.default:
      n361 {derived}: [t606 f32 [H=7 W=7 C=288] {derived}] =
        permute x=t605 {pt2=root:mul_19} perm=[H<-W, W<-C, C<-H]
    n566 {derived}: [t871 f32 [H=7 W=7 C=96] {derived}] =
      convolution
        x=t606 {derived}
        weight=t785 {folded from=[p_features_9_block_3_0_weight,p_features_9_block_3_1_weight,b_features_9_block_3_1_running_var]}
        bias=t786 {folded from=[p_features_9_block_3_1_weight,p_features_9_block_3_1_bias,b_features_9_block_3_1_running_mean,b_features_9_block_3_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n567 {derived}: [t872 f32 [H=7 W=7 C=576] {derived}] =
      convolution
        x=t871 {derived}
        weight=t787 {folded from=[p_features_10_block_0_0_weight,p_features_10_block_0_1_weight,b_features_10_block_0_1_running_var]}
        bias=t788 {folded from=[p_features_10_block_0_1_weight,p_features_10_block_0_1_bias,b_features_10_block_0_1_running_mean,b_features_10_block_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g70 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n374 {pt2=root[192] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t619 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_81}] =
        permute x=t872 {derived} perm=[H<-C, W<-H, C<-W]
    n375 {pt2=root[193] torch.ops.aten.add.Tensor}: [t620 f32 [H=576 W=7 C=7] {pt2=root:add_24}] =
      add_scalar x=t619 {pt2=root:getitem_81} scalar=3
    n376 {pt2=root[194] torch.ops.aten.clamp.default}: [t621 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_40}] =
      clamp x=t620 {pt2=root:add_24} params={min=0; max=none}
    n377 {pt2=root[195] torch.ops.aten.clamp.default}: [t622 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_41}] =
      clamp x=t621 {pt2=root:clamp_40} params={min=none; max=6}
    n378 {pt2=root[196] torch.ops.aten.mul.Tensor}: [t623 f32 [H=576 W=7 C=7] {pt2=root:mul_20}] =
      mul a=t619 {pt2=root:getitem_81} b=t622 {pt2=root:clamp_41}
    n379 {pt2=root[197] torch.ops.aten.div.Tensor}: [t624 f32 [H=576 W=7 C=7] {pt2=root:div_20}] =
      div_scalar x=t623 {pt2=root:mul_20} scalar=6
    group g71 torch.ops.aten.convolution.default:
      n380 {derived}: [t625 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t624 {pt2=root:div_20} perm=[H<-W, W<-C, C<-H]
    n568 {derived}: [t873 f32 [H=7 W=7 C=576] {derived}] =
      convolution
        x=t625 {derived}
        weight=t789 {folded from=[p_features_10_block_1_0_weight,p_features_10_block_1_1_weight,b_features_10_block_1_1_running_var]}
        bias=t790 {folded from=[p_features_10_block_1_1_weight,p_features_10_block_1_1_bias,b_features_10_block_1_1_running_mean,b_features_10_block_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=2; w=2};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=576}
    group g72 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n386 {pt2=root[199] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t631 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_84}] =
        permute x=t873 {derived} perm=[H<-C, W<-H, C<-W]
    n387 {pt2=root[200] torch.ops.aten.add.Tensor}: [t632 f32 [H=576 W=7 C=7] {pt2=root:add_25}] =
      add_scalar x=t631 {pt2=root:getitem_84} scalar=3
    n388 {pt2=root[201] torch.ops.aten.clamp.default}: [t633 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_42}] =
      clamp x=t632 {pt2=root:add_25} params={min=0; max=none}
    n389 {pt2=root[202] torch.ops.aten.clamp.default}: [t634 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_43}] =
      clamp x=t633 {pt2=root:clamp_42} params={min=none; max=6}
    n390 {pt2=root[203] torch.ops.aten.mul.Tensor}: [t635 f32 [H=576 W=7 C=7] {pt2=root:mul_21}] =
      mul a=t631 {pt2=root:getitem_84} b=t634 {pt2=root:clamp_43}
    n391 {pt2=root[204] torch.ops.aten.div.Tensor}: [t636 f32 [H=576 W=7 C=7] {pt2=root:div_21}] =
      div_scalar x=t635 {pt2=root:mul_21} scalar=6
    n392 {pt2=root[205] torch.ops.aten.mean.dim}: [t637 f32 [H=576 W=1 C=1] {pt2=root:mean_7}] =
      mean x=t636 {pt2=root:div_21} params={dims=[C, W]; keepdim=true}
    group g73 torch.ops.aten.convolution.default:
      n393 {derived}: [t638 f32 [C=576] {derived}] =
        permute x=t637 {pt2=root:mean_7} perm=[H<-W, W<-C, C<-H]
      n395 {derived}: [t640 f32 [C=144] {derived}] =
        convolution
          x=t638 {derived}
          weight=t639 {folded from=[p_features_10_block_2_fc1_weight]}
          bias=t116 {pt2=root:p_features_10_block_2_fc1_bias target=features.10.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n569 {pt2=root[207] torch.ops.aten.relu.default}: [t874 f32 [C=144] {derived}] =
      relu x=t640 {derived}
    group g74 torch.ops.aten.convolution.default:
      n400 {derived}: [t645 f32 [C=576] {derived}] =
        convolution
          x=t874 {derived}
          weight=t644 {folded from=[p_features_10_block_2_fc2_weight]}
          bias=t118 {pt2=root:p_features_10_block_2_fc2_bias target=features.10.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n570 {pt2=root[209] torch.ops.aten.add.Tensor}: [t875 f32 [C=576] {derived}] =
      add_scalar x=t645 {derived} scalar=3
    n571 {pt2=root[210] torch.ops.aten.clamp.default}: [t876 f32 [C=576] {derived}] =
      clamp x=t875 {derived} params={min=0; max=none}
    n572 {pt2=root[211] torch.ops.aten.clamp.default}: [t877 f32 [C=576] {derived}] =
      clamp x=t876 {derived} params={min=none; max=6}
    n573 {pt2=root[212] torch.ops.aten.div.Tensor}: [t878 f32 [C=576] {derived}] =
      div_scalar x=t877 {derived} scalar=6
    n574 {pt2=root[208] torch.ops.aten.convolution.default}: [t650 f32 [H=576
                                                                      W=1 C=1] {pt2=root:div_22}] =
      permute x=t878 {derived} perm=[H<-C, W<-H, C<-W]
    n406 {pt2=root[213] torch.ops.aten.mul.Tensor}: [t651 f32 [H=576 W=7 C=7] {pt2=root:mul_22}] =
      mul a=t650 {pt2=root:div_22} b=t636 {pt2=root:div_21}
    group g75 torch.ops.aten.convolution.default:
      n407 {derived}: [t652 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t651 {pt2=root:mul_22} perm=[H<-W, W<-C, C<-H]
    n575 {derived}: [t879 f32 [H=7 W=7 C=96] {derived}] =
      convolution
        x=t652 {derived}
        weight=t791 {folded from=[p_features_10_block_3_0_weight,p_features_10_block_3_1_weight,b_features_10_block_3_1_running_var]}
        bias=t792 {folded from=[p_features_10_block_3_1_weight,p_features_10_block_3_1_bias,b_features_10_block_3_1_running_mean,b_features_10_block_3_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n576 {pt2=root[216] torch.ops.aten.add.Tensor}: [t880 f32 [H=7 W=7 C=96] {derived}] =
      add a=t879 {derived} b=t871 {derived}
    n577 {derived}: [t881 f32 [H=7 W=7 C=576] {derived}] =
      convolution
        x=t880 {derived}
        weight=t793 {folded from=[p_features_11_block_0_0_weight,p_features_11_block_0_1_weight,b_features_11_block_0_1_running_var]}
        bias=t794 {folded from=[p_features_11_block_0_1_weight,p_features_11_block_0_1_bias,b_features_11_block_0_1_running_mean,b_features_11_block_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g78 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n421 {pt2=root[218] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t666 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_90}] =
        permute x=t881 {derived} perm=[H<-C, W<-H, C<-W]
    n422 {pt2=root[219] torch.ops.aten.add.Tensor}: [t667 f32 [H=576 W=7 C=7] {pt2=root:add_28}] =
      add_scalar x=t666 {pt2=root:getitem_90} scalar=3
    n423 {pt2=root[220] torch.ops.aten.clamp.default}: [t668 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_46}] =
      clamp x=t667 {pt2=root:add_28} params={min=0; max=none}
    n424 {pt2=root[221] torch.ops.aten.clamp.default}: [t669 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_47}] =
      clamp x=t668 {pt2=root:clamp_46} params={min=none; max=6}
    n425 {pt2=root[222] torch.ops.aten.mul.Tensor}: [t670 f32 [H=576 W=7 C=7] {pt2=root:mul_23}] =
      mul a=t666 {pt2=root:getitem_90} b=t669 {pt2=root:clamp_47}
    n426 {pt2=root[223] torch.ops.aten.div.Tensor}: [t671 f32 [H=576 W=7 C=7] {pt2=root:div_23}] =
      div_scalar x=t670 {pt2=root:mul_23} scalar=6
    group g79 torch.ops.aten.convolution.default:
      n427 {derived}: [t672 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t671 {pt2=root:div_23} perm=[H<-W, W<-C, C<-H]
    n578 {derived}: [t882 f32 [H=7 W=7 C=576] {derived}] =
      convolution
        x=t672 {derived}
        weight=t795 {folded from=[p_features_11_block_1_0_weight,p_features_11_block_1_1_weight,b_features_11_block_1_1_running_var]}
        bias=t796 {folded from=[p_features_11_block_1_1_weight,p_features_11_block_1_1_bias,b_features_11_block_1_1_running_mean,b_features_11_block_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=2; w=2};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=576}
    group g80 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n433 {pt2=root[225] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t678 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_93}] =
        permute x=t882 {derived} perm=[H<-C, W<-H, C<-W]
    n434 {pt2=root[226] torch.ops.aten.add.Tensor}: [t679 f32 [H=576 W=7 C=7] {pt2=root:add_29}] =
      add_scalar x=t678 {pt2=root:getitem_93} scalar=3
    n435 {pt2=root[227] torch.ops.aten.clamp.default}: [t680 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_48}] =
      clamp x=t679 {pt2=root:add_29} params={min=0; max=none}
    n436 {pt2=root[228] torch.ops.aten.clamp.default}: [t681 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_49}] =
      clamp x=t680 {pt2=root:clamp_48} params={min=none; max=6}
    n437 {pt2=root[229] torch.ops.aten.mul.Tensor}: [t682 f32 [H=576 W=7 C=7] {pt2=root:mul_24}] =
      mul a=t678 {pt2=root:getitem_93} b=t681 {pt2=root:clamp_49}
    n438 {pt2=root[230] torch.ops.aten.div.Tensor}: [t683 f32 [H=576 W=7 C=7] {pt2=root:div_24}] =
      div_scalar x=t682 {pt2=root:mul_24} scalar=6
    n439 {pt2=root[231] torch.ops.aten.mean.dim}: [t684 f32 [H=576 W=1 C=1] {pt2=root:mean_8}] =
      mean x=t683 {pt2=root:div_24} params={dims=[C, W]; keepdim=true}
    group g81 torch.ops.aten.convolution.default:
      n440 {derived}: [t685 f32 [C=576] {derived}] =
        permute x=t684 {pt2=root:mean_8} perm=[H<-W, W<-C, C<-H]
      n442 {derived}: [t687 f32 [C=144] {derived}] =
        convolution
          x=t685 {derived}
          weight=t686 {folded from=[p_features_11_block_2_fc1_weight]}
          bias=t129 {pt2=root:p_features_11_block_2_fc1_bias target=features.11.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n579 {pt2=root[233] torch.ops.aten.relu.default}: [t883 f32 [C=144] {derived}] =
      relu x=t687 {derived}
    group g82 torch.ops.aten.convolution.default:
      n447 {derived}: [t692 f32 [C=576] {derived}] =
        convolution
          x=t883 {derived}
          weight=t691 {folded from=[p_features_11_block_2_fc2_weight]}
          bias=t131 {pt2=root:p_features_11_block_2_fc2_bias target=features.11.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n580 {pt2=root[235] torch.ops.aten.add.Tensor}: [t884 f32 [C=576] {derived}] =
      add_scalar x=t692 {derived} scalar=3
    n581 {pt2=root[236] torch.ops.aten.clamp.default}: [t885 f32 [C=576] {derived}] =
      clamp x=t884 {derived} params={min=0; max=none}
    n582 {pt2=root[237] torch.ops.aten.clamp.default}: [t886 f32 [C=576] {derived}] =
      clamp x=t885 {derived} params={min=none; max=6}
    n583 {pt2=root[238] torch.ops.aten.div.Tensor}: [t887 f32 [C=576] {derived}] =
      div_scalar x=t886 {derived} scalar=6
    n584 {pt2=root[234] torch.ops.aten.convolution.default}: [t697 f32 [H=576
                                                                      W=1 C=1] {pt2=root:div_25}] =
      permute x=t887 {derived} perm=[H<-C, W<-H, C<-W]
    n453 {pt2=root[239] torch.ops.aten.mul.Tensor}: [t698 f32 [H=576 W=7 C=7] {pt2=root:mul_25}] =
      mul a=t697 {pt2=root:div_25} b=t683 {pt2=root:div_24}
    group g83 torch.ops.aten.convolution.default:
      n454 {derived}: [t699 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t698 {pt2=root:mul_25} perm=[H<-W, W<-C, C<-H]
    n585 {derived}: [t888 f32 [H=7 W=7 C=96] {derived}] =
      convolution
        x=t699 {derived}
        weight=t797 {folded from=[p_features_11_block_3_0_weight,p_features_11_block_3_1_weight,b_features_11_block_3_1_running_var]}
        bias=t798 {folded from=[p_features_11_block_3_1_weight,p_features_11_block_3_1_bias,b_features_11_block_3_1_running_mean,b_features_11_block_3_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n586 {pt2=root[242] torch.ops.aten.add.Tensor}: [t889 f32 [H=7 W=7 C=96] {derived}] =
      add a=t888 {derived} b=t880 {derived}
    n587 {derived}: [t890 f32 [H=7 W=7 C=576] {derived}] =
      convolution
        x=t889 {derived}
        weight=t799 {folded from=[p_features_12_0_weight,p_features_12_1_weight,b_features_12_1_running_var]}
        bias=t800 {folded from=[p_features_12_1_weight,p_features_12_1_bias,b_features_12_1_running_mean,b_features_12_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g86 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n468 {pt2=root[244] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t713 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_99}] =
        permute x=t890 {derived} perm=[H<-C, W<-H, C<-W]
    n469 {pt2=root[245] torch.ops.aten.add.Tensor}: [t714 f32 [H=576 W=7 C=7] {pt2=root:add_32}] =
      add_scalar x=t713 {pt2=root:getitem_99} scalar=3
    n470 {pt2=root[246] torch.ops.aten.clamp.default}: [t715 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_52}] =
      clamp x=t714 {pt2=root:add_32} params={min=0; max=none}
    n471 {pt2=root[247] torch.ops.aten.clamp.default}: [t716 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_53}] =
      clamp x=t715 {pt2=root:clamp_52} params={min=none; max=6}
    n472 {pt2=root[248] torch.ops.aten.mul.Tensor}: [t717 f32 [H=576 W=7 C=7] {pt2=root:mul_26}] =
      mul a=t713 {pt2=root:getitem_99} b=t716 {pt2=root:clamp_53}
    n473 {pt2=root[249] torch.ops.aten.div.Tensor}: [t718 f32 [H=576 W=7 C=7] {pt2=root:div_26}] =
      div_scalar x=t717 {pt2=root:mul_26} scalar=6
    n474 {pt2=root[250] torch.ops.aten.mean.dim}: [t719 f32 [H=576 W=1 C=1] {pt2=root:mean_9}] =
      mean x=t718 {pt2=root:div_26} params={dims=[C, W]; keepdim=true}
    n588 {pt2=root[251] torch.ops.aten.view.default}: [t720 f32 [C=576] {pt2=root:view}] =
      permute x=t719 {pt2=root:mean_9} perm=[H<-W, W<-C, C<-H]
    group g87 torch.ops.aten.addmm.default:
      n478 {pt2=root[253] torch.ops.aten.addmm.default}: [t723 f32 [C=1024] {pt2=root:addmm}] =
        linear
          x=t720 {pt2=root:view}
          weight=t722 {folded from=[p_classifier_0_weight]}
          bias=t139 {pt2=root:p_classifier_0_bias target=classifier.0.bias}
          params={in_features=576}
    n479 {pt2=root[254] torch.ops.aten.add.Tensor}: [t724 f32 [C=1024] {pt2=root:add_33}] =
      add_scalar x=t723 {pt2=root:addmm} scalar=3
    n480 {pt2=root[255] torch.ops.aten.clamp.default}: [t725 f32 [C=1024] {pt2=root:clamp_54}] =
      clamp x=t724 {pt2=root:add_33} params={min=0; max=none}
    n481 {pt2=root[256] torch.ops.aten.clamp.default}: [t726 f32 [C=1024] {pt2=root:clamp_55}] =
      clamp x=t725 {pt2=root:clamp_54} params={min=none; max=6}
    n482 {pt2=root[257] torch.ops.aten.mul.Tensor}: [t727 f32 [C=1024] {pt2=root:mul_27}] =
      mul a=t723 {pt2=root:addmm} b=t726 {pt2=root:clamp_55}
    n483 {pt2=root[258] torch.ops.aten.div.Tensor}: [t728 f32 [C=1024] {pt2=root:div_27}] =
      div_scalar x=t727 {pt2=root:mul_27} scalar=6
    n484 {pt2=root[259] torch.ops.aten.clone.default}: [t729 f32 [C=1024] {pt2=root:clone}] =
      clone x=t728 {pt2=root:div_27}
    group g88 torch.ops.aten.addmm.default:
      n487 {pt2=root[261] torch.ops.aten.addmm.default}: [t732 f32 [C=1000] {pt2=root:addmm_1}] =
        linear
          x=t729 {pt2=root:clone}
          weight=t731 {folded from=[p_classifier_3_weight]}
          bias=t141 {pt2=root:p_classifier_3_bias target=classifier.3.bias}
          params={in_features=1024}
  outputs: [t732 f32 [C=1000] {pt2=root:addmm_1}]

MobileNet-v2 folds furthest of the three: all 52 batch norms disappear into their
convolutions and every weight relayout becomes data, leaving a single permute in
the whole graph. The 35 hardtanh nodes all survive, which is the point — relu6
is real per-activation work, not something folding can hoist.

  $ ../bin/native_graph.exe transform --fold --pt2 "$PT2_DATA/mobilenet_v2/mobilenet_v2.pt2"
  nodes: 415 -> 101
  constants: 106, of which 105 folded
  graph
  inputs:
    [t157 f32 [C=1000] {pt2=root:p_classifier_1_bias target=classifier.1.bias} constant,
     t314 f32 [H=3 W=224 C=224] {pt2=root:x},
     t728 f32 [N=1000 T=1 D=1 H=1 W=1 C=1280] {folded from=[p_classifier_1_weight]} constant,
     t730 f32 [N=32 T=1 D=1 H=3 W=3 C=3] {folded from=[p_features_0_0_weight,p_features_0_1_weight,b_features_0_1_running_var]} constant,
     t731 f32 [C=32] {folded from=[p_features_0_1_weight,p_features_0_1_bias,b_features_0_1_running_mean,b_features_0_1_running_var]} constant,
     t732 f32 [N=32 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_1_conv_0_0_weight,p_features_1_conv_0_1_weight,b_features_1_conv_0_1_running_var]} constant,
     t733 f32 [C=32] {folded from=[p_features_1_conv_0_1_weight,p_features_1_conv_0_1_bias,b_features_1_conv_0_1_running_mean,b_features_1_conv_0_1_running_var]} constant,
     t734 f32 [N=16 T=1 D=1 H=1 W=1 C=32] {folded from=[p_features_1_conv_1_weight,p_features_1_conv_2_weight,b_features_1_conv_2_running_var]} constant,
     t735 f32 [C=16] {folded from=[p_features_1_conv_2_weight,p_features_1_conv_2_bias,b_features_1_conv_2_running_mean,b_features_1_conv_2_running_var]} constant,
     t736 f32 [N=96 T=1 D=1 H=1 W=1 C=16] {folded from=[p_features_2_conv_0_0_weight,p_features_2_conv_0_1_weight,b_features_2_conv_0_1_running_var]} constant,
     t737 f32 [C=96] {folded from=[p_features_2_conv_0_1_weight,p_features_2_conv_0_1_bias,b_features_2_conv_0_1_running_mean,b_features_2_conv_0_1_running_var]} constant,
     t738 f32 [N=96 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_2_conv_1_0_weight,p_features_2_conv_1_1_weight,b_features_2_conv_1_1_running_var]} constant,
     t739 f32 [C=96] {folded from=[p_features_2_conv_1_1_weight,p_features_2_conv_1_1_bias,b_features_2_conv_1_1_running_mean,b_features_2_conv_1_1_running_var]} constant,
     t740 f32 [N=24 T=1 D=1 H=1 W=1 C=96] {folded from=[p_features_2_conv_2_weight,p_features_2_conv_3_weight,b_features_2_conv_3_running_var]} constant,
     t741 f32 [C=24] {folded from=[p_features_2_conv_3_weight,p_features_2_conv_3_bias,b_features_2_conv_3_running_mean,b_features_2_conv_3_running_var]} constant,
     t742 f32 [N=144 T=1 D=1 H=1 W=1 C=24] {folded from=[p_features_3_conv_0_0_weight,p_features_3_conv_0_1_weight,b_features_3_conv_0_1_running_var]} constant,
     t743 f32 [C=144] {folded from=[p_features_3_conv_0_1_weight,p_features_3_conv_0_1_bias,b_features_3_conv_0_1_running_mean,b_features_3_conv_0_1_running_var]} constant,
     t744 f32 [N=144 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_3_conv_1_0_weight,p_features_3_conv_1_1_weight,b_features_3_conv_1_1_running_var]} constant,
     t745 f32 [C=144] {folded from=[p_features_3_conv_1_1_weight,p_features_3_conv_1_1_bias,b_features_3_conv_1_1_running_mean,b_features_3_conv_1_1_running_var]} constant,
     t746 f32 [N=24 T=1 D=1 H=1 W=1 C=144] {folded from=[p_features_3_conv_2_weight,p_features_3_conv_3_weight,b_features_3_conv_3_running_var]} constant,
     t747 f32 [C=24] {folded from=[p_features_3_conv_3_weight,p_features_3_conv_3_bias,b_features_3_conv_3_running_mean,b_features_3_conv_3_running_var]} constant,
     t748 f32 [N=144 T=1 D=1 H=1 W=1 C=24] {folded from=[p_features_4_conv_0_0_weight,p_features_4_conv_0_1_weight,b_features_4_conv_0_1_running_var]} constant,
     t749 f32 [C=144] {folded from=[p_features_4_conv_0_1_weight,p_features_4_conv_0_1_bias,b_features_4_conv_0_1_running_mean,b_features_4_conv_0_1_running_var]} constant,
     t750 f32 [N=144 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_4_conv_1_0_weight,p_features_4_conv_1_1_weight,b_features_4_conv_1_1_running_var]} constant,
     t751 f32 [C=144] {folded from=[p_features_4_conv_1_1_weight,p_features_4_conv_1_1_bias,b_features_4_conv_1_1_running_mean,b_features_4_conv_1_1_running_var]} constant,
     t752 f32 [N=32 T=1 D=1 H=1 W=1 C=144] {folded from=[p_features_4_conv_2_weight,p_features_4_conv_3_weight,b_features_4_conv_3_running_var]} constant,
     t753 f32 [C=32] {folded from=[p_features_4_conv_3_weight,p_features_4_conv_3_bias,b_features_4_conv_3_running_mean,b_features_4_conv_3_running_var]} constant,
     t754 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {folded from=[p_features_5_conv_0_0_weight,p_features_5_conv_0_1_weight,b_features_5_conv_0_1_running_var]} constant,
     t755 f32 [C=192] {folded from=[p_features_5_conv_0_1_weight,p_features_5_conv_0_1_bias,b_features_5_conv_0_1_running_mean,b_features_5_conv_0_1_running_var]} constant,
     t756 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_5_conv_1_0_weight,p_features_5_conv_1_1_weight,b_features_5_conv_1_1_running_var]} constant,
     t757 f32 [C=192] {folded from=[p_features_5_conv_1_1_weight,p_features_5_conv_1_1_bias,b_features_5_conv_1_1_running_mean,b_features_5_conv_1_1_running_var]} constant,
     t758 f32 [N=32 T=1 D=1 H=1 W=1 C=192] {folded from=[p_features_5_conv_2_weight,p_features_5_conv_3_weight,b_features_5_conv_3_running_var]} constant,
     t759 f32 [C=32] {folded from=[p_features_5_conv_3_weight,p_features_5_conv_3_bias,b_features_5_conv_3_running_mean,b_features_5_conv_3_running_var]} constant,
     t760 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {folded from=[p_features_6_conv_0_0_weight,p_features_6_conv_0_1_weight,b_features_6_conv_0_1_running_var]} constant,
     t761 f32 [C=192] {folded from=[p_features_6_conv_0_1_weight,p_features_6_conv_0_1_bias,b_features_6_conv_0_1_running_mean,b_features_6_conv_0_1_running_var]} constant,
     t762 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_6_conv_1_0_weight,p_features_6_conv_1_1_weight,b_features_6_conv_1_1_running_var]} constant,
     t763 f32 [C=192] {folded from=[p_features_6_conv_1_1_weight,p_features_6_conv_1_1_bias,b_features_6_conv_1_1_running_mean,b_features_6_conv_1_1_running_var]} constant,
     t764 f32 [N=32 T=1 D=1 H=1 W=1 C=192] {folded from=[p_features_6_conv_2_weight,p_features_6_conv_3_weight,b_features_6_conv_3_running_var]} constant,
     t765 f32 [C=32] {folded from=[p_features_6_conv_3_weight,p_features_6_conv_3_bias,b_features_6_conv_3_running_mean,b_features_6_conv_3_running_var]} constant,
     t766 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {folded from=[p_features_7_conv_0_0_weight,p_features_7_conv_0_1_weight,b_features_7_conv_0_1_running_var]} constant,
     t767 f32 [C=192] {folded from=[p_features_7_conv_0_1_weight,p_features_7_conv_0_1_bias,b_features_7_conv_0_1_running_mean,b_features_7_conv_0_1_running_var]} constant,
     t768 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_7_conv_1_0_weight,p_features_7_conv_1_1_weight,b_features_7_conv_1_1_running_var]} constant,
     t769 f32 [C=192] {folded from=[p_features_7_conv_1_1_weight,p_features_7_conv_1_1_bias,b_features_7_conv_1_1_running_mean,b_features_7_conv_1_1_running_var]} constant,
     t770 f32 [N=64 T=1 D=1 H=1 W=1 C=192] {folded from=[p_features_7_conv_2_weight,p_features_7_conv_3_weight,b_features_7_conv_3_running_var]} constant,
     t771 f32 [C=64] {folded from=[p_features_7_conv_3_weight,p_features_7_conv_3_bias,b_features_7_conv_3_running_mean,b_features_7_conv_3_running_var]} constant,
     t772 f32 [N=384 T=1 D=1 H=1 W=1 C=64] {folded from=[p_features_8_conv_0_0_weight,p_features_8_conv_0_1_weight,b_features_8_conv_0_1_running_var]} constant,
     t773 f32 [C=384] {folded from=[p_features_8_conv_0_1_weight,p_features_8_conv_0_1_bias,b_features_8_conv_0_1_running_mean,b_features_8_conv_0_1_running_var]} constant,
     t774 f32 [N=384 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_8_conv_1_0_weight,p_features_8_conv_1_1_weight,b_features_8_conv_1_1_running_var]} constant,
     t775 f32 [C=384] {folded from=[p_features_8_conv_1_1_weight,p_features_8_conv_1_1_bias,b_features_8_conv_1_1_running_mean,b_features_8_conv_1_1_running_var]} constant,
     t776 f32 [N=64 T=1 D=1 H=1 W=1 C=384] {folded from=[p_features_8_conv_2_weight,p_features_8_conv_3_weight,b_features_8_conv_3_running_var]} constant,
     t777 f32 [C=64] {folded from=[p_features_8_conv_3_weight,p_features_8_conv_3_bias,b_features_8_conv_3_running_mean,b_features_8_conv_3_running_var]} constant,
     t778 f32 [N=384 T=1 D=1 H=1 W=1 C=64] {folded from=[p_features_9_conv_0_0_weight,p_features_9_conv_0_1_weight,b_features_9_conv_0_1_running_var]} constant,
     t779 f32 [C=384] {folded from=[p_features_9_conv_0_1_weight,p_features_9_conv_0_1_bias,b_features_9_conv_0_1_running_mean,b_features_9_conv_0_1_running_var]} constant,
     t780 f32 [N=384 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_9_conv_1_0_weight,p_features_9_conv_1_1_weight,b_features_9_conv_1_1_running_var]} constant,
     t781 f32 [C=384] {folded from=[p_features_9_conv_1_1_weight,p_features_9_conv_1_1_bias,b_features_9_conv_1_1_running_mean,b_features_9_conv_1_1_running_var]} constant,
     t782 f32 [N=64 T=1 D=1 H=1 W=1 C=384] {folded from=[p_features_9_conv_2_weight,p_features_9_conv_3_weight,b_features_9_conv_3_running_var]} constant,
     t783 f32 [C=64] {folded from=[p_features_9_conv_3_weight,p_features_9_conv_3_bias,b_features_9_conv_3_running_mean,b_features_9_conv_3_running_var]} constant,
     t784 f32 [N=384 T=1 D=1 H=1 W=1 C=64] {folded from=[p_features_10_conv_0_0_weight,p_features_10_conv_0_1_weight,b_features_10_conv_0_1_running_var]} constant,
     t785 f32 [C=384] {folded from=[p_features_10_conv_0_1_weight,p_features_10_conv_0_1_bias,b_features_10_conv_0_1_running_mean,b_features_10_conv_0_1_running_var]} constant,
     t786 f32 [N=384 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_10_conv_1_0_weight,p_features_10_conv_1_1_weight,b_features_10_conv_1_1_running_var]} constant,
     t787 f32 [C=384] {folded from=[p_features_10_conv_1_1_weight,p_features_10_conv_1_1_bias,b_features_10_conv_1_1_running_mean,b_features_10_conv_1_1_running_var]} constant,
     t788 f32 [N=64 T=1 D=1 H=1 W=1 C=384] {folded from=[p_features_10_conv_2_weight,p_features_10_conv_3_weight,b_features_10_conv_3_running_var]} constant,
     t789 f32 [C=64] {folded from=[p_features_10_conv_3_weight,p_features_10_conv_3_bias,b_features_10_conv_3_running_mean,b_features_10_conv_3_running_var]} constant,
     t790 f32 [N=384 T=1 D=1 H=1 W=1 C=64] {folded from=[p_features_11_conv_0_0_weight,p_features_11_conv_0_1_weight,b_features_11_conv_0_1_running_var]} constant,
     t791 f32 [C=384] {folded from=[p_features_11_conv_0_1_weight,p_features_11_conv_0_1_bias,b_features_11_conv_0_1_running_mean,b_features_11_conv_0_1_running_var]} constant,
     t792 f32 [N=384 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_11_conv_1_0_weight,p_features_11_conv_1_1_weight,b_features_11_conv_1_1_running_var]} constant,
     t793 f32 [C=384] {folded from=[p_features_11_conv_1_1_weight,p_features_11_conv_1_1_bias,b_features_11_conv_1_1_running_mean,b_features_11_conv_1_1_running_var]} constant,
     t794 f32 [N=96 T=1 D=1 H=1 W=1 C=384] {folded from=[p_features_11_conv_2_weight,p_features_11_conv_3_weight,b_features_11_conv_3_running_var]} constant,
     t795 f32 [C=96] {folded from=[p_features_11_conv_3_weight,p_features_11_conv_3_bias,b_features_11_conv_3_running_mean,b_features_11_conv_3_running_var]} constant,
     t796 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {folded from=[p_features_12_conv_0_0_weight,p_features_12_conv_0_1_weight,b_features_12_conv_0_1_running_var]} constant,
     t797 f32 [C=576] {folded from=[p_features_12_conv_0_1_weight,p_features_12_conv_0_1_bias,b_features_12_conv_0_1_running_mean,b_features_12_conv_0_1_running_var]} constant,
     t798 f32 [N=576 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_12_conv_1_0_weight,p_features_12_conv_1_1_weight,b_features_12_conv_1_1_running_var]} constant,
     t799 f32 [C=576] {folded from=[p_features_12_conv_1_1_weight,p_features_12_conv_1_1_bias,b_features_12_conv_1_1_running_mean,b_features_12_conv_1_1_running_var]} constant,
     t800 f32 [N=96 T=1 D=1 H=1 W=1 C=576] {folded from=[p_features_12_conv_2_weight,p_features_12_conv_3_weight,b_features_12_conv_3_running_var]} constant,
     t801 f32 [C=96] {folded from=[p_features_12_conv_3_weight,p_features_12_conv_3_bias,b_features_12_conv_3_running_mean,b_features_12_conv_3_running_var]} constant,
     t802 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {folded from=[p_features_13_conv_0_0_weight,p_features_13_conv_0_1_weight,b_features_13_conv_0_1_running_var]} constant,
     t803 f32 [C=576] {folded from=[p_features_13_conv_0_1_weight,p_features_13_conv_0_1_bias,b_features_13_conv_0_1_running_mean,b_features_13_conv_0_1_running_var]} constant,
     t804 f32 [N=576 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_13_conv_1_0_weight,p_features_13_conv_1_1_weight,b_features_13_conv_1_1_running_var]} constant,
     t805 f32 [C=576] {folded from=[p_features_13_conv_1_1_weight,p_features_13_conv_1_1_bias,b_features_13_conv_1_1_running_mean,b_features_13_conv_1_1_running_var]} constant,
     t806 f32 [N=96 T=1 D=1 H=1 W=1 C=576] {folded from=[p_features_13_conv_2_weight,p_features_13_conv_3_weight,b_features_13_conv_3_running_var]} constant,
     t807 f32 [C=96] {folded from=[p_features_13_conv_3_weight,p_features_13_conv_3_bias,b_features_13_conv_3_running_mean,b_features_13_conv_3_running_var]} constant,
     t808 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {folded from=[p_features_14_conv_0_0_weight,p_features_14_conv_0_1_weight,b_features_14_conv_0_1_running_var]} constant,
     t809 f32 [C=576] {folded from=[p_features_14_conv_0_1_weight,p_features_14_conv_0_1_bias,b_features_14_conv_0_1_running_mean,b_features_14_conv_0_1_running_var]} constant,
     t810 f32 [N=576 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_14_conv_1_0_weight,p_features_14_conv_1_1_weight,b_features_14_conv_1_1_running_var]} constant,
     t811 f32 [C=576] {folded from=[p_features_14_conv_1_1_weight,p_features_14_conv_1_1_bias,b_features_14_conv_1_1_running_mean,b_features_14_conv_1_1_running_var]} constant,
     t812 f32 [N=160 T=1 D=1 H=1 W=1 C=576] {folded from=[p_features_14_conv_2_weight,p_features_14_conv_3_weight,b_features_14_conv_3_running_var]} constant,
     t813 f32 [C=160] {folded from=[p_features_14_conv_3_weight,p_features_14_conv_3_bias,b_features_14_conv_3_running_mean,b_features_14_conv_3_running_var]} constant,
     t814 f32 [N=960 T=1 D=1 H=1 W=1 C=160] {folded from=[p_features_15_conv_0_0_weight,p_features_15_conv_0_1_weight,b_features_15_conv_0_1_running_var]} constant,
     t815 f32 [C=960] {folded from=[p_features_15_conv_0_1_weight,p_features_15_conv_0_1_bias,b_features_15_conv_0_1_running_mean,b_features_15_conv_0_1_running_var]} constant,
     t816 f32 [N=960 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_15_conv_1_0_weight,p_features_15_conv_1_1_weight,b_features_15_conv_1_1_running_var]} constant,
     t817 f32 [C=960] {folded from=[p_features_15_conv_1_1_weight,p_features_15_conv_1_1_bias,b_features_15_conv_1_1_running_mean,b_features_15_conv_1_1_running_var]} constant,
     t818 f32 [N=160 T=1 D=1 H=1 W=1 C=960] {folded from=[p_features_15_conv_2_weight,p_features_15_conv_3_weight,b_features_15_conv_3_running_var]} constant,
     t819 f32 [C=160] {folded from=[p_features_15_conv_3_weight,p_features_15_conv_3_bias,b_features_15_conv_3_running_mean,b_features_15_conv_3_running_var]} constant,
     t820 f32 [N=960 T=1 D=1 H=1 W=1 C=160] {folded from=[p_features_16_conv_0_0_weight,p_features_16_conv_0_1_weight,b_features_16_conv_0_1_running_var]} constant,
     t821 f32 [C=960] {folded from=[p_features_16_conv_0_1_weight,p_features_16_conv_0_1_bias,b_features_16_conv_0_1_running_mean,b_features_16_conv_0_1_running_var]} constant,
     t822 f32 [N=960 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_16_conv_1_0_weight,p_features_16_conv_1_1_weight,b_features_16_conv_1_1_running_var]} constant,
     t823 f32 [C=960] {folded from=[p_features_16_conv_1_1_weight,p_features_16_conv_1_1_bias,b_features_16_conv_1_1_running_mean,b_features_16_conv_1_1_running_var]} constant,
     t824 f32 [N=160 T=1 D=1 H=1 W=1 C=960] {folded from=[p_features_16_conv_2_weight,p_features_16_conv_3_weight,b_features_16_conv_3_running_var]} constant,
     t825 f32 [C=160] {folded from=[p_features_16_conv_3_weight,p_features_16_conv_3_bias,b_features_16_conv_3_running_mean,b_features_16_conv_3_running_var]} constant,
     t826 f32 [N=960 T=1 D=1 H=1 W=1 C=160] {folded from=[p_features_17_conv_0_0_weight,p_features_17_conv_0_1_weight,b_features_17_conv_0_1_running_var]} constant,
     t827 f32 [C=960] {folded from=[p_features_17_conv_0_1_weight,p_features_17_conv_0_1_bias,b_features_17_conv_0_1_running_mean,b_features_17_conv_0_1_running_var]} constant,
     t828 f32 [N=960 T=1 D=1 H=3 W=3 C=1] {folded from=[p_features_17_conv_1_0_weight,p_features_17_conv_1_1_weight,b_features_17_conv_1_1_running_var]} constant,
     t829 f32 [C=960] {folded from=[p_features_17_conv_1_1_weight,p_features_17_conv_1_1_bias,b_features_17_conv_1_1_running_mean,b_features_17_conv_1_1_running_var]} constant,
     t830 f32 [N=320 T=1 D=1 H=1 W=1 C=960] {folded from=[p_features_17_conv_2_weight,p_features_17_conv_3_weight,b_features_17_conv_3_running_var]} constant,
     t831 f32 [C=320] {folded from=[p_features_17_conv_3_weight,p_features_17_conv_3_bias,b_features_17_conv_3_running_mean,b_features_17_conv_3_running_var]} constant,
     t832 f32 [N=1280 T=1 D=1 H=1 W=1 C=320] {folded from=[p_features_18_0_weight,p_features_18_1_weight,b_features_18_1_running_var]} constant,
     t833 f32 [C=1280] {folded from=[p_features_18_1_weight,p_features_18_1_bias,b_features_18_1_running_mean,b_features_18_1_running_var]} constant]
  nodes:
    group g1 torch.ops.aten.convolution.default:
      n0 {derived}: [t315 f32 [H=224 W=224 C=3] {derived}] =
        permute x=t314 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
    n415 {derived}: [t834 f32 [H=112 W=112 C=32] {derived}] =
      convolution
        x=t315 {derived}
        weight=t730 {folded from=[p_features_0_0_weight,p_features_0_1_weight,b_features_0_1_running_var]}
        bias=t731 {folded from=[p_features_0_1_weight,p_features_0_1_bias,b_features_0_1_running_mean,b_features_0_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n416 {pt2=root[2] torch.ops.aten.hardtanh.default}: [t835 f32 [H=112 W=112
                                                                   C=32] {derived}] =
      hardtanh x=t834 {derived} params={min_val=0; max_val=6}
    n417 {derived}: [t836 f32 [H=112 W=112 C=32] {derived}] =
      convolution
        x=t835 {derived}
        weight=t732 {folded from=[p_features_1_conv_0_0_weight,p_features_1_conv_0_1_weight,b_features_1_conv_0_1_running_var]}
        bias=t733 {folded from=[p_features_1_conv_0_1_weight,p_features_1_conv_0_1_bias,b_features_1_conv_0_1_running_mean,b_features_1_conv_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=32}
    n418 {pt2=root[5] torch.ops.aten.hardtanh.default}: [t837 f32 [H=112 W=112
                                                                   C=32] {derived}] =
      hardtanh x=t836 {derived} params={min_val=0; max_val=6}
    n419 {derived}: [t838 f32 [H=112 W=112 C=16] {derived}] =
      convolution
        x=t837 {derived}
        weight=t734 {folded from=[p_features_1_conv_1_weight,p_features_1_conv_2_weight,b_features_1_conv_2_running_var]}
        bias=t735 {folded from=[p_features_1_conv_2_weight,p_features_1_conv_2_bias,b_features_1_conv_2_running_mean,b_features_1_conv_2_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n420 {derived}: [t839 f32 [H=112 W=112 C=96] {derived}] =
      convolution
        x=t838 {derived}
        weight=t736 {folded from=[p_features_2_conv_0_0_weight,p_features_2_conv_0_1_weight,b_features_2_conv_0_1_running_var]}
        bias=t737 {folded from=[p_features_2_conv_0_1_weight,p_features_2_conv_0_1_bias,b_features_2_conv_0_1_running_mean,b_features_2_conv_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n421 {pt2=root[10] torch.ops.aten.hardtanh.default}: [t840 f32 [H=112 W=112
                                                                    C=96] {derived}] =
      hardtanh x=t839 {derived} params={min_val=0; max_val=6}
    n422 {derived}: [t841 f32 [H=56 W=56 C=96] {derived}] =
      convolution
        x=t840 {derived}
        weight=t738 {folded from=[p_features_2_conv_1_0_weight,p_features_2_conv_1_1_weight,b_features_2_conv_1_1_running_var]}
        bias=t739 {folded from=[p_features_2_conv_1_1_weight,p_features_2_conv_1_1_bias,b_features_2_conv_1_1_running_mean,b_features_2_conv_1_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=96}
    n423 {pt2=root[13] torch.ops.aten.hardtanh.default}: [t842 f32 [H=56 W=56
                                                                    C=96] {derived}] =
      hardtanh x=t841 {derived} params={min_val=0; max_val=6}
    n424 {derived}: [t843 f32 [H=56 W=56 C=24] {derived}] =
      convolution
        x=t842 {derived}
        weight=t740 {folded from=[p_features_2_conv_2_weight,p_features_2_conv_3_weight,b_features_2_conv_3_running_var]}
        bias=t741 {folded from=[p_features_2_conv_3_weight,p_features_2_conv_3_bias,b_features_2_conv_3_running_mean,b_features_2_conv_3_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n425 {derived}: [t844 f32 [H=56 W=56 C=144] {derived}] =
      convolution
        x=t843 {derived}
        weight=t742 {folded from=[p_features_3_conv_0_0_weight,p_features_3_conv_0_1_weight,b_features_3_conv_0_1_running_var]}
        bias=t743 {folded from=[p_features_3_conv_0_1_weight,p_features_3_conv_0_1_bias,b_features_3_conv_0_1_running_mean,b_features_3_conv_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n426 {pt2=root[18] torch.ops.aten.hardtanh.default}: [t845 f32 [H=56 W=56
                                                                    C=144] {derived}] =
      hardtanh x=t844 {derived} params={min_val=0; max_val=6}
    n427 {derived}: [t846 f32 [H=56 W=56 C=144] {derived}] =
      convolution
        x=t845 {derived}
        weight=t744 {folded from=[p_features_3_conv_1_0_weight,p_features_3_conv_1_1_weight,b_features_3_conv_1_1_running_var]}
        bias=t745 {folded from=[p_features_3_conv_1_1_weight,p_features_3_conv_1_1_bias,b_features_3_conv_1_1_running_mean,b_features_3_conv_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=144}
    n428 {pt2=root[21] torch.ops.aten.hardtanh.default}: [t847 f32 [H=56 W=56
                                                                    C=144] {derived}] =
      hardtanh x=t846 {derived} params={min_val=0; max_val=6}
    n429 {derived}: [t848 f32 [H=56 W=56 C=24] {derived}] =
      convolution
        x=t847 {derived}
        weight=t746 {folded from=[p_features_3_conv_2_weight,p_features_3_conv_3_weight,b_features_3_conv_3_running_var]}
        bias=t747 {folded from=[p_features_3_conv_3_weight,p_features_3_conv_3_bias,b_features_3_conv_3_running_mean,b_features_3_conv_3_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n430 {pt2=root[24] torch.ops.aten.add.Tensor}: [t849 f32 [H=56 W=56 C=24] {derived}] =
      add a=t843 {derived} b=t848 {derived}
    n431 {derived}: [t850 f32 [H=56 W=56 C=144] {derived}] =
      convolution
        x=t849 {derived}
        weight=t748 {folded from=[p_features_4_conv_0_0_weight,p_features_4_conv_0_1_weight,b_features_4_conv_0_1_running_var]}
        bias=t749 {folded from=[p_features_4_conv_0_1_weight,p_features_4_conv_0_1_bias,b_features_4_conv_0_1_running_mean,b_features_4_conv_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n432 {pt2=root[27] torch.ops.aten.hardtanh.default}: [t851 f32 [H=56 W=56
                                                                    C=144] {derived}] =
      hardtanh x=t850 {derived} params={min_val=0; max_val=6}
    n433 {derived}: [t852 f32 [H=28 W=28 C=144] {derived}] =
      convolution
        x=t851 {derived}
        weight=t750 {folded from=[p_features_4_conv_1_0_weight,p_features_4_conv_1_1_weight,b_features_4_conv_1_1_running_var]}
        bias=t751 {folded from=[p_features_4_conv_1_1_weight,p_features_4_conv_1_1_bias,b_features_4_conv_1_1_running_mean,b_features_4_conv_1_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=144}
    n434 {pt2=root[30] torch.ops.aten.hardtanh.default}: [t853 f32 [H=28 W=28
                                                                    C=144] {derived}] =
      hardtanh x=t852 {derived} params={min_val=0; max_val=6}
    n435 {derived}: [t854 f32 [H=28 W=28 C=32] {derived}] =
      convolution
        x=t853 {derived}
        weight=t752 {folded from=[p_features_4_conv_2_weight,p_features_4_conv_3_weight,b_features_4_conv_3_running_var]}
        bias=t753 {folded from=[p_features_4_conv_3_weight,p_features_4_conv_3_bias,b_features_4_conv_3_running_mean,b_features_4_conv_3_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n436 {derived}: [t855 f32 [H=28 W=28 C=192] {derived}] =
      convolution
        x=t854 {derived}
        weight=t754 {folded from=[p_features_5_conv_0_0_weight,p_features_5_conv_0_1_weight,b_features_5_conv_0_1_running_var]}
        bias=t755 {folded from=[p_features_5_conv_0_1_weight,p_features_5_conv_0_1_bias,b_features_5_conv_0_1_running_mean,b_features_5_conv_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n437 {pt2=root[35] torch.ops.aten.hardtanh.default}: [t856 f32 [H=28 W=28
                                                                    C=192] {derived}] =
      hardtanh x=t855 {derived} params={min_val=0; max_val=6}
    n438 {derived}: [t857 f32 [H=28 W=28 C=192] {derived}] =
      convolution
        x=t856 {derived}
        weight=t756 {folded from=[p_features_5_conv_1_0_weight,p_features_5_conv_1_1_weight,b_features_5_conv_1_1_running_var]}
        bias=t757 {folded from=[p_features_5_conv_1_1_weight,p_features_5_conv_1_1_bias,b_features_5_conv_1_1_running_mean,b_features_5_conv_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=192}
    n439 {pt2=root[38] torch.ops.aten.hardtanh.default}: [t858 f32 [H=28 W=28
                                                                    C=192] {derived}] =
      hardtanh x=t857 {derived} params={min_val=0; max_val=6}
    n440 {derived}: [t859 f32 [H=28 W=28 C=32] {derived}] =
      convolution
        x=t858 {derived}
        weight=t758 {folded from=[p_features_5_conv_2_weight,p_features_5_conv_3_weight,b_features_5_conv_3_running_var]}
        bias=t759 {folded from=[p_features_5_conv_3_weight,p_features_5_conv_3_bias,b_features_5_conv_3_running_mean,b_features_5_conv_3_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n441 {pt2=root[41] torch.ops.aten.add.Tensor}: [t860 f32 [H=28 W=28 C=32] {derived}] =
      add a=t854 {derived} b=t859 {derived}
    n442 {derived}: [t861 f32 [H=28 W=28 C=192] {derived}] =
      convolution
        x=t860 {derived}
        weight=t760 {folded from=[p_features_6_conv_0_0_weight,p_features_6_conv_0_1_weight,b_features_6_conv_0_1_running_var]}
        bias=t761 {folded from=[p_features_6_conv_0_1_weight,p_features_6_conv_0_1_bias,b_features_6_conv_0_1_running_mean,b_features_6_conv_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n443 {pt2=root[44] torch.ops.aten.hardtanh.default}: [t862 f32 [H=28 W=28
                                                                    C=192] {derived}] =
      hardtanh x=t861 {derived} params={min_val=0; max_val=6}
    n444 {derived}: [t863 f32 [H=28 W=28 C=192] {derived}] =
      convolution
        x=t862 {derived}
        weight=t762 {folded from=[p_features_6_conv_1_0_weight,p_features_6_conv_1_1_weight,b_features_6_conv_1_1_running_var]}
        bias=t763 {folded from=[p_features_6_conv_1_1_weight,p_features_6_conv_1_1_bias,b_features_6_conv_1_1_running_mean,b_features_6_conv_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=192}
    n445 {pt2=root[47] torch.ops.aten.hardtanh.default}: [t864 f32 [H=28 W=28
                                                                    C=192] {derived}] =
      hardtanh x=t863 {derived} params={min_val=0; max_val=6}
    n446 {derived}: [t865 f32 [H=28 W=28 C=32] {derived}] =
      convolution
        x=t864 {derived}
        weight=t764 {folded from=[p_features_6_conv_2_weight,p_features_6_conv_3_weight,b_features_6_conv_3_running_var]}
        bias=t765 {folded from=[p_features_6_conv_3_weight,p_features_6_conv_3_bias,b_features_6_conv_3_running_mean,b_features_6_conv_3_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n447 {pt2=root[50] torch.ops.aten.add.Tensor}: [t866 f32 [H=28 W=28 C=32] {derived}] =
      add a=t860 {derived} b=t865 {derived}
    n448 {derived}: [t867 f32 [H=28 W=28 C=192] {derived}] =
      convolution
        x=t866 {derived}
        weight=t766 {folded from=[p_features_7_conv_0_0_weight,p_features_7_conv_0_1_weight,b_features_7_conv_0_1_running_var]}
        bias=t767 {folded from=[p_features_7_conv_0_1_weight,p_features_7_conv_0_1_bias,b_features_7_conv_0_1_running_mean,b_features_7_conv_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n449 {pt2=root[53] torch.ops.aten.hardtanh.default}: [t868 f32 [H=28 W=28
                                                                    C=192] {derived}] =
      hardtanh x=t867 {derived} params={min_val=0; max_val=6}
    n450 {derived}: [t869 f32 [H=14 W=14 C=192] {derived}] =
      convolution
        x=t868 {derived}
        weight=t768 {folded from=[p_features_7_conv_1_0_weight,p_features_7_conv_1_1_weight,b_features_7_conv_1_1_running_var]}
        bias=t769 {folded from=[p_features_7_conv_1_1_weight,p_features_7_conv_1_1_bias,b_features_7_conv_1_1_running_mean,b_features_7_conv_1_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=192}
    n451 {pt2=root[56] torch.ops.aten.hardtanh.default}: [t870 f32 [H=14 W=14
                                                                    C=192] {derived}] =
      hardtanh x=t869 {derived} params={min_val=0; max_val=6}
    n452 {derived}: [t871 f32 [H=14 W=14 C=64] {derived}] =
      convolution
        x=t870 {derived}
        weight=t770 {folded from=[p_features_7_conv_2_weight,p_features_7_conv_3_weight,b_features_7_conv_3_running_var]}
        bias=t771 {folded from=[p_features_7_conv_3_weight,p_features_7_conv_3_bias,b_features_7_conv_3_running_mean,b_features_7_conv_3_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n453 {derived}: [t872 f32 [H=14 W=14 C=384] {derived}] =
      convolution
        x=t871 {derived}
        weight=t772 {folded from=[p_features_8_conv_0_0_weight,p_features_8_conv_0_1_weight,b_features_8_conv_0_1_running_var]}
        bias=t773 {folded from=[p_features_8_conv_0_1_weight,p_features_8_conv_0_1_bias,b_features_8_conv_0_1_running_mean,b_features_8_conv_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n454 {pt2=root[61] torch.ops.aten.hardtanh.default}: [t873 f32 [H=14 W=14
                                                                    C=384] {derived}] =
      hardtanh x=t872 {derived} params={min_val=0; max_val=6}
    n455 {derived}: [t874 f32 [H=14 W=14 C=384] {derived}] =
      convolution
        x=t873 {derived}
        weight=t774 {folded from=[p_features_8_conv_1_0_weight,p_features_8_conv_1_1_weight,b_features_8_conv_1_1_running_var]}
        bias=t775 {folded from=[p_features_8_conv_1_1_weight,p_features_8_conv_1_1_bias,b_features_8_conv_1_1_running_mean,b_features_8_conv_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=384}
    n456 {pt2=root[64] torch.ops.aten.hardtanh.default}: [t875 f32 [H=14 W=14
                                                                    C=384] {derived}] =
      hardtanh x=t874 {derived} params={min_val=0; max_val=6}
    n457 {derived}: [t876 f32 [H=14 W=14 C=64] {derived}] =
      convolution
        x=t875 {derived}
        weight=t776 {folded from=[p_features_8_conv_2_weight,p_features_8_conv_3_weight,b_features_8_conv_3_running_var]}
        bias=t777 {folded from=[p_features_8_conv_3_weight,p_features_8_conv_3_bias,b_features_8_conv_3_running_mean,b_features_8_conv_3_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n458 {pt2=root[67] torch.ops.aten.add.Tensor}: [t877 f32 [H=14 W=14 C=64] {derived}] =
      add a=t871 {derived} b=t876 {derived}
    n459 {derived}: [t878 f32 [H=14 W=14 C=384] {derived}] =
      convolution
        x=t877 {derived}
        weight=t778 {folded from=[p_features_9_conv_0_0_weight,p_features_9_conv_0_1_weight,b_features_9_conv_0_1_running_var]}
        bias=t779 {folded from=[p_features_9_conv_0_1_weight,p_features_9_conv_0_1_bias,b_features_9_conv_0_1_running_mean,b_features_9_conv_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n460 {pt2=root[70] torch.ops.aten.hardtanh.default}: [t879 f32 [H=14 W=14
                                                                    C=384] {derived}] =
      hardtanh x=t878 {derived} params={min_val=0; max_val=6}
    n461 {derived}: [t880 f32 [H=14 W=14 C=384] {derived}] =
      convolution
        x=t879 {derived}
        weight=t780 {folded from=[p_features_9_conv_1_0_weight,p_features_9_conv_1_1_weight,b_features_9_conv_1_1_running_var]}
        bias=t781 {folded from=[p_features_9_conv_1_1_weight,p_features_9_conv_1_1_bias,b_features_9_conv_1_1_running_mean,b_features_9_conv_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=384}
    n462 {pt2=root[73] torch.ops.aten.hardtanh.default}: [t881 f32 [H=14 W=14
                                                                    C=384] {derived}] =
      hardtanh x=t880 {derived} params={min_val=0; max_val=6}
    n463 {derived}: [t882 f32 [H=14 W=14 C=64] {derived}] =
      convolution
        x=t881 {derived}
        weight=t782 {folded from=[p_features_9_conv_2_weight,p_features_9_conv_3_weight,b_features_9_conv_3_running_var]}
        bias=t783 {folded from=[p_features_9_conv_3_weight,p_features_9_conv_3_bias,b_features_9_conv_3_running_mean,b_features_9_conv_3_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n464 {pt2=root[76] torch.ops.aten.add.Tensor}: [t883 f32 [H=14 W=14 C=64] {derived}] =
      add a=t877 {derived} b=t882 {derived}
    n465 {derived}: [t884 f32 [H=14 W=14 C=384] {derived}] =
      convolution
        x=t883 {derived}
        weight=t784 {folded from=[p_features_10_conv_0_0_weight,p_features_10_conv_0_1_weight,b_features_10_conv_0_1_running_var]}
        bias=t785 {folded from=[p_features_10_conv_0_1_weight,p_features_10_conv_0_1_bias,b_features_10_conv_0_1_running_mean,b_features_10_conv_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n466 {pt2=root[79] torch.ops.aten.hardtanh.default}: [t885 f32 [H=14 W=14
                                                                    C=384] {derived}] =
      hardtanh x=t884 {derived} params={min_val=0; max_val=6}
    n467 {derived}: [t886 f32 [H=14 W=14 C=384] {derived}] =
      convolution
        x=t885 {derived}
        weight=t786 {folded from=[p_features_10_conv_1_0_weight,p_features_10_conv_1_1_weight,b_features_10_conv_1_1_running_var]}
        bias=t787 {folded from=[p_features_10_conv_1_1_weight,p_features_10_conv_1_1_bias,b_features_10_conv_1_1_running_mean,b_features_10_conv_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=384}
    n468 {pt2=root[82] torch.ops.aten.hardtanh.default}: [t887 f32 [H=14 W=14
                                                                    C=384] {derived}] =
      hardtanh x=t886 {derived} params={min_val=0; max_val=6}
    n469 {derived}: [t888 f32 [H=14 W=14 C=64] {derived}] =
      convolution
        x=t887 {derived}
        weight=t788 {folded from=[p_features_10_conv_2_weight,p_features_10_conv_3_weight,b_features_10_conv_3_running_var]}
        bias=t789 {folded from=[p_features_10_conv_3_weight,p_features_10_conv_3_bias,b_features_10_conv_3_running_mean,b_features_10_conv_3_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n470 {pt2=root[85] torch.ops.aten.add.Tensor}: [t889 f32 [H=14 W=14 C=64] {derived}] =
      add a=t883 {derived} b=t888 {derived}
    n471 {derived}: [t890 f32 [H=14 W=14 C=384] {derived}] =
      convolution
        x=t889 {derived}
        weight=t790 {folded from=[p_features_11_conv_0_0_weight,p_features_11_conv_0_1_weight,b_features_11_conv_0_1_running_var]}
        bias=t791 {folded from=[p_features_11_conv_0_1_weight,p_features_11_conv_0_1_bias,b_features_11_conv_0_1_running_mean,b_features_11_conv_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n472 {pt2=root[88] torch.ops.aten.hardtanh.default}: [t891 f32 [H=14 W=14
                                                                    C=384] {derived}] =
      hardtanh x=t890 {derived} params={min_val=0; max_val=6}
    n473 {derived}: [t892 f32 [H=14 W=14 C=384] {derived}] =
      convolution
        x=t891 {derived}
        weight=t792 {folded from=[p_features_11_conv_1_0_weight,p_features_11_conv_1_1_weight,b_features_11_conv_1_1_running_var]}
        bias=t793 {folded from=[p_features_11_conv_1_1_weight,p_features_11_conv_1_1_bias,b_features_11_conv_1_1_running_mean,b_features_11_conv_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=384}
    n474 {pt2=root[91] torch.ops.aten.hardtanh.default}: [t893 f32 [H=14 W=14
                                                                    C=384] {derived}] =
      hardtanh x=t892 {derived} params={min_val=0; max_val=6}
    n475 {derived}: [t894 f32 [H=14 W=14 C=96] {derived}] =
      convolution
        x=t893 {derived}
        weight=t794 {folded from=[p_features_11_conv_2_weight,p_features_11_conv_3_weight,b_features_11_conv_3_running_var]}
        bias=t795 {folded from=[p_features_11_conv_3_weight,p_features_11_conv_3_bias,b_features_11_conv_3_running_mean,b_features_11_conv_3_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n476 {derived}: [t895 f32 [H=14 W=14 C=576] {derived}] =
      convolution
        x=t894 {derived}
        weight=t796 {folded from=[p_features_12_conv_0_0_weight,p_features_12_conv_0_1_weight,b_features_12_conv_0_1_running_var]}
        bias=t797 {folded from=[p_features_12_conv_0_1_weight,p_features_12_conv_0_1_bias,b_features_12_conv_0_1_running_mean,b_features_12_conv_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n477 {pt2=root[96] torch.ops.aten.hardtanh.default}: [t896 f32 [H=14 W=14
                                                                    C=576] {derived}] =
      hardtanh x=t895 {derived} params={min_val=0; max_val=6}
    n478 {derived}: [t897 f32 [H=14 W=14 C=576] {derived}] =
      convolution
        x=t896 {derived}
        weight=t798 {folded from=[p_features_12_conv_1_0_weight,p_features_12_conv_1_1_weight,b_features_12_conv_1_1_running_var]}
        bias=t799 {folded from=[p_features_12_conv_1_1_weight,p_features_12_conv_1_1_bias,b_features_12_conv_1_1_running_mean,b_features_12_conv_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=576}
    n479 {pt2=root[99] torch.ops.aten.hardtanh.default}: [t898 f32 [H=14 W=14
                                                                    C=576] {derived}] =
      hardtanh x=t897 {derived} params={min_val=0; max_val=6}
    n480 {derived}: [t899 f32 [H=14 W=14 C=96] {derived}] =
      convolution
        x=t898 {derived}
        weight=t800 {folded from=[p_features_12_conv_2_weight,p_features_12_conv_3_weight,b_features_12_conv_3_running_var]}
        bias=t801 {folded from=[p_features_12_conv_3_weight,p_features_12_conv_3_bias,b_features_12_conv_3_running_mean,b_features_12_conv_3_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n481 {pt2=root[102] torch.ops.aten.add.Tensor}: [t900 f32 [H=14 W=14 C=96] {derived}] =
      add a=t894 {derived} b=t899 {derived}
    n482 {derived}: [t901 f32 [H=14 W=14 C=576] {derived}] =
      convolution
        x=t900 {derived}
        weight=t802 {folded from=[p_features_13_conv_0_0_weight,p_features_13_conv_0_1_weight,b_features_13_conv_0_1_running_var]}
        bias=t803 {folded from=[p_features_13_conv_0_1_weight,p_features_13_conv_0_1_bias,b_features_13_conv_0_1_running_mean,b_features_13_conv_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n483 {pt2=root[105] torch.ops.aten.hardtanh.default}: [t902 f32 [H=14 W=14
                                                                     C=576] {derived}] =
      hardtanh x=t901 {derived} params={min_val=0; max_val=6}
    n484 {derived}: [t903 f32 [H=14 W=14 C=576] {derived}] =
      convolution
        x=t902 {derived}
        weight=t804 {folded from=[p_features_13_conv_1_0_weight,p_features_13_conv_1_1_weight,b_features_13_conv_1_1_running_var]}
        bias=t805 {folded from=[p_features_13_conv_1_1_weight,p_features_13_conv_1_1_bias,b_features_13_conv_1_1_running_mean,b_features_13_conv_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=576}
    n485 {pt2=root[108] torch.ops.aten.hardtanh.default}: [t904 f32 [H=14 W=14
                                                                     C=576] {derived}] =
      hardtanh x=t903 {derived} params={min_val=0; max_val=6}
    n486 {derived}: [t905 f32 [H=14 W=14 C=96] {derived}] =
      convolution
        x=t904 {derived}
        weight=t806 {folded from=[p_features_13_conv_2_weight,p_features_13_conv_3_weight,b_features_13_conv_3_running_var]}
        bias=t807 {folded from=[p_features_13_conv_3_weight,p_features_13_conv_3_bias,b_features_13_conv_3_running_mean,b_features_13_conv_3_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n487 {pt2=root[111] torch.ops.aten.add.Tensor}: [t906 f32 [H=14 W=14 C=96] {derived}] =
      add a=t900 {derived} b=t905 {derived}
    n488 {derived}: [t907 f32 [H=14 W=14 C=576] {derived}] =
      convolution
        x=t906 {derived}
        weight=t808 {folded from=[p_features_14_conv_0_0_weight,p_features_14_conv_0_1_weight,b_features_14_conv_0_1_running_var]}
        bias=t809 {folded from=[p_features_14_conv_0_1_weight,p_features_14_conv_0_1_bias,b_features_14_conv_0_1_running_mean,b_features_14_conv_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n489 {pt2=root[114] torch.ops.aten.hardtanh.default}: [t908 f32 [H=14 W=14
                                                                     C=576] {derived}] =
      hardtanh x=t907 {derived} params={min_val=0; max_val=6}
    n490 {derived}: [t909 f32 [H=7 W=7 C=576] {derived}] =
      convolution
        x=t908 {derived}
        weight=t810 {folded from=[p_features_14_conv_1_0_weight,p_features_14_conv_1_1_weight,b_features_14_conv_1_1_running_var]}
        bias=t811 {folded from=[p_features_14_conv_1_1_weight,p_features_14_conv_1_1_bias,b_features_14_conv_1_1_running_mean,b_features_14_conv_1_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=576}
    n491 {pt2=root[117] torch.ops.aten.hardtanh.default}: [t910 f32 [H=7 W=7
                                                                     C=576] {derived}] =
      hardtanh x=t909 {derived} params={min_val=0; max_val=6}
    n492 {derived}: [t911 f32 [H=7 W=7 C=160] {derived}] =
      convolution
        x=t910 {derived}
        weight=t812 {folded from=[p_features_14_conv_2_weight,p_features_14_conv_3_weight,b_features_14_conv_3_running_var]}
        bias=t813 {folded from=[p_features_14_conv_3_weight,p_features_14_conv_3_bias,b_features_14_conv_3_running_mean,b_features_14_conv_3_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n493 {derived}: [t912 f32 [H=7 W=7 C=960] {derived}] =
      convolution
        x=t911 {derived}
        weight=t814 {folded from=[p_features_15_conv_0_0_weight,p_features_15_conv_0_1_weight,b_features_15_conv_0_1_running_var]}
        bias=t815 {folded from=[p_features_15_conv_0_1_weight,p_features_15_conv_0_1_bias,b_features_15_conv_0_1_running_mean,b_features_15_conv_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n494 {pt2=root[122] torch.ops.aten.hardtanh.default}: [t913 f32 [H=7 W=7
                                                                     C=960] {derived}] =
      hardtanh x=t912 {derived} params={min_val=0; max_val=6}
    n495 {derived}: [t914 f32 [H=7 W=7 C=960] {derived}] =
      convolution
        x=t913 {derived}
        weight=t816 {folded from=[p_features_15_conv_1_0_weight,p_features_15_conv_1_1_weight,b_features_15_conv_1_1_running_var]}
        bias=t817 {folded from=[p_features_15_conv_1_1_weight,p_features_15_conv_1_1_bias,b_features_15_conv_1_1_running_mean,b_features_15_conv_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=960}
    n496 {pt2=root[125] torch.ops.aten.hardtanh.default}: [t915 f32 [H=7 W=7
                                                                     C=960] {derived}] =
      hardtanh x=t914 {derived} params={min_val=0; max_val=6}
    n497 {derived}: [t916 f32 [H=7 W=7 C=160] {derived}] =
      convolution
        x=t915 {derived}
        weight=t818 {folded from=[p_features_15_conv_2_weight,p_features_15_conv_3_weight,b_features_15_conv_3_running_var]}
        bias=t819 {folded from=[p_features_15_conv_3_weight,p_features_15_conv_3_bias,b_features_15_conv_3_running_mean,b_features_15_conv_3_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n498 {pt2=root[128] torch.ops.aten.add.Tensor}: [t917 f32 [H=7 W=7 C=160] {derived}] =
      add a=t911 {derived} b=t916 {derived}
    n499 {derived}: [t918 f32 [H=7 W=7 C=960] {derived}] =
      convolution
        x=t917 {derived}
        weight=t820 {folded from=[p_features_16_conv_0_0_weight,p_features_16_conv_0_1_weight,b_features_16_conv_0_1_running_var]}
        bias=t821 {folded from=[p_features_16_conv_0_1_weight,p_features_16_conv_0_1_bias,b_features_16_conv_0_1_running_mean,b_features_16_conv_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n500 {pt2=root[131] torch.ops.aten.hardtanh.default}: [t919 f32 [H=7 W=7
                                                                     C=960] {derived}] =
      hardtanh x=t918 {derived} params={min_val=0; max_val=6}
    n501 {derived}: [t920 f32 [H=7 W=7 C=960] {derived}] =
      convolution
        x=t919 {derived}
        weight=t822 {folded from=[p_features_16_conv_1_0_weight,p_features_16_conv_1_1_weight,b_features_16_conv_1_1_running_var]}
        bias=t823 {folded from=[p_features_16_conv_1_1_weight,p_features_16_conv_1_1_bias,b_features_16_conv_1_1_running_mean,b_features_16_conv_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=960}
    n502 {pt2=root[134] torch.ops.aten.hardtanh.default}: [t921 f32 [H=7 W=7
                                                                     C=960] {derived}] =
      hardtanh x=t920 {derived} params={min_val=0; max_val=6}
    n503 {derived}: [t922 f32 [H=7 W=7 C=160] {derived}] =
      convolution
        x=t921 {derived}
        weight=t824 {folded from=[p_features_16_conv_2_weight,p_features_16_conv_3_weight,b_features_16_conv_3_running_var]}
        bias=t825 {folded from=[p_features_16_conv_3_weight,p_features_16_conv_3_bias,b_features_16_conv_3_running_mean,b_features_16_conv_3_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n504 {pt2=root[137] torch.ops.aten.add.Tensor}: [t923 f32 [H=7 W=7 C=160] {derived}] =
      add a=t917 {derived} b=t922 {derived}
    n505 {derived}: [t924 f32 [H=7 W=7 C=960] {derived}] =
      convolution
        x=t923 {derived}
        weight=t826 {folded from=[p_features_17_conv_0_0_weight,p_features_17_conv_0_1_weight,b_features_17_conv_0_1_running_var]}
        bias=t827 {folded from=[p_features_17_conv_0_1_weight,p_features_17_conv_0_1_bias,b_features_17_conv_0_1_running_mean,b_features_17_conv_0_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n506 {pt2=root[140] torch.ops.aten.hardtanh.default}: [t925 f32 [H=7 W=7
                                                                     C=960] {derived}] =
      hardtanh x=t924 {derived} params={min_val=0; max_val=6}
    n507 {derived}: [t926 f32 [H=7 W=7 C=960] {derived}] =
      convolution
        x=t925 {derived}
        weight=t828 {folded from=[p_features_17_conv_1_0_weight,p_features_17_conv_1_1_weight,b_features_17_conv_1_1_running_var]}
        bias=t829 {folded from=[p_features_17_conv_1_1_weight,p_features_17_conv_1_1_bias,b_features_17_conv_1_1_running_mean,b_features_17_conv_1_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=960}
    n508 {pt2=root[143] torch.ops.aten.hardtanh.default}: [t927 f32 [H=7 W=7
                                                                     C=960] {derived}] =
      hardtanh x=t926 {derived} params={min_val=0; max_val=6}
    n509 {derived}: [t928 f32 [H=7 W=7 C=320] {derived}] =
      convolution
        x=t927 {derived}
        weight=t830 {folded from=[p_features_17_conv_2_weight,p_features_17_conv_3_weight,b_features_17_conv_3_running_var]}
        bias=t831 {folded from=[p_features_17_conv_3_weight,p_features_17_conv_3_bias,b_features_17_conv_3_running_mean,b_features_17_conv_3_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n510 {derived}: [t929 f32 [H=7 W=7 C=1280] {derived}] =
      convolution
        x=t928 {derived}
        weight=t832 {folded from=[p_features_18_0_weight,p_features_18_1_weight,b_features_18_1_running_var]}
        bias=t833 {folded from=[p_features_18_1_weight,p_features_18_1_bias,b_features_18_1_running_mean,b_features_18_1_running_var]}
        params={stride={h=1; w=1};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n511 {pt2=root[148] torch.ops.aten.hardtanh.default}: [t930 f32 [H=7 W=7
                                                                     C=1280] {derived}] =
      hardtanh x=t929 {derived} params={min_val=0; max_val=6}
    n512 {pt2=root[149] torch.ops.aten.mean.dim}: [t931 f32 [C=1280] {derived}] =
      mean x=t930 {derived} params={dims=[W, H]; keepdim=true}
    n513 {pt2=root[151] torch.ops.aten.clone.default}: [t932 f32 [C=1280] {pt2=root:clone}] =
      clone x=t931 {derived}
    group g105 torch.ops.aten.addmm.default:
      n414 {pt2=root[153] torch.ops.aten.addmm.default}: [t729 f32 [C=1000] {pt2=root:addmm}] =
        linear
          x=t932 {pt2=root:clone}
          weight=t728 {folded from=[p_classifier_1_weight]}
          bias=t157 {pt2=root:p_classifier_1_bias target=classifier.1.bias}
          params={in_features=1280}
  outputs: [t729 f32 [C=1000] {pt2=root:addmm}]
