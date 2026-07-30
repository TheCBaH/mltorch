The same symbolic verification with `--fold`, so constant folding and batch-norm
folding also run. Gated on PT2_DATA; run with `make pt2.runtest` after
`make pt2.download-cram`.

Folding is what makes the constant-shaped clusters checkable, and the contrast
with the structural run is the point of having both: with payloads bound, a fold
can be compared against the number the pass computed, and those clusters come
back `proved (for these constants)` — a weaker claim than `proved (structural)`
and deliberately labelled apart. It holds for every input, but only for the
weights this model carries, where a structural proof holds for any payload at
all.

`tested` is not a proof either. A batch-norm fold re-associates, so the two
sides agree as polynomials but not bit for bit; the honest verdict is that their
coefficients agree within a tolerance. Only `Identical` claims can be refuted
outright, which is why a disagreement here would be reported as evidence rather
than as a counterexample.

As above, every tensor and node carries its claim inline — with one limit worth
reading carefully. A folded constant is a CREATION in the composed
origin-to-final map (`{} -> {t297}`), and a creation claims nothing, so its
inline verdict is `vacuous origins=0` rather than the proof one might expect.

That is not the verifier declining to check it. The fold IS checked, by the pass
that performed it, and shows up as `proved (for these constants)` in the
fold_const summaries above. What the inline annotation can show is bounded by
what the composed map says, and end to end the composed map's honest statement
about a folded weight is that the destination graph has an edge the source did
not. The per-pass summary and the inline annotation answer different questions,
which is why both are printed.

  $ ../bin/native_graph.exe transform --fold --verify-symbolic quick \
  >   --pt2 "$PT2_DATA/resnet18/resnet18.pt2"
  nodes: 174 -> 49
  constants: 42, of which 41 folded
  symbolic verification: reshape_to_permute
    (root)
         81  proved (for these constants) [sampled 4]
          2  proved (structural) [sampled 4]
         48  unproved (too large)
         20  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         60  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
          1  unproved (too large)
    torch.ops.aten.convolution.default
         80  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: chain_permute
    (root)
         81  proved (for these constants) [sampled 4]
          2  proved (structural) [sampled 4]
         68  unproved (too large)
         21  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         40  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: trim_permute
    (root)
         81  proved (for these constants) [sampled 4]
          2  proved (structural) [sampled 4]
         48  unproved (too large)
    torch.ops.aten._native_batch_norm_legit_no_training.default
         40  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: sink_permute
    (root)
         81  proved (for these constants) [sampled 4]
          2  proved (structural) [sampled 4]
         48  unproved (too large)
         27  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         25  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: sink_permute
    (root)
         81  proved (for these constants) [sampled 4]
          2  proved (structural) [sampled 4]
         57  unproved (too large)
          6  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         25  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: reuse_permute
    (root)
         81  proved (for these constants) [sampled 4]
          2  proved (structural) [sampled 4]
         60  unproved (too large)
         10  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: sink_permute
    (root)
         81  proved (for these constants) [sampled 4]
          2  proved (structural) [sampled 4]
         60  unproved (too large)
         10  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: bypass_permute
    (root)
         81  proved (for these constants) [sampled 4]
          2  proved (structural) [sampled 4]
         49  unproved (too large)
         17  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         41  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          2  unproved (too large)
  symbolic verification: sink_permute_mean
    (root)
         81  proved (for these constants) [sampled 4]
          2  proved (structural) [sampled 4]
         48  unproved (too large)
          2  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         41  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          2  unproved (too large)
  symbolic verification: bypass_permute
    (root)
         81  proved (for these constants) [sampled 4]
          1  unproved (over max_rounds) [sampled 4]
         48  unproved (too large)
          1  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         41  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          2  unproved (too large)
  symbolic verification: dce
    (root)
         81  proved (for these constants) [sampled 4]
          1  proved (structural) [sampled 4]
         48  unproved (too large)
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         41  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          2  unproved (too large)
  symbolic verification: drop_pool_indices
    (root)
         81  proved (for these constants) [sampled 4]
          1  proved (structural) [sampled 4]
         48  unproved (too large)
          1  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         41  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          1  unproved (too large)
  symbolic verification: fold_const
    (root)
         81  proved (for these constants) [sampled 4]
          1  proved (structural) [sampled 4]
         47  unproved (too large)
         21  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         21  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          1  unproved (too large)
  symbolic verification: fold_batch_norm
    (root)
         81  proved (for these constants) [sampled 4]
          1  proved (structural) [sampled 4]
         67  unproved (too large)
        220  vacuous
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
          1  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          1  unproved (too large)
  symbolic verification: fold_const
    (root)
         81  proved (for these constants) [sampled 4]
        101  proved (structural) [sampled 4]
         87  unproved (too large)
         80  vacuous
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
          1  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          1  unproved (too large)
  symbolic verification: fold_const
    (root)
         81  proved (for these constants) [sampled 4]
         81  proved (structural) [sampled 4]
         87  unproved (too large)
         20  vacuous
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
          1  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          1  unproved (too large)
  symbolic verification: fold_const
    (root)
         61  proved (for these constants) [sampled 4]
         61  proved (structural) [sampled 4]
         87  unproved (too large)
         40  vacuous
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
          1  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          1  unproved (too large)
  symbolic verification: fold_const
    (root)
         61  proved (for these constants) [sampled 4]
         21  proved (structural) [sampled 4]
         87  unproved (too large)
         40  vacuous
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
          1  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          1  unproved (too large)
  symbolic verification: fold_const
    (root)
         21  proved (for these constants) [sampled 4]
          1  proved (structural) [sampled 4]
         67  unproved (too large)
         80  vacuous
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
          1  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          1  unproved (too large)
  symbolic verification: total
     1439  proved (for these constants) [sampled 4]
      306  proved (structural) [sampled 4]
        1  unproved (over max_rounds) [sampled 4]
     2226  unproved (too large)
      616  vacuous
  graph
  inputs:
    [t61 f32 [C=1000] {pt2=root:p_fc_bias target=fc.bias verify=proved (for these constants) [sampled 4]} ->[n173] constant,
     t122 f32 [H=3 W=224 C=224] {pt2=root:x verify=unproved (too large)} ->[n0],
     t295 f32 [N=1000 T=1 D=1 H=1 W=1 C=512] {folded from=[p_fc_weight] verify=unproved (too large)} ->[n173] constant,
     t297 f32 [N=64 T=1 D=1 H=7 W=7 C=3] {folded from=[p_conv1_weight,p_bn1_weight,b_bn1_running_var] verify=vacuous origins=0} ->[n174] constant,
     t298 f32 [C=64] {folded from=[p_bn1_weight,p_bn1_bias,b_bn1_running_mean,b_bn1_running_var] verify=vacuous origins=0} ->[n174] constant,
     t299 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_0_conv1_weight,p_layer1_0_bn1_weight,b_layer1_0_bn1_running_var] verify=vacuous origins=0} ->[n177] constant,
     t300 f32 [C=64] {folded from=[p_layer1_0_bn1_weight,p_layer1_0_bn1_bias,b_layer1_0_bn1_running_mean,b_layer1_0_bn1_running_var] verify=vacuous origins=0} ->[n177] constant,
     t301 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_0_conv2_weight,p_layer1_0_bn2_weight,b_layer1_0_bn2_running_var] verify=vacuous origins=0} ->[n179] constant,
     t302 f32 [C=64] {folded from=[p_layer1_0_bn2_weight,p_layer1_0_bn2_bias,b_layer1_0_bn2_running_mean,b_layer1_0_bn2_running_var] verify=vacuous origins=0} ->[n179] constant,
     t303 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_1_conv1_weight,p_layer1_1_bn1_weight,b_layer1_1_bn1_running_var] verify=vacuous origins=0} ->[n182] constant,
     t304 f32 [C=64] {folded from=[p_layer1_1_bn1_weight,p_layer1_1_bn1_bias,b_layer1_1_bn1_running_mean,b_layer1_1_bn1_running_var] verify=vacuous origins=0} ->[n182] constant,
     t305 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_1_conv2_weight,p_layer1_1_bn2_weight,b_layer1_1_bn2_running_var] verify=vacuous origins=0} ->[n184] constant,
     t306 f32 [C=64] {folded from=[p_layer1_1_bn2_weight,p_layer1_1_bn2_bias,b_layer1_1_bn2_running_mean,b_layer1_1_bn2_running_var] verify=vacuous origins=0} ->[n184] constant,
     t307 f32 [N=128 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer2_0_conv1_weight,p_layer2_0_bn1_weight,b_layer2_0_bn1_running_var] verify=vacuous origins=0} ->[n187] constant,
     t308 f32 [C=128] {folded from=[p_layer2_0_bn1_weight,p_layer2_0_bn1_bias,b_layer2_0_bn1_running_mean,b_layer2_0_bn1_running_var] verify=vacuous origins=0} ->[n187] constant,
     t309 f32 [N=128 T=1 D=1 H=1 W=1 C=64] {folded from=[p_layer2_0_downsample_0_weight,p_layer2_0_downsample_1_weight,b_layer2_0_downsample_1_running_var] verify=vacuous origins=0} ->[n188] constant,
     t310 f32 [C=128] {folded from=[p_layer2_0_downsample_1_weight,p_layer2_0_downsample_1_bias,b_layer2_0_downsample_1_running_mean,b_layer2_0_downsample_1_running_var] verify=vacuous origins=0} ->[n188] constant,
     t311 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer2_0_conv2_weight,p_layer2_0_bn2_weight,b_layer2_0_bn2_running_var] verify=vacuous origins=0} ->[n190] constant,
     t312 f32 [C=128] {folded from=[p_layer2_0_bn2_weight,p_layer2_0_bn2_bias,b_layer2_0_bn2_running_mean,b_layer2_0_bn2_running_var] verify=vacuous origins=0} ->[n190] constant,
     t313 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer2_1_conv1_weight,p_layer2_1_bn1_weight,b_layer2_1_bn1_running_var] verify=vacuous origins=0} ->[n193] constant,
     t314 f32 [C=128] {folded from=[p_layer2_1_bn1_weight,p_layer2_1_bn1_bias,b_layer2_1_bn1_running_mean,b_layer2_1_bn1_running_var] verify=vacuous origins=0} ->[n193] constant,
     t315 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer2_1_conv2_weight,p_layer2_1_bn2_weight,b_layer2_1_bn2_running_var] verify=vacuous origins=0} ->[n195] constant,
     t316 f32 [C=128] {folded from=[p_layer2_1_bn2_weight,p_layer2_1_bn2_bias,b_layer2_1_bn2_running_mean,b_layer2_1_bn2_running_var] verify=vacuous origins=0} ->[n195] constant,
     t317 f32 [N=256 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer3_0_conv1_weight,p_layer3_0_bn1_weight,b_layer3_0_bn1_running_var] verify=vacuous origins=0} ->[n198] constant,
     t318 f32 [C=256] {folded from=[p_layer3_0_bn1_weight,p_layer3_0_bn1_bias,b_layer3_0_bn1_running_mean,b_layer3_0_bn1_running_var] verify=vacuous origins=0} ->[n198] constant,
     t319 f32 [N=256 T=1 D=1 H=1 W=1 C=128] {folded from=[p_layer3_0_downsample_0_weight,p_layer3_0_downsample_1_weight,b_layer3_0_downsample_1_running_var] verify=vacuous origins=0} ->[n199] constant,
     t320 f32 [C=256] {folded from=[p_layer3_0_downsample_1_weight,p_layer3_0_downsample_1_bias,b_layer3_0_downsample_1_running_mean,b_layer3_0_downsample_1_running_var] verify=vacuous origins=0} ->[n199] constant,
     t321 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer3_0_conv2_weight,p_layer3_0_bn2_weight,b_layer3_0_bn2_running_var] verify=vacuous origins=0} ->[n201] constant,
     t322 f32 [C=256] {folded from=[p_layer3_0_bn2_weight,p_layer3_0_bn2_bias,b_layer3_0_bn2_running_mean,b_layer3_0_bn2_running_var] verify=vacuous origins=0} ->[n201] constant,
     t323 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer3_1_conv1_weight,p_layer3_1_bn1_weight,b_layer3_1_bn1_running_var] verify=vacuous origins=0} ->[n204] constant,
     t324 f32 [C=256] {folded from=[p_layer3_1_bn1_weight,p_layer3_1_bn1_bias,b_layer3_1_bn1_running_mean,b_layer3_1_bn1_running_var] verify=vacuous origins=0} ->[n204] constant,
     t325 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer3_1_conv2_weight,p_layer3_1_bn2_weight,b_layer3_1_bn2_running_var] verify=vacuous origins=0} ->[n206] constant,
     t326 f32 [C=256] {folded from=[p_layer3_1_bn2_weight,p_layer3_1_bn2_bias,b_layer3_1_bn2_running_mean,b_layer3_1_bn2_running_var] verify=vacuous origins=0} ->[n206] constant,
     t327 f32 [N=512 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer4_0_conv1_weight,p_layer4_0_bn1_weight,b_layer4_0_bn1_running_var] verify=vacuous origins=0} ->[n209] constant,
     t328 f32 [C=512] {folded from=[p_layer4_0_bn1_weight,p_layer4_0_bn1_bias,b_layer4_0_bn1_running_mean,b_layer4_0_bn1_running_var] verify=vacuous origins=0} ->[n209] constant,
     t329 f32 [N=512 T=1 D=1 H=1 W=1 C=256] {folded from=[p_layer4_0_downsample_0_weight,p_layer4_0_downsample_1_weight,b_layer4_0_downsample_1_running_var] verify=vacuous origins=0} ->[n210] constant,
     t330 f32 [C=512] {folded from=[p_layer4_0_downsample_1_weight,p_layer4_0_downsample_1_bias,b_layer4_0_downsample_1_running_mean,b_layer4_0_downsample_1_running_var] verify=vacuous origins=0} ->[n210] constant,
     t331 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {folded from=[p_layer4_0_conv2_weight,p_layer4_0_bn2_weight,b_layer4_0_bn2_running_var] verify=vacuous origins=0} ->[n212] constant,
     t332 f32 [C=512] {folded from=[p_layer4_0_bn2_weight,p_layer4_0_bn2_bias,b_layer4_0_bn2_running_mean,b_layer4_0_bn2_running_var] verify=vacuous origins=0} ->[n212] constant,
     t333 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {folded from=[p_layer4_1_conv1_weight,p_layer4_1_bn1_weight,b_layer4_1_bn1_running_var] verify=vacuous origins=0} ->[n215] constant,
     t334 f32 [C=512] {folded from=[p_layer4_1_bn1_weight,p_layer4_1_bn1_bias,b_layer4_1_bn1_running_mean,b_layer4_1_bn1_running_var] verify=vacuous origins=0} ->[n215] constant,
     t335 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {folded from=[p_layer4_1_conv2_weight,p_layer4_1_bn2_weight,b_layer4_1_bn2_running_var] verify=vacuous origins=0} ->[n217] constant,
     t336 f32 [C=512] {folded from=[p_layer4_1_bn2_weight,p_layer4_1_bn2_bias,b_layer4_1_bn2_running_mean,b_layer4_1_bn2_running_var] verify=vacuous origins=0} ->[n217] constant]
  nodes:
    group g1 torch.ops.aten.convolution.default:
      n0 {derived verify=unproved (too large)}: [t123 f32 [H=224 W=224 C=3] {derived verify=unproved (too large)} ->[n174]] =
        permute
          x=t122 {pt2=root:x verify=unproved (too large)}
          perm=[H<-W, W<-C, C<-H]
    n174 {derived verify=unproved (too large)}: [t337 f32 [H=112 W=112 C=64] {derived verify=unproved (too large)} ->[n175]] =
      convolution
        x=t123 {derived verify=unproved (too large)} <-n0
        weight=t297 {folded from=[p_conv1_weight,p_bn1_weight,b_bn1_running_var] verify=vacuous origins=0}
        bias=t298 {folded from=[p_bn1_weight,p_bn1_bias,b_bn1_running_mean,b_bn1_running_var] verify=vacuous origins=0}
        params={stride={h=2; w=2};
               padding={h=3; w=3};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n175 {pt2=root[2] torch.ops.aten.relu.default verify=unproved (too large)}: [t338 f32 [H=112
                                                                      W=112
                                                                      C=64] {derived verify=unproved (too large)} ->[n176]] =
      relu x=t337 {derived verify=unproved (too large)} <-n174
    group g3 torch.ops.aten.max_pool2d_with_indices.default:
      n176 {derived verify=unproved (too large)}: [t132 f32 [H=56 W=56 C=64] {derived verify=unproved (too large) origins=2} ->[n177,
                                                                      n180]] =
        max_pool2d
          x=t338 {derived verify=unproved (too large)} <-n175
          params={kernel={h=3; w=3}; stride={h=2; w=2}; pad={h=1; w=1}}
    n177 {derived verify=unproved (too large)}: [t339 f32 [H=56 W=56 C=64] {derived verify=unproved (too large)} ->[n178]] =
      convolution
        x=t132 {derived verify=unproved (too large) origins=2} <-n176
        weight=t299 {folded from=[p_layer1_0_conv1_weight,p_layer1_0_bn1_weight,b_layer1_0_bn1_running_var] verify=vacuous origins=0}
        bias=t300 {folded from=[p_layer1_0_bn1_weight,p_layer1_0_bn1_bias,b_layer1_0_bn1_running_mean,b_layer1_0_bn1_running_var] verify=vacuous origins=0}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n178 {pt2=root[6] torch.ops.aten.relu.default verify=unproved (too large)}: [t340 f32 [H=56
                                                                      W=56
                                                                      C=64] {derived verify=unproved (too large)} ->[n179]] =
      relu x=t339 {derived verify=unproved (too large)} <-n177
    n179 {derived verify=unproved (too large)}: [t341 f32 [H=56 W=56 C=64] {derived verify=unproved (too large)} ->[n180]] =
      convolution
        x=t340 {derived verify=unproved (too large)} <-n178
        weight=t301 {folded from=[p_layer1_0_conv2_weight,p_layer1_0_bn2_weight,b_layer1_0_bn2_running_var] verify=vacuous origins=0}
        bias=t302 {folded from=[p_layer1_0_bn2_weight,p_layer1_0_bn2_bias,b_layer1_0_bn2_running_mean,b_layer1_0_bn2_running_var] verify=vacuous origins=0}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n180 {pt2=root[9] torch.ops.aten.add.Tensor verify=vacuous}: [t342 f32 [H=56
                                                                      W=56
                                                                      C=64] {derived verify=vacuous origins=0} ->[n181]] =
      add
        a=t341 {derived verify=unproved (too large)} <-n179
        b=t132 {derived verify=unproved (too large) origins=2} <-n176
    n181 {pt2=root[10] torch.ops.aten.relu.default verify=unproved (too large)}: [t343 f32 [H=56
                                                                      W=56
                                                                      C=64] {derived verify=unproved (too large)} ->[n182,
                                                                      n185]] =
      relu x=t342 {derived verify=vacuous origins=0} <-n180
    n182 {derived verify=unproved (too large)}: [t344 f32 [H=56 W=56 C=64] {derived verify=unproved (too large)} ->[n183]] =
      convolution
        x=t343 {derived verify=unproved (too large)} <-n181
        weight=t303 {folded from=[p_layer1_1_conv1_weight,p_layer1_1_bn1_weight,b_layer1_1_bn1_running_var] verify=vacuous origins=0}
        bias=t304 {folded from=[p_layer1_1_bn1_weight,p_layer1_1_bn1_bias,b_layer1_1_bn1_running_mean,b_layer1_1_bn1_running_var] verify=vacuous origins=0}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n183 {pt2=root[13] torch.ops.aten.relu.default verify=unproved (too large)}: [t345 f32 [H=56
                                                                      W=56
                                                                      C=64] {derived verify=unproved (too large)} ->[n184]] =
      relu x=t344 {derived verify=unproved (too large)} <-n182
    n184 {derived verify=unproved (too large)}: [t346 f32 [H=56 W=56 C=64] {derived verify=unproved (too large)} ->[n185]] =
      convolution
        x=t345 {derived verify=unproved (too large)} <-n183
        weight=t305 {folded from=[p_layer1_1_conv2_weight,p_layer1_1_bn2_weight,b_layer1_1_bn2_running_var] verify=vacuous origins=0}
        bias=t306 {folded from=[p_layer1_1_bn2_weight,p_layer1_1_bn2_bias,b_layer1_1_bn2_running_mean,b_layer1_1_bn2_running_var] verify=vacuous origins=0}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n185 {pt2=root[16] torch.ops.aten.add.Tensor verify=vacuous}: [t347 f32 [H=56
                                                                      W=56
                                                                      C=64] {derived verify=vacuous origins=0} ->[n186]] =
      add
        a=t346 {derived verify=unproved (too large)} <-n184
        b=t343 {derived verify=unproved (too large)} <-n181
    n186 {pt2=root[17] torch.ops.aten.relu.default verify=unproved (too large)}: [t348 f32 [H=56
                                                                      W=56
                                                                      C=64] {derived verify=unproved (too large) origins=2} ->[n187,
                                                                      n188]] =
      relu x=t347 {derived verify=vacuous origins=0} <-n185
    n187 {derived verify=unproved (too large)}: [t349 f32 [H=28 W=28 C=128] {derived verify=unproved (too large)} ->[n189]] =
      convolution
        x=t348 {derived verify=unproved (too large) origins=2} <-n186
        weight=t307 {folded from=[p_layer2_0_conv1_weight,p_layer2_0_bn1_weight,b_layer2_0_bn1_running_var] verify=vacuous origins=0}
        bias=t308 {folded from=[p_layer2_0_bn1_weight,p_layer2_0_bn1_bias,b_layer2_0_bn1_running_mean,b_layer2_0_bn1_running_var] verify=vacuous origins=0}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n188 {derived verify=unproved (too large)}: [t350 f32 [H=28 W=28 C=128] {derived verify=unproved (too large)} ->[n191]] =
      convolution
        x=t348 {derived verify=unproved (too large) origins=2} <-n186
        weight=t309 {folded from=[p_layer2_0_downsample_0_weight,p_layer2_0_downsample_1_weight,b_layer2_0_downsample_1_running_var] verify=vacuous origins=0}
        bias=t310 {folded from=[p_layer2_0_downsample_1_weight,p_layer2_0_downsample_1_bias,b_layer2_0_downsample_1_running_mean,b_layer2_0_downsample_1_running_var] verify=vacuous origins=0}
        params={stride={h=2; w=2};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n189 {pt2=root[20] torch.ops.aten.relu.default verify=unproved (too large)}: [t351 f32 [H=28
                                                                      W=28
                                                                      C=128] {derived verify=unproved (too large)} ->[n190]] =
      relu x=t349 {derived verify=unproved (too large)} <-n187
    n190 {derived verify=unproved (too large)}: [t352 f32 [H=28 W=28 C=128] {derived verify=unproved (too large)} ->[n191]] =
      convolution
        x=t351 {derived verify=unproved (too large)} <-n189
        weight=t311 {folded from=[p_layer2_0_conv2_weight,p_layer2_0_bn2_weight,b_layer2_0_bn2_running_var] verify=vacuous origins=0}
        bias=t312 {folded from=[p_layer2_0_bn2_weight,p_layer2_0_bn2_bias,b_layer2_0_bn2_running_mean,b_layer2_0_bn2_running_var] verify=vacuous origins=0}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n191 {pt2=root[25] torch.ops.aten.add.Tensor verify=vacuous}: [t353 f32 [H=28
                                                                      W=28
                                                                      C=128] {derived verify=vacuous origins=0} ->[n192]] =
      add
        a=t352 {derived verify=unproved (too large)} <-n190
        b=t350 {derived verify=unproved (too large)} <-n188
    n192 {pt2=root[26] torch.ops.aten.relu.default verify=unproved (too large)}: [t354 f32 [H=28
                                                                      W=28
                                                                      C=128] {derived verify=unproved (too large)} ->[n193,
                                                                      n196]] =
      relu x=t353 {derived verify=vacuous origins=0} <-n191
    n193 {derived verify=unproved (too large)}: [t355 f32 [H=28 W=28 C=128] {derived verify=unproved (too large)} ->[n194]] =
      convolution
        x=t354 {derived verify=unproved (too large)} <-n192
        weight=t313 {folded from=[p_layer2_1_conv1_weight,p_layer2_1_bn1_weight,b_layer2_1_bn1_running_var] verify=vacuous origins=0}
        bias=t314 {folded from=[p_layer2_1_bn1_weight,p_layer2_1_bn1_bias,b_layer2_1_bn1_running_mean,b_layer2_1_bn1_running_var] verify=vacuous origins=0}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n194 {pt2=root[29] torch.ops.aten.relu.default verify=unproved (too large)}: [t356 f32 [H=28
                                                                      W=28
                                                                      C=128] {derived verify=unproved (too large)} ->[n195]] =
      relu x=t355 {derived verify=unproved (too large)} <-n193
    n195 {derived verify=unproved (too large)}: [t357 f32 [H=28 W=28 C=128] {derived verify=unproved (too large)} ->[n196]] =
      convolution
        x=t356 {derived verify=unproved (too large)} <-n194
        weight=t315 {folded from=[p_layer2_1_conv2_weight,p_layer2_1_bn2_weight,b_layer2_1_bn2_running_var] verify=vacuous origins=0}
        bias=t316 {folded from=[p_layer2_1_bn2_weight,p_layer2_1_bn2_bias,b_layer2_1_bn2_running_mean,b_layer2_1_bn2_running_var] verify=vacuous origins=0}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n196 {pt2=root[32] torch.ops.aten.add.Tensor verify=vacuous}: [t358 f32 [H=28
                                                                      W=28
                                                                      C=128] {derived verify=vacuous origins=0} ->[n197]] =
      add
        a=t357 {derived verify=unproved (too large)} <-n195
        b=t354 {derived verify=unproved (too large)} <-n192
    n197 {pt2=root[33] torch.ops.aten.relu.default verify=unproved (too large)}: [t359 f32 [H=28
                                                                      W=28
                                                                      C=128] {derived verify=unproved (too large) origins=2} ->[n198,
                                                                      n199]] =
      relu x=t358 {derived verify=vacuous origins=0} <-n196
    n198 {derived verify=unproved (too large)}: [t360 f32 [H=14 W=14 C=256] {derived verify=unproved (too large)} ->[n200]] =
      convolution
        x=t359 {derived verify=unproved (too large) origins=2} <-n197
        weight=t317 {folded from=[p_layer3_0_conv1_weight,p_layer3_0_bn1_weight,b_layer3_0_bn1_running_var] verify=vacuous origins=0}
        bias=t318 {folded from=[p_layer3_0_bn1_weight,p_layer3_0_bn1_bias,b_layer3_0_bn1_running_mean,b_layer3_0_bn1_running_var] verify=vacuous origins=0}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n199 {derived verify=unproved (too large)}: [t361 f32 [H=14 W=14 C=256] {derived verify=unproved (too large)} ->[n202]] =
      convolution
        x=t359 {derived verify=unproved (too large) origins=2} <-n197
        weight=t319 {folded from=[p_layer3_0_downsample_0_weight,p_layer3_0_downsample_1_weight,b_layer3_0_downsample_1_running_var] verify=vacuous origins=0}
        bias=t320 {folded from=[p_layer3_0_downsample_1_weight,p_layer3_0_downsample_1_bias,b_layer3_0_downsample_1_running_mean,b_layer3_0_downsample_1_running_var] verify=vacuous origins=0}
        params={stride={h=2; w=2};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n200 {pt2=root[36] torch.ops.aten.relu.default verify=unproved (too large)}: [t362 f32 [H=14
                                                                      W=14
                                                                      C=256] {derived verify=unproved (too large)} ->[n201]] =
      relu x=t360 {derived verify=unproved (too large)} <-n198
    n201 {derived verify=unproved (too large)}: [t363 f32 [H=14 W=14 C=256] {derived verify=unproved (too large)} ->[n202]] =
      convolution
        x=t362 {derived verify=unproved (too large)} <-n200
        weight=t321 {folded from=[p_layer3_0_conv2_weight,p_layer3_0_bn2_weight,b_layer3_0_bn2_running_var] verify=vacuous origins=0}
        bias=t322 {folded from=[p_layer3_0_bn2_weight,p_layer3_0_bn2_bias,b_layer3_0_bn2_running_mean,b_layer3_0_bn2_running_var] verify=vacuous origins=0}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n202 {pt2=root[41] torch.ops.aten.add.Tensor verify=vacuous}: [t364 f32 [H=14
                                                                      W=14
                                                                      C=256] {derived verify=vacuous origins=0} ->[n203]] =
      add
        a=t363 {derived verify=unproved (too large)} <-n201
        b=t361 {derived verify=unproved (too large)} <-n199
    n203 {pt2=root[42] torch.ops.aten.relu.default verify=unproved (too large)}: [t365 f32 [H=14
                                                                      W=14
                                                                      C=256] {derived verify=unproved (too large)} ->[n204,
                                                                      n207]] =
      relu x=t364 {derived verify=vacuous origins=0} <-n202
    n204 {derived verify=unproved (too large)}: [t366 f32 [H=14 W=14 C=256] {derived verify=unproved (too large)} ->[n205]] =
      convolution
        x=t365 {derived verify=unproved (too large)} <-n203
        weight=t323 {folded from=[p_layer3_1_conv1_weight,p_layer3_1_bn1_weight,b_layer3_1_bn1_running_var] verify=vacuous origins=0}
        bias=t324 {folded from=[p_layer3_1_bn1_weight,p_layer3_1_bn1_bias,b_layer3_1_bn1_running_mean,b_layer3_1_bn1_running_var] verify=vacuous origins=0}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n205 {pt2=root[45] torch.ops.aten.relu.default verify=unproved (too large)}: [t367 f32 [H=14
                                                                      W=14
                                                                      C=256] {derived verify=unproved (too large)} ->[n206]] =
      relu x=t366 {derived verify=unproved (too large)} <-n204
    n206 {derived verify=unproved (too large)}: [t368 f32 [H=14 W=14 C=256] {derived verify=unproved (too large)} ->[n207]] =
      convolution
        x=t367 {derived verify=unproved (too large)} <-n205
        weight=t325 {folded from=[p_layer3_1_conv2_weight,p_layer3_1_bn2_weight,b_layer3_1_bn2_running_var] verify=vacuous origins=0}
        bias=t326 {folded from=[p_layer3_1_bn2_weight,p_layer3_1_bn2_bias,b_layer3_1_bn2_running_mean,b_layer3_1_bn2_running_var] verify=vacuous origins=0}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n207 {pt2=root[48] torch.ops.aten.add.Tensor verify=vacuous}: [t369 f32 [H=14
                                                                      W=14
                                                                      C=256] {derived verify=vacuous origins=0} ->[n208]] =
      add
        a=t368 {derived verify=unproved (too large)} <-n206
        b=t365 {derived verify=unproved (too large)} <-n203
    n208 {pt2=root[49] torch.ops.aten.relu.default verify=unproved (too large)}: [t370 f32 [H=14
                                                                      W=14
                                                                      C=256] {derived verify=unproved (too large) origins=2} ->[n209,
                                                                      n210]] =
      relu x=t369 {derived verify=vacuous origins=0} <-n207
    n209 {derived verify=unproved (too large)}: [t371 f32 [H=7 W=7 C=512] {derived verify=unproved (too large)} ->[n211]] =
      convolution
        x=t370 {derived verify=unproved (too large) origins=2} <-n208
        weight=t327 {folded from=[p_layer4_0_conv1_weight,p_layer4_0_bn1_weight,b_layer4_0_bn1_running_var] verify=vacuous origins=0}
        bias=t328 {folded from=[p_layer4_0_bn1_weight,p_layer4_0_bn1_bias,b_layer4_0_bn1_running_mean,b_layer4_0_bn1_running_var] verify=vacuous origins=0}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n210 {derived verify=unproved (too large)}: [t372 f32 [H=7 W=7 C=512] {derived verify=unproved (too large)} ->[n213]] =
      convolution
        x=t370 {derived verify=unproved (too large) origins=2} <-n208
        weight=t329 {folded from=[p_layer4_0_downsample_0_weight,p_layer4_0_downsample_1_weight,b_layer4_0_downsample_1_running_var] verify=vacuous origins=0}
        bias=t330 {folded from=[p_layer4_0_downsample_1_weight,p_layer4_0_downsample_1_bias,b_layer4_0_downsample_1_running_mean,b_layer4_0_downsample_1_running_var] verify=vacuous origins=0}
        params={stride={h=2; w=2};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n211 {pt2=root[52] torch.ops.aten.relu.default verify=unproved (too large)}: [t373 f32 [H=7
                                                                      W=7
                                                                      C=512] {derived verify=unproved (too large)} ->[n212]] =
      relu x=t371 {derived verify=unproved (too large)} <-n209
    n212 {derived verify=unproved (too large)}: [t374 f32 [H=7 W=7 C=512] {derived verify=unproved (too large)} ->[n213]] =
      convolution
        x=t373 {derived verify=unproved (too large)} <-n211
        weight=t331 {folded from=[p_layer4_0_conv2_weight,p_layer4_0_bn2_weight,b_layer4_0_bn2_running_var] verify=vacuous origins=0}
        bias=t332 {folded from=[p_layer4_0_bn2_weight,p_layer4_0_bn2_bias,b_layer4_0_bn2_running_mean,b_layer4_0_bn2_running_var] verify=vacuous origins=0}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n213 {pt2=root[57] torch.ops.aten.add.Tensor verify=vacuous}: [t375 f32 [H=7
                                                                      W=7
                                                                      C=512] {derived verify=vacuous origins=0} ->[n214]] =
      add
        a=t374 {derived verify=unproved (too large)} <-n212
        b=t372 {derived verify=unproved (too large)} <-n210
    n214 {pt2=root[58] torch.ops.aten.relu.default verify=unproved (too large)}: [t376 f32 [H=7
                                                                      W=7
                                                                      C=512] {derived verify=unproved (too large)} ->[n215,
                                                                      n218]] =
      relu x=t375 {derived verify=vacuous origins=0} <-n213
    n215 {derived verify=unproved (too large)}: [t377 f32 [H=7 W=7 C=512] {derived verify=unproved (too large)} ->[n216]] =
      convolution
        x=t376 {derived verify=unproved (too large)} <-n214
        weight=t333 {folded from=[p_layer4_1_conv1_weight,p_layer4_1_bn1_weight,b_layer4_1_bn1_running_var] verify=vacuous origins=0}
        bias=t334 {folded from=[p_layer4_1_bn1_weight,p_layer4_1_bn1_bias,b_layer4_1_bn1_running_mean,b_layer4_1_bn1_running_var] verify=vacuous origins=0}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n216 {pt2=root[61] torch.ops.aten.relu.default verify=unproved (too large)}: [t378 f32 [H=7
                                                                      W=7
                                                                      C=512] {derived verify=unproved (too large)} ->[n217]] =
      relu x=t377 {derived verify=unproved (too large)} <-n215
    n217 {derived verify=unproved (too large)}: [t379 f32 [H=7 W=7 C=512] {derived verify=unproved (too large)} ->[n218]] =
      convolution
        x=t378 {derived verify=unproved (too large)} <-n216
        weight=t335 {folded from=[p_layer4_1_conv2_weight,p_layer4_1_bn2_weight,b_layer4_1_bn2_running_var] verify=vacuous origins=0}
        bias=t336 {folded from=[p_layer4_1_bn2_weight,p_layer4_1_bn2_bias,b_layer4_1_bn2_running_mean,b_layer4_1_bn2_running_var] verify=vacuous origins=0}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n218 {pt2=root[64] torch.ops.aten.add.Tensor verify=vacuous}: [t380 f32 [H=7
                                                                      W=7
                                                                      C=512] {derived verify=vacuous origins=0} ->[n219]] =
      add
        a=t379 {derived verify=unproved (too large)} <-n217
        b=t376 {derived verify=unproved (too large)} <-n214
    n219 {pt2=root[65] torch.ops.aten.relu.default verify=vacuous}: [t381 f32 [H=7
                                                                      W=7
                                                                      C=512] {derived verify=vacuous origins=0} ->[n220]] =
      relu x=t380 {derived verify=vacuous origins=0} <-n218
    n220 {pt2=root[66] torch.ops.aten.mean.dim verify=unproved (over max_rounds) [sampled 4]}: [t382 f32 [C=512] {pt2=root:view verify=unproved (over max_rounds) [sampled 4]} ->[n173]] =
      mean
        x=t381 {derived verify=vacuous origins=0} <-n219
        params={dims=[W, H]; keepdim=true}
    group g42 torch.ops.aten.addmm.default:
      n173 {pt2=root[69] torch.ops.aten.addmm.default verify=proved (structural) [sampled 4]}: [t296 f32 [C=1000] {pt2=root:addmm verify=proved (structural) [sampled 4]}] =
        linear
          x=t382 {pt2=root:view verify=unproved (over max_rounds) [sampled 4]} <-n220
          weight=t295 {folded from=[p_fc_weight] verify=unproved (too large)}
          bias=t61 {pt2=root:p_fc_bias target=fc.bias verify=proved (for these constants) [sampled 4]}
          params={in_features=512}
  outputs:
    [t296 f32 [C=1000] {pt2=root:addmm verify=proved (structural) [sampled 4]} <-n173]
