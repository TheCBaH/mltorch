Symbolically verify RegNetX-002's structural rewrite: every pass's mapping is
checked against the state it came from, WITHOUT payloads for the graph inputs,
so a proved cluster is a statement about every input rather than about the one
tensor a numeric run happens to use. See .ai/native_transform_verify.md. Gated
on PT2_DATA; run with `make pt2.runtest` after `make pt2.download-cram`.
RegNetX-002 stands in for the retired resnet18 role model here.

This is the companion to native_transform_regnetx_002_cram.t, which pins the
resulting graph. Here the graph is dropped and only the verification is kept —
the two would otherwise duplicate a 300-line dump.

Results are per pass and per group, because a flat count over the model's
clusters says nothing about which part of the model was covered. The groups
are the PT2 call sites the importer recorded, so they name what a reader
recognises.

Read `unproved (too large)` as "not looked at": a whole activation tensor is
refused by the coordinate budget before any expansion, which is what stops the
verifier being pointed at a real model and hanging. The clusters that DO get
checked here are the layout-shaped ones the permute passes actually rewrite,
which is exactly where these passes could be wrong.

Every tensor and node carries its own claim inline, so the graph shows WHICH
nodes were verified rather than only how many. `verify=` on a node is the
weakest verdict over its outputs, since a node is only as verified as its
least-verified result. `origins=n` marks an edge that several origin edges
collapsed into — that is a node more than one pass rewrote — and `origins=0` an
edge a pass created outright.

The claims are the composed origin-to-final ones, not per-pass: those speak in
intermediate id spaces that no longer exist here, whereas the composed map's
destination ids ARE this graph's. Reading a node therefore gives what the whole
pipeline established about it, not a pile of intermediate steps.

The group tree the dump already prints is what turns this into per-group
results: clusters are attributed to those same groups in the summaries above,
so a group reads the same way in both.

  $ ../bin/native_graph.exe transform --verify-symbolic quick \
  >   --pt2 "$PT2_DATA/regnetx_002/regnetx_002.pt2"
  nodes: 368 -> 101
  constants: 90, of which 89 folded
  symbolic verification: canonical/reshape_to_permute#0
    (root)
        188  proved (structural) [sampled 4]
         90  unproved (too large)
         44  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
        132  unproved (too large)
    torch.ops.aten.adaptive_avg_pool2d.default
          2  proved (structural) [sampled 4]
          1  unproved (too large)
    torch.ops.aten.conv2d.default
          9  proved (structural) [sampled 4]
        167  unproved (too large)
    torch.ops.aten.linear.default
          1  proved (structural) [sampled 4]
          1  unproved (too large)
  symbolic verification: canonical/relayout[0]/relayout/chain_permute[0]/chain_permute#1
    (root)
        187  proved (structural) [sampled 4]
          1  unproved (over max_rounds) [sampled 4]
        134  unproved (too large)
         45  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         88  unproved (too large)
    torch.ops.aten.adaptive_avg_pool2d.default
          1  proved (structural) [sampled 4]
          1  unproved (too large)
    torch.ops.aten.conv2d.default
          9  proved (structural) [sampled 4]
        123  unproved (too large)
    torch.ops.aten.linear.default
          1  proved (structural) [sampled 4]
          1  unproved (too large)
  symbolic verification: canonical/relayout[0]/relayout/trim_permute[0]/trim_permute#2
    (root)
        187  proved (structural) [sampled 4]
         90  unproved (too large)
    torch.ops.aten._native_batch_norm_legit_no_training.default
         88  unproved (too large)
    torch.ops.aten.adaptive_avg_pool2d.default
          1  unproved (over max_rounds) [sampled 4]
          1  unproved (too large)
    torch.ops.aten.conv2d.default
          9  proved (structural) [sampled 4]
        123  unproved (too large)
    torch.ops.aten.linear.default
          1  proved (structural) [sampled 4]
          1  unproved (too large)
  symbolic verification: canonical/relayout[0]/relayout/sink_permute[0]/sink_permute#3
    (root)
        187  proved (structural) [sampled 4]
         90  unproved (too large)
         66  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         53  unproved (too large)
    torch.ops.aten.adaptive_avg_pool2d.default
          1  proved (structural) [sampled 4]
          1  unproved (too large)
    torch.ops.aten.conv2d.default
          9  proved (structural) [sampled 4]
        123  unproved (too large)
    torch.ops.aten.linear.default
          1  proved (structural) [sampled 4]
          1  unproved (too large)
  symbolic verification: canonical/relayout[0]/relayout/sink_permute[1]/sink_permute#4
    (root)
        187  proved (structural) [sampled 4]
        117  unproved (too large)
          8  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         53  unproved (too large)
    torch.ops.aten.adaptive_avg_pool2d.default
          1  proved (structural) [sampled 4]
          1  unproved (too large)
    torch.ops.aten.conv2d.default
          9  proved (structural) [sampled 4]
        123  unproved (too large)
    torch.ops.aten.linear.default
          1  proved (structural) [sampled 4]
          1  unproved (too large)
  symbolic verification: canonical/relayout[0]/relayout/reuse_permute[0]/reuse_permute#5
    (root)
        187  proved (structural) [sampled 4]
        121  unproved (too large)
         18  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         44  unproved (too large)
    torch.ops.aten.adaptive_avg_pool2d.default
          1  proved (structural) [sampled 4]
          1  unproved (too large)
    torch.ops.aten.conv2d.default
          9  proved (structural) [sampled 4]
        123  unproved (too large)
    torch.ops.aten.linear.default
          1  proved (structural) [sampled 4]
          1  unproved (too large)
  symbolic verification: canonical/relayout[0]/relayout/sink_permute[0]/sink_permute#6
    (root)
        187  proved (structural) [sampled 4]
        121  unproved (too large)
         18  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         44  unproved (too large)
    torch.ops.aten.adaptive_avg_pool2d.default
          1  proved (structural) [sampled 4]
          1  unproved (too large)
    torch.ops.aten.conv2d.default
          9  proved (structural) [sampled 4]
        123  unproved (too large)
    torch.ops.aten.linear.default
          1  proved (structural) [sampled 4]
          1  unproved (too large)
  symbolic verification: canonical/relayout[0]/relayout/bypass_permute[0]/bypass_permute#7
    (root)
        187  proved (structural) [sampled 4]
         90  unproved (too large)
         40  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         44  unproved (too large)
    torch.ops.aten.adaptive_avg_pool2d.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.conv2d.default
          9  proved (structural) [sampled 4]
         80  unproved (too large)
    torch.ops.aten.linear.default
          1  proved (structural) [sampled 4]
          1  unproved (too large)
  symbolic verification: canonical/fold_const[0]/fold_const#8
    (root)
        187  proved (structural) [sampled 4]
         90  unproved (too large)
         45  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         44  unproved (too large)
    torch.ops.aten.adaptive_avg_pool2d.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.conv2d.default
         45  unproved (too large)
    torch.ops.aten.linear.default
          1  proved (structural) [sampled 4]
  symbolic verification: canonical/fold_batch_norm#9
    (root)
        187  proved (structural) [sampled 4]
        134  unproved (too large)
        484  vacuous
    torch.ops.aten.adaptive_avg_pool2d.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.conv2d.default
          1  unproved (too large)
    torch.ops.aten.linear.default
          1  proved (structural) [sampled 4]
  symbolic verification: canonical/fold_const[0]/fold_const#10
    (root)
        416  proved (structural) [sampled 4]
        169  unproved (too large)
        176  vacuous
    torch.ops.aten.adaptive_avg_pool2d.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.conv2d.default
          1  unproved (too large)
    torch.ops.aten.linear.default
          1  proved (structural) [sampled 4]
  symbolic verification: canonical/fold_const[1]/fold_const#11
    (root)
        372  proved (structural) [sampled 4]
        169  unproved (too large)
         44  vacuous
    torch.ops.aten.adaptive_avg_pool2d.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.conv2d.default
          1  unproved (too large)
    torch.ops.aten.linear.default
          1  proved (structural) [sampled 4]
  symbolic verification: canonical/fold_const[2]/fold_const#12
    (root)
        284  proved (structural) [sampled 4]
        169  unproved (too large)
         88  vacuous
    torch.ops.aten.adaptive_avg_pool2d.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.conv2d.default
          1  unproved (too large)
    torch.ops.aten.linear.default
          1  proved (structural) [sampled 4]
  symbolic verification: canonical/fold_const[3]/fold_const#13
    (root)
        196  proved (structural) [sampled 4]
        169  unproved (too large)
         88  vacuous
    torch.ops.aten.adaptive_avg_pool2d.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.conv2d.default
          1  unproved (too large)
    torch.ops.aten.linear.default
          1  proved (structural) [sampled 4]
  symbolic verification: canonical/fold_const[4]/fold_const#14
    (root)
         55  proved (structural) [sampled 4]
        134  unproved (too large)
        176  vacuous
    torch.ops.aten.adaptive_avg_pool2d.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.conv2d.default
          1  unproved (too large)
    torch.ops.aten.linear.default
          1  proved (structural) [sampled 4]
  symbolic verification: total
     3296  proved (structural) [sampled 4]
        2  unproved (over max_rounds) [sampled 4]
     3528  unproved (too large)
     1340  vacuous
  graph
  inputs:
    [t133 f32 [C=1000] {pt2=root:p_head_fc_bias target=head.fc.bias verify=proved (structural) [sampled 4]} ->[n367] constant,
     t266 f32 [H=3 W=224 C=224] {pt2=root:x verify=unproved (too large)} ->[n0],
     t633 f32 [N=1000 T=1 D=1 H=1 W=1 C=368] {folded from=[p_head_fc_weight] verify=unproved (too large)} ->[n367] constant,
     t635 f32 [N=32 T=1 D=1 H=3 W=3 C=3] {folded from=[p_stem_conv_weight,p_stem_bn_weight,b_stem_bn_running_var] verify=vacuous origins=0} ->[n368] constant,
     t636 f32 [C=32] {folded from=[p_stem_bn_weight,p_stem_bn_bias,b_stem_bn_running_mean,b_stem_bn_running_var] verify=vacuous origins=0} ->[n368] constant,
     t637 f32 [N=24 T=1 D=1 H=1 W=1 C=32] {folded from=[p_s1_b1_conv1_conv_weight,p_s1_b1_conv1_bn_weight,b_s1_b1_conv1_bn_running_var] verify=vacuous origins=0} ->[n370] constant,
     t638 f32 [C=24] {folded from=[p_s1_b1_conv1_bn_weight,p_s1_b1_conv1_bn_bias,b_s1_b1_conv1_bn_running_mean,b_s1_b1_conv1_bn_running_var] verify=vacuous origins=0} ->[n370] constant,
     t639 f32 [N=24 T=1 D=1 H=1 W=1 C=32] {folded from=[p_s1_b1_downsample_conv_weight,p_s1_b1_downsample_bn_weight,b_s1_b1_downsample_bn_running_var] verify=vacuous origins=0} ->[n371] constant,
     t640 f32 [C=24] {folded from=[p_s1_b1_downsample_bn_weight,p_s1_b1_downsample_bn_bias,b_s1_b1_downsample_bn_running_mean,b_s1_b1_downsample_bn_running_var] verify=vacuous origins=0} ->[n371] constant,
     t641 f32 [N=24 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s1_b1_conv2_conv_weight,p_s1_b1_conv2_bn_weight,b_s1_b1_conv2_bn_running_var] verify=vacuous origins=0} ->[n373] constant,
     t642 f32 [C=24] {folded from=[p_s1_b1_conv2_bn_weight,p_s1_b1_conv2_bn_bias,b_s1_b1_conv2_bn_running_mean,b_s1_b1_conv2_bn_running_var] verify=vacuous origins=0} ->[n373] constant,
     t643 f32 [N=24 T=1 D=1 H=1 W=1 C=24] {folded from=[p_s1_b1_conv3_conv_weight,p_s1_b1_conv3_bn_weight,b_s1_b1_conv3_bn_running_var] verify=vacuous origins=0} ->[n375] constant,
     t644 f32 [C=24] {folded from=[p_s1_b1_conv3_bn_weight,p_s1_b1_conv3_bn_bias,b_s1_b1_conv3_bn_running_mean,b_s1_b1_conv3_bn_running_var] verify=vacuous origins=0} ->[n375] constant,
     t645 f32 [N=56 T=1 D=1 H=1 W=1 C=24] {folded from=[p_s2_b1_conv1_conv_weight,p_s2_b1_conv1_bn_weight,b_s2_b1_conv1_bn_running_var] verify=vacuous origins=0} ->[n378] constant,
     t646 f32 [C=56] {folded from=[p_s2_b1_conv1_bn_weight,p_s2_b1_conv1_bn_bias,b_s2_b1_conv1_bn_running_mean,b_s2_b1_conv1_bn_running_var] verify=vacuous origins=0} ->[n378] constant,
     t647 f32 [N=56 T=1 D=1 H=1 W=1 C=24] {folded from=[p_s2_b1_downsample_conv_weight,p_s2_b1_downsample_bn_weight,b_s2_b1_downsample_bn_running_var] verify=vacuous origins=0} ->[n379] constant,
     t648 f32 [C=56] {folded from=[p_s2_b1_downsample_bn_weight,p_s2_b1_downsample_bn_bias,b_s2_b1_downsample_bn_running_mean,b_s2_b1_downsample_bn_running_var] verify=vacuous origins=0} ->[n379] constant,
     t649 f32 [N=56 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s2_b1_conv2_conv_weight,p_s2_b1_conv2_bn_weight,b_s2_b1_conv2_bn_running_var] verify=vacuous origins=0} ->[n381] constant,
     t650 f32 [C=56] {folded from=[p_s2_b1_conv2_bn_weight,p_s2_b1_conv2_bn_bias,b_s2_b1_conv2_bn_running_mean,b_s2_b1_conv2_bn_running_var] verify=vacuous origins=0} ->[n381] constant,
     t651 f32 [N=56 T=1 D=1 H=1 W=1 C=56] {folded from=[p_s2_b1_conv3_conv_weight,p_s2_b1_conv3_bn_weight,b_s2_b1_conv3_bn_running_var] verify=vacuous origins=0} ->[n383] constant,
     t652 f32 [C=56] {folded from=[p_s2_b1_conv3_bn_weight,p_s2_b1_conv3_bn_bias,b_s2_b1_conv3_bn_running_mean,b_s2_b1_conv3_bn_running_var] verify=vacuous origins=0} ->[n383] constant,
     t653 f32 [N=152 T=1 D=1 H=1 W=1 C=56] {folded from=[p_s3_b1_conv1_conv_weight,p_s3_b1_conv1_bn_weight,b_s3_b1_conv1_bn_running_var] verify=vacuous origins=0} ->[n386] constant,
     t654 f32 [C=152] {folded from=[p_s3_b1_conv1_bn_weight,p_s3_b1_conv1_bn_bias,b_s3_b1_conv1_bn_running_mean,b_s3_b1_conv1_bn_running_var] verify=vacuous origins=0} ->[n386] constant,
     t655 f32 [N=152 T=1 D=1 H=1 W=1 C=56] {folded from=[p_s3_b1_downsample_conv_weight,p_s3_b1_downsample_bn_weight,b_s3_b1_downsample_bn_running_var] verify=vacuous origins=0} ->[n387] constant,
     t656 f32 [C=152] {folded from=[p_s3_b1_downsample_bn_weight,p_s3_b1_downsample_bn_bias,b_s3_b1_downsample_bn_running_mean,b_s3_b1_downsample_bn_running_var] verify=vacuous origins=0} ->[n387] constant,
     t657 f32 [N=152 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s3_b1_conv2_conv_weight,p_s3_b1_conv2_bn_weight,b_s3_b1_conv2_bn_running_var] verify=vacuous origins=0} ->[n389] constant,
     t658 f32 [C=152] {folded from=[p_s3_b1_conv2_bn_weight,p_s3_b1_conv2_bn_bias,b_s3_b1_conv2_bn_running_mean,b_s3_b1_conv2_bn_running_var] verify=vacuous origins=0} ->[n389] constant,
     t659 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s3_b1_conv3_conv_weight,p_s3_b1_conv3_bn_weight,b_s3_b1_conv3_bn_running_var] verify=vacuous origins=0} ->[n391] constant,
     t660 f32 [C=152] {folded from=[p_s3_b1_conv3_bn_weight,p_s3_b1_conv3_bn_bias,b_s3_b1_conv3_bn_running_mean,b_s3_b1_conv3_bn_running_var] verify=vacuous origins=0} ->[n391] constant,
     t661 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s3_b2_conv1_conv_weight,p_s3_b2_conv1_bn_weight,b_s3_b2_conv1_bn_running_var] verify=vacuous origins=0} ->[n394] constant,
     t662 f32 [C=152] {folded from=[p_s3_b2_conv1_bn_weight,p_s3_b2_conv1_bn_bias,b_s3_b2_conv1_bn_running_mean,b_s3_b2_conv1_bn_running_var] verify=vacuous origins=0} ->[n394] constant,
     t663 f32 [N=152 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s3_b2_conv2_conv_weight,p_s3_b2_conv2_bn_weight,b_s3_b2_conv2_bn_running_var] verify=vacuous origins=0} ->[n396] constant,
     t664 f32 [C=152] {folded from=[p_s3_b2_conv2_bn_weight,p_s3_b2_conv2_bn_bias,b_s3_b2_conv2_bn_running_mean,b_s3_b2_conv2_bn_running_var] verify=vacuous origins=0} ->[n396] constant,
     t665 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s3_b2_conv3_conv_weight,p_s3_b2_conv3_bn_weight,b_s3_b2_conv3_bn_running_var] verify=vacuous origins=0} ->[n398] constant,
     t666 f32 [C=152] {folded from=[p_s3_b2_conv3_bn_weight,p_s3_b2_conv3_bn_bias,b_s3_b2_conv3_bn_running_mean,b_s3_b2_conv3_bn_running_var] verify=vacuous origins=0} ->[n398] constant,
     t667 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s3_b3_conv1_conv_weight,p_s3_b3_conv1_bn_weight,b_s3_b3_conv1_bn_running_var] verify=vacuous origins=0} ->[n401] constant,
     t668 f32 [C=152] {folded from=[p_s3_b3_conv1_bn_weight,p_s3_b3_conv1_bn_bias,b_s3_b3_conv1_bn_running_mean,b_s3_b3_conv1_bn_running_var] verify=vacuous origins=0} ->[n401] constant,
     t669 f32 [N=152 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s3_b3_conv2_conv_weight,p_s3_b3_conv2_bn_weight,b_s3_b3_conv2_bn_running_var] verify=vacuous origins=0} ->[n403] constant,
     t670 f32 [C=152] {folded from=[p_s3_b3_conv2_bn_weight,p_s3_b3_conv2_bn_bias,b_s3_b3_conv2_bn_running_mean,b_s3_b3_conv2_bn_running_var] verify=vacuous origins=0} ->[n403] constant,
     t671 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s3_b3_conv3_conv_weight,p_s3_b3_conv3_bn_weight,b_s3_b3_conv3_bn_running_var] verify=vacuous origins=0} ->[n405] constant,
     t672 f32 [C=152] {folded from=[p_s3_b3_conv3_bn_weight,p_s3_b3_conv3_bn_bias,b_s3_b3_conv3_bn_running_mean,b_s3_b3_conv3_bn_running_var] verify=vacuous origins=0} ->[n405] constant,
     t673 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s3_b4_conv1_conv_weight,p_s3_b4_conv1_bn_weight,b_s3_b4_conv1_bn_running_var] verify=vacuous origins=0} ->[n408] constant,
     t674 f32 [C=152] {folded from=[p_s3_b4_conv1_bn_weight,p_s3_b4_conv1_bn_bias,b_s3_b4_conv1_bn_running_mean,b_s3_b4_conv1_bn_running_var] verify=vacuous origins=0} ->[n408] constant,
     t675 f32 [N=152 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s3_b4_conv2_conv_weight,p_s3_b4_conv2_bn_weight,b_s3_b4_conv2_bn_running_var] verify=vacuous origins=0} ->[n410] constant,
     t676 f32 [C=152] {folded from=[p_s3_b4_conv2_bn_weight,p_s3_b4_conv2_bn_bias,b_s3_b4_conv2_bn_running_mean,b_s3_b4_conv2_bn_running_var] verify=vacuous origins=0} ->[n410] constant,
     t677 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s3_b4_conv3_conv_weight,p_s3_b4_conv3_bn_weight,b_s3_b4_conv3_bn_running_var] verify=vacuous origins=0} ->[n412] constant,
     t678 f32 [C=152] {folded from=[p_s3_b4_conv3_bn_weight,p_s3_b4_conv3_bn_bias,b_s3_b4_conv3_bn_running_mean,b_s3_b4_conv3_bn_running_var] verify=vacuous origins=0} ->[n412] constant,
     t679 f32 [N=368 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s4_b1_conv1_conv_weight,p_s4_b1_conv1_bn_weight,b_s4_b1_conv1_bn_running_var] verify=vacuous origins=0} ->[n415] constant,
     t680 f32 [C=368] {folded from=[p_s4_b1_conv1_bn_weight,p_s4_b1_conv1_bn_bias,b_s4_b1_conv1_bn_running_mean,b_s4_b1_conv1_bn_running_var] verify=vacuous origins=0} ->[n415] constant,
     t681 f32 [N=368 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s4_b1_downsample_conv_weight,p_s4_b1_downsample_bn_weight,b_s4_b1_downsample_bn_running_var] verify=vacuous origins=0} ->[n416] constant,
     t682 f32 [C=368] {folded from=[p_s4_b1_downsample_bn_weight,p_s4_b1_downsample_bn_bias,b_s4_b1_downsample_bn_running_mean,b_s4_b1_downsample_bn_running_var] verify=vacuous origins=0} ->[n416] constant,
     t683 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s4_b1_conv2_conv_weight,p_s4_b1_conv2_bn_weight,b_s4_b1_conv2_bn_running_var] verify=vacuous origins=0} ->[n418] constant,
     t684 f32 [C=368] {folded from=[p_s4_b1_conv2_bn_weight,p_s4_b1_conv2_bn_bias,b_s4_b1_conv2_bn_running_mean,b_s4_b1_conv2_bn_running_var] verify=vacuous origins=0} ->[n418] constant,
     t685 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b1_conv3_conv_weight,p_s4_b1_conv3_bn_weight,b_s4_b1_conv3_bn_running_var] verify=vacuous origins=0} ->[n420] constant,
     t686 f32 [C=368] {folded from=[p_s4_b1_conv3_bn_weight,p_s4_b1_conv3_bn_bias,b_s4_b1_conv3_bn_running_mean,b_s4_b1_conv3_bn_running_var] verify=vacuous origins=0} ->[n420] constant,
     t687 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b2_conv1_conv_weight,p_s4_b2_conv1_bn_weight,b_s4_b2_conv1_bn_running_var] verify=vacuous origins=0} ->[n423] constant,
     t688 f32 [C=368] {folded from=[p_s4_b2_conv1_bn_weight,p_s4_b2_conv1_bn_bias,b_s4_b2_conv1_bn_running_mean,b_s4_b2_conv1_bn_running_var] verify=vacuous origins=0} ->[n423] constant,
     t689 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s4_b2_conv2_conv_weight,p_s4_b2_conv2_bn_weight,b_s4_b2_conv2_bn_running_var] verify=vacuous origins=0} ->[n425] constant,
     t690 f32 [C=368] {folded from=[p_s4_b2_conv2_bn_weight,p_s4_b2_conv2_bn_bias,b_s4_b2_conv2_bn_running_mean,b_s4_b2_conv2_bn_running_var] verify=vacuous origins=0} ->[n425] constant,
     t691 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b2_conv3_conv_weight,p_s4_b2_conv3_bn_weight,b_s4_b2_conv3_bn_running_var] verify=vacuous origins=0} ->[n427] constant,
     t692 f32 [C=368] {folded from=[p_s4_b2_conv3_bn_weight,p_s4_b2_conv3_bn_bias,b_s4_b2_conv3_bn_running_mean,b_s4_b2_conv3_bn_running_var] verify=vacuous origins=0} ->[n427] constant,
     t693 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b3_conv1_conv_weight,p_s4_b3_conv1_bn_weight,b_s4_b3_conv1_bn_running_var] verify=vacuous origins=0} ->[n430] constant,
     t694 f32 [C=368] {folded from=[p_s4_b3_conv1_bn_weight,p_s4_b3_conv1_bn_bias,b_s4_b3_conv1_bn_running_mean,b_s4_b3_conv1_bn_running_var] verify=vacuous origins=0} ->[n430] constant,
     t695 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s4_b3_conv2_conv_weight,p_s4_b3_conv2_bn_weight,b_s4_b3_conv2_bn_running_var] verify=vacuous origins=0} ->[n432] constant,
     t696 f32 [C=368] {folded from=[p_s4_b3_conv2_bn_weight,p_s4_b3_conv2_bn_bias,b_s4_b3_conv2_bn_running_mean,b_s4_b3_conv2_bn_running_var] verify=vacuous origins=0} ->[n432] constant,
     t697 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b3_conv3_conv_weight,p_s4_b3_conv3_bn_weight,b_s4_b3_conv3_bn_running_var] verify=vacuous origins=0} ->[n434] constant,
     t698 f32 [C=368] {folded from=[p_s4_b3_conv3_bn_weight,p_s4_b3_conv3_bn_bias,b_s4_b3_conv3_bn_running_mean,b_s4_b3_conv3_bn_running_var] verify=vacuous origins=0} ->[n434] constant,
     t699 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b4_conv1_conv_weight,p_s4_b4_conv1_bn_weight,b_s4_b4_conv1_bn_running_var] verify=vacuous origins=0} ->[n437] constant,
     t700 f32 [C=368] {folded from=[p_s4_b4_conv1_bn_weight,p_s4_b4_conv1_bn_bias,b_s4_b4_conv1_bn_running_mean,b_s4_b4_conv1_bn_running_var] verify=vacuous origins=0} ->[n437] constant,
     t701 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s4_b4_conv2_conv_weight,p_s4_b4_conv2_bn_weight,b_s4_b4_conv2_bn_running_var] verify=vacuous origins=0} ->[n439] constant,
     t702 f32 [C=368] {folded from=[p_s4_b4_conv2_bn_weight,p_s4_b4_conv2_bn_bias,b_s4_b4_conv2_bn_running_mean,b_s4_b4_conv2_bn_running_var] verify=vacuous origins=0} ->[n439] constant,
     t703 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b4_conv3_conv_weight,p_s4_b4_conv3_bn_weight,b_s4_b4_conv3_bn_running_var] verify=vacuous origins=0} ->[n441] constant,
     t704 f32 [C=368] {folded from=[p_s4_b4_conv3_bn_weight,p_s4_b4_conv3_bn_bias,b_s4_b4_conv3_bn_running_mean,b_s4_b4_conv3_bn_running_var] verify=vacuous origins=0} ->[n441] constant,
     t705 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b5_conv1_conv_weight,p_s4_b5_conv1_bn_weight,b_s4_b5_conv1_bn_running_var] verify=vacuous origins=0} ->[n444] constant,
     t706 f32 [C=368] {folded from=[p_s4_b5_conv1_bn_weight,p_s4_b5_conv1_bn_bias,b_s4_b5_conv1_bn_running_mean,b_s4_b5_conv1_bn_running_var] verify=vacuous origins=0} ->[n444] constant,
     t707 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s4_b5_conv2_conv_weight,p_s4_b5_conv2_bn_weight,b_s4_b5_conv2_bn_running_var] verify=vacuous origins=0} ->[n446] constant,
     t708 f32 [C=368] {folded from=[p_s4_b5_conv2_bn_weight,p_s4_b5_conv2_bn_bias,b_s4_b5_conv2_bn_running_mean,b_s4_b5_conv2_bn_running_var] verify=vacuous origins=0} ->[n446] constant,
     t709 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b5_conv3_conv_weight,p_s4_b5_conv3_bn_weight,b_s4_b5_conv3_bn_running_var] verify=vacuous origins=0} ->[n448] constant,
     t710 f32 [C=368] {folded from=[p_s4_b5_conv3_bn_weight,p_s4_b5_conv3_bn_bias,b_s4_b5_conv3_bn_running_mean,b_s4_b5_conv3_bn_running_var] verify=vacuous origins=0} ->[n448] constant,
     t711 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b6_conv1_conv_weight,p_s4_b6_conv1_bn_weight,b_s4_b6_conv1_bn_running_var] verify=vacuous origins=0} ->[n451] constant,
     t712 f32 [C=368] {folded from=[p_s4_b6_conv1_bn_weight,p_s4_b6_conv1_bn_bias,b_s4_b6_conv1_bn_running_mean,b_s4_b6_conv1_bn_running_var] verify=vacuous origins=0} ->[n451] constant,
     t713 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s4_b6_conv2_conv_weight,p_s4_b6_conv2_bn_weight,b_s4_b6_conv2_bn_running_var] verify=vacuous origins=0} ->[n453] constant,
     t714 f32 [C=368] {folded from=[p_s4_b6_conv2_bn_weight,p_s4_b6_conv2_bn_bias,b_s4_b6_conv2_bn_running_mean,b_s4_b6_conv2_bn_running_var] verify=vacuous origins=0} ->[n453] constant,
     t715 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b6_conv3_conv_weight,p_s4_b6_conv3_bn_weight,b_s4_b6_conv3_bn_running_var] verify=vacuous origins=0} ->[n455] constant,
     t716 f32 [C=368] {folded from=[p_s4_b6_conv3_bn_weight,p_s4_b6_conv3_bn_bias,b_s4_b6_conv3_bn_running_mean,b_s4_b6_conv3_bn_running_var] verify=vacuous origins=0} ->[n455] constant,
     t717 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b7_conv1_conv_weight,p_s4_b7_conv1_bn_weight,b_s4_b7_conv1_bn_running_var] verify=vacuous origins=0} ->[n458] constant,
     t718 f32 [C=368] {folded from=[p_s4_b7_conv1_bn_weight,p_s4_b7_conv1_bn_bias,b_s4_b7_conv1_bn_running_mean,b_s4_b7_conv1_bn_running_var] verify=vacuous origins=0} ->[n458] constant,
     t719 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s4_b7_conv2_conv_weight,p_s4_b7_conv2_bn_weight,b_s4_b7_conv2_bn_running_var] verify=vacuous origins=0} ->[n460] constant,
     t720 f32 [C=368] {folded from=[p_s4_b7_conv2_bn_weight,p_s4_b7_conv2_bn_bias,b_s4_b7_conv2_bn_running_mean,b_s4_b7_conv2_bn_running_var] verify=vacuous origins=0} ->[n460] constant,
     t721 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b7_conv3_conv_weight,p_s4_b7_conv3_bn_weight,b_s4_b7_conv3_bn_running_var] verify=vacuous origins=0} ->[n462] constant,
     t722 f32 [C=368] {folded from=[p_s4_b7_conv3_bn_weight,p_s4_b7_conv3_bn_bias,b_s4_b7_conv3_bn_running_mean,b_s4_b7_conv3_bn_running_var] verify=vacuous origins=0} ->[n462] constant]
  nodes:
    group g1 torch.ops.aten.conv2d.default:
      n0 {derived verify=unproved (too large)}: [t267 f32 [H=224 W=224 C=3] {derived verify=unproved (too large)} ->[n368]] =
        permute
          x=t266 {pt2=root:x verify=unproved (too large)}
          perm=[H<-W, W<-C, C<-H]
    n368 {derived verify=unproved (too large)}: [t723 f32 [H=112 W=112 C=32] {derived verify=unproved (too large)} ->[n369]] =
      conv2d
        x=t267 {derived verify=unproved (too large)} <-n0
        weight=t635 {folded from=[p_stem_conv_weight,p_stem_bn_weight,b_stem_bn_running_var] verify=vacuous origins=0}
        bias=t636 {folded from=[p_stem_bn_weight,p_stem_bn_bias,b_stem_bn_running_mean,b_stem_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=3;
               groups=1}
    n369 {pt2=root[2] torch.ops.aten.relu.default verify=unproved (too large)}: [t724 f32 [H=112
                                                                      W=112
                                                                      C=32] {derived verify=unproved (too large) origins=2} ->[n370,
                                                                      n371]] =
      relu x=t723 {derived verify=unproved (too large)} <-n368
    n370 {derived verify=unproved (too large)}: [t725 f32 [H=112 W=112 C=24] {derived verify=unproved (too large)} ->[n372]] =
      conv2d
        x=t724 {derived verify=unproved (too large) origins=2} <-n369
        weight=t637 {folded from=[p_s1_b1_conv1_conv_weight,p_s1_b1_conv1_bn_weight,b_s1_b1_conv1_bn_running_var] verify=vacuous origins=0}
        bias=t638 {folded from=[p_s1_b1_conv1_bn_weight,p_s1_b1_conv1_bn_bias,b_s1_b1_conv1_bn_running_mean,b_s1_b1_conv1_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=32;
               groups=1}
    n371 {derived verify=unproved (too large)}: [t726 f32 [H=56 W=56 C=24] {derived verify=unproved (too large)} ->[n376]] =
      conv2d
        x=t724 {derived verify=unproved (too large) origins=2} <-n369
        weight=t639 {folded from=[p_s1_b1_downsample_conv_weight,p_s1_b1_downsample_bn_weight,b_s1_b1_downsample_bn_running_var] verify=vacuous origins=0}
        bias=t640 {folded from=[p_s1_b1_downsample_bn_weight,p_s1_b1_downsample_bn_bias,b_s1_b1_downsample_bn_running_mean,b_s1_b1_downsample_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               in_channels=32;
               groups=1}
    n372 {pt2=root[5] torch.ops.aten.relu.default verify=unproved (too large)}: [t727 f32 [H=112
                                                                      W=112
                                                                      C=24] {derived verify=unproved (too large)} ->[n373]] =
      relu x=t725 {derived verify=unproved (too large)} <-n370
    n373 {derived verify=unproved (too large)}: [t728 f32 [H=56 W=56 C=24] {derived verify=unproved (too large)} ->[n374]] =
      conv2d
        x=t727 {derived verify=unproved (too large)} <-n372
        weight=t641 {folded from=[p_s1_b1_conv2_conv_weight,p_s1_b1_conv2_bn_weight,b_s1_b1_conv2_bn_running_var] verify=vacuous origins=0}
        bias=t642 {folded from=[p_s1_b1_conv2_bn_weight,p_s1_b1_conv2_bn_bias,b_s1_b1_conv2_bn_running_mean,b_s1_b1_conv2_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=24;
               groups=3}
    n374 {pt2=root[8] torch.ops.aten.relu.default verify=unproved (too large)}: [t729 f32 [H=56
                                                                      W=56
                                                                      C=24] {derived verify=unproved (too large)} ->[n375]] =
      relu x=t728 {derived verify=unproved (too large)} <-n373
    n375 {derived verify=unproved (too large)}: [t730 f32 [H=56 W=56 C=24] {derived verify=unproved (too large)} ->[n376]] =
      conv2d
        x=t729 {derived verify=unproved (too large)} <-n374
        weight=t643 {folded from=[p_s1_b1_conv3_conv_weight,p_s1_b1_conv3_bn_weight,b_s1_b1_conv3_bn_running_var] verify=vacuous origins=0}
        bias=t644 {folded from=[p_s1_b1_conv3_bn_weight,p_s1_b1_conv3_bn_bias,b_s1_b1_conv3_bn_running_mean,b_s1_b1_conv3_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=24;
               groups=1}
    n376 {pt2=root[13] torch.ops.aten.add.Tensor verify=vacuous}: [t731 f32 [H=56
                                                                      W=56
                                                                      C=24] {derived verify=vacuous origins=0} ->[n377]] =
      add
        a=t730 {derived verify=unproved (too large)} <-n375
        b=t726 {derived verify=unproved (too large)} <-n371
    n377 {pt2=root[14] torch.ops.aten.relu.default verify=unproved (too large)}: [t732 f32 [H=56
                                                                      W=56
                                                                      C=24] {derived verify=unproved (too large) origins=2} ->[n378,
                                                                      n379]] =
      relu x=t731 {derived verify=vacuous origins=0} <-n376
    n378 {derived verify=unproved (too large)}: [t733 f32 [H=56 W=56 C=56] {derived verify=unproved (too large)} ->[n380]] =
      conv2d
        x=t732 {derived verify=unproved (too large) origins=2} <-n377
        weight=t645 {folded from=[p_s2_b1_conv1_conv_weight,p_s2_b1_conv1_bn_weight,b_s2_b1_conv1_bn_running_var] verify=vacuous origins=0}
        bias=t646 {folded from=[p_s2_b1_conv1_bn_weight,p_s2_b1_conv1_bn_bias,b_s2_b1_conv1_bn_running_mean,b_s2_b1_conv1_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=24;
               groups=1}
    n379 {derived verify=unproved (too large)}: [t734 f32 [H=28 W=28 C=56] {derived verify=unproved (too large)} ->[n384]] =
      conv2d
        x=t732 {derived verify=unproved (too large) origins=2} <-n377
        weight=t647 {folded from=[p_s2_b1_downsample_conv_weight,p_s2_b1_downsample_bn_weight,b_s2_b1_downsample_bn_running_var] verify=vacuous origins=0}
        bias=t648 {folded from=[p_s2_b1_downsample_bn_weight,p_s2_b1_downsample_bn_bias,b_s2_b1_downsample_bn_running_mean,b_s2_b1_downsample_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               in_channels=24;
               groups=1}
    n380 {pt2=root[17] torch.ops.aten.relu.default verify=unproved (too large)}: [t735 f32 [H=56
                                                                      W=56
                                                                      C=56] {derived verify=unproved (too large)} ->[n381]] =
      relu x=t733 {derived verify=unproved (too large)} <-n378
    n381 {derived verify=unproved (too large)}: [t736 f32 [H=28 W=28 C=56] {derived verify=unproved (too large)} ->[n382]] =
      conv2d
        x=t735 {derived verify=unproved (too large)} <-n380
        weight=t649 {folded from=[p_s2_b1_conv2_conv_weight,p_s2_b1_conv2_bn_weight,b_s2_b1_conv2_bn_running_var] verify=vacuous origins=0}
        bias=t650 {folded from=[p_s2_b1_conv2_bn_weight,p_s2_b1_conv2_bn_bias,b_s2_b1_conv2_bn_running_mean,b_s2_b1_conv2_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=56;
               groups=7}
    n382 {pt2=root[20] torch.ops.aten.relu.default verify=unproved (too large)}: [t737 f32 [H=28
                                                                      W=28
                                                                      C=56] {derived verify=unproved (too large)} ->[n383]] =
      relu x=t736 {derived verify=unproved (too large)} <-n381
    n383 {derived verify=unproved (too large)}: [t738 f32 [H=28 W=28 C=56] {derived verify=unproved (too large)} ->[n384]] =
      conv2d
        x=t737 {derived verify=unproved (too large)} <-n382
        weight=t651 {folded from=[p_s2_b1_conv3_conv_weight,p_s2_b1_conv3_bn_weight,b_s2_b1_conv3_bn_running_var] verify=vacuous origins=0}
        bias=t652 {folded from=[p_s2_b1_conv3_bn_weight,p_s2_b1_conv3_bn_bias,b_s2_b1_conv3_bn_running_mean,b_s2_b1_conv3_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=56;
               groups=1}
    n384 {pt2=root[25] torch.ops.aten.add.Tensor verify=vacuous}: [t739 f32 [H=28
                                                                      W=28
                                                                      C=56] {derived verify=vacuous origins=0} ->[n385]] =
      add
        a=t738 {derived verify=unproved (too large)} <-n383
        b=t734 {derived verify=unproved (too large)} <-n379
    n385 {pt2=root[26] torch.ops.aten.relu.default verify=unproved (too large)}: [t740 f32 [H=28
                                                                      W=28
                                                                      C=56] {derived verify=unproved (too large) origins=2} ->[n386,
                                                                      n387]] =
      relu x=t739 {derived verify=vacuous origins=0} <-n384
    n386 {derived verify=unproved (too large)}: [t741 f32 [H=28 W=28 C=152] {derived verify=unproved (too large)} ->[n388]] =
      conv2d
        x=t740 {derived verify=unproved (too large) origins=2} <-n385
        weight=t653 {folded from=[p_s3_b1_conv1_conv_weight,p_s3_b1_conv1_bn_weight,b_s3_b1_conv1_bn_running_var] verify=vacuous origins=0}
        bias=t654 {folded from=[p_s3_b1_conv1_bn_weight,p_s3_b1_conv1_bn_bias,b_s3_b1_conv1_bn_running_mean,b_s3_b1_conv1_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=56;
               groups=1}
    n387 {derived verify=unproved (too large)}: [t742 f32 [H=14 W=14 C=152] {derived verify=unproved (too large)} ->[n392]] =
      conv2d
        x=t740 {derived verify=unproved (too large) origins=2} <-n385
        weight=t655 {folded from=[p_s3_b1_downsample_conv_weight,p_s3_b1_downsample_bn_weight,b_s3_b1_downsample_bn_running_var] verify=vacuous origins=0}
        bias=t656 {folded from=[p_s3_b1_downsample_bn_weight,p_s3_b1_downsample_bn_bias,b_s3_b1_downsample_bn_running_mean,b_s3_b1_downsample_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               in_channels=56;
               groups=1}
    n388 {pt2=root[29] torch.ops.aten.relu.default verify=unproved (too large)}: [t743 f32 [H=28
                                                                      W=28
                                                                      C=152] {derived verify=unproved (too large)} ->[n389]] =
      relu x=t741 {derived verify=unproved (too large)} <-n386
    n389 {derived verify=unproved (too large)}: [t744 f32 [H=14 W=14 C=152] {derived verify=unproved (too large)} ->[n390]] =
      conv2d
        x=t743 {derived verify=unproved (too large)} <-n388
        weight=t657 {folded from=[p_s3_b1_conv2_conv_weight,p_s3_b1_conv2_bn_weight,b_s3_b1_conv2_bn_running_var] verify=vacuous origins=0}
        bias=t658 {folded from=[p_s3_b1_conv2_bn_weight,p_s3_b1_conv2_bn_bias,b_s3_b1_conv2_bn_running_mean,b_s3_b1_conv2_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=152;
               groups=19}
    n390 {pt2=root[32] torch.ops.aten.relu.default verify=unproved (too large)}: [t745 f32 [H=14
                                                                      W=14
                                                                      C=152] {derived verify=unproved (too large)} ->[n391]] =
      relu x=t744 {derived verify=unproved (too large)} <-n389
    n391 {derived verify=unproved (too large)}: [t746 f32 [H=14 W=14 C=152] {derived verify=unproved (too large)} ->[n392]] =
      conv2d
        x=t745 {derived verify=unproved (too large)} <-n390
        weight=t659 {folded from=[p_s3_b1_conv3_conv_weight,p_s3_b1_conv3_bn_weight,b_s3_b1_conv3_bn_running_var] verify=vacuous origins=0}
        bias=t660 {folded from=[p_s3_b1_conv3_bn_weight,p_s3_b1_conv3_bn_bias,b_s3_b1_conv3_bn_running_mean,b_s3_b1_conv3_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n392 {pt2=root[37] torch.ops.aten.add.Tensor verify=vacuous}: [t747 f32 [H=14
                                                                      W=14
                                                                      C=152] {derived verify=vacuous origins=0} ->[n393]] =
      add
        a=t746 {derived verify=unproved (too large)} <-n391
        b=t742 {derived verify=unproved (too large)} <-n387
    n393 {pt2=root[38] torch.ops.aten.relu.default verify=unproved (too large)}: [t748 f32 [H=14
                                                                      W=14
                                                                      C=152] {derived verify=unproved (too large)} ->[n394,
                                                                      n399]] =
      relu x=t747 {derived verify=vacuous origins=0} <-n392
    n394 {derived verify=unproved (too large)}: [t749 f32 [H=14 W=14 C=152] {derived verify=unproved (too large)} ->[n395]] =
      conv2d
        x=t748 {derived verify=unproved (too large)} <-n393
        weight=t661 {folded from=[p_s3_b2_conv1_conv_weight,p_s3_b2_conv1_bn_weight,b_s3_b2_conv1_bn_running_var] verify=vacuous origins=0}
        bias=t662 {folded from=[p_s3_b2_conv1_bn_weight,p_s3_b2_conv1_bn_bias,b_s3_b2_conv1_bn_running_mean,b_s3_b2_conv1_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n395 {pt2=root[41] torch.ops.aten.relu.default verify=unproved (too large)}: [t750 f32 [H=14
                                                                      W=14
                                                                      C=152] {derived verify=unproved (too large)} ->[n396]] =
      relu x=t749 {derived verify=unproved (too large)} <-n394
    n396 {derived verify=unproved (too large)}: [t751 f32 [H=14 W=14 C=152] {derived verify=unproved (too large)} ->[n397]] =
      conv2d
        x=t750 {derived verify=unproved (too large)} <-n395
        weight=t663 {folded from=[p_s3_b2_conv2_conv_weight,p_s3_b2_conv2_bn_weight,b_s3_b2_conv2_bn_running_var] verify=vacuous origins=0}
        bias=t664 {folded from=[p_s3_b2_conv2_bn_weight,p_s3_b2_conv2_bn_bias,b_s3_b2_conv2_bn_running_mean,b_s3_b2_conv2_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=152;
               groups=19}
    n397 {pt2=root[44] torch.ops.aten.relu.default verify=unproved (too large)}: [t752 f32 [H=14
                                                                      W=14
                                                                      C=152] {derived verify=unproved (too large)} ->[n398]] =
      relu x=t751 {derived verify=unproved (too large)} <-n396
    n398 {derived verify=unproved (too large)}: [t753 f32 [H=14 W=14 C=152] {derived verify=unproved (too large)} ->[n399]] =
      conv2d
        x=t752 {derived verify=unproved (too large)} <-n397
        weight=t665 {folded from=[p_s3_b2_conv3_conv_weight,p_s3_b2_conv3_bn_weight,b_s3_b2_conv3_bn_running_var] verify=vacuous origins=0}
        bias=t666 {folded from=[p_s3_b2_conv3_bn_weight,p_s3_b2_conv3_bn_bias,b_s3_b2_conv3_bn_running_mean,b_s3_b2_conv3_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n399 {pt2=root[47] torch.ops.aten.add.Tensor verify=vacuous}: [t754 f32 [H=14
                                                                      W=14
                                                                      C=152] {derived verify=vacuous origins=0} ->[n400]] =
      add
        a=t753 {derived verify=unproved (too large)} <-n398
        b=t748 {derived verify=unproved (too large)} <-n393
    n400 {pt2=root[48] torch.ops.aten.relu.default verify=unproved (too large)}: [t755 f32 [H=14
                                                                      W=14
                                                                      C=152] {derived verify=unproved (too large)} ->[n401,
                                                                      n406]] =
      relu x=t754 {derived verify=vacuous origins=0} <-n399
    n401 {derived verify=unproved (too large)}: [t756 f32 [H=14 W=14 C=152] {derived verify=unproved (too large)} ->[n402]] =
      conv2d
        x=t755 {derived verify=unproved (too large)} <-n400
        weight=t667 {folded from=[p_s3_b3_conv1_conv_weight,p_s3_b3_conv1_bn_weight,b_s3_b3_conv1_bn_running_var] verify=vacuous origins=0}
        bias=t668 {folded from=[p_s3_b3_conv1_bn_weight,p_s3_b3_conv1_bn_bias,b_s3_b3_conv1_bn_running_mean,b_s3_b3_conv1_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n402 {pt2=root[51] torch.ops.aten.relu.default verify=unproved (too large)}: [t757 f32 [H=14
                                                                      W=14
                                                                      C=152] {derived verify=unproved (too large)} ->[n403]] =
      relu x=t756 {derived verify=unproved (too large)} <-n401
    n403 {derived verify=unproved (too large)}: [t758 f32 [H=14 W=14 C=152] {derived verify=unproved (too large)} ->[n404]] =
      conv2d
        x=t757 {derived verify=unproved (too large)} <-n402
        weight=t669 {folded from=[p_s3_b3_conv2_conv_weight,p_s3_b3_conv2_bn_weight,b_s3_b3_conv2_bn_running_var] verify=vacuous origins=0}
        bias=t670 {folded from=[p_s3_b3_conv2_bn_weight,p_s3_b3_conv2_bn_bias,b_s3_b3_conv2_bn_running_mean,b_s3_b3_conv2_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=152;
               groups=19}
    n404 {pt2=root[54] torch.ops.aten.relu.default verify=unproved (too large)}: [t759 f32 [H=14
                                                                      W=14
                                                                      C=152] {derived verify=unproved (too large)} ->[n405]] =
      relu x=t758 {derived verify=unproved (too large)} <-n403
    n405 {derived verify=unproved (too large)}: [t760 f32 [H=14 W=14 C=152] {derived verify=unproved (too large)} ->[n406]] =
      conv2d
        x=t759 {derived verify=unproved (too large)} <-n404
        weight=t671 {folded from=[p_s3_b3_conv3_conv_weight,p_s3_b3_conv3_bn_weight,b_s3_b3_conv3_bn_running_var] verify=vacuous origins=0}
        bias=t672 {folded from=[p_s3_b3_conv3_bn_weight,p_s3_b3_conv3_bn_bias,b_s3_b3_conv3_bn_running_mean,b_s3_b3_conv3_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n406 {pt2=root[57] torch.ops.aten.add.Tensor verify=vacuous}: [t761 f32 [H=14
                                                                      W=14
                                                                      C=152] {derived verify=vacuous origins=0} ->[n407]] =
      add
        a=t760 {derived verify=unproved (too large)} <-n405
        b=t755 {derived verify=unproved (too large)} <-n400
    n407 {pt2=root[58] torch.ops.aten.relu.default verify=unproved (too large)}: [t762 f32 [H=14
                                                                      W=14
                                                                      C=152] {derived verify=unproved (too large)} ->[n408,
                                                                      n413]] =
      relu x=t761 {derived verify=vacuous origins=0} <-n406
    n408 {derived verify=unproved (too large)}: [t763 f32 [H=14 W=14 C=152] {derived verify=unproved (too large)} ->[n409]] =
      conv2d
        x=t762 {derived verify=unproved (too large)} <-n407
        weight=t673 {folded from=[p_s3_b4_conv1_conv_weight,p_s3_b4_conv1_bn_weight,b_s3_b4_conv1_bn_running_var] verify=vacuous origins=0}
        bias=t674 {folded from=[p_s3_b4_conv1_bn_weight,p_s3_b4_conv1_bn_bias,b_s3_b4_conv1_bn_running_mean,b_s3_b4_conv1_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n409 {pt2=root[61] torch.ops.aten.relu.default verify=unproved (too large)}: [t764 f32 [H=14
                                                                      W=14
                                                                      C=152] {derived verify=unproved (too large)} ->[n410]] =
      relu x=t763 {derived verify=unproved (too large)} <-n408
    n410 {derived verify=unproved (too large)}: [t765 f32 [H=14 W=14 C=152] {derived verify=unproved (too large)} ->[n411]] =
      conv2d
        x=t764 {derived verify=unproved (too large)} <-n409
        weight=t675 {folded from=[p_s3_b4_conv2_conv_weight,p_s3_b4_conv2_bn_weight,b_s3_b4_conv2_bn_running_var] verify=vacuous origins=0}
        bias=t676 {folded from=[p_s3_b4_conv2_bn_weight,p_s3_b4_conv2_bn_bias,b_s3_b4_conv2_bn_running_mean,b_s3_b4_conv2_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=152;
               groups=19}
    n411 {pt2=root[64] torch.ops.aten.relu.default verify=unproved (too large)}: [t766 f32 [H=14
                                                                      W=14
                                                                      C=152] {derived verify=unproved (too large)} ->[n412]] =
      relu x=t765 {derived verify=unproved (too large)} <-n410
    n412 {derived verify=unproved (too large)}: [t767 f32 [H=14 W=14 C=152] {derived verify=unproved (too large)} ->[n413]] =
      conv2d
        x=t766 {derived verify=unproved (too large)} <-n411
        weight=t677 {folded from=[p_s3_b4_conv3_conv_weight,p_s3_b4_conv3_bn_weight,b_s3_b4_conv3_bn_running_var] verify=vacuous origins=0}
        bias=t678 {folded from=[p_s3_b4_conv3_bn_weight,p_s3_b4_conv3_bn_bias,b_s3_b4_conv3_bn_running_mean,b_s3_b4_conv3_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n413 {pt2=root[67] torch.ops.aten.add.Tensor verify=vacuous}: [t768 f32 [H=14
                                                                      W=14
                                                                      C=152] {derived verify=vacuous origins=0} ->[n414]] =
      add
        a=t767 {derived verify=unproved (too large)} <-n412
        b=t762 {derived verify=unproved (too large)} <-n407
    n414 {pt2=root[68] torch.ops.aten.relu.default verify=unproved (too large)}: [t769 f32 [H=14
                                                                      W=14
                                                                      C=152] {derived verify=unproved (too large) origins=2} ->[n415,
                                                                      n416]] =
      relu x=t768 {derived verify=vacuous origins=0} <-n413
    n415 {derived verify=unproved (too large)}: [t770 f32 [H=14 W=14 C=368] {derived verify=unproved (too large)} ->[n417]] =
      conv2d
        x=t769 {derived verify=unproved (too large) origins=2} <-n414
        weight=t679 {folded from=[p_s4_b1_conv1_conv_weight,p_s4_b1_conv1_bn_weight,b_s4_b1_conv1_bn_running_var] verify=vacuous origins=0}
        bias=t680 {folded from=[p_s4_b1_conv1_bn_weight,p_s4_b1_conv1_bn_bias,b_s4_b1_conv1_bn_running_mean,b_s4_b1_conv1_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n416 {derived verify=unproved (too large)}: [t771 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n421]] =
      conv2d
        x=t769 {derived verify=unproved (too large) origins=2} <-n414
        weight=t681 {folded from=[p_s4_b1_downsample_conv_weight,p_s4_b1_downsample_bn_weight,b_s4_b1_downsample_bn_running_var] verify=vacuous origins=0}
        bias=t682 {folded from=[p_s4_b1_downsample_bn_weight,p_s4_b1_downsample_bn_bias,b_s4_b1_downsample_bn_running_mean,b_s4_b1_downsample_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n417 {pt2=root[71] torch.ops.aten.relu.default verify=unproved (too large)}: [t772 f32 [H=14
                                                                      W=14
                                                                      C=368] {derived verify=unproved (too large)} ->[n418]] =
      relu x=t770 {derived verify=unproved (too large)} <-n415
    n418 {derived verify=unproved (too large)}: [t773 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n419]] =
      conv2d
        x=t772 {derived verify=unproved (too large)} <-n417
        weight=t683 {folded from=[p_s4_b1_conv2_conv_weight,p_s4_b1_conv2_bn_weight,b_s4_b1_conv2_bn_running_var] verify=vacuous origins=0}
        bias=t684 {folded from=[p_s4_b1_conv2_bn_weight,p_s4_b1_conv2_bn_bias,b_s4_b1_conv2_bn_running_mean,b_s4_b1_conv2_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=368;
               groups=46}
    n419 {pt2=root[74] torch.ops.aten.relu.default verify=unproved (too large)}: [t774 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n420]] =
      relu x=t773 {derived verify=unproved (too large)} <-n418
    n420 {derived verify=unproved (too large)}: [t775 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n421]] =
      conv2d
        x=t774 {derived verify=unproved (too large)} <-n419
        weight=t685 {folded from=[p_s4_b1_conv3_conv_weight,p_s4_b1_conv3_bn_weight,b_s4_b1_conv3_bn_running_var] verify=vacuous origins=0}
        bias=t686 {folded from=[p_s4_b1_conv3_bn_weight,p_s4_b1_conv3_bn_bias,b_s4_b1_conv3_bn_running_mean,b_s4_b1_conv3_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n421 {pt2=root[79] torch.ops.aten.add.Tensor verify=vacuous}: [t776 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=vacuous origins=0} ->[n422]] =
      add
        a=t775 {derived verify=unproved (too large)} <-n420
        b=t771 {derived verify=unproved (too large)} <-n416
    n422 {pt2=root[80] torch.ops.aten.relu.default verify=unproved (too large)}: [t777 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n423,
                                                                      n428]] =
      relu x=t776 {derived verify=vacuous origins=0} <-n421
    n423 {derived verify=unproved (too large)}: [t778 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n424]] =
      conv2d
        x=t777 {derived verify=unproved (too large)} <-n422
        weight=t687 {folded from=[p_s4_b2_conv1_conv_weight,p_s4_b2_conv1_bn_weight,b_s4_b2_conv1_bn_running_var] verify=vacuous origins=0}
        bias=t688 {folded from=[p_s4_b2_conv1_bn_weight,p_s4_b2_conv1_bn_bias,b_s4_b2_conv1_bn_running_mean,b_s4_b2_conv1_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n424 {pt2=root[83] torch.ops.aten.relu.default verify=unproved (too large)}: [t779 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n425]] =
      relu x=t778 {derived verify=unproved (too large)} <-n423
    n425 {derived verify=unproved (too large)}: [t780 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n426]] =
      conv2d
        x=t779 {derived verify=unproved (too large)} <-n424
        weight=t689 {folded from=[p_s4_b2_conv2_conv_weight,p_s4_b2_conv2_bn_weight,b_s4_b2_conv2_bn_running_var] verify=vacuous origins=0}
        bias=t690 {folded from=[p_s4_b2_conv2_bn_weight,p_s4_b2_conv2_bn_bias,b_s4_b2_conv2_bn_running_mean,b_s4_b2_conv2_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=368;
               groups=46}
    n426 {pt2=root[86] torch.ops.aten.relu.default verify=unproved (too large)}: [t781 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n427]] =
      relu x=t780 {derived verify=unproved (too large)} <-n425
    n427 {derived verify=unproved (too large)}: [t782 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n428]] =
      conv2d
        x=t781 {derived verify=unproved (too large)} <-n426
        weight=t691 {folded from=[p_s4_b2_conv3_conv_weight,p_s4_b2_conv3_bn_weight,b_s4_b2_conv3_bn_running_var] verify=vacuous origins=0}
        bias=t692 {folded from=[p_s4_b2_conv3_bn_weight,p_s4_b2_conv3_bn_bias,b_s4_b2_conv3_bn_running_mean,b_s4_b2_conv3_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n428 {pt2=root[89] torch.ops.aten.add.Tensor verify=vacuous}: [t783 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=vacuous origins=0} ->[n429]] =
      add
        a=t782 {derived verify=unproved (too large)} <-n427
        b=t777 {derived verify=unproved (too large)} <-n422
    n429 {pt2=root[90] torch.ops.aten.relu.default verify=unproved (too large)}: [t784 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n430,
                                                                      n435]] =
      relu x=t783 {derived verify=vacuous origins=0} <-n428
    n430 {derived verify=unproved (too large)}: [t785 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n431]] =
      conv2d
        x=t784 {derived verify=unproved (too large)} <-n429
        weight=t693 {folded from=[p_s4_b3_conv1_conv_weight,p_s4_b3_conv1_bn_weight,b_s4_b3_conv1_bn_running_var] verify=vacuous origins=0}
        bias=t694 {folded from=[p_s4_b3_conv1_bn_weight,p_s4_b3_conv1_bn_bias,b_s4_b3_conv1_bn_running_mean,b_s4_b3_conv1_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n431 {pt2=root[93] torch.ops.aten.relu.default verify=unproved (too large)}: [t786 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n432]] =
      relu x=t785 {derived verify=unproved (too large)} <-n430
    n432 {derived verify=unproved (too large)}: [t787 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n433]] =
      conv2d
        x=t786 {derived verify=unproved (too large)} <-n431
        weight=t695 {folded from=[p_s4_b3_conv2_conv_weight,p_s4_b3_conv2_bn_weight,b_s4_b3_conv2_bn_running_var] verify=vacuous origins=0}
        bias=t696 {folded from=[p_s4_b3_conv2_bn_weight,p_s4_b3_conv2_bn_bias,b_s4_b3_conv2_bn_running_mean,b_s4_b3_conv2_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=368;
               groups=46}
    n433 {pt2=root[96] torch.ops.aten.relu.default verify=unproved (too large)}: [t788 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n434]] =
      relu x=t787 {derived verify=unproved (too large)} <-n432
    n434 {derived verify=unproved (too large)}: [t789 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n435]] =
      conv2d
        x=t788 {derived verify=unproved (too large)} <-n433
        weight=t697 {folded from=[p_s4_b3_conv3_conv_weight,p_s4_b3_conv3_bn_weight,b_s4_b3_conv3_bn_running_var] verify=vacuous origins=0}
        bias=t698 {folded from=[p_s4_b3_conv3_bn_weight,p_s4_b3_conv3_bn_bias,b_s4_b3_conv3_bn_running_mean,b_s4_b3_conv3_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n435 {pt2=root[99] torch.ops.aten.add.Tensor verify=vacuous}: [t790 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=vacuous origins=0} ->[n436]] =
      add
        a=t789 {derived verify=unproved (too large)} <-n434
        b=t784 {derived verify=unproved (too large)} <-n429
    n436 {pt2=root[100] torch.ops.aten.relu.default verify=unproved (too large)}: [t791 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n437,
                                                                      n442]] =
      relu x=t790 {derived verify=vacuous origins=0} <-n435
    n437 {derived verify=unproved (too large)}: [t792 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n438]] =
      conv2d
        x=t791 {derived verify=unproved (too large)} <-n436
        weight=t699 {folded from=[p_s4_b4_conv1_conv_weight,p_s4_b4_conv1_bn_weight,b_s4_b4_conv1_bn_running_var] verify=vacuous origins=0}
        bias=t700 {folded from=[p_s4_b4_conv1_bn_weight,p_s4_b4_conv1_bn_bias,b_s4_b4_conv1_bn_running_mean,b_s4_b4_conv1_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n438 {pt2=root[103] torch.ops.aten.relu.default verify=unproved (too large)}: [t793 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n439]] =
      relu x=t792 {derived verify=unproved (too large)} <-n437
    n439 {derived verify=unproved (too large)}: [t794 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n440]] =
      conv2d
        x=t793 {derived verify=unproved (too large)} <-n438
        weight=t701 {folded from=[p_s4_b4_conv2_conv_weight,p_s4_b4_conv2_bn_weight,b_s4_b4_conv2_bn_running_var] verify=vacuous origins=0}
        bias=t702 {folded from=[p_s4_b4_conv2_bn_weight,p_s4_b4_conv2_bn_bias,b_s4_b4_conv2_bn_running_mean,b_s4_b4_conv2_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=368;
               groups=46}
    n440 {pt2=root[106] torch.ops.aten.relu.default verify=unproved (too large)}: [t795 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n441]] =
      relu x=t794 {derived verify=unproved (too large)} <-n439
    n441 {derived verify=unproved (too large)}: [t796 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n442]] =
      conv2d
        x=t795 {derived verify=unproved (too large)} <-n440
        weight=t703 {folded from=[p_s4_b4_conv3_conv_weight,p_s4_b4_conv3_bn_weight,b_s4_b4_conv3_bn_running_var] verify=vacuous origins=0}
        bias=t704 {folded from=[p_s4_b4_conv3_bn_weight,p_s4_b4_conv3_bn_bias,b_s4_b4_conv3_bn_running_mean,b_s4_b4_conv3_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n442 {pt2=root[109] torch.ops.aten.add.Tensor verify=vacuous}: [t797 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=vacuous origins=0} ->[n443]] =
      add
        a=t796 {derived verify=unproved (too large)} <-n441
        b=t791 {derived verify=unproved (too large)} <-n436
    n443 {pt2=root[110] torch.ops.aten.relu.default verify=unproved (too large)}: [t798 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n444,
                                                                      n449]] =
      relu x=t797 {derived verify=vacuous origins=0} <-n442
    n444 {derived verify=unproved (too large)}: [t799 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n445]] =
      conv2d
        x=t798 {derived verify=unproved (too large)} <-n443
        weight=t705 {folded from=[p_s4_b5_conv1_conv_weight,p_s4_b5_conv1_bn_weight,b_s4_b5_conv1_bn_running_var] verify=vacuous origins=0}
        bias=t706 {folded from=[p_s4_b5_conv1_bn_weight,p_s4_b5_conv1_bn_bias,b_s4_b5_conv1_bn_running_mean,b_s4_b5_conv1_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n445 {pt2=root[113] torch.ops.aten.relu.default verify=unproved (too large)}: [t800 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n446]] =
      relu x=t799 {derived verify=unproved (too large)} <-n444
    n446 {derived verify=unproved (too large)}: [t801 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n447]] =
      conv2d
        x=t800 {derived verify=unproved (too large)} <-n445
        weight=t707 {folded from=[p_s4_b5_conv2_conv_weight,p_s4_b5_conv2_bn_weight,b_s4_b5_conv2_bn_running_var] verify=vacuous origins=0}
        bias=t708 {folded from=[p_s4_b5_conv2_bn_weight,p_s4_b5_conv2_bn_bias,b_s4_b5_conv2_bn_running_mean,b_s4_b5_conv2_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=368;
               groups=46}
    n447 {pt2=root[116] torch.ops.aten.relu.default verify=unproved (too large)}: [t802 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n448]] =
      relu x=t801 {derived verify=unproved (too large)} <-n446
    n448 {derived verify=unproved (too large)}: [t803 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n449]] =
      conv2d
        x=t802 {derived verify=unproved (too large)} <-n447
        weight=t709 {folded from=[p_s4_b5_conv3_conv_weight,p_s4_b5_conv3_bn_weight,b_s4_b5_conv3_bn_running_var] verify=vacuous origins=0}
        bias=t710 {folded from=[p_s4_b5_conv3_bn_weight,p_s4_b5_conv3_bn_bias,b_s4_b5_conv3_bn_running_mean,b_s4_b5_conv3_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n449 {pt2=root[119] torch.ops.aten.add.Tensor verify=vacuous}: [t804 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=vacuous origins=0} ->[n450]] =
      add
        a=t803 {derived verify=unproved (too large)} <-n448
        b=t798 {derived verify=unproved (too large)} <-n443
    n450 {pt2=root[120] torch.ops.aten.relu.default verify=unproved (too large)}: [t805 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n451,
                                                                      n456]] =
      relu x=t804 {derived verify=vacuous origins=0} <-n449
    n451 {derived verify=unproved (too large)}: [t806 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n452]] =
      conv2d
        x=t805 {derived verify=unproved (too large)} <-n450
        weight=t711 {folded from=[p_s4_b6_conv1_conv_weight,p_s4_b6_conv1_bn_weight,b_s4_b6_conv1_bn_running_var] verify=vacuous origins=0}
        bias=t712 {folded from=[p_s4_b6_conv1_bn_weight,p_s4_b6_conv1_bn_bias,b_s4_b6_conv1_bn_running_mean,b_s4_b6_conv1_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n452 {pt2=root[123] torch.ops.aten.relu.default verify=unproved (too large)}: [t807 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n453]] =
      relu x=t806 {derived verify=unproved (too large)} <-n451
    n453 {derived verify=unproved (too large)}: [t808 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n454]] =
      conv2d
        x=t807 {derived verify=unproved (too large)} <-n452
        weight=t713 {folded from=[p_s4_b6_conv2_conv_weight,p_s4_b6_conv2_bn_weight,b_s4_b6_conv2_bn_running_var] verify=vacuous origins=0}
        bias=t714 {folded from=[p_s4_b6_conv2_bn_weight,p_s4_b6_conv2_bn_bias,b_s4_b6_conv2_bn_running_mean,b_s4_b6_conv2_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=368;
               groups=46}
    n454 {pt2=root[126] torch.ops.aten.relu.default verify=unproved (too large)}: [t809 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n455]] =
      relu x=t808 {derived verify=unproved (too large)} <-n453
    n455 {derived verify=unproved (too large)}: [t810 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n456]] =
      conv2d
        x=t809 {derived verify=unproved (too large)} <-n454
        weight=t715 {folded from=[p_s4_b6_conv3_conv_weight,p_s4_b6_conv3_bn_weight,b_s4_b6_conv3_bn_running_var] verify=vacuous origins=0}
        bias=t716 {folded from=[p_s4_b6_conv3_bn_weight,p_s4_b6_conv3_bn_bias,b_s4_b6_conv3_bn_running_mean,b_s4_b6_conv3_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n456 {pt2=root[129] torch.ops.aten.add.Tensor verify=vacuous}: [t811 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=vacuous origins=0} ->[n457]] =
      add
        a=t810 {derived verify=unproved (too large)} <-n455
        b=t805 {derived verify=unproved (too large)} <-n450
    n457 {pt2=root[130] torch.ops.aten.relu.default verify=unproved (too large)}: [t812 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n458,
                                                                      n463]] =
      relu x=t811 {derived verify=vacuous origins=0} <-n456
    n458 {derived verify=unproved (too large)}: [t813 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n459]] =
      conv2d
        x=t812 {derived verify=unproved (too large)} <-n457
        weight=t717 {folded from=[p_s4_b7_conv1_conv_weight,p_s4_b7_conv1_bn_weight,b_s4_b7_conv1_bn_running_var] verify=vacuous origins=0}
        bias=t718 {folded from=[p_s4_b7_conv1_bn_weight,p_s4_b7_conv1_bn_bias,b_s4_b7_conv1_bn_running_mean,b_s4_b7_conv1_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n459 {pt2=root[133] torch.ops.aten.relu.default verify=unproved (too large)}: [t814 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n460]] =
      relu x=t813 {derived verify=unproved (too large)} <-n458
    n460 {derived verify=unproved (too large)}: [t815 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n461]] =
      conv2d
        x=t814 {derived verify=unproved (too large)} <-n459
        weight=t719 {folded from=[p_s4_b7_conv2_conv_weight,p_s4_b7_conv2_bn_weight,b_s4_b7_conv2_bn_running_var] verify=vacuous origins=0}
        bias=t720 {folded from=[p_s4_b7_conv2_bn_weight,p_s4_b7_conv2_bn_bias,b_s4_b7_conv2_bn_running_mean,b_s4_b7_conv2_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=368;
               groups=46}
    n461 {pt2=root[136] torch.ops.aten.relu.default verify=unproved (too large)}: [t816 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n462]] =
      relu x=t815 {derived verify=unproved (too large)} <-n460
    n462 {derived verify=unproved (too large)}: [t817 f32 [H=7 W=7 C=368] {derived verify=unproved (too large)} ->[n463]] =
      conv2d
        x=t816 {derived verify=unproved (too large)} <-n461
        weight=t721 {folded from=[p_s4_b7_conv3_conv_weight,p_s4_b7_conv3_bn_weight,b_s4_b7_conv3_bn_running_var] verify=vacuous origins=0}
        bias=t722 {folded from=[p_s4_b7_conv3_bn_weight,p_s4_b7_conv3_bn_bias,b_s4_b7_conv3_bn_running_mean,b_s4_b7_conv3_bn_running_var] verify=vacuous origins=0}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n463 {pt2=root[139] torch.ops.aten.add.Tensor verify=vacuous}: [t818 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=vacuous origins=0} ->[n464]] =
      add
        a=t817 {derived verify=unproved (too large)} <-n462
        b=t812 {derived verify=unproved (too large)} <-n457
    n464 {pt2=root[140] torch.ops.aten.relu.default verify=unproved (too large)}: [t819 f32 [H=7
                                                                      W=7
                                                                      C=368] {derived verify=unproved (too large)} ->[n362]] =
      relu x=t818 {derived verify=vacuous origins=0} <-n463
    group g89 torch.ops.aten.adaptive_avg_pool2d.default:
      n362 {derived verify=unproved (over max_rounds) [sampled 4]}: [t629 f32 [C=368] {pt2=root:view verify=unproved (over max_rounds) [sampled 4] origins=2} ->[n365]] =
        adaptive_avg_pool2d
          x=t819 {derived verify=unproved (too large)} <-n464
          params={output_size={h=1; w=1}}
    n365 {pt2=root[143] torch.ops.aten.clone.default verify=proved (structural) [sampled 4]}: [t632 f32 [C=368] {pt2=root:clone verify=proved (structural) [sampled 4]} ->[n367]] =
      clone
        x=t629 {pt2=root:view verify=unproved (over max_rounds) [sampled 4] origins=2} <-n362
    group g90 torch.ops.aten.linear.default:
      n367 {pt2=root[144] torch.ops.aten.linear.default verify=proved (structural) [sampled 4]}: [t634 f32 [C=1000] {pt2=root:linear verify=proved (structural) [sampled 4]}] =
        linear
          x=t632 {pt2=root:clone verify=proved (structural) [sampled 4]} <-n365
          weight=t633 {folded from=[p_head_fc_weight] verify=unproved (too large)}
          bias=t133 {pt2=root:p_head_fc_bias target=head.fc.bias verify=proved (structural) [sampled 4]}
          params={in_features=368}
  outputs:
    [t634 f32 [C=1000] {pt2=root:linear verify=proved (structural) [sampled 4]} <-n367]
