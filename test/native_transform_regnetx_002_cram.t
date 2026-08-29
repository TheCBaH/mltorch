Rewrite RegNetX-002's imported graph with the transformation framework,
resolving PT2 provenance through the composed map (see
.ai/native_transform_design.md §10). Structure only, as
`test/native_graph_regnetx_002_cram.t` is: executing the result is
`make native-transform-verify.regnetx_002`, because a full inference is slow
and the residual it reports is floating point, neither of which belongs in a
golden.

The canonical pipeline (`lib/native/transform/pipeline.ml`) always runs
constant folding and batch-norm folding, regardless of `--fold` — that flag
now only controls whether payload bytes are preloaded for `--verify`/`--input`
evaluation, and is deliberately ignored by the pass list itself
(`~fold:_` in `canonical_with_trace`). So this one dump already reflects the
folded structure; there is no separate `--fold` golden.

  $ ../bin/native_graph.exe transform --pt2 "$PT2_DATA/regnetx_002/regnetx_002.pt2"
  nodes: 368 -> 101
  constants: 90, of which 89 folded
  graph
  inputs:
    [t133 f32 [C=1000] {pt2=root:p_head_fc_bias target=head.fc.bias} ->[n367] constant,
     t266 f32 [H=3 W=224 C=224] {pt2=root:x} ->[n0],
     t633 f32 [N=1000 T=1 D=1 H=1 W=1 C=368] {folded from=[p_head_fc_weight]} ->[n367] constant,
     t635 f32 [N=32 T=1 D=1 H=3 W=3 C=3] {folded from=[p_stem_conv_weight,p_stem_bn_weight,b_stem_bn_running_var]} ->[n368] constant,
     t636 f32 [C=32] {folded from=[p_stem_bn_weight,p_stem_bn_bias,b_stem_bn_running_mean,b_stem_bn_running_var]} ->[n368] constant,
     t637 f32 [N=24 T=1 D=1 H=1 W=1 C=32] {folded from=[p_s1_b1_conv1_conv_weight,p_s1_b1_conv1_bn_weight,b_s1_b1_conv1_bn_running_var]} ->[n370] constant,
     t638 f32 [C=24] {folded from=[p_s1_b1_conv1_bn_weight,p_s1_b1_conv1_bn_bias,b_s1_b1_conv1_bn_running_mean,b_s1_b1_conv1_bn_running_var]} ->[n370] constant,
     t639 f32 [N=24 T=1 D=1 H=1 W=1 C=32] {folded from=[p_s1_b1_downsample_conv_weight,p_s1_b1_downsample_bn_weight,b_s1_b1_downsample_bn_running_var]} ->[n371] constant,
     t640 f32 [C=24] {folded from=[p_s1_b1_downsample_bn_weight,p_s1_b1_downsample_bn_bias,b_s1_b1_downsample_bn_running_mean,b_s1_b1_downsample_bn_running_var]} ->[n371] constant,
     t641 f32 [N=24 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s1_b1_conv2_conv_weight,p_s1_b1_conv2_bn_weight,b_s1_b1_conv2_bn_running_var]} ->[n373] constant,
     t642 f32 [C=24] {folded from=[p_s1_b1_conv2_bn_weight,p_s1_b1_conv2_bn_bias,b_s1_b1_conv2_bn_running_mean,b_s1_b1_conv2_bn_running_var]} ->[n373] constant,
     t643 f32 [N=24 T=1 D=1 H=1 W=1 C=24] {folded from=[p_s1_b1_conv3_conv_weight,p_s1_b1_conv3_bn_weight,b_s1_b1_conv3_bn_running_var]} ->[n375] constant,
     t644 f32 [C=24] {folded from=[p_s1_b1_conv3_bn_weight,p_s1_b1_conv3_bn_bias,b_s1_b1_conv3_bn_running_mean,b_s1_b1_conv3_bn_running_var]} ->[n375] constant,
     t645 f32 [N=56 T=1 D=1 H=1 W=1 C=24] {folded from=[p_s2_b1_conv1_conv_weight,p_s2_b1_conv1_bn_weight,b_s2_b1_conv1_bn_running_var]} ->[n378] constant,
     t646 f32 [C=56] {folded from=[p_s2_b1_conv1_bn_weight,p_s2_b1_conv1_bn_bias,b_s2_b1_conv1_bn_running_mean,b_s2_b1_conv1_bn_running_var]} ->[n378] constant,
     t647 f32 [N=56 T=1 D=1 H=1 W=1 C=24] {folded from=[p_s2_b1_downsample_conv_weight,p_s2_b1_downsample_bn_weight,b_s2_b1_downsample_bn_running_var]} ->[n379] constant,
     t648 f32 [C=56] {folded from=[p_s2_b1_downsample_bn_weight,p_s2_b1_downsample_bn_bias,b_s2_b1_downsample_bn_running_mean,b_s2_b1_downsample_bn_running_var]} ->[n379] constant,
     t649 f32 [N=56 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s2_b1_conv2_conv_weight,p_s2_b1_conv2_bn_weight,b_s2_b1_conv2_bn_running_var]} ->[n381] constant,
     t650 f32 [C=56] {folded from=[p_s2_b1_conv2_bn_weight,p_s2_b1_conv2_bn_bias,b_s2_b1_conv2_bn_running_mean,b_s2_b1_conv2_bn_running_var]} ->[n381] constant,
     t651 f32 [N=56 T=1 D=1 H=1 W=1 C=56] {folded from=[p_s2_b1_conv3_conv_weight,p_s2_b1_conv3_bn_weight,b_s2_b1_conv3_bn_running_var]} ->[n383] constant,
     t652 f32 [C=56] {folded from=[p_s2_b1_conv3_bn_weight,p_s2_b1_conv3_bn_bias,b_s2_b1_conv3_bn_running_mean,b_s2_b1_conv3_bn_running_var]} ->[n383] constant,
     t653 f32 [N=152 T=1 D=1 H=1 W=1 C=56] {folded from=[p_s3_b1_conv1_conv_weight,p_s3_b1_conv1_bn_weight,b_s3_b1_conv1_bn_running_var]} ->[n386] constant,
     t654 f32 [C=152] {folded from=[p_s3_b1_conv1_bn_weight,p_s3_b1_conv1_bn_bias,b_s3_b1_conv1_bn_running_mean,b_s3_b1_conv1_bn_running_var]} ->[n386] constant,
     t655 f32 [N=152 T=1 D=1 H=1 W=1 C=56] {folded from=[p_s3_b1_downsample_conv_weight,p_s3_b1_downsample_bn_weight,b_s3_b1_downsample_bn_running_var]} ->[n387] constant,
     t656 f32 [C=152] {folded from=[p_s3_b1_downsample_bn_weight,p_s3_b1_downsample_bn_bias,b_s3_b1_downsample_bn_running_mean,b_s3_b1_downsample_bn_running_var]} ->[n387] constant,
     t657 f32 [N=152 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s3_b1_conv2_conv_weight,p_s3_b1_conv2_bn_weight,b_s3_b1_conv2_bn_running_var]} ->[n389] constant,
     t658 f32 [C=152] {folded from=[p_s3_b1_conv2_bn_weight,p_s3_b1_conv2_bn_bias,b_s3_b1_conv2_bn_running_mean,b_s3_b1_conv2_bn_running_var]} ->[n389] constant,
     t659 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s3_b1_conv3_conv_weight,p_s3_b1_conv3_bn_weight,b_s3_b1_conv3_bn_running_var]} ->[n391] constant,
     t660 f32 [C=152] {folded from=[p_s3_b1_conv3_bn_weight,p_s3_b1_conv3_bn_bias,b_s3_b1_conv3_bn_running_mean,b_s3_b1_conv3_bn_running_var]} ->[n391] constant,
     t661 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s3_b2_conv1_conv_weight,p_s3_b2_conv1_bn_weight,b_s3_b2_conv1_bn_running_var]} ->[n394] constant,
     t662 f32 [C=152] {folded from=[p_s3_b2_conv1_bn_weight,p_s3_b2_conv1_bn_bias,b_s3_b2_conv1_bn_running_mean,b_s3_b2_conv1_bn_running_var]} ->[n394] constant,
     t663 f32 [N=152 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s3_b2_conv2_conv_weight,p_s3_b2_conv2_bn_weight,b_s3_b2_conv2_bn_running_var]} ->[n396] constant,
     t664 f32 [C=152] {folded from=[p_s3_b2_conv2_bn_weight,p_s3_b2_conv2_bn_bias,b_s3_b2_conv2_bn_running_mean,b_s3_b2_conv2_bn_running_var]} ->[n396] constant,
     t665 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s3_b2_conv3_conv_weight,p_s3_b2_conv3_bn_weight,b_s3_b2_conv3_bn_running_var]} ->[n398] constant,
     t666 f32 [C=152] {folded from=[p_s3_b2_conv3_bn_weight,p_s3_b2_conv3_bn_bias,b_s3_b2_conv3_bn_running_mean,b_s3_b2_conv3_bn_running_var]} ->[n398] constant,
     t667 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s3_b3_conv1_conv_weight,p_s3_b3_conv1_bn_weight,b_s3_b3_conv1_bn_running_var]} ->[n401] constant,
     t668 f32 [C=152] {folded from=[p_s3_b3_conv1_bn_weight,p_s3_b3_conv1_bn_bias,b_s3_b3_conv1_bn_running_mean,b_s3_b3_conv1_bn_running_var]} ->[n401] constant,
     t669 f32 [N=152 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s3_b3_conv2_conv_weight,p_s3_b3_conv2_bn_weight,b_s3_b3_conv2_bn_running_var]} ->[n403] constant,
     t670 f32 [C=152] {folded from=[p_s3_b3_conv2_bn_weight,p_s3_b3_conv2_bn_bias,b_s3_b3_conv2_bn_running_mean,b_s3_b3_conv2_bn_running_var]} ->[n403] constant,
     t671 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s3_b3_conv3_conv_weight,p_s3_b3_conv3_bn_weight,b_s3_b3_conv3_bn_running_var]} ->[n405] constant,
     t672 f32 [C=152] {folded from=[p_s3_b3_conv3_bn_weight,p_s3_b3_conv3_bn_bias,b_s3_b3_conv3_bn_running_mean,b_s3_b3_conv3_bn_running_var]} ->[n405] constant,
     t673 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s3_b4_conv1_conv_weight,p_s3_b4_conv1_bn_weight,b_s3_b4_conv1_bn_running_var]} ->[n408] constant,
     t674 f32 [C=152] {folded from=[p_s3_b4_conv1_bn_weight,p_s3_b4_conv1_bn_bias,b_s3_b4_conv1_bn_running_mean,b_s3_b4_conv1_bn_running_var]} ->[n408] constant,
     t675 f32 [N=152 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s3_b4_conv2_conv_weight,p_s3_b4_conv2_bn_weight,b_s3_b4_conv2_bn_running_var]} ->[n410] constant,
     t676 f32 [C=152] {folded from=[p_s3_b4_conv2_bn_weight,p_s3_b4_conv2_bn_bias,b_s3_b4_conv2_bn_running_mean,b_s3_b4_conv2_bn_running_var]} ->[n410] constant,
     t677 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s3_b4_conv3_conv_weight,p_s3_b4_conv3_bn_weight,b_s3_b4_conv3_bn_running_var]} ->[n412] constant,
     t678 f32 [C=152] {folded from=[p_s3_b4_conv3_bn_weight,p_s3_b4_conv3_bn_bias,b_s3_b4_conv3_bn_running_mean,b_s3_b4_conv3_bn_running_var]} ->[n412] constant,
     t679 f32 [N=368 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s4_b1_conv1_conv_weight,p_s4_b1_conv1_bn_weight,b_s4_b1_conv1_bn_running_var]} ->[n415] constant,
     t680 f32 [C=368] {folded from=[p_s4_b1_conv1_bn_weight,p_s4_b1_conv1_bn_bias,b_s4_b1_conv1_bn_running_mean,b_s4_b1_conv1_bn_running_var]} ->[n415] constant,
     t681 f32 [N=368 T=1 D=1 H=1 W=1 C=152] {folded from=[p_s4_b1_downsample_conv_weight,p_s4_b1_downsample_bn_weight,b_s4_b1_downsample_bn_running_var]} ->[n416] constant,
     t682 f32 [C=368] {folded from=[p_s4_b1_downsample_bn_weight,p_s4_b1_downsample_bn_bias,b_s4_b1_downsample_bn_running_mean,b_s4_b1_downsample_bn_running_var]} ->[n416] constant,
     t683 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s4_b1_conv2_conv_weight,p_s4_b1_conv2_bn_weight,b_s4_b1_conv2_bn_running_var]} ->[n418] constant,
     t684 f32 [C=368] {folded from=[p_s4_b1_conv2_bn_weight,p_s4_b1_conv2_bn_bias,b_s4_b1_conv2_bn_running_mean,b_s4_b1_conv2_bn_running_var]} ->[n418] constant,
     t685 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b1_conv3_conv_weight,p_s4_b1_conv3_bn_weight,b_s4_b1_conv3_bn_running_var]} ->[n420] constant,
     t686 f32 [C=368] {folded from=[p_s4_b1_conv3_bn_weight,p_s4_b1_conv3_bn_bias,b_s4_b1_conv3_bn_running_mean,b_s4_b1_conv3_bn_running_var]} ->[n420] constant,
     t687 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b2_conv1_conv_weight,p_s4_b2_conv1_bn_weight,b_s4_b2_conv1_bn_running_var]} ->[n423] constant,
     t688 f32 [C=368] {folded from=[p_s4_b2_conv1_bn_weight,p_s4_b2_conv1_bn_bias,b_s4_b2_conv1_bn_running_mean,b_s4_b2_conv1_bn_running_var]} ->[n423] constant,
     t689 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s4_b2_conv2_conv_weight,p_s4_b2_conv2_bn_weight,b_s4_b2_conv2_bn_running_var]} ->[n425] constant,
     t690 f32 [C=368] {folded from=[p_s4_b2_conv2_bn_weight,p_s4_b2_conv2_bn_bias,b_s4_b2_conv2_bn_running_mean,b_s4_b2_conv2_bn_running_var]} ->[n425] constant,
     t691 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b2_conv3_conv_weight,p_s4_b2_conv3_bn_weight,b_s4_b2_conv3_bn_running_var]} ->[n427] constant,
     t692 f32 [C=368] {folded from=[p_s4_b2_conv3_bn_weight,p_s4_b2_conv3_bn_bias,b_s4_b2_conv3_bn_running_mean,b_s4_b2_conv3_bn_running_var]} ->[n427] constant,
     t693 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b3_conv1_conv_weight,p_s4_b3_conv1_bn_weight,b_s4_b3_conv1_bn_running_var]} ->[n430] constant,
     t694 f32 [C=368] {folded from=[p_s4_b3_conv1_bn_weight,p_s4_b3_conv1_bn_bias,b_s4_b3_conv1_bn_running_mean,b_s4_b3_conv1_bn_running_var]} ->[n430] constant,
     t695 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s4_b3_conv2_conv_weight,p_s4_b3_conv2_bn_weight,b_s4_b3_conv2_bn_running_var]} ->[n432] constant,
     t696 f32 [C=368] {folded from=[p_s4_b3_conv2_bn_weight,p_s4_b3_conv2_bn_bias,b_s4_b3_conv2_bn_running_mean,b_s4_b3_conv2_bn_running_var]} ->[n432] constant,
     t697 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b3_conv3_conv_weight,p_s4_b3_conv3_bn_weight,b_s4_b3_conv3_bn_running_var]} ->[n434] constant,
     t698 f32 [C=368] {folded from=[p_s4_b3_conv3_bn_weight,p_s4_b3_conv3_bn_bias,b_s4_b3_conv3_bn_running_mean,b_s4_b3_conv3_bn_running_var]} ->[n434] constant,
     t699 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b4_conv1_conv_weight,p_s4_b4_conv1_bn_weight,b_s4_b4_conv1_bn_running_var]} ->[n437] constant,
     t700 f32 [C=368] {folded from=[p_s4_b4_conv1_bn_weight,p_s4_b4_conv1_bn_bias,b_s4_b4_conv1_bn_running_mean,b_s4_b4_conv1_bn_running_var]} ->[n437] constant,
     t701 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s4_b4_conv2_conv_weight,p_s4_b4_conv2_bn_weight,b_s4_b4_conv2_bn_running_var]} ->[n439] constant,
     t702 f32 [C=368] {folded from=[p_s4_b4_conv2_bn_weight,p_s4_b4_conv2_bn_bias,b_s4_b4_conv2_bn_running_mean,b_s4_b4_conv2_bn_running_var]} ->[n439] constant,
     t703 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b4_conv3_conv_weight,p_s4_b4_conv3_bn_weight,b_s4_b4_conv3_bn_running_var]} ->[n441] constant,
     t704 f32 [C=368] {folded from=[p_s4_b4_conv3_bn_weight,p_s4_b4_conv3_bn_bias,b_s4_b4_conv3_bn_running_mean,b_s4_b4_conv3_bn_running_var]} ->[n441] constant,
     t705 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b5_conv1_conv_weight,p_s4_b5_conv1_bn_weight,b_s4_b5_conv1_bn_running_var]} ->[n444] constant,
     t706 f32 [C=368] {folded from=[p_s4_b5_conv1_bn_weight,p_s4_b5_conv1_bn_bias,b_s4_b5_conv1_bn_running_mean,b_s4_b5_conv1_bn_running_var]} ->[n444] constant,
     t707 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s4_b5_conv2_conv_weight,p_s4_b5_conv2_bn_weight,b_s4_b5_conv2_bn_running_var]} ->[n446] constant,
     t708 f32 [C=368] {folded from=[p_s4_b5_conv2_bn_weight,p_s4_b5_conv2_bn_bias,b_s4_b5_conv2_bn_running_mean,b_s4_b5_conv2_bn_running_var]} ->[n446] constant,
     t709 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b5_conv3_conv_weight,p_s4_b5_conv3_bn_weight,b_s4_b5_conv3_bn_running_var]} ->[n448] constant,
     t710 f32 [C=368] {folded from=[p_s4_b5_conv3_bn_weight,p_s4_b5_conv3_bn_bias,b_s4_b5_conv3_bn_running_mean,b_s4_b5_conv3_bn_running_var]} ->[n448] constant,
     t711 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b6_conv1_conv_weight,p_s4_b6_conv1_bn_weight,b_s4_b6_conv1_bn_running_var]} ->[n451] constant,
     t712 f32 [C=368] {folded from=[p_s4_b6_conv1_bn_weight,p_s4_b6_conv1_bn_bias,b_s4_b6_conv1_bn_running_mean,b_s4_b6_conv1_bn_running_var]} ->[n451] constant,
     t713 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s4_b6_conv2_conv_weight,p_s4_b6_conv2_bn_weight,b_s4_b6_conv2_bn_running_var]} ->[n453] constant,
     t714 f32 [C=368] {folded from=[p_s4_b6_conv2_bn_weight,p_s4_b6_conv2_bn_bias,b_s4_b6_conv2_bn_running_mean,b_s4_b6_conv2_bn_running_var]} ->[n453] constant,
     t715 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b6_conv3_conv_weight,p_s4_b6_conv3_bn_weight,b_s4_b6_conv3_bn_running_var]} ->[n455] constant,
     t716 f32 [C=368] {folded from=[p_s4_b6_conv3_bn_weight,p_s4_b6_conv3_bn_bias,b_s4_b6_conv3_bn_running_mean,b_s4_b6_conv3_bn_running_var]} ->[n455] constant,
     t717 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b7_conv1_conv_weight,p_s4_b7_conv1_bn_weight,b_s4_b7_conv1_bn_running_var]} ->[n458] constant,
     t718 f32 [C=368] {folded from=[p_s4_b7_conv1_bn_weight,p_s4_b7_conv1_bn_bias,b_s4_b7_conv1_bn_running_mean,b_s4_b7_conv1_bn_running_var]} ->[n458] constant,
     t719 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {folded from=[p_s4_b7_conv2_conv_weight,p_s4_b7_conv2_bn_weight,b_s4_b7_conv2_bn_running_var]} ->[n460] constant,
     t720 f32 [C=368] {folded from=[p_s4_b7_conv2_bn_weight,p_s4_b7_conv2_bn_bias,b_s4_b7_conv2_bn_running_mean,b_s4_b7_conv2_bn_running_var]} ->[n460] constant,
     t721 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {folded from=[p_s4_b7_conv3_conv_weight,p_s4_b7_conv3_bn_weight,b_s4_b7_conv3_bn_running_var]} ->[n462] constant,
     t722 f32 [C=368] {folded from=[p_s4_b7_conv3_bn_weight,p_s4_b7_conv3_bn_bias,b_s4_b7_conv3_bn_running_mean,b_s4_b7_conv3_bn_running_var]} ->[n462] constant]
  nodes:
    group g1 torch.ops.aten.conv2d.default:
      n0 {derived}: [t267 f32 [H=224 W=224 C=3] {derived} ->[n368]] =
        permute x=t266 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
    n368 {derived}: [t723 f32 [H=112 W=112 C=32] {derived} ->[n369]] =
      conv2d
        x=t267 {derived} <-n0
        weight=t635 {folded from=[p_stem_conv_weight,p_stem_bn_weight,b_stem_bn_running_var]}
        bias=t636 {folded from=[p_stem_bn_weight,p_stem_bn_bias,b_stem_bn_running_mean,b_stem_bn_running_var]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=3;
               groups=1}
    n369 {pt2=root[2] torch.ops.aten.relu.default}: [t724 f32 [H=112 W=112
                                                               C=32] {derived} ->[n370,
                                                                      n371]] =
      relu x=t723 {derived} <-n368
    n370 {derived}: [t725 f32 [H=112 W=112 C=24] {derived} ->[n372]] =
      conv2d
        x=t724 {derived} <-n369
        weight=t637 {folded from=[p_s1_b1_conv1_conv_weight,p_s1_b1_conv1_bn_weight,b_s1_b1_conv1_bn_running_var]}
        bias=t638 {folded from=[p_s1_b1_conv1_bn_weight,p_s1_b1_conv1_bn_bias,b_s1_b1_conv1_bn_running_mean,b_s1_b1_conv1_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=32;
               groups=1}
    n371 {derived}: [t726 f32 [H=56 W=56 C=24] {derived} ->[n376]] =
      conv2d
        x=t724 {derived} <-n369
        weight=t639 {folded from=[p_s1_b1_downsample_conv_weight,p_s1_b1_downsample_bn_weight,b_s1_b1_downsample_bn_running_var]}
        bias=t640 {folded from=[p_s1_b1_downsample_bn_weight,p_s1_b1_downsample_bn_bias,b_s1_b1_downsample_bn_running_mean,b_s1_b1_downsample_bn_running_var]}
        params={h={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               in_channels=32;
               groups=1}
    n372 {pt2=root[5] torch.ops.aten.relu.default}: [t727 f32 [H=112 W=112
                                                               C=24] {derived} ->[n373]] =
      relu x=t725 {derived} <-n370
    n373 {derived}: [t728 f32 [H=56 W=56 C=24] {derived} ->[n374]] =
      conv2d
        x=t727 {derived} <-n372
        weight=t641 {folded from=[p_s1_b1_conv2_conv_weight,p_s1_b1_conv2_bn_weight,b_s1_b1_conv2_bn_running_var]}
        bias=t642 {folded from=[p_s1_b1_conv2_bn_weight,p_s1_b1_conv2_bn_bias,b_s1_b1_conv2_bn_running_mean,b_s1_b1_conv2_bn_running_var]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=24;
               groups=3}
    n374 {pt2=root[8] torch.ops.aten.relu.default}: [t729 f32 [H=56 W=56 C=24] {derived} ->[n375]] =
      relu x=t728 {derived} <-n373
    n375 {derived}: [t730 f32 [H=56 W=56 C=24] {derived} ->[n376]] =
      conv2d
        x=t729 {derived} <-n374
        weight=t643 {folded from=[p_s1_b1_conv3_conv_weight,p_s1_b1_conv3_bn_weight,b_s1_b1_conv3_bn_running_var]}
        bias=t644 {folded from=[p_s1_b1_conv3_bn_weight,p_s1_b1_conv3_bn_bias,b_s1_b1_conv3_bn_running_mean,b_s1_b1_conv3_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=24;
               groups=1}
    n376 {pt2=root[13] torch.ops.aten.add.Tensor}: [t731 f32 [H=56 W=56 C=24] {derived} ->[n377]] =
      add a=t730 {derived} <-n375 b=t726 {derived} <-n371
    n377 {pt2=root[14] torch.ops.aten.relu.default}: [t732 f32 [H=56 W=56 C=24] {derived} ->[n378,
                                                                      n379]] =
      relu x=t731 {derived} <-n376
    n378 {derived}: [t733 f32 [H=56 W=56 C=56] {derived} ->[n380]] =
      conv2d
        x=t732 {derived} <-n377
        weight=t645 {folded from=[p_s2_b1_conv1_conv_weight,p_s2_b1_conv1_bn_weight,b_s2_b1_conv1_bn_running_var]}
        bias=t646 {folded from=[p_s2_b1_conv1_bn_weight,p_s2_b1_conv1_bn_bias,b_s2_b1_conv1_bn_running_mean,b_s2_b1_conv1_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=24;
               groups=1}
    n379 {derived}: [t734 f32 [H=28 W=28 C=56] {derived} ->[n384]] =
      conv2d
        x=t732 {derived} <-n377
        weight=t647 {folded from=[p_s2_b1_downsample_conv_weight,p_s2_b1_downsample_bn_weight,b_s2_b1_downsample_bn_running_var]}
        bias=t648 {folded from=[p_s2_b1_downsample_bn_weight,p_s2_b1_downsample_bn_bias,b_s2_b1_downsample_bn_running_mean,b_s2_b1_downsample_bn_running_var]}
        params={h={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               in_channels=24;
               groups=1}
    n380 {pt2=root[17] torch.ops.aten.relu.default}: [t735 f32 [H=56 W=56 C=56] {derived} ->[n381]] =
      relu x=t733 {derived} <-n378
    n381 {derived}: [t736 f32 [H=28 W=28 C=56] {derived} ->[n382]] =
      conv2d
        x=t735 {derived} <-n380
        weight=t649 {folded from=[p_s2_b1_conv2_conv_weight,p_s2_b1_conv2_bn_weight,b_s2_b1_conv2_bn_running_var]}
        bias=t650 {folded from=[p_s2_b1_conv2_bn_weight,p_s2_b1_conv2_bn_bias,b_s2_b1_conv2_bn_running_mean,b_s2_b1_conv2_bn_running_var]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=56;
               groups=7}
    n382 {pt2=root[20] torch.ops.aten.relu.default}: [t737 f32 [H=28 W=28 C=56] {derived} ->[n383]] =
      relu x=t736 {derived} <-n381
    n383 {derived}: [t738 f32 [H=28 W=28 C=56] {derived} ->[n384]] =
      conv2d
        x=t737 {derived} <-n382
        weight=t651 {folded from=[p_s2_b1_conv3_conv_weight,p_s2_b1_conv3_bn_weight,b_s2_b1_conv3_bn_running_var]}
        bias=t652 {folded from=[p_s2_b1_conv3_bn_weight,p_s2_b1_conv3_bn_bias,b_s2_b1_conv3_bn_running_mean,b_s2_b1_conv3_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=56;
               groups=1}
    n384 {pt2=root[25] torch.ops.aten.add.Tensor}: [t739 f32 [H=28 W=28 C=56] {derived} ->[n385]] =
      add a=t738 {derived} <-n383 b=t734 {derived} <-n379
    n385 {pt2=root[26] torch.ops.aten.relu.default}: [t740 f32 [H=28 W=28 C=56] {derived} ->[n386,
                                                                      n387]] =
      relu x=t739 {derived} <-n384
    n386 {derived}: [t741 f32 [H=28 W=28 C=152] {derived} ->[n388]] =
      conv2d
        x=t740 {derived} <-n385
        weight=t653 {folded from=[p_s3_b1_conv1_conv_weight,p_s3_b1_conv1_bn_weight,b_s3_b1_conv1_bn_running_var]}
        bias=t654 {folded from=[p_s3_b1_conv1_bn_weight,p_s3_b1_conv1_bn_bias,b_s3_b1_conv1_bn_running_mean,b_s3_b1_conv1_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=56;
               groups=1}
    n387 {derived}: [t742 f32 [H=14 W=14 C=152] {derived} ->[n392]] =
      conv2d
        x=t740 {derived} <-n385
        weight=t655 {folded from=[p_s3_b1_downsample_conv_weight,p_s3_b1_downsample_bn_weight,b_s3_b1_downsample_bn_running_var]}
        bias=t656 {folded from=[p_s3_b1_downsample_bn_weight,p_s3_b1_downsample_bn_bias,b_s3_b1_downsample_bn_running_mean,b_s3_b1_downsample_bn_running_var]}
        params={h={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               in_channels=56;
               groups=1}
    n388 {pt2=root[29] torch.ops.aten.relu.default}: [t743 f32 [H=28 W=28
                                                                C=152] {derived} ->[n389]] =
      relu x=t741 {derived} <-n386
    n389 {derived}: [t744 f32 [H=14 W=14 C=152] {derived} ->[n390]] =
      conv2d
        x=t743 {derived} <-n388
        weight=t657 {folded from=[p_s3_b1_conv2_conv_weight,p_s3_b1_conv2_bn_weight,b_s3_b1_conv2_bn_running_var]}
        bias=t658 {folded from=[p_s3_b1_conv2_bn_weight,p_s3_b1_conv2_bn_bias,b_s3_b1_conv2_bn_running_mean,b_s3_b1_conv2_bn_running_var]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=152;
               groups=19}
    n390 {pt2=root[32] torch.ops.aten.relu.default}: [t745 f32 [H=14 W=14
                                                                C=152] {derived} ->[n391]] =
      relu x=t744 {derived} <-n389
    n391 {derived}: [t746 f32 [H=14 W=14 C=152] {derived} ->[n392]] =
      conv2d
        x=t745 {derived} <-n390
        weight=t659 {folded from=[p_s3_b1_conv3_conv_weight,p_s3_b1_conv3_bn_weight,b_s3_b1_conv3_bn_running_var]}
        bias=t660 {folded from=[p_s3_b1_conv3_bn_weight,p_s3_b1_conv3_bn_bias,b_s3_b1_conv3_bn_running_mean,b_s3_b1_conv3_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n392 {pt2=root[37] torch.ops.aten.add.Tensor}: [t747 f32 [H=14 W=14 C=152] {derived} ->[n393]] =
      add a=t746 {derived} <-n391 b=t742 {derived} <-n387
    n393 {pt2=root[38] torch.ops.aten.relu.default}: [t748 f32 [H=14 W=14
                                                                C=152] {derived} ->[n394,
                                                                      n399]] =
      relu x=t747 {derived} <-n392
    n394 {derived}: [t749 f32 [H=14 W=14 C=152] {derived} ->[n395]] =
      conv2d
        x=t748 {derived} <-n393
        weight=t661 {folded from=[p_s3_b2_conv1_conv_weight,p_s3_b2_conv1_bn_weight,b_s3_b2_conv1_bn_running_var]}
        bias=t662 {folded from=[p_s3_b2_conv1_bn_weight,p_s3_b2_conv1_bn_bias,b_s3_b2_conv1_bn_running_mean,b_s3_b2_conv1_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n395 {pt2=root[41] torch.ops.aten.relu.default}: [t750 f32 [H=14 W=14
                                                                C=152] {derived} ->[n396]] =
      relu x=t749 {derived} <-n394
    n396 {derived}: [t751 f32 [H=14 W=14 C=152] {derived} ->[n397]] =
      conv2d
        x=t750 {derived} <-n395
        weight=t663 {folded from=[p_s3_b2_conv2_conv_weight,p_s3_b2_conv2_bn_weight,b_s3_b2_conv2_bn_running_var]}
        bias=t664 {folded from=[p_s3_b2_conv2_bn_weight,p_s3_b2_conv2_bn_bias,b_s3_b2_conv2_bn_running_mean,b_s3_b2_conv2_bn_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=152;
               groups=19}
    n397 {pt2=root[44] torch.ops.aten.relu.default}: [t752 f32 [H=14 W=14
                                                                C=152] {derived} ->[n398]] =
      relu x=t751 {derived} <-n396
    n398 {derived}: [t753 f32 [H=14 W=14 C=152] {derived} ->[n399]] =
      conv2d
        x=t752 {derived} <-n397
        weight=t665 {folded from=[p_s3_b2_conv3_conv_weight,p_s3_b2_conv3_bn_weight,b_s3_b2_conv3_bn_running_var]}
        bias=t666 {folded from=[p_s3_b2_conv3_bn_weight,p_s3_b2_conv3_bn_bias,b_s3_b2_conv3_bn_running_mean,b_s3_b2_conv3_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n399 {pt2=root[47] torch.ops.aten.add.Tensor}: [t754 f32 [H=14 W=14 C=152] {derived} ->[n400]] =
      add a=t753 {derived} <-n398 b=t748 {derived} <-n393
    n400 {pt2=root[48] torch.ops.aten.relu.default}: [t755 f32 [H=14 W=14
                                                                C=152] {derived} ->[n401,
                                                                      n406]] =
      relu x=t754 {derived} <-n399
    n401 {derived}: [t756 f32 [H=14 W=14 C=152] {derived} ->[n402]] =
      conv2d
        x=t755 {derived} <-n400
        weight=t667 {folded from=[p_s3_b3_conv1_conv_weight,p_s3_b3_conv1_bn_weight,b_s3_b3_conv1_bn_running_var]}
        bias=t668 {folded from=[p_s3_b3_conv1_bn_weight,p_s3_b3_conv1_bn_bias,b_s3_b3_conv1_bn_running_mean,b_s3_b3_conv1_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n402 {pt2=root[51] torch.ops.aten.relu.default}: [t757 f32 [H=14 W=14
                                                                C=152] {derived} ->[n403]] =
      relu x=t756 {derived} <-n401
    n403 {derived}: [t758 f32 [H=14 W=14 C=152] {derived} ->[n404]] =
      conv2d
        x=t757 {derived} <-n402
        weight=t669 {folded from=[p_s3_b3_conv2_conv_weight,p_s3_b3_conv2_bn_weight,b_s3_b3_conv2_bn_running_var]}
        bias=t670 {folded from=[p_s3_b3_conv2_bn_weight,p_s3_b3_conv2_bn_bias,b_s3_b3_conv2_bn_running_mean,b_s3_b3_conv2_bn_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=152;
               groups=19}
    n404 {pt2=root[54] torch.ops.aten.relu.default}: [t759 f32 [H=14 W=14
                                                                C=152] {derived} ->[n405]] =
      relu x=t758 {derived} <-n403
    n405 {derived}: [t760 f32 [H=14 W=14 C=152] {derived} ->[n406]] =
      conv2d
        x=t759 {derived} <-n404
        weight=t671 {folded from=[p_s3_b3_conv3_conv_weight,p_s3_b3_conv3_bn_weight,b_s3_b3_conv3_bn_running_var]}
        bias=t672 {folded from=[p_s3_b3_conv3_bn_weight,p_s3_b3_conv3_bn_bias,b_s3_b3_conv3_bn_running_mean,b_s3_b3_conv3_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n406 {pt2=root[57] torch.ops.aten.add.Tensor}: [t761 f32 [H=14 W=14 C=152] {derived} ->[n407]] =
      add a=t760 {derived} <-n405 b=t755 {derived} <-n400
    n407 {pt2=root[58] torch.ops.aten.relu.default}: [t762 f32 [H=14 W=14
                                                                C=152] {derived} ->[n408,
                                                                      n413]] =
      relu x=t761 {derived} <-n406
    n408 {derived}: [t763 f32 [H=14 W=14 C=152] {derived} ->[n409]] =
      conv2d
        x=t762 {derived} <-n407
        weight=t673 {folded from=[p_s3_b4_conv1_conv_weight,p_s3_b4_conv1_bn_weight,b_s3_b4_conv1_bn_running_var]}
        bias=t674 {folded from=[p_s3_b4_conv1_bn_weight,p_s3_b4_conv1_bn_bias,b_s3_b4_conv1_bn_running_mean,b_s3_b4_conv1_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n409 {pt2=root[61] torch.ops.aten.relu.default}: [t764 f32 [H=14 W=14
                                                                C=152] {derived} ->[n410]] =
      relu x=t763 {derived} <-n408
    n410 {derived}: [t765 f32 [H=14 W=14 C=152] {derived} ->[n411]] =
      conv2d
        x=t764 {derived} <-n409
        weight=t675 {folded from=[p_s3_b4_conv2_conv_weight,p_s3_b4_conv2_bn_weight,b_s3_b4_conv2_bn_running_var]}
        bias=t676 {folded from=[p_s3_b4_conv2_bn_weight,p_s3_b4_conv2_bn_bias,b_s3_b4_conv2_bn_running_mean,b_s3_b4_conv2_bn_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=152;
               groups=19}
    n411 {pt2=root[64] torch.ops.aten.relu.default}: [t766 f32 [H=14 W=14
                                                                C=152] {derived} ->[n412]] =
      relu x=t765 {derived} <-n410
    n412 {derived}: [t767 f32 [H=14 W=14 C=152] {derived} ->[n413]] =
      conv2d
        x=t766 {derived} <-n411
        weight=t677 {folded from=[p_s3_b4_conv3_conv_weight,p_s3_b4_conv3_bn_weight,b_s3_b4_conv3_bn_running_var]}
        bias=t678 {folded from=[p_s3_b4_conv3_bn_weight,p_s3_b4_conv3_bn_bias,b_s3_b4_conv3_bn_running_mean,b_s3_b4_conv3_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n413 {pt2=root[67] torch.ops.aten.add.Tensor}: [t768 f32 [H=14 W=14 C=152] {derived} ->[n414]] =
      add a=t767 {derived} <-n412 b=t762 {derived} <-n407
    n414 {pt2=root[68] torch.ops.aten.relu.default}: [t769 f32 [H=14 W=14
                                                                C=152] {derived} ->[n415,
                                                                      n416]] =
      relu x=t768 {derived} <-n413
    n415 {derived}: [t770 f32 [H=14 W=14 C=368] {derived} ->[n417]] =
      conv2d
        x=t769 {derived} <-n414
        weight=t679 {folded from=[p_s4_b1_conv1_conv_weight,p_s4_b1_conv1_bn_weight,b_s4_b1_conv1_bn_running_var]}
        bias=t680 {folded from=[p_s4_b1_conv1_bn_weight,p_s4_b1_conv1_bn_bias,b_s4_b1_conv1_bn_running_mean,b_s4_b1_conv1_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n416 {derived}: [t771 f32 [H=7 W=7 C=368] {derived} ->[n421]] =
      conv2d
        x=t769 {derived} <-n414
        weight=t681 {folded from=[p_s4_b1_downsample_conv_weight,p_s4_b1_downsample_bn_weight,b_s4_b1_downsample_bn_running_var]}
        bias=t682 {folded from=[p_s4_b1_downsample_bn_weight,p_s4_b1_downsample_bn_bias,b_s4_b1_downsample_bn_running_mean,b_s4_b1_downsample_bn_running_var]}
        params={h={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
               in_channels=152;
               groups=1}
    n417 {pt2=root[71] torch.ops.aten.relu.default}: [t772 f32 [H=14 W=14
                                                                C=368] {derived} ->[n418]] =
      relu x=t770 {derived} <-n415
    n418 {derived}: [t773 f32 [H=7 W=7 C=368] {derived} ->[n419]] =
      conv2d
        x=t772 {derived} <-n417
        weight=t683 {folded from=[p_s4_b1_conv2_conv_weight,p_s4_b1_conv2_bn_weight,b_s4_b1_conv2_bn_running_var]}
        bias=t684 {folded from=[p_s4_b1_conv2_bn_weight,p_s4_b1_conv2_bn_bias,b_s4_b1_conv2_bn_running_mean,b_s4_b1_conv2_bn_running_var]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=368;
               groups=46}
    n419 {pt2=root[74] torch.ops.aten.relu.default}: [t774 f32 [H=7 W=7 C=368] {derived} ->[n420]] =
      relu x=t773 {derived} <-n418
    n420 {derived}: [t775 f32 [H=7 W=7 C=368] {derived} ->[n421]] =
      conv2d
        x=t774 {derived} <-n419
        weight=t685 {folded from=[p_s4_b1_conv3_conv_weight,p_s4_b1_conv3_bn_weight,b_s4_b1_conv3_bn_running_var]}
        bias=t686 {folded from=[p_s4_b1_conv3_bn_weight,p_s4_b1_conv3_bn_bias,b_s4_b1_conv3_bn_running_mean,b_s4_b1_conv3_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n421 {pt2=root[79] torch.ops.aten.add.Tensor}: [t776 f32 [H=7 W=7 C=368] {derived} ->[n422]] =
      add a=t775 {derived} <-n420 b=t771 {derived} <-n416
    n422 {pt2=root[80] torch.ops.aten.relu.default}: [t777 f32 [H=7 W=7 C=368] {derived} ->[n423,
                                                                      n428]] =
      relu x=t776 {derived} <-n421
    n423 {derived}: [t778 f32 [H=7 W=7 C=368] {derived} ->[n424]] =
      conv2d
        x=t777 {derived} <-n422
        weight=t687 {folded from=[p_s4_b2_conv1_conv_weight,p_s4_b2_conv1_bn_weight,b_s4_b2_conv1_bn_running_var]}
        bias=t688 {folded from=[p_s4_b2_conv1_bn_weight,p_s4_b2_conv1_bn_bias,b_s4_b2_conv1_bn_running_mean,b_s4_b2_conv1_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n424 {pt2=root[83] torch.ops.aten.relu.default}: [t779 f32 [H=7 W=7 C=368] {derived} ->[n425]] =
      relu x=t778 {derived} <-n423
    n425 {derived}: [t780 f32 [H=7 W=7 C=368] {derived} ->[n426]] =
      conv2d
        x=t779 {derived} <-n424
        weight=t689 {folded from=[p_s4_b2_conv2_conv_weight,p_s4_b2_conv2_bn_weight,b_s4_b2_conv2_bn_running_var]}
        bias=t690 {folded from=[p_s4_b2_conv2_bn_weight,p_s4_b2_conv2_bn_bias,b_s4_b2_conv2_bn_running_mean,b_s4_b2_conv2_bn_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=368;
               groups=46}
    n426 {pt2=root[86] torch.ops.aten.relu.default}: [t781 f32 [H=7 W=7 C=368] {derived} ->[n427]] =
      relu x=t780 {derived} <-n425
    n427 {derived}: [t782 f32 [H=7 W=7 C=368] {derived} ->[n428]] =
      conv2d
        x=t781 {derived} <-n426
        weight=t691 {folded from=[p_s4_b2_conv3_conv_weight,p_s4_b2_conv3_bn_weight,b_s4_b2_conv3_bn_running_var]}
        bias=t692 {folded from=[p_s4_b2_conv3_bn_weight,p_s4_b2_conv3_bn_bias,b_s4_b2_conv3_bn_running_mean,b_s4_b2_conv3_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n428 {pt2=root[89] torch.ops.aten.add.Tensor}: [t783 f32 [H=7 W=7 C=368] {derived} ->[n429]] =
      add a=t782 {derived} <-n427 b=t777 {derived} <-n422
    n429 {pt2=root[90] torch.ops.aten.relu.default}: [t784 f32 [H=7 W=7 C=368] {derived} ->[n430,
                                                                      n435]] =
      relu x=t783 {derived} <-n428
    n430 {derived}: [t785 f32 [H=7 W=7 C=368] {derived} ->[n431]] =
      conv2d
        x=t784 {derived} <-n429
        weight=t693 {folded from=[p_s4_b3_conv1_conv_weight,p_s4_b3_conv1_bn_weight,b_s4_b3_conv1_bn_running_var]}
        bias=t694 {folded from=[p_s4_b3_conv1_bn_weight,p_s4_b3_conv1_bn_bias,b_s4_b3_conv1_bn_running_mean,b_s4_b3_conv1_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n431 {pt2=root[93] torch.ops.aten.relu.default}: [t786 f32 [H=7 W=7 C=368] {derived} ->[n432]] =
      relu x=t785 {derived} <-n430
    n432 {derived}: [t787 f32 [H=7 W=7 C=368] {derived} ->[n433]] =
      conv2d
        x=t786 {derived} <-n431
        weight=t695 {folded from=[p_s4_b3_conv2_conv_weight,p_s4_b3_conv2_bn_weight,b_s4_b3_conv2_bn_running_var]}
        bias=t696 {folded from=[p_s4_b3_conv2_bn_weight,p_s4_b3_conv2_bn_bias,b_s4_b3_conv2_bn_running_mean,b_s4_b3_conv2_bn_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=368;
               groups=46}
    n433 {pt2=root[96] torch.ops.aten.relu.default}: [t788 f32 [H=7 W=7 C=368] {derived} ->[n434]] =
      relu x=t787 {derived} <-n432
    n434 {derived}: [t789 f32 [H=7 W=7 C=368] {derived} ->[n435]] =
      conv2d
        x=t788 {derived} <-n433
        weight=t697 {folded from=[p_s4_b3_conv3_conv_weight,p_s4_b3_conv3_bn_weight,b_s4_b3_conv3_bn_running_var]}
        bias=t698 {folded from=[p_s4_b3_conv3_bn_weight,p_s4_b3_conv3_bn_bias,b_s4_b3_conv3_bn_running_mean,b_s4_b3_conv3_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n435 {pt2=root[99] torch.ops.aten.add.Tensor}: [t790 f32 [H=7 W=7 C=368] {derived} ->[n436]] =
      add a=t789 {derived} <-n434 b=t784 {derived} <-n429
    n436 {pt2=root[100] torch.ops.aten.relu.default}: [t791 f32 [H=7 W=7 C=368] {derived} ->[n437,
                                                                      n442]] =
      relu x=t790 {derived} <-n435
    n437 {derived}: [t792 f32 [H=7 W=7 C=368] {derived} ->[n438]] =
      conv2d
        x=t791 {derived} <-n436
        weight=t699 {folded from=[p_s4_b4_conv1_conv_weight,p_s4_b4_conv1_bn_weight,b_s4_b4_conv1_bn_running_var]}
        bias=t700 {folded from=[p_s4_b4_conv1_bn_weight,p_s4_b4_conv1_bn_bias,b_s4_b4_conv1_bn_running_mean,b_s4_b4_conv1_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n438 {pt2=root[103] torch.ops.aten.relu.default}: [t793 f32 [H=7 W=7 C=368] {derived} ->[n439]] =
      relu x=t792 {derived} <-n437
    n439 {derived}: [t794 f32 [H=7 W=7 C=368] {derived} ->[n440]] =
      conv2d
        x=t793 {derived} <-n438
        weight=t701 {folded from=[p_s4_b4_conv2_conv_weight,p_s4_b4_conv2_bn_weight,b_s4_b4_conv2_bn_running_var]}
        bias=t702 {folded from=[p_s4_b4_conv2_bn_weight,p_s4_b4_conv2_bn_bias,b_s4_b4_conv2_bn_running_mean,b_s4_b4_conv2_bn_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=368;
               groups=46}
    n440 {pt2=root[106] torch.ops.aten.relu.default}: [t795 f32 [H=7 W=7 C=368] {derived} ->[n441]] =
      relu x=t794 {derived} <-n439
    n441 {derived}: [t796 f32 [H=7 W=7 C=368] {derived} ->[n442]] =
      conv2d
        x=t795 {derived} <-n440
        weight=t703 {folded from=[p_s4_b4_conv3_conv_weight,p_s4_b4_conv3_bn_weight,b_s4_b4_conv3_bn_running_var]}
        bias=t704 {folded from=[p_s4_b4_conv3_bn_weight,p_s4_b4_conv3_bn_bias,b_s4_b4_conv3_bn_running_mean,b_s4_b4_conv3_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n442 {pt2=root[109] torch.ops.aten.add.Tensor}: [t797 f32 [H=7 W=7 C=368] {derived} ->[n443]] =
      add a=t796 {derived} <-n441 b=t791 {derived} <-n436
    n443 {pt2=root[110] torch.ops.aten.relu.default}: [t798 f32 [H=7 W=7 C=368] {derived} ->[n444,
                                                                      n449]] =
      relu x=t797 {derived} <-n442
    n444 {derived}: [t799 f32 [H=7 W=7 C=368] {derived} ->[n445]] =
      conv2d
        x=t798 {derived} <-n443
        weight=t705 {folded from=[p_s4_b5_conv1_conv_weight,p_s4_b5_conv1_bn_weight,b_s4_b5_conv1_bn_running_var]}
        bias=t706 {folded from=[p_s4_b5_conv1_bn_weight,p_s4_b5_conv1_bn_bias,b_s4_b5_conv1_bn_running_mean,b_s4_b5_conv1_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n445 {pt2=root[113] torch.ops.aten.relu.default}: [t800 f32 [H=7 W=7 C=368] {derived} ->[n446]] =
      relu x=t799 {derived} <-n444
    n446 {derived}: [t801 f32 [H=7 W=7 C=368] {derived} ->[n447]] =
      conv2d
        x=t800 {derived} <-n445
        weight=t707 {folded from=[p_s4_b5_conv2_conv_weight,p_s4_b5_conv2_bn_weight,b_s4_b5_conv2_bn_running_var]}
        bias=t708 {folded from=[p_s4_b5_conv2_bn_weight,p_s4_b5_conv2_bn_bias,b_s4_b5_conv2_bn_running_mean,b_s4_b5_conv2_bn_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=368;
               groups=46}
    n447 {pt2=root[116] torch.ops.aten.relu.default}: [t802 f32 [H=7 W=7 C=368] {derived} ->[n448]] =
      relu x=t801 {derived} <-n446
    n448 {derived}: [t803 f32 [H=7 W=7 C=368] {derived} ->[n449]] =
      conv2d
        x=t802 {derived} <-n447
        weight=t709 {folded from=[p_s4_b5_conv3_conv_weight,p_s4_b5_conv3_bn_weight,b_s4_b5_conv3_bn_running_var]}
        bias=t710 {folded from=[p_s4_b5_conv3_bn_weight,p_s4_b5_conv3_bn_bias,b_s4_b5_conv3_bn_running_mean,b_s4_b5_conv3_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n449 {pt2=root[119] torch.ops.aten.add.Tensor}: [t804 f32 [H=7 W=7 C=368] {derived} ->[n450]] =
      add a=t803 {derived} <-n448 b=t798 {derived} <-n443
    n450 {pt2=root[120] torch.ops.aten.relu.default}: [t805 f32 [H=7 W=7 C=368] {derived} ->[n451,
                                                                      n456]] =
      relu x=t804 {derived} <-n449
    n451 {derived}: [t806 f32 [H=7 W=7 C=368] {derived} ->[n452]] =
      conv2d
        x=t805 {derived} <-n450
        weight=t711 {folded from=[p_s4_b6_conv1_conv_weight,p_s4_b6_conv1_bn_weight,b_s4_b6_conv1_bn_running_var]}
        bias=t712 {folded from=[p_s4_b6_conv1_bn_weight,p_s4_b6_conv1_bn_bias,b_s4_b6_conv1_bn_running_mean,b_s4_b6_conv1_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n452 {pt2=root[123] torch.ops.aten.relu.default}: [t807 f32 [H=7 W=7 C=368] {derived} ->[n453]] =
      relu x=t806 {derived} <-n451
    n453 {derived}: [t808 f32 [H=7 W=7 C=368] {derived} ->[n454]] =
      conv2d
        x=t807 {derived} <-n452
        weight=t713 {folded from=[p_s4_b6_conv2_conv_weight,p_s4_b6_conv2_bn_weight,b_s4_b6_conv2_bn_running_var]}
        bias=t714 {folded from=[p_s4_b6_conv2_bn_weight,p_s4_b6_conv2_bn_bias,b_s4_b6_conv2_bn_running_mean,b_s4_b6_conv2_bn_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=368;
               groups=46}
    n454 {pt2=root[126] torch.ops.aten.relu.default}: [t809 f32 [H=7 W=7 C=368] {derived} ->[n455]] =
      relu x=t808 {derived} <-n453
    n455 {derived}: [t810 f32 [H=7 W=7 C=368] {derived} ->[n456]] =
      conv2d
        x=t809 {derived} <-n454
        weight=t715 {folded from=[p_s4_b6_conv3_conv_weight,p_s4_b6_conv3_bn_weight,b_s4_b6_conv3_bn_running_var]}
        bias=t716 {folded from=[p_s4_b6_conv3_bn_weight,p_s4_b6_conv3_bn_bias,b_s4_b6_conv3_bn_running_mean,b_s4_b6_conv3_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n456 {pt2=root[129] torch.ops.aten.add.Tensor}: [t811 f32 [H=7 W=7 C=368] {derived} ->[n457]] =
      add a=t810 {derived} <-n455 b=t805 {derived} <-n450
    n457 {pt2=root[130] torch.ops.aten.relu.default}: [t812 f32 [H=7 W=7 C=368] {derived} ->[n458,
                                                                      n463]] =
      relu x=t811 {derived} <-n456
    n458 {derived}: [t813 f32 [H=7 W=7 C=368] {derived} ->[n459]] =
      conv2d
        x=t812 {derived} <-n457
        weight=t717 {folded from=[p_s4_b7_conv1_conv_weight,p_s4_b7_conv1_bn_weight,b_s4_b7_conv1_bn_running_var]}
        bias=t718 {folded from=[p_s4_b7_conv1_bn_weight,p_s4_b7_conv1_bn_bias,b_s4_b7_conv1_bn_running_mean,b_s4_b7_conv1_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n459 {pt2=root[133] torch.ops.aten.relu.default}: [t814 f32 [H=7 W=7 C=368] {derived} ->[n460]] =
      relu x=t813 {derived} <-n458
    n460 {derived}: [t815 f32 [H=7 W=7 C=368] {derived} ->[n461]] =
      conv2d
        x=t814 {derived} <-n459
        weight=t719 {folded from=[p_s4_b7_conv2_conv_weight,p_s4_b7_conv2_bn_weight,b_s4_b7_conv2_bn_running_var]}
        bias=t720 {folded from=[p_s4_b7_conv2_bn_weight,p_s4_b7_conv2_bn_bias,b_s4_b7_conv2_bn_running_mean,b_s4_b7_conv2_bn_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=368;
               groups=46}
    n461 {pt2=root[136] torch.ops.aten.relu.default}: [t816 f32 [H=7 W=7 C=368] {derived} ->[n462]] =
      relu x=t815 {derived} <-n460
    n462 {derived}: [t817 f32 [H=7 W=7 C=368] {derived} ->[n463]] =
      conv2d
        x=t816 {derived} <-n461
        weight=t721 {folded from=[p_s4_b7_conv3_conv_weight,p_s4_b7_conv3_bn_weight,b_s4_b7_conv3_bn_running_var]}
        bias=t722 {folded from=[p_s4_b7_conv3_bn_weight,p_s4_b7_conv3_bn_bias,b_s4_b7_conv3_bn_running_mean,b_s4_b7_conv3_bn_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=368;
               groups=1}
    n463 {pt2=root[139] torch.ops.aten.add.Tensor}: [t818 f32 [H=7 W=7 C=368] {derived} ->[n464]] =
      add a=t817 {derived} <-n462 b=t812 {derived} <-n457
    n464 {pt2=root[140] torch.ops.aten.relu.default}: [t819 f32 [H=7 W=7 C=368] {derived} ->[n362]] =
      relu x=t818 {derived} <-n463
    group g89 torch.ops.aten.adaptive_avg_pool2d.default:
      n362 {derived}: [t629 f32 [C=368] {pt2=root:view} ->[n365]] =
        adaptive_avg_pool2d
          x=t819 {derived} <-n464
          params={output_size={h=1; w=1}}
    n365 {pt2=root[143] torch.ops.aten.clone.default}: [t632 f32 [C=368] {pt2=root:clone} ->[n367]] =
      clone x=t629 {pt2=root:view} <-n362
    group g90 torch.ops.aten.linear.default:
      n367 {pt2=root[144] torch.ops.aten.linear.default}: [t634 f32 [C=1000] {pt2=root:linear}] =
        linear
          x=t632 {pt2=root:clone} <-n365
          weight=t633 {folded from=[p_head_fc_weight]}
          bias=t133 {pt2=root:p_head_fc_bias target=head.fc.bias}
          params={in_features=368}
  outputs: [t634 f32 [C=1000] {pt2=root:linear} <-n367]
