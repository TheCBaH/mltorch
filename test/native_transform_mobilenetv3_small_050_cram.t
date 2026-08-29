MobileNetV3-small-050 through the same structural pipeline as RegNetX-002
(`test/native_transform_regnetx_002_cram.t`).

  $ ../bin/native_graph.exe transform --pt2 "$PT2_DATA/mobilenetv3_small_050/mobilenetv3_small_050.pt2"
  nodes: 387 -> 159
  constants: 108, of which 88 folded
  graph
  inputs:
    [t7 f32 [C=8] {pt2=root:p_blocks_0_0_se_conv_reduce_bias target=blocks.0.0.se.conv_reduce.bias} ->[n19] constant,
     t9 f32 [C=16] {pt2=root:p_blocks_0_0_se_conv_expand_bias target=blocks.0.0.se.conv_expand.bias} ->[n24] constant,
     t38 f32 [C=16] {pt2=root:p_blocks_2_0_se_conv_reduce_bias target=blocks.2.0.se.conv_reduce.bias} ->[n101] constant,
     t40 f32 [C=64] {pt2=root:p_blocks_2_0_se_conv_expand_bias target=blocks.2.0.se.conv_expand.bias} ->[n106] constant,
     t51 f32 [C=40] {pt2=root:p_blocks_2_1_se_conv_reduce_bias target=blocks.2.1.se.conv_reduce.bias} ->[n136] constant,
     t53 f32 [C=144] {pt2=root:p_blocks_2_1_se_conv_expand_bias target=blocks.2.1.se.conv_expand.bias} ->[n141] constant,
     t64 f32 [C=40] {pt2=root:p_blocks_2_2_se_conv_reduce_bias target=blocks.2.2.se.conv_reduce.bias} ->[n172] constant,
     t66 f32 [C=144] {pt2=root:p_blocks_2_2_se_conv_expand_bias target=blocks.2.2.se.conv_expand.bias} ->[n177] constant,
     t77 f32 [C=24] {pt2=root:p_blocks_3_0_se_conv_reduce_bias target=blocks.3.0.se.conv_reduce.bias} ->[n208] constant,
     t79 f32 [C=72] {pt2=root:p_blocks_3_0_se_conv_expand_bias target=blocks.3.0.se.conv_expand.bias} ->[n213] constant,
     t90 f32 [C=24] {pt2=root:p_blocks_3_1_se_conv_reduce_bias target=blocks.3.1.se.conv_reduce.bias} ->[n244] constant,
     t92 f32 [C=72] {pt2=root:p_blocks_3_1_se_conv_expand_bias target=blocks.3.1.se.conv_expand.bias} ->[n249] constant,
     t103 f32 [C=40] {pt2=root:p_blocks_4_0_se_conv_reduce_bias target=blocks.4.0.se.conv_reduce.bias} ->[n280] constant,
     t105 f32 [C=144] {pt2=root:p_blocks_4_0_se_conv_expand_bias target=blocks.4.0.se.conv_expand.bias} ->[n285] constant,
     t116 f32 [C=72] {pt2=root:p_blocks_4_1_se_conv_reduce_bias target=blocks.4.1.se.conv_reduce.bias} ->[n315] constant,
     t118 f32 [C=288] {pt2=root:p_blocks_4_1_se_conv_expand_bias target=blocks.4.1.se.conv_expand.bias} ->[n320] constant,
     t129 f32 [C=72] {pt2=root:p_blocks_4_2_se_conv_reduce_bias target=blocks.4.2.se.conv_reduce.bias} ->[n351] constant,
     t131 f32 [C=288] {pt2=root:p_blocks_4_2_se_conv_expand_bias target=blocks.4.2.se.conv_expand.bias} ->[n356] constant,
     t139 f32 [C=1024] {pt2=root:p_conv_head_bias target=conv_head.bias} ->[n381] constant,
     t141 f32 [C=1000] {pt2=root:p_classifier_bias target=classifier.bias} ->[n386] constant,
     t244 f32 [H=3 W=224 C=224] {pt2=root:x} ->[n0],
     t263 f32 [N=8 T=1 D=1 H=1 W=1 C=16] {folded from=[p_blocks_0_0_se_conv_reduce_weight]} ->[n19] constant,
     t268 f32 [N=16 T=1 D=1 H=1 W=1 C=8] {folded from=[p_blocks_0_0_se_conv_expand_weight]} ->[n24] constant,
     t345 f32 [N=16 T=1 D=1 H=1 W=1 C=64] {folded from=[p_blocks_2_0_se_conv_reduce_weight]} ->[n101] constant,
     t350 f32 [N=64 T=1 D=1 H=1 W=1 C=16] {folded from=[p_blocks_2_0_se_conv_expand_weight]} ->[n106] constant,
     t380 f32 [N=40 T=1 D=1 H=1 W=1 C=144] {folded from=[p_blocks_2_1_se_conv_reduce_weight]} ->[n136] constant,
     t385 f32 [N=144 T=1 D=1 H=1 W=1 C=40] {folded from=[p_blocks_2_1_se_conv_expand_weight]} ->[n141] constant,
     t416 f32 [N=40 T=1 D=1 H=1 W=1 C=144] {folded from=[p_blocks_2_2_se_conv_reduce_weight]} ->[n172] constant,
     t421 f32 [N=144 T=1 D=1 H=1 W=1 C=40] {folded from=[p_blocks_2_2_se_conv_expand_weight]} ->[n177] constant,
     t452 f32 [N=24 T=1 D=1 H=1 W=1 C=72] {folded from=[p_blocks_3_0_se_conv_reduce_weight]} ->[n208] constant,
     t457 f32 [N=72 T=1 D=1 H=1 W=1 C=24] {folded from=[p_blocks_3_0_se_conv_expand_weight]} ->[n213] constant,
     t488 f32 [N=24 T=1 D=1 H=1 W=1 C=72] {folded from=[p_blocks_3_1_se_conv_reduce_weight]} ->[n244] constant,
     t493 f32 [N=72 T=1 D=1 H=1 W=1 C=24] {folded from=[p_blocks_3_1_se_conv_expand_weight]} ->[n249] constant,
     t524 f32 [N=40 T=1 D=1 H=1 W=1 C=144] {folded from=[p_blocks_4_0_se_conv_reduce_weight]} ->[n280] constant,
     t529 f32 [N=144 T=1 D=1 H=1 W=1 C=40] {folded from=[p_blocks_4_0_se_conv_expand_weight]} ->[n285] constant,
     t559 f32 [N=72 T=1 D=1 H=1 W=1 C=288] {folded from=[p_blocks_4_1_se_conv_reduce_weight]} ->[n315] constant,
     t564 f32 [N=288 T=1 D=1 H=1 W=1 C=72] {folded from=[p_blocks_4_1_se_conv_expand_weight]} ->[n320] constant,
     t595 f32 [N=72 T=1 D=1 H=1 W=1 C=288] {folded from=[p_blocks_4_2_se_conv_reduce_weight]} ->[n351] constant,
     t600 f32 [N=288 T=1 D=1 H=1 W=1 C=72] {folded from=[p_blocks_4_2_se_conv_expand_weight]} ->[n356] constant,
     t625 f32 [N=1024 T=1 D=1 H=1 W=1 C=288] {folded from=[p_conv_head_weight]} ->[n381] constant,
     t630 f32 [N=1000 T=1 D=1 H=1 W=1 C=1024] {folded from=[p_classifier_weight]} ->[n386] constant,
     t632 f32 [N=16 T=1 D=1 H=3 W=3 C=3] {folded from=[p_conv_stem_weight,p_bn1_weight,b_bn1_running_var]} ->[n387] constant,
     t633 f32 [C=16] {folded from=[p_bn1_weight,p_bn1_bias,b_bn1_running_mean,b_bn1_running_var]} ->[n387] constant,
     t634 f32 [N=16 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_0_0_conv_dw_weight,p_blocks_0_0_bn1_weight,b_blocks_0_0_bn1_running_var]} ->[n389] constant,
     t635 f32 [C=16] {folded from=[p_blocks_0_0_bn1_weight,p_blocks_0_0_bn1_bias,b_blocks_0_0_bn1_running_mean,b_blocks_0_0_bn1_running_var]} ->[n389] constant,
     t636 f32 [N=8 T=1 D=1 H=1 W=1 C=16] {folded from=[p_blocks_0_0_conv_pw_weight,p_blocks_0_0_bn2_weight,b_blocks_0_0_bn2_running_var]} ->[n395] constant,
     t637 f32 [C=8] {folded from=[p_blocks_0_0_bn2_weight,p_blocks_0_0_bn2_bias,b_blocks_0_0_bn2_running_mean,b_blocks_0_0_bn2_running_var]} ->[n395] constant,
     t638 f32 [N=40 T=1 D=1 H=1 W=1 C=8] {folded from=[p_blocks_1_0_conv_pw_weight,p_blocks_1_0_bn1_weight,b_blocks_1_0_bn1_running_var]} ->[n396] constant,
     t639 f32 [C=40] {folded from=[p_blocks_1_0_bn1_weight,p_blocks_1_0_bn1_bias,b_blocks_1_0_bn1_running_mean,b_blocks_1_0_bn1_running_var]} ->[n396] constant,
     t640 f32 [N=40 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_1_0_conv_dw_weight,p_blocks_1_0_bn2_weight,b_blocks_1_0_bn2_running_var]} ->[n398] constant,
     t641 f32 [C=40] {folded from=[p_blocks_1_0_bn2_weight,p_blocks_1_0_bn2_bias,b_blocks_1_0_bn2_running_mean,b_blocks_1_0_bn2_running_var]} ->[n398] constant,
     t642 f32 [N=16 T=1 D=1 H=1 W=1 C=40] {folded from=[p_blocks_1_0_conv_pwl_weight,p_blocks_1_0_bn3_weight,b_blocks_1_0_bn3_running_var]} ->[n400] constant,
     t643 f32 [C=16] {folded from=[p_blocks_1_0_bn3_weight,p_blocks_1_0_bn3_bias,b_blocks_1_0_bn3_running_mean,b_blocks_1_0_bn3_running_var]} ->[n400] constant,
     t644 f32 [N=56 T=1 D=1 H=1 W=1 C=16] {folded from=[p_blocks_1_1_conv_pw_weight,p_blocks_1_1_bn1_weight,b_blocks_1_1_bn1_running_var]} ->[n401] constant,
     t645 f32 [C=56] {folded from=[p_blocks_1_1_bn1_weight,p_blocks_1_1_bn1_bias,b_blocks_1_1_bn1_running_mean,b_blocks_1_1_bn1_running_var]} ->[n401] constant,
     t646 f32 [N=56 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_1_1_conv_dw_weight,p_blocks_1_1_bn2_weight,b_blocks_1_1_bn2_running_var]} ->[n403] constant,
     t647 f32 [C=56] {folded from=[p_blocks_1_1_bn2_weight,p_blocks_1_1_bn2_bias,b_blocks_1_1_bn2_running_mean,b_blocks_1_1_bn2_running_var]} ->[n403] constant,
     t648 f32 [N=16 T=1 D=1 H=1 W=1 C=56] {folded from=[p_blocks_1_1_conv_pwl_weight,p_blocks_1_1_bn3_weight,b_blocks_1_1_bn3_running_var]} ->[n405] constant,
     t649 f32 [C=16] {folded from=[p_blocks_1_1_bn3_weight,p_blocks_1_1_bn3_bias,b_blocks_1_1_bn3_running_mean,b_blocks_1_1_bn3_running_var]} ->[n405] constant,
     t650 f32 [N=64 T=1 D=1 H=1 W=1 C=16] {folded from=[p_blocks_2_0_conv_pw_weight,p_blocks_2_0_bn1_weight,b_blocks_2_0_bn1_running_var]} ->[n407] constant,
     t651 f32 [C=64] {folded from=[p_blocks_2_0_bn1_weight,p_blocks_2_0_bn1_bias,b_blocks_2_0_bn1_running_mean,b_blocks_2_0_bn1_running_var]} ->[n407] constant,
     t652 f32 [N=64 T=1 D=1 H=5 W=5 C=1] {folded from=[p_blocks_2_0_conv_dw_weight,p_blocks_2_0_bn2_weight,b_blocks_2_0_bn2_running_var]} ->[n409] constant,
     t653 f32 [C=64] {folded from=[p_blocks_2_0_bn2_weight,p_blocks_2_0_bn2_bias,b_blocks_2_0_bn2_running_mean,b_blocks_2_0_bn2_running_var]} ->[n409] constant,
     t654 f32 [N=24 T=1 D=1 H=1 W=1 C=64] {folded from=[p_blocks_2_0_conv_pwl_weight,p_blocks_2_0_bn3_weight,b_blocks_2_0_bn3_running_var]} ->[n415] constant,
     t655 f32 [C=24] {folded from=[p_blocks_2_0_bn3_weight,p_blocks_2_0_bn3_bias,b_blocks_2_0_bn3_running_mean,b_blocks_2_0_bn3_running_var]} ->[n415] constant,
     t656 f32 [N=144 T=1 D=1 H=1 W=1 C=24] {folded from=[p_blocks_2_1_conv_pw_weight,p_blocks_2_1_bn1_weight,b_blocks_2_1_bn1_running_var]} ->[n416] constant,
     t657 f32 [C=144] {folded from=[p_blocks_2_1_bn1_weight,p_blocks_2_1_bn1_bias,b_blocks_2_1_bn1_running_mean,b_blocks_2_1_bn1_running_var]} ->[n416] constant,
     t658 f32 [N=144 T=1 D=1 H=5 W=5 C=1] {folded from=[p_blocks_2_1_conv_dw_weight,p_blocks_2_1_bn2_weight,b_blocks_2_1_bn2_running_var]} ->[n418] constant,
     t659 f32 [C=144] {folded from=[p_blocks_2_1_bn2_weight,p_blocks_2_1_bn2_bias,b_blocks_2_1_bn2_running_mean,b_blocks_2_1_bn2_running_var]} ->[n418] constant,
     t660 f32 [N=24 T=1 D=1 H=1 W=1 C=144] {folded from=[p_blocks_2_1_conv_pwl_weight,p_blocks_2_1_bn3_weight,b_blocks_2_1_bn3_running_var]} ->[n424] constant,
     t661 f32 [C=24] {folded from=[p_blocks_2_1_bn3_weight,p_blocks_2_1_bn3_bias,b_blocks_2_1_bn3_running_mean,b_blocks_2_1_bn3_running_var]} ->[n424] constant,
     t662 f32 [N=144 T=1 D=1 H=1 W=1 C=24] {folded from=[p_blocks_2_2_conv_pw_weight,p_blocks_2_2_bn1_weight,b_blocks_2_2_bn1_running_var]} ->[n426] constant,
     t663 f32 [C=144] {folded from=[p_blocks_2_2_bn1_weight,p_blocks_2_2_bn1_bias,b_blocks_2_2_bn1_running_mean,b_blocks_2_2_bn1_running_var]} ->[n426] constant,
     t664 f32 [N=144 T=1 D=1 H=5 W=5 C=1] {folded from=[p_blocks_2_2_conv_dw_weight,p_blocks_2_2_bn2_weight,b_blocks_2_2_bn2_running_var]} ->[n428] constant,
     t665 f32 [C=144] {folded from=[p_blocks_2_2_bn2_weight,p_blocks_2_2_bn2_bias,b_blocks_2_2_bn2_running_mean,b_blocks_2_2_bn2_running_var]} ->[n428] constant,
     t666 f32 [N=24 T=1 D=1 H=1 W=1 C=144] {folded from=[p_blocks_2_2_conv_pwl_weight,p_blocks_2_2_bn3_weight,b_blocks_2_2_bn3_running_var]} ->[n434] constant,
     t667 f32 [C=24] {folded from=[p_blocks_2_2_bn3_weight,p_blocks_2_2_bn3_bias,b_blocks_2_2_bn3_running_mean,b_blocks_2_2_bn3_running_var]} ->[n434] constant,
     t668 f32 [N=72 T=1 D=1 H=1 W=1 C=24] {folded from=[p_blocks_3_0_conv_pw_weight,p_blocks_3_0_bn1_weight,b_blocks_3_0_bn1_running_var]} ->[n436] constant,
     t669 f32 [C=72] {folded from=[p_blocks_3_0_bn1_weight,p_blocks_3_0_bn1_bias,b_blocks_3_0_bn1_running_mean,b_blocks_3_0_bn1_running_var]} ->[n436] constant,
     t670 f32 [N=72 T=1 D=1 H=5 W=5 C=1] {folded from=[p_blocks_3_0_conv_dw_weight,p_blocks_3_0_bn2_weight,b_blocks_3_0_bn2_running_var]} ->[n438] constant,
     t671 f32 [C=72] {folded from=[p_blocks_3_0_bn2_weight,p_blocks_3_0_bn2_bias,b_blocks_3_0_bn2_running_mean,b_blocks_3_0_bn2_running_var]} ->[n438] constant,
     t672 f32 [N=24 T=1 D=1 H=1 W=1 C=72] {folded from=[p_blocks_3_0_conv_pwl_weight,p_blocks_3_0_bn3_weight,b_blocks_3_0_bn3_running_var]} ->[n444] constant,
     t673 f32 [C=24] {folded from=[p_blocks_3_0_bn3_weight,p_blocks_3_0_bn3_bias,b_blocks_3_0_bn3_running_mean,b_blocks_3_0_bn3_running_var]} ->[n444] constant,
     t674 f32 [N=72 T=1 D=1 H=1 W=1 C=24] {folded from=[p_blocks_3_1_conv_pw_weight,p_blocks_3_1_bn1_weight,b_blocks_3_1_bn1_running_var]} ->[n446] constant,
     t675 f32 [C=72] {folded from=[p_blocks_3_1_bn1_weight,p_blocks_3_1_bn1_bias,b_blocks_3_1_bn1_running_mean,b_blocks_3_1_bn1_running_var]} ->[n446] constant,
     t676 f32 [N=72 T=1 D=1 H=5 W=5 C=1] {folded from=[p_blocks_3_1_conv_dw_weight,p_blocks_3_1_bn2_weight,b_blocks_3_1_bn2_running_var]} ->[n448] constant,
     t677 f32 [C=72] {folded from=[p_blocks_3_1_bn2_weight,p_blocks_3_1_bn2_bias,b_blocks_3_1_bn2_running_mean,b_blocks_3_1_bn2_running_var]} ->[n448] constant,
     t678 f32 [N=24 T=1 D=1 H=1 W=1 C=72] {folded from=[p_blocks_3_1_conv_pwl_weight,p_blocks_3_1_bn3_weight,b_blocks_3_1_bn3_running_var]} ->[n454] constant,
     t679 f32 [C=24] {folded from=[p_blocks_3_1_bn3_weight,p_blocks_3_1_bn3_bias,b_blocks_3_1_bn3_running_mean,b_blocks_3_1_bn3_running_var]} ->[n454] constant,
     t680 f32 [N=144 T=1 D=1 H=1 W=1 C=24] {folded from=[p_blocks_4_0_conv_pw_weight,p_blocks_4_0_bn1_weight,b_blocks_4_0_bn1_running_var]} ->[n456] constant,
     t681 f32 [C=144] {folded from=[p_blocks_4_0_bn1_weight,p_blocks_4_0_bn1_bias,b_blocks_4_0_bn1_running_mean,b_blocks_4_0_bn1_running_var]} ->[n456] constant,
     t682 f32 [N=144 T=1 D=1 H=5 W=5 C=1] {folded from=[p_blocks_4_0_conv_dw_weight,p_blocks_4_0_bn2_weight,b_blocks_4_0_bn2_running_var]} ->[n458] constant,
     t683 f32 [C=144] {folded from=[p_blocks_4_0_bn2_weight,p_blocks_4_0_bn2_bias,b_blocks_4_0_bn2_running_mean,b_blocks_4_0_bn2_running_var]} ->[n458] constant,
     t684 f32 [N=48 T=1 D=1 H=1 W=1 C=144] {folded from=[p_blocks_4_0_conv_pwl_weight,p_blocks_4_0_bn3_weight,b_blocks_4_0_bn3_running_var]} ->[n464] constant,
     t685 f32 [C=48] {folded from=[p_blocks_4_0_bn3_weight,p_blocks_4_0_bn3_bias,b_blocks_4_0_bn3_running_mean,b_blocks_4_0_bn3_running_var]} ->[n464] constant,
     t686 f32 [N=288 T=1 D=1 H=1 W=1 C=48] {folded from=[p_blocks_4_1_conv_pw_weight,p_blocks_4_1_bn1_weight,b_blocks_4_1_bn1_running_var]} ->[n465] constant,
     t687 f32 [C=288] {folded from=[p_blocks_4_1_bn1_weight,p_blocks_4_1_bn1_bias,b_blocks_4_1_bn1_running_mean,b_blocks_4_1_bn1_running_var]} ->[n465] constant,
     t688 f32 [N=288 T=1 D=1 H=5 W=5 C=1] {folded from=[p_blocks_4_1_conv_dw_weight,p_blocks_4_1_bn2_weight,b_blocks_4_1_bn2_running_var]} ->[n467] constant,
     t689 f32 [C=288] {folded from=[p_blocks_4_1_bn2_weight,p_blocks_4_1_bn2_bias,b_blocks_4_1_bn2_running_mean,b_blocks_4_1_bn2_running_var]} ->[n467] constant,
     t690 f32 [N=48 T=1 D=1 H=1 W=1 C=288] {folded from=[p_blocks_4_1_conv_pwl_weight,p_blocks_4_1_bn3_weight,b_blocks_4_1_bn3_running_var]} ->[n473] constant,
     t691 f32 [C=48] {folded from=[p_blocks_4_1_bn3_weight,p_blocks_4_1_bn3_bias,b_blocks_4_1_bn3_running_mean,b_blocks_4_1_bn3_running_var]} ->[n473] constant,
     t692 f32 [N=288 T=1 D=1 H=1 W=1 C=48] {folded from=[p_blocks_4_2_conv_pw_weight,p_blocks_4_2_bn1_weight,b_blocks_4_2_bn1_running_var]} ->[n475] constant,
     t693 f32 [C=288] {folded from=[p_blocks_4_2_bn1_weight,p_blocks_4_2_bn1_bias,b_blocks_4_2_bn1_running_mean,b_blocks_4_2_bn1_running_var]} ->[n475] constant,
     t694 f32 [N=288 T=1 D=1 H=5 W=5 C=1] {folded from=[p_blocks_4_2_conv_dw_weight,p_blocks_4_2_bn2_weight,b_blocks_4_2_bn2_running_var]} ->[n477] constant,
     t695 f32 [C=288] {folded from=[p_blocks_4_2_bn2_weight,p_blocks_4_2_bn2_bias,b_blocks_4_2_bn2_running_mean,b_blocks_4_2_bn2_running_var]} ->[n477] constant,
     t696 f32 [N=48 T=1 D=1 H=1 W=1 C=288] {folded from=[p_blocks_4_2_conv_pwl_weight,p_blocks_4_2_bn3_weight,b_blocks_4_2_bn3_running_var]} ->[n483] constant,
     t697 f32 [C=48] {folded from=[p_blocks_4_2_bn3_weight,p_blocks_4_2_bn3_bias,b_blocks_4_2_bn3_running_mean,b_blocks_4_2_bn3_running_var]} ->[n483] constant,
     t698 f32 [N=288 T=1 D=1 H=1 W=1 C=48] {folded from=[p_blocks_5_0_conv_weight,p_blocks_5_0_bn1_weight,b_blocks_5_0_bn1_running_var]} ->[n485] constant,
     t699 f32 [C=288] {folded from=[p_blocks_5_0_bn1_weight,p_blocks_5_0_bn1_bias,b_blocks_5_0_bn1_running_mean,b_blocks_5_0_bn1_running_var]} ->[n485] constant]
  nodes:
    group g1 torch.ops.aten.conv2d.default:
      n0 {derived}: [t245 f32 [H=224 W=224 C=3] {derived} ->[n387]] =
        permute x=t244 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
    n387 {derived}: [t700 f32 [H=112 W=112 C=16] {derived} ->[n388]] =
      conv2d
        x=t245 {derived} <-n0
        weight=t632 {folded from=[p_conv_stem_weight,p_bn1_weight,b_bn1_running_var]}
        bias=t633 {folded from=[p_bn1_weight,p_bn1_bias,b_bn1_running_mean,b_bn1_running_var]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=3;
               groups=1}
    n388 {pt2=root[2] torch.ops.aten.hardswish.default}: [t701 f32 [H=112 W=112
                                                                    C=16] {derived} ->[n389]] =
      hardswish x=t700 {derived} <-n387
    n389 {derived}: [t702 f32 [H=56 W=56 C=16] {derived} ->[n390]] =
      conv2d
        x=t701 {derived} <-n388
        weight=t634 {folded from=[p_blocks_0_0_conv_dw_weight,p_blocks_0_0_bn1_weight,b_blocks_0_0_bn1_running_var]}
        bias=t635 {folded from=[p_blocks_0_0_bn1_weight,p_blocks_0_0_bn1_bias,b_blocks_0_0_bn1_running_mean,b_blocks_0_0_bn1_running_var]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=16;
               groups=16}
    n390 {pt2=root[5] torch.ops.aten.relu.default}: [t703 f32 [H=56 W=56 C=16] {derived} ->[n391]] =
      relu x=t702 {derived} <-n389
    n391 {pt2=root[4] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t260 f32 [H=16
                                                                      W=56
                                                                      C=56] {pt2=root:relu} ->[n16,
                                                                      n27]] =
      permute x=t703 {derived} <-n390 perm=[H<-C, W<-H, C<-W]
    n16 {pt2=root[6] torch.ops.aten.mean.dim}: [t261 f32 [H=16 W=1 C=1] {pt2=root:mean} ->[n17]] =
      mean x=t260 {pt2=root:relu} <-n391 params={dims=[W, C]; keepdim=true}
    group g5 torch.ops.aten.conv2d.default:
      n17 {derived}: [t262 f32 [C=16] {derived} ->[n19]] =
        permute x=t261 {pt2=root:mean} <-n16 perm=[H<-W, W<-C, C<-H]
      n19 {derived}: [t264 f32 [C=8] {derived} ->[n392]] =
        conv2d
          x=t262 {derived} <-n17
          weight=t263 {folded from=[p_blocks_0_0_se_conv_reduce_weight]}
          bias=t7 {pt2=root:p_blocks_0_0_se_conv_reduce_bias target=blocks.0.0.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=16;
                 groups=1}
    n392 {pt2=root[8] torch.ops.aten.relu.default}: [t704 f32 [C=8] {derived} ->[n24]] =
      relu x=t264 {derived} <-n19
    group g6 torch.ops.aten.conv2d.default:
      n24 {derived}: [t269 f32 [C=16] {derived} ->[n393]] =
        conv2d
          x=t704 {derived} <-n392
          weight=t268 {folded from=[p_blocks_0_0_se_conv_expand_weight]}
          bias=t9 {pt2=root:p_blocks_0_0_se_conv_expand_bias target=blocks.0.0.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=8;
                 groups=1}
    n393 {pt2=root[10] torch.ops.aten.hardsigmoid.default}: [t705 f32 [C=16] {derived} ->[n394]] =
      hardsigmoid x=t269 {derived} <-n24
    n394 {pt2=root[9] torch.ops.aten.conv2d.default}: [t271 f32 [H=16 W=1 C=1] {pt2=root:hardsigmoid} ->[n27]] =
      permute x=t705 {derived} <-n393 perm=[H<-C, W<-H, C<-W]
    n27 {pt2=root[11] torch.ops.aten.mul.Tensor}: [t272 f32 [H=16 W=56 C=56] {pt2=root:mul} ->[n28]] =
      mul a=t260 {pt2=root:relu} <-n391 b=t271 {pt2=root:hardsigmoid} <-n394
    group g7 torch.ops.aten.conv2d.default:
      n28 {derived}: [t273 f32 [H=56 W=56 C=16] {derived} ->[n395]] =
        permute x=t272 {pt2=root:mul} <-n27 perm=[H<-W, W<-C, C<-H]
    n395 {derived}: [t706 f32 [H=56 W=56 C=8] {derived} ->[n396]] =
      conv2d
        x=t273 {derived} <-n28
        weight=t636 {folded from=[p_blocks_0_0_conv_pw_weight,p_blocks_0_0_bn2_weight,b_blocks_0_0_bn2_running_var]}
        bias=t637 {folded from=[p_blocks_0_0_bn2_weight,p_blocks_0_0_bn2_bias,b_blocks_0_0_bn2_running_mean,b_blocks_0_0_bn2_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=16;
               groups=1}
    n396 {derived}: [t707 f32 [H=56 W=56 C=40] {derived} ->[n397]] =
      conv2d
        x=t706 {derived} <-n395
        weight=t638 {folded from=[p_blocks_1_0_conv_pw_weight,p_blocks_1_0_bn1_weight,b_blocks_1_0_bn1_running_var]}
        bias=t639 {folded from=[p_blocks_1_0_bn1_weight,p_blocks_1_0_bn1_bias,b_blocks_1_0_bn1_running_mean,b_blocks_1_0_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=8;
               groups=1}
    n397 {pt2=root[16] torch.ops.aten.relu.default}: [t708 f32 [H=56 W=56 C=40] {derived} ->[n398]] =
      relu x=t707 {derived} <-n396
    n398 {derived}: [t709 f32 [H=28 W=28 C=40] {derived} ->[n399]] =
      conv2d
        x=t708 {derived} <-n397
        weight=t640 {folded from=[p_blocks_1_0_conv_dw_weight,p_blocks_1_0_bn2_weight,b_blocks_1_0_bn2_running_var]}
        bias=t641 {folded from=[p_blocks_1_0_bn2_weight,p_blocks_1_0_bn2_bias,b_blocks_1_0_bn2_running_mean,b_blocks_1_0_bn2_running_var]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=40;
               groups=40}
    n399 {pt2=root[19] torch.ops.aten.relu.default}: [t710 f32 [H=28 W=28 C=40] {derived} ->[n400]] =
      relu x=t709 {derived} <-n398
    n400 {derived}: [t711 f32 [H=28 W=28 C=16] {derived} ->[n401, n406]] =
      conv2d
        x=t710 {derived} <-n399
        weight=t642 {folded from=[p_blocks_1_0_conv_pwl_weight,p_blocks_1_0_bn3_weight,b_blocks_1_0_bn3_running_var]}
        bias=t643 {folded from=[p_blocks_1_0_bn3_weight,p_blocks_1_0_bn3_bias,b_blocks_1_0_bn3_running_mean,b_blocks_1_0_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=40;
               groups=1}
    n401 {derived}: [t712 f32 [H=28 W=28 C=56] {derived} ->[n402]] =
      conv2d
        x=t711 {derived} <-n400
        weight=t644 {folded from=[p_blocks_1_1_conv_pw_weight,p_blocks_1_1_bn1_weight,b_blocks_1_1_bn1_running_var]}
        bias=t645 {folded from=[p_blocks_1_1_bn1_weight,p_blocks_1_1_bn1_bias,b_blocks_1_1_bn1_running_mean,b_blocks_1_1_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=16;
               groups=1}
    n402 {pt2=root[24] torch.ops.aten.relu.default}: [t713 f32 [H=28 W=28 C=56] {derived} ->[n403]] =
      relu x=t712 {derived} <-n401
    n403 {derived}: [t714 f32 [H=28 W=28 C=56] {derived} ->[n404]] =
      conv2d
        x=t713 {derived} <-n402
        weight=t646 {folded from=[p_blocks_1_1_conv_dw_weight,p_blocks_1_1_bn2_weight,b_blocks_1_1_bn2_running_var]}
        bias=t647 {folded from=[p_blocks_1_1_bn2_weight,p_blocks_1_1_bn2_bias,b_blocks_1_1_bn2_running_mean,b_blocks_1_1_bn2_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=56;
               groups=56}
    n404 {pt2=root[27] torch.ops.aten.relu.default}: [t715 f32 [H=28 W=28 C=56] {derived} ->[n405]] =
      relu x=t714 {derived} <-n403
    n405 {derived}: [t716 f32 [H=28 W=28 C=16] {derived} ->[n406]] =
      conv2d
        x=t715 {derived} <-n404
        weight=t648 {folded from=[p_blocks_1_1_conv_pwl_weight,p_blocks_1_1_bn3_weight,b_blocks_1_1_bn3_running_var]}
        bias=t649 {folded from=[p_blocks_1_1_bn3_weight,p_blocks_1_1_bn3_bias,b_blocks_1_1_bn3_running_mean,b_blocks_1_1_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=56;
               groups=1}
    n406 {pt2=root[30] torch.ops.aten.add.Tensor}: [t717 f32 [H=28 W=28 C=16] {derived} ->[n407]] =
      add a=t716 {derived} <-n405 b=t711 {derived} <-n400
    n407 {derived}: [t718 f32 [H=28 W=28 C=64] {derived} ->[n408]] =
      conv2d
        x=t717 {derived} <-n406
        weight=t650 {folded from=[p_blocks_2_0_conv_pw_weight,p_blocks_2_0_bn1_weight,b_blocks_2_0_bn1_running_var]}
        bias=t651 {folded from=[p_blocks_2_0_bn1_weight,p_blocks_2_0_bn1_bias,b_blocks_2_0_bn1_running_mean,b_blocks_2_0_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=16;
               groups=1}
    n408 {pt2=root[33] torch.ops.aten.hardswish.default}: [t719 f32 [H=28 W=28
                                                                     C=64] {derived} ->[n409]] =
      hardswish x=t718 {derived} <-n407
    n409 {derived}: [t720 f32 [H=14 W=14 C=64] {derived} ->[n410]] =
      conv2d
        x=t719 {derived} <-n408
        weight=t652 {folded from=[p_blocks_2_0_conv_dw_weight,p_blocks_2_0_bn2_weight,b_blocks_2_0_bn2_running_var]}
        bias=t653 {folded from=[p_blocks_2_0_bn2_weight,p_blocks_2_0_bn2_bias,b_blocks_2_0_bn2_running_mean,b_blocks_2_0_bn2_running_var]}
        params={h={kernel=5; stride=2; pad_before=2; pad_after=2; dilation=1};
               w={kernel=5; stride=2; pad_before=2; pad_after=2; dilation=1};
               in_channels=64;
               groups=64}
    n410 {pt2=root[36] torch.ops.aten.hardswish.default}: [t721 f32 [H=14 W=14
                                                                     C=64] {derived} ->[n411]] =
      hardswish x=t720 {derived} <-n409
    n411 {pt2=root[35] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t342 f32 [H=64
                                                                      W=14
                                                                      C=14] {pt2=root:hardswish_2} ->[n98,
                                                                      n109]] =
      permute x=t721 {derived} <-n410 perm=[H<-C, W<-H, C<-W]
    n98 {pt2=root[37] torch.ops.aten.mean.dim}: [t343 f32 [H=64 W=1 C=1] {pt2=root:mean_1} ->[n99]] =
      mean
        x=t342 {pt2=root:hardswish_2} <-n411
        params={dims=[W, C]; keepdim=true}
    group g25 torch.ops.aten.conv2d.default:
      n99 {derived}: [t344 f32 [C=64] {derived} ->[n101]] =
        permute x=t343 {pt2=root:mean_1} <-n98 perm=[H<-W, W<-C, C<-H]
      n101 {derived}: [t346 f32 [C=16] {derived} ->[n412]] =
        conv2d
          x=t344 {derived} <-n99
          weight=t345 {folded from=[p_blocks_2_0_se_conv_reduce_weight]}
          bias=t38 {pt2=root:p_blocks_2_0_se_conv_reduce_bias target=blocks.2.0.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=64;
                 groups=1}
    n412 {pt2=root[39] torch.ops.aten.relu.default}: [t722 f32 [C=16] {derived} ->[n106]] =
      relu x=t346 {derived} <-n101
    group g26 torch.ops.aten.conv2d.default:
      n106 {derived}: [t351 f32 [C=64] {derived} ->[n413]] =
        conv2d
          x=t722 {derived} <-n412
          weight=t350 {folded from=[p_blocks_2_0_se_conv_expand_weight]}
          bias=t40 {pt2=root:p_blocks_2_0_se_conv_expand_bias target=blocks.2.0.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=16;
                 groups=1}
    n413 {pt2=root[41] torch.ops.aten.hardsigmoid.default}: [t723 f32 [C=64] {derived} ->[n414]] =
      hardsigmoid x=t351 {derived} <-n106
    n414 {pt2=root[40] torch.ops.aten.conv2d.default}: [t353 f32 [H=64 W=1 C=1] {pt2=root:hardsigmoid_1} ->[n109]] =
      permute x=t723 {derived} <-n413 perm=[H<-C, W<-H, C<-W]
    n109 {pt2=root[42] torch.ops.aten.mul.Tensor}: [t354 f32 [H=64 W=14 C=14] {pt2=root:mul_1} ->[n110]] =
      mul
        a=t342 {pt2=root:hardswish_2} <-n411
        b=t353 {pt2=root:hardsigmoid_1} <-n414
    group g27 torch.ops.aten.conv2d.default:
      n110 {derived}: [t355 f32 [H=14 W=14 C=64] {derived} ->[n415]] =
        permute x=t354 {pt2=root:mul_1} <-n109 perm=[H<-W, W<-C, C<-H]
    n415 {derived}: [t724 f32 [H=14 W=14 C=24] {derived} ->[n416, n425]] =
      conv2d
        x=t355 {derived} <-n110
        weight=t654 {folded from=[p_blocks_2_0_conv_pwl_weight,p_blocks_2_0_bn3_weight,b_blocks_2_0_bn3_running_var]}
        bias=t655 {folded from=[p_blocks_2_0_bn3_weight,p_blocks_2_0_bn3_bias,b_blocks_2_0_bn3_running_mean,b_blocks_2_0_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=64;
               groups=1}
    n416 {derived}: [t725 f32 [H=14 W=14 C=144] {derived} ->[n417]] =
      conv2d
        x=t724 {derived} <-n415
        weight=t656 {folded from=[p_blocks_2_1_conv_pw_weight,p_blocks_2_1_bn1_weight,b_blocks_2_1_bn1_running_var]}
        bias=t657 {folded from=[p_blocks_2_1_bn1_weight,p_blocks_2_1_bn1_bias,b_blocks_2_1_bn1_running_mean,b_blocks_2_1_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=24;
               groups=1}
    n417 {pt2=root[47] torch.ops.aten.hardswish.default}: [t726 f32 [H=14 W=14
                                                                     C=144] {derived} ->[n418]] =
      hardswish x=t725 {derived} <-n416
    n418 {derived}: [t727 f32 [H=14 W=14 C=144] {derived} ->[n419]] =
      conv2d
        x=t726 {derived} <-n417
        weight=t658 {folded from=[p_blocks_2_1_conv_dw_weight,p_blocks_2_1_bn2_weight,b_blocks_2_1_bn2_running_var]}
        bias=t659 {folded from=[p_blocks_2_1_bn2_weight,p_blocks_2_1_bn2_bias,b_blocks_2_1_bn2_running_mean,b_blocks_2_1_bn2_running_var]}
        params={h={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
               w={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
               in_channels=144;
               groups=144}
    n419 {pt2=root[50] torch.ops.aten.hardswish.default}: [t728 f32 [H=14 W=14
                                                                     C=144] {derived} ->[n420]] =
      hardswish x=t727 {derived} <-n418
    n420 {pt2=root[49] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t377 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:hardswish_4} ->[n133,
                                                                      n144]] =
      permute x=t728 {derived} <-n419 perm=[H<-C, W<-H, C<-W]
    n133 {pt2=root[51] torch.ops.aten.mean.dim}: [t378 f32 [H=144 W=1 C=1] {pt2=root:mean_2} ->[n134]] =
      mean
        x=t377 {pt2=root:hardswish_4} <-n420
        params={dims=[W, C]; keepdim=true}
    group g33 torch.ops.aten.conv2d.default:
      n134 {derived}: [t379 f32 [C=144] {derived} ->[n136]] =
        permute x=t378 {pt2=root:mean_2} <-n133 perm=[H<-W, W<-C, C<-H]
      n136 {derived}: [t381 f32 [C=40] {derived} ->[n421]] =
        conv2d
          x=t379 {derived} <-n134
          weight=t380 {folded from=[p_blocks_2_1_se_conv_reduce_weight]}
          bias=t51 {pt2=root:p_blocks_2_1_se_conv_reduce_bias target=blocks.2.1.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=144;
                 groups=1}
    n421 {pt2=root[53] torch.ops.aten.relu.default}: [t729 f32 [C=40] {derived} ->[n141]] =
      relu x=t381 {derived} <-n136
    group g34 torch.ops.aten.conv2d.default:
      n141 {derived}: [t386 f32 [C=144] {derived} ->[n422]] =
        conv2d
          x=t729 {derived} <-n421
          weight=t385 {folded from=[p_blocks_2_1_se_conv_expand_weight]}
          bias=t53 {pt2=root:p_blocks_2_1_se_conv_expand_bias target=blocks.2.1.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=40;
                 groups=1}
    n422 {pt2=root[55] torch.ops.aten.hardsigmoid.default}: [t730 f32 [C=144] {derived} ->[n423]] =
      hardsigmoid x=t386 {derived} <-n141
    n423 {pt2=root[54] torch.ops.aten.conv2d.default}: [t388 f32 [H=144 W=1
                                                                  C=1] {pt2=root:hardsigmoid_2} ->[n144]] =
      permute x=t730 {derived} <-n422 perm=[H<-C, W<-H, C<-W]
    n144 {pt2=root[56] torch.ops.aten.mul.Tensor}: [t389 f32 [H=144 W=14 C=14] {pt2=root:mul_2} ->[n145]] =
      mul
        a=t377 {pt2=root:hardswish_4} <-n420
        b=t388 {pt2=root:hardsigmoid_2} <-n423
    group g35 torch.ops.aten.conv2d.default:
      n145 {derived}: [t390 f32 [H=14 W=14 C=144] {derived} ->[n424]] =
        permute x=t389 {pt2=root:mul_2} <-n144 perm=[H<-W, W<-C, C<-H]
    n424 {derived}: [t731 f32 [H=14 W=14 C=24] {derived} ->[n425]] =
      conv2d
        x=t390 {derived} <-n145
        weight=t660 {folded from=[p_blocks_2_1_conv_pwl_weight,p_blocks_2_1_bn3_weight,b_blocks_2_1_bn3_running_var]}
        bias=t661 {folded from=[p_blocks_2_1_bn3_weight,p_blocks_2_1_bn3_bias,b_blocks_2_1_bn3_running_mean,b_blocks_2_1_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=144;
               groups=1}
    n425 {pt2=root[59] torch.ops.aten.add.Tensor}: [t732 f32 [H=14 W=14 C=24] {derived} ->[n426,
                                                                      n435]] =
      add a=t731 {derived} <-n424 b=t724 {derived} <-n415
    n426 {derived}: [t733 f32 [H=14 W=14 C=144] {derived} ->[n427]] =
      conv2d
        x=t732 {derived} <-n425
        weight=t662 {folded from=[p_blocks_2_2_conv_pw_weight,p_blocks_2_2_bn1_weight,b_blocks_2_2_bn1_running_var]}
        bias=t663 {folded from=[p_blocks_2_2_bn1_weight,p_blocks_2_2_bn1_bias,b_blocks_2_2_bn1_running_mean,b_blocks_2_2_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=24;
               groups=1}
    n427 {pt2=root[62] torch.ops.aten.hardswish.default}: [t734 f32 [H=14 W=14
                                                                     C=144] {derived} ->[n428]] =
      hardswish x=t733 {derived} <-n426
    n428 {derived}: [t735 f32 [H=14 W=14 C=144] {derived} ->[n429]] =
      conv2d
        x=t734 {derived} <-n427
        weight=t664 {folded from=[p_blocks_2_2_conv_dw_weight,p_blocks_2_2_bn2_weight,b_blocks_2_2_bn2_running_var]}
        bias=t665 {folded from=[p_blocks_2_2_bn2_weight,p_blocks_2_2_bn2_bias,b_blocks_2_2_bn2_running_mean,b_blocks_2_2_bn2_running_var]}
        params={h={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
               w={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
               in_channels=144;
               groups=144}
    n429 {pt2=root[65] torch.ops.aten.hardswish.default}: [t736 f32 [H=14 W=14
                                                                     C=144] {derived} ->[n430]] =
      hardswish x=t735 {derived} <-n428
    n430 {pt2=root[64] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t413 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:hardswish_6} ->[n169,
                                                                      n180]] =
      permute x=t736 {derived} <-n429 perm=[H<-C, W<-H, C<-W]
    n169 {pt2=root[66] torch.ops.aten.mean.dim}: [t414 f32 [H=144 W=1 C=1] {pt2=root:mean_3} ->[n170]] =
      mean
        x=t413 {pt2=root:hardswish_6} <-n430
        params={dims=[W, C]; keepdim=true}
    group g41 torch.ops.aten.conv2d.default:
      n170 {derived}: [t415 f32 [C=144] {derived} ->[n172]] =
        permute x=t414 {pt2=root:mean_3} <-n169 perm=[H<-W, W<-C, C<-H]
      n172 {derived}: [t417 f32 [C=40] {derived} ->[n431]] =
        conv2d
          x=t415 {derived} <-n170
          weight=t416 {folded from=[p_blocks_2_2_se_conv_reduce_weight]}
          bias=t64 {pt2=root:p_blocks_2_2_se_conv_reduce_bias target=blocks.2.2.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=144;
                 groups=1}
    n431 {pt2=root[68] torch.ops.aten.relu.default}: [t737 f32 [C=40] {derived} ->[n177]] =
      relu x=t417 {derived} <-n172
    group g42 torch.ops.aten.conv2d.default:
      n177 {derived}: [t422 f32 [C=144] {derived} ->[n432]] =
        conv2d
          x=t737 {derived} <-n431
          weight=t421 {folded from=[p_blocks_2_2_se_conv_expand_weight]}
          bias=t66 {pt2=root:p_blocks_2_2_se_conv_expand_bias target=blocks.2.2.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=40;
                 groups=1}
    n432 {pt2=root[70] torch.ops.aten.hardsigmoid.default}: [t738 f32 [C=144] {derived} ->[n433]] =
      hardsigmoid x=t422 {derived} <-n177
    n433 {pt2=root[69] torch.ops.aten.conv2d.default}: [t424 f32 [H=144 W=1
                                                                  C=1] {pt2=root:hardsigmoid_3} ->[n180]] =
      permute x=t738 {derived} <-n432 perm=[H<-C, W<-H, C<-W]
    n180 {pt2=root[71] torch.ops.aten.mul.Tensor}: [t425 f32 [H=144 W=14 C=14] {pt2=root:mul_3} ->[n181]] =
      mul
        a=t413 {pt2=root:hardswish_6} <-n430
        b=t424 {pt2=root:hardsigmoid_3} <-n433
    group g43 torch.ops.aten.conv2d.default:
      n181 {derived}: [t426 f32 [H=14 W=14 C=144] {derived} ->[n434]] =
        permute x=t425 {pt2=root:mul_3} <-n180 perm=[H<-W, W<-C, C<-H]
    n434 {derived}: [t739 f32 [H=14 W=14 C=24] {derived} ->[n435]] =
      conv2d
        x=t426 {derived} <-n181
        weight=t666 {folded from=[p_blocks_2_2_conv_pwl_weight,p_blocks_2_2_bn3_weight,b_blocks_2_2_bn3_running_var]}
        bias=t667 {folded from=[p_blocks_2_2_bn3_weight,p_blocks_2_2_bn3_bias,b_blocks_2_2_bn3_running_mean,b_blocks_2_2_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=144;
               groups=1}
    n435 {pt2=root[74] torch.ops.aten.add.Tensor}: [t740 f32 [H=14 W=14 C=24] {derived} ->[n436,
                                                                      n445]] =
      add a=t739 {derived} <-n434 b=t732 {derived} <-n425
    n436 {derived}: [t741 f32 [H=14 W=14 C=72] {derived} ->[n437]] =
      conv2d
        x=t740 {derived} <-n435
        weight=t668 {folded from=[p_blocks_3_0_conv_pw_weight,p_blocks_3_0_bn1_weight,b_blocks_3_0_bn1_running_var]}
        bias=t669 {folded from=[p_blocks_3_0_bn1_weight,p_blocks_3_0_bn1_bias,b_blocks_3_0_bn1_running_mean,b_blocks_3_0_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=24;
               groups=1}
    n437 {pt2=root[77] torch.ops.aten.hardswish.default}: [t742 f32 [H=14 W=14
                                                                     C=72] {derived} ->[n438]] =
      hardswish x=t741 {derived} <-n436
    n438 {derived}: [t743 f32 [H=14 W=14 C=72] {derived} ->[n439]] =
      conv2d
        x=t742 {derived} <-n437
        weight=t670 {folded from=[p_blocks_3_0_conv_dw_weight,p_blocks_3_0_bn2_weight,b_blocks_3_0_bn2_running_var]}
        bias=t671 {folded from=[p_blocks_3_0_bn2_weight,p_blocks_3_0_bn2_bias,b_blocks_3_0_bn2_running_mean,b_blocks_3_0_bn2_running_var]}
        params={h={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
               w={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
               in_channels=72;
               groups=72}
    n439 {pt2=root[80] torch.ops.aten.hardswish.default}: [t744 f32 [H=14 W=14
                                                                     C=72] {derived} ->[n440]] =
      hardswish x=t743 {derived} <-n438
    n440 {pt2=root[79] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t449 f32 [H=72
                                                                      W=14
                                                                      C=14] {pt2=root:hardswish_8} ->[n205,
                                                                      n216]] =
      permute x=t744 {derived} <-n439 perm=[H<-C, W<-H, C<-W]
    n205 {pt2=root[81] torch.ops.aten.mean.dim}: [t450 f32 [H=72 W=1 C=1] {pt2=root:mean_4} ->[n206]] =
      mean
        x=t449 {pt2=root:hardswish_8} <-n440
        params={dims=[W, C]; keepdim=true}
    group g49 torch.ops.aten.conv2d.default:
      n206 {derived}: [t451 f32 [C=72] {derived} ->[n208]] =
        permute x=t450 {pt2=root:mean_4} <-n205 perm=[H<-W, W<-C, C<-H]
      n208 {derived}: [t453 f32 [C=24] {derived} ->[n441]] =
        conv2d
          x=t451 {derived} <-n206
          weight=t452 {folded from=[p_blocks_3_0_se_conv_reduce_weight]}
          bias=t77 {pt2=root:p_blocks_3_0_se_conv_reduce_bias target=blocks.3.0.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=72;
                 groups=1}
    n441 {pt2=root[83] torch.ops.aten.relu.default}: [t745 f32 [C=24] {derived} ->[n213]] =
      relu x=t453 {derived} <-n208
    group g50 torch.ops.aten.conv2d.default:
      n213 {derived}: [t458 f32 [C=72] {derived} ->[n442]] =
        conv2d
          x=t745 {derived} <-n441
          weight=t457 {folded from=[p_blocks_3_0_se_conv_expand_weight]}
          bias=t79 {pt2=root:p_blocks_3_0_se_conv_expand_bias target=blocks.3.0.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=24;
                 groups=1}
    n442 {pt2=root[85] torch.ops.aten.hardsigmoid.default}: [t746 f32 [C=72] {derived} ->[n443]] =
      hardsigmoid x=t458 {derived} <-n213
    n443 {pt2=root[84] torch.ops.aten.conv2d.default}: [t460 f32 [H=72 W=1 C=1] {pt2=root:hardsigmoid_4} ->[n216]] =
      permute x=t746 {derived} <-n442 perm=[H<-C, W<-H, C<-W]
    n216 {pt2=root[86] torch.ops.aten.mul.Tensor}: [t461 f32 [H=72 W=14 C=14] {pt2=root:mul_4} ->[n217]] =
      mul
        a=t449 {pt2=root:hardswish_8} <-n440
        b=t460 {pt2=root:hardsigmoid_4} <-n443
    group g51 torch.ops.aten.conv2d.default:
      n217 {derived}: [t462 f32 [H=14 W=14 C=72] {derived} ->[n444]] =
        permute x=t461 {pt2=root:mul_4} <-n216 perm=[H<-W, W<-C, C<-H]
    n444 {derived}: [t747 f32 [H=14 W=14 C=24] {derived} ->[n445]] =
      conv2d
        x=t462 {derived} <-n217
        weight=t672 {folded from=[p_blocks_3_0_conv_pwl_weight,p_blocks_3_0_bn3_weight,b_blocks_3_0_bn3_running_var]}
        bias=t673 {folded from=[p_blocks_3_0_bn3_weight,p_blocks_3_0_bn3_bias,b_blocks_3_0_bn3_running_mean,b_blocks_3_0_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=72;
               groups=1}
    n445 {pt2=root[89] torch.ops.aten.add.Tensor}: [t748 f32 [H=14 W=14 C=24] {derived} ->[n446,
                                                                      n455]] =
      add a=t747 {derived} <-n444 b=t740 {derived} <-n435
    n446 {derived}: [t749 f32 [H=14 W=14 C=72] {derived} ->[n447]] =
      conv2d
        x=t748 {derived} <-n445
        weight=t674 {folded from=[p_blocks_3_1_conv_pw_weight,p_blocks_3_1_bn1_weight,b_blocks_3_1_bn1_running_var]}
        bias=t675 {folded from=[p_blocks_3_1_bn1_weight,p_blocks_3_1_bn1_bias,b_blocks_3_1_bn1_running_mean,b_blocks_3_1_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=24;
               groups=1}
    n447 {pt2=root[92] torch.ops.aten.hardswish.default}: [t750 f32 [H=14 W=14
                                                                     C=72] {derived} ->[n448]] =
      hardswish x=t749 {derived} <-n446
    n448 {derived}: [t751 f32 [H=14 W=14 C=72] {derived} ->[n449]] =
      conv2d
        x=t750 {derived} <-n447
        weight=t676 {folded from=[p_blocks_3_1_conv_dw_weight,p_blocks_3_1_bn2_weight,b_blocks_3_1_bn2_running_var]}
        bias=t677 {folded from=[p_blocks_3_1_bn2_weight,p_blocks_3_1_bn2_bias,b_blocks_3_1_bn2_running_mean,b_blocks_3_1_bn2_running_var]}
        params={h={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
               w={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
               in_channels=72;
               groups=72}
    n449 {pt2=root[95] torch.ops.aten.hardswish.default}: [t752 f32 [H=14 W=14
                                                                     C=72] {derived} ->[n450]] =
      hardswish x=t751 {derived} <-n448
    n450 {pt2=root[94] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t485 f32 [H=72
                                                                      W=14
                                                                      C=14] {pt2=root:hardswish_10} ->[n241,
                                                                      n252]] =
      permute x=t752 {derived} <-n449 perm=[H<-C, W<-H, C<-W]
    n241 {pt2=root[96] torch.ops.aten.mean.dim}: [t486 f32 [H=72 W=1 C=1] {pt2=root:mean_5} ->[n242]] =
      mean
        x=t485 {pt2=root:hardswish_10} <-n450
        params={dims=[W, C]; keepdim=true}
    group g57 torch.ops.aten.conv2d.default:
      n242 {derived}: [t487 f32 [C=72] {derived} ->[n244]] =
        permute x=t486 {pt2=root:mean_5} <-n241 perm=[H<-W, W<-C, C<-H]
      n244 {derived}: [t489 f32 [C=24] {derived} ->[n451]] =
        conv2d
          x=t487 {derived} <-n242
          weight=t488 {folded from=[p_blocks_3_1_se_conv_reduce_weight]}
          bias=t90 {pt2=root:p_blocks_3_1_se_conv_reduce_bias target=blocks.3.1.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=72;
                 groups=1}
    n451 {pt2=root[98] torch.ops.aten.relu.default}: [t753 f32 [C=24] {derived} ->[n249]] =
      relu x=t489 {derived} <-n244
    group g58 torch.ops.aten.conv2d.default:
      n249 {derived}: [t494 f32 [C=72] {derived} ->[n452]] =
        conv2d
          x=t753 {derived} <-n451
          weight=t493 {folded from=[p_blocks_3_1_se_conv_expand_weight]}
          bias=t92 {pt2=root:p_blocks_3_1_se_conv_expand_bias target=blocks.3.1.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=24;
                 groups=1}
    n452 {pt2=root[100] torch.ops.aten.hardsigmoid.default}: [t754 f32 [C=72] {derived} ->[n453]] =
      hardsigmoid x=t494 {derived} <-n249
    n453 {pt2=root[99] torch.ops.aten.conv2d.default}: [t496 f32 [H=72 W=1 C=1] {pt2=root:hardsigmoid_5} ->[n252]] =
      permute x=t754 {derived} <-n452 perm=[H<-C, W<-H, C<-W]
    n252 {pt2=root[101] torch.ops.aten.mul.Tensor}: [t497 f32 [H=72 W=14 C=14] {pt2=root:mul_5} ->[n253]] =
      mul
        a=t485 {pt2=root:hardswish_10} <-n450
        b=t496 {pt2=root:hardsigmoid_5} <-n453
    group g59 torch.ops.aten.conv2d.default:
      n253 {derived}: [t498 f32 [H=14 W=14 C=72] {derived} ->[n454]] =
        permute x=t497 {pt2=root:mul_5} <-n252 perm=[H<-W, W<-C, C<-H]
    n454 {derived}: [t755 f32 [H=14 W=14 C=24] {derived} ->[n455]] =
      conv2d
        x=t498 {derived} <-n253
        weight=t678 {folded from=[p_blocks_3_1_conv_pwl_weight,p_blocks_3_1_bn3_weight,b_blocks_3_1_bn3_running_var]}
        bias=t679 {folded from=[p_blocks_3_1_bn3_weight,p_blocks_3_1_bn3_bias,b_blocks_3_1_bn3_running_mean,b_blocks_3_1_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=72;
               groups=1}
    n455 {pt2=root[104] torch.ops.aten.add.Tensor}: [t756 f32 [H=14 W=14 C=24] {derived} ->[n456]] =
      add a=t755 {derived} <-n454 b=t748 {derived} <-n445
    n456 {derived}: [t757 f32 [H=14 W=14 C=144] {derived} ->[n457]] =
      conv2d
        x=t756 {derived} <-n455
        weight=t680 {folded from=[p_blocks_4_0_conv_pw_weight,p_blocks_4_0_bn1_weight,b_blocks_4_0_bn1_running_var]}
        bias=t681 {folded from=[p_blocks_4_0_bn1_weight,p_blocks_4_0_bn1_bias,b_blocks_4_0_bn1_running_mean,b_blocks_4_0_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=24;
               groups=1}
    n457 {pt2=root[107] torch.ops.aten.hardswish.default}: [t758 f32 [H=14 W=14
                                                                      C=144] {derived} ->[n458]] =
      hardswish x=t757 {derived} <-n456
    n458 {derived}: [t759 f32 [H=7 W=7 C=144] {derived} ->[n459]] =
      conv2d
        x=t758 {derived} <-n457
        weight=t682 {folded from=[p_blocks_4_0_conv_dw_weight,p_blocks_4_0_bn2_weight,b_blocks_4_0_bn2_running_var]}
        bias=t683 {folded from=[p_blocks_4_0_bn2_weight,p_blocks_4_0_bn2_bias,b_blocks_4_0_bn2_running_mean,b_blocks_4_0_bn2_running_var]}
        params={h={kernel=5; stride=2; pad_before=2; pad_after=2; dilation=1};
               w={kernel=5; stride=2; pad_before=2; pad_after=2; dilation=1};
               in_channels=144;
               groups=144}
    n459 {pt2=root[110] torch.ops.aten.hardswish.default}: [t760 f32 [H=7 W=7
                                                                      C=144] {derived} ->[n460]] =
      hardswish x=t759 {derived} <-n458
    n460 {pt2=root[109] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t521 f32 [H=144
                                                                      W=7 C=7] {pt2=root:hardswish_12} ->[n277,
                                                                      n288]] =
      permute x=t760 {derived} <-n459 perm=[H<-C, W<-H, C<-W]
    n277 {pt2=root[111] torch.ops.aten.mean.dim}: [t522 f32 [H=144 W=1 C=1] {pt2=root:mean_6} ->[n278]] =
      mean
        x=t521 {pt2=root:hardswish_12} <-n460
        params={dims=[W, C]; keepdim=true}
    group g65 torch.ops.aten.conv2d.default:
      n278 {derived}: [t523 f32 [C=144] {derived} ->[n280]] =
        permute x=t522 {pt2=root:mean_6} <-n277 perm=[H<-W, W<-C, C<-H]
      n280 {derived}: [t525 f32 [C=40] {derived} ->[n461]] =
        conv2d
          x=t523 {derived} <-n278
          weight=t524 {folded from=[p_blocks_4_0_se_conv_reduce_weight]}
          bias=t103 {pt2=root:p_blocks_4_0_se_conv_reduce_bias target=blocks.4.0.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=144;
                 groups=1}
    n461 {pt2=root[113] torch.ops.aten.relu.default}: [t761 f32 [C=40] {derived} ->[n285]] =
      relu x=t525 {derived} <-n280
    group g66 torch.ops.aten.conv2d.default:
      n285 {derived}: [t530 f32 [C=144] {derived} ->[n462]] =
        conv2d
          x=t761 {derived} <-n461
          weight=t529 {folded from=[p_blocks_4_0_se_conv_expand_weight]}
          bias=t105 {pt2=root:p_blocks_4_0_se_conv_expand_bias target=blocks.4.0.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=40;
                 groups=1}
    n462 {pt2=root[115] torch.ops.aten.hardsigmoid.default}: [t762 f32 [C=144] {derived} ->[n463]] =
      hardsigmoid x=t530 {derived} <-n285
    n463 {pt2=root[114] torch.ops.aten.conv2d.default}: [t532 f32 [H=144 W=1
                                                                   C=1] {pt2=root:hardsigmoid_6} ->[n288]] =
      permute x=t762 {derived} <-n462 perm=[H<-C, W<-H, C<-W]
    n288 {pt2=root[116] torch.ops.aten.mul.Tensor}: [t533 f32 [H=144 W=7 C=7] {pt2=root:mul_6} ->[n289]] =
      mul
        a=t521 {pt2=root:hardswish_12} <-n460
        b=t532 {pt2=root:hardsigmoid_6} <-n463
    group g67 torch.ops.aten.conv2d.default:
      n289 {derived}: [t534 f32 [H=7 W=7 C=144] {derived} ->[n464]] =
        permute x=t533 {pt2=root:mul_6} <-n288 perm=[H<-W, W<-C, C<-H]
    n464 {derived}: [t763 f32 [H=7 W=7 C=48] {derived} ->[n465, n474]] =
      conv2d
        x=t534 {derived} <-n289
        weight=t684 {folded from=[p_blocks_4_0_conv_pwl_weight,p_blocks_4_0_bn3_weight,b_blocks_4_0_bn3_running_var]}
        bias=t685 {folded from=[p_blocks_4_0_bn3_weight,p_blocks_4_0_bn3_bias,b_blocks_4_0_bn3_running_mean,b_blocks_4_0_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=144;
               groups=1}
    n465 {derived}: [t764 f32 [H=7 W=7 C=288] {derived} ->[n466]] =
      conv2d
        x=t763 {derived} <-n464
        weight=t686 {folded from=[p_blocks_4_1_conv_pw_weight,p_blocks_4_1_bn1_weight,b_blocks_4_1_bn1_running_var]}
        bias=t687 {folded from=[p_blocks_4_1_bn1_weight,p_blocks_4_1_bn1_bias,b_blocks_4_1_bn1_running_mean,b_blocks_4_1_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=48;
               groups=1}
    n466 {pt2=root[121] torch.ops.aten.hardswish.default}: [t765 f32 [H=7 W=7
                                                                      C=288] {derived} ->[n467]] =
      hardswish x=t764 {derived} <-n465
    n467 {derived}: [t766 f32 [H=7 W=7 C=288] {derived} ->[n468]] =
      conv2d
        x=t765 {derived} <-n466
        weight=t688 {folded from=[p_blocks_4_1_conv_dw_weight,p_blocks_4_1_bn2_weight,b_blocks_4_1_bn2_running_var]}
        bias=t689 {folded from=[p_blocks_4_1_bn2_weight,p_blocks_4_1_bn2_bias,b_blocks_4_1_bn2_running_mean,b_blocks_4_1_bn2_running_var]}
        params={h={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
               w={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
               in_channels=288;
               groups=288}
    n468 {pt2=root[124] torch.ops.aten.hardswish.default}: [t767 f32 [H=7 W=7
                                                                      C=288] {derived} ->[n469]] =
      hardswish x=t766 {derived} <-n467
    n469 {pt2=root[123] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t556 f32 [H=288
                                                                      W=7 C=7] {pt2=root:hardswish_14} ->[n312,
                                                                      n323]] =
      permute x=t767 {derived} <-n468 perm=[H<-C, W<-H, C<-W]
    n312 {pt2=root[125] torch.ops.aten.mean.dim}: [t557 f32 [H=288 W=1 C=1] {pt2=root:mean_7} ->[n313]] =
      mean
        x=t556 {pt2=root:hardswish_14} <-n469
        params={dims=[W, C]; keepdim=true}
    group g73 torch.ops.aten.conv2d.default:
      n313 {derived}: [t558 f32 [C=288] {derived} ->[n315]] =
        permute x=t557 {pt2=root:mean_7} <-n312 perm=[H<-W, W<-C, C<-H]
      n315 {derived}: [t560 f32 [C=72] {derived} ->[n470]] =
        conv2d
          x=t558 {derived} <-n313
          weight=t559 {folded from=[p_blocks_4_1_se_conv_reduce_weight]}
          bias=t116 {pt2=root:p_blocks_4_1_se_conv_reduce_bias target=blocks.4.1.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=288;
                 groups=1}
    n470 {pt2=root[127] torch.ops.aten.relu.default}: [t768 f32 [C=72] {derived} ->[n320]] =
      relu x=t560 {derived} <-n315
    group g74 torch.ops.aten.conv2d.default:
      n320 {derived}: [t565 f32 [C=288] {derived} ->[n471]] =
        conv2d
          x=t768 {derived} <-n470
          weight=t564 {folded from=[p_blocks_4_1_se_conv_expand_weight]}
          bias=t118 {pt2=root:p_blocks_4_1_se_conv_expand_bias target=blocks.4.1.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=72;
                 groups=1}
    n471 {pt2=root[129] torch.ops.aten.hardsigmoid.default}: [t769 f32 [C=288] {derived} ->[n472]] =
      hardsigmoid x=t565 {derived} <-n320
    n472 {pt2=root[128] torch.ops.aten.conv2d.default}: [t567 f32 [H=288 W=1
                                                                   C=1] {pt2=root:hardsigmoid_7} ->[n323]] =
      permute x=t769 {derived} <-n471 perm=[H<-C, W<-H, C<-W]
    n323 {pt2=root[130] torch.ops.aten.mul.Tensor}: [t568 f32 [H=288 W=7 C=7] {pt2=root:mul_7} ->[n324]] =
      mul
        a=t556 {pt2=root:hardswish_14} <-n469
        b=t567 {pt2=root:hardsigmoid_7} <-n472
    group g75 torch.ops.aten.conv2d.default:
      n324 {derived}: [t569 f32 [H=7 W=7 C=288] {derived} ->[n473]] =
        permute x=t568 {pt2=root:mul_7} <-n323 perm=[H<-W, W<-C, C<-H]
    n473 {derived}: [t770 f32 [H=7 W=7 C=48] {derived} ->[n474]] =
      conv2d
        x=t569 {derived} <-n324
        weight=t690 {folded from=[p_blocks_4_1_conv_pwl_weight,p_blocks_4_1_bn3_weight,b_blocks_4_1_bn3_running_var]}
        bias=t691 {folded from=[p_blocks_4_1_bn3_weight,p_blocks_4_1_bn3_bias,b_blocks_4_1_bn3_running_mean,b_blocks_4_1_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=288;
               groups=1}
    n474 {pt2=root[133] torch.ops.aten.add.Tensor}: [t771 f32 [H=7 W=7 C=48] {derived} ->[n475,
                                                                      n484]] =
      add a=t770 {derived} <-n473 b=t763 {derived} <-n464
    n475 {derived}: [t772 f32 [H=7 W=7 C=288] {derived} ->[n476]] =
      conv2d
        x=t771 {derived} <-n474
        weight=t692 {folded from=[p_blocks_4_2_conv_pw_weight,p_blocks_4_2_bn1_weight,b_blocks_4_2_bn1_running_var]}
        bias=t693 {folded from=[p_blocks_4_2_bn1_weight,p_blocks_4_2_bn1_bias,b_blocks_4_2_bn1_running_mean,b_blocks_4_2_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=48;
               groups=1}
    n476 {pt2=root[136] torch.ops.aten.hardswish.default}: [t773 f32 [H=7 W=7
                                                                      C=288] {derived} ->[n477]] =
      hardswish x=t772 {derived} <-n475
    n477 {derived}: [t774 f32 [H=7 W=7 C=288] {derived} ->[n478]] =
      conv2d
        x=t773 {derived} <-n476
        weight=t694 {folded from=[p_blocks_4_2_conv_dw_weight,p_blocks_4_2_bn2_weight,b_blocks_4_2_bn2_running_var]}
        bias=t695 {folded from=[p_blocks_4_2_bn2_weight,p_blocks_4_2_bn2_bias,b_blocks_4_2_bn2_running_mean,b_blocks_4_2_bn2_running_var]}
        params={h={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
               w={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
               in_channels=288;
               groups=288}
    n478 {pt2=root[139] torch.ops.aten.hardswish.default}: [t775 f32 [H=7 W=7
                                                                      C=288] {derived} ->[n479]] =
      hardswish x=t774 {derived} <-n477
    n479 {pt2=root[138] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t592 f32 [H=288
                                                                      W=7 C=7] {pt2=root:hardswish_16} ->[n348,
                                                                      n359]] =
      permute x=t775 {derived} <-n478 perm=[H<-C, W<-H, C<-W]
    n348 {pt2=root[140] torch.ops.aten.mean.dim}: [t593 f32 [H=288 W=1 C=1] {pt2=root:mean_8} ->[n349]] =
      mean
        x=t592 {pt2=root:hardswish_16} <-n479
        params={dims=[W, C]; keepdim=true}
    group g81 torch.ops.aten.conv2d.default:
      n349 {derived}: [t594 f32 [C=288] {derived} ->[n351]] =
        permute x=t593 {pt2=root:mean_8} <-n348 perm=[H<-W, W<-C, C<-H]
      n351 {derived}: [t596 f32 [C=72] {derived} ->[n480]] =
        conv2d
          x=t594 {derived} <-n349
          weight=t595 {folded from=[p_blocks_4_2_se_conv_reduce_weight]}
          bias=t129 {pt2=root:p_blocks_4_2_se_conv_reduce_bias target=blocks.4.2.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=288;
                 groups=1}
    n480 {pt2=root[142] torch.ops.aten.relu.default}: [t776 f32 [C=72] {derived} ->[n356]] =
      relu x=t596 {derived} <-n351
    group g82 torch.ops.aten.conv2d.default:
      n356 {derived}: [t601 f32 [C=288] {derived} ->[n481]] =
        conv2d
          x=t776 {derived} <-n480
          weight=t600 {folded from=[p_blocks_4_2_se_conv_expand_weight]}
          bias=t131 {pt2=root:p_blocks_4_2_se_conv_expand_bias target=blocks.4.2.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=72;
                 groups=1}
    n481 {pt2=root[144] torch.ops.aten.hardsigmoid.default}: [t777 f32 [C=288] {derived} ->[n482]] =
      hardsigmoid x=t601 {derived} <-n356
    n482 {pt2=root[143] torch.ops.aten.conv2d.default}: [t603 f32 [H=288 W=1
                                                                   C=1] {pt2=root:hardsigmoid_8} ->[n359]] =
      permute x=t777 {derived} <-n481 perm=[H<-C, W<-H, C<-W]
    n359 {pt2=root[145] torch.ops.aten.mul.Tensor}: [t604 f32 [H=288 W=7 C=7] {pt2=root:mul_8} ->[n360]] =
      mul
        a=t592 {pt2=root:hardswish_16} <-n479
        b=t603 {pt2=root:hardsigmoid_8} <-n482
    group g83 torch.ops.aten.conv2d.default:
      n360 {derived}: [t605 f32 [H=7 W=7 C=288] {derived} ->[n483]] =
        permute x=t604 {pt2=root:mul_8} <-n359 perm=[H<-W, W<-C, C<-H]
    n483 {derived}: [t778 f32 [H=7 W=7 C=48] {derived} ->[n484]] =
      conv2d
        x=t605 {derived} <-n360
        weight=t696 {folded from=[p_blocks_4_2_conv_pwl_weight,p_blocks_4_2_bn3_weight,b_blocks_4_2_bn3_running_var]}
        bias=t697 {folded from=[p_blocks_4_2_bn3_weight,p_blocks_4_2_bn3_bias,b_blocks_4_2_bn3_running_mean,b_blocks_4_2_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=288;
               groups=1}
    n484 {pt2=root[148] torch.ops.aten.add.Tensor}: [t779 f32 [H=7 W=7 C=48] {derived} ->[n485]] =
      add a=t778 {derived} <-n483 b=t771 {derived} <-n474
    n485 {derived}: [t780 f32 [H=7 W=7 C=288] {derived} ->[n486]] =
      conv2d
        x=t779 {derived} <-n484
        weight=t698 {folded from=[p_blocks_5_0_conv_weight,p_blocks_5_0_bn1_weight,b_blocks_5_0_bn1_running_var]}
        bias=t699 {folded from=[p_blocks_5_0_bn1_weight,p_blocks_5_0_bn1_bias,b_blocks_5_0_bn1_running_mean,b_blocks_5_0_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=48;
               groups=1}
    n486 {pt2=root[151] torch.ops.aten.hardswish.default}: [t781 f32 [H=7 W=7
                                                                      C=288] {derived} ->[n377]] =
      hardswish x=t780 {derived} <-n485
    group g87 torch.ops.aten.adaptive_avg_pool2d.default:
      n377 {derived}: [t622 f32 [C=288] {derived} ->[n381]] =
        adaptive_avg_pool2d
          x=t781 {derived} <-n486
          params={output_size={h=1; w=1}}
    group g88 torch.ops.aten.conv2d.default:
      n381 {derived}: [t626 f32 [C=1024] {derived} ->[n487]] =
        conv2d
          x=t622 {derived} <-n377
          weight=t625 {folded from=[p_conv_head_weight]}
          bias=t139 {pt2=root:p_conv_head_bias target=conv_head.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=288;
                 groups=1}
    n487 {pt2=root[154] torch.ops.aten.hardswish.default}: [t782 f32 [C=1024] {pt2=root:view_1} ->[n386]] =
      hardswish x=t626 {derived} <-n381
    group g89 torch.ops.aten.linear.default:
      n386 {pt2=root[156] torch.ops.aten.linear.default}: [t631 f32 [C=1000] {pt2=root:linear}] =
        linear
          x=t782 {pt2=root:view_1} <-n487
          weight=t630 {folded from=[p_classifier_weight]}
          bias=t141 {pt2=root:p_classifier_bias target=classifier.bias}
          params={in_features=1024}
  outputs: [t631 f32 [C=1000] {pt2=root:linear} <-n386]
