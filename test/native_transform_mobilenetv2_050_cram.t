MobileNetV2-050 through the same structural pipeline as RegNetX-002
(`test/native_transform_regnetx_002_cram.t`). The graph is printed in full,
same reason as there: the point of these goldens is to be able to read the
resulting layout, and a node count alone cannot show that.

  $ ../bin/native_graph.exe transform --pt2 "$PT2_DATA/mobilenetv2_050/mobilenetv2_050.pt2"
  nodes: 415 -> 100
  constants: 106, of which 105 folded
  graph
  inputs:
    [t157 f32 [C=1000] {pt2=root:p_classifier_bias target=classifier.bias} ->[n414] constant,
     t314 f32 [H=3 W=224 C=224] {pt2=root:x} ->[n0],
     t728 f32 [N=1000 T=1 D=1 H=1 W=1 C=1280] {folded from=[p_classifier_weight]} ->[n414] constant,
     t730 f32 [N=16 T=1 D=1 H=3 W=3 C=3] {folded from=[p_conv_stem_weight,p_bn1_weight,b_bn1_running_var]} ->[n415] constant,
     t731 f32 [C=16] {folded from=[p_bn1_weight,p_bn1_bias,b_bn1_running_mean,b_bn1_running_var]} ->[n415] constant,
     t732 f32 [N=16 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_0_0_conv_dw_weight,p_blocks_0_0_bn1_weight,b_blocks_0_0_bn1_running_var]} ->[n417] constant,
     t733 f32 [C=16] {folded from=[p_blocks_0_0_bn1_weight,p_blocks_0_0_bn1_bias,b_blocks_0_0_bn1_running_mean,b_blocks_0_0_bn1_running_var]} ->[n417] constant,
     t734 f32 [N=8 T=1 D=1 H=1 W=1 C=16] {folded from=[p_blocks_0_0_conv_pw_weight,p_blocks_0_0_bn2_weight,b_blocks_0_0_bn2_running_var]} ->[n419] constant,
     t735 f32 [C=8] {folded from=[p_blocks_0_0_bn2_weight,p_blocks_0_0_bn2_bias,b_blocks_0_0_bn2_running_mean,b_blocks_0_0_bn2_running_var]} ->[n419] constant,
     t736 f32 [N=48 T=1 D=1 H=1 W=1 C=8] {folded from=[p_blocks_1_0_conv_pw_weight,p_blocks_1_0_bn1_weight,b_blocks_1_0_bn1_running_var]} ->[n420] constant,
     t737 f32 [C=48] {folded from=[p_blocks_1_0_bn1_weight,p_blocks_1_0_bn1_bias,b_blocks_1_0_bn1_running_mean,b_blocks_1_0_bn1_running_var]} ->[n420] constant,
     t738 f32 [N=48 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_1_0_conv_dw_weight,p_blocks_1_0_bn2_weight,b_blocks_1_0_bn2_running_var]} ->[n422] constant,
     t739 f32 [C=48] {folded from=[p_blocks_1_0_bn2_weight,p_blocks_1_0_bn2_bias,b_blocks_1_0_bn2_running_mean,b_blocks_1_0_bn2_running_var]} ->[n422] constant,
     t740 f32 [N=16 T=1 D=1 H=1 W=1 C=48] {folded from=[p_blocks_1_0_conv_pwl_weight,p_blocks_1_0_bn3_weight,b_blocks_1_0_bn3_running_var]} ->[n424] constant,
     t741 f32 [C=16] {folded from=[p_blocks_1_0_bn3_weight,p_blocks_1_0_bn3_bias,b_blocks_1_0_bn3_running_mean,b_blocks_1_0_bn3_running_var]} ->[n424] constant,
     t742 f32 [N=96 T=1 D=1 H=1 W=1 C=16] {folded from=[p_blocks_1_1_conv_pw_weight,p_blocks_1_1_bn1_weight,b_blocks_1_1_bn1_running_var]} ->[n425] constant,
     t743 f32 [C=96] {folded from=[p_blocks_1_1_bn1_weight,p_blocks_1_1_bn1_bias,b_blocks_1_1_bn1_running_mean,b_blocks_1_1_bn1_running_var]} ->[n425] constant,
     t744 f32 [N=96 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_1_1_conv_dw_weight,p_blocks_1_1_bn2_weight,b_blocks_1_1_bn2_running_var]} ->[n427] constant,
     t745 f32 [C=96] {folded from=[p_blocks_1_1_bn2_weight,p_blocks_1_1_bn2_bias,b_blocks_1_1_bn2_running_mean,b_blocks_1_1_bn2_running_var]} ->[n427] constant,
     t746 f32 [N=16 T=1 D=1 H=1 W=1 C=96] {folded from=[p_blocks_1_1_conv_pwl_weight,p_blocks_1_1_bn3_weight,b_blocks_1_1_bn3_running_var]} ->[n429] constant,
     t747 f32 [C=16] {folded from=[p_blocks_1_1_bn3_weight,p_blocks_1_1_bn3_bias,b_blocks_1_1_bn3_running_mean,b_blocks_1_1_bn3_running_var]} ->[n429] constant,
     t748 f32 [N=96 T=1 D=1 H=1 W=1 C=16] {folded from=[p_blocks_2_0_conv_pw_weight,p_blocks_2_0_bn1_weight,b_blocks_2_0_bn1_running_var]} ->[n431] constant,
     t749 f32 [C=96] {folded from=[p_blocks_2_0_bn1_weight,p_blocks_2_0_bn1_bias,b_blocks_2_0_bn1_running_mean,b_blocks_2_0_bn1_running_var]} ->[n431] constant,
     t750 f32 [N=96 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_2_0_conv_dw_weight,p_blocks_2_0_bn2_weight,b_blocks_2_0_bn2_running_var]} ->[n433] constant,
     t751 f32 [C=96] {folded from=[p_blocks_2_0_bn2_weight,p_blocks_2_0_bn2_bias,b_blocks_2_0_bn2_running_mean,b_blocks_2_0_bn2_running_var]} ->[n433] constant,
     t752 f32 [N=16 T=1 D=1 H=1 W=1 C=96] {folded from=[p_blocks_2_0_conv_pwl_weight,p_blocks_2_0_bn3_weight,b_blocks_2_0_bn3_running_var]} ->[n435] constant,
     t753 f32 [C=16] {folded from=[p_blocks_2_0_bn3_weight,p_blocks_2_0_bn3_bias,b_blocks_2_0_bn3_running_mean,b_blocks_2_0_bn3_running_var]} ->[n435] constant,
     t754 f32 [N=96 T=1 D=1 H=1 W=1 C=16] {folded from=[p_blocks_2_1_conv_pw_weight,p_blocks_2_1_bn1_weight,b_blocks_2_1_bn1_running_var]} ->[n436] constant,
     t755 f32 [C=96] {folded from=[p_blocks_2_1_bn1_weight,p_blocks_2_1_bn1_bias,b_blocks_2_1_bn1_running_mean,b_blocks_2_1_bn1_running_var]} ->[n436] constant,
     t756 f32 [N=96 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_2_1_conv_dw_weight,p_blocks_2_1_bn2_weight,b_blocks_2_1_bn2_running_var]} ->[n438] constant,
     t757 f32 [C=96] {folded from=[p_blocks_2_1_bn2_weight,p_blocks_2_1_bn2_bias,b_blocks_2_1_bn2_running_mean,b_blocks_2_1_bn2_running_var]} ->[n438] constant,
     t758 f32 [N=16 T=1 D=1 H=1 W=1 C=96] {folded from=[p_blocks_2_1_conv_pwl_weight,p_blocks_2_1_bn3_weight,b_blocks_2_1_bn3_running_var]} ->[n440] constant,
     t759 f32 [C=16] {folded from=[p_blocks_2_1_bn3_weight,p_blocks_2_1_bn3_bias,b_blocks_2_1_bn3_running_mean,b_blocks_2_1_bn3_running_var]} ->[n440] constant,
     t760 f32 [N=96 T=1 D=1 H=1 W=1 C=16] {folded from=[p_blocks_2_2_conv_pw_weight,p_blocks_2_2_bn1_weight,b_blocks_2_2_bn1_running_var]} ->[n442] constant,
     t761 f32 [C=96] {folded from=[p_blocks_2_2_bn1_weight,p_blocks_2_2_bn1_bias,b_blocks_2_2_bn1_running_mean,b_blocks_2_2_bn1_running_var]} ->[n442] constant,
     t762 f32 [N=96 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_2_2_conv_dw_weight,p_blocks_2_2_bn2_weight,b_blocks_2_2_bn2_running_var]} ->[n444] constant,
     t763 f32 [C=96] {folded from=[p_blocks_2_2_bn2_weight,p_blocks_2_2_bn2_bias,b_blocks_2_2_bn2_running_mean,b_blocks_2_2_bn2_running_var]} ->[n444] constant,
     t764 f32 [N=16 T=1 D=1 H=1 W=1 C=96] {folded from=[p_blocks_2_2_conv_pwl_weight,p_blocks_2_2_bn3_weight,b_blocks_2_2_bn3_running_var]} ->[n446] constant,
     t765 f32 [C=16] {folded from=[p_blocks_2_2_bn3_weight,p_blocks_2_2_bn3_bias,b_blocks_2_2_bn3_running_mean,b_blocks_2_2_bn3_running_var]} ->[n446] constant,
     t766 f32 [N=96 T=1 D=1 H=1 W=1 C=16] {folded from=[p_blocks_3_0_conv_pw_weight,p_blocks_3_0_bn1_weight,b_blocks_3_0_bn1_running_var]} ->[n448] constant,
     t767 f32 [C=96] {folded from=[p_blocks_3_0_bn1_weight,p_blocks_3_0_bn1_bias,b_blocks_3_0_bn1_running_mean,b_blocks_3_0_bn1_running_var]} ->[n448] constant,
     t768 f32 [N=96 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_3_0_conv_dw_weight,p_blocks_3_0_bn2_weight,b_blocks_3_0_bn2_running_var]} ->[n450] constant,
     t769 f32 [C=96] {folded from=[p_blocks_3_0_bn2_weight,p_blocks_3_0_bn2_bias,b_blocks_3_0_bn2_running_mean,b_blocks_3_0_bn2_running_var]} ->[n450] constant,
     t770 f32 [N=32 T=1 D=1 H=1 W=1 C=96] {folded from=[p_blocks_3_0_conv_pwl_weight,p_blocks_3_0_bn3_weight,b_blocks_3_0_bn3_running_var]} ->[n452] constant,
     t771 f32 [C=32] {folded from=[p_blocks_3_0_bn3_weight,p_blocks_3_0_bn3_bias,b_blocks_3_0_bn3_running_mean,b_blocks_3_0_bn3_running_var]} ->[n452] constant,
     t772 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {folded from=[p_blocks_3_1_conv_pw_weight,p_blocks_3_1_bn1_weight,b_blocks_3_1_bn1_running_var]} ->[n453] constant,
     t773 f32 [C=192] {folded from=[p_blocks_3_1_bn1_weight,p_blocks_3_1_bn1_bias,b_blocks_3_1_bn1_running_mean,b_blocks_3_1_bn1_running_var]} ->[n453] constant,
     t774 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_3_1_conv_dw_weight,p_blocks_3_1_bn2_weight,b_blocks_3_1_bn2_running_var]} ->[n455] constant,
     t775 f32 [C=192] {folded from=[p_blocks_3_1_bn2_weight,p_blocks_3_1_bn2_bias,b_blocks_3_1_bn2_running_mean,b_blocks_3_1_bn2_running_var]} ->[n455] constant,
     t776 f32 [N=32 T=1 D=1 H=1 W=1 C=192] {folded from=[p_blocks_3_1_conv_pwl_weight,p_blocks_3_1_bn3_weight,b_blocks_3_1_bn3_running_var]} ->[n457] constant,
     t777 f32 [C=32] {folded from=[p_blocks_3_1_bn3_weight,p_blocks_3_1_bn3_bias,b_blocks_3_1_bn3_running_mean,b_blocks_3_1_bn3_running_var]} ->[n457] constant,
     t778 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {folded from=[p_blocks_3_2_conv_pw_weight,p_blocks_3_2_bn1_weight,b_blocks_3_2_bn1_running_var]} ->[n459] constant,
     t779 f32 [C=192] {folded from=[p_blocks_3_2_bn1_weight,p_blocks_3_2_bn1_bias,b_blocks_3_2_bn1_running_mean,b_blocks_3_2_bn1_running_var]} ->[n459] constant,
     t780 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_3_2_conv_dw_weight,p_blocks_3_2_bn2_weight,b_blocks_3_2_bn2_running_var]} ->[n461] constant,
     t781 f32 [C=192] {folded from=[p_blocks_3_2_bn2_weight,p_blocks_3_2_bn2_bias,b_blocks_3_2_bn2_running_mean,b_blocks_3_2_bn2_running_var]} ->[n461] constant,
     t782 f32 [N=32 T=1 D=1 H=1 W=1 C=192] {folded from=[p_blocks_3_2_conv_pwl_weight,p_blocks_3_2_bn3_weight,b_blocks_3_2_bn3_running_var]} ->[n463] constant,
     t783 f32 [C=32] {folded from=[p_blocks_3_2_bn3_weight,p_blocks_3_2_bn3_bias,b_blocks_3_2_bn3_running_mean,b_blocks_3_2_bn3_running_var]} ->[n463] constant,
     t784 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {folded from=[p_blocks_3_3_conv_pw_weight,p_blocks_3_3_bn1_weight,b_blocks_3_3_bn1_running_var]} ->[n465] constant,
     t785 f32 [C=192] {folded from=[p_blocks_3_3_bn1_weight,p_blocks_3_3_bn1_bias,b_blocks_3_3_bn1_running_mean,b_blocks_3_3_bn1_running_var]} ->[n465] constant,
     t786 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_3_3_conv_dw_weight,p_blocks_3_3_bn2_weight,b_blocks_3_3_bn2_running_var]} ->[n467] constant,
     t787 f32 [C=192] {folded from=[p_blocks_3_3_bn2_weight,p_blocks_3_3_bn2_bias,b_blocks_3_3_bn2_running_mean,b_blocks_3_3_bn2_running_var]} ->[n467] constant,
     t788 f32 [N=32 T=1 D=1 H=1 W=1 C=192] {folded from=[p_blocks_3_3_conv_pwl_weight,p_blocks_3_3_bn3_weight,b_blocks_3_3_bn3_running_var]} ->[n469] constant,
     t789 f32 [C=32] {folded from=[p_blocks_3_3_bn3_weight,p_blocks_3_3_bn3_bias,b_blocks_3_3_bn3_running_mean,b_blocks_3_3_bn3_running_var]} ->[n469] constant,
     t790 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {folded from=[p_blocks_4_0_conv_pw_weight,p_blocks_4_0_bn1_weight,b_blocks_4_0_bn1_running_var]} ->[n471] constant,
     t791 f32 [C=192] {folded from=[p_blocks_4_0_bn1_weight,p_blocks_4_0_bn1_bias,b_blocks_4_0_bn1_running_mean,b_blocks_4_0_bn1_running_var]} ->[n471] constant,
     t792 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_4_0_conv_dw_weight,p_blocks_4_0_bn2_weight,b_blocks_4_0_bn2_running_var]} ->[n473] constant,
     t793 f32 [C=192] {folded from=[p_blocks_4_0_bn2_weight,p_blocks_4_0_bn2_bias,b_blocks_4_0_bn2_running_mean,b_blocks_4_0_bn2_running_var]} ->[n473] constant,
     t794 f32 [N=48 T=1 D=1 H=1 W=1 C=192] {folded from=[p_blocks_4_0_conv_pwl_weight,p_blocks_4_0_bn3_weight,b_blocks_4_0_bn3_running_var]} ->[n475] constant,
     t795 f32 [C=48] {folded from=[p_blocks_4_0_bn3_weight,p_blocks_4_0_bn3_bias,b_blocks_4_0_bn3_running_mean,b_blocks_4_0_bn3_running_var]} ->[n475] constant,
     t796 f32 [N=288 T=1 D=1 H=1 W=1 C=48] {folded from=[p_blocks_4_1_conv_pw_weight,p_blocks_4_1_bn1_weight,b_blocks_4_1_bn1_running_var]} ->[n476] constant,
     t797 f32 [C=288] {folded from=[p_blocks_4_1_bn1_weight,p_blocks_4_1_bn1_bias,b_blocks_4_1_bn1_running_mean,b_blocks_4_1_bn1_running_var]} ->[n476] constant,
     t798 f32 [N=288 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_4_1_conv_dw_weight,p_blocks_4_1_bn2_weight,b_blocks_4_1_bn2_running_var]} ->[n478] constant,
     t799 f32 [C=288] {folded from=[p_blocks_4_1_bn2_weight,p_blocks_4_1_bn2_bias,b_blocks_4_1_bn2_running_mean,b_blocks_4_1_bn2_running_var]} ->[n478] constant,
     t800 f32 [N=48 T=1 D=1 H=1 W=1 C=288] {folded from=[p_blocks_4_1_conv_pwl_weight,p_blocks_4_1_bn3_weight,b_blocks_4_1_bn3_running_var]} ->[n480] constant,
     t801 f32 [C=48] {folded from=[p_blocks_4_1_bn3_weight,p_blocks_4_1_bn3_bias,b_blocks_4_1_bn3_running_mean,b_blocks_4_1_bn3_running_var]} ->[n480] constant,
     t802 f32 [N=288 T=1 D=1 H=1 W=1 C=48] {folded from=[p_blocks_4_2_conv_pw_weight,p_blocks_4_2_bn1_weight,b_blocks_4_2_bn1_running_var]} ->[n482] constant,
     t803 f32 [C=288] {folded from=[p_blocks_4_2_bn1_weight,p_blocks_4_2_bn1_bias,b_blocks_4_2_bn1_running_mean,b_blocks_4_2_bn1_running_var]} ->[n482] constant,
     t804 f32 [N=288 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_4_2_conv_dw_weight,p_blocks_4_2_bn2_weight,b_blocks_4_2_bn2_running_var]} ->[n484] constant,
     t805 f32 [C=288] {folded from=[p_blocks_4_2_bn2_weight,p_blocks_4_2_bn2_bias,b_blocks_4_2_bn2_running_mean,b_blocks_4_2_bn2_running_var]} ->[n484] constant,
     t806 f32 [N=48 T=1 D=1 H=1 W=1 C=288] {folded from=[p_blocks_4_2_conv_pwl_weight,p_blocks_4_2_bn3_weight,b_blocks_4_2_bn3_running_var]} ->[n486] constant,
     t807 f32 [C=48] {folded from=[p_blocks_4_2_bn3_weight,p_blocks_4_2_bn3_bias,b_blocks_4_2_bn3_running_mean,b_blocks_4_2_bn3_running_var]} ->[n486] constant,
     t808 f32 [N=288 T=1 D=1 H=1 W=1 C=48] {folded from=[p_blocks_5_0_conv_pw_weight,p_blocks_5_0_bn1_weight,b_blocks_5_0_bn1_running_var]} ->[n488] constant,
     t809 f32 [C=288] {folded from=[p_blocks_5_0_bn1_weight,p_blocks_5_0_bn1_bias,b_blocks_5_0_bn1_running_mean,b_blocks_5_0_bn1_running_var]} ->[n488] constant,
     t810 f32 [N=288 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_5_0_conv_dw_weight,p_blocks_5_0_bn2_weight,b_blocks_5_0_bn2_running_var]} ->[n490] constant,
     t811 f32 [C=288] {folded from=[p_blocks_5_0_bn2_weight,p_blocks_5_0_bn2_bias,b_blocks_5_0_bn2_running_mean,b_blocks_5_0_bn2_running_var]} ->[n490] constant,
     t812 f32 [N=80 T=1 D=1 H=1 W=1 C=288] {folded from=[p_blocks_5_0_conv_pwl_weight,p_blocks_5_0_bn3_weight,b_blocks_5_0_bn3_running_var]} ->[n492] constant,
     t813 f32 [C=80] {folded from=[p_blocks_5_0_bn3_weight,p_blocks_5_0_bn3_bias,b_blocks_5_0_bn3_running_mean,b_blocks_5_0_bn3_running_var]} ->[n492] constant,
     t814 f32 [N=480 T=1 D=1 H=1 W=1 C=80] {folded from=[p_blocks_5_1_conv_pw_weight,p_blocks_5_1_bn1_weight,b_blocks_5_1_bn1_running_var]} ->[n493] constant,
     t815 f32 [C=480] {folded from=[p_blocks_5_1_bn1_weight,p_blocks_5_1_bn1_bias,b_blocks_5_1_bn1_running_mean,b_blocks_5_1_bn1_running_var]} ->[n493] constant,
     t816 f32 [N=480 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_5_1_conv_dw_weight,p_blocks_5_1_bn2_weight,b_blocks_5_1_bn2_running_var]} ->[n495] constant,
     t817 f32 [C=480] {folded from=[p_blocks_5_1_bn2_weight,p_blocks_5_1_bn2_bias,b_blocks_5_1_bn2_running_mean,b_blocks_5_1_bn2_running_var]} ->[n495] constant,
     t818 f32 [N=80 T=1 D=1 H=1 W=1 C=480] {folded from=[p_blocks_5_1_conv_pwl_weight,p_blocks_5_1_bn3_weight,b_blocks_5_1_bn3_running_var]} ->[n497] constant,
     t819 f32 [C=80] {folded from=[p_blocks_5_1_bn3_weight,p_blocks_5_1_bn3_bias,b_blocks_5_1_bn3_running_mean,b_blocks_5_1_bn3_running_var]} ->[n497] constant,
     t820 f32 [N=480 T=1 D=1 H=1 W=1 C=80] {folded from=[p_blocks_5_2_conv_pw_weight,p_blocks_5_2_bn1_weight,b_blocks_5_2_bn1_running_var]} ->[n499] constant,
     t821 f32 [C=480] {folded from=[p_blocks_5_2_bn1_weight,p_blocks_5_2_bn1_bias,b_blocks_5_2_bn1_running_mean,b_blocks_5_2_bn1_running_var]} ->[n499] constant,
     t822 f32 [N=480 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_5_2_conv_dw_weight,p_blocks_5_2_bn2_weight,b_blocks_5_2_bn2_running_var]} ->[n501] constant,
     t823 f32 [C=480] {folded from=[p_blocks_5_2_bn2_weight,p_blocks_5_2_bn2_bias,b_blocks_5_2_bn2_running_mean,b_blocks_5_2_bn2_running_var]} ->[n501] constant,
     t824 f32 [N=80 T=1 D=1 H=1 W=1 C=480] {folded from=[p_blocks_5_2_conv_pwl_weight,p_blocks_5_2_bn3_weight,b_blocks_5_2_bn3_running_var]} ->[n503] constant,
     t825 f32 [C=80] {folded from=[p_blocks_5_2_bn3_weight,p_blocks_5_2_bn3_bias,b_blocks_5_2_bn3_running_mean,b_blocks_5_2_bn3_running_var]} ->[n503] constant,
     t826 f32 [N=480 T=1 D=1 H=1 W=1 C=80] {folded from=[p_blocks_6_0_conv_pw_weight,p_blocks_6_0_bn1_weight,b_blocks_6_0_bn1_running_var]} ->[n505] constant,
     t827 f32 [C=480] {folded from=[p_blocks_6_0_bn1_weight,p_blocks_6_0_bn1_bias,b_blocks_6_0_bn1_running_mean,b_blocks_6_0_bn1_running_var]} ->[n505] constant,
     t828 f32 [N=480 T=1 D=1 H=3 W=3 C=1] {folded from=[p_blocks_6_0_conv_dw_weight,p_blocks_6_0_bn2_weight,b_blocks_6_0_bn2_running_var]} ->[n507] constant,
     t829 f32 [C=480] {folded from=[p_blocks_6_0_bn2_weight,p_blocks_6_0_bn2_bias,b_blocks_6_0_bn2_running_mean,b_blocks_6_0_bn2_running_var]} ->[n507] constant,
     t830 f32 [N=160 T=1 D=1 H=1 W=1 C=480] {folded from=[p_blocks_6_0_conv_pwl_weight,p_blocks_6_0_bn3_weight,b_blocks_6_0_bn3_running_var]} ->[n509] constant,
     t831 f32 [C=160] {folded from=[p_blocks_6_0_bn3_weight,p_blocks_6_0_bn3_bias,b_blocks_6_0_bn3_running_mean,b_blocks_6_0_bn3_running_var]} ->[n509] constant,
     t832 f32 [N=1280 T=1 D=1 H=1 W=1 C=160] {folded from=[p_conv_head_weight,p_bn2_weight,b_bn2_running_var]} ->[n510] constant,
     t833 f32 [C=1280] {folded from=[p_bn2_weight,p_bn2_bias,b_bn2_running_mean,b_bn2_running_var]} ->[n510] constant]
  nodes:
    group g1 torch.ops.aten.conv2d.default:
      n0 {derived}: [t315 f32 [H=224 W=224 C=3] {derived} ->[n415]] =
        permute x=t314 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
    n415 {derived}: [t834 f32 [H=112 W=112 C=16] {derived} ->[n416]] =
      conv2d
        x=t315 {derived} <-n0
        weight=t730 {folded from=[p_conv_stem_weight,p_bn1_weight,b_bn1_running_var]}
        bias=t731 {folded from=[p_bn1_weight,p_bn1_bias,b_bn1_running_mean,b_bn1_running_var]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=3;
               groups=1}
    n416 {pt2=root[2] torch.ops.aten.hardtanh.default}: [t835 f32 [H=112 W=112
                                                                   C=16] {derived} ->[n417]] =
      hardtanh x=t834 {derived} <-n415 params={min_val=0; max_val=6}
    n417 {derived}: [t836 f32 [H=112 W=112 C=16] {derived} ->[n418]] =
      conv2d
        x=t835 {derived} <-n416
        weight=t732 {folded from=[p_blocks_0_0_conv_dw_weight,p_blocks_0_0_bn1_weight,b_blocks_0_0_bn1_running_var]}
        bias=t733 {folded from=[p_blocks_0_0_bn1_weight,p_blocks_0_0_bn1_bias,b_blocks_0_0_bn1_running_mean,b_blocks_0_0_bn1_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=16;
               groups=16}
    n418 {pt2=root[5] torch.ops.aten.hardtanh.default}: [t837 f32 [H=112 W=112
                                                                   C=16] {derived} ->[n419]] =
      hardtanh x=t836 {derived} <-n417 params={min_val=0; max_val=6}
    n419 {derived}: [t838 f32 [H=112 W=112 C=8] {derived} ->[n420]] =
      conv2d
        x=t837 {derived} <-n418
        weight=t734 {folded from=[p_blocks_0_0_conv_pw_weight,p_blocks_0_0_bn2_weight,b_blocks_0_0_bn2_running_var]}
        bias=t735 {folded from=[p_blocks_0_0_bn2_weight,p_blocks_0_0_bn2_bias,b_blocks_0_0_bn2_running_mean,b_blocks_0_0_bn2_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=16;
               groups=1}
    n420 {derived}: [t839 f32 [H=112 W=112 C=48] {derived} ->[n421]] =
      conv2d
        x=t838 {derived} <-n419
        weight=t736 {folded from=[p_blocks_1_0_conv_pw_weight,p_blocks_1_0_bn1_weight,b_blocks_1_0_bn1_running_var]}
        bias=t737 {folded from=[p_blocks_1_0_bn1_weight,p_blocks_1_0_bn1_bias,b_blocks_1_0_bn1_running_mean,b_blocks_1_0_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=8;
               groups=1}
    n421 {pt2=root[10] torch.ops.aten.hardtanh.default}: [t840 f32 [H=112 W=112
                                                                    C=48] {derived} ->[n422]] =
      hardtanh x=t839 {derived} <-n420 params={min_val=0; max_val=6}
    n422 {derived}: [t841 f32 [H=56 W=56 C=48] {derived} ->[n423]] =
      conv2d
        x=t840 {derived} <-n421
        weight=t738 {folded from=[p_blocks_1_0_conv_dw_weight,p_blocks_1_0_bn2_weight,b_blocks_1_0_bn2_running_var]}
        bias=t739 {folded from=[p_blocks_1_0_bn2_weight,p_blocks_1_0_bn2_bias,b_blocks_1_0_bn2_running_mean,b_blocks_1_0_bn2_running_var]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=48;
               groups=48}
    n423 {pt2=root[13] torch.ops.aten.hardtanh.default}: [t842 f32 [H=56 W=56
                                                                    C=48] {derived} ->[n424]] =
      hardtanh x=t841 {derived} <-n422 params={min_val=0; max_val=6}
    n424 {derived}: [t843 f32 [H=56 W=56 C=16] {derived} ->[n425, n430]] =
      conv2d
        x=t842 {derived} <-n423
        weight=t740 {folded from=[p_blocks_1_0_conv_pwl_weight,p_blocks_1_0_bn3_weight,b_blocks_1_0_bn3_running_var]}
        bias=t741 {folded from=[p_blocks_1_0_bn3_weight,p_blocks_1_0_bn3_bias,b_blocks_1_0_bn3_running_mean,b_blocks_1_0_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=48;
               groups=1}
    n425 {derived}: [t844 f32 [H=56 W=56 C=96] {derived} ->[n426]] =
      conv2d
        x=t843 {derived} <-n424
        weight=t742 {folded from=[p_blocks_1_1_conv_pw_weight,p_blocks_1_1_bn1_weight,b_blocks_1_1_bn1_running_var]}
        bias=t743 {folded from=[p_blocks_1_1_bn1_weight,p_blocks_1_1_bn1_bias,b_blocks_1_1_bn1_running_mean,b_blocks_1_1_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=16;
               groups=1}
    n426 {pt2=root[18] torch.ops.aten.hardtanh.default}: [t845 f32 [H=56 W=56
                                                                    C=96] {derived} ->[n427]] =
      hardtanh x=t844 {derived} <-n425 params={min_val=0; max_val=6}
    n427 {derived}: [t846 f32 [H=56 W=56 C=96] {derived} ->[n428]] =
      conv2d
        x=t845 {derived} <-n426
        weight=t744 {folded from=[p_blocks_1_1_conv_dw_weight,p_blocks_1_1_bn2_weight,b_blocks_1_1_bn2_running_var]}
        bias=t745 {folded from=[p_blocks_1_1_bn2_weight,p_blocks_1_1_bn2_bias,b_blocks_1_1_bn2_running_mean,b_blocks_1_1_bn2_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=96;
               groups=96}
    n428 {pt2=root[21] torch.ops.aten.hardtanh.default}: [t847 f32 [H=56 W=56
                                                                    C=96] {derived} ->[n429]] =
      hardtanh x=t846 {derived} <-n427 params={min_val=0; max_val=6}
    n429 {derived}: [t848 f32 [H=56 W=56 C=16] {derived} ->[n430]] =
      conv2d
        x=t847 {derived} <-n428
        weight=t746 {folded from=[p_blocks_1_1_conv_pwl_weight,p_blocks_1_1_bn3_weight,b_blocks_1_1_bn3_running_var]}
        bias=t747 {folded from=[p_blocks_1_1_bn3_weight,p_blocks_1_1_bn3_bias,b_blocks_1_1_bn3_running_mean,b_blocks_1_1_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=96;
               groups=1}
    n430 {pt2=root[24] torch.ops.aten.add.Tensor}: [t849 f32 [H=56 W=56 C=16] {derived} ->[n431]] =
      add a=t848 {derived} <-n429 b=t843 {derived} <-n424
    n431 {derived}: [t850 f32 [H=56 W=56 C=96] {derived} ->[n432]] =
      conv2d
        x=t849 {derived} <-n430
        weight=t748 {folded from=[p_blocks_2_0_conv_pw_weight,p_blocks_2_0_bn1_weight,b_blocks_2_0_bn1_running_var]}
        bias=t749 {folded from=[p_blocks_2_0_bn1_weight,p_blocks_2_0_bn1_bias,b_blocks_2_0_bn1_running_mean,b_blocks_2_0_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=16;
               groups=1}
    n432 {pt2=root[27] torch.ops.aten.hardtanh.default}: [t851 f32 [H=56 W=56
                                                                    C=96] {derived} ->[n433]] =
      hardtanh x=t850 {derived} <-n431 params={min_val=0; max_val=6}
    n433 {derived}: [t852 f32 [H=28 W=28 C=96] {derived} ->[n434]] =
      conv2d
        x=t851 {derived} <-n432
        weight=t750 {folded from=[p_blocks_2_0_conv_dw_weight,p_blocks_2_0_bn2_weight,b_blocks_2_0_bn2_running_var]}
        bias=t751 {folded from=[p_blocks_2_0_bn2_weight,p_blocks_2_0_bn2_bias,b_blocks_2_0_bn2_running_mean,b_blocks_2_0_bn2_running_var]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=96;
               groups=96}
    n434 {pt2=root[30] torch.ops.aten.hardtanh.default}: [t853 f32 [H=28 W=28
                                                                    C=96] {derived} ->[n435]] =
      hardtanh x=t852 {derived} <-n433 params={min_val=0; max_val=6}
    n435 {derived}: [t854 f32 [H=28 W=28 C=16] {derived} ->[n436, n441]] =
      conv2d
        x=t853 {derived} <-n434
        weight=t752 {folded from=[p_blocks_2_0_conv_pwl_weight,p_blocks_2_0_bn3_weight,b_blocks_2_0_bn3_running_var]}
        bias=t753 {folded from=[p_blocks_2_0_bn3_weight,p_blocks_2_0_bn3_bias,b_blocks_2_0_bn3_running_mean,b_blocks_2_0_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=96;
               groups=1}
    n436 {derived}: [t855 f32 [H=28 W=28 C=96] {derived} ->[n437]] =
      conv2d
        x=t854 {derived} <-n435
        weight=t754 {folded from=[p_blocks_2_1_conv_pw_weight,p_blocks_2_1_bn1_weight,b_blocks_2_1_bn1_running_var]}
        bias=t755 {folded from=[p_blocks_2_1_bn1_weight,p_blocks_2_1_bn1_bias,b_blocks_2_1_bn1_running_mean,b_blocks_2_1_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=16;
               groups=1}
    n437 {pt2=root[35] torch.ops.aten.hardtanh.default}: [t856 f32 [H=28 W=28
                                                                    C=96] {derived} ->[n438]] =
      hardtanh x=t855 {derived} <-n436 params={min_val=0; max_val=6}
    n438 {derived}: [t857 f32 [H=28 W=28 C=96] {derived} ->[n439]] =
      conv2d
        x=t856 {derived} <-n437
        weight=t756 {folded from=[p_blocks_2_1_conv_dw_weight,p_blocks_2_1_bn2_weight,b_blocks_2_1_bn2_running_var]}
        bias=t757 {folded from=[p_blocks_2_1_bn2_weight,p_blocks_2_1_bn2_bias,b_blocks_2_1_bn2_running_mean,b_blocks_2_1_bn2_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=96;
               groups=96}
    n439 {pt2=root[38] torch.ops.aten.hardtanh.default}: [t858 f32 [H=28 W=28
                                                                    C=96] {derived} ->[n440]] =
      hardtanh x=t857 {derived} <-n438 params={min_val=0; max_val=6}
    n440 {derived}: [t859 f32 [H=28 W=28 C=16] {derived} ->[n441]] =
      conv2d
        x=t858 {derived} <-n439
        weight=t758 {folded from=[p_blocks_2_1_conv_pwl_weight,p_blocks_2_1_bn3_weight,b_blocks_2_1_bn3_running_var]}
        bias=t759 {folded from=[p_blocks_2_1_bn3_weight,p_blocks_2_1_bn3_bias,b_blocks_2_1_bn3_running_mean,b_blocks_2_1_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=96;
               groups=1}
    n441 {pt2=root[41] torch.ops.aten.add.Tensor}: [t860 f32 [H=28 W=28 C=16] {derived} ->[n442,
                                                                      n447]] =
      add a=t859 {derived} <-n440 b=t854 {derived} <-n435
    n442 {derived}: [t861 f32 [H=28 W=28 C=96] {derived} ->[n443]] =
      conv2d
        x=t860 {derived} <-n441
        weight=t760 {folded from=[p_blocks_2_2_conv_pw_weight,p_blocks_2_2_bn1_weight,b_blocks_2_2_bn1_running_var]}
        bias=t761 {folded from=[p_blocks_2_2_bn1_weight,p_blocks_2_2_bn1_bias,b_blocks_2_2_bn1_running_mean,b_blocks_2_2_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=16;
               groups=1}
    n443 {pt2=root[44] torch.ops.aten.hardtanh.default}: [t862 f32 [H=28 W=28
                                                                    C=96] {derived} ->[n444]] =
      hardtanh x=t861 {derived} <-n442 params={min_val=0; max_val=6}
    n444 {derived}: [t863 f32 [H=28 W=28 C=96] {derived} ->[n445]] =
      conv2d
        x=t862 {derived} <-n443
        weight=t762 {folded from=[p_blocks_2_2_conv_dw_weight,p_blocks_2_2_bn2_weight,b_blocks_2_2_bn2_running_var]}
        bias=t763 {folded from=[p_blocks_2_2_bn2_weight,p_blocks_2_2_bn2_bias,b_blocks_2_2_bn2_running_mean,b_blocks_2_2_bn2_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=96;
               groups=96}
    n445 {pt2=root[47] torch.ops.aten.hardtanh.default}: [t864 f32 [H=28 W=28
                                                                    C=96] {derived} ->[n446]] =
      hardtanh x=t863 {derived} <-n444 params={min_val=0; max_val=6}
    n446 {derived}: [t865 f32 [H=28 W=28 C=16] {derived} ->[n447]] =
      conv2d
        x=t864 {derived} <-n445
        weight=t764 {folded from=[p_blocks_2_2_conv_pwl_weight,p_blocks_2_2_bn3_weight,b_blocks_2_2_bn3_running_var]}
        bias=t765 {folded from=[p_blocks_2_2_bn3_weight,p_blocks_2_2_bn3_bias,b_blocks_2_2_bn3_running_mean,b_blocks_2_2_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=96;
               groups=1}
    n447 {pt2=root[50] torch.ops.aten.add.Tensor}: [t866 f32 [H=28 W=28 C=16] {derived} ->[n448]] =
      add a=t865 {derived} <-n446 b=t860 {derived} <-n441
    n448 {derived}: [t867 f32 [H=28 W=28 C=96] {derived} ->[n449]] =
      conv2d
        x=t866 {derived} <-n447
        weight=t766 {folded from=[p_blocks_3_0_conv_pw_weight,p_blocks_3_0_bn1_weight,b_blocks_3_0_bn1_running_var]}
        bias=t767 {folded from=[p_blocks_3_0_bn1_weight,p_blocks_3_0_bn1_bias,b_blocks_3_0_bn1_running_mean,b_blocks_3_0_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=16;
               groups=1}
    n449 {pt2=root[53] torch.ops.aten.hardtanh.default}: [t868 f32 [H=28 W=28
                                                                    C=96] {derived} ->[n450]] =
      hardtanh x=t867 {derived} <-n448 params={min_val=0; max_val=6}
    n450 {derived}: [t869 f32 [H=14 W=14 C=96] {derived} ->[n451]] =
      conv2d
        x=t868 {derived} <-n449
        weight=t768 {folded from=[p_blocks_3_0_conv_dw_weight,p_blocks_3_0_bn2_weight,b_blocks_3_0_bn2_running_var]}
        bias=t769 {folded from=[p_blocks_3_0_bn2_weight,p_blocks_3_0_bn2_bias,b_blocks_3_0_bn2_running_mean,b_blocks_3_0_bn2_running_var]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=96;
               groups=96}
    n451 {pt2=root[56] torch.ops.aten.hardtanh.default}: [t870 f32 [H=14 W=14
                                                                    C=96] {derived} ->[n452]] =
      hardtanh x=t869 {derived} <-n450 params={min_val=0; max_val=6}
    n452 {derived}: [t871 f32 [H=14 W=14 C=32] {derived} ->[n453, n458]] =
      conv2d
        x=t870 {derived} <-n451
        weight=t770 {folded from=[p_blocks_3_0_conv_pwl_weight,p_blocks_3_0_bn3_weight,b_blocks_3_0_bn3_running_var]}
        bias=t771 {folded from=[p_blocks_3_0_bn3_weight,p_blocks_3_0_bn3_bias,b_blocks_3_0_bn3_running_mean,b_blocks_3_0_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=96;
               groups=1}
    n453 {derived}: [t872 f32 [H=14 W=14 C=192] {derived} ->[n454]] =
      conv2d
        x=t871 {derived} <-n452
        weight=t772 {folded from=[p_blocks_3_1_conv_pw_weight,p_blocks_3_1_bn1_weight,b_blocks_3_1_bn1_running_var]}
        bias=t773 {folded from=[p_blocks_3_1_bn1_weight,p_blocks_3_1_bn1_bias,b_blocks_3_1_bn1_running_mean,b_blocks_3_1_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=32;
               groups=1}
    n454 {pt2=root[61] torch.ops.aten.hardtanh.default}: [t873 f32 [H=14 W=14
                                                                    C=192] {derived} ->[n455]] =
      hardtanh x=t872 {derived} <-n453 params={min_val=0; max_val=6}
    n455 {derived}: [t874 f32 [H=14 W=14 C=192] {derived} ->[n456]] =
      conv2d
        x=t873 {derived} <-n454
        weight=t774 {folded from=[p_blocks_3_1_conv_dw_weight,p_blocks_3_1_bn2_weight,b_blocks_3_1_bn2_running_var]}
        bias=t775 {folded from=[p_blocks_3_1_bn2_weight,p_blocks_3_1_bn2_bias,b_blocks_3_1_bn2_running_mean,b_blocks_3_1_bn2_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=192;
               groups=192}
    n456 {pt2=root[64] torch.ops.aten.hardtanh.default}: [t875 f32 [H=14 W=14
                                                                    C=192] {derived} ->[n457]] =
      hardtanh x=t874 {derived} <-n455 params={min_val=0; max_val=6}
    n457 {derived}: [t876 f32 [H=14 W=14 C=32] {derived} ->[n458]] =
      conv2d
        x=t875 {derived} <-n456
        weight=t776 {folded from=[p_blocks_3_1_conv_pwl_weight,p_blocks_3_1_bn3_weight,b_blocks_3_1_bn3_running_var]}
        bias=t777 {folded from=[p_blocks_3_1_bn3_weight,p_blocks_3_1_bn3_bias,b_blocks_3_1_bn3_running_mean,b_blocks_3_1_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=192;
               groups=1}
    n458 {pt2=root[67] torch.ops.aten.add.Tensor}: [t877 f32 [H=14 W=14 C=32] {derived} ->[n459,
                                                                      n464]] =
      add a=t876 {derived} <-n457 b=t871 {derived} <-n452
    n459 {derived}: [t878 f32 [H=14 W=14 C=192] {derived} ->[n460]] =
      conv2d
        x=t877 {derived} <-n458
        weight=t778 {folded from=[p_blocks_3_2_conv_pw_weight,p_blocks_3_2_bn1_weight,b_blocks_3_2_bn1_running_var]}
        bias=t779 {folded from=[p_blocks_3_2_bn1_weight,p_blocks_3_2_bn1_bias,b_blocks_3_2_bn1_running_mean,b_blocks_3_2_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=32;
               groups=1}
    n460 {pt2=root[70] torch.ops.aten.hardtanh.default}: [t879 f32 [H=14 W=14
                                                                    C=192] {derived} ->[n461]] =
      hardtanh x=t878 {derived} <-n459 params={min_val=0; max_val=6}
    n461 {derived}: [t880 f32 [H=14 W=14 C=192] {derived} ->[n462]] =
      conv2d
        x=t879 {derived} <-n460
        weight=t780 {folded from=[p_blocks_3_2_conv_dw_weight,p_blocks_3_2_bn2_weight,b_blocks_3_2_bn2_running_var]}
        bias=t781 {folded from=[p_blocks_3_2_bn2_weight,p_blocks_3_2_bn2_bias,b_blocks_3_2_bn2_running_mean,b_blocks_3_2_bn2_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=192;
               groups=192}
    n462 {pt2=root[73] torch.ops.aten.hardtanh.default}: [t881 f32 [H=14 W=14
                                                                    C=192] {derived} ->[n463]] =
      hardtanh x=t880 {derived} <-n461 params={min_val=0; max_val=6}
    n463 {derived}: [t882 f32 [H=14 W=14 C=32] {derived} ->[n464]] =
      conv2d
        x=t881 {derived} <-n462
        weight=t782 {folded from=[p_blocks_3_2_conv_pwl_weight,p_blocks_3_2_bn3_weight,b_blocks_3_2_bn3_running_var]}
        bias=t783 {folded from=[p_blocks_3_2_bn3_weight,p_blocks_3_2_bn3_bias,b_blocks_3_2_bn3_running_mean,b_blocks_3_2_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=192;
               groups=1}
    n464 {pt2=root[76] torch.ops.aten.add.Tensor}: [t883 f32 [H=14 W=14 C=32] {derived} ->[n465,
                                                                      n470]] =
      add a=t882 {derived} <-n463 b=t877 {derived} <-n458
    n465 {derived}: [t884 f32 [H=14 W=14 C=192] {derived} ->[n466]] =
      conv2d
        x=t883 {derived} <-n464
        weight=t784 {folded from=[p_blocks_3_3_conv_pw_weight,p_blocks_3_3_bn1_weight,b_blocks_3_3_bn1_running_var]}
        bias=t785 {folded from=[p_blocks_3_3_bn1_weight,p_blocks_3_3_bn1_bias,b_blocks_3_3_bn1_running_mean,b_blocks_3_3_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=32;
               groups=1}
    n466 {pt2=root[79] torch.ops.aten.hardtanh.default}: [t885 f32 [H=14 W=14
                                                                    C=192] {derived} ->[n467]] =
      hardtanh x=t884 {derived} <-n465 params={min_val=0; max_val=6}
    n467 {derived}: [t886 f32 [H=14 W=14 C=192] {derived} ->[n468]] =
      conv2d
        x=t885 {derived} <-n466
        weight=t786 {folded from=[p_blocks_3_3_conv_dw_weight,p_blocks_3_3_bn2_weight,b_blocks_3_3_bn2_running_var]}
        bias=t787 {folded from=[p_blocks_3_3_bn2_weight,p_blocks_3_3_bn2_bias,b_blocks_3_3_bn2_running_mean,b_blocks_3_3_bn2_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=192;
               groups=192}
    n468 {pt2=root[82] torch.ops.aten.hardtanh.default}: [t887 f32 [H=14 W=14
                                                                    C=192] {derived} ->[n469]] =
      hardtanh x=t886 {derived} <-n467 params={min_val=0; max_val=6}
    n469 {derived}: [t888 f32 [H=14 W=14 C=32] {derived} ->[n470]] =
      conv2d
        x=t887 {derived} <-n468
        weight=t788 {folded from=[p_blocks_3_3_conv_pwl_weight,p_blocks_3_3_bn3_weight,b_blocks_3_3_bn3_running_var]}
        bias=t789 {folded from=[p_blocks_3_3_bn3_weight,p_blocks_3_3_bn3_bias,b_blocks_3_3_bn3_running_mean,b_blocks_3_3_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=192;
               groups=1}
    n470 {pt2=root[85] torch.ops.aten.add.Tensor}: [t889 f32 [H=14 W=14 C=32] {derived} ->[n471]] =
      add a=t888 {derived} <-n469 b=t883 {derived} <-n464
    n471 {derived}: [t890 f32 [H=14 W=14 C=192] {derived} ->[n472]] =
      conv2d
        x=t889 {derived} <-n470
        weight=t790 {folded from=[p_blocks_4_0_conv_pw_weight,p_blocks_4_0_bn1_weight,b_blocks_4_0_bn1_running_var]}
        bias=t791 {folded from=[p_blocks_4_0_bn1_weight,p_blocks_4_0_bn1_bias,b_blocks_4_0_bn1_running_mean,b_blocks_4_0_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=32;
               groups=1}
    n472 {pt2=root[88] torch.ops.aten.hardtanh.default}: [t891 f32 [H=14 W=14
                                                                    C=192] {derived} ->[n473]] =
      hardtanh x=t890 {derived} <-n471 params={min_val=0; max_val=6}
    n473 {derived}: [t892 f32 [H=14 W=14 C=192] {derived} ->[n474]] =
      conv2d
        x=t891 {derived} <-n472
        weight=t792 {folded from=[p_blocks_4_0_conv_dw_weight,p_blocks_4_0_bn2_weight,b_blocks_4_0_bn2_running_var]}
        bias=t793 {folded from=[p_blocks_4_0_bn2_weight,p_blocks_4_0_bn2_bias,b_blocks_4_0_bn2_running_mean,b_blocks_4_0_bn2_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=192;
               groups=192}
    n474 {pt2=root[91] torch.ops.aten.hardtanh.default}: [t893 f32 [H=14 W=14
                                                                    C=192] {derived} ->[n475]] =
      hardtanh x=t892 {derived} <-n473 params={min_val=0; max_val=6}
    n475 {derived}: [t894 f32 [H=14 W=14 C=48] {derived} ->[n476, n481]] =
      conv2d
        x=t893 {derived} <-n474
        weight=t794 {folded from=[p_blocks_4_0_conv_pwl_weight,p_blocks_4_0_bn3_weight,b_blocks_4_0_bn3_running_var]}
        bias=t795 {folded from=[p_blocks_4_0_bn3_weight,p_blocks_4_0_bn3_bias,b_blocks_4_0_bn3_running_mean,b_blocks_4_0_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=192;
               groups=1}
    n476 {derived}: [t895 f32 [H=14 W=14 C=288] {derived} ->[n477]] =
      conv2d
        x=t894 {derived} <-n475
        weight=t796 {folded from=[p_blocks_4_1_conv_pw_weight,p_blocks_4_1_bn1_weight,b_blocks_4_1_bn1_running_var]}
        bias=t797 {folded from=[p_blocks_4_1_bn1_weight,p_blocks_4_1_bn1_bias,b_blocks_4_1_bn1_running_mean,b_blocks_4_1_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=48;
               groups=1}
    n477 {pt2=root[96] torch.ops.aten.hardtanh.default}: [t896 f32 [H=14 W=14
                                                                    C=288] {derived} ->[n478]] =
      hardtanh x=t895 {derived} <-n476 params={min_val=0; max_val=6}
    n478 {derived}: [t897 f32 [H=14 W=14 C=288] {derived} ->[n479]] =
      conv2d
        x=t896 {derived} <-n477
        weight=t798 {folded from=[p_blocks_4_1_conv_dw_weight,p_blocks_4_1_bn2_weight,b_blocks_4_1_bn2_running_var]}
        bias=t799 {folded from=[p_blocks_4_1_bn2_weight,p_blocks_4_1_bn2_bias,b_blocks_4_1_bn2_running_mean,b_blocks_4_1_bn2_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=288;
               groups=288}
    n479 {pt2=root[99] torch.ops.aten.hardtanh.default}: [t898 f32 [H=14 W=14
                                                                    C=288] {derived} ->[n480]] =
      hardtanh x=t897 {derived} <-n478 params={min_val=0; max_val=6}
    n480 {derived}: [t899 f32 [H=14 W=14 C=48] {derived} ->[n481]] =
      conv2d
        x=t898 {derived} <-n479
        weight=t800 {folded from=[p_blocks_4_1_conv_pwl_weight,p_blocks_4_1_bn3_weight,b_blocks_4_1_bn3_running_var]}
        bias=t801 {folded from=[p_blocks_4_1_bn3_weight,p_blocks_4_1_bn3_bias,b_blocks_4_1_bn3_running_mean,b_blocks_4_1_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=288;
               groups=1}
    n481 {pt2=root[102] torch.ops.aten.add.Tensor}: [t900 f32 [H=14 W=14 C=48] {derived} ->[n482,
                                                                      n487]] =
      add a=t899 {derived} <-n480 b=t894 {derived} <-n475
    n482 {derived}: [t901 f32 [H=14 W=14 C=288] {derived} ->[n483]] =
      conv2d
        x=t900 {derived} <-n481
        weight=t802 {folded from=[p_blocks_4_2_conv_pw_weight,p_blocks_4_2_bn1_weight,b_blocks_4_2_bn1_running_var]}
        bias=t803 {folded from=[p_blocks_4_2_bn1_weight,p_blocks_4_2_bn1_bias,b_blocks_4_2_bn1_running_mean,b_blocks_4_2_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=48;
               groups=1}
    n483 {pt2=root[105] torch.ops.aten.hardtanh.default}: [t902 f32 [H=14 W=14
                                                                     C=288] {derived} ->[n484]] =
      hardtanh x=t901 {derived} <-n482 params={min_val=0; max_val=6}
    n484 {derived}: [t903 f32 [H=14 W=14 C=288] {derived} ->[n485]] =
      conv2d
        x=t902 {derived} <-n483
        weight=t804 {folded from=[p_blocks_4_2_conv_dw_weight,p_blocks_4_2_bn2_weight,b_blocks_4_2_bn2_running_var]}
        bias=t805 {folded from=[p_blocks_4_2_bn2_weight,p_blocks_4_2_bn2_bias,b_blocks_4_2_bn2_running_mean,b_blocks_4_2_bn2_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=288;
               groups=288}
    n485 {pt2=root[108] torch.ops.aten.hardtanh.default}: [t904 f32 [H=14 W=14
                                                                     C=288] {derived} ->[n486]] =
      hardtanh x=t903 {derived} <-n484 params={min_val=0; max_val=6}
    n486 {derived}: [t905 f32 [H=14 W=14 C=48] {derived} ->[n487]] =
      conv2d
        x=t904 {derived} <-n485
        weight=t806 {folded from=[p_blocks_4_2_conv_pwl_weight,p_blocks_4_2_bn3_weight,b_blocks_4_2_bn3_running_var]}
        bias=t807 {folded from=[p_blocks_4_2_bn3_weight,p_blocks_4_2_bn3_bias,b_blocks_4_2_bn3_running_mean,b_blocks_4_2_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=288;
               groups=1}
    n487 {pt2=root[111] torch.ops.aten.add.Tensor}: [t906 f32 [H=14 W=14 C=48] {derived} ->[n488]] =
      add a=t905 {derived} <-n486 b=t900 {derived} <-n481
    n488 {derived}: [t907 f32 [H=14 W=14 C=288] {derived} ->[n489]] =
      conv2d
        x=t906 {derived} <-n487
        weight=t808 {folded from=[p_blocks_5_0_conv_pw_weight,p_blocks_5_0_bn1_weight,b_blocks_5_0_bn1_running_var]}
        bias=t809 {folded from=[p_blocks_5_0_bn1_weight,p_blocks_5_0_bn1_bias,b_blocks_5_0_bn1_running_mean,b_blocks_5_0_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=48;
               groups=1}
    n489 {pt2=root[114] torch.ops.aten.hardtanh.default}: [t908 f32 [H=14 W=14
                                                                     C=288] {derived} ->[n490]] =
      hardtanh x=t907 {derived} <-n488 params={min_val=0; max_val=6}
    n490 {derived}: [t909 f32 [H=7 W=7 C=288] {derived} ->[n491]] =
      conv2d
        x=t908 {derived} <-n489
        weight=t810 {folded from=[p_blocks_5_0_conv_dw_weight,p_blocks_5_0_bn2_weight,b_blocks_5_0_bn2_running_var]}
        bias=t811 {folded from=[p_blocks_5_0_bn2_weight,p_blocks_5_0_bn2_bias,b_blocks_5_0_bn2_running_mean,b_blocks_5_0_bn2_running_var]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=288;
               groups=288}
    n491 {pt2=root[117] torch.ops.aten.hardtanh.default}: [t910 f32 [H=7 W=7
                                                                     C=288] {derived} ->[n492]] =
      hardtanh x=t909 {derived} <-n490 params={min_val=0; max_val=6}
    n492 {derived}: [t911 f32 [H=7 W=7 C=80] {derived} ->[n493, n498]] =
      conv2d
        x=t910 {derived} <-n491
        weight=t812 {folded from=[p_blocks_5_0_conv_pwl_weight,p_blocks_5_0_bn3_weight,b_blocks_5_0_bn3_running_var]}
        bias=t813 {folded from=[p_blocks_5_0_bn3_weight,p_blocks_5_0_bn3_bias,b_blocks_5_0_bn3_running_mean,b_blocks_5_0_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=288;
               groups=1}
    n493 {derived}: [t912 f32 [H=7 W=7 C=480] {derived} ->[n494]] =
      conv2d
        x=t911 {derived} <-n492
        weight=t814 {folded from=[p_blocks_5_1_conv_pw_weight,p_blocks_5_1_bn1_weight,b_blocks_5_1_bn1_running_var]}
        bias=t815 {folded from=[p_blocks_5_1_bn1_weight,p_blocks_5_1_bn1_bias,b_blocks_5_1_bn1_running_mean,b_blocks_5_1_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=80;
               groups=1}
    n494 {pt2=root[122] torch.ops.aten.hardtanh.default}: [t913 f32 [H=7 W=7
                                                                     C=480] {derived} ->[n495]] =
      hardtanh x=t912 {derived} <-n493 params={min_val=0; max_val=6}
    n495 {derived}: [t914 f32 [H=7 W=7 C=480] {derived} ->[n496]] =
      conv2d
        x=t913 {derived} <-n494
        weight=t816 {folded from=[p_blocks_5_1_conv_dw_weight,p_blocks_5_1_bn2_weight,b_blocks_5_1_bn2_running_var]}
        bias=t817 {folded from=[p_blocks_5_1_bn2_weight,p_blocks_5_1_bn2_bias,b_blocks_5_1_bn2_running_mean,b_blocks_5_1_bn2_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=480;
               groups=480}
    n496 {pt2=root[125] torch.ops.aten.hardtanh.default}: [t915 f32 [H=7 W=7
                                                                     C=480] {derived} ->[n497]] =
      hardtanh x=t914 {derived} <-n495 params={min_val=0; max_val=6}
    n497 {derived}: [t916 f32 [H=7 W=7 C=80] {derived} ->[n498]] =
      conv2d
        x=t915 {derived} <-n496
        weight=t818 {folded from=[p_blocks_5_1_conv_pwl_weight,p_blocks_5_1_bn3_weight,b_blocks_5_1_bn3_running_var]}
        bias=t819 {folded from=[p_blocks_5_1_bn3_weight,p_blocks_5_1_bn3_bias,b_blocks_5_1_bn3_running_mean,b_blocks_5_1_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=480;
               groups=1}
    n498 {pt2=root[128] torch.ops.aten.add.Tensor}: [t917 f32 [H=7 W=7 C=80] {derived} ->[n499,
                                                                      n504]] =
      add a=t916 {derived} <-n497 b=t911 {derived} <-n492
    n499 {derived}: [t918 f32 [H=7 W=7 C=480] {derived} ->[n500]] =
      conv2d
        x=t917 {derived} <-n498
        weight=t820 {folded from=[p_blocks_5_2_conv_pw_weight,p_blocks_5_2_bn1_weight,b_blocks_5_2_bn1_running_var]}
        bias=t821 {folded from=[p_blocks_5_2_bn1_weight,p_blocks_5_2_bn1_bias,b_blocks_5_2_bn1_running_mean,b_blocks_5_2_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=80;
               groups=1}
    n500 {pt2=root[131] torch.ops.aten.hardtanh.default}: [t919 f32 [H=7 W=7
                                                                     C=480] {derived} ->[n501]] =
      hardtanh x=t918 {derived} <-n499 params={min_val=0; max_val=6}
    n501 {derived}: [t920 f32 [H=7 W=7 C=480] {derived} ->[n502]] =
      conv2d
        x=t919 {derived} <-n500
        weight=t822 {folded from=[p_blocks_5_2_conv_dw_weight,p_blocks_5_2_bn2_weight,b_blocks_5_2_bn2_running_var]}
        bias=t823 {folded from=[p_blocks_5_2_bn2_weight,p_blocks_5_2_bn2_bias,b_blocks_5_2_bn2_running_mean,b_blocks_5_2_bn2_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=480;
               groups=480}
    n502 {pt2=root[134] torch.ops.aten.hardtanh.default}: [t921 f32 [H=7 W=7
                                                                     C=480] {derived} ->[n503]] =
      hardtanh x=t920 {derived} <-n501 params={min_val=0; max_val=6}
    n503 {derived}: [t922 f32 [H=7 W=7 C=80] {derived} ->[n504]] =
      conv2d
        x=t921 {derived} <-n502
        weight=t824 {folded from=[p_blocks_5_2_conv_pwl_weight,p_blocks_5_2_bn3_weight,b_blocks_5_2_bn3_running_var]}
        bias=t825 {folded from=[p_blocks_5_2_bn3_weight,p_blocks_5_2_bn3_bias,b_blocks_5_2_bn3_running_mean,b_blocks_5_2_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=480;
               groups=1}
    n504 {pt2=root[137] torch.ops.aten.add.Tensor}: [t923 f32 [H=7 W=7 C=80] {derived} ->[n505]] =
      add a=t922 {derived} <-n503 b=t917 {derived} <-n498
    n505 {derived}: [t924 f32 [H=7 W=7 C=480] {derived} ->[n506]] =
      conv2d
        x=t923 {derived} <-n504
        weight=t826 {folded from=[p_blocks_6_0_conv_pw_weight,p_blocks_6_0_bn1_weight,b_blocks_6_0_bn1_running_var]}
        bias=t827 {folded from=[p_blocks_6_0_bn1_weight,p_blocks_6_0_bn1_bias,b_blocks_6_0_bn1_running_mean,b_blocks_6_0_bn1_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=80;
               groups=1}
    n506 {pt2=root[140] torch.ops.aten.hardtanh.default}: [t925 f32 [H=7 W=7
                                                                     C=480] {derived} ->[n507]] =
      hardtanh x=t924 {derived} <-n505 params={min_val=0; max_val=6}
    n507 {derived}: [t926 f32 [H=7 W=7 C=480] {derived} ->[n508]] =
      conv2d
        x=t925 {derived} <-n506
        weight=t828 {folded from=[p_blocks_6_0_conv_dw_weight,p_blocks_6_0_bn2_weight,b_blocks_6_0_bn2_running_var]}
        bias=t829 {folded from=[p_blocks_6_0_bn2_weight,p_blocks_6_0_bn2_bias,b_blocks_6_0_bn2_running_mean,b_blocks_6_0_bn2_running_var]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=480;
               groups=480}
    n508 {pt2=root[143] torch.ops.aten.hardtanh.default}: [t927 f32 [H=7 W=7
                                                                     C=480] {derived} ->[n509]] =
      hardtanh x=t926 {derived} <-n507 params={min_val=0; max_val=6}
    n509 {derived}: [t928 f32 [H=7 W=7 C=160] {derived} ->[n510]] =
      conv2d
        x=t927 {derived} <-n508
        weight=t830 {folded from=[p_blocks_6_0_conv_pwl_weight,p_blocks_6_0_bn3_weight,b_blocks_6_0_bn3_running_var]}
        bias=t831 {folded from=[p_blocks_6_0_bn3_weight,p_blocks_6_0_bn3_bias,b_blocks_6_0_bn3_running_mean,b_blocks_6_0_bn3_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=480;
               groups=1}
    n510 {derived}: [t929 f32 [H=7 W=7 C=1280] {derived} ->[n511]] =
      conv2d
        x=t928 {derived} <-n509
        weight=t832 {folded from=[p_conv_head_weight,p_bn2_weight,b_bn2_running_var]}
        bias=t833 {folded from=[p_bn2_weight,p_bn2_bias,b_bn2_running_mean,b_bn2_running_var]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=160;
               groups=1}
    n511 {pt2=root[148] torch.ops.aten.hardtanh.default}: [t930 f32 [H=7 W=7
                                                                     C=1280] {derived} ->[n410]] =
      hardtanh x=t929 {derived} <-n510 params={min_val=0; max_val=6}
    group g105 torch.ops.aten.adaptive_avg_pool2d.default:
      n410 {derived}: [t725 f32 [C=1280] {pt2=root:view} ->[n414]] =
        adaptive_avg_pool2d
          x=t930 {derived} <-n511
          params={output_size={h=1; w=1}}
    group g106 torch.ops.aten.linear.default:
      n414 {pt2=root[151] torch.ops.aten.linear.default}: [t729 f32 [C=1000] {pt2=root:linear}] =
        linear
          x=t725 {pt2=root:view} <-n410
          weight=t728 {folded from=[p_classifier_weight]}
          bias=t157 {pt2=root:p_classifier_bias target=classifier.bias}
          params={in_features=1280}
  outputs: [t729 f32 [C=1000] {pt2=root:linear} <-n414]
