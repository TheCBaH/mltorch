Convert two real models to the Native4D four-axis dialect, and verify the
conversion map for one of them. Gated on PT2_DATA; run with `make pt2.runtest`
after `make pt2.download-cram`. See .ai/native4d_plan.md stage 8.

RegNetX-002 and MobileNetV2-050 stand in for the retired resnet18/mobilenet_v2
role models. Unlike that pair, which both fully converted,
RegNetX-002 hits a real Native4D domain limit here: its first stage-3
convolution has 3 groups, which is neither `1` (dense) nor
`in_channels` (depthwise) — the only two groupings this dialect recognizes
(.ai/native4d_design.md). That is a genuine limit, not a bug, so showing it
alongside a model that DOES convert is more informative than the old pair,
which never exercised the rejection path at all.

Conversion is only meaningful on a CANONICAL graph: the importer emits every
conv weight right-aligned from ATen's [out, in, kh, kw], which lands on
D/H/W/C, so an unpipelined graph has non-unit D on every weight and is outside
the dialect entirely. Constant folding is what materialises the relaid
weights, and batch-norm folding is what removes the BatchNorms that would
otherwise each need their own precomputation. Both now run unconditionally in
the canonical pipeline (`lib/native/transform/pipeline.ml`'s `~fold:_`), so
`--fold` below no longer gates this — it is kept only because it still
controls whether real payload bytes are preloaded (see
test/me_visualize_cram.t's `feature:fold` capability).

RegNetX-002: rejected before Native4D produces any graph at all, so there is
nothing to verify.

  $ ../bin/native_graph.exe to4d --pt2 "$PT2_DATA/regnetx_002/regnetx_002.pt2" --fold --verify-symbolic standard
  canonical native: nodes=101
  outside the dialect: node n373: convolution has 3 groups, which is neither 1 nor depthwise

MobileNetV2-050, whole structure, with each edge's verification verdict beside
it. `DepthwiseConv2D` is the point of including this model: grouping is a
CONSTRUCTOR in this dialect rather than a parameter, so a depthwise
convolution is a different op from a dense one and the classification has to
be right for the graph to convert at all. `Hardtanh` is ReLU6.

The verdicts are what make this more than a shape dump. `proved (structural)`
holds for every input; `proved (structural, for these constants)` is the
weaker tier that substitutes the folded weights; `too large` is the budget
refusing an activation tensor before doing any work, which is a decline rather
than a doubt. What must not appear anywhere is `refuted`.

  $ ../bin/native_graph.exe to4d --pt2 "$PT2_DATA/mobilenetv2_050/mobilenetv2_050.pt2" --fold --verify-symbolic standard
  canonical native: nodes=100
  native4d: nodes=100 inputs=107
    Adaptive_avg_pool2d 1
    Add                10
    Conv2D             36
    DepthwiseConv2D    17
    Hardtanh           35
    Permute4           1
  map verification: 207 clusters: 1 proved (for these constants) [sampled 8], 71 proved (structural) [sampled 8], 32 unproved (too large), 1 unproved (unbound constant), 102 unproved (unbound constant) [sampled 8]
  graph4
  inputs: [t157 {proved (structural, for these constants) [sampled 8]} [C=1000],
  t314 {unproved: too large (150528 coords)} [H=3 W=224 C=224],
  t728 {unproved: too large (1280000 coords)} [N=1000 T=1 D=1 H=1 W=1 C=1280],
  t730 {unproved: unbound constant: src.t730(0) [sampled 8]} [N=16 T=1 D=1 H=3 W=3 C=3],
  t731 {unproved: unbound constant: src.t731(0) [sampled 8]} [C=16],
  t732 {unproved: unbound constant: src.t732(0) [sampled 8]} [N=16 T=1 D=1 H=3 W=3 C=1],
  t733 {unproved: unbound constant: src.t733(0) [sampled 8]} [C=16],
  t734 {unproved: unbound constant: src.t734(0) [sampled 8]} [N=8 T=1 D=1 H=1 W=1 C=16],
  t735 {unproved: unbound constant: src.t735(0) [exhaustive]} [C=8],
  t736 {unproved: unbound constant: src.t736(0) [sampled 8]} [N=48 T=1 D=1 H=1 W=1 C=8],
  t737 {unproved: unbound constant: src.t737(0) [sampled 8]} [C=48],
  t738 {unproved: unbound constant: src.t738(0) [sampled 8]} [N=48 T=1 D=1 H=3 W=3 C=1],
  t739 {unproved: unbound constant: src.t739(0) [sampled 8]} [C=48],
  t740 {unproved: unbound constant: src.t740(0) [sampled 8]} [N=16 T=1 D=1 H=1 W=1 C=48],
  t741 {unproved: unbound constant: src.t741(0) [sampled 8]} [C=16],
  t742 {unproved: unbound constant: src.t742(0) [sampled 8]} [N=96 T=1 D=1 H=1 W=1 C=16],
  t743 {unproved: unbound constant: src.t743(0) [sampled 8]} [C=96],
  t744 {unproved: unbound constant: src.t744(0) [sampled 8]} [N=96 T=1 D=1 H=3 W=3 C=1],
  t745 {unproved: unbound constant: src.t745(0) [sampled 8]} [C=96],
  t746 {unproved: unbound constant: src.t746(0) [sampled 8]} [N=16 T=1 D=1 H=1 W=1 C=96],
  t747 {unproved: unbound constant: src.t747(0) [sampled 8]} [C=16],
  t748 {unproved: unbound constant: src.t748(0) [sampled 8]} [N=96 T=1 D=1 H=1 W=1 C=16],
  t749 {unproved: unbound constant: src.t749(0) [sampled 8]} [C=96],
  t750 {unproved: unbound constant: src.t750(0) [sampled 8]} [N=96 T=1 D=1 H=3 W=3 C=1],
  t751 {unproved: unbound constant: src.t751(0) [sampled 8]} [C=96],
  t752 {unproved: unbound constant: src.t752(0) [sampled 8]} [N=16 T=1 D=1 H=1 W=1 C=96],
  t753 {unproved: unbound constant: src.t753(0) [sampled 8]} [C=16],
  t754 {unproved: unbound constant: src.t754(0) [sampled 8]} [N=96 T=1 D=1 H=1 W=1 C=16],
  t755 {unproved: unbound constant: src.t755(0) [sampled 8]} [C=96],
  t756 {unproved: unbound constant: src.t756(0) [sampled 8]} [N=96 T=1 D=1 H=3 W=3 C=1],
  t757 {unproved: unbound constant: src.t757(0) [sampled 8]} [C=96],
  t758 {unproved: unbound constant: src.t758(0) [sampled 8]} [N=16 T=1 D=1 H=1 W=1 C=96],
  t759 {unproved: unbound constant: src.t759(0) [sampled 8]} [C=16],
  t760 {unproved: unbound constant: src.t760(0) [sampled 8]} [N=96 T=1 D=1 H=1 W=1 C=16],
  t761 {unproved: unbound constant: src.t761(0) [sampled 8]} [C=96],
  t762 {unproved: unbound constant: src.t762(0) [sampled 8]} [N=96 T=1 D=1 H=3 W=3 C=1],
  t763 {unproved: unbound constant: src.t763(0) [sampled 8]} [C=96],
  t764 {unproved: unbound constant: src.t764(0) [sampled 8]} [N=16 T=1 D=1 H=1 W=1 C=96],
  t765 {unproved: unbound constant: src.t765(0) [sampled 8]} [C=16],
  t766 {unproved: unbound constant: src.t766(0) [sampled 8]} [N=96 T=1 D=1 H=1 W=1 C=16],
  t767 {unproved: unbound constant: src.t767(0) [sampled 8]} [C=96],
  t768 {unproved: unbound constant: src.t768(0) [sampled 8]} [N=96 T=1 D=1 H=3 W=3 C=1],
  t769 {unproved: unbound constant: src.t769(0) [sampled 8]} [C=96],
  t770 {unproved: unbound constant: src.t770(0) [sampled 8]} [N=32 T=1 D=1 H=1 W=1 C=96],
  t771 {unproved: unbound constant: src.t771(0) [sampled 8]} [C=32],
  t772 {unproved: unbound constant: src.t772(0) [sampled 8]} [N=192 T=1 D=1 H=1 W=1 C=32],
  t773 {unproved: unbound constant: src.t773(0) [sampled 8]} [C=192],
  t774 {unproved: unbound constant: src.t774(0) [sampled 8]} [N=192 T=1 D=1 H=3 W=3 C=1],
  t775 {unproved: unbound constant: src.t775(0) [sampled 8]} [C=192],
  t776 {unproved: unbound constant: src.t776(0) [sampled 8]} [N=32 T=1 D=1 H=1 W=1 C=192],
  t777 {unproved: unbound constant: src.t777(0) [sampled 8]} [C=32],
  t778 {unproved: unbound constant: src.t778(0) [sampled 8]} [N=192 T=1 D=1 H=1 W=1 C=32],
  t779 {unproved: unbound constant: src.t779(0) [sampled 8]} [C=192],
  t780 {unproved: unbound constant: src.t780(0) [sampled 8]} [N=192 T=1 D=1 H=3 W=3 C=1],
  t781 {unproved: unbound constant: src.t781(0) [sampled 8]} [C=192],
  t782 {unproved: unbound constant: src.t782(0) [sampled 8]} [N=32 T=1 D=1 H=1 W=1 C=192],
  t783 {unproved: unbound constant: src.t783(0) [sampled 8]} [C=32],
  t784 {unproved: unbound constant: src.t784(0) [sampled 8]} [N=192 T=1 D=1 H=1 W=1 C=32],
  t785 {unproved: unbound constant: src.t785(0) [sampled 8]} [C=192],
  t786 {unproved: unbound constant: src.t786(0) [sampled 8]} [N=192 T=1 D=1 H=3 W=3 C=1],
  t787 {unproved: unbound constant: src.t787(0) [sampled 8]} [C=192],
  t788 {unproved: unbound constant: src.t788(0) [sampled 8]} [N=32 T=1 D=1 H=1 W=1 C=192],
  t789 {unproved: unbound constant: src.t789(0) [sampled 8]} [C=32],
  t790 {unproved: unbound constant: src.t790(0) [sampled 8]} [N=192 T=1 D=1 H=1 W=1 C=32],
  t791 {unproved: unbound constant: src.t791(0) [sampled 8]} [C=192],
  t792 {unproved: unbound constant: src.t792(0) [sampled 8]} [N=192 T=1 D=1 H=3 W=3 C=1],
  t793 {unproved: unbound constant: src.t793(0) [sampled 8]} [C=192],
  t794 {unproved: unbound constant: src.t794(0) [sampled 8]} [N=48 T=1 D=1 H=1 W=1 C=192],
  t795 {unproved: unbound constant: src.t795(0) [sampled 8]} [C=48],
  t796 {unproved: unbound constant: src.t796(0) [sampled 8]} [N=288 T=1 D=1 H=1 W=1 C=48],
  t797 {unproved: unbound constant: src.t797(0) [sampled 8]} [C=288],
  t798 {unproved: unbound constant: src.t798(0) [sampled 8]} [N=288 T=1 D=1 H=3 W=3 C=1],
  t799 {unproved: unbound constant: src.t799(0) [sampled 8]} [C=288],
  t800 {unproved: unbound constant: src.t800(0) [sampled 8]} [N=48 T=1 D=1 H=1 W=1 C=288],
  t801 {unproved: unbound constant: src.t801(0) [sampled 8]} [C=48],
  t802 {unproved: unbound constant: src.t802(0) [sampled 8]} [N=288 T=1 D=1 H=1 W=1 C=48],
  t803 {unproved: unbound constant: src.t803(0) [sampled 8]} [C=288],
  t804 {unproved: unbound constant: src.t804(0) [sampled 8]} [N=288 T=1 D=1 H=3 W=3 C=1],
  t805 {unproved: unbound constant: src.t805(0) [sampled 8]} [C=288],
  t806 {unproved: unbound constant: src.t806(0) [sampled 8]} [N=48 T=1 D=1 H=1 W=1 C=288],
  t807 {unproved: unbound constant: src.t807(0) [sampled 8]} [C=48],
  t808 {unproved: unbound constant: src.t808(0) [sampled 8]} [N=288 T=1 D=1 H=1 W=1 C=48],
  t809 {unproved: unbound constant: src.t809(0) [sampled 8]} [C=288],
  t810 {unproved: unbound constant: src.t810(0) [sampled 8]} [N=288 T=1 D=1 H=3 W=3 C=1],
  t811 {unproved: unbound constant: src.t811(0) [sampled 8]} [C=288],
  t812 {unproved: unbound constant: src.t812(0) [sampled 8]} [N=80 T=1 D=1 H=1 W=1 C=288],
  t813 {unproved: unbound constant: src.t813(0) [sampled 8]} [C=80],
  t814 {unproved: unbound constant: src.t814(0) [sampled 8]} [N=480 T=1 D=1 H=1 W=1 C=80],
  t815 {unproved: unbound constant: src.t815(0) [sampled 8]} [C=480],
  t816 {unproved: unbound constant: src.t816(0) [sampled 8]} [N=480 T=1 D=1 H=3 W=3 C=1],
  t817 {unproved: unbound constant: src.t817(0) [sampled 8]} [C=480],
  t818 {unproved: unbound constant: src.t818(0) [sampled 8]} [N=80 T=1 D=1 H=1 W=1 C=480],
  t819 {unproved: unbound constant: src.t819(0) [sampled 8]} [C=80],
  t820 {unproved: unbound constant: src.t820(0) [sampled 8]} [N=480 T=1 D=1 H=1 W=1 C=80],
  t821 {unproved: unbound constant: src.t821(0) [sampled 8]} [C=480],
  t822 {unproved: unbound constant: src.t822(0) [sampled 8]} [N=480 T=1 D=1 H=3 W=3 C=1],
  t823 {unproved: unbound constant: src.t823(0) [sampled 8]} [C=480],
  t824 {unproved: unbound constant: src.t824(0) [sampled 8]} [N=80 T=1 D=1 H=1 W=1 C=480],
  t825 {unproved: unbound constant: src.t825(0) [sampled 8]} [C=80],
  t826 {unproved: unbound constant: src.t826(0) [sampled 8]} [N=480 T=1 D=1 H=1 W=1 C=80],
  t827 {unproved: unbound constant: src.t827(0) [sampled 8]} [C=480],
  t828 {unproved: unbound constant: src.t828(0) [sampled 8]} [N=480 T=1 D=1 H=3 W=3 C=1],
  t829 {unproved: unbound constant: src.t829(0) [sampled 8]} [C=480],
  t830 {unproved: too large (76800 coords)} [N=160 T=1 D=1 H=1 W=1 C=480],
  t831 {unproved: unbound constant: src.t831(0) [sampled 8]} [C=160],
  t832 {unproved: too large (204800 coords)} [N=1280 T=1 D=1 H=1 W=1 C=160],
  t833 {unproved: unbound constant: src.t833(0) [sampled 8]} [C=1280]]
  nodes:
    n0: [t315 {unproved: too large (150528 coords)}] =
      permute4
        x=t314 {unproved: too large (150528 coords)}
        perm=[H<-W, W<-C, C<-H]
    n415: [t834 {unproved: too large (200704 coords)}] =
      conv2d
        x=t315 {unproved: too large (150528 coords)}
        weight=t730 {unproved: unbound constant: src.t730(0) [sampled 8]}
        bias=t731 {unproved: unbound constant: src.t731(0) [sampled 8]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=3}
    n416: [t835 {unproved: too large (200704 coords)}] =
      hardtanh
        x=t834 {unproved: too large (200704 coords)}
        params={min_val=0; max_val=6}
    n417: [t836 {unproved: too large (200704 coords)}] =
      depthwise_conv2d
        x=t835 {unproved: too large (200704 coords)}
        weight=t732 {unproved: unbound constant: src.t732(0) [sampled 8]}
        bias=t733 {unproved: unbound constant: src.t733(0) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=16}
    n418: [t837 {unproved: too large (200704 coords)}] =
      hardtanh
        x=t836 {unproved: too large (200704 coords)}
        params={min_val=0; max_val=6}
    n419: [t838 {unproved: too large (100352 coords)}] =
      conv2d
        x=t837 {unproved: too large (200704 coords)}
        weight=t734 {unproved: unbound constant: src.t734(0) [sampled 8]}
        bias=t735 {unproved: unbound constant: src.t735(0) [exhaustive]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=16}
    n420: [t839 {unproved: too large (602112 coords)}] =
      conv2d
        x=t838 {unproved: too large (100352 coords)}
        weight=t736 {unproved: unbound constant: src.t736(0) [sampled 8]}
        bias=t737 {unproved: unbound constant: src.t737(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=8}
    n421: [t840 {unproved: too large (602112 coords)}] =
      hardtanh
        x=t839 {unproved: too large (602112 coords)}
        params={min_val=0; max_val=6}
    n422: [t841 {unproved: too large (150528 coords)}] =
      depthwise_conv2d
        x=t840 {unproved: too large (602112 coords)}
        weight=t738 {unproved: unbound constant: src.t738(0) [sampled 8]}
        bias=t739 {unproved: unbound constant: src.t739(0) [sampled 8]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=48}
    n423: [t842 {unproved: too large (150528 coords)}] =
      hardtanh
        x=t841 {unproved: too large (150528 coords)}
        params={min_val=0; max_val=6}
    n424: [t843 {proved (structural) [sampled 8]}] =
      conv2d
        x=t842 {unproved: too large (150528 coords)}
        weight=t740 {unproved: unbound constant: src.t740(0) [sampled 8]}
        bias=t741 {unproved: unbound constant: src.t741(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=48}
    n425: [t844 {unproved: too large (301056 coords)}] =
      conv2d
        x=t843 {proved (structural) [sampled 8]}
        weight=t742 {unproved: unbound constant: src.t742(0) [sampled 8]}
        bias=t743 {unproved: unbound constant: src.t743(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=16}
    n426: [t845 {unproved: too large (301056 coords)}] =
      hardtanh
        x=t844 {unproved: too large (301056 coords)}
        params={min_val=0; max_val=6}
    n427: [t846 {unproved: too large (301056 coords)}] =
      depthwise_conv2d
        x=t845 {unproved: too large (301056 coords)}
        weight=t744 {unproved: unbound constant: src.t744(0) [sampled 8]}
        bias=t745 {unproved: unbound constant: src.t745(0) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=96}
    n428: [t847 {unproved: too large (301056 coords)}] =
      hardtanh
        x=t846 {unproved: too large (301056 coords)}
        params={min_val=0; max_val=6}
    n429: [t848 {proved (structural) [sampled 8]}] =
      conv2d
        x=t847 {unproved: too large (301056 coords)}
        weight=t746 {unproved: unbound constant: src.t746(0) [sampled 8]}
        bias=t747 {unproved: unbound constant: src.t747(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=96}
    n430: [t849 {proved (structural) [sampled 8]}] =
      add
        a=t848 {proved (structural) [sampled 8]}
        b=t843 {proved (structural) [sampled 8]}
    n431: [t850 {unproved: too large (301056 coords)}] =
      conv2d
        x=t849 {proved (structural) [sampled 8]}
        weight=t748 {unproved: unbound constant: src.t748(0) [sampled 8]}
        bias=t749 {unproved: unbound constant: src.t749(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=16}
    n432: [t851 {unproved: too large (301056 coords)}] =
      hardtanh
        x=t850 {unproved: too large (301056 coords)}
        params={min_val=0; max_val=6}
    n433: [t852 {unproved: too large (75264 coords)}] =
      depthwise_conv2d
        x=t851 {unproved: too large (301056 coords)}
        weight=t750 {unproved: unbound constant: src.t750(0) [sampled 8]}
        bias=t751 {unproved: unbound constant: src.t751(0) [sampled 8]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=96}
    n434: [t853 {unproved: too large (75264 coords)}] =
      hardtanh
        x=t852 {unproved: too large (75264 coords)}
        params={min_val=0; max_val=6}
    n435: [t854 {proved (structural) [sampled 8]}] =
      conv2d
        x=t853 {unproved: too large (75264 coords)}
        weight=t752 {unproved: unbound constant: src.t752(0) [sampled 8]}
        bias=t753 {unproved: unbound constant: src.t753(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=96}
    n436: [t855 {unproved: too large (75264 coords)}] =
      conv2d
        x=t854 {proved (structural) [sampled 8]}
        weight=t754 {unproved: unbound constant: src.t754(0) [sampled 8]}
        bias=t755 {unproved: unbound constant: src.t755(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=16}
    n437: [t856 {unproved: too large (75264 coords)}] =
      hardtanh
        x=t855 {unproved: too large (75264 coords)}
        params={min_val=0; max_val=6}
    n438: [t857 {unproved: too large (75264 coords)}] =
      depthwise_conv2d
        x=t856 {unproved: too large (75264 coords)}
        weight=t756 {unproved: unbound constant: src.t756(0) [sampled 8]}
        bias=t757 {unproved: unbound constant: src.t757(0) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=96}
    n439: [t858 {unproved: too large (75264 coords)}] =
      hardtanh
        x=t857 {unproved: too large (75264 coords)}
        params={min_val=0; max_val=6}
    n440: [t859 {proved (structural) [sampled 8]}] =
      conv2d
        x=t858 {unproved: too large (75264 coords)}
        weight=t758 {unproved: unbound constant: src.t758(0) [sampled 8]}
        bias=t759 {unproved: unbound constant: src.t759(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=96}
    n441: [t860 {proved (structural) [sampled 8]}] =
      add
        a=t859 {proved (structural) [sampled 8]}
        b=t854 {proved (structural) [sampled 8]}
    n442: [t861 {unproved: too large (75264 coords)}] =
      conv2d
        x=t860 {proved (structural) [sampled 8]}
        weight=t760 {unproved: unbound constant: src.t760(0) [sampled 8]}
        bias=t761 {unproved: unbound constant: src.t761(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=16}
    n443: [t862 {unproved: too large (75264 coords)}] =
      hardtanh
        x=t861 {unproved: too large (75264 coords)}
        params={min_val=0; max_val=6}
    n444: [t863 {unproved: too large (75264 coords)}] =
      depthwise_conv2d
        x=t862 {unproved: too large (75264 coords)}
        weight=t762 {unproved: unbound constant: src.t762(0) [sampled 8]}
        bias=t763 {unproved: unbound constant: src.t763(0) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=96}
    n445: [t864 {unproved: too large (75264 coords)}] =
      hardtanh
        x=t863 {unproved: too large (75264 coords)}
        params={min_val=0; max_val=6}
    n446: [t865 {proved (structural) [sampled 8]}] =
      conv2d
        x=t864 {unproved: too large (75264 coords)}
        weight=t764 {unproved: unbound constant: src.t764(0) [sampled 8]}
        bias=t765 {unproved: unbound constant: src.t765(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=96}
    n447: [t866 {proved (structural) [sampled 8]}] =
      add
        a=t865 {proved (structural) [sampled 8]}
        b=t860 {proved (structural) [sampled 8]}
    n448: [t867 {unproved: too large (75264 coords)}] =
      conv2d
        x=t866 {proved (structural) [sampled 8]}
        weight=t766 {unproved: unbound constant: src.t766(0) [sampled 8]}
        bias=t767 {unproved: unbound constant: src.t767(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=16}
    n449: [t868 {unproved: too large (75264 coords)}] =
      hardtanh
        x=t867 {unproved: too large (75264 coords)}
        params={min_val=0; max_val=6}
    n450: [t869 {proved (structural) [sampled 8]}] =
      depthwise_conv2d
        x=t868 {unproved: too large (75264 coords)}
        weight=t768 {unproved: unbound constant: src.t768(0) [sampled 8]}
        bias=t769 {unproved: unbound constant: src.t769(0) [sampled 8]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=96}
    n451: [t870 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t869 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n452: [t871 {proved (structural) [sampled 8]}] =
      conv2d
        x=t870 {proved (structural) [sampled 8]}
        weight=t770 {unproved: unbound constant: src.t770(0) [sampled 8]}
        bias=t771 {unproved: unbound constant: src.t771(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=96}
    n453: [t872 {proved (structural) [sampled 8]}] =
      conv2d
        x=t871 {proved (structural) [sampled 8]}
        weight=t772 {unproved: unbound constant: src.t772(0) [sampled 8]}
        bias=t773 {unproved: unbound constant: src.t773(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=32}
    n454: [t873 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t872 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n455: [t874 {proved (structural) [sampled 8]}] =
      depthwise_conv2d
        x=t873 {proved (structural) [sampled 8]}
        weight=t774 {unproved: unbound constant: src.t774(0) [sampled 8]}
        bias=t775 {unproved: unbound constant: src.t775(0) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=192}
    n456: [t875 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t874 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n457: [t876 {proved (structural) [sampled 8]}] =
      conv2d
        x=t875 {proved (structural) [sampled 8]}
        weight=t776 {unproved: unbound constant: src.t776(0) [sampled 8]}
        bias=t777 {unproved: unbound constant: src.t777(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=192}
    n458: [t877 {proved (structural) [sampled 8]}] =
      add
        a=t876 {proved (structural) [sampled 8]}
        b=t871 {proved (structural) [sampled 8]}
    n459: [t878 {proved (structural) [sampled 8]}] =
      conv2d
        x=t877 {proved (structural) [sampled 8]}
        weight=t778 {unproved: unbound constant: src.t778(0) [sampled 8]}
        bias=t779 {unproved: unbound constant: src.t779(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=32}
    n460: [t879 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t878 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n461: [t880 {proved (structural) [sampled 8]}] =
      depthwise_conv2d
        x=t879 {proved (structural) [sampled 8]}
        weight=t780 {unproved: unbound constant: src.t780(0) [sampled 8]}
        bias=t781 {unproved: unbound constant: src.t781(0) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=192}
    n462: [t881 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t880 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n463: [t882 {proved (structural) [sampled 8]}] =
      conv2d
        x=t881 {proved (structural) [sampled 8]}
        weight=t782 {unproved: unbound constant: src.t782(0) [sampled 8]}
        bias=t783 {unproved: unbound constant: src.t783(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=192}
    n464: [t883 {proved (structural) [sampled 8]}] =
      add
        a=t882 {proved (structural) [sampled 8]}
        b=t877 {proved (structural) [sampled 8]}
    n465: [t884 {proved (structural) [sampled 8]}] =
      conv2d
        x=t883 {proved (structural) [sampled 8]}
        weight=t784 {unproved: unbound constant: src.t784(0) [sampled 8]}
        bias=t785 {unproved: unbound constant: src.t785(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=32}
    n466: [t885 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t884 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n467: [t886 {proved (structural) [sampled 8]}] =
      depthwise_conv2d
        x=t885 {proved (structural) [sampled 8]}
        weight=t786 {unproved: unbound constant: src.t786(0) [sampled 8]}
        bias=t787 {unproved: unbound constant: src.t787(0) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=192}
    n468: [t887 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t886 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n469: [t888 {proved (structural) [sampled 8]}] =
      conv2d
        x=t887 {proved (structural) [sampled 8]}
        weight=t788 {unproved: unbound constant: src.t788(0) [sampled 8]}
        bias=t789 {unproved: unbound constant: src.t789(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=192}
    n470: [t889 {proved (structural) [sampled 8]}] =
      add
        a=t888 {proved (structural) [sampled 8]}
        b=t883 {proved (structural) [sampled 8]}
    n471: [t890 {proved (structural) [sampled 8]}] =
      conv2d
        x=t889 {proved (structural) [sampled 8]}
        weight=t790 {unproved: unbound constant: src.t790(0) [sampled 8]}
        bias=t791 {unproved: unbound constant: src.t791(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=32}
    n472: [t891 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t890 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n473: [t892 {proved (structural) [sampled 8]}] =
      depthwise_conv2d
        x=t891 {proved (structural) [sampled 8]}
        weight=t792 {unproved: unbound constant: src.t792(0) [sampled 8]}
        bias=t793 {unproved: unbound constant: src.t793(0) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=192}
    n474: [t893 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t892 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n475: [t894 {proved (structural) [sampled 8]}] =
      conv2d
        x=t893 {proved (structural) [sampled 8]}
        weight=t794 {unproved: unbound constant: src.t794(0) [sampled 8]}
        bias=t795 {unproved: unbound constant: src.t795(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=192}
    n476: [t895 {proved (structural) [sampled 8]}] =
      conv2d
        x=t894 {proved (structural) [sampled 8]}
        weight=t796 {unproved: unbound constant: src.t796(0) [sampled 8]}
        bias=t797 {unproved: unbound constant: src.t797(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=48}
    n477: [t896 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t895 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n478: [t897 {proved (structural) [sampled 8]}] =
      depthwise_conv2d
        x=t896 {proved (structural) [sampled 8]}
        weight=t798 {unproved: unbound constant: src.t798(0) [sampled 8]}
        bias=t799 {unproved: unbound constant: src.t799(0) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=288}
    n479: [t898 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t897 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n480: [t899 {proved (structural) [sampled 8]}] =
      conv2d
        x=t898 {proved (structural) [sampled 8]}
        weight=t800 {unproved: unbound constant: src.t800(0) [sampled 8]}
        bias=t801 {unproved: unbound constant: src.t801(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=288}
    n481: [t900 {proved (structural) [sampled 8]}] =
      add
        a=t899 {proved (structural) [sampled 8]}
        b=t894 {proved (structural) [sampled 8]}
    n482: [t901 {proved (structural) [sampled 8]}] =
      conv2d
        x=t900 {proved (structural) [sampled 8]}
        weight=t802 {unproved: unbound constant: src.t802(0) [sampled 8]}
        bias=t803 {unproved: unbound constant: src.t803(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=48}
    n483: [t902 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t901 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n484: [t903 {proved (structural) [sampled 8]}] =
      depthwise_conv2d
        x=t902 {proved (structural) [sampled 8]}
        weight=t804 {unproved: unbound constant: src.t804(0) [sampled 8]}
        bias=t805 {unproved: unbound constant: src.t805(0) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=288}
    n485: [t904 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t903 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n486: [t905 {proved (structural) [sampled 8]}] =
      conv2d
        x=t904 {proved (structural) [sampled 8]}
        weight=t806 {unproved: unbound constant: src.t806(0) [sampled 8]}
        bias=t807 {unproved: unbound constant: src.t807(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=288}
    n487: [t906 {proved (structural) [sampled 8]}] =
      add
        a=t905 {proved (structural) [sampled 8]}
        b=t900 {proved (structural) [sampled 8]}
    n488: [t907 {proved (structural) [sampled 8]}] =
      conv2d
        x=t906 {proved (structural) [sampled 8]}
        weight=t808 {unproved: unbound constant: src.t808(0) [sampled 8]}
        bias=t809 {unproved: unbound constant: src.t809(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=48}
    n489: [t908 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t907 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n490: [t909 {proved (structural) [sampled 8]}] =
      depthwise_conv2d
        x=t908 {proved (structural) [sampled 8]}
        weight=t810 {unproved: unbound constant: src.t810(0) [sampled 8]}
        bias=t811 {unproved: unbound constant: src.t811(0) [sampled 8]}
        params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
               in_channels=288}
    n491: [t910 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t909 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n492: [t911 {proved (structural) [sampled 8]}] =
      conv2d
        x=t910 {proved (structural) [sampled 8]}
        weight=t812 {unproved: unbound constant: src.t812(0) [sampled 8]}
        bias=t813 {unproved: unbound constant: src.t813(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=288}
    n493: [t912 {proved (structural) [sampled 8]}] =
      conv2d
        x=t911 {proved (structural) [sampled 8]}
        weight=t814 {unproved: unbound constant: src.t814(0) [sampled 8]}
        bias=t815 {unproved: unbound constant: src.t815(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=80}
    n494: [t913 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t912 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n495: [t914 {proved (structural) [sampled 8]}] =
      depthwise_conv2d
        x=t913 {proved (structural) [sampled 8]}
        weight=t816 {unproved: unbound constant: src.t816(0) [sampled 8]}
        bias=t817 {unproved: unbound constant: src.t817(0) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=480}
    n496: [t915 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t914 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n497: [t916 {proved (structural) [sampled 8]}] =
      conv2d
        x=t915 {proved (structural) [sampled 8]}
        weight=t818 {unproved: unbound constant: src.t818(0) [sampled 8]}
        bias=t819 {unproved: unbound constant: src.t819(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=480}
    n498: [t917 {proved (structural) [sampled 8]}] =
      add
        a=t916 {proved (structural) [sampled 8]}
        b=t911 {proved (structural) [sampled 8]}
    n499: [t918 {proved (structural) [sampled 8]}] =
      conv2d
        x=t917 {proved (structural) [sampled 8]}
        weight=t820 {unproved: unbound constant: src.t820(0) [sampled 8]}
        bias=t821 {unproved: unbound constant: src.t821(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=80}
    n500: [t919 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t918 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n501: [t920 {proved (structural) [sampled 8]}] =
      depthwise_conv2d
        x=t919 {proved (structural) [sampled 8]}
        weight=t822 {unproved: unbound constant: src.t822(0) [sampled 8]}
        bias=t823 {unproved: unbound constant: src.t823(0) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=480}
    n502: [t921 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t920 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n503: [t922 {proved (structural) [sampled 8]}] =
      conv2d
        x=t921 {proved (structural) [sampled 8]}
        weight=t824 {unproved: unbound constant: src.t824(0) [sampled 8]}
        bias=t825 {unproved: unbound constant: src.t825(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=480}
    n504: [t923 {proved (structural) [sampled 8]}] =
      add
        a=t922 {proved (structural) [sampled 8]}
        b=t917 {proved (structural) [sampled 8]}
    n505: [t924 {proved (structural) [sampled 8]}] =
      conv2d
        x=t923 {proved (structural) [sampled 8]}
        weight=t826 {unproved: unbound constant: src.t826(0) [sampled 8]}
        bias=t827 {unproved: unbound constant: src.t827(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=80}
    n506: [t925 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t924 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n507: [t926 {proved (structural) [sampled 8]}] =
      depthwise_conv2d
        x=t925 {proved (structural) [sampled 8]}
        weight=t828 {unproved: unbound constant: src.t828(0) [sampled 8]}
        bias=t829 {unproved: unbound constant: src.t829(0) [sampled 8]}
        params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
               in_channels=480}
    n508: [t927 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t926 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n509: [t928 {proved (structural) [sampled 8]}] =
      conv2d
        x=t927 {proved (structural) [sampled 8]}
        weight=t830 {unproved: too large (76800 coords)}
        bias=t831 {unproved: unbound constant: src.t831(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=480}
    n510: [t929 {proved (structural) [sampled 8]}] =
      conv2d
        x=t928 {proved (structural) [sampled 8]}
        weight=t832 {unproved: too large (204800 coords)}
        bias=t833 {unproved: unbound constant: src.t833(0) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=160}
    n511: [t930 {proved (structural) [sampled 8]}] =
      hardtanh
        x=t929 {proved (structural) [sampled 8]}
        params={min_val=0; max_val=6}
    n410: [t725 {proved (structural) [sampled 8]}] =
      adaptive_avg_pool2d
        x=t930 {proved (structural) [sampled 8]}
        params={output_size={h=1; w=1}}
    n414: [t729 {unproved: unbound constant: src.t728(0) [sampled 8]}] =
      conv2d
        x=t725 {proved (structural) [sampled 8]}
        weight=t728 {unproved: too large (1280000 coords)}
        bias=t157 {proved (structural, for these constants) [sampled 8]}
        params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
               in_channels=1280}
  outputs: [t729 {unproved: unbound constant: src.t728(0) [sampled 8]} [C=1000]]

This is not the same claim as the numeric check `transform --verify` makes.
That one runs a model twice on one real input and compares outputs; this one
checks every corresponding edge without payloads for the graph inputs, so it
speaks about every input within its budget. Neither replaces the other.

  $ for m in regnetx_002 mobilenetv2_050; do
  >   n=$(../bin/native_graph.exe to4d --pt2 "$PT2_DATA/$m/$m.pt2" --fold \
  >       --verify-symbolic standard | grep -c refuted || true)
  >   echo "$m: $n refuted"
  > done
  regnetx_002: 0 refuted
  mobilenetv2_050: 0 refuted
