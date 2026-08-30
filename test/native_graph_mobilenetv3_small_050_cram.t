MobileNetV3-small-050's import, printed in full as RegNetX-002's is above
(`test/native_graph_regnetx_002_cram.t`) — it stands in for the retired
mobilenet_v3_small role model. Unlike that retired archive, this exporter
emits hardswish and its SE-block gate directly as the functional
`hardswish.default`/`hardsigmoid.default` ops rather than a decomposed
`mul(x, div_scalar(clamp(add_scalar(x,3),0,6), 6))` chain — real-model
coverage for exactly the functional spellings `.ai/testing_strategy.md`'s
Group 5 section discusses.

  $ ../bin/native_graph.exe print --pt2 "$PT2_DATA/mobilenetv3_small_050/mobilenetv3_small_050.pt2"
  native graph: inputs=245 constants=244 nodes=387 outputs=1
  PT2 provenance: tensor-origins=402 captured-targets=244 node-origins=157
  graph
  inputs:
    [t0 f32 [D=16 H=3 W=3 C=3] {pt2=root:p_conv_stem_weight target=conv_stem.weight} ->[n1] constant,
     t1 f32 [C=16] {pt2=root:p_bn1_weight target=bn1.weight} ->[n5] constant,
     t2 f32 [C=16] {pt2=root:p_bn1_bias target=bn1.bias} ->[n5] constant,
     t3 f32 [D=16 H=1 W=3 C=3] {pt2=root:p_blocks_0_0_conv_dw_weight target=blocks.0.0.conv_dw.weight} ->[n9] constant,
     t4 f32 [C=16] {pt2=root:p_blocks_0_0_bn1_weight target=blocks.0.0.bn1.weight} ->[n13] constant,
     t5 f32 [C=16] {pt2=root:p_blocks_0_0_bn1_bias target=blocks.0.0.bn1.bias} ->[n13] constant,
     t6 f32 [D=8 H=16 W=1 C=1] {pt2=root:p_blocks_0_0_se_conv_reduce_weight target=blocks.0.0.se.conv_reduce.weight} ->[n18] constant,
     t7 f32 [C=8] {pt2=root:p_blocks_0_0_se_conv_reduce_bias target=blocks.0.0.se.conv_reduce.bias} ->[n19] constant,
     t8 f32 [D=16 H=8 W=1 C=1] {pt2=root:p_blocks_0_0_se_conv_expand_weight target=blocks.0.0.se.conv_expand.weight} ->[n23] constant,
     t9 f32 [C=16] {pt2=root:p_blocks_0_0_se_conv_expand_bias target=blocks.0.0.se.conv_expand.bias} ->[n24] constant,
     t10 f32 [D=8 H=16 W=1 C=1] {pt2=root:p_blocks_0_0_conv_pw_weight target=blocks.0.0.conv_pw.weight} ->[n29] constant,
     t11 f32 [C=8] {pt2=root:p_blocks_0_0_bn2_weight target=blocks.0.0.bn2.weight} ->[n33] constant,
     t12 f32 [C=8] {pt2=root:p_blocks_0_0_bn2_bias target=blocks.0.0.bn2.bias} ->[n33] constant,
     t13 f32 [D=40 H=8 W=1 C=1] {pt2=root:p_blocks_1_0_conv_pw_weight target=blocks.1.0.conv_pw.weight} ->[n36] constant,
     t14 f32 [C=40] {pt2=root:p_blocks_1_0_bn1_weight target=blocks.1.0.bn1.weight} ->[n40] constant,
     t15 f32 [C=40] {pt2=root:p_blocks_1_0_bn1_bias target=blocks.1.0.bn1.bias} ->[n40] constant,
     t16 f32 [D=40 H=1 W=3 C=3] {pt2=root:p_blocks_1_0_conv_dw_weight target=blocks.1.0.conv_dw.weight} ->[n44] constant,
     t17 f32 [C=40] {pt2=root:p_blocks_1_0_bn2_weight target=blocks.1.0.bn2.weight} ->[n48] constant,
     t18 f32 [C=40] {pt2=root:p_blocks_1_0_bn2_bias target=blocks.1.0.bn2.bias} ->[n48] constant,
     t19 f32 [D=16 H=40 W=1 C=1] {pt2=root:p_blocks_1_0_conv_pwl_weight target=blocks.1.0.conv_pwl.weight} ->[n52] constant,
     t20 f32 [C=16] {pt2=root:p_blocks_1_0_bn3_weight target=blocks.1.0.bn3.weight} ->[n56] constant,
     t21 f32 [C=16] {pt2=root:p_blocks_1_0_bn3_bias target=blocks.1.0.bn3.bias} ->[n56] constant,
     t22 f32 [D=56 H=16 W=1 C=1] {pt2=root:p_blocks_1_1_conv_pw_weight target=blocks.1.1.conv_pw.weight} ->[n59] constant,
     t23 f32 [C=56] {pt2=root:p_blocks_1_1_bn1_weight target=blocks.1.1.bn1.weight} ->[n63] constant,
     t24 f32 [C=56] {pt2=root:p_blocks_1_1_bn1_bias target=blocks.1.1.bn1.bias} ->[n63] constant,
     t25 f32 [D=56 H=1 W=3 C=3] {pt2=root:p_blocks_1_1_conv_dw_weight target=blocks.1.1.conv_dw.weight} ->[n67] constant,
     t26 f32 [C=56] {pt2=root:p_blocks_1_1_bn2_weight target=blocks.1.1.bn2.weight} ->[n71] constant,
     t27 f32 [C=56] {pt2=root:p_blocks_1_1_bn2_bias target=blocks.1.1.bn2.bias} ->[n71] constant,
     t28 f32 [D=16 H=56 W=1 C=1] {pt2=root:p_blocks_1_1_conv_pwl_weight target=blocks.1.1.conv_pwl.weight} ->[n75] constant,
     t29 f32 [C=16] {pt2=root:p_blocks_1_1_bn3_weight target=blocks.1.1.bn3.weight} ->[n79] constant,
     t30 f32 [C=16] {pt2=root:p_blocks_1_1_bn3_bias target=blocks.1.1.bn3.bias} ->[n79] constant,
     t31 f32 [D=64 H=16 W=1 C=1] {pt2=root:p_blocks_2_0_conv_pw_weight target=blocks.2.0.conv_pw.weight} ->[n83] constant,
     t32 f32 [C=64] {pt2=root:p_blocks_2_0_bn1_weight target=blocks.2.0.bn1.weight} ->[n87] constant,
     t33 f32 [C=64] {pt2=root:p_blocks_2_0_bn1_bias target=blocks.2.0.bn1.bias} ->[n87] constant,
     t34 f32 [D=64 H=1 W=5 C=5] {pt2=root:p_blocks_2_0_conv_dw_weight target=blocks.2.0.conv_dw.weight} ->[n91] constant,
     t35 f32 [C=64] {pt2=root:p_blocks_2_0_bn2_weight target=blocks.2.0.bn2.weight} ->[n95] constant,
     t36 f32 [C=64] {pt2=root:p_blocks_2_0_bn2_bias target=blocks.2.0.bn2.bias} ->[n95] constant,
     t37 f32 [D=16 H=64 W=1 C=1] {pt2=root:p_blocks_2_0_se_conv_reduce_weight target=blocks.2.0.se.conv_reduce.weight} ->[n100] constant,
     t38 f32 [C=16] {pt2=root:p_blocks_2_0_se_conv_reduce_bias target=blocks.2.0.se.conv_reduce.bias} ->[n101] constant,
     t39 f32 [D=64 H=16 W=1 C=1] {pt2=root:p_blocks_2_0_se_conv_expand_weight target=blocks.2.0.se.conv_expand.weight} ->[n105] constant,
     t40 f32 [C=64] {pt2=root:p_blocks_2_0_se_conv_expand_bias target=blocks.2.0.se.conv_expand.bias} ->[n106] constant,
     t41 f32 [D=24 H=64 W=1 C=1] {pt2=root:p_blocks_2_0_conv_pwl_weight target=blocks.2.0.conv_pwl.weight} ->[n111] constant,
     t42 f32 [C=24] {pt2=root:p_blocks_2_0_bn3_weight target=blocks.2.0.bn3.weight} ->[n115] constant,
     t43 f32 [C=24] {pt2=root:p_blocks_2_0_bn3_bias target=blocks.2.0.bn3.bias} ->[n115] constant,
     t44 f32 [D=144 H=24 W=1 C=1] {pt2=root:p_blocks_2_1_conv_pw_weight target=blocks.2.1.conv_pw.weight} ->[n118] constant,
     t45 f32 [C=144] {pt2=root:p_blocks_2_1_bn1_weight target=blocks.2.1.bn1.weight} ->[n122] constant,
     t46 f32 [C=144] {pt2=root:p_blocks_2_1_bn1_bias target=blocks.2.1.bn1.bias} ->[n122] constant,
     t47 f32 [D=144 H=1 W=5 C=5] {pt2=root:p_blocks_2_1_conv_dw_weight target=blocks.2.1.conv_dw.weight} ->[n126] constant,
     t48 f32 [C=144] {pt2=root:p_blocks_2_1_bn2_weight target=blocks.2.1.bn2.weight} ->[n130] constant,
     t49 f32 [C=144] {pt2=root:p_blocks_2_1_bn2_bias target=blocks.2.1.bn2.bias} ->[n130] constant,
     t50 f32 [D=40 H=144 W=1 C=1] {pt2=root:p_blocks_2_1_se_conv_reduce_weight target=blocks.2.1.se.conv_reduce.weight} ->[n135] constant,
     t51 f32 [C=40] {pt2=root:p_blocks_2_1_se_conv_reduce_bias target=blocks.2.1.se.conv_reduce.bias} ->[n136] constant,
     t52 f32 [D=144 H=40 W=1 C=1] {pt2=root:p_blocks_2_1_se_conv_expand_weight target=blocks.2.1.se.conv_expand.weight} ->[n140] constant,
     t53 f32 [C=144] {pt2=root:p_blocks_2_1_se_conv_expand_bias target=blocks.2.1.se.conv_expand.bias} ->[n141] constant,
     t54 f32 [D=24 H=144 W=1 C=1] {pt2=root:p_blocks_2_1_conv_pwl_weight target=blocks.2.1.conv_pwl.weight} ->[n146] constant,
     t55 f32 [C=24] {pt2=root:p_blocks_2_1_bn3_weight target=blocks.2.1.bn3.weight} ->[n150] constant,
     t56 f32 [C=24] {pt2=root:p_blocks_2_1_bn3_bias target=blocks.2.1.bn3.bias} ->[n150] constant,
     t57 f32 [D=144 H=24 W=1 C=1] {pt2=root:p_blocks_2_2_conv_pw_weight target=blocks.2.2.conv_pw.weight} ->[n154] constant,
     t58 f32 [C=144] {pt2=root:p_blocks_2_2_bn1_weight target=blocks.2.2.bn1.weight} ->[n158] constant,
     t59 f32 [C=144] {pt2=root:p_blocks_2_2_bn1_bias target=blocks.2.2.bn1.bias} ->[n158] constant,
     t60 f32 [D=144 H=1 W=5 C=5] {pt2=root:p_blocks_2_2_conv_dw_weight target=blocks.2.2.conv_dw.weight} ->[n162] constant,
     t61 f32 [C=144] {pt2=root:p_blocks_2_2_bn2_weight target=blocks.2.2.bn2.weight} ->[n166] constant,
     t62 f32 [C=144] {pt2=root:p_blocks_2_2_bn2_bias target=blocks.2.2.bn2.bias} ->[n166] constant,
     t63 f32 [D=40 H=144 W=1 C=1] {pt2=root:p_blocks_2_2_se_conv_reduce_weight target=blocks.2.2.se.conv_reduce.weight} ->[n171] constant,
     t64 f32 [C=40] {pt2=root:p_blocks_2_2_se_conv_reduce_bias target=blocks.2.2.se.conv_reduce.bias} ->[n172] constant,
     t65 f32 [D=144 H=40 W=1 C=1] {pt2=root:p_blocks_2_2_se_conv_expand_weight target=blocks.2.2.se.conv_expand.weight} ->[n176] constant,
     t66 f32 [C=144] {pt2=root:p_blocks_2_2_se_conv_expand_bias target=blocks.2.2.se.conv_expand.bias} ->[n177] constant,
     t67 f32 [D=24 H=144 W=1 C=1] {pt2=root:p_blocks_2_2_conv_pwl_weight target=blocks.2.2.conv_pwl.weight} ->[n182] constant,
     t68 f32 [C=24] {pt2=root:p_blocks_2_2_bn3_weight target=blocks.2.2.bn3.weight} ->[n186] constant,
     t69 f32 [C=24] {pt2=root:p_blocks_2_2_bn3_bias target=blocks.2.2.bn3.bias} ->[n186] constant,
     t70 f32 [D=72 H=24 W=1 C=1] {pt2=root:p_blocks_3_0_conv_pw_weight target=blocks.3.0.conv_pw.weight} ->[n190] constant,
     t71 f32 [C=72] {pt2=root:p_blocks_3_0_bn1_weight target=blocks.3.0.bn1.weight} ->[n194] constant,
     t72 f32 [C=72] {pt2=root:p_blocks_3_0_bn1_bias target=blocks.3.0.bn1.bias} ->[n194] constant,
     t73 f32 [D=72 H=1 W=5 C=5] {pt2=root:p_blocks_3_0_conv_dw_weight target=blocks.3.0.conv_dw.weight} ->[n198] constant,
     t74 f32 [C=72] {pt2=root:p_blocks_3_0_bn2_weight target=blocks.3.0.bn2.weight} ->[n202] constant,
     t75 f32 [C=72] {pt2=root:p_blocks_3_0_bn2_bias target=blocks.3.0.bn2.bias} ->[n202] constant,
     t76 f32 [D=24 H=72 W=1 C=1] {pt2=root:p_blocks_3_0_se_conv_reduce_weight target=blocks.3.0.se.conv_reduce.weight} ->[n207] constant,
     t77 f32 [C=24] {pt2=root:p_blocks_3_0_se_conv_reduce_bias target=blocks.3.0.se.conv_reduce.bias} ->[n208] constant,
     t78 f32 [D=72 H=24 W=1 C=1] {pt2=root:p_blocks_3_0_se_conv_expand_weight target=blocks.3.0.se.conv_expand.weight} ->[n212] constant,
     t79 f32 [C=72] {pt2=root:p_blocks_3_0_se_conv_expand_bias target=blocks.3.0.se.conv_expand.bias} ->[n213] constant,
     t80 f32 [D=24 H=72 W=1 C=1] {pt2=root:p_blocks_3_0_conv_pwl_weight target=blocks.3.0.conv_pwl.weight} ->[n218] constant,
     t81 f32 [C=24] {pt2=root:p_blocks_3_0_bn3_weight target=blocks.3.0.bn3.weight} ->[n222] constant,
     t82 f32 [C=24] {pt2=root:p_blocks_3_0_bn3_bias target=blocks.3.0.bn3.bias} ->[n222] constant,
     t83 f32 [D=72 H=24 W=1 C=1] {pt2=root:p_blocks_3_1_conv_pw_weight target=blocks.3.1.conv_pw.weight} ->[n226] constant,
     t84 f32 [C=72] {pt2=root:p_blocks_3_1_bn1_weight target=blocks.3.1.bn1.weight} ->[n230] constant,
     t85 f32 [C=72] {pt2=root:p_blocks_3_1_bn1_bias target=blocks.3.1.bn1.bias} ->[n230] constant,
     t86 f32 [D=72 H=1 W=5 C=5] {pt2=root:p_blocks_3_1_conv_dw_weight target=blocks.3.1.conv_dw.weight} ->[n234] constant,
     t87 f32 [C=72] {pt2=root:p_blocks_3_1_bn2_weight target=blocks.3.1.bn2.weight} ->[n238] constant,
     t88 f32 [C=72] {pt2=root:p_blocks_3_1_bn2_bias target=blocks.3.1.bn2.bias} ->[n238] constant,
     t89 f32 [D=24 H=72 W=1 C=1] {pt2=root:p_blocks_3_1_se_conv_reduce_weight target=blocks.3.1.se.conv_reduce.weight} ->[n243] constant,
     t90 f32 [C=24] {pt2=root:p_blocks_3_1_se_conv_reduce_bias target=blocks.3.1.se.conv_reduce.bias} ->[n244] constant,
     t91 f32 [D=72 H=24 W=1 C=1] {pt2=root:p_blocks_3_1_se_conv_expand_weight target=blocks.3.1.se.conv_expand.weight} ->[n248] constant,
     t92 f32 [C=72] {pt2=root:p_blocks_3_1_se_conv_expand_bias target=blocks.3.1.se.conv_expand.bias} ->[n249] constant,
     t93 f32 [D=24 H=72 W=1 C=1] {pt2=root:p_blocks_3_1_conv_pwl_weight target=blocks.3.1.conv_pwl.weight} ->[n254] constant,
     t94 f32 [C=24] {pt2=root:p_blocks_3_1_bn3_weight target=blocks.3.1.bn3.weight} ->[n258] constant,
     t95 f32 [C=24] {pt2=root:p_blocks_3_1_bn3_bias target=blocks.3.1.bn3.bias} ->[n258] constant,
     t96 f32 [D=144 H=24 W=1 C=1] {pt2=root:p_blocks_4_0_conv_pw_weight target=blocks.4.0.conv_pw.weight} ->[n262] constant,
     t97 f32 [C=144] {pt2=root:p_blocks_4_0_bn1_weight target=blocks.4.0.bn1.weight} ->[n266] constant,
     t98 f32 [C=144] {pt2=root:p_blocks_4_0_bn1_bias target=blocks.4.0.bn1.bias} ->[n266] constant,
     t99 f32 [D=144 H=1 W=5 C=5] {pt2=root:p_blocks_4_0_conv_dw_weight target=blocks.4.0.conv_dw.weight} ->[n270] constant,
     t100 f32 [C=144] {pt2=root:p_blocks_4_0_bn2_weight target=blocks.4.0.bn2.weight} ->[n274] constant,
     t101 f32 [C=144] {pt2=root:p_blocks_4_0_bn2_bias target=blocks.4.0.bn2.bias} ->[n274] constant,
     t102 f32 [D=40 H=144 W=1 C=1] {pt2=root:p_blocks_4_0_se_conv_reduce_weight target=blocks.4.0.se.conv_reduce.weight} ->[n279] constant,
     t103 f32 [C=40] {pt2=root:p_blocks_4_0_se_conv_reduce_bias target=blocks.4.0.se.conv_reduce.bias} ->[n280] constant,
     t104 f32 [D=144 H=40 W=1 C=1] {pt2=root:p_blocks_4_0_se_conv_expand_weight target=blocks.4.0.se.conv_expand.weight} ->[n284] constant,
     t105 f32 [C=144] {pt2=root:p_blocks_4_0_se_conv_expand_bias target=blocks.4.0.se.conv_expand.bias} ->[n285] constant,
     t106 f32 [D=48 H=144 W=1 C=1] {pt2=root:p_blocks_4_0_conv_pwl_weight target=blocks.4.0.conv_pwl.weight} ->[n290] constant,
     t107 f32 [C=48] {pt2=root:p_blocks_4_0_bn3_weight target=blocks.4.0.bn3.weight} ->[n294] constant,
     t108 f32 [C=48] {pt2=root:p_blocks_4_0_bn3_bias target=blocks.4.0.bn3.bias} ->[n294] constant,
     t109 f32 [D=288 H=48 W=1 C=1] {pt2=root:p_blocks_4_1_conv_pw_weight target=blocks.4.1.conv_pw.weight} ->[n297] constant,
     t110 f32 [C=288] {pt2=root:p_blocks_4_1_bn1_weight target=blocks.4.1.bn1.weight} ->[n301] constant,
     t111 f32 [C=288] {pt2=root:p_blocks_4_1_bn1_bias target=blocks.4.1.bn1.bias} ->[n301] constant,
     t112 f32 [D=288 H=1 W=5 C=5] {pt2=root:p_blocks_4_1_conv_dw_weight target=blocks.4.1.conv_dw.weight} ->[n305] constant,
     t113 f32 [C=288] {pt2=root:p_blocks_4_1_bn2_weight target=blocks.4.1.bn2.weight} ->[n309] constant,
     t114 f32 [C=288] {pt2=root:p_blocks_4_1_bn2_bias target=blocks.4.1.bn2.bias} ->[n309] constant,
     t115 f32 [D=72 H=288 W=1 C=1] {pt2=root:p_blocks_4_1_se_conv_reduce_weight target=blocks.4.1.se.conv_reduce.weight} ->[n314] constant,
     t116 f32 [C=72] {pt2=root:p_blocks_4_1_se_conv_reduce_bias target=blocks.4.1.se.conv_reduce.bias} ->[n315] constant,
     t117 f32 [D=288 H=72 W=1 C=1] {pt2=root:p_blocks_4_1_se_conv_expand_weight target=blocks.4.1.se.conv_expand.weight} ->[n319] constant,
     t118 f32 [C=288] {pt2=root:p_blocks_4_1_se_conv_expand_bias target=blocks.4.1.se.conv_expand.bias} ->[n320] constant,
     t119 f32 [D=48 H=288 W=1 C=1] {pt2=root:p_blocks_4_1_conv_pwl_weight target=blocks.4.1.conv_pwl.weight} ->[n325] constant,
     t120 f32 [C=48] {pt2=root:p_blocks_4_1_bn3_weight target=blocks.4.1.bn3.weight} ->[n329] constant,
     t121 f32 [C=48] {pt2=root:p_blocks_4_1_bn3_bias target=blocks.4.1.bn3.bias} ->[n329] constant,
     t122 f32 [D=288 H=48 W=1 C=1] {pt2=root:p_blocks_4_2_conv_pw_weight target=blocks.4.2.conv_pw.weight} ->[n333] constant,
     t123 f32 [C=288] {pt2=root:p_blocks_4_2_bn1_weight target=blocks.4.2.bn1.weight} ->[n337] constant,
     t124 f32 [C=288] {pt2=root:p_blocks_4_2_bn1_bias target=blocks.4.2.bn1.bias} ->[n337] constant,
     t125 f32 [D=288 H=1 W=5 C=5] {pt2=root:p_blocks_4_2_conv_dw_weight target=blocks.4.2.conv_dw.weight} ->[n341] constant,
     t126 f32 [C=288] {pt2=root:p_blocks_4_2_bn2_weight target=blocks.4.2.bn2.weight} ->[n345] constant,
     t127 f32 [C=288] {pt2=root:p_blocks_4_2_bn2_bias target=blocks.4.2.bn2.bias} ->[n345] constant,
     t128 f32 [D=72 H=288 W=1 C=1] {pt2=root:p_blocks_4_2_se_conv_reduce_weight target=blocks.4.2.se.conv_reduce.weight} ->[n350] constant,
     t129 f32 [C=72] {pt2=root:p_blocks_4_2_se_conv_reduce_bias target=blocks.4.2.se.conv_reduce.bias} ->[n351] constant,
     t130 f32 [D=288 H=72 W=1 C=1] {pt2=root:p_blocks_4_2_se_conv_expand_weight target=blocks.4.2.se.conv_expand.weight} ->[n355] constant,
     t131 f32 [C=288] {pt2=root:p_blocks_4_2_se_conv_expand_bias target=blocks.4.2.se.conv_expand.bias} ->[n356] constant,
     t132 f32 [D=48 H=288 W=1 C=1] {pt2=root:p_blocks_4_2_conv_pwl_weight target=blocks.4.2.conv_pwl.weight} ->[n361] constant,
     t133 f32 [C=48] {pt2=root:p_blocks_4_2_bn3_weight target=blocks.4.2.bn3.weight} ->[n365] constant,
     t134 f32 [C=48] {pt2=root:p_blocks_4_2_bn3_bias target=blocks.4.2.bn3.bias} ->[n365] constant,
     t135 f32 [D=288 H=48 W=1 C=1] {pt2=root:p_blocks_5_0_conv_weight target=blocks.5.0.conv.weight} ->[n369] constant,
     t136 f32 [C=288] {pt2=root:p_blocks_5_0_bn1_weight target=blocks.5.0.bn1.weight} ->[n373] constant,
     t137 f32 [C=288] {pt2=root:p_blocks_5_0_bn1_bias target=blocks.5.0.bn1.bias} ->[n373] constant,
     t138 f32 [D=1024 H=288 W=1 C=1] {pt2=root:p_conv_head_weight target=conv_head.weight} ->[n380] constant,
     t139 f32 [C=1024] {pt2=root:p_conv_head_bias target=conv_head.bias} ->[n381] constant,
     t140 f32 [W=1000 C=1024] {pt2=root:p_classifier_weight target=classifier.weight} ->[n385] constant,
     t141 f32 [C=1000] {pt2=root:p_classifier_bias target=classifier.bias} ->[n386] constant,
     t142 f32 [C=16] {pt2=root:b_bn1_running_mean target=bn1.running_mean} ->[n5] constant,
     t143 f32 [C=16] {pt2=root:b_bn1_running_var target=bn1.running_var} ->[n5] constant,
     t144 i64 [C=1] {pt2=root:b_bn1_num_batches_tracked target=bn1.num_batches_tracked} constant,
     t145 f32 [C=16] {pt2=root:b_blocks_0_0_bn1_running_mean target=blocks.0.0.bn1.running_mean} ->[n13] constant,
     t146 f32 [C=16] {pt2=root:b_blocks_0_0_bn1_running_var target=blocks.0.0.bn1.running_var} ->[n13] constant,
     t147 i64 [C=1] {pt2=root:b_blocks_0_0_bn1_num_batches_tracked target=blocks.0.0.bn1.num_batches_tracked} constant,
     t148 f32 [C=8] {pt2=root:b_blocks_0_0_bn2_running_mean target=blocks.0.0.bn2.running_mean} ->[n33] constant,
     t149 f32 [C=8] {pt2=root:b_blocks_0_0_bn2_running_var target=blocks.0.0.bn2.running_var} ->[n33] constant,
     t150 i64 [C=1] {pt2=root:b_blocks_0_0_bn2_num_batches_tracked target=blocks.0.0.bn2.num_batches_tracked} constant,
     t151 f32 [C=40] {pt2=root:b_blocks_1_0_bn1_running_mean target=blocks.1.0.bn1.running_mean} ->[n40] constant,
     t152 f32 [C=40] {pt2=root:b_blocks_1_0_bn1_running_var target=blocks.1.0.bn1.running_var} ->[n40] constant,
     t153 i64 [C=1] {pt2=root:b_blocks_1_0_bn1_num_batches_tracked target=blocks.1.0.bn1.num_batches_tracked} constant,
     t154 f32 [C=40] {pt2=root:b_blocks_1_0_bn2_running_mean target=blocks.1.0.bn2.running_mean} ->[n48] constant,
     t155 f32 [C=40] {pt2=root:b_blocks_1_0_bn2_running_var target=blocks.1.0.bn2.running_var} ->[n48] constant,
     t156 i64 [C=1] {pt2=root:b_blocks_1_0_bn2_num_batches_tracked target=blocks.1.0.bn2.num_batches_tracked} constant,
     t157 f32 [C=16] {pt2=root:b_blocks_1_0_bn3_running_mean target=blocks.1.0.bn3.running_mean} ->[n56] constant,
     t158 f32 [C=16] {pt2=root:b_blocks_1_0_bn3_running_var target=blocks.1.0.bn3.running_var} ->[n56] constant,
     t159 i64 [C=1] {pt2=root:b_blocks_1_0_bn3_num_batches_tracked target=blocks.1.0.bn3.num_batches_tracked} constant,
     t160 f32 [C=56] {pt2=root:b_blocks_1_1_bn1_running_mean target=blocks.1.1.bn1.running_mean} ->[n63] constant,
     t161 f32 [C=56] {pt2=root:b_blocks_1_1_bn1_running_var target=blocks.1.1.bn1.running_var} ->[n63] constant,
     t162 i64 [C=1] {pt2=root:b_blocks_1_1_bn1_num_batches_tracked target=blocks.1.1.bn1.num_batches_tracked} constant,
     t163 f32 [C=56] {pt2=root:b_blocks_1_1_bn2_running_mean target=blocks.1.1.bn2.running_mean} ->[n71] constant,
     t164 f32 [C=56] {pt2=root:b_blocks_1_1_bn2_running_var target=blocks.1.1.bn2.running_var} ->[n71] constant,
     t165 i64 [C=1] {pt2=root:b_blocks_1_1_bn2_num_batches_tracked target=blocks.1.1.bn2.num_batches_tracked} constant,
     t166 f32 [C=16] {pt2=root:b_blocks_1_1_bn3_running_mean target=blocks.1.1.bn3.running_mean} ->[n79] constant,
     t167 f32 [C=16] {pt2=root:b_blocks_1_1_bn3_running_var target=blocks.1.1.bn3.running_var} ->[n79] constant,
     t168 i64 [C=1] {pt2=root:b_blocks_1_1_bn3_num_batches_tracked target=blocks.1.1.bn3.num_batches_tracked} constant,
     t169 f32 [C=64] {pt2=root:b_blocks_2_0_bn1_running_mean target=blocks.2.0.bn1.running_mean} ->[n87] constant,
     t170 f32 [C=64] {pt2=root:b_blocks_2_0_bn1_running_var target=blocks.2.0.bn1.running_var} ->[n87] constant,
     t171 i64 [C=1] {pt2=root:b_blocks_2_0_bn1_num_batches_tracked target=blocks.2.0.bn1.num_batches_tracked} constant,
     t172 f32 [C=64] {pt2=root:b_blocks_2_0_bn2_running_mean target=blocks.2.0.bn2.running_mean} ->[n95] constant,
     t173 f32 [C=64] {pt2=root:b_blocks_2_0_bn2_running_var target=blocks.2.0.bn2.running_var} ->[n95] constant,
     t174 i64 [C=1] {pt2=root:b_blocks_2_0_bn2_num_batches_tracked target=blocks.2.0.bn2.num_batches_tracked} constant,
     t175 f32 [C=24] {pt2=root:b_blocks_2_0_bn3_running_mean target=blocks.2.0.bn3.running_mean} ->[n115] constant,
     t176 f32 [C=24] {pt2=root:b_blocks_2_0_bn3_running_var target=blocks.2.0.bn3.running_var} ->[n115] constant,
     t177 i64 [C=1] {pt2=root:b_blocks_2_0_bn3_num_batches_tracked target=blocks.2.0.bn3.num_batches_tracked} constant,
     t178 f32 [C=144] {pt2=root:b_blocks_2_1_bn1_running_mean target=blocks.2.1.bn1.running_mean} ->[n122] constant,
     t179 f32 [C=144] {pt2=root:b_blocks_2_1_bn1_running_var target=blocks.2.1.bn1.running_var} ->[n122] constant,
     t180 i64 [C=1] {pt2=root:b_blocks_2_1_bn1_num_batches_tracked target=blocks.2.1.bn1.num_batches_tracked} constant,
     t181 f32 [C=144] {pt2=root:b_blocks_2_1_bn2_running_mean target=blocks.2.1.bn2.running_mean} ->[n130] constant,
     t182 f32 [C=144] {pt2=root:b_blocks_2_1_bn2_running_var target=blocks.2.1.bn2.running_var} ->[n130] constant,
     t183 i64 [C=1] {pt2=root:b_blocks_2_1_bn2_num_batches_tracked target=blocks.2.1.bn2.num_batches_tracked} constant,
     t184 f32 [C=24] {pt2=root:b_blocks_2_1_bn3_running_mean target=blocks.2.1.bn3.running_mean} ->[n150] constant,
     t185 f32 [C=24] {pt2=root:b_blocks_2_1_bn3_running_var target=blocks.2.1.bn3.running_var} ->[n150] constant,
     t186 i64 [C=1] {pt2=root:b_blocks_2_1_bn3_num_batches_tracked target=blocks.2.1.bn3.num_batches_tracked} constant,
     t187 f32 [C=144] {pt2=root:b_blocks_2_2_bn1_running_mean target=blocks.2.2.bn1.running_mean} ->[n158] constant,
     t188 f32 [C=144] {pt2=root:b_blocks_2_2_bn1_running_var target=blocks.2.2.bn1.running_var} ->[n158] constant,
     t189 i64 [C=1] {pt2=root:b_blocks_2_2_bn1_num_batches_tracked target=blocks.2.2.bn1.num_batches_tracked} constant,
     t190 f32 [C=144] {pt2=root:b_blocks_2_2_bn2_running_mean target=blocks.2.2.bn2.running_mean} ->[n166] constant,
     t191 f32 [C=144] {pt2=root:b_blocks_2_2_bn2_running_var target=blocks.2.2.bn2.running_var} ->[n166] constant,
     t192 i64 [C=1] {pt2=root:b_blocks_2_2_bn2_num_batches_tracked target=blocks.2.2.bn2.num_batches_tracked} constant,
     t193 f32 [C=24] {pt2=root:b_blocks_2_2_bn3_running_mean target=blocks.2.2.bn3.running_mean} ->[n186] constant,
     t194 f32 [C=24] {pt2=root:b_blocks_2_2_bn3_running_var target=blocks.2.2.bn3.running_var} ->[n186] constant,
     t195 i64 [C=1] {pt2=root:b_blocks_2_2_bn3_num_batches_tracked target=blocks.2.2.bn3.num_batches_tracked} constant,
     t196 f32 [C=72] {pt2=root:b_blocks_3_0_bn1_running_mean target=blocks.3.0.bn1.running_mean} ->[n194] constant,
     t197 f32 [C=72] {pt2=root:b_blocks_3_0_bn1_running_var target=blocks.3.0.bn1.running_var} ->[n194] constant,
     t198 i64 [C=1] {pt2=root:b_blocks_3_0_bn1_num_batches_tracked target=blocks.3.0.bn1.num_batches_tracked} constant,
     t199 f32 [C=72] {pt2=root:b_blocks_3_0_bn2_running_mean target=blocks.3.0.bn2.running_mean} ->[n202] constant,
     t200 f32 [C=72] {pt2=root:b_blocks_3_0_bn2_running_var target=blocks.3.0.bn2.running_var} ->[n202] constant,
     t201 i64 [C=1] {pt2=root:b_blocks_3_0_bn2_num_batches_tracked target=blocks.3.0.bn2.num_batches_tracked} constant,
     t202 f32 [C=24] {pt2=root:b_blocks_3_0_bn3_running_mean target=blocks.3.0.bn3.running_mean} ->[n222] constant,
     t203 f32 [C=24] {pt2=root:b_blocks_3_0_bn3_running_var target=blocks.3.0.bn3.running_var} ->[n222] constant,
     t204 i64 [C=1] {pt2=root:b_blocks_3_0_bn3_num_batches_tracked target=blocks.3.0.bn3.num_batches_tracked} constant,
     t205 f32 [C=72] {pt2=root:b_blocks_3_1_bn1_running_mean target=blocks.3.1.bn1.running_mean} ->[n230] constant,
     t206 f32 [C=72] {pt2=root:b_blocks_3_1_bn1_running_var target=blocks.3.1.bn1.running_var} ->[n230] constant,
     t207 i64 [C=1] {pt2=root:b_blocks_3_1_bn1_num_batches_tracked target=blocks.3.1.bn1.num_batches_tracked} constant,
     t208 f32 [C=72] {pt2=root:b_blocks_3_1_bn2_running_mean target=blocks.3.1.bn2.running_mean} ->[n238] constant,
     t209 f32 [C=72] {pt2=root:b_blocks_3_1_bn2_running_var target=blocks.3.1.bn2.running_var} ->[n238] constant,
     t210 i64 [C=1] {pt2=root:b_blocks_3_1_bn2_num_batches_tracked target=blocks.3.1.bn2.num_batches_tracked} constant,
     t211 f32 [C=24] {pt2=root:b_blocks_3_1_bn3_running_mean target=blocks.3.1.bn3.running_mean} ->[n258] constant,
     t212 f32 [C=24] {pt2=root:b_blocks_3_1_bn3_running_var target=blocks.3.1.bn3.running_var} ->[n258] constant,
     t213 i64 [C=1] {pt2=root:b_blocks_3_1_bn3_num_batches_tracked target=blocks.3.1.bn3.num_batches_tracked} constant,
     t214 f32 [C=144] {pt2=root:b_blocks_4_0_bn1_running_mean target=blocks.4.0.bn1.running_mean} ->[n266] constant,
     t215 f32 [C=144] {pt2=root:b_blocks_4_0_bn1_running_var target=blocks.4.0.bn1.running_var} ->[n266] constant,
     t216 i64 [C=1] {pt2=root:b_blocks_4_0_bn1_num_batches_tracked target=blocks.4.0.bn1.num_batches_tracked} constant,
     t217 f32 [C=144] {pt2=root:b_blocks_4_0_bn2_running_mean target=blocks.4.0.bn2.running_mean} ->[n274] constant,
     t218 f32 [C=144] {pt2=root:b_blocks_4_0_bn2_running_var target=blocks.4.0.bn2.running_var} ->[n274] constant,
     t219 i64 [C=1] {pt2=root:b_blocks_4_0_bn2_num_batches_tracked target=blocks.4.0.bn2.num_batches_tracked} constant,
     t220 f32 [C=48] {pt2=root:b_blocks_4_0_bn3_running_mean target=blocks.4.0.bn3.running_mean} ->[n294] constant,
     t221 f32 [C=48] {pt2=root:b_blocks_4_0_bn3_running_var target=blocks.4.0.bn3.running_var} ->[n294] constant,
     t222 i64 [C=1] {pt2=root:b_blocks_4_0_bn3_num_batches_tracked target=blocks.4.0.bn3.num_batches_tracked} constant,
     t223 f32 [C=288] {pt2=root:b_blocks_4_1_bn1_running_mean target=blocks.4.1.bn1.running_mean} ->[n301] constant,
     t224 f32 [C=288] {pt2=root:b_blocks_4_1_bn1_running_var target=blocks.4.1.bn1.running_var} ->[n301] constant,
     t225 i64 [C=1] {pt2=root:b_blocks_4_1_bn1_num_batches_tracked target=blocks.4.1.bn1.num_batches_tracked} constant,
     t226 f32 [C=288] {pt2=root:b_blocks_4_1_bn2_running_mean target=blocks.4.1.bn2.running_mean} ->[n309] constant,
     t227 f32 [C=288] {pt2=root:b_blocks_4_1_bn2_running_var target=blocks.4.1.bn2.running_var} ->[n309] constant,
     t228 i64 [C=1] {pt2=root:b_blocks_4_1_bn2_num_batches_tracked target=blocks.4.1.bn2.num_batches_tracked} constant,
     t229 f32 [C=48] {pt2=root:b_blocks_4_1_bn3_running_mean target=blocks.4.1.bn3.running_mean} ->[n329] constant,
     t230 f32 [C=48] {pt2=root:b_blocks_4_1_bn3_running_var target=blocks.4.1.bn3.running_var} ->[n329] constant,
     t231 i64 [C=1] {pt2=root:b_blocks_4_1_bn3_num_batches_tracked target=blocks.4.1.bn3.num_batches_tracked} constant,
     t232 f32 [C=288] {pt2=root:b_blocks_4_2_bn1_running_mean target=blocks.4.2.bn1.running_mean} ->[n337] constant,
     t233 f32 [C=288] {pt2=root:b_blocks_4_2_bn1_running_var target=blocks.4.2.bn1.running_var} ->[n337] constant,
     t234 i64 [C=1] {pt2=root:b_blocks_4_2_bn1_num_batches_tracked target=blocks.4.2.bn1.num_batches_tracked} constant,
     t235 f32 [C=288] {pt2=root:b_blocks_4_2_bn2_running_mean target=blocks.4.2.bn2.running_mean} ->[n345] constant,
     t236 f32 [C=288] {pt2=root:b_blocks_4_2_bn2_running_var target=blocks.4.2.bn2.running_var} ->[n345] constant,
     t237 i64 [C=1] {pt2=root:b_blocks_4_2_bn2_num_batches_tracked target=blocks.4.2.bn2.num_batches_tracked} constant,
     t238 f32 [C=48] {pt2=root:b_blocks_4_2_bn3_running_mean target=blocks.4.2.bn3.running_mean} ->[n365] constant,
     t239 f32 [C=48] {pt2=root:b_blocks_4_2_bn3_running_var target=blocks.4.2.bn3.running_var} ->[n365] constant,
     t240 i64 [C=1] {pt2=root:b_blocks_4_2_bn3_num_batches_tracked target=blocks.4.2.bn3.num_batches_tracked} constant,
     t241 f32 [C=288] {pt2=root:b_blocks_5_0_bn1_running_mean target=blocks.5.0.bn1.running_mean} ->[n373] constant,
     t242 f32 [C=288] {pt2=root:b_blocks_5_0_bn1_running_var target=blocks.5.0.bn1.running_var} ->[n373] constant,
     t243 i64 [C=1] {pt2=root:b_blocks_5_0_bn1_num_batches_tracked target=blocks.5.0.bn1.num_batches_tracked} constant,
     t244 f32 [H=3 W=224 C=224] {pt2=root:x} ->[n0]]
  nodes:
    group g1 torch.ops.aten.conv2d.default:
      n0 {derived}: [t245 f32 [H=224 W=224 C=3] {derived} ->[n2]] =
        permute x=t244 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
      n1 {derived}: [t246 f32 [N=16 T=1 D=1 H=3 W=3 C=3] {derived} ->[n2]] =
        permute
          x=t0 {pt2=root:p_conv_stem_weight target=conv_stem.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2 {derived}: [t247 f32 [H=112 W=112 C=16] {derived} ->[n3]] =
        conv2d
          x=t245 {derived} <-n0
          weight=t246 {derived} <-n1
          bias=none
          params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 in_channels=3;
                 groups=1}
      n3 {pt2=root[0] torch.ops.aten.conv2d.default (conv2d)}: [t248 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:conv2d} ->[n4]] =
        permute x=t247 {derived} <-n2 perm=[H<-C, W<-H, C<-W]
    group g2 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n4 {derived}: [t249 f32 [H=112 W=112 C=16] {derived} ->[n5]] =
        permute x=t248 {pt2=root:conv2d} <-n3 perm=[H<-W, W<-C, C<-H]
      n5 {derived}: [t250 f32 [H=112 W=112 C=16] {derived} ->[n6]] =
        batch_norm
          x=t249 {derived} <-n4
          weight=t1 {pt2=root:p_bn1_weight target=bn1.weight}
          bias=t2 {pt2=root:p_bn1_bias target=bn1.bias}
          running_mean=t142 {pt2=root:b_bn1_running_mean target=bn1.running_mean}
          running_var=t143 {pt2=root:b_bn1_running_var target=bn1.running_var}
          params={channel=C; eps=1e-05}
      n6 {pt2=root[1] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training)}: [t251 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:getitem} ->[n7]] =
        permute x=t250 {derived} <-n5 perm=[H<-C, W<-H, C<-W]
    n7 {pt2=root[2] torch.ops.aten.hardswish.default (hardswish)}: [t252 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:hardswish} ->[n8]] =
      hardswish x=t251 {pt2=root:getitem} <-n6
    group g3 torch.ops.aten.conv2d.default:
      n8 {derived}: [t253 f32 [H=112 W=112 C=16] {derived} ->[n10]] =
        permute x=t252 {pt2=root:hardswish} <-n7 perm=[H<-W, W<-C, C<-H]
      n9 {derived}: [t254 f32 [N=16 T=1 D=1 H=3 W=3 C=1] {derived} ->[n10]] =
        permute
          x=t3 {pt2=root:p_blocks_0_0_conv_dw_weight target=blocks.0.0.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n10 {derived}: [t255 f32 [H=56 W=56 C=16] {derived} ->[n11]] =
        conv2d
          x=t253 {derived} <-n8
          weight=t254 {derived} <-n9
          bias=none
          params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 in_channels=16;
                 groups=16}
      n11 {pt2=root[3] torch.ops.aten.conv2d.default (conv2d_1)}: [t256 f32 [H=16
                                                                      W=56
                                                                      C=56] {pt2=root:conv2d_1} ->[n12]] =
        permute x=t255 {derived} <-n10 perm=[H<-C, W<-H, C<-W]
    group g4 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n12 {derived}: [t257 f32 [H=56 W=56 C=16] {derived} ->[n13]] =
        permute x=t256 {pt2=root:conv2d_1} <-n11 perm=[H<-W, W<-C, C<-H]
      n13 {derived}: [t258 f32 [H=56 W=56 C=16] {derived} ->[n14]] =
        batch_norm
          x=t257 {derived} <-n12
          weight=t4 {pt2=root:p_blocks_0_0_bn1_weight target=blocks.0.0.bn1.weight}
          bias=t5 {pt2=root:p_blocks_0_0_bn1_bias target=blocks.0.0.bn1.bias}
          running_mean=t145 {pt2=root:b_blocks_0_0_bn1_running_mean target=blocks.0.0.bn1.running_mean}
          running_var=t146 {pt2=root:b_blocks_0_0_bn1_running_var target=blocks.0.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n14 {pt2=root[4] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_1)}: [t259 f32 [H=16
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_3} ->[n15]] =
        permute x=t258 {derived} <-n13 perm=[H<-C, W<-H, C<-W]
    n15 {pt2=root[5] torch.ops.aten.relu.default (relu)}: [t260 f32 [H=16 W=56
                                                                     C=56] {pt2=root:relu} ->[n16,
                                                                      n27]] =
      relu x=t259 {pt2=root:getitem_3} <-n14
    n16 {pt2=root[6] torch.ops.aten.mean.dim (mean)}: [t261 f32 [H=16 W=1 C=1] {pt2=root:mean} ->[n17]] =
      mean x=t260 {pt2=root:relu} <-n15 params={dims=[W, C]; keepdim=true}
    group g5 torch.ops.aten.conv2d.default:
      n17 {derived}: [t262 f32 [C=16] {derived} ->[n19]] =
        permute x=t261 {pt2=root:mean} <-n16 perm=[H<-W, W<-C, C<-H]
      n18 {derived}: [t263 f32 [N=8 T=1 D=1 H=1 W=1 C=16] {derived} ->[n19]] =
        permute
          x=t6 {pt2=root:p_blocks_0_0_se_conv_reduce_weight target=blocks.0.0.se.conv_reduce.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n19 {derived}: [t264 f32 [C=8] {derived} ->[n20]] =
        conv2d
          x=t262 {derived} <-n17
          weight=t263 {derived} <-n18
          bias=t7 {pt2=root:p_blocks_0_0_se_conv_reduce_bias target=blocks.0.0.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=16;
                 groups=1}
      n20 {pt2=root[7] torch.ops.aten.conv2d.default (conv2d_2)}: [t265 f32 [H=8
                                                                      W=1 C=1] {pt2=root:conv2d_2} ->[n21]] =
        permute x=t264 {derived} <-n19 perm=[H<-C, W<-H, C<-W]
    n21 {pt2=root[8] torch.ops.aten.relu.default (relu_1)}: [t266 f32 [H=8 W=1
                                                                      C=1] {pt2=root:relu_1} ->[n22]] =
      relu x=t265 {pt2=root:conv2d_2} <-n20
    group g6 torch.ops.aten.conv2d.default:
      n22 {derived}: [t267 f32 [C=8] {derived} ->[n24]] =
        permute x=t266 {pt2=root:relu_1} <-n21 perm=[H<-W, W<-C, C<-H]
      n23 {derived}: [t268 f32 [N=16 T=1 D=1 H=1 W=1 C=8] {derived} ->[n24]] =
        permute
          x=t8 {pt2=root:p_blocks_0_0_se_conv_expand_weight target=blocks.0.0.se.conv_expand.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n24 {derived}: [t269 f32 [C=16] {derived} ->[n25]] =
        conv2d
          x=t267 {derived} <-n22
          weight=t268 {derived} <-n23
          bias=t9 {pt2=root:p_blocks_0_0_se_conv_expand_bias target=blocks.0.0.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=8;
                 groups=1}
      n25 {pt2=root[9] torch.ops.aten.conv2d.default (conv2d_3)}: [t270 f32 [H=16
                                                                      W=1 C=1] {pt2=root:conv2d_3} ->[n26]] =
        permute x=t269 {derived} <-n24 perm=[H<-C, W<-H, C<-W]
    n26 {pt2=root[10] torch.ops.aten.hardsigmoid.default (hardsigmoid)}: [t271 f32 [H=16
                                                                      W=1 C=1] {pt2=root:hardsigmoid} ->[n27]] =
      hardsigmoid x=t270 {pt2=root:conv2d_3} <-n25
    n27 {pt2=root[11] torch.ops.aten.mul.Tensor (mul)}: [t272 f32 [H=16 W=56
                                                                   C=56] {pt2=root:mul} ->[n28]] =
      mul a=t260 {pt2=root:relu} <-n15 b=t271 {pt2=root:hardsigmoid} <-n26
    group g7 torch.ops.aten.conv2d.default:
      n28 {derived}: [t273 f32 [H=56 W=56 C=16] {derived} ->[n30]] =
        permute x=t272 {pt2=root:mul} <-n27 perm=[H<-W, W<-C, C<-H]
      n29 {derived}: [t274 f32 [N=8 T=1 D=1 H=1 W=1 C=16] {derived} ->[n30]] =
        permute
          x=t10 {pt2=root:p_blocks_0_0_conv_pw_weight target=blocks.0.0.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n30 {derived}: [t275 f32 [H=56 W=56 C=8] {derived} ->[n31]] =
        conv2d
          x=t273 {derived} <-n28
          weight=t274 {derived} <-n29
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=16;
                 groups=1}
      n31 {pt2=root[12] torch.ops.aten.conv2d.default (conv2d_4)}: [t276 f32 [H=8
                                                                      W=56
                                                                      C=56] {pt2=root:conv2d_4} ->[n32]] =
        permute x=t275 {derived} <-n30 perm=[H<-C, W<-H, C<-W]
    group g8 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n32 {derived}: [t277 f32 [H=56 W=56 C=8] {derived} ->[n33]] =
        permute x=t276 {pt2=root:conv2d_4} <-n31 perm=[H<-W, W<-C, C<-H]
      n33 {derived}: [t278 f32 [H=56 W=56 C=8] {derived} ->[n34]] =
        batch_norm
          x=t277 {derived} <-n32
          weight=t11 {pt2=root:p_blocks_0_0_bn2_weight target=blocks.0.0.bn2.weight}
          bias=t12 {pt2=root:p_blocks_0_0_bn2_bias target=blocks.0.0.bn2.bias}
          running_mean=t148 {pt2=root:b_blocks_0_0_bn2_running_mean target=blocks.0.0.bn2.running_mean}
          running_var=t149 {pt2=root:b_blocks_0_0_bn2_running_var target=blocks.0.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n34 {pt2=root[13] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_2)}: [t279 f32 [H=8
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_6} ->[n35]] =
        permute x=t278 {derived} <-n33 perm=[H<-C, W<-H, C<-W]
    group g9 torch.ops.aten.conv2d.default:
      n35 {derived}: [t280 f32 [H=56 W=56 C=8] {derived} ->[n37]] =
        permute x=t279 {pt2=root:getitem_6} <-n34 perm=[H<-W, W<-C, C<-H]
      n36 {derived}: [t281 f32 [N=40 T=1 D=1 H=1 W=1 C=8] {derived} ->[n37]] =
        permute
          x=t13 {pt2=root:p_blocks_1_0_conv_pw_weight target=blocks.1.0.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n37 {derived}: [t282 f32 [H=56 W=56 C=40] {derived} ->[n38]] =
        conv2d
          x=t280 {derived} <-n35
          weight=t281 {derived} <-n36
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=8;
                 groups=1}
      n38 {pt2=root[14] torch.ops.aten.conv2d.default (conv2d_5)}: [t283 f32 [H=40
                                                                      W=56
                                                                      C=56] {pt2=root:conv2d_5} ->[n39]] =
        permute x=t282 {derived} <-n37 perm=[H<-C, W<-H, C<-W]
    group g10 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n39 {derived}: [t284 f32 [H=56 W=56 C=40] {derived} ->[n40]] =
        permute x=t283 {pt2=root:conv2d_5} <-n38 perm=[H<-W, W<-C, C<-H]
      n40 {derived}: [t285 f32 [H=56 W=56 C=40] {derived} ->[n41]] =
        batch_norm
          x=t284 {derived} <-n39
          weight=t14 {pt2=root:p_blocks_1_0_bn1_weight target=blocks.1.0.bn1.weight}
          bias=t15 {pt2=root:p_blocks_1_0_bn1_bias target=blocks.1.0.bn1.bias}
          running_mean=t151 {pt2=root:b_blocks_1_0_bn1_running_mean target=blocks.1.0.bn1.running_mean}
          running_var=t152 {pt2=root:b_blocks_1_0_bn1_running_var target=blocks.1.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n41 {pt2=root[15] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_3)}: [t286 f32 [H=40
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_9} ->[n42]] =
        permute x=t285 {derived} <-n40 perm=[H<-C, W<-H, C<-W]
    n42 {pt2=root[16] torch.ops.aten.relu.default (relu_2)}: [t287 f32 [H=40
                                                                      W=56
                                                                      C=56] {pt2=root:relu_2} ->[n43]] =
      relu x=t286 {pt2=root:getitem_9} <-n41
    group g11 torch.ops.aten.conv2d.default:
      n43 {derived}: [t288 f32 [H=56 W=56 C=40] {derived} ->[n45]] =
        permute x=t287 {pt2=root:relu_2} <-n42 perm=[H<-W, W<-C, C<-H]
      n44 {derived}: [t289 f32 [N=40 T=1 D=1 H=3 W=3 C=1] {derived} ->[n45]] =
        permute
          x=t16 {pt2=root:p_blocks_1_0_conv_dw_weight target=blocks.1.0.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n45 {derived}: [t290 f32 [H=28 W=28 C=40] {derived} ->[n46]] =
        conv2d
          x=t288 {derived} <-n43
          weight=t289 {derived} <-n44
          bias=none
          params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 in_channels=40;
                 groups=40}
      n46 {pt2=root[17] torch.ops.aten.conv2d.default (conv2d_6)}: [t291 f32 [H=40
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_6} ->[n47]] =
        permute x=t290 {derived} <-n45 perm=[H<-C, W<-H, C<-W]
    group g12 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n47 {derived}: [t292 f32 [H=28 W=28 C=40] {derived} ->[n48]] =
        permute x=t291 {pt2=root:conv2d_6} <-n46 perm=[H<-W, W<-C, C<-H]
      n48 {derived}: [t293 f32 [H=28 W=28 C=40] {derived} ->[n49]] =
        batch_norm
          x=t292 {derived} <-n47
          weight=t17 {pt2=root:p_blocks_1_0_bn2_weight target=blocks.1.0.bn2.weight}
          bias=t18 {pt2=root:p_blocks_1_0_bn2_bias target=blocks.1.0.bn2.bias}
          running_mean=t154 {pt2=root:b_blocks_1_0_bn2_running_mean target=blocks.1.0.bn2.running_mean}
          running_var=t155 {pt2=root:b_blocks_1_0_bn2_running_var target=blocks.1.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n49 {pt2=root[18] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_4)}: [t294 f32 [H=40
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_12} ->[n50]] =
        permute x=t293 {derived} <-n48 perm=[H<-C, W<-H, C<-W]
    n50 {pt2=root[19] torch.ops.aten.relu.default (relu_3)}: [t295 f32 [H=40
                                                                      W=28
                                                                      C=28] {pt2=root:relu_3} ->[n51]] =
      relu x=t294 {pt2=root:getitem_12} <-n49
    group g13 torch.ops.aten.conv2d.default:
      n51 {derived}: [t296 f32 [H=28 W=28 C=40] {derived} ->[n53]] =
        permute x=t295 {pt2=root:relu_3} <-n50 perm=[H<-W, W<-C, C<-H]
      n52 {derived}: [t297 f32 [N=16 T=1 D=1 H=1 W=1 C=40] {derived} ->[n53]] =
        permute
          x=t19 {pt2=root:p_blocks_1_0_conv_pwl_weight target=blocks.1.0.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n53 {derived}: [t298 f32 [H=28 W=28 C=16] {derived} ->[n54]] =
        conv2d
          x=t296 {derived} <-n51
          weight=t297 {derived} <-n52
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=40;
                 groups=1}
      n54 {pt2=root[20] torch.ops.aten.conv2d.default (conv2d_7)}: [t299 f32 [H=16
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_7} ->[n55]] =
        permute x=t298 {derived} <-n53 perm=[H<-C, W<-H, C<-W]
    group g14 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n55 {derived}: [t300 f32 [H=28 W=28 C=16] {derived} ->[n56]] =
        permute x=t299 {pt2=root:conv2d_7} <-n54 perm=[H<-W, W<-C, C<-H]
      n56 {derived}: [t301 f32 [H=28 W=28 C=16] {derived} ->[n57]] =
        batch_norm
          x=t300 {derived} <-n55
          weight=t20 {pt2=root:p_blocks_1_0_bn3_weight target=blocks.1.0.bn3.weight}
          bias=t21 {pt2=root:p_blocks_1_0_bn3_bias target=blocks.1.0.bn3.bias}
          running_mean=t157 {pt2=root:b_blocks_1_0_bn3_running_mean target=blocks.1.0.bn3.running_mean}
          running_var=t158 {pt2=root:b_blocks_1_0_bn3_running_var target=blocks.1.0.bn3.running_var}
          params={channel=C; eps=1e-05}
      n57 {pt2=root[21] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_5)}: [t302 f32 [H=16
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_15} ->[n58,
                                                                      n81]] =
        permute x=t301 {derived} <-n56 perm=[H<-C, W<-H, C<-W]
    group g15 torch.ops.aten.conv2d.default:
      n58 {derived}: [t303 f32 [H=28 W=28 C=16] {derived} ->[n60]] =
        permute x=t302 {pt2=root:getitem_15} <-n57 perm=[H<-W, W<-C, C<-H]
      n59 {derived}: [t304 f32 [N=56 T=1 D=1 H=1 W=1 C=16] {derived} ->[n60]] =
        permute
          x=t22 {pt2=root:p_blocks_1_1_conv_pw_weight target=blocks.1.1.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n60 {derived}: [t305 f32 [H=28 W=28 C=56] {derived} ->[n61]] =
        conv2d
          x=t303 {derived} <-n58
          weight=t304 {derived} <-n59
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=16;
                 groups=1}
      n61 {pt2=root[22] torch.ops.aten.conv2d.default (conv2d_8)}: [t306 f32 [H=56
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_8} ->[n62]] =
        permute x=t305 {derived} <-n60 perm=[H<-C, W<-H, C<-W]
    group g16 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n62 {derived}: [t307 f32 [H=28 W=28 C=56] {derived} ->[n63]] =
        permute x=t306 {pt2=root:conv2d_8} <-n61 perm=[H<-W, W<-C, C<-H]
      n63 {derived}: [t308 f32 [H=28 W=28 C=56] {derived} ->[n64]] =
        batch_norm
          x=t307 {derived} <-n62
          weight=t23 {pt2=root:p_blocks_1_1_bn1_weight target=blocks.1.1.bn1.weight}
          bias=t24 {pt2=root:p_blocks_1_1_bn1_bias target=blocks.1.1.bn1.bias}
          running_mean=t160 {pt2=root:b_blocks_1_1_bn1_running_mean target=blocks.1.1.bn1.running_mean}
          running_var=t161 {pt2=root:b_blocks_1_1_bn1_running_var target=blocks.1.1.bn1.running_var}
          params={channel=C; eps=1e-05}
      n64 {pt2=root[23] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_6)}: [t309 f32 [H=56
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_18} ->[n65]] =
        permute x=t308 {derived} <-n63 perm=[H<-C, W<-H, C<-W]
    n65 {pt2=root[24] torch.ops.aten.relu.default (relu_4)}: [t310 f32 [H=56
                                                                      W=28
                                                                      C=28] {pt2=root:relu_4} ->[n66]] =
      relu x=t309 {pt2=root:getitem_18} <-n64
    group g17 torch.ops.aten.conv2d.default:
      n66 {derived}: [t311 f32 [H=28 W=28 C=56] {derived} ->[n68]] =
        permute x=t310 {pt2=root:relu_4} <-n65 perm=[H<-W, W<-C, C<-H]
      n67 {derived}: [t312 f32 [N=56 T=1 D=1 H=3 W=3 C=1] {derived} ->[n68]] =
        permute
          x=t25 {pt2=root:p_blocks_1_1_conv_dw_weight target=blocks.1.1.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n68 {derived}: [t313 f32 [H=28 W=28 C=56] {derived} ->[n69]] =
        conv2d
          x=t311 {derived} <-n66
          weight=t312 {derived} <-n67
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=56;
                 groups=56}
      n69 {pt2=root[25] torch.ops.aten.conv2d.default (conv2d_9)}: [t314 f32 [H=56
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_9} ->[n70]] =
        permute x=t313 {derived} <-n68 perm=[H<-C, W<-H, C<-W]
    group g18 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n70 {derived}: [t315 f32 [H=28 W=28 C=56] {derived} ->[n71]] =
        permute x=t314 {pt2=root:conv2d_9} <-n69 perm=[H<-W, W<-C, C<-H]
      n71 {derived}: [t316 f32 [H=28 W=28 C=56] {derived} ->[n72]] =
        batch_norm
          x=t315 {derived} <-n70
          weight=t26 {pt2=root:p_blocks_1_1_bn2_weight target=blocks.1.1.bn2.weight}
          bias=t27 {pt2=root:p_blocks_1_1_bn2_bias target=blocks.1.1.bn2.bias}
          running_mean=t163 {pt2=root:b_blocks_1_1_bn2_running_mean target=blocks.1.1.bn2.running_mean}
          running_var=t164 {pt2=root:b_blocks_1_1_bn2_running_var target=blocks.1.1.bn2.running_var}
          params={channel=C; eps=1e-05}
      n72 {pt2=root[26] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_7)}: [t317 f32 [H=56
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_21} ->[n73]] =
        permute x=t316 {derived} <-n71 perm=[H<-C, W<-H, C<-W]
    n73 {pt2=root[27] torch.ops.aten.relu.default (relu_5)}: [t318 f32 [H=56
                                                                      W=28
                                                                      C=28] {pt2=root:relu_5} ->[n74]] =
      relu x=t317 {pt2=root:getitem_21} <-n72
    group g19 torch.ops.aten.conv2d.default:
      n74 {derived}: [t319 f32 [H=28 W=28 C=56] {derived} ->[n76]] =
        permute x=t318 {pt2=root:relu_5} <-n73 perm=[H<-W, W<-C, C<-H]
      n75 {derived}: [t320 f32 [N=16 T=1 D=1 H=1 W=1 C=56] {derived} ->[n76]] =
        permute
          x=t28 {pt2=root:p_blocks_1_1_conv_pwl_weight target=blocks.1.1.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n76 {derived}: [t321 f32 [H=28 W=28 C=16] {derived} ->[n77]] =
        conv2d
          x=t319 {derived} <-n74
          weight=t320 {derived} <-n75
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=56;
                 groups=1}
      n77 {pt2=root[28] torch.ops.aten.conv2d.default (conv2d_10)}: [t322 f32 [H=16
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_10} ->[n78]] =
        permute x=t321 {derived} <-n76 perm=[H<-C, W<-H, C<-W]
    group g20 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n78 {derived}: [t323 f32 [H=28 W=28 C=16] {derived} ->[n79]] =
        permute x=t322 {pt2=root:conv2d_10} <-n77 perm=[H<-W, W<-C, C<-H]
      n79 {derived}: [t324 f32 [H=28 W=28 C=16] {derived} ->[n80]] =
        batch_norm
          x=t323 {derived} <-n78
          weight=t29 {pt2=root:p_blocks_1_1_bn3_weight target=blocks.1.1.bn3.weight}
          bias=t30 {pt2=root:p_blocks_1_1_bn3_bias target=blocks.1.1.bn3.bias}
          running_mean=t166 {pt2=root:b_blocks_1_1_bn3_running_mean target=blocks.1.1.bn3.running_mean}
          running_var=t167 {pt2=root:b_blocks_1_1_bn3_running_var target=blocks.1.1.bn3.running_var}
          params={channel=C; eps=1e-05}
      n80 {pt2=root[29] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_8)}: [t325 f32 [H=16
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_24} ->[n81]] =
        permute x=t324 {derived} <-n79 perm=[H<-C, W<-H, C<-W]
    n81 {pt2=root[30] torch.ops.aten.add.Tensor (add)}: [t326 f32 [H=16 W=28
                                                                   C=28] {pt2=root:add} ->[n82]] =
      add a=t325 {pt2=root:getitem_24} <-n80 b=t302 {pt2=root:getitem_15} <-n57
    group g21 torch.ops.aten.conv2d.default:
      n82 {derived}: [t327 f32 [H=28 W=28 C=16] {derived} ->[n84]] =
        permute x=t326 {pt2=root:add} <-n81 perm=[H<-W, W<-C, C<-H]
      n83 {derived}: [t328 f32 [N=64 T=1 D=1 H=1 W=1 C=16] {derived} ->[n84]] =
        permute
          x=t31 {pt2=root:p_blocks_2_0_conv_pw_weight target=blocks.2.0.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n84 {derived}: [t329 f32 [H=28 W=28 C=64] {derived} ->[n85]] =
        conv2d
          x=t327 {derived} <-n82
          weight=t328 {derived} <-n83
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=16;
                 groups=1}
      n85 {pt2=root[31] torch.ops.aten.conv2d.default (conv2d_11)}: [t330 f32 [H=64
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_11} ->[n86]] =
        permute x=t329 {derived} <-n84 perm=[H<-C, W<-H, C<-W]
    group g22 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n86 {derived}: [t331 f32 [H=28 W=28 C=64] {derived} ->[n87]] =
        permute x=t330 {pt2=root:conv2d_11} <-n85 perm=[H<-W, W<-C, C<-H]
      n87 {derived}: [t332 f32 [H=28 W=28 C=64] {derived} ->[n88]] =
        batch_norm
          x=t331 {derived} <-n86
          weight=t32 {pt2=root:p_blocks_2_0_bn1_weight target=blocks.2.0.bn1.weight}
          bias=t33 {pt2=root:p_blocks_2_0_bn1_bias target=blocks.2.0.bn1.bias}
          running_mean=t169 {pt2=root:b_blocks_2_0_bn1_running_mean target=blocks.2.0.bn1.running_mean}
          running_var=t170 {pt2=root:b_blocks_2_0_bn1_running_var target=blocks.2.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n88 {pt2=root[32] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_9)}: [t333 f32 [H=64
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_27} ->[n89]] =
        permute x=t332 {derived} <-n87 perm=[H<-C, W<-H, C<-W]
    n89 {pt2=root[33] torch.ops.aten.hardswish.default (hardswish_1)}: [t334 f32 [H=64
                                                                      W=28
                                                                      C=28] {pt2=root:hardswish_1} ->[n90]] =
      hardswish x=t333 {pt2=root:getitem_27} <-n88
    group g23 torch.ops.aten.conv2d.default:
      n90 {derived}: [t335 f32 [H=28 W=28 C=64] {derived} ->[n92]] =
        permute x=t334 {pt2=root:hardswish_1} <-n89 perm=[H<-W, W<-C, C<-H]
      n91 {derived}: [t336 f32 [N=64 T=1 D=1 H=5 W=5 C=1] {derived} ->[n92]] =
        permute
          x=t34 {pt2=root:p_blocks_2_0_conv_dw_weight target=blocks.2.0.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n92 {derived}: [t337 f32 [H=14 W=14 C=64] {derived} ->[n93]] =
        conv2d
          x=t335 {derived} <-n90
          weight=t336 {derived} <-n91
          bias=none
          params={h={kernel=5; stride=2; pad_before=2; pad_after=2; dilation=1};
                 w={kernel=5; stride=2; pad_before=2; pad_after=2; dilation=1};
                 in_channels=64;
                 groups=64}
      n93 {pt2=root[34] torch.ops.aten.conv2d.default (conv2d_12)}: [t338 f32 [H=64
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_12} ->[n94]] =
        permute x=t337 {derived} <-n92 perm=[H<-C, W<-H, C<-W]
    group g24 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n94 {derived}: [t339 f32 [H=14 W=14 C=64] {derived} ->[n95]] =
        permute x=t338 {pt2=root:conv2d_12} <-n93 perm=[H<-W, W<-C, C<-H]
      n95 {derived}: [t340 f32 [H=14 W=14 C=64] {derived} ->[n96]] =
        batch_norm
          x=t339 {derived} <-n94
          weight=t35 {pt2=root:p_blocks_2_0_bn2_weight target=blocks.2.0.bn2.weight}
          bias=t36 {pt2=root:p_blocks_2_0_bn2_bias target=blocks.2.0.bn2.bias}
          running_mean=t172 {pt2=root:b_blocks_2_0_bn2_running_mean target=blocks.2.0.bn2.running_mean}
          running_var=t173 {pt2=root:b_blocks_2_0_bn2_running_var target=blocks.2.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n96 {pt2=root[35] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_10)}: [t341 f32 [H=64
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_30} ->[n97]] =
        permute x=t340 {derived} <-n95 perm=[H<-C, W<-H, C<-W]
    n97 {pt2=root[36] torch.ops.aten.hardswish.default (hardswish_2)}: [t342 f32 [H=64
                                                                      W=14
                                                                      C=14] {pt2=root:hardswish_2} ->[n98,
                                                                      n109]] =
      hardswish x=t341 {pt2=root:getitem_30} <-n96
    n98 {pt2=root[37] torch.ops.aten.mean.dim (mean_1)}: [t343 f32 [H=64 W=1
                                                                    C=1] {pt2=root:mean_1} ->[n99]] =
      mean
        x=t342 {pt2=root:hardswish_2} <-n97
        params={dims=[W, C]; keepdim=true}
    group g25 torch.ops.aten.conv2d.default:
      n99 {derived}: [t344 f32 [C=64] {derived} ->[n101]] =
        permute x=t343 {pt2=root:mean_1} <-n98 perm=[H<-W, W<-C, C<-H]
      n100 {derived}: [t345 f32 [N=16 T=1 D=1 H=1 W=1 C=64] {derived} ->[n101]] =
        permute
          x=t37 {pt2=root:p_blocks_2_0_se_conv_reduce_weight target=blocks.2.0.se.conv_reduce.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n101 {derived}: [t346 f32 [C=16] {derived} ->[n102]] =
        conv2d
          x=t344 {derived} <-n99
          weight=t345 {derived} <-n100
          bias=t38 {pt2=root:p_blocks_2_0_se_conv_reduce_bias target=blocks.2.0.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=64;
                 groups=1}
      n102 {pt2=root[38] torch.ops.aten.conv2d.default (conv2d_13)}: [t347 f32 [H=16
                                                                      W=1 C=1] {pt2=root:conv2d_13} ->[n103]] =
        permute x=t346 {derived} <-n101 perm=[H<-C, W<-H, C<-W]
    n103 {pt2=root[39] torch.ops.aten.relu.default (relu_6)}: [t348 f32 [H=16
                                                                      W=1 C=1] {pt2=root:relu_6} ->[n104]] =
      relu x=t347 {pt2=root:conv2d_13} <-n102
    group g26 torch.ops.aten.conv2d.default:
      n104 {derived}: [t349 f32 [C=16] {derived} ->[n106]] =
        permute x=t348 {pt2=root:relu_6} <-n103 perm=[H<-W, W<-C, C<-H]
      n105 {derived}: [t350 f32 [N=64 T=1 D=1 H=1 W=1 C=16] {derived} ->[n106]] =
        permute
          x=t39 {pt2=root:p_blocks_2_0_se_conv_expand_weight target=blocks.2.0.se.conv_expand.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n106 {derived}: [t351 f32 [C=64] {derived} ->[n107]] =
        conv2d
          x=t349 {derived} <-n104
          weight=t350 {derived} <-n105
          bias=t40 {pt2=root:p_blocks_2_0_se_conv_expand_bias target=blocks.2.0.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=16;
                 groups=1}
      n107 {pt2=root[40] torch.ops.aten.conv2d.default (conv2d_14)}: [t352 f32 [H=64
                                                                      W=1 C=1] {pt2=root:conv2d_14} ->[n108]] =
        permute x=t351 {derived} <-n106 perm=[H<-C, W<-H, C<-W]
    n108 {pt2=root[41] torch.ops.aten.hardsigmoid.default (hardsigmoid_1)}: [t353 f32 [H=64
                                                                      W=1 C=1] {pt2=root:hardsigmoid_1} ->[n109]] =
      hardsigmoid x=t352 {pt2=root:conv2d_14} <-n107
    n109 {pt2=root[42] torch.ops.aten.mul.Tensor (mul_1)}: [t354 f32 [H=64 W=14
                                                                      C=14] {pt2=root:mul_1} ->[n110]] =
      mul
        a=t342 {pt2=root:hardswish_2} <-n97
        b=t353 {pt2=root:hardsigmoid_1} <-n108
    group g27 torch.ops.aten.conv2d.default:
      n110 {derived}: [t355 f32 [H=14 W=14 C=64] {derived} ->[n112]] =
        permute x=t354 {pt2=root:mul_1} <-n109 perm=[H<-W, W<-C, C<-H]
      n111 {derived}: [t356 f32 [N=24 T=1 D=1 H=1 W=1 C=64] {derived} ->[n112]] =
        permute
          x=t41 {pt2=root:p_blocks_2_0_conv_pwl_weight target=blocks.2.0.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n112 {derived}: [t357 f32 [H=14 W=14 C=24] {derived} ->[n113]] =
        conv2d
          x=t355 {derived} <-n110
          weight=t356 {derived} <-n111
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=64;
                 groups=1}
      n113 {pt2=root[43] torch.ops.aten.conv2d.default (conv2d_15)}: [t358 f32 [H=24
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_15} ->[n114]] =
        permute x=t357 {derived} <-n112 perm=[H<-C, W<-H, C<-W]
    group g28 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n114 {derived}: [t359 f32 [H=14 W=14 C=24] {derived} ->[n115]] =
        permute x=t358 {pt2=root:conv2d_15} <-n113 perm=[H<-W, W<-C, C<-H]
      n115 {derived}: [t360 f32 [H=14 W=14 C=24] {derived} ->[n116]] =
        batch_norm
          x=t359 {derived} <-n114
          weight=t42 {pt2=root:p_blocks_2_0_bn3_weight target=blocks.2.0.bn3.weight}
          bias=t43 {pt2=root:p_blocks_2_0_bn3_bias target=blocks.2.0.bn3.bias}
          running_mean=t175 {pt2=root:b_blocks_2_0_bn3_running_mean target=blocks.2.0.bn3.running_mean}
          running_var=t176 {pt2=root:b_blocks_2_0_bn3_running_var target=blocks.2.0.bn3.running_var}
          params={channel=C; eps=1e-05}
      n116 {pt2=root[44] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_11)}: [t361 f32 [H=24
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_33} ->[n117,
                                                                      n152]] =
        permute x=t360 {derived} <-n115 perm=[H<-C, W<-H, C<-W]
    group g29 torch.ops.aten.conv2d.default:
      n117 {derived}: [t362 f32 [H=14 W=14 C=24] {derived} ->[n119]] =
        permute x=t361 {pt2=root:getitem_33} <-n116 perm=[H<-W, W<-C, C<-H]
      n118 {derived}: [t363 f32 [N=144 T=1 D=1 H=1 W=1 C=24] {derived} ->[n119]] =
        permute
          x=t44 {pt2=root:p_blocks_2_1_conv_pw_weight target=blocks.2.1.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n119 {derived}: [t364 f32 [H=14 W=14 C=144] {derived} ->[n120]] =
        conv2d
          x=t362 {derived} <-n117
          weight=t363 {derived} <-n118
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=24;
                 groups=1}
      n120 {pt2=root[45] torch.ops.aten.conv2d.default (conv2d_16)}: [t365 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_16} ->[n121]] =
        permute x=t364 {derived} <-n119 perm=[H<-C, W<-H, C<-W]
    group g30 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n121 {derived}: [t366 f32 [H=14 W=14 C=144] {derived} ->[n122]] =
        permute x=t365 {pt2=root:conv2d_16} <-n120 perm=[H<-W, W<-C, C<-H]
      n122 {derived}: [t367 f32 [H=14 W=14 C=144] {derived} ->[n123]] =
        batch_norm
          x=t366 {derived} <-n121
          weight=t45 {pt2=root:p_blocks_2_1_bn1_weight target=blocks.2.1.bn1.weight}
          bias=t46 {pt2=root:p_blocks_2_1_bn1_bias target=blocks.2.1.bn1.bias}
          running_mean=t178 {pt2=root:b_blocks_2_1_bn1_running_mean target=blocks.2.1.bn1.running_mean}
          running_var=t179 {pt2=root:b_blocks_2_1_bn1_running_var target=blocks.2.1.bn1.running_var}
          params={channel=C; eps=1e-05}
      n123 {pt2=root[46] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_12)}: [t368 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_36} ->[n124]] =
        permute x=t367 {derived} <-n122 perm=[H<-C, W<-H, C<-W]
    n124 {pt2=root[47] torch.ops.aten.hardswish.default (hardswish_3)}: [t369 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:hardswish_3} ->[n125]] =
      hardswish x=t368 {pt2=root:getitem_36} <-n123
    group g31 torch.ops.aten.conv2d.default:
      n125 {derived}: [t370 f32 [H=14 W=14 C=144] {derived} ->[n127]] =
        permute x=t369 {pt2=root:hardswish_3} <-n124 perm=[H<-W, W<-C, C<-H]
      n126 {derived}: [t371 f32 [N=144 T=1 D=1 H=5 W=5 C=1] {derived} ->[n127]] =
        permute
          x=t47 {pt2=root:p_blocks_2_1_conv_dw_weight target=blocks.2.1.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n127 {derived}: [t372 f32 [H=14 W=14 C=144] {derived} ->[n128]] =
        conv2d
          x=t370 {derived} <-n125
          weight=t371 {derived} <-n126
          bias=none
          params={h={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
                 w={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
                 in_channels=144;
                 groups=144}
      n128 {pt2=root[48] torch.ops.aten.conv2d.default (conv2d_17)}: [t373 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_17} ->[n129]] =
        permute x=t372 {derived} <-n127 perm=[H<-C, W<-H, C<-W]
    group g32 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n129 {derived}: [t374 f32 [H=14 W=14 C=144] {derived} ->[n130]] =
        permute x=t373 {pt2=root:conv2d_17} <-n128 perm=[H<-W, W<-C, C<-H]
      n130 {derived}: [t375 f32 [H=14 W=14 C=144] {derived} ->[n131]] =
        batch_norm
          x=t374 {derived} <-n129
          weight=t48 {pt2=root:p_blocks_2_1_bn2_weight target=blocks.2.1.bn2.weight}
          bias=t49 {pt2=root:p_blocks_2_1_bn2_bias target=blocks.2.1.bn2.bias}
          running_mean=t181 {pt2=root:b_blocks_2_1_bn2_running_mean target=blocks.2.1.bn2.running_mean}
          running_var=t182 {pt2=root:b_blocks_2_1_bn2_running_var target=blocks.2.1.bn2.running_var}
          params={channel=C; eps=1e-05}
      n131 {pt2=root[49] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_13)}: [t376 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_39} ->[n132]] =
        permute x=t375 {derived} <-n130 perm=[H<-C, W<-H, C<-W]
    n132 {pt2=root[50] torch.ops.aten.hardswish.default (hardswish_4)}: [t377 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:hardswish_4} ->[n133,
                                                                      n144]] =
      hardswish x=t376 {pt2=root:getitem_39} <-n131
    n133 {pt2=root[51] torch.ops.aten.mean.dim (mean_2)}: [t378 f32 [H=144 W=1
                                                                     C=1] {pt2=root:mean_2} ->[n134]] =
      mean
        x=t377 {pt2=root:hardswish_4} <-n132
        params={dims=[W, C]; keepdim=true}
    group g33 torch.ops.aten.conv2d.default:
      n134 {derived}: [t379 f32 [C=144] {derived} ->[n136]] =
        permute x=t378 {pt2=root:mean_2} <-n133 perm=[H<-W, W<-C, C<-H]
      n135 {derived}: [t380 f32 [N=40 T=1 D=1 H=1 W=1 C=144] {derived} ->[n136]] =
        permute
          x=t50 {pt2=root:p_blocks_2_1_se_conv_reduce_weight target=blocks.2.1.se.conv_reduce.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n136 {derived}: [t381 f32 [C=40] {derived} ->[n137]] =
        conv2d
          x=t379 {derived} <-n134
          weight=t380 {derived} <-n135
          bias=t51 {pt2=root:p_blocks_2_1_se_conv_reduce_bias target=blocks.2.1.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=144;
                 groups=1}
      n137 {pt2=root[52] torch.ops.aten.conv2d.default (conv2d_18)}: [t382 f32 [H=40
                                                                      W=1 C=1] {pt2=root:conv2d_18} ->[n138]] =
        permute x=t381 {derived} <-n136 perm=[H<-C, W<-H, C<-W]
    n138 {pt2=root[53] torch.ops.aten.relu.default (relu_7)}: [t383 f32 [H=40
                                                                      W=1 C=1] {pt2=root:relu_7} ->[n139]] =
      relu x=t382 {pt2=root:conv2d_18} <-n137
    group g34 torch.ops.aten.conv2d.default:
      n139 {derived}: [t384 f32 [C=40] {derived} ->[n141]] =
        permute x=t383 {pt2=root:relu_7} <-n138 perm=[H<-W, W<-C, C<-H]
      n140 {derived}: [t385 f32 [N=144 T=1 D=1 H=1 W=1 C=40] {derived} ->[n141]] =
        permute
          x=t52 {pt2=root:p_blocks_2_1_se_conv_expand_weight target=blocks.2.1.se.conv_expand.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n141 {derived}: [t386 f32 [C=144] {derived} ->[n142]] =
        conv2d
          x=t384 {derived} <-n139
          weight=t385 {derived} <-n140
          bias=t53 {pt2=root:p_blocks_2_1_se_conv_expand_bias target=blocks.2.1.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=40;
                 groups=1}
      n142 {pt2=root[54] torch.ops.aten.conv2d.default (conv2d_19)}: [t387 f32 [H=144
                                                                      W=1 C=1] {pt2=root:conv2d_19} ->[n143]] =
        permute x=t386 {derived} <-n141 perm=[H<-C, W<-H, C<-W]
    n143 {pt2=root[55] torch.ops.aten.hardsigmoid.default (hardsigmoid_2)}: [t388 f32 [H=144
                                                                      W=1 C=1] {pt2=root:hardsigmoid_2} ->[n144]] =
      hardsigmoid x=t387 {pt2=root:conv2d_19} <-n142
    n144 {pt2=root[56] torch.ops.aten.mul.Tensor (mul_2)}: [t389 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:mul_2} ->[n145]] =
      mul
        a=t377 {pt2=root:hardswish_4} <-n132
        b=t388 {pt2=root:hardsigmoid_2} <-n143
    group g35 torch.ops.aten.conv2d.default:
      n145 {derived}: [t390 f32 [H=14 W=14 C=144] {derived} ->[n147]] =
        permute x=t389 {pt2=root:mul_2} <-n144 perm=[H<-W, W<-C, C<-H]
      n146 {derived}: [t391 f32 [N=24 T=1 D=1 H=1 W=1 C=144] {derived} ->[n147]] =
        permute
          x=t54 {pt2=root:p_blocks_2_1_conv_pwl_weight target=blocks.2.1.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n147 {derived}: [t392 f32 [H=14 W=14 C=24] {derived} ->[n148]] =
        conv2d
          x=t390 {derived} <-n145
          weight=t391 {derived} <-n146
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=144;
                 groups=1}
      n148 {pt2=root[57] torch.ops.aten.conv2d.default (conv2d_20)}: [t393 f32 [H=24
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_20} ->[n149]] =
        permute x=t392 {derived} <-n147 perm=[H<-C, W<-H, C<-W]
    group g36 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n149 {derived}: [t394 f32 [H=14 W=14 C=24] {derived} ->[n150]] =
        permute x=t393 {pt2=root:conv2d_20} <-n148 perm=[H<-W, W<-C, C<-H]
      n150 {derived}: [t395 f32 [H=14 W=14 C=24] {derived} ->[n151]] =
        batch_norm
          x=t394 {derived} <-n149
          weight=t55 {pt2=root:p_blocks_2_1_bn3_weight target=blocks.2.1.bn3.weight}
          bias=t56 {pt2=root:p_blocks_2_1_bn3_bias target=blocks.2.1.bn3.bias}
          running_mean=t184 {pt2=root:b_blocks_2_1_bn3_running_mean target=blocks.2.1.bn3.running_mean}
          running_var=t185 {pt2=root:b_blocks_2_1_bn3_running_var target=blocks.2.1.bn3.running_var}
          params={channel=C; eps=1e-05}
      n151 {pt2=root[58] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_14)}: [t396 f32 [H=24
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_42} ->[n152]] =
        permute x=t395 {derived} <-n150 perm=[H<-C, W<-H, C<-W]
    n152 {pt2=root[59] torch.ops.aten.add.Tensor (add_1)}: [t397 f32 [H=24 W=14
                                                                      C=14] {pt2=root:add_1} ->[n153,
                                                                      n188]] =
      add
        a=t396 {pt2=root:getitem_42} <-n151
        b=t361 {pt2=root:getitem_33} <-n116
    group g37 torch.ops.aten.conv2d.default:
      n153 {derived}: [t398 f32 [H=14 W=14 C=24] {derived} ->[n155]] =
        permute x=t397 {pt2=root:add_1} <-n152 perm=[H<-W, W<-C, C<-H]
      n154 {derived}: [t399 f32 [N=144 T=1 D=1 H=1 W=1 C=24] {derived} ->[n155]] =
        permute
          x=t57 {pt2=root:p_blocks_2_2_conv_pw_weight target=blocks.2.2.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n155 {derived}: [t400 f32 [H=14 W=14 C=144] {derived} ->[n156]] =
        conv2d
          x=t398 {derived} <-n153
          weight=t399 {derived} <-n154
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=24;
                 groups=1}
      n156 {pt2=root[60] torch.ops.aten.conv2d.default (conv2d_21)}: [t401 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_21} ->[n157]] =
        permute x=t400 {derived} <-n155 perm=[H<-C, W<-H, C<-W]
    group g38 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n157 {derived}: [t402 f32 [H=14 W=14 C=144] {derived} ->[n158]] =
        permute x=t401 {pt2=root:conv2d_21} <-n156 perm=[H<-W, W<-C, C<-H]
      n158 {derived}: [t403 f32 [H=14 W=14 C=144] {derived} ->[n159]] =
        batch_norm
          x=t402 {derived} <-n157
          weight=t58 {pt2=root:p_blocks_2_2_bn1_weight target=blocks.2.2.bn1.weight}
          bias=t59 {pt2=root:p_blocks_2_2_bn1_bias target=blocks.2.2.bn1.bias}
          running_mean=t187 {pt2=root:b_blocks_2_2_bn1_running_mean target=blocks.2.2.bn1.running_mean}
          running_var=t188 {pt2=root:b_blocks_2_2_bn1_running_var target=blocks.2.2.bn1.running_var}
          params={channel=C; eps=1e-05}
      n159 {pt2=root[61] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_15)}: [t404 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_45} ->[n160]] =
        permute x=t403 {derived} <-n158 perm=[H<-C, W<-H, C<-W]
    n160 {pt2=root[62] torch.ops.aten.hardswish.default (hardswish_5)}: [t405 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:hardswish_5} ->[n161]] =
      hardswish x=t404 {pt2=root:getitem_45} <-n159
    group g39 torch.ops.aten.conv2d.default:
      n161 {derived}: [t406 f32 [H=14 W=14 C=144] {derived} ->[n163]] =
        permute x=t405 {pt2=root:hardswish_5} <-n160 perm=[H<-W, W<-C, C<-H]
      n162 {derived}: [t407 f32 [N=144 T=1 D=1 H=5 W=5 C=1] {derived} ->[n163]] =
        permute
          x=t60 {pt2=root:p_blocks_2_2_conv_dw_weight target=blocks.2.2.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n163 {derived}: [t408 f32 [H=14 W=14 C=144] {derived} ->[n164]] =
        conv2d
          x=t406 {derived} <-n161
          weight=t407 {derived} <-n162
          bias=none
          params={h={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
                 w={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
                 in_channels=144;
                 groups=144}
      n164 {pt2=root[63] torch.ops.aten.conv2d.default (conv2d_22)}: [t409 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_22} ->[n165]] =
        permute x=t408 {derived} <-n163 perm=[H<-C, W<-H, C<-W]
    group g40 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n165 {derived}: [t410 f32 [H=14 W=14 C=144] {derived} ->[n166]] =
        permute x=t409 {pt2=root:conv2d_22} <-n164 perm=[H<-W, W<-C, C<-H]
      n166 {derived}: [t411 f32 [H=14 W=14 C=144] {derived} ->[n167]] =
        batch_norm
          x=t410 {derived} <-n165
          weight=t61 {pt2=root:p_blocks_2_2_bn2_weight target=blocks.2.2.bn2.weight}
          bias=t62 {pt2=root:p_blocks_2_2_bn2_bias target=blocks.2.2.bn2.bias}
          running_mean=t190 {pt2=root:b_blocks_2_2_bn2_running_mean target=blocks.2.2.bn2.running_mean}
          running_var=t191 {pt2=root:b_blocks_2_2_bn2_running_var target=blocks.2.2.bn2.running_var}
          params={channel=C; eps=1e-05}
      n167 {pt2=root[64] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_16)}: [t412 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_48} ->[n168]] =
        permute x=t411 {derived} <-n166 perm=[H<-C, W<-H, C<-W]
    n168 {pt2=root[65] torch.ops.aten.hardswish.default (hardswish_6)}: [t413 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:hardswish_6} ->[n169,
                                                                      n180]] =
      hardswish x=t412 {pt2=root:getitem_48} <-n167
    n169 {pt2=root[66] torch.ops.aten.mean.dim (mean_3)}: [t414 f32 [H=144 W=1
                                                                     C=1] {pt2=root:mean_3} ->[n170]] =
      mean
        x=t413 {pt2=root:hardswish_6} <-n168
        params={dims=[W, C]; keepdim=true}
    group g41 torch.ops.aten.conv2d.default:
      n170 {derived}: [t415 f32 [C=144] {derived} ->[n172]] =
        permute x=t414 {pt2=root:mean_3} <-n169 perm=[H<-W, W<-C, C<-H]
      n171 {derived}: [t416 f32 [N=40 T=1 D=1 H=1 W=1 C=144] {derived} ->[n172]] =
        permute
          x=t63 {pt2=root:p_blocks_2_2_se_conv_reduce_weight target=blocks.2.2.se.conv_reduce.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n172 {derived}: [t417 f32 [C=40] {derived} ->[n173]] =
        conv2d
          x=t415 {derived} <-n170
          weight=t416 {derived} <-n171
          bias=t64 {pt2=root:p_blocks_2_2_se_conv_reduce_bias target=blocks.2.2.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=144;
                 groups=1}
      n173 {pt2=root[67] torch.ops.aten.conv2d.default (conv2d_23)}: [t418 f32 [H=40
                                                                      W=1 C=1] {pt2=root:conv2d_23} ->[n174]] =
        permute x=t417 {derived} <-n172 perm=[H<-C, W<-H, C<-W]
    n174 {pt2=root[68] torch.ops.aten.relu.default (relu_8)}: [t419 f32 [H=40
                                                                      W=1 C=1] {pt2=root:relu_8} ->[n175]] =
      relu x=t418 {pt2=root:conv2d_23} <-n173
    group g42 torch.ops.aten.conv2d.default:
      n175 {derived}: [t420 f32 [C=40] {derived} ->[n177]] =
        permute x=t419 {pt2=root:relu_8} <-n174 perm=[H<-W, W<-C, C<-H]
      n176 {derived}: [t421 f32 [N=144 T=1 D=1 H=1 W=1 C=40] {derived} ->[n177]] =
        permute
          x=t65 {pt2=root:p_blocks_2_2_se_conv_expand_weight target=blocks.2.2.se.conv_expand.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n177 {derived}: [t422 f32 [C=144] {derived} ->[n178]] =
        conv2d
          x=t420 {derived} <-n175
          weight=t421 {derived} <-n176
          bias=t66 {pt2=root:p_blocks_2_2_se_conv_expand_bias target=blocks.2.2.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=40;
                 groups=1}
      n178 {pt2=root[69] torch.ops.aten.conv2d.default (conv2d_24)}: [t423 f32 [H=144
                                                                      W=1 C=1] {pt2=root:conv2d_24} ->[n179]] =
        permute x=t422 {derived} <-n177 perm=[H<-C, W<-H, C<-W]
    n179 {pt2=root[70] torch.ops.aten.hardsigmoid.default (hardsigmoid_3)}: [t424 f32 [H=144
                                                                      W=1 C=1] {pt2=root:hardsigmoid_3} ->[n180]] =
      hardsigmoid x=t423 {pt2=root:conv2d_24} <-n178
    n180 {pt2=root[71] torch.ops.aten.mul.Tensor (mul_3)}: [t425 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:mul_3} ->[n181]] =
      mul
        a=t413 {pt2=root:hardswish_6} <-n168
        b=t424 {pt2=root:hardsigmoid_3} <-n179
    group g43 torch.ops.aten.conv2d.default:
      n181 {derived}: [t426 f32 [H=14 W=14 C=144] {derived} ->[n183]] =
        permute x=t425 {pt2=root:mul_3} <-n180 perm=[H<-W, W<-C, C<-H]
      n182 {derived}: [t427 f32 [N=24 T=1 D=1 H=1 W=1 C=144] {derived} ->[n183]] =
        permute
          x=t67 {pt2=root:p_blocks_2_2_conv_pwl_weight target=blocks.2.2.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n183 {derived}: [t428 f32 [H=14 W=14 C=24] {derived} ->[n184]] =
        conv2d
          x=t426 {derived} <-n181
          weight=t427 {derived} <-n182
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=144;
                 groups=1}
      n184 {pt2=root[72] torch.ops.aten.conv2d.default (conv2d_25)}: [t429 f32 [H=24
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_25} ->[n185]] =
        permute x=t428 {derived} <-n183 perm=[H<-C, W<-H, C<-W]
    group g44 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n185 {derived}: [t430 f32 [H=14 W=14 C=24] {derived} ->[n186]] =
        permute x=t429 {pt2=root:conv2d_25} <-n184 perm=[H<-W, W<-C, C<-H]
      n186 {derived}: [t431 f32 [H=14 W=14 C=24] {derived} ->[n187]] =
        batch_norm
          x=t430 {derived} <-n185
          weight=t68 {pt2=root:p_blocks_2_2_bn3_weight target=blocks.2.2.bn3.weight}
          bias=t69 {pt2=root:p_blocks_2_2_bn3_bias target=blocks.2.2.bn3.bias}
          running_mean=t193 {pt2=root:b_blocks_2_2_bn3_running_mean target=blocks.2.2.bn3.running_mean}
          running_var=t194 {pt2=root:b_blocks_2_2_bn3_running_var target=blocks.2.2.bn3.running_var}
          params={channel=C; eps=1e-05}
      n187 {pt2=root[73] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_17)}: [t432 f32 [H=24
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_51} ->[n188]] =
        permute x=t431 {derived} <-n186 perm=[H<-C, W<-H, C<-W]
    n188 {pt2=root[74] torch.ops.aten.add.Tensor (add_2)}: [t433 f32 [H=24 W=14
                                                                      C=14] {pt2=root:add_2} ->[n189,
                                                                      n224]] =
      add a=t432 {pt2=root:getitem_51} <-n187 b=t397 {pt2=root:add_1} <-n152
    group g45 torch.ops.aten.conv2d.default:
      n189 {derived}: [t434 f32 [H=14 W=14 C=24] {derived} ->[n191]] =
        permute x=t433 {pt2=root:add_2} <-n188 perm=[H<-W, W<-C, C<-H]
      n190 {derived}: [t435 f32 [N=72 T=1 D=1 H=1 W=1 C=24] {derived} ->[n191]] =
        permute
          x=t70 {pt2=root:p_blocks_3_0_conv_pw_weight target=blocks.3.0.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n191 {derived}: [t436 f32 [H=14 W=14 C=72] {derived} ->[n192]] =
        conv2d
          x=t434 {derived} <-n189
          weight=t435 {derived} <-n190
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=24;
                 groups=1}
      n192 {pt2=root[75] torch.ops.aten.conv2d.default (conv2d_26)}: [t437 f32 [H=72
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_26} ->[n193]] =
        permute x=t436 {derived} <-n191 perm=[H<-C, W<-H, C<-W]
    group g46 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n193 {derived}: [t438 f32 [H=14 W=14 C=72] {derived} ->[n194]] =
        permute x=t437 {pt2=root:conv2d_26} <-n192 perm=[H<-W, W<-C, C<-H]
      n194 {derived}: [t439 f32 [H=14 W=14 C=72] {derived} ->[n195]] =
        batch_norm
          x=t438 {derived} <-n193
          weight=t71 {pt2=root:p_blocks_3_0_bn1_weight target=blocks.3.0.bn1.weight}
          bias=t72 {pt2=root:p_blocks_3_0_bn1_bias target=blocks.3.0.bn1.bias}
          running_mean=t196 {pt2=root:b_blocks_3_0_bn1_running_mean target=blocks.3.0.bn1.running_mean}
          running_var=t197 {pt2=root:b_blocks_3_0_bn1_running_var target=blocks.3.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n195 {pt2=root[76] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_18)}: [t440 f32 [H=72
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_54} ->[n196]] =
        permute x=t439 {derived} <-n194 perm=[H<-C, W<-H, C<-W]
    n196 {pt2=root[77] torch.ops.aten.hardswish.default (hardswish_7)}: [t441 f32 [H=72
                                                                      W=14
                                                                      C=14] {pt2=root:hardswish_7} ->[n197]] =
      hardswish x=t440 {pt2=root:getitem_54} <-n195
    group g47 torch.ops.aten.conv2d.default:
      n197 {derived}: [t442 f32 [H=14 W=14 C=72] {derived} ->[n199]] =
        permute x=t441 {pt2=root:hardswish_7} <-n196 perm=[H<-W, W<-C, C<-H]
      n198 {derived}: [t443 f32 [N=72 T=1 D=1 H=5 W=5 C=1] {derived} ->[n199]] =
        permute
          x=t73 {pt2=root:p_blocks_3_0_conv_dw_weight target=blocks.3.0.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n199 {derived}: [t444 f32 [H=14 W=14 C=72] {derived} ->[n200]] =
        conv2d
          x=t442 {derived} <-n197
          weight=t443 {derived} <-n198
          bias=none
          params={h={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
                 w={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
                 in_channels=72;
                 groups=72}
      n200 {pt2=root[78] torch.ops.aten.conv2d.default (conv2d_27)}: [t445 f32 [H=72
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_27} ->[n201]] =
        permute x=t444 {derived} <-n199 perm=[H<-C, W<-H, C<-W]
    group g48 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n201 {derived}: [t446 f32 [H=14 W=14 C=72] {derived} ->[n202]] =
        permute x=t445 {pt2=root:conv2d_27} <-n200 perm=[H<-W, W<-C, C<-H]
      n202 {derived}: [t447 f32 [H=14 W=14 C=72] {derived} ->[n203]] =
        batch_norm
          x=t446 {derived} <-n201
          weight=t74 {pt2=root:p_blocks_3_0_bn2_weight target=blocks.3.0.bn2.weight}
          bias=t75 {pt2=root:p_blocks_3_0_bn2_bias target=blocks.3.0.bn2.bias}
          running_mean=t199 {pt2=root:b_blocks_3_0_bn2_running_mean target=blocks.3.0.bn2.running_mean}
          running_var=t200 {pt2=root:b_blocks_3_0_bn2_running_var target=blocks.3.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n203 {pt2=root[79] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_19)}: [t448 f32 [H=72
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_57} ->[n204]] =
        permute x=t447 {derived} <-n202 perm=[H<-C, W<-H, C<-W]
    n204 {pt2=root[80] torch.ops.aten.hardswish.default (hardswish_8)}: [t449 f32 [H=72
                                                                      W=14
                                                                      C=14] {pt2=root:hardswish_8} ->[n205,
                                                                      n216]] =
      hardswish x=t448 {pt2=root:getitem_57} <-n203
    n205 {pt2=root[81] torch.ops.aten.mean.dim (mean_4)}: [t450 f32 [H=72 W=1
                                                                     C=1] {pt2=root:mean_4} ->[n206]] =
      mean
        x=t449 {pt2=root:hardswish_8} <-n204
        params={dims=[W, C]; keepdim=true}
    group g49 torch.ops.aten.conv2d.default:
      n206 {derived}: [t451 f32 [C=72] {derived} ->[n208]] =
        permute x=t450 {pt2=root:mean_4} <-n205 perm=[H<-W, W<-C, C<-H]
      n207 {derived}: [t452 f32 [N=24 T=1 D=1 H=1 W=1 C=72] {derived} ->[n208]] =
        permute
          x=t76 {pt2=root:p_blocks_3_0_se_conv_reduce_weight target=blocks.3.0.se.conv_reduce.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n208 {derived}: [t453 f32 [C=24] {derived} ->[n209]] =
        conv2d
          x=t451 {derived} <-n206
          weight=t452 {derived} <-n207
          bias=t77 {pt2=root:p_blocks_3_0_se_conv_reduce_bias target=blocks.3.0.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=72;
                 groups=1}
      n209 {pt2=root[82] torch.ops.aten.conv2d.default (conv2d_28)}: [t454 f32 [H=24
                                                                      W=1 C=1] {pt2=root:conv2d_28} ->[n210]] =
        permute x=t453 {derived} <-n208 perm=[H<-C, W<-H, C<-W]
    n210 {pt2=root[83] torch.ops.aten.relu.default (relu_9)}: [t455 f32 [H=24
                                                                      W=1 C=1] {pt2=root:relu_9} ->[n211]] =
      relu x=t454 {pt2=root:conv2d_28} <-n209
    group g50 torch.ops.aten.conv2d.default:
      n211 {derived}: [t456 f32 [C=24] {derived} ->[n213]] =
        permute x=t455 {pt2=root:relu_9} <-n210 perm=[H<-W, W<-C, C<-H]
      n212 {derived}: [t457 f32 [N=72 T=1 D=1 H=1 W=1 C=24] {derived} ->[n213]] =
        permute
          x=t78 {pt2=root:p_blocks_3_0_se_conv_expand_weight target=blocks.3.0.se.conv_expand.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n213 {derived}: [t458 f32 [C=72] {derived} ->[n214]] =
        conv2d
          x=t456 {derived} <-n211
          weight=t457 {derived} <-n212
          bias=t79 {pt2=root:p_blocks_3_0_se_conv_expand_bias target=blocks.3.0.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=24;
                 groups=1}
      n214 {pt2=root[84] torch.ops.aten.conv2d.default (conv2d_29)}: [t459 f32 [H=72
                                                                      W=1 C=1] {pt2=root:conv2d_29} ->[n215]] =
        permute x=t458 {derived} <-n213 perm=[H<-C, W<-H, C<-W]
    n215 {pt2=root[85] torch.ops.aten.hardsigmoid.default (hardsigmoid_4)}: [t460 f32 [H=72
                                                                      W=1 C=1] {pt2=root:hardsigmoid_4} ->[n216]] =
      hardsigmoid x=t459 {pt2=root:conv2d_29} <-n214
    n216 {pt2=root[86] torch.ops.aten.mul.Tensor (mul_4)}: [t461 f32 [H=72 W=14
                                                                      C=14] {pt2=root:mul_4} ->[n217]] =
      mul
        a=t449 {pt2=root:hardswish_8} <-n204
        b=t460 {pt2=root:hardsigmoid_4} <-n215
    group g51 torch.ops.aten.conv2d.default:
      n217 {derived}: [t462 f32 [H=14 W=14 C=72] {derived} ->[n219]] =
        permute x=t461 {pt2=root:mul_4} <-n216 perm=[H<-W, W<-C, C<-H]
      n218 {derived}: [t463 f32 [N=24 T=1 D=1 H=1 W=1 C=72] {derived} ->[n219]] =
        permute
          x=t80 {pt2=root:p_blocks_3_0_conv_pwl_weight target=blocks.3.0.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n219 {derived}: [t464 f32 [H=14 W=14 C=24] {derived} ->[n220]] =
        conv2d
          x=t462 {derived} <-n217
          weight=t463 {derived} <-n218
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=72;
                 groups=1}
      n220 {pt2=root[87] torch.ops.aten.conv2d.default (conv2d_30)}: [t465 f32 [H=24
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_30} ->[n221]] =
        permute x=t464 {derived} <-n219 perm=[H<-C, W<-H, C<-W]
    group g52 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n221 {derived}: [t466 f32 [H=14 W=14 C=24] {derived} ->[n222]] =
        permute x=t465 {pt2=root:conv2d_30} <-n220 perm=[H<-W, W<-C, C<-H]
      n222 {derived}: [t467 f32 [H=14 W=14 C=24] {derived} ->[n223]] =
        batch_norm
          x=t466 {derived} <-n221
          weight=t81 {pt2=root:p_blocks_3_0_bn3_weight target=blocks.3.0.bn3.weight}
          bias=t82 {pt2=root:p_blocks_3_0_bn3_bias target=blocks.3.0.bn3.bias}
          running_mean=t202 {pt2=root:b_blocks_3_0_bn3_running_mean target=blocks.3.0.bn3.running_mean}
          running_var=t203 {pt2=root:b_blocks_3_0_bn3_running_var target=blocks.3.0.bn3.running_var}
          params={channel=C; eps=1e-05}
      n223 {pt2=root[88] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_20)}: [t468 f32 [H=24
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_60} ->[n224]] =
        permute x=t467 {derived} <-n222 perm=[H<-C, W<-H, C<-W]
    n224 {pt2=root[89] torch.ops.aten.add.Tensor (add_3)}: [t469 f32 [H=24 W=14
                                                                      C=14] {pt2=root:add_3} ->[n225,
                                                                      n260]] =
      add a=t468 {pt2=root:getitem_60} <-n223 b=t433 {pt2=root:add_2} <-n188
    group g53 torch.ops.aten.conv2d.default:
      n225 {derived}: [t470 f32 [H=14 W=14 C=24] {derived} ->[n227]] =
        permute x=t469 {pt2=root:add_3} <-n224 perm=[H<-W, W<-C, C<-H]
      n226 {derived}: [t471 f32 [N=72 T=1 D=1 H=1 W=1 C=24] {derived} ->[n227]] =
        permute
          x=t83 {pt2=root:p_blocks_3_1_conv_pw_weight target=blocks.3.1.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n227 {derived}: [t472 f32 [H=14 W=14 C=72] {derived} ->[n228]] =
        conv2d
          x=t470 {derived} <-n225
          weight=t471 {derived} <-n226
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=24;
                 groups=1}
      n228 {pt2=root[90] torch.ops.aten.conv2d.default (conv2d_31)}: [t473 f32 [H=72
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_31} ->[n229]] =
        permute x=t472 {derived} <-n227 perm=[H<-C, W<-H, C<-W]
    group g54 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n229 {derived}: [t474 f32 [H=14 W=14 C=72] {derived} ->[n230]] =
        permute x=t473 {pt2=root:conv2d_31} <-n228 perm=[H<-W, W<-C, C<-H]
      n230 {derived}: [t475 f32 [H=14 W=14 C=72] {derived} ->[n231]] =
        batch_norm
          x=t474 {derived} <-n229
          weight=t84 {pt2=root:p_blocks_3_1_bn1_weight target=blocks.3.1.bn1.weight}
          bias=t85 {pt2=root:p_blocks_3_1_bn1_bias target=blocks.3.1.bn1.bias}
          running_mean=t205 {pt2=root:b_blocks_3_1_bn1_running_mean target=blocks.3.1.bn1.running_mean}
          running_var=t206 {pt2=root:b_blocks_3_1_bn1_running_var target=blocks.3.1.bn1.running_var}
          params={channel=C; eps=1e-05}
      n231 {pt2=root[91] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_21)}: [t476 f32 [H=72
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_63} ->[n232]] =
        permute x=t475 {derived} <-n230 perm=[H<-C, W<-H, C<-W]
    n232 {pt2=root[92] torch.ops.aten.hardswish.default (hardswish_9)}: [t477 f32 [H=72
                                                                      W=14
                                                                      C=14] {pt2=root:hardswish_9} ->[n233]] =
      hardswish x=t476 {pt2=root:getitem_63} <-n231
    group g55 torch.ops.aten.conv2d.default:
      n233 {derived}: [t478 f32 [H=14 W=14 C=72] {derived} ->[n235]] =
        permute x=t477 {pt2=root:hardswish_9} <-n232 perm=[H<-W, W<-C, C<-H]
      n234 {derived}: [t479 f32 [N=72 T=1 D=1 H=5 W=5 C=1] {derived} ->[n235]] =
        permute
          x=t86 {pt2=root:p_blocks_3_1_conv_dw_weight target=blocks.3.1.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n235 {derived}: [t480 f32 [H=14 W=14 C=72] {derived} ->[n236]] =
        conv2d
          x=t478 {derived} <-n233
          weight=t479 {derived} <-n234
          bias=none
          params={h={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
                 w={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
                 in_channels=72;
                 groups=72}
      n236 {pt2=root[93] torch.ops.aten.conv2d.default (conv2d_32)}: [t481 f32 [H=72
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_32} ->[n237]] =
        permute x=t480 {derived} <-n235 perm=[H<-C, W<-H, C<-W]
    group g56 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n237 {derived}: [t482 f32 [H=14 W=14 C=72] {derived} ->[n238]] =
        permute x=t481 {pt2=root:conv2d_32} <-n236 perm=[H<-W, W<-C, C<-H]
      n238 {derived}: [t483 f32 [H=14 W=14 C=72] {derived} ->[n239]] =
        batch_norm
          x=t482 {derived} <-n237
          weight=t87 {pt2=root:p_blocks_3_1_bn2_weight target=blocks.3.1.bn2.weight}
          bias=t88 {pt2=root:p_blocks_3_1_bn2_bias target=blocks.3.1.bn2.bias}
          running_mean=t208 {pt2=root:b_blocks_3_1_bn2_running_mean target=blocks.3.1.bn2.running_mean}
          running_var=t209 {pt2=root:b_blocks_3_1_bn2_running_var target=blocks.3.1.bn2.running_var}
          params={channel=C; eps=1e-05}
      n239 {pt2=root[94] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_22)}: [t484 f32 [H=72
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_66} ->[n240]] =
        permute x=t483 {derived} <-n238 perm=[H<-C, W<-H, C<-W]
    n240 {pt2=root[95] torch.ops.aten.hardswish.default (hardswish_10)}: [t485 f32 [H=72
                                                                      W=14
                                                                      C=14] {pt2=root:hardswish_10} ->[n241,
                                                                      n252]] =
      hardswish x=t484 {pt2=root:getitem_66} <-n239
    n241 {pt2=root[96] torch.ops.aten.mean.dim (mean_5)}: [t486 f32 [H=72 W=1
                                                                     C=1] {pt2=root:mean_5} ->[n242]] =
      mean
        x=t485 {pt2=root:hardswish_10} <-n240
        params={dims=[W, C]; keepdim=true}
    group g57 torch.ops.aten.conv2d.default:
      n242 {derived}: [t487 f32 [C=72] {derived} ->[n244]] =
        permute x=t486 {pt2=root:mean_5} <-n241 perm=[H<-W, W<-C, C<-H]
      n243 {derived}: [t488 f32 [N=24 T=1 D=1 H=1 W=1 C=72] {derived} ->[n244]] =
        permute
          x=t89 {pt2=root:p_blocks_3_1_se_conv_reduce_weight target=blocks.3.1.se.conv_reduce.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n244 {derived}: [t489 f32 [C=24] {derived} ->[n245]] =
        conv2d
          x=t487 {derived} <-n242
          weight=t488 {derived} <-n243
          bias=t90 {pt2=root:p_blocks_3_1_se_conv_reduce_bias target=blocks.3.1.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=72;
                 groups=1}
      n245 {pt2=root[97] torch.ops.aten.conv2d.default (conv2d_33)}: [t490 f32 [H=24
                                                                      W=1 C=1] {pt2=root:conv2d_33} ->[n246]] =
        permute x=t489 {derived} <-n244 perm=[H<-C, W<-H, C<-W]
    n246 {pt2=root[98] torch.ops.aten.relu.default (relu_10)}: [t491 f32 [H=24
                                                                      W=1 C=1] {pt2=root:relu_10} ->[n247]] =
      relu x=t490 {pt2=root:conv2d_33} <-n245
    group g58 torch.ops.aten.conv2d.default:
      n247 {derived}: [t492 f32 [C=24] {derived} ->[n249]] =
        permute x=t491 {pt2=root:relu_10} <-n246 perm=[H<-W, W<-C, C<-H]
      n248 {derived}: [t493 f32 [N=72 T=1 D=1 H=1 W=1 C=24] {derived} ->[n249]] =
        permute
          x=t91 {pt2=root:p_blocks_3_1_se_conv_expand_weight target=blocks.3.1.se.conv_expand.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n249 {derived}: [t494 f32 [C=72] {derived} ->[n250]] =
        conv2d
          x=t492 {derived} <-n247
          weight=t493 {derived} <-n248
          bias=t92 {pt2=root:p_blocks_3_1_se_conv_expand_bias target=blocks.3.1.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=24;
                 groups=1}
      n250 {pt2=root[99] torch.ops.aten.conv2d.default (conv2d_34)}: [t495 f32 [H=72
                                                                      W=1 C=1] {pt2=root:conv2d_34} ->[n251]] =
        permute x=t494 {derived} <-n249 perm=[H<-C, W<-H, C<-W]
    n251 {pt2=root[100] torch.ops.aten.hardsigmoid.default (hardsigmoid_5)}: [t496 f32 [H=72
                                                                      W=1 C=1] {pt2=root:hardsigmoid_5} ->[n252]] =
      hardsigmoid x=t495 {pt2=root:conv2d_34} <-n250
    n252 {pt2=root[101] torch.ops.aten.mul.Tensor (mul_5)}: [t497 f32 [H=72
                                                                      W=14
                                                                      C=14] {pt2=root:mul_5} ->[n253]] =
      mul
        a=t485 {pt2=root:hardswish_10} <-n240
        b=t496 {pt2=root:hardsigmoid_5} <-n251
    group g59 torch.ops.aten.conv2d.default:
      n253 {derived}: [t498 f32 [H=14 W=14 C=72] {derived} ->[n255]] =
        permute x=t497 {pt2=root:mul_5} <-n252 perm=[H<-W, W<-C, C<-H]
      n254 {derived}: [t499 f32 [N=24 T=1 D=1 H=1 W=1 C=72] {derived} ->[n255]] =
        permute
          x=t93 {pt2=root:p_blocks_3_1_conv_pwl_weight target=blocks.3.1.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n255 {derived}: [t500 f32 [H=14 W=14 C=24] {derived} ->[n256]] =
        conv2d
          x=t498 {derived} <-n253
          weight=t499 {derived} <-n254
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=72;
                 groups=1}
      n256 {pt2=root[102] torch.ops.aten.conv2d.default (conv2d_35)}: [t501 f32 [H=24
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_35} ->[n257]] =
        permute x=t500 {derived} <-n255 perm=[H<-C, W<-H, C<-W]
    group g60 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n257 {derived}: [t502 f32 [H=14 W=14 C=24] {derived} ->[n258]] =
        permute x=t501 {pt2=root:conv2d_35} <-n256 perm=[H<-W, W<-C, C<-H]
      n258 {derived}: [t503 f32 [H=14 W=14 C=24] {derived} ->[n259]] =
        batch_norm
          x=t502 {derived} <-n257
          weight=t94 {pt2=root:p_blocks_3_1_bn3_weight target=blocks.3.1.bn3.weight}
          bias=t95 {pt2=root:p_blocks_3_1_bn3_bias target=blocks.3.1.bn3.bias}
          running_mean=t211 {pt2=root:b_blocks_3_1_bn3_running_mean target=blocks.3.1.bn3.running_mean}
          running_var=t212 {pt2=root:b_blocks_3_1_bn3_running_var target=blocks.3.1.bn3.running_var}
          params={channel=C; eps=1e-05}
      n259 {pt2=root[103] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_23)}: [t504 f32 [H=24
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_69} ->[n260]] =
        permute x=t503 {derived} <-n258 perm=[H<-C, W<-H, C<-W]
    n260 {pt2=root[104] torch.ops.aten.add.Tensor (add_4)}: [t505 f32 [H=24
                                                                      W=14
                                                                      C=14] {pt2=root:add_4} ->[n261]] =
      add a=t504 {pt2=root:getitem_69} <-n259 b=t469 {pt2=root:add_3} <-n224
    group g61 torch.ops.aten.conv2d.default:
      n261 {derived}: [t506 f32 [H=14 W=14 C=24] {derived} ->[n263]] =
        permute x=t505 {pt2=root:add_4} <-n260 perm=[H<-W, W<-C, C<-H]
      n262 {derived}: [t507 f32 [N=144 T=1 D=1 H=1 W=1 C=24] {derived} ->[n263]] =
        permute
          x=t96 {pt2=root:p_blocks_4_0_conv_pw_weight target=blocks.4.0.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n263 {derived}: [t508 f32 [H=14 W=14 C=144] {derived} ->[n264]] =
        conv2d
          x=t506 {derived} <-n261
          weight=t507 {derived} <-n262
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=24;
                 groups=1}
      n264 {pt2=root[105] torch.ops.aten.conv2d.default (conv2d_36)}: [t509 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_36} ->[n265]] =
        permute x=t508 {derived} <-n263 perm=[H<-C, W<-H, C<-W]
    group g62 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n265 {derived}: [t510 f32 [H=14 W=14 C=144] {derived} ->[n266]] =
        permute x=t509 {pt2=root:conv2d_36} <-n264 perm=[H<-W, W<-C, C<-H]
      n266 {derived}: [t511 f32 [H=14 W=14 C=144] {derived} ->[n267]] =
        batch_norm
          x=t510 {derived} <-n265
          weight=t97 {pt2=root:p_blocks_4_0_bn1_weight target=blocks.4.0.bn1.weight}
          bias=t98 {pt2=root:p_blocks_4_0_bn1_bias target=blocks.4.0.bn1.bias}
          running_mean=t214 {pt2=root:b_blocks_4_0_bn1_running_mean target=blocks.4.0.bn1.running_mean}
          running_var=t215 {pt2=root:b_blocks_4_0_bn1_running_var target=blocks.4.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n267 {pt2=root[106] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_24)}: [t512 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_72} ->[n268]] =
        permute x=t511 {derived} <-n266 perm=[H<-C, W<-H, C<-W]
    n268 {pt2=root[107] torch.ops.aten.hardswish.default (hardswish_11)}: [t513 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:hardswish_11} ->[n269]] =
      hardswish x=t512 {pt2=root:getitem_72} <-n267
    group g63 torch.ops.aten.conv2d.default:
      n269 {derived}: [t514 f32 [H=14 W=14 C=144] {derived} ->[n271]] =
        permute x=t513 {pt2=root:hardswish_11} <-n268 perm=[H<-W, W<-C, C<-H]
      n270 {derived}: [t515 f32 [N=144 T=1 D=1 H=5 W=5 C=1] {derived} ->[n271]] =
        permute
          x=t99 {pt2=root:p_blocks_4_0_conv_dw_weight target=blocks.4.0.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n271 {derived}: [t516 f32 [H=7 W=7 C=144] {derived} ->[n272]] =
        conv2d
          x=t514 {derived} <-n269
          weight=t515 {derived} <-n270
          bias=none
          params={h={kernel=5; stride=2; pad_before=2; pad_after=2; dilation=1};
                 w={kernel=5; stride=2; pad_before=2; pad_after=2; dilation=1};
                 in_channels=144;
                 groups=144}
      n272 {pt2=root[108] torch.ops.aten.conv2d.default (conv2d_37)}: [t517 f32 [H=144
                                                                      W=7 C=7] {pt2=root:conv2d_37} ->[n273]] =
        permute x=t516 {derived} <-n271 perm=[H<-C, W<-H, C<-W]
    group g64 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n273 {derived}: [t518 f32 [H=7 W=7 C=144] {derived} ->[n274]] =
        permute x=t517 {pt2=root:conv2d_37} <-n272 perm=[H<-W, W<-C, C<-H]
      n274 {derived}: [t519 f32 [H=7 W=7 C=144] {derived} ->[n275]] =
        batch_norm
          x=t518 {derived} <-n273
          weight=t100 {pt2=root:p_blocks_4_0_bn2_weight target=blocks.4.0.bn2.weight}
          bias=t101 {pt2=root:p_blocks_4_0_bn2_bias target=blocks.4.0.bn2.bias}
          running_mean=t217 {pt2=root:b_blocks_4_0_bn2_running_mean target=blocks.4.0.bn2.running_mean}
          running_var=t218 {pt2=root:b_blocks_4_0_bn2_running_var target=blocks.4.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n275 {pt2=root[109] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_25)}: [t520 f32 [H=144
                                                                      W=7 C=7] {pt2=root:getitem_75} ->[n276]] =
        permute x=t519 {derived} <-n274 perm=[H<-C, W<-H, C<-W]
    n276 {pt2=root[110] torch.ops.aten.hardswish.default (hardswish_12)}: [t521 f32 [H=144
                                                                      W=7 C=7] {pt2=root:hardswish_12} ->[n277,
                                                                      n288]] =
      hardswish x=t520 {pt2=root:getitem_75} <-n275
    n277 {pt2=root[111] torch.ops.aten.mean.dim (mean_6)}: [t522 f32 [H=144 W=1
                                                                      C=1] {pt2=root:mean_6} ->[n278]] =
      mean
        x=t521 {pt2=root:hardswish_12} <-n276
        params={dims=[W, C]; keepdim=true}
    group g65 torch.ops.aten.conv2d.default:
      n278 {derived}: [t523 f32 [C=144] {derived} ->[n280]] =
        permute x=t522 {pt2=root:mean_6} <-n277 perm=[H<-W, W<-C, C<-H]
      n279 {derived}: [t524 f32 [N=40 T=1 D=1 H=1 W=1 C=144] {derived} ->[n280]] =
        permute
          x=t102 {pt2=root:p_blocks_4_0_se_conv_reduce_weight target=blocks.4.0.se.conv_reduce.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n280 {derived}: [t525 f32 [C=40] {derived} ->[n281]] =
        conv2d
          x=t523 {derived} <-n278
          weight=t524 {derived} <-n279
          bias=t103 {pt2=root:p_blocks_4_0_se_conv_reduce_bias target=blocks.4.0.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=144;
                 groups=1}
      n281 {pt2=root[112] torch.ops.aten.conv2d.default (conv2d_38)}: [t526 f32 [H=40
                                                                      W=1 C=1] {pt2=root:conv2d_38} ->[n282]] =
        permute x=t525 {derived} <-n280 perm=[H<-C, W<-H, C<-W]
    n282 {pt2=root[113] torch.ops.aten.relu.default (relu_11)}: [t527 f32 [H=40
                                                                      W=1 C=1] {pt2=root:relu_11} ->[n283]] =
      relu x=t526 {pt2=root:conv2d_38} <-n281
    group g66 torch.ops.aten.conv2d.default:
      n283 {derived}: [t528 f32 [C=40] {derived} ->[n285]] =
        permute x=t527 {pt2=root:relu_11} <-n282 perm=[H<-W, W<-C, C<-H]
      n284 {derived}: [t529 f32 [N=144 T=1 D=1 H=1 W=1 C=40] {derived} ->[n285]] =
        permute
          x=t104 {pt2=root:p_blocks_4_0_se_conv_expand_weight target=blocks.4.0.se.conv_expand.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n285 {derived}: [t530 f32 [C=144] {derived} ->[n286]] =
        conv2d
          x=t528 {derived} <-n283
          weight=t529 {derived} <-n284
          bias=t105 {pt2=root:p_blocks_4_0_se_conv_expand_bias target=blocks.4.0.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=40;
                 groups=1}
      n286 {pt2=root[114] torch.ops.aten.conv2d.default (conv2d_39)}: [t531 f32 [H=144
                                                                      W=1 C=1] {pt2=root:conv2d_39} ->[n287]] =
        permute x=t530 {derived} <-n285 perm=[H<-C, W<-H, C<-W]
    n287 {pt2=root[115] torch.ops.aten.hardsigmoid.default (hardsigmoid_6)}: [t532 f32 [H=144
                                                                      W=1 C=1] {pt2=root:hardsigmoid_6} ->[n288]] =
      hardsigmoid x=t531 {pt2=root:conv2d_39} <-n286
    n288 {pt2=root[116] torch.ops.aten.mul.Tensor (mul_6)}: [t533 f32 [H=144
                                                                      W=7 C=7] {pt2=root:mul_6} ->[n289]] =
      mul
        a=t521 {pt2=root:hardswish_12} <-n276
        b=t532 {pt2=root:hardsigmoid_6} <-n287
    group g67 torch.ops.aten.conv2d.default:
      n289 {derived}: [t534 f32 [H=7 W=7 C=144] {derived} ->[n291]] =
        permute x=t533 {pt2=root:mul_6} <-n288 perm=[H<-W, W<-C, C<-H]
      n290 {derived}: [t535 f32 [N=48 T=1 D=1 H=1 W=1 C=144] {derived} ->[n291]] =
        permute
          x=t106 {pt2=root:p_blocks_4_0_conv_pwl_weight target=blocks.4.0.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n291 {derived}: [t536 f32 [H=7 W=7 C=48] {derived} ->[n292]] =
        conv2d
          x=t534 {derived} <-n289
          weight=t535 {derived} <-n290
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=144;
                 groups=1}
      n292 {pt2=root[117] torch.ops.aten.conv2d.default (conv2d_40)}: [t537 f32 [H=48
                                                                      W=7 C=7] {pt2=root:conv2d_40} ->[n293]] =
        permute x=t536 {derived} <-n291 perm=[H<-C, W<-H, C<-W]
    group g68 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n293 {derived}: [t538 f32 [H=7 W=7 C=48] {derived} ->[n294]] =
        permute x=t537 {pt2=root:conv2d_40} <-n292 perm=[H<-W, W<-C, C<-H]
      n294 {derived}: [t539 f32 [H=7 W=7 C=48] {derived} ->[n295]] =
        batch_norm
          x=t538 {derived} <-n293
          weight=t107 {pt2=root:p_blocks_4_0_bn3_weight target=blocks.4.0.bn3.weight}
          bias=t108 {pt2=root:p_blocks_4_0_bn3_bias target=blocks.4.0.bn3.bias}
          running_mean=t220 {pt2=root:b_blocks_4_0_bn3_running_mean target=blocks.4.0.bn3.running_mean}
          running_var=t221 {pt2=root:b_blocks_4_0_bn3_running_var target=blocks.4.0.bn3.running_var}
          params={channel=C; eps=1e-05}
      n295 {pt2=root[118] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_26)}: [t540 f32 [H=48
                                                                      W=7 C=7] {pt2=root:getitem_78} ->[n296,
                                                                      n331]] =
        permute x=t539 {derived} <-n294 perm=[H<-C, W<-H, C<-W]
    group g69 torch.ops.aten.conv2d.default:
      n296 {derived}: [t541 f32 [H=7 W=7 C=48] {derived} ->[n298]] =
        permute x=t540 {pt2=root:getitem_78} <-n295 perm=[H<-W, W<-C, C<-H]
      n297 {derived}: [t542 f32 [N=288 T=1 D=1 H=1 W=1 C=48] {derived} ->[n298]] =
        permute
          x=t109 {pt2=root:p_blocks_4_1_conv_pw_weight target=blocks.4.1.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n298 {derived}: [t543 f32 [H=7 W=7 C=288] {derived} ->[n299]] =
        conv2d
          x=t541 {derived} <-n296
          weight=t542 {derived} <-n297
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=48;
                 groups=1}
      n299 {pt2=root[119] torch.ops.aten.conv2d.default (conv2d_41)}: [t544 f32 [H=288
                                                                      W=7 C=7] {pt2=root:conv2d_41} ->[n300]] =
        permute x=t543 {derived} <-n298 perm=[H<-C, W<-H, C<-W]
    group g70 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n300 {derived}: [t545 f32 [H=7 W=7 C=288] {derived} ->[n301]] =
        permute x=t544 {pt2=root:conv2d_41} <-n299 perm=[H<-W, W<-C, C<-H]
      n301 {derived}: [t546 f32 [H=7 W=7 C=288] {derived} ->[n302]] =
        batch_norm
          x=t545 {derived} <-n300
          weight=t110 {pt2=root:p_blocks_4_1_bn1_weight target=blocks.4.1.bn1.weight}
          bias=t111 {pt2=root:p_blocks_4_1_bn1_bias target=blocks.4.1.bn1.bias}
          running_mean=t223 {pt2=root:b_blocks_4_1_bn1_running_mean target=blocks.4.1.bn1.running_mean}
          running_var=t224 {pt2=root:b_blocks_4_1_bn1_running_var target=blocks.4.1.bn1.running_var}
          params={channel=C; eps=1e-05}
      n302 {pt2=root[120] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_27)}: [t547 f32 [H=288
                                                                      W=7 C=7] {pt2=root:getitem_81} ->[n303]] =
        permute x=t546 {derived} <-n301 perm=[H<-C, W<-H, C<-W]
    n303 {pt2=root[121] torch.ops.aten.hardswish.default (hardswish_13)}: [t548 f32 [H=288
                                                                      W=7 C=7] {pt2=root:hardswish_13} ->[n304]] =
      hardswish x=t547 {pt2=root:getitem_81} <-n302
    group g71 torch.ops.aten.conv2d.default:
      n304 {derived}: [t549 f32 [H=7 W=7 C=288] {derived} ->[n306]] =
        permute x=t548 {pt2=root:hardswish_13} <-n303 perm=[H<-W, W<-C, C<-H]
      n305 {derived}: [t550 f32 [N=288 T=1 D=1 H=5 W=5 C=1] {derived} ->[n306]] =
        permute
          x=t112 {pt2=root:p_blocks_4_1_conv_dw_weight target=blocks.4.1.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n306 {derived}: [t551 f32 [H=7 W=7 C=288] {derived} ->[n307]] =
        conv2d
          x=t549 {derived} <-n304
          weight=t550 {derived} <-n305
          bias=none
          params={h={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
                 w={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
                 in_channels=288;
                 groups=288}
      n307 {pt2=root[122] torch.ops.aten.conv2d.default (conv2d_42)}: [t552 f32 [H=288
                                                                      W=7 C=7] {pt2=root:conv2d_42} ->[n308]] =
        permute x=t551 {derived} <-n306 perm=[H<-C, W<-H, C<-W]
    group g72 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n308 {derived}: [t553 f32 [H=7 W=7 C=288] {derived} ->[n309]] =
        permute x=t552 {pt2=root:conv2d_42} <-n307 perm=[H<-W, W<-C, C<-H]
      n309 {derived}: [t554 f32 [H=7 W=7 C=288] {derived} ->[n310]] =
        batch_norm
          x=t553 {derived} <-n308
          weight=t113 {pt2=root:p_blocks_4_1_bn2_weight target=blocks.4.1.bn2.weight}
          bias=t114 {pt2=root:p_blocks_4_1_bn2_bias target=blocks.4.1.bn2.bias}
          running_mean=t226 {pt2=root:b_blocks_4_1_bn2_running_mean target=blocks.4.1.bn2.running_mean}
          running_var=t227 {pt2=root:b_blocks_4_1_bn2_running_var target=blocks.4.1.bn2.running_var}
          params={channel=C; eps=1e-05}
      n310 {pt2=root[123] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_28)}: [t555 f32 [H=288
                                                                      W=7 C=7] {pt2=root:getitem_84} ->[n311]] =
        permute x=t554 {derived} <-n309 perm=[H<-C, W<-H, C<-W]
    n311 {pt2=root[124] torch.ops.aten.hardswish.default (hardswish_14)}: [t556 f32 [H=288
                                                                      W=7 C=7] {pt2=root:hardswish_14} ->[n312,
                                                                      n323]] =
      hardswish x=t555 {pt2=root:getitem_84} <-n310
    n312 {pt2=root[125] torch.ops.aten.mean.dim (mean_7)}: [t557 f32 [H=288 W=1
                                                                      C=1] {pt2=root:mean_7} ->[n313]] =
      mean
        x=t556 {pt2=root:hardswish_14} <-n311
        params={dims=[W, C]; keepdim=true}
    group g73 torch.ops.aten.conv2d.default:
      n313 {derived}: [t558 f32 [C=288] {derived} ->[n315]] =
        permute x=t557 {pt2=root:mean_7} <-n312 perm=[H<-W, W<-C, C<-H]
      n314 {derived}: [t559 f32 [N=72 T=1 D=1 H=1 W=1 C=288] {derived} ->[n315]] =
        permute
          x=t115 {pt2=root:p_blocks_4_1_se_conv_reduce_weight target=blocks.4.1.se.conv_reduce.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n315 {derived}: [t560 f32 [C=72] {derived} ->[n316]] =
        conv2d
          x=t558 {derived} <-n313
          weight=t559 {derived} <-n314
          bias=t116 {pt2=root:p_blocks_4_1_se_conv_reduce_bias target=blocks.4.1.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=288;
                 groups=1}
      n316 {pt2=root[126] torch.ops.aten.conv2d.default (conv2d_43)}: [t561 f32 [H=72
                                                                      W=1 C=1] {pt2=root:conv2d_43} ->[n317]] =
        permute x=t560 {derived} <-n315 perm=[H<-C, W<-H, C<-W]
    n317 {pt2=root[127] torch.ops.aten.relu.default (relu_12)}: [t562 f32 [H=72
                                                                      W=1 C=1] {pt2=root:relu_12} ->[n318]] =
      relu x=t561 {pt2=root:conv2d_43} <-n316
    group g74 torch.ops.aten.conv2d.default:
      n318 {derived}: [t563 f32 [C=72] {derived} ->[n320]] =
        permute x=t562 {pt2=root:relu_12} <-n317 perm=[H<-W, W<-C, C<-H]
      n319 {derived}: [t564 f32 [N=288 T=1 D=1 H=1 W=1 C=72] {derived} ->[n320]] =
        permute
          x=t117 {pt2=root:p_blocks_4_1_se_conv_expand_weight target=blocks.4.1.se.conv_expand.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n320 {derived}: [t565 f32 [C=288] {derived} ->[n321]] =
        conv2d
          x=t563 {derived} <-n318
          weight=t564 {derived} <-n319
          bias=t118 {pt2=root:p_blocks_4_1_se_conv_expand_bias target=blocks.4.1.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=72;
                 groups=1}
      n321 {pt2=root[128] torch.ops.aten.conv2d.default (conv2d_44)}: [t566 f32 [H=288
                                                                      W=1 C=1] {pt2=root:conv2d_44} ->[n322]] =
        permute x=t565 {derived} <-n320 perm=[H<-C, W<-H, C<-W]
    n322 {pt2=root[129] torch.ops.aten.hardsigmoid.default (hardsigmoid_7)}: [t567 f32 [H=288
                                                                      W=1 C=1] {pt2=root:hardsigmoid_7} ->[n323]] =
      hardsigmoid x=t566 {pt2=root:conv2d_44} <-n321
    n323 {pt2=root[130] torch.ops.aten.mul.Tensor (mul_7)}: [t568 f32 [H=288
                                                                      W=7 C=7] {pt2=root:mul_7} ->[n324]] =
      mul
        a=t556 {pt2=root:hardswish_14} <-n311
        b=t567 {pt2=root:hardsigmoid_7} <-n322
    group g75 torch.ops.aten.conv2d.default:
      n324 {derived}: [t569 f32 [H=7 W=7 C=288] {derived} ->[n326]] =
        permute x=t568 {pt2=root:mul_7} <-n323 perm=[H<-W, W<-C, C<-H]
      n325 {derived}: [t570 f32 [N=48 T=1 D=1 H=1 W=1 C=288] {derived} ->[n326]] =
        permute
          x=t119 {pt2=root:p_blocks_4_1_conv_pwl_weight target=blocks.4.1.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n326 {derived}: [t571 f32 [H=7 W=7 C=48] {derived} ->[n327]] =
        conv2d
          x=t569 {derived} <-n324
          weight=t570 {derived} <-n325
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=288;
                 groups=1}
      n327 {pt2=root[131] torch.ops.aten.conv2d.default (conv2d_45)}: [t572 f32 [H=48
                                                                      W=7 C=7] {pt2=root:conv2d_45} ->[n328]] =
        permute x=t571 {derived} <-n326 perm=[H<-C, W<-H, C<-W]
    group g76 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n328 {derived}: [t573 f32 [H=7 W=7 C=48] {derived} ->[n329]] =
        permute x=t572 {pt2=root:conv2d_45} <-n327 perm=[H<-W, W<-C, C<-H]
      n329 {derived}: [t574 f32 [H=7 W=7 C=48] {derived} ->[n330]] =
        batch_norm
          x=t573 {derived} <-n328
          weight=t120 {pt2=root:p_blocks_4_1_bn3_weight target=blocks.4.1.bn3.weight}
          bias=t121 {pt2=root:p_blocks_4_1_bn3_bias target=blocks.4.1.bn3.bias}
          running_mean=t229 {pt2=root:b_blocks_4_1_bn3_running_mean target=blocks.4.1.bn3.running_mean}
          running_var=t230 {pt2=root:b_blocks_4_1_bn3_running_var target=blocks.4.1.bn3.running_var}
          params={channel=C; eps=1e-05}
      n330 {pt2=root[132] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_29)}: [t575 f32 [H=48
                                                                      W=7 C=7] {pt2=root:getitem_87} ->[n331]] =
        permute x=t574 {derived} <-n329 perm=[H<-C, W<-H, C<-W]
    n331 {pt2=root[133] torch.ops.aten.add.Tensor (add_5)}: [t576 f32 [H=48 W=7
                                                                      C=7] {pt2=root:add_5} ->[n332,
                                                                      n367]] =
      add
        a=t575 {pt2=root:getitem_87} <-n330
        b=t540 {pt2=root:getitem_78} <-n295
    group g77 torch.ops.aten.conv2d.default:
      n332 {derived}: [t577 f32 [H=7 W=7 C=48] {derived} ->[n334]] =
        permute x=t576 {pt2=root:add_5} <-n331 perm=[H<-W, W<-C, C<-H]
      n333 {derived}: [t578 f32 [N=288 T=1 D=1 H=1 W=1 C=48] {derived} ->[n334]] =
        permute
          x=t122 {pt2=root:p_blocks_4_2_conv_pw_weight target=blocks.4.2.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n334 {derived}: [t579 f32 [H=7 W=7 C=288] {derived} ->[n335]] =
        conv2d
          x=t577 {derived} <-n332
          weight=t578 {derived} <-n333
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=48;
                 groups=1}
      n335 {pt2=root[134] torch.ops.aten.conv2d.default (conv2d_46)}: [t580 f32 [H=288
                                                                      W=7 C=7] {pt2=root:conv2d_46} ->[n336]] =
        permute x=t579 {derived} <-n334 perm=[H<-C, W<-H, C<-W]
    group g78 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n336 {derived}: [t581 f32 [H=7 W=7 C=288] {derived} ->[n337]] =
        permute x=t580 {pt2=root:conv2d_46} <-n335 perm=[H<-W, W<-C, C<-H]
      n337 {derived}: [t582 f32 [H=7 W=7 C=288] {derived} ->[n338]] =
        batch_norm
          x=t581 {derived} <-n336
          weight=t123 {pt2=root:p_blocks_4_2_bn1_weight target=blocks.4.2.bn1.weight}
          bias=t124 {pt2=root:p_blocks_4_2_bn1_bias target=blocks.4.2.bn1.bias}
          running_mean=t232 {pt2=root:b_blocks_4_2_bn1_running_mean target=blocks.4.2.bn1.running_mean}
          running_var=t233 {pt2=root:b_blocks_4_2_bn1_running_var target=blocks.4.2.bn1.running_var}
          params={channel=C; eps=1e-05}
      n338 {pt2=root[135] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_30)}: [t583 f32 [H=288
                                                                      W=7 C=7] {pt2=root:getitem_90} ->[n339]] =
        permute x=t582 {derived} <-n337 perm=[H<-C, W<-H, C<-W]
    n339 {pt2=root[136] torch.ops.aten.hardswish.default (hardswish_15)}: [t584 f32 [H=288
                                                                      W=7 C=7] {pt2=root:hardswish_15} ->[n340]] =
      hardswish x=t583 {pt2=root:getitem_90} <-n338
    group g79 torch.ops.aten.conv2d.default:
      n340 {derived}: [t585 f32 [H=7 W=7 C=288] {derived} ->[n342]] =
        permute x=t584 {pt2=root:hardswish_15} <-n339 perm=[H<-W, W<-C, C<-H]
      n341 {derived}: [t586 f32 [N=288 T=1 D=1 H=5 W=5 C=1] {derived} ->[n342]] =
        permute
          x=t125 {pt2=root:p_blocks_4_2_conv_dw_weight target=blocks.4.2.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n342 {derived}: [t587 f32 [H=7 W=7 C=288] {derived} ->[n343]] =
        conv2d
          x=t585 {derived} <-n340
          weight=t586 {derived} <-n341
          bias=none
          params={h={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
                 w={kernel=5; stride=1; pad_before=2; pad_after=2; dilation=1};
                 in_channels=288;
                 groups=288}
      n343 {pt2=root[137] torch.ops.aten.conv2d.default (conv2d_47)}: [t588 f32 [H=288
                                                                      W=7 C=7] {pt2=root:conv2d_47} ->[n344]] =
        permute x=t587 {derived} <-n342 perm=[H<-C, W<-H, C<-W]
    group g80 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n344 {derived}: [t589 f32 [H=7 W=7 C=288] {derived} ->[n345]] =
        permute x=t588 {pt2=root:conv2d_47} <-n343 perm=[H<-W, W<-C, C<-H]
      n345 {derived}: [t590 f32 [H=7 W=7 C=288] {derived} ->[n346]] =
        batch_norm
          x=t589 {derived} <-n344
          weight=t126 {pt2=root:p_blocks_4_2_bn2_weight target=blocks.4.2.bn2.weight}
          bias=t127 {pt2=root:p_blocks_4_2_bn2_bias target=blocks.4.2.bn2.bias}
          running_mean=t235 {pt2=root:b_blocks_4_2_bn2_running_mean target=blocks.4.2.bn2.running_mean}
          running_var=t236 {pt2=root:b_blocks_4_2_bn2_running_var target=blocks.4.2.bn2.running_var}
          params={channel=C; eps=1e-05}
      n346 {pt2=root[138] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_31)}: [t591 f32 [H=288
                                                                      W=7 C=7] {pt2=root:getitem_93} ->[n347]] =
        permute x=t590 {derived} <-n345 perm=[H<-C, W<-H, C<-W]
    n347 {pt2=root[139] torch.ops.aten.hardswish.default (hardswish_16)}: [t592 f32 [H=288
                                                                      W=7 C=7] {pt2=root:hardswish_16} ->[n348,
                                                                      n359]] =
      hardswish x=t591 {pt2=root:getitem_93} <-n346
    n348 {pt2=root[140] torch.ops.aten.mean.dim (mean_8)}: [t593 f32 [H=288 W=1
                                                                      C=1] {pt2=root:mean_8} ->[n349]] =
      mean
        x=t592 {pt2=root:hardswish_16} <-n347
        params={dims=[W, C]; keepdim=true}
    group g81 torch.ops.aten.conv2d.default:
      n349 {derived}: [t594 f32 [C=288] {derived} ->[n351]] =
        permute x=t593 {pt2=root:mean_8} <-n348 perm=[H<-W, W<-C, C<-H]
      n350 {derived}: [t595 f32 [N=72 T=1 D=1 H=1 W=1 C=288] {derived} ->[n351]] =
        permute
          x=t128 {pt2=root:p_blocks_4_2_se_conv_reduce_weight target=blocks.4.2.se.conv_reduce.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n351 {derived}: [t596 f32 [C=72] {derived} ->[n352]] =
        conv2d
          x=t594 {derived} <-n349
          weight=t595 {derived} <-n350
          bias=t129 {pt2=root:p_blocks_4_2_se_conv_reduce_bias target=blocks.4.2.se.conv_reduce.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=288;
                 groups=1}
      n352 {pt2=root[141] torch.ops.aten.conv2d.default (conv2d_48)}: [t597 f32 [H=72
                                                                      W=1 C=1] {pt2=root:conv2d_48} ->[n353]] =
        permute x=t596 {derived} <-n351 perm=[H<-C, W<-H, C<-W]
    n353 {pt2=root[142] torch.ops.aten.relu.default (relu_13)}: [t598 f32 [H=72
                                                                      W=1 C=1] {pt2=root:relu_13} ->[n354]] =
      relu x=t597 {pt2=root:conv2d_48} <-n352
    group g82 torch.ops.aten.conv2d.default:
      n354 {derived}: [t599 f32 [C=72] {derived} ->[n356]] =
        permute x=t598 {pt2=root:relu_13} <-n353 perm=[H<-W, W<-C, C<-H]
      n355 {derived}: [t600 f32 [N=288 T=1 D=1 H=1 W=1 C=72] {derived} ->[n356]] =
        permute
          x=t130 {pt2=root:p_blocks_4_2_se_conv_expand_weight target=blocks.4.2.se.conv_expand.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n356 {derived}: [t601 f32 [C=288] {derived} ->[n357]] =
        conv2d
          x=t599 {derived} <-n354
          weight=t600 {derived} <-n355
          bias=t131 {pt2=root:p_blocks_4_2_se_conv_expand_bias target=blocks.4.2.se.conv_expand.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=72;
                 groups=1}
      n357 {pt2=root[143] torch.ops.aten.conv2d.default (conv2d_49)}: [t602 f32 [H=288
                                                                      W=1 C=1] {pt2=root:conv2d_49} ->[n358]] =
        permute x=t601 {derived} <-n356 perm=[H<-C, W<-H, C<-W]
    n358 {pt2=root[144] torch.ops.aten.hardsigmoid.default (hardsigmoid_8)}: [t603 f32 [H=288
                                                                      W=1 C=1] {pt2=root:hardsigmoid_8} ->[n359]] =
      hardsigmoid x=t602 {pt2=root:conv2d_49} <-n357
    n359 {pt2=root[145] torch.ops.aten.mul.Tensor (mul_8)}: [t604 f32 [H=288
                                                                      W=7 C=7] {pt2=root:mul_8} ->[n360]] =
      mul
        a=t592 {pt2=root:hardswish_16} <-n347
        b=t603 {pt2=root:hardsigmoid_8} <-n358
    group g83 torch.ops.aten.conv2d.default:
      n360 {derived}: [t605 f32 [H=7 W=7 C=288] {derived} ->[n362]] =
        permute x=t604 {pt2=root:mul_8} <-n359 perm=[H<-W, W<-C, C<-H]
      n361 {derived}: [t606 f32 [N=48 T=1 D=1 H=1 W=1 C=288] {derived} ->[n362]] =
        permute
          x=t132 {pt2=root:p_blocks_4_2_conv_pwl_weight target=blocks.4.2.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n362 {derived}: [t607 f32 [H=7 W=7 C=48] {derived} ->[n363]] =
        conv2d
          x=t605 {derived} <-n360
          weight=t606 {derived} <-n361
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=288;
                 groups=1}
      n363 {pt2=root[146] torch.ops.aten.conv2d.default (conv2d_50)}: [t608 f32 [H=48
                                                                      W=7 C=7] {pt2=root:conv2d_50} ->[n364]] =
        permute x=t607 {derived} <-n362 perm=[H<-C, W<-H, C<-W]
    group g84 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n364 {derived}: [t609 f32 [H=7 W=7 C=48] {derived} ->[n365]] =
        permute x=t608 {pt2=root:conv2d_50} <-n363 perm=[H<-W, W<-C, C<-H]
      n365 {derived}: [t610 f32 [H=7 W=7 C=48] {derived} ->[n366]] =
        batch_norm
          x=t609 {derived} <-n364
          weight=t133 {pt2=root:p_blocks_4_2_bn3_weight target=blocks.4.2.bn3.weight}
          bias=t134 {pt2=root:p_blocks_4_2_bn3_bias target=blocks.4.2.bn3.bias}
          running_mean=t238 {pt2=root:b_blocks_4_2_bn3_running_mean target=blocks.4.2.bn3.running_mean}
          running_var=t239 {pt2=root:b_blocks_4_2_bn3_running_var target=blocks.4.2.bn3.running_var}
          params={channel=C; eps=1e-05}
      n366 {pt2=root[147] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_32)}: [t611 f32 [H=48
                                                                      W=7 C=7] {pt2=root:getitem_96} ->[n367]] =
        permute x=t610 {derived} <-n365 perm=[H<-C, W<-H, C<-W]
    n367 {pt2=root[148] torch.ops.aten.add.Tensor (add_6)}: [t612 f32 [H=48 W=7
                                                                      C=7] {pt2=root:add_6} ->[n368]] =
      add a=t611 {pt2=root:getitem_96} <-n366 b=t576 {pt2=root:add_5} <-n331
    group g85 torch.ops.aten.conv2d.default:
      n368 {derived}: [t613 f32 [H=7 W=7 C=48] {derived} ->[n370]] =
        permute x=t612 {pt2=root:add_6} <-n367 perm=[H<-W, W<-C, C<-H]
      n369 {derived}: [t614 f32 [N=288 T=1 D=1 H=1 W=1 C=48] {derived} ->[n370]] =
        permute
          x=t135 {pt2=root:p_blocks_5_0_conv_weight target=blocks.5.0.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n370 {derived}: [t615 f32 [H=7 W=7 C=288] {derived} ->[n371]] =
        conv2d
          x=t613 {derived} <-n368
          weight=t614 {derived} <-n369
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=48;
                 groups=1}
      n371 {pt2=root[149] torch.ops.aten.conv2d.default (conv2d_51)}: [t616 f32 [H=288
                                                                      W=7 C=7] {pt2=root:conv2d_51} ->[n372]] =
        permute x=t615 {derived} <-n370 perm=[H<-C, W<-H, C<-W]
    group g86 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n372 {derived}: [t617 f32 [H=7 W=7 C=288] {derived} ->[n373]] =
        permute x=t616 {pt2=root:conv2d_51} <-n371 perm=[H<-W, W<-C, C<-H]
      n373 {derived}: [t618 f32 [H=7 W=7 C=288] {derived} ->[n374]] =
        batch_norm
          x=t617 {derived} <-n372
          weight=t136 {pt2=root:p_blocks_5_0_bn1_weight target=blocks.5.0.bn1.weight}
          bias=t137 {pt2=root:p_blocks_5_0_bn1_bias target=blocks.5.0.bn1.bias}
          running_mean=t241 {pt2=root:b_blocks_5_0_bn1_running_mean target=blocks.5.0.bn1.running_mean}
          running_var=t242 {pt2=root:b_blocks_5_0_bn1_running_var target=blocks.5.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n374 {pt2=root[150] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_33)}: [t619 f32 [H=288
                                                                      W=7 C=7] {pt2=root:getitem_99} ->[n375]] =
        permute x=t618 {derived} <-n373 perm=[H<-C, W<-H, C<-W]
    n375 {pt2=root[151] torch.ops.aten.hardswish.default (hardswish_17)}: [t620 f32 [H=288
                                                                      W=7 C=7] {pt2=root:hardswish_17} ->[n376]] =
      hardswish x=t619 {pt2=root:getitem_99} <-n374
    group g87 torch.ops.aten.adaptive_avg_pool2d.default:
      n376 {derived}: [t621 f32 [H=7 W=7 C=288] {derived} ->[n377]] =
        permute x=t620 {pt2=root:hardswish_17} <-n375 perm=[H<-W, W<-C, C<-H]
      n377 {derived}: [t622 f32 [C=288] {derived} ->[n378]] =
        adaptive_avg_pool2d
          x=t621 {derived} <-n376
          params={output_size={h=1; w=1}}
      n378 {pt2=root[152] torch.ops.aten.adaptive_avg_pool2d.default (adaptive_avg_pool2d)}: [t623 f32 [H=288
                                                                      W=1 C=1] {pt2=root:adaptive_avg_pool2d} ->[n379]] =
        permute x=t622 {derived} <-n377 perm=[H<-C, W<-H, C<-W]
    group g88 torch.ops.aten.conv2d.default:
      n379 {derived}: [t624 f32 [C=288] {derived} ->[n381]] =
        permute
          x=t623 {pt2=root:adaptive_avg_pool2d} <-n378
          perm=[H<-W, W<-C, C<-H]
      n380 {derived}: [t625 f32 [N=1024 T=1 D=1 H=1 W=1 C=288] {derived} ->[n381]] =
        permute
          x=t138 {pt2=root:p_conv_head_weight target=conv_head.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n381 {derived}: [t626 f32 [C=1024] {derived} ->[n382]] =
        conv2d
          x=t624 {derived} <-n379
          weight=t625 {derived} <-n380
          bias=t139 {pt2=root:p_conv_head_bias target=conv_head.bias}
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=288;
                 groups=1}
      n382 {pt2=root[153] torch.ops.aten.conv2d.default (conv2d_52)}: [t627 f32 [H=1024
                                                                      W=1 C=1] {pt2=root:conv2d_52} ->[n383]] =
        permute x=t626 {derived} <-n381 perm=[H<-C, W<-H, C<-W]
    n383 {pt2=root[154] torch.ops.aten.hardswish.default (hardswish_18)}: [t628 f32 [H=1024
                                                                      W=1 C=1] {pt2=root:hardswish_18} ->[n384]] =
      hardswish x=t627 {pt2=root:conv2d_52} <-n382
    n384 {pt2=root[155] torch.ops.aten.view.default (view_1)}: [t629 f32 [C=1024] {pt2=root:view_1} ->[n386]] =
      reshape x=t628 {pt2=root:hardswish_18} <-n383 params={shape=[C=1024]}
    group g89 torch.ops.aten.linear.default:
      n385 {derived}: [t630 f32 [N=1000 T=1 D=1 H=1 W=1 C=1024] {derived} ->[n386]] =
        permute
          x=t140 {pt2=root:p_classifier_weight target=classifier.weight}
          perm=[N<-W, W<-N]
      n386 {pt2=root[156] torch.ops.aten.linear.default (linear)}: [t631 f32 [C=1000] {pt2=root:linear}] =
        linear
          x=t629 {pt2=root:view_1} <-n384
          weight=t630 {derived} <-n385
          bias=t141 {pt2=root:p_classifier_bias target=classifier.bias}
          params={in_features=1024}
  outputs: [t631 f32 [C=1000] {pt2=root:linear} <-n386]
