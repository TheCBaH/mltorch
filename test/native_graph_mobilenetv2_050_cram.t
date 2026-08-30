MobileNetV2-050's import, printed in full as RegNetX-002's is above
(`test/native_graph_regnetx_002_cram.t`) — it stands in for the retired
mobilenet_v2 role model, same architecture, just narrower (`width_mult`).
MobileNetV2 uses `hardtanh` (relu6, bounds `[0, 6]`) rather than RegNetX's
plain `relu`.

  $ ../bin/native_graph.exe print --pt2 "$PT2_DATA/mobilenetv2_050/mobilenetv2_050.pt2"
  native graph: inputs=315 constants=314 nodes=415 outputs=1
  PT2 provenance: tensor-origins=467 captured-targets=314 node-origins=152
  graph
  inputs:
    [t0 f32 [D=16 H=3 W=3 C=3] {pt2=root:p_conv_stem_weight target=conv_stem.weight} ->[n1] constant,
     t1 f32 [C=16] {pt2=root:p_bn1_weight target=bn1.weight} ->[n5] constant,
     t2 f32 [C=16] {pt2=root:p_bn1_bias target=bn1.bias} ->[n5] constant,
     t3 f32 [D=16 H=1 W=3 C=3] {pt2=root:p_blocks_0_0_conv_dw_weight target=blocks.0.0.conv_dw.weight} ->[n9] constant,
     t4 f32 [C=16] {pt2=root:p_blocks_0_0_bn1_weight target=blocks.0.0.bn1.weight} ->[n13] constant,
     t5 f32 [C=16] {pt2=root:p_blocks_0_0_bn1_bias target=blocks.0.0.bn1.bias} ->[n13] constant,
     t6 f32 [D=8 H=16 W=1 C=1] {pt2=root:p_blocks_0_0_conv_pw_weight target=blocks.0.0.conv_pw.weight} ->[n17] constant,
     t7 f32 [C=8] {pt2=root:p_blocks_0_0_bn2_weight target=blocks.0.0.bn2.weight} ->[n21] constant,
     t8 f32 [C=8] {pt2=root:p_blocks_0_0_bn2_bias target=blocks.0.0.bn2.bias} ->[n21] constant,
     t9 f32 [D=48 H=8 W=1 C=1] {pt2=root:p_blocks_1_0_conv_pw_weight target=blocks.1.0.conv_pw.weight} ->[n24] constant,
     t10 f32 [C=48] {pt2=root:p_blocks_1_0_bn1_weight target=blocks.1.0.bn1.weight} ->[n28] constant,
     t11 f32 [C=48] {pt2=root:p_blocks_1_0_bn1_bias target=blocks.1.0.bn1.bias} ->[n28] constant,
     t12 f32 [D=48 H=1 W=3 C=3] {pt2=root:p_blocks_1_0_conv_dw_weight target=blocks.1.0.conv_dw.weight} ->[n32] constant,
     t13 f32 [C=48] {pt2=root:p_blocks_1_0_bn2_weight target=blocks.1.0.bn2.weight} ->[n36] constant,
     t14 f32 [C=48] {pt2=root:p_blocks_1_0_bn2_bias target=blocks.1.0.bn2.bias} ->[n36] constant,
     t15 f32 [D=16 H=48 W=1 C=1] {pt2=root:p_blocks_1_0_conv_pwl_weight target=blocks.1.0.conv_pwl.weight} ->[n40] constant,
     t16 f32 [C=16] {pt2=root:p_blocks_1_0_bn3_weight target=blocks.1.0.bn3.weight} ->[n44] constant,
     t17 f32 [C=16] {pt2=root:p_blocks_1_0_bn3_bias target=blocks.1.0.bn3.bias} ->[n44] constant,
     t18 f32 [D=96 H=16 W=1 C=1] {pt2=root:p_blocks_1_1_conv_pw_weight target=blocks.1.1.conv_pw.weight} ->[n47] constant,
     t19 f32 [C=96] {pt2=root:p_blocks_1_1_bn1_weight target=blocks.1.1.bn1.weight} ->[n51] constant,
     t20 f32 [C=96] {pt2=root:p_blocks_1_1_bn1_bias target=blocks.1.1.bn1.bias} ->[n51] constant,
     t21 f32 [D=96 H=1 W=3 C=3] {pt2=root:p_blocks_1_1_conv_dw_weight target=blocks.1.1.conv_dw.weight} ->[n55] constant,
     t22 f32 [C=96] {pt2=root:p_blocks_1_1_bn2_weight target=blocks.1.1.bn2.weight} ->[n59] constant,
     t23 f32 [C=96] {pt2=root:p_blocks_1_1_bn2_bias target=blocks.1.1.bn2.bias} ->[n59] constant,
     t24 f32 [D=16 H=96 W=1 C=1] {pt2=root:p_blocks_1_1_conv_pwl_weight target=blocks.1.1.conv_pwl.weight} ->[n63] constant,
     t25 f32 [C=16] {pt2=root:p_blocks_1_1_bn3_weight target=blocks.1.1.bn3.weight} ->[n67] constant,
     t26 f32 [C=16] {pt2=root:p_blocks_1_1_bn3_bias target=blocks.1.1.bn3.bias} ->[n67] constant,
     t27 f32 [D=96 H=16 W=1 C=1] {pt2=root:p_blocks_2_0_conv_pw_weight target=blocks.2.0.conv_pw.weight} ->[n71] constant,
     t28 f32 [C=96] {pt2=root:p_blocks_2_0_bn1_weight target=blocks.2.0.bn1.weight} ->[n75] constant,
     t29 f32 [C=96] {pt2=root:p_blocks_2_0_bn1_bias target=blocks.2.0.bn1.bias} ->[n75] constant,
     t30 f32 [D=96 H=1 W=3 C=3] {pt2=root:p_blocks_2_0_conv_dw_weight target=blocks.2.0.conv_dw.weight} ->[n79] constant,
     t31 f32 [C=96] {pt2=root:p_blocks_2_0_bn2_weight target=blocks.2.0.bn2.weight} ->[n83] constant,
     t32 f32 [C=96] {pt2=root:p_blocks_2_0_bn2_bias target=blocks.2.0.bn2.bias} ->[n83] constant,
     t33 f32 [D=16 H=96 W=1 C=1] {pt2=root:p_blocks_2_0_conv_pwl_weight target=blocks.2.0.conv_pwl.weight} ->[n87] constant,
     t34 f32 [C=16] {pt2=root:p_blocks_2_0_bn3_weight target=blocks.2.0.bn3.weight} ->[n91] constant,
     t35 f32 [C=16] {pt2=root:p_blocks_2_0_bn3_bias target=blocks.2.0.bn3.bias} ->[n91] constant,
     t36 f32 [D=96 H=16 W=1 C=1] {pt2=root:p_blocks_2_1_conv_pw_weight target=blocks.2.1.conv_pw.weight} ->[n94] constant,
     t37 f32 [C=96] {pt2=root:p_blocks_2_1_bn1_weight target=blocks.2.1.bn1.weight} ->[n98] constant,
     t38 f32 [C=96] {pt2=root:p_blocks_2_1_bn1_bias target=blocks.2.1.bn1.bias} ->[n98] constant,
     t39 f32 [D=96 H=1 W=3 C=3] {pt2=root:p_blocks_2_1_conv_dw_weight target=blocks.2.1.conv_dw.weight} ->[n102] constant,
     t40 f32 [C=96] {pt2=root:p_blocks_2_1_bn2_weight target=blocks.2.1.bn2.weight} ->[n106] constant,
     t41 f32 [C=96] {pt2=root:p_blocks_2_1_bn2_bias target=blocks.2.1.bn2.bias} ->[n106] constant,
     t42 f32 [D=16 H=96 W=1 C=1] {pt2=root:p_blocks_2_1_conv_pwl_weight target=blocks.2.1.conv_pwl.weight} ->[n110] constant,
     t43 f32 [C=16] {pt2=root:p_blocks_2_1_bn3_weight target=blocks.2.1.bn3.weight} ->[n114] constant,
     t44 f32 [C=16] {pt2=root:p_blocks_2_1_bn3_bias target=blocks.2.1.bn3.bias} ->[n114] constant,
     t45 f32 [D=96 H=16 W=1 C=1] {pt2=root:p_blocks_2_2_conv_pw_weight target=blocks.2.2.conv_pw.weight} ->[n118] constant,
     t46 f32 [C=96] {pt2=root:p_blocks_2_2_bn1_weight target=blocks.2.2.bn1.weight} ->[n122] constant,
     t47 f32 [C=96] {pt2=root:p_blocks_2_2_bn1_bias target=blocks.2.2.bn1.bias} ->[n122] constant,
     t48 f32 [D=96 H=1 W=3 C=3] {pt2=root:p_blocks_2_2_conv_dw_weight target=blocks.2.2.conv_dw.weight} ->[n126] constant,
     t49 f32 [C=96] {pt2=root:p_blocks_2_2_bn2_weight target=blocks.2.2.bn2.weight} ->[n130] constant,
     t50 f32 [C=96] {pt2=root:p_blocks_2_2_bn2_bias target=blocks.2.2.bn2.bias} ->[n130] constant,
     t51 f32 [D=16 H=96 W=1 C=1] {pt2=root:p_blocks_2_2_conv_pwl_weight target=blocks.2.2.conv_pwl.weight} ->[n134] constant,
     t52 f32 [C=16] {pt2=root:p_blocks_2_2_bn3_weight target=blocks.2.2.bn3.weight} ->[n138] constant,
     t53 f32 [C=16] {pt2=root:p_blocks_2_2_bn3_bias target=blocks.2.2.bn3.bias} ->[n138] constant,
     t54 f32 [D=96 H=16 W=1 C=1] {pt2=root:p_blocks_3_0_conv_pw_weight target=blocks.3.0.conv_pw.weight} ->[n142] constant,
     t55 f32 [C=96] {pt2=root:p_blocks_3_0_bn1_weight target=blocks.3.0.bn1.weight} ->[n146] constant,
     t56 f32 [C=96] {pt2=root:p_blocks_3_0_bn1_bias target=blocks.3.0.bn1.bias} ->[n146] constant,
     t57 f32 [D=96 H=1 W=3 C=3] {pt2=root:p_blocks_3_0_conv_dw_weight target=blocks.3.0.conv_dw.weight} ->[n150] constant,
     t58 f32 [C=96] {pt2=root:p_blocks_3_0_bn2_weight target=blocks.3.0.bn2.weight} ->[n154] constant,
     t59 f32 [C=96] {pt2=root:p_blocks_3_0_bn2_bias target=blocks.3.0.bn2.bias} ->[n154] constant,
     t60 f32 [D=32 H=96 W=1 C=1] {pt2=root:p_blocks_3_0_conv_pwl_weight target=blocks.3.0.conv_pwl.weight} ->[n158] constant,
     t61 f32 [C=32] {pt2=root:p_blocks_3_0_bn3_weight target=blocks.3.0.bn3.weight} ->[n162] constant,
     t62 f32 [C=32] {pt2=root:p_blocks_3_0_bn3_bias target=blocks.3.0.bn3.bias} ->[n162] constant,
     t63 f32 [D=192 H=32 W=1 C=1] {pt2=root:p_blocks_3_1_conv_pw_weight target=blocks.3.1.conv_pw.weight} ->[n165] constant,
     t64 f32 [C=192] {pt2=root:p_blocks_3_1_bn1_weight target=blocks.3.1.bn1.weight} ->[n169] constant,
     t65 f32 [C=192] {pt2=root:p_blocks_3_1_bn1_bias target=blocks.3.1.bn1.bias} ->[n169] constant,
     t66 f32 [D=192 H=1 W=3 C=3] {pt2=root:p_blocks_3_1_conv_dw_weight target=blocks.3.1.conv_dw.weight} ->[n173] constant,
     t67 f32 [C=192] {pt2=root:p_blocks_3_1_bn2_weight target=blocks.3.1.bn2.weight} ->[n177] constant,
     t68 f32 [C=192] {pt2=root:p_blocks_3_1_bn2_bias target=blocks.3.1.bn2.bias} ->[n177] constant,
     t69 f32 [D=32 H=192 W=1 C=1] {pt2=root:p_blocks_3_1_conv_pwl_weight target=blocks.3.1.conv_pwl.weight} ->[n181] constant,
     t70 f32 [C=32] {pt2=root:p_blocks_3_1_bn3_weight target=blocks.3.1.bn3.weight} ->[n185] constant,
     t71 f32 [C=32] {pt2=root:p_blocks_3_1_bn3_bias target=blocks.3.1.bn3.bias} ->[n185] constant,
     t72 f32 [D=192 H=32 W=1 C=1] {pt2=root:p_blocks_3_2_conv_pw_weight target=blocks.3.2.conv_pw.weight} ->[n189] constant,
     t73 f32 [C=192] {pt2=root:p_blocks_3_2_bn1_weight target=blocks.3.2.bn1.weight} ->[n193] constant,
     t74 f32 [C=192] {pt2=root:p_blocks_3_2_bn1_bias target=blocks.3.2.bn1.bias} ->[n193] constant,
     t75 f32 [D=192 H=1 W=3 C=3] {pt2=root:p_blocks_3_2_conv_dw_weight target=blocks.3.2.conv_dw.weight} ->[n197] constant,
     t76 f32 [C=192] {pt2=root:p_blocks_3_2_bn2_weight target=blocks.3.2.bn2.weight} ->[n201] constant,
     t77 f32 [C=192] {pt2=root:p_blocks_3_2_bn2_bias target=blocks.3.2.bn2.bias} ->[n201] constant,
     t78 f32 [D=32 H=192 W=1 C=1] {pt2=root:p_blocks_3_2_conv_pwl_weight target=blocks.3.2.conv_pwl.weight} ->[n205] constant,
     t79 f32 [C=32] {pt2=root:p_blocks_3_2_bn3_weight target=blocks.3.2.bn3.weight} ->[n209] constant,
     t80 f32 [C=32] {pt2=root:p_blocks_3_2_bn3_bias target=blocks.3.2.bn3.bias} ->[n209] constant,
     t81 f32 [D=192 H=32 W=1 C=1] {pt2=root:p_blocks_3_3_conv_pw_weight target=blocks.3.3.conv_pw.weight} ->[n213] constant,
     t82 f32 [C=192] {pt2=root:p_blocks_3_3_bn1_weight target=blocks.3.3.bn1.weight} ->[n217] constant,
     t83 f32 [C=192] {pt2=root:p_blocks_3_3_bn1_bias target=blocks.3.3.bn1.bias} ->[n217] constant,
     t84 f32 [D=192 H=1 W=3 C=3] {pt2=root:p_blocks_3_3_conv_dw_weight target=blocks.3.3.conv_dw.weight} ->[n221] constant,
     t85 f32 [C=192] {pt2=root:p_blocks_3_3_bn2_weight target=blocks.3.3.bn2.weight} ->[n225] constant,
     t86 f32 [C=192] {pt2=root:p_blocks_3_3_bn2_bias target=blocks.3.3.bn2.bias} ->[n225] constant,
     t87 f32 [D=32 H=192 W=1 C=1] {pt2=root:p_blocks_3_3_conv_pwl_weight target=blocks.3.3.conv_pwl.weight} ->[n229] constant,
     t88 f32 [C=32] {pt2=root:p_blocks_3_3_bn3_weight target=blocks.3.3.bn3.weight} ->[n233] constant,
     t89 f32 [C=32] {pt2=root:p_blocks_3_3_bn3_bias target=blocks.3.3.bn3.bias} ->[n233] constant,
     t90 f32 [D=192 H=32 W=1 C=1] {pt2=root:p_blocks_4_0_conv_pw_weight target=blocks.4.0.conv_pw.weight} ->[n237] constant,
     t91 f32 [C=192] {pt2=root:p_blocks_4_0_bn1_weight target=blocks.4.0.bn1.weight} ->[n241] constant,
     t92 f32 [C=192] {pt2=root:p_blocks_4_0_bn1_bias target=blocks.4.0.bn1.bias} ->[n241] constant,
     t93 f32 [D=192 H=1 W=3 C=3] {pt2=root:p_blocks_4_0_conv_dw_weight target=blocks.4.0.conv_dw.weight} ->[n245] constant,
     t94 f32 [C=192] {pt2=root:p_blocks_4_0_bn2_weight target=blocks.4.0.bn2.weight} ->[n249] constant,
     t95 f32 [C=192] {pt2=root:p_blocks_4_0_bn2_bias target=blocks.4.0.bn2.bias} ->[n249] constant,
     t96 f32 [D=48 H=192 W=1 C=1] {pt2=root:p_blocks_4_0_conv_pwl_weight target=blocks.4.0.conv_pwl.weight} ->[n253] constant,
     t97 f32 [C=48] {pt2=root:p_blocks_4_0_bn3_weight target=blocks.4.0.bn3.weight} ->[n257] constant,
     t98 f32 [C=48] {pt2=root:p_blocks_4_0_bn3_bias target=blocks.4.0.bn3.bias} ->[n257] constant,
     t99 f32 [D=288 H=48 W=1 C=1] {pt2=root:p_blocks_4_1_conv_pw_weight target=blocks.4.1.conv_pw.weight} ->[n260] constant,
     t100 f32 [C=288] {pt2=root:p_blocks_4_1_bn1_weight target=blocks.4.1.bn1.weight} ->[n264] constant,
     t101 f32 [C=288] {pt2=root:p_blocks_4_1_bn1_bias target=blocks.4.1.bn1.bias} ->[n264] constant,
     t102 f32 [D=288 H=1 W=3 C=3] {pt2=root:p_blocks_4_1_conv_dw_weight target=blocks.4.1.conv_dw.weight} ->[n268] constant,
     t103 f32 [C=288] {pt2=root:p_blocks_4_1_bn2_weight target=blocks.4.1.bn2.weight} ->[n272] constant,
     t104 f32 [C=288] {pt2=root:p_blocks_4_1_bn2_bias target=blocks.4.1.bn2.bias} ->[n272] constant,
     t105 f32 [D=48 H=288 W=1 C=1] {pt2=root:p_blocks_4_1_conv_pwl_weight target=blocks.4.1.conv_pwl.weight} ->[n276] constant,
     t106 f32 [C=48] {pt2=root:p_blocks_4_1_bn3_weight target=blocks.4.1.bn3.weight} ->[n280] constant,
     t107 f32 [C=48] {pt2=root:p_blocks_4_1_bn3_bias target=blocks.4.1.bn3.bias} ->[n280] constant,
     t108 f32 [D=288 H=48 W=1 C=1] {pt2=root:p_blocks_4_2_conv_pw_weight target=blocks.4.2.conv_pw.weight} ->[n284] constant,
     t109 f32 [C=288] {pt2=root:p_blocks_4_2_bn1_weight target=blocks.4.2.bn1.weight} ->[n288] constant,
     t110 f32 [C=288] {pt2=root:p_blocks_4_2_bn1_bias target=blocks.4.2.bn1.bias} ->[n288] constant,
     t111 f32 [D=288 H=1 W=3 C=3] {pt2=root:p_blocks_4_2_conv_dw_weight target=blocks.4.2.conv_dw.weight} ->[n292] constant,
     t112 f32 [C=288] {pt2=root:p_blocks_4_2_bn2_weight target=blocks.4.2.bn2.weight} ->[n296] constant,
     t113 f32 [C=288] {pt2=root:p_blocks_4_2_bn2_bias target=blocks.4.2.bn2.bias} ->[n296] constant,
     t114 f32 [D=48 H=288 W=1 C=1] {pt2=root:p_blocks_4_2_conv_pwl_weight target=blocks.4.2.conv_pwl.weight} ->[n300] constant,
     t115 f32 [C=48] {pt2=root:p_blocks_4_2_bn3_weight target=blocks.4.2.bn3.weight} ->[n304] constant,
     t116 f32 [C=48] {pt2=root:p_blocks_4_2_bn3_bias target=blocks.4.2.bn3.bias} ->[n304] constant,
     t117 f32 [D=288 H=48 W=1 C=1] {pt2=root:p_blocks_5_0_conv_pw_weight target=blocks.5.0.conv_pw.weight} ->[n308] constant,
     t118 f32 [C=288] {pt2=root:p_blocks_5_0_bn1_weight target=blocks.5.0.bn1.weight} ->[n312] constant,
     t119 f32 [C=288] {pt2=root:p_blocks_5_0_bn1_bias target=blocks.5.0.bn1.bias} ->[n312] constant,
     t120 f32 [D=288 H=1 W=3 C=3] {pt2=root:p_blocks_5_0_conv_dw_weight target=blocks.5.0.conv_dw.weight} ->[n316] constant,
     t121 f32 [C=288] {pt2=root:p_blocks_5_0_bn2_weight target=blocks.5.0.bn2.weight} ->[n320] constant,
     t122 f32 [C=288] {pt2=root:p_blocks_5_0_bn2_bias target=blocks.5.0.bn2.bias} ->[n320] constant,
     t123 f32 [D=80 H=288 W=1 C=1] {pt2=root:p_blocks_5_0_conv_pwl_weight target=blocks.5.0.conv_pwl.weight} ->[n324] constant,
     t124 f32 [C=80] {pt2=root:p_blocks_5_0_bn3_weight target=blocks.5.0.bn3.weight} ->[n328] constant,
     t125 f32 [C=80] {pt2=root:p_blocks_5_0_bn3_bias target=blocks.5.0.bn3.bias} ->[n328] constant,
     t126 f32 [D=480 H=80 W=1 C=1] {pt2=root:p_blocks_5_1_conv_pw_weight target=blocks.5.1.conv_pw.weight} ->[n331] constant,
     t127 f32 [C=480] {pt2=root:p_blocks_5_1_bn1_weight target=blocks.5.1.bn1.weight} ->[n335] constant,
     t128 f32 [C=480] {pt2=root:p_blocks_5_1_bn1_bias target=blocks.5.1.bn1.bias} ->[n335] constant,
     t129 f32 [D=480 H=1 W=3 C=3] {pt2=root:p_blocks_5_1_conv_dw_weight target=blocks.5.1.conv_dw.weight} ->[n339] constant,
     t130 f32 [C=480] {pt2=root:p_blocks_5_1_bn2_weight target=blocks.5.1.bn2.weight} ->[n343] constant,
     t131 f32 [C=480] {pt2=root:p_blocks_5_1_bn2_bias target=blocks.5.1.bn2.bias} ->[n343] constant,
     t132 f32 [D=80 H=480 W=1 C=1] {pt2=root:p_blocks_5_1_conv_pwl_weight target=blocks.5.1.conv_pwl.weight} ->[n347] constant,
     t133 f32 [C=80] {pt2=root:p_blocks_5_1_bn3_weight target=blocks.5.1.bn3.weight} ->[n351] constant,
     t134 f32 [C=80] {pt2=root:p_blocks_5_1_bn3_bias target=blocks.5.1.bn3.bias} ->[n351] constant,
     t135 f32 [D=480 H=80 W=1 C=1] {pt2=root:p_blocks_5_2_conv_pw_weight target=blocks.5.2.conv_pw.weight} ->[n355] constant,
     t136 f32 [C=480] {pt2=root:p_blocks_5_2_bn1_weight target=blocks.5.2.bn1.weight} ->[n359] constant,
     t137 f32 [C=480] {pt2=root:p_blocks_5_2_bn1_bias target=blocks.5.2.bn1.bias} ->[n359] constant,
     t138 f32 [D=480 H=1 W=3 C=3] {pt2=root:p_blocks_5_2_conv_dw_weight target=blocks.5.2.conv_dw.weight} ->[n363] constant,
     t139 f32 [C=480] {pt2=root:p_blocks_5_2_bn2_weight target=blocks.5.2.bn2.weight} ->[n367] constant,
     t140 f32 [C=480] {pt2=root:p_blocks_5_2_bn2_bias target=blocks.5.2.bn2.bias} ->[n367] constant,
     t141 f32 [D=80 H=480 W=1 C=1] {pt2=root:p_blocks_5_2_conv_pwl_weight target=blocks.5.2.conv_pwl.weight} ->[n371] constant,
     t142 f32 [C=80] {pt2=root:p_blocks_5_2_bn3_weight target=blocks.5.2.bn3.weight} ->[n375] constant,
     t143 f32 [C=80] {pt2=root:p_blocks_5_2_bn3_bias target=blocks.5.2.bn3.bias} ->[n375] constant,
     t144 f32 [D=480 H=80 W=1 C=1] {pt2=root:p_blocks_6_0_conv_pw_weight target=blocks.6.0.conv_pw.weight} ->[n379] constant,
     t145 f32 [C=480] {pt2=root:p_blocks_6_0_bn1_weight target=blocks.6.0.bn1.weight} ->[n383] constant,
     t146 f32 [C=480] {pt2=root:p_blocks_6_0_bn1_bias target=blocks.6.0.bn1.bias} ->[n383] constant,
     t147 f32 [D=480 H=1 W=3 C=3] {pt2=root:p_blocks_6_0_conv_dw_weight target=blocks.6.0.conv_dw.weight} ->[n387] constant,
     t148 f32 [C=480] {pt2=root:p_blocks_6_0_bn2_weight target=blocks.6.0.bn2.weight} ->[n391] constant,
     t149 f32 [C=480] {pt2=root:p_blocks_6_0_bn2_bias target=blocks.6.0.bn2.bias} ->[n391] constant,
     t150 f32 [D=160 H=480 W=1 C=1] {pt2=root:p_blocks_6_0_conv_pwl_weight target=blocks.6.0.conv_pwl.weight} ->[n395] constant,
     t151 f32 [C=160] {pt2=root:p_blocks_6_0_bn3_weight target=blocks.6.0.bn3.weight} ->[n399] constant,
     t152 f32 [C=160] {pt2=root:p_blocks_6_0_bn3_bias target=blocks.6.0.bn3.bias} ->[n399] constant,
     t153 f32 [D=1280 H=160 W=1 C=1] {pt2=root:p_conv_head_weight target=conv_head.weight} ->[n402] constant,
     t154 f32 [C=1280] {pt2=root:p_bn2_weight target=bn2.weight} ->[n406] constant,
     t155 f32 [C=1280] {pt2=root:p_bn2_bias target=bn2.bias} ->[n406] constant,
     t156 f32 [W=1000 C=1280] {pt2=root:p_classifier_weight target=classifier.weight} ->[n413] constant,
     t157 f32 [C=1000] {pt2=root:p_classifier_bias target=classifier.bias} ->[n414] constant,
     t158 f32 [C=16] {pt2=root:b_bn1_running_mean target=bn1.running_mean} ->[n5] constant,
     t159 f32 [C=16] {pt2=root:b_bn1_running_var target=bn1.running_var} ->[n5] constant,
     t160 i64 [C=1] {pt2=root:b_bn1_num_batches_tracked target=bn1.num_batches_tracked} constant,
     t161 f32 [C=16] {pt2=root:b_blocks_0_0_bn1_running_mean target=blocks.0.0.bn1.running_mean} ->[n13] constant,
     t162 f32 [C=16] {pt2=root:b_blocks_0_0_bn1_running_var target=blocks.0.0.bn1.running_var} ->[n13] constant,
     t163 i64 [C=1] {pt2=root:b_blocks_0_0_bn1_num_batches_tracked target=blocks.0.0.bn1.num_batches_tracked} constant,
     t164 f32 [C=8] {pt2=root:b_blocks_0_0_bn2_running_mean target=blocks.0.0.bn2.running_mean} ->[n21] constant,
     t165 f32 [C=8] {pt2=root:b_blocks_0_0_bn2_running_var target=blocks.0.0.bn2.running_var} ->[n21] constant,
     t166 i64 [C=1] {pt2=root:b_blocks_0_0_bn2_num_batches_tracked target=blocks.0.0.bn2.num_batches_tracked} constant,
     t167 f32 [C=48] {pt2=root:b_blocks_1_0_bn1_running_mean target=blocks.1.0.bn1.running_mean} ->[n28] constant,
     t168 f32 [C=48] {pt2=root:b_blocks_1_0_bn1_running_var target=blocks.1.0.bn1.running_var} ->[n28] constant,
     t169 i64 [C=1] {pt2=root:b_blocks_1_0_bn1_num_batches_tracked target=blocks.1.0.bn1.num_batches_tracked} constant,
     t170 f32 [C=48] {pt2=root:b_blocks_1_0_bn2_running_mean target=blocks.1.0.bn2.running_mean} ->[n36] constant,
     t171 f32 [C=48] {pt2=root:b_blocks_1_0_bn2_running_var target=blocks.1.0.bn2.running_var} ->[n36] constant,
     t172 i64 [C=1] {pt2=root:b_blocks_1_0_bn2_num_batches_tracked target=blocks.1.0.bn2.num_batches_tracked} constant,
     t173 f32 [C=16] {pt2=root:b_blocks_1_0_bn3_running_mean target=blocks.1.0.bn3.running_mean} ->[n44] constant,
     t174 f32 [C=16] {pt2=root:b_blocks_1_0_bn3_running_var target=blocks.1.0.bn3.running_var} ->[n44] constant,
     t175 i64 [C=1] {pt2=root:b_blocks_1_0_bn3_num_batches_tracked target=blocks.1.0.bn3.num_batches_tracked} constant,
     t176 f32 [C=96] {pt2=root:b_blocks_1_1_bn1_running_mean target=blocks.1.1.bn1.running_mean} ->[n51] constant,
     t177 f32 [C=96] {pt2=root:b_blocks_1_1_bn1_running_var target=blocks.1.1.bn1.running_var} ->[n51] constant,
     t178 i64 [C=1] {pt2=root:b_blocks_1_1_bn1_num_batches_tracked target=blocks.1.1.bn1.num_batches_tracked} constant,
     t179 f32 [C=96] {pt2=root:b_blocks_1_1_bn2_running_mean target=blocks.1.1.bn2.running_mean} ->[n59] constant,
     t180 f32 [C=96] {pt2=root:b_blocks_1_1_bn2_running_var target=blocks.1.1.bn2.running_var} ->[n59] constant,
     t181 i64 [C=1] {pt2=root:b_blocks_1_1_bn2_num_batches_tracked target=blocks.1.1.bn2.num_batches_tracked} constant,
     t182 f32 [C=16] {pt2=root:b_blocks_1_1_bn3_running_mean target=blocks.1.1.bn3.running_mean} ->[n67] constant,
     t183 f32 [C=16] {pt2=root:b_blocks_1_1_bn3_running_var target=blocks.1.1.bn3.running_var} ->[n67] constant,
     t184 i64 [C=1] {pt2=root:b_blocks_1_1_bn3_num_batches_tracked target=blocks.1.1.bn3.num_batches_tracked} constant,
     t185 f32 [C=96] {pt2=root:b_blocks_2_0_bn1_running_mean target=blocks.2.0.bn1.running_mean} ->[n75] constant,
     t186 f32 [C=96] {pt2=root:b_blocks_2_0_bn1_running_var target=blocks.2.0.bn1.running_var} ->[n75] constant,
     t187 i64 [C=1] {pt2=root:b_blocks_2_0_bn1_num_batches_tracked target=blocks.2.0.bn1.num_batches_tracked} constant,
     t188 f32 [C=96] {pt2=root:b_blocks_2_0_bn2_running_mean target=blocks.2.0.bn2.running_mean} ->[n83] constant,
     t189 f32 [C=96] {pt2=root:b_blocks_2_0_bn2_running_var target=blocks.2.0.bn2.running_var} ->[n83] constant,
     t190 i64 [C=1] {pt2=root:b_blocks_2_0_bn2_num_batches_tracked target=blocks.2.0.bn2.num_batches_tracked} constant,
     t191 f32 [C=16] {pt2=root:b_blocks_2_0_bn3_running_mean target=blocks.2.0.bn3.running_mean} ->[n91] constant,
     t192 f32 [C=16] {pt2=root:b_blocks_2_0_bn3_running_var target=blocks.2.0.bn3.running_var} ->[n91] constant,
     t193 i64 [C=1] {pt2=root:b_blocks_2_0_bn3_num_batches_tracked target=blocks.2.0.bn3.num_batches_tracked} constant,
     t194 f32 [C=96] {pt2=root:b_blocks_2_1_bn1_running_mean target=blocks.2.1.bn1.running_mean} ->[n98] constant,
     t195 f32 [C=96] {pt2=root:b_blocks_2_1_bn1_running_var target=blocks.2.1.bn1.running_var} ->[n98] constant,
     t196 i64 [C=1] {pt2=root:b_blocks_2_1_bn1_num_batches_tracked target=blocks.2.1.bn1.num_batches_tracked} constant,
     t197 f32 [C=96] {pt2=root:b_blocks_2_1_bn2_running_mean target=blocks.2.1.bn2.running_mean} ->[n106] constant,
     t198 f32 [C=96] {pt2=root:b_blocks_2_1_bn2_running_var target=blocks.2.1.bn2.running_var} ->[n106] constant,
     t199 i64 [C=1] {pt2=root:b_blocks_2_1_bn2_num_batches_tracked target=blocks.2.1.bn2.num_batches_tracked} constant,
     t200 f32 [C=16] {pt2=root:b_blocks_2_1_bn3_running_mean target=blocks.2.1.bn3.running_mean} ->[n114] constant,
     t201 f32 [C=16] {pt2=root:b_blocks_2_1_bn3_running_var target=blocks.2.1.bn3.running_var} ->[n114] constant,
     t202 i64 [C=1] {pt2=root:b_blocks_2_1_bn3_num_batches_tracked target=blocks.2.1.bn3.num_batches_tracked} constant,
     t203 f32 [C=96] {pt2=root:b_blocks_2_2_bn1_running_mean target=blocks.2.2.bn1.running_mean} ->[n122] constant,
     t204 f32 [C=96] {pt2=root:b_blocks_2_2_bn1_running_var target=blocks.2.2.bn1.running_var} ->[n122] constant,
     t205 i64 [C=1] {pt2=root:b_blocks_2_2_bn1_num_batches_tracked target=blocks.2.2.bn1.num_batches_tracked} constant,
     t206 f32 [C=96] {pt2=root:b_blocks_2_2_bn2_running_mean target=blocks.2.2.bn2.running_mean} ->[n130] constant,
     t207 f32 [C=96] {pt2=root:b_blocks_2_2_bn2_running_var target=blocks.2.2.bn2.running_var} ->[n130] constant,
     t208 i64 [C=1] {pt2=root:b_blocks_2_2_bn2_num_batches_tracked target=blocks.2.2.bn2.num_batches_tracked} constant,
     t209 f32 [C=16] {pt2=root:b_blocks_2_2_bn3_running_mean target=blocks.2.2.bn3.running_mean} ->[n138] constant,
     t210 f32 [C=16] {pt2=root:b_blocks_2_2_bn3_running_var target=blocks.2.2.bn3.running_var} ->[n138] constant,
     t211 i64 [C=1] {pt2=root:b_blocks_2_2_bn3_num_batches_tracked target=blocks.2.2.bn3.num_batches_tracked} constant,
     t212 f32 [C=96] {pt2=root:b_blocks_3_0_bn1_running_mean target=blocks.3.0.bn1.running_mean} ->[n146] constant,
     t213 f32 [C=96] {pt2=root:b_blocks_3_0_bn1_running_var target=blocks.3.0.bn1.running_var} ->[n146] constant,
     t214 i64 [C=1] {pt2=root:b_blocks_3_0_bn1_num_batches_tracked target=blocks.3.0.bn1.num_batches_tracked} constant,
     t215 f32 [C=96] {pt2=root:b_blocks_3_0_bn2_running_mean target=blocks.3.0.bn2.running_mean} ->[n154] constant,
     t216 f32 [C=96] {pt2=root:b_blocks_3_0_bn2_running_var target=blocks.3.0.bn2.running_var} ->[n154] constant,
     t217 i64 [C=1] {pt2=root:b_blocks_3_0_bn2_num_batches_tracked target=blocks.3.0.bn2.num_batches_tracked} constant,
     t218 f32 [C=32] {pt2=root:b_blocks_3_0_bn3_running_mean target=blocks.3.0.bn3.running_mean} ->[n162] constant,
     t219 f32 [C=32] {pt2=root:b_blocks_3_0_bn3_running_var target=blocks.3.0.bn3.running_var} ->[n162] constant,
     t220 i64 [C=1] {pt2=root:b_blocks_3_0_bn3_num_batches_tracked target=blocks.3.0.bn3.num_batches_tracked} constant,
     t221 f32 [C=192] {pt2=root:b_blocks_3_1_bn1_running_mean target=blocks.3.1.bn1.running_mean} ->[n169] constant,
     t222 f32 [C=192] {pt2=root:b_blocks_3_1_bn1_running_var target=blocks.3.1.bn1.running_var} ->[n169] constant,
     t223 i64 [C=1] {pt2=root:b_blocks_3_1_bn1_num_batches_tracked target=blocks.3.1.bn1.num_batches_tracked} constant,
     t224 f32 [C=192] {pt2=root:b_blocks_3_1_bn2_running_mean target=blocks.3.1.bn2.running_mean} ->[n177] constant,
     t225 f32 [C=192] {pt2=root:b_blocks_3_1_bn2_running_var target=blocks.3.1.bn2.running_var} ->[n177] constant,
     t226 i64 [C=1] {pt2=root:b_blocks_3_1_bn2_num_batches_tracked target=blocks.3.1.bn2.num_batches_tracked} constant,
     t227 f32 [C=32] {pt2=root:b_blocks_3_1_bn3_running_mean target=blocks.3.1.bn3.running_mean} ->[n185] constant,
     t228 f32 [C=32] {pt2=root:b_blocks_3_1_bn3_running_var target=blocks.3.1.bn3.running_var} ->[n185] constant,
     t229 i64 [C=1] {pt2=root:b_blocks_3_1_bn3_num_batches_tracked target=blocks.3.1.bn3.num_batches_tracked} constant,
     t230 f32 [C=192] {pt2=root:b_blocks_3_2_bn1_running_mean target=blocks.3.2.bn1.running_mean} ->[n193] constant,
     t231 f32 [C=192] {pt2=root:b_blocks_3_2_bn1_running_var target=blocks.3.2.bn1.running_var} ->[n193] constant,
     t232 i64 [C=1] {pt2=root:b_blocks_3_2_bn1_num_batches_tracked target=blocks.3.2.bn1.num_batches_tracked} constant,
     t233 f32 [C=192] {pt2=root:b_blocks_3_2_bn2_running_mean target=blocks.3.2.bn2.running_mean} ->[n201] constant,
     t234 f32 [C=192] {pt2=root:b_blocks_3_2_bn2_running_var target=blocks.3.2.bn2.running_var} ->[n201] constant,
     t235 i64 [C=1] {pt2=root:b_blocks_3_2_bn2_num_batches_tracked target=blocks.3.2.bn2.num_batches_tracked} constant,
     t236 f32 [C=32] {pt2=root:b_blocks_3_2_bn3_running_mean target=blocks.3.2.bn3.running_mean} ->[n209] constant,
     t237 f32 [C=32] {pt2=root:b_blocks_3_2_bn3_running_var target=blocks.3.2.bn3.running_var} ->[n209] constant,
     t238 i64 [C=1] {pt2=root:b_blocks_3_2_bn3_num_batches_tracked target=blocks.3.2.bn3.num_batches_tracked} constant,
     t239 f32 [C=192] {pt2=root:b_blocks_3_3_bn1_running_mean target=blocks.3.3.bn1.running_mean} ->[n217] constant,
     t240 f32 [C=192] {pt2=root:b_blocks_3_3_bn1_running_var target=blocks.3.3.bn1.running_var} ->[n217] constant,
     t241 i64 [C=1] {pt2=root:b_blocks_3_3_bn1_num_batches_tracked target=blocks.3.3.bn1.num_batches_tracked} constant,
     t242 f32 [C=192] {pt2=root:b_blocks_3_3_bn2_running_mean target=blocks.3.3.bn2.running_mean} ->[n225] constant,
     t243 f32 [C=192] {pt2=root:b_blocks_3_3_bn2_running_var target=blocks.3.3.bn2.running_var} ->[n225] constant,
     t244 i64 [C=1] {pt2=root:b_blocks_3_3_bn2_num_batches_tracked target=blocks.3.3.bn2.num_batches_tracked} constant,
     t245 f32 [C=32] {pt2=root:b_blocks_3_3_bn3_running_mean target=blocks.3.3.bn3.running_mean} ->[n233] constant,
     t246 f32 [C=32] {pt2=root:b_blocks_3_3_bn3_running_var target=blocks.3.3.bn3.running_var} ->[n233] constant,
     t247 i64 [C=1] {pt2=root:b_blocks_3_3_bn3_num_batches_tracked target=blocks.3.3.bn3.num_batches_tracked} constant,
     t248 f32 [C=192] {pt2=root:b_blocks_4_0_bn1_running_mean target=blocks.4.0.bn1.running_mean} ->[n241] constant,
     t249 f32 [C=192] {pt2=root:b_blocks_4_0_bn1_running_var target=blocks.4.0.bn1.running_var} ->[n241] constant,
     t250 i64 [C=1] {pt2=root:b_blocks_4_0_bn1_num_batches_tracked target=blocks.4.0.bn1.num_batches_tracked} constant,
     t251 f32 [C=192] {pt2=root:b_blocks_4_0_bn2_running_mean target=blocks.4.0.bn2.running_mean} ->[n249] constant,
     t252 f32 [C=192] {pt2=root:b_blocks_4_0_bn2_running_var target=blocks.4.0.bn2.running_var} ->[n249] constant,
     t253 i64 [C=1] {pt2=root:b_blocks_4_0_bn2_num_batches_tracked target=blocks.4.0.bn2.num_batches_tracked} constant,
     t254 f32 [C=48] {pt2=root:b_blocks_4_0_bn3_running_mean target=blocks.4.0.bn3.running_mean} ->[n257] constant,
     t255 f32 [C=48] {pt2=root:b_blocks_4_0_bn3_running_var target=blocks.4.0.bn3.running_var} ->[n257] constant,
     t256 i64 [C=1] {pt2=root:b_blocks_4_0_bn3_num_batches_tracked target=blocks.4.0.bn3.num_batches_tracked} constant,
     t257 f32 [C=288] {pt2=root:b_blocks_4_1_bn1_running_mean target=blocks.4.1.bn1.running_mean} ->[n264] constant,
     t258 f32 [C=288] {pt2=root:b_blocks_4_1_bn1_running_var target=blocks.4.1.bn1.running_var} ->[n264] constant,
     t259 i64 [C=1] {pt2=root:b_blocks_4_1_bn1_num_batches_tracked target=blocks.4.1.bn1.num_batches_tracked} constant,
     t260 f32 [C=288] {pt2=root:b_blocks_4_1_bn2_running_mean target=blocks.4.1.bn2.running_mean} ->[n272] constant,
     t261 f32 [C=288] {pt2=root:b_blocks_4_1_bn2_running_var target=blocks.4.1.bn2.running_var} ->[n272] constant,
     t262 i64 [C=1] {pt2=root:b_blocks_4_1_bn2_num_batches_tracked target=blocks.4.1.bn2.num_batches_tracked} constant,
     t263 f32 [C=48] {pt2=root:b_blocks_4_1_bn3_running_mean target=blocks.4.1.bn3.running_mean} ->[n280] constant,
     t264 f32 [C=48] {pt2=root:b_blocks_4_1_bn3_running_var target=blocks.4.1.bn3.running_var} ->[n280] constant,
     t265 i64 [C=1] {pt2=root:b_blocks_4_1_bn3_num_batches_tracked target=blocks.4.1.bn3.num_batches_tracked} constant,
     t266 f32 [C=288] {pt2=root:b_blocks_4_2_bn1_running_mean target=blocks.4.2.bn1.running_mean} ->[n288] constant,
     t267 f32 [C=288] {pt2=root:b_blocks_4_2_bn1_running_var target=blocks.4.2.bn1.running_var} ->[n288] constant,
     t268 i64 [C=1] {pt2=root:b_blocks_4_2_bn1_num_batches_tracked target=blocks.4.2.bn1.num_batches_tracked} constant,
     t269 f32 [C=288] {pt2=root:b_blocks_4_2_bn2_running_mean target=blocks.4.2.bn2.running_mean} ->[n296] constant,
     t270 f32 [C=288] {pt2=root:b_blocks_4_2_bn2_running_var target=blocks.4.2.bn2.running_var} ->[n296] constant,
     t271 i64 [C=1] {pt2=root:b_blocks_4_2_bn2_num_batches_tracked target=blocks.4.2.bn2.num_batches_tracked} constant,
     t272 f32 [C=48] {pt2=root:b_blocks_4_2_bn3_running_mean target=blocks.4.2.bn3.running_mean} ->[n304] constant,
     t273 f32 [C=48] {pt2=root:b_blocks_4_2_bn3_running_var target=blocks.4.2.bn3.running_var} ->[n304] constant,
     t274 i64 [C=1] {pt2=root:b_blocks_4_2_bn3_num_batches_tracked target=blocks.4.2.bn3.num_batches_tracked} constant,
     t275 f32 [C=288] {pt2=root:b_blocks_5_0_bn1_running_mean target=blocks.5.0.bn1.running_mean} ->[n312] constant,
     t276 f32 [C=288] {pt2=root:b_blocks_5_0_bn1_running_var target=blocks.5.0.bn1.running_var} ->[n312] constant,
     t277 i64 [C=1] {pt2=root:b_blocks_5_0_bn1_num_batches_tracked target=blocks.5.0.bn1.num_batches_tracked} constant,
     t278 f32 [C=288] {pt2=root:b_blocks_5_0_bn2_running_mean target=blocks.5.0.bn2.running_mean} ->[n320] constant,
     t279 f32 [C=288] {pt2=root:b_blocks_5_0_bn2_running_var target=blocks.5.0.bn2.running_var} ->[n320] constant,
     t280 i64 [C=1] {pt2=root:b_blocks_5_0_bn2_num_batches_tracked target=blocks.5.0.bn2.num_batches_tracked} constant,
     t281 f32 [C=80] {pt2=root:b_blocks_5_0_bn3_running_mean target=blocks.5.0.bn3.running_mean} ->[n328] constant,
     t282 f32 [C=80] {pt2=root:b_blocks_5_0_bn3_running_var target=blocks.5.0.bn3.running_var} ->[n328] constant,
     t283 i64 [C=1] {pt2=root:b_blocks_5_0_bn3_num_batches_tracked target=blocks.5.0.bn3.num_batches_tracked} constant,
     t284 f32 [C=480] {pt2=root:b_blocks_5_1_bn1_running_mean target=blocks.5.1.bn1.running_mean} ->[n335] constant,
     t285 f32 [C=480] {pt2=root:b_blocks_5_1_bn1_running_var target=blocks.5.1.bn1.running_var} ->[n335] constant,
     t286 i64 [C=1] {pt2=root:b_blocks_5_1_bn1_num_batches_tracked target=blocks.5.1.bn1.num_batches_tracked} constant,
     t287 f32 [C=480] {pt2=root:b_blocks_5_1_bn2_running_mean target=blocks.5.1.bn2.running_mean} ->[n343] constant,
     t288 f32 [C=480] {pt2=root:b_blocks_5_1_bn2_running_var target=blocks.5.1.bn2.running_var} ->[n343] constant,
     t289 i64 [C=1] {pt2=root:b_blocks_5_1_bn2_num_batches_tracked target=blocks.5.1.bn2.num_batches_tracked} constant,
     t290 f32 [C=80] {pt2=root:b_blocks_5_1_bn3_running_mean target=blocks.5.1.bn3.running_mean} ->[n351] constant,
     t291 f32 [C=80] {pt2=root:b_blocks_5_1_bn3_running_var target=blocks.5.1.bn3.running_var} ->[n351] constant,
     t292 i64 [C=1] {pt2=root:b_blocks_5_1_bn3_num_batches_tracked target=blocks.5.1.bn3.num_batches_tracked} constant,
     t293 f32 [C=480] {pt2=root:b_blocks_5_2_bn1_running_mean target=blocks.5.2.bn1.running_mean} ->[n359] constant,
     t294 f32 [C=480] {pt2=root:b_blocks_5_2_bn1_running_var target=blocks.5.2.bn1.running_var} ->[n359] constant,
     t295 i64 [C=1] {pt2=root:b_blocks_5_2_bn1_num_batches_tracked target=blocks.5.2.bn1.num_batches_tracked} constant,
     t296 f32 [C=480] {pt2=root:b_blocks_5_2_bn2_running_mean target=blocks.5.2.bn2.running_mean} ->[n367] constant,
     t297 f32 [C=480] {pt2=root:b_blocks_5_2_bn2_running_var target=blocks.5.2.bn2.running_var} ->[n367] constant,
     t298 i64 [C=1] {pt2=root:b_blocks_5_2_bn2_num_batches_tracked target=blocks.5.2.bn2.num_batches_tracked} constant,
     t299 f32 [C=80] {pt2=root:b_blocks_5_2_bn3_running_mean target=blocks.5.2.bn3.running_mean} ->[n375] constant,
     t300 f32 [C=80] {pt2=root:b_blocks_5_2_bn3_running_var target=blocks.5.2.bn3.running_var} ->[n375] constant,
     t301 i64 [C=1] {pt2=root:b_blocks_5_2_bn3_num_batches_tracked target=blocks.5.2.bn3.num_batches_tracked} constant,
     t302 f32 [C=480] {pt2=root:b_blocks_6_0_bn1_running_mean target=blocks.6.0.bn1.running_mean} ->[n383] constant,
     t303 f32 [C=480] {pt2=root:b_blocks_6_0_bn1_running_var target=blocks.6.0.bn1.running_var} ->[n383] constant,
     t304 i64 [C=1] {pt2=root:b_blocks_6_0_bn1_num_batches_tracked target=blocks.6.0.bn1.num_batches_tracked} constant,
     t305 f32 [C=480] {pt2=root:b_blocks_6_0_bn2_running_mean target=blocks.6.0.bn2.running_mean} ->[n391] constant,
     t306 f32 [C=480] {pt2=root:b_blocks_6_0_bn2_running_var target=blocks.6.0.bn2.running_var} ->[n391] constant,
     t307 i64 [C=1] {pt2=root:b_blocks_6_0_bn2_num_batches_tracked target=blocks.6.0.bn2.num_batches_tracked} constant,
     t308 f32 [C=160] {pt2=root:b_blocks_6_0_bn3_running_mean target=blocks.6.0.bn3.running_mean} ->[n399] constant,
     t309 f32 [C=160] {pt2=root:b_blocks_6_0_bn3_running_var target=blocks.6.0.bn3.running_var} ->[n399] constant,
     t310 i64 [C=1] {pt2=root:b_blocks_6_0_bn3_num_batches_tracked target=blocks.6.0.bn3.num_batches_tracked} constant,
     t311 f32 [C=1280] {pt2=root:b_bn2_running_mean target=bn2.running_mean} ->[n406] constant,
     t312 f32 [C=1280] {pt2=root:b_bn2_running_var target=bn2.running_var} ->[n406] constant,
     t313 i64 [C=1] {pt2=root:b_bn2_num_batches_tracked target=bn2.num_batches_tracked} constant,
     t314 f32 [H=3 W=224 C=224] {pt2=root:x} ->[n0]]
  nodes:
    group g1 torch.ops.aten.conv2d.default:
      n0 {derived}: [t315 f32 [H=224 W=224 C=3] {derived} ->[n2]] =
        permute x=t314 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
      n1 {derived}: [t316 f32 [N=16 T=1 D=1 H=3 W=3 C=3] {derived} ->[n2]] =
        permute
          x=t0 {pt2=root:p_conv_stem_weight target=conv_stem.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2 {derived}: [t317 f32 [H=112 W=112 C=16] {derived} ->[n3]] =
        conv2d
          x=t315 {derived} <-n0
          weight=t316 {derived} <-n1
          bias=none
          params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 in_channels=3;
                 groups=1}
      n3 {pt2=root[0] torch.ops.aten.conv2d.default (conv2d)}: [t318 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:conv2d} ->[n4]] =
        permute x=t317 {derived} <-n2 perm=[H<-C, W<-H, C<-W]
    group g2 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n4 {derived}: [t319 f32 [H=112 W=112 C=16] {derived} ->[n5]] =
        permute x=t318 {pt2=root:conv2d} <-n3 perm=[H<-W, W<-C, C<-H]
      n5 {derived}: [t320 f32 [H=112 W=112 C=16] {derived} ->[n6]] =
        batch_norm
          x=t319 {derived} <-n4
          weight=t1 {pt2=root:p_bn1_weight target=bn1.weight}
          bias=t2 {pt2=root:p_bn1_bias target=bn1.bias}
          running_mean=t158 {pt2=root:b_bn1_running_mean target=bn1.running_mean}
          running_var=t159 {pt2=root:b_bn1_running_var target=bn1.running_var}
          params={channel=C; eps=1e-05}
      n6 {pt2=root[1] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training)}: [t321 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:getitem} ->[n7]] =
        permute x=t320 {derived} <-n5 perm=[H<-C, W<-H, C<-W]
    n7 {pt2=root[2] torch.ops.aten.hardtanh.default (hardtanh)}: [t322 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:hardtanh} ->[n8]] =
      hardtanh x=t321 {pt2=root:getitem} <-n6 params={min_val=0; max_val=6}
    group g3 torch.ops.aten.conv2d.default:
      n8 {derived}: [t323 f32 [H=112 W=112 C=16] {derived} ->[n10]] =
        permute x=t322 {pt2=root:hardtanh} <-n7 perm=[H<-W, W<-C, C<-H]
      n9 {derived}: [t324 f32 [N=16 T=1 D=1 H=3 W=3 C=1] {derived} ->[n10]] =
        permute
          x=t3 {pt2=root:p_blocks_0_0_conv_dw_weight target=blocks.0.0.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n10 {derived}: [t325 f32 [H=112 W=112 C=16] {derived} ->[n11]] =
        conv2d
          x=t323 {derived} <-n8
          weight=t324 {derived} <-n9
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=16;
                 groups=16}
      n11 {pt2=root[3] torch.ops.aten.conv2d.default (conv2d_1)}: [t326 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:conv2d_1} ->[n12]] =
        permute x=t325 {derived} <-n10 perm=[H<-C, W<-H, C<-W]
    group g4 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n12 {derived}: [t327 f32 [H=112 W=112 C=16] {derived} ->[n13]] =
        permute x=t326 {pt2=root:conv2d_1} <-n11 perm=[H<-W, W<-C, C<-H]
      n13 {derived}: [t328 f32 [H=112 W=112 C=16] {derived} ->[n14]] =
        batch_norm
          x=t327 {derived} <-n12
          weight=t4 {pt2=root:p_blocks_0_0_bn1_weight target=blocks.0.0.bn1.weight}
          bias=t5 {pt2=root:p_blocks_0_0_bn1_bias target=blocks.0.0.bn1.bias}
          running_mean=t161 {pt2=root:b_blocks_0_0_bn1_running_mean target=blocks.0.0.bn1.running_mean}
          running_var=t162 {pt2=root:b_blocks_0_0_bn1_running_var target=blocks.0.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n14 {pt2=root[4] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_1)}: [t329 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:getitem_3} ->[n15]] =
        permute x=t328 {derived} <-n13 perm=[H<-C, W<-H, C<-W]
    n15 {pt2=root[5] torch.ops.aten.hardtanh.default (hardtanh_1)}: [t330 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:hardtanh_1} ->[n16]] =
      hardtanh x=t329 {pt2=root:getitem_3} <-n14 params={min_val=0; max_val=6}
    group g5 torch.ops.aten.conv2d.default:
      n16 {derived}: [t331 f32 [H=112 W=112 C=16] {derived} ->[n18]] =
        permute x=t330 {pt2=root:hardtanh_1} <-n15 perm=[H<-W, W<-C, C<-H]
      n17 {derived}: [t332 f32 [N=8 T=1 D=1 H=1 W=1 C=16] {derived} ->[n18]] =
        permute
          x=t6 {pt2=root:p_blocks_0_0_conv_pw_weight target=blocks.0.0.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n18 {derived}: [t333 f32 [H=112 W=112 C=8] {derived} ->[n19]] =
        conv2d
          x=t331 {derived} <-n16
          weight=t332 {derived} <-n17
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=16;
                 groups=1}
      n19 {pt2=root[6] torch.ops.aten.conv2d.default (conv2d_2)}: [t334 f32 [H=8
                                                                      W=112
                                                                      C=112] {pt2=root:conv2d_2} ->[n20]] =
        permute x=t333 {derived} <-n18 perm=[H<-C, W<-H, C<-W]
    group g6 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n20 {derived}: [t335 f32 [H=112 W=112 C=8] {derived} ->[n21]] =
        permute x=t334 {pt2=root:conv2d_2} <-n19 perm=[H<-W, W<-C, C<-H]
      n21 {derived}: [t336 f32 [H=112 W=112 C=8] {derived} ->[n22]] =
        batch_norm
          x=t335 {derived} <-n20
          weight=t7 {pt2=root:p_blocks_0_0_bn2_weight target=blocks.0.0.bn2.weight}
          bias=t8 {pt2=root:p_blocks_0_0_bn2_bias target=blocks.0.0.bn2.bias}
          running_mean=t164 {pt2=root:b_blocks_0_0_bn2_running_mean target=blocks.0.0.bn2.running_mean}
          running_var=t165 {pt2=root:b_blocks_0_0_bn2_running_var target=blocks.0.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n22 {pt2=root[7] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_2)}: [t337 f32 [H=8
                                                                      W=112
                                                                      C=112] {pt2=root:getitem_6} ->[n23]] =
        permute x=t336 {derived} <-n21 perm=[H<-C, W<-H, C<-W]
    group g7 torch.ops.aten.conv2d.default:
      n23 {derived}: [t338 f32 [H=112 W=112 C=8] {derived} ->[n25]] =
        permute x=t337 {pt2=root:getitem_6} <-n22 perm=[H<-W, W<-C, C<-H]
      n24 {derived}: [t339 f32 [N=48 T=1 D=1 H=1 W=1 C=8] {derived} ->[n25]] =
        permute
          x=t9 {pt2=root:p_blocks_1_0_conv_pw_weight target=blocks.1.0.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n25 {derived}: [t340 f32 [H=112 W=112 C=48] {derived} ->[n26]] =
        conv2d
          x=t338 {derived} <-n23
          weight=t339 {derived} <-n24
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=8;
                 groups=1}
      n26 {pt2=root[8] torch.ops.aten.conv2d.default (conv2d_3)}: [t341 f32 [H=48
                                                                      W=112
                                                                      C=112] {pt2=root:conv2d_3} ->[n27]] =
        permute x=t340 {derived} <-n25 perm=[H<-C, W<-H, C<-W]
    group g8 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n27 {derived}: [t342 f32 [H=112 W=112 C=48] {derived} ->[n28]] =
        permute x=t341 {pt2=root:conv2d_3} <-n26 perm=[H<-W, W<-C, C<-H]
      n28 {derived}: [t343 f32 [H=112 W=112 C=48] {derived} ->[n29]] =
        batch_norm
          x=t342 {derived} <-n27
          weight=t10 {pt2=root:p_blocks_1_0_bn1_weight target=blocks.1.0.bn1.weight}
          bias=t11 {pt2=root:p_blocks_1_0_bn1_bias target=blocks.1.0.bn1.bias}
          running_mean=t167 {pt2=root:b_blocks_1_0_bn1_running_mean target=blocks.1.0.bn1.running_mean}
          running_var=t168 {pt2=root:b_blocks_1_0_bn1_running_var target=blocks.1.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n29 {pt2=root[9] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_3)}: [t344 f32 [H=48
                                                                      W=112
                                                                      C=112] {pt2=root:getitem_9} ->[n30]] =
        permute x=t343 {derived} <-n28 perm=[H<-C, W<-H, C<-W]
    n30 {pt2=root[10] torch.ops.aten.hardtanh.default (hardtanh_2)}: [t345 f32 [H=48
                                                                      W=112
                                                                      C=112] {pt2=root:hardtanh_2} ->[n31]] =
      hardtanh x=t344 {pt2=root:getitem_9} <-n29 params={min_val=0; max_val=6}
    group g9 torch.ops.aten.conv2d.default:
      n31 {derived}: [t346 f32 [H=112 W=112 C=48] {derived} ->[n33]] =
        permute x=t345 {pt2=root:hardtanh_2} <-n30 perm=[H<-W, W<-C, C<-H]
      n32 {derived}: [t347 f32 [N=48 T=1 D=1 H=3 W=3 C=1] {derived} ->[n33]] =
        permute
          x=t12 {pt2=root:p_blocks_1_0_conv_dw_weight target=blocks.1.0.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n33 {derived}: [t348 f32 [H=56 W=56 C=48] {derived} ->[n34]] =
        conv2d
          x=t346 {derived} <-n31
          weight=t347 {derived} <-n32
          bias=none
          params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 in_channels=48;
                 groups=48}
      n34 {pt2=root[11] torch.ops.aten.conv2d.default (conv2d_4)}: [t349 f32 [H=48
                                                                      W=56
                                                                      C=56] {pt2=root:conv2d_4} ->[n35]] =
        permute x=t348 {derived} <-n33 perm=[H<-C, W<-H, C<-W]
    group g10 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n35 {derived}: [t350 f32 [H=56 W=56 C=48] {derived} ->[n36]] =
        permute x=t349 {pt2=root:conv2d_4} <-n34 perm=[H<-W, W<-C, C<-H]
      n36 {derived}: [t351 f32 [H=56 W=56 C=48] {derived} ->[n37]] =
        batch_norm
          x=t350 {derived} <-n35
          weight=t13 {pt2=root:p_blocks_1_0_bn2_weight target=blocks.1.0.bn2.weight}
          bias=t14 {pt2=root:p_blocks_1_0_bn2_bias target=blocks.1.0.bn2.bias}
          running_mean=t170 {pt2=root:b_blocks_1_0_bn2_running_mean target=blocks.1.0.bn2.running_mean}
          running_var=t171 {pt2=root:b_blocks_1_0_bn2_running_var target=blocks.1.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n37 {pt2=root[12] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_4)}: [t352 f32 [H=48
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_12} ->[n38]] =
        permute x=t351 {derived} <-n36 perm=[H<-C, W<-H, C<-W]
    n38 {pt2=root[13] torch.ops.aten.hardtanh.default (hardtanh_3)}: [t353 f32 [H=48
                                                                      W=56
                                                                      C=56] {pt2=root:hardtanh_3} ->[n39]] =
      hardtanh x=t352 {pt2=root:getitem_12} <-n37 params={min_val=0; max_val=6}
    group g11 torch.ops.aten.conv2d.default:
      n39 {derived}: [t354 f32 [H=56 W=56 C=48] {derived} ->[n41]] =
        permute x=t353 {pt2=root:hardtanh_3} <-n38 perm=[H<-W, W<-C, C<-H]
      n40 {derived}: [t355 f32 [N=16 T=1 D=1 H=1 W=1 C=48] {derived} ->[n41]] =
        permute
          x=t15 {pt2=root:p_blocks_1_0_conv_pwl_weight target=blocks.1.0.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n41 {derived}: [t356 f32 [H=56 W=56 C=16] {derived} ->[n42]] =
        conv2d
          x=t354 {derived} <-n39
          weight=t355 {derived} <-n40
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=48;
                 groups=1}
      n42 {pt2=root[14] torch.ops.aten.conv2d.default (conv2d_5)}: [t357 f32 [H=16
                                                                      W=56
                                                                      C=56] {pt2=root:conv2d_5} ->[n43]] =
        permute x=t356 {derived} <-n41 perm=[H<-C, W<-H, C<-W]
    group g12 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n43 {derived}: [t358 f32 [H=56 W=56 C=16] {derived} ->[n44]] =
        permute x=t357 {pt2=root:conv2d_5} <-n42 perm=[H<-W, W<-C, C<-H]
      n44 {derived}: [t359 f32 [H=56 W=56 C=16] {derived} ->[n45]] =
        batch_norm
          x=t358 {derived} <-n43
          weight=t16 {pt2=root:p_blocks_1_0_bn3_weight target=blocks.1.0.bn3.weight}
          bias=t17 {pt2=root:p_blocks_1_0_bn3_bias target=blocks.1.0.bn3.bias}
          running_mean=t173 {pt2=root:b_blocks_1_0_bn3_running_mean target=blocks.1.0.bn3.running_mean}
          running_var=t174 {pt2=root:b_blocks_1_0_bn3_running_var target=blocks.1.0.bn3.running_var}
          params={channel=C; eps=1e-05}
      n45 {pt2=root[15] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_5)}: [t360 f32 [H=16
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_15} ->[n46,
                                                                      n69]] =
        permute x=t359 {derived} <-n44 perm=[H<-C, W<-H, C<-W]
    group g13 torch.ops.aten.conv2d.default:
      n46 {derived}: [t361 f32 [H=56 W=56 C=16] {derived} ->[n48]] =
        permute x=t360 {pt2=root:getitem_15} <-n45 perm=[H<-W, W<-C, C<-H]
      n47 {derived}: [t362 f32 [N=96 T=1 D=1 H=1 W=1 C=16] {derived} ->[n48]] =
        permute
          x=t18 {pt2=root:p_blocks_1_1_conv_pw_weight target=blocks.1.1.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n48 {derived}: [t363 f32 [H=56 W=56 C=96] {derived} ->[n49]] =
        conv2d
          x=t361 {derived} <-n46
          weight=t362 {derived} <-n47
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=16;
                 groups=1}
      n49 {pt2=root[16] torch.ops.aten.conv2d.default (conv2d_6)}: [t364 f32 [H=96
                                                                      W=56
                                                                      C=56] {pt2=root:conv2d_6} ->[n50]] =
        permute x=t363 {derived} <-n48 perm=[H<-C, W<-H, C<-W]
    group g14 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n50 {derived}: [t365 f32 [H=56 W=56 C=96] {derived} ->[n51]] =
        permute x=t364 {pt2=root:conv2d_6} <-n49 perm=[H<-W, W<-C, C<-H]
      n51 {derived}: [t366 f32 [H=56 W=56 C=96] {derived} ->[n52]] =
        batch_norm
          x=t365 {derived} <-n50
          weight=t19 {pt2=root:p_blocks_1_1_bn1_weight target=blocks.1.1.bn1.weight}
          bias=t20 {pt2=root:p_blocks_1_1_bn1_bias target=blocks.1.1.bn1.bias}
          running_mean=t176 {pt2=root:b_blocks_1_1_bn1_running_mean target=blocks.1.1.bn1.running_mean}
          running_var=t177 {pt2=root:b_blocks_1_1_bn1_running_var target=blocks.1.1.bn1.running_var}
          params={channel=C; eps=1e-05}
      n52 {pt2=root[17] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_6)}: [t367 f32 [H=96
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_18} ->[n53]] =
        permute x=t366 {derived} <-n51 perm=[H<-C, W<-H, C<-W]
    n53 {pt2=root[18] torch.ops.aten.hardtanh.default (hardtanh_4)}: [t368 f32 [H=96
                                                                      W=56
                                                                      C=56] {pt2=root:hardtanh_4} ->[n54]] =
      hardtanh x=t367 {pt2=root:getitem_18} <-n52 params={min_val=0; max_val=6}
    group g15 torch.ops.aten.conv2d.default:
      n54 {derived}: [t369 f32 [H=56 W=56 C=96] {derived} ->[n56]] =
        permute x=t368 {pt2=root:hardtanh_4} <-n53 perm=[H<-W, W<-C, C<-H]
      n55 {derived}: [t370 f32 [N=96 T=1 D=1 H=3 W=3 C=1] {derived} ->[n56]] =
        permute
          x=t21 {pt2=root:p_blocks_1_1_conv_dw_weight target=blocks.1.1.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n56 {derived}: [t371 f32 [H=56 W=56 C=96] {derived} ->[n57]] =
        conv2d
          x=t369 {derived} <-n54
          weight=t370 {derived} <-n55
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=96;
                 groups=96}
      n57 {pt2=root[19] torch.ops.aten.conv2d.default (conv2d_7)}: [t372 f32 [H=96
                                                                      W=56
                                                                      C=56] {pt2=root:conv2d_7} ->[n58]] =
        permute x=t371 {derived} <-n56 perm=[H<-C, W<-H, C<-W]
    group g16 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n58 {derived}: [t373 f32 [H=56 W=56 C=96] {derived} ->[n59]] =
        permute x=t372 {pt2=root:conv2d_7} <-n57 perm=[H<-W, W<-C, C<-H]
      n59 {derived}: [t374 f32 [H=56 W=56 C=96] {derived} ->[n60]] =
        batch_norm
          x=t373 {derived} <-n58
          weight=t22 {pt2=root:p_blocks_1_1_bn2_weight target=blocks.1.1.bn2.weight}
          bias=t23 {pt2=root:p_blocks_1_1_bn2_bias target=blocks.1.1.bn2.bias}
          running_mean=t179 {pt2=root:b_blocks_1_1_bn2_running_mean target=blocks.1.1.bn2.running_mean}
          running_var=t180 {pt2=root:b_blocks_1_1_bn2_running_var target=blocks.1.1.bn2.running_var}
          params={channel=C; eps=1e-05}
      n60 {pt2=root[20] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_7)}: [t375 f32 [H=96
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_21} ->[n61]] =
        permute x=t374 {derived} <-n59 perm=[H<-C, W<-H, C<-W]
    n61 {pt2=root[21] torch.ops.aten.hardtanh.default (hardtanh_5)}: [t376 f32 [H=96
                                                                      W=56
                                                                      C=56] {pt2=root:hardtanh_5} ->[n62]] =
      hardtanh x=t375 {pt2=root:getitem_21} <-n60 params={min_val=0; max_val=6}
    group g17 torch.ops.aten.conv2d.default:
      n62 {derived}: [t377 f32 [H=56 W=56 C=96] {derived} ->[n64]] =
        permute x=t376 {pt2=root:hardtanh_5} <-n61 perm=[H<-W, W<-C, C<-H]
      n63 {derived}: [t378 f32 [N=16 T=1 D=1 H=1 W=1 C=96] {derived} ->[n64]] =
        permute
          x=t24 {pt2=root:p_blocks_1_1_conv_pwl_weight target=blocks.1.1.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n64 {derived}: [t379 f32 [H=56 W=56 C=16] {derived} ->[n65]] =
        conv2d
          x=t377 {derived} <-n62
          weight=t378 {derived} <-n63
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=96;
                 groups=1}
      n65 {pt2=root[22] torch.ops.aten.conv2d.default (conv2d_8)}: [t380 f32 [H=16
                                                                      W=56
                                                                      C=56] {pt2=root:conv2d_8} ->[n66]] =
        permute x=t379 {derived} <-n64 perm=[H<-C, W<-H, C<-W]
    group g18 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n66 {derived}: [t381 f32 [H=56 W=56 C=16] {derived} ->[n67]] =
        permute x=t380 {pt2=root:conv2d_8} <-n65 perm=[H<-W, W<-C, C<-H]
      n67 {derived}: [t382 f32 [H=56 W=56 C=16] {derived} ->[n68]] =
        batch_norm
          x=t381 {derived} <-n66
          weight=t25 {pt2=root:p_blocks_1_1_bn3_weight target=blocks.1.1.bn3.weight}
          bias=t26 {pt2=root:p_blocks_1_1_bn3_bias target=blocks.1.1.bn3.bias}
          running_mean=t182 {pt2=root:b_blocks_1_1_bn3_running_mean target=blocks.1.1.bn3.running_mean}
          running_var=t183 {pt2=root:b_blocks_1_1_bn3_running_var target=blocks.1.1.bn3.running_var}
          params={channel=C; eps=1e-05}
      n68 {pt2=root[23] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_8)}: [t383 f32 [H=16
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_24} ->[n69]] =
        permute x=t382 {derived} <-n67 perm=[H<-C, W<-H, C<-W]
    n69 {pt2=root[24] torch.ops.aten.add.Tensor (add)}: [t384 f32 [H=16 W=56
                                                                   C=56] {pt2=root:add} ->[n70]] =
      add a=t383 {pt2=root:getitem_24} <-n68 b=t360 {pt2=root:getitem_15} <-n45
    group g19 torch.ops.aten.conv2d.default:
      n70 {derived}: [t385 f32 [H=56 W=56 C=16] {derived} ->[n72]] =
        permute x=t384 {pt2=root:add} <-n69 perm=[H<-W, W<-C, C<-H]
      n71 {derived}: [t386 f32 [N=96 T=1 D=1 H=1 W=1 C=16] {derived} ->[n72]] =
        permute
          x=t27 {pt2=root:p_blocks_2_0_conv_pw_weight target=blocks.2.0.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n72 {derived}: [t387 f32 [H=56 W=56 C=96] {derived} ->[n73]] =
        conv2d
          x=t385 {derived} <-n70
          weight=t386 {derived} <-n71
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=16;
                 groups=1}
      n73 {pt2=root[25] torch.ops.aten.conv2d.default (conv2d_9)}: [t388 f32 [H=96
                                                                      W=56
                                                                      C=56] {pt2=root:conv2d_9} ->[n74]] =
        permute x=t387 {derived} <-n72 perm=[H<-C, W<-H, C<-W]
    group g20 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n74 {derived}: [t389 f32 [H=56 W=56 C=96] {derived} ->[n75]] =
        permute x=t388 {pt2=root:conv2d_9} <-n73 perm=[H<-W, W<-C, C<-H]
      n75 {derived}: [t390 f32 [H=56 W=56 C=96] {derived} ->[n76]] =
        batch_norm
          x=t389 {derived} <-n74
          weight=t28 {pt2=root:p_blocks_2_0_bn1_weight target=blocks.2.0.bn1.weight}
          bias=t29 {pt2=root:p_blocks_2_0_bn1_bias target=blocks.2.0.bn1.bias}
          running_mean=t185 {pt2=root:b_blocks_2_0_bn1_running_mean target=blocks.2.0.bn1.running_mean}
          running_var=t186 {pt2=root:b_blocks_2_0_bn1_running_var target=blocks.2.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n76 {pt2=root[26] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_9)}: [t391 f32 [H=96
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_27} ->[n77]] =
        permute x=t390 {derived} <-n75 perm=[H<-C, W<-H, C<-W]
    n77 {pt2=root[27] torch.ops.aten.hardtanh.default (hardtanh_6)}: [t392 f32 [H=96
                                                                      W=56
                                                                      C=56] {pt2=root:hardtanh_6} ->[n78]] =
      hardtanh x=t391 {pt2=root:getitem_27} <-n76 params={min_val=0; max_val=6}
    group g21 torch.ops.aten.conv2d.default:
      n78 {derived}: [t393 f32 [H=56 W=56 C=96] {derived} ->[n80]] =
        permute x=t392 {pt2=root:hardtanh_6} <-n77 perm=[H<-W, W<-C, C<-H]
      n79 {derived}: [t394 f32 [N=96 T=1 D=1 H=3 W=3 C=1] {derived} ->[n80]] =
        permute
          x=t30 {pt2=root:p_blocks_2_0_conv_dw_weight target=blocks.2.0.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n80 {derived}: [t395 f32 [H=28 W=28 C=96] {derived} ->[n81]] =
        conv2d
          x=t393 {derived} <-n78
          weight=t394 {derived} <-n79
          bias=none
          params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 in_channels=96;
                 groups=96}
      n81 {pt2=root[28] torch.ops.aten.conv2d.default (conv2d_10)}: [t396 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_10} ->[n82]] =
        permute x=t395 {derived} <-n80 perm=[H<-C, W<-H, C<-W]
    group g22 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n82 {derived}: [t397 f32 [H=28 W=28 C=96] {derived} ->[n83]] =
        permute x=t396 {pt2=root:conv2d_10} <-n81 perm=[H<-W, W<-C, C<-H]
      n83 {derived}: [t398 f32 [H=28 W=28 C=96] {derived} ->[n84]] =
        batch_norm
          x=t397 {derived} <-n82
          weight=t31 {pt2=root:p_blocks_2_0_bn2_weight target=blocks.2.0.bn2.weight}
          bias=t32 {pt2=root:p_blocks_2_0_bn2_bias target=blocks.2.0.bn2.bias}
          running_mean=t188 {pt2=root:b_blocks_2_0_bn2_running_mean target=blocks.2.0.bn2.running_mean}
          running_var=t189 {pt2=root:b_blocks_2_0_bn2_running_var target=blocks.2.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n84 {pt2=root[29] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_10)}: [t399 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_30} ->[n85]] =
        permute x=t398 {derived} <-n83 perm=[H<-C, W<-H, C<-W]
    n85 {pt2=root[30] torch.ops.aten.hardtanh.default (hardtanh_7)}: [t400 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:hardtanh_7} ->[n86]] =
      hardtanh x=t399 {pt2=root:getitem_30} <-n84 params={min_val=0; max_val=6}
    group g23 torch.ops.aten.conv2d.default:
      n86 {derived}: [t401 f32 [H=28 W=28 C=96] {derived} ->[n88]] =
        permute x=t400 {pt2=root:hardtanh_7} <-n85 perm=[H<-W, W<-C, C<-H]
      n87 {derived}: [t402 f32 [N=16 T=1 D=1 H=1 W=1 C=96] {derived} ->[n88]] =
        permute
          x=t33 {pt2=root:p_blocks_2_0_conv_pwl_weight target=blocks.2.0.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n88 {derived}: [t403 f32 [H=28 W=28 C=16] {derived} ->[n89]] =
        conv2d
          x=t401 {derived} <-n86
          weight=t402 {derived} <-n87
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=96;
                 groups=1}
      n89 {pt2=root[31] torch.ops.aten.conv2d.default (conv2d_11)}: [t404 f32 [H=16
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_11} ->[n90]] =
        permute x=t403 {derived} <-n88 perm=[H<-C, W<-H, C<-W]
    group g24 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n90 {derived}: [t405 f32 [H=28 W=28 C=16] {derived} ->[n91]] =
        permute x=t404 {pt2=root:conv2d_11} <-n89 perm=[H<-W, W<-C, C<-H]
      n91 {derived}: [t406 f32 [H=28 W=28 C=16] {derived} ->[n92]] =
        batch_norm
          x=t405 {derived} <-n90
          weight=t34 {pt2=root:p_blocks_2_0_bn3_weight target=blocks.2.0.bn3.weight}
          bias=t35 {pt2=root:p_blocks_2_0_bn3_bias target=blocks.2.0.bn3.bias}
          running_mean=t191 {pt2=root:b_blocks_2_0_bn3_running_mean target=blocks.2.0.bn3.running_mean}
          running_var=t192 {pt2=root:b_blocks_2_0_bn3_running_var target=blocks.2.0.bn3.running_var}
          params={channel=C; eps=1e-05}
      n92 {pt2=root[32] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_11)}: [t407 f32 [H=16
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_33} ->[n93,
                                                                      n116]] =
        permute x=t406 {derived} <-n91 perm=[H<-C, W<-H, C<-W]
    group g25 torch.ops.aten.conv2d.default:
      n93 {derived}: [t408 f32 [H=28 W=28 C=16] {derived} ->[n95]] =
        permute x=t407 {pt2=root:getitem_33} <-n92 perm=[H<-W, W<-C, C<-H]
      n94 {derived}: [t409 f32 [N=96 T=1 D=1 H=1 W=1 C=16] {derived} ->[n95]] =
        permute
          x=t36 {pt2=root:p_blocks_2_1_conv_pw_weight target=blocks.2.1.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n95 {derived}: [t410 f32 [H=28 W=28 C=96] {derived} ->[n96]] =
        conv2d
          x=t408 {derived} <-n93
          weight=t409 {derived} <-n94
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=16;
                 groups=1}
      n96 {pt2=root[33] torch.ops.aten.conv2d.default (conv2d_12)}: [t411 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_12} ->[n97]] =
        permute x=t410 {derived} <-n95 perm=[H<-C, W<-H, C<-W]
    group g26 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n97 {derived}: [t412 f32 [H=28 W=28 C=96] {derived} ->[n98]] =
        permute x=t411 {pt2=root:conv2d_12} <-n96 perm=[H<-W, W<-C, C<-H]
      n98 {derived}: [t413 f32 [H=28 W=28 C=96] {derived} ->[n99]] =
        batch_norm
          x=t412 {derived} <-n97
          weight=t37 {pt2=root:p_blocks_2_1_bn1_weight target=blocks.2.1.bn1.weight}
          bias=t38 {pt2=root:p_blocks_2_1_bn1_bias target=blocks.2.1.bn1.bias}
          running_mean=t194 {pt2=root:b_blocks_2_1_bn1_running_mean target=blocks.2.1.bn1.running_mean}
          running_var=t195 {pt2=root:b_blocks_2_1_bn1_running_var target=blocks.2.1.bn1.running_var}
          params={channel=C; eps=1e-05}
      n99 {pt2=root[34] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_12)}: [t414 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_36} ->[n100]] =
        permute x=t413 {derived} <-n98 perm=[H<-C, W<-H, C<-W]
    n100 {pt2=root[35] torch.ops.aten.hardtanh.default (hardtanh_8)}: [t415 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:hardtanh_8} ->[n101]] =
      hardtanh x=t414 {pt2=root:getitem_36} <-n99 params={min_val=0; max_val=6}
    group g27 torch.ops.aten.conv2d.default:
      n101 {derived}: [t416 f32 [H=28 W=28 C=96] {derived} ->[n103]] =
        permute x=t415 {pt2=root:hardtanh_8} <-n100 perm=[H<-W, W<-C, C<-H]
      n102 {derived}: [t417 f32 [N=96 T=1 D=1 H=3 W=3 C=1] {derived} ->[n103]] =
        permute
          x=t39 {pt2=root:p_blocks_2_1_conv_dw_weight target=blocks.2.1.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n103 {derived}: [t418 f32 [H=28 W=28 C=96] {derived} ->[n104]] =
        conv2d
          x=t416 {derived} <-n101
          weight=t417 {derived} <-n102
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=96;
                 groups=96}
      n104 {pt2=root[36] torch.ops.aten.conv2d.default (conv2d_13)}: [t419 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_13} ->[n105]] =
        permute x=t418 {derived} <-n103 perm=[H<-C, W<-H, C<-W]
    group g28 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n105 {derived}: [t420 f32 [H=28 W=28 C=96] {derived} ->[n106]] =
        permute x=t419 {pt2=root:conv2d_13} <-n104 perm=[H<-W, W<-C, C<-H]
      n106 {derived}: [t421 f32 [H=28 W=28 C=96] {derived} ->[n107]] =
        batch_norm
          x=t420 {derived} <-n105
          weight=t40 {pt2=root:p_blocks_2_1_bn2_weight target=blocks.2.1.bn2.weight}
          bias=t41 {pt2=root:p_blocks_2_1_bn2_bias target=blocks.2.1.bn2.bias}
          running_mean=t197 {pt2=root:b_blocks_2_1_bn2_running_mean target=blocks.2.1.bn2.running_mean}
          running_var=t198 {pt2=root:b_blocks_2_1_bn2_running_var target=blocks.2.1.bn2.running_var}
          params={channel=C; eps=1e-05}
      n107 {pt2=root[37] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_13)}: [t422 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_39} ->[n108]] =
        permute x=t421 {derived} <-n106 perm=[H<-C, W<-H, C<-W]
    n108 {pt2=root[38] torch.ops.aten.hardtanh.default (hardtanh_9)}: [t423 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:hardtanh_9} ->[n109]] =
      hardtanh
        x=t422 {pt2=root:getitem_39} <-n107
        params={min_val=0; max_val=6}
    group g29 torch.ops.aten.conv2d.default:
      n109 {derived}: [t424 f32 [H=28 W=28 C=96] {derived} ->[n111]] =
        permute x=t423 {pt2=root:hardtanh_9} <-n108 perm=[H<-W, W<-C, C<-H]
      n110 {derived}: [t425 f32 [N=16 T=1 D=1 H=1 W=1 C=96] {derived} ->[n111]] =
        permute
          x=t42 {pt2=root:p_blocks_2_1_conv_pwl_weight target=blocks.2.1.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n111 {derived}: [t426 f32 [H=28 W=28 C=16] {derived} ->[n112]] =
        conv2d
          x=t424 {derived} <-n109
          weight=t425 {derived} <-n110
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=96;
                 groups=1}
      n112 {pt2=root[39] torch.ops.aten.conv2d.default (conv2d_14)}: [t427 f32 [H=16
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_14} ->[n113]] =
        permute x=t426 {derived} <-n111 perm=[H<-C, W<-H, C<-W]
    group g30 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n113 {derived}: [t428 f32 [H=28 W=28 C=16] {derived} ->[n114]] =
        permute x=t427 {pt2=root:conv2d_14} <-n112 perm=[H<-W, W<-C, C<-H]
      n114 {derived}: [t429 f32 [H=28 W=28 C=16] {derived} ->[n115]] =
        batch_norm
          x=t428 {derived} <-n113
          weight=t43 {pt2=root:p_blocks_2_1_bn3_weight target=blocks.2.1.bn3.weight}
          bias=t44 {pt2=root:p_blocks_2_1_bn3_bias target=blocks.2.1.bn3.bias}
          running_mean=t200 {pt2=root:b_blocks_2_1_bn3_running_mean target=blocks.2.1.bn3.running_mean}
          running_var=t201 {pt2=root:b_blocks_2_1_bn3_running_var target=blocks.2.1.bn3.running_var}
          params={channel=C; eps=1e-05}
      n115 {pt2=root[40] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_14)}: [t430 f32 [H=16
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_42} ->[n116]] =
        permute x=t429 {derived} <-n114 perm=[H<-C, W<-H, C<-W]
    n116 {pt2=root[41] torch.ops.aten.add.Tensor (add_1)}: [t431 f32 [H=16 W=28
                                                                      C=28] {pt2=root:add_1} ->[n117,
                                                                      n140]] =
      add
        a=t430 {pt2=root:getitem_42} <-n115
        b=t407 {pt2=root:getitem_33} <-n92
    group g31 torch.ops.aten.conv2d.default:
      n117 {derived}: [t432 f32 [H=28 W=28 C=16] {derived} ->[n119]] =
        permute x=t431 {pt2=root:add_1} <-n116 perm=[H<-W, W<-C, C<-H]
      n118 {derived}: [t433 f32 [N=96 T=1 D=1 H=1 W=1 C=16] {derived} ->[n119]] =
        permute
          x=t45 {pt2=root:p_blocks_2_2_conv_pw_weight target=blocks.2.2.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n119 {derived}: [t434 f32 [H=28 W=28 C=96] {derived} ->[n120]] =
        conv2d
          x=t432 {derived} <-n117
          weight=t433 {derived} <-n118
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=16;
                 groups=1}
      n120 {pt2=root[42] torch.ops.aten.conv2d.default (conv2d_15)}: [t435 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_15} ->[n121]] =
        permute x=t434 {derived} <-n119 perm=[H<-C, W<-H, C<-W]
    group g32 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n121 {derived}: [t436 f32 [H=28 W=28 C=96] {derived} ->[n122]] =
        permute x=t435 {pt2=root:conv2d_15} <-n120 perm=[H<-W, W<-C, C<-H]
      n122 {derived}: [t437 f32 [H=28 W=28 C=96] {derived} ->[n123]] =
        batch_norm
          x=t436 {derived} <-n121
          weight=t46 {pt2=root:p_blocks_2_2_bn1_weight target=blocks.2.2.bn1.weight}
          bias=t47 {pt2=root:p_blocks_2_2_bn1_bias target=blocks.2.2.bn1.bias}
          running_mean=t203 {pt2=root:b_blocks_2_2_bn1_running_mean target=blocks.2.2.bn1.running_mean}
          running_var=t204 {pt2=root:b_blocks_2_2_bn1_running_var target=blocks.2.2.bn1.running_var}
          params={channel=C; eps=1e-05}
      n123 {pt2=root[43] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_15)}: [t438 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_45} ->[n124]] =
        permute x=t437 {derived} <-n122 perm=[H<-C, W<-H, C<-W]
    n124 {pt2=root[44] torch.ops.aten.hardtanh.default (hardtanh_10)}: [t439 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:hardtanh_10} ->[n125]] =
      hardtanh
        x=t438 {pt2=root:getitem_45} <-n123
        params={min_val=0; max_val=6}
    group g33 torch.ops.aten.conv2d.default:
      n125 {derived}: [t440 f32 [H=28 W=28 C=96] {derived} ->[n127]] =
        permute x=t439 {pt2=root:hardtanh_10} <-n124 perm=[H<-W, W<-C, C<-H]
      n126 {derived}: [t441 f32 [N=96 T=1 D=1 H=3 W=3 C=1] {derived} ->[n127]] =
        permute
          x=t48 {pt2=root:p_blocks_2_2_conv_dw_weight target=blocks.2.2.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n127 {derived}: [t442 f32 [H=28 W=28 C=96] {derived} ->[n128]] =
        conv2d
          x=t440 {derived} <-n125
          weight=t441 {derived} <-n126
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=96;
                 groups=96}
      n128 {pt2=root[45] torch.ops.aten.conv2d.default (conv2d_16)}: [t443 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_16} ->[n129]] =
        permute x=t442 {derived} <-n127 perm=[H<-C, W<-H, C<-W]
    group g34 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n129 {derived}: [t444 f32 [H=28 W=28 C=96] {derived} ->[n130]] =
        permute x=t443 {pt2=root:conv2d_16} <-n128 perm=[H<-W, W<-C, C<-H]
      n130 {derived}: [t445 f32 [H=28 W=28 C=96] {derived} ->[n131]] =
        batch_norm
          x=t444 {derived} <-n129
          weight=t49 {pt2=root:p_blocks_2_2_bn2_weight target=blocks.2.2.bn2.weight}
          bias=t50 {pt2=root:p_blocks_2_2_bn2_bias target=blocks.2.2.bn2.bias}
          running_mean=t206 {pt2=root:b_blocks_2_2_bn2_running_mean target=blocks.2.2.bn2.running_mean}
          running_var=t207 {pt2=root:b_blocks_2_2_bn2_running_var target=blocks.2.2.bn2.running_var}
          params={channel=C; eps=1e-05}
      n131 {pt2=root[46] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_16)}: [t446 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_48} ->[n132]] =
        permute x=t445 {derived} <-n130 perm=[H<-C, W<-H, C<-W]
    n132 {pt2=root[47] torch.ops.aten.hardtanh.default (hardtanh_11)}: [t447 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:hardtanh_11} ->[n133]] =
      hardtanh
        x=t446 {pt2=root:getitem_48} <-n131
        params={min_val=0; max_val=6}
    group g35 torch.ops.aten.conv2d.default:
      n133 {derived}: [t448 f32 [H=28 W=28 C=96] {derived} ->[n135]] =
        permute x=t447 {pt2=root:hardtanh_11} <-n132 perm=[H<-W, W<-C, C<-H]
      n134 {derived}: [t449 f32 [N=16 T=1 D=1 H=1 W=1 C=96] {derived} ->[n135]] =
        permute
          x=t51 {pt2=root:p_blocks_2_2_conv_pwl_weight target=blocks.2.2.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n135 {derived}: [t450 f32 [H=28 W=28 C=16] {derived} ->[n136]] =
        conv2d
          x=t448 {derived} <-n133
          weight=t449 {derived} <-n134
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=96;
                 groups=1}
      n136 {pt2=root[48] torch.ops.aten.conv2d.default (conv2d_17)}: [t451 f32 [H=16
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_17} ->[n137]] =
        permute x=t450 {derived} <-n135 perm=[H<-C, W<-H, C<-W]
    group g36 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n137 {derived}: [t452 f32 [H=28 W=28 C=16] {derived} ->[n138]] =
        permute x=t451 {pt2=root:conv2d_17} <-n136 perm=[H<-W, W<-C, C<-H]
      n138 {derived}: [t453 f32 [H=28 W=28 C=16] {derived} ->[n139]] =
        batch_norm
          x=t452 {derived} <-n137
          weight=t52 {pt2=root:p_blocks_2_2_bn3_weight target=blocks.2.2.bn3.weight}
          bias=t53 {pt2=root:p_blocks_2_2_bn3_bias target=blocks.2.2.bn3.bias}
          running_mean=t209 {pt2=root:b_blocks_2_2_bn3_running_mean target=blocks.2.2.bn3.running_mean}
          running_var=t210 {pt2=root:b_blocks_2_2_bn3_running_var target=blocks.2.2.bn3.running_var}
          params={channel=C; eps=1e-05}
      n139 {pt2=root[49] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_17)}: [t454 f32 [H=16
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_51} ->[n140]] =
        permute x=t453 {derived} <-n138 perm=[H<-C, W<-H, C<-W]
    n140 {pt2=root[50] torch.ops.aten.add.Tensor (add_2)}: [t455 f32 [H=16 W=28
                                                                      C=28] {pt2=root:add_2} ->[n141]] =
      add a=t454 {pt2=root:getitem_51} <-n139 b=t431 {pt2=root:add_1} <-n116
    group g37 torch.ops.aten.conv2d.default:
      n141 {derived}: [t456 f32 [H=28 W=28 C=16] {derived} ->[n143]] =
        permute x=t455 {pt2=root:add_2} <-n140 perm=[H<-W, W<-C, C<-H]
      n142 {derived}: [t457 f32 [N=96 T=1 D=1 H=1 W=1 C=16] {derived} ->[n143]] =
        permute
          x=t54 {pt2=root:p_blocks_3_0_conv_pw_weight target=blocks.3.0.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n143 {derived}: [t458 f32 [H=28 W=28 C=96] {derived} ->[n144]] =
        conv2d
          x=t456 {derived} <-n141
          weight=t457 {derived} <-n142
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=16;
                 groups=1}
      n144 {pt2=root[51] torch.ops.aten.conv2d.default (conv2d_18)}: [t459 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_18} ->[n145]] =
        permute x=t458 {derived} <-n143 perm=[H<-C, W<-H, C<-W]
    group g38 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n145 {derived}: [t460 f32 [H=28 W=28 C=96] {derived} ->[n146]] =
        permute x=t459 {pt2=root:conv2d_18} <-n144 perm=[H<-W, W<-C, C<-H]
      n146 {derived}: [t461 f32 [H=28 W=28 C=96] {derived} ->[n147]] =
        batch_norm
          x=t460 {derived} <-n145
          weight=t55 {pt2=root:p_blocks_3_0_bn1_weight target=blocks.3.0.bn1.weight}
          bias=t56 {pt2=root:p_blocks_3_0_bn1_bias target=blocks.3.0.bn1.bias}
          running_mean=t212 {pt2=root:b_blocks_3_0_bn1_running_mean target=blocks.3.0.bn1.running_mean}
          running_var=t213 {pt2=root:b_blocks_3_0_bn1_running_var target=blocks.3.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n147 {pt2=root[52] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_18)}: [t462 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_54} ->[n148]] =
        permute x=t461 {derived} <-n146 perm=[H<-C, W<-H, C<-W]
    n148 {pt2=root[53] torch.ops.aten.hardtanh.default (hardtanh_12)}: [t463 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:hardtanh_12} ->[n149]] =
      hardtanh
        x=t462 {pt2=root:getitem_54} <-n147
        params={min_val=0; max_val=6}
    group g39 torch.ops.aten.conv2d.default:
      n149 {derived}: [t464 f32 [H=28 W=28 C=96] {derived} ->[n151]] =
        permute x=t463 {pt2=root:hardtanh_12} <-n148 perm=[H<-W, W<-C, C<-H]
      n150 {derived}: [t465 f32 [N=96 T=1 D=1 H=3 W=3 C=1] {derived} ->[n151]] =
        permute
          x=t57 {pt2=root:p_blocks_3_0_conv_dw_weight target=blocks.3.0.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n151 {derived}: [t466 f32 [H=14 W=14 C=96] {derived} ->[n152]] =
        conv2d
          x=t464 {derived} <-n149
          weight=t465 {derived} <-n150
          bias=none
          params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 in_channels=96;
                 groups=96}
      n152 {pt2=root[54] torch.ops.aten.conv2d.default (conv2d_19)}: [t467 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_19} ->[n153]] =
        permute x=t466 {derived} <-n151 perm=[H<-C, W<-H, C<-W]
    group g40 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n153 {derived}: [t468 f32 [H=14 W=14 C=96] {derived} ->[n154]] =
        permute x=t467 {pt2=root:conv2d_19} <-n152 perm=[H<-W, W<-C, C<-H]
      n154 {derived}: [t469 f32 [H=14 W=14 C=96] {derived} ->[n155]] =
        batch_norm
          x=t468 {derived} <-n153
          weight=t58 {pt2=root:p_blocks_3_0_bn2_weight target=blocks.3.0.bn2.weight}
          bias=t59 {pt2=root:p_blocks_3_0_bn2_bias target=blocks.3.0.bn2.bias}
          running_mean=t215 {pt2=root:b_blocks_3_0_bn2_running_mean target=blocks.3.0.bn2.running_mean}
          running_var=t216 {pt2=root:b_blocks_3_0_bn2_running_var target=blocks.3.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n155 {pt2=root[55] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_19)}: [t470 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_57} ->[n156]] =
        permute x=t469 {derived} <-n154 perm=[H<-C, W<-H, C<-W]
    n156 {pt2=root[56] torch.ops.aten.hardtanh.default (hardtanh_13)}: [t471 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_13} ->[n157]] =
      hardtanh
        x=t470 {pt2=root:getitem_57} <-n155
        params={min_val=0; max_val=6}
    group g41 torch.ops.aten.conv2d.default:
      n157 {derived}: [t472 f32 [H=14 W=14 C=96] {derived} ->[n159]] =
        permute x=t471 {pt2=root:hardtanh_13} <-n156 perm=[H<-W, W<-C, C<-H]
      n158 {derived}: [t473 f32 [N=32 T=1 D=1 H=1 W=1 C=96] {derived} ->[n159]] =
        permute
          x=t60 {pt2=root:p_blocks_3_0_conv_pwl_weight target=blocks.3.0.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n159 {derived}: [t474 f32 [H=14 W=14 C=32] {derived} ->[n160]] =
        conv2d
          x=t472 {derived} <-n157
          weight=t473 {derived} <-n158
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=96;
                 groups=1}
      n160 {pt2=root[57] torch.ops.aten.conv2d.default (conv2d_20)}: [t475 f32 [H=32
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_20} ->[n161]] =
        permute x=t474 {derived} <-n159 perm=[H<-C, W<-H, C<-W]
    group g42 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n161 {derived}: [t476 f32 [H=14 W=14 C=32] {derived} ->[n162]] =
        permute x=t475 {pt2=root:conv2d_20} <-n160 perm=[H<-W, W<-C, C<-H]
      n162 {derived}: [t477 f32 [H=14 W=14 C=32] {derived} ->[n163]] =
        batch_norm
          x=t476 {derived} <-n161
          weight=t61 {pt2=root:p_blocks_3_0_bn3_weight target=blocks.3.0.bn3.weight}
          bias=t62 {pt2=root:p_blocks_3_0_bn3_bias target=blocks.3.0.bn3.bias}
          running_mean=t218 {pt2=root:b_blocks_3_0_bn3_running_mean target=blocks.3.0.bn3.running_mean}
          running_var=t219 {pt2=root:b_blocks_3_0_bn3_running_var target=blocks.3.0.bn3.running_var}
          params={channel=C; eps=1e-05}
      n163 {pt2=root[58] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_20)}: [t478 f32 [H=32
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_60} ->[n164,
                                                                      n187]] =
        permute x=t477 {derived} <-n162 perm=[H<-C, W<-H, C<-W]
    group g43 torch.ops.aten.conv2d.default:
      n164 {derived}: [t479 f32 [H=14 W=14 C=32] {derived} ->[n166]] =
        permute x=t478 {pt2=root:getitem_60} <-n163 perm=[H<-W, W<-C, C<-H]
      n165 {derived}: [t480 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {derived} ->[n166]] =
        permute
          x=t63 {pt2=root:p_blocks_3_1_conv_pw_weight target=blocks.3.1.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n166 {derived}: [t481 f32 [H=14 W=14 C=192] {derived} ->[n167]] =
        conv2d
          x=t479 {derived} <-n164
          weight=t480 {derived} <-n165
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=32;
                 groups=1}
      n167 {pt2=root[59] torch.ops.aten.conv2d.default (conv2d_21)}: [t482 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_21} ->[n168]] =
        permute x=t481 {derived} <-n166 perm=[H<-C, W<-H, C<-W]
    group g44 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n168 {derived}: [t483 f32 [H=14 W=14 C=192] {derived} ->[n169]] =
        permute x=t482 {pt2=root:conv2d_21} <-n167 perm=[H<-W, W<-C, C<-H]
      n169 {derived}: [t484 f32 [H=14 W=14 C=192] {derived} ->[n170]] =
        batch_norm
          x=t483 {derived} <-n168
          weight=t64 {pt2=root:p_blocks_3_1_bn1_weight target=blocks.3.1.bn1.weight}
          bias=t65 {pt2=root:p_blocks_3_1_bn1_bias target=blocks.3.1.bn1.bias}
          running_mean=t221 {pt2=root:b_blocks_3_1_bn1_running_mean target=blocks.3.1.bn1.running_mean}
          running_var=t222 {pt2=root:b_blocks_3_1_bn1_running_var target=blocks.3.1.bn1.running_var}
          params={channel=C; eps=1e-05}
      n170 {pt2=root[60] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_21)}: [t485 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_63} ->[n171]] =
        permute x=t484 {derived} <-n169 perm=[H<-C, W<-H, C<-W]
    n171 {pt2=root[61] torch.ops.aten.hardtanh.default (hardtanh_14)}: [t486 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_14} ->[n172]] =
      hardtanh
        x=t485 {pt2=root:getitem_63} <-n170
        params={min_val=0; max_val=6}
    group g45 torch.ops.aten.conv2d.default:
      n172 {derived}: [t487 f32 [H=14 W=14 C=192] {derived} ->[n174]] =
        permute x=t486 {pt2=root:hardtanh_14} <-n171 perm=[H<-W, W<-C, C<-H]
      n173 {derived}: [t488 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {derived} ->[n174]] =
        permute
          x=t66 {pt2=root:p_blocks_3_1_conv_dw_weight target=blocks.3.1.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n174 {derived}: [t489 f32 [H=14 W=14 C=192] {derived} ->[n175]] =
        conv2d
          x=t487 {derived} <-n172
          weight=t488 {derived} <-n173
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=192;
                 groups=192}
      n175 {pt2=root[62] torch.ops.aten.conv2d.default (conv2d_22)}: [t490 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_22} ->[n176]] =
        permute x=t489 {derived} <-n174 perm=[H<-C, W<-H, C<-W]
    group g46 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n176 {derived}: [t491 f32 [H=14 W=14 C=192] {derived} ->[n177]] =
        permute x=t490 {pt2=root:conv2d_22} <-n175 perm=[H<-W, W<-C, C<-H]
      n177 {derived}: [t492 f32 [H=14 W=14 C=192] {derived} ->[n178]] =
        batch_norm
          x=t491 {derived} <-n176
          weight=t67 {pt2=root:p_blocks_3_1_bn2_weight target=blocks.3.1.bn2.weight}
          bias=t68 {pt2=root:p_blocks_3_1_bn2_bias target=blocks.3.1.bn2.bias}
          running_mean=t224 {pt2=root:b_blocks_3_1_bn2_running_mean target=blocks.3.1.bn2.running_mean}
          running_var=t225 {pt2=root:b_blocks_3_1_bn2_running_var target=blocks.3.1.bn2.running_var}
          params={channel=C; eps=1e-05}
      n178 {pt2=root[63] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_22)}: [t493 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_66} ->[n179]] =
        permute x=t492 {derived} <-n177 perm=[H<-C, W<-H, C<-W]
    n179 {pt2=root[64] torch.ops.aten.hardtanh.default (hardtanh_15)}: [t494 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_15} ->[n180]] =
      hardtanh
        x=t493 {pt2=root:getitem_66} <-n178
        params={min_val=0; max_val=6}
    group g47 torch.ops.aten.conv2d.default:
      n180 {derived}: [t495 f32 [H=14 W=14 C=192] {derived} ->[n182]] =
        permute x=t494 {pt2=root:hardtanh_15} <-n179 perm=[H<-W, W<-C, C<-H]
      n181 {derived}: [t496 f32 [N=32 T=1 D=1 H=1 W=1 C=192] {derived} ->[n182]] =
        permute
          x=t69 {pt2=root:p_blocks_3_1_conv_pwl_weight target=blocks.3.1.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n182 {derived}: [t497 f32 [H=14 W=14 C=32] {derived} ->[n183]] =
        conv2d
          x=t495 {derived} <-n180
          weight=t496 {derived} <-n181
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=192;
                 groups=1}
      n183 {pt2=root[65] torch.ops.aten.conv2d.default (conv2d_23)}: [t498 f32 [H=32
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_23} ->[n184]] =
        permute x=t497 {derived} <-n182 perm=[H<-C, W<-H, C<-W]
    group g48 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n184 {derived}: [t499 f32 [H=14 W=14 C=32] {derived} ->[n185]] =
        permute x=t498 {pt2=root:conv2d_23} <-n183 perm=[H<-W, W<-C, C<-H]
      n185 {derived}: [t500 f32 [H=14 W=14 C=32] {derived} ->[n186]] =
        batch_norm
          x=t499 {derived} <-n184
          weight=t70 {pt2=root:p_blocks_3_1_bn3_weight target=blocks.3.1.bn3.weight}
          bias=t71 {pt2=root:p_blocks_3_1_bn3_bias target=blocks.3.1.bn3.bias}
          running_mean=t227 {pt2=root:b_blocks_3_1_bn3_running_mean target=blocks.3.1.bn3.running_mean}
          running_var=t228 {pt2=root:b_blocks_3_1_bn3_running_var target=blocks.3.1.bn3.running_var}
          params={channel=C; eps=1e-05}
      n186 {pt2=root[66] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_23)}: [t501 f32 [H=32
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_69} ->[n187]] =
        permute x=t500 {derived} <-n185 perm=[H<-C, W<-H, C<-W]
    n187 {pt2=root[67] torch.ops.aten.add.Tensor (add_3)}: [t502 f32 [H=32 W=14
                                                                      C=14] {pt2=root:add_3} ->[n188,
                                                                      n211]] =
      add
        a=t501 {pt2=root:getitem_69} <-n186
        b=t478 {pt2=root:getitem_60} <-n163
    group g49 torch.ops.aten.conv2d.default:
      n188 {derived}: [t503 f32 [H=14 W=14 C=32] {derived} ->[n190]] =
        permute x=t502 {pt2=root:add_3} <-n187 perm=[H<-W, W<-C, C<-H]
      n189 {derived}: [t504 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {derived} ->[n190]] =
        permute
          x=t72 {pt2=root:p_blocks_3_2_conv_pw_weight target=blocks.3.2.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n190 {derived}: [t505 f32 [H=14 W=14 C=192] {derived} ->[n191]] =
        conv2d
          x=t503 {derived} <-n188
          weight=t504 {derived} <-n189
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=32;
                 groups=1}
      n191 {pt2=root[68] torch.ops.aten.conv2d.default (conv2d_24)}: [t506 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_24} ->[n192]] =
        permute x=t505 {derived} <-n190 perm=[H<-C, W<-H, C<-W]
    group g50 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n192 {derived}: [t507 f32 [H=14 W=14 C=192] {derived} ->[n193]] =
        permute x=t506 {pt2=root:conv2d_24} <-n191 perm=[H<-W, W<-C, C<-H]
      n193 {derived}: [t508 f32 [H=14 W=14 C=192] {derived} ->[n194]] =
        batch_norm
          x=t507 {derived} <-n192
          weight=t73 {pt2=root:p_blocks_3_2_bn1_weight target=blocks.3.2.bn1.weight}
          bias=t74 {pt2=root:p_blocks_3_2_bn1_bias target=blocks.3.2.bn1.bias}
          running_mean=t230 {pt2=root:b_blocks_3_2_bn1_running_mean target=blocks.3.2.bn1.running_mean}
          running_var=t231 {pt2=root:b_blocks_3_2_bn1_running_var target=blocks.3.2.bn1.running_var}
          params={channel=C; eps=1e-05}
      n194 {pt2=root[69] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_24)}: [t509 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_72} ->[n195]] =
        permute x=t508 {derived} <-n193 perm=[H<-C, W<-H, C<-W]
    n195 {pt2=root[70] torch.ops.aten.hardtanh.default (hardtanh_16)}: [t510 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_16} ->[n196]] =
      hardtanh
        x=t509 {pt2=root:getitem_72} <-n194
        params={min_val=0; max_val=6}
    group g51 torch.ops.aten.conv2d.default:
      n196 {derived}: [t511 f32 [H=14 W=14 C=192] {derived} ->[n198]] =
        permute x=t510 {pt2=root:hardtanh_16} <-n195 perm=[H<-W, W<-C, C<-H]
      n197 {derived}: [t512 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {derived} ->[n198]] =
        permute
          x=t75 {pt2=root:p_blocks_3_2_conv_dw_weight target=blocks.3.2.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n198 {derived}: [t513 f32 [H=14 W=14 C=192] {derived} ->[n199]] =
        conv2d
          x=t511 {derived} <-n196
          weight=t512 {derived} <-n197
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=192;
                 groups=192}
      n199 {pt2=root[71] torch.ops.aten.conv2d.default (conv2d_25)}: [t514 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_25} ->[n200]] =
        permute x=t513 {derived} <-n198 perm=[H<-C, W<-H, C<-W]
    group g52 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n200 {derived}: [t515 f32 [H=14 W=14 C=192] {derived} ->[n201]] =
        permute x=t514 {pt2=root:conv2d_25} <-n199 perm=[H<-W, W<-C, C<-H]
      n201 {derived}: [t516 f32 [H=14 W=14 C=192] {derived} ->[n202]] =
        batch_norm
          x=t515 {derived} <-n200
          weight=t76 {pt2=root:p_blocks_3_2_bn2_weight target=blocks.3.2.bn2.weight}
          bias=t77 {pt2=root:p_blocks_3_2_bn2_bias target=blocks.3.2.bn2.bias}
          running_mean=t233 {pt2=root:b_blocks_3_2_bn2_running_mean target=blocks.3.2.bn2.running_mean}
          running_var=t234 {pt2=root:b_blocks_3_2_bn2_running_var target=blocks.3.2.bn2.running_var}
          params={channel=C; eps=1e-05}
      n202 {pt2=root[72] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_25)}: [t517 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_75} ->[n203]] =
        permute x=t516 {derived} <-n201 perm=[H<-C, W<-H, C<-W]
    n203 {pt2=root[73] torch.ops.aten.hardtanh.default (hardtanh_17)}: [t518 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_17} ->[n204]] =
      hardtanh
        x=t517 {pt2=root:getitem_75} <-n202
        params={min_val=0; max_val=6}
    group g53 torch.ops.aten.conv2d.default:
      n204 {derived}: [t519 f32 [H=14 W=14 C=192] {derived} ->[n206]] =
        permute x=t518 {pt2=root:hardtanh_17} <-n203 perm=[H<-W, W<-C, C<-H]
      n205 {derived}: [t520 f32 [N=32 T=1 D=1 H=1 W=1 C=192] {derived} ->[n206]] =
        permute
          x=t78 {pt2=root:p_blocks_3_2_conv_pwl_weight target=blocks.3.2.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n206 {derived}: [t521 f32 [H=14 W=14 C=32] {derived} ->[n207]] =
        conv2d
          x=t519 {derived} <-n204
          weight=t520 {derived} <-n205
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=192;
                 groups=1}
      n207 {pt2=root[74] torch.ops.aten.conv2d.default (conv2d_26)}: [t522 f32 [H=32
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_26} ->[n208]] =
        permute x=t521 {derived} <-n206 perm=[H<-C, W<-H, C<-W]
    group g54 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n208 {derived}: [t523 f32 [H=14 W=14 C=32] {derived} ->[n209]] =
        permute x=t522 {pt2=root:conv2d_26} <-n207 perm=[H<-W, W<-C, C<-H]
      n209 {derived}: [t524 f32 [H=14 W=14 C=32] {derived} ->[n210]] =
        batch_norm
          x=t523 {derived} <-n208
          weight=t79 {pt2=root:p_blocks_3_2_bn3_weight target=blocks.3.2.bn3.weight}
          bias=t80 {pt2=root:p_blocks_3_2_bn3_bias target=blocks.3.2.bn3.bias}
          running_mean=t236 {pt2=root:b_blocks_3_2_bn3_running_mean target=blocks.3.2.bn3.running_mean}
          running_var=t237 {pt2=root:b_blocks_3_2_bn3_running_var target=blocks.3.2.bn3.running_var}
          params={channel=C; eps=1e-05}
      n210 {pt2=root[75] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_26)}: [t525 f32 [H=32
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_78} ->[n211]] =
        permute x=t524 {derived} <-n209 perm=[H<-C, W<-H, C<-W]
    n211 {pt2=root[76] torch.ops.aten.add.Tensor (add_4)}: [t526 f32 [H=32 W=14
                                                                      C=14] {pt2=root:add_4} ->[n212,
                                                                      n235]] =
      add a=t525 {pt2=root:getitem_78} <-n210 b=t502 {pt2=root:add_3} <-n187
    group g55 torch.ops.aten.conv2d.default:
      n212 {derived}: [t527 f32 [H=14 W=14 C=32] {derived} ->[n214]] =
        permute x=t526 {pt2=root:add_4} <-n211 perm=[H<-W, W<-C, C<-H]
      n213 {derived}: [t528 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {derived} ->[n214]] =
        permute
          x=t81 {pt2=root:p_blocks_3_3_conv_pw_weight target=blocks.3.3.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n214 {derived}: [t529 f32 [H=14 W=14 C=192] {derived} ->[n215]] =
        conv2d
          x=t527 {derived} <-n212
          weight=t528 {derived} <-n213
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=32;
                 groups=1}
      n215 {pt2=root[77] torch.ops.aten.conv2d.default (conv2d_27)}: [t530 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_27} ->[n216]] =
        permute x=t529 {derived} <-n214 perm=[H<-C, W<-H, C<-W]
    group g56 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n216 {derived}: [t531 f32 [H=14 W=14 C=192] {derived} ->[n217]] =
        permute x=t530 {pt2=root:conv2d_27} <-n215 perm=[H<-W, W<-C, C<-H]
      n217 {derived}: [t532 f32 [H=14 W=14 C=192] {derived} ->[n218]] =
        batch_norm
          x=t531 {derived} <-n216
          weight=t82 {pt2=root:p_blocks_3_3_bn1_weight target=blocks.3.3.bn1.weight}
          bias=t83 {pt2=root:p_blocks_3_3_bn1_bias target=blocks.3.3.bn1.bias}
          running_mean=t239 {pt2=root:b_blocks_3_3_bn1_running_mean target=blocks.3.3.bn1.running_mean}
          running_var=t240 {pt2=root:b_blocks_3_3_bn1_running_var target=blocks.3.3.bn1.running_var}
          params={channel=C; eps=1e-05}
      n218 {pt2=root[78] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_27)}: [t533 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_81} ->[n219]] =
        permute x=t532 {derived} <-n217 perm=[H<-C, W<-H, C<-W]
    n219 {pt2=root[79] torch.ops.aten.hardtanh.default (hardtanh_18)}: [t534 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_18} ->[n220]] =
      hardtanh
        x=t533 {pt2=root:getitem_81} <-n218
        params={min_val=0; max_val=6}
    group g57 torch.ops.aten.conv2d.default:
      n220 {derived}: [t535 f32 [H=14 W=14 C=192] {derived} ->[n222]] =
        permute x=t534 {pt2=root:hardtanh_18} <-n219 perm=[H<-W, W<-C, C<-H]
      n221 {derived}: [t536 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {derived} ->[n222]] =
        permute
          x=t84 {pt2=root:p_blocks_3_3_conv_dw_weight target=blocks.3.3.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n222 {derived}: [t537 f32 [H=14 W=14 C=192] {derived} ->[n223]] =
        conv2d
          x=t535 {derived} <-n220
          weight=t536 {derived} <-n221
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=192;
                 groups=192}
      n223 {pt2=root[80] torch.ops.aten.conv2d.default (conv2d_28)}: [t538 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_28} ->[n224]] =
        permute x=t537 {derived} <-n222 perm=[H<-C, W<-H, C<-W]
    group g58 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n224 {derived}: [t539 f32 [H=14 W=14 C=192] {derived} ->[n225]] =
        permute x=t538 {pt2=root:conv2d_28} <-n223 perm=[H<-W, W<-C, C<-H]
      n225 {derived}: [t540 f32 [H=14 W=14 C=192] {derived} ->[n226]] =
        batch_norm
          x=t539 {derived} <-n224
          weight=t85 {pt2=root:p_blocks_3_3_bn2_weight target=blocks.3.3.bn2.weight}
          bias=t86 {pt2=root:p_blocks_3_3_bn2_bias target=blocks.3.3.bn2.bias}
          running_mean=t242 {pt2=root:b_blocks_3_3_bn2_running_mean target=blocks.3.3.bn2.running_mean}
          running_var=t243 {pt2=root:b_blocks_3_3_bn2_running_var target=blocks.3.3.bn2.running_var}
          params={channel=C; eps=1e-05}
      n226 {pt2=root[81] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_28)}: [t541 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_84} ->[n227]] =
        permute x=t540 {derived} <-n225 perm=[H<-C, W<-H, C<-W]
    n227 {pt2=root[82] torch.ops.aten.hardtanh.default (hardtanh_19)}: [t542 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_19} ->[n228]] =
      hardtanh
        x=t541 {pt2=root:getitem_84} <-n226
        params={min_val=0; max_val=6}
    group g59 torch.ops.aten.conv2d.default:
      n228 {derived}: [t543 f32 [H=14 W=14 C=192] {derived} ->[n230]] =
        permute x=t542 {pt2=root:hardtanh_19} <-n227 perm=[H<-W, W<-C, C<-H]
      n229 {derived}: [t544 f32 [N=32 T=1 D=1 H=1 W=1 C=192] {derived} ->[n230]] =
        permute
          x=t87 {pt2=root:p_blocks_3_3_conv_pwl_weight target=blocks.3.3.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n230 {derived}: [t545 f32 [H=14 W=14 C=32] {derived} ->[n231]] =
        conv2d
          x=t543 {derived} <-n228
          weight=t544 {derived} <-n229
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=192;
                 groups=1}
      n231 {pt2=root[83] torch.ops.aten.conv2d.default (conv2d_29)}: [t546 f32 [H=32
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_29} ->[n232]] =
        permute x=t545 {derived} <-n230 perm=[H<-C, W<-H, C<-W]
    group g60 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n232 {derived}: [t547 f32 [H=14 W=14 C=32] {derived} ->[n233]] =
        permute x=t546 {pt2=root:conv2d_29} <-n231 perm=[H<-W, W<-C, C<-H]
      n233 {derived}: [t548 f32 [H=14 W=14 C=32] {derived} ->[n234]] =
        batch_norm
          x=t547 {derived} <-n232
          weight=t88 {pt2=root:p_blocks_3_3_bn3_weight target=blocks.3.3.bn3.weight}
          bias=t89 {pt2=root:p_blocks_3_3_bn3_bias target=blocks.3.3.bn3.bias}
          running_mean=t245 {pt2=root:b_blocks_3_3_bn3_running_mean target=blocks.3.3.bn3.running_mean}
          running_var=t246 {pt2=root:b_blocks_3_3_bn3_running_var target=blocks.3.3.bn3.running_var}
          params={channel=C; eps=1e-05}
      n234 {pt2=root[84] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_29)}: [t549 f32 [H=32
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_87} ->[n235]] =
        permute x=t548 {derived} <-n233 perm=[H<-C, W<-H, C<-W]
    n235 {pt2=root[85] torch.ops.aten.add.Tensor (add_5)}: [t550 f32 [H=32 W=14
                                                                      C=14] {pt2=root:add_5} ->[n236]] =
      add a=t549 {pt2=root:getitem_87} <-n234 b=t526 {pt2=root:add_4} <-n211
    group g61 torch.ops.aten.conv2d.default:
      n236 {derived}: [t551 f32 [H=14 W=14 C=32] {derived} ->[n238]] =
        permute x=t550 {pt2=root:add_5} <-n235 perm=[H<-W, W<-C, C<-H]
      n237 {derived}: [t552 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {derived} ->[n238]] =
        permute
          x=t90 {pt2=root:p_blocks_4_0_conv_pw_weight target=blocks.4.0.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n238 {derived}: [t553 f32 [H=14 W=14 C=192] {derived} ->[n239]] =
        conv2d
          x=t551 {derived} <-n236
          weight=t552 {derived} <-n237
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=32;
                 groups=1}
      n239 {pt2=root[86] torch.ops.aten.conv2d.default (conv2d_30)}: [t554 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_30} ->[n240]] =
        permute x=t553 {derived} <-n238 perm=[H<-C, W<-H, C<-W]
    group g62 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n240 {derived}: [t555 f32 [H=14 W=14 C=192] {derived} ->[n241]] =
        permute x=t554 {pt2=root:conv2d_30} <-n239 perm=[H<-W, W<-C, C<-H]
      n241 {derived}: [t556 f32 [H=14 W=14 C=192] {derived} ->[n242]] =
        batch_norm
          x=t555 {derived} <-n240
          weight=t91 {pt2=root:p_blocks_4_0_bn1_weight target=blocks.4.0.bn1.weight}
          bias=t92 {pt2=root:p_blocks_4_0_bn1_bias target=blocks.4.0.bn1.bias}
          running_mean=t248 {pt2=root:b_blocks_4_0_bn1_running_mean target=blocks.4.0.bn1.running_mean}
          running_var=t249 {pt2=root:b_blocks_4_0_bn1_running_var target=blocks.4.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n242 {pt2=root[87] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_30)}: [t557 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_90} ->[n243]] =
        permute x=t556 {derived} <-n241 perm=[H<-C, W<-H, C<-W]
    n243 {pt2=root[88] torch.ops.aten.hardtanh.default (hardtanh_20)}: [t558 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_20} ->[n244]] =
      hardtanh
        x=t557 {pt2=root:getitem_90} <-n242
        params={min_val=0; max_val=6}
    group g63 torch.ops.aten.conv2d.default:
      n244 {derived}: [t559 f32 [H=14 W=14 C=192] {derived} ->[n246]] =
        permute x=t558 {pt2=root:hardtanh_20} <-n243 perm=[H<-W, W<-C, C<-H]
      n245 {derived}: [t560 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {derived} ->[n246]] =
        permute
          x=t93 {pt2=root:p_blocks_4_0_conv_dw_weight target=blocks.4.0.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n246 {derived}: [t561 f32 [H=14 W=14 C=192] {derived} ->[n247]] =
        conv2d
          x=t559 {derived} <-n244
          weight=t560 {derived} <-n245
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=192;
                 groups=192}
      n247 {pt2=root[89] torch.ops.aten.conv2d.default (conv2d_31)}: [t562 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_31} ->[n248]] =
        permute x=t561 {derived} <-n246 perm=[H<-C, W<-H, C<-W]
    group g64 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n248 {derived}: [t563 f32 [H=14 W=14 C=192] {derived} ->[n249]] =
        permute x=t562 {pt2=root:conv2d_31} <-n247 perm=[H<-W, W<-C, C<-H]
      n249 {derived}: [t564 f32 [H=14 W=14 C=192] {derived} ->[n250]] =
        batch_norm
          x=t563 {derived} <-n248
          weight=t94 {pt2=root:p_blocks_4_0_bn2_weight target=blocks.4.0.bn2.weight}
          bias=t95 {pt2=root:p_blocks_4_0_bn2_bias target=blocks.4.0.bn2.bias}
          running_mean=t251 {pt2=root:b_blocks_4_0_bn2_running_mean target=blocks.4.0.bn2.running_mean}
          running_var=t252 {pt2=root:b_blocks_4_0_bn2_running_var target=blocks.4.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n250 {pt2=root[90] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_31)}: [t565 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_93} ->[n251]] =
        permute x=t564 {derived} <-n249 perm=[H<-C, W<-H, C<-W]
    n251 {pt2=root[91] torch.ops.aten.hardtanh.default (hardtanh_21)}: [t566 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_21} ->[n252]] =
      hardtanh
        x=t565 {pt2=root:getitem_93} <-n250
        params={min_val=0; max_val=6}
    group g65 torch.ops.aten.conv2d.default:
      n252 {derived}: [t567 f32 [H=14 W=14 C=192] {derived} ->[n254]] =
        permute x=t566 {pt2=root:hardtanh_21} <-n251 perm=[H<-W, W<-C, C<-H]
      n253 {derived}: [t568 f32 [N=48 T=1 D=1 H=1 W=1 C=192] {derived} ->[n254]] =
        permute
          x=t96 {pt2=root:p_blocks_4_0_conv_pwl_weight target=blocks.4.0.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n254 {derived}: [t569 f32 [H=14 W=14 C=48] {derived} ->[n255]] =
        conv2d
          x=t567 {derived} <-n252
          weight=t568 {derived} <-n253
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=192;
                 groups=1}
      n255 {pt2=root[92] torch.ops.aten.conv2d.default (conv2d_32)}: [t570 f32 [H=48
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_32} ->[n256]] =
        permute x=t569 {derived} <-n254 perm=[H<-C, W<-H, C<-W]
    group g66 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n256 {derived}: [t571 f32 [H=14 W=14 C=48] {derived} ->[n257]] =
        permute x=t570 {pt2=root:conv2d_32} <-n255 perm=[H<-W, W<-C, C<-H]
      n257 {derived}: [t572 f32 [H=14 W=14 C=48] {derived} ->[n258]] =
        batch_norm
          x=t571 {derived} <-n256
          weight=t97 {pt2=root:p_blocks_4_0_bn3_weight target=blocks.4.0.bn3.weight}
          bias=t98 {pt2=root:p_blocks_4_0_bn3_bias target=blocks.4.0.bn3.bias}
          running_mean=t254 {pt2=root:b_blocks_4_0_bn3_running_mean target=blocks.4.0.bn3.running_mean}
          running_var=t255 {pt2=root:b_blocks_4_0_bn3_running_var target=blocks.4.0.bn3.running_var}
          params={channel=C; eps=1e-05}
      n258 {pt2=root[93] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_32)}: [t573 f32 [H=48
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_96} ->[n259,
                                                                      n282]] =
        permute x=t572 {derived} <-n257 perm=[H<-C, W<-H, C<-W]
    group g67 torch.ops.aten.conv2d.default:
      n259 {derived}: [t574 f32 [H=14 W=14 C=48] {derived} ->[n261]] =
        permute x=t573 {pt2=root:getitem_96} <-n258 perm=[H<-W, W<-C, C<-H]
      n260 {derived}: [t575 f32 [N=288 T=1 D=1 H=1 W=1 C=48] {derived} ->[n261]] =
        permute
          x=t99 {pt2=root:p_blocks_4_1_conv_pw_weight target=blocks.4.1.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n261 {derived}: [t576 f32 [H=14 W=14 C=288] {derived} ->[n262]] =
        conv2d
          x=t574 {derived} <-n259
          weight=t575 {derived} <-n260
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=48;
                 groups=1}
      n262 {pt2=root[94] torch.ops.aten.conv2d.default (conv2d_33)}: [t577 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_33} ->[n263]] =
        permute x=t576 {derived} <-n261 perm=[H<-C, W<-H, C<-W]
    group g68 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n263 {derived}: [t578 f32 [H=14 W=14 C=288] {derived} ->[n264]] =
        permute x=t577 {pt2=root:conv2d_33} <-n262 perm=[H<-W, W<-C, C<-H]
      n264 {derived}: [t579 f32 [H=14 W=14 C=288] {derived} ->[n265]] =
        batch_norm
          x=t578 {derived} <-n263
          weight=t100 {pt2=root:p_blocks_4_1_bn1_weight target=blocks.4.1.bn1.weight}
          bias=t101 {pt2=root:p_blocks_4_1_bn1_bias target=blocks.4.1.bn1.bias}
          running_mean=t257 {pt2=root:b_blocks_4_1_bn1_running_mean target=blocks.4.1.bn1.running_mean}
          running_var=t258 {pt2=root:b_blocks_4_1_bn1_running_var target=blocks.4.1.bn1.running_var}
          params={channel=C; eps=1e-05}
      n265 {pt2=root[95] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_33)}: [t580 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_99} ->[n266]] =
        permute x=t579 {derived} <-n264 perm=[H<-C, W<-H, C<-W]
    n266 {pt2=root[96] torch.ops.aten.hardtanh.default (hardtanh_22)}: [t581 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_22} ->[n267]] =
      hardtanh
        x=t580 {pt2=root:getitem_99} <-n265
        params={min_val=0; max_val=6}
    group g69 torch.ops.aten.conv2d.default:
      n267 {derived}: [t582 f32 [H=14 W=14 C=288] {derived} ->[n269]] =
        permute x=t581 {pt2=root:hardtanh_22} <-n266 perm=[H<-W, W<-C, C<-H]
      n268 {derived}: [t583 f32 [N=288 T=1 D=1 H=3 W=3 C=1] {derived} ->[n269]] =
        permute
          x=t102 {pt2=root:p_blocks_4_1_conv_dw_weight target=blocks.4.1.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n269 {derived}: [t584 f32 [H=14 W=14 C=288] {derived} ->[n270]] =
        conv2d
          x=t582 {derived} <-n267
          weight=t583 {derived} <-n268
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=288;
                 groups=288}
      n270 {pt2=root[97] torch.ops.aten.conv2d.default (conv2d_34)}: [t585 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_34} ->[n271]] =
        permute x=t584 {derived} <-n269 perm=[H<-C, W<-H, C<-W]
    group g70 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n271 {derived}: [t586 f32 [H=14 W=14 C=288] {derived} ->[n272]] =
        permute x=t585 {pt2=root:conv2d_34} <-n270 perm=[H<-W, W<-C, C<-H]
      n272 {derived}: [t587 f32 [H=14 W=14 C=288] {derived} ->[n273]] =
        batch_norm
          x=t586 {derived} <-n271
          weight=t103 {pt2=root:p_blocks_4_1_bn2_weight target=blocks.4.1.bn2.weight}
          bias=t104 {pt2=root:p_blocks_4_1_bn2_bias target=blocks.4.1.bn2.bias}
          running_mean=t260 {pt2=root:b_blocks_4_1_bn2_running_mean target=blocks.4.1.bn2.running_mean}
          running_var=t261 {pt2=root:b_blocks_4_1_bn2_running_var target=blocks.4.1.bn2.running_var}
          params={channel=C; eps=1e-05}
      n273 {pt2=root[98] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_34)}: [t588 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_102} ->[n274]] =
        permute x=t587 {derived} <-n272 perm=[H<-C, W<-H, C<-W]
    n274 {pt2=root[99] torch.ops.aten.hardtanh.default (hardtanh_23)}: [t589 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_23} ->[n275]] =
      hardtanh
        x=t588 {pt2=root:getitem_102} <-n273
        params={min_val=0; max_val=6}
    group g71 torch.ops.aten.conv2d.default:
      n275 {derived}: [t590 f32 [H=14 W=14 C=288] {derived} ->[n277]] =
        permute x=t589 {pt2=root:hardtanh_23} <-n274 perm=[H<-W, W<-C, C<-H]
      n276 {derived}: [t591 f32 [N=48 T=1 D=1 H=1 W=1 C=288] {derived} ->[n277]] =
        permute
          x=t105 {pt2=root:p_blocks_4_1_conv_pwl_weight target=blocks.4.1.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n277 {derived}: [t592 f32 [H=14 W=14 C=48] {derived} ->[n278]] =
        conv2d
          x=t590 {derived} <-n275
          weight=t591 {derived} <-n276
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=288;
                 groups=1}
      n278 {pt2=root[100] torch.ops.aten.conv2d.default (conv2d_35)}: [t593 f32 [H=48
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_35} ->[n279]] =
        permute x=t592 {derived} <-n277 perm=[H<-C, W<-H, C<-W]
    group g72 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n279 {derived}: [t594 f32 [H=14 W=14 C=48] {derived} ->[n280]] =
        permute x=t593 {pt2=root:conv2d_35} <-n278 perm=[H<-W, W<-C, C<-H]
      n280 {derived}: [t595 f32 [H=14 W=14 C=48] {derived} ->[n281]] =
        batch_norm
          x=t594 {derived} <-n279
          weight=t106 {pt2=root:p_blocks_4_1_bn3_weight target=blocks.4.1.bn3.weight}
          bias=t107 {pt2=root:p_blocks_4_1_bn3_bias target=blocks.4.1.bn3.bias}
          running_mean=t263 {pt2=root:b_blocks_4_1_bn3_running_mean target=blocks.4.1.bn3.running_mean}
          running_var=t264 {pt2=root:b_blocks_4_1_bn3_running_var target=blocks.4.1.bn3.running_var}
          params={channel=C; eps=1e-05}
      n281 {pt2=root[101] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_35)}: [t596 f32 [H=48
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_105} ->[n282]] =
        permute x=t595 {derived} <-n280 perm=[H<-C, W<-H, C<-W]
    n282 {pt2=root[102] torch.ops.aten.add.Tensor (add_6)}: [t597 f32 [H=48
                                                                      W=14
                                                                      C=14] {pt2=root:add_6} ->[n283,
                                                                      n306]] =
      add
        a=t596 {pt2=root:getitem_105} <-n281
        b=t573 {pt2=root:getitem_96} <-n258
    group g73 torch.ops.aten.conv2d.default:
      n283 {derived}: [t598 f32 [H=14 W=14 C=48] {derived} ->[n285]] =
        permute x=t597 {pt2=root:add_6} <-n282 perm=[H<-W, W<-C, C<-H]
      n284 {derived}: [t599 f32 [N=288 T=1 D=1 H=1 W=1 C=48] {derived} ->[n285]] =
        permute
          x=t108 {pt2=root:p_blocks_4_2_conv_pw_weight target=blocks.4.2.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n285 {derived}: [t600 f32 [H=14 W=14 C=288] {derived} ->[n286]] =
        conv2d
          x=t598 {derived} <-n283
          weight=t599 {derived} <-n284
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=48;
                 groups=1}
      n286 {pt2=root[103] torch.ops.aten.conv2d.default (conv2d_36)}: [t601 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_36} ->[n287]] =
        permute x=t600 {derived} <-n285 perm=[H<-C, W<-H, C<-W]
    group g74 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n287 {derived}: [t602 f32 [H=14 W=14 C=288] {derived} ->[n288]] =
        permute x=t601 {pt2=root:conv2d_36} <-n286 perm=[H<-W, W<-C, C<-H]
      n288 {derived}: [t603 f32 [H=14 W=14 C=288] {derived} ->[n289]] =
        batch_norm
          x=t602 {derived} <-n287
          weight=t109 {pt2=root:p_blocks_4_2_bn1_weight target=blocks.4.2.bn1.weight}
          bias=t110 {pt2=root:p_blocks_4_2_bn1_bias target=blocks.4.2.bn1.bias}
          running_mean=t266 {pt2=root:b_blocks_4_2_bn1_running_mean target=blocks.4.2.bn1.running_mean}
          running_var=t267 {pt2=root:b_blocks_4_2_bn1_running_var target=blocks.4.2.bn1.running_var}
          params={channel=C; eps=1e-05}
      n289 {pt2=root[104] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_36)}: [t604 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_108} ->[n290]] =
        permute x=t603 {derived} <-n288 perm=[H<-C, W<-H, C<-W]
    n290 {pt2=root[105] torch.ops.aten.hardtanh.default (hardtanh_24)}: [t605 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_24} ->[n291]] =
      hardtanh
        x=t604 {pt2=root:getitem_108} <-n289
        params={min_val=0; max_val=6}
    group g75 torch.ops.aten.conv2d.default:
      n291 {derived}: [t606 f32 [H=14 W=14 C=288] {derived} ->[n293]] =
        permute x=t605 {pt2=root:hardtanh_24} <-n290 perm=[H<-W, W<-C, C<-H]
      n292 {derived}: [t607 f32 [N=288 T=1 D=1 H=3 W=3 C=1] {derived} ->[n293]] =
        permute
          x=t111 {pt2=root:p_blocks_4_2_conv_dw_weight target=blocks.4.2.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n293 {derived}: [t608 f32 [H=14 W=14 C=288] {derived} ->[n294]] =
        conv2d
          x=t606 {derived} <-n291
          weight=t607 {derived} <-n292
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=288;
                 groups=288}
      n294 {pt2=root[106] torch.ops.aten.conv2d.default (conv2d_37)}: [t609 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_37} ->[n295]] =
        permute x=t608 {derived} <-n293 perm=[H<-C, W<-H, C<-W]
    group g76 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n295 {derived}: [t610 f32 [H=14 W=14 C=288] {derived} ->[n296]] =
        permute x=t609 {pt2=root:conv2d_37} <-n294 perm=[H<-W, W<-C, C<-H]
      n296 {derived}: [t611 f32 [H=14 W=14 C=288] {derived} ->[n297]] =
        batch_norm
          x=t610 {derived} <-n295
          weight=t112 {pt2=root:p_blocks_4_2_bn2_weight target=blocks.4.2.bn2.weight}
          bias=t113 {pt2=root:p_blocks_4_2_bn2_bias target=blocks.4.2.bn2.bias}
          running_mean=t269 {pt2=root:b_blocks_4_2_bn2_running_mean target=blocks.4.2.bn2.running_mean}
          running_var=t270 {pt2=root:b_blocks_4_2_bn2_running_var target=blocks.4.2.bn2.running_var}
          params={channel=C; eps=1e-05}
      n297 {pt2=root[107] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_37)}: [t612 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_111} ->[n298]] =
        permute x=t611 {derived} <-n296 perm=[H<-C, W<-H, C<-W]
    n298 {pt2=root[108] torch.ops.aten.hardtanh.default (hardtanh_25)}: [t613 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_25} ->[n299]] =
      hardtanh
        x=t612 {pt2=root:getitem_111} <-n297
        params={min_val=0; max_val=6}
    group g77 torch.ops.aten.conv2d.default:
      n299 {derived}: [t614 f32 [H=14 W=14 C=288] {derived} ->[n301]] =
        permute x=t613 {pt2=root:hardtanh_25} <-n298 perm=[H<-W, W<-C, C<-H]
      n300 {derived}: [t615 f32 [N=48 T=1 D=1 H=1 W=1 C=288] {derived} ->[n301]] =
        permute
          x=t114 {pt2=root:p_blocks_4_2_conv_pwl_weight target=blocks.4.2.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n301 {derived}: [t616 f32 [H=14 W=14 C=48] {derived} ->[n302]] =
        conv2d
          x=t614 {derived} <-n299
          weight=t615 {derived} <-n300
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=288;
                 groups=1}
      n302 {pt2=root[109] torch.ops.aten.conv2d.default (conv2d_38)}: [t617 f32 [H=48
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_38} ->[n303]] =
        permute x=t616 {derived} <-n301 perm=[H<-C, W<-H, C<-W]
    group g78 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n303 {derived}: [t618 f32 [H=14 W=14 C=48] {derived} ->[n304]] =
        permute x=t617 {pt2=root:conv2d_38} <-n302 perm=[H<-W, W<-C, C<-H]
      n304 {derived}: [t619 f32 [H=14 W=14 C=48] {derived} ->[n305]] =
        batch_norm
          x=t618 {derived} <-n303
          weight=t115 {pt2=root:p_blocks_4_2_bn3_weight target=blocks.4.2.bn3.weight}
          bias=t116 {pt2=root:p_blocks_4_2_bn3_bias target=blocks.4.2.bn3.bias}
          running_mean=t272 {pt2=root:b_blocks_4_2_bn3_running_mean target=blocks.4.2.bn3.running_mean}
          running_var=t273 {pt2=root:b_blocks_4_2_bn3_running_var target=blocks.4.2.bn3.running_var}
          params={channel=C; eps=1e-05}
      n305 {pt2=root[110] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_38)}: [t620 f32 [H=48
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_114} ->[n306]] =
        permute x=t619 {derived} <-n304 perm=[H<-C, W<-H, C<-W]
    n306 {pt2=root[111] torch.ops.aten.add.Tensor (add_7)}: [t621 f32 [H=48
                                                                      W=14
                                                                      C=14] {pt2=root:add_7} ->[n307]] =
      add a=t620 {pt2=root:getitem_114} <-n305 b=t597 {pt2=root:add_6} <-n282
    group g79 torch.ops.aten.conv2d.default:
      n307 {derived}: [t622 f32 [H=14 W=14 C=48] {derived} ->[n309]] =
        permute x=t621 {pt2=root:add_7} <-n306 perm=[H<-W, W<-C, C<-H]
      n308 {derived}: [t623 f32 [N=288 T=1 D=1 H=1 W=1 C=48] {derived} ->[n309]] =
        permute
          x=t117 {pt2=root:p_blocks_5_0_conv_pw_weight target=blocks.5.0.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n309 {derived}: [t624 f32 [H=14 W=14 C=288] {derived} ->[n310]] =
        conv2d
          x=t622 {derived} <-n307
          weight=t623 {derived} <-n308
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=48;
                 groups=1}
      n310 {pt2=root[112] torch.ops.aten.conv2d.default (conv2d_39)}: [t625 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_39} ->[n311]] =
        permute x=t624 {derived} <-n309 perm=[H<-C, W<-H, C<-W]
    group g80 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n311 {derived}: [t626 f32 [H=14 W=14 C=288] {derived} ->[n312]] =
        permute x=t625 {pt2=root:conv2d_39} <-n310 perm=[H<-W, W<-C, C<-H]
      n312 {derived}: [t627 f32 [H=14 W=14 C=288] {derived} ->[n313]] =
        batch_norm
          x=t626 {derived} <-n311
          weight=t118 {pt2=root:p_blocks_5_0_bn1_weight target=blocks.5.0.bn1.weight}
          bias=t119 {pt2=root:p_blocks_5_0_bn1_bias target=blocks.5.0.bn1.bias}
          running_mean=t275 {pt2=root:b_blocks_5_0_bn1_running_mean target=blocks.5.0.bn1.running_mean}
          running_var=t276 {pt2=root:b_blocks_5_0_bn1_running_var target=blocks.5.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n313 {pt2=root[113] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_39)}: [t628 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_117} ->[n314]] =
        permute x=t627 {derived} <-n312 perm=[H<-C, W<-H, C<-W]
    n314 {pt2=root[114] torch.ops.aten.hardtanh.default (hardtanh_26)}: [t629 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_26} ->[n315]] =
      hardtanh
        x=t628 {pt2=root:getitem_117} <-n313
        params={min_val=0; max_val=6}
    group g81 torch.ops.aten.conv2d.default:
      n315 {derived}: [t630 f32 [H=14 W=14 C=288] {derived} ->[n317]] =
        permute x=t629 {pt2=root:hardtanh_26} <-n314 perm=[H<-W, W<-C, C<-H]
      n316 {derived}: [t631 f32 [N=288 T=1 D=1 H=3 W=3 C=1] {derived} ->[n317]] =
        permute
          x=t120 {pt2=root:p_blocks_5_0_conv_dw_weight target=blocks.5.0.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n317 {derived}: [t632 f32 [H=7 W=7 C=288] {derived} ->[n318]] =
        conv2d
          x=t630 {derived} <-n315
          weight=t631 {derived} <-n316
          bias=none
          params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 in_channels=288;
                 groups=288}
      n318 {pt2=root[115] torch.ops.aten.conv2d.default (conv2d_40)}: [t633 f32 [H=288
                                                                      W=7 C=7] {pt2=root:conv2d_40} ->[n319]] =
        permute x=t632 {derived} <-n317 perm=[H<-C, W<-H, C<-W]
    group g82 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n319 {derived}: [t634 f32 [H=7 W=7 C=288] {derived} ->[n320]] =
        permute x=t633 {pt2=root:conv2d_40} <-n318 perm=[H<-W, W<-C, C<-H]
      n320 {derived}: [t635 f32 [H=7 W=7 C=288] {derived} ->[n321]] =
        batch_norm
          x=t634 {derived} <-n319
          weight=t121 {pt2=root:p_blocks_5_0_bn2_weight target=blocks.5.0.bn2.weight}
          bias=t122 {pt2=root:p_blocks_5_0_bn2_bias target=blocks.5.0.bn2.bias}
          running_mean=t278 {pt2=root:b_blocks_5_0_bn2_running_mean target=blocks.5.0.bn2.running_mean}
          running_var=t279 {pt2=root:b_blocks_5_0_bn2_running_var target=blocks.5.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n321 {pt2=root[116] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_40)}: [t636 f32 [H=288
                                                                      W=7 C=7] {pt2=root:getitem_120} ->[n322]] =
        permute x=t635 {derived} <-n320 perm=[H<-C, W<-H, C<-W]
    n322 {pt2=root[117] torch.ops.aten.hardtanh.default (hardtanh_27)}: [t637 f32 [H=288
                                                                      W=7 C=7] {pt2=root:hardtanh_27} ->[n323]] =
      hardtanh
        x=t636 {pt2=root:getitem_120} <-n321
        params={min_val=0; max_val=6}
    group g83 torch.ops.aten.conv2d.default:
      n323 {derived}: [t638 f32 [H=7 W=7 C=288] {derived} ->[n325]] =
        permute x=t637 {pt2=root:hardtanh_27} <-n322 perm=[H<-W, W<-C, C<-H]
      n324 {derived}: [t639 f32 [N=80 T=1 D=1 H=1 W=1 C=288] {derived} ->[n325]] =
        permute
          x=t123 {pt2=root:p_blocks_5_0_conv_pwl_weight target=blocks.5.0.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n325 {derived}: [t640 f32 [H=7 W=7 C=80] {derived} ->[n326]] =
        conv2d
          x=t638 {derived} <-n323
          weight=t639 {derived} <-n324
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=288;
                 groups=1}
      n326 {pt2=root[118] torch.ops.aten.conv2d.default (conv2d_41)}: [t641 f32 [H=80
                                                                      W=7 C=7] {pt2=root:conv2d_41} ->[n327]] =
        permute x=t640 {derived} <-n325 perm=[H<-C, W<-H, C<-W]
    group g84 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n327 {derived}: [t642 f32 [H=7 W=7 C=80] {derived} ->[n328]] =
        permute x=t641 {pt2=root:conv2d_41} <-n326 perm=[H<-W, W<-C, C<-H]
      n328 {derived}: [t643 f32 [H=7 W=7 C=80] {derived} ->[n329]] =
        batch_norm
          x=t642 {derived} <-n327
          weight=t124 {pt2=root:p_blocks_5_0_bn3_weight target=blocks.5.0.bn3.weight}
          bias=t125 {pt2=root:p_blocks_5_0_bn3_bias target=blocks.5.0.bn3.bias}
          running_mean=t281 {pt2=root:b_blocks_5_0_bn3_running_mean target=blocks.5.0.bn3.running_mean}
          running_var=t282 {pt2=root:b_blocks_5_0_bn3_running_var target=blocks.5.0.bn3.running_var}
          params={channel=C; eps=1e-05}
      n329 {pt2=root[119] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_41)}: [t644 f32 [H=80
                                                                      W=7 C=7] {pt2=root:getitem_123} ->[n330,
                                                                      n353]] =
        permute x=t643 {derived} <-n328 perm=[H<-C, W<-H, C<-W]
    group g85 torch.ops.aten.conv2d.default:
      n330 {derived}: [t645 f32 [H=7 W=7 C=80] {derived} ->[n332]] =
        permute x=t644 {pt2=root:getitem_123} <-n329 perm=[H<-W, W<-C, C<-H]
      n331 {derived}: [t646 f32 [N=480 T=1 D=1 H=1 W=1 C=80] {derived} ->[n332]] =
        permute
          x=t126 {pt2=root:p_blocks_5_1_conv_pw_weight target=blocks.5.1.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n332 {derived}: [t647 f32 [H=7 W=7 C=480] {derived} ->[n333]] =
        conv2d
          x=t645 {derived} <-n330
          weight=t646 {derived} <-n331
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=80;
                 groups=1}
      n333 {pt2=root[120] torch.ops.aten.conv2d.default (conv2d_42)}: [t648 f32 [H=480
                                                                      W=7 C=7] {pt2=root:conv2d_42} ->[n334]] =
        permute x=t647 {derived} <-n332 perm=[H<-C, W<-H, C<-W]
    group g86 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n334 {derived}: [t649 f32 [H=7 W=7 C=480] {derived} ->[n335]] =
        permute x=t648 {pt2=root:conv2d_42} <-n333 perm=[H<-W, W<-C, C<-H]
      n335 {derived}: [t650 f32 [H=7 W=7 C=480] {derived} ->[n336]] =
        batch_norm
          x=t649 {derived} <-n334
          weight=t127 {pt2=root:p_blocks_5_1_bn1_weight target=blocks.5.1.bn1.weight}
          bias=t128 {pt2=root:p_blocks_5_1_bn1_bias target=blocks.5.1.bn1.bias}
          running_mean=t284 {pt2=root:b_blocks_5_1_bn1_running_mean target=blocks.5.1.bn1.running_mean}
          running_var=t285 {pt2=root:b_blocks_5_1_bn1_running_var target=blocks.5.1.bn1.running_var}
          params={channel=C; eps=1e-05}
      n336 {pt2=root[121] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_42)}: [t651 f32 [H=480
                                                                      W=7 C=7] {pt2=root:getitem_126} ->[n337]] =
        permute x=t650 {derived} <-n335 perm=[H<-C, W<-H, C<-W]
    n337 {pt2=root[122] torch.ops.aten.hardtanh.default (hardtanh_28)}: [t652 f32 [H=480
                                                                      W=7 C=7] {pt2=root:hardtanh_28} ->[n338]] =
      hardtanh
        x=t651 {pt2=root:getitem_126} <-n336
        params={min_val=0; max_val=6}
    group g87 torch.ops.aten.conv2d.default:
      n338 {derived}: [t653 f32 [H=7 W=7 C=480] {derived} ->[n340]] =
        permute x=t652 {pt2=root:hardtanh_28} <-n337 perm=[H<-W, W<-C, C<-H]
      n339 {derived}: [t654 f32 [N=480 T=1 D=1 H=3 W=3 C=1] {derived} ->[n340]] =
        permute
          x=t129 {pt2=root:p_blocks_5_1_conv_dw_weight target=blocks.5.1.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n340 {derived}: [t655 f32 [H=7 W=7 C=480] {derived} ->[n341]] =
        conv2d
          x=t653 {derived} <-n338
          weight=t654 {derived} <-n339
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=480;
                 groups=480}
      n341 {pt2=root[123] torch.ops.aten.conv2d.default (conv2d_43)}: [t656 f32 [H=480
                                                                      W=7 C=7] {pt2=root:conv2d_43} ->[n342]] =
        permute x=t655 {derived} <-n340 perm=[H<-C, W<-H, C<-W]
    group g88 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n342 {derived}: [t657 f32 [H=7 W=7 C=480] {derived} ->[n343]] =
        permute x=t656 {pt2=root:conv2d_43} <-n341 perm=[H<-W, W<-C, C<-H]
      n343 {derived}: [t658 f32 [H=7 W=7 C=480] {derived} ->[n344]] =
        batch_norm
          x=t657 {derived} <-n342
          weight=t130 {pt2=root:p_blocks_5_1_bn2_weight target=blocks.5.1.bn2.weight}
          bias=t131 {pt2=root:p_blocks_5_1_bn2_bias target=blocks.5.1.bn2.bias}
          running_mean=t287 {pt2=root:b_blocks_5_1_bn2_running_mean target=blocks.5.1.bn2.running_mean}
          running_var=t288 {pt2=root:b_blocks_5_1_bn2_running_var target=blocks.5.1.bn2.running_var}
          params={channel=C; eps=1e-05}
      n344 {pt2=root[124] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_43)}: [t659 f32 [H=480
                                                                      W=7 C=7] {pt2=root:getitem_129} ->[n345]] =
        permute x=t658 {derived} <-n343 perm=[H<-C, W<-H, C<-W]
    n345 {pt2=root[125] torch.ops.aten.hardtanh.default (hardtanh_29)}: [t660 f32 [H=480
                                                                      W=7 C=7] {pt2=root:hardtanh_29} ->[n346]] =
      hardtanh
        x=t659 {pt2=root:getitem_129} <-n344
        params={min_val=0; max_val=6}
    group g89 torch.ops.aten.conv2d.default:
      n346 {derived}: [t661 f32 [H=7 W=7 C=480] {derived} ->[n348]] =
        permute x=t660 {pt2=root:hardtanh_29} <-n345 perm=[H<-W, W<-C, C<-H]
      n347 {derived}: [t662 f32 [N=80 T=1 D=1 H=1 W=1 C=480] {derived} ->[n348]] =
        permute
          x=t132 {pt2=root:p_blocks_5_1_conv_pwl_weight target=blocks.5.1.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n348 {derived}: [t663 f32 [H=7 W=7 C=80] {derived} ->[n349]] =
        conv2d
          x=t661 {derived} <-n346
          weight=t662 {derived} <-n347
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=480;
                 groups=1}
      n349 {pt2=root[126] torch.ops.aten.conv2d.default (conv2d_44)}: [t664 f32 [H=80
                                                                      W=7 C=7] {pt2=root:conv2d_44} ->[n350]] =
        permute x=t663 {derived} <-n348 perm=[H<-C, W<-H, C<-W]
    group g90 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n350 {derived}: [t665 f32 [H=7 W=7 C=80] {derived} ->[n351]] =
        permute x=t664 {pt2=root:conv2d_44} <-n349 perm=[H<-W, W<-C, C<-H]
      n351 {derived}: [t666 f32 [H=7 W=7 C=80] {derived} ->[n352]] =
        batch_norm
          x=t665 {derived} <-n350
          weight=t133 {pt2=root:p_blocks_5_1_bn3_weight target=blocks.5.1.bn3.weight}
          bias=t134 {pt2=root:p_blocks_5_1_bn3_bias target=blocks.5.1.bn3.bias}
          running_mean=t290 {pt2=root:b_blocks_5_1_bn3_running_mean target=blocks.5.1.bn3.running_mean}
          running_var=t291 {pt2=root:b_blocks_5_1_bn3_running_var target=blocks.5.1.bn3.running_var}
          params={channel=C; eps=1e-05}
      n352 {pt2=root[127] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_44)}: [t667 f32 [H=80
                                                                      W=7 C=7] {pt2=root:getitem_132} ->[n353]] =
        permute x=t666 {derived} <-n351 perm=[H<-C, W<-H, C<-W]
    n353 {pt2=root[128] torch.ops.aten.add.Tensor (add_8)}: [t668 f32 [H=80 W=7
                                                                      C=7] {pt2=root:add_8} ->[n354,
                                                                      n377]] =
      add
        a=t667 {pt2=root:getitem_132} <-n352
        b=t644 {pt2=root:getitem_123} <-n329
    group g91 torch.ops.aten.conv2d.default:
      n354 {derived}: [t669 f32 [H=7 W=7 C=80] {derived} ->[n356]] =
        permute x=t668 {pt2=root:add_8} <-n353 perm=[H<-W, W<-C, C<-H]
      n355 {derived}: [t670 f32 [N=480 T=1 D=1 H=1 W=1 C=80] {derived} ->[n356]] =
        permute
          x=t135 {pt2=root:p_blocks_5_2_conv_pw_weight target=blocks.5.2.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n356 {derived}: [t671 f32 [H=7 W=7 C=480] {derived} ->[n357]] =
        conv2d
          x=t669 {derived} <-n354
          weight=t670 {derived} <-n355
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=80;
                 groups=1}
      n357 {pt2=root[129] torch.ops.aten.conv2d.default (conv2d_45)}: [t672 f32 [H=480
                                                                      W=7 C=7] {pt2=root:conv2d_45} ->[n358]] =
        permute x=t671 {derived} <-n356 perm=[H<-C, W<-H, C<-W]
    group g92 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n358 {derived}: [t673 f32 [H=7 W=7 C=480] {derived} ->[n359]] =
        permute x=t672 {pt2=root:conv2d_45} <-n357 perm=[H<-W, W<-C, C<-H]
      n359 {derived}: [t674 f32 [H=7 W=7 C=480] {derived} ->[n360]] =
        batch_norm
          x=t673 {derived} <-n358
          weight=t136 {pt2=root:p_blocks_5_2_bn1_weight target=blocks.5.2.bn1.weight}
          bias=t137 {pt2=root:p_blocks_5_2_bn1_bias target=blocks.5.2.bn1.bias}
          running_mean=t293 {pt2=root:b_blocks_5_2_bn1_running_mean target=blocks.5.2.bn1.running_mean}
          running_var=t294 {pt2=root:b_blocks_5_2_bn1_running_var target=blocks.5.2.bn1.running_var}
          params={channel=C; eps=1e-05}
      n360 {pt2=root[130] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_45)}: [t675 f32 [H=480
                                                                      W=7 C=7] {pt2=root:getitem_135} ->[n361]] =
        permute x=t674 {derived} <-n359 perm=[H<-C, W<-H, C<-W]
    n361 {pt2=root[131] torch.ops.aten.hardtanh.default (hardtanh_30)}: [t676 f32 [H=480
                                                                      W=7 C=7] {pt2=root:hardtanh_30} ->[n362]] =
      hardtanh
        x=t675 {pt2=root:getitem_135} <-n360
        params={min_val=0; max_val=6}
    group g93 torch.ops.aten.conv2d.default:
      n362 {derived}: [t677 f32 [H=7 W=7 C=480] {derived} ->[n364]] =
        permute x=t676 {pt2=root:hardtanh_30} <-n361 perm=[H<-W, W<-C, C<-H]
      n363 {derived}: [t678 f32 [N=480 T=1 D=1 H=3 W=3 C=1] {derived} ->[n364]] =
        permute
          x=t138 {pt2=root:p_blocks_5_2_conv_dw_weight target=blocks.5.2.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n364 {derived}: [t679 f32 [H=7 W=7 C=480] {derived} ->[n365]] =
        conv2d
          x=t677 {derived} <-n362
          weight=t678 {derived} <-n363
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=480;
                 groups=480}
      n365 {pt2=root[132] torch.ops.aten.conv2d.default (conv2d_46)}: [t680 f32 [H=480
                                                                      W=7 C=7] {pt2=root:conv2d_46} ->[n366]] =
        permute x=t679 {derived} <-n364 perm=[H<-C, W<-H, C<-W]
    group g94 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n366 {derived}: [t681 f32 [H=7 W=7 C=480] {derived} ->[n367]] =
        permute x=t680 {pt2=root:conv2d_46} <-n365 perm=[H<-W, W<-C, C<-H]
      n367 {derived}: [t682 f32 [H=7 W=7 C=480] {derived} ->[n368]] =
        batch_norm
          x=t681 {derived} <-n366
          weight=t139 {pt2=root:p_blocks_5_2_bn2_weight target=blocks.5.2.bn2.weight}
          bias=t140 {pt2=root:p_blocks_5_2_bn2_bias target=blocks.5.2.bn2.bias}
          running_mean=t296 {pt2=root:b_blocks_5_2_bn2_running_mean target=blocks.5.2.bn2.running_mean}
          running_var=t297 {pt2=root:b_blocks_5_2_bn2_running_var target=blocks.5.2.bn2.running_var}
          params={channel=C; eps=1e-05}
      n368 {pt2=root[133] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_46)}: [t683 f32 [H=480
                                                                      W=7 C=7] {pt2=root:getitem_138} ->[n369]] =
        permute x=t682 {derived} <-n367 perm=[H<-C, W<-H, C<-W]
    n369 {pt2=root[134] torch.ops.aten.hardtanh.default (hardtanh_31)}: [t684 f32 [H=480
                                                                      W=7 C=7] {pt2=root:hardtanh_31} ->[n370]] =
      hardtanh
        x=t683 {pt2=root:getitem_138} <-n368
        params={min_val=0; max_val=6}
    group g95 torch.ops.aten.conv2d.default:
      n370 {derived}: [t685 f32 [H=7 W=7 C=480] {derived} ->[n372]] =
        permute x=t684 {pt2=root:hardtanh_31} <-n369 perm=[H<-W, W<-C, C<-H]
      n371 {derived}: [t686 f32 [N=80 T=1 D=1 H=1 W=1 C=480] {derived} ->[n372]] =
        permute
          x=t141 {pt2=root:p_blocks_5_2_conv_pwl_weight target=blocks.5.2.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n372 {derived}: [t687 f32 [H=7 W=7 C=80] {derived} ->[n373]] =
        conv2d
          x=t685 {derived} <-n370
          weight=t686 {derived} <-n371
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=480;
                 groups=1}
      n373 {pt2=root[135] torch.ops.aten.conv2d.default (conv2d_47)}: [t688 f32 [H=80
                                                                      W=7 C=7] {pt2=root:conv2d_47} ->[n374]] =
        permute x=t687 {derived} <-n372 perm=[H<-C, W<-H, C<-W]
    group g96 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n374 {derived}: [t689 f32 [H=7 W=7 C=80] {derived} ->[n375]] =
        permute x=t688 {pt2=root:conv2d_47} <-n373 perm=[H<-W, W<-C, C<-H]
      n375 {derived}: [t690 f32 [H=7 W=7 C=80] {derived} ->[n376]] =
        batch_norm
          x=t689 {derived} <-n374
          weight=t142 {pt2=root:p_blocks_5_2_bn3_weight target=blocks.5.2.bn3.weight}
          bias=t143 {pt2=root:p_blocks_5_2_bn3_bias target=blocks.5.2.bn3.bias}
          running_mean=t299 {pt2=root:b_blocks_5_2_bn3_running_mean target=blocks.5.2.bn3.running_mean}
          running_var=t300 {pt2=root:b_blocks_5_2_bn3_running_var target=blocks.5.2.bn3.running_var}
          params={channel=C; eps=1e-05}
      n376 {pt2=root[136] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_47)}: [t691 f32 [H=80
                                                                      W=7 C=7] {pt2=root:getitem_141} ->[n377]] =
        permute x=t690 {derived} <-n375 perm=[H<-C, W<-H, C<-W]
    n377 {pt2=root[137] torch.ops.aten.add.Tensor (add_9)}: [t692 f32 [H=80 W=7
                                                                      C=7] {pt2=root:add_9} ->[n378]] =
      add a=t691 {pt2=root:getitem_141} <-n376 b=t668 {pt2=root:add_8} <-n353
    group g97 torch.ops.aten.conv2d.default:
      n378 {derived}: [t693 f32 [H=7 W=7 C=80] {derived} ->[n380]] =
        permute x=t692 {pt2=root:add_9} <-n377 perm=[H<-W, W<-C, C<-H]
      n379 {derived}: [t694 f32 [N=480 T=1 D=1 H=1 W=1 C=80] {derived} ->[n380]] =
        permute
          x=t144 {pt2=root:p_blocks_6_0_conv_pw_weight target=blocks.6.0.conv_pw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n380 {derived}: [t695 f32 [H=7 W=7 C=480] {derived} ->[n381]] =
        conv2d
          x=t693 {derived} <-n378
          weight=t694 {derived} <-n379
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=80;
                 groups=1}
      n381 {pt2=root[138] torch.ops.aten.conv2d.default (conv2d_48)}: [t696 f32 [H=480
                                                                      W=7 C=7] {pt2=root:conv2d_48} ->[n382]] =
        permute x=t695 {derived} <-n380 perm=[H<-C, W<-H, C<-W]
    group g98 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n382 {derived}: [t697 f32 [H=7 W=7 C=480] {derived} ->[n383]] =
        permute x=t696 {pt2=root:conv2d_48} <-n381 perm=[H<-W, W<-C, C<-H]
      n383 {derived}: [t698 f32 [H=7 W=7 C=480] {derived} ->[n384]] =
        batch_norm
          x=t697 {derived} <-n382
          weight=t145 {pt2=root:p_blocks_6_0_bn1_weight target=blocks.6.0.bn1.weight}
          bias=t146 {pt2=root:p_blocks_6_0_bn1_bias target=blocks.6.0.bn1.bias}
          running_mean=t302 {pt2=root:b_blocks_6_0_bn1_running_mean target=blocks.6.0.bn1.running_mean}
          running_var=t303 {pt2=root:b_blocks_6_0_bn1_running_var target=blocks.6.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n384 {pt2=root[139] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_48)}: [t699 f32 [H=480
                                                                      W=7 C=7] {pt2=root:getitem_144} ->[n385]] =
        permute x=t698 {derived} <-n383 perm=[H<-C, W<-H, C<-W]
    n385 {pt2=root[140] torch.ops.aten.hardtanh.default (hardtanh_32)}: [t700 f32 [H=480
                                                                      W=7 C=7] {pt2=root:hardtanh_32} ->[n386]] =
      hardtanh
        x=t699 {pt2=root:getitem_144} <-n384
        params={min_val=0; max_val=6}
    group g99 torch.ops.aten.conv2d.default:
      n386 {derived}: [t701 f32 [H=7 W=7 C=480] {derived} ->[n388]] =
        permute x=t700 {pt2=root:hardtanh_32} <-n385 perm=[H<-W, W<-C, C<-H]
      n387 {derived}: [t702 f32 [N=480 T=1 D=1 H=3 W=3 C=1] {derived} ->[n388]] =
        permute
          x=t147 {pt2=root:p_blocks_6_0_conv_dw_weight target=blocks.6.0.conv_dw.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n388 {derived}: [t703 f32 [H=7 W=7 C=480] {derived} ->[n389]] =
        conv2d
          x=t701 {derived} <-n386
          weight=t702 {derived} <-n387
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=480;
                 groups=480}
      n389 {pt2=root[141] torch.ops.aten.conv2d.default (conv2d_49)}: [t704 f32 [H=480
                                                                      W=7 C=7] {pt2=root:conv2d_49} ->[n390]] =
        permute x=t703 {derived} <-n388 perm=[H<-C, W<-H, C<-W]
    group g100 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n390 {derived}: [t705 f32 [H=7 W=7 C=480] {derived} ->[n391]] =
        permute x=t704 {pt2=root:conv2d_49} <-n389 perm=[H<-W, W<-C, C<-H]
      n391 {derived}: [t706 f32 [H=7 W=7 C=480] {derived} ->[n392]] =
        batch_norm
          x=t705 {derived} <-n390
          weight=t148 {pt2=root:p_blocks_6_0_bn2_weight target=blocks.6.0.bn2.weight}
          bias=t149 {pt2=root:p_blocks_6_0_bn2_bias target=blocks.6.0.bn2.bias}
          running_mean=t305 {pt2=root:b_blocks_6_0_bn2_running_mean target=blocks.6.0.bn2.running_mean}
          running_var=t306 {pt2=root:b_blocks_6_0_bn2_running_var target=blocks.6.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n392 {pt2=root[142] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_49)}: [t707 f32 [H=480
                                                                      W=7 C=7] {pt2=root:getitem_147} ->[n393]] =
        permute x=t706 {derived} <-n391 perm=[H<-C, W<-H, C<-W]
    n393 {pt2=root[143] torch.ops.aten.hardtanh.default (hardtanh_33)}: [t708 f32 [H=480
                                                                      W=7 C=7] {pt2=root:hardtanh_33} ->[n394]] =
      hardtanh
        x=t707 {pt2=root:getitem_147} <-n392
        params={min_val=0; max_val=6}
    group g101 torch.ops.aten.conv2d.default:
      n394 {derived}: [t709 f32 [H=7 W=7 C=480] {derived} ->[n396]] =
        permute x=t708 {pt2=root:hardtanh_33} <-n393 perm=[H<-W, W<-C, C<-H]
      n395 {derived}: [t710 f32 [N=160 T=1 D=1 H=1 W=1 C=480] {derived} ->[n396]] =
        permute
          x=t150 {pt2=root:p_blocks_6_0_conv_pwl_weight target=blocks.6.0.conv_pwl.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n396 {derived}: [t711 f32 [H=7 W=7 C=160] {derived} ->[n397]] =
        conv2d
          x=t709 {derived} <-n394
          weight=t710 {derived} <-n395
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=480;
                 groups=1}
      n397 {pt2=root[144] torch.ops.aten.conv2d.default (conv2d_50)}: [t712 f32 [H=160
                                                                      W=7 C=7] {pt2=root:conv2d_50} ->[n398]] =
        permute x=t711 {derived} <-n396 perm=[H<-C, W<-H, C<-W]
    group g102 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n398 {derived}: [t713 f32 [H=7 W=7 C=160] {derived} ->[n399]] =
        permute x=t712 {pt2=root:conv2d_50} <-n397 perm=[H<-W, W<-C, C<-H]
      n399 {derived}: [t714 f32 [H=7 W=7 C=160] {derived} ->[n400]] =
        batch_norm
          x=t713 {derived} <-n398
          weight=t151 {pt2=root:p_blocks_6_0_bn3_weight target=blocks.6.0.bn3.weight}
          bias=t152 {pt2=root:p_blocks_6_0_bn3_bias target=blocks.6.0.bn3.bias}
          running_mean=t308 {pt2=root:b_blocks_6_0_bn3_running_mean target=blocks.6.0.bn3.running_mean}
          running_var=t309 {pt2=root:b_blocks_6_0_bn3_running_var target=blocks.6.0.bn3.running_var}
          params={channel=C; eps=1e-05}
      n400 {pt2=root[145] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_50)}: [t715 f32 [H=160
                                                                      W=7 C=7] {pt2=root:getitem_150} ->[n401]] =
        permute x=t714 {derived} <-n399 perm=[H<-C, W<-H, C<-W]
    group g103 torch.ops.aten.conv2d.default:
      n401 {derived}: [t716 f32 [H=7 W=7 C=160] {derived} ->[n403]] =
        permute x=t715 {pt2=root:getitem_150} <-n400 perm=[H<-W, W<-C, C<-H]
      n402 {derived}: [t717 f32 [N=1280 T=1 D=1 H=1 W=1 C=160] {derived} ->[n403]] =
        permute
          x=t153 {pt2=root:p_conv_head_weight target=conv_head.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n403 {derived}: [t718 f32 [H=7 W=7 C=1280] {derived} ->[n404]] =
        conv2d
          x=t716 {derived} <-n401
          weight=t717 {derived} <-n402
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=160;
                 groups=1}
      n404 {pt2=root[146] torch.ops.aten.conv2d.default (conv2d_51)}: [t719 f32 [H=1280
                                                                      W=7 C=7] {pt2=root:conv2d_51} ->[n405]] =
        permute x=t718 {derived} <-n403 perm=[H<-C, W<-H, C<-W]
    group g104 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n405 {derived}: [t720 f32 [H=7 W=7 C=1280] {derived} ->[n406]] =
        permute x=t719 {pt2=root:conv2d_51} <-n404 perm=[H<-W, W<-C, C<-H]
      n406 {derived}: [t721 f32 [H=7 W=7 C=1280] {derived} ->[n407]] =
        batch_norm
          x=t720 {derived} <-n405
          weight=t154 {pt2=root:p_bn2_weight target=bn2.weight}
          bias=t155 {pt2=root:p_bn2_bias target=bn2.bias}
          running_mean=t311 {pt2=root:b_bn2_running_mean target=bn2.running_mean}
          running_var=t312 {pt2=root:b_bn2_running_var target=bn2.running_var}
          params={channel=C; eps=1e-05}
      n407 {pt2=root[147] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_51)}: [t722 f32 [H=1280
                                                                      W=7 C=7] {pt2=root:getitem_153} ->[n408]] =
        permute x=t721 {derived} <-n406 perm=[H<-C, W<-H, C<-W]
    n408 {pt2=root[148] torch.ops.aten.hardtanh.default (hardtanh_34)}: [t723 f32 [H=1280
                                                                      W=7 C=7] {pt2=root:hardtanh_34} ->[n409]] =
      hardtanh
        x=t722 {pt2=root:getitem_153} <-n407
        params={min_val=0; max_val=6}
    group g105 torch.ops.aten.adaptive_avg_pool2d.default:
      n409 {derived}: [t724 f32 [H=7 W=7 C=1280] {derived} ->[n410]] =
        permute x=t723 {pt2=root:hardtanh_34} <-n408 perm=[H<-W, W<-C, C<-H]
      n410 {derived}: [t725 f32 [C=1280] {derived} ->[n411]] =
        adaptive_avg_pool2d
          x=t724 {derived} <-n409
          params={output_size={h=1; w=1}}
      n411 {pt2=root[149] torch.ops.aten.adaptive_avg_pool2d.default (adaptive_avg_pool2d)}: [t726 f32 [H=1280
                                                                      W=1 C=1] {pt2=root:adaptive_avg_pool2d} ->[n412]] =
        permute x=t725 {derived} <-n410 perm=[H<-C, W<-H, C<-W]
    n412 {pt2=root[150] torch.ops.aten.view.default (view)}: [t727 f32 [C=1280] {pt2=root:view} ->[n414]] =
      reshape
        x=t726 {pt2=root:adaptive_avg_pool2d} <-n411
        params={shape=[C=1280]}
    group g106 torch.ops.aten.linear.default:
      n413 {derived}: [t728 f32 [N=1000 T=1 D=1 H=1 W=1 C=1280] {derived} ->[n414]] =
        permute
          x=t156 {pt2=root:p_classifier_weight target=classifier.weight}
          perm=[N<-W, W<-N]
      n414 {pt2=root[151] torch.ops.aten.linear.default (linear)}: [t729 f32 [C=1000] {pt2=root:linear}] =
        linear
          x=t727 {pt2=root:view} <-n412
          weight=t728 {derived} <-n413
          bias=t157 {pt2=root:p_classifier_bias target=classifier.bias}
          params={in_features=1280}
  outputs: [t729 f32 [C=1000] {pt2=root:linear} <-n414]
