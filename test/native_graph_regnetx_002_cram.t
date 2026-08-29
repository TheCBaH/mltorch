Import RegNetX-002's real exported graph as one native graph. `native_graph
print` uses the pure PT2 importer and keeps exporter-facing names and
captured payload targets in `Pt2_native_graph`.  RegNetX-002 stands in for
the retired resnet18 role model here: both are residual
conv/batch-norm/relu CNNs, and this is the primary, most-detailed model of
the three in this family (see also
`test/native_graph_mobilenetv2_050_cram.t` and
`test/native_graph_mobilenetv3_small_050_cram.t`).

`print` renders the native graph structure and every retained native-id-to-PT2
tensor/node mapping. Gated on PT2_DATA; run with `make pt2.runtest` after
`make pt2.download-cram`.

  $ ../bin/native_graph.exe print --pt2 "$PT2_DATA/regnetx_002/regnetx_002.pt2"
  native graph: inputs=267 constants=266 nodes=368 outputs=1
  PT2 provenance: tensor-origins=412 captured-targets=266 node-origins=145
  graph
  inputs:
    [t0 f32 [D=32 H=3 W=3 C=3] {pt2=root:p_stem_conv_weight target=stem.conv.weight} ->[n1] constant,
     t1 f32 [C=32] {pt2=root:p_stem_bn_weight target=stem.bn.weight} ->[n5] constant,
     t2 f32 [C=32] {pt2=root:p_stem_bn_bias target=stem.bn.bias} ->[n5] constant,
     t3 f32 [D=24 H=32 W=1 C=1] {pt2=root:p_s1_b1_conv1_conv_weight target=s1.b1.conv1.conv.weight} ->[n9] constant,
     t4 f32 [C=24] {pt2=root:p_s1_b1_conv1_bn_weight target=s1.b1.conv1.bn.weight} ->[n13] constant,
     t5 f32 [C=24] {pt2=root:p_s1_b1_conv1_bn_bias target=s1.b1.conv1.bn.bias} ->[n13] constant,
     t6 f32 [D=24 H=8 W=3 C=3] {pt2=root:p_s1_b1_conv2_conv_weight target=s1.b1.conv2.conv.weight} ->[n17] constant,
     t7 f32 [C=24] {pt2=root:p_s1_b1_conv2_bn_weight target=s1.b1.conv2.bn.weight} ->[n21] constant,
     t8 f32 [C=24] {pt2=root:p_s1_b1_conv2_bn_bias target=s1.b1.conv2.bn.bias} ->[n21] constant,
     t9 f32 [D=24 H=24 W=1 C=1] {pt2=root:p_s1_b1_conv3_conv_weight target=s1.b1.conv3.conv.weight} ->[n25] constant,
     t10 f32 [C=24] {pt2=root:p_s1_b1_conv3_bn_weight target=s1.b1.conv3.bn.weight} ->[n29] constant,
     t11 f32 [C=24] {pt2=root:p_s1_b1_conv3_bn_bias target=s1.b1.conv3.bn.bias} ->[n29] constant,
     t12 f32 [D=24 H=32 W=1 C=1] {pt2=root:p_s1_b1_downsample_conv_weight target=s1.b1.downsample.conv.weight} ->[n32] constant,
     t13 f32 [C=24] {pt2=root:p_s1_b1_downsample_bn_weight target=s1.b1.downsample.bn.weight} ->[n36] constant,
     t14 f32 [C=24] {pt2=root:p_s1_b1_downsample_bn_bias target=s1.b1.downsample.bn.bias} ->[n36] constant,
     t15 f32 [D=56 H=24 W=1 C=1] {pt2=root:p_s2_b1_conv1_conv_weight target=s2.b1.conv1.conv.weight} ->[n41] constant,
     t16 f32 [C=56] {pt2=root:p_s2_b1_conv1_bn_weight target=s2.b1.conv1.bn.weight} ->[n45] constant,
     t17 f32 [C=56] {pt2=root:p_s2_b1_conv1_bn_bias target=s2.b1.conv1.bn.bias} ->[n45] constant,
     t18 f32 [D=56 H=8 W=3 C=3] {pt2=root:p_s2_b1_conv2_conv_weight target=s2.b1.conv2.conv.weight} ->[n49] constant,
     t19 f32 [C=56] {pt2=root:p_s2_b1_conv2_bn_weight target=s2.b1.conv2.bn.weight} ->[n53] constant,
     t20 f32 [C=56] {pt2=root:p_s2_b1_conv2_bn_bias target=s2.b1.conv2.bn.bias} ->[n53] constant,
     t21 f32 [D=56 H=56 W=1 C=1] {pt2=root:p_s2_b1_conv3_conv_weight target=s2.b1.conv3.conv.weight} ->[n57] constant,
     t22 f32 [C=56] {pt2=root:p_s2_b1_conv3_bn_weight target=s2.b1.conv3.bn.weight} ->[n61] constant,
     t23 f32 [C=56] {pt2=root:p_s2_b1_conv3_bn_bias target=s2.b1.conv3.bn.bias} ->[n61] constant,
     t24 f32 [D=56 H=24 W=1 C=1] {pt2=root:p_s2_b1_downsample_conv_weight target=s2.b1.downsample.conv.weight} ->[n64] constant,
     t25 f32 [C=56] {pt2=root:p_s2_b1_downsample_bn_weight target=s2.b1.downsample.bn.weight} ->[n68] constant,
     t26 f32 [C=56] {pt2=root:p_s2_b1_downsample_bn_bias target=s2.b1.downsample.bn.bias} ->[n68] constant,
     t27 f32 [D=152 H=56 W=1 C=1] {pt2=root:p_s3_b1_conv1_conv_weight target=s3.b1.conv1.conv.weight} ->[n73] constant,
     t28 f32 [C=152] {pt2=root:p_s3_b1_conv1_bn_weight target=s3.b1.conv1.bn.weight} ->[n77] constant,
     t29 f32 [C=152] {pt2=root:p_s3_b1_conv1_bn_bias target=s3.b1.conv1.bn.bias} ->[n77] constant,
     t30 f32 [D=152 H=8 W=3 C=3] {pt2=root:p_s3_b1_conv2_conv_weight target=s3.b1.conv2.conv.weight} ->[n81] constant,
     t31 f32 [C=152] {pt2=root:p_s3_b1_conv2_bn_weight target=s3.b1.conv2.bn.weight} ->[n85] constant,
     t32 f32 [C=152] {pt2=root:p_s3_b1_conv2_bn_bias target=s3.b1.conv2.bn.bias} ->[n85] constant,
     t33 f32 [D=152 H=152 W=1 C=1] {pt2=root:p_s3_b1_conv3_conv_weight target=s3.b1.conv3.conv.weight} ->[n89] constant,
     t34 f32 [C=152] {pt2=root:p_s3_b1_conv3_bn_weight target=s3.b1.conv3.bn.weight} ->[n93] constant,
     t35 f32 [C=152] {pt2=root:p_s3_b1_conv3_bn_bias target=s3.b1.conv3.bn.bias} ->[n93] constant,
     t36 f32 [D=152 H=56 W=1 C=1] {pt2=root:p_s3_b1_downsample_conv_weight target=s3.b1.downsample.conv.weight} ->[n96] constant,
     t37 f32 [C=152] {pt2=root:p_s3_b1_downsample_bn_weight target=s3.b1.downsample.bn.weight} ->[n100] constant,
     t38 f32 [C=152] {pt2=root:p_s3_b1_downsample_bn_bias target=s3.b1.downsample.bn.bias} ->[n100] constant,
     t39 f32 [D=152 H=152 W=1 C=1] {pt2=root:p_s3_b2_conv1_conv_weight target=s3.b2.conv1.conv.weight} ->[n105] constant,
     t40 f32 [C=152] {pt2=root:p_s3_b2_conv1_bn_weight target=s3.b2.conv1.bn.weight} ->[n109] constant,
     t41 f32 [C=152] {pt2=root:p_s3_b2_conv1_bn_bias target=s3.b2.conv1.bn.bias} ->[n109] constant,
     t42 f32 [D=152 H=8 W=3 C=3] {pt2=root:p_s3_b2_conv2_conv_weight target=s3.b2.conv2.conv.weight} ->[n113] constant,
     t43 f32 [C=152] {pt2=root:p_s3_b2_conv2_bn_weight target=s3.b2.conv2.bn.weight} ->[n117] constant,
     t44 f32 [C=152] {pt2=root:p_s3_b2_conv2_bn_bias target=s3.b2.conv2.bn.bias} ->[n117] constant,
     t45 f32 [D=152 H=152 W=1 C=1] {pt2=root:p_s3_b2_conv3_conv_weight target=s3.b2.conv3.conv.weight} ->[n121] constant,
     t46 f32 [C=152] {pt2=root:p_s3_b2_conv3_bn_weight target=s3.b2.conv3.bn.weight} ->[n125] constant,
     t47 f32 [C=152] {pt2=root:p_s3_b2_conv3_bn_bias target=s3.b2.conv3.bn.bias} ->[n125] constant,
     t48 f32 [D=152 H=152 W=1 C=1] {pt2=root:p_s3_b3_conv1_conv_weight target=s3.b3.conv1.conv.weight} ->[n130] constant,
     t49 f32 [C=152] {pt2=root:p_s3_b3_conv1_bn_weight target=s3.b3.conv1.bn.weight} ->[n134] constant,
     t50 f32 [C=152] {pt2=root:p_s3_b3_conv1_bn_bias target=s3.b3.conv1.bn.bias} ->[n134] constant,
     t51 f32 [D=152 H=8 W=3 C=3] {pt2=root:p_s3_b3_conv2_conv_weight target=s3.b3.conv2.conv.weight} ->[n138] constant,
     t52 f32 [C=152] {pt2=root:p_s3_b3_conv2_bn_weight target=s3.b3.conv2.bn.weight} ->[n142] constant,
     t53 f32 [C=152] {pt2=root:p_s3_b3_conv2_bn_bias target=s3.b3.conv2.bn.bias} ->[n142] constant,
     t54 f32 [D=152 H=152 W=1 C=1] {pt2=root:p_s3_b3_conv3_conv_weight target=s3.b3.conv3.conv.weight} ->[n146] constant,
     t55 f32 [C=152] {pt2=root:p_s3_b3_conv3_bn_weight target=s3.b3.conv3.bn.weight} ->[n150] constant,
     t56 f32 [C=152] {pt2=root:p_s3_b3_conv3_bn_bias target=s3.b3.conv3.bn.bias} ->[n150] constant,
     t57 f32 [D=152 H=152 W=1 C=1] {pt2=root:p_s3_b4_conv1_conv_weight target=s3.b4.conv1.conv.weight} ->[n155] constant,
     t58 f32 [C=152] {pt2=root:p_s3_b4_conv1_bn_weight target=s3.b4.conv1.bn.weight} ->[n159] constant,
     t59 f32 [C=152] {pt2=root:p_s3_b4_conv1_bn_bias target=s3.b4.conv1.bn.bias} ->[n159] constant,
     t60 f32 [D=152 H=8 W=3 C=3] {pt2=root:p_s3_b4_conv2_conv_weight target=s3.b4.conv2.conv.weight} ->[n163] constant,
     t61 f32 [C=152] {pt2=root:p_s3_b4_conv2_bn_weight target=s3.b4.conv2.bn.weight} ->[n167] constant,
     t62 f32 [C=152] {pt2=root:p_s3_b4_conv2_bn_bias target=s3.b4.conv2.bn.bias} ->[n167] constant,
     t63 f32 [D=152 H=152 W=1 C=1] {pt2=root:p_s3_b4_conv3_conv_weight target=s3.b4.conv3.conv.weight} ->[n171] constant,
     t64 f32 [C=152] {pt2=root:p_s3_b4_conv3_bn_weight target=s3.b4.conv3.bn.weight} ->[n175] constant,
     t65 f32 [C=152] {pt2=root:p_s3_b4_conv3_bn_bias target=s3.b4.conv3.bn.bias} ->[n175] constant,
     t66 f32 [D=368 H=152 W=1 C=1] {pt2=root:p_s4_b1_conv1_conv_weight target=s4.b1.conv1.conv.weight} ->[n180] constant,
     t67 f32 [C=368] {pt2=root:p_s4_b1_conv1_bn_weight target=s4.b1.conv1.bn.weight} ->[n184] constant,
     t68 f32 [C=368] {pt2=root:p_s4_b1_conv1_bn_bias target=s4.b1.conv1.bn.bias} ->[n184] constant,
     t69 f32 [D=368 H=8 W=3 C=3] {pt2=root:p_s4_b1_conv2_conv_weight target=s4.b1.conv2.conv.weight} ->[n188] constant,
     t70 f32 [C=368] {pt2=root:p_s4_b1_conv2_bn_weight target=s4.b1.conv2.bn.weight} ->[n192] constant,
     t71 f32 [C=368] {pt2=root:p_s4_b1_conv2_bn_bias target=s4.b1.conv2.bn.bias} ->[n192] constant,
     t72 f32 [D=368 H=368 W=1 C=1] {pt2=root:p_s4_b1_conv3_conv_weight target=s4.b1.conv3.conv.weight} ->[n196] constant,
     t73 f32 [C=368] {pt2=root:p_s4_b1_conv3_bn_weight target=s4.b1.conv3.bn.weight} ->[n200] constant,
     t74 f32 [C=368] {pt2=root:p_s4_b1_conv3_bn_bias target=s4.b1.conv3.bn.bias} ->[n200] constant,
     t75 f32 [D=368 H=152 W=1 C=1] {pt2=root:p_s4_b1_downsample_conv_weight target=s4.b1.downsample.conv.weight} ->[n203] constant,
     t76 f32 [C=368] {pt2=root:p_s4_b1_downsample_bn_weight target=s4.b1.downsample.bn.weight} ->[n207] constant,
     t77 f32 [C=368] {pt2=root:p_s4_b1_downsample_bn_bias target=s4.b1.downsample.bn.bias} ->[n207] constant,
     t78 f32 [D=368 H=368 W=1 C=1] {pt2=root:p_s4_b2_conv1_conv_weight target=s4.b2.conv1.conv.weight} ->[n212] constant,
     t79 f32 [C=368] {pt2=root:p_s4_b2_conv1_bn_weight target=s4.b2.conv1.bn.weight} ->[n216] constant,
     t80 f32 [C=368] {pt2=root:p_s4_b2_conv1_bn_bias target=s4.b2.conv1.bn.bias} ->[n216] constant,
     t81 f32 [D=368 H=8 W=3 C=3] {pt2=root:p_s4_b2_conv2_conv_weight target=s4.b2.conv2.conv.weight} ->[n220] constant,
     t82 f32 [C=368] {pt2=root:p_s4_b2_conv2_bn_weight target=s4.b2.conv2.bn.weight} ->[n224] constant,
     t83 f32 [C=368] {pt2=root:p_s4_b2_conv2_bn_bias target=s4.b2.conv2.bn.bias} ->[n224] constant,
     t84 f32 [D=368 H=368 W=1 C=1] {pt2=root:p_s4_b2_conv3_conv_weight target=s4.b2.conv3.conv.weight} ->[n228] constant,
     t85 f32 [C=368] {pt2=root:p_s4_b2_conv3_bn_weight target=s4.b2.conv3.bn.weight} ->[n232] constant,
     t86 f32 [C=368] {pt2=root:p_s4_b2_conv3_bn_bias target=s4.b2.conv3.bn.bias} ->[n232] constant,
     t87 f32 [D=368 H=368 W=1 C=1] {pt2=root:p_s4_b3_conv1_conv_weight target=s4.b3.conv1.conv.weight} ->[n237] constant,
     t88 f32 [C=368] {pt2=root:p_s4_b3_conv1_bn_weight target=s4.b3.conv1.bn.weight} ->[n241] constant,
     t89 f32 [C=368] {pt2=root:p_s4_b3_conv1_bn_bias target=s4.b3.conv1.bn.bias} ->[n241] constant,
     t90 f32 [D=368 H=8 W=3 C=3] {pt2=root:p_s4_b3_conv2_conv_weight target=s4.b3.conv2.conv.weight} ->[n245] constant,
     t91 f32 [C=368] {pt2=root:p_s4_b3_conv2_bn_weight target=s4.b3.conv2.bn.weight} ->[n249] constant,
     t92 f32 [C=368] {pt2=root:p_s4_b3_conv2_bn_bias target=s4.b3.conv2.bn.bias} ->[n249] constant,
     t93 f32 [D=368 H=368 W=1 C=1] {pt2=root:p_s4_b3_conv3_conv_weight target=s4.b3.conv3.conv.weight} ->[n253] constant,
     t94 f32 [C=368] {pt2=root:p_s4_b3_conv3_bn_weight target=s4.b3.conv3.bn.weight} ->[n257] constant,
     t95 f32 [C=368] {pt2=root:p_s4_b3_conv3_bn_bias target=s4.b3.conv3.bn.bias} ->[n257] constant,
     t96 f32 [D=368 H=368 W=1 C=1] {pt2=root:p_s4_b4_conv1_conv_weight target=s4.b4.conv1.conv.weight} ->[n262] constant,
     t97 f32 [C=368] {pt2=root:p_s4_b4_conv1_bn_weight target=s4.b4.conv1.bn.weight} ->[n266] constant,
     t98 f32 [C=368] {pt2=root:p_s4_b4_conv1_bn_bias target=s4.b4.conv1.bn.bias} ->[n266] constant,
     t99 f32 [D=368 H=8 W=3 C=3] {pt2=root:p_s4_b4_conv2_conv_weight target=s4.b4.conv2.conv.weight} ->[n270] constant,
     t100 f32 [C=368] {pt2=root:p_s4_b4_conv2_bn_weight target=s4.b4.conv2.bn.weight} ->[n274] constant,
     t101 f32 [C=368] {pt2=root:p_s4_b4_conv2_bn_bias target=s4.b4.conv2.bn.bias} ->[n274] constant,
     t102 f32 [D=368 H=368 W=1 C=1] {pt2=root:p_s4_b4_conv3_conv_weight target=s4.b4.conv3.conv.weight} ->[n278] constant,
     t103 f32 [C=368] {pt2=root:p_s4_b4_conv3_bn_weight target=s4.b4.conv3.bn.weight} ->[n282] constant,
     t104 f32 [C=368] {pt2=root:p_s4_b4_conv3_bn_bias target=s4.b4.conv3.bn.bias} ->[n282] constant,
     t105 f32 [D=368 H=368 W=1 C=1] {pt2=root:p_s4_b5_conv1_conv_weight target=s4.b5.conv1.conv.weight} ->[n287] constant,
     t106 f32 [C=368] {pt2=root:p_s4_b5_conv1_bn_weight target=s4.b5.conv1.bn.weight} ->[n291] constant,
     t107 f32 [C=368] {pt2=root:p_s4_b5_conv1_bn_bias target=s4.b5.conv1.bn.bias} ->[n291] constant,
     t108 f32 [D=368 H=8 W=3 C=3] {pt2=root:p_s4_b5_conv2_conv_weight target=s4.b5.conv2.conv.weight} ->[n295] constant,
     t109 f32 [C=368] {pt2=root:p_s4_b5_conv2_bn_weight target=s4.b5.conv2.bn.weight} ->[n299] constant,
     t110 f32 [C=368] {pt2=root:p_s4_b5_conv2_bn_bias target=s4.b5.conv2.bn.bias} ->[n299] constant,
     t111 f32 [D=368 H=368 W=1 C=1] {pt2=root:p_s4_b5_conv3_conv_weight target=s4.b5.conv3.conv.weight} ->[n303] constant,
     t112 f32 [C=368] {pt2=root:p_s4_b5_conv3_bn_weight target=s4.b5.conv3.bn.weight} ->[n307] constant,
     t113 f32 [C=368] {pt2=root:p_s4_b5_conv3_bn_bias target=s4.b5.conv3.bn.bias} ->[n307] constant,
     t114 f32 [D=368 H=368 W=1 C=1] {pt2=root:p_s4_b6_conv1_conv_weight target=s4.b6.conv1.conv.weight} ->[n312] constant,
     t115 f32 [C=368] {pt2=root:p_s4_b6_conv1_bn_weight target=s4.b6.conv1.bn.weight} ->[n316] constant,
     t116 f32 [C=368] {pt2=root:p_s4_b6_conv1_bn_bias target=s4.b6.conv1.bn.bias} ->[n316] constant,
     t117 f32 [D=368 H=8 W=3 C=3] {pt2=root:p_s4_b6_conv2_conv_weight target=s4.b6.conv2.conv.weight} ->[n320] constant,
     t118 f32 [C=368] {pt2=root:p_s4_b6_conv2_bn_weight target=s4.b6.conv2.bn.weight} ->[n324] constant,
     t119 f32 [C=368] {pt2=root:p_s4_b6_conv2_bn_bias target=s4.b6.conv2.bn.bias} ->[n324] constant,
     t120 f32 [D=368 H=368 W=1 C=1] {pt2=root:p_s4_b6_conv3_conv_weight target=s4.b6.conv3.conv.weight} ->[n328] constant,
     t121 f32 [C=368] {pt2=root:p_s4_b6_conv3_bn_weight target=s4.b6.conv3.bn.weight} ->[n332] constant,
     t122 f32 [C=368] {pt2=root:p_s4_b6_conv3_bn_bias target=s4.b6.conv3.bn.bias} ->[n332] constant,
     t123 f32 [D=368 H=368 W=1 C=1] {pt2=root:p_s4_b7_conv1_conv_weight target=s4.b7.conv1.conv.weight} ->[n337] constant,
     t124 f32 [C=368] {pt2=root:p_s4_b7_conv1_bn_weight target=s4.b7.conv1.bn.weight} ->[n341] constant,
     t125 f32 [C=368] {pt2=root:p_s4_b7_conv1_bn_bias target=s4.b7.conv1.bn.bias} ->[n341] constant,
     t126 f32 [D=368 H=8 W=3 C=3] {pt2=root:p_s4_b7_conv2_conv_weight target=s4.b7.conv2.conv.weight} ->[n345] constant,
     t127 f32 [C=368] {pt2=root:p_s4_b7_conv2_bn_weight target=s4.b7.conv2.bn.weight} ->[n349] constant,
     t128 f32 [C=368] {pt2=root:p_s4_b7_conv2_bn_bias target=s4.b7.conv2.bn.bias} ->[n349] constant,
     t129 f32 [D=368 H=368 W=1 C=1] {pt2=root:p_s4_b7_conv3_conv_weight target=s4.b7.conv3.conv.weight} ->[n353] constant,
     t130 f32 [C=368] {pt2=root:p_s4_b7_conv3_bn_weight target=s4.b7.conv3.bn.weight} ->[n357] constant,
     t131 f32 [C=368] {pt2=root:p_s4_b7_conv3_bn_bias target=s4.b7.conv3.bn.bias} ->[n357] constant,
     t132 f32 [W=1000 C=368] {pt2=root:p_head_fc_weight target=head.fc.weight} ->[n366] constant,
     t133 f32 [C=1000] {pt2=root:p_head_fc_bias target=head.fc.bias} ->[n367] constant,
     t134 f32 [C=32] {pt2=root:b_stem_bn_running_mean target=stem.bn.running_mean} ->[n5] constant,
     t135 f32 [C=32] {pt2=root:b_stem_bn_running_var target=stem.bn.running_var} ->[n5] constant,
     t136 f32 [C=1] {pt2=root:b_stem_bn_num_batches_tracked target=stem.bn.num_batches_tracked} constant,
     t137 f32 [C=24] {pt2=root:b_s1_b1_conv1_bn_running_mean target=s1.b1.conv1.bn.running_mean} ->[n13] constant,
     t138 f32 [C=24] {pt2=root:b_s1_b1_conv1_bn_running_var target=s1.b1.conv1.bn.running_var} ->[n13] constant,
     t139 f32 [C=1] {pt2=root:b_s1_b1_conv1_bn_num_batches_tracked target=s1.b1.conv1.bn.num_batches_tracked} constant,
     t140 f32 [C=24] {pt2=root:b_s1_b1_conv2_bn_running_mean target=s1.b1.conv2.bn.running_mean} ->[n21] constant,
     t141 f32 [C=24] {pt2=root:b_s1_b1_conv2_bn_running_var target=s1.b1.conv2.bn.running_var} ->[n21] constant,
     t142 f32 [C=1] {pt2=root:b_s1_b1_conv2_bn_num_batches_tracked target=s1.b1.conv2.bn.num_batches_tracked} constant,
     t143 f32 [C=24] {pt2=root:b_s1_b1_conv3_bn_running_mean target=s1.b1.conv3.bn.running_mean} ->[n29] constant,
     t144 f32 [C=24] {pt2=root:b_s1_b1_conv3_bn_running_var target=s1.b1.conv3.bn.running_var} ->[n29] constant,
     t145 f32 [C=1] {pt2=root:b_s1_b1_conv3_bn_num_batches_tracked target=s1.b1.conv3.bn.num_batches_tracked} constant,
     t146 f32 [C=24] {pt2=root:b_s1_b1_downsample_bn_running_mean target=s1.b1.downsample.bn.running_mean} ->[n36] constant,
     t147 f32 [C=24] {pt2=root:b_s1_b1_downsample_bn_running_var target=s1.b1.downsample.bn.running_var} ->[n36] constant,
     t148 f32 [C=1] {pt2=root:b_s1_b1_downsample_bn_num_batches_tracked target=s1.b1.downsample.bn.num_batches_tracked} constant,
     t149 f32 [C=56] {pt2=root:b_s2_b1_conv1_bn_running_mean target=s2.b1.conv1.bn.running_mean} ->[n45] constant,
     t150 f32 [C=56] {pt2=root:b_s2_b1_conv1_bn_running_var target=s2.b1.conv1.bn.running_var} ->[n45] constant,
     t151 f32 [C=1] {pt2=root:b_s2_b1_conv1_bn_num_batches_tracked target=s2.b1.conv1.bn.num_batches_tracked} constant,
     t152 f32 [C=56] {pt2=root:b_s2_b1_conv2_bn_running_mean target=s2.b1.conv2.bn.running_mean} ->[n53] constant,
     t153 f32 [C=56] {pt2=root:b_s2_b1_conv2_bn_running_var target=s2.b1.conv2.bn.running_var} ->[n53] constant,
     t154 f32 [C=1] {pt2=root:b_s2_b1_conv2_bn_num_batches_tracked target=s2.b1.conv2.bn.num_batches_tracked} constant,
     t155 f32 [C=56] {pt2=root:b_s2_b1_conv3_bn_running_mean target=s2.b1.conv3.bn.running_mean} ->[n61] constant,
     t156 f32 [C=56] {pt2=root:b_s2_b1_conv3_bn_running_var target=s2.b1.conv3.bn.running_var} ->[n61] constant,
     t157 f32 [C=1] {pt2=root:b_s2_b1_conv3_bn_num_batches_tracked target=s2.b1.conv3.bn.num_batches_tracked} constant,
     t158 f32 [C=56] {pt2=root:b_s2_b1_downsample_bn_running_mean target=s2.b1.downsample.bn.running_mean} ->[n68] constant,
     t159 f32 [C=56] {pt2=root:b_s2_b1_downsample_bn_running_var target=s2.b1.downsample.bn.running_var} ->[n68] constant,
     t160 f32 [C=1] {pt2=root:b_s2_b1_downsample_bn_num_batches_tracked target=s2.b1.downsample.bn.num_batches_tracked} constant,
     t161 f32 [C=152] {pt2=root:b_s3_b1_conv1_bn_running_mean target=s3.b1.conv1.bn.running_mean} ->[n77] constant,
     t162 f32 [C=152] {pt2=root:b_s3_b1_conv1_bn_running_var target=s3.b1.conv1.bn.running_var} ->[n77] constant,
     t163 f32 [C=1] {pt2=root:b_s3_b1_conv1_bn_num_batches_tracked target=s3.b1.conv1.bn.num_batches_tracked} constant,
     t164 f32 [C=152] {pt2=root:b_s3_b1_conv2_bn_running_mean target=s3.b1.conv2.bn.running_mean} ->[n85] constant,
     t165 f32 [C=152] {pt2=root:b_s3_b1_conv2_bn_running_var target=s3.b1.conv2.bn.running_var} ->[n85] constant,
     t166 f32 [C=1] {pt2=root:b_s3_b1_conv2_bn_num_batches_tracked target=s3.b1.conv2.bn.num_batches_tracked} constant,
     t167 f32 [C=152] {pt2=root:b_s3_b1_conv3_bn_running_mean target=s3.b1.conv3.bn.running_mean} ->[n93] constant,
     t168 f32 [C=152] {pt2=root:b_s3_b1_conv3_bn_running_var target=s3.b1.conv3.bn.running_var} ->[n93] constant,
     t169 f32 [C=1] {pt2=root:b_s3_b1_conv3_bn_num_batches_tracked target=s3.b1.conv3.bn.num_batches_tracked} constant,
     t170 f32 [C=152] {pt2=root:b_s3_b1_downsample_bn_running_mean target=s3.b1.downsample.bn.running_mean} ->[n100] constant,
     t171 f32 [C=152] {pt2=root:b_s3_b1_downsample_bn_running_var target=s3.b1.downsample.bn.running_var} ->[n100] constant,
     t172 f32 [C=1] {pt2=root:b_s3_b1_downsample_bn_num_batches_tracked target=s3.b1.downsample.bn.num_batches_tracked} constant,
     t173 f32 [C=152] {pt2=root:b_s3_b2_conv1_bn_running_mean target=s3.b2.conv1.bn.running_mean} ->[n109] constant,
     t174 f32 [C=152] {pt2=root:b_s3_b2_conv1_bn_running_var target=s3.b2.conv1.bn.running_var} ->[n109] constant,
     t175 f32 [C=1] {pt2=root:b_s3_b2_conv1_bn_num_batches_tracked target=s3.b2.conv1.bn.num_batches_tracked} constant,
     t176 f32 [C=152] {pt2=root:b_s3_b2_conv2_bn_running_mean target=s3.b2.conv2.bn.running_mean} ->[n117] constant,
     t177 f32 [C=152] {pt2=root:b_s3_b2_conv2_bn_running_var target=s3.b2.conv2.bn.running_var} ->[n117] constant,
     t178 f32 [C=1] {pt2=root:b_s3_b2_conv2_bn_num_batches_tracked target=s3.b2.conv2.bn.num_batches_tracked} constant,
     t179 f32 [C=152] {pt2=root:b_s3_b2_conv3_bn_running_mean target=s3.b2.conv3.bn.running_mean} ->[n125] constant,
     t180 f32 [C=152] {pt2=root:b_s3_b2_conv3_bn_running_var target=s3.b2.conv3.bn.running_var} ->[n125] constant,
     t181 f32 [C=1] {pt2=root:b_s3_b2_conv3_bn_num_batches_tracked target=s3.b2.conv3.bn.num_batches_tracked} constant,
     t182 f32 [C=152] {pt2=root:b_s3_b3_conv1_bn_running_mean target=s3.b3.conv1.bn.running_mean} ->[n134] constant,
     t183 f32 [C=152] {pt2=root:b_s3_b3_conv1_bn_running_var target=s3.b3.conv1.bn.running_var} ->[n134] constant,
     t184 f32 [C=1] {pt2=root:b_s3_b3_conv1_bn_num_batches_tracked target=s3.b3.conv1.bn.num_batches_tracked} constant,
     t185 f32 [C=152] {pt2=root:b_s3_b3_conv2_bn_running_mean target=s3.b3.conv2.bn.running_mean} ->[n142] constant,
     t186 f32 [C=152] {pt2=root:b_s3_b3_conv2_bn_running_var target=s3.b3.conv2.bn.running_var} ->[n142] constant,
     t187 f32 [C=1] {pt2=root:b_s3_b3_conv2_bn_num_batches_tracked target=s3.b3.conv2.bn.num_batches_tracked} constant,
     t188 f32 [C=152] {pt2=root:b_s3_b3_conv3_bn_running_mean target=s3.b3.conv3.bn.running_mean} ->[n150] constant,
     t189 f32 [C=152] {pt2=root:b_s3_b3_conv3_bn_running_var target=s3.b3.conv3.bn.running_var} ->[n150] constant,
     t190 f32 [C=1] {pt2=root:b_s3_b3_conv3_bn_num_batches_tracked target=s3.b3.conv3.bn.num_batches_tracked} constant,
     t191 f32 [C=152] {pt2=root:b_s3_b4_conv1_bn_running_mean target=s3.b4.conv1.bn.running_mean} ->[n159] constant,
     t192 f32 [C=152] {pt2=root:b_s3_b4_conv1_bn_running_var target=s3.b4.conv1.bn.running_var} ->[n159] constant,
     t193 f32 [C=1] {pt2=root:b_s3_b4_conv1_bn_num_batches_tracked target=s3.b4.conv1.bn.num_batches_tracked} constant,
     t194 f32 [C=152] {pt2=root:b_s3_b4_conv2_bn_running_mean target=s3.b4.conv2.bn.running_mean} ->[n167] constant,
     t195 f32 [C=152] {pt2=root:b_s3_b4_conv2_bn_running_var target=s3.b4.conv2.bn.running_var} ->[n167] constant,
     t196 f32 [C=1] {pt2=root:b_s3_b4_conv2_bn_num_batches_tracked target=s3.b4.conv2.bn.num_batches_tracked} constant,
     t197 f32 [C=152] {pt2=root:b_s3_b4_conv3_bn_running_mean target=s3.b4.conv3.bn.running_mean} ->[n175] constant,
     t198 f32 [C=152] {pt2=root:b_s3_b4_conv3_bn_running_var target=s3.b4.conv3.bn.running_var} ->[n175] constant,
     t199 f32 [C=1] {pt2=root:b_s3_b4_conv3_bn_num_batches_tracked target=s3.b4.conv3.bn.num_batches_tracked} constant,
     t200 f32 [C=368] {pt2=root:b_s4_b1_conv1_bn_running_mean target=s4.b1.conv1.bn.running_mean} ->[n184] constant,
     t201 f32 [C=368] {pt2=root:b_s4_b1_conv1_bn_running_var target=s4.b1.conv1.bn.running_var} ->[n184] constant,
     t202 f32 [C=1] {pt2=root:b_s4_b1_conv1_bn_num_batches_tracked target=s4.b1.conv1.bn.num_batches_tracked} constant,
     t203 f32 [C=368] {pt2=root:b_s4_b1_conv2_bn_running_mean target=s4.b1.conv2.bn.running_mean} ->[n192] constant,
     t204 f32 [C=368] {pt2=root:b_s4_b1_conv2_bn_running_var target=s4.b1.conv2.bn.running_var} ->[n192] constant,
     t205 f32 [C=1] {pt2=root:b_s4_b1_conv2_bn_num_batches_tracked target=s4.b1.conv2.bn.num_batches_tracked} constant,
     t206 f32 [C=368] {pt2=root:b_s4_b1_conv3_bn_running_mean target=s4.b1.conv3.bn.running_mean} ->[n200] constant,
     t207 f32 [C=368] {pt2=root:b_s4_b1_conv3_bn_running_var target=s4.b1.conv3.bn.running_var} ->[n200] constant,
     t208 f32 [C=1] {pt2=root:b_s4_b1_conv3_bn_num_batches_tracked target=s4.b1.conv3.bn.num_batches_tracked} constant,
     t209 f32 [C=368] {pt2=root:b_s4_b1_downsample_bn_running_mean target=s4.b1.downsample.bn.running_mean} ->[n207] constant,
     t210 f32 [C=368] {pt2=root:b_s4_b1_downsample_bn_running_var target=s4.b1.downsample.bn.running_var} ->[n207] constant,
     t211 f32 [C=1] {pt2=root:b_s4_b1_downsample_bn_num_batches_tracked target=s4.b1.downsample.bn.num_batches_tracked} constant,
     t212 f32 [C=368] {pt2=root:b_s4_b2_conv1_bn_running_mean target=s4.b2.conv1.bn.running_mean} ->[n216] constant,
     t213 f32 [C=368] {pt2=root:b_s4_b2_conv1_bn_running_var target=s4.b2.conv1.bn.running_var} ->[n216] constant,
     t214 f32 [C=1] {pt2=root:b_s4_b2_conv1_bn_num_batches_tracked target=s4.b2.conv1.bn.num_batches_tracked} constant,
     t215 f32 [C=368] {pt2=root:b_s4_b2_conv2_bn_running_mean target=s4.b2.conv2.bn.running_mean} ->[n224] constant,
     t216 f32 [C=368] {pt2=root:b_s4_b2_conv2_bn_running_var target=s4.b2.conv2.bn.running_var} ->[n224] constant,
     t217 f32 [C=1] {pt2=root:b_s4_b2_conv2_bn_num_batches_tracked target=s4.b2.conv2.bn.num_batches_tracked} constant,
     t218 f32 [C=368] {pt2=root:b_s4_b2_conv3_bn_running_mean target=s4.b2.conv3.bn.running_mean} ->[n232] constant,
     t219 f32 [C=368] {pt2=root:b_s4_b2_conv3_bn_running_var target=s4.b2.conv3.bn.running_var} ->[n232] constant,
     t220 f32 [C=1] {pt2=root:b_s4_b2_conv3_bn_num_batches_tracked target=s4.b2.conv3.bn.num_batches_tracked} constant,
     t221 f32 [C=368] {pt2=root:b_s4_b3_conv1_bn_running_mean target=s4.b3.conv1.bn.running_mean} ->[n241] constant,
     t222 f32 [C=368] {pt2=root:b_s4_b3_conv1_bn_running_var target=s4.b3.conv1.bn.running_var} ->[n241] constant,
     t223 f32 [C=1] {pt2=root:b_s4_b3_conv1_bn_num_batches_tracked target=s4.b3.conv1.bn.num_batches_tracked} constant,
     t224 f32 [C=368] {pt2=root:b_s4_b3_conv2_bn_running_mean target=s4.b3.conv2.bn.running_mean} ->[n249] constant,
     t225 f32 [C=368] {pt2=root:b_s4_b3_conv2_bn_running_var target=s4.b3.conv2.bn.running_var} ->[n249] constant,
     t226 f32 [C=1] {pt2=root:b_s4_b3_conv2_bn_num_batches_tracked target=s4.b3.conv2.bn.num_batches_tracked} constant,
     t227 f32 [C=368] {pt2=root:b_s4_b3_conv3_bn_running_mean target=s4.b3.conv3.bn.running_mean} ->[n257] constant,
     t228 f32 [C=368] {pt2=root:b_s4_b3_conv3_bn_running_var target=s4.b3.conv3.bn.running_var} ->[n257] constant,
     t229 f32 [C=1] {pt2=root:b_s4_b3_conv3_bn_num_batches_tracked target=s4.b3.conv3.bn.num_batches_tracked} constant,
     t230 f32 [C=368] {pt2=root:b_s4_b4_conv1_bn_running_mean target=s4.b4.conv1.bn.running_mean} ->[n266] constant,
     t231 f32 [C=368] {pt2=root:b_s4_b4_conv1_bn_running_var target=s4.b4.conv1.bn.running_var} ->[n266] constant,
     t232 f32 [C=1] {pt2=root:b_s4_b4_conv1_bn_num_batches_tracked target=s4.b4.conv1.bn.num_batches_tracked} constant,
     t233 f32 [C=368] {pt2=root:b_s4_b4_conv2_bn_running_mean target=s4.b4.conv2.bn.running_mean} ->[n274] constant,
     t234 f32 [C=368] {pt2=root:b_s4_b4_conv2_bn_running_var target=s4.b4.conv2.bn.running_var} ->[n274] constant,
     t235 f32 [C=1] {pt2=root:b_s4_b4_conv2_bn_num_batches_tracked target=s4.b4.conv2.bn.num_batches_tracked} constant,
     t236 f32 [C=368] {pt2=root:b_s4_b4_conv3_bn_running_mean target=s4.b4.conv3.bn.running_mean} ->[n282] constant,
     t237 f32 [C=368] {pt2=root:b_s4_b4_conv3_bn_running_var target=s4.b4.conv3.bn.running_var} ->[n282] constant,
     t238 f32 [C=1] {pt2=root:b_s4_b4_conv3_bn_num_batches_tracked target=s4.b4.conv3.bn.num_batches_tracked} constant,
     t239 f32 [C=368] {pt2=root:b_s4_b5_conv1_bn_running_mean target=s4.b5.conv1.bn.running_mean} ->[n291] constant,
     t240 f32 [C=368] {pt2=root:b_s4_b5_conv1_bn_running_var target=s4.b5.conv1.bn.running_var} ->[n291] constant,
     t241 f32 [C=1] {pt2=root:b_s4_b5_conv1_bn_num_batches_tracked target=s4.b5.conv1.bn.num_batches_tracked} constant,
     t242 f32 [C=368] {pt2=root:b_s4_b5_conv2_bn_running_mean target=s4.b5.conv2.bn.running_mean} ->[n299] constant,
     t243 f32 [C=368] {pt2=root:b_s4_b5_conv2_bn_running_var target=s4.b5.conv2.bn.running_var} ->[n299] constant,
     t244 f32 [C=1] {pt2=root:b_s4_b5_conv2_bn_num_batches_tracked target=s4.b5.conv2.bn.num_batches_tracked} constant,
     t245 f32 [C=368] {pt2=root:b_s4_b5_conv3_bn_running_mean target=s4.b5.conv3.bn.running_mean} ->[n307] constant,
     t246 f32 [C=368] {pt2=root:b_s4_b5_conv3_bn_running_var target=s4.b5.conv3.bn.running_var} ->[n307] constant,
     t247 f32 [C=1] {pt2=root:b_s4_b5_conv3_bn_num_batches_tracked target=s4.b5.conv3.bn.num_batches_tracked} constant,
     t248 f32 [C=368] {pt2=root:b_s4_b6_conv1_bn_running_mean target=s4.b6.conv1.bn.running_mean} ->[n316] constant,
     t249 f32 [C=368] {pt2=root:b_s4_b6_conv1_bn_running_var target=s4.b6.conv1.bn.running_var} ->[n316] constant,
     t250 f32 [C=1] {pt2=root:b_s4_b6_conv1_bn_num_batches_tracked target=s4.b6.conv1.bn.num_batches_tracked} constant,
     t251 f32 [C=368] {pt2=root:b_s4_b6_conv2_bn_running_mean target=s4.b6.conv2.bn.running_mean} ->[n324] constant,
     t252 f32 [C=368] {pt2=root:b_s4_b6_conv2_bn_running_var target=s4.b6.conv2.bn.running_var} ->[n324] constant,
     t253 f32 [C=1] {pt2=root:b_s4_b6_conv2_bn_num_batches_tracked target=s4.b6.conv2.bn.num_batches_tracked} constant,
     t254 f32 [C=368] {pt2=root:b_s4_b6_conv3_bn_running_mean target=s4.b6.conv3.bn.running_mean} ->[n332] constant,
     t255 f32 [C=368] {pt2=root:b_s4_b6_conv3_bn_running_var target=s4.b6.conv3.bn.running_var} ->[n332] constant,
     t256 f32 [C=1] {pt2=root:b_s4_b6_conv3_bn_num_batches_tracked target=s4.b6.conv3.bn.num_batches_tracked} constant,
     t257 f32 [C=368] {pt2=root:b_s4_b7_conv1_bn_running_mean target=s4.b7.conv1.bn.running_mean} ->[n341] constant,
     t258 f32 [C=368] {pt2=root:b_s4_b7_conv1_bn_running_var target=s4.b7.conv1.bn.running_var} ->[n341] constant,
     t259 f32 [C=1] {pt2=root:b_s4_b7_conv1_bn_num_batches_tracked target=s4.b7.conv1.bn.num_batches_tracked} constant,
     t260 f32 [C=368] {pt2=root:b_s4_b7_conv2_bn_running_mean target=s4.b7.conv2.bn.running_mean} ->[n349] constant,
     t261 f32 [C=368] {pt2=root:b_s4_b7_conv2_bn_running_var target=s4.b7.conv2.bn.running_var} ->[n349] constant,
     t262 f32 [C=1] {pt2=root:b_s4_b7_conv2_bn_num_batches_tracked target=s4.b7.conv2.bn.num_batches_tracked} constant,
     t263 f32 [C=368] {pt2=root:b_s4_b7_conv3_bn_running_mean target=s4.b7.conv3.bn.running_mean} ->[n357] constant,
     t264 f32 [C=368] {pt2=root:b_s4_b7_conv3_bn_running_var target=s4.b7.conv3.bn.running_var} ->[n357] constant,
     t265 f32 [C=1] {pt2=root:b_s4_b7_conv3_bn_num_batches_tracked target=s4.b7.conv3.bn.num_batches_tracked} constant,
     t266 f32 [H=3 W=224 C=224] {pt2=root:x} ->[n0]]
  nodes:
    group g1 torch.ops.aten.conv2d.default:
      n0 {derived}: [t267 f32 [H=224 W=224 C=3] {derived} ->[n2]] =
        permute x=t266 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
      n1 {derived}: [t268 f32 [N=32 T=1 D=1 H=3 W=3 C=3] {derived} ->[n2]] =
        permute
          x=t0 {pt2=root:p_stem_conv_weight target=stem.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2 {derived}: [t269 f32 [H=112 W=112 C=32] {derived} ->[n3]] =
        conv2d
          x=t267 {derived} <-n0
          weight=t268 {derived} <-n1
          bias=none
          params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 in_channels=3;
                 groups=1}
      n3 {pt2=root[0] torch.ops.aten.conv2d.default (conv2d)}: [t270 f32 [H=32
                                                                      W=112
                                                                      C=112] {pt2=root:conv2d} ->[n4]] =
        permute x=t269 {derived} <-n2 perm=[H<-C, W<-H, C<-W]
    group g2 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n4 {derived}: [t271 f32 [H=112 W=112 C=32] {derived} ->[n5]] =
        permute x=t270 {pt2=root:conv2d} <-n3 perm=[H<-W, W<-C, C<-H]
      n5 {derived}: [t272 f32 [H=112 W=112 C=32] {derived} ->[n6]] =
        batch_norm
          x=t271 {derived} <-n4
          weight=t1 {pt2=root:p_stem_bn_weight target=stem.bn.weight}
          bias=t2 {pt2=root:p_stem_bn_bias target=stem.bn.bias}
          running_mean=t134 {pt2=root:b_stem_bn_running_mean target=stem.bn.running_mean}
          running_var=t135 {pt2=root:b_stem_bn_running_var target=stem.bn.running_var}
          params={channel=C; eps=1e-05}
      n6 {pt2=root[1] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training)}: [t273 f32 [H=32
                                                                      W=112
                                                                      C=112] {pt2=root:getitem} ->[n7]] =
        permute x=t272 {derived} <-n5 perm=[H<-C, W<-H, C<-W]
    n7 {pt2=root[2] torch.ops.aten.relu.default (relu)}: [t274 f32 [H=32 W=112
                                                                    C=112] {pt2=root:relu} ->[n8,
                                                                      n31]] =
      relu x=t273 {pt2=root:getitem} <-n6
    group g3 torch.ops.aten.conv2d.default:
      n8 {derived}: [t275 f32 [H=112 W=112 C=32] {derived} ->[n10]] =
        permute x=t274 {pt2=root:relu} <-n7 perm=[H<-W, W<-C, C<-H]
      n9 {derived}: [t276 f32 [N=24 T=1 D=1 H=1 W=1 C=32] {derived} ->[n10]] =
        permute
          x=t3 {pt2=root:p_s1_b1_conv1_conv_weight target=s1.b1.conv1.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n10 {derived}: [t277 f32 [H=112 W=112 C=24] {derived} ->[n11]] =
        conv2d
          x=t275 {derived} <-n8
          weight=t276 {derived} <-n9
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=32;
                 groups=1}
      n11 {pt2=root[3] torch.ops.aten.conv2d.default (conv2d_1)}: [t278 f32 [H=24
                                                                      W=112
                                                                      C=112] {pt2=root:conv2d_1} ->[n12]] =
        permute x=t277 {derived} <-n10 perm=[H<-C, W<-H, C<-W]
    group g4 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n12 {derived}: [t279 f32 [H=112 W=112 C=24] {derived} ->[n13]] =
        permute x=t278 {pt2=root:conv2d_1} <-n11 perm=[H<-W, W<-C, C<-H]
      n13 {derived}: [t280 f32 [H=112 W=112 C=24] {derived} ->[n14]] =
        batch_norm
          x=t279 {derived} <-n12
          weight=t4 {pt2=root:p_s1_b1_conv1_bn_weight target=s1.b1.conv1.bn.weight}
          bias=t5 {pt2=root:p_s1_b1_conv1_bn_bias target=s1.b1.conv1.bn.bias}
          running_mean=t137 {pt2=root:b_s1_b1_conv1_bn_running_mean target=s1.b1.conv1.bn.running_mean}
          running_var=t138 {pt2=root:b_s1_b1_conv1_bn_running_var target=s1.b1.conv1.bn.running_var}
          params={channel=C; eps=1e-05}
      n14 {pt2=root[4] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_1)}: [t281 f32 [H=24
                                                                      W=112
                                                                      C=112] {pt2=root:getitem_3} ->[n15]] =
        permute x=t280 {derived} <-n13 perm=[H<-C, W<-H, C<-W]
    n15 {pt2=root[5] torch.ops.aten.relu.default (relu_1)}: [t282 f32 [H=24
                                                                      W=112
                                                                      C=112] {pt2=root:relu_1} ->[n16]] =
      relu x=t281 {pt2=root:getitem_3} <-n14
    group g5 torch.ops.aten.conv2d.default:
      n16 {derived}: [t283 f32 [H=112 W=112 C=24] {derived} ->[n18]] =
        permute x=t282 {pt2=root:relu_1} <-n15 perm=[H<-W, W<-C, C<-H]
      n17 {derived}: [t284 f32 [N=24 T=1 D=1 H=3 W=3 C=8] {derived} ->[n18]] =
        permute
          x=t6 {pt2=root:p_s1_b1_conv2_conv_weight target=s1.b1.conv2.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n18 {derived}: [t285 f32 [H=56 W=56 C=24] {derived} ->[n19]] =
        conv2d
          x=t283 {derived} <-n16
          weight=t284 {derived} <-n17
          bias=none
          params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 in_channels=24;
                 groups=3}
      n19 {pt2=root[6] torch.ops.aten.conv2d.default (conv2d_2)}: [t286 f32 [H=24
                                                                      W=56
                                                                      C=56] {pt2=root:conv2d_2} ->[n20]] =
        permute x=t285 {derived} <-n18 perm=[H<-C, W<-H, C<-W]
    group g6 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n20 {derived}: [t287 f32 [H=56 W=56 C=24] {derived} ->[n21]] =
        permute x=t286 {pt2=root:conv2d_2} <-n19 perm=[H<-W, W<-C, C<-H]
      n21 {derived}: [t288 f32 [H=56 W=56 C=24] {derived} ->[n22]] =
        batch_norm
          x=t287 {derived} <-n20
          weight=t7 {pt2=root:p_s1_b1_conv2_bn_weight target=s1.b1.conv2.bn.weight}
          bias=t8 {pt2=root:p_s1_b1_conv2_bn_bias target=s1.b1.conv2.bn.bias}
          running_mean=t140 {pt2=root:b_s1_b1_conv2_bn_running_mean target=s1.b1.conv2.bn.running_mean}
          running_var=t141 {pt2=root:b_s1_b1_conv2_bn_running_var target=s1.b1.conv2.bn.running_var}
          params={channel=C; eps=1e-05}
      n22 {pt2=root[7] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_2)}: [t289 f32 [H=24
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_6} ->[n23]] =
        permute x=t288 {derived} <-n21 perm=[H<-C, W<-H, C<-W]
    n23 {pt2=root[8] torch.ops.aten.relu.default (relu_2)}: [t290 f32 [H=24
                                                                      W=56
                                                                      C=56] {pt2=root:relu_2} ->[n24]] =
      relu x=t289 {pt2=root:getitem_6} <-n22
    group g7 torch.ops.aten.conv2d.default:
      n24 {derived}: [t291 f32 [H=56 W=56 C=24] {derived} ->[n26]] =
        permute x=t290 {pt2=root:relu_2} <-n23 perm=[H<-W, W<-C, C<-H]
      n25 {derived}: [t292 f32 [N=24 T=1 D=1 H=1 W=1 C=24] {derived} ->[n26]] =
        permute
          x=t9 {pt2=root:p_s1_b1_conv3_conv_weight target=s1.b1.conv3.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n26 {derived}: [t293 f32 [H=56 W=56 C=24] {derived} ->[n27]] =
        conv2d
          x=t291 {derived} <-n24
          weight=t292 {derived} <-n25
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=24;
                 groups=1}
      n27 {pt2=root[9] torch.ops.aten.conv2d.default (conv2d_3)}: [t294 f32 [H=24
                                                                      W=56
                                                                      C=56] {pt2=root:conv2d_3} ->[n28]] =
        permute x=t293 {derived} <-n26 perm=[H<-C, W<-H, C<-W]
    group g8 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n28 {derived}: [t295 f32 [H=56 W=56 C=24] {derived} ->[n29]] =
        permute x=t294 {pt2=root:conv2d_3} <-n27 perm=[H<-W, W<-C, C<-H]
      n29 {derived}: [t296 f32 [H=56 W=56 C=24] {derived} ->[n30]] =
        batch_norm
          x=t295 {derived} <-n28
          weight=t10 {pt2=root:p_s1_b1_conv3_bn_weight target=s1.b1.conv3.bn.weight}
          bias=t11 {pt2=root:p_s1_b1_conv3_bn_bias target=s1.b1.conv3.bn.bias}
          running_mean=t143 {pt2=root:b_s1_b1_conv3_bn_running_mean target=s1.b1.conv3.bn.running_mean}
          running_var=t144 {pt2=root:b_s1_b1_conv3_bn_running_var target=s1.b1.conv3.bn.running_var}
          params={channel=C; eps=1e-05}
      n30 {pt2=root[10] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_3)}: [t297 f32 [H=24
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_9} ->[n38]] =
        permute x=t296 {derived} <-n29 perm=[H<-C, W<-H, C<-W]
    group g9 torch.ops.aten.conv2d.default:
      n31 {derived}: [t298 f32 [H=112 W=112 C=32] {derived} ->[n33]] =
        permute x=t274 {pt2=root:relu} <-n7 perm=[H<-W, W<-C, C<-H]
      n32 {derived}: [t299 f32 [N=24 T=1 D=1 H=1 W=1 C=32] {derived} ->[n33]] =
        permute
          x=t12 {pt2=root:p_s1_b1_downsample_conv_weight target=s1.b1.downsample.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n33 {derived}: [t300 f32 [H=56 W=56 C=24] {derived} ->[n34]] =
        conv2d
          x=t298 {derived} <-n31
          weight=t299 {derived} <-n32
          bias=none
          params={h={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
                 in_channels=32;
                 groups=1}
      n34 {pt2=root[11] torch.ops.aten.conv2d.default (conv2d_4)}: [t301 f32 [H=24
                                                                      W=56
                                                                      C=56] {pt2=root:conv2d_4} ->[n35]] =
        permute x=t300 {derived} <-n33 perm=[H<-C, W<-H, C<-W]
    group g10 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n35 {derived}: [t302 f32 [H=56 W=56 C=24] {derived} ->[n36]] =
        permute x=t301 {pt2=root:conv2d_4} <-n34 perm=[H<-W, W<-C, C<-H]
      n36 {derived}: [t303 f32 [H=56 W=56 C=24] {derived} ->[n37]] =
        batch_norm
          x=t302 {derived} <-n35
          weight=t13 {pt2=root:p_s1_b1_downsample_bn_weight target=s1.b1.downsample.bn.weight}
          bias=t14 {pt2=root:p_s1_b1_downsample_bn_bias target=s1.b1.downsample.bn.bias}
          running_mean=t146 {pt2=root:b_s1_b1_downsample_bn_running_mean target=s1.b1.downsample.bn.running_mean}
          running_var=t147 {pt2=root:b_s1_b1_downsample_bn_running_var target=s1.b1.downsample.bn.running_var}
          params={channel=C; eps=1e-05}
      n37 {pt2=root[12] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_4)}: [t304 f32 [H=24
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_12} ->[n38]] =
        permute x=t303 {derived} <-n36 perm=[H<-C, W<-H, C<-W]
    n38 {pt2=root[13] torch.ops.aten.add.Tensor (add)}: [t305 f32 [H=24 W=56
                                                                   C=56] {pt2=root:add} ->[n39]] =
      add a=t297 {pt2=root:getitem_9} <-n30 b=t304 {pt2=root:getitem_12} <-n37
    n39 {pt2=root[14] torch.ops.aten.relu.default (relu_3)}: [t306 f32 [H=24
                                                                      W=56
                                                                      C=56] {pt2=root:relu_3} ->[n40,
                                                                      n63]] =
      relu x=t305 {pt2=root:add} <-n38
    group g11 torch.ops.aten.conv2d.default:
      n40 {derived}: [t307 f32 [H=56 W=56 C=24] {derived} ->[n42]] =
        permute x=t306 {pt2=root:relu_3} <-n39 perm=[H<-W, W<-C, C<-H]
      n41 {derived}: [t308 f32 [N=56 T=1 D=1 H=1 W=1 C=24] {derived} ->[n42]] =
        permute
          x=t15 {pt2=root:p_s2_b1_conv1_conv_weight target=s2.b1.conv1.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n42 {derived}: [t309 f32 [H=56 W=56 C=56] {derived} ->[n43]] =
        conv2d
          x=t307 {derived} <-n40
          weight=t308 {derived} <-n41
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=24;
                 groups=1}
      n43 {pt2=root[15] torch.ops.aten.conv2d.default (conv2d_5)}: [t310 f32 [H=56
                                                                      W=56
                                                                      C=56] {pt2=root:conv2d_5} ->[n44]] =
        permute x=t309 {derived} <-n42 perm=[H<-C, W<-H, C<-W]
    group g12 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n44 {derived}: [t311 f32 [H=56 W=56 C=56] {derived} ->[n45]] =
        permute x=t310 {pt2=root:conv2d_5} <-n43 perm=[H<-W, W<-C, C<-H]
      n45 {derived}: [t312 f32 [H=56 W=56 C=56] {derived} ->[n46]] =
        batch_norm
          x=t311 {derived} <-n44
          weight=t16 {pt2=root:p_s2_b1_conv1_bn_weight target=s2.b1.conv1.bn.weight}
          bias=t17 {pt2=root:p_s2_b1_conv1_bn_bias target=s2.b1.conv1.bn.bias}
          running_mean=t149 {pt2=root:b_s2_b1_conv1_bn_running_mean target=s2.b1.conv1.bn.running_mean}
          running_var=t150 {pt2=root:b_s2_b1_conv1_bn_running_var target=s2.b1.conv1.bn.running_var}
          params={channel=C; eps=1e-05}
      n46 {pt2=root[16] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_5)}: [t313 f32 [H=56
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_15} ->[n47]] =
        permute x=t312 {derived} <-n45 perm=[H<-C, W<-H, C<-W]
    n47 {pt2=root[17] torch.ops.aten.relu.default (relu_4)}: [t314 f32 [H=56
                                                                      W=56
                                                                      C=56] {pt2=root:relu_4} ->[n48]] =
      relu x=t313 {pt2=root:getitem_15} <-n46
    group g13 torch.ops.aten.conv2d.default:
      n48 {derived}: [t315 f32 [H=56 W=56 C=56] {derived} ->[n50]] =
        permute x=t314 {pt2=root:relu_4} <-n47 perm=[H<-W, W<-C, C<-H]
      n49 {derived}: [t316 f32 [N=56 T=1 D=1 H=3 W=3 C=8] {derived} ->[n50]] =
        permute
          x=t18 {pt2=root:p_s2_b1_conv2_conv_weight target=s2.b1.conv2.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n50 {derived}: [t317 f32 [H=28 W=28 C=56] {derived} ->[n51]] =
        conv2d
          x=t315 {derived} <-n48
          weight=t316 {derived} <-n49
          bias=none
          params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 in_channels=56;
                 groups=7}
      n51 {pt2=root[18] torch.ops.aten.conv2d.default (conv2d_6)}: [t318 f32 [H=56
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_6} ->[n52]] =
        permute x=t317 {derived} <-n50 perm=[H<-C, W<-H, C<-W]
    group g14 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n52 {derived}: [t319 f32 [H=28 W=28 C=56] {derived} ->[n53]] =
        permute x=t318 {pt2=root:conv2d_6} <-n51 perm=[H<-W, W<-C, C<-H]
      n53 {derived}: [t320 f32 [H=28 W=28 C=56] {derived} ->[n54]] =
        batch_norm
          x=t319 {derived} <-n52
          weight=t19 {pt2=root:p_s2_b1_conv2_bn_weight target=s2.b1.conv2.bn.weight}
          bias=t20 {pt2=root:p_s2_b1_conv2_bn_bias target=s2.b1.conv2.bn.bias}
          running_mean=t152 {pt2=root:b_s2_b1_conv2_bn_running_mean target=s2.b1.conv2.bn.running_mean}
          running_var=t153 {pt2=root:b_s2_b1_conv2_bn_running_var target=s2.b1.conv2.bn.running_var}
          params={channel=C; eps=1e-05}
      n54 {pt2=root[19] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_6)}: [t321 f32 [H=56
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_18} ->[n55]] =
        permute x=t320 {derived} <-n53 perm=[H<-C, W<-H, C<-W]
    n55 {pt2=root[20] torch.ops.aten.relu.default (relu_5)}: [t322 f32 [H=56
                                                                      W=28
                                                                      C=28] {pt2=root:relu_5} ->[n56]] =
      relu x=t321 {pt2=root:getitem_18} <-n54
    group g15 torch.ops.aten.conv2d.default:
      n56 {derived}: [t323 f32 [H=28 W=28 C=56] {derived} ->[n58]] =
        permute x=t322 {pt2=root:relu_5} <-n55 perm=[H<-W, W<-C, C<-H]
      n57 {derived}: [t324 f32 [N=56 T=1 D=1 H=1 W=1 C=56] {derived} ->[n58]] =
        permute
          x=t21 {pt2=root:p_s2_b1_conv3_conv_weight target=s2.b1.conv3.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n58 {derived}: [t325 f32 [H=28 W=28 C=56] {derived} ->[n59]] =
        conv2d
          x=t323 {derived} <-n56
          weight=t324 {derived} <-n57
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=56;
                 groups=1}
      n59 {pt2=root[21] torch.ops.aten.conv2d.default (conv2d_7)}: [t326 f32 [H=56
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_7} ->[n60]] =
        permute x=t325 {derived} <-n58 perm=[H<-C, W<-H, C<-W]
    group g16 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n60 {derived}: [t327 f32 [H=28 W=28 C=56] {derived} ->[n61]] =
        permute x=t326 {pt2=root:conv2d_7} <-n59 perm=[H<-W, W<-C, C<-H]
      n61 {derived}: [t328 f32 [H=28 W=28 C=56] {derived} ->[n62]] =
        batch_norm
          x=t327 {derived} <-n60
          weight=t22 {pt2=root:p_s2_b1_conv3_bn_weight target=s2.b1.conv3.bn.weight}
          bias=t23 {pt2=root:p_s2_b1_conv3_bn_bias target=s2.b1.conv3.bn.bias}
          running_mean=t155 {pt2=root:b_s2_b1_conv3_bn_running_mean target=s2.b1.conv3.bn.running_mean}
          running_var=t156 {pt2=root:b_s2_b1_conv3_bn_running_var target=s2.b1.conv3.bn.running_var}
          params={channel=C; eps=1e-05}
      n62 {pt2=root[22] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_7)}: [t329 f32 [H=56
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_21} ->[n70]] =
        permute x=t328 {derived} <-n61 perm=[H<-C, W<-H, C<-W]
    group g17 torch.ops.aten.conv2d.default:
      n63 {derived}: [t330 f32 [H=56 W=56 C=24] {derived} ->[n65]] =
        permute x=t306 {pt2=root:relu_3} <-n39 perm=[H<-W, W<-C, C<-H]
      n64 {derived}: [t331 f32 [N=56 T=1 D=1 H=1 W=1 C=24] {derived} ->[n65]] =
        permute
          x=t24 {pt2=root:p_s2_b1_downsample_conv_weight target=s2.b1.downsample.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n65 {derived}: [t332 f32 [H=28 W=28 C=56] {derived} ->[n66]] =
        conv2d
          x=t330 {derived} <-n63
          weight=t331 {derived} <-n64
          bias=none
          params={h={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
                 in_channels=24;
                 groups=1}
      n66 {pt2=root[23] torch.ops.aten.conv2d.default (conv2d_8)}: [t333 f32 [H=56
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_8} ->[n67]] =
        permute x=t332 {derived} <-n65 perm=[H<-C, W<-H, C<-W]
    group g18 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n67 {derived}: [t334 f32 [H=28 W=28 C=56] {derived} ->[n68]] =
        permute x=t333 {pt2=root:conv2d_8} <-n66 perm=[H<-W, W<-C, C<-H]
      n68 {derived}: [t335 f32 [H=28 W=28 C=56] {derived} ->[n69]] =
        batch_norm
          x=t334 {derived} <-n67
          weight=t25 {pt2=root:p_s2_b1_downsample_bn_weight target=s2.b1.downsample.bn.weight}
          bias=t26 {pt2=root:p_s2_b1_downsample_bn_bias target=s2.b1.downsample.bn.bias}
          running_mean=t158 {pt2=root:b_s2_b1_downsample_bn_running_mean target=s2.b1.downsample.bn.running_mean}
          running_var=t159 {pt2=root:b_s2_b1_downsample_bn_running_var target=s2.b1.downsample.bn.running_var}
          params={channel=C; eps=1e-05}
      n69 {pt2=root[24] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_8)}: [t336 f32 [H=56
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_24} ->[n70]] =
        permute x=t335 {derived} <-n68 perm=[H<-C, W<-H, C<-W]
    n70 {pt2=root[25] torch.ops.aten.add.Tensor (add_1)}: [t337 f32 [H=56 W=28
                                                                     C=28] {pt2=root:add_1} ->[n71]] =
      add a=t329 {pt2=root:getitem_21} <-n62 b=t336 {pt2=root:getitem_24} <-n69
    n71 {pt2=root[26] torch.ops.aten.relu.default (relu_6)}: [t338 f32 [H=56
                                                                      W=28
                                                                      C=28] {pt2=root:relu_6} ->[n72,
                                                                      n95]] =
      relu x=t337 {pt2=root:add_1} <-n70
    group g19 torch.ops.aten.conv2d.default:
      n72 {derived}: [t339 f32 [H=28 W=28 C=56] {derived} ->[n74]] =
        permute x=t338 {pt2=root:relu_6} <-n71 perm=[H<-W, W<-C, C<-H]
      n73 {derived}: [t340 f32 [N=152 T=1 D=1 H=1 W=1 C=56] {derived} ->[n74]] =
        permute
          x=t27 {pt2=root:p_s3_b1_conv1_conv_weight target=s3.b1.conv1.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n74 {derived}: [t341 f32 [H=28 W=28 C=152] {derived} ->[n75]] =
        conv2d
          x=t339 {derived} <-n72
          weight=t340 {derived} <-n73
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=56;
                 groups=1}
      n75 {pt2=root[27] torch.ops.aten.conv2d.default (conv2d_9)}: [t342 f32 [H=152
                                                                      W=28
                                                                      C=28] {pt2=root:conv2d_9} ->[n76]] =
        permute x=t341 {derived} <-n74 perm=[H<-C, W<-H, C<-W]
    group g20 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n76 {derived}: [t343 f32 [H=28 W=28 C=152] {derived} ->[n77]] =
        permute x=t342 {pt2=root:conv2d_9} <-n75 perm=[H<-W, W<-C, C<-H]
      n77 {derived}: [t344 f32 [H=28 W=28 C=152] {derived} ->[n78]] =
        batch_norm
          x=t343 {derived} <-n76
          weight=t28 {pt2=root:p_s3_b1_conv1_bn_weight target=s3.b1.conv1.bn.weight}
          bias=t29 {pt2=root:p_s3_b1_conv1_bn_bias target=s3.b1.conv1.bn.bias}
          running_mean=t161 {pt2=root:b_s3_b1_conv1_bn_running_mean target=s3.b1.conv1.bn.running_mean}
          running_var=t162 {pt2=root:b_s3_b1_conv1_bn_running_var target=s3.b1.conv1.bn.running_var}
          params={channel=C; eps=1e-05}
      n78 {pt2=root[28] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_9)}: [t345 f32 [H=152
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_27} ->[n79]] =
        permute x=t344 {derived} <-n77 perm=[H<-C, W<-H, C<-W]
    n79 {pt2=root[29] torch.ops.aten.relu.default (relu_7)}: [t346 f32 [H=152
                                                                      W=28
                                                                      C=28] {pt2=root:relu_7} ->[n80]] =
      relu x=t345 {pt2=root:getitem_27} <-n78
    group g21 torch.ops.aten.conv2d.default:
      n80 {derived}: [t347 f32 [H=28 W=28 C=152] {derived} ->[n82]] =
        permute x=t346 {pt2=root:relu_7} <-n79 perm=[H<-W, W<-C, C<-H]
      n81 {derived}: [t348 f32 [N=152 T=1 D=1 H=3 W=3 C=8] {derived} ->[n82]] =
        permute
          x=t30 {pt2=root:p_s3_b1_conv2_conv_weight target=s3.b1.conv2.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n82 {derived}: [t349 f32 [H=14 W=14 C=152] {derived} ->[n83]] =
        conv2d
          x=t347 {derived} <-n80
          weight=t348 {derived} <-n81
          bias=none
          params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 in_channels=152;
                 groups=19}
      n83 {pt2=root[30] torch.ops.aten.conv2d.default (conv2d_10)}: [t350 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_10} ->[n84]] =
        permute x=t349 {derived} <-n82 perm=[H<-C, W<-H, C<-W]
    group g22 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n84 {derived}: [t351 f32 [H=14 W=14 C=152] {derived} ->[n85]] =
        permute x=t350 {pt2=root:conv2d_10} <-n83 perm=[H<-W, W<-C, C<-H]
      n85 {derived}: [t352 f32 [H=14 W=14 C=152] {derived} ->[n86]] =
        batch_norm
          x=t351 {derived} <-n84
          weight=t31 {pt2=root:p_s3_b1_conv2_bn_weight target=s3.b1.conv2.bn.weight}
          bias=t32 {pt2=root:p_s3_b1_conv2_bn_bias target=s3.b1.conv2.bn.bias}
          running_mean=t164 {pt2=root:b_s3_b1_conv2_bn_running_mean target=s3.b1.conv2.bn.running_mean}
          running_var=t165 {pt2=root:b_s3_b1_conv2_bn_running_var target=s3.b1.conv2.bn.running_var}
          params={channel=C; eps=1e-05}
      n86 {pt2=root[31] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_10)}: [t353 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_30} ->[n87]] =
        permute x=t352 {derived} <-n85 perm=[H<-C, W<-H, C<-W]
    n87 {pt2=root[32] torch.ops.aten.relu.default (relu_8)}: [t354 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:relu_8} ->[n88]] =
      relu x=t353 {pt2=root:getitem_30} <-n86
    group g23 torch.ops.aten.conv2d.default:
      n88 {derived}: [t355 f32 [H=14 W=14 C=152] {derived} ->[n90]] =
        permute x=t354 {pt2=root:relu_8} <-n87 perm=[H<-W, W<-C, C<-H]
      n89 {derived}: [t356 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {derived} ->[n90]] =
        permute
          x=t33 {pt2=root:p_s3_b1_conv3_conv_weight target=s3.b1.conv3.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n90 {derived}: [t357 f32 [H=14 W=14 C=152] {derived} ->[n91]] =
        conv2d
          x=t355 {derived} <-n88
          weight=t356 {derived} <-n89
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=152;
                 groups=1}
      n91 {pt2=root[33] torch.ops.aten.conv2d.default (conv2d_11)}: [t358 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_11} ->[n92]] =
        permute x=t357 {derived} <-n90 perm=[H<-C, W<-H, C<-W]
    group g24 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n92 {derived}: [t359 f32 [H=14 W=14 C=152] {derived} ->[n93]] =
        permute x=t358 {pt2=root:conv2d_11} <-n91 perm=[H<-W, W<-C, C<-H]
      n93 {derived}: [t360 f32 [H=14 W=14 C=152] {derived} ->[n94]] =
        batch_norm
          x=t359 {derived} <-n92
          weight=t34 {pt2=root:p_s3_b1_conv3_bn_weight target=s3.b1.conv3.bn.weight}
          bias=t35 {pt2=root:p_s3_b1_conv3_bn_bias target=s3.b1.conv3.bn.bias}
          running_mean=t167 {pt2=root:b_s3_b1_conv3_bn_running_mean target=s3.b1.conv3.bn.running_mean}
          running_var=t168 {pt2=root:b_s3_b1_conv3_bn_running_var target=s3.b1.conv3.bn.running_var}
          params={channel=C; eps=1e-05}
      n94 {pt2=root[34] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_11)}: [t361 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_33} ->[n102]] =
        permute x=t360 {derived} <-n93 perm=[H<-C, W<-H, C<-W]
    group g25 torch.ops.aten.conv2d.default:
      n95 {derived}: [t362 f32 [H=28 W=28 C=56] {derived} ->[n97]] =
        permute x=t338 {pt2=root:relu_6} <-n71 perm=[H<-W, W<-C, C<-H]
      n96 {derived}: [t363 f32 [N=152 T=1 D=1 H=1 W=1 C=56] {derived} ->[n97]] =
        permute
          x=t36 {pt2=root:p_s3_b1_downsample_conv_weight target=s3.b1.downsample.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n97 {derived}: [t364 f32 [H=14 W=14 C=152] {derived} ->[n98]] =
        conv2d
          x=t362 {derived} <-n95
          weight=t363 {derived} <-n96
          bias=none
          params={h={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
                 in_channels=56;
                 groups=1}
      n98 {pt2=root[35] torch.ops.aten.conv2d.default (conv2d_12)}: [t365 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_12} ->[n99]] =
        permute x=t364 {derived} <-n97 perm=[H<-C, W<-H, C<-W]
    group g26 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n99 {derived}: [t366 f32 [H=14 W=14 C=152] {derived} ->[n100]] =
        permute x=t365 {pt2=root:conv2d_12} <-n98 perm=[H<-W, W<-C, C<-H]
      n100 {derived}: [t367 f32 [H=14 W=14 C=152] {derived} ->[n101]] =
        batch_norm
          x=t366 {derived} <-n99
          weight=t37 {pt2=root:p_s3_b1_downsample_bn_weight target=s3.b1.downsample.bn.weight}
          bias=t38 {pt2=root:p_s3_b1_downsample_bn_bias target=s3.b1.downsample.bn.bias}
          running_mean=t170 {pt2=root:b_s3_b1_downsample_bn_running_mean target=s3.b1.downsample.bn.running_mean}
          running_var=t171 {pt2=root:b_s3_b1_downsample_bn_running_var target=s3.b1.downsample.bn.running_var}
          params={channel=C; eps=1e-05}
      n101 {pt2=root[36] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_12)}: [t368 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_36} ->[n102]] =
        permute x=t367 {derived} <-n100 perm=[H<-C, W<-H, C<-W]
    n102 {pt2=root[37] torch.ops.aten.add.Tensor (add_2)}: [t369 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:add_2} ->[n103]] =
      add
        a=t361 {pt2=root:getitem_33} <-n94
        b=t368 {pt2=root:getitem_36} <-n101
    n103 {pt2=root[38] torch.ops.aten.relu.default (relu_9)}: [t370 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:relu_9} ->[n104,
                                                                      n127]] =
      relu x=t369 {pt2=root:add_2} <-n102
    group g27 torch.ops.aten.conv2d.default:
      n104 {derived}: [t371 f32 [H=14 W=14 C=152] {derived} ->[n106]] =
        permute x=t370 {pt2=root:relu_9} <-n103 perm=[H<-W, W<-C, C<-H]
      n105 {derived}: [t372 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {derived} ->[n106]] =
        permute
          x=t39 {pt2=root:p_s3_b2_conv1_conv_weight target=s3.b2.conv1.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n106 {derived}: [t373 f32 [H=14 W=14 C=152] {derived} ->[n107]] =
        conv2d
          x=t371 {derived} <-n104
          weight=t372 {derived} <-n105
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=152;
                 groups=1}
      n107 {pt2=root[39] torch.ops.aten.conv2d.default (conv2d_13)}: [t374 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_13} ->[n108]] =
        permute x=t373 {derived} <-n106 perm=[H<-C, W<-H, C<-W]
    group g28 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n108 {derived}: [t375 f32 [H=14 W=14 C=152] {derived} ->[n109]] =
        permute x=t374 {pt2=root:conv2d_13} <-n107 perm=[H<-W, W<-C, C<-H]
      n109 {derived}: [t376 f32 [H=14 W=14 C=152] {derived} ->[n110]] =
        batch_norm
          x=t375 {derived} <-n108
          weight=t40 {pt2=root:p_s3_b2_conv1_bn_weight target=s3.b2.conv1.bn.weight}
          bias=t41 {pt2=root:p_s3_b2_conv1_bn_bias target=s3.b2.conv1.bn.bias}
          running_mean=t173 {pt2=root:b_s3_b2_conv1_bn_running_mean target=s3.b2.conv1.bn.running_mean}
          running_var=t174 {pt2=root:b_s3_b2_conv1_bn_running_var target=s3.b2.conv1.bn.running_var}
          params={channel=C; eps=1e-05}
      n110 {pt2=root[40] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_13)}: [t377 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_39} ->[n111]] =
        permute x=t376 {derived} <-n109 perm=[H<-C, W<-H, C<-W]
    n111 {pt2=root[41] torch.ops.aten.relu.default (relu_10)}: [t378 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:relu_10} ->[n112]] =
      relu x=t377 {pt2=root:getitem_39} <-n110
    group g29 torch.ops.aten.conv2d.default:
      n112 {derived}: [t379 f32 [H=14 W=14 C=152] {derived} ->[n114]] =
        permute x=t378 {pt2=root:relu_10} <-n111 perm=[H<-W, W<-C, C<-H]
      n113 {derived}: [t380 f32 [N=152 T=1 D=1 H=3 W=3 C=8] {derived} ->[n114]] =
        permute
          x=t42 {pt2=root:p_s3_b2_conv2_conv_weight target=s3.b2.conv2.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n114 {derived}: [t381 f32 [H=14 W=14 C=152] {derived} ->[n115]] =
        conv2d
          x=t379 {derived} <-n112
          weight=t380 {derived} <-n113
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=152;
                 groups=19}
      n115 {pt2=root[42] torch.ops.aten.conv2d.default (conv2d_14)}: [t382 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_14} ->[n116]] =
        permute x=t381 {derived} <-n114 perm=[H<-C, W<-H, C<-W]
    group g30 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n116 {derived}: [t383 f32 [H=14 W=14 C=152] {derived} ->[n117]] =
        permute x=t382 {pt2=root:conv2d_14} <-n115 perm=[H<-W, W<-C, C<-H]
      n117 {derived}: [t384 f32 [H=14 W=14 C=152] {derived} ->[n118]] =
        batch_norm
          x=t383 {derived} <-n116
          weight=t43 {pt2=root:p_s3_b2_conv2_bn_weight target=s3.b2.conv2.bn.weight}
          bias=t44 {pt2=root:p_s3_b2_conv2_bn_bias target=s3.b2.conv2.bn.bias}
          running_mean=t176 {pt2=root:b_s3_b2_conv2_bn_running_mean target=s3.b2.conv2.bn.running_mean}
          running_var=t177 {pt2=root:b_s3_b2_conv2_bn_running_var target=s3.b2.conv2.bn.running_var}
          params={channel=C; eps=1e-05}
      n118 {pt2=root[43] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_14)}: [t385 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_42} ->[n119]] =
        permute x=t384 {derived} <-n117 perm=[H<-C, W<-H, C<-W]
    n119 {pt2=root[44] torch.ops.aten.relu.default (relu_11)}: [t386 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:relu_11} ->[n120]] =
      relu x=t385 {pt2=root:getitem_42} <-n118
    group g31 torch.ops.aten.conv2d.default:
      n120 {derived}: [t387 f32 [H=14 W=14 C=152] {derived} ->[n122]] =
        permute x=t386 {pt2=root:relu_11} <-n119 perm=[H<-W, W<-C, C<-H]
      n121 {derived}: [t388 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {derived} ->[n122]] =
        permute
          x=t45 {pt2=root:p_s3_b2_conv3_conv_weight target=s3.b2.conv3.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n122 {derived}: [t389 f32 [H=14 W=14 C=152] {derived} ->[n123]] =
        conv2d
          x=t387 {derived} <-n120
          weight=t388 {derived} <-n121
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=152;
                 groups=1}
      n123 {pt2=root[45] torch.ops.aten.conv2d.default (conv2d_15)}: [t390 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_15} ->[n124]] =
        permute x=t389 {derived} <-n122 perm=[H<-C, W<-H, C<-W]
    group g32 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n124 {derived}: [t391 f32 [H=14 W=14 C=152] {derived} ->[n125]] =
        permute x=t390 {pt2=root:conv2d_15} <-n123 perm=[H<-W, W<-C, C<-H]
      n125 {derived}: [t392 f32 [H=14 W=14 C=152] {derived} ->[n126]] =
        batch_norm
          x=t391 {derived} <-n124
          weight=t46 {pt2=root:p_s3_b2_conv3_bn_weight target=s3.b2.conv3.bn.weight}
          bias=t47 {pt2=root:p_s3_b2_conv3_bn_bias target=s3.b2.conv3.bn.bias}
          running_mean=t179 {pt2=root:b_s3_b2_conv3_bn_running_mean target=s3.b2.conv3.bn.running_mean}
          running_var=t180 {pt2=root:b_s3_b2_conv3_bn_running_var target=s3.b2.conv3.bn.running_var}
          params={channel=C; eps=1e-05}
      n126 {pt2=root[46] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_15)}: [t393 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_45} ->[n127]] =
        permute x=t392 {derived} <-n125 perm=[H<-C, W<-H, C<-W]
    n127 {pt2=root[47] torch.ops.aten.add.Tensor (add_3)}: [t394 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:add_3} ->[n128]] =
      add a=t393 {pt2=root:getitem_45} <-n126 b=t370 {pt2=root:relu_9} <-n103
    n128 {pt2=root[48] torch.ops.aten.relu.default (relu_12)}: [t395 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:relu_12} ->[n129,
                                                                      n152]] =
      relu x=t394 {pt2=root:add_3} <-n127
    group g33 torch.ops.aten.conv2d.default:
      n129 {derived}: [t396 f32 [H=14 W=14 C=152] {derived} ->[n131]] =
        permute x=t395 {pt2=root:relu_12} <-n128 perm=[H<-W, W<-C, C<-H]
      n130 {derived}: [t397 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {derived} ->[n131]] =
        permute
          x=t48 {pt2=root:p_s3_b3_conv1_conv_weight target=s3.b3.conv1.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n131 {derived}: [t398 f32 [H=14 W=14 C=152] {derived} ->[n132]] =
        conv2d
          x=t396 {derived} <-n129
          weight=t397 {derived} <-n130
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=152;
                 groups=1}
      n132 {pt2=root[49] torch.ops.aten.conv2d.default (conv2d_16)}: [t399 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_16} ->[n133]] =
        permute x=t398 {derived} <-n131 perm=[H<-C, W<-H, C<-W]
    group g34 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n133 {derived}: [t400 f32 [H=14 W=14 C=152] {derived} ->[n134]] =
        permute x=t399 {pt2=root:conv2d_16} <-n132 perm=[H<-W, W<-C, C<-H]
      n134 {derived}: [t401 f32 [H=14 W=14 C=152] {derived} ->[n135]] =
        batch_norm
          x=t400 {derived} <-n133
          weight=t49 {pt2=root:p_s3_b3_conv1_bn_weight target=s3.b3.conv1.bn.weight}
          bias=t50 {pt2=root:p_s3_b3_conv1_bn_bias target=s3.b3.conv1.bn.bias}
          running_mean=t182 {pt2=root:b_s3_b3_conv1_bn_running_mean target=s3.b3.conv1.bn.running_mean}
          running_var=t183 {pt2=root:b_s3_b3_conv1_bn_running_var target=s3.b3.conv1.bn.running_var}
          params={channel=C; eps=1e-05}
      n135 {pt2=root[50] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_16)}: [t402 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_48} ->[n136]] =
        permute x=t401 {derived} <-n134 perm=[H<-C, W<-H, C<-W]
    n136 {pt2=root[51] torch.ops.aten.relu.default (relu_13)}: [t403 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:relu_13} ->[n137]] =
      relu x=t402 {pt2=root:getitem_48} <-n135
    group g35 torch.ops.aten.conv2d.default:
      n137 {derived}: [t404 f32 [H=14 W=14 C=152] {derived} ->[n139]] =
        permute x=t403 {pt2=root:relu_13} <-n136 perm=[H<-W, W<-C, C<-H]
      n138 {derived}: [t405 f32 [N=152 T=1 D=1 H=3 W=3 C=8] {derived} ->[n139]] =
        permute
          x=t51 {pt2=root:p_s3_b3_conv2_conv_weight target=s3.b3.conv2.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n139 {derived}: [t406 f32 [H=14 W=14 C=152] {derived} ->[n140]] =
        conv2d
          x=t404 {derived} <-n137
          weight=t405 {derived} <-n138
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=152;
                 groups=19}
      n140 {pt2=root[52] torch.ops.aten.conv2d.default (conv2d_17)}: [t407 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_17} ->[n141]] =
        permute x=t406 {derived} <-n139 perm=[H<-C, W<-H, C<-W]
    group g36 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n141 {derived}: [t408 f32 [H=14 W=14 C=152] {derived} ->[n142]] =
        permute x=t407 {pt2=root:conv2d_17} <-n140 perm=[H<-W, W<-C, C<-H]
      n142 {derived}: [t409 f32 [H=14 W=14 C=152] {derived} ->[n143]] =
        batch_norm
          x=t408 {derived} <-n141
          weight=t52 {pt2=root:p_s3_b3_conv2_bn_weight target=s3.b3.conv2.bn.weight}
          bias=t53 {pt2=root:p_s3_b3_conv2_bn_bias target=s3.b3.conv2.bn.bias}
          running_mean=t185 {pt2=root:b_s3_b3_conv2_bn_running_mean target=s3.b3.conv2.bn.running_mean}
          running_var=t186 {pt2=root:b_s3_b3_conv2_bn_running_var target=s3.b3.conv2.bn.running_var}
          params={channel=C; eps=1e-05}
      n143 {pt2=root[53] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_17)}: [t410 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_51} ->[n144]] =
        permute x=t409 {derived} <-n142 perm=[H<-C, W<-H, C<-W]
    n144 {pt2=root[54] torch.ops.aten.relu.default (relu_14)}: [t411 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:relu_14} ->[n145]] =
      relu x=t410 {pt2=root:getitem_51} <-n143
    group g37 torch.ops.aten.conv2d.default:
      n145 {derived}: [t412 f32 [H=14 W=14 C=152] {derived} ->[n147]] =
        permute x=t411 {pt2=root:relu_14} <-n144 perm=[H<-W, W<-C, C<-H]
      n146 {derived}: [t413 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {derived} ->[n147]] =
        permute
          x=t54 {pt2=root:p_s3_b3_conv3_conv_weight target=s3.b3.conv3.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n147 {derived}: [t414 f32 [H=14 W=14 C=152] {derived} ->[n148]] =
        conv2d
          x=t412 {derived} <-n145
          weight=t413 {derived} <-n146
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=152;
                 groups=1}
      n148 {pt2=root[55] torch.ops.aten.conv2d.default (conv2d_18)}: [t415 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_18} ->[n149]] =
        permute x=t414 {derived} <-n147 perm=[H<-C, W<-H, C<-W]
    group g38 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n149 {derived}: [t416 f32 [H=14 W=14 C=152] {derived} ->[n150]] =
        permute x=t415 {pt2=root:conv2d_18} <-n148 perm=[H<-W, W<-C, C<-H]
      n150 {derived}: [t417 f32 [H=14 W=14 C=152] {derived} ->[n151]] =
        batch_norm
          x=t416 {derived} <-n149
          weight=t55 {pt2=root:p_s3_b3_conv3_bn_weight target=s3.b3.conv3.bn.weight}
          bias=t56 {pt2=root:p_s3_b3_conv3_bn_bias target=s3.b3.conv3.bn.bias}
          running_mean=t188 {pt2=root:b_s3_b3_conv3_bn_running_mean target=s3.b3.conv3.bn.running_mean}
          running_var=t189 {pt2=root:b_s3_b3_conv3_bn_running_var target=s3.b3.conv3.bn.running_var}
          params={channel=C; eps=1e-05}
      n151 {pt2=root[56] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_18)}: [t418 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_54} ->[n152]] =
        permute x=t417 {derived} <-n150 perm=[H<-C, W<-H, C<-W]
    n152 {pt2=root[57] torch.ops.aten.add.Tensor (add_4)}: [t419 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:add_4} ->[n153]] =
      add a=t418 {pt2=root:getitem_54} <-n151 b=t395 {pt2=root:relu_12} <-n128
    n153 {pt2=root[58] torch.ops.aten.relu.default (relu_15)}: [t420 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:relu_15} ->[n154,
                                                                      n177]] =
      relu x=t419 {pt2=root:add_4} <-n152
    group g39 torch.ops.aten.conv2d.default:
      n154 {derived}: [t421 f32 [H=14 W=14 C=152] {derived} ->[n156]] =
        permute x=t420 {pt2=root:relu_15} <-n153 perm=[H<-W, W<-C, C<-H]
      n155 {derived}: [t422 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {derived} ->[n156]] =
        permute
          x=t57 {pt2=root:p_s3_b4_conv1_conv_weight target=s3.b4.conv1.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n156 {derived}: [t423 f32 [H=14 W=14 C=152] {derived} ->[n157]] =
        conv2d
          x=t421 {derived} <-n154
          weight=t422 {derived} <-n155
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=152;
                 groups=1}
      n157 {pt2=root[59] torch.ops.aten.conv2d.default (conv2d_19)}: [t424 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_19} ->[n158]] =
        permute x=t423 {derived} <-n156 perm=[H<-C, W<-H, C<-W]
    group g40 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n158 {derived}: [t425 f32 [H=14 W=14 C=152] {derived} ->[n159]] =
        permute x=t424 {pt2=root:conv2d_19} <-n157 perm=[H<-W, W<-C, C<-H]
      n159 {derived}: [t426 f32 [H=14 W=14 C=152] {derived} ->[n160]] =
        batch_norm
          x=t425 {derived} <-n158
          weight=t58 {pt2=root:p_s3_b4_conv1_bn_weight target=s3.b4.conv1.bn.weight}
          bias=t59 {pt2=root:p_s3_b4_conv1_bn_bias target=s3.b4.conv1.bn.bias}
          running_mean=t191 {pt2=root:b_s3_b4_conv1_bn_running_mean target=s3.b4.conv1.bn.running_mean}
          running_var=t192 {pt2=root:b_s3_b4_conv1_bn_running_var target=s3.b4.conv1.bn.running_var}
          params={channel=C; eps=1e-05}
      n160 {pt2=root[60] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_19)}: [t427 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_57} ->[n161]] =
        permute x=t426 {derived} <-n159 perm=[H<-C, W<-H, C<-W]
    n161 {pt2=root[61] torch.ops.aten.relu.default (relu_16)}: [t428 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:relu_16} ->[n162]] =
      relu x=t427 {pt2=root:getitem_57} <-n160
    group g41 torch.ops.aten.conv2d.default:
      n162 {derived}: [t429 f32 [H=14 W=14 C=152] {derived} ->[n164]] =
        permute x=t428 {pt2=root:relu_16} <-n161 perm=[H<-W, W<-C, C<-H]
      n163 {derived}: [t430 f32 [N=152 T=1 D=1 H=3 W=3 C=8] {derived} ->[n164]] =
        permute
          x=t60 {pt2=root:p_s3_b4_conv2_conv_weight target=s3.b4.conv2.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n164 {derived}: [t431 f32 [H=14 W=14 C=152] {derived} ->[n165]] =
        conv2d
          x=t429 {derived} <-n162
          weight=t430 {derived} <-n163
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=152;
                 groups=19}
      n165 {pt2=root[62] torch.ops.aten.conv2d.default (conv2d_20)}: [t432 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_20} ->[n166]] =
        permute x=t431 {derived} <-n164 perm=[H<-C, W<-H, C<-W]
    group g42 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n166 {derived}: [t433 f32 [H=14 W=14 C=152] {derived} ->[n167]] =
        permute x=t432 {pt2=root:conv2d_20} <-n165 perm=[H<-W, W<-C, C<-H]
      n167 {derived}: [t434 f32 [H=14 W=14 C=152] {derived} ->[n168]] =
        batch_norm
          x=t433 {derived} <-n166
          weight=t61 {pt2=root:p_s3_b4_conv2_bn_weight target=s3.b4.conv2.bn.weight}
          bias=t62 {pt2=root:p_s3_b4_conv2_bn_bias target=s3.b4.conv2.bn.bias}
          running_mean=t194 {pt2=root:b_s3_b4_conv2_bn_running_mean target=s3.b4.conv2.bn.running_mean}
          running_var=t195 {pt2=root:b_s3_b4_conv2_bn_running_var target=s3.b4.conv2.bn.running_var}
          params={channel=C; eps=1e-05}
      n168 {pt2=root[63] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_20)}: [t435 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_60} ->[n169]] =
        permute x=t434 {derived} <-n167 perm=[H<-C, W<-H, C<-W]
    n169 {pt2=root[64] torch.ops.aten.relu.default (relu_17)}: [t436 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:relu_17} ->[n170]] =
      relu x=t435 {pt2=root:getitem_60} <-n168
    group g43 torch.ops.aten.conv2d.default:
      n170 {derived}: [t437 f32 [H=14 W=14 C=152] {derived} ->[n172]] =
        permute x=t436 {pt2=root:relu_17} <-n169 perm=[H<-W, W<-C, C<-H]
      n171 {derived}: [t438 f32 [N=152 T=1 D=1 H=1 W=1 C=152] {derived} ->[n172]] =
        permute
          x=t63 {pt2=root:p_s3_b4_conv3_conv_weight target=s3.b4.conv3.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n172 {derived}: [t439 f32 [H=14 W=14 C=152] {derived} ->[n173]] =
        conv2d
          x=t437 {derived} <-n170
          weight=t438 {derived} <-n171
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=152;
                 groups=1}
      n173 {pt2=root[65] torch.ops.aten.conv2d.default (conv2d_21)}: [t440 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_21} ->[n174]] =
        permute x=t439 {derived} <-n172 perm=[H<-C, W<-H, C<-W]
    group g44 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n174 {derived}: [t441 f32 [H=14 W=14 C=152] {derived} ->[n175]] =
        permute x=t440 {pt2=root:conv2d_21} <-n173 perm=[H<-W, W<-C, C<-H]
      n175 {derived}: [t442 f32 [H=14 W=14 C=152] {derived} ->[n176]] =
        batch_norm
          x=t441 {derived} <-n174
          weight=t64 {pt2=root:p_s3_b4_conv3_bn_weight target=s3.b4.conv3.bn.weight}
          bias=t65 {pt2=root:p_s3_b4_conv3_bn_bias target=s3.b4.conv3.bn.bias}
          running_mean=t197 {pt2=root:b_s3_b4_conv3_bn_running_mean target=s3.b4.conv3.bn.running_mean}
          running_var=t198 {pt2=root:b_s3_b4_conv3_bn_running_var target=s3.b4.conv3.bn.running_var}
          params={channel=C; eps=1e-05}
      n176 {pt2=root[66] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_21)}: [t443 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_63} ->[n177]] =
        permute x=t442 {derived} <-n175 perm=[H<-C, W<-H, C<-W]
    n177 {pt2=root[67] torch.ops.aten.add.Tensor (add_5)}: [t444 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:add_5} ->[n178]] =
      add a=t443 {pt2=root:getitem_63} <-n176 b=t420 {pt2=root:relu_15} <-n153
    n178 {pt2=root[68] torch.ops.aten.relu.default (relu_18)}: [t445 f32 [H=152
                                                                      W=14
                                                                      C=14] {pt2=root:relu_18} ->[n179,
                                                                      n202]] =
      relu x=t444 {pt2=root:add_5} <-n177
    group g45 torch.ops.aten.conv2d.default:
      n179 {derived}: [t446 f32 [H=14 W=14 C=152] {derived} ->[n181]] =
        permute x=t445 {pt2=root:relu_18} <-n178 perm=[H<-W, W<-C, C<-H]
      n180 {derived}: [t447 f32 [N=368 T=1 D=1 H=1 W=1 C=152] {derived} ->[n181]] =
        permute
          x=t66 {pt2=root:p_s4_b1_conv1_conv_weight target=s4.b1.conv1.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n181 {derived}: [t448 f32 [H=14 W=14 C=368] {derived} ->[n182]] =
        conv2d
          x=t446 {derived} <-n179
          weight=t447 {derived} <-n180
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=152;
                 groups=1}
      n182 {pt2=root[69] torch.ops.aten.conv2d.default (conv2d_22)}: [t449 f32 [H=368
                                                                      W=14
                                                                      C=14] {pt2=root:conv2d_22} ->[n183]] =
        permute x=t448 {derived} <-n181 perm=[H<-C, W<-H, C<-W]
    group g46 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n183 {derived}: [t450 f32 [H=14 W=14 C=368] {derived} ->[n184]] =
        permute x=t449 {pt2=root:conv2d_22} <-n182 perm=[H<-W, W<-C, C<-H]
      n184 {derived}: [t451 f32 [H=14 W=14 C=368] {derived} ->[n185]] =
        batch_norm
          x=t450 {derived} <-n183
          weight=t67 {pt2=root:p_s4_b1_conv1_bn_weight target=s4.b1.conv1.bn.weight}
          bias=t68 {pt2=root:p_s4_b1_conv1_bn_bias target=s4.b1.conv1.bn.bias}
          running_mean=t200 {pt2=root:b_s4_b1_conv1_bn_running_mean target=s4.b1.conv1.bn.running_mean}
          running_var=t201 {pt2=root:b_s4_b1_conv1_bn_running_var target=s4.b1.conv1.bn.running_var}
          params={channel=C; eps=1e-05}
      n185 {pt2=root[70] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_22)}: [t452 f32 [H=368
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_66} ->[n186]] =
        permute x=t451 {derived} <-n184 perm=[H<-C, W<-H, C<-W]
    n186 {pt2=root[71] torch.ops.aten.relu.default (relu_19)}: [t453 f32 [H=368
                                                                      W=14
                                                                      C=14] {pt2=root:relu_19} ->[n187]] =
      relu x=t452 {pt2=root:getitem_66} <-n185
    group g47 torch.ops.aten.conv2d.default:
      n187 {derived}: [t454 f32 [H=14 W=14 C=368] {derived} ->[n189]] =
        permute x=t453 {pt2=root:relu_19} <-n186 perm=[H<-W, W<-C, C<-H]
      n188 {derived}: [t455 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {derived} ->[n189]] =
        permute
          x=t69 {pt2=root:p_s4_b1_conv2_conv_weight target=s4.b1.conv2.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n189 {derived}: [t456 f32 [H=7 W=7 C=368] {derived} ->[n190]] =
        conv2d
          x=t454 {derived} <-n187
          weight=t455 {derived} <-n188
          bias=none
          params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
                 in_channels=368;
                 groups=46}
      n190 {pt2=root[72] torch.ops.aten.conv2d.default (conv2d_23)}: [t457 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_23} ->[n191]] =
        permute x=t456 {derived} <-n189 perm=[H<-C, W<-H, C<-W]
    group g48 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n191 {derived}: [t458 f32 [H=7 W=7 C=368] {derived} ->[n192]] =
        permute x=t457 {pt2=root:conv2d_23} <-n190 perm=[H<-W, W<-C, C<-H]
      n192 {derived}: [t459 f32 [H=7 W=7 C=368] {derived} ->[n193]] =
        batch_norm
          x=t458 {derived} <-n191
          weight=t70 {pt2=root:p_s4_b1_conv2_bn_weight target=s4.b1.conv2.bn.weight}
          bias=t71 {pt2=root:p_s4_b1_conv2_bn_bias target=s4.b1.conv2.bn.bias}
          running_mean=t203 {pt2=root:b_s4_b1_conv2_bn_running_mean target=s4.b1.conv2.bn.running_mean}
          running_var=t204 {pt2=root:b_s4_b1_conv2_bn_running_var target=s4.b1.conv2.bn.running_var}
          params={channel=C; eps=1e-05}
      n193 {pt2=root[73] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_23)}: [t460 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_69} ->[n194]] =
        permute x=t459 {derived} <-n192 perm=[H<-C, W<-H, C<-W]
    n194 {pt2=root[74] torch.ops.aten.relu.default (relu_20)}: [t461 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_20} ->[n195]] =
      relu x=t460 {pt2=root:getitem_69} <-n193
    group g49 torch.ops.aten.conv2d.default:
      n195 {derived}: [t462 f32 [H=7 W=7 C=368] {derived} ->[n197]] =
        permute x=t461 {pt2=root:relu_20} <-n194 perm=[H<-W, W<-C, C<-H]
      n196 {derived}: [t463 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {derived} ->[n197]] =
        permute
          x=t72 {pt2=root:p_s4_b1_conv3_conv_weight target=s4.b1.conv3.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n197 {derived}: [t464 f32 [H=7 W=7 C=368] {derived} ->[n198]] =
        conv2d
          x=t462 {derived} <-n195
          weight=t463 {derived} <-n196
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=368;
                 groups=1}
      n198 {pt2=root[75] torch.ops.aten.conv2d.default (conv2d_24)}: [t465 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_24} ->[n199]] =
        permute x=t464 {derived} <-n197 perm=[H<-C, W<-H, C<-W]
    group g50 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n199 {derived}: [t466 f32 [H=7 W=7 C=368] {derived} ->[n200]] =
        permute x=t465 {pt2=root:conv2d_24} <-n198 perm=[H<-W, W<-C, C<-H]
      n200 {derived}: [t467 f32 [H=7 W=7 C=368] {derived} ->[n201]] =
        batch_norm
          x=t466 {derived} <-n199
          weight=t73 {pt2=root:p_s4_b1_conv3_bn_weight target=s4.b1.conv3.bn.weight}
          bias=t74 {pt2=root:p_s4_b1_conv3_bn_bias target=s4.b1.conv3.bn.bias}
          running_mean=t206 {pt2=root:b_s4_b1_conv3_bn_running_mean target=s4.b1.conv3.bn.running_mean}
          running_var=t207 {pt2=root:b_s4_b1_conv3_bn_running_var target=s4.b1.conv3.bn.running_var}
          params={channel=C; eps=1e-05}
      n201 {pt2=root[76] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_24)}: [t468 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_72} ->[n209]] =
        permute x=t467 {derived} <-n200 perm=[H<-C, W<-H, C<-W]
    group g51 torch.ops.aten.conv2d.default:
      n202 {derived}: [t469 f32 [H=14 W=14 C=152] {derived} ->[n204]] =
        permute x=t445 {pt2=root:relu_18} <-n178 perm=[H<-W, W<-C, C<-H]
      n203 {derived}: [t470 f32 [N=368 T=1 D=1 H=1 W=1 C=152] {derived} ->[n204]] =
        permute
          x=t75 {pt2=root:p_s4_b1_downsample_conv_weight target=s4.b1.downsample.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n204 {derived}: [t471 f32 [H=7 W=7 C=368] {derived} ->[n205]] =
        conv2d
          x=t469 {derived} <-n202
          weight=t470 {derived} <-n203
          bias=none
          params={h={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=2; pad_before=0; pad_after=0; dilation=1};
                 in_channels=152;
                 groups=1}
      n205 {pt2=root[77] torch.ops.aten.conv2d.default (conv2d_25)}: [t472 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_25} ->[n206]] =
        permute x=t471 {derived} <-n204 perm=[H<-C, W<-H, C<-W]
    group g52 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n206 {derived}: [t473 f32 [H=7 W=7 C=368] {derived} ->[n207]] =
        permute x=t472 {pt2=root:conv2d_25} <-n205 perm=[H<-W, W<-C, C<-H]
      n207 {derived}: [t474 f32 [H=7 W=7 C=368] {derived} ->[n208]] =
        batch_norm
          x=t473 {derived} <-n206
          weight=t76 {pt2=root:p_s4_b1_downsample_bn_weight target=s4.b1.downsample.bn.weight}
          bias=t77 {pt2=root:p_s4_b1_downsample_bn_bias target=s4.b1.downsample.bn.bias}
          running_mean=t209 {pt2=root:b_s4_b1_downsample_bn_running_mean target=s4.b1.downsample.bn.running_mean}
          running_var=t210 {pt2=root:b_s4_b1_downsample_bn_running_var target=s4.b1.downsample.bn.running_var}
          params={channel=C; eps=1e-05}
      n208 {pt2=root[78] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_25)}: [t475 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_75} ->[n209]] =
        permute x=t474 {derived} <-n207 perm=[H<-C, W<-H, C<-W]
    n209 {pt2=root[79] torch.ops.aten.add.Tensor (add_6)}: [t476 f32 [H=368 W=7
                                                                      C=7] {pt2=root:add_6} ->[n210]] =
      add
        a=t468 {pt2=root:getitem_72} <-n201
        b=t475 {pt2=root:getitem_75} <-n208
    n210 {pt2=root[80] torch.ops.aten.relu.default (relu_21)}: [t477 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_21} ->[n211,
                                                                      n234]] =
      relu x=t476 {pt2=root:add_6} <-n209
    group g53 torch.ops.aten.conv2d.default:
      n211 {derived}: [t478 f32 [H=7 W=7 C=368] {derived} ->[n213]] =
        permute x=t477 {pt2=root:relu_21} <-n210 perm=[H<-W, W<-C, C<-H]
      n212 {derived}: [t479 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {derived} ->[n213]] =
        permute
          x=t78 {pt2=root:p_s4_b2_conv1_conv_weight target=s4.b2.conv1.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n213 {derived}: [t480 f32 [H=7 W=7 C=368] {derived} ->[n214]] =
        conv2d
          x=t478 {derived} <-n211
          weight=t479 {derived} <-n212
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=368;
                 groups=1}
      n214 {pt2=root[81] torch.ops.aten.conv2d.default (conv2d_26)}: [t481 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_26} ->[n215]] =
        permute x=t480 {derived} <-n213 perm=[H<-C, W<-H, C<-W]
    group g54 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n215 {derived}: [t482 f32 [H=7 W=7 C=368] {derived} ->[n216]] =
        permute x=t481 {pt2=root:conv2d_26} <-n214 perm=[H<-W, W<-C, C<-H]
      n216 {derived}: [t483 f32 [H=7 W=7 C=368] {derived} ->[n217]] =
        batch_norm
          x=t482 {derived} <-n215
          weight=t79 {pt2=root:p_s4_b2_conv1_bn_weight target=s4.b2.conv1.bn.weight}
          bias=t80 {pt2=root:p_s4_b2_conv1_bn_bias target=s4.b2.conv1.bn.bias}
          running_mean=t212 {pt2=root:b_s4_b2_conv1_bn_running_mean target=s4.b2.conv1.bn.running_mean}
          running_var=t213 {pt2=root:b_s4_b2_conv1_bn_running_var target=s4.b2.conv1.bn.running_var}
          params={channel=C; eps=1e-05}
      n217 {pt2=root[82] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_26)}: [t484 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_78} ->[n218]] =
        permute x=t483 {derived} <-n216 perm=[H<-C, W<-H, C<-W]
    n218 {pt2=root[83] torch.ops.aten.relu.default (relu_22)}: [t485 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_22} ->[n219]] =
      relu x=t484 {pt2=root:getitem_78} <-n217
    group g55 torch.ops.aten.conv2d.default:
      n219 {derived}: [t486 f32 [H=7 W=7 C=368] {derived} ->[n221]] =
        permute x=t485 {pt2=root:relu_22} <-n218 perm=[H<-W, W<-C, C<-H]
      n220 {derived}: [t487 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {derived} ->[n221]] =
        permute
          x=t81 {pt2=root:p_s4_b2_conv2_conv_weight target=s4.b2.conv2.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n221 {derived}: [t488 f32 [H=7 W=7 C=368] {derived} ->[n222]] =
        conv2d
          x=t486 {derived} <-n219
          weight=t487 {derived} <-n220
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=368;
                 groups=46}
      n222 {pt2=root[84] torch.ops.aten.conv2d.default (conv2d_27)}: [t489 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_27} ->[n223]] =
        permute x=t488 {derived} <-n221 perm=[H<-C, W<-H, C<-W]
    group g56 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n223 {derived}: [t490 f32 [H=7 W=7 C=368] {derived} ->[n224]] =
        permute x=t489 {pt2=root:conv2d_27} <-n222 perm=[H<-W, W<-C, C<-H]
      n224 {derived}: [t491 f32 [H=7 W=7 C=368] {derived} ->[n225]] =
        batch_norm
          x=t490 {derived} <-n223
          weight=t82 {pt2=root:p_s4_b2_conv2_bn_weight target=s4.b2.conv2.bn.weight}
          bias=t83 {pt2=root:p_s4_b2_conv2_bn_bias target=s4.b2.conv2.bn.bias}
          running_mean=t215 {pt2=root:b_s4_b2_conv2_bn_running_mean target=s4.b2.conv2.bn.running_mean}
          running_var=t216 {pt2=root:b_s4_b2_conv2_bn_running_var target=s4.b2.conv2.bn.running_var}
          params={channel=C; eps=1e-05}
      n225 {pt2=root[85] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_27)}: [t492 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_81} ->[n226]] =
        permute x=t491 {derived} <-n224 perm=[H<-C, W<-H, C<-W]
    n226 {pt2=root[86] torch.ops.aten.relu.default (relu_23)}: [t493 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_23} ->[n227]] =
      relu x=t492 {pt2=root:getitem_81} <-n225
    group g57 torch.ops.aten.conv2d.default:
      n227 {derived}: [t494 f32 [H=7 W=7 C=368] {derived} ->[n229]] =
        permute x=t493 {pt2=root:relu_23} <-n226 perm=[H<-W, W<-C, C<-H]
      n228 {derived}: [t495 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {derived} ->[n229]] =
        permute
          x=t84 {pt2=root:p_s4_b2_conv3_conv_weight target=s4.b2.conv3.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n229 {derived}: [t496 f32 [H=7 W=7 C=368] {derived} ->[n230]] =
        conv2d
          x=t494 {derived} <-n227
          weight=t495 {derived} <-n228
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=368;
                 groups=1}
      n230 {pt2=root[87] torch.ops.aten.conv2d.default (conv2d_28)}: [t497 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_28} ->[n231]] =
        permute x=t496 {derived} <-n229 perm=[H<-C, W<-H, C<-W]
    group g58 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n231 {derived}: [t498 f32 [H=7 W=7 C=368] {derived} ->[n232]] =
        permute x=t497 {pt2=root:conv2d_28} <-n230 perm=[H<-W, W<-C, C<-H]
      n232 {derived}: [t499 f32 [H=7 W=7 C=368] {derived} ->[n233]] =
        batch_norm
          x=t498 {derived} <-n231
          weight=t85 {pt2=root:p_s4_b2_conv3_bn_weight target=s4.b2.conv3.bn.weight}
          bias=t86 {pt2=root:p_s4_b2_conv3_bn_bias target=s4.b2.conv3.bn.bias}
          running_mean=t218 {pt2=root:b_s4_b2_conv3_bn_running_mean target=s4.b2.conv3.bn.running_mean}
          running_var=t219 {pt2=root:b_s4_b2_conv3_bn_running_var target=s4.b2.conv3.bn.running_var}
          params={channel=C; eps=1e-05}
      n233 {pt2=root[88] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_28)}: [t500 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_84} ->[n234]] =
        permute x=t499 {derived} <-n232 perm=[H<-C, W<-H, C<-W]
    n234 {pt2=root[89] torch.ops.aten.add.Tensor (add_7)}: [t501 f32 [H=368 W=7
                                                                      C=7] {pt2=root:add_7} ->[n235]] =
      add a=t500 {pt2=root:getitem_84} <-n233 b=t477 {pt2=root:relu_21} <-n210
    n235 {pt2=root[90] torch.ops.aten.relu.default (relu_24)}: [t502 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_24} ->[n236,
                                                                      n259]] =
      relu x=t501 {pt2=root:add_7} <-n234
    group g59 torch.ops.aten.conv2d.default:
      n236 {derived}: [t503 f32 [H=7 W=7 C=368] {derived} ->[n238]] =
        permute x=t502 {pt2=root:relu_24} <-n235 perm=[H<-W, W<-C, C<-H]
      n237 {derived}: [t504 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {derived} ->[n238]] =
        permute
          x=t87 {pt2=root:p_s4_b3_conv1_conv_weight target=s4.b3.conv1.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n238 {derived}: [t505 f32 [H=7 W=7 C=368] {derived} ->[n239]] =
        conv2d
          x=t503 {derived} <-n236
          weight=t504 {derived} <-n237
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=368;
                 groups=1}
      n239 {pt2=root[91] torch.ops.aten.conv2d.default (conv2d_29)}: [t506 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_29} ->[n240]] =
        permute x=t505 {derived} <-n238 perm=[H<-C, W<-H, C<-W]
    group g60 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n240 {derived}: [t507 f32 [H=7 W=7 C=368] {derived} ->[n241]] =
        permute x=t506 {pt2=root:conv2d_29} <-n239 perm=[H<-W, W<-C, C<-H]
      n241 {derived}: [t508 f32 [H=7 W=7 C=368] {derived} ->[n242]] =
        batch_norm
          x=t507 {derived} <-n240
          weight=t88 {pt2=root:p_s4_b3_conv1_bn_weight target=s4.b3.conv1.bn.weight}
          bias=t89 {pt2=root:p_s4_b3_conv1_bn_bias target=s4.b3.conv1.bn.bias}
          running_mean=t221 {pt2=root:b_s4_b3_conv1_bn_running_mean target=s4.b3.conv1.bn.running_mean}
          running_var=t222 {pt2=root:b_s4_b3_conv1_bn_running_var target=s4.b3.conv1.bn.running_var}
          params={channel=C; eps=1e-05}
      n242 {pt2=root[92] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_29)}: [t509 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_87} ->[n243]] =
        permute x=t508 {derived} <-n241 perm=[H<-C, W<-H, C<-W]
    n243 {pt2=root[93] torch.ops.aten.relu.default (relu_25)}: [t510 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_25} ->[n244]] =
      relu x=t509 {pt2=root:getitem_87} <-n242
    group g61 torch.ops.aten.conv2d.default:
      n244 {derived}: [t511 f32 [H=7 W=7 C=368] {derived} ->[n246]] =
        permute x=t510 {pt2=root:relu_25} <-n243 perm=[H<-W, W<-C, C<-H]
      n245 {derived}: [t512 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {derived} ->[n246]] =
        permute
          x=t90 {pt2=root:p_s4_b3_conv2_conv_weight target=s4.b3.conv2.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n246 {derived}: [t513 f32 [H=7 W=7 C=368] {derived} ->[n247]] =
        conv2d
          x=t511 {derived} <-n244
          weight=t512 {derived} <-n245
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=368;
                 groups=46}
      n247 {pt2=root[94] torch.ops.aten.conv2d.default (conv2d_30)}: [t514 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_30} ->[n248]] =
        permute x=t513 {derived} <-n246 perm=[H<-C, W<-H, C<-W]
    group g62 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n248 {derived}: [t515 f32 [H=7 W=7 C=368] {derived} ->[n249]] =
        permute x=t514 {pt2=root:conv2d_30} <-n247 perm=[H<-W, W<-C, C<-H]
      n249 {derived}: [t516 f32 [H=7 W=7 C=368] {derived} ->[n250]] =
        batch_norm
          x=t515 {derived} <-n248
          weight=t91 {pt2=root:p_s4_b3_conv2_bn_weight target=s4.b3.conv2.bn.weight}
          bias=t92 {pt2=root:p_s4_b3_conv2_bn_bias target=s4.b3.conv2.bn.bias}
          running_mean=t224 {pt2=root:b_s4_b3_conv2_bn_running_mean target=s4.b3.conv2.bn.running_mean}
          running_var=t225 {pt2=root:b_s4_b3_conv2_bn_running_var target=s4.b3.conv2.bn.running_var}
          params={channel=C; eps=1e-05}
      n250 {pt2=root[95] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_30)}: [t517 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_90} ->[n251]] =
        permute x=t516 {derived} <-n249 perm=[H<-C, W<-H, C<-W]
    n251 {pt2=root[96] torch.ops.aten.relu.default (relu_26)}: [t518 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_26} ->[n252]] =
      relu x=t517 {pt2=root:getitem_90} <-n250
    group g63 torch.ops.aten.conv2d.default:
      n252 {derived}: [t519 f32 [H=7 W=7 C=368] {derived} ->[n254]] =
        permute x=t518 {pt2=root:relu_26} <-n251 perm=[H<-W, W<-C, C<-H]
      n253 {derived}: [t520 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {derived} ->[n254]] =
        permute
          x=t93 {pt2=root:p_s4_b3_conv3_conv_weight target=s4.b3.conv3.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n254 {derived}: [t521 f32 [H=7 W=7 C=368] {derived} ->[n255]] =
        conv2d
          x=t519 {derived} <-n252
          weight=t520 {derived} <-n253
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=368;
                 groups=1}
      n255 {pt2=root[97] torch.ops.aten.conv2d.default (conv2d_31)}: [t522 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_31} ->[n256]] =
        permute x=t521 {derived} <-n254 perm=[H<-C, W<-H, C<-W]
    group g64 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n256 {derived}: [t523 f32 [H=7 W=7 C=368] {derived} ->[n257]] =
        permute x=t522 {pt2=root:conv2d_31} <-n255 perm=[H<-W, W<-C, C<-H]
      n257 {derived}: [t524 f32 [H=7 W=7 C=368] {derived} ->[n258]] =
        batch_norm
          x=t523 {derived} <-n256
          weight=t94 {pt2=root:p_s4_b3_conv3_bn_weight target=s4.b3.conv3.bn.weight}
          bias=t95 {pt2=root:p_s4_b3_conv3_bn_bias target=s4.b3.conv3.bn.bias}
          running_mean=t227 {pt2=root:b_s4_b3_conv3_bn_running_mean target=s4.b3.conv3.bn.running_mean}
          running_var=t228 {pt2=root:b_s4_b3_conv3_bn_running_var target=s4.b3.conv3.bn.running_var}
          params={channel=C; eps=1e-05}
      n258 {pt2=root[98] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_31)}: [t525 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_93} ->[n259]] =
        permute x=t524 {derived} <-n257 perm=[H<-C, W<-H, C<-W]
    n259 {pt2=root[99] torch.ops.aten.add.Tensor (add_8)}: [t526 f32 [H=368 W=7
                                                                      C=7] {pt2=root:add_8} ->[n260]] =
      add a=t525 {pt2=root:getitem_93} <-n258 b=t502 {pt2=root:relu_24} <-n235
    n260 {pt2=root[100] torch.ops.aten.relu.default (relu_27)}: [t527 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_27} ->[n261,
                                                                      n284]] =
      relu x=t526 {pt2=root:add_8} <-n259
    group g65 torch.ops.aten.conv2d.default:
      n261 {derived}: [t528 f32 [H=7 W=7 C=368] {derived} ->[n263]] =
        permute x=t527 {pt2=root:relu_27} <-n260 perm=[H<-W, W<-C, C<-H]
      n262 {derived}: [t529 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {derived} ->[n263]] =
        permute
          x=t96 {pt2=root:p_s4_b4_conv1_conv_weight target=s4.b4.conv1.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n263 {derived}: [t530 f32 [H=7 W=7 C=368] {derived} ->[n264]] =
        conv2d
          x=t528 {derived} <-n261
          weight=t529 {derived} <-n262
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=368;
                 groups=1}
      n264 {pt2=root[101] torch.ops.aten.conv2d.default (conv2d_32)}: [t531 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_32} ->[n265]] =
        permute x=t530 {derived} <-n263 perm=[H<-C, W<-H, C<-W]
    group g66 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n265 {derived}: [t532 f32 [H=7 W=7 C=368] {derived} ->[n266]] =
        permute x=t531 {pt2=root:conv2d_32} <-n264 perm=[H<-W, W<-C, C<-H]
      n266 {derived}: [t533 f32 [H=7 W=7 C=368] {derived} ->[n267]] =
        batch_norm
          x=t532 {derived} <-n265
          weight=t97 {pt2=root:p_s4_b4_conv1_bn_weight target=s4.b4.conv1.bn.weight}
          bias=t98 {pt2=root:p_s4_b4_conv1_bn_bias target=s4.b4.conv1.bn.bias}
          running_mean=t230 {pt2=root:b_s4_b4_conv1_bn_running_mean target=s4.b4.conv1.bn.running_mean}
          running_var=t231 {pt2=root:b_s4_b4_conv1_bn_running_var target=s4.b4.conv1.bn.running_var}
          params={channel=C; eps=1e-05}
      n267 {pt2=root[102] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_32)}: [t534 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_96} ->[n268]] =
        permute x=t533 {derived} <-n266 perm=[H<-C, W<-H, C<-W]
    n268 {pt2=root[103] torch.ops.aten.relu.default (relu_28)}: [t535 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_28} ->[n269]] =
      relu x=t534 {pt2=root:getitem_96} <-n267
    group g67 torch.ops.aten.conv2d.default:
      n269 {derived}: [t536 f32 [H=7 W=7 C=368] {derived} ->[n271]] =
        permute x=t535 {pt2=root:relu_28} <-n268 perm=[H<-W, W<-C, C<-H]
      n270 {derived}: [t537 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {derived} ->[n271]] =
        permute
          x=t99 {pt2=root:p_s4_b4_conv2_conv_weight target=s4.b4.conv2.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n271 {derived}: [t538 f32 [H=7 W=7 C=368] {derived} ->[n272]] =
        conv2d
          x=t536 {derived} <-n269
          weight=t537 {derived} <-n270
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=368;
                 groups=46}
      n272 {pt2=root[104] torch.ops.aten.conv2d.default (conv2d_33)}: [t539 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_33} ->[n273]] =
        permute x=t538 {derived} <-n271 perm=[H<-C, W<-H, C<-W]
    group g68 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n273 {derived}: [t540 f32 [H=7 W=7 C=368] {derived} ->[n274]] =
        permute x=t539 {pt2=root:conv2d_33} <-n272 perm=[H<-W, W<-C, C<-H]
      n274 {derived}: [t541 f32 [H=7 W=7 C=368] {derived} ->[n275]] =
        batch_norm
          x=t540 {derived} <-n273
          weight=t100 {pt2=root:p_s4_b4_conv2_bn_weight target=s4.b4.conv2.bn.weight}
          bias=t101 {pt2=root:p_s4_b4_conv2_bn_bias target=s4.b4.conv2.bn.bias}
          running_mean=t233 {pt2=root:b_s4_b4_conv2_bn_running_mean target=s4.b4.conv2.bn.running_mean}
          running_var=t234 {pt2=root:b_s4_b4_conv2_bn_running_var target=s4.b4.conv2.bn.running_var}
          params={channel=C; eps=1e-05}
      n275 {pt2=root[105] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_33)}: [t542 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_99} ->[n276]] =
        permute x=t541 {derived} <-n274 perm=[H<-C, W<-H, C<-W]
    n276 {pt2=root[106] torch.ops.aten.relu.default (relu_29)}: [t543 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_29} ->[n277]] =
      relu x=t542 {pt2=root:getitem_99} <-n275
    group g69 torch.ops.aten.conv2d.default:
      n277 {derived}: [t544 f32 [H=7 W=7 C=368] {derived} ->[n279]] =
        permute x=t543 {pt2=root:relu_29} <-n276 perm=[H<-W, W<-C, C<-H]
      n278 {derived}: [t545 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {derived} ->[n279]] =
        permute
          x=t102 {pt2=root:p_s4_b4_conv3_conv_weight target=s4.b4.conv3.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n279 {derived}: [t546 f32 [H=7 W=7 C=368] {derived} ->[n280]] =
        conv2d
          x=t544 {derived} <-n277
          weight=t545 {derived} <-n278
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=368;
                 groups=1}
      n280 {pt2=root[107] torch.ops.aten.conv2d.default (conv2d_34)}: [t547 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_34} ->[n281]] =
        permute x=t546 {derived} <-n279 perm=[H<-C, W<-H, C<-W]
    group g70 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n281 {derived}: [t548 f32 [H=7 W=7 C=368] {derived} ->[n282]] =
        permute x=t547 {pt2=root:conv2d_34} <-n280 perm=[H<-W, W<-C, C<-H]
      n282 {derived}: [t549 f32 [H=7 W=7 C=368] {derived} ->[n283]] =
        batch_norm
          x=t548 {derived} <-n281
          weight=t103 {pt2=root:p_s4_b4_conv3_bn_weight target=s4.b4.conv3.bn.weight}
          bias=t104 {pt2=root:p_s4_b4_conv3_bn_bias target=s4.b4.conv3.bn.bias}
          running_mean=t236 {pt2=root:b_s4_b4_conv3_bn_running_mean target=s4.b4.conv3.bn.running_mean}
          running_var=t237 {pt2=root:b_s4_b4_conv3_bn_running_var target=s4.b4.conv3.bn.running_var}
          params={channel=C; eps=1e-05}
      n283 {pt2=root[108] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_34)}: [t550 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_102} ->[n284]] =
        permute x=t549 {derived} <-n282 perm=[H<-C, W<-H, C<-W]
    n284 {pt2=root[109] torch.ops.aten.add.Tensor (add_9)}: [t551 f32 [H=368
                                                                      W=7 C=7] {pt2=root:add_9} ->[n285]] =
      add a=t550 {pt2=root:getitem_102} <-n283 b=t527 {pt2=root:relu_27} <-n260
    n285 {pt2=root[110] torch.ops.aten.relu.default (relu_30)}: [t552 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_30} ->[n286,
                                                                      n309]] =
      relu x=t551 {pt2=root:add_9} <-n284
    group g71 torch.ops.aten.conv2d.default:
      n286 {derived}: [t553 f32 [H=7 W=7 C=368] {derived} ->[n288]] =
        permute x=t552 {pt2=root:relu_30} <-n285 perm=[H<-W, W<-C, C<-H]
      n287 {derived}: [t554 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {derived} ->[n288]] =
        permute
          x=t105 {pt2=root:p_s4_b5_conv1_conv_weight target=s4.b5.conv1.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n288 {derived}: [t555 f32 [H=7 W=7 C=368] {derived} ->[n289]] =
        conv2d
          x=t553 {derived} <-n286
          weight=t554 {derived} <-n287
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=368;
                 groups=1}
      n289 {pt2=root[111] torch.ops.aten.conv2d.default (conv2d_35)}: [t556 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_35} ->[n290]] =
        permute x=t555 {derived} <-n288 perm=[H<-C, W<-H, C<-W]
    group g72 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n290 {derived}: [t557 f32 [H=7 W=7 C=368] {derived} ->[n291]] =
        permute x=t556 {pt2=root:conv2d_35} <-n289 perm=[H<-W, W<-C, C<-H]
      n291 {derived}: [t558 f32 [H=7 W=7 C=368] {derived} ->[n292]] =
        batch_norm
          x=t557 {derived} <-n290
          weight=t106 {pt2=root:p_s4_b5_conv1_bn_weight target=s4.b5.conv1.bn.weight}
          bias=t107 {pt2=root:p_s4_b5_conv1_bn_bias target=s4.b5.conv1.bn.bias}
          running_mean=t239 {pt2=root:b_s4_b5_conv1_bn_running_mean target=s4.b5.conv1.bn.running_mean}
          running_var=t240 {pt2=root:b_s4_b5_conv1_bn_running_var target=s4.b5.conv1.bn.running_var}
          params={channel=C; eps=1e-05}
      n292 {pt2=root[112] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_35)}: [t559 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_105} ->[n293]] =
        permute x=t558 {derived} <-n291 perm=[H<-C, W<-H, C<-W]
    n293 {pt2=root[113] torch.ops.aten.relu.default (relu_31)}: [t560 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_31} ->[n294]] =
      relu x=t559 {pt2=root:getitem_105} <-n292
    group g73 torch.ops.aten.conv2d.default:
      n294 {derived}: [t561 f32 [H=7 W=7 C=368] {derived} ->[n296]] =
        permute x=t560 {pt2=root:relu_31} <-n293 perm=[H<-W, W<-C, C<-H]
      n295 {derived}: [t562 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {derived} ->[n296]] =
        permute
          x=t108 {pt2=root:p_s4_b5_conv2_conv_weight target=s4.b5.conv2.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n296 {derived}: [t563 f32 [H=7 W=7 C=368] {derived} ->[n297]] =
        conv2d
          x=t561 {derived} <-n294
          weight=t562 {derived} <-n295
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=368;
                 groups=46}
      n297 {pt2=root[114] torch.ops.aten.conv2d.default (conv2d_36)}: [t564 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_36} ->[n298]] =
        permute x=t563 {derived} <-n296 perm=[H<-C, W<-H, C<-W]
    group g74 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n298 {derived}: [t565 f32 [H=7 W=7 C=368] {derived} ->[n299]] =
        permute x=t564 {pt2=root:conv2d_36} <-n297 perm=[H<-W, W<-C, C<-H]
      n299 {derived}: [t566 f32 [H=7 W=7 C=368] {derived} ->[n300]] =
        batch_norm
          x=t565 {derived} <-n298
          weight=t109 {pt2=root:p_s4_b5_conv2_bn_weight target=s4.b5.conv2.bn.weight}
          bias=t110 {pt2=root:p_s4_b5_conv2_bn_bias target=s4.b5.conv2.bn.bias}
          running_mean=t242 {pt2=root:b_s4_b5_conv2_bn_running_mean target=s4.b5.conv2.bn.running_mean}
          running_var=t243 {pt2=root:b_s4_b5_conv2_bn_running_var target=s4.b5.conv2.bn.running_var}
          params={channel=C; eps=1e-05}
      n300 {pt2=root[115] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_36)}: [t567 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_108} ->[n301]] =
        permute x=t566 {derived} <-n299 perm=[H<-C, W<-H, C<-W]
    n301 {pt2=root[116] torch.ops.aten.relu.default (relu_32)}: [t568 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_32} ->[n302]] =
      relu x=t567 {pt2=root:getitem_108} <-n300
    group g75 torch.ops.aten.conv2d.default:
      n302 {derived}: [t569 f32 [H=7 W=7 C=368] {derived} ->[n304]] =
        permute x=t568 {pt2=root:relu_32} <-n301 perm=[H<-W, W<-C, C<-H]
      n303 {derived}: [t570 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {derived} ->[n304]] =
        permute
          x=t111 {pt2=root:p_s4_b5_conv3_conv_weight target=s4.b5.conv3.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n304 {derived}: [t571 f32 [H=7 W=7 C=368] {derived} ->[n305]] =
        conv2d
          x=t569 {derived} <-n302
          weight=t570 {derived} <-n303
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=368;
                 groups=1}
      n305 {pt2=root[117] torch.ops.aten.conv2d.default (conv2d_37)}: [t572 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_37} ->[n306]] =
        permute x=t571 {derived} <-n304 perm=[H<-C, W<-H, C<-W]
    group g76 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n306 {derived}: [t573 f32 [H=7 W=7 C=368] {derived} ->[n307]] =
        permute x=t572 {pt2=root:conv2d_37} <-n305 perm=[H<-W, W<-C, C<-H]
      n307 {derived}: [t574 f32 [H=7 W=7 C=368] {derived} ->[n308]] =
        batch_norm
          x=t573 {derived} <-n306
          weight=t112 {pt2=root:p_s4_b5_conv3_bn_weight target=s4.b5.conv3.bn.weight}
          bias=t113 {pt2=root:p_s4_b5_conv3_bn_bias target=s4.b5.conv3.bn.bias}
          running_mean=t245 {pt2=root:b_s4_b5_conv3_bn_running_mean target=s4.b5.conv3.bn.running_mean}
          running_var=t246 {pt2=root:b_s4_b5_conv3_bn_running_var target=s4.b5.conv3.bn.running_var}
          params={channel=C; eps=1e-05}
      n308 {pt2=root[118] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_37)}: [t575 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_111} ->[n309]] =
        permute x=t574 {derived} <-n307 perm=[H<-C, W<-H, C<-W]
    n309 {pt2=root[119] torch.ops.aten.add.Tensor (add_10)}: [t576 f32 [H=368
                                                                      W=7 C=7] {pt2=root:add_10} ->[n310]] =
      add a=t575 {pt2=root:getitem_111} <-n308 b=t552 {pt2=root:relu_30} <-n285
    n310 {pt2=root[120] torch.ops.aten.relu.default (relu_33)}: [t577 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_33} ->[n311,
                                                                      n334]] =
      relu x=t576 {pt2=root:add_10} <-n309
    group g77 torch.ops.aten.conv2d.default:
      n311 {derived}: [t578 f32 [H=7 W=7 C=368] {derived} ->[n313]] =
        permute x=t577 {pt2=root:relu_33} <-n310 perm=[H<-W, W<-C, C<-H]
      n312 {derived}: [t579 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {derived} ->[n313]] =
        permute
          x=t114 {pt2=root:p_s4_b6_conv1_conv_weight target=s4.b6.conv1.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n313 {derived}: [t580 f32 [H=7 W=7 C=368] {derived} ->[n314]] =
        conv2d
          x=t578 {derived} <-n311
          weight=t579 {derived} <-n312
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=368;
                 groups=1}
      n314 {pt2=root[121] torch.ops.aten.conv2d.default (conv2d_38)}: [t581 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_38} ->[n315]] =
        permute x=t580 {derived} <-n313 perm=[H<-C, W<-H, C<-W]
    group g78 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n315 {derived}: [t582 f32 [H=7 W=7 C=368] {derived} ->[n316]] =
        permute x=t581 {pt2=root:conv2d_38} <-n314 perm=[H<-W, W<-C, C<-H]
      n316 {derived}: [t583 f32 [H=7 W=7 C=368] {derived} ->[n317]] =
        batch_norm
          x=t582 {derived} <-n315
          weight=t115 {pt2=root:p_s4_b6_conv1_bn_weight target=s4.b6.conv1.bn.weight}
          bias=t116 {pt2=root:p_s4_b6_conv1_bn_bias target=s4.b6.conv1.bn.bias}
          running_mean=t248 {pt2=root:b_s4_b6_conv1_bn_running_mean target=s4.b6.conv1.bn.running_mean}
          running_var=t249 {pt2=root:b_s4_b6_conv1_bn_running_var target=s4.b6.conv1.bn.running_var}
          params={channel=C; eps=1e-05}
      n317 {pt2=root[122] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_38)}: [t584 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_114} ->[n318]] =
        permute x=t583 {derived} <-n316 perm=[H<-C, W<-H, C<-W]
    n318 {pt2=root[123] torch.ops.aten.relu.default (relu_34)}: [t585 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_34} ->[n319]] =
      relu x=t584 {pt2=root:getitem_114} <-n317
    group g79 torch.ops.aten.conv2d.default:
      n319 {derived}: [t586 f32 [H=7 W=7 C=368] {derived} ->[n321]] =
        permute x=t585 {pt2=root:relu_34} <-n318 perm=[H<-W, W<-C, C<-H]
      n320 {derived}: [t587 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {derived} ->[n321]] =
        permute
          x=t117 {pt2=root:p_s4_b6_conv2_conv_weight target=s4.b6.conv2.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n321 {derived}: [t588 f32 [H=7 W=7 C=368] {derived} ->[n322]] =
        conv2d
          x=t586 {derived} <-n319
          weight=t587 {derived} <-n320
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=368;
                 groups=46}
      n322 {pt2=root[124] torch.ops.aten.conv2d.default (conv2d_39)}: [t589 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_39} ->[n323]] =
        permute x=t588 {derived} <-n321 perm=[H<-C, W<-H, C<-W]
    group g80 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n323 {derived}: [t590 f32 [H=7 W=7 C=368] {derived} ->[n324]] =
        permute x=t589 {pt2=root:conv2d_39} <-n322 perm=[H<-W, W<-C, C<-H]
      n324 {derived}: [t591 f32 [H=7 W=7 C=368] {derived} ->[n325]] =
        batch_norm
          x=t590 {derived} <-n323
          weight=t118 {pt2=root:p_s4_b6_conv2_bn_weight target=s4.b6.conv2.bn.weight}
          bias=t119 {pt2=root:p_s4_b6_conv2_bn_bias target=s4.b6.conv2.bn.bias}
          running_mean=t251 {pt2=root:b_s4_b6_conv2_bn_running_mean target=s4.b6.conv2.bn.running_mean}
          running_var=t252 {pt2=root:b_s4_b6_conv2_bn_running_var target=s4.b6.conv2.bn.running_var}
          params={channel=C; eps=1e-05}
      n325 {pt2=root[125] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_39)}: [t592 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_117} ->[n326]] =
        permute x=t591 {derived} <-n324 perm=[H<-C, W<-H, C<-W]
    n326 {pt2=root[126] torch.ops.aten.relu.default (relu_35)}: [t593 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_35} ->[n327]] =
      relu x=t592 {pt2=root:getitem_117} <-n325
    group g81 torch.ops.aten.conv2d.default:
      n327 {derived}: [t594 f32 [H=7 W=7 C=368] {derived} ->[n329]] =
        permute x=t593 {pt2=root:relu_35} <-n326 perm=[H<-W, W<-C, C<-H]
      n328 {derived}: [t595 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {derived} ->[n329]] =
        permute
          x=t120 {pt2=root:p_s4_b6_conv3_conv_weight target=s4.b6.conv3.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n329 {derived}: [t596 f32 [H=7 W=7 C=368] {derived} ->[n330]] =
        conv2d
          x=t594 {derived} <-n327
          weight=t595 {derived} <-n328
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=368;
                 groups=1}
      n330 {pt2=root[127] torch.ops.aten.conv2d.default (conv2d_40)}: [t597 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_40} ->[n331]] =
        permute x=t596 {derived} <-n329 perm=[H<-C, W<-H, C<-W]
    group g82 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n331 {derived}: [t598 f32 [H=7 W=7 C=368] {derived} ->[n332]] =
        permute x=t597 {pt2=root:conv2d_40} <-n330 perm=[H<-W, W<-C, C<-H]
      n332 {derived}: [t599 f32 [H=7 W=7 C=368] {derived} ->[n333]] =
        batch_norm
          x=t598 {derived} <-n331
          weight=t121 {pt2=root:p_s4_b6_conv3_bn_weight target=s4.b6.conv3.bn.weight}
          bias=t122 {pt2=root:p_s4_b6_conv3_bn_bias target=s4.b6.conv3.bn.bias}
          running_mean=t254 {pt2=root:b_s4_b6_conv3_bn_running_mean target=s4.b6.conv3.bn.running_mean}
          running_var=t255 {pt2=root:b_s4_b6_conv3_bn_running_var target=s4.b6.conv3.bn.running_var}
          params={channel=C; eps=1e-05}
      n333 {pt2=root[128] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_40)}: [t600 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_120} ->[n334]] =
        permute x=t599 {derived} <-n332 perm=[H<-C, W<-H, C<-W]
    n334 {pt2=root[129] torch.ops.aten.add.Tensor (add_11)}: [t601 f32 [H=368
                                                                      W=7 C=7] {pt2=root:add_11} ->[n335]] =
      add a=t600 {pt2=root:getitem_120} <-n333 b=t577 {pt2=root:relu_33} <-n310
    n335 {pt2=root[130] torch.ops.aten.relu.default (relu_36)}: [t602 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_36} ->[n336,
                                                                      n359]] =
      relu x=t601 {pt2=root:add_11} <-n334
    group g83 torch.ops.aten.conv2d.default:
      n336 {derived}: [t603 f32 [H=7 W=7 C=368] {derived} ->[n338]] =
        permute x=t602 {pt2=root:relu_36} <-n335 perm=[H<-W, W<-C, C<-H]
      n337 {derived}: [t604 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {derived} ->[n338]] =
        permute
          x=t123 {pt2=root:p_s4_b7_conv1_conv_weight target=s4.b7.conv1.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n338 {derived}: [t605 f32 [H=7 W=7 C=368] {derived} ->[n339]] =
        conv2d
          x=t603 {derived} <-n336
          weight=t604 {derived} <-n337
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=368;
                 groups=1}
      n339 {pt2=root[131] torch.ops.aten.conv2d.default (conv2d_41)}: [t606 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_41} ->[n340]] =
        permute x=t605 {derived} <-n338 perm=[H<-C, W<-H, C<-W]
    group g84 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n340 {derived}: [t607 f32 [H=7 W=7 C=368] {derived} ->[n341]] =
        permute x=t606 {pt2=root:conv2d_41} <-n339 perm=[H<-W, W<-C, C<-H]
      n341 {derived}: [t608 f32 [H=7 W=7 C=368] {derived} ->[n342]] =
        batch_norm
          x=t607 {derived} <-n340
          weight=t124 {pt2=root:p_s4_b7_conv1_bn_weight target=s4.b7.conv1.bn.weight}
          bias=t125 {pt2=root:p_s4_b7_conv1_bn_bias target=s4.b7.conv1.bn.bias}
          running_mean=t257 {pt2=root:b_s4_b7_conv1_bn_running_mean target=s4.b7.conv1.bn.running_mean}
          running_var=t258 {pt2=root:b_s4_b7_conv1_bn_running_var target=s4.b7.conv1.bn.running_var}
          params={channel=C; eps=1e-05}
      n342 {pt2=root[132] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_41)}: [t609 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_123} ->[n343]] =
        permute x=t608 {derived} <-n341 perm=[H<-C, W<-H, C<-W]
    n343 {pt2=root[133] torch.ops.aten.relu.default (relu_37)}: [t610 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_37} ->[n344]] =
      relu x=t609 {pt2=root:getitem_123} <-n342
    group g85 torch.ops.aten.conv2d.default:
      n344 {derived}: [t611 f32 [H=7 W=7 C=368] {derived} ->[n346]] =
        permute x=t610 {pt2=root:relu_37} <-n343 perm=[H<-W, W<-C, C<-H]
      n345 {derived}: [t612 f32 [N=368 T=1 D=1 H=3 W=3 C=8] {derived} ->[n346]] =
        permute
          x=t126 {pt2=root:p_s4_b7_conv2_conv_weight target=s4.b7.conv2.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n346 {derived}: [t613 f32 [H=7 W=7 C=368] {derived} ->[n347]] =
        conv2d
          x=t611 {derived} <-n344
          weight=t612 {derived} <-n345
          bias=none
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=368;
                 groups=46}
      n347 {pt2=root[134] torch.ops.aten.conv2d.default (conv2d_42)}: [t614 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_42} ->[n348]] =
        permute x=t613 {derived} <-n346 perm=[H<-C, W<-H, C<-W]
    group g86 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n348 {derived}: [t615 f32 [H=7 W=7 C=368] {derived} ->[n349]] =
        permute x=t614 {pt2=root:conv2d_42} <-n347 perm=[H<-W, W<-C, C<-H]
      n349 {derived}: [t616 f32 [H=7 W=7 C=368] {derived} ->[n350]] =
        batch_norm
          x=t615 {derived} <-n348
          weight=t127 {pt2=root:p_s4_b7_conv2_bn_weight target=s4.b7.conv2.bn.weight}
          bias=t128 {pt2=root:p_s4_b7_conv2_bn_bias target=s4.b7.conv2.bn.bias}
          running_mean=t260 {pt2=root:b_s4_b7_conv2_bn_running_mean target=s4.b7.conv2.bn.running_mean}
          running_var=t261 {pt2=root:b_s4_b7_conv2_bn_running_var target=s4.b7.conv2.bn.running_var}
          params={channel=C; eps=1e-05}
      n350 {pt2=root[135] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_42)}: [t617 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_126} ->[n351]] =
        permute x=t616 {derived} <-n349 perm=[H<-C, W<-H, C<-W]
    n351 {pt2=root[136] torch.ops.aten.relu.default (relu_38)}: [t618 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_38} ->[n352]] =
      relu x=t617 {pt2=root:getitem_126} <-n350
    group g87 torch.ops.aten.conv2d.default:
      n352 {derived}: [t619 f32 [H=7 W=7 C=368] {derived} ->[n354]] =
        permute x=t618 {pt2=root:relu_38} <-n351 perm=[H<-W, W<-C, C<-H]
      n353 {derived}: [t620 f32 [N=368 T=1 D=1 H=1 W=1 C=368] {derived} ->[n354]] =
        permute
          x=t129 {pt2=root:p_s4_b7_conv3_conv_weight target=s4.b7.conv3.conv.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n354 {derived}: [t621 f32 [H=7 W=7 C=368] {derived} ->[n355]] =
        conv2d
          x=t619 {derived} <-n352
          weight=t620 {derived} <-n353
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=368;
                 groups=1}
      n355 {pt2=root[137] torch.ops.aten.conv2d.default (conv2d_43)}: [t622 f32 [H=368
                                                                      W=7 C=7] {pt2=root:conv2d_43} ->[n356]] =
        permute x=t621 {derived} <-n354 perm=[H<-C, W<-H, C<-W]
    group g88 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n356 {derived}: [t623 f32 [H=7 W=7 C=368] {derived} ->[n357]] =
        permute x=t622 {pt2=root:conv2d_43} <-n355 perm=[H<-W, W<-C, C<-H]
      n357 {derived}: [t624 f32 [H=7 W=7 C=368] {derived} ->[n358]] =
        batch_norm
          x=t623 {derived} <-n356
          weight=t130 {pt2=root:p_s4_b7_conv3_bn_weight target=s4.b7.conv3.bn.weight}
          bias=t131 {pt2=root:p_s4_b7_conv3_bn_bias target=s4.b7.conv3.bn.bias}
          running_mean=t263 {pt2=root:b_s4_b7_conv3_bn_running_mean target=s4.b7.conv3.bn.running_mean}
          running_var=t264 {pt2=root:b_s4_b7_conv3_bn_running_var target=s4.b7.conv3.bn.running_var}
          params={channel=C; eps=1e-05}
      n358 {pt2=root[138] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_43)}: [t625 f32 [H=368
                                                                      W=7 C=7] {pt2=root:getitem_129} ->[n359]] =
        permute x=t624 {derived} <-n357 perm=[H<-C, W<-H, C<-W]
    n359 {pt2=root[139] torch.ops.aten.add.Tensor (add_12)}: [t626 f32 [H=368
                                                                      W=7 C=7] {pt2=root:add_12} ->[n360]] =
      add a=t625 {pt2=root:getitem_129} <-n358 b=t602 {pt2=root:relu_36} <-n335
    n360 {pt2=root[140] torch.ops.aten.relu.default (relu_39)}: [t627 f32 [H=368
                                                                      W=7 C=7] {pt2=root:relu_39} ->[n361]] =
      relu x=t626 {pt2=root:add_12} <-n359
    group g89 torch.ops.aten.adaptive_avg_pool2d.default:
      n361 {derived}: [t628 f32 [H=7 W=7 C=368] {derived} ->[n362]] =
        permute x=t627 {pt2=root:relu_39} <-n360 perm=[H<-W, W<-C, C<-H]
      n362 {derived}: [t629 f32 [C=368] {derived} ->[n363]] =
        adaptive_avg_pool2d
          x=t628 {derived} <-n361
          params={output_size={h=1; w=1}}
      n363 {pt2=root[141] torch.ops.aten.adaptive_avg_pool2d.default (adaptive_avg_pool2d)}: [t630 f32 [H=368
                                                                      W=1 C=1] {pt2=root:adaptive_avg_pool2d} ->[n364]] =
        permute x=t629 {derived} <-n362 perm=[H<-C, W<-H, C<-W]
    n364 {pt2=root[142] torch.ops.aten.view.default (view)}: [t631 f32 [C=368] {pt2=root:view} ->[n365]] =
      reshape
        x=t630 {pt2=root:adaptive_avg_pool2d} <-n363
        params={shape=[C=368]}
    n365 {pt2=root[143] torch.ops.aten.clone.default (clone)}: [t632 f32 [C=368] {pt2=root:clone} ->[n367]] =
      clone x=t631 {pt2=root:view} <-n364
    group g90 torch.ops.aten.linear.default:
      n366 {derived}: [t633 f32 [N=1000 T=1 D=1 H=1 W=1 C=368] {derived} ->[n367]] =
        permute
          x=t132 {pt2=root:p_head_fc_weight target=head.fc.weight}
          perm=[N<-W, W<-N]
      n367 {pt2=root[144] torch.ops.aten.linear.default (linear)}: [t634 f32 [C=1000] {pt2=root:linear}] =
        linear
          x=t632 {pt2=root:clone} <-n365
          weight=t633 {derived} <-n366
          bias=t133 {pt2=root:p_head_fc_bias target=head.fc.bias}
          params={in_features=368}
  outputs: [t634 f32 [C=1000] {pt2=root:linear} <-n367]
