Import ResNet-18's real exported graph as one native graph.  Unlike the former
per-node ATen bridge report, `native_graph print` uses the pure PT2 importer and
keeps exporter-facing names and captured payload targets in `Pt2_native_graph`.
`print` renders the native graph structure and every retained native-id-to-PT2
tensor/node mapping. Gated on PT2_DATA; run with `make pt2.runtest` after
`make pt2.download-cram`.

  $ ../bin/native_graph.exe print --pt2 "$PT2_DATA/resnet18/resnet18.pt2"
  native graph: inputs=123 constants=122 nodes=174 outputs=1
  PT2 provenance: tensor-origins=193 captured-targets=122 node-origins=70
  graph
  inputs:
    [t0 f32 [D=64 H=3 W=7 C=7] {pt2=root:p_conv1_weight target=conv1.weight} constant,
     t1 f32 [C=64] {pt2=root:p_bn1_weight target=bn1.weight} constant,
     t2 f32 [C=64] {pt2=root:p_bn1_bias target=bn1.bias} constant,
     t3 f32 [D=64 H=64 W=3 C=3] {pt2=root:p_layer1_0_conv1_weight target=layer1.0.conv1.weight} constant,
     t4 f32 [C=64] {pt2=root:p_layer1_0_bn1_weight target=layer1.0.bn1.weight} constant,
     t5 f32 [C=64] {pt2=root:p_layer1_0_bn1_bias target=layer1.0.bn1.bias} constant,
     t6 f32 [D=64 H=64 W=3 C=3] {pt2=root:p_layer1_0_conv2_weight target=layer1.0.conv2.weight} constant,
     t7 f32 [C=64] {pt2=root:p_layer1_0_bn2_weight target=layer1.0.bn2.weight} constant,
     t8 f32 [C=64] {pt2=root:p_layer1_0_bn2_bias target=layer1.0.bn2.bias} constant,
     t9 f32 [D=64 H=64 W=3 C=3] {pt2=root:p_layer1_1_conv1_weight target=layer1.1.conv1.weight} constant,
     t10 f32 [C=64] {pt2=root:p_layer1_1_bn1_weight target=layer1.1.bn1.weight} constant,
     t11 f32 [C=64] {pt2=root:p_layer1_1_bn1_bias target=layer1.1.bn1.bias} constant,
     t12 f32 [D=64 H=64 W=3 C=3] {pt2=root:p_layer1_1_conv2_weight target=layer1.1.conv2.weight} constant,
     t13 f32 [C=64] {pt2=root:p_layer1_1_bn2_weight target=layer1.1.bn2.weight} constant,
     t14 f32 [C=64] {pt2=root:p_layer1_1_bn2_bias target=layer1.1.bn2.bias} constant,
     t15 f32 [D=128 H=64 W=3 C=3] {pt2=root:p_layer2_0_conv1_weight target=layer2.0.conv1.weight} constant,
     t16 f32 [C=128] {pt2=root:p_layer2_0_bn1_weight target=layer2.0.bn1.weight} constant,
     t17 f32 [C=128] {pt2=root:p_layer2_0_bn1_bias target=layer2.0.bn1.bias} constant,
     t18 f32 [D=128 H=128 W=3 C=3] {pt2=root:p_layer2_0_conv2_weight target=layer2.0.conv2.weight} constant,
     t19 f32 [C=128] {pt2=root:p_layer2_0_bn2_weight target=layer2.0.bn2.weight} constant,
     t20 f32 [C=128] {pt2=root:p_layer2_0_bn2_bias target=layer2.0.bn2.bias} constant,
     t21 f32 [D=128 H=64 W=1 C=1] {pt2=root:p_layer2_0_downsample_0_weight target=layer2.0.downsample.0.weight} constant,
     t22 f32 [C=128] {pt2=root:p_layer2_0_downsample_1_weight target=layer2.0.downsample.1.weight} constant,
     t23 f32 [C=128] {pt2=root:p_layer2_0_downsample_1_bias target=layer2.0.downsample.1.bias} constant,
     t24 f32 [D=128 H=128 W=3 C=3] {pt2=root:p_layer2_1_conv1_weight target=layer2.1.conv1.weight} constant,
     t25 f32 [C=128] {pt2=root:p_layer2_1_bn1_weight target=layer2.1.bn1.weight} constant,
     t26 f32 [C=128] {pt2=root:p_layer2_1_bn1_bias target=layer2.1.bn1.bias} constant,
     t27 f32 [D=128 H=128 W=3 C=3] {pt2=root:p_layer2_1_conv2_weight target=layer2.1.conv2.weight} constant,
     t28 f32 [C=128] {pt2=root:p_layer2_1_bn2_weight target=layer2.1.bn2.weight} constant,
     t29 f32 [C=128] {pt2=root:p_layer2_1_bn2_bias target=layer2.1.bn2.bias} constant,
     t30 f32 [D=256 H=128 W=3 C=3] {pt2=root:p_layer3_0_conv1_weight target=layer3.0.conv1.weight} constant,
     t31 f32 [C=256] {pt2=root:p_layer3_0_bn1_weight target=layer3.0.bn1.weight} constant,
     t32 f32 [C=256] {pt2=root:p_layer3_0_bn1_bias target=layer3.0.bn1.bias} constant,
     t33 f32 [D=256 H=256 W=3 C=3] {pt2=root:p_layer3_0_conv2_weight target=layer3.0.conv2.weight} constant,
     t34 f32 [C=256] {pt2=root:p_layer3_0_bn2_weight target=layer3.0.bn2.weight} constant,
     t35 f32 [C=256] {pt2=root:p_layer3_0_bn2_bias target=layer3.0.bn2.bias} constant,
     t36 f32 [D=256 H=128 W=1 C=1] {pt2=root:p_layer3_0_downsample_0_weight target=layer3.0.downsample.0.weight} constant,
     t37 f32 [C=256] {pt2=root:p_layer3_0_downsample_1_weight target=layer3.0.downsample.1.weight} constant,
     t38 f32 [C=256] {pt2=root:p_layer3_0_downsample_1_bias target=layer3.0.downsample.1.bias} constant,
     t39 f32 [D=256 H=256 W=3 C=3] {pt2=root:p_layer3_1_conv1_weight target=layer3.1.conv1.weight} constant,
     t40 f32 [C=256] {pt2=root:p_layer3_1_bn1_weight target=layer3.1.bn1.weight} constant,
     t41 f32 [C=256] {pt2=root:p_layer3_1_bn1_bias target=layer3.1.bn1.bias} constant,
     t42 f32 [D=256 H=256 W=3 C=3] {pt2=root:p_layer3_1_conv2_weight target=layer3.1.conv2.weight} constant,
     t43 f32 [C=256] {pt2=root:p_layer3_1_bn2_weight target=layer3.1.bn2.weight} constant,
     t44 f32 [C=256] {pt2=root:p_layer3_1_bn2_bias target=layer3.1.bn2.bias} constant,
     t45 f32 [D=512 H=256 W=3 C=3] {pt2=root:p_layer4_0_conv1_weight target=layer4.0.conv1.weight} constant,
     t46 f32 [C=512] {pt2=root:p_layer4_0_bn1_weight target=layer4.0.bn1.weight} constant,
     t47 f32 [C=512] {pt2=root:p_layer4_0_bn1_bias target=layer4.0.bn1.bias} constant,
     t48 f32 [D=512 H=512 W=3 C=3] {pt2=root:p_layer4_0_conv2_weight target=layer4.0.conv2.weight} constant,
     t49 f32 [C=512] {pt2=root:p_layer4_0_bn2_weight target=layer4.0.bn2.weight} constant,
     t50 f32 [C=512] {pt2=root:p_layer4_0_bn2_bias target=layer4.0.bn2.bias} constant,
     t51 f32 [D=512 H=256 W=1 C=1] {pt2=root:p_layer4_0_downsample_0_weight target=layer4.0.downsample.0.weight} constant,
     t52 f32 [C=512] {pt2=root:p_layer4_0_downsample_1_weight target=layer4.0.downsample.1.weight} constant,
     t53 f32 [C=512] {pt2=root:p_layer4_0_downsample_1_bias target=layer4.0.downsample.1.bias} constant,
     t54 f32 [D=512 H=512 W=3 C=3] {pt2=root:p_layer4_1_conv1_weight target=layer4.1.conv1.weight} constant,
     t55 f32 [C=512] {pt2=root:p_layer4_1_bn1_weight target=layer4.1.bn1.weight} constant,
     t56 f32 [C=512] {pt2=root:p_layer4_1_bn1_bias target=layer4.1.bn1.bias} constant,
     t57 f32 [D=512 H=512 W=3 C=3] {pt2=root:p_layer4_1_conv2_weight target=layer4.1.conv2.weight} constant,
     t58 f32 [C=512] {pt2=root:p_layer4_1_bn2_weight target=layer4.1.bn2.weight} constant,
     t59 f32 [C=512] {pt2=root:p_layer4_1_bn2_bias target=layer4.1.bn2.bias} constant,
     t60 f32 [W=1000 C=512] {pt2=root:p_fc_weight target=fc.weight} constant,
     t61 f32 [C=1000] {pt2=root:p_fc_bias target=fc.bias} constant,
     t62 f32 [C=64] {pt2=root:b_bn1_running_mean target=bn1.running_mean} constant,
     t63 f32 [C=64] {pt2=root:b_bn1_running_var target=bn1.running_var} constant,
     t64 f32 [C=1] {pt2=root:b_bn1_num_batches_tracked target=bn1.num_batches_tracked} constant,
     t65 f32 [C=64] {pt2=root:b_layer1_0_bn1_running_mean target=layer1.0.bn1.running_mean} constant,
     t66 f32 [C=64] {pt2=root:b_layer1_0_bn1_running_var target=layer1.0.bn1.running_var} constant,
     t67 f32 [C=1] {pt2=root:b_layer1_0_bn1_num_batches_tracked target=layer1.0.bn1.num_batches_tracked} constant,
     t68 f32 [C=64] {pt2=root:b_layer1_0_bn2_running_mean target=layer1.0.bn2.running_mean} constant,
     t69 f32 [C=64] {pt2=root:b_layer1_0_bn2_running_var target=layer1.0.bn2.running_var} constant,
     t70 f32 [C=1] {pt2=root:b_layer1_0_bn2_num_batches_tracked target=layer1.0.bn2.num_batches_tracked} constant,
     t71 f32 [C=64] {pt2=root:b_layer1_1_bn1_running_mean target=layer1.1.bn1.running_mean} constant,
     t72 f32 [C=64] {pt2=root:b_layer1_1_bn1_running_var target=layer1.1.bn1.running_var} constant,
     t73 f32 [C=1] {pt2=root:b_layer1_1_bn1_num_batches_tracked target=layer1.1.bn1.num_batches_tracked} constant,
     t74 f32 [C=64] {pt2=root:b_layer1_1_bn2_running_mean target=layer1.1.bn2.running_mean} constant,
     t75 f32 [C=64] {pt2=root:b_layer1_1_bn2_running_var target=layer1.1.bn2.running_var} constant,
     t76 f32 [C=1] {pt2=root:b_layer1_1_bn2_num_batches_tracked target=layer1.1.bn2.num_batches_tracked} constant,
     t77 f32 [C=128] {pt2=root:b_layer2_0_bn1_running_mean target=layer2.0.bn1.running_mean} constant,
     t78 f32 [C=128] {pt2=root:b_layer2_0_bn1_running_var target=layer2.0.bn1.running_var} constant,
     t79 f32 [C=1] {pt2=root:b_layer2_0_bn1_num_batches_tracked target=layer2.0.bn1.num_batches_tracked} constant,
     t80 f32 [C=128] {pt2=root:b_layer2_0_bn2_running_mean target=layer2.0.bn2.running_mean} constant,
     t81 f32 [C=128] {pt2=root:b_layer2_0_bn2_running_var target=layer2.0.bn2.running_var} constant,
     t82 f32 [C=1] {pt2=root:b_layer2_0_bn2_num_batches_tracked target=layer2.0.bn2.num_batches_tracked} constant,
     t83 f32 [C=128] {pt2=root:b_layer2_0_downsample_1_running_mean target=layer2.0.downsample.1.running_mean} constant,
     t84 f32 [C=128] {pt2=root:b_layer2_0_downsample_1_running_var target=layer2.0.downsample.1.running_var} constant,
     t85 f32 [C=1] {pt2=root:b_layer2_0_downsample_1_num_batches_tracked target=layer2.0.downsample.1.num_batches_tracked} constant,
     t86 f32 [C=128] {pt2=root:b_layer2_1_bn1_running_mean target=layer2.1.bn1.running_mean} constant,
     t87 f32 [C=128] {pt2=root:b_layer2_1_bn1_running_var target=layer2.1.bn1.running_var} constant,
     t88 f32 [C=1] {pt2=root:b_layer2_1_bn1_num_batches_tracked target=layer2.1.bn1.num_batches_tracked} constant,
     t89 f32 [C=128] {pt2=root:b_layer2_1_bn2_running_mean target=layer2.1.bn2.running_mean} constant,
     t90 f32 [C=128] {pt2=root:b_layer2_1_bn2_running_var target=layer2.1.bn2.running_var} constant,
     t91 f32 [C=1] {pt2=root:b_layer2_1_bn2_num_batches_tracked target=layer2.1.bn2.num_batches_tracked} constant,
     t92 f32 [C=256] {pt2=root:b_layer3_0_bn1_running_mean target=layer3.0.bn1.running_mean} constant,
     t93 f32 [C=256] {pt2=root:b_layer3_0_bn1_running_var target=layer3.0.bn1.running_var} constant,
     t94 f32 [C=1] {pt2=root:b_layer3_0_bn1_num_batches_tracked target=layer3.0.bn1.num_batches_tracked} constant,
     t95 f32 [C=256] {pt2=root:b_layer3_0_bn2_running_mean target=layer3.0.bn2.running_mean} constant,
     t96 f32 [C=256] {pt2=root:b_layer3_0_bn2_running_var target=layer3.0.bn2.running_var} constant,
     t97 f32 [C=1] {pt2=root:b_layer3_0_bn2_num_batches_tracked target=layer3.0.bn2.num_batches_tracked} constant,
     t98 f32 [C=256] {pt2=root:b_layer3_0_downsample_1_running_mean target=layer3.0.downsample.1.running_mean} constant,
     t99 f32 [C=256] {pt2=root:b_layer3_0_downsample_1_running_var target=layer3.0.downsample.1.running_var} constant,
     t100 f32 [C=1] {pt2=root:b_layer3_0_downsample_1_num_batches_tracked target=layer3.0.downsample.1.num_batches_tracked} constant,
     t101 f32 [C=256] {pt2=root:b_layer3_1_bn1_running_mean target=layer3.1.bn1.running_mean} constant,
     t102 f32 [C=256] {pt2=root:b_layer3_1_bn1_running_var target=layer3.1.bn1.running_var} constant,
     t103 f32 [C=1] {pt2=root:b_layer3_1_bn1_num_batches_tracked target=layer3.1.bn1.num_batches_tracked} constant,
     t104 f32 [C=256] {pt2=root:b_layer3_1_bn2_running_mean target=layer3.1.bn2.running_mean} constant,
     t105 f32 [C=256] {pt2=root:b_layer3_1_bn2_running_var target=layer3.1.bn2.running_var} constant,
     t106 f32 [C=1] {pt2=root:b_layer3_1_bn2_num_batches_tracked target=layer3.1.bn2.num_batches_tracked} constant,
     t107 f32 [C=512] {pt2=root:b_layer4_0_bn1_running_mean target=layer4.0.bn1.running_mean} constant,
     t108 f32 [C=512] {pt2=root:b_layer4_0_bn1_running_var target=layer4.0.bn1.running_var} constant,
     t109 f32 [C=1] {pt2=root:b_layer4_0_bn1_num_batches_tracked target=layer4.0.bn1.num_batches_tracked} constant,
     t110 f32 [C=512] {pt2=root:b_layer4_0_bn2_running_mean target=layer4.0.bn2.running_mean} constant,
     t111 f32 [C=512] {pt2=root:b_layer4_0_bn2_running_var target=layer4.0.bn2.running_var} constant,
     t112 f32 [C=1] {pt2=root:b_layer4_0_bn2_num_batches_tracked target=layer4.0.bn2.num_batches_tracked} constant,
     t113 f32 [C=512] {pt2=root:b_layer4_0_downsample_1_running_mean target=layer4.0.downsample.1.running_mean} constant,
     t114 f32 [C=512] {pt2=root:b_layer4_0_downsample_1_running_var target=layer4.0.downsample.1.running_var} constant,
     t115 f32 [C=1] {pt2=root:b_layer4_0_downsample_1_num_batches_tracked target=layer4.0.downsample.1.num_batches_tracked} constant,
     t116 f32 [C=512] {pt2=root:b_layer4_1_bn1_running_mean target=layer4.1.bn1.running_mean} constant,
     t117 f32 [C=512] {pt2=root:b_layer4_1_bn1_running_var target=layer4.1.bn1.running_var} constant,
     t118 f32 [C=1] {pt2=root:b_layer4_1_bn1_num_batches_tracked target=layer4.1.bn1.num_batches_tracked} constant,
     t119 f32 [C=512] {pt2=root:b_layer4_1_bn2_running_mean target=layer4.1.bn2.running_mean} constant,
     t120 f32 [C=512] {pt2=root:b_layer4_1_bn2_running_var target=layer4.1.bn2.running_var} constant,
     t121 f32 [C=1] {pt2=root:b_layer4_1_bn2_num_batches_tracked target=layer4.1.bn2.num_batches_tracked} constant,
     t122 f32 [H=3 W=224 C=224] {pt2=root:x}]
  nodes:
    group g1 torch.ops.aten.convolution.default:
      n0 {derived}: [t123 f32 [H=224 W=224 C=3] {derived}] =
        permute x=t122 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
      n1 {derived}: [t124 f32 [N=64 T=1 D=1 H=7 W=7 C=3] {derived}] =
        permute
          x=t0 {pt2=root:p_conv1_weight target=conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2 {derived}: [t125 f32 [H=112 W=112 C=64] {derived}] =
        convolution
          x=t123 {derived}
          weight=t124 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=3; w=3};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n3 {pt2=root[0] torch.ops.aten.convolution.default (convolution)}: [t126 f32 [H=64
                                                                      W=112
                                                                      C=112] {pt2=root:convolution}] =
        permute x=t125 {derived} perm=[H<-C, W<-H, C<-W]
    group g2 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n4 {derived}: [t127 f32 [H=112 W=112 C=64] {derived}] =
        permute x=t126 {pt2=root:convolution} perm=[H<-W, W<-C, C<-H]
      n5 {derived}: [t128 f32 [H=112 W=112 C=64] {derived}] =
        batch_norm
          x=t127 {derived}
          weight=t1 {pt2=root:p_bn1_weight target=bn1.weight}
          bias=t2 {pt2=root:p_bn1_bias target=bn1.bias}
          running_mean=t62 {pt2=root:b_bn1_running_mean target=bn1.running_mean}
          running_var=t63 {pt2=root:b_bn1_running_var target=bn1.running_var}
          params={channel=C; eps=1e-05}
      n6 {pt2=root[1] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training)}: [t129 f32 [H=64
                                                                      W=112
                                                                      C=112] {pt2=root:getitem}] =
        permute x=t128 {derived} perm=[H<-C, W<-H, C<-W]
    n7 {pt2=root[2] torch.ops.aten.relu.default (relu)}: [t130 f32 [H=64 W=112
                                                                    C=112] {pt2=root:relu}] =
      relu x=t129 {pt2=root:getitem}
    group g3 torch.ops.aten.max_pool2d_with_indices.default:
      n8 {derived}: [t131 f32 [H=112 W=112 C=64] {derived}] =
        permute x=t130 {pt2=root:relu} perm=[H<-W, W<-C, C<-H]
      n9 {derived}: [t132 f32 [H=56 W=56 C=64] {derived},
                     t133 f32 [H=56 W=56 C=64] {derived}] =
        max_pool2d_with_indices
          x=t131 {derived}
          params={kernel={h=3; w=3}; stride={h=2; w=2}; pad={h=1; w=1}}
      n10 {derived}: [] = discard x=t133 {derived}
      n11 {pt2=root[3] torch.ops.aten.max_pool2d_with_indices.default (max_pool2d_with_indices)}: [t134 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_3}] =
        permute x=t132 {derived} perm=[H<-C, W<-H, C<-W]
    group g4 torch.ops.aten.convolution.default:
      n12 {derived}: [t135 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t134 {pt2=root:getitem_3} perm=[H<-W, W<-C, C<-H]
      n13 {derived}: [t136 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {derived}] =
        permute
          x=t3 {pt2=root:p_layer1_0_conv1_weight target=layer1.0.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n14 {derived}: [t137 f32 [H=56 W=56 C=64] {derived}] =
        convolution
          x=t135 {derived}
          weight=t136 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n15 {pt2=root[4] torch.ops.aten.convolution.default (convolution_1)}: [t138 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:convolution_1}] =
        permute x=t137 {derived} perm=[H<-C, W<-H, C<-W]
    group g5 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n16 {derived}: [t139 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t138 {pt2=root:convolution_1} perm=[H<-W, W<-C, C<-H]
      n17 {derived}: [t140 f32 [H=56 W=56 C=64] {derived}] =
        batch_norm
          x=t139 {derived}
          weight=t4 {pt2=root:p_layer1_0_bn1_weight target=layer1.0.bn1.weight}
          bias=t5 {pt2=root:p_layer1_0_bn1_bias target=layer1.0.bn1.bias}
          running_mean=t65 {pt2=root:b_layer1_0_bn1_running_mean target=layer1.0.bn1.running_mean}
          running_var=t66 {pt2=root:b_layer1_0_bn1_running_var target=layer1.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n18 {pt2=root[5] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_1)}: [t141 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_5}] =
        permute x=t140 {derived} perm=[H<-C, W<-H, C<-W]
    n19 {pt2=root[6] torch.ops.aten.relu.default (relu_1)}: [t142 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:relu_1}] =
      relu x=t141 {pt2=root:getitem_5}
    group g6 torch.ops.aten.convolution.default:
      n20 {derived}: [t143 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t142 {pt2=root:relu_1} perm=[H<-W, W<-C, C<-H]
      n21 {derived}: [t144 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {derived}] =
        permute
          x=t6 {pt2=root:p_layer1_0_conv2_weight target=layer1.0.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n22 {derived}: [t145 f32 [H=56 W=56 C=64] {derived}] =
        convolution
          x=t143 {derived}
          weight=t144 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n23 {pt2=root[7] torch.ops.aten.convolution.default (convolution_2)}: [t146 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:convolution_2}] =
        permute x=t145 {derived} perm=[H<-C, W<-H, C<-W]
    group g7 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n24 {derived}: [t147 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t146 {pt2=root:convolution_2} perm=[H<-W, W<-C, C<-H]
      n25 {derived}: [t148 f32 [H=56 W=56 C=64] {derived}] =
        batch_norm
          x=t147 {derived}
          weight=t7 {pt2=root:p_layer1_0_bn2_weight target=layer1.0.bn2.weight}
          bias=t8 {pt2=root:p_layer1_0_bn2_bias target=layer1.0.bn2.bias}
          running_mean=t68 {pt2=root:b_layer1_0_bn2_running_mean target=layer1.0.bn2.running_mean}
          running_var=t69 {pt2=root:b_layer1_0_bn2_running_var target=layer1.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n26 {pt2=root[8] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_2)}: [t149 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_8}] =
        permute x=t148 {derived} perm=[H<-C, W<-H, C<-W]
    n27 {pt2=root[9] torch.ops.aten.add.Tensor (add)}: [t150 f32 [H=64 W=56
                                                                  C=56] {pt2=root:add}] =
      add a=t149 {pt2=root:getitem_8} b=t134 {pt2=root:getitem_3}
    n28 {pt2=root[10] torch.ops.aten.relu.default (relu_2)}: [t151 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:relu_2}] =
      relu x=t150 {pt2=root:add}
    group g8 torch.ops.aten.convolution.default:
      n29 {derived}: [t152 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t151 {pt2=root:relu_2} perm=[H<-W, W<-C, C<-H]
      n30 {derived}: [t153 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {derived}] =
        permute
          x=t9 {pt2=root:p_layer1_1_conv1_weight target=layer1.1.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n31 {derived}: [t154 f32 [H=56 W=56 C=64] {derived}] =
        convolution
          x=t152 {derived}
          weight=t153 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n32 {pt2=root[11] torch.ops.aten.convolution.default (convolution_3)}: [t155 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:convolution_3}] =
        permute x=t154 {derived} perm=[H<-C, W<-H, C<-W]
    group g9 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n33 {derived}: [t156 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t155 {pt2=root:convolution_3} perm=[H<-W, W<-C, C<-H]
      n34 {derived}: [t157 f32 [H=56 W=56 C=64] {derived}] =
        batch_norm
          x=t156 {derived}
          weight=t10 {pt2=root:p_layer1_1_bn1_weight target=layer1.1.bn1.weight}
          bias=t11 {pt2=root:p_layer1_1_bn1_bias target=layer1.1.bn1.bias}
          running_mean=t71 {pt2=root:b_layer1_1_bn1_running_mean target=layer1.1.bn1.running_mean}
          running_var=t72 {pt2=root:b_layer1_1_bn1_running_var target=layer1.1.bn1.running_var}
          params={channel=C; eps=1e-05}
      n35 {pt2=root[12] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_3)}: [t158 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_11}] =
        permute x=t157 {derived} perm=[H<-C, W<-H, C<-W]
    n36 {pt2=root[13] torch.ops.aten.relu.default (relu_3)}: [t159 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:relu_3}] =
      relu x=t158 {pt2=root:getitem_11}
    group g10 torch.ops.aten.convolution.default:
      n37 {derived}: [t160 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t159 {pt2=root:relu_3} perm=[H<-W, W<-C, C<-H]
      n38 {derived}: [t161 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {derived}] =
        permute
          x=t12 {pt2=root:p_layer1_1_conv2_weight target=layer1.1.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n39 {derived}: [t162 f32 [H=56 W=56 C=64] {derived}] =
        convolution
          x=t160 {derived}
          weight=t161 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n40 {pt2=root[14] torch.ops.aten.convolution.default (convolution_4)}: [t163 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:convolution_4}] =
        permute x=t162 {derived} perm=[H<-C, W<-H, C<-W]
    group g11 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n41 {derived}: [t164 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t163 {pt2=root:convolution_4} perm=[H<-W, W<-C, C<-H]
      n42 {derived}: [t165 f32 [H=56 W=56 C=64] {derived}] =
        batch_norm
          x=t164 {derived}
          weight=t13 {pt2=root:p_layer1_1_bn2_weight target=layer1.1.bn2.weight}
          bias=t14 {pt2=root:p_layer1_1_bn2_bias target=layer1.1.bn2.bias}
          running_mean=t74 {pt2=root:b_layer1_1_bn2_running_mean target=layer1.1.bn2.running_mean}
          running_var=t75 {pt2=root:b_layer1_1_bn2_running_var target=layer1.1.bn2.running_var}
          params={channel=C; eps=1e-05}
      n43 {pt2=root[15] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_4)}: [t166 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_14}] =
        permute x=t165 {derived} perm=[H<-C, W<-H, C<-W]
    n44 {pt2=root[16] torch.ops.aten.add.Tensor (add_1)}: [t167 f32 [H=64 W=56
                                                                     C=56] {pt2=root:add_1}] =
      add a=t166 {pt2=root:getitem_14} b=t151 {pt2=root:relu_2}
    n45 {pt2=root[17] torch.ops.aten.relu.default (relu_4)}: [t168 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:relu_4}] =
      relu x=t167 {pt2=root:add_1}
    group g12 torch.ops.aten.convolution.default:
      n46 {derived}: [t169 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t168 {pt2=root:relu_4} perm=[H<-W, W<-C, C<-H]
      n47 {derived}: [t170 f32 [N=128 T=1 D=1 H=3 W=3 C=64] {derived}] =
        permute
          x=t15 {pt2=root:p_layer2_0_conv1_weight target=layer2.0.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n48 {derived}: [t171 f32 [H=28 W=28 C=128] {derived}] =
        convolution
          x=t169 {derived}
          weight=t170 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n49 {pt2=root[18] torch.ops.aten.convolution.default (convolution_5)}: [t172 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_5}] =
        permute x=t171 {derived} perm=[H<-C, W<-H, C<-W]
    group g13 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n50 {derived}: [t173 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t172 {pt2=root:convolution_5} perm=[H<-W, W<-C, C<-H]
      n51 {derived}: [t174 f32 [H=28 W=28 C=128] {derived}] =
        batch_norm
          x=t173 {derived}
          weight=t16 {pt2=root:p_layer2_0_bn1_weight target=layer2.0.bn1.weight}
          bias=t17 {pt2=root:p_layer2_0_bn1_bias target=layer2.0.bn1.bias}
          running_mean=t77 {pt2=root:b_layer2_0_bn1_running_mean target=layer2.0.bn1.running_mean}
          running_var=t78 {pt2=root:b_layer2_0_bn1_running_var target=layer2.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n52 {pt2=root[19] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_5)}: [t175 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_17}] =
        permute x=t174 {derived} perm=[H<-C, W<-H, C<-W]
    n53 {pt2=root[20] torch.ops.aten.relu.default (relu_5)}: [t176 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:relu_5}] =
      relu x=t175 {pt2=root:getitem_17}
    group g14 torch.ops.aten.convolution.default:
      n54 {derived}: [t177 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t176 {pt2=root:relu_5} perm=[H<-W, W<-C, C<-H]
      n55 {derived}: [t178 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {derived}] =
        permute
          x=t18 {pt2=root:p_layer2_0_conv2_weight target=layer2.0.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n56 {derived}: [t179 f32 [H=28 W=28 C=128] {derived}] =
        convolution
          x=t177 {derived}
          weight=t178 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n57 {pt2=root[21] torch.ops.aten.convolution.default (convolution_6)}: [t180 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_6}] =
        permute x=t179 {derived} perm=[H<-C, W<-H, C<-W]
    group g15 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n58 {derived}: [t181 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t180 {pt2=root:convolution_6} perm=[H<-W, W<-C, C<-H]
      n59 {derived}: [t182 f32 [H=28 W=28 C=128] {derived}] =
        batch_norm
          x=t181 {derived}
          weight=t19 {pt2=root:p_layer2_0_bn2_weight target=layer2.0.bn2.weight}
          bias=t20 {pt2=root:p_layer2_0_bn2_bias target=layer2.0.bn2.bias}
          running_mean=t80 {pt2=root:b_layer2_0_bn2_running_mean target=layer2.0.bn2.running_mean}
          running_var=t81 {pt2=root:b_layer2_0_bn2_running_var target=layer2.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n60 {pt2=root[22] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_6)}: [t183 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_20}] =
        permute x=t182 {derived} perm=[H<-C, W<-H, C<-W]
    group g16 torch.ops.aten.convolution.default:
      n61 {derived}: [t184 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t168 {pt2=root:relu_4} perm=[H<-W, W<-C, C<-H]
      n62 {derived}: [t185 f32 [N=128 T=1 D=1 H=1 W=1 C=64] {derived}] =
        permute
          x=t21 {pt2=root:p_layer2_0_downsample_0_weight target=layer2.0.downsample.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n63 {derived}: [t186 f32 [H=28 W=28 C=128] {derived}] =
        convolution
          x=t184 {derived}
          weight=t185 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n64 {pt2=root[23] torch.ops.aten.convolution.default (convolution_7)}: [t187 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_7}] =
        permute x=t186 {derived} perm=[H<-C, W<-H, C<-W]
    group g17 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n65 {derived}: [t188 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t187 {pt2=root:convolution_7} perm=[H<-W, W<-C, C<-H]
      n66 {derived}: [t189 f32 [H=28 W=28 C=128] {derived}] =
        batch_norm
          x=t188 {derived}
          weight=t22 {pt2=root:p_layer2_0_downsample_1_weight target=layer2.0.downsample.1.weight}
          bias=t23 {pt2=root:p_layer2_0_downsample_1_bias target=layer2.0.downsample.1.bias}
          running_mean=t83 {pt2=root:b_layer2_0_downsample_1_running_mean target=layer2.0.downsample.1.running_mean}
          running_var=t84 {pt2=root:b_layer2_0_downsample_1_running_var target=layer2.0.downsample.1.running_var}
          params={channel=C; eps=1e-05}
      n67 {pt2=root[24] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_7)}: [t190 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_23}] =
        permute x=t189 {derived} perm=[H<-C, W<-H, C<-W]
    n68 {pt2=root[25] torch.ops.aten.add.Tensor (add_2)}: [t191 f32 [H=128 W=28
                                                                     C=28] {pt2=root:add_2}] =
      add a=t183 {pt2=root:getitem_20} b=t190 {pt2=root:getitem_23}
    n69 {pt2=root[26] torch.ops.aten.relu.default (relu_6)}: [t192 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:relu_6}] =
      relu x=t191 {pt2=root:add_2}
    group g18 torch.ops.aten.convolution.default:
      n70 {derived}: [t193 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t192 {pt2=root:relu_6} perm=[H<-W, W<-C, C<-H]
      n71 {derived}: [t194 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {derived}] =
        permute
          x=t24 {pt2=root:p_layer2_1_conv1_weight target=layer2.1.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n72 {derived}: [t195 f32 [H=28 W=28 C=128] {derived}] =
        convolution
          x=t193 {derived}
          weight=t194 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n73 {pt2=root[27] torch.ops.aten.convolution.default (convolution_8)}: [t196 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_8}] =
        permute x=t195 {derived} perm=[H<-C, W<-H, C<-W]
    group g19 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n74 {derived}: [t197 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t196 {pt2=root:convolution_8} perm=[H<-W, W<-C, C<-H]
      n75 {derived}: [t198 f32 [H=28 W=28 C=128] {derived}] =
        batch_norm
          x=t197 {derived}
          weight=t25 {pt2=root:p_layer2_1_bn1_weight target=layer2.1.bn1.weight}
          bias=t26 {pt2=root:p_layer2_1_bn1_bias target=layer2.1.bn1.bias}
          running_mean=t86 {pt2=root:b_layer2_1_bn1_running_mean target=layer2.1.bn1.running_mean}
          running_var=t87 {pt2=root:b_layer2_1_bn1_running_var target=layer2.1.bn1.running_var}
          params={channel=C; eps=1e-05}
      n76 {pt2=root[28] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_8)}: [t199 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_26}] =
        permute x=t198 {derived} perm=[H<-C, W<-H, C<-W]
    n77 {pt2=root[29] torch.ops.aten.relu.default (relu_7)}: [t200 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:relu_7}] =
      relu x=t199 {pt2=root:getitem_26}
    group g20 torch.ops.aten.convolution.default:
      n78 {derived}: [t201 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t200 {pt2=root:relu_7} perm=[H<-W, W<-C, C<-H]
      n79 {derived}: [t202 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {derived}] =
        permute
          x=t27 {pt2=root:p_layer2_1_conv2_weight target=layer2.1.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n80 {derived}: [t203 f32 [H=28 W=28 C=128] {derived}] =
        convolution
          x=t201 {derived}
          weight=t202 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n81 {pt2=root[30] torch.ops.aten.convolution.default (convolution_9)}: [t204 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_9}] =
        permute x=t203 {derived} perm=[H<-C, W<-H, C<-W]
    group g21 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n82 {derived}: [t205 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t204 {pt2=root:convolution_9} perm=[H<-W, W<-C, C<-H]
      n83 {derived}: [t206 f32 [H=28 W=28 C=128] {derived}] =
        batch_norm
          x=t205 {derived}
          weight=t28 {pt2=root:p_layer2_1_bn2_weight target=layer2.1.bn2.weight}
          bias=t29 {pt2=root:p_layer2_1_bn2_bias target=layer2.1.bn2.bias}
          running_mean=t89 {pt2=root:b_layer2_1_bn2_running_mean target=layer2.1.bn2.running_mean}
          running_var=t90 {pt2=root:b_layer2_1_bn2_running_var target=layer2.1.bn2.running_var}
          params={channel=C; eps=1e-05}
      n84 {pt2=root[31] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_9)}: [t207 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_29}] =
        permute x=t206 {derived} perm=[H<-C, W<-H, C<-W]
    n85 {pt2=root[32] torch.ops.aten.add.Tensor (add_3)}: [t208 f32 [H=128 W=28
                                                                     C=28] {pt2=root:add_3}] =
      add a=t207 {pt2=root:getitem_29} b=t192 {pt2=root:relu_6}
    n86 {pt2=root[33] torch.ops.aten.relu.default (relu_8)}: [t209 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:relu_8}] =
      relu x=t208 {pt2=root:add_3}
    group g22 torch.ops.aten.convolution.default:
      n87 {derived}: [t210 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t209 {pt2=root:relu_8} perm=[H<-W, W<-C, C<-H]
      n88 {derived}: [t211 f32 [N=256 T=1 D=1 H=3 W=3 C=128] {derived}] =
        permute
          x=t30 {pt2=root:p_layer3_0_conv1_weight target=layer3.0.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n89 {derived}: [t212 f32 [H=14 W=14 C=256] {derived}] =
        convolution
          x=t210 {derived}
          weight=t211 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n90 {pt2=root[34] torch.ops.aten.convolution.default (convolution_10)}: [t213 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_10}] =
        permute x=t212 {derived} perm=[H<-C, W<-H, C<-W]
    group g23 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n91 {derived}: [t214 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t213 {pt2=root:convolution_10} perm=[H<-W, W<-C, C<-H]
      n92 {derived}: [t215 f32 [H=14 W=14 C=256] {derived}] =
        batch_norm
          x=t214 {derived}
          weight=t31 {pt2=root:p_layer3_0_bn1_weight target=layer3.0.bn1.weight}
          bias=t32 {pt2=root:p_layer3_0_bn1_bias target=layer3.0.bn1.bias}
          running_mean=t92 {pt2=root:b_layer3_0_bn1_running_mean target=layer3.0.bn1.running_mean}
          running_var=t93 {pt2=root:b_layer3_0_bn1_running_var target=layer3.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n93 {pt2=root[35] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_10)}: [t216 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_32}] =
        permute x=t215 {derived} perm=[H<-C, W<-H, C<-W]
    n94 {pt2=root[36] torch.ops.aten.relu.default (relu_9)}: [t217 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:relu_9}] =
      relu x=t216 {pt2=root:getitem_32}
    group g24 torch.ops.aten.convolution.default:
      n95 {derived}: [t218 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t217 {pt2=root:relu_9} perm=[H<-W, W<-C, C<-H]
      n96 {derived}: [t219 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {derived}] =
        permute
          x=t33 {pt2=root:p_layer3_0_conv2_weight target=layer3.0.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n97 {derived}: [t220 f32 [H=14 W=14 C=256] {derived}] =
        convolution
          x=t218 {derived}
          weight=t219 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n98 {pt2=root[37] torch.ops.aten.convolution.default (convolution_11)}: [t221 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_11}] =
        permute x=t220 {derived} perm=[H<-C, W<-H, C<-W]
    group g25 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n99 {derived}: [t222 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t221 {pt2=root:convolution_11} perm=[H<-W, W<-C, C<-H]
      n100 {derived}: [t223 f32 [H=14 W=14 C=256] {derived}] =
        batch_norm
          x=t222 {derived}
          weight=t34 {pt2=root:p_layer3_0_bn2_weight target=layer3.0.bn2.weight}
          bias=t35 {pt2=root:p_layer3_0_bn2_bias target=layer3.0.bn2.bias}
          running_mean=t95 {pt2=root:b_layer3_0_bn2_running_mean target=layer3.0.bn2.running_mean}
          running_var=t96 {pt2=root:b_layer3_0_bn2_running_var target=layer3.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n101 {pt2=root[38] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_11)}: [t224 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_35}] =
        permute x=t223 {derived} perm=[H<-C, W<-H, C<-W]
    group g26 torch.ops.aten.convolution.default:
      n102 {derived}: [t225 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t209 {pt2=root:relu_8} perm=[H<-W, W<-C, C<-H]
      n103 {derived}: [t226 f32 [N=256 T=1 D=1 H=1 W=1 C=128] {derived}] =
        permute
          x=t36 {pt2=root:p_layer3_0_downsample_0_weight target=layer3.0.downsample.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n104 {derived}: [t227 f32 [H=14 W=14 C=256] {derived}] =
        convolution
          x=t225 {derived}
          weight=t226 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n105 {pt2=root[39] torch.ops.aten.convolution.default (convolution_12)}: [t228 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_12}] =
        permute x=t227 {derived} perm=[H<-C, W<-H, C<-W]
    group g27 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n106 {derived}: [t229 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t228 {pt2=root:convolution_12} perm=[H<-W, W<-C, C<-H]
      n107 {derived}: [t230 f32 [H=14 W=14 C=256] {derived}] =
        batch_norm
          x=t229 {derived}
          weight=t37 {pt2=root:p_layer3_0_downsample_1_weight target=layer3.0.downsample.1.weight}
          bias=t38 {pt2=root:p_layer3_0_downsample_1_bias target=layer3.0.downsample.1.bias}
          running_mean=t98 {pt2=root:b_layer3_0_downsample_1_running_mean target=layer3.0.downsample.1.running_mean}
          running_var=t99 {pt2=root:b_layer3_0_downsample_1_running_var target=layer3.0.downsample.1.running_var}
          params={channel=C; eps=1e-05}
      n108 {pt2=root[40] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_12)}: [t231 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_38}] =
        permute x=t230 {derived} perm=[H<-C, W<-H, C<-W]
    n109 {pt2=root[41] torch.ops.aten.add.Tensor (add_4)}: [t232 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:add_4}] =
      add a=t224 {pt2=root:getitem_35} b=t231 {pt2=root:getitem_38}
    n110 {pt2=root[42] torch.ops.aten.relu.default (relu_10)}: [t233 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:relu_10}] =
      relu x=t232 {pt2=root:add_4}
    group g28 torch.ops.aten.convolution.default:
      n111 {derived}: [t234 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t233 {pt2=root:relu_10} perm=[H<-W, W<-C, C<-H]
      n112 {derived}: [t235 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {derived}] =
        permute
          x=t39 {pt2=root:p_layer3_1_conv1_weight target=layer3.1.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n113 {derived}: [t236 f32 [H=14 W=14 C=256] {derived}] =
        convolution
          x=t234 {derived}
          weight=t235 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n114 {pt2=root[43] torch.ops.aten.convolution.default (convolution_13)}: [t237 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_13}] =
        permute x=t236 {derived} perm=[H<-C, W<-H, C<-W]
    group g29 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n115 {derived}: [t238 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t237 {pt2=root:convolution_13} perm=[H<-W, W<-C, C<-H]
      n116 {derived}: [t239 f32 [H=14 W=14 C=256] {derived}] =
        batch_norm
          x=t238 {derived}
          weight=t40 {pt2=root:p_layer3_1_bn1_weight target=layer3.1.bn1.weight}
          bias=t41 {pt2=root:p_layer3_1_bn1_bias target=layer3.1.bn1.bias}
          running_mean=t101 {pt2=root:b_layer3_1_bn1_running_mean target=layer3.1.bn1.running_mean}
          running_var=t102 {pt2=root:b_layer3_1_bn1_running_var target=layer3.1.bn1.running_var}
          params={channel=C; eps=1e-05}
      n117 {pt2=root[44] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_13)}: [t240 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_41}] =
        permute x=t239 {derived} perm=[H<-C, W<-H, C<-W]
    n118 {pt2=root[45] torch.ops.aten.relu.default (relu_11)}: [t241 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:relu_11}] =
      relu x=t240 {pt2=root:getitem_41}
    group g30 torch.ops.aten.convolution.default:
      n119 {derived}: [t242 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t241 {pt2=root:relu_11} perm=[H<-W, W<-C, C<-H]
      n120 {derived}: [t243 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {derived}] =
        permute
          x=t42 {pt2=root:p_layer3_1_conv2_weight target=layer3.1.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n121 {derived}: [t244 f32 [H=14 W=14 C=256] {derived}] =
        convolution
          x=t242 {derived}
          weight=t243 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n122 {pt2=root[46] torch.ops.aten.convolution.default (convolution_14)}: [t245 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_14}] =
        permute x=t244 {derived} perm=[H<-C, W<-H, C<-W]
    group g31 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n123 {derived}: [t246 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t245 {pt2=root:convolution_14} perm=[H<-W, W<-C, C<-H]
      n124 {derived}: [t247 f32 [H=14 W=14 C=256] {derived}] =
        batch_norm
          x=t246 {derived}
          weight=t43 {pt2=root:p_layer3_1_bn2_weight target=layer3.1.bn2.weight}
          bias=t44 {pt2=root:p_layer3_1_bn2_bias target=layer3.1.bn2.bias}
          running_mean=t104 {pt2=root:b_layer3_1_bn2_running_mean target=layer3.1.bn2.running_mean}
          running_var=t105 {pt2=root:b_layer3_1_bn2_running_var target=layer3.1.bn2.running_var}
          params={channel=C; eps=1e-05}
      n125 {pt2=root[47] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_14)}: [t248 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_44}] =
        permute x=t247 {derived} perm=[H<-C, W<-H, C<-W]
    n126 {pt2=root[48] torch.ops.aten.add.Tensor (add_5)}: [t249 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:add_5}] =
      add a=t248 {pt2=root:getitem_44} b=t233 {pt2=root:relu_10}
    n127 {pt2=root[49] torch.ops.aten.relu.default (relu_12)}: [t250 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:relu_12}] =
      relu x=t249 {pt2=root:add_5}
    group g32 torch.ops.aten.convolution.default:
      n128 {derived}: [t251 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t250 {pt2=root:relu_12} perm=[H<-W, W<-C, C<-H]
      n129 {derived}: [t252 f32 [N=512 T=1 D=1 H=3 W=3 C=256] {derived}] =
        permute
          x=t45 {pt2=root:p_layer4_0_conv1_weight target=layer4.0.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n130 {derived}: [t253 f32 [H=7 W=7 C=512] {derived}] =
        convolution
          x=t251 {derived}
          weight=t252 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n131 {pt2=root[50] torch.ops.aten.convolution.default (convolution_15)}: [t254 f32 [H=512
                                                                      W=7 C=7] {pt2=root:convolution_15}] =
        permute x=t253 {derived} perm=[H<-C, W<-H, C<-W]
    group g33 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n132 {derived}: [t255 f32 [H=7 W=7 C=512] {derived}] =
        permute x=t254 {pt2=root:convolution_15} perm=[H<-W, W<-C, C<-H]
      n133 {derived}: [t256 f32 [H=7 W=7 C=512] {derived}] =
        batch_norm
          x=t255 {derived}
          weight=t46 {pt2=root:p_layer4_0_bn1_weight target=layer4.0.bn1.weight}
          bias=t47 {pt2=root:p_layer4_0_bn1_bias target=layer4.0.bn1.bias}
          running_mean=t107 {pt2=root:b_layer4_0_bn1_running_mean target=layer4.0.bn1.running_mean}
          running_var=t108 {pt2=root:b_layer4_0_bn1_running_var target=layer4.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n134 {pt2=root[51] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_15)}: [t257 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_47}] =
        permute x=t256 {derived} perm=[H<-C, W<-H, C<-W]
    n135 {pt2=root[52] torch.ops.aten.relu.default (relu_13)}: [t258 f32 [H=512
                                                                      W=7 C=7] {pt2=root:relu_13}] =
      relu x=t257 {pt2=root:getitem_47}
    group g34 torch.ops.aten.convolution.default:
      n136 {derived}: [t259 f32 [H=7 W=7 C=512] {derived}] =
        permute x=t258 {pt2=root:relu_13} perm=[H<-W, W<-C, C<-H]
      n137 {derived}: [t260 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {derived}] =
        permute
          x=t48 {pt2=root:p_layer4_0_conv2_weight target=layer4.0.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n138 {derived}: [t261 f32 [H=7 W=7 C=512] {derived}] =
        convolution
          x=t259 {derived}
          weight=t260 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n139 {pt2=root[53] torch.ops.aten.convolution.default (convolution_16)}: [t262 f32 [H=512
                                                                      W=7 C=7] {pt2=root:convolution_16}] =
        permute x=t261 {derived} perm=[H<-C, W<-H, C<-W]
    group g35 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n140 {derived}: [t263 f32 [H=7 W=7 C=512] {derived}] =
        permute x=t262 {pt2=root:convolution_16} perm=[H<-W, W<-C, C<-H]
      n141 {derived}: [t264 f32 [H=7 W=7 C=512] {derived}] =
        batch_norm
          x=t263 {derived}
          weight=t49 {pt2=root:p_layer4_0_bn2_weight target=layer4.0.bn2.weight}
          bias=t50 {pt2=root:p_layer4_0_bn2_bias target=layer4.0.bn2.bias}
          running_mean=t110 {pt2=root:b_layer4_0_bn2_running_mean target=layer4.0.bn2.running_mean}
          running_var=t111 {pt2=root:b_layer4_0_bn2_running_var target=layer4.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n142 {pt2=root[54] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_16)}: [t265 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_50}] =
        permute x=t264 {derived} perm=[H<-C, W<-H, C<-W]
    group g36 torch.ops.aten.convolution.default:
      n143 {derived}: [t266 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t250 {pt2=root:relu_12} perm=[H<-W, W<-C, C<-H]
      n144 {derived}: [t267 f32 [N=512 T=1 D=1 H=1 W=1 C=256] {derived}] =
        permute
          x=t51 {pt2=root:p_layer4_0_downsample_0_weight target=layer4.0.downsample.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n145 {derived}: [t268 f32 [H=7 W=7 C=512] {derived}] =
        convolution
          x=t266 {derived}
          weight=t267 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n146 {pt2=root[55] torch.ops.aten.convolution.default (convolution_17)}: [t269 f32 [H=512
                                                                      W=7 C=7] {pt2=root:convolution_17}] =
        permute x=t268 {derived} perm=[H<-C, W<-H, C<-W]
    group g37 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n147 {derived}: [t270 f32 [H=7 W=7 C=512] {derived}] =
        permute x=t269 {pt2=root:convolution_17} perm=[H<-W, W<-C, C<-H]
      n148 {derived}: [t271 f32 [H=7 W=7 C=512] {derived}] =
        batch_norm
          x=t270 {derived}
          weight=t52 {pt2=root:p_layer4_0_downsample_1_weight target=layer4.0.downsample.1.weight}
          bias=t53 {pt2=root:p_layer4_0_downsample_1_bias target=layer4.0.downsample.1.bias}
          running_mean=t113 {pt2=root:b_layer4_0_downsample_1_running_mean target=layer4.0.downsample.1.running_mean}
          running_var=t114 {pt2=root:b_layer4_0_downsample_1_running_var target=layer4.0.downsample.1.running_var}
          params={channel=C; eps=1e-05}
      n149 {pt2=root[56] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_17)}: [t272 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_53}] =
        permute x=t271 {derived} perm=[H<-C, W<-H, C<-W]
    n150 {pt2=root[57] torch.ops.aten.add.Tensor (add_6)}: [t273 f32 [H=512 W=7
                                                                      C=7] {pt2=root:add_6}] =
      add a=t265 {pt2=root:getitem_50} b=t272 {pt2=root:getitem_53}
    n151 {pt2=root[58] torch.ops.aten.relu.default (relu_14)}: [t274 f32 [H=512
                                                                      W=7 C=7] {pt2=root:relu_14}] =
      relu x=t273 {pt2=root:add_6}
    group g38 torch.ops.aten.convolution.default:
      n152 {derived}: [t275 f32 [H=7 W=7 C=512] {derived}] =
        permute x=t274 {pt2=root:relu_14} perm=[H<-W, W<-C, C<-H]
      n153 {derived}: [t276 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {derived}] =
        permute
          x=t54 {pt2=root:p_layer4_1_conv1_weight target=layer4.1.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n154 {derived}: [t277 f32 [H=7 W=7 C=512] {derived}] =
        convolution
          x=t275 {derived}
          weight=t276 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n155 {pt2=root[59] torch.ops.aten.convolution.default (convolution_18)}: [t278 f32 [H=512
                                                                      W=7 C=7] {pt2=root:convolution_18}] =
        permute x=t277 {derived} perm=[H<-C, W<-H, C<-W]
    group g39 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n156 {derived}: [t279 f32 [H=7 W=7 C=512] {derived}] =
        permute x=t278 {pt2=root:convolution_18} perm=[H<-W, W<-C, C<-H]
      n157 {derived}: [t280 f32 [H=7 W=7 C=512] {derived}] =
        batch_norm
          x=t279 {derived}
          weight=t55 {pt2=root:p_layer4_1_bn1_weight target=layer4.1.bn1.weight}
          bias=t56 {pt2=root:p_layer4_1_bn1_bias target=layer4.1.bn1.bias}
          running_mean=t116 {pt2=root:b_layer4_1_bn1_running_mean target=layer4.1.bn1.running_mean}
          running_var=t117 {pt2=root:b_layer4_1_bn1_running_var target=layer4.1.bn1.running_var}
          params={channel=C; eps=1e-05}
      n158 {pt2=root[60] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_18)}: [t281 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_56}] =
        permute x=t280 {derived} perm=[H<-C, W<-H, C<-W]
    n159 {pt2=root[61] torch.ops.aten.relu.default (relu_15)}: [t282 f32 [H=512
                                                                      W=7 C=7] {pt2=root:relu_15}] =
      relu x=t281 {pt2=root:getitem_56}
    group g40 torch.ops.aten.convolution.default:
      n160 {derived}: [t283 f32 [H=7 W=7 C=512] {derived}] =
        permute x=t282 {pt2=root:relu_15} perm=[H<-W, W<-C, C<-H]
      n161 {derived}: [t284 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {derived}] =
        permute
          x=t57 {pt2=root:p_layer4_1_conv2_weight target=layer4.1.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n162 {derived}: [t285 f32 [H=7 W=7 C=512] {derived}] =
        convolution
          x=t283 {derived}
          weight=t284 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n163 {pt2=root[62] torch.ops.aten.convolution.default (convolution_19)}: [t286 f32 [H=512
                                                                      W=7 C=7] {pt2=root:convolution_19}] =
        permute x=t285 {derived} perm=[H<-C, W<-H, C<-W]
    group g41 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n164 {derived}: [t287 f32 [H=7 W=7 C=512] {derived}] =
        permute x=t286 {pt2=root:convolution_19} perm=[H<-W, W<-C, C<-H]
      n165 {derived}: [t288 f32 [H=7 W=7 C=512] {derived}] =
        batch_norm
          x=t287 {derived}
          weight=t58 {pt2=root:p_layer4_1_bn2_weight target=layer4.1.bn2.weight}
          bias=t59 {pt2=root:p_layer4_1_bn2_bias target=layer4.1.bn2.bias}
          running_mean=t119 {pt2=root:b_layer4_1_bn2_running_mean target=layer4.1.bn2.running_mean}
          running_var=t120 {pt2=root:b_layer4_1_bn2_running_var target=layer4.1.bn2.running_var}
          params={channel=C; eps=1e-05}
      n166 {pt2=root[63] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_19)}: [t289 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_59}] =
        permute x=t288 {derived} perm=[H<-C, W<-H, C<-W]
    n167 {pt2=root[64] torch.ops.aten.add.Tensor (add_7)}: [t290 f32 [H=512 W=7
                                                                      C=7] {pt2=root:add_7}] =
      add a=t289 {pt2=root:getitem_59} b=t274 {pt2=root:relu_14}
    n168 {pt2=root[65] torch.ops.aten.relu.default (relu_16)}: [t291 f32 [H=512
                                                                      W=7 C=7] {pt2=root:relu_16}] =
      relu x=t290 {pt2=root:add_7}
    n169 {pt2=root[66] torch.ops.aten.mean.dim (mean)}: [t292 f32 [H=512 W=1
                                                                   C=1] {pt2=root:mean}] =
      mean x=t291 {pt2=root:relu_16} params={dims=[C, W]; keepdim=true}
    n170 {pt2=root[67] torch.ops.aten.view.default (view)}: [t293 f32 [C=512] {pt2=root:view}] =
      reshape x=t292 {pt2=root:mean} params={shape=[C=512]}
    n171 {pt2=root[68] torch.ops.aten.permute.default (permute)}: [t294 f32 [W=512
                                                                      C=1000] {pt2=root:permute}] =
      permute x=t60 {pt2=root:p_fc_weight target=fc.weight} perm=[W<-C, C<-W]
    group g42 torch.ops.aten.addmm.default:
      n172 {derived}: [t295 f32 [N=1000 T=1 D=1 H=1 W=1 C=512] {derived}] =
        permute x=t294 {pt2=root:permute} perm=[N<-C, W<-N, C<-W]
      n173 {pt2=root[69] torch.ops.aten.addmm.default (addmm)}: [t296 f32 [C=1000] {pt2=root:addmm}] =
        linear
          x=t293 {pt2=root:view}
          weight=t295 {derived}
          bias=t61 {pt2=root:p_fc_bias target=fc.bias}
          params={in_features=512}
  outputs: [t296 f32 [C=1000] {pt2=root:addmm}]

MobileNet-v3-small's import, printed in full as ResNet-18's is above. The
importer arms this exercises that ResNet-18 does not are the ones for
compile-time scalars, clamp and clone; reading the graph is also how the
hardswish blocks (x * clamp(x + 3, 0, 6) / 6) can be seen to keep their PT2
provenance node for node, including the +3 and /6 that arrive as bare `as_int`
arguments in Tensor slots and become [add_scalar] / [div_scalar] parameters
rather than edges needing a payload.

  $ ../bin/native_graph.exe print --pt2 "$PT2_DATA/mobilenet_v3_small/mobilenet_v3_small.pt2"
  native graph: inputs=245 constants=244 nodes=488 outputs=1
  PT2 provenance: tensor-origins=507 captured-targets=244 node-origins=262
  graph
  inputs:
    [t0 f32 [D=16 H=3 W=3 C=3] {pt2=root:p_features_0_0_weight target=features.0.0.weight} constant,
     t1 f32 [C=16] {pt2=root:p_features_0_1_weight target=features.0.1.weight} constant,
     t2 f32 [C=16] {pt2=root:p_features_0_1_bias target=features.0.1.bias} constant,
     t3 f32 [D=16 H=1 W=3 C=3] {pt2=root:p_features_1_block_0_0_weight target=features.1.block.0.0.weight} constant,
     t4 f32 [C=16] {pt2=root:p_features_1_block_0_1_weight target=features.1.block.0.1.weight} constant,
     t5 f32 [C=16] {pt2=root:p_features_1_block_0_1_bias target=features.1.block.0.1.bias} constant,
     t6 f32 [D=8 H=16 W=1 C=1] {pt2=root:p_features_1_block_1_fc1_weight target=features.1.block.1.fc1.weight} constant,
     t7 f32 [C=8] {pt2=root:p_features_1_block_1_fc1_bias target=features.1.block.1.fc1.bias} constant,
     t8 f32 [D=16 H=8 W=1 C=1] {pt2=root:p_features_1_block_1_fc2_weight target=features.1.block.1.fc2.weight} constant,
     t9 f32 [C=16] {pt2=root:p_features_1_block_1_fc2_bias target=features.1.block.1.fc2.bias} constant,
     t10 f32 [D=16 H=16 W=1 C=1] {pt2=root:p_features_1_block_2_0_weight target=features.1.block.2.0.weight} constant,
     t11 f32 [C=16] {pt2=root:p_features_1_block_2_1_weight target=features.1.block.2.1.weight} constant,
     t12 f32 [C=16] {pt2=root:p_features_1_block_2_1_bias target=features.1.block.2.1.bias} constant,
     t13 f32 [D=72 H=16 W=1 C=1] {pt2=root:p_features_2_block_0_0_weight target=features.2.block.0.0.weight} constant,
     t14 f32 [C=72] {pt2=root:p_features_2_block_0_1_weight target=features.2.block.0.1.weight} constant,
     t15 f32 [C=72] {pt2=root:p_features_2_block_0_1_bias target=features.2.block.0.1.bias} constant,
     t16 f32 [D=72 H=1 W=3 C=3] {pt2=root:p_features_2_block_1_0_weight target=features.2.block.1.0.weight} constant,
     t17 f32 [C=72] {pt2=root:p_features_2_block_1_1_weight target=features.2.block.1.1.weight} constant,
     t18 f32 [C=72] {pt2=root:p_features_2_block_1_1_bias target=features.2.block.1.1.bias} constant,
     t19 f32 [D=24 H=72 W=1 C=1] {pt2=root:p_features_2_block_2_0_weight target=features.2.block.2.0.weight} constant,
     t20 f32 [C=24] {pt2=root:p_features_2_block_2_1_weight target=features.2.block.2.1.weight} constant,
     t21 f32 [C=24] {pt2=root:p_features_2_block_2_1_bias target=features.2.block.2.1.bias} constant,
     t22 f32 [D=88 H=24 W=1 C=1] {pt2=root:p_features_3_block_0_0_weight target=features.3.block.0.0.weight} constant,
     t23 f32 [C=88] {pt2=root:p_features_3_block_0_1_weight target=features.3.block.0.1.weight} constant,
     t24 f32 [C=88] {pt2=root:p_features_3_block_0_1_bias target=features.3.block.0.1.bias} constant,
     t25 f32 [D=88 H=1 W=3 C=3] {pt2=root:p_features_3_block_1_0_weight target=features.3.block.1.0.weight} constant,
     t26 f32 [C=88] {pt2=root:p_features_3_block_1_1_weight target=features.3.block.1.1.weight} constant,
     t27 f32 [C=88] {pt2=root:p_features_3_block_1_1_bias target=features.3.block.1.1.bias} constant,
     t28 f32 [D=24 H=88 W=1 C=1] {pt2=root:p_features_3_block_2_0_weight target=features.3.block.2.0.weight} constant,
     t29 f32 [C=24] {pt2=root:p_features_3_block_2_1_weight target=features.3.block.2.1.weight} constant,
     t30 f32 [C=24] {pt2=root:p_features_3_block_2_1_bias target=features.3.block.2.1.bias} constant,
     t31 f32 [D=96 H=24 W=1 C=1] {pt2=root:p_features_4_block_0_0_weight target=features.4.block.0.0.weight} constant,
     t32 f32 [C=96] {pt2=root:p_features_4_block_0_1_weight target=features.4.block.0.1.weight} constant,
     t33 f32 [C=96] {pt2=root:p_features_4_block_0_1_bias target=features.4.block.0.1.bias} constant,
     t34 f32 [D=96 H=1 W=5 C=5] {pt2=root:p_features_4_block_1_0_weight target=features.4.block.1.0.weight} constant,
     t35 f32 [C=96] {pt2=root:p_features_4_block_1_1_weight target=features.4.block.1.1.weight} constant,
     t36 f32 [C=96] {pt2=root:p_features_4_block_1_1_bias target=features.4.block.1.1.bias} constant,
     t37 f32 [D=24 H=96 W=1 C=1] {pt2=root:p_features_4_block_2_fc1_weight target=features.4.block.2.fc1.weight} constant,
     t38 f32 [C=24] {pt2=root:p_features_4_block_2_fc1_bias target=features.4.block.2.fc1.bias} constant,
     t39 f32 [D=96 H=24 W=1 C=1] {pt2=root:p_features_4_block_2_fc2_weight target=features.4.block.2.fc2.weight} constant,
     t40 f32 [C=96] {pt2=root:p_features_4_block_2_fc2_bias target=features.4.block.2.fc2.bias} constant,
     t41 f32 [D=40 H=96 W=1 C=1] {pt2=root:p_features_4_block_3_0_weight target=features.4.block.3.0.weight} constant,
     t42 f32 [C=40] {pt2=root:p_features_4_block_3_1_weight target=features.4.block.3.1.weight} constant,
     t43 f32 [C=40] {pt2=root:p_features_4_block_3_1_bias target=features.4.block.3.1.bias} constant,
     t44 f32 [D=240 H=40 W=1 C=1] {pt2=root:p_features_5_block_0_0_weight target=features.5.block.0.0.weight} constant,
     t45 f32 [C=240] {pt2=root:p_features_5_block_0_1_weight target=features.5.block.0.1.weight} constant,
     t46 f32 [C=240] {pt2=root:p_features_5_block_0_1_bias target=features.5.block.0.1.bias} constant,
     t47 f32 [D=240 H=1 W=5 C=5] {pt2=root:p_features_5_block_1_0_weight target=features.5.block.1.0.weight} constant,
     t48 f32 [C=240] {pt2=root:p_features_5_block_1_1_weight target=features.5.block.1.1.weight} constant,
     t49 f32 [C=240] {pt2=root:p_features_5_block_1_1_bias target=features.5.block.1.1.bias} constant,
     t50 f32 [D=64 H=240 W=1 C=1] {pt2=root:p_features_5_block_2_fc1_weight target=features.5.block.2.fc1.weight} constant,
     t51 f32 [C=64] {pt2=root:p_features_5_block_2_fc1_bias target=features.5.block.2.fc1.bias} constant,
     t52 f32 [D=240 H=64 W=1 C=1] {pt2=root:p_features_5_block_2_fc2_weight target=features.5.block.2.fc2.weight} constant,
     t53 f32 [C=240] {pt2=root:p_features_5_block_2_fc2_bias target=features.5.block.2.fc2.bias} constant,
     t54 f32 [D=40 H=240 W=1 C=1] {pt2=root:p_features_5_block_3_0_weight target=features.5.block.3.0.weight} constant,
     t55 f32 [C=40] {pt2=root:p_features_5_block_3_1_weight target=features.5.block.3.1.weight} constant,
     t56 f32 [C=40] {pt2=root:p_features_5_block_3_1_bias target=features.5.block.3.1.bias} constant,
     t57 f32 [D=240 H=40 W=1 C=1] {pt2=root:p_features_6_block_0_0_weight target=features.6.block.0.0.weight} constant,
     t58 f32 [C=240] {pt2=root:p_features_6_block_0_1_weight target=features.6.block.0.1.weight} constant,
     t59 f32 [C=240] {pt2=root:p_features_6_block_0_1_bias target=features.6.block.0.1.bias} constant,
     t60 f32 [D=240 H=1 W=5 C=5] {pt2=root:p_features_6_block_1_0_weight target=features.6.block.1.0.weight} constant,
     t61 f32 [C=240] {pt2=root:p_features_6_block_1_1_weight target=features.6.block.1.1.weight} constant,
     t62 f32 [C=240] {pt2=root:p_features_6_block_1_1_bias target=features.6.block.1.1.bias} constant,
     t63 f32 [D=64 H=240 W=1 C=1] {pt2=root:p_features_6_block_2_fc1_weight target=features.6.block.2.fc1.weight} constant,
     t64 f32 [C=64] {pt2=root:p_features_6_block_2_fc1_bias target=features.6.block.2.fc1.bias} constant,
     t65 f32 [D=240 H=64 W=1 C=1] {pt2=root:p_features_6_block_2_fc2_weight target=features.6.block.2.fc2.weight} constant,
     t66 f32 [C=240] {pt2=root:p_features_6_block_2_fc2_bias target=features.6.block.2.fc2.bias} constant,
     t67 f32 [D=40 H=240 W=1 C=1] {pt2=root:p_features_6_block_3_0_weight target=features.6.block.3.0.weight} constant,
     t68 f32 [C=40] {pt2=root:p_features_6_block_3_1_weight target=features.6.block.3.1.weight} constant,
     t69 f32 [C=40] {pt2=root:p_features_6_block_3_1_bias target=features.6.block.3.1.bias} constant,
     t70 f32 [D=120 H=40 W=1 C=1] {pt2=root:p_features_7_block_0_0_weight target=features.7.block.0.0.weight} constant,
     t71 f32 [C=120] {pt2=root:p_features_7_block_0_1_weight target=features.7.block.0.1.weight} constant,
     t72 f32 [C=120] {pt2=root:p_features_7_block_0_1_bias target=features.7.block.0.1.bias} constant,
     t73 f32 [D=120 H=1 W=5 C=5] {pt2=root:p_features_7_block_1_0_weight target=features.7.block.1.0.weight} constant,
     t74 f32 [C=120] {pt2=root:p_features_7_block_1_1_weight target=features.7.block.1.1.weight} constant,
     t75 f32 [C=120] {pt2=root:p_features_7_block_1_1_bias target=features.7.block.1.1.bias} constant,
     t76 f32 [D=32 H=120 W=1 C=1] {pt2=root:p_features_7_block_2_fc1_weight target=features.7.block.2.fc1.weight} constant,
     t77 f32 [C=32] {pt2=root:p_features_7_block_2_fc1_bias target=features.7.block.2.fc1.bias} constant,
     t78 f32 [D=120 H=32 W=1 C=1] {pt2=root:p_features_7_block_2_fc2_weight target=features.7.block.2.fc2.weight} constant,
     t79 f32 [C=120] {pt2=root:p_features_7_block_2_fc2_bias target=features.7.block.2.fc2.bias} constant,
     t80 f32 [D=48 H=120 W=1 C=1] {pt2=root:p_features_7_block_3_0_weight target=features.7.block.3.0.weight} constant,
     t81 f32 [C=48] {pt2=root:p_features_7_block_3_1_weight target=features.7.block.3.1.weight} constant,
     t82 f32 [C=48] {pt2=root:p_features_7_block_3_1_bias target=features.7.block.3.1.bias} constant,
     t83 f32 [D=144 H=48 W=1 C=1] {pt2=root:p_features_8_block_0_0_weight target=features.8.block.0.0.weight} constant,
     t84 f32 [C=144] {pt2=root:p_features_8_block_0_1_weight target=features.8.block.0.1.weight} constant,
     t85 f32 [C=144] {pt2=root:p_features_8_block_0_1_bias target=features.8.block.0.1.bias} constant,
     t86 f32 [D=144 H=1 W=5 C=5] {pt2=root:p_features_8_block_1_0_weight target=features.8.block.1.0.weight} constant,
     t87 f32 [C=144] {pt2=root:p_features_8_block_1_1_weight target=features.8.block.1.1.weight} constant,
     t88 f32 [C=144] {pt2=root:p_features_8_block_1_1_bias target=features.8.block.1.1.bias} constant,
     t89 f32 [D=40 H=144 W=1 C=1] {pt2=root:p_features_8_block_2_fc1_weight target=features.8.block.2.fc1.weight} constant,
     t90 f32 [C=40] {pt2=root:p_features_8_block_2_fc1_bias target=features.8.block.2.fc1.bias} constant,
     t91 f32 [D=144 H=40 W=1 C=1] {pt2=root:p_features_8_block_2_fc2_weight target=features.8.block.2.fc2.weight} constant,
     t92 f32 [C=144] {pt2=root:p_features_8_block_2_fc2_bias target=features.8.block.2.fc2.bias} constant,
     t93 f32 [D=48 H=144 W=1 C=1] {pt2=root:p_features_8_block_3_0_weight target=features.8.block.3.0.weight} constant,
     t94 f32 [C=48] {pt2=root:p_features_8_block_3_1_weight target=features.8.block.3.1.weight} constant,
     t95 f32 [C=48] {pt2=root:p_features_8_block_3_1_bias target=features.8.block.3.1.bias} constant,
     t96 f32 [D=288 H=48 W=1 C=1] {pt2=root:p_features_9_block_0_0_weight target=features.9.block.0.0.weight} constant,
     t97 f32 [C=288] {pt2=root:p_features_9_block_0_1_weight target=features.9.block.0.1.weight} constant,
     t98 f32 [C=288] {pt2=root:p_features_9_block_0_1_bias target=features.9.block.0.1.bias} constant,
     t99 f32 [D=288 H=1 W=5 C=5] {pt2=root:p_features_9_block_1_0_weight target=features.9.block.1.0.weight} constant,
     t100 f32 [C=288] {pt2=root:p_features_9_block_1_1_weight target=features.9.block.1.1.weight} constant,
     t101 f32 [C=288] {pt2=root:p_features_9_block_1_1_bias target=features.9.block.1.1.bias} constant,
     t102 f32 [D=72 H=288 W=1 C=1] {pt2=root:p_features_9_block_2_fc1_weight target=features.9.block.2.fc1.weight} constant,
     t103 f32 [C=72] {pt2=root:p_features_9_block_2_fc1_bias target=features.9.block.2.fc1.bias} constant,
     t104 f32 [D=288 H=72 W=1 C=1] {pt2=root:p_features_9_block_2_fc2_weight target=features.9.block.2.fc2.weight} constant,
     t105 f32 [C=288] {pt2=root:p_features_9_block_2_fc2_bias target=features.9.block.2.fc2.bias} constant,
     t106 f32 [D=96 H=288 W=1 C=1] {pt2=root:p_features_9_block_3_0_weight target=features.9.block.3.0.weight} constant,
     t107 f32 [C=96] {pt2=root:p_features_9_block_3_1_weight target=features.9.block.3.1.weight} constant,
     t108 f32 [C=96] {pt2=root:p_features_9_block_3_1_bias target=features.9.block.3.1.bias} constant,
     t109 f32 [D=576 H=96 W=1 C=1] {pt2=root:p_features_10_block_0_0_weight target=features.10.block.0.0.weight} constant,
     t110 f32 [C=576] {pt2=root:p_features_10_block_0_1_weight target=features.10.block.0.1.weight} constant,
     t111 f32 [C=576] {pt2=root:p_features_10_block_0_1_bias target=features.10.block.0.1.bias} constant,
     t112 f32 [D=576 H=1 W=5 C=5] {pt2=root:p_features_10_block_1_0_weight target=features.10.block.1.0.weight} constant,
     t113 f32 [C=576] {pt2=root:p_features_10_block_1_1_weight target=features.10.block.1.1.weight} constant,
     t114 f32 [C=576] {pt2=root:p_features_10_block_1_1_bias target=features.10.block.1.1.bias} constant,
     t115 f32 [D=144 H=576 W=1 C=1] {pt2=root:p_features_10_block_2_fc1_weight target=features.10.block.2.fc1.weight} constant,
     t116 f32 [C=144] {pt2=root:p_features_10_block_2_fc1_bias target=features.10.block.2.fc1.bias} constant,
     t117 f32 [D=576 H=144 W=1 C=1] {pt2=root:p_features_10_block_2_fc2_weight target=features.10.block.2.fc2.weight} constant,
     t118 f32 [C=576] {pt2=root:p_features_10_block_2_fc2_bias target=features.10.block.2.fc2.bias} constant,
     t119 f32 [D=96 H=576 W=1 C=1] {pt2=root:p_features_10_block_3_0_weight target=features.10.block.3.0.weight} constant,
     t120 f32 [C=96] {pt2=root:p_features_10_block_3_1_weight target=features.10.block.3.1.weight} constant,
     t121 f32 [C=96] {pt2=root:p_features_10_block_3_1_bias target=features.10.block.3.1.bias} constant,
     t122 f32 [D=576 H=96 W=1 C=1] {pt2=root:p_features_11_block_0_0_weight target=features.11.block.0.0.weight} constant,
     t123 f32 [C=576] {pt2=root:p_features_11_block_0_1_weight target=features.11.block.0.1.weight} constant,
     t124 f32 [C=576] {pt2=root:p_features_11_block_0_1_bias target=features.11.block.0.1.bias} constant,
     t125 f32 [D=576 H=1 W=5 C=5] {pt2=root:p_features_11_block_1_0_weight target=features.11.block.1.0.weight} constant,
     t126 f32 [C=576] {pt2=root:p_features_11_block_1_1_weight target=features.11.block.1.1.weight} constant,
     t127 f32 [C=576] {pt2=root:p_features_11_block_1_1_bias target=features.11.block.1.1.bias} constant,
     t128 f32 [D=144 H=576 W=1 C=1] {pt2=root:p_features_11_block_2_fc1_weight target=features.11.block.2.fc1.weight} constant,
     t129 f32 [C=144] {pt2=root:p_features_11_block_2_fc1_bias target=features.11.block.2.fc1.bias} constant,
     t130 f32 [D=576 H=144 W=1 C=1] {pt2=root:p_features_11_block_2_fc2_weight target=features.11.block.2.fc2.weight} constant,
     t131 f32 [C=576] {pt2=root:p_features_11_block_2_fc2_bias target=features.11.block.2.fc2.bias} constant,
     t132 f32 [D=96 H=576 W=1 C=1] {pt2=root:p_features_11_block_3_0_weight target=features.11.block.3.0.weight} constant,
     t133 f32 [C=96] {pt2=root:p_features_11_block_3_1_weight target=features.11.block.3.1.weight} constant,
     t134 f32 [C=96] {pt2=root:p_features_11_block_3_1_bias target=features.11.block.3.1.bias} constant,
     t135 f32 [D=576 H=96 W=1 C=1] {pt2=root:p_features_12_0_weight target=features.12.0.weight} constant,
     t136 f32 [C=576] {pt2=root:p_features_12_1_weight target=features.12.1.weight} constant,
     t137 f32 [C=576] {pt2=root:p_features_12_1_bias target=features.12.1.bias} constant,
     t138 f32 [W=1024 C=576] {pt2=root:p_classifier_0_weight target=classifier.0.weight} constant,
     t139 f32 [C=1024] {pt2=root:p_classifier_0_bias target=classifier.0.bias} constant,
     t140 f32 [W=1000 C=1024] {pt2=root:p_classifier_3_weight target=classifier.3.weight} constant,
     t141 f32 [C=1000] {pt2=root:p_classifier_3_bias target=classifier.3.bias} constant,
     t142 f32 [C=16] {pt2=root:b_features_0_1_running_mean target=features.0.1.running_mean} constant,
     t143 f32 [C=16] {pt2=root:b_features_0_1_running_var target=features.0.1.running_var} constant,
     t144 f32 [C=1] {pt2=root:b_features_0_1_num_batches_tracked target=features.0.1.num_batches_tracked} constant,
     t145 f32 [C=16] {pt2=root:b_features_1_block_0_1_running_mean target=features.1.block.0.1.running_mean} constant,
     t146 f32 [C=16] {pt2=root:b_features_1_block_0_1_running_var target=features.1.block.0.1.running_var} constant,
     t147 f32 [C=1] {pt2=root:b_features_1_block_0_1_num_batches_tracked target=features.1.block.0.1.num_batches_tracked} constant,
     t148 f32 [C=16] {pt2=root:b_features_1_block_2_1_running_mean target=features.1.block.2.1.running_mean} constant,
     t149 f32 [C=16] {pt2=root:b_features_1_block_2_1_running_var target=features.1.block.2.1.running_var} constant,
     t150 f32 [C=1] {pt2=root:b_features_1_block_2_1_num_batches_tracked target=features.1.block.2.1.num_batches_tracked} constant,
     t151 f32 [C=72] {pt2=root:b_features_2_block_0_1_running_mean target=features.2.block.0.1.running_mean} constant,
     t152 f32 [C=72] {pt2=root:b_features_2_block_0_1_running_var target=features.2.block.0.1.running_var} constant,
     t153 f32 [C=1] {pt2=root:b_features_2_block_0_1_num_batches_tracked target=features.2.block.0.1.num_batches_tracked} constant,
     t154 f32 [C=72] {pt2=root:b_features_2_block_1_1_running_mean target=features.2.block.1.1.running_mean} constant,
     t155 f32 [C=72] {pt2=root:b_features_2_block_1_1_running_var target=features.2.block.1.1.running_var} constant,
     t156 f32 [C=1] {pt2=root:b_features_2_block_1_1_num_batches_tracked target=features.2.block.1.1.num_batches_tracked} constant,
     t157 f32 [C=24] {pt2=root:b_features_2_block_2_1_running_mean target=features.2.block.2.1.running_mean} constant,
     t158 f32 [C=24] {pt2=root:b_features_2_block_2_1_running_var target=features.2.block.2.1.running_var} constant,
     t159 f32 [C=1] {pt2=root:b_features_2_block_2_1_num_batches_tracked target=features.2.block.2.1.num_batches_tracked} constant,
     t160 f32 [C=88] {pt2=root:b_features_3_block_0_1_running_mean target=features.3.block.0.1.running_mean} constant,
     t161 f32 [C=88] {pt2=root:b_features_3_block_0_1_running_var target=features.3.block.0.1.running_var} constant,
     t162 f32 [C=1] {pt2=root:b_features_3_block_0_1_num_batches_tracked target=features.3.block.0.1.num_batches_tracked} constant,
     t163 f32 [C=88] {pt2=root:b_features_3_block_1_1_running_mean target=features.3.block.1.1.running_mean} constant,
     t164 f32 [C=88] {pt2=root:b_features_3_block_1_1_running_var target=features.3.block.1.1.running_var} constant,
     t165 f32 [C=1] {pt2=root:b_features_3_block_1_1_num_batches_tracked target=features.3.block.1.1.num_batches_tracked} constant,
     t166 f32 [C=24] {pt2=root:b_features_3_block_2_1_running_mean target=features.3.block.2.1.running_mean} constant,
     t167 f32 [C=24] {pt2=root:b_features_3_block_2_1_running_var target=features.3.block.2.1.running_var} constant,
     t168 f32 [C=1] {pt2=root:b_features_3_block_2_1_num_batches_tracked target=features.3.block.2.1.num_batches_tracked} constant,
     t169 f32 [C=96] {pt2=root:b_features_4_block_0_1_running_mean target=features.4.block.0.1.running_mean} constant,
     t170 f32 [C=96] {pt2=root:b_features_4_block_0_1_running_var target=features.4.block.0.1.running_var} constant,
     t171 f32 [C=1] {pt2=root:b_features_4_block_0_1_num_batches_tracked target=features.4.block.0.1.num_batches_tracked} constant,
     t172 f32 [C=96] {pt2=root:b_features_4_block_1_1_running_mean target=features.4.block.1.1.running_mean} constant,
     t173 f32 [C=96] {pt2=root:b_features_4_block_1_1_running_var target=features.4.block.1.1.running_var} constant,
     t174 f32 [C=1] {pt2=root:b_features_4_block_1_1_num_batches_tracked target=features.4.block.1.1.num_batches_tracked} constant,
     t175 f32 [C=40] {pt2=root:b_features_4_block_3_1_running_mean target=features.4.block.3.1.running_mean} constant,
     t176 f32 [C=40] {pt2=root:b_features_4_block_3_1_running_var target=features.4.block.3.1.running_var} constant,
     t177 f32 [C=1] {pt2=root:b_features_4_block_3_1_num_batches_tracked target=features.4.block.3.1.num_batches_tracked} constant,
     t178 f32 [C=240] {pt2=root:b_features_5_block_0_1_running_mean target=features.5.block.0.1.running_mean} constant,
     t179 f32 [C=240] {pt2=root:b_features_5_block_0_1_running_var target=features.5.block.0.1.running_var} constant,
     t180 f32 [C=1] {pt2=root:b_features_5_block_0_1_num_batches_tracked target=features.5.block.0.1.num_batches_tracked} constant,
     t181 f32 [C=240] {pt2=root:b_features_5_block_1_1_running_mean target=features.5.block.1.1.running_mean} constant,
     t182 f32 [C=240] {pt2=root:b_features_5_block_1_1_running_var target=features.5.block.1.1.running_var} constant,
     t183 f32 [C=1] {pt2=root:b_features_5_block_1_1_num_batches_tracked target=features.5.block.1.1.num_batches_tracked} constant,
     t184 f32 [C=40] {pt2=root:b_features_5_block_3_1_running_mean target=features.5.block.3.1.running_mean} constant,
     t185 f32 [C=40] {pt2=root:b_features_5_block_3_1_running_var target=features.5.block.3.1.running_var} constant,
     t186 f32 [C=1] {pt2=root:b_features_5_block_3_1_num_batches_tracked target=features.5.block.3.1.num_batches_tracked} constant,
     t187 f32 [C=240] {pt2=root:b_features_6_block_0_1_running_mean target=features.6.block.0.1.running_mean} constant,
     t188 f32 [C=240] {pt2=root:b_features_6_block_0_1_running_var target=features.6.block.0.1.running_var} constant,
     t189 f32 [C=1] {pt2=root:b_features_6_block_0_1_num_batches_tracked target=features.6.block.0.1.num_batches_tracked} constant,
     t190 f32 [C=240] {pt2=root:b_features_6_block_1_1_running_mean target=features.6.block.1.1.running_mean} constant,
     t191 f32 [C=240] {pt2=root:b_features_6_block_1_1_running_var target=features.6.block.1.1.running_var} constant,
     t192 f32 [C=1] {pt2=root:b_features_6_block_1_1_num_batches_tracked target=features.6.block.1.1.num_batches_tracked} constant,
     t193 f32 [C=40] {pt2=root:b_features_6_block_3_1_running_mean target=features.6.block.3.1.running_mean} constant,
     t194 f32 [C=40] {pt2=root:b_features_6_block_3_1_running_var target=features.6.block.3.1.running_var} constant,
     t195 f32 [C=1] {pt2=root:b_features_6_block_3_1_num_batches_tracked target=features.6.block.3.1.num_batches_tracked} constant,
     t196 f32 [C=120] {pt2=root:b_features_7_block_0_1_running_mean target=features.7.block.0.1.running_mean} constant,
     t197 f32 [C=120] {pt2=root:b_features_7_block_0_1_running_var target=features.7.block.0.1.running_var} constant,
     t198 f32 [C=1] {pt2=root:b_features_7_block_0_1_num_batches_tracked target=features.7.block.0.1.num_batches_tracked} constant,
     t199 f32 [C=120] {pt2=root:b_features_7_block_1_1_running_mean target=features.7.block.1.1.running_mean} constant,
     t200 f32 [C=120] {pt2=root:b_features_7_block_1_1_running_var target=features.7.block.1.1.running_var} constant,
     t201 f32 [C=1] {pt2=root:b_features_7_block_1_1_num_batches_tracked target=features.7.block.1.1.num_batches_tracked} constant,
     t202 f32 [C=48] {pt2=root:b_features_7_block_3_1_running_mean target=features.7.block.3.1.running_mean} constant,
     t203 f32 [C=48] {pt2=root:b_features_7_block_3_1_running_var target=features.7.block.3.1.running_var} constant,
     t204 f32 [C=1] {pt2=root:b_features_7_block_3_1_num_batches_tracked target=features.7.block.3.1.num_batches_tracked} constant,
     t205 f32 [C=144] {pt2=root:b_features_8_block_0_1_running_mean target=features.8.block.0.1.running_mean} constant,
     t206 f32 [C=144] {pt2=root:b_features_8_block_0_1_running_var target=features.8.block.0.1.running_var} constant,
     t207 f32 [C=1] {pt2=root:b_features_8_block_0_1_num_batches_tracked target=features.8.block.0.1.num_batches_tracked} constant,
     t208 f32 [C=144] {pt2=root:b_features_8_block_1_1_running_mean target=features.8.block.1.1.running_mean} constant,
     t209 f32 [C=144] {pt2=root:b_features_8_block_1_1_running_var target=features.8.block.1.1.running_var} constant,
     t210 f32 [C=1] {pt2=root:b_features_8_block_1_1_num_batches_tracked target=features.8.block.1.1.num_batches_tracked} constant,
     t211 f32 [C=48] {pt2=root:b_features_8_block_3_1_running_mean target=features.8.block.3.1.running_mean} constant,
     t212 f32 [C=48] {pt2=root:b_features_8_block_3_1_running_var target=features.8.block.3.1.running_var} constant,
     t213 f32 [C=1] {pt2=root:b_features_8_block_3_1_num_batches_tracked target=features.8.block.3.1.num_batches_tracked} constant,
     t214 f32 [C=288] {pt2=root:b_features_9_block_0_1_running_mean target=features.9.block.0.1.running_mean} constant,
     t215 f32 [C=288] {pt2=root:b_features_9_block_0_1_running_var target=features.9.block.0.1.running_var} constant,
     t216 f32 [C=1] {pt2=root:b_features_9_block_0_1_num_batches_tracked target=features.9.block.0.1.num_batches_tracked} constant,
     t217 f32 [C=288] {pt2=root:b_features_9_block_1_1_running_mean target=features.9.block.1.1.running_mean} constant,
     t218 f32 [C=288] {pt2=root:b_features_9_block_1_1_running_var target=features.9.block.1.1.running_var} constant,
     t219 f32 [C=1] {pt2=root:b_features_9_block_1_1_num_batches_tracked target=features.9.block.1.1.num_batches_tracked} constant,
     t220 f32 [C=96] {pt2=root:b_features_9_block_3_1_running_mean target=features.9.block.3.1.running_mean} constant,
     t221 f32 [C=96] {pt2=root:b_features_9_block_3_1_running_var target=features.9.block.3.1.running_var} constant,
     t222 f32 [C=1] {pt2=root:b_features_9_block_3_1_num_batches_tracked target=features.9.block.3.1.num_batches_tracked} constant,
     t223 f32 [C=576] {pt2=root:b_features_10_block_0_1_running_mean target=features.10.block.0.1.running_mean} constant,
     t224 f32 [C=576] {pt2=root:b_features_10_block_0_1_running_var target=features.10.block.0.1.running_var} constant,
     t225 f32 [C=1] {pt2=root:b_features_10_block_0_1_num_batches_tracked target=features.10.block.0.1.num_batches_tracked} constant,
     t226 f32 [C=576] {pt2=root:b_features_10_block_1_1_running_mean target=features.10.block.1.1.running_mean} constant,
     t227 f32 [C=576] {pt2=root:b_features_10_block_1_1_running_var target=features.10.block.1.1.running_var} constant,
     t228 f32 [C=1] {pt2=root:b_features_10_block_1_1_num_batches_tracked target=features.10.block.1.1.num_batches_tracked} constant,
     t229 f32 [C=96] {pt2=root:b_features_10_block_3_1_running_mean target=features.10.block.3.1.running_mean} constant,
     t230 f32 [C=96] {pt2=root:b_features_10_block_3_1_running_var target=features.10.block.3.1.running_var} constant,
     t231 f32 [C=1] {pt2=root:b_features_10_block_3_1_num_batches_tracked target=features.10.block.3.1.num_batches_tracked} constant,
     t232 f32 [C=576] {pt2=root:b_features_11_block_0_1_running_mean target=features.11.block.0.1.running_mean} constant,
     t233 f32 [C=576] {pt2=root:b_features_11_block_0_1_running_var target=features.11.block.0.1.running_var} constant,
     t234 f32 [C=1] {pt2=root:b_features_11_block_0_1_num_batches_tracked target=features.11.block.0.1.num_batches_tracked} constant,
     t235 f32 [C=576] {pt2=root:b_features_11_block_1_1_running_mean target=features.11.block.1.1.running_mean} constant,
     t236 f32 [C=576] {pt2=root:b_features_11_block_1_1_running_var target=features.11.block.1.1.running_var} constant,
     t237 f32 [C=1] {pt2=root:b_features_11_block_1_1_num_batches_tracked target=features.11.block.1.1.num_batches_tracked} constant,
     t238 f32 [C=96] {pt2=root:b_features_11_block_3_1_running_mean target=features.11.block.3.1.running_mean} constant,
     t239 f32 [C=96] {pt2=root:b_features_11_block_3_1_running_var target=features.11.block.3.1.running_var} constant,
     t240 f32 [C=1] {pt2=root:b_features_11_block_3_1_num_batches_tracked target=features.11.block.3.1.num_batches_tracked} constant,
     t241 f32 [C=576] {pt2=root:b_features_12_1_running_mean target=features.12.1.running_mean} constant,
     t242 f32 [C=576] {pt2=root:b_features_12_1_running_var target=features.12.1.running_var} constant,
     t243 f32 [C=1] {pt2=root:b_features_12_1_num_batches_tracked target=features.12.1.num_batches_tracked} constant,
     t244 f32 [H=3 W=224 C=224] {pt2=root:x}]
  nodes:
    group g1 torch.ops.aten.convolution.default:
      n0 {derived}: [t245 f32 [H=224 W=224 C=3] {derived}] =
        permute x=t244 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
      n1 {derived}: [t246 f32 [N=16 T=1 D=1 H=3 W=3 C=3] {derived}] =
        permute
          x=t0 {pt2=root:p_features_0_0_weight target=features.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2 {derived}: [t247 f32 [H=112 W=112 C=16] {derived}] =
        convolution
          x=t245 {derived}
          weight=t246 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n3 {pt2=root[0] torch.ops.aten.convolution.default (convolution)}: [t248 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:convolution}] =
        permute x=t247 {derived} perm=[H<-C, W<-H, C<-W]
    group g2 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n4 {derived}: [t249 f32 [H=112 W=112 C=16] {derived}] =
        permute x=t248 {pt2=root:convolution} perm=[H<-W, W<-C, C<-H]
      n5 {derived}: [t250 f32 [H=112 W=112 C=16] {derived}] =
        batch_norm
          x=t249 {derived}
          weight=t1 {pt2=root:p_features_0_1_weight target=features.0.1.weight}
          bias=t2 {pt2=root:p_features_0_1_bias target=features.0.1.bias}
          running_mean=t142 {pt2=root:b_features_0_1_running_mean target=features.0.1.running_mean}
          running_var=t143 {pt2=root:b_features_0_1_running_var target=features.0.1.running_var}
          params={channel=C; eps=0.001}
      n6 {pt2=root[1] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training)}: [t251 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:getitem}] =
        permute x=t250 {derived} perm=[H<-C, W<-H, C<-W]
    n7 {pt2=root[2] torch.ops.aten.add.Tensor (add)}: [t252 f32 [H=16 W=112
                                                                 C=112] {pt2=root:add}] =
      add_scalar x=t251 {pt2=root:getitem} scalar=3
    n8 {pt2=root[3] torch.ops.aten.clamp.default (clamp)}: [t253 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:clamp}] =
      clamp x=t252 {pt2=root:add} params={min=0; max=none}
    n9 {pt2=root[4] torch.ops.aten.clamp.default (clamp_1)}: [t254 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:clamp_1}] =
      clamp x=t253 {pt2=root:clamp} params={min=none; max=6}
    n10 {pt2=root[5] torch.ops.aten.mul.Tensor (mul)}: [t255 f32 [H=16 W=112
                                                                  C=112] {pt2=root:mul}] =
      mul a=t251 {pt2=root:getitem} b=t254 {pt2=root:clamp_1}
    n11 {pt2=root[6] torch.ops.aten.div.Tensor (div)}: [t256 f32 [H=16 W=112
                                                                  C=112] {pt2=root:div}] =
      div_scalar x=t255 {pt2=root:mul} scalar=6
    group g3 torch.ops.aten.convolution.default:
      n12 {derived}: [t257 f32 [H=112 W=112 C=16] {derived}] =
        permute x=t256 {pt2=root:div} perm=[H<-W, W<-C, C<-H]
      n13 {derived}: [t258 f32 [N=16 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t3 {pt2=root:p_features_1_block_0_0_weight target=features.1.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n14 {derived}: [t259 f32 [H=56 W=56 C=16] {derived}] =
        convolution
          x=t257 {derived}
          weight=t258 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=16}
      n15 {pt2=root[7] torch.ops.aten.convolution.default (convolution_1)}: [t260 f32 [H=16
                                                                      W=56
                                                                      C=56] {pt2=root:convolution_1}] =
        permute x=t259 {derived} perm=[H<-C, W<-H, C<-W]
    group g4 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n16 {derived}: [t261 f32 [H=56 W=56 C=16] {derived}] =
        permute x=t260 {pt2=root:convolution_1} perm=[H<-W, W<-C, C<-H]
      n17 {derived}: [t262 f32 [H=56 W=56 C=16] {derived}] =
        batch_norm
          x=t261 {derived}
          weight=t4 {pt2=root:p_features_1_block_0_1_weight target=features.1.block.0.1.weight}
          bias=t5 {pt2=root:p_features_1_block_0_1_bias target=features.1.block.0.1.bias}
          running_mean=t145 {pt2=root:b_features_1_block_0_1_running_mean target=features.1.block.0.1.running_mean}
          running_var=t146 {pt2=root:b_features_1_block_0_1_running_var target=features.1.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n18 {pt2=root[8] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_1)}: [t263 f32 [H=16
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_3}] =
        permute x=t262 {derived} perm=[H<-C, W<-H, C<-W]
    n19 {pt2=root[9] torch.ops.aten.relu.default (relu)}: [t264 f32 [H=16 W=56
                                                                     C=56] {pt2=root:relu}] =
      relu x=t263 {pt2=root:getitem_3}
    n20 {pt2=root[10] torch.ops.aten.mean.dim (mean)}: [t265 f32 [H=16 W=1 C=1] {pt2=root:mean}] =
      mean x=t264 {pt2=root:relu} params={dims=[C, W]; keepdim=true}
    group g5 torch.ops.aten.convolution.default:
      n21 {derived}: [t266 f32 [C=16] {derived}] =
        permute x=t265 {pt2=root:mean} perm=[H<-W, W<-C, C<-H]
      n22 {derived}: [t267 f32 [N=8 T=1 D=1 H=1 W=1 C=16] {derived}] =
        permute
          x=t6 {pt2=root:p_features_1_block_1_fc1_weight target=features.1.block.1.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n23 {derived}: [t268 f32 [C=8] {derived}] =
        convolution
          x=t266 {derived}
          weight=t267 {derived}
          bias=t7 {pt2=root:p_features_1_block_1_fc1_bias target=features.1.block.1.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n24 {pt2=root[11] torch.ops.aten.convolution.default (convolution_2)}: [t269 f32 [H=8
                                                                      W=1 C=1] {pt2=root:convolution_2}] =
        permute x=t268 {derived} perm=[H<-C, W<-H, C<-W]
    n25 {pt2=root[12] torch.ops.aten.relu.default (relu_1)}: [t270 f32 [H=8 W=1
                                                                      C=1] {pt2=root:relu_1}] =
      relu x=t269 {pt2=root:convolution_2}
    group g6 torch.ops.aten.convolution.default:
      n26 {derived}: [t271 f32 [C=8] {derived}] =
        permute x=t270 {pt2=root:relu_1} perm=[H<-W, W<-C, C<-H]
      n27 {derived}: [t272 f32 [N=16 T=1 D=1 H=1 W=1 C=8] {derived}] =
        permute
          x=t8 {pt2=root:p_features_1_block_1_fc2_weight target=features.1.block.1.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n28 {derived}: [t273 f32 [C=16] {derived}] =
        convolution
          x=t271 {derived}
          weight=t272 {derived}
          bias=t9 {pt2=root:p_features_1_block_1_fc2_bias target=features.1.block.1.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n29 {pt2=root[13] torch.ops.aten.convolution.default (convolution_3)}: [t274 f32 [H=16
                                                                      W=1 C=1] {pt2=root:convolution_3}] =
        permute x=t273 {derived} perm=[H<-C, W<-H, C<-W]
    n30 {pt2=root[14] torch.ops.aten.add.Tensor (add_1)}: [t275 f32 [H=16 W=1
                                                                     C=1] {pt2=root:add_1}] =
      add_scalar x=t274 {pt2=root:convolution_3} scalar=3
    n31 {pt2=root[15] torch.ops.aten.clamp.default (clamp_2)}: [t276 f32 [H=16
                                                                      W=1 C=1] {pt2=root:clamp_2}] =
      clamp x=t275 {pt2=root:add_1} params={min=0; max=none}
    n32 {pt2=root[16] torch.ops.aten.clamp.default (clamp_3)}: [t277 f32 [H=16
                                                                      W=1 C=1] {pt2=root:clamp_3}] =
      clamp x=t276 {pt2=root:clamp_2} params={min=none; max=6}
    n33 {pt2=root[17] torch.ops.aten.div.Tensor (div_1)}: [t278 f32 [H=16 W=1
                                                                     C=1] {pt2=root:div_1}] =
      div_scalar x=t277 {pt2=root:clamp_3} scalar=6
    n34 {pt2=root[18] torch.ops.aten.mul.Tensor (mul_1)}: [t279 f32 [H=16 W=56
                                                                     C=56] {pt2=root:mul_1}] =
      mul a=t278 {pt2=root:div_1} b=t264 {pt2=root:relu}
    group g7 torch.ops.aten.convolution.default:
      n35 {derived}: [t280 f32 [H=56 W=56 C=16] {derived}] =
        permute x=t279 {pt2=root:mul_1} perm=[H<-W, W<-C, C<-H]
      n36 {derived}: [t281 f32 [N=16 T=1 D=1 H=1 W=1 C=16] {derived}] =
        permute
          x=t10 {pt2=root:p_features_1_block_2_0_weight target=features.1.block.2.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n37 {derived}: [t282 f32 [H=56 W=56 C=16] {derived}] =
        convolution
          x=t280 {derived}
          weight=t281 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n38 {pt2=root[19] torch.ops.aten.convolution.default (convolution_4)}: [t283 f32 [H=16
                                                                      W=56
                                                                      C=56] {pt2=root:convolution_4}] =
        permute x=t282 {derived} perm=[H<-C, W<-H, C<-W]
    group g8 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n39 {derived}: [t284 f32 [H=56 W=56 C=16] {derived}] =
        permute x=t283 {pt2=root:convolution_4} perm=[H<-W, W<-C, C<-H]
      n40 {derived}: [t285 f32 [H=56 W=56 C=16] {derived}] =
        batch_norm
          x=t284 {derived}
          weight=t11 {pt2=root:p_features_1_block_2_1_weight target=features.1.block.2.1.weight}
          bias=t12 {pt2=root:p_features_1_block_2_1_bias target=features.1.block.2.1.bias}
          running_mean=t148 {pt2=root:b_features_1_block_2_1_running_mean target=features.1.block.2.1.running_mean}
          running_var=t149 {pt2=root:b_features_1_block_2_1_running_var target=features.1.block.2.1.running_var}
          params={channel=C; eps=0.001}
      n41 {pt2=root[20] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_2)}: [t286 f32 [H=16
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_6}] =
        permute x=t285 {derived} perm=[H<-C, W<-H, C<-W]
    group g9 torch.ops.aten.convolution.default:
      n42 {derived}: [t287 f32 [H=56 W=56 C=16] {derived}] =
        permute x=t286 {pt2=root:getitem_6} perm=[H<-W, W<-C, C<-H]
      n43 {derived}: [t288 f32 [N=72 T=1 D=1 H=1 W=1 C=16] {derived}] =
        permute
          x=t13 {pt2=root:p_features_2_block_0_0_weight target=features.2.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n44 {derived}: [t289 f32 [H=56 W=56 C=72] {derived}] =
        convolution
          x=t287 {derived}
          weight=t288 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n45 {pt2=root[21] torch.ops.aten.convolution.default (convolution_5)}: [t290 f32 [H=72
                                                                      W=56
                                                                      C=56] {pt2=root:convolution_5}] =
        permute x=t289 {derived} perm=[H<-C, W<-H, C<-W]
    group g10 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n46 {derived}: [t291 f32 [H=56 W=56 C=72] {derived}] =
        permute x=t290 {pt2=root:convolution_5} perm=[H<-W, W<-C, C<-H]
      n47 {derived}: [t292 f32 [H=56 W=56 C=72] {derived}] =
        batch_norm
          x=t291 {derived}
          weight=t14 {pt2=root:p_features_2_block_0_1_weight target=features.2.block.0.1.weight}
          bias=t15 {pt2=root:p_features_2_block_0_1_bias target=features.2.block.0.1.bias}
          running_mean=t151 {pt2=root:b_features_2_block_0_1_running_mean target=features.2.block.0.1.running_mean}
          running_var=t152 {pt2=root:b_features_2_block_0_1_running_var target=features.2.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n48 {pt2=root[22] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_3)}: [t293 f32 [H=72
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_9}] =
        permute x=t292 {derived} perm=[H<-C, W<-H, C<-W]
    n49 {pt2=root[23] torch.ops.aten.relu.default (relu_2)}: [t294 f32 [H=72
                                                                      W=56
                                                                      C=56] {pt2=root:relu_2}] =
      relu x=t293 {pt2=root:getitem_9}
    group g11 torch.ops.aten.convolution.default:
      n50 {derived}: [t295 f32 [H=56 W=56 C=72] {derived}] =
        permute x=t294 {pt2=root:relu_2} perm=[H<-W, W<-C, C<-H]
      n51 {derived}: [t296 f32 [N=72 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t16 {pt2=root:p_features_2_block_1_0_weight target=features.2.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n52 {derived}: [t297 f32 [H=28 W=28 C=72] {derived}] =
        convolution
          x=t295 {derived}
          weight=t296 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=72}
      n53 {pt2=root[24] torch.ops.aten.convolution.default (convolution_6)}: [t298 f32 [H=72
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_6}] =
        permute x=t297 {derived} perm=[H<-C, W<-H, C<-W]
    group g12 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n54 {derived}: [t299 f32 [H=28 W=28 C=72] {derived}] =
        permute x=t298 {pt2=root:convolution_6} perm=[H<-W, W<-C, C<-H]
      n55 {derived}: [t300 f32 [H=28 W=28 C=72] {derived}] =
        batch_norm
          x=t299 {derived}
          weight=t17 {pt2=root:p_features_2_block_1_1_weight target=features.2.block.1.1.weight}
          bias=t18 {pt2=root:p_features_2_block_1_1_bias target=features.2.block.1.1.bias}
          running_mean=t154 {pt2=root:b_features_2_block_1_1_running_mean target=features.2.block.1.1.running_mean}
          running_var=t155 {pt2=root:b_features_2_block_1_1_running_var target=features.2.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n56 {pt2=root[25] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_4)}: [t301 f32 [H=72
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_12}] =
        permute x=t300 {derived} perm=[H<-C, W<-H, C<-W]
    n57 {pt2=root[26] torch.ops.aten.relu.default (relu_3)}: [t302 f32 [H=72
                                                                      W=28
                                                                      C=28] {pt2=root:relu_3}] =
      relu x=t301 {pt2=root:getitem_12}
    group g13 torch.ops.aten.convolution.default:
      n58 {derived}: [t303 f32 [H=28 W=28 C=72] {derived}] =
        permute x=t302 {pt2=root:relu_3} perm=[H<-W, W<-C, C<-H]
      n59 {derived}: [t304 f32 [N=24 T=1 D=1 H=1 W=1 C=72] {derived}] =
        permute
          x=t19 {pt2=root:p_features_2_block_2_0_weight target=features.2.block.2.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n60 {derived}: [t305 f32 [H=28 W=28 C=24] {derived}] =
        convolution
          x=t303 {derived}
          weight=t304 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n61 {pt2=root[27] torch.ops.aten.convolution.default (convolution_7)}: [t306 f32 [H=24
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_7}] =
        permute x=t305 {derived} perm=[H<-C, W<-H, C<-W]
    group g14 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n62 {derived}: [t307 f32 [H=28 W=28 C=24] {derived}] =
        permute x=t306 {pt2=root:convolution_7} perm=[H<-W, W<-C, C<-H]
      n63 {derived}: [t308 f32 [H=28 W=28 C=24] {derived}] =
        batch_norm
          x=t307 {derived}
          weight=t20 {pt2=root:p_features_2_block_2_1_weight target=features.2.block.2.1.weight}
          bias=t21 {pt2=root:p_features_2_block_2_1_bias target=features.2.block.2.1.bias}
          running_mean=t157 {pt2=root:b_features_2_block_2_1_running_mean target=features.2.block.2.1.running_mean}
          running_var=t158 {pt2=root:b_features_2_block_2_1_running_var target=features.2.block.2.1.running_var}
          params={channel=C; eps=0.001}
      n64 {pt2=root[28] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_5)}: [t309 f32 [H=24
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_15}] =
        permute x=t308 {derived} perm=[H<-C, W<-H, C<-W]
    group g15 torch.ops.aten.convolution.default:
      n65 {derived}: [t310 f32 [H=28 W=28 C=24] {derived}] =
        permute x=t309 {pt2=root:getitem_15} perm=[H<-W, W<-C, C<-H]
      n66 {derived}: [t311 f32 [N=88 T=1 D=1 H=1 W=1 C=24] {derived}] =
        permute
          x=t22 {pt2=root:p_features_3_block_0_0_weight target=features.3.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n67 {derived}: [t312 f32 [H=28 W=28 C=88] {derived}] =
        convolution
          x=t310 {derived}
          weight=t311 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n68 {pt2=root[29] torch.ops.aten.convolution.default (convolution_8)}: [t313 f32 [H=88
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_8}] =
        permute x=t312 {derived} perm=[H<-C, W<-H, C<-W]
    group g16 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n69 {derived}: [t314 f32 [H=28 W=28 C=88] {derived}] =
        permute x=t313 {pt2=root:convolution_8} perm=[H<-W, W<-C, C<-H]
      n70 {derived}: [t315 f32 [H=28 W=28 C=88] {derived}] =
        batch_norm
          x=t314 {derived}
          weight=t23 {pt2=root:p_features_3_block_0_1_weight target=features.3.block.0.1.weight}
          bias=t24 {pt2=root:p_features_3_block_0_1_bias target=features.3.block.0.1.bias}
          running_mean=t160 {pt2=root:b_features_3_block_0_1_running_mean target=features.3.block.0.1.running_mean}
          running_var=t161 {pt2=root:b_features_3_block_0_1_running_var target=features.3.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n71 {pt2=root[30] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_6)}: [t316 f32 [H=88
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_18}] =
        permute x=t315 {derived} perm=[H<-C, W<-H, C<-W]
    n72 {pt2=root[31] torch.ops.aten.relu.default (relu_4)}: [t317 f32 [H=88
                                                                      W=28
                                                                      C=28] {pt2=root:relu_4}] =
      relu x=t316 {pt2=root:getitem_18}
    group g17 torch.ops.aten.convolution.default:
      n73 {derived}: [t318 f32 [H=28 W=28 C=88] {derived}] =
        permute x=t317 {pt2=root:relu_4} perm=[H<-W, W<-C, C<-H]
      n74 {derived}: [t319 f32 [N=88 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t25 {pt2=root:p_features_3_block_1_0_weight target=features.3.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n75 {derived}: [t320 f32 [H=28 W=28 C=88] {derived}] =
        convolution
          x=t318 {derived}
          weight=t319 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=88}
      n76 {pt2=root[32] torch.ops.aten.convolution.default (convolution_9)}: [t321 f32 [H=88
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_9}] =
        permute x=t320 {derived} perm=[H<-C, W<-H, C<-W]
    group g18 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n77 {derived}: [t322 f32 [H=28 W=28 C=88] {derived}] =
        permute x=t321 {pt2=root:convolution_9} perm=[H<-W, W<-C, C<-H]
      n78 {derived}: [t323 f32 [H=28 W=28 C=88] {derived}] =
        batch_norm
          x=t322 {derived}
          weight=t26 {pt2=root:p_features_3_block_1_1_weight target=features.3.block.1.1.weight}
          bias=t27 {pt2=root:p_features_3_block_1_1_bias target=features.3.block.1.1.bias}
          running_mean=t163 {pt2=root:b_features_3_block_1_1_running_mean target=features.3.block.1.1.running_mean}
          running_var=t164 {pt2=root:b_features_3_block_1_1_running_var target=features.3.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n79 {pt2=root[33] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_7)}: [t324 f32 [H=88
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_21}] =
        permute x=t323 {derived} perm=[H<-C, W<-H, C<-W]
    n80 {pt2=root[34] torch.ops.aten.relu.default (relu_5)}: [t325 f32 [H=88
                                                                      W=28
                                                                      C=28] {pt2=root:relu_5}] =
      relu x=t324 {pt2=root:getitem_21}
    group g19 torch.ops.aten.convolution.default:
      n81 {derived}: [t326 f32 [H=28 W=28 C=88] {derived}] =
        permute x=t325 {pt2=root:relu_5} perm=[H<-W, W<-C, C<-H]
      n82 {derived}: [t327 f32 [N=24 T=1 D=1 H=1 W=1 C=88] {derived}] =
        permute
          x=t28 {pt2=root:p_features_3_block_2_0_weight target=features.3.block.2.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n83 {derived}: [t328 f32 [H=28 W=28 C=24] {derived}] =
        convolution
          x=t326 {derived}
          weight=t327 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n84 {pt2=root[35] torch.ops.aten.convolution.default (convolution_10)}: [t329 f32 [H=24
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_10}] =
        permute x=t328 {derived} perm=[H<-C, W<-H, C<-W]
    group g20 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n85 {derived}: [t330 f32 [H=28 W=28 C=24] {derived}] =
        permute x=t329 {pt2=root:convolution_10} perm=[H<-W, W<-C, C<-H]
      n86 {derived}: [t331 f32 [H=28 W=28 C=24] {derived}] =
        batch_norm
          x=t330 {derived}
          weight=t29 {pt2=root:p_features_3_block_2_1_weight target=features.3.block.2.1.weight}
          bias=t30 {pt2=root:p_features_3_block_2_1_bias target=features.3.block.2.1.bias}
          running_mean=t166 {pt2=root:b_features_3_block_2_1_running_mean target=features.3.block.2.1.running_mean}
          running_var=t167 {pt2=root:b_features_3_block_2_1_running_var target=features.3.block.2.1.running_var}
          params={channel=C; eps=0.001}
      n87 {pt2=root[36] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_8)}: [t332 f32 [H=24
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_24}] =
        permute x=t331 {derived} perm=[H<-C, W<-H, C<-W]
    n88 {pt2=root[37] torch.ops.aten.add.Tensor (add_2)}: [t333 f32 [H=24 W=28
                                                                     C=28] {pt2=root:add_2}] =
      add a=t332 {pt2=root:getitem_24} b=t309 {pt2=root:getitem_15}
    group g21 torch.ops.aten.convolution.default:
      n89 {derived}: [t334 f32 [H=28 W=28 C=24] {derived}] =
        permute x=t333 {pt2=root:add_2} perm=[H<-W, W<-C, C<-H]
      n90 {derived}: [t335 f32 [N=96 T=1 D=1 H=1 W=1 C=24] {derived}] =
        permute
          x=t31 {pt2=root:p_features_4_block_0_0_weight target=features.4.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n91 {derived}: [t336 f32 [H=28 W=28 C=96] {derived}] =
        convolution
          x=t334 {derived}
          weight=t335 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n92 {pt2=root[38] torch.ops.aten.convolution.default (convolution_11)}: [t337 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_11}] =
        permute x=t336 {derived} perm=[H<-C, W<-H, C<-W]
    group g22 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n93 {derived}: [t338 f32 [H=28 W=28 C=96] {derived}] =
        permute x=t337 {pt2=root:convolution_11} perm=[H<-W, W<-C, C<-H]
      n94 {derived}: [t339 f32 [H=28 W=28 C=96] {derived}] =
        batch_norm
          x=t338 {derived}
          weight=t32 {pt2=root:p_features_4_block_0_1_weight target=features.4.block.0.1.weight}
          bias=t33 {pt2=root:p_features_4_block_0_1_bias target=features.4.block.0.1.bias}
          running_mean=t169 {pt2=root:b_features_4_block_0_1_running_mean target=features.4.block.0.1.running_mean}
          running_var=t170 {pt2=root:b_features_4_block_0_1_running_var target=features.4.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n95 {pt2=root[39] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_9)}: [t340 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_27}] =
        permute x=t339 {derived} perm=[H<-C, W<-H, C<-W]
    n96 {pt2=root[40] torch.ops.aten.add.Tensor (add_3)}: [t341 f32 [H=96 W=28
                                                                     C=28] {pt2=root:add_3}] =
      add_scalar x=t340 {pt2=root:getitem_27} scalar=3
    n97 {pt2=root[41] torch.ops.aten.clamp.default (clamp_4)}: [t342 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:clamp_4}] =
      clamp x=t341 {pt2=root:add_3} params={min=0; max=none}
    n98 {pt2=root[42] torch.ops.aten.clamp.default (clamp_5)}: [t343 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:clamp_5}] =
      clamp x=t342 {pt2=root:clamp_4} params={min=none; max=6}
    n99 {pt2=root[43] torch.ops.aten.mul.Tensor (mul_2)}: [t344 f32 [H=96 W=28
                                                                     C=28] {pt2=root:mul_2}] =
      mul a=t340 {pt2=root:getitem_27} b=t343 {pt2=root:clamp_5}
    n100 {pt2=root[44] torch.ops.aten.div.Tensor (div_2)}: [t345 f32 [H=96 W=28
                                                                      C=28] {pt2=root:div_2}] =
      div_scalar x=t344 {pt2=root:mul_2} scalar=6
    group g23 torch.ops.aten.convolution.default:
      n101 {derived}: [t346 f32 [H=28 W=28 C=96] {derived}] =
        permute x=t345 {pt2=root:div_2} perm=[H<-W, W<-C, C<-H]
      n102 {derived}: [t347 f32 [N=96 T=1 D=1 H=5 W=5 C=1] {derived}] =
        permute
          x=t34 {pt2=root:p_features_4_block_1_0_weight target=features.4.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n103 {derived}: [t348 f32 [H=14 W=14 C=96] {derived}] =
        convolution
          x=t346 {derived}
          weight=t347 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=2; w=2};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=96}
      n104 {pt2=root[45] torch.ops.aten.convolution.default (convolution_12)}: [t349 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_12}] =
        permute x=t348 {derived} perm=[H<-C, W<-H, C<-W]
    group g24 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n105 {derived}: [t350 f32 [H=14 W=14 C=96] {derived}] =
        permute x=t349 {pt2=root:convolution_12} perm=[H<-W, W<-C, C<-H]
      n106 {derived}: [t351 f32 [H=14 W=14 C=96] {derived}] =
        batch_norm
          x=t350 {derived}
          weight=t35 {pt2=root:p_features_4_block_1_1_weight target=features.4.block.1.1.weight}
          bias=t36 {pt2=root:p_features_4_block_1_1_bias target=features.4.block.1.1.bias}
          running_mean=t172 {pt2=root:b_features_4_block_1_1_running_mean target=features.4.block.1.1.running_mean}
          running_var=t173 {pt2=root:b_features_4_block_1_1_running_var target=features.4.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n107 {pt2=root[46] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_10)}: [t352 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_30}] =
        permute x=t351 {derived} perm=[H<-C, W<-H, C<-W]
    n108 {pt2=root[47] torch.ops.aten.add.Tensor (add_4)}: [t353 f32 [H=96 W=14
                                                                      C=14] {pt2=root:add_4}] =
      add_scalar x=t352 {pt2=root:getitem_30} scalar=3
    n109 {pt2=root[48] torch.ops.aten.clamp.default (clamp_6)}: [t354 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_6}] =
      clamp x=t353 {pt2=root:add_4} params={min=0; max=none}
    n110 {pt2=root[49] torch.ops.aten.clamp.default (clamp_7)}: [t355 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_7}] =
      clamp x=t354 {pt2=root:clamp_6} params={min=none; max=6}
    n111 {pt2=root[50] torch.ops.aten.mul.Tensor (mul_3)}: [t356 f32 [H=96 W=14
                                                                      C=14] {pt2=root:mul_3}] =
      mul a=t352 {pt2=root:getitem_30} b=t355 {pt2=root:clamp_7}
    n112 {pt2=root[51] torch.ops.aten.div.Tensor (div_3)}: [t357 f32 [H=96 W=14
                                                                      C=14] {pt2=root:div_3}] =
      div_scalar x=t356 {pt2=root:mul_3} scalar=6
    n113 {pt2=root[52] torch.ops.aten.mean.dim (mean_1)}: [t358 f32 [H=96 W=1
                                                                     C=1] {pt2=root:mean_1}] =
      mean x=t357 {pt2=root:div_3} params={dims=[C, W]; keepdim=true}
    group g25 torch.ops.aten.convolution.default:
      n114 {derived}: [t359 f32 [C=96] {derived}] =
        permute x=t358 {pt2=root:mean_1} perm=[H<-W, W<-C, C<-H]
      n115 {derived}: [t360 f32 [N=24 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t37 {pt2=root:p_features_4_block_2_fc1_weight target=features.4.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n116 {derived}: [t361 f32 [C=24] {derived}] =
        convolution
          x=t359 {derived}
          weight=t360 {derived}
          bias=t38 {pt2=root:p_features_4_block_2_fc1_bias target=features.4.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n117 {pt2=root[53] torch.ops.aten.convolution.default (convolution_13)}: [t362 f32 [H=24
                                                                      W=1 C=1] {pt2=root:convolution_13}] =
        permute x=t361 {derived} perm=[H<-C, W<-H, C<-W]
    n118 {pt2=root[54] torch.ops.aten.relu.default (relu_6)}: [t363 f32 [H=24
                                                                      W=1 C=1] {pt2=root:relu_6}] =
      relu x=t362 {pt2=root:convolution_13}
    group g26 torch.ops.aten.convolution.default:
      n119 {derived}: [t364 f32 [C=24] {derived}] =
        permute x=t363 {pt2=root:relu_6} perm=[H<-W, W<-C, C<-H]
      n120 {derived}: [t365 f32 [N=96 T=1 D=1 H=1 W=1 C=24] {derived}] =
        permute
          x=t39 {pt2=root:p_features_4_block_2_fc2_weight target=features.4.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n121 {derived}: [t366 f32 [C=96] {derived}] =
        convolution
          x=t364 {derived}
          weight=t365 {derived}
          bias=t40 {pt2=root:p_features_4_block_2_fc2_bias target=features.4.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n122 {pt2=root[55] torch.ops.aten.convolution.default (convolution_14)}: [t367 f32 [H=96
                                                                      W=1 C=1] {pt2=root:convolution_14}] =
        permute x=t366 {derived} perm=[H<-C, W<-H, C<-W]
    n123 {pt2=root[56] torch.ops.aten.add.Tensor (add_5)}: [t368 f32 [H=96 W=1
                                                                      C=1] {pt2=root:add_5}] =
      add_scalar x=t367 {pt2=root:convolution_14} scalar=3
    n124 {pt2=root[57] torch.ops.aten.clamp.default (clamp_8)}: [t369 f32 [H=96
                                                                      W=1 C=1] {pt2=root:clamp_8}] =
      clamp x=t368 {pt2=root:add_5} params={min=0; max=none}
    n125 {pt2=root[58] torch.ops.aten.clamp.default (clamp_9)}: [t370 f32 [H=96
                                                                      W=1 C=1] {pt2=root:clamp_9}] =
      clamp x=t369 {pt2=root:clamp_8} params={min=none; max=6}
    n126 {pt2=root[59] torch.ops.aten.div.Tensor (div_4)}: [t371 f32 [H=96 W=1
                                                                      C=1] {pt2=root:div_4}] =
      div_scalar x=t370 {pt2=root:clamp_9} scalar=6
    n127 {pt2=root[60] torch.ops.aten.mul.Tensor (mul_4)}: [t372 f32 [H=96 W=14
                                                                      C=14] {pt2=root:mul_4}] =
      mul a=t371 {pt2=root:div_4} b=t357 {pt2=root:div_3}
    group g27 torch.ops.aten.convolution.default:
      n128 {derived}: [t373 f32 [H=14 W=14 C=96] {derived}] =
        permute x=t372 {pt2=root:mul_4} perm=[H<-W, W<-C, C<-H]
      n129 {derived}: [t374 f32 [N=40 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t41 {pt2=root:p_features_4_block_3_0_weight target=features.4.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n130 {derived}: [t375 f32 [H=14 W=14 C=40] {derived}] =
        convolution
          x=t373 {derived}
          weight=t374 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n131 {pt2=root[61] torch.ops.aten.convolution.default (convolution_15)}: [t376 f32 [H=40
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_15}] =
        permute x=t375 {derived} perm=[H<-C, W<-H, C<-W]
    group g28 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n132 {derived}: [t377 f32 [H=14 W=14 C=40] {derived}] =
        permute x=t376 {pt2=root:convolution_15} perm=[H<-W, W<-C, C<-H]
      n133 {derived}: [t378 f32 [H=14 W=14 C=40] {derived}] =
        batch_norm
          x=t377 {derived}
          weight=t42 {pt2=root:p_features_4_block_3_1_weight target=features.4.block.3.1.weight}
          bias=t43 {pt2=root:p_features_4_block_3_1_bias target=features.4.block.3.1.bias}
          running_mean=t175 {pt2=root:b_features_4_block_3_1_running_mean target=features.4.block.3.1.running_mean}
          running_var=t176 {pt2=root:b_features_4_block_3_1_running_var target=features.4.block.3.1.running_var}
          params={channel=C; eps=0.001}
      n134 {pt2=root[62] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_11)}: [t379 f32 [H=40
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_33}] =
        permute x=t378 {derived} perm=[H<-C, W<-H, C<-W]
    group g29 torch.ops.aten.convolution.default:
      n135 {derived}: [t380 f32 [H=14 W=14 C=40] {derived}] =
        permute x=t379 {pt2=root:getitem_33} perm=[H<-W, W<-C, C<-H]
      n136 {derived}: [t381 f32 [N=240 T=1 D=1 H=1 W=1 C=40] {derived}] =
        permute
          x=t44 {pt2=root:p_features_5_block_0_0_weight target=features.5.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n137 {derived}: [t382 f32 [H=14 W=14 C=240] {derived}] =
        convolution
          x=t380 {derived}
          weight=t381 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n138 {pt2=root[63] torch.ops.aten.convolution.default (convolution_16)}: [t383 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_16}] =
        permute x=t382 {derived} perm=[H<-C, W<-H, C<-W]
    group g30 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n139 {derived}: [t384 f32 [H=14 W=14 C=240] {derived}] =
        permute x=t383 {pt2=root:convolution_16} perm=[H<-W, W<-C, C<-H]
      n140 {derived}: [t385 f32 [H=14 W=14 C=240] {derived}] =
        batch_norm
          x=t384 {derived}
          weight=t45 {pt2=root:p_features_5_block_0_1_weight target=features.5.block.0.1.weight}
          bias=t46 {pt2=root:p_features_5_block_0_1_bias target=features.5.block.0.1.bias}
          running_mean=t178 {pt2=root:b_features_5_block_0_1_running_mean target=features.5.block.0.1.running_mean}
          running_var=t179 {pt2=root:b_features_5_block_0_1_running_var target=features.5.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n141 {pt2=root[64] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_12)}: [t386 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_36}] =
        permute x=t385 {derived} perm=[H<-C, W<-H, C<-W]
    n142 {pt2=root[65] torch.ops.aten.add.Tensor (add_6)}: [t387 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:add_6}] =
      add_scalar x=t386 {pt2=root:getitem_36} scalar=3
    n143 {pt2=root[66] torch.ops.aten.clamp.default (clamp_10)}: [t388 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_10}] =
      clamp x=t387 {pt2=root:add_6} params={min=0; max=none}
    n144 {pt2=root[67] torch.ops.aten.clamp.default (clamp_11)}: [t389 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_11}] =
      clamp x=t388 {pt2=root:clamp_10} params={min=none; max=6}
    n145 {pt2=root[68] torch.ops.aten.mul.Tensor (mul_5)}: [t390 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:mul_5}] =
      mul a=t386 {pt2=root:getitem_36} b=t389 {pt2=root:clamp_11}
    n146 {pt2=root[69] torch.ops.aten.div.Tensor (div_5)}: [t391 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:div_5}] =
      div_scalar x=t390 {pt2=root:mul_5} scalar=6
    group g31 torch.ops.aten.convolution.default:
      n147 {derived}: [t392 f32 [H=14 W=14 C=240] {derived}] =
        permute x=t391 {pt2=root:div_5} perm=[H<-W, W<-C, C<-H]
      n148 {derived}: [t393 f32 [N=240 T=1 D=1 H=5 W=5 C=1] {derived}] =
        permute
          x=t47 {pt2=root:p_features_5_block_1_0_weight target=features.5.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n149 {derived}: [t394 f32 [H=14 W=14 C=240] {derived}] =
        convolution
          x=t392 {derived}
          weight=t393 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=2; w=2};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=240}
      n150 {pt2=root[70] torch.ops.aten.convolution.default (convolution_17)}: [t395 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_17}] =
        permute x=t394 {derived} perm=[H<-C, W<-H, C<-W]
    group g32 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n151 {derived}: [t396 f32 [H=14 W=14 C=240] {derived}] =
        permute x=t395 {pt2=root:convolution_17} perm=[H<-W, W<-C, C<-H]
      n152 {derived}: [t397 f32 [H=14 W=14 C=240] {derived}] =
        batch_norm
          x=t396 {derived}
          weight=t48 {pt2=root:p_features_5_block_1_1_weight target=features.5.block.1.1.weight}
          bias=t49 {pt2=root:p_features_5_block_1_1_bias target=features.5.block.1.1.bias}
          running_mean=t181 {pt2=root:b_features_5_block_1_1_running_mean target=features.5.block.1.1.running_mean}
          running_var=t182 {pt2=root:b_features_5_block_1_1_running_var target=features.5.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n153 {pt2=root[71] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_13)}: [t398 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_39}] =
        permute x=t397 {derived} perm=[H<-C, W<-H, C<-W]
    n154 {pt2=root[72] torch.ops.aten.add.Tensor (add_7)}: [t399 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:add_7}] =
      add_scalar x=t398 {pt2=root:getitem_39} scalar=3
    n155 {pt2=root[73] torch.ops.aten.clamp.default (clamp_12)}: [t400 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_12}] =
      clamp x=t399 {pt2=root:add_7} params={min=0; max=none}
    n156 {pt2=root[74] torch.ops.aten.clamp.default (clamp_13)}: [t401 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_13}] =
      clamp x=t400 {pt2=root:clamp_12} params={min=none; max=6}
    n157 {pt2=root[75] torch.ops.aten.mul.Tensor (mul_6)}: [t402 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:mul_6}] =
      mul a=t398 {pt2=root:getitem_39} b=t401 {pt2=root:clamp_13}
    n158 {pt2=root[76] torch.ops.aten.div.Tensor (div_6)}: [t403 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:div_6}] =
      div_scalar x=t402 {pt2=root:mul_6} scalar=6
    n159 {pt2=root[77] torch.ops.aten.mean.dim (mean_2)}: [t404 f32 [H=240 W=1
                                                                     C=1] {pt2=root:mean_2}] =
      mean x=t403 {pt2=root:div_6} params={dims=[C, W]; keepdim=true}
    group g33 torch.ops.aten.convolution.default:
      n160 {derived}: [t405 f32 [C=240] {derived}] =
        permute x=t404 {pt2=root:mean_2} perm=[H<-W, W<-C, C<-H]
      n161 {derived}: [t406 f32 [N=64 T=1 D=1 H=1 W=1 C=240] {derived}] =
        permute
          x=t50 {pt2=root:p_features_5_block_2_fc1_weight target=features.5.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n162 {derived}: [t407 f32 [C=64] {derived}] =
        convolution
          x=t405 {derived}
          weight=t406 {derived}
          bias=t51 {pt2=root:p_features_5_block_2_fc1_bias target=features.5.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n163 {pt2=root[78] torch.ops.aten.convolution.default (convolution_18)}: [t408 f32 [H=64
                                                                      W=1 C=1] {pt2=root:convolution_18}] =
        permute x=t407 {derived} perm=[H<-C, W<-H, C<-W]
    n164 {pt2=root[79] torch.ops.aten.relu.default (relu_7)}: [t409 f32 [H=64
                                                                      W=1 C=1] {pt2=root:relu_7}] =
      relu x=t408 {pt2=root:convolution_18}
    group g34 torch.ops.aten.convolution.default:
      n165 {derived}: [t410 f32 [C=64] {derived}] =
        permute x=t409 {pt2=root:relu_7} perm=[H<-W, W<-C, C<-H]
      n166 {derived}: [t411 f32 [N=240 T=1 D=1 H=1 W=1 C=64] {derived}] =
        permute
          x=t52 {pt2=root:p_features_5_block_2_fc2_weight target=features.5.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n167 {derived}: [t412 f32 [C=240] {derived}] =
        convolution
          x=t410 {derived}
          weight=t411 {derived}
          bias=t53 {pt2=root:p_features_5_block_2_fc2_bias target=features.5.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n168 {pt2=root[80] torch.ops.aten.convolution.default (convolution_19)}: [t413 f32 [H=240
                                                                      W=1 C=1] {pt2=root:convolution_19}] =
        permute x=t412 {derived} perm=[H<-C, W<-H, C<-W]
    n169 {pt2=root[81] torch.ops.aten.add.Tensor (add_8)}: [t414 f32 [H=240 W=1
                                                                      C=1] {pt2=root:add_8}] =
      add_scalar x=t413 {pt2=root:convolution_19} scalar=3
    n170 {pt2=root[82] torch.ops.aten.clamp.default (clamp_14)}: [t415 f32 [H=240
                                                                      W=1 C=1] {pt2=root:clamp_14}] =
      clamp x=t414 {pt2=root:add_8} params={min=0; max=none}
    n171 {pt2=root[83] torch.ops.aten.clamp.default (clamp_15)}: [t416 f32 [H=240
                                                                      W=1 C=1] {pt2=root:clamp_15}] =
      clamp x=t415 {pt2=root:clamp_14} params={min=none; max=6}
    n172 {pt2=root[84] torch.ops.aten.div.Tensor (div_7)}: [t417 f32 [H=240 W=1
                                                                      C=1] {pt2=root:div_7}] =
      div_scalar x=t416 {pt2=root:clamp_15} scalar=6
    n173 {pt2=root[85] torch.ops.aten.mul.Tensor (mul_7)}: [t418 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:mul_7}] =
      mul a=t417 {pt2=root:div_7} b=t403 {pt2=root:div_6}
    group g35 torch.ops.aten.convolution.default:
      n174 {derived}: [t419 f32 [H=14 W=14 C=240] {derived}] =
        permute x=t418 {pt2=root:mul_7} perm=[H<-W, W<-C, C<-H]
      n175 {derived}: [t420 f32 [N=40 T=1 D=1 H=1 W=1 C=240] {derived}] =
        permute
          x=t54 {pt2=root:p_features_5_block_3_0_weight target=features.5.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n176 {derived}: [t421 f32 [H=14 W=14 C=40] {derived}] =
        convolution
          x=t419 {derived}
          weight=t420 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n177 {pt2=root[86] torch.ops.aten.convolution.default (convolution_20)}: [t422 f32 [H=40
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_20}] =
        permute x=t421 {derived} perm=[H<-C, W<-H, C<-W]
    group g36 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n178 {derived}: [t423 f32 [H=14 W=14 C=40] {derived}] =
        permute x=t422 {pt2=root:convolution_20} perm=[H<-W, W<-C, C<-H]
      n179 {derived}: [t424 f32 [H=14 W=14 C=40] {derived}] =
        batch_norm
          x=t423 {derived}
          weight=t55 {pt2=root:p_features_5_block_3_1_weight target=features.5.block.3.1.weight}
          bias=t56 {pt2=root:p_features_5_block_3_1_bias target=features.5.block.3.1.bias}
          running_mean=t184 {pt2=root:b_features_5_block_3_1_running_mean target=features.5.block.3.1.running_mean}
          running_var=t185 {pt2=root:b_features_5_block_3_1_running_var target=features.5.block.3.1.running_var}
          params={channel=C; eps=0.001}
      n180 {pt2=root[87] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_14)}: [t425 f32 [H=40
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_42}] =
        permute x=t424 {derived} perm=[H<-C, W<-H, C<-W]
    n181 {pt2=root[88] torch.ops.aten.add.Tensor (add_9)}: [t426 f32 [H=40 W=14
                                                                      C=14] {pt2=root:add_9}] =
      add a=t425 {pt2=root:getitem_42} b=t379 {pt2=root:getitem_33}
    group g37 torch.ops.aten.convolution.default:
      n182 {derived}: [t427 f32 [H=14 W=14 C=40] {derived}] =
        permute x=t426 {pt2=root:add_9} perm=[H<-W, W<-C, C<-H]
      n183 {derived}: [t428 f32 [N=240 T=1 D=1 H=1 W=1 C=40] {derived}] =
        permute
          x=t57 {pt2=root:p_features_6_block_0_0_weight target=features.6.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n184 {derived}: [t429 f32 [H=14 W=14 C=240] {derived}] =
        convolution
          x=t427 {derived}
          weight=t428 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n185 {pt2=root[89] torch.ops.aten.convolution.default (convolution_21)}: [t430 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_21}] =
        permute x=t429 {derived} perm=[H<-C, W<-H, C<-W]
    group g38 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n186 {derived}: [t431 f32 [H=14 W=14 C=240] {derived}] =
        permute x=t430 {pt2=root:convolution_21} perm=[H<-W, W<-C, C<-H]
      n187 {derived}: [t432 f32 [H=14 W=14 C=240] {derived}] =
        batch_norm
          x=t431 {derived}
          weight=t58 {pt2=root:p_features_6_block_0_1_weight target=features.6.block.0.1.weight}
          bias=t59 {pt2=root:p_features_6_block_0_1_bias target=features.6.block.0.1.bias}
          running_mean=t187 {pt2=root:b_features_6_block_0_1_running_mean target=features.6.block.0.1.running_mean}
          running_var=t188 {pt2=root:b_features_6_block_0_1_running_var target=features.6.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n188 {pt2=root[90] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_15)}: [t433 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_45}] =
        permute x=t432 {derived} perm=[H<-C, W<-H, C<-W]
    n189 {pt2=root[91] torch.ops.aten.add.Tensor (add_10)}: [t434 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:add_10}] =
      add_scalar x=t433 {pt2=root:getitem_45} scalar=3
    n190 {pt2=root[92] torch.ops.aten.clamp.default (clamp_16)}: [t435 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_16}] =
      clamp x=t434 {pt2=root:add_10} params={min=0; max=none}
    n191 {pt2=root[93] torch.ops.aten.clamp.default (clamp_17)}: [t436 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_17}] =
      clamp x=t435 {pt2=root:clamp_16} params={min=none; max=6}
    n192 {pt2=root[94] torch.ops.aten.mul.Tensor (mul_8)}: [t437 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:mul_8}] =
      mul a=t433 {pt2=root:getitem_45} b=t436 {pt2=root:clamp_17}
    n193 {pt2=root[95] torch.ops.aten.div.Tensor (div_8)}: [t438 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:div_8}] =
      div_scalar x=t437 {pt2=root:mul_8} scalar=6
    group g39 torch.ops.aten.convolution.default:
      n194 {derived}: [t439 f32 [H=14 W=14 C=240] {derived}] =
        permute x=t438 {pt2=root:div_8} perm=[H<-W, W<-C, C<-H]
      n195 {derived}: [t440 f32 [N=240 T=1 D=1 H=5 W=5 C=1] {derived}] =
        permute
          x=t60 {pt2=root:p_features_6_block_1_0_weight target=features.6.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n196 {derived}: [t441 f32 [H=14 W=14 C=240] {derived}] =
        convolution
          x=t439 {derived}
          weight=t440 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=2; w=2};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=240}
      n197 {pt2=root[96] torch.ops.aten.convolution.default (convolution_22)}: [t442 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_22}] =
        permute x=t441 {derived} perm=[H<-C, W<-H, C<-W]
    group g40 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n198 {derived}: [t443 f32 [H=14 W=14 C=240] {derived}] =
        permute x=t442 {pt2=root:convolution_22} perm=[H<-W, W<-C, C<-H]
      n199 {derived}: [t444 f32 [H=14 W=14 C=240] {derived}] =
        batch_norm
          x=t443 {derived}
          weight=t61 {pt2=root:p_features_6_block_1_1_weight target=features.6.block.1.1.weight}
          bias=t62 {pt2=root:p_features_6_block_1_1_bias target=features.6.block.1.1.bias}
          running_mean=t190 {pt2=root:b_features_6_block_1_1_running_mean target=features.6.block.1.1.running_mean}
          running_var=t191 {pt2=root:b_features_6_block_1_1_running_var target=features.6.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n200 {pt2=root[97] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_16)}: [t445 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_48}] =
        permute x=t444 {derived} perm=[H<-C, W<-H, C<-W]
    n201 {pt2=root[98] torch.ops.aten.add.Tensor (add_11)}: [t446 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:add_11}] =
      add_scalar x=t445 {pt2=root:getitem_48} scalar=3
    n202 {pt2=root[99] torch.ops.aten.clamp.default (clamp_18)}: [t447 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_18}] =
      clamp x=t446 {pt2=root:add_11} params={min=0; max=none}
    n203 {pt2=root[100] torch.ops.aten.clamp.default (clamp_19)}: [t448 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_19}] =
      clamp x=t447 {pt2=root:clamp_18} params={min=none; max=6}
    n204 {pt2=root[101] torch.ops.aten.mul.Tensor (mul_9)}: [t449 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:mul_9}] =
      mul a=t445 {pt2=root:getitem_48} b=t448 {pt2=root:clamp_19}
    n205 {pt2=root[102] torch.ops.aten.div.Tensor (div_9)}: [t450 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:div_9}] =
      div_scalar x=t449 {pt2=root:mul_9} scalar=6
    n206 {pt2=root[103] torch.ops.aten.mean.dim (mean_3)}: [t451 f32 [H=240 W=1
                                                                      C=1] {pt2=root:mean_3}] =
      mean x=t450 {pt2=root:div_9} params={dims=[C, W]; keepdim=true}
    group g41 torch.ops.aten.convolution.default:
      n207 {derived}: [t452 f32 [C=240] {derived}] =
        permute x=t451 {pt2=root:mean_3} perm=[H<-W, W<-C, C<-H]
      n208 {derived}: [t453 f32 [N=64 T=1 D=1 H=1 W=1 C=240] {derived}] =
        permute
          x=t63 {pt2=root:p_features_6_block_2_fc1_weight target=features.6.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n209 {derived}: [t454 f32 [C=64] {derived}] =
        convolution
          x=t452 {derived}
          weight=t453 {derived}
          bias=t64 {pt2=root:p_features_6_block_2_fc1_bias target=features.6.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n210 {pt2=root[104] torch.ops.aten.convolution.default (convolution_23)}: [t455 f32 [H=64
                                                                      W=1 C=1] {pt2=root:convolution_23}] =
        permute x=t454 {derived} perm=[H<-C, W<-H, C<-W]
    n211 {pt2=root[105] torch.ops.aten.relu.default (relu_8)}: [t456 f32 [H=64
                                                                      W=1 C=1] {pt2=root:relu_8}] =
      relu x=t455 {pt2=root:convolution_23}
    group g42 torch.ops.aten.convolution.default:
      n212 {derived}: [t457 f32 [C=64] {derived}] =
        permute x=t456 {pt2=root:relu_8} perm=[H<-W, W<-C, C<-H]
      n213 {derived}: [t458 f32 [N=240 T=1 D=1 H=1 W=1 C=64] {derived}] =
        permute
          x=t65 {pt2=root:p_features_6_block_2_fc2_weight target=features.6.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n214 {derived}: [t459 f32 [C=240] {derived}] =
        convolution
          x=t457 {derived}
          weight=t458 {derived}
          bias=t66 {pt2=root:p_features_6_block_2_fc2_bias target=features.6.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n215 {pt2=root[106] torch.ops.aten.convolution.default (convolution_24)}: [t460 f32 [H=240
                                                                      W=1 C=1] {pt2=root:convolution_24}] =
        permute x=t459 {derived} perm=[H<-C, W<-H, C<-W]
    n216 {pt2=root[107] torch.ops.aten.add.Tensor (add_12)}: [t461 f32 [H=240
                                                                      W=1 C=1] {pt2=root:add_12}] =
      add_scalar x=t460 {pt2=root:convolution_24} scalar=3
    n217 {pt2=root[108] torch.ops.aten.clamp.default (clamp_20)}: [t462 f32 [H=240
                                                                      W=1 C=1] {pt2=root:clamp_20}] =
      clamp x=t461 {pt2=root:add_12} params={min=0; max=none}
    n218 {pt2=root[109] torch.ops.aten.clamp.default (clamp_21)}: [t463 f32 [H=240
                                                                      W=1 C=1] {pt2=root:clamp_21}] =
      clamp x=t462 {pt2=root:clamp_20} params={min=none; max=6}
    n219 {pt2=root[110] torch.ops.aten.div.Tensor (div_10)}: [t464 f32 [H=240
                                                                      W=1 C=1] {pt2=root:div_10}] =
      div_scalar x=t463 {pt2=root:clamp_21} scalar=6
    n220 {pt2=root[111] torch.ops.aten.mul.Tensor (mul_10)}: [t465 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:mul_10}] =
      mul a=t464 {pt2=root:div_10} b=t450 {pt2=root:div_9}
    group g43 torch.ops.aten.convolution.default:
      n221 {derived}: [t466 f32 [H=14 W=14 C=240] {derived}] =
        permute x=t465 {pt2=root:mul_10} perm=[H<-W, W<-C, C<-H]
      n222 {derived}: [t467 f32 [N=40 T=1 D=1 H=1 W=1 C=240] {derived}] =
        permute
          x=t67 {pt2=root:p_features_6_block_3_0_weight target=features.6.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n223 {derived}: [t468 f32 [H=14 W=14 C=40] {derived}] =
        convolution
          x=t466 {derived}
          weight=t467 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n224 {pt2=root[112] torch.ops.aten.convolution.default (convolution_25)}: [t469 f32 [H=40
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_25}] =
        permute x=t468 {derived} perm=[H<-C, W<-H, C<-W]
    group g44 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n225 {derived}: [t470 f32 [H=14 W=14 C=40] {derived}] =
        permute x=t469 {pt2=root:convolution_25} perm=[H<-W, W<-C, C<-H]
      n226 {derived}: [t471 f32 [H=14 W=14 C=40] {derived}] =
        batch_norm
          x=t470 {derived}
          weight=t68 {pt2=root:p_features_6_block_3_1_weight target=features.6.block.3.1.weight}
          bias=t69 {pt2=root:p_features_6_block_3_1_bias target=features.6.block.3.1.bias}
          running_mean=t193 {pt2=root:b_features_6_block_3_1_running_mean target=features.6.block.3.1.running_mean}
          running_var=t194 {pt2=root:b_features_6_block_3_1_running_var target=features.6.block.3.1.running_var}
          params={channel=C; eps=0.001}
      n227 {pt2=root[113] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_17)}: [t472 f32 [H=40
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_51}] =
        permute x=t471 {derived} perm=[H<-C, W<-H, C<-W]
    n228 {pt2=root[114] torch.ops.aten.add.Tensor (add_13)}: [t473 f32 [H=40
                                                                      W=14
                                                                      C=14] {pt2=root:add_13}] =
      add a=t472 {pt2=root:getitem_51} b=t426 {pt2=root:add_9}
    group g45 torch.ops.aten.convolution.default:
      n229 {derived}: [t474 f32 [H=14 W=14 C=40] {derived}] =
        permute x=t473 {pt2=root:add_13} perm=[H<-W, W<-C, C<-H]
      n230 {derived}: [t475 f32 [N=120 T=1 D=1 H=1 W=1 C=40] {derived}] =
        permute
          x=t70 {pt2=root:p_features_7_block_0_0_weight target=features.7.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n231 {derived}: [t476 f32 [H=14 W=14 C=120] {derived}] =
        convolution
          x=t474 {derived}
          weight=t475 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n232 {pt2=root[115] torch.ops.aten.convolution.default (convolution_26)}: [t477 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_26}] =
        permute x=t476 {derived} perm=[H<-C, W<-H, C<-W]
    group g46 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n233 {derived}: [t478 f32 [H=14 W=14 C=120] {derived}] =
        permute x=t477 {pt2=root:convolution_26} perm=[H<-W, W<-C, C<-H]
      n234 {derived}: [t479 f32 [H=14 W=14 C=120] {derived}] =
        batch_norm
          x=t478 {derived}
          weight=t71 {pt2=root:p_features_7_block_0_1_weight target=features.7.block.0.1.weight}
          bias=t72 {pt2=root:p_features_7_block_0_1_bias target=features.7.block.0.1.bias}
          running_mean=t196 {pt2=root:b_features_7_block_0_1_running_mean target=features.7.block.0.1.running_mean}
          running_var=t197 {pt2=root:b_features_7_block_0_1_running_var target=features.7.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n235 {pt2=root[116] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_18)}: [t480 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_54}] =
        permute x=t479 {derived} perm=[H<-C, W<-H, C<-W]
    n236 {pt2=root[117] torch.ops.aten.add.Tensor (add_14)}: [t481 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:add_14}] =
      add_scalar x=t480 {pt2=root:getitem_54} scalar=3
    n237 {pt2=root[118] torch.ops.aten.clamp.default (clamp_22)}: [t482 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_22}] =
      clamp x=t481 {pt2=root:add_14} params={min=0; max=none}
    n238 {pt2=root[119] torch.ops.aten.clamp.default (clamp_23)}: [t483 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_23}] =
      clamp x=t482 {pt2=root:clamp_22} params={min=none; max=6}
    n239 {pt2=root[120] torch.ops.aten.mul.Tensor (mul_11)}: [t484 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:mul_11}] =
      mul a=t480 {pt2=root:getitem_54} b=t483 {pt2=root:clamp_23}
    n240 {pt2=root[121] torch.ops.aten.div.Tensor (div_11)}: [t485 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:div_11}] =
      div_scalar x=t484 {pt2=root:mul_11} scalar=6
    group g47 torch.ops.aten.convolution.default:
      n241 {derived}: [t486 f32 [H=14 W=14 C=120] {derived}] =
        permute x=t485 {pt2=root:div_11} perm=[H<-W, W<-C, C<-H]
      n242 {derived}: [t487 f32 [N=120 T=1 D=1 H=5 W=5 C=1] {derived}] =
        permute
          x=t73 {pt2=root:p_features_7_block_1_0_weight target=features.7.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n243 {derived}: [t488 f32 [H=14 W=14 C=120] {derived}] =
        convolution
          x=t486 {derived}
          weight=t487 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=2; w=2};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=120}
      n244 {pt2=root[122] torch.ops.aten.convolution.default (convolution_27)}: [t489 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_27}] =
        permute x=t488 {derived} perm=[H<-C, W<-H, C<-W]
    group g48 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n245 {derived}: [t490 f32 [H=14 W=14 C=120] {derived}] =
        permute x=t489 {pt2=root:convolution_27} perm=[H<-W, W<-C, C<-H]
      n246 {derived}: [t491 f32 [H=14 W=14 C=120] {derived}] =
        batch_norm
          x=t490 {derived}
          weight=t74 {pt2=root:p_features_7_block_1_1_weight target=features.7.block.1.1.weight}
          bias=t75 {pt2=root:p_features_7_block_1_1_bias target=features.7.block.1.1.bias}
          running_mean=t199 {pt2=root:b_features_7_block_1_1_running_mean target=features.7.block.1.1.running_mean}
          running_var=t200 {pt2=root:b_features_7_block_1_1_running_var target=features.7.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n247 {pt2=root[123] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_19)}: [t492 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_57}] =
        permute x=t491 {derived} perm=[H<-C, W<-H, C<-W]
    n248 {pt2=root[124] torch.ops.aten.add.Tensor (add_15)}: [t493 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:add_15}] =
      add_scalar x=t492 {pt2=root:getitem_57} scalar=3
    n249 {pt2=root[125] torch.ops.aten.clamp.default (clamp_24)}: [t494 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_24}] =
      clamp x=t493 {pt2=root:add_15} params={min=0; max=none}
    n250 {pt2=root[126] torch.ops.aten.clamp.default (clamp_25)}: [t495 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_25}] =
      clamp x=t494 {pt2=root:clamp_24} params={min=none; max=6}
    n251 {pt2=root[127] torch.ops.aten.mul.Tensor (mul_12)}: [t496 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:mul_12}] =
      mul a=t492 {pt2=root:getitem_57} b=t495 {pt2=root:clamp_25}
    n252 {pt2=root[128] torch.ops.aten.div.Tensor (div_12)}: [t497 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:div_12}] =
      div_scalar x=t496 {pt2=root:mul_12} scalar=6
    n253 {pt2=root[129] torch.ops.aten.mean.dim (mean_4)}: [t498 f32 [H=120 W=1
                                                                      C=1] {pt2=root:mean_4}] =
      mean x=t497 {pt2=root:div_12} params={dims=[C, W]; keepdim=true}
    group g49 torch.ops.aten.convolution.default:
      n254 {derived}: [t499 f32 [C=120] {derived}] =
        permute x=t498 {pt2=root:mean_4} perm=[H<-W, W<-C, C<-H]
      n255 {derived}: [t500 f32 [N=32 T=1 D=1 H=1 W=1 C=120] {derived}] =
        permute
          x=t76 {pt2=root:p_features_7_block_2_fc1_weight target=features.7.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n256 {derived}: [t501 f32 [C=32] {derived}] =
        convolution
          x=t499 {derived}
          weight=t500 {derived}
          bias=t77 {pt2=root:p_features_7_block_2_fc1_bias target=features.7.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n257 {pt2=root[130] torch.ops.aten.convolution.default (convolution_28)}: [t502 f32 [H=32
                                                                      W=1 C=1] {pt2=root:convolution_28}] =
        permute x=t501 {derived} perm=[H<-C, W<-H, C<-W]
    n258 {pt2=root[131] torch.ops.aten.relu.default (relu_9)}: [t503 f32 [H=32
                                                                      W=1 C=1] {pt2=root:relu_9}] =
      relu x=t502 {pt2=root:convolution_28}
    group g50 torch.ops.aten.convolution.default:
      n259 {derived}: [t504 f32 [C=32] {derived}] =
        permute x=t503 {pt2=root:relu_9} perm=[H<-W, W<-C, C<-H]
      n260 {derived}: [t505 f32 [N=120 T=1 D=1 H=1 W=1 C=32] {derived}] =
        permute
          x=t78 {pt2=root:p_features_7_block_2_fc2_weight target=features.7.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n261 {derived}: [t506 f32 [C=120] {derived}] =
        convolution
          x=t504 {derived}
          weight=t505 {derived}
          bias=t79 {pt2=root:p_features_7_block_2_fc2_bias target=features.7.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n262 {pt2=root[132] torch.ops.aten.convolution.default (convolution_29)}: [t507 f32 [H=120
                                                                      W=1 C=1] {pt2=root:convolution_29}] =
        permute x=t506 {derived} perm=[H<-C, W<-H, C<-W]
    n263 {pt2=root[133] torch.ops.aten.add.Tensor (add_16)}: [t508 f32 [H=120
                                                                      W=1 C=1] {pt2=root:add_16}] =
      add_scalar x=t507 {pt2=root:convolution_29} scalar=3
    n264 {pt2=root[134] torch.ops.aten.clamp.default (clamp_26)}: [t509 f32 [H=120
                                                                      W=1 C=1] {pt2=root:clamp_26}] =
      clamp x=t508 {pt2=root:add_16} params={min=0; max=none}
    n265 {pt2=root[135] torch.ops.aten.clamp.default (clamp_27)}: [t510 f32 [H=120
                                                                      W=1 C=1] {pt2=root:clamp_27}] =
      clamp x=t509 {pt2=root:clamp_26} params={min=none; max=6}
    n266 {pt2=root[136] torch.ops.aten.div.Tensor (div_13)}: [t511 f32 [H=120
                                                                      W=1 C=1] {pt2=root:div_13}] =
      div_scalar x=t510 {pt2=root:clamp_27} scalar=6
    n267 {pt2=root[137] torch.ops.aten.mul.Tensor (mul_13)}: [t512 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:mul_13}] =
      mul a=t511 {pt2=root:div_13} b=t497 {pt2=root:div_12}
    group g51 torch.ops.aten.convolution.default:
      n268 {derived}: [t513 f32 [H=14 W=14 C=120] {derived}] =
        permute x=t512 {pt2=root:mul_13} perm=[H<-W, W<-C, C<-H]
      n269 {derived}: [t514 f32 [N=48 T=1 D=1 H=1 W=1 C=120] {derived}] =
        permute
          x=t80 {pt2=root:p_features_7_block_3_0_weight target=features.7.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n270 {derived}: [t515 f32 [H=14 W=14 C=48] {derived}] =
        convolution
          x=t513 {derived}
          weight=t514 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n271 {pt2=root[138] torch.ops.aten.convolution.default (convolution_30)}: [t516 f32 [H=48
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_30}] =
        permute x=t515 {derived} perm=[H<-C, W<-H, C<-W]
    group g52 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n272 {derived}: [t517 f32 [H=14 W=14 C=48] {derived}] =
        permute x=t516 {pt2=root:convolution_30} perm=[H<-W, W<-C, C<-H]
      n273 {derived}: [t518 f32 [H=14 W=14 C=48] {derived}] =
        batch_norm
          x=t517 {derived}
          weight=t81 {pt2=root:p_features_7_block_3_1_weight target=features.7.block.3.1.weight}
          bias=t82 {pt2=root:p_features_7_block_3_1_bias target=features.7.block.3.1.bias}
          running_mean=t202 {pt2=root:b_features_7_block_3_1_running_mean target=features.7.block.3.1.running_mean}
          running_var=t203 {pt2=root:b_features_7_block_3_1_running_var target=features.7.block.3.1.running_var}
          params={channel=C; eps=0.001}
      n274 {pt2=root[139] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_20)}: [t519 f32 [H=48
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_60}] =
        permute x=t518 {derived} perm=[H<-C, W<-H, C<-W]
    group g53 torch.ops.aten.convolution.default:
      n275 {derived}: [t520 f32 [H=14 W=14 C=48] {derived}] =
        permute x=t519 {pt2=root:getitem_60} perm=[H<-W, W<-C, C<-H]
      n276 {derived}: [t521 f32 [N=144 T=1 D=1 H=1 W=1 C=48] {derived}] =
        permute
          x=t83 {pt2=root:p_features_8_block_0_0_weight target=features.8.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n277 {derived}: [t522 f32 [H=14 W=14 C=144] {derived}] =
        convolution
          x=t520 {derived}
          weight=t521 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n278 {pt2=root[140] torch.ops.aten.convolution.default (convolution_31)}: [t523 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_31}] =
        permute x=t522 {derived} perm=[H<-C, W<-H, C<-W]
    group g54 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n279 {derived}: [t524 f32 [H=14 W=14 C=144] {derived}] =
        permute x=t523 {pt2=root:convolution_31} perm=[H<-W, W<-C, C<-H]
      n280 {derived}: [t525 f32 [H=14 W=14 C=144] {derived}] =
        batch_norm
          x=t524 {derived}
          weight=t84 {pt2=root:p_features_8_block_0_1_weight target=features.8.block.0.1.weight}
          bias=t85 {pt2=root:p_features_8_block_0_1_bias target=features.8.block.0.1.bias}
          running_mean=t205 {pt2=root:b_features_8_block_0_1_running_mean target=features.8.block.0.1.running_mean}
          running_var=t206 {pt2=root:b_features_8_block_0_1_running_var target=features.8.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n281 {pt2=root[141] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_21)}: [t526 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_63}] =
        permute x=t525 {derived} perm=[H<-C, W<-H, C<-W]
    n282 {pt2=root[142] torch.ops.aten.add.Tensor (add_17)}: [t527 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:add_17}] =
      add_scalar x=t526 {pt2=root:getitem_63} scalar=3
    n283 {pt2=root[143] torch.ops.aten.clamp.default (clamp_28)}: [t528 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_28}] =
      clamp x=t527 {pt2=root:add_17} params={min=0; max=none}
    n284 {pt2=root[144] torch.ops.aten.clamp.default (clamp_29)}: [t529 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_29}] =
      clamp x=t528 {pt2=root:clamp_28} params={min=none; max=6}
    n285 {pt2=root[145] torch.ops.aten.mul.Tensor (mul_14)}: [t530 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:mul_14}] =
      mul a=t526 {pt2=root:getitem_63} b=t529 {pt2=root:clamp_29}
    n286 {pt2=root[146] torch.ops.aten.div.Tensor (div_14)}: [t531 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:div_14}] =
      div_scalar x=t530 {pt2=root:mul_14} scalar=6
    group g55 torch.ops.aten.convolution.default:
      n287 {derived}: [t532 f32 [H=14 W=14 C=144] {derived}] =
        permute x=t531 {pt2=root:div_14} perm=[H<-W, W<-C, C<-H]
      n288 {derived}: [t533 f32 [N=144 T=1 D=1 H=5 W=5 C=1] {derived}] =
        permute
          x=t86 {pt2=root:p_features_8_block_1_0_weight target=features.8.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n289 {derived}: [t534 f32 [H=14 W=14 C=144] {derived}] =
        convolution
          x=t532 {derived}
          weight=t533 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=2; w=2};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=144}
      n290 {pt2=root[147] torch.ops.aten.convolution.default (convolution_32)}: [t535 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_32}] =
        permute x=t534 {derived} perm=[H<-C, W<-H, C<-W]
    group g56 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n291 {derived}: [t536 f32 [H=14 W=14 C=144] {derived}] =
        permute x=t535 {pt2=root:convolution_32} perm=[H<-W, W<-C, C<-H]
      n292 {derived}: [t537 f32 [H=14 W=14 C=144] {derived}] =
        batch_norm
          x=t536 {derived}
          weight=t87 {pt2=root:p_features_8_block_1_1_weight target=features.8.block.1.1.weight}
          bias=t88 {pt2=root:p_features_8_block_1_1_bias target=features.8.block.1.1.bias}
          running_mean=t208 {pt2=root:b_features_8_block_1_1_running_mean target=features.8.block.1.1.running_mean}
          running_var=t209 {pt2=root:b_features_8_block_1_1_running_var target=features.8.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n293 {pt2=root[148] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_22)}: [t538 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_66}] =
        permute x=t537 {derived} perm=[H<-C, W<-H, C<-W]
    n294 {pt2=root[149] torch.ops.aten.add.Tensor (add_18)}: [t539 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:add_18}] =
      add_scalar x=t538 {pt2=root:getitem_66} scalar=3
    n295 {pt2=root[150] torch.ops.aten.clamp.default (clamp_30)}: [t540 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_30}] =
      clamp x=t539 {pt2=root:add_18} params={min=0; max=none}
    n296 {pt2=root[151] torch.ops.aten.clamp.default (clamp_31)}: [t541 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_31}] =
      clamp x=t540 {pt2=root:clamp_30} params={min=none; max=6}
    n297 {pt2=root[152] torch.ops.aten.mul.Tensor (mul_15)}: [t542 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:mul_15}] =
      mul a=t538 {pt2=root:getitem_66} b=t541 {pt2=root:clamp_31}
    n298 {pt2=root[153] torch.ops.aten.div.Tensor (div_15)}: [t543 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:div_15}] =
      div_scalar x=t542 {pt2=root:mul_15} scalar=6
    n299 {pt2=root[154] torch.ops.aten.mean.dim (mean_5)}: [t544 f32 [H=144 W=1
                                                                      C=1] {pt2=root:mean_5}] =
      mean x=t543 {pt2=root:div_15} params={dims=[C, W]; keepdim=true}
    group g57 torch.ops.aten.convolution.default:
      n300 {derived}: [t545 f32 [C=144] {derived}] =
        permute x=t544 {pt2=root:mean_5} perm=[H<-W, W<-C, C<-H]
      n301 {derived}: [t546 f32 [N=40 T=1 D=1 H=1 W=1 C=144] {derived}] =
        permute
          x=t89 {pt2=root:p_features_8_block_2_fc1_weight target=features.8.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n302 {derived}: [t547 f32 [C=40] {derived}] =
        convolution
          x=t545 {derived}
          weight=t546 {derived}
          bias=t90 {pt2=root:p_features_8_block_2_fc1_bias target=features.8.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n303 {pt2=root[155] torch.ops.aten.convolution.default (convolution_33)}: [t548 f32 [H=40
                                                                      W=1 C=1] {pt2=root:convolution_33}] =
        permute x=t547 {derived} perm=[H<-C, W<-H, C<-W]
    n304 {pt2=root[156] torch.ops.aten.relu.default (relu_10)}: [t549 f32 [H=40
                                                                      W=1 C=1] {pt2=root:relu_10}] =
      relu x=t548 {pt2=root:convolution_33}
    group g58 torch.ops.aten.convolution.default:
      n305 {derived}: [t550 f32 [C=40] {derived}] =
        permute x=t549 {pt2=root:relu_10} perm=[H<-W, W<-C, C<-H]
      n306 {derived}: [t551 f32 [N=144 T=1 D=1 H=1 W=1 C=40] {derived}] =
        permute
          x=t91 {pt2=root:p_features_8_block_2_fc2_weight target=features.8.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n307 {derived}: [t552 f32 [C=144] {derived}] =
        convolution
          x=t550 {derived}
          weight=t551 {derived}
          bias=t92 {pt2=root:p_features_8_block_2_fc2_bias target=features.8.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n308 {pt2=root[157] torch.ops.aten.convolution.default (convolution_34)}: [t553 f32 [H=144
                                                                      W=1 C=1] {pt2=root:convolution_34}] =
        permute x=t552 {derived} perm=[H<-C, W<-H, C<-W]
    n309 {pt2=root[158] torch.ops.aten.add.Tensor (add_19)}: [t554 f32 [H=144
                                                                      W=1 C=1] {pt2=root:add_19}] =
      add_scalar x=t553 {pt2=root:convolution_34} scalar=3
    n310 {pt2=root[159] torch.ops.aten.clamp.default (clamp_32)}: [t555 f32 [H=144
                                                                      W=1 C=1] {pt2=root:clamp_32}] =
      clamp x=t554 {pt2=root:add_19} params={min=0; max=none}
    n311 {pt2=root[160] torch.ops.aten.clamp.default (clamp_33)}: [t556 f32 [H=144
                                                                      W=1 C=1] {pt2=root:clamp_33}] =
      clamp x=t555 {pt2=root:clamp_32} params={min=none; max=6}
    n312 {pt2=root[161] torch.ops.aten.div.Tensor (div_16)}: [t557 f32 [H=144
                                                                      W=1 C=1] {pt2=root:div_16}] =
      div_scalar x=t556 {pt2=root:clamp_33} scalar=6
    n313 {pt2=root[162] torch.ops.aten.mul.Tensor (mul_16)}: [t558 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:mul_16}] =
      mul a=t557 {pt2=root:div_16} b=t543 {pt2=root:div_15}
    group g59 torch.ops.aten.convolution.default:
      n314 {derived}: [t559 f32 [H=14 W=14 C=144] {derived}] =
        permute x=t558 {pt2=root:mul_16} perm=[H<-W, W<-C, C<-H]
      n315 {derived}: [t560 f32 [N=48 T=1 D=1 H=1 W=1 C=144] {derived}] =
        permute
          x=t93 {pt2=root:p_features_8_block_3_0_weight target=features.8.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n316 {derived}: [t561 f32 [H=14 W=14 C=48] {derived}] =
        convolution
          x=t559 {derived}
          weight=t560 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n317 {pt2=root[163] torch.ops.aten.convolution.default (convolution_35)}: [t562 f32 [H=48
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_35}] =
        permute x=t561 {derived} perm=[H<-C, W<-H, C<-W]
    group g60 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n318 {derived}: [t563 f32 [H=14 W=14 C=48] {derived}] =
        permute x=t562 {pt2=root:convolution_35} perm=[H<-W, W<-C, C<-H]
      n319 {derived}: [t564 f32 [H=14 W=14 C=48] {derived}] =
        batch_norm
          x=t563 {derived}
          weight=t94 {pt2=root:p_features_8_block_3_1_weight target=features.8.block.3.1.weight}
          bias=t95 {pt2=root:p_features_8_block_3_1_bias target=features.8.block.3.1.bias}
          running_mean=t211 {pt2=root:b_features_8_block_3_1_running_mean target=features.8.block.3.1.running_mean}
          running_var=t212 {pt2=root:b_features_8_block_3_1_running_var target=features.8.block.3.1.running_var}
          params={channel=C; eps=0.001}
      n320 {pt2=root[164] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_23)}: [t565 f32 [H=48
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_69}] =
        permute x=t564 {derived} perm=[H<-C, W<-H, C<-W]
    n321 {pt2=root[165] torch.ops.aten.add.Tensor (add_20)}: [t566 f32 [H=48
                                                                      W=14
                                                                      C=14] {pt2=root:add_20}] =
      add a=t565 {pt2=root:getitem_69} b=t519 {pt2=root:getitem_60}
    group g61 torch.ops.aten.convolution.default:
      n322 {derived}: [t567 f32 [H=14 W=14 C=48] {derived}] =
        permute x=t566 {pt2=root:add_20} perm=[H<-W, W<-C, C<-H]
      n323 {derived}: [t568 f32 [N=288 T=1 D=1 H=1 W=1 C=48] {derived}] =
        permute
          x=t96 {pt2=root:p_features_9_block_0_0_weight target=features.9.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n324 {derived}: [t569 f32 [H=14 W=14 C=288] {derived}] =
        convolution
          x=t567 {derived}
          weight=t568 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n325 {pt2=root[166] torch.ops.aten.convolution.default (convolution_36)}: [t570 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_36}] =
        permute x=t569 {derived} perm=[H<-C, W<-H, C<-W]
    group g62 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n326 {derived}: [t571 f32 [H=14 W=14 C=288] {derived}] =
        permute x=t570 {pt2=root:convolution_36} perm=[H<-W, W<-C, C<-H]
      n327 {derived}: [t572 f32 [H=14 W=14 C=288] {derived}] =
        batch_norm
          x=t571 {derived}
          weight=t97 {pt2=root:p_features_9_block_0_1_weight target=features.9.block.0.1.weight}
          bias=t98 {pt2=root:p_features_9_block_0_1_bias target=features.9.block.0.1.bias}
          running_mean=t214 {pt2=root:b_features_9_block_0_1_running_mean target=features.9.block.0.1.running_mean}
          running_var=t215 {pt2=root:b_features_9_block_0_1_running_var target=features.9.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n328 {pt2=root[167] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_24)}: [t573 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_72}] =
        permute x=t572 {derived} perm=[H<-C, W<-H, C<-W]
    n329 {pt2=root[168] torch.ops.aten.add.Tensor (add_21)}: [t574 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:add_21}] =
      add_scalar x=t573 {pt2=root:getitem_72} scalar=3
    n330 {pt2=root[169] torch.ops.aten.clamp.default (clamp_34)}: [t575 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_34}] =
      clamp x=t574 {pt2=root:add_21} params={min=0; max=none}
    n331 {pt2=root[170] torch.ops.aten.clamp.default (clamp_35)}: [t576 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:clamp_35}] =
      clamp x=t575 {pt2=root:clamp_34} params={min=none; max=6}
    n332 {pt2=root[171] torch.ops.aten.mul.Tensor (mul_17)}: [t577 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:mul_17}] =
      mul a=t573 {pt2=root:getitem_72} b=t576 {pt2=root:clamp_35}
    n333 {pt2=root[172] torch.ops.aten.div.Tensor (div_17)}: [t578 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:div_17}] =
      div_scalar x=t577 {pt2=root:mul_17} scalar=6
    group g63 torch.ops.aten.convolution.default:
      n334 {derived}: [t579 f32 [H=14 W=14 C=288] {derived}] =
        permute x=t578 {pt2=root:div_17} perm=[H<-W, W<-C, C<-H]
      n335 {derived}: [t580 f32 [N=288 T=1 D=1 H=5 W=5 C=1] {derived}] =
        permute
          x=t99 {pt2=root:p_features_9_block_1_0_weight target=features.9.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n336 {derived}: [t581 f32 [H=7 W=7 C=288] {derived}] =
        convolution
          x=t579 {derived}
          weight=t580 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=2; w=2};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=288}
      n337 {pt2=root[173] torch.ops.aten.convolution.default (convolution_37)}: [t582 f32 [H=288
                                                                      W=7 C=7] {pt2=root:convolution_37}] =
        permute x=t581 {derived} perm=[H<-C, W<-H, C<-W]
    group g64 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n338 {derived}: [t583 f32 [H=7 W=7 C=288] {derived}] =
        permute x=t582 {pt2=root:convolution_37} perm=[H<-W, W<-C, C<-H]
      n339 {derived}: [t584 f32 [H=7 W=7 C=288] {derived}] =
        batch_norm
          x=t583 {derived}
          weight=t100 {pt2=root:p_features_9_block_1_1_weight target=features.9.block.1.1.weight}
          bias=t101 {pt2=root:p_features_9_block_1_1_bias target=features.9.block.1.1.bias}
          running_mean=t217 {pt2=root:b_features_9_block_1_1_running_mean target=features.9.block.1.1.running_mean}
          running_var=t218 {pt2=root:b_features_9_block_1_1_running_var target=features.9.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n340 {pt2=root[174] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_25)}: [t585 f32 [H=288
                                                                      W=7 C=7] {pt2=root:getitem_75}] =
        permute x=t584 {derived} perm=[H<-C, W<-H, C<-W]
    n341 {pt2=root[175] torch.ops.aten.add.Tensor (add_22)}: [t586 f32 [H=288
                                                                      W=7 C=7] {pt2=root:add_22}] =
      add_scalar x=t585 {pt2=root:getitem_75} scalar=3
    n342 {pt2=root[176] torch.ops.aten.clamp.default (clamp_36)}: [t587 f32 [H=288
                                                                      W=7 C=7] {pt2=root:clamp_36}] =
      clamp x=t586 {pt2=root:add_22} params={min=0; max=none}
    n343 {pt2=root[177] torch.ops.aten.clamp.default (clamp_37)}: [t588 f32 [H=288
                                                                      W=7 C=7] {pt2=root:clamp_37}] =
      clamp x=t587 {pt2=root:clamp_36} params={min=none; max=6}
    n344 {pt2=root[178] torch.ops.aten.mul.Tensor (mul_18)}: [t589 f32 [H=288
                                                                      W=7 C=7] {pt2=root:mul_18}] =
      mul a=t585 {pt2=root:getitem_75} b=t588 {pt2=root:clamp_37}
    n345 {pt2=root[179] torch.ops.aten.div.Tensor (div_18)}: [t590 f32 [H=288
                                                                      W=7 C=7] {pt2=root:div_18}] =
      div_scalar x=t589 {pt2=root:mul_18} scalar=6
    n346 {pt2=root[180] torch.ops.aten.mean.dim (mean_6)}: [t591 f32 [H=288 W=1
                                                                      C=1] {pt2=root:mean_6}] =
      mean x=t590 {pt2=root:div_18} params={dims=[C, W]; keepdim=true}
    group g65 torch.ops.aten.convolution.default:
      n347 {derived}: [t592 f32 [C=288] {derived}] =
        permute x=t591 {pt2=root:mean_6} perm=[H<-W, W<-C, C<-H]
      n348 {derived}: [t593 f32 [N=72 T=1 D=1 H=1 W=1 C=288] {derived}] =
        permute
          x=t102 {pt2=root:p_features_9_block_2_fc1_weight target=features.9.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n349 {derived}: [t594 f32 [C=72] {derived}] =
        convolution
          x=t592 {derived}
          weight=t593 {derived}
          bias=t103 {pt2=root:p_features_9_block_2_fc1_bias target=features.9.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n350 {pt2=root[181] torch.ops.aten.convolution.default (convolution_38)}: [t595 f32 [H=72
                                                                      W=1 C=1] {pt2=root:convolution_38}] =
        permute x=t594 {derived} perm=[H<-C, W<-H, C<-W]
    n351 {pt2=root[182] torch.ops.aten.relu.default (relu_11)}: [t596 f32 [H=72
                                                                      W=1 C=1] {pt2=root:relu_11}] =
      relu x=t595 {pt2=root:convolution_38}
    group g66 torch.ops.aten.convolution.default:
      n352 {derived}: [t597 f32 [C=72] {derived}] =
        permute x=t596 {pt2=root:relu_11} perm=[H<-W, W<-C, C<-H]
      n353 {derived}: [t598 f32 [N=288 T=1 D=1 H=1 W=1 C=72] {derived}] =
        permute
          x=t104 {pt2=root:p_features_9_block_2_fc2_weight target=features.9.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n354 {derived}: [t599 f32 [C=288] {derived}] =
        convolution
          x=t597 {derived}
          weight=t598 {derived}
          bias=t105 {pt2=root:p_features_9_block_2_fc2_bias target=features.9.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n355 {pt2=root[183] torch.ops.aten.convolution.default (convolution_39)}: [t600 f32 [H=288
                                                                      W=1 C=1] {pt2=root:convolution_39}] =
        permute x=t599 {derived} perm=[H<-C, W<-H, C<-W]
    n356 {pt2=root[184] torch.ops.aten.add.Tensor (add_23)}: [t601 f32 [H=288
                                                                      W=1 C=1] {pt2=root:add_23}] =
      add_scalar x=t600 {pt2=root:convolution_39} scalar=3
    n357 {pt2=root[185] torch.ops.aten.clamp.default (clamp_38)}: [t602 f32 [H=288
                                                                      W=1 C=1] {pt2=root:clamp_38}] =
      clamp x=t601 {pt2=root:add_23} params={min=0; max=none}
    n358 {pt2=root[186] torch.ops.aten.clamp.default (clamp_39)}: [t603 f32 [H=288
                                                                      W=1 C=1] {pt2=root:clamp_39}] =
      clamp x=t602 {pt2=root:clamp_38} params={min=none; max=6}
    n359 {pt2=root[187] torch.ops.aten.div.Tensor (div_19)}: [t604 f32 [H=288
                                                                      W=1 C=1] {pt2=root:div_19}] =
      div_scalar x=t603 {pt2=root:clamp_39} scalar=6
    n360 {pt2=root[188] torch.ops.aten.mul.Tensor (mul_19)}: [t605 f32 [H=288
                                                                      W=7 C=7] {pt2=root:mul_19}] =
      mul a=t604 {pt2=root:div_19} b=t590 {pt2=root:div_18}
    group g67 torch.ops.aten.convolution.default:
      n361 {derived}: [t606 f32 [H=7 W=7 C=288] {derived}] =
        permute x=t605 {pt2=root:mul_19} perm=[H<-W, W<-C, C<-H]
      n362 {derived}: [t607 f32 [N=96 T=1 D=1 H=1 W=1 C=288] {derived}] =
        permute
          x=t106 {pt2=root:p_features_9_block_3_0_weight target=features.9.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n363 {derived}: [t608 f32 [H=7 W=7 C=96] {derived}] =
        convolution
          x=t606 {derived}
          weight=t607 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n364 {pt2=root[189] torch.ops.aten.convolution.default (convolution_40)}: [t609 f32 [H=96
                                                                      W=7 C=7] {pt2=root:convolution_40}] =
        permute x=t608 {derived} perm=[H<-C, W<-H, C<-W]
    group g68 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n365 {derived}: [t610 f32 [H=7 W=7 C=96] {derived}] =
        permute x=t609 {pt2=root:convolution_40} perm=[H<-W, W<-C, C<-H]
      n366 {derived}: [t611 f32 [H=7 W=7 C=96] {derived}] =
        batch_norm
          x=t610 {derived}
          weight=t107 {pt2=root:p_features_9_block_3_1_weight target=features.9.block.3.1.weight}
          bias=t108 {pt2=root:p_features_9_block_3_1_bias target=features.9.block.3.1.bias}
          running_mean=t220 {pt2=root:b_features_9_block_3_1_running_mean target=features.9.block.3.1.running_mean}
          running_var=t221 {pt2=root:b_features_9_block_3_1_running_var target=features.9.block.3.1.running_var}
          params={channel=C; eps=0.001}
      n367 {pt2=root[190] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_26)}: [t612 f32 [H=96
                                                                      W=7 C=7] {pt2=root:getitem_78}] =
        permute x=t611 {derived} perm=[H<-C, W<-H, C<-W]
    group g69 torch.ops.aten.convolution.default:
      n368 {derived}: [t613 f32 [H=7 W=7 C=96] {derived}] =
        permute x=t612 {pt2=root:getitem_78} perm=[H<-W, W<-C, C<-H]
      n369 {derived}: [t614 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t109 {pt2=root:p_features_10_block_0_0_weight target=features.10.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n370 {derived}: [t615 f32 [H=7 W=7 C=576] {derived}] =
        convolution
          x=t613 {derived}
          weight=t614 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n371 {pt2=root[191] torch.ops.aten.convolution.default (convolution_41)}: [t616 f32 [H=576
                                                                      W=7 C=7] {pt2=root:convolution_41}] =
        permute x=t615 {derived} perm=[H<-C, W<-H, C<-W]
    group g70 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n372 {derived}: [t617 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t616 {pt2=root:convolution_41} perm=[H<-W, W<-C, C<-H]
      n373 {derived}: [t618 f32 [H=7 W=7 C=576] {derived}] =
        batch_norm
          x=t617 {derived}
          weight=t110 {pt2=root:p_features_10_block_0_1_weight target=features.10.block.0.1.weight}
          bias=t111 {pt2=root:p_features_10_block_0_1_bias target=features.10.block.0.1.bias}
          running_mean=t223 {pt2=root:b_features_10_block_0_1_running_mean target=features.10.block.0.1.running_mean}
          running_var=t224 {pt2=root:b_features_10_block_0_1_running_var target=features.10.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n374 {pt2=root[192] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_27)}: [t619 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_81}] =
        permute x=t618 {derived} perm=[H<-C, W<-H, C<-W]
    n375 {pt2=root[193] torch.ops.aten.add.Tensor (add_24)}: [t620 f32 [H=576
                                                                      W=7 C=7] {pt2=root:add_24}] =
      add_scalar x=t619 {pt2=root:getitem_81} scalar=3
    n376 {pt2=root[194] torch.ops.aten.clamp.default (clamp_40)}: [t621 f32 [H=576
                                                                      W=7 C=7] {pt2=root:clamp_40}] =
      clamp x=t620 {pt2=root:add_24} params={min=0; max=none}
    n377 {pt2=root[195] torch.ops.aten.clamp.default (clamp_41)}: [t622 f32 [H=576
                                                                      W=7 C=7] {pt2=root:clamp_41}] =
      clamp x=t621 {pt2=root:clamp_40} params={min=none; max=6}
    n378 {pt2=root[196] torch.ops.aten.mul.Tensor (mul_20)}: [t623 f32 [H=576
                                                                      W=7 C=7] {pt2=root:mul_20}] =
      mul a=t619 {pt2=root:getitem_81} b=t622 {pt2=root:clamp_41}
    n379 {pt2=root[197] torch.ops.aten.div.Tensor (div_20)}: [t624 f32 [H=576
                                                                      W=7 C=7] {pt2=root:div_20}] =
      div_scalar x=t623 {pt2=root:mul_20} scalar=6
    group g71 torch.ops.aten.convolution.default:
      n380 {derived}: [t625 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t624 {pt2=root:div_20} perm=[H<-W, W<-C, C<-H]
      n381 {derived}: [t626 f32 [N=576 T=1 D=1 H=5 W=5 C=1] {derived}] =
        permute
          x=t112 {pt2=root:p_features_10_block_1_0_weight target=features.10.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n382 {derived}: [t627 f32 [H=7 W=7 C=576] {derived}] =
        convolution
          x=t625 {derived}
          weight=t626 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=2; w=2};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=576}
      n383 {pt2=root[198] torch.ops.aten.convolution.default (convolution_42)}: [t628 f32 [H=576
                                                                      W=7 C=7] {pt2=root:convolution_42}] =
        permute x=t627 {derived} perm=[H<-C, W<-H, C<-W]
    group g72 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n384 {derived}: [t629 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t628 {pt2=root:convolution_42} perm=[H<-W, W<-C, C<-H]
      n385 {derived}: [t630 f32 [H=7 W=7 C=576] {derived}] =
        batch_norm
          x=t629 {derived}
          weight=t113 {pt2=root:p_features_10_block_1_1_weight target=features.10.block.1.1.weight}
          bias=t114 {pt2=root:p_features_10_block_1_1_bias target=features.10.block.1.1.bias}
          running_mean=t226 {pt2=root:b_features_10_block_1_1_running_mean target=features.10.block.1.1.running_mean}
          running_var=t227 {pt2=root:b_features_10_block_1_1_running_var target=features.10.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n386 {pt2=root[199] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_28)}: [t631 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_84}] =
        permute x=t630 {derived} perm=[H<-C, W<-H, C<-W]
    n387 {pt2=root[200] torch.ops.aten.add.Tensor (add_25)}: [t632 f32 [H=576
                                                                      W=7 C=7] {pt2=root:add_25}] =
      add_scalar x=t631 {pt2=root:getitem_84} scalar=3
    n388 {pt2=root[201] torch.ops.aten.clamp.default (clamp_42)}: [t633 f32 [H=576
                                                                      W=7 C=7] {pt2=root:clamp_42}] =
      clamp x=t632 {pt2=root:add_25} params={min=0; max=none}
    n389 {pt2=root[202] torch.ops.aten.clamp.default (clamp_43)}: [t634 f32 [H=576
                                                                      W=7 C=7] {pt2=root:clamp_43}] =
      clamp x=t633 {pt2=root:clamp_42} params={min=none; max=6}
    n390 {pt2=root[203] torch.ops.aten.mul.Tensor (mul_21)}: [t635 f32 [H=576
                                                                      W=7 C=7] {pt2=root:mul_21}] =
      mul a=t631 {pt2=root:getitem_84} b=t634 {pt2=root:clamp_43}
    n391 {pt2=root[204] torch.ops.aten.div.Tensor (div_21)}: [t636 f32 [H=576
                                                                      W=7 C=7] {pt2=root:div_21}] =
      div_scalar x=t635 {pt2=root:mul_21} scalar=6
    n392 {pt2=root[205] torch.ops.aten.mean.dim (mean_7)}: [t637 f32 [H=576 W=1
                                                                      C=1] {pt2=root:mean_7}] =
      mean x=t636 {pt2=root:div_21} params={dims=[C, W]; keepdim=true}
    group g73 torch.ops.aten.convolution.default:
      n393 {derived}: [t638 f32 [C=576] {derived}] =
        permute x=t637 {pt2=root:mean_7} perm=[H<-W, W<-C, C<-H]
      n394 {derived}: [t639 f32 [N=144 T=1 D=1 H=1 W=1 C=576] {derived}] =
        permute
          x=t115 {pt2=root:p_features_10_block_2_fc1_weight target=features.10.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n395 {derived}: [t640 f32 [C=144] {derived}] =
        convolution
          x=t638 {derived}
          weight=t639 {derived}
          bias=t116 {pt2=root:p_features_10_block_2_fc1_bias target=features.10.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n396 {pt2=root[206] torch.ops.aten.convolution.default (convolution_43)}: [t641 f32 [H=144
                                                                      W=1 C=1] {pt2=root:convolution_43}] =
        permute x=t640 {derived} perm=[H<-C, W<-H, C<-W]
    n397 {pt2=root[207] torch.ops.aten.relu.default (relu_12)}: [t642 f32 [H=144
                                                                      W=1 C=1] {pt2=root:relu_12}] =
      relu x=t641 {pt2=root:convolution_43}
    group g74 torch.ops.aten.convolution.default:
      n398 {derived}: [t643 f32 [C=144] {derived}] =
        permute x=t642 {pt2=root:relu_12} perm=[H<-W, W<-C, C<-H]
      n399 {derived}: [t644 f32 [N=576 T=1 D=1 H=1 W=1 C=144] {derived}] =
        permute
          x=t117 {pt2=root:p_features_10_block_2_fc2_weight target=features.10.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n400 {derived}: [t645 f32 [C=576] {derived}] =
        convolution
          x=t643 {derived}
          weight=t644 {derived}
          bias=t118 {pt2=root:p_features_10_block_2_fc2_bias target=features.10.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n401 {pt2=root[208] torch.ops.aten.convolution.default (convolution_44)}: [t646 f32 [H=576
                                                                      W=1 C=1] {pt2=root:convolution_44}] =
        permute x=t645 {derived} perm=[H<-C, W<-H, C<-W]
    n402 {pt2=root[209] torch.ops.aten.add.Tensor (add_26)}: [t647 f32 [H=576
                                                                      W=1 C=1] {pt2=root:add_26}] =
      add_scalar x=t646 {pt2=root:convolution_44} scalar=3
    n403 {pt2=root[210] torch.ops.aten.clamp.default (clamp_44)}: [t648 f32 [H=576
                                                                      W=1 C=1] {pt2=root:clamp_44}] =
      clamp x=t647 {pt2=root:add_26} params={min=0; max=none}
    n404 {pt2=root[211] torch.ops.aten.clamp.default (clamp_45)}: [t649 f32 [H=576
                                                                      W=1 C=1] {pt2=root:clamp_45}] =
      clamp x=t648 {pt2=root:clamp_44} params={min=none; max=6}
    n405 {pt2=root[212] torch.ops.aten.div.Tensor (div_22)}: [t650 f32 [H=576
                                                                      W=1 C=1] {pt2=root:div_22}] =
      div_scalar x=t649 {pt2=root:clamp_45} scalar=6
    n406 {pt2=root[213] torch.ops.aten.mul.Tensor (mul_22)}: [t651 f32 [H=576
                                                                      W=7 C=7] {pt2=root:mul_22}] =
      mul a=t650 {pt2=root:div_22} b=t636 {pt2=root:div_21}
    group g75 torch.ops.aten.convolution.default:
      n407 {derived}: [t652 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t651 {pt2=root:mul_22} perm=[H<-W, W<-C, C<-H]
      n408 {derived}: [t653 f32 [N=96 T=1 D=1 H=1 W=1 C=576] {derived}] =
        permute
          x=t119 {pt2=root:p_features_10_block_3_0_weight target=features.10.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n409 {derived}: [t654 f32 [H=7 W=7 C=96] {derived}] =
        convolution
          x=t652 {derived}
          weight=t653 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n410 {pt2=root[214] torch.ops.aten.convolution.default (convolution_45)}: [t655 f32 [H=96
                                                                      W=7 C=7] {pt2=root:convolution_45}] =
        permute x=t654 {derived} perm=[H<-C, W<-H, C<-W]
    group g76 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n411 {derived}: [t656 f32 [H=7 W=7 C=96] {derived}] =
        permute x=t655 {pt2=root:convolution_45} perm=[H<-W, W<-C, C<-H]
      n412 {derived}: [t657 f32 [H=7 W=7 C=96] {derived}] =
        batch_norm
          x=t656 {derived}
          weight=t120 {pt2=root:p_features_10_block_3_1_weight target=features.10.block.3.1.weight}
          bias=t121 {pt2=root:p_features_10_block_3_1_bias target=features.10.block.3.1.bias}
          running_mean=t229 {pt2=root:b_features_10_block_3_1_running_mean target=features.10.block.3.1.running_mean}
          running_var=t230 {pt2=root:b_features_10_block_3_1_running_var target=features.10.block.3.1.running_var}
          params={channel=C; eps=0.001}
      n413 {pt2=root[215] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_29)}: [t658 f32 [H=96
                                                                      W=7 C=7] {pt2=root:getitem_87}] =
        permute x=t657 {derived} perm=[H<-C, W<-H, C<-W]
    n414 {pt2=root[216] torch.ops.aten.add.Tensor (add_27)}: [t659 f32 [H=96
                                                                      W=7 C=7] {pt2=root:add_27}] =
      add a=t658 {pt2=root:getitem_87} b=t612 {pt2=root:getitem_78}
    group g77 torch.ops.aten.convolution.default:
      n415 {derived}: [t660 f32 [H=7 W=7 C=96] {derived}] =
        permute x=t659 {pt2=root:add_27} perm=[H<-W, W<-C, C<-H]
      n416 {derived}: [t661 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t122 {pt2=root:p_features_11_block_0_0_weight target=features.11.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n417 {derived}: [t662 f32 [H=7 W=7 C=576] {derived}] =
        convolution
          x=t660 {derived}
          weight=t661 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n418 {pt2=root[217] torch.ops.aten.convolution.default (convolution_46)}: [t663 f32 [H=576
                                                                      W=7 C=7] {pt2=root:convolution_46}] =
        permute x=t662 {derived} perm=[H<-C, W<-H, C<-W]
    group g78 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n419 {derived}: [t664 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t663 {pt2=root:convolution_46} perm=[H<-W, W<-C, C<-H]
      n420 {derived}: [t665 f32 [H=7 W=7 C=576] {derived}] =
        batch_norm
          x=t664 {derived}
          weight=t123 {pt2=root:p_features_11_block_0_1_weight target=features.11.block.0.1.weight}
          bias=t124 {pt2=root:p_features_11_block_0_1_bias target=features.11.block.0.1.bias}
          running_mean=t232 {pt2=root:b_features_11_block_0_1_running_mean target=features.11.block.0.1.running_mean}
          running_var=t233 {pt2=root:b_features_11_block_0_1_running_var target=features.11.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n421 {pt2=root[218] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_30)}: [t666 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_90}] =
        permute x=t665 {derived} perm=[H<-C, W<-H, C<-W]
    n422 {pt2=root[219] torch.ops.aten.add.Tensor (add_28)}: [t667 f32 [H=576
                                                                      W=7 C=7] {pt2=root:add_28}] =
      add_scalar x=t666 {pt2=root:getitem_90} scalar=3
    n423 {pt2=root[220] torch.ops.aten.clamp.default (clamp_46)}: [t668 f32 [H=576
                                                                      W=7 C=7] {pt2=root:clamp_46}] =
      clamp x=t667 {pt2=root:add_28} params={min=0; max=none}
    n424 {pt2=root[221] torch.ops.aten.clamp.default (clamp_47)}: [t669 f32 [H=576
                                                                      W=7 C=7] {pt2=root:clamp_47}] =
      clamp x=t668 {pt2=root:clamp_46} params={min=none; max=6}
    n425 {pt2=root[222] torch.ops.aten.mul.Tensor (mul_23)}: [t670 f32 [H=576
                                                                      W=7 C=7] {pt2=root:mul_23}] =
      mul a=t666 {pt2=root:getitem_90} b=t669 {pt2=root:clamp_47}
    n426 {pt2=root[223] torch.ops.aten.div.Tensor (div_23)}: [t671 f32 [H=576
                                                                      W=7 C=7] {pt2=root:div_23}] =
      div_scalar x=t670 {pt2=root:mul_23} scalar=6
    group g79 torch.ops.aten.convolution.default:
      n427 {derived}: [t672 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t671 {pt2=root:div_23} perm=[H<-W, W<-C, C<-H]
      n428 {derived}: [t673 f32 [N=576 T=1 D=1 H=5 W=5 C=1] {derived}] =
        permute
          x=t125 {pt2=root:p_features_11_block_1_0_weight target=features.11.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n429 {derived}: [t674 f32 [H=7 W=7 C=576] {derived}] =
        convolution
          x=t672 {derived}
          weight=t673 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=2; w=2};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=576}
      n430 {pt2=root[224] torch.ops.aten.convolution.default (convolution_47)}: [t675 f32 [H=576
                                                                      W=7 C=7] {pt2=root:convolution_47}] =
        permute x=t674 {derived} perm=[H<-C, W<-H, C<-W]
    group g80 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n431 {derived}: [t676 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t675 {pt2=root:convolution_47} perm=[H<-W, W<-C, C<-H]
      n432 {derived}: [t677 f32 [H=7 W=7 C=576] {derived}] =
        batch_norm
          x=t676 {derived}
          weight=t126 {pt2=root:p_features_11_block_1_1_weight target=features.11.block.1.1.weight}
          bias=t127 {pt2=root:p_features_11_block_1_1_bias target=features.11.block.1.1.bias}
          running_mean=t235 {pt2=root:b_features_11_block_1_1_running_mean target=features.11.block.1.1.running_mean}
          running_var=t236 {pt2=root:b_features_11_block_1_1_running_var target=features.11.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n433 {pt2=root[225] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_31)}: [t678 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_93}] =
        permute x=t677 {derived} perm=[H<-C, W<-H, C<-W]
    n434 {pt2=root[226] torch.ops.aten.add.Tensor (add_29)}: [t679 f32 [H=576
                                                                      W=7 C=7] {pt2=root:add_29}] =
      add_scalar x=t678 {pt2=root:getitem_93} scalar=3
    n435 {pt2=root[227] torch.ops.aten.clamp.default (clamp_48)}: [t680 f32 [H=576
                                                                      W=7 C=7] {pt2=root:clamp_48}] =
      clamp x=t679 {pt2=root:add_29} params={min=0; max=none}
    n436 {pt2=root[228] torch.ops.aten.clamp.default (clamp_49)}: [t681 f32 [H=576
                                                                      W=7 C=7] {pt2=root:clamp_49}] =
      clamp x=t680 {pt2=root:clamp_48} params={min=none; max=6}
    n437 {pt2=root[229] torch.ops.aten.mul.Tensor (mul_24)}: [t682 f32 [H=576
                                                                      W=7 C=7] {pt2=root:mul_24}] =
      mul a=t678 {pt2=root:getitem_93} b=t681 {pt2=root:clamp_49}
    n438 {pt2=root[230] torch.ops.aten.div.Tensor (div_24)}: [t683 f32 [H=576
                                                                      W=7 C=7] {pt2=root:div_24}] =
      div_scalar x=t682 {pt2=root:mul_24} scalar=6
    n439 {pt2=root[231] torch.ops.aten.mean.dim (mean_8)}: [t684 f32 [H=576 W=1
                                                                      C=1] {pt2=root:mean_8}] =
      mean x=t683 {pt2=root:div_24} params={dims=[C, W]; keepdim=true}
    group g81 torch.ops.aten.convolution.default:
      n440 {derived}: [t685 f32 [C=576] {derived}] =
        permute x=t684 {pt2=root:mean_8} perm=[H<-W, W<-C, C<-H]
      n441 {derived}: [t686 f32 [N=144 T=1 D=1 H=1 W=1 C=576] {derived}] =
        permute
          x=t128 {pt2=root:p_features_11_block_2_fc1_weight target=features.11.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n442 {derived}: [t687 f32 [C=144] {derived}] =
        convolution
          x=t685 {derived}
          weight=t686 {derived}
          bias=t129 {pt2=root:p_features_11_block_2_fc1_bias target=features.11.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n443 {pt2=root[232] torch.ops.aten.convolution.default (convolution_48)}: [t688 f32 [H=144
                                                                      W=1 C=1] {pt2=root:convolution_48}] =
        permute x=t687 {derived} perm=[H<-C, W<-H, C<-W]
    n444 {pt2=root[233] torch.ops.aten.relu.default (relu_13)}: [t689 f32 [H=144
                                                                      W=1 C=1] {pt2=root:relu_13}] =
      relu x=t688 {pt2=root:convolution_48}
    group g82 torch.ops.aten.convolution.default:
      n445 {derived}: [t690 f32 [C=144] {derived}] =
        permute x=t689 {pt2=root:relu_13} perm=[H<-W, W<-C, C<-H]
      n446 {derived}: [t691 f32 [N=576 T=1 D=1 H=1 W=1 C=144] {derived}] =
        permute
          x=t130 {pt2=root:p_features_11_block_2_fc2_weight target=features.11.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n447 {derived}: [t692 f32 [C=576] {derived}] =
        convolution
          x=t690 {derived}
          weight=t691 {derived}
          bias=t131 {pt2=root:p_features_11_block_2_fc2_bias target=features.11.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n448 {pt2=root[234] torch.ops.aten.convolution.default (convolution_49)}: [t693 f32 [H=576
                                                                      W=1 C=1] {pt2=root:convolution_49}] =
        permute x=t692 {derived} perm=[H<-C, W<-H, C<-W]
    n449 {pt2=root[235] torch.ops.aten.add.Tensor (add_30)}: [t694 f32 [H=576
                                                                      W=1 C=1] {pt2=root:add_30}] =
      add_scalar x=t693 {pt2=root:convolution_49} scalar=3
    n450 {pt2=root[236] torch.ops.aten.clamp.default (clamp_50)}: [t695 f32 [H=576
                                                                      W=1 C=1] {pt2=root:clamp_50}] =
      clamp x=t694 {pt2=root:add_30} params={min=0; max=none}
    n451 {pt2=root[237] torch.ops.aten.clamp.default (clamp_51)}: [t696 f32 [H=576
                                                                      W=1 C=1] {pt2=root:clamp_51}] =
      clamp x=t695 {pt2=root:clamp_50} params={min=none; max=6}
    n452 {pt2=root[238] torch.ops.aten.div.Tensor (div_25)}: [t697 f32 [H=576
                                                                      W=1 C=1] {pt2=root:div_25}] =
      div_scalar x=t696 {pt2=root:clamp_51} scalar=6
    n453 {pt2=root[239] torch.ops.aten.mul.Tensor (mul_25)}: [t698 f32 [H=576
                                                                      W=7 C=7] {pt2=root:mul_25}] =
      mul a=t697 {pt2=root:div_25} b=t683 {pt2=root:div_24}
    group g83 torch.ops.aten.convolution.default:
      n454 {derived}: [t699 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t698 {pt2=root:mul_25} perm=[H<-W, W<-C, C<-H]
      n455 {derived}: [t700 f32 [N=96 T=1 D=1 H=1 W=1 C=576] {derived}] =
        permute
          x=t132 {pt2=root:p_features_11_block_3_0_weight target=features.11.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n456 {derived}: [t701 f32 [H=7 W=7 C=96] {derived}] =
        convolution
          x=t699 {derived}
          weight=t700 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n457 {pt2=root[240] torch.ops.aten.convolution.default (convolution_50)}: [t702 f32 [H=96
                                                                      W=7 C=7] {pt2=root:convolution_50}] =
        permute x=t701 {derived} perm=[H<-C, W<-H, C<-W]
    group g84 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n458 {derived}: [t703 f32 [H=7 W=7 C=96] {derived}] =
        permute x=t702 {pt2=root:convolution_50} perm=[H<-W, W<-C, C<-H]
      n459 {derived}: [t704 f32 [H=7 W=7 C=96] {derived}] =
        batch_norm
          x=t703 {derived}
          weight=t133 {pt2=root:p_features_11_block_3_1_weight target=features.11.block.3.1.weight}
          bias=t134 {pt2=root:p_features_11_block_3_1_bias target=features.11.block.3.1.bias}
          running_mean=t238 {pt2=root:b_features_11_block_3_1_running_mean target=features.11.block.3.1.running_mean}
          running_var=t239 {pt2=root:b_features_11_block_3_1_running_var target=features.11.block.3.1.running_var}
          params={channel=C; eps=0.001}
      n460 {pt2=root[241] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_32)}: [t705 f32 [H=96
                                                                      W=7 C=7] {pt2=root:getitem_96}] =
        permute x=t704 {derived} perm=[H<-C, W<-H, C<-W]
    n461 {pt2=root[242] torch.ops.aten.add.Tensor (add_31)}: [t706 f32 [H=96
                                                                      W=7 C=7] {pt2=root:add_31}] =
      add a=t705 {pt2=root:getitem_96} b=t659 {pt2=root:add_27}
    group g85 torch.ops.aten.convolution.default:
      n462 {derived}: [t707 f32 [H=7 W=7 C=96] {derived}] =
        permute x=t706 {pt2=root:add_31} perm=[H<-W, W<-C, C<-H]
      n463 {derived}: [t708 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t135 {pt2=root:p_features_12_0_weight target=features.12.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n464 {derived}: [t709 f32 [H=7 W=7 C=576] {derived}] =
        convolution
          x=t707 {derived}
          weight=t708 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n465 {pt2=root[243] torch.ops.aten.convolution.default (convolution_51)}: [t710 f32 [H=576
                                                                      W=7 C=7] {pt2=root:convolution_51}] =
        permute x=t709 {derived} perm=[H<-C, W<-H, C<-W]
    group g86 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n466 {derived}: [t711 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t710 {pt2=root:convolution_51} perm=[H<-W, W<-C, C<-H]
      n467 {derived}: [t712 f32 [H=7 W=7 C=576] {derived}] =
        batch_norm
          x=t711 {derived}
          weight=t136 {pt2=root:p_features_12_1_weight target=features.12.1.weight}
          bias=t137 {pt2=root:p_features_12_1_bias target=features.12.1.bias}
          running_mean=t241 {pt2=root:b_features_12_1_running_mean target=features.12.1.running_mean}
          running_var=t242 {pt2=root:b_features_12_1_running_var target=features.12.1.running_var}
          params={channel=C; eps=0.001}
      n468 {pt2=root[244] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_33)}: [t713 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_99}] =
        permute x=t712 {derived} perm=[H<-C, W<-H, C<-W]
    n469 {pt2=root[245] torch.ops.aten.add.Tensor (add_32)}: [t714 f32 [H=576
                                                                      W=7 C=7] {pt2=root:add_32}] =
      add_scalar x=t713 {pt2=root:getitem_99} scalar=3
    n470 {pt2=root[246] torch.ops.aten.clamp.default (clamp_52)}: [t715 f32 [H=576
                                                                      W=7 C=7] {pt2=root:clamp_52}] =
      clamp x=t714 {pt2=root:add_32} params={min=0; max=none}
    n471 {pt2=root[247] torch.ops.aten.clamp.default (clamp_53)}: [t716 f32 [H=576
                                                                      W=7 C=7] {pt2=root:clamp_53}] =
      clamp x=t715 {pt2=root:clamp_52} params={min=none; max=6}
    n472 {pt2=root[248] torch.ops.aten.mul.Tensor (mul_26)}: [t717 f32 [H=576
                                                                      W=7 C=7] {pt2=root:mul_26}] =
      mul a=t713 {pt2=root:getitem_99} b=t716 {pt2=root:clamp_53}
    n473 {pt2=root[249] torch.ops.aten.div.Tensor (div_26)}: [t718 f32 [H=576
                                                                      W=7 C=7] {pt2=root:div_26}] =
      div_scalar x=t717 {pt2=root:mul_26} scalar=6
    n474 {pt2=root[250] torch.ops.aten.mean.dim (mean_9)}: [t719 f32 [H=576 W=1
                                                                      C=1] {pt2=root:mean_9}] =
      mean x=t718 {pt2=root:div_26} params={dims=[C, W]; keepdim=true}
    n475 {pt2=root[251] torch.ops.aten.view.default (view)}: [t720 f32 [C=576] {pt2=root:view}] =
      reshape x=t719 {pt2=root:mean_9} params={shape=[C=576]}
    n476 {pt2=root[252] torch.ops.aten.permute.default (permute)}: [t721 f32 [W=576
                                                                      C=1024] {pt2=root:permute}] =
      permute
        x=t138 {pt2=root:p_classifier_0_weight target=classifier.0.weight}
        perm=[W<-C, C<-W]
    group g87 torch.ops.aten.addmm.default:
      n477 {derived}: [t722 f32 [N=1024 T=1 D=1 H=1 W=1 C=576] {derived}] =
        permute x=t721 {pt2=root:permute} perm=[N<-C, W<-N, C<-W]
      n478 {pt2=root[253] torch.ops.aten.addmm.default (addmm)}: [t723 f32 [C=1024] {pt2=root:addmm}] =
        linear
          x=t720 {pt2=root:view}
          weight=t722 {derived}
          bias=t139 {pt2=root:p_classifier_0_bias target=classifier.0.bias}
          params={in_features=576}
    n479 {pt2=root[254] torch.ops.aten.add.Tensor (add_33)}: [t724 f32 [C=1024] {pt2=root:add_33}] =
      add_scalar x=t723 {pt2=root:addmm} scalar=3
    n480 {pt2=root[255] torch.ops.aten.clamp.default (clamp_54)}: [t725 f32 [C=1024] {pt2=root:clamp_54}] =
      clamp x=t724 {pt2=root:add_33} params={min=0; max=none}
    n481 {pt2=root[256] torch.ops.aten.clamp.default (clamp_55)}: [t726 f32 [C=1024] {pt2=root:clamp_55}] =
      clamp x=t725 {pt2=root:clamp_54} params={min=none; max=6}
    n482 {pt2=root[257] torch.ops.aten.mul.Tensor (mul_27)}: [t727 f32 [C=1024] {pt2=root:mul_27}] =
      mul a=t723 {pt2=root:addmm} b=t726 {pt2=root:clamp_55}
    n483 {pt2=root[258] torch.ops.aten.div.Tensor (div_27)}: [t728 f32 [C=1024] {pt2=root:div_27}] =
      div_scalar x=t727 {pt2=root:mul_27} scalar=6
    n484 {pt2=root[259] torch.ops.aten.clone.default (clone)}: [t729 f32 [C=1024] {pt2=root:clone}] =
      clone x=t728 {pt2=root:div_27}
    n485 {pt2=root[260] torch.ops.aten.permute.default (permute_1)}: [t730 f32 [W=1024
                                                                      C=1000] {pt2=root:permute_1}] =
      permute
        x=t140 {pt2=root:p_classifier_3_weight target=classifier.3.weight}
        perm=[W<-C, C<-W]
    group g88 torch.ops.aten.addmm.default:
      n486 {derived}: [t731 f32 [N=1000 T=1 D=1 H=1 W=1 C=1024] {derived}] =
        permute x=t730 {pt2=root:permute_1} perm=[N<-C, W<-N, C<-W]
      n487 {pt2=root[261] torch.ops.aten.addmm.default (addmm_1)}: [t732 f32 [C=1000] {pt2=root:addmm_1}] =
        linear
          x=t729 {pt2=root:clone}
          weight=t731 {derived}
          bias=t141 {pt2=root:p_classifier_3_bias target=classifier.3.bias}
          params={in_features=1024}
  outputs: [t732 f32 [C=1000] {pt2=root:addmm_1}]

MobileNet-v2 uses relu6 instead, which the exporter emits as hardtanh with float
bounds — the same parameter kind clamp receives as ints above.

  $ ../bin/native_graph.exe print --pt2 "$PT2_DATA/mobilenet_v2/mobilenet_v2.pt2"
  native graph: inputs=315 constants=314 nodes=415 outputs=1
  PT2 provenance: tensor-origins=469 captured-targets=314 node-origins=154
  graph
  inputs:
    [t0 f32 [D=32 H=3 W=3 C=3] {pt2=root:p_features_0_0_weight target=features.0.0.weight} constant,
     t1 f32 [C=32] {pt2=root:p_features_0_1_weight target=features.0.1.weight} constant,
     t2 f32 [C=32] {pt2=root:p_features_0_1_bias target=features.0.1.bias} constant,
     t3 f32 [D=32 H=1 W=3 C=3] {pt2=root:p_features_1_conv_0_0_weight target=features.1.conv.0.0.weight} constant,
     t4 f32 [C=32] {pt2=root:p_features_1_conv_0_1_weight target=features.1.conv.0.1.weight} constant,
     t5 f32 [C=32] {pt2=root:p_features_1_conv_0_1_bias target=features.1.conv.0.1.bias} constant,
     t6 f32 [D=16 H=32 W=1 C=1] {pt2=root:p_features_1_conv_1_weight target=features.1.conv.1.weight} constant,
     t7 f32 [C=16] {pt2=root:p_features_1_conv_2_weight target=features.1.conv.2.weight} constant,
     t8 f32 [C=16] {pt2=root:p_features_1_conv_2_bias target=features.1.conv.2.bias} constant,
     t9 f32 [D=96 H=16 W=1 C=1] {pt2=root:p_features_2_conv_0_0_weight target=features.2.conv.0.0.weight} constant,
     t10 f32 [C=96] {pt2=root:p_features_2_conv_0_1_weight target=features.2.conv.0.1.weight} constant,
     t11 f32 [C=96] {pt2=root:p_features_2_conv_0_1_bias target=features.2.conv.0.1.bias} constant,
     t12 f32 [D=96 H=1 W=3 C=3] {pt2=root:p_features_2_conv_1_0_weight target=features.2.conv.1.0.weight} constant,
     t13 f32 [C=96] {pt2=root:p_features_2_conv_1_1_weight target=features.2.conv.1.1.weight} constant,
     t14 f32 [C=96] {pt2=root:p_features_2_conv_1_1_bias target=features.2.conv.1.1.bias} constant,
     t15 f32 [D=24 H=96 W=1 C=1] {pt2=root:p_features_2_conv_2_weight target=features.2.conv.2.weight} constant,
     t16 f32 [C=24] {pt2=root:p_features_2_conv_3_weight target=features.2.conv.3.weight} constant,
     t17 f32 [C=24] {pt2=root:p_features_2_conv_3_bias target=features.2.conv.3.bias} constant,
     t18 f32 [D=144 H=24 W=1 C=1] {pt2=root:p_features_3_conv_0_0_weight target=features.3.conv.0.0.weight} constant,
     t19 f32 [C=144] {pt2=root:p_features_3_conv_0_1_weight target=features.3.conv.0.1.weight} constant,
     t20 f32 [C=144] {pt2=root:p_features_3_conv_0_1_bias target=features.3.conv.0.1.bias} constant,
     t21 f32 [D=144 H=1 W=3 C=3] {pt2=root:p_features_3_conv_1_0_weight target=features.3.conv.1.0.weight} constant,
     t22 f32 [C=144] {pt2=root:p_features_3_conv_1_1_weight target=features.3.conv.1.1.weight} constant,
     t23 f32 [C=144] {pt2=root:p_features_3_conv_1_1_bias target=features.3.conv.1.1.bias} constant,
     t24 f32 [D=24 H=144 W=1 C=1] {pt2=root:p_features_3_conv_2_weight target=features.3.conv.2.weight} constant,
     t25 f32 [C=24] {pt2=root:p_features_3_conv_3_weight target=features.3.conv.3.weight} constant,
     t26 f32 [C=24] {pt2=root:p_features_3_conv_3_bias target=features.3.conv.3.bias} constant,
     t27 f32 [D=144 H=24 W=1 C=1] {pt2=root:p_features_4_conv_0_0_weight target=features.4.conv.0.0.weight} constant,
     t28 f32 [C=144] {pt2=root:p_features_4_conv_0_1_weight target=features.4.conv.0.1.weight} constant,
     t29 f32 [C=144] {pt2=root:p_features_4_conv_0_1_bias target=features.4.conv.0.1.bias} constant,
     t30 f32 [D=144 H=1 W=3 C=3] {pt2=root:p_features_4_conv_1_0_weight target=features.4.conv.1.0.weight} constant,
     t31 f32 [C=144] {pt2=root:p_features_4_conv_1_1_weight target=features.4.conv.1.1.weight} constant,
     t32 f32 [C=144] {pt2=root:p_features_4_conv_1_1_bias target=features.4.conv.1.1.bias} constant,
     t33 f32 [D=32 H=144 W=1 C=1] {pt2=root:p_features_4_conv_2_weight target=features.4.conv.2.weight} constant,
     t34 f32 [C=32] {pt2=root:p_features_4_conv_3_weight target=features.4.conv.3.weight} constant,
     t35 f32 [C=32] {pt2=root:p_features_4_conv_3_bias target=features.4.conv.3.bias} constant,
     t36 f32 [D=192 H=32 W=1 C=1] {pt2=root:p_features_5_conv_0_0_weight target=features.5.conv.0.0.weight} constant,
     t37 f32 [C=192] {pt2=root:p_features_5_conv_0_1_weight target=features.5.conv.0.1.weight} constant,
     t38 f32 [C=192] {pt2=root:p_features_5_conv_0_1_bias target=features.5.conv.0.1.bias} constant,
     t39 f32 [D=192 H=1 W=3 C=3] {pt2=root:p_features_5_conv_1_0_weight target=features.5.conv.1.0.weight} constant,
     t40 f32 [C=192] {pt2=root:p_features_5_conv_1_1_weight target=features.5.conv.1.1.weight} constant,
     t41 f32 [C=192] {pt2=root:p_features_5_conv_1_1_bias target=features.5.conv.1.1.bias} constant,
     t42 f32 [D=32 H=192 W=1 C=1] {pt2=root:p_features_5_conv_2_weight target=features.5.conv.2.weight} constant,
     t43 f32 [C=32] {pt2=root:p_features_5_conv_3_weight target=features.5.conv.3.weight} constant,
     t44 f32 [C=32] {pt2=root:p_features_5_conv_3_bias target=features.5.conv.3.bias} constant,
     t45 f32 [D=192 H=32 W=1 C=1] {pt2=root:p_features_6_conv_0_0_weight target=features.6.conv.0.0.weight} constant,
     t46 f32 [C=192] {pt2=root:p_features_6_conv_0_1_weight target=features.6.conv.0.1.weight} constant,
     t47 f32 [C=192] {pt2=root:p_features_6_conv_0_1_bias target=features.6.conv.0.1.bias} constant,
     t48 f32 [D=192 H=1 W=3 C=3] {pt2=root:p_features_6_conv_1_0_weight target=features.6.conv.1.0.weight} constant,
     t49 f32 [C=192] {pt2=root:p_features_6_conv_1_1_weight target=features.6.conv.1.1.weight} constant,
     t50 f32 [C=192] {pt2=root:p_features_6_conv_1_1_bias target=features.6.conv.1.1.bias} constant,
     t51 f32 [D=32 H=192 W=1 C=1] {pt2=root:p_features_6_conv_2_weight target=features.6.conv.2.weight} constant,
     t52 f32 [C=32] {pt2=root:p_features_6_conv_3_weight target=features.6.conv.3.weight} constant,
     t53 f32 [C=32] {pt2=root:p_features_6_conv_3_bias target=features.6.conv.3.bias} constant,
     t54 f32 [D=192 H=32 W=1 C=1] {pt2=root:p_features_7_conv_0_0_weight target=features.7.conv.0.0.weight} constant,
     t55 f32 [C=192] {pt2=root:p_features_7_conv_0_1_weight target=features.7.conv.0.1.weight} constant,
     t56 f32 [C=192] {pt2=root:p_features_7_conv_0_1_bias target=features.7.conv.0.1.bias} constant,
     t57 f32 [D=192 H=1 W=3 C=3] {pt2=root:p_features_7_conv_1_0_weight target=features.7.conv.1.0.weight} constant,
     t58 f32 [C=192] {pt2=root:p_features_7_conv_1_1_weight target=features.7.conv.1.1.weight} constant,
     t59 f32 [C=192] {pt2=root:p_features_7_conv_1_1_bias target=features.7.conv.1.1.bias} constant,
     t60 f32 [D=64 H=192 W=1 C=1] {pt2=root:p_features_7_conv_2_weight target=features.7.conv.2.weight} constant,
     t61 f32 [C=64] {pt2=root:p_features_7_conv_3_weight target=features.7.conv.3.weight} constant,
     t62 f32 [C=64] {pt2=root:p_features_7_conv_3_bias target=features.7.conv.3.bias} constant,
     t63 f32 [D=384 H=64 W=1 C=1] {pt2=root:p_features_8_conv_0_0_weight target=features.8.conv.0.0.weight} constant,
     t64 f32 [C=384] {pt2=root:p_features_8_conv_0_1_weight target=features.8.conv.0.1.weight} constant,
     t65 f32 [C=384] {pt2=root:p_features_8_conv_0_1_bias target=features.8.conv.0.1.bias} constant,
     t66 f32 [D=384 H=1 W=3 C=3] {pt2=root:p_features_8_conv_1_0_weight target=features.8.conv.1.0.weight} constant,
     t67 f32 [C=384] {pt2=root:p_features_8_conv_1_1_weight target=features.8.conv.1.1.weight} constant,
     t68 f32 [C=384] {pt2=root:p_features_8_conv_1_1_bias target=features.8.conv.1.1.bias} constant,
     t69 f32 [D=64 H=384 W=1 C=1] {pt2=root:p_features_8_conv_2_weight target=features.8.conv.2.weight} constant,
     t70 f32 [C=64] {pt2=root:p_features_8_conv_3_weight target=features.8.conv.3.weight} constant,
     t71 f32 [C=64] {pt2=root:p_features_8_conv_3_bias target=features.8.conv.3.bias} constant,
     t72 f32 [D=384 H=64 W=1 C=1] {pt2=root:p_features_9_conv_0_0_weight target=features.9.conv.0.0.weight} constant,
     t73 f32 [C=384] {pt2=root:p_features_9_conv_0_1_weight target=features.9.conv.0.1.weight} constant,
     t74 f32 [C=384] {pt2=root:p_features_9_conv_0_1_bias target=features.9.conv.0.1.bias} constant,
     t75 f32 [D=384 H=1 W=3 C=3] {pt2=root:p_features_9_conv_1_0_weight target=features.9.conv.1.0.weight} constant,
     t76 f32 [C=384] {pt2=root:p_features_9_conv_1_1_weight target=features.9.conv.1.1.weight} constant,
     t77 f32 [C=384] {pt2=root:p_features_9_conv_1_1_bias target=features.9.conv.1.1.bias} constant,
     t78 f32 [D=64 H=384 W=1 C=1] {pt2=root:p_features_9_conv_2_weight target=features.9.conv.2.weight} constant,
     t79 f32 [C=64] {pt2=root:p_features_9_conv_3_weight target=features.9.conv.3.weight} constant,
     t80 f32 [C=64] {pt2=root:p_features_9_conv_3_bias target=features.9.conv.3.bias} constant,
     t81 f32 [D=384 H=64 W=1 C=1] {pt2=root:p_features_10_conv_0_0_weight target=features.10.conv.0.0.weight} constant,
     t82 f32 [C=384] {pt2=root:p_features_10_conv_0_1_weight target=features.10.conv.0.1.weight} constant,
     t83 f32 [C=384] {pt2=root:p_features_10_conv_0_1_bias target=features.10.conv.0.1.bias} constant,
     t84 f32 [D=384 H=1 W=3 C=3] {pt2=root:p_features_10_conv_1_0_weight target=features.10.conv.1.0.weight} constant,
     t85 f32 [C=384] {pt2=root:p_features_10_conv_1_1_weight target=features.10.conv.1.1.weight} constant,
     t86 f32 [C=384] {pt2=root:p_features_10_conv_1_1_bias target=features.10.conv.1.1.bias} constant,
     t87 f32 [D=64 H=384 W=1 C=1] {pt2=root:p_features_10_conv_2_weight target=features.10.conv.2.weight} constant,
     t88 f32 [C=64] {pt2=root:p_features_10_conv_3_weight target=features.10.conv.3.weight} constant,
     t89 f32 [C=64] {pt2=root:p_features_10_conv_3_bias target=features.10.conv.3.bias} constant,
     t90 f32 [D=384 H=64 W=1 C=1] {pt2=root:p_features_11_conv_0_0_weight target=features.11.conv.0.0.weight} constant,
     t91 f32 [C=384] {pt2=root:p_features_11_conv_0_1_weight target=features.11.conv.0.1.weight} constant,
     t92 f32 [C=384] {pt2=root:p_features_11_conv_0_1_bias target=features.11.conv.0.1.bias} constant,
     t93 f32 [D=384 H=1 W=3 C=3] {pt2=root:p_features_11_conv_1_0_weight target=features.11.conv.1.0.weight} constant,
     t94 f32 [C=384] {pt2=root:p_features_11_conv_1_1_weight target=features.11.conv.1.1.weight} constant,
     t95 f32 [C=384] {pt2=root:p_features_11_conv_1_1_bias target=features.11.conv.1.1.bias} constant,
     t96 f32 [D=96 H=384 W=1 C=1] {pt2=root:p_features_11_conv_2_weight target=features.11.conv.2.weight} constant,
     t97 f32 [C=96] {pt2=root:p_features_11_conv_3_weight target=features.11.conv.3.weight} constant,
     t98 f32 [C=96] {pt2=root:p_features_11_conv_3_bias target=features.11.conv.3.bias} constant,
     t99 f32 [D=576 H=96 W=1 C=1] {pt2=root:p_features_12_conv_0_0_weight target=features.12.conv.0.0.weight} constant,
     t100 f32 [C=576] {pt2=root:p_features_12_conv_0_1_weight target=features.12.conv.0.1.weight} constant,
     t101 f32 [C=576] {pt2=root:p_features_12_conv_0_1_bias target=features.12.conv.0.1.bias} constant,
     t102 f32 [D=576 H=1 W=3 C=3] {pt2=root:p_features_12_conv_1_0_weight target=features.12.conv.1.0.weight} constant,
     t103 f32 [C=576] {pt2=root:p_features_12_conv_1_1_weight target=features.12.conv.1.1.weight} constant,
     t104 f32 [C=576] {pt2=root:p_features_12_conv_1_1_bias target=features.12.conv.1.1.bias} constant,
     t105 f32 [D=96 H=576 W=1 C=1] {pt2=root:p_features_12_conv_2_weight target=features.12.conv.2.weight} constant,
     t106 f32 [C=96] {pt2=root:p_features_12_conv_3_weight target=features.12.conv.3.weight} constant,
     t107 f32 [C=96] {pt2=root:p_features_12_conv_3_bias target=features.12.conv.3.bias} constant,
     t108 f32 [D=576 H=96 W=1 C=1] {pt2=root:p_features_13_conv_0_0_weight target=features.13.conv.0.0.weight} constant,
     t109 f32 [C=576] {pt2=root:p_features_13_conv_0_1_weight target=features.13.conv.0.1.weight} constant,
     t110 f32 [C=576] {pt2=root:p_features_13_conv_0_1_bias target=features.13.conv.0.1.bias} constant,
     t111 f32 [D=576 H=1 W=3 C=3] {pt2=root:p_features_13_conv_1_0_weight target=features.13.conv.1.0.weight} constant,
     t112 f32 [C=576] {pt2=root:p_features_13_conv_1_1_weight target=features.13.conv.1.1.weight} constant,
     t113 f32 [C=576] {pt2=root:p_features_13_conv_1_1_bias target=features.13.conv.1.1.bias} constant,
     t114 f32 [D=96 H=576 W=1 C=1] {pt2=root:p_features_13_conv_2_weight target=features.13.conv.2.weight} constant,
     t115 f32 [C=96] {pt2=root:p_features_13_conv_3_weight target=features.13.conv.3.weight} constant,
     t116 f32 [C=96] {pt2=root:p_features_13_conv_3_bias target=features.13.conv.3.bias} constant,
     t117 f32 [D=576 H=96 W=1 C=1] {pt2=root:p_features_14_conv_0_0_weight target=features.14.conv.0.0.weight} constant,
     t118 f32 [C=576] {pt2=root:p_features_14_conv_0_1_weight target=features.14.conv.0.1.weight} constant,
     t119 f32 [C=576] {pt2=root:p_features_14_conv_0_1_bias target=features.14.conv.0.1.bias} constant,
     t120 f32 [D=576 H=1 W=3 C=3] {pt2=root:p_features_14_conv_1_0_weight target=features.14.conv.1.0.weight} constant,
     t121 f32 [C=576] {pt2=root:p_features_14_conv_1_1_weight target=features.14.conv.1.1.weight} constant,
     t122 f32 [C=576] {pt2=root:p_features_14_conv_1_1_bias target=features.14.conv.1.1.bias} constant,
     t123 f32 [D=160 H=576 W=1 C=1] {pt2=root:p_features_14_conv_2_weight target=features.14.conv.2.weight} constant,
     t124 f32 [C=160] {pt2=root:p_features_14_conv_3_weight target=features.14.conv.3.weight} constant,
     t125 f32 [C=160] {pt2=root:p_features_14_conv_3_bias target=features.14.conv.3.bias} constant,
     t126 f32 [D=960 H=160 W=1 C=1] {pt2=root:p_features_15_conv_0_0_weight target=features.15.conv.0.0.weight} constant,
     t127 f32 [C=960] {pt2=root:p_features_15_conv_0_1_weight target=features.15.conv.0.1.weight} constant,
     t128 f32 [C=960] {pt2=root:p_features_15_conv_0_1_bias target=features.15.conv.0.1.bias} constant,
     t129 f32 [D=960 H=1 W=3 C=3] {pt2=root:p_features_15_conv_1_0_weight target=features.15.conv.1.0.weight} constant,
     t130 f32 [C=960] {pt2=root:p_features_15_conv_1_1_weight target=features.15.conv.1.1.weight} constant,
     t131 f32 [C=960] {pt2=root:p_features_15_conv_1_1_bias target=features.15.conv.1.1.bias} constant,
     t132 f32 [D=160 H=960 W=1 C=1] {pt2=root:p_features_15_conv_2_weight target=features.15.conv.2.weight} constant,
     t133 f32 [C=160] {pt2=root:p_features_15_conv_3_weight target=features.15.conv.3.weight} constant,
     t134 f32 [C=160] {pt2=root:p_features_15_conv_3_bias target=features.15.conv.3.bias} constant,
     t135 f32 [D=960 H=160 W=1 C=1] {pt2=root:p_features_16_conv_0_0_weight target=features.16.conv.0.0.weight} constant,
     t136 f32 [C=960] {pt2=root:p_features_16_conv_0_1_weight target=features.16.conv.0.1.weight} constant,
     t137 f32 [C=960] {pt2=root:p_features_16_conv_0_1_bias target=features.16.conv.0.1.bias} constant,
     t138 f32 [D=960 H=1 W=3 C=3] {pt2=root:p_features_16_conv_1_0_weight target=features.16.conv.1.0.weight} constant,
     t139 f32 [C=960] {pt2=root:p_features_16_conv_1_1_weight target=features.16.conv.1.1.weight} constant,
     t140 f32 [C=960] {pt2=root:p_features_16_conv_1_1_bias target=features.16.conv.1.1.bias} constant,
     t141 f32 [D=160 H=960 W=1 C=1] {pt2=root:p_features_16_conv_2_weight target=features.16.conv.2.weight} constant,
     t142 f32 [C=160] {pt2=root:p_features_16_conv_3_weight target=features.16.conv.3.weight} constant,
     t143 f32 [C=160] {pt2=root:p_features_16_conv_3_bias target=features.16.conv.3.bias} constant,
     t144 f32 [D=960 H=160 W=1 C=1] {pt2=root:p_features_17_conv_0_0_weight target=features.17.conv.0.0.weight} constant,
     t145 f32 [C=960] {pt2=root:p_features_17_conv_0_1_weight target=features.17.conv.0.1.weight} constant,
     t146 f32 [C=960] {pt2=root:p_features_17_conv_0_1_bias target=features.17.conv.0.1.bias} constant,
     t147 f32 [D=960 H=1 W=3 C=3] {pt2=root:p_features_17_conv_1_0_weight target=features.17.conv.1.0.weight} constant,
     t148 f32 [C=960] {pt2=root:p_features_17_conv_1_1_weight target=features.17.conv.1.1.weight} constant,
     t149 f32 [C=960] {pt2=root:p_features_17_conv_1_1_bias target=features.17.conv.1.1.bias} constant,
     t150 f32 [D=320 H=960 W=1 C=1] {pt2=root:p_features_17_conv_2_weight target=features.17.conv.2.weight} constant,
     t151 f32 [C=320] {pt2=root:p_features_17_conv_3_weight target=features.17.conv.3.weight} constant,
     t152 f32 [C=320] {pt2=root:p_features_17_conv_3_bias target=features.17.conv.3.bias} constant,
     t153 f32 [D=1280 H=320 W=1 C=1] {pt2=root:p_features_18_0_weight target=features.18.0.weight} constant,
     t154 f32 [C=1280] {pt2=root:p_features_18_1_weight target=features.18.1.weight} constant,
     t155 f32 [C=1280] {pt2=root:p_features_18_1_bias target=features.18.1.bias} constant,
     t156 f32 [W=1000 C=1280] {pt2=root:p_classifier_1_weight target=classifier.1.weight} constant,
     t157 f32 [C=1000] {pt2=root:p_classifier_1_bias target=classifier.1.bias} constant,
     t158 f32 [C=32] {pt2=root:b_features_0_1_running_mean target=features.0.1.running_mean} constant,
     t159 f32 [C=32] {pt2=root:b_features_0_1_running_var target=features.0.1.running_var} constant,
     t160 f32 [C=1] {pt2=root:b_features_0_1_num_batches_tracked target=features.0.1.num_batches_tracked} constant,
     t161 f32 [C=32] {pt2=root:b_features_1_conv_0_1_running_mean target=features.1.conv.0.1.running_mean} constant,
     t162 f32 [C=32] {pt2=root:b_features_1_conv_0_1_running_var target=features.1.conv.0.1.running_var} constant,
     t163 f32 [C=1] {pt2=root:b_features_1_conv_0_1_num_batches_tracked target=features.1.conv.0.1.num_batches_tracked} constant,
     t164 f32 [C=16] {pt2=root:b_features_1_conv_2_running_mean target=features.1.conv.2.running_mean} constant,
     t165 f32 [C=16] {pt2=root:b_features_1_conv_2_running_var target=features.1.conv.2.running_var} constant,
     t166 f32 [C=1] {pt2=root:b_features_1_conv_2_num_batches_tracked target=features.1.conv.2.num_batches_tracked} constant,
     t167 f32 [C=96] {pt2=root:b_features_2_conv_0_1_running_mean target=features.2.conv.0.1.running_mean} constant,
     t168 f32 [C=96] {pt2=root:b_features_2_conv_0_1_running_var target=features.2.conv.0.1.running_var} constant,
     t169 f32 [C=1] {pt2=root:b_features_2_conv_0_1_num_batches_tracked target=features.2.conv.0.1.num_batches_tracked} constant,
     t170 f32 [C=96] {pt2=root:b_features_2_conv_1_1_running_mean target=features.2.conv.1.1.running_mean} constant,
     t171 f32 [C=96] {pt2=root:b_features_2_conv_1_1_running_var target=features.2.conv.1.1.running_var} constant,
     t172 f32 [C=1] {pt2=root:b_features_2_conv_1_1_num_batches_tracked target=features.2.conv.1.1.num_batches_tracked} constant,
     t173 f32 [C=24] {pt2=root:b_features_2_conv_3_running_mean target=features.2.conv.3.running_mean} constant,
     t174 f32 [C=24] {pt2=root:b_features_2_conv_3_running_var target=features.2.conv.3.running_var} constant,
     t175 f32 [C=1] {pt2=root:b_features_2_conv_3_num_batches_tracked target=features.2.conv.3.num_batches_tracked} constant,
     t176 f32 [C=144] {pt2=root:b_features_3_conv_0_1_running_mean target=features.3.conv.0.1.running_mean} constant,
     t177 f32 [C=144] {pt2=root:b_features_3_conv_0_1_running_var target=features.3.conv.0.1.running_var} constant,
     t178 f32 [C=1] {pt2=root:b_features_3_conv_0_1_num_batches_tracked target=features.3.conv.0.1.num_batches_tracked} constant,
     t179 f32 [C=144] {pt2=root:b_features_3_conv_1_1_running_mean target=features.3.conv.1.1.running_mean} constant,
     t180 f32 [C=144] {pt2=root:b_features_3_conv_1_1_running_var target=features.3.conv.1.1.running_var} constant,
     t181 f32 [C=1] {pt2=root:b_features_3_conv_1_1_num_batches_tracked target=features.3.conv.1.1.num_batches_tracked} constant,
     t182 f32 [C=24] {pt2=root:b_features_3_conv_3_running_mean target=features.3.conv.3.running_mean} constant,
     t183 f32 [C=24] {pt2=root:b_features_3_conv_3_running_var target=features.3.conv.3.running_var} constant,
     t184 f32 [C=1] {pt2=root:b_features_3_conv_3_num_batches_tracked target=features.3.conv.3.num_batches_tracked} constant,
     t185 f32 [C=144] {pt2=root:b_features_4_conv_0_1_running_mean target=features.4.conv.0.1.running_mean} constant,
     t186 f32 [C=144] {pt2=root:b_features_4_conv_0_1_running_var target=features.4.conv.0.1.running_var} constant,
     t187 f32 [C=1] {pt2=root:b_features_4_conv_0_1_num_batches_tracked target=features.4.conv.0.1.num_batches_tracked} constant,
     t188 f32 [C=144] {pt2=root:b_features_4_conv_1_1_running_mean target=features.4.conv.1.1.running_mean} constant,
     t189 f32 [C=144] {pt2=root:b_features_4_conv_1_1_running_var target=features.4.conv.1.1.running_var} constant,
     t190 f32 [C=1] {pt2=root:b_features_4_conv_1_1_num_batches_tracked target=features.4.conv.1.1.num_batches_tracked} constant,
     t191 f32 [C=32] {pt2=root:b_features_4_conv_3_running_mean target=features.4.conv.3.running_mean} constant,
     t192 f32 [C=32] {pt2=root:b_features_4_conv_3_running_var target=features.4.conv.3.running_var} constant,
     t193 f32 [C=1] {pt2=root:b_features_4_conv_3_num_batches_tracked target=features.4.conv.3.num_batches_tracked} constant,
     t194 f32 [C=192] {pt2=root:b_features_5_conv_0_1_running_mean target=features.5.conv.0.1.running_mean} constant,
     t195 f32 [C=192] {pt2=root:b_features_5_conv_0_1_running_var target=features.5.conv.0.1.running_var} constant,
     t196 f32 [C=1] {pt2=root:b_features_5_conv_0_1_num_batches_tracked target=features.5.conv.0.1.num_batches_tracked} constant,
     t197 f32 [C=192] {pt2=root:b_features_5_conv_1_1_running_mean target=features.5.conv.1.1.running_mean} constant,
     t198 f32 [C=192] {pt2=root:b_features_5_conv_1_1_running_var target=features.5.conv.1.1.running_var} constant,
     t199 f32 [C=1] {pt2=root:b_features_5_conv_1_1_num_batches_tracked target=features.5.conv.1.1.num_batches_tracked} constant,
     t200 f32 [C=32] {pt2=root:b_features_5_conv_3_running_mean target=features.5.conv.3.running_mean} constant,
     t201 f32 [C=32] {pt2=root:b_features_5_conv_3_running_var target=features.5.conv.3.running_var} constant,
     t202 f32 [C=1] {pt2=root:b_features_5_conv_3_num_batches_tracked target=features.5.conv.3.num_batches_tracked} constant,
     t203 f32 [C=192] {pt2=root:b_features_6_conv_0_1_running_mean target=features.6.conv.0.1.running_mean} constant,
     t204 f32 [C=192] {pt2=root:b_features_6_conv_0_1_running_var target=features.6.conv.0.1.running_var} constant,
     t205 f32 [C=1] {pt2=root:b_features_6_conv_0_1_num_batches_tracked target=features.6.conv.0.1.num_batches_tracked} constant,
     t206 f32 [C=192] {pt2=root:b_features_6_conv_1_1_running_mean target=features.6.conv.1.1.running_mean} constant,
     t207 f32 [C=192] {pt2=root:b_features_6_conv_1_1_running_var target=features.6.conv.1.1.running_var} constant,
     t208 f32 [C=1] {pt2=root:b_features_6_conv_1_1_num_batches_tracked target=features.6.conv.1.1.num_batches_tracked} constant,
     t209 f32 [C=32] {pt2=root:b_features_6_conv_3_running_mean target=features.6.conv.3.running_mean} constant,
     t210 f32 [C=32] {pt2=root:b_features_6_conv_3_running_var target=features.6.conv.3.running_var} constant,
     t211 f32 [C=1] {pt2=root:b_features_6_conv_3_num_batches_tracked target=features.6.conv.3.num_batches_tracked} constant,
     t212 f32 [C=192] {pt2=root:b_features_7_conv_0_1_running_mean target=features.7.conv.0.1.running_mean} constant,
     t213 f32 [C=192] {pt2=root:b_features_7_conv_0_1_running_var target=features.7.conv.0.1.running_var} constant,
     t214 f32 [C=1] {pt2=root:b_features_7_conv_0_1_num_batches_tracked target=features.7.conv.0.1.num_batches_tracked} constant,
     t215 f32 [C=192] {pt2=root:b_features_7_conv_1_1_running_mean target=features.7.conv.1.1.running_mean} constant,
     t216 f32 [C=192] {pt2=root:b_features_7_conv_1_1_running_var target=features.7.conv.1.1.running_var} constant,
     t217 f32 [C=1] {pt2=root:b_features_7_conv_1_1_num_batches_tracked target=features.7.conv.1.1.num_batches_tracked} constant,
     t218 f32 [C=64] {pt2=root:b_features_7_conv_3_running_mean target=features.7.conv.3.running_mean} constant,
     t219 f32 [C=64] {pt2=root:b_features_7_conv_3_running_var target=features.7.conv.3.running_var} constant,
     t220 f32 [C=1] {pt2=root:b_features_7_conv_3_num_batches_tracked target=features.7.conv.3.num_batches_tracked} constant,
     t221 f32 [C=384] {pt2=root:b_features_8_conv_0_1_running_mean target=features.8.conv.0.1.running_mean} constant,
     t222 f32 [C=384] {pt2=root:b_features_8_conv_0_1_running_var target=features.8.conv.0.1.running_var} constant,
     t223 f32 [C=1] {pt2=root:b_features_8_conv_0_1_num_batches_tracked target=features.8.conv.0.1.num_batches_tracked} constant,
     t224 f32 [C=384] {pt2=root:b_features_8_conv_1_1_running_mean target=features.8.conv.1.1.running_mean} constant,
     t225 f32 [C=384] {pt2=root:b_features_8_conv_1_1_running_var target=features.8.conv.1.1.running_var} constant,
     t226 f32 [C=1] {pt2=root:b_features_8_conv_1_1_num_batches_tracked target=features.8.conv.1.1.num_batches_tracked} constant,
     t227 f32 [C=64] {pt2=root:b_features_8_conv_3_running_mean target=features.8.conv.3.running_mean} constant,
     t228 f32 [C=64] {pt2=root:b_features_8_conv_3_running_var target=features.8.conv.3.running_var} constant,
     t229 f32 [C=1] {pt2=root:b_features_8_conv_3_num_batches_tracked target=features.8.conv.3.num_batches_tracked} constant,
     t230 f32 [C=384] {pt2=root:b_features_9_conv_0_1_running_mean target=features.9.conv.0.1.running_mean} constant,
     t231 f32 [C=384] {pt2=root:b_features_9_conv_0_1_running_var target=features.9.conv.0.1.running_var} constant,
     t232 f32 [C=1] {pt2=root:b_features_9_conv_0_1_num_batches_tracked target=features.9.conv.0.1.num_batches_tracked} constant,
     t233 f32 [C=384] {pt2=root:b_features_9_conv_1_1_running_mean target=features.9.conv.1.1.running_mean} constant,
     t234 f32 [C=384] {pt2=root:b_features_9_conv_1_1_running_var target=features.9.conv.1.1.running_var} constant,
     t235 f32 [C=1] {pt2=root:b_features_9_conv_1_1_num_batches_tracked target=features.9.conv.1.1.num_batches_tracked} constant,
     t236 f32 [C=64] {pt2=root:b_features_9_conv_3_running_mean target=features.9.conv.3.running_mean} constant,
     t237 f32 [C=64] {pt2=root:b_features_9_conv_3_running_var target=features.9.conv.3.running_var} constant,
     t238 f32 [C=1] {pt2=root:b_features_9_conv_3_num_batches_tracked target=features.9.conv.3.num_batches_tracked} constant,
     t239 f32 [C=384] {pt2=root:b_features_10_conv_0_1_running_mean target=features.10.conv.0.1.running_mean} constant,
     t240 f32 [C=384] {pt2=root:b_features_10_conv_0_1_running_var target=features.10.conv.0.1.running_var} constant,
     t241 f32 [C=1] {pt2=root:b_features_10_conv_0_1_num_batches_tracked target=features.10.conv.0.1.num_batches_tracked} constant,
     t242 f32 [C=384] {pt2=root:b_features_10_conv_1_1_running_mean target=features.10.conv.1.1.running_mean} constant,
     t243 f32 [C=384] {pt2=root:b_features_10_conv_1_1_running_var target=features.10.conv.1.1.running_var} constant,
     t244 f32 [C=1] {pt2=root:b_features_10_conv_1_1_num_batches_tracked target=features.10.conv.1.1.num_batches_tracked} constant,
     t245 f32 [C=64] {pt2=root:b_features_10_conv_3_running_mean target=features.10.conv.3.running_mean} constant,
     t246 f32 [C=64] {pt2=root:b_features_10_conv_3_running_var target=features.10.conv.3.running_var} constant,
     t247 f32 [C=1] {pt2=root:b_features_10_conv_3_num_batches_tracked target=features.10.conv.3.num_batches_tracked} constant,
     t248 f32 [C=384] {pt2=root:b_features_11_conv_0_1_running_mean target=features.11.conv.0.1.running_mean} constant,
     t249 f32 [C=384] {pt2=root:b_features_11_conv_0_1_running_var target=features.11.conv.0.1.running_var} constant,
     t250 f32 [C=1] {pt2=root:b_features_11_conv_0_1_num_batches_tracked target=features.11.conv.0.1.num_batches_tracked} constant,
     t251 f32 [C=384] {pt2=root:b_features_11_conv_1_1_running_mean target=features.11.conv.1.1.running_mean} constant,
     t252 f32 [C=384] {pt2=root:b_features_11_conv_1_1_running_var target=features.11.conv.1.1.running_var} constant,
     t253 f32 [C=1] {pt2=root:b_features_11_conv_1_1_num_batches_tracked target=features.11.conv.1.1.num_batches_tracked} constant,
     t254 f32 [C=96] {pt2=root:b_features_11_conv_3_running_mean target=features.11.conv.3.running_mean} constant,
     t255 f32 [C=96] {pt2=root:b_features_11_conv_3_running_var target=features.11.conv.3.running_var} constant,
     t256 f32 [C=1] {pt2=root:b_features_11_conv_3_num_batches_tracked target=features.11.conv.3.num_batches_tracked} constant,
     t257 f32 [C=576] {pt2=root:b_features_12_conv_0_1_running_mean target=features.12.conv.0.1.running_mean} constant,
     t258 f32 [C=576] {pt2=root:b_features_12_conv_0_1_running_var target=features.12.conv.0.1.running_var} constant,
     t259 f32 [C=1] {pt2=root:b_features_12_conv_0_1_num_batches_tracked target=features.12.conv.0.1.num_batches_tracked} constant,
     t260 f32 [C=576] {pt2=root:b_features_12_conv_1_1_running_mean target=features.12.conv.1.1.running_mean} constant,
     t261 f32 [C=576] {pt2=root:b_features_12_conv_1_1_running_var target=features.12.conv.1.1.running_var} constant,
     t262 f32 [C=1] {pt2=root:b_features_12_conv_1_1_num_batches_tracked target=features.12.conv.1.1.num_batches_tracked} constant,
     t263 f32 [C=96] {pt2=root:b_features_12_conv_3_running_mean target=features.12.conv.3.running_mean} constant,
     t264 f32 [C=96] {pt2=root:b_features_12_conv_3_running_var target=features.12.conv.3.running_var} constant,
     t265 f32 [C=1] {pt2=root:b_features_12_conv_3_num_batches_tracked target=features.12.conv.3.num_batches_tracked} constant,
     t266 f32 [C=576] {pt2=root:b_features_13_conv_0_1_running_mean target=features.13.conv.0.1.running_mean} constant,
     t267 f32 [C=576] {pt2=root:b_features_13_conv_0_1_running_var target=features.13.conv.0.1.running_var} constant,
     t268 f32 [C=1] {pt2=root:b_features_13_conv_0_1_num_batches_tracked target=features.13.conv.0.1.num_batches_tracked} constant,
     t269 f32 [C=576] {pt2=root:b_features_13_conv_1_1_running_mean target=features.13.conv.1.1.running_mean} constant,
     t270 f32 [C=576] {pt2=root:b_features_13_conv_1_1_running_var target=features.13.conv.1.1.running_var} constant,
     t271 f32 [C=1] {pt2=root:b_features_13_conv_1_1_num_batches_tracked target=features.13.conv.1.1.num_batches_tracked} constant,
     t272 f32 [C=96] {pt2=root:b_features_13_conv_3_running_mean target=features.13.conv.3.running_mean} constant,
     t273 f32 [C=96] {pt2=root:b_features_13_conv_3_running_var target=features.13.conv.3.running_var} constant,
     t274 f32 [C=1] {pt2=root:b_features_13_conv_3_num_batches_tracked target=features.13.conv.3.num_batches_tracked} constant,
     t275 f32 [C=576] {pt2=root:b_features_14_conv_0_1_running_mean target=features.14.conv.0.1.running_mean} constant,
     t276 f32 [C=576] {pt2=root:b_features_14_conv_0_1_running_var target=features.14.conv.0.1.running_var} constant,
     t277 f32 [C=1] {pt2=root:b_features_14_conv_0_1_num_batches_tracked target=features.14.conv.0.1.num_batches_tracked} constant,
     t278 f32 [C=576] {pt2=root:b_features_14_conv_1_1_running_mean target=features.14.conv.1.1.running_mean} constant,
     t279 f32 [C=576] {pt2=root:b_features_14_conv_1_1_running_var target=features.14.conv.1.1.running_var} constant,
     t280 f32 [C=1] {pt2=root:b_features_14_conv_1_1_num_batches_tracked target=features.14.conv.1.1.num_batches_tracked} constant,
     t281 f32 [C=160] {pt2=root:b_features_14_conv_3_running_mean target=features.14.conv.3.running_mean} constant,
     t282 f32 [C=160] {pt2=root:b_features_14_conv_3_running_var target=features.14.conv.3.running_var} constant,
     t283 f32 [C=1] {pt2=root:b_features_14_conv_3_num_batches_tracked target=features.14.conv.3.num_batches_tracked} constant,
     t284 f32 [C=960] {pt2=root:b_features_15_conv_0_1_running_mean target=features.15.conv.0.1.running_mean} constant,
     t285 f32 [C=960] {pt2=root:b_features_15_conv_0_1_running_var target=features.15.conv.0.1.running_var} constant,
     t286 f32 [C=1] {pt2=root:b_features_15_conv_0_1_num_batches_tracked target=features.15.conv.0.1.num_batches_tracked} constant,
     t287 f32 [C=960] {pt2=root:b_features_15_conv_1_1_running_mean target=features.15.conv.1.1.running_mean} constant,
     t288 f32 [C=960] {pt2=root:b_features_15_conv_1_1_running_var target=features.15.conv.1.1.running_var} constant,
     t289 f32 [C=1] {pt2=root:b_features_15_conv_1_1_num_batches_tracked target=features.15.conv.1.1.num_batches_tracked} constant,
     t290 f32 [C=160] {pt2=root:b_features_15_conv_3_running_mean target=features.15.conv.3.running_mean} constant,
     t291 f32 [C=160] {pt2=root:b_features_15_conv_3_running_var target=features.15.conv.3.running_var} constant,
     t292 f32 [C=1] {pt2=root:b_features_15_conv_3_num_batches_tracked target=features.15.conv.3.num_batches_tracked} constant,
     t293 f32 [C=960] {pt2=root:b_features_16_conv_0_1_running_mean target=features.16.conv.0.1.running_mean} constant,
     t294 f32 [C=960] {pt2=root:b_features_16_conv_0_1_running_var target=features.16.conv.0.1.running_var} constant,
     t295 f32 [C=1] {pt2=root:b_features_16_conv_0_1_num_batches_tracked target=features.16.conv.0.1.num_batches_tracked} constant,
     t296 f32 [C=960] {pt2=root:b_features_16_conv_1_1_running_mean target=features.16.conv.1.1.running_mean} constant,
     t297 f32 [C=960] {pt2=root:b_features_16_conv_1_1_running_var target=features.16.conv.1.1.running_var} constant,
     t298 f32 [C=1] {pt2=root:b_features_16_conv_1_1_num_batches_tracked target=features.16.conv.1.1.num_batches_tracked} constant,
     t299 f32 [C=160] {pt2=root:b_features_16_conv_3_running_mean target=features.16.conv.3.running_mean} constant,
     t300 f32 [C=160] {pt2=root:b_features_16_conv_3_running_var target=features.16.conv.3.running_var} constant,
     t301 f32 [C=1] {pt2=root:b_features_16_conv_3_num_batches_tracked target=features.16.conv.3.num_batches_tracked} constant,
     t302 f32 [C=960] {pt2=root:b_features_17_conv_0_1_running_mean target=features.17.conv.0.1.running_mean} constant,
     t303 f32 [C=960] {pt2=root:b_features_17_conv_0_1_running_var target=features.17.conv.0.1.running_var} constant,
     t304 f32 [C=1] {pt2=root:b_features_17_conv_0_1_num_batches_tracked target=features.17.conv.0.1.num_batches_tracked} constant,
     t305 f32 [C=960] {pt2=root:b_features_17_conv_1_1_running_mean target=features.17.conv.1.1.running_mean} constant,
     t306 f32 [C=960] {pt2=root:b_features_17_conv_1_1_running_var target=features.17.conv.1.1.running_var} constant,
     t307 f32 [C=1] {pt2=root:b_features_17_conv_1_1_num_batches_tracked target=features.17.conv.1.1.num_batches_tracked} constant,
     t308 f32 [C=320] {pt2=root:b_features_17_conv_3_running_mean target=features.17.conv.3.running_mean} constant,
     t309 f32 [C=320] {pt2=root:b_features_17_conv_3_running_var target=features.17.conv.3.running_var} constant,
     t310 f32 [C=1] {pt2=root:b_features_17_conv_3_num_batches_tracked target=features.17.conv.3.num_batches_tracked} constant,
     t311 f32 [C=1280] {pt2=root:b_features_18_1_running_mean target=features.18.1.running_mean} constant,
     t312 f32 [C=1280] {pt2=root:b_features_18_1_running_var target=features.18.1.running_var} constant,
     t313 f32 [C=1] {pt2=root:b_features_18_1_num_batches_tracked target=features.18.1.num_batches_tracked} constant,
     t314 f32 [H=3 W=224 C=224] {pt2=root:x}]
  nodes:
    group g1 torch.ops.aten.convolution.default:
      n0 {derived}: [t315 f32 [H=224 W=224 C=3] {derived}] =
        permute x=t314 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
      n1 {derived}: [t316 f32 [N=32 T=1 D=1 H=3 W=3 C=3] {derived}] =
        permute
          x=t0 {pt2=root:p_features_0_0_weight target=features.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2 {derived}: [t317 f32 [H=112 W=112 C=32] {derived}] =
        convolution
          x=t315 {derived}
          weight=t316 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n3 {pt2=root[0] torch.ops.aten.convolution.default (convolution)}: [t318 f32 [H=32
                                                                      W=112
                                                                      C=112] {pt2=root:convolution}] =
        permute x=t317 {derived} perm=[H<-C, W<-H, C<-W]
    group g2 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n4 {derived}: [t319 f32 [H=112 W=112 C=32] {derived}] =
        permute x=t318 {pt2=root:convolution} perm=[H<-W, W<-C, C<-H]
      n5 {derived}: [t320 f32 [H=112 W=112 C=32] {derived}] =
        batch_norm
          x=t319 {derived}
          weight=t1 {pt2=root:p_features_0_1_weight target=features.0.1.weight}
          bias=t2 {pt2=root:p_features_0_1_bias target=features.0.1.bias}
          running_mean=t158 {pt2=root:b_features_0_1_running_mean target=features.0.1.running_mean}
          running_var=t159 {pt2=root:b_features_0_1_running_var target=features.0.1.running_var}
          params={channel=C; eps=1e-05}
      n6 {pt2=root[1] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training)}: [t321 f32 [H=32
                                                                      W=112
                                                                      C=112] {pt2=root:getitem}] =
        permute x=t320 {derived} perm=[H<-C, W<-H, C<-W]
    n7 {pt2=root[2] torch.ops.aten.hardtanh.default (hardtanh)}: [t322 f32 [H=32
                                                                      W=112
                                                                      C=112] {pt2=root:hardtanh}] =
      hardtanh x=t321 {pt2=root:getitem} params={min_val=0; max_val=6}
    group g3 torch.ops.aten.convolution.default:
      n8 {derived}: [t323 f32 [H=112 W=112 C=32] {derived}] =
        permute x=t322 {pt2=root:hardtanh} perm=[H<-W, W<-C, C<-H]
      n9 {derived}: [t324 f32 [N=32 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t3 {pt2=root:p_features_1_conv_0_0_weight target=features.1.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n10 {derived}: [t325 f32 [H=112 W=112 C=32] {derived}] =
        convolution
          x=t323 {derived}
          weight=t324 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=32}
      n11 {pt2=root[3] torch.ops.aten.convolution.default (convolution_1)}: [t326 f32 [H=32
                                                                      W=112
                                                                      C=112] {pt2=root:convolution_1}] =
        permute x=t325 {derived} perm=[H<-C, W<-H, C<-W]
    group g4 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n12 {derived}: [t327 f32 [H=112 W=112 C=32] {derived}] =
        permute x=t326 {pt2=root:convolution_1} perm=[H<-W, W<-C, C<-H]
      n13 {derived}: [t328 f32 [H=112 W=112 C=32] {derived}] =
        batch_norm
          x=t327 {derived}
          weight=t4 {pt2=root:p_features_1_conv_0_1_weight target=features.1.conv.0.1.weight}
          bias=t5 {pt2=root:p_features_1_conv_0_1_bias target=features.1.conv.0.1.bias}
          running_mean=t161 {pt2=root:b_features_1_conv_0_1_running_mean target=features.1.conv.0.1.running_mean}
          running_var=t162 {pt2=root:b_features_1_conv_0_1_running_var target=features.1.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
      n14 {pt2=root[4] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_1)}: [t329 f32 [H=32
                                                                      W=112
                                                                      C=112] {pt2=root:getitem_3}] =
        permute x=t328 {derived} perm=[H<-C, W<-H, C<-W]
    n15 {pt2=root[5] torch.ops.aten.hardtanh.default (hardtanh_1)}: [t330 f32 [H=32
                                                                      W=112
                                                                      C=112] {pt2=root:hardtanh_1}] =
      hardtanh x=t329 {pt2=root:getitem_3} params={min_val=0; max_val=6}
    group g5 torch.ops.aten.convolution.default:
      n16 {derived}: [t331 f32 [H=112 W=112 C=32] {derived}] =
        permute x=t330 {pt2=root:hardtanh_1} perm=[H<-W, W<-C, C<-H]
      n17 {derived}: [t332 f32 [N=16 T=1 D=1 H=1 W=1 C=32] {derived}] =
        permute
          x=t6 {pt2=root:p_features_1_conv_1_weight target=features.1.conv.1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n18 {derived}: [t333 f32 [H=112 W=112 C=16] {derived}] =
        convolution
          x=t331 {derived}
          weight=t332 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n19 {pt2=root[6] torch.ops.aten.convolution.default (convolution_2)}: [t334 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:convolution_2}] =
        permute x=t333 {derived} perm=[H<-C, W<-H, C<-W]
    group g6 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n20 {derived}: [t335 f32 [H=112 W=112 C=16] {derived}] =
        permute x=t334 {pt2=root:convolution_2} perm=[H<-W, W<-C, C<-H]
      n21 {derived}: [t336 f32 [H=112 W=112 C=16] {derived}] =
        batch_norm
          x=t335 {derived}
          weight=t7 {pt2=root:p_features_1_conv_2_weight target=features.1.conv.2.weight}
          bias=t8 {pt2=root:p_features_1_conv_2_bias target=features.1.conv.2.bias}
          running_mean=t164 {pt2=root:b_features_1_conv_2_running_mean target=features.1.conv.2.running_mean}
          running_var=t165 {pt2=root:b_features_1_conv_2_running_var target=features.1.conv.2.running_var}
          params={channel=C; eps=1e-05}
      n22 {pt2=root[7] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_2)}: [t337 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:getitem_6}] =
        permute x=t336 {derived} perm=[H<-C, W<-H, C<-W]
    group g7 torch.ops.aten.convolution.default:
      n23 {derived}: [t338 f32 [H=112 W=112 C=16] {derived}] =
        permute x=t337 {pt2=root:getitem_6} perm=[H<-W, W<-C, C<-H]
      n24 {derived}: [t339 f32 [N=96 T=1 D=1 H=1 W=1 C=16] {derived}] =
        permute
          x=t9 {pt2=root:p_features_2_conv_0_0_weight target=features.2.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n25 {derived}: [t340 f32 [H=112 W=112 C=96] {derived}] =
        convolution
          x=t338 {derived}
          weight=t339 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n26 {pt2=root[8] torch.ops.aten.convolution.default (convolution_3)}: [t341 f32 [H=96
                                                                      W=112
                                                                      C=112] {pt2=root:convolution_3}] =
        permute x=t340 {derived} perm=[H<-C, W<-H, C<-W]
    group g8 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n27 {derived}: [t342 f32 [H=112 W=112 C=96] {derived}] =
        permute x=t341 {pt2=root:convolution_3} perm=[H<-W, W<-C, C<-H]
      n28 {derived}: [t343 f32 [H=112 W=112 C=96] {derived}] =
        batch_norm
          x=t342 {derived}
          weight=t10 {pt2=root:p_features_2_conv_0_1_weight target=features.2.conv.0.1.weight}
          bias=t11 {pt2=root:p_features_2_conv_0_1_bias target=features.2.conv.0.1.bias}
          running_mean=t167 {pt2=root:b_features_2_conv_0_1_running_mean target=features.2.conv.0.1.running_mean}
          running_var=t168 {pt2=root:b_features_2_conv_0_1_running_var target=features.2.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
      n29 {pt2=root[9] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_3)}: [t344 f32 [H=96
                                                                      W=112
                                                                      C=112] {pt2=root:getitem_9}] =
        permute x=t343 {derived} perm=[H<-C, W<-H, C<-W]
    n30 {pt2=root[10] torch.ops.aten.hardtanh.default (hardtanh_2)}: [t345 f32 [H=96
                                                                      W=112
                                                                      C=112] {pt2=root:hardtanh_2}] =
      hardtanh x=t344 {pt2=root:getitem_9} params={min_val=0; max_val=6}
    group g9 torch.ops.aten.convolution.default:
      n31 {derived}: [t346 f32 [H=112 W=112 C=96] {derived}] =
        permute x=t345 {pt2=root:hardtanh_2} perm=[H<-W, W<-C, C<-H]
      n32 {derived}: [t347 f32 [N=96 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t12 {pt2=root:p_features_2_conv_1_0_weight target=features.2.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n33 {derived}: [t348 f32 [H=56 W=56 C=96] {derived}] =
        convolution
          x=t346 {derived}
          weight=t347 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=96}
      n34 {pt2=root[11] torch.ops.aten.convolution.default (convolution_4)}: [t349 f32 [H=96
                                                                      W=56
                                                                      C=56] {pt2=root:convolution_4}] =
        permute x=t348 {derived} perm=[H<-C, W<-H, C<-W]
    group g10 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n35 {derived}: [t350 f32 [H=56 W=56 C=96] {derived}] =
        permute x=t349 {pt2=root:convolution_4} perm=[H<-W, W<-C, C<-H]
      n36 {derived}: [t351 f32 [H=56 W=56 C=96] {derived}] =
        batch_norm
          x=t350 {derived}
          weight=t13 {pt2=root:p_features_2_conv_1_1_weight target=features.2.conv.1.1.weight}
          bias=t14 {pt2=root:p_features_2_conv_1_1_bias target=features.2.conv.1.1.bias}
          running_mean=t170 {pt2=root:b_features_2_conv_1_1_running_mean target=features.2.conv.1.1.running_mean}
          running_var=t171 {pt2=root:b_features_2_conv_1_1_running_var target=features.2.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
      n37 {pt2=root[12] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_4)}: [t352 f32 [H=96
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_12}] =
        permute x=t351 {derived} perm=[H<-C, W<-H, C<-W]
    n38 {pt2=root[13] torch.ops.aten.hardtanh.default (hardtanh_3)}: [t353 f32 [H=96
                                                                      W=56
                                                                      C=56] {pt2=root:hardtanh_3}] =
      hardtanh x=t352 {pt2=root:getitem_12} params={min_val=0; max_val=6}
    group g11 torch.ops.aten.convolution.default:
      n39 {derived}: [t354 f32 [H=56 W=56 C=96] {derived}] =
        permute x=t353 {pt2=root:hardtanh_3} perm=[H<-W, W<-C, C<-H]
      n40 {derived}: [t355 f32 [N=24 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t15 {pt2=root:p_features_2_conv_2_weight target=features.2.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n41 {derived}: [t356 f32 [H=56 W=56 C=24] {derived}] =
        convolution
          x=t354 {derived}
          weight=t355 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n42 {pt2=root[14] torch.ops.aten.convolution.default (convolution_5)}: [t357 f32 [H=24
                                                                      W=56
                                                                      C=56] {pt2=root:convolution_5}] =
        permute x=t356 {derived} perm=[H<-C, W<-H, C<-W]
    group g12 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n43 {derived}: [t358 f32 [H=56 W=56 C=24] {derived}] =
        permute x=t357 {pt2=root:convolution_5} perm=[H<-W, W<-C, C<-H]
      n44 {derived}: [t359 f32 [H=56 W=56 C=24] {derived}] =
        batch_norm
          x=t358 {derived}
          weight=t16 {pt2=root:p_features_2_conv_3_weight target=features.2.conv.3.weight}
          bias=t17 {pt2=root:p_features_2_conv_3_bias target=features.2.conv.3.bias}
          running_mean=t173 {pt2=root:b_features_2_conv_3_running_mean target=features.2.conv.3.running_mean}
          running_var=t174 {pt2=root:b_features_2_conv_3_running_var target=features.2.conv.3.running_var}
          params={channel=C; eps=1e-05}
      n45 {pt2=root[15] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_5)}: [t360 f32 [H=24
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_15}] =
        permute x=t359 {derived} perm=[H<-C, W<-H, C<-W]
    group g13 torch.ops.aten.convolution.default:
      n46 {derived}: [t361 f32 [H=56 W=56 C=24] {derived}] =
        permute x=t360 {pt2=root:getitem_15} perm=[H<-W, W<-C, C<-H]
      n47 {derived}: [t362 f32 [N=144 T=1 D=1 H=1 W=1 C=24] {derived}] =
        permute
          x=t18 {pt2=root:p_features_3_conv_0_0_weight target=features.3.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n48 {derived}: [t363 f32 [H=56 W=56 C=144] {derived}] =
        convolution
          x=t361 {derived}
          weight=t362 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n49 {pt2=root[16] torch.ops.aten.convolution.default (convolution_6)}: [t364 f32 [H=144
                                                                      W=56
                                                                      C=56] {pt2=root:convolution_6}] =
        permute x=t363 {derived} perm=[H<-C, W<-H, C<-W]
    group g14 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n50 {derived}: [t365 f32 [H=56 W=56 C=144] {derived}] =
        permute x=t364 {pt2=root:convolution_6} perm=[H<-W, W<-C, C<-H]
      n51 {derived}: [t366 f32 [H=56 W=56 C=144] {derived}] =
        batch_norm
          x=t365 {derived}
          weight=t19 {pt2=root:p_features_3_conv_0_1_weight target=features.3.conv.0.1.weight}
          bias=t20 {pt2=root:p_features_3_conv_0_1_bias target=features.3.conv.0.1.bias}
          running_mean=t176 {pt2=root:b_features_3_conv_0_1_running_mean target=features.3.conv.0.1.running_mean}
          running_var=t177 {pt2=root:b_features_3_conv_0_1_running_var target=features.3.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
      n52 {pt2=root[17] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_6)}: [t367 f32 [H=144
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_18}] =
        permute x=t366 {derived} perm=[H<-C, W<-H, C<-W]
    n53 {pt2=root[18] torch.ops.aten.hardtanh.default (hardtanh_4)}: [t368 f32 [H=144
                                                                      W=56
                                                                      C=56] {pt2=root:hardtanh_4}] =
      hardtanh x=t367 {pt2=root:getitem_18} params={min_val=0; max_val=6}
    group g15 torch.ops.aten.convolution.default:
      n54 {derived}: [t369 f32 [H=56 W=56 C=144] {derived}] =
        permute x=t368 {pt2=root:hardtanh_4} perm=[H<-W, W<-C, C<-H]
      n55 {derived}: [t370 f32 [N=144 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t21 {pt2=root:p_features_3_conv_1_0_weight target=features.3.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n56 {derived}: [t371 f32 [H=56 W=56 C=144] {derived}] =
        convolution
          x=t369 {derived}
          weight=t370 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=144}
      n57 {pt2=root[19] torch.ops.aten.convolution.default (convolution_7)}: [t372 f32 [H=144
                                                                      W=56
                                                                      C=56] {pt2=root:convolution_7}] =
        permute x=t371 {derived} perm=[H<-C, W<-H, C<-W]
    group g16 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n58 {derived}: [t373 f32 [H=56 W=56 C=144] {derived}] =
        permute x=t372 {pt2=root:convolution_7} perm=[H<-W, W<-C, C<-H]
      n59 {derived}: [t374 f32 [H=56 W=56 C=144] {derived}] =
        batch_norm
          x=t373 {derived}
          weight=t22 {pt2=root:p_features_3_conv_1_1_weight target=features.3.conv.1.1.weight}
          bias=t23 {pt2=root:p_features_3_conv_1_1_bias target=features.3.conv.1.1.bias}
          running_mean=t179 {pt2=root:b_features_3_conv_1_1_running_mean target=features.3.conv.1.1.running_mean}
          running_var=t180 {pt2=root:b_features_3_conv_1_1_running_var target=features.3.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
      n60 {pt2=root[20] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_7)}: [t375 f32 [H=144
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_21}] =
        permute x=t374 {derived} perm=[H<-C, W<-H, C<-W]
    n61 {pt2=root[21] torch.ops.aten.hardtanh.default (hardtanh_5)}: [t376 f32 [H=144
                                                                      W=56
                                                                      C=56] {pt2=root:hardtanh_5}] =
      hardtanh x=t375 {pt2=root:getitem_21} params={min_val=0; max_val=6}
    group g17 torch.ops.aten.convolution.default:
      n62 {derived}: [t377 f32 [H=56 W=56 C=144] {derived}] =
        permute x=t376 {pt2=root:hardtanh_5} perm=[H<-W, W<-C, C<-H]
      n63 {derived}: [t378 f32 [N=24 T=1 D=1 H=1 W=1 C=144] {derived}] =
        permute
          x=t24 {pt2=root:p_features_3_conv_2_weight target=features.3.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n64 {derived}: [t379 f32 [H=56 W=56 C=24] {derived}] =
        convolution
          x=t377 {derived}
          weight=t378 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n65 {pt2=root[22] torch.ops.aten.convolution.default (convolution_8)}: [t380 f32 [H=24
                                                                      W=56
                                                                      C=56] {pt2=root:convolution_8}] =
        permute x=t379 {derived} perm=[H<-C, W<-H, C<-W]
    group g18 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n66 {derived}: [t381 f32 [H=56 W=56 C=24] {derived}] =
        permute x=t380 {pt2=root:convolution_8} perm=[H<-W, W<-C, C<-H]
      n67 {derived}: [t382 f32 [H=56 W=56 C=24] {derived}] =
        batch_norm
          x=t381 {derived}
          weight=t25 {pt2=root:p_features_3_conv_3_weight target=features.3.conv.3.weight}
          bias=t26 {pt2=root:p_features_3_conv_3_bias target=features.3.conv.3.bias}
          running_mean=t182 {pt2=root:b_features_3_conv_3_running_mean target=features.3.conv.3.running_mean}
          running_var=t183 {pt2=root:b_features_3_conv_3_running_var target=features.3.conv.3.running_var}
          params={channel=C; eps=1e-05}
      n68 {pt2=root[23] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_8)}: [t383 f32 [H=24
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_24}] =
        permute x=t382 {derived} perm=[H<-C, W<-H, C<-W]
    n69 {pt2=root[24] torch.ops.aten.add.Tensor (add)}: [t384 f32 [H=24 W=56
                                                                   C=56] {pt2=root:add}] =
      add a=t360 {pt2=root:getitem_15} b=t383 {pt2=root:getitem_24}
    group g19 torch.ops.aten.convolution.default:
      n70 {derived}: [t385 f32 [H=56 W=56 C=24] {derived}] =
        permute x=t384 {pt2=root:add} perm=[H<-W, W<-C, C<-H]
      n71 {derived}: [t386 f32 [N=144 T=1 D=1 H=1 W=1 C=24] {derived}] =
        permute
          x=t27 {pt2=root:p_features_4_conv_0_0_weight target=features.4.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n72 {derived}: [t387 f32 [H=56 W=56 C=144] {derived}] =
        convolution
          x=t385 {derived}
          weight=t386 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n73 {pt2=root[25] torch.ops.aten.convolution.default (convolution_9)}: [t388 f32 [H=144
                                                                      W=56
                                                                      C=56] {pt2=root:convolution_9}] =
        permute x=t387 {derived} perm=[H<-C, W<-H, C<-W]
    group g20 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n74 {derived}: [t389 f32 [H=56 W=56 C=144] {derived}] =
        permute x=t388 {pt2=root:convolution_9} perm=[H<-W, W<-C, C<-H]
      n75 {derived}: [t390 f32 [H=56 W=56 C=144] {derived}] =
        batch_norm
          x=t389 {derived}
          weight=t28 {pt2=root:p_features_4_conv_0_1_weight target=features.4.conv.0.1.weight}
          bias=t29 {pt2=root:p_features_4_conv_0_1_bias target=features.4.conv.0.1.bias}
          running_mean=t185 {pt2=root:b_features_4_conv_0_1_running_mean target=features.4.conv.0.1.running_mean}
          running_var=t186 {pt2=root:b_features_4_conv_0_1_running_var target=features.4.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
      n76 {pt2=root[26] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_9)}: [t391 f32 [H=144
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_27}] =
        permute x=t390 {derived} perm=[H<-C, W<-H, C<-W]
    n77 {pt2=root[27] torch.ops.aten.hardtanh.default (hardtanh_6)}: [t392 f32 [H=144
                                                                      W=56
                                                                      C=56] {pt2=root:hardtanh_6}] =
      hardtanh x=t391 {pt2=root:getitem_27} params={min_val=0; max_val=6}
    group g21 torch.ops.aten.convolution.default:
      n78 {derived}: [t393 f32 [H=56 W=56 C=144] {derived}] =
        permute x=t392 {pt2=root:hardtanh_6} perm=[H<-W, W<-C, C<-H]
      n79 {derived}: [t394 f32 [N=144 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t30 {pt2=root:p_features_4_conv_1_0_weight target=features.4.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n80 {derived}: [t395 f32 [H=28 W=28 C=144] {derived}] =
        convolution
          x=t393 {derived}
          weight=t394 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=144}
      n81 {pt2=root[28] torch.ops.aten.convolution.default (convolution_10)}: [t396 f32 [H=144
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_10}] =
        permute x=t395 {derived} perm=[H<-C, W<-H, C<-W]
    group g22 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n82 {derived}: [t397 f32 [H=28 W=28 C=144] {derived}] =
        permute x=t396 {pt2=root:convolution_10} perm=[H<-W, W<-C, C<-H]
      n83 {derived}: [t398 f32 [H=28 W=28 C=144] {derived}] =
        batch_norm
          x=t397 {derived}
          weight=t31 {pt2=root:p_features_4_conv_1_1_weight target=features.4.conv.1.1.weight}
          bias=t32 {pt2=root:p_features_4_conv_1_1_bias target=features.4.conv.1.1.bias}
          running_mean=t188 {pt2=root:b_features_4_conv_1_1_running_mean target=features.4.conv.1.1.running_mean}
          running_var=t189 {pt2=root:b_features_4_conv_1_1_running_var target=features.4.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
      n84 {pt2=root[29] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_10)}: [t399 f32 [H=144
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_30}] =
        permute x=t398 {derived} perm=[H<-C, W<-H, C<-W]
    n85 {pt2=root[30] torch.ops.aten.hardtanh.default (hardtanh_7)}: [t400 f32 [H=144
                                                                      W=28
                                                                      C=28] {pt2=root:hardtanh_7}] =
      hardtanh x=t399 {pt2=root:getitem_30} params={min_val=0; max_val=6}
    group g23 torch.ops.aten.convolution.default:
      n86 {derived}: [t401 f32 [H=28 W=28 C=144] {derived}] =
        permute x=t400 {pt2=root:hardtanh_7} perm=[H<-W, W<-C, C<-H]
      n87 {derived}: [t402 f32 [N=32 T=1 D=1 H=1 W=1 C=144] {derived}] =
        permute
          x=t33 {pt2=root:p_features_4_conv_2_weight target=features.4.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n88 {derived}: [t403 f32 [H=28 W=28 C=32] {derived}] =
        convolution
          x=t401 {derived}
          weight=t402 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n89 {pt2=root[31] torch.ops.aten.convolution.default (convolution_11)}: [t404 f32 [H=32
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_11}] =
        permute x=t403 {derived} perm=[H<-C, W<-H, C<-W]
    group g24 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n90 {derived}: [t405 f32 [H=28 W=28 C=32] {derived}] =
        permute x=t404 {pt2=root:convolution_11} perm=[H<-W, W<-C, C<-H]
      n91 {derived}: [t406 f32 [H=28 W=28 C=32] {derived}] =
        batch_norm
          x=t405 {derived}
          weight=t34 {pt2=root:p_features_4_conv_3_weight target=features.4.conv.3.weight}
          bias=t35 {pt2=root:p_features_4_conv_3_bias target=features.4.conv.3.bias}
          running_mean=t191 {pt2=root:b_features_4_conv_3_running_mean target=features.4.conv.3.running_mean}
          running_var=t192 {pt2=root:b_features_4_conv_3_running_var target=features.4.conv.3.running_var}
          params={channel=C; eps=1e-05}
      n92 {pt2=root[32] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_11)}: [t407 f32 [H=32
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_33}] =
        permute x=t406 {derived} perm=[H<-C, W<-H, C<-W]
    group g25 torch.ops.aten.convolution.default:
      n93 {derived}: [t408 f32 [H=28 W=28 C=32] {derived}] =
        permute x=t407 {pt2=root:getitem_33} perm=[H<-W, W<-C, C<-H]
      n94 {derived}: [t409 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {derived}] =
        permute
          x=t36 {pt2=root:p_features_5_conv_0_0_weight target=features.5.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n95 {derived}: [t410 f32 [H=28 W=28 C=192] {derived}] =
        convolution
          x=t408 {derived}
          weight=t409 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n96 {pt2=root[33] torch.ops.aten.convolution.default (convolution_12)}: [t411 f32 [H=192
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_12}] =
        permute x=t410 {derived} perm=[H<-C, W<-H, C<-W]
    group g26 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n97 {derived}: [t412 f32 [H=28 W=28 C=192] {derived}] =
        permute x=t411 {pt2=root:convolution_12} perm=[H<-W, W<-C, C<-H]
      n98 {derived}: [t413 f32 [H=28 W=28 C=192] {derived}] =
        batch_norm
          x=t412 {derived}
          weight=t37 {pt2=root:p_features_5_conv_0_1_weight target=features.5.conv.0.1.weight}
          bias=t38 {pt2=root:p_features_5_conv_0_1_bias target=features.5.conv.0.1.bias}
          running_mean=t194 {pt2=root:b_features_5_conv_0_1_running_mean target=features.5.conv.0.1.running_mean}
          running_var=t195 {pt2=root:b_features_5_conv_0_1_running_var target=features.5.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
      n99 {pt2=root[34] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_12)}: [t414 f32 [H=192
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_36}] =
        permute x=t413 {derived} perm=[H<-C, W<-H, C<-W]
    n100 {pt2=root[35] torch.ops.aten.hardtanh.default (hardtanh_8)}: [t415 f32 [H=192
                                                                      W=28
                                                                      C=28] {pt2=root:hardtanh_8}] =
      hardtanh x=t414 {pt2=root:getitem_36} params={min_val=0; max_val=6}
    group g27 torch.ops.aten.convolution.default:
      n101 {derived}: [t416 f32 [H=28 W=28 C=192] {derived}] =
        permute x=t415 {pt2=root:hardtanh_8} perm=[H<-W, W<-C, C<-H]
      n102 {derived}: [t417 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t39 {pt2=root:p_features_5_conv_1_0_weight target=features.5.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n103 {derived}: [t418 f32 [H=28 W=28 C=192] {derived}] =
        convolution
          x=t416 {derived}
          weight=t417 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=192}
      n104 {pt2=root[36] torch.ops.aten.convolution.default (convolution_13)}: [t419 f32 [H=192
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_13}] =
        permute x=t418 {derived} perm=[H<-C, W<-H, C<-W]
    group g28 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n105 {derived}: [t420 f32 [H=28 W=28 C=192] {derived}] =
        permute x=t419 {pt2=root:convolution_13} perm=[H<-W, W<-C, C<-H]
      n106 {derived}: [t421 f32 [H=28 W=28 C=192] {derived}] =
        batch_norm
          x=t420 {derived}
          weight=t40 {pt2=root:p_features_5_conv_1_1_weight target=features.5.conv.1.1.weight}
          bias=t41 {pt2=root:p_features_5_conv_1_1_bias target=features.5.conv.1.1.bias}
          running_mean=t197 {pt2=root:b_features_5_conv_1_1_running_mean target=features.5.conv.1.1.running_mean}
          running_var=t198 {pt2=root:b_features_5_conv_1_1_running_var target=features.5.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
      n107 {pt2=root[37] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_13)}: [t422 f32 [H=192
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_39}] =
        permute x=t421 {derived} perm=[H<-C, W<-H, C<-W]
    n108 {pt2=root[38] torch.ops.aten.hardtanh.default (hardtanh_9)}: [t423 f32 [H=192
                                                                      W=28
                                                                      C=28] {pt2=root:hardtanh_9}] =
      hardtanh x=t422 {pt2=root:getitem_39} params={min_val=0; max_val=6}
    group g29 torch.ops.aten.convolution.default:
      n109 {derived}: [t424 f32 [H=28 W=28 C=192] {derived}] =
        permute x=t423 {pt2=root:hardtanh_9} perm=[H<-W, W<-C, C<-H]
      n110 {derived}: [t425 f32 [N=32 T=1 D=1 H=1 W=1 C=192] {derived}] =
        permute
          x=t42 {pt2=root:p_features_5_conv_2_weight target=features.5.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n111 {derived}: [t426 f32 [H=28 W=28 C=32] {derived}] =
        convolution
          x=t424 {derived}
          weight=t425 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n112 {pt2=root[39] torch.ops.aten.convolution.default (convolution_14)}: [t427 f32 [H=32
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_14}] =
        permute x=t426 {derived} perm=[H<-C, W<-H, C<-W]
    group g30 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n113 {derived}: [t428 f32 [H=28 W=28 C=32] {derived}] =
        permute x=t427 {pt2=root:convolution_14} perm=[H<-W, W<-C, C<-H]
      n114 {derived}: [t429 f32 [H=28 W=28 C=32] {derived}] =
        batch_norm
          x=t428 {derived}
          weight=t43 {pt2=root:p_features_5_conv_3_weight target=features.5.conv.3.weight}
          bias=t44 {pt2=root:p_features_5_conv_3_bias target=features.5.conv.3.bias}
          running_mean=t200 {pt2=root:b_features_5_conv_3_running_mean target=features.5.conv.3.running_mean}
          running_var=t201 {pt2=root:b_features_5_conv_3_running_var target=features.5.conv.3.running_var}
          params={channel=C; eps=1e-05}
      n115 {pt2=root[40] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_14)}: [t430 f32 [H=32
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_42}] =
        permute x=t429 {derived} perm=[H<-C, W<-H, C<-W]
    n116 {pt2=root[41] torch.ops.aten.add.Tensor (add_1)}: [t431 f32 [H=32 W=28
                                                                      C=28] {pt2=root:add_1}] =
      add a=t407 {pt2=root:getitem_33} b=t430 {pt2=root:getitem_42}
    group g31 torch.ops.aten.convolution.default:
      n117 {derived}: [t432 f32 [H=28 W=28 C=32] {derived}] =
        permute x=t431 {pt2=root:add_1} perm=[H<-W, W<-C, C<-H]
      n118 {derived}: [t433 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {derived}] =
        permute
          x=t45 {pt2=root:p_features_6_conv_0_0_weight target=features.6.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n119 {derived}: [t434 f32 [H=28 W=28 C=192] {derived}] =
        convolution
          x=t432 {derived}
          weight=t433 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n120 {pt2=root[42] torch.ops.aten.convolution.default (convolution_15)}: [t435 f32 [H=192
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_15}] =
        permute x=t434 {derived} perm=[H<-C, W<-H, C<-W]
    group g32 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n121 {derived}: [t436 f32 [H=28 W=28 C=192] {derived}] =
        permute x=t435 {pt2=root:convolution_15} perm=[H<-W, W<-C, C<-H]
      n122 {derived}: [t437 f32 [H=28 W=28 C=192] {derived}] =
        batch_norm
          x=t436 {derived}
          weight=t46 {pt2=root:p_features_6_conv_0_1_weight target=features.6.conv.0.1.weight}
          bias=t47 {pt2=root:p_features_6_conv_0_1_bias target=features.6.conv.0.1.bias}
          running_mean=t203 {pt2=root:b_features_6_conv_0_1_running_mean target=features.6.conv.0.1.running_mean}
          running_var=t204 {pt2=root:b_features_6_conv_0_1_running_var target=features.6.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
      n123 {pt2=root[43] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_15)}: [t438 f32 [H=192
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_45}] =
        permute x=t437 {derived} perm=[H<-C, W<-H, C<-W]
    n124 {pt2=root[44] torch.ops.aten.hardtanh.default (hardtanh_10)}: [t439 f32 [H=192
                                                                      W=28
                                                                      C=28] {pt2=root:hardtanh_10}] =
      hardtanh x=t438 {pt2=root:getitem_45} params={min_val=0; max_val=6}
    group g33 torch.ops.aten.convolution.default:
      n125 {derived}: [t440 f32 [H=28 W=28 C=192] {derived}] =
        permute x=t439 {pt2=root:hardtanh_10} perm=[H<-W, W<-C, C<-H]
      n126 {derived}: [t441 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t48 {pt2=root:p_features_6_conv_1_0_weight target=features.6.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n127 {derived}: [t442 f32 [H=28 W=28 C=192] {derived}] =
        convolution
          x=t440 {derived}
          weight=t441 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=192}
      n128 {pt2=root[45] torch.ops.aten.convolution.default (convolution_16)}: [t443 f32 [H=192
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_16}] =
        permute x=t442 {derived} perm=[H<-C, W<-H, C<-W]
    group g34 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n129 {derived}: [t444 f32 [H=28 W=28 C=192] {derived}] =
        permute x=t443 {pt2=root:convolution_16} perm=[H<-W, W<-C, C<-H]
      n130 {derived}: [t445 f32 [H=28 W=28 C=192] {derived}] =
        batch_norm
          x=t444 {derived}
          weight=t49 {pt2=root:p_features_6_conv_1_1_weight target=features.6.conv.1.1.weight}
          bias=t50 {pt2=root:p_features_6_conv_1_1_bias target=features.6.conv.1.1.bias}
          running_mean=t206 {pt2=root:b_features_6_conv_1_1_running_mean target=features.6.conv.1.1.running_mean}
          running_var=t207 {pt2=root:b_features_6_conv_1_1_running_var target=features.6.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
      n131 {pt2=root[46] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_16)}: [t446 f32 [H=192
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_48}] =
        permute x=t445 {derived} perm=[H<-C, W<-H, C<-W]
    n132 {pt2=root[47] torch.ops.aten.hardtanh.default (hardtanh_11)}: [t447 f32 [H=192
                                                                      W=28
                                                                      C=28] {pt2=root:hardtanh_11}] =
      hardtanh x=t446 {pt2=root:getitem_48} params={min_val=0; max_val=6}
    group g35 torch.ops.aten.convolution.default:
      n133 {derived}: [t448 f32 [H=28 W=28 C=192] {derived}] =
        permute x=t447 {pt2=root:hardtanh_11} perm=[H<-W, W<-C, C<-H]
      n134 {derived}: [t449 f32 [N=32 T=1 D=1 H=1 W=1 C=192] {derived}] =
        permute
          x=t51 {pt2=root:p_features_6_conv_2_weight target=features.6.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n135 {derived}: [t450 f32 [H=28 W=28 C=32] {derived}] =
        convolution
          x=t448 {derived}
          weight=t449 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n136 {pt2=root[48] torch.ops.aten.convolution.default (convolution_17)}: [t451 f32 [H=32
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_17}] =
        permute x=t450 {derived} perm=[H<-C, W<-H, C<-W]
    group g36 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n137 {derived}: [t452 f32 [H=28 W=28 C=32] {derived}] =
        permute x=t451 {pt2=root:convolution_17} perm=[H<-W, W<-C, C<-H]
      n138 {derived}: [t453 f32 [H=28 W=28 C=32] {derived}] =
        batch_norm
          x=t452 {derived}
          weight=t52 {pt2=root:p_features_6_conv_3_weight target=features.6.conv.3.weight}
          bias=t53 {pt2=root:p_features_6_conv_3_bias target=features.6.conv.3.bias}
          running_mean=t209 {pt2=root:b_features_6_conv_3_running_mean target=features.6.conv.3.running_mean}
          running_var=t210 {pt2=root:b_features_6_conv_3_running_var target=features.6.conv.3.running_var}
          params={channel=C; eps=1e-05}
      n139 {pt2=root[49] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_17)}: [t454 f32 [H=32
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_51}] =
        permute x=t453 {derived} perm=[H<-C, W<-H, C<-W]
    n140 {pt2=root[50] torch.ops.aten.add.Tensor (add_2)}: [t455 f32 [H=32 W=28
                                                                      C=28] {pt2=root:add_2}] =
      add a=t431 {pt2=root:add_1} b=t454 {pt2=root:getitem_51}
    group g37 torch.ops.aten.convolution.default:
      n141 {derived}: [t456 f32 [H=28 W=28 C=32] {derived}] =
        permute x=t455 {pt2=root:add_2} perm=[H<-W, W<-C, C<-H]
      n142 {derived}: [t457 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {derived}] =
        permute
          x=t54 {pt2=root:p_features_7_conv_0_0_weight target=features.7.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n143 {derived}: [t458 f32 [H=28 W=28 C=192] {derived}] =
        convolution
          x=t456 {derived}
          weight=t457 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n144 {pt2=root[51] torch.ops.aten.convolution.default (convolution_18)}: [t459 f32 [H=192
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_18}] =
        permute x=t458 {derived} perm=[H<-C, W<-H, C<-W]
    group g38 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n145 {derived}: [t460 f32 [H=28 W=28 C=192] {derived}] =
        permute x=t459 {pt2=root:convolution_18} perm=[H<-W, W<-C, C<-H]
      n146 {derived}: [t461 f32 [H=28 W=28 C=192] {derived}] =
        batch_norm
          x=t460 {derived}
          weight=t55 {pt2=root:p_features_7_conv_0_1_weight target=features.7.conv.0.1.weight}
          bias=t56 {pt2=root:p_features_7_conv_0_1_bias target=features.7.conv.0.1.bias}
          running_mean=t212 {pt2=root:b_features_7_conv_0_1_running_mean target=features.7.conv.0.1.running_mean}
          running_var=t213 {pt2=root:b_features_7_conv_0_1_running_var target=features.7.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
      n147 {pt2=root[52] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_18)}: [t462 f32 [H=192
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_54}] =
        permute x=t461 {derived} perm=[H<-C, W<-H, C<-W]
    n148 {pt2=root[53] torch.ops.aten.hardtanh.default (hardtanh_12)}: [t463 f32 [H=192
                                                                      W=28
                                                                      C=28] {pt2=root:hardtanh_12}] =
      hardtanh x=t462 {pt2=root:getitem_54} params={min_val=0; max_val=6}
    group g39 torch.ops.aten.convolution.default:
      n149 {derived}: [t464 f32 [H=28 W=28 C=192] {derived}] =
        permute x=t463 {pt2=root:hardtanh_12} perm=[H<-W, W<-C, C<-H]
      n150 {derived}: [t465 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t57 {pt2=root:p_features_7_conv_1_0_weight target=features.7.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n151 {derived}: [t466 f32 [H=14 W=14 C=192] {derived}] =
        convolution
          x=t464 {derived}
          weight=t465 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=192}
      n152 {pt2=root[54] torch.ops.aten.convolution.default (convolution_19)}: [t467 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_19}] =
        permute x=t466 {derived} perm=[H<-C, W<-H, C<-W]
    group g40 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n153 {derived}: [t468 f32 [H=14 W=14 C=192] {derived}] =
        permute x=t467 {pt2=root:convolution_19} perm=[H<-W, W<-C, C<-H]
      n154 {derived}: [t469 f32 [H=14 W=14 C=192] {derived}] =
        batch_norm
          x=t468 {derived}
          weight=t58 {pt2=root:p_features_7_conv_1_1_weight target=features.7.conv.1.1.weight}
          bias=t59 {pt2=root:p_features_7_conv_1_1_bias target=features.7.conv.1.1.bias}
          running_mean=t215 {pt2=root:b_features_7_conv_1_1_running_mean target=features.7.conv.1.1.running_mean}
          running_var=t216 {pt2=root:b_features_7_conv_1_1_running_var target=features.7.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
      n155 {pt2=root[55] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_19)}: [t470 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_57}] =
        permute x=t469 {derived} perm=[H<-C, W<-H, C<-W]
    n156 {pt2=root[56] torch.ops.aten.hardtanh.default (hardtanh_13)}: [t471 f32 [H=192
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_13}] =
      hardtanh x=t470 {pt2=root:getitem_57} params={min_val=0; max_val=6}
    group g41 torch.ops.aten.convolution.default:
      n157 {derived}: [t472 f32 [H=14 W=14 C=192] {derived}] =
        permute x=t471 {pt2=root:hardtanh_13} perm=[H<-W, W<-C, C<-H]
      n158 {derived}: [t473 f32 [N=64 T=1 D=1 H=1 W=1 C=192] {derived}] =
        permute
          x=t60 {pt2=root:p_features_7_conv_2_weight target=features.7.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n159 {derived}: [t474 f32 [H=14 W=14 C=64] {derived}] =
        convolution
          x=t472 {derived}
          weight=t473 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n160 {pt2=root[57] torch.ops.aten.convolution.default (convolution_20)}: [t475 f32 [H=64
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_20}] =
        permute x=t474 {derived} perm=[H<-C, W<-H, C<-W]
    group g42 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n161 {derived}: [t476 f32 [H=14 W=14 C=64] {derived}] =
        permute x=t475 {pt2=root:convolution_20} perm=[H<-W, W<-C, C<-H]
      n162 {derived}: [t477 f32 [H=14 W=14 C=64] {derived}] =
        batch_norm
          x=t476 {derived}
          weight=t61 {pt2=root:p_features_7_conv_3_weight target=features.7.conv.3.weight}
          bias=t62 {pt2=root:p_features_7_conv_3_bias target=features.7.conv.3.bias}
          running_mean=t218 {pt2=root:b_features_7_conv_3_running_mean target=features.7.conv.3.running_mean}
          running_var=t219 {pt2=root:b_features_7_conv_3_running_var target=features.7.conv.3.running_var}
          params={channel=C; eps=1e-05}
      n163 {pt2=root[58] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_20)}: [t478 f32 [H=64
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_60}] =
        permute x=t477 {derived} perm=[H<-C, W<-H, C<-W]
    group g43 torch.ops.aten.convolution.default:
      n164 {derived}: [t479 f32 [H=14 W=14 C=64] {derived}] =
        permute x=t478 {pt2=root:getitem_60} perm=[H<-W, W<-C, C<-H]
      n165 {derived}: [t480 f32 [N=384 T=1 D=1 H=1 W=1 C=64] {derived}] =
        permute
          x=t63 {pt2=root:p_features_8_conv_0_0_weight target=features.8.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n166 {derived}: [t481 f32 [H=14 W=14 C=384] {derived}] =
        convolution
          x=t479 {derived}
          weight=t480 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n167 {pt2=root[59] torch.ops.aten.convolution.default (convolution_21)}: [t482 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_21}] =
        permute x=t481 {derived} perm=[H<-C, W<-H, C<-W]
    group g44 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n168 {derived}: [t483 f32 [H=14 W=14 C=384] {derived}] =
        permute x=t482 {pt2=root:convolution_21} perm=[H<-W, W<-C, C<-H]
      n169 {derived}: [t484 f32 [H=14 W=14 C=384] {derived}] =
        batch_norm
          x=t483 {derived}
          weight=t64 {pt2=root:p_features_8_conv_0_1_weight target=features.8.conv.0.1.weight}
          bias=t65 {pt2=root:p_features_8_conv_0_1_bias target=features.8.conv.0.1.bias}
          running_mean=t221 {pt2=root:b_features_8_conv_0_1_running_mean target=features.8.conv.0.1.running_mean}
          running_var=t222 {pt2=root:b_features_8_conv_0_1_running_var target=features.8.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
      n170 {pt2=root[60] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_21)}: [t485 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_63}] =
        permute x=t484 {derived} perm=[H<-C, W<-H, C<-W]
    n171 {pt2=root[61] torch.ops.aten.hardtanh.default (hardtanh_14)}: [t486 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_14}] =
      hardtanh x=t485 {pt2=root:getitem_63} params={min_val=0; max_val=6}
    group g45 torch.ops.aten.convolution.default:
      n172 {derived}: [t487 f32 [H=14 W=14 C=384] {derived}] =
        permute x=t486 {pt2=root:hardtanh_14} perm=[H<-W, W<-C, C<-H]
      n173 {derived}: [t488 f32 [N=384 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t66 {pt2=root:p_features_8_conv_1_0_weight target=features.8.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n174 {derived}: [t489 f32 [H=14 W=14 C=384] {derived}] =
        convolution
          x=t487 {derived}
          weight=t488 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=384}
      n175 {pt2=root[62] torch.ops.aten.convolution.default (convolution_22)}: [t490 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_22}] =
        permute x=t489 {derived} perm=[H<-C, W<-H, C<-W]
    group g46 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n176 {derived}: [t491 f32 [H=14 W=14 C=384] {derived}] =
        permute x=t490 {pt2=root:convolution_22} perm=[H<-W, W<-C, C<-H]
      n177 {derived}: [t492 f32 [H=14 W=14 C=384] {derived}] =
        batch_norm
          x=t491 {derived}
          weight=t67 {pt2=root:p_features_8_conv_1_1_weight target=features.8.conv.1.1.weight}
          bias=t68 {pt2=root:p_features_8_conv_1_1_bias target=features.8.conv.1.1.bias}
          running_mean=t224 {pt2=root:b_features_8_conv_1_1_running_mean target=features.8.conv.1.1.running_mean}
          running_var=t225 {pt2=root:b_features_8_conv_1_1_running_var target=features.8.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
      n178 {pt2=root[63] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_22)}: [t493 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_66}] =
        permute x=t492 {derived} perm=[H<-C, W<-H, C<-W]
    n179 {pt2=root[64] torch.ops.aten.hardtanh.default (hardtanh_15)}: [t494 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_15}] =
      hardtanh x=t493 {pt2=root:getitem_66} params={min_val=0; max_val=6}
    group g47 torch.ops.aten.convolution.default:
      n180 {derived}: [t495 f32 [H=14 W=14 C=384] {derived}] =
        permute x=t494 {pt2=root:hardtanh_15} perm=[H<-W, W<-C, C<-H]
      n181 {derived}: [t496 f32 [N=64 T=1 D=1 H=1 W=1 C=384] {derived}] =
        permute
          x=t69 {pt2=root:p_features_8_conv_2_weight target=features.8.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n182 {derived}: [t497 f32 [H=14 W=14 C=64] {derived}] =
        convolution
          x=t495 {derived}
          weight=t496 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n183 {pt2=root[65] torch.ops.aten.convolution.default (convolution_23)}: [t498 f32 [H=64
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_23}] =
        permute x=t497 {derived} perm=[H<-C, W<-H, C<-W]
    group g48 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n184 {derived}: [t499 f32 [H=14 W=14 C=64] {derived}] =
        permute x=t498 {pt2=root:convolution_23} perm=[H<-W, W<-C, C<-H]
      n185 {derived}: [t500 f32 [H=14 W=14 C=64] {derived}] =
        batch_norm
          x=t499 {derived}
          weight=t70 {pt2=root:p_features_8_conv_3_weight target=features.8.conv.3.weight}
          bias=t71 {pt2=root:p_features_8_conv_3_bias target=features.8.conv.3.bias}
          running_mean=t227 {pt2=root:b_features_8_conv_3_running_mean target=features.8.conv.3.running_mean}
          running_var=t228 {pt2=root:b_features_8_conv_3_running_var target=features.8.conv.3.running_var}
          params={channel=C; eps=1e-05}
      n186 {pt2=root[66] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_23)}: [t501 f32 [H=64
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_69}] =
        permute x=t500 {derived} perm=[H<-C, W<-H, C<-W]
    n187 {pt2=root[67] torch.ops.aten.add.Tensor (add_3)}: [t502 f32 [H=64 W=14
                                                                      C=14] {pt2=root:add_3}] =
      add a=t478 {pt2=root:getitem_60} b=t501 {pt2=root:getitem_69}
    group g49 torch.ops.aten.convolution.default:
      n188 {derived}: [t503 f32 [H=14 W=14 C=64] {derived}] =
        permute x=t502 {pt2=root:add_3} perm=[H<-W, W<-C, C<-H]
      n189 {derived}: [t504 f32 [N=384 T=1 D=1 H=1 W=1 C=64] {derived}] =
        permute
          x=t72 {pt2=root:p_features_9_conv_0_0_weight target=features.9.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n190 {derived}: [t505 f32 [H=14 W=14 C=384] {derived}] =
        convolution
          x=t503 {derived}
          weight=t504 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n191 {pt2=root[68] torch.ops.aten.convolution.default (convolution_24)}: [t506 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_24}] =
        permute x=t505 {derived} perm=[H<-C, W<-H, C<-W]
    group g50 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n192 {derived}: [t507 f32 [H=14 W=14 C=384] {derived}] =
        permute x=t506 {pt2=root:convolution_24} perm=[H<-W, W<-C, C<-H]
      n193 {derived}: [t508 f32 [H=14 W=14 C=384] {derived}] =
        batch_norm
          x=t507 {derived}
          weight=t73 {pt2=root:p_features_9_conv_0_1_weight target=features.9.conv.0.1.weight}
          bias=t74 {pt2=root:p_features_9_conv_0_1_bias target=features.9.conv.0.1.bias}
          running_mean=t230 {pt2=root:b_features_9_conv_0_1_running_mean target=features.9.conv.0.1.running_mean}
          running_var=t231 {pt2=root:b_features_9_conv_0_1_running_var target=features.9.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
      n194 {pt2=root[69] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_24)}: [t509 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_72}] =
        permute x=t508 {derived} perm=[H<-C, W<-H, C<-W]
    n195 {pt2=root[70] torch.ops.aten.hardtanh.default (hardtanh_16)}: [t510 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_16}] =
      hardtanh x=t509 {pt2=root:getitem_72} params={min_val=0; max_val=6}
    group g51 torch.ops.aten.convolution.default:
      n196 {derived}: [t511 f32 [H=14 W=14 C=384] {derived}] =
        permute x=t510 {pt2=root:hardtanh_16} perm=[H<-W, W<-C, C<-H]
      n197 {derived}: [t512 f32 [N=384 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t75 {pt2=root:p_features_9_conv_1_0_weight target=features.9.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n198 {derived}: [t513 f32 [H=14 W=14 C=384] {derived}] =
        convolution
          x=t511 {derived}
          weight=t512 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=384}
      n199 {pt2=root[71] torch.ops.aten.convolution.default (convolution_25)}: [t514 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_25}] =
        permute x=t513 {derived} perm=[H<-C, W<-H, C<-W]
    group g52 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n200 {derived}: [t515 f32 [H=14 W=14 C=384] {derived}] =
        permute x=t514 {pt2=root:convolution_25} perm=[H<-W, W<-C, C<-H]
      n201 {derived}: [t516 f32 [H=14 W=14 C=384] {derived}] =
        batch_norm
          x=t515 {derived}
          weight=t76 {pt2=root:p_features_9_conv_1_1_weight target=features.9.conv.1.1.weight}
          bias=t77 {pt2=root:p_features_9_conv_1_1_bias target=features.9.conv.1.1.bias}
          running_mean=t233 {pt2=root:b_features_9_conv_1_1_running_mean target=features.9.conv.1.1.running_mean}
          running_var=t234 {pt2=root:b_features_9_conv_1_1_running_var target=features.9.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
      n202 {pt2=root[72] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_25)}: [t517 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_75}] =
        permute x=t516 {derived} perm=[H<-C, W<-H, C<-W]
    n203 {pt2=root[73] torch.ops.aten.hardtanh.default (hardtanh_17)}: [t518 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_17}] =
      hardtanh x=t517 {pt2=root:getitem_75} params={min_val=0; max_val=6}
    group g53 torch.ops.aten.convolution.default:
      n204 {derived}: [t519 f32 [H=14 W=14 C=384] {derived}] =
        permute x=t518 {pt2=root:hardtanh_17} perm=[H<-W, W<-C, C<-H]
      n205 {derived}: [t520 f32 [N=64 T=1 D=1 H=1 W=1 C=384] {derived}] =
        permute
          x=t78 {pt2=root:p_features_9_conv_2_weight target=features.9.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n206 {derived}: [t521 f32 [H=14 W=14 C=64] {derived}] =
        convolution
          x=t519 {derived}
          weight=t520 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n207 {pt2=root[74] torch.ops.aten.convolution.default (convolution_26)}: [t522 f32 [H=64
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_26}] =
        permute x=t521 {derived} perm=[H<-C, W<-H, C<-W]
    group g54 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n208 {derived}: [t523 f32 [H=14 W=14 C=64] {derived}] =
        permute x=t522 {pt2=root:convolution_26} perm=[H<-W, W<-C, C<-H]
      n209 {derived}: [t524 f32 [H=14 W=14 C=64] {derived}] =
        batch_norm
          x=t523 {derived}
          weight=t79 {pt2=root:p_features_9_conv_3_weight target=features.9.conv.3.weight}
          bias=t80 {pt2=root:p_features_9_conv_3_bias target=features.9.conv.3.bias}
          running_mean=t236 {pt2=root:b_features_9_conv_3_running_mean target=features.9.conv.3.running_mean}
          running_var=t237 {pt2=root:b_features_9_conv_3_running_var target=features.9.conv.3.running_var}
          params={channel=C; eps=1e-05}
      n210 {pt2=root[75] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_26)}: [t525 f32 [H=64
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_78}] =
        permute x=t524 {derived} perm=[H<-C, W<-H, C<-W]
    n211 {pt2=root[76] torch.ops.aten.add.Tensor (add_4)}: [t526 f32 [H=64 W=14
                                                                      C=14] {pt2=root:add_4}] =
      add a=t502 {pt2=root:add_3} b=t525 {pt2=root:getitem_78}
    group g55 torch.ops.aten.convolution.default:
      n212 {derived}: [t527 f32 [H=14 W=14 C=64] {derived}] =
        permute x=t526 {pt2=root:add_4} perm=[H<-W, W<-C, C<-H]
      n213 {derived}: [t528 f32 [N=384 T=1 D=1 H=1 W=1 C=64] {derived}] =
        permute
          x=t81 {pt2=root:p_features_10_conv_0_0_weight target=features.10.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n214 {derived}: [t529 f32 [H=14 W=14 C=384] {derived}] =
        convolution
          x=t527 {derived}
          weight=t528 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n215 {pt2=root[77] torch.ops.aten.convolution.default (convolution_27)}: [t530 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_27}] =
        permute x=t529 {derived} perm=[H<-C, W<-H, C<-W]
    group g56 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n216 {derived}: [t531 f32 [H=14 W=14 C=384] {derived}] =
        permute x=t530 {pt2=root:convolution_27} perm=[H<-W, W<-C, C<-H]
      n217 {derived}: [t532 f32 [H=14 W=14 C=384] {derived}] =
        batch_norm
          x=t531 {derived}
          weight=t82 {pt2=root:p_features_10_conv_0_1_weight target=features.10.conv.0.1.weight}
          bias=t83 {pt2=root:p_features_10_conv_0_1_bias target=features.10.conv.0.1.bias}
          running_mean=t239 {pt2=root:b_features_10_conv_0_1_running_mean target=features.10.conv.0.1.running_mean}
          running_var=t240 {pt2=root:b_features_10_conv_0_1_running_var target=features.10.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
      n218 {pt2=root[78] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_27)}: [t533 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_81}] =
        permute x=t532 {derived} perm=[H<-C, W<-H, C<-W]
    n219 {pt2=root[79] torch.ops.aten.hardtanh.default (hardtanh_18)}: [t534 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_18}] =
      hardtanh x=t533 {pt2=root:getitem_81} params={min_val=0; max_val=6}
    group g57 torch.ops.aten.convolution.default:
      n220 {derived}: [t535 f32 [H=14 W=14 C=384] {derived}] =
        permute x=t534 {pt2=root:hardtanh_18} perm=[H<-W, W<-C, C<-H]
      n221 {derived}: [t536 f32 [N=384 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t84 {pt2=root:p_features_10_conv_1_0_weight target=features.10.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n222 {derived}: [t537 f32 [H=14 W=14 C=384] {derived}] =
        convolution
          x=t535 {derived}
          weight=t536 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=384}
      n223 {pt2=root[80] torch.ops.aten.convolution.default (convolution_28)}: [t538 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_28}] =
        permute x=t537 {derived} perm=[H<-C, W<-H, C<-W]
    group g58 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n224 {derived}: [t539 f32 [H=14 W=14 C=384] {derived}] =
        permute x=t538 {pt2=root:convolution_28} perm=[H<-W, W<-C, C<-H]
      n225 {derived}: [t540 f32 [H=14 W=14 C=384] {derived}] =
        batch_norm
          x=t539 {derived}
          weight=t85 {pt2=root:p_features_10_conv_1_1_weight target=features.10.conv.1.1.weight}
          bias=t86 {pt2=root:p_features_10_conv_1_1_bias target=features.10.conv.1.1.bias}
          running_mean=t242 {pt2=root:b_features_10_conv_1_1_running_mean target=features.10.conv.1.1.running_mean}
          running_var=t243 {pt2=root:b_features_10_conv_1_1_running_var target=features.10.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
      n226 {pt2=root[81] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_28)}: [t541 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_84}] =
        permute x=t540 {derived} perm=[H<-C, W<-H, C<-W]
    n227 {pt2=root[82] torch.ops.aten.hardtanh.default (hardtanh_19)}: [t542 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_19}] =
      hardtanh x=t541 {pt2=root:getitem_84} params={min_val=0; max_val=6}
    group g59 torch.ops.aten.convolution.default:
      n228 {derived}: [t543 f32 [H=14 W=14 C=384] {derived}] =
        permute x=t542 {pt2=root:hardtanh_19} perm=[H<-W, W<-C, C<-H]
      n229 {derived}: [t544 f32 [N=64 T=1 D=1 H=1 W=1 C=384] {derived}] =
        permute
          x=t87 {pt2=root:p_features_10_conv_2_weight target=features.10.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n230 {derived}: [t545 f32 [H=14 W=14 C=64] {derived}] =
        convolution
          x=t543 {derived}
          weight=t544 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n231 {pt2=root[83] torch.ops.aten.convolution.default (convolution_29)}: [t546 f32 [H=64
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_29}] =
        permute x=t545 {derived} perm=[H<-C, W<-H, C<-W]
    group g60 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n232 {derived}: [t547 f32 [H=14 W=14 C=64] {derived}] =
        permute x=t546 {pt2=root:convolution_29} perm=[H<-W, W<-C, C<-H]
      n233 {derived}: [t548 f32 [H=14 W=14 C=64] {derived}] =
        batch_norm
          x=t547 {derived}
          weight=t88 {pt2=root:p_features_10_conv_3_weight target=features.10.conv.3.weight}
          bias=t89 {pt2=root:p_features_10_conv_3_bias target=features.10.conv.3.bias}
          running_mean=t245 {pt2=root:b_features_10_conv_3_running_mean target=features.10.conv.3.running_mean}
          running_var=t246 {pt2=root:b_features_10_conv_3_running_var target=features.10.conv.3.running_var}
          params={channel=C; eps=1e-05}
      n234 {pt2=root[84] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_29)}: [t549 f32 [H=64
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_87}] =
        permute x=t548 {derived} perm=[H<-C, W<-H, C<-W]
    n235 {pt2=root[85] torch.ops.aten.add.Tensor (add_5)}: [t550 f32 [H=64 W=14
                                                                      C=14] {pt2=root:add_5}] =
      add a=t526 {pt2=root:add_4} b=t549 {pt2=root:getitem_87}
    group g61 torch.ops.aten.convolution.default:
      n236 {derived}: [t551 f32 [H=14 W=14 C=64] {derived}] =
        permute x=t550 {pt2=root:add_5} perm=[H<-W, W<-C, C<-H]
      n237 {derived}: [t552 f32 [N=384 T=1 D=1 H=1 W=1 C=64] {derived}] =
        permute
          x=t90 {pt2=root:p_features_11_conv_0_0_weight target=features.11.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n238 {derived}: [t553 f32 [H=14 W=14 C=384] {derived}] =
        convolution
          x=t551 {derived}
          weight=t552 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n239 {pt2=root[86] torch.ops.aten.convolution.default (convolution_30)}: [t554 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_30}] =
        permute x=t553 {derived} perm=[H<-C, W<-H, C<-W]
    group g62 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n240 {derived}: [t555 f32 [H=14 W=14 C=384] {derived}] =
        permute x=t554 {pt2=root:convolution_30} perm=[H<-W, W<-C, C<-H]
      n241 {derived}: [t556 f32 [H=14 W=14 C=384] {derived}] =
        batch_norm
          x=t555 {derived}
          weight=t91 {pt2=root:p_features_11_conv_0_1_weight target=features.11.conv.0.1.weight}
          bias=t92 {pt2=root:p_features_11_conv_0_1_bias target=features.11.conv.0.1.bias}
          running_mean=t248 {pt2=root:b_features_11_conv_0_1_running_mean target=features.11.conv.0.1.running_mean}
          running_var=t249 {pt2=root:b_features_11_conv_0_1_running_var target=features.11.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
      n242 {pt2=root[87] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_30)}: [t557 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_90}] =
        permute x=t556 {derived} perm=[H<-C, W<-H, C<-W]
    n243 {pt2=root[88] torch.ops.aten.hardtanh.default (hardtanh_20)}: [t558 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_20}] =
      hardtanh x=t557 {pt2=root:getitem_90} params={min_val=0; max_val=6}
    group g63 torch.ops.aten.convolution.default:
      n244 {derived}: [t559 f32 [H=14 W=14 C=384] {derived}] =
        permute x=t558 {pt2=root:hardtanh_20} perm=[H<-W, W<-C, C<-H]
      n245 {derived}: [t560 f32 [N=384 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t93 {pt2=root:p_features_11_conv_1_0_weight target=features.11.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n246 {derived}: [t561 f32 [H=14 W=14 C=384] {derived}] =
        convolution
          x=t559 {derived}
          weight=t560 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=384}
      n247 {pt2=root[89] torch.ops.aten.convolution.default (convolution_31)}: [t562 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_31}] =
        permute x=t561 {derived} perm=[H<-C, W<-H, C<-W]
    group g64 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n248 {derived}: [t563 f32 [H=14 W=14 C=384] {derived}] =
        permute x=t562 {pt2=root:convolution_31} perm=[H<-W, W<-C, C<-H]
      n249 {derived}: [t564 f32 [H=14 W=14 C=384] {derived}] =
        batch_norm
          x=t563 {derived}
          weight=t94 {pt2=root:p_features_11_conv_1_1_weight target=features.11.conv.1.1.weight}
          bias=t95 {pt2=root:p_features_11_conv_1_1_bias target=features.11.conv.1.1.bias}
          running_mean=t251 {pt2=root:b_features_11_conv_1_1_running_mean target=features.11.conv.1.1.running_mean}
          running_var=t252 {pt2=root:b_features_11_conv_1_1_running_var target=features.11.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
      n250 {pt2=root[90] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_31)}: [t565 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_93}] =
        permute x=t564 {derived} perm=[H<-C, W<-H, C<-W]
    n251 {pt2=root[91] torch.ops.aten.hardtanh.default (hardtanh_21)}: [t566 f32 [H=384
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_21}] =
      hardtanh x=t565 {pt2=root:getitem_93} params={min_val=0; max_val=6}
    group g65 torch.ops.aten.convolution.default:
      n252 {derived}: [t567 f32 [H=14 W=14 C=384] {derived}] =
        permute x=t566 {pt2=root:hardtanh_21} perm=[H<-W, W<-C, C<-H]
      n253 {derived}: [t568 f32 [N=96 T=1 D=1 H=1 W=1 C=384] {derived}] =
        permute
          x=t96 {pt2=root:p_features_11_conv_2_weight target=features.11.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n254 {derived}: [t569 f32 [H=14 W=14 C=96] {derived}] =
        convolution
          x=t567 {derived}
          weight=t568 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n255 {pt2=root[92] torch.ops.aten.convolution.default (convolution_32)}: [t570 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_32}] =
        permute x=t569 {derived} perm=[H<-C, W<-H, C<-W]
    group g66 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n256 {derived}: [t571 f32 [H=14 W=14 C=96] {derived}] =
        permute x=t570 {pt2=root:convolution_32} perm=[H<-W, W<-C, C<-H]
      n257 {derived}: [t572 f32 [H=14 W=14 C=96] {derived}] =
        batch_norm
          x=t571 {derived}
          weight=t97 {pt2=root:p_features_11_conv_3_weight target=features.11.conv.3.weight}
          bias=t98 {pt2=root:p_features_11_conv_3_bias target=features.11.conv.3.bias}
          running_mean=t254 {pt2=root:b_features_11_conv_3_running_mean target=features.11.conv.3.running_mean}
          running_var=t255 {pt2=root:b_features_11_conv_3_running_var target=features.11.conv.3.running_var}
          params={channel=C; eps=1e-05}
      n258 {pt2=root[93] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_32)}: [t573 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_96}] =
        permute x=t572 {derived} perm=[H<-C, W<-H, C<-W]
    group g67 torch.ops.aten.convolution.default:
      n259 {derived}: [t574 f32 [H=14 W=14 C=96] {derived}] =
        permute x=t573 {pt2=root:getitem_96} perm=[H<-W, W<-C, C<-H]
      n260 {derived}: [t575 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t99 {pt2=root:p_features_12_conv_0_0_weight target=features.12.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n261 {derived}: [t576 f32 [H=14 W=14 C=576] {derived}] =
        convolution
          x=t574 {derived}
          weight=t575 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n262 {pt2=root[94] torch.ops.aten.convolution.default (convolution_33)}: [t577 f32 [H=576
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_33}] =
        permute x=t576 {derived} perm=[H<-C, W<-H, C<-W]
    group g68 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n263 {derived}: [t578 f32 [H=14 W=14 C=576] {derived}] =
        permute x=t577 {pt2=root:convolution_33} perm=[H<-W, W<-C, C<-H]
      n264 {derived}: [t579 f32 [H=14 W=14 C=576] {derived}] =
        batch_norm
          x=t578 {derived}
          weight=t100 {pt2=root:p_features_12_conv_0_1_weight target=features.12.conv.0.1.weight}
          bias=t101 {pt2=root:p_features_12_conv_0_1_bias target=features.12.conv.0.1.bias}
          running_mean=t257 {pt2=root:b_features_12_conv_0_1_running_mean target=features.12.conv.0.1.running_mean}
          running_var=t258 {pt2=root:b_features_12_conv_0_1_running_var target=features.12.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
      n265 {pt2=root[95] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_33)}: [t580 f32 [H=576
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_99}] =
        permute x=t579 {derived} perm=[H<-C, W<-H, C<-W]
    n266 {pt2=root[96] torch.ops.aten.hardtanh.default (hardtanh_22)}: [t581 f32 [H=576
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_22}] =
      hardtanh x=t580 {pt2=root:getitem_99} params={min_val=0; max_val=6}
    group g69 torch.ops.aten.convolution.default:
      n267 {derived}: [t582 f32 [H=14 W=14 C=576] {derived}] =
        permute x=t581 {pt2=root:hardtanh_22} perm=[H<-W, W<-C, C<-H]
      n268 {derived}: [t583 f32 [N=576 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t102 {pt2=root:p_features_12_conv_1_0_weight target=features.12.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n269 {derived}: [t584 f32 [H=14 W=14 C=576] {derived}] =
        convolution
          x=t582 {derived}
          weight=t583 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=576}
      n270 {pt2=root[97] torch.ops.aten.convolution.default (convolution_34)}: [t585 f32 [H=576
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_34}] =
        permute x=t584 {derived} perm=[H<-C, W<-H, C<-W]
    group g70 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n271 {derived}: [t586 f32 [H=14 W=14 C=576] {derived}] =
        permute x=t585 {pt2=root:convolution_34} perm=[H<-W, W<-C, C<-H]
      n272 {derived}: [t587 f32 [H=14 W=14 C=576] {derived}] =
        batch_norm
          x=t586 {derived}
          weight=t103 {pt2=root:p_features_12_conv_1_1_weight target=features.12.conv.1.1.weight}
          bias=t104 {pt2=root:p_features_12_conv_1_1_bias target=features.12.conv.1.1.bias}
          running_mean=t260 {pt2=root:b_features_12_conv_1_1_running_mean target=features.12.conv.1.1.running_mean}
          running_var=t261 {pt2=root:b_features_12_conv_1_1_running_var target=features.12.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
      n273 {pt2=root[98] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_34)}: [t588 f32 [H=576
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_102}] =
        permute x=t587 {derived} perm=[H<-C, W<-H, C<-W]
    n274 {pt2=root[99] torch.ops.aten.hardtanh.default (hardtanh_23)}: [t589 f32 [H=576
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_23}] =
      hardtanh x=t588 {pt2=root:getitem_102} params={min_val=0; max_val=6}
    group g71 torch.ops.aten.convolution.default:
      n275 {derived}: [t590 f32 [H=14 W=14 C=576] {derived}] =
        permute x=t589 {pt2=root:hardtanh_23} perm=[H<-W, W<-C, C<-H]
      n276 {derived}: [t591 f32 [N=96 T=1 D=1 H=1 W=1 C=576] {derived}] =
        permute
          x=t105 {pt2=root:p_features_12_conv_2_weight target=features.12.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n277 {derived}: [t592 f32 [H=14 W=14 C=96] {derived}] =
        convolution
          x=t590 {derived}
          weight=t591 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n278 {pt2=root[100] torch.ops.aten.convolution.default (convolution_35)}: [t593 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_35}] =
        permute x=t592 {derived} perm=[H<-C, W<-H, C<-W]
    group g72 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n279 {derived}: [t594 f32 [H=14 W=14 C=96] {derived}] =
        permute x=t593 {pt2=root:convolution_35} perm=[H<-W, W<-C, C<-H]
      n280 {derived}: [t595 f32 [H=14 W=14 C=96] {derived}] =
        batch_norm
          x=t594 {derived}
          weight=t106 {pt2=root:p_features_12_conv_3_weight target=features.12.conv.3.weight}
          bias=t107 {pt2=root:p_features_12_conv_3_bias target=features.12.conv.3.bias}
          running_mean=t263 {pt2=root:b_features_12_conv_3_running_mean target=features.12.conv.3.running_mean}
          running_var=t264 {pt2=root:b_features_12_conv_3_running_var target=features.12.conv.3.running_var}
          params={channel=C; eps=1e-05}
      n281 {pt2=root[101] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_35)}: [t596 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_105}] =
        permute x=t595 {derived} perm=[H<-C, W<-H, C<-W]
    n282 {pt2=root[102] torch.ops.aten.add.Tensor (add_6)}: [t597 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:add_6}] =
      add a=t573 {pt2=root:getitem_96} b=t596 {pt2=root:getitem_105}
    group g73 torch.ops.aten.convolution.default:
      n283 {derived}: [t598 f32 [H=14 W=14 C=96] {derived}] =
        permute x=t597 {pt2=root:add_6} perm=[H<-W, W<-C, C<-H]
      n284 {derived}: [t599 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t108 {pt2=root:p_features_13_conv_0_0_weight target=features.13.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n285 {derived}: [t600 f32 [H=14 W=14 C=576] {derived}] =
        convolution
          x=t598 {derived}
          weight=t599 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n286 {pt2=root[103] torch.ops.aten.convolution.default (convolution_36)}: [t601 f32 [H=576
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_36}] =
        permute x=t600 {derived} perm=[H<-C, W<-H, C<-W]
    group g74 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n287 {derived}: [t602 f32 [H=14 W=14 C=576] {derived}] =
        permute x=t601 {pt2=root:convolution_36} perm=[H<-W, W<-C, C<-H]
      n288 {derived}: [t603 f32 [H=14 W=14 C=576] {derived}] =
        batch_norm
          x=t602 {derived}
          weight=t109 {pt2=root:p_features_13_conv_0_1_weight target=features.13.conv.0.1.weight}
          bias=t110 {pt2=root:p_features_13_conv_0_1_bias target=features.13.conv.0.1.bias}
          running_mean=t266 {pt2=root:b_features_13_conv_0_1_running_mean target=features.13.conv.0.1.running_mean}
          running_var=t267 {pt2=root:b_features_13_conv_0_1_running_var target=features.13.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
      n289 {pt2=root[104] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_36)}: [t604 f32 [H=576
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_108}] =
        permute x=t603 {derived} perm=[H<-C, W<-H, C<-W]
    n290 {pt2=root[105] torch.ops.aten.hardtanh.default (hardtanh_24)}: [t605 f32 [H=576
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_24}] =
      hardtanh x=t604 {pt2=root:getitem_108} params={min_val=0; max_val=6}
    group g75 torch.ops.aten.convolution.default:
      n291 {derived}: [t606 f32 [H=14 W=14 C=576] {derived}] =
        permute x=t605 {pt2=root:hardtanh_24} perm=[H<-W, W<-C, C<-H]
      n292 {derived}: [t607 f32 [N=576 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t111 {pt2=root:p_features_13_conv_1_0_weight target=features.13.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n293 {derived}: [t608 f32 [H=14 W=14 C=576] {derived}] =
        convolution
          x=t606 {derived}
          weight=t607 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=576}
      n294 {pt2=root[106] torch.ops.aten.convolution.default (convolution_37)}: [t609 f32 [H=576
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_37}] =
        permute x=t608 {derived} perm=[H<-C, W<-H, C<-W]
    group g76 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n295 {derived}: [t610 f32 [H=14 W=14 C=576] {derived}] =
        permute x=t609 {pt2=root:convolution_37} perm=[H<-W, W<-C, C<-H]
      n296 {derived}: [t611 f32 [H=14 W=14 C=576] {derived}] =
        batch_norm
          x=t610 {derived}
          weight=t112 {pt2=root:p_features_13_conv_1_1_weight target=features.13.conv.1.1.weight}
          bias=t113 {pt2=root:p_features_13_conv_1_1_bias target=features.13.conv.1.1.bias}
          running_mean=t269 {pt2=root:b_features_13_conv_1_1_running_mean target=features.13.conv.1.1.running_mean}
          running_var=t270 {pt2=root:b_features_13_conv_1_1_running_var target=features.13.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
      n297 {pt2=root[107] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_37)}: [t612 f32 [H=576
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_111}] =
        permute x=t611 {derived} perm=[H<-C, W<-H, C<-W]
    n298 {pt2=root[108] torch.ops.aten.hardtanh.default (hardtanh_25)}: [t613 f32 [H=576
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_25}] =
      hardtanh x=t612 {pt2=root:getitem_111} params={min_val=0; max_val=6}
    group g77 torch.ops.aten.convolution.default:
      n299 {derived}: [t614 f32 [H=14 W=14 C=576] {derived}] =
        permute x=t613 {pt2=root:hardtanh_25} perm=[H<-W, W<-C, C<-H]
      n300 {derived}: [t615 f32 [N=96 T=1 D=1 H=1 W=1 C=576] {derived}] =
        permute
          x=t114 {pt2=root:p_features_13_conv_2_weight target=features.13.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n301 {derived}: [t616 f32 [H=14 W=14 C=96] {derived}] =
        convolution
          x=t614 {derived}
          weight=t615 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n302 {pt2=root[109] torch.ops.aten.convolution.default (convolution_38)}: [t617 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_38}] =
        permute x=t616 {derived} perm=[H<-C, W<-H, C<-W]
    group g78 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n303 {derived}: [t618 f32 [H=14 W=14 C=96] {derived}] =
        permute x=t617 {pt2=root:convolution_38} perm=[H<-W, W<-C, C<-H]
      n304 {derived}: [t619 f32 [H=14 W=14 C=96] {derived}] =
        batch_norm
          x=t618 {derived}
          weight=t115 {pt2=root:p_features_13_conv_3_weight target=features.13.conv.3.weight}
          bias=t116 {pt2=root:p_features_13_conv_3_bias target=features.13.conv.3.bias}
          running_mean=t272 {pt2=root:b_features_13_conv_3_running_mean target=features.13.conv.3.running_mean}
          running_var=t273 {pt2=root:b_features_13_conv_3_running_var target=features.13.conv.3.running_var}
          params={channel=C; eps=1e-05}
      n305 {pt2=root[110] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_38)}: [t620 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_114}] =
        permute x=t619 {derived} perm=[H<-C, W<-H, C<-W]
    n306 {pt2=root[111] torch.ops.aten.add.Tensor (add_7)}: [t621 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:add_7}] =
      add a=t597 {pt2=root:add_6} b=t620 {pt2=root:getitem_114}
    group g79 torch.ops.aten.convolution.default:
      n307 {derived}: [t622 f32 [H=14 W=14 C=96] {derived}] =
        permute x=t621 {pt2=root:add_7} perm=[H<-W, W<-C, C<-H]
      n308 {derived}: [t623 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t117 {pt2=root:p_features_14_conv_0_0_weight target=features.14.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n309 {derived}: [t624 f32 [H=14 W=14 C=576] {derived}] =
        convolution
          x=t622 {derived}
          weight=t623 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n310 {pt2=root[112] torch.ops.aten.convolution.default (convolution_39)}: [t625 f32 [H=576
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_39}] =
        permute x=t624 {derived} perm=[H<-C, W<-H, C<-W]
    group g80 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n311 {derived}: [t626 f32 [H=14 W=14 C=576] {derived}] =
        permute x=t625 {pt2=root:convolution_39} perm=[H<-W, W<-C, C<-H]
      n312 {derived}: [t627 f32 [H=14 W=14 C=576] {derived}] =
        batch_norm
          x=t626 {derived}
          weight=t118 {pt2=root:p_features_14_conv_0_1_weight target=features.14.conv.0.1.weight}
          bias=t119 {pt2=root:p_features_14_conv_0_1_bias target=features.14.conv.0.1.bias}
          running_mean=t275 {pt2=root:b_features_14_conv_0_1_running_mean target=features.14.conv.0.1.running_mean}
          running_var=t276 {pt2=root:b_features_14_conv_0_1_running_var target=features.14.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
      n313 {pt2=root[113] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_39)}: [t628 f32 [H=576
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_117}] =
        permute x=t627 {derived} perm=[H<-C, W<-H, C<-W]
    n314 {pt2=root[114] torch.ops.aten.hardtanh.default (hardtanh_26)}: [t629 f32 [H=576
                                                                      W=14
                                                                      C=14] {pt2=root:hardtanh_26}] =
      hardtanh x=t628 {pt2=root:getitem_117} params={min_val=0; max_val=6}
    group g81 torch.ops.aten.convolution.default:
      n315 {derived}: [t630 f32 [H=14 W=14 C=576] {derived}] =
        permute x=t629 {pt2=root:hardtanh_26} perm=[H<-W, W<-C, C<-H]
      n316 {derived}: [t631 f32 [N=576 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t120 {pt2=root:p_features_14_conv_1_0_weight target=features.14.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n317 {derived}: [t632 f32 [H=7 W=7 C=576] {derived}] =
        convolution
          x=t630 {derived}
          weight=t631 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=576}
      n318 {pt2=root[115] torch.ops.aten.convolution.default (convolution_40)}: [t633 f32 [H=576
                                                                      W=7 C=7] {pt2=root:convolution_40}] =
        permute x=t632 {derived} perm=[H<-C, W<-H, C<-W]
    group g82 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n319 {derived}: [t634 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t633 {pt2=root:convolution_40} perm=[H<-W, W<-C, C<-H]
      n320 {derived}: [t635 f32 [H=7 W=7 C=576] {derived}] =
        batch_norm
          x=t634 {derived}
          weight=t121 {pt2=root:p_features_14_conv_1_1_weight target=features.14.conv.1.1.weight}
          bias=t122 {pt2=root:p_features_14_conv_1_1_bias target=features.14.conv.1.1.bias}
          running_mean=t278 {pt2=root:b_features_14_conv_1_1_running_mean target=features.14.conv.1.1.running_mean}
          running_var=t279 {pt2=root:b_features_14_conv_1_1_running_var target=features.14.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
      n321 {pt2=root[116] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_40)}: [t636 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_120}] =
        permute x=t635 {derived} perm=[H<-C, W<-H, C<-W]
    n322 {pt2=root[117] torch.ops.aten.hardtanh.default (hardtanh_27)}: [t637 f32 [H=576
                                                                      W=7 C=7] {pt2=root:hardtanh_27}] =
      hardtanh x=t636 {pt2=root:getitem_120} params={min_val=0; max_val=6}
    group g83 torch.ops.aten.convolution.default:
      n323 {derived}: [t638 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t637 {pt2=root:hardtanh_27} perm=[H<-W, W<-C, C<-H]
      n324 {derived}: [t639 f32 [N=160 T=1 D=1 H=1 W=1 C=576] {derived}] =
        permute
          x=t123 {pt2=root:p_features_14_conv_2_weight target=features.14.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n325 {derived}: [t640 f32 [H=7 W=7 C=160] {derived}] =
        convolution
          x=t638 {derived}
          weight=t639 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n326 {pt2=root[118] torch.ops.aten.convolution.default (convolution_41)}: [t641 f32 [H=160
                                                                      W=7 C=7] {pt2=root:convolution_41}] =
        permute x=t640 {derived} perm=[H<-C, W<-H, C<-W]
    group g84 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n327 {derived}: [t642 f32 [H=7 W=7 C=160] {derived}] =
        permute x=t641 {pt2=root:convolution_41} perm=[H<-W, W<-C, C<-H]
      n328 {derived}: [t643 f32 [H=7 W=7 C=160] {derived}] =
        batch_norm
          x=t642 {derived}
          weight=t124 {pt2=root:p_features_14_conv_3_weight target=features.14.conv.3.weight}
          bias=t125 {pt2=root:p_features_14_conv_3_bias target=features.14.conv.3.bias}
          running_mean=t281 {pt2=root:b_features_14_conv_3_running_mean target=features.14.conv.3.running_mean}
          running_var=t282 {pt2=root:b_features_14_conv_3_running_var target=features.14.conv.3.running_var}
          params={channel=C; eps=1e-05}
      n329 {pt2=root[119] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_41)}: [t644 f32 [H=160
                                                                      W=7 C=7] {pt2=root:getitem_123}] =
        permute x=t643 {derived} perm=[H<-C, W<-H, C<-W]
    group g85 torch.ops.aten.convolution.default:
      n330 {derived}: [t645 f32 [H=7 W=7 C=160] {derived}] =
        permute x=t644 {pt2=root:getitem_123} perm=[H<-W, W<-C, C<-H]
      n331 {derived}: [t646 f32 [N=960 T=1 D=1 H=1 W=1 C=160] {derived}] =
        permute
          x=t126 {pt2=root:p_features_15_conv_0_0_weight target=features.15.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n332 {derived}: [t647 f32 [H=7 W=7 C=960] {derived}] =
        convolution
          x=t645 {derived}
          weight=t646 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n333 {pt2=root[120] torch.ops.aten.convolution.default (convolution_42)}: [t648 f32 [H=960
                                                                      W=7 C=7] {pt2=root:convolution_42}] =
        permute x=t647 {derived} perm=[H<-C, W<-H, C<-W]
    group g86 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n334 {derived}: [t649 f32 [H=7 W=7 C=960] {derived}] =
        permute x=t648 {pt2=root:convolution_42} perm=[H<-W, W<-C, C<-H]
      n335 {derived}: [t650 f32 [H=7 W=7 C=960] {derived}] =
        batch_norm
          x=t649 {derived}
          weight=t127 {pt2=root:p_features_15_conv_0_1_weight target=features.15.conv.0.1.weight}
          bias=t128 {pt2=root:p_features_15_conv_0_1_bias target=features.15.conv.0.1.bias}
          running_mean=t284 {pt2=root:b_features_15_conv_0_1_running_mean target=features.15.conv.0.1.running_mean}
          running_var=t285 {pt2=root:b_features_15_conv_0_1_running_var target=features.15.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
      n336 {pt2=root[121] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_42)}: [t651 f32 [H=960
                                                                      W=7 C=7] {pt2=root:getitem_126}] =
        permute x=t650 {derived} perm=[H<-C, W<-H, C<-W]
    n337 {pt2=root[122] torch.ops.aten.hardtanh.default (hardtanh_28)}: [t652 f32 [H=960
                                                                      W=7 C=7] {pt2=root:hardtanh_28}] =
      hardtanh x=t651 {pt2=root:getitem_126} params={min_val=0; max_val=6}
    group g87 torch.ops.aten.convolution.default:
      n338 {derived}: [t653 f32 [H=7 W=7 C=960] {derived}] =
        permute x=t652 {pt2=root:hardtanh_28} perm=[H<-W, W<-C, C<-H]
      n339 {derived}: [t654 f32 [N=960 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t129 {pt2=root:p_features_15_conv_1_0_weight target=features.15.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n340 {derived}: [t655 f32 [H=7 W=7 C=960] {derived}] =
        convolution
          x=t653 {derived}
          weight=t654 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=960}
      n341 {pt2=root[123] torch.ops.aten.convolution.default (convolution_43)}: [t656 f32 [H=960
                                                                      W=7 C=7] {pt2=root:convolution_43}] =
        permute x=t655 {derived} perm=[H<-C, W<-H, C<-W]
    group g88 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n342 {derived}: [t657 f32 [H=7 W=7 C=960] {derived}] =
        permute x=t656 {pt2=root:convolution_43} perm=[H<-W, W<-C, C<-H]
      n343 {derived}: [t658 f32 [H=7 W=7 C=960] {derived}] =
        batch_norm
          x=t657 {derived}
          weight=t130 {pt2=root:p_features_15_conv_1_1_weight target=features.15.conv.1.1.weight}
          bias=t131 {pt2=root:p_features_15_conv_1_1_bias target=features.15.conv.1.1.bias}
          running_mean=t287 {pt2=root:b_features_15_conv_1_1_running_mean target=features.15.conv.1.1.running_mean}
          running_var=t288 {pt2=root:b_features_15_conv_1_1_running_var target=features.15.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
      n344 {pt2=root[124] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_43)}: [t659 f32 [H=960
                                                                      W=7 C=7] {pt2=root:getitem_129}] =
        permute x=t658 {derived} perm=[H<-C, W<-H, C<-W]
    n345 {pt2=root[125] torch.ops.aten.hardtanh.default (hardtanh_29)}: [t660 f32 [H=960
                                                                      W=7 C=7] {pt2=root:hardtanh_29}] =
      hardtanh x=t659 {pt2=root:getitem_129} params={min_val=0; max_val=6}
    group g89 torch.ops.aten.convolution.default:
      n346 {derived}: [t661 f32 [H=7 W=7 C=960] {derived}] =
        permute x=t660 {pt2=root:hardtanh_29} perm=[H<-W, W<-C, C<-H]
      n347 {derived}: [t662 f32 [N=160 T=1 D=1 H=1 W=1 C=960] {derived}] =
        permute
          x=t132 {pt2=root:p_features_15_conv_2_weight target=features.15.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n348 {derived}: [t663 f32 [H=7 W=7 C=160] {derived}] =
        convolution
          x=t661 {derived}
          weight=t662 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n349 {pt2=root[126] torch.ops.aten.convolution.default (convolution_44)}: [t664 f32 [H=160
                                                                      W=7 C=7] {pt2=root:convolution_44}] =
        permute x=t663 {derived} perm=[H<-C, W<-H, C<-W]
    group g90 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n350 {derived}: [t665 f32 [H=7 W=7 C=160] {derived}] =
        permute x=t664 {pt2=root:convolution_44} perm=[H<-W, W<-C, C<-H]
      n351 {derived}: [t666 f32 [H=7 W=7 C=160] {derived}] =
        batch_norm
          x=t665 {derived}
          weight=t133 {pt2=root:p_features_15_conv_3_weight target=features.15.conv.3.weight}
          bias=t134 {pt2=root:p_features_15_conv_3_bias target=features.15.conv.3.bias}
          running_mean=t290 {pt2=root:b_features_15_conv_3_running_mean target=features.15.conv.3.running_mean}
          running_var=t291 {pt2=root:b_features_15_conv_3_running_var target=features.15.conv.3.running_var}
          params={channel=C; eps=1e-05}
      n352 {pt2=root[127] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_44)}: [t667 f32 [H=160
                                                                      W=7 C=7] {pt2=root:getitem_132}] =
        permute x=t666 {derived} perm=[H<-C, W<-H, C<-W]
    n353 {pt2=root[128] torch.ops.aten.add.Tensor (add_8)}: [t668 f32 [H=160
                                                                      W=7 C=7] {pt2=root:add_8}] =
      add a=t644 {pt2=root:getitem_123} b=t667 {pt2=root:getitem_132}
    group g91 torch.ops.aten.convolution.default:
      n354 {derived}: [t669 f32 [H=7 W=7 C=160] {derived}] =
        permute x=t668 {pt2=root:add_8} perm=[H<-W, W<-C, C<-H]
      n355 {derived}: [t670 f32 [N=960 T=1 D=1 H=1 W=1 C=160] {derived}] =
        permute
          x=t135 {pt2=root:p_features_16_conv_0_0_weight target=features.16.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n356 {derived}: [t671 f32 [H=7 W=7 C=960] {derived}] =
        convolution
          x=t669 {derived}
          weight=t670 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n357 {pt2=root[129] torch.ops.aten.convolution.default (convolution_45)}: [t672 f32 [H=960
                                                                      W=7 C=7] {pt2=root:convolution_45}] =
        permute x=t671 {derived} perm=[H<-C, W<-H, C<-W]
    group g92 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n358 {derived}: [t673 f32 [H=7 W=7 C=960] {derived}] =
        permute x=t672 {pt2=root:convolution_45} perm=[H<-W, W<-C, C<-H]
      n359 {derived}: [t674 f32 [H=7 W=7 C=960] {derived}] =
        batch_norm
          x=t673 {derived}
          weight=t136 {pt2=root:p_features_16_conv_0_1_weight target=features.16.conv.0.1.weight}
          bias=t137 {pt2=root:p_features_16_conv_0_1_bias target=features.16.conv.0.1.bias}
          running_mean=t293 {pt2=root:b_features_16_conv_0_1_running_mean target=features.16.conv.0.1.running_mean}
          running_var=t294 {pt2=root:b_features_16_conv_0_1_running_var target=features.16.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
      n360 {pt2=root[130] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_45)}: [t675 f32 [H=960
                                                                      W=7 C=7] {pt2=root:getitem_135}] =
        permute x=t674 {derived} perm=[H<-C, W<-H, C<-W]
    n361 {pt2=root[131] torch.ops.aten.hardtanh.default (hardtanh_30)}: [t676 f32 [H=960
                                                                      W=7 C=7] {pt2=root:hardtanh_30}] =
      hardtanh x=t675 {pt2=root:getitem_135} params={min_val=0; max_val=6}
    group g93 torch.ops.aten.convolution.default:
      n362 {derived}: [t677 f32 [H=7 W=7 C=960] {derived}] =
        permute x=t676 {pt2=root:hardtanh_30} perm=[H<-W, W<-C, C<-H]
      n363 {derived}: [t678 f32 [N=960 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t138 {pt2=root:p_features_16_conv_1_0_weight target=features.16.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n364 {derived}: [t679 f32 [H=7 W=7 C=960] {derived}] =
        convolution
          x=t677 {derived}
          weight=t678 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=960}
      n365 {pt2=root[132] torch.ops.aten.convolution.default (convolution_46)}: [t680 f32 [H=960
                                                                      W=7 C=7] {pt2=root:convolution_46}] =
        permute x=t679 {derived} perm=[H<-C, W<-H, C<-W]
    group g94 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n366 {derived}: [t681 f32 [H=7 W=7 C=960] {derived}] =
        permute x=t680 {pt2=root:convolution_46} perm=[H<-W, W<-C, C<-H]
      n367 {derived}: [t682 f32 [H=7 W=7 C=960] {derived}] =
        batch_norm
          x=t681 {derived}
          weight=t139 {pt2=root:p_features_16_conv_1_1_weight target=features.16.conv.1.1.weight}
          bias=t140 {pt2=root:p_features_16_conv_1_1_bias target=features.16.conv.1.1.bias}
          running_mean=t296 {pt2=root:b_features_16_conv_1_1_running_mean target=features.16.conv.1.1.running_mean}
          running_var=t297 {pt2=root:b_features_16_conv_1_1_running_var target=features.16.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
      n368 {pt2=root[133] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_46)}: [t683 f32 [H=960
                                                                      W=7 C=7] {pt2=root:getitem_138}] =
        permute x=t682 {derived} perm=[H<-C, W<-H, C<-W]
    n369 {pt2=root[134] torch.ops.aten.hardtanh.default (hardtanh_31)}: [t684 f32 [H=960
                                                                      W=7 C=7] {pt2=root:hardtanh_31}] =
      hardtanh x=t683 {pt2=root:getitem_138} params={min_val=0; max_val=6}
    group g95 torch.ops.aten.convolution.default:
      n370 {derived}: [t685 f32 [H=7 W=7 C=960] {derived}] =
        permute x=t684 {pt2=root:hardtanh_31} perm=[H<-W, W<-C, C<-H]
      n371 {derived}: [t686 f32 [N=160 T=1 D=1 H=1 W=1 C=960] {derived}] =
        permute
          x=t141 {pt2=root:p_features_16_conv_2_weight target=features.16.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n372 {derived}: [t687 f32 [H=7 W=7 C=160] {derived}] =
        convolution
          x=t685 {derived}
          weight=t686 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n373 {pt2=root[135] torch.ops.aten.convolution.default (convolution_47)}: [t688 f32 [H=160
                                                                      W=7 C=7] {pt2=root:convolution_47}] =
        permute x=t687 {derived} perm=[H<-C, W<-H, C<-W]
    group g96 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n374 {derived}: [t689 f32 [H=7 W=7 C=160] {derived}] =
        permute x=t688 {pt2=root:convolution_47} perm=[H<-W, W<-C, C<-H]
      n375 {derived}: [t690 f32 [H=7 W=7 C=160] {derived}] =
        batch_norm
          x=t689 {derived}
          weight=t142 {pt2=root:p_features_16_conv_3_weight target=features.16.conv.3.weight}
          bias=t143 {pt2=root:p_features_16_conv_3_bias target=features.16.conv.3.bias}
          running_mean=t299 {pt2=root:b_features_16_conv_3_running_mean target=features.16.conv.3.running_mean}
          running_var=t300 {pt2=root:b_features_16_conv_3_running_var target=features.16.conv.3.running_var}
          params={channel=C; eps=1e-05}
      n376 {pt2=root[136] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_47)}: [t691 f32 [H=160
                                                                      W=7 C=7] {pt2=root:getitem_141}] =
        permute x=t690 {derived} perm=[H<-C, W<-H, C<-W]
    n377 {pt2=root[137] torch.ops.aten.add.Tensor (add_9)}: [t692 f32 [H=160
                                                                      W=7 C=7] {pt2=root:add_9}] =
      add a=t668 {pt2=root:add_8} b=t691 {pt2=root:getitem_141}
    group g97 torch.ops.aten.convolution.default:
      n378 {derived}: [t693 f32 [H=7 W=7 C=160] {derived}] =
        permute x=t692 {pt2=root:add_9} perm=[H<-W, W<-C, C<-H]
      n379 {derived}: [t694 f32 [N=960 T=1 D=1 H=1 W=1 C=160] {derived}] =
        permute
          x=t144 {pt2=root:p_features_17_conv_0_0_weight target=features.17.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n380 {derived}: [t695 f32 [H=7 W=7 C=960] {derived}] =
        convolution
          x=t693 {derived}
          weight=t694 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n381 {pt2=root[138] torch.ops.aten.convolution.default (convolution_48)}: [t696 f32 [H=960
                                                                      W=7 C=7] {pt2=root:convolution_48}] =
        permute x=t695 {derived} perm=[H<-C, W<-H, C<-W]
    group g98 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n382 {derived}: [t697 f32 [H=7 W=7 C=960] {derived}] =
        permute x=t696 {pt2=root:convolution_48} perm=[H<-W, W<-C, C<-H]
      n383 {derived}: [t698 f32 [H=7 W=7 C=960] {derived}] =
        batch_norm
          x=t697 {derived}
          weight=t145 {pt2=root:p_features_17_conv_0_1_weight target=features.17.conv.0.1.weight}
          bias=t146 {pt2=root:p_features_17_conv_0_1_bias target=features.17.conv.0.1.bias}
          running_mean=t302 {pt2=root:b_features_17_conv_0_1_running_mean target=features.17.conv.0.1.running_mean}
          running_var=t303 {pt2=root:b_features_17_conv_0_1_running_var target=features.17.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
      n384 {pt2=root[139] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_48)}: [t699 f32 [H=960
                                                                      W=7 C=7] {pt2=root:getitem_144}] =
        permute x=t698 {derived} perm=[H<-C, W<-H, C<-W]
    n385 {pt2=root[140] torch.ops.aten.hardtanh.default (hardtanh_32)}: [t700 f32 [H=960
                                                                      W=7 C=7] {pt2=root:hardtanh_32}] =
      hardtanh x=t699 {pt2=root:getitem_144} params={min_val=0; max_val=6}
    group g99 torch.ops.aten.convolution.default:
      n386 {derived}: [t701 f32 [H=7 W=7 C=960] {derived}] =
        permute x=t700 {pt2=root:hardtanh_32} perm=[H<-W, W<-C, C<-H]
      n387 {derived}: [t702 f32 [N=960 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t147 {pt2=root:p_features_17_conv_1_0_weight target=features.17.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n388 {derived}: [t703 f32 [H=7 W=7 C=960] {derived}] =
        convolution
          x=t701 {derived}
          weight=t702 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=960}
      n389 {pt2=root[141] torch.ops.aten.convolution.default (convolution_49)}: [t704 f32 [H=960
                                                                      W=7 C=7] {pt2=root:convolution_49}] =
        permute x=t703 {derived} perm=[H<-C, W<-H, C<-W]
    group g100 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n390 {derived}: [t705 f32 [H=7 W=7 C=960] {derived}] =
        permute x=t704 {pt2=root:convolution_49} perm=[H<-W, W<-C, C<-H]
      n391 {derived}: [t706 f32 [H=7 W=7 C=960] {derived}] =
        batch_norm
          x=t705 {derived}
          weight=t148 {pt2=root:p_features_17_conv_1_1_weight target=features.17.conv.1.1.weight}
          bias=t149 {pt2=root:p_features_17_conv_1_1_bias target=features.17.conv.1.1.bias}
          running_mean=t305 {pt2=root:b_features_17_conv_1_1_running_mean target=features.17.conv.1.1.running_mean}
          running_var=t306 {pt2=root:b_features_17_conv_1_1_running_var target=features.17.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
      n392 {pt2=root[142] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_49)}: [t707 f32 [H=960
                                                                      W=7 C=7] {pt2=root:getitem_147}] =
        permute x=t706 {derived} perm=[H<-C, W<-H, C<-W]
    n393 {pt2=root[143] torch.ops.aten.hardtanh.default (hardtanh_33)}: [t708 f32 [H=960
                                                                      W=7 C=7] {pt2=root:hardtanh_33}] =
      hardtanh x=t707 {pt2=root:getitem_147} params={min_val=0; max_val=6}
    group g101 torch.ops.aten.convolution.default:
      n394 {derived}: [t709 f32 [H=7 W=7 C=960] {derived}] =
        permute x=t708 {pt2=root:hardtanh_33} perm=[H<-W, W<-C, C<-H]
      n395 {derived}: [t710 f32 [N=320 T=1 D=1 H=1 W=1 C=960] {derived}] =
        permute
          x=t150 {pt2=root:p_features_17_conv_2_weight target=features.17.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n396 {derived}: [t711 f32 [H=7 W=7 C=320] {derived}] =
        convolution
          x=t709 {derived}
          weight=t710 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n397 {pt2=root[144] torch.ops.aten.convolution.default (convolution_50)}: [t712 f32 [H=320
                                                                      W=7 C=7] {pt2=root:convolution_50}] =
        permute x=t711 {derived} perm=[H<-C, W<-H, C<-W]
    group g102 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n398 {derived}: [t713 f32 [H=7 W=7 C=320] {derived}] =
        permute x=t712 {pt2=root:convolution_50} perm=[H<-W, W<-C, C<-H]
      n399 {derived}: [t714 f32 [H=7 W=7 C=320] {derived}] =
        batch_norm
          x=t713 {derived}
          weight=t151 {pt2=root:p_features_17_conv_3_weight target=features.17.conv.3.weight}
          bias=t152 {pt2=root:p_features_17_conv_3_bias target=features.17.conv.3.bias}
          running_mean=t308 {pt2=root:b_features_17_conv_3_running_mean target=features.17.conv.3.running_mean}
          running_var=t309 {pt2=root:b_features_17_conv_3_running_var target=features.17.conv.3.running_var}
          params={channel=C; eps=1e-05}
      n400 {pt2=root[145] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_50)}: [t715 f32 [H=320
                                                                      W=7 C=7] {pt2=root:getitem_150}] =
        permute x=t714 {derived} perm=[H<-C, W<-H, C<-W]
    group g103 torch.ops.aten.convolution.default:
      n401 {derived}: [t716 f32 [H=7 W=7 C=320] {derived}] =
        permute x=t715 {pt2=root:getitem_150} perm=[H<-W, W<-C, C<-H]
      n402 {derived}: [t717 f32 [N=1280 T=1 D=1 H=1 W=1 C=320] {derived}] =
        permute
          x=t153 {pt2=root:p_features_18_0_weight target=features.18.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n403 {derived}: [t718 f32 [H=7 W=7 C=1280] {derived}] =
        convolution
          x=t716 {derived}
          weight=t717 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n404 {pt2=root[146] torch.ops.aten.convolution.default (convolution_51)}: [t719 f32 [H=1280
                                                                      W=7 C=7] {pt2=root:convolution_51}] =
        permute x=t718 {derived} perm=[H<-C, W<-H, C<-W]
    group g104 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n405 {derived}: [t720 f32 [H=7 W=7 C=1280] {derived}] =
        permute x=t719 {pt2=root:convolution_51} perm=[H<-W, W<-C, C<-H]
      n406 {derived}: [t721 f32 [H=7 W=7 C=1280] {derived}] =
        batch_norm
          x=t720 {derived}
          weight=t154 {pt2=root:p_features_18_1_weight target=features.18.1.weight}
          bias=t155 {pt2=root:p_features_18_1_bias target=features.18.1.bias}
          running_mean=t311 {pt2=root:b_features_18_1_running_mean target=features.18.1.running_mean}
          running_var=t312 {pt2=root:b_features_18_1_running_var target=features.18.1.running_var}
          params={channel=C; eps=1e-05}
      n407 {pt2=root[147] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_51)}: [t722 f32 [H=1280
                                                                      W=7 C=7] {pt2=root:getitem_153}] =
        permute x=t721 {derived} perm=[H<-C, W<-H, C<-W]
    n408 {pt2=root[148] torch.ops.aten.hardtanh.default (hardtanh_34)}: [t723 f32 [H=1280
                                                                      W=7 C=7] {pt2=root:hardtanh_34}] =
      hardtanh x=t722 {pt2=root:getitem_153} params={min_val=0; max_val=6}
    n409 {pt2=root[149] torch.ops.aten.mean.dim (mean)}: [t724 f32 [H=1280 W=1
                                                                    C=1] {pt2=root:mean}] =
      mean x=t723 {pt2=root:hardtanh_34} params={dims=[C, W]; keepdim=true}
    n410 {pt2=root[150] torch.ops.aten.view.default (view)}: [t725 f32 [C=1280] {pt2=root:view}] =
      reshape x=t724 {pt2=root:mean} params={shape=[C=1280]}
    n411 {pt2=root[151] torch.ops.aten.clone.default (clone)}: [t726 f32 [C=1280] {pt2=root:clone}] =
      clone x=t725 {pt2=root:view}
    n412 {pt2=root[152] torch.ops.aten.permute.default (permute)}: [t727 f32 [W=1280
                                                                      C=1000] {pt2=root:permute}] =
      permute
        x=t156 {pt2=root:p_classifier_1_weight target=classifier.1.weight}
        perm=[W<-C, C<-W]
    group g105 torch.ops.aten.addmm.default:
      n413 {derived}: [t728 f32 [N=1000 T=1 D=1 H=1 W=1 C=1280] {derived}] =
        permute x=t727 {pt2=root:permute} perm=[N<-C, W<-N, C<-W]
      n414 {pt2=root[153] torch.ops.aten.addmm.default (addmm)}: [t729 f32 [C=1000] {pt2=root:addmm}] =
        linear
          x=t726 {pt2=root:clone}
          weight=t728 {derived}
          bias=t157 {pt2=root:p_classifier_1_bias target=classifier.1.bias}
          params={in_features=1280}
  outputs: [t729 f32 [C=1000] {pt2=root:addmm}]
