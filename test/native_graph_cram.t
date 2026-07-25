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
