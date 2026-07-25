Import ResNet-18's real exported graph as one native graph.  Unlike the former
per-node ATen bridge report, `native_graph print` uses the pure PT2 importer and
keeps exporter-facing names and captured payload targets in `Pt2_native_graph`.
`print` renders the native graph structure and every retained native-id-to-PT2
tensor/node mapping. Gated on PT2_DATA; run with `make pt2.runtest` after
`make pt2.download-cram`.

  $ ../bin/native_graph.exe print --pt2 "$PT2_DATA/resnet18/resnet18.pt2"
  native graph: inputs=123 constants=122 nodes=70 outputs=1
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
    n4 {pt2=root[0] torch.ops.aten.convolution.default (convolution)}: [t129 f32 [H=64
                                                                      W=112
                                                                      C=112] {pt2=root:convolution}] =
      subgraph
        args=[t122 {pt2=root:x}
                -> t123 {derived},
              t0 {pt2=root:p_conv1_weight target=conv1.weight}
                -> t124 {derived}]
      graph
      inputs:
        [t123 f32 [H=3 W=224 C=224] {derived},
         t124 f32 [D=64 H=3 W=7 C=7] {derived}]
      nodes:
        n0 {derived}: [t125 f32 [H=224 W=224 C=3] {derived}] =
          permute x=t123 {derived} perm=[H<-W, W<-C, C<-H]
        n1 {derived}: [t126 f32 [N=64 T=1 D=1 H=7 W=7 C=3] {derived}] =
          permute x=t124 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n2 {derived}: [t127 f32 [H=112 W=112 C=64] {derived}] =
          convolution
            x=t125 {derived}
            weight=t126 {derived}
            bias=none
            params={stride={h=2; w=2};
                   padding={h=3; w=3};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n3 {derived}: [t128 f32 [H=64 W=112 C=112] {derived}] =
          permute x=t127 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t128 f32 [H=64 W=112 C=112] {derived}]
    n8 {pt2=root[1] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training)}: [t138 f32 [H=64
                                                                      W=112
                                                                      C=112] {pt2=root:getitem}] =
      subgraph
        args=[t129 {pt2=root:convolution}
                -> t130 {derived},
              t1 {pt2=root:p_bn1_weight target=bn1.weight}
                -> t131 {derived},
              t2 {pt2=root:p_bn1_bias target=bn1.bias}
                -> t132 {derived},
              t62 {pt2=root:b_bn1_running_mean target=bn1.running_mean}
                -> t133 {derived},
              t63 {pt2=root:b_bn1_running_var target=bn1.running_var}
                -> t134 {derived}]
      graph
      inputs:
        [t130 f32 [H=64 W=112 C=112] {derived}, t131 f32 [C=64] {derived},
         t132 f32 [C=64] {derived}, t133 f32 [C=64] {derived},
         t134 f32 [C=64] {derived}]
      nodes:
        n5 {derived}: [t135 f32 [H=112 W=112 C=64] {derived}] =
          permute x=t130 {derived} perm=[H<-W, W<-C, C<-H]
        n6 {derived}: [t136 f32 [H=112 W=112 C=64] {derived}] =
          batch_norm
            x=t135 {derived}
            weight=t131 {derived}
            bias=t132 {derived}
            running_mean=t133 {derived}
            running_var=t134 {derived}
            params={channel=C; eps=1e-05}
        n7 {derived}: [t137 f32 [H=64 W=112 C=112] {derived}] =
          permute x=t136 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t137 f32 [H=64 W=112 C=112] {derived}]
    n9 {pt2=root[2] torch.ops.aten.relu.default (relu)}: [t139 f32 [H=64 W=112
                                                                    C=112] {pt2=root:relu}] =
      relu x=t138 {pt2=root:getitem}
    n14 {pt2=root[3] torch.ops.aten.max_pool2d_with_indices.default (max_pool2d_with_indices)}: [t145 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_3}] =
      subgraph args=[t139 {pt2=root:relu}
                       -> t140 {derived}]
      graph
      inputs: [t140 f32 [H=64 W=112 C=112] {derived}]
      nodes:
        n10 {derived}: [t141 f32 [H=112 W=112 C=64] {derived}] =
          permute x=t140 {derived} perm=[H<-W, W<-C, C<-H]
        n11 {derived}: [t142 f32 [H=56 W=56 C=64] {derived},
                        t143 f32 [H=56 W=56 C=64] {derived}] =
          max_pool2d_with_indices
            x=t141 {derived}
            params={kernel={h=3; w=3}; stride={h=2; w=2}; pad={h=1; w=1}}
        n12 {derived}: [] = discard x=t143 {derived}
        n13 {derived}: [t144 f32 [H=64 W=56 C=56] {derived}] =
          permute x=t142 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t144 f32 [H=64 W=56 C=56] {derived}]
    n19 {pt2=root[4] torch.ops.aten.convolution.default (convolution_1)}: [t152 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:convolution_1}] =
      subgraph
        args=[t145 {pt2=root:getitem_3}
                -> t146 {derived},
              t3 {pt2=root:p_layer1_0_conv1_weight target=layer1.0.conv1.weight}
                -> t147 {derived}]
      graph
      inputs:
        [t146 f32 [H=64 W=56 C=56] {derived},
         t147 f32 [D=64 H=64 W=3 C=3] {derived}]
      nodes:
        n15 {derived}: [t148 f32 [H=56 W=56 C=64] {derived}] =
          permute x=t146 {derived} perm=[H<-W, W<-C, C<-H]
        n16 {derived}: [t149 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {derived}] =
          permute x=t147 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n17 {derived}: [t150 f32 [H=56 W=56 C=64] {derived}] =
          convolution
            x=t148 {derived}
            weight=t149 {derived}
            bias=none
            params={stride={h=1; w=1};
                   padding={h=1; w=1};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n18 {derived}: [t151 f32 [H=64 W=56 C=56] {derived}] =
          permute x=t150 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t151 f32 [H=64 W=56 C=56] {derived}]
    n23 {pt2=root[5] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_1)}: [t161 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_5}] =
      subgraph
        args=[t152 {pt2=root:convolution_1}
                -> t153 {derived},
              t4 {pt2=root:p_layer1_0_bn1_weight target=layer1.0.bn1.weight}
                -> t154 {derived},
              t5 {pt2=root:p_layer1_0_bn1_bias target=layer1.0.bn1.bias}
                -> t155 {derived},
              t65 {pt2=root:b_layer1_0_bn1_running_mean target=layer1.0.bn1.running_mean}
                -> t156 {derived},
              t66 {pt2=root:b_layer1_0_bn1_running_var target=layer1.0.bn1.running_var}
                -> t157 {derived}]
      graph
      inputs:
        [t153 f32 [H=64 W=56 C=56] {derived}, t154 f32 [C=64] {derived},
         t155 f32 [C=64] {derived}, t156 f32 [C=64] {derived},
         t157 f32 [C=64] {derived}]
      nodes:
        n20 {derived}: [t158 f32 [H=56 W=56 C=64] {derived}] =
          permute x=t153 {derived} perm=[H<-W, W<-C, C<-H]
        n21 {derived}: [t159 f32 [H=56 W=56 C=64] {derived}] =
          batch_norm
            x=t158 {derived}
            weight=t154 {derived}
            bias=t155 {derived}
            running_mean=t156 {derived}
            running_var=t157 {derived}
            params={channel=C; eps=1e-05}
        n22 {derived}: [t160 f32 [H=64 W=56 C=56] {derived}] =
          permute x=t159 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t160 f32 [H=64 W=56 C=56] {derived}]
    n24 {pt2=root[6] torch.ops.aten.relu.default (relu_1)}: [t162 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:relu_1}] =
      relu x=t161 {pt2=root:getitem_5}
    n29 {pt2=root[7] torch.ops.aten.convolution.default (convolution_2)}: [t169 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:convolution_2}] =
      subgraph
        args=[t162 {pt2=root:relu_1}
                -> t163 {derived},
              t6 {pt2=root:p_layer1_0_conv2_weight target=layer1.0.conv2.weight}
                -> t164 {derived}]
      graph
      inputs:
        [t163 f32 [H=64 W=56 C=56] {derived},
         t164 f32 [D=64 H=64 W=3 C=3] {derived}]
      nodes:
        n25 {derived}: [t165 f32 [H=56 W=56 C=64] {derived}] =
          permute x=t163 {derived} perm=[H<-W, W<-C, C<-H]
        n26 {derived}: [t166 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {derived}] =
          permute x=t164 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n27 {derived}: [t167 f32 [H=56 W=56 C=64] {derived}] =
          convolution
            x=t165 {derived}
            weight=t166 {derived}
            bias=none
            params={stride={h=1; w=1};
                   padding={h=1; w=1};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n28 {derived}: [t168 f32 [H=64 W=56 C=56] {derived}] =
          permute x=t167 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t168 f32 [H=64 W=56 C=56] {derived}]
    n33 {pt2=root[8] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_2)}: [t178 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_8}] =
      subgraph
        args=[t169 {pt2=root:convolution_2}
                -> t170 {derived},
              t7 {pt2=root:p_layer1_0_bn2_weight target=layer1.0.bn2.weight}
                -> t171 {derived},
              t8 {pt2=root:p_layer1_0_bn2_bias target=layer1.0.bn2.bias}
                -> t172 {derived},
              t68 {pt2=root:b_layer1_0_bn2_running_mean target=layer1.0.bn2.running_mean}
                -> t173 {derived},
              t69 {pt2=root:b_layer1_0_bn2_running_var target=layer1.0.bn2.running_var}
                -> t174 {derived}]
      graph
      inputs:
        [t170 f32 [H=64 W=56 C=56] {derived}, t171 f32 [C=64] {derived},
         t172 f32 [C=64] {derived}, t173 f32 [C=64] {derived},
         t174 f32 [C=64] {derived}]
      nodes:
        n30 {derived}: [t175 f32 [H=56 W=56 C=64] {derived}] =
          permute x=t170 {derived} perm=[H<-W, W<-C, C<-H]
        n31 {derived}: [t176 f32 [H=56 W=56 C=64] {derived}] =
          batch_norm
            x=t175 {derived}
            weight=t171 {derived}
            bias=t172 {derived}
            running_mean=t173 {derived}
            running_var=t174 {derived}
            params={channel=C; eps=1e-05}
        n32 {derived}: [t177 f32 [H=64 W=56 C=56] {derived}] =
          permute x=t176 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t177 f32 [H=64 W=56 C=56] {derived}]
    n34 {pt2=root[9] torch.ops.aten.add.Tensor (add)}: [t179 f32 [H=64 W=56
                                                                  C=56] {pt2=root:add}] =
      add a=t178 {pt2=root:getitem_8} b=t145 {pt2=root:getitem_3}
    n35 {pt2=root[10] torch.ops.aten.relu.default (relu_2)}: [t180 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:relu_2}] =
      relu x=t179 {pt2=root:add}
    n40 {pt2=root[11] torch.ops.aten.convolution.default (convolution_3)}: [t187 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:convolution_3}] =
      subgraph
        args=[t180 {pt2=root:relu_2}
                -> t181 {derived},
              t9 {pt2=root:p_layer1_1_conv1_weight target=layer1.1.conv1.weight}
                -> t182 {derived}]
      graph
      inputs:
        [t181 f32 [H=64 W=56 C=56] {derived},
         t182 f32 [D=64 H=64 W=3 C=3] {derived}]
      nodes:
        n36 {derived}: [t183 f32 [H=56 W=56 C=64] {derived}] =
          permute x=t181 {derived} perm=[H<-W, W<-C, C<-H]
        n37 {derived}: [t184 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {derived}] =
          permute x=t182 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n38 {derived}: [t185 f32 [H=56 W=56 C=64] {derived}] =
          convolution
            x=t183 {derived}
            weight=t184 {derived}
            bias=none
            params={stride={h=1; w=1};
                   padding={h=1; w=1};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n39 {derived}: [t186 f32 [H=64 W=56 C=56] {derived}] =
          permute x=t185 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t186 f32 [H=64 W=56 C=56] {derived}]
    n44 {pt2=root[12] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_3)}: [t196 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_11}] =
      subgraph
        args=[t187 {pt2=root:convolution_3}
                -> t188 {derived},
              t10 {pt2=root:p_layer1_1_bn1_weight target=layer1.1.bn1.weight}
                -> t189 {derived},
              t11 {pt2=root:p_layer1_1_bn1_bias target=layer1.1.bn1.bias}
                -> t190 {derived},
              t71 {pt2=root:b_layer1_1_bn1_running_mean target=layer1.1.bn1.running_mean}
                -> t191 {derived},
              t72 {pt2=root:b_layer1_1_bn1_running_var target=layer1.1.bn1.running_var}
                -> t192 {derived}]
      graph
      inputs:
        [t188 f32 [H=64 W=56 C=56] {derived}, t189 f32 [C=64] {derived},
         t190 f32 [C=64] {derived}, t191 f32 [C=64] {derived},
         t192 f32 [C=64] {derived}]
      nodes:
        n41 {derived}: [t193 f32 [H=56 W=56 C=64] {derived}] =
          permute x=t188 {derived} perm=[H<-W, W<-C, C<-H]
        n42 {derived}: [t194 f32 [H=56 W=56 C=64] {derived}] =
          batch_norm
            x=t193 {derived}
            weight=t189 {derived}
            bias=t190 {derived}
            running_mean=t191 {derived}
            running_var=t192 {derived}
            params={channel=C; eps=1e-05}
        n43 {derived}: [t195 f32 [H=64 W=56 C=56] {derived}] =
          permute x=t194 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t195 f32 [H=64 W=56 C=56] {derived}]
    n45 {pt2=root[13] torch.ops.aten.relu.default (relu_3)}: [t197 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:relu_3}] =
      relu x=t196 {pt2=root:getitem_11}
    n50 {pt2=root[14] torch.ops.aten.convolution.default (convolution_4)}: [t204 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:convolution_4}] =
      subgraph
        args=[t197 {pt2=root:relu_3}
                -> t198 {derived},
              t12 {pt2=root:p_layer1_1_conv2_weight target=layer1.1.conv2.weight}
                -> t199 {derived}]
      graph
      inputs:
        [t198 f32 [H=64 W=56 C=56] {derived},
         t199 f32 [D=64 H=64 W=3 C=3] {derived}]
      nodes:
        n46 {derived}: [t200 f32 [H=56 W=56 C=64] {derived}] =
          permute x=t198 {derived} perm=[H<-W, W<-C, C<-H]
        n47 {derived}: [t201 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {derived}] =
          permute x=t199 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n48 {derived}: [t202 f32 [H=56 W=56 C=64] {derived}] =
          convolution
            x=t200 {derived}
            weight=t201 {derived}
            bias=none
            params={stride={h=1; w=1};
                   padding={h=1; w=1};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n49 {derived}: [t203 f32 [H=64 W=56 C=56] {derived}] =
          permute x=t202 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t203 f32 [H=64 W=56 C=56] {derived}]
    n54 {pt2=root[15] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_4)}: [t213 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_14}] =
      subgraph
        args=[t204 {pt2=root:convolution_4}
                -> t205 {derived},
              t13 {pt2=root:p_layer1_1_bn2_weight target=layer1.1.bn2.weight}
                -> t206 {derived},
              t14 {pt2=root:p_layer1_1_bn2_bias target=layer1.1.bn2.bias}
                -> t207 {derived},
              t74 {pt2=root:b_layer1_1_bn2_running_mean target=layer1.1.bn2.running_mean}
                -> t208 {derived},
              t75 {pt2=root:b_layer1_1_bn2_running_var target=layer1.1.bn2.running_var}
                -> t209 {derived}]
      graph
      inputs:
        [t205 f32 [H=64 W=56 C=56] {derived}, t206 f32 [C=64] {derived},
         t207 f32 [C=64] {derived}, t208 f32 [C=64] {derived},
         t209 f32 [C=64] {derived}]
      nodes:
        n51 {derived}: [t210 f32 [H=56 W=56 C=64] {derived}] =
          permute x=t205 {derived} perm=[H<-W, W<-C, C<-H]
        n52 {derived}: [t211 f32 [H=56 W=56 C=64] {derived}] =
          batch_norm
            x=t210 {derived}
            weight=t206 {derived}
            bias=t207 {derived}
            running_mean=t208 {derived}
            running_var=t209 {derived}
            params={channel=C; eps=1e-05}
        n53 {derived}: [t212 f32 [H=64 W=56 C=56] {derived}] =
          permute x=t211 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t212 f32 [H=64 W=56 C=56] {derived}]
    n55 {pt2=root[16] torch.ops.aten.add.Tensor (add_1)}: [t214 f32 [H=64 W=56
                                                                     C=56] {pt2=root:add_1}] =
      add a=t213 {pt2=root:getitem_14} b=t180 {pt2=root:relu_2}
    n56 {pt2=root[17] torch.ops.aten.relu.default (relu_4)}: [t215 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:relu_4}] =
      relu x=t214 {pt2=root:add_1}
    n61 {pt2=root[18] torch.ops.aten.convolution.default (convolution_5)}: [t222 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_5}] =
      subgraph
        args=[t215 {pt2=root:relu_4}
                -> t216 {derived},
              t15 {pt2=root:p_layer2_0_conv1_weight target=layer2.0.conv1.weight}
                -> t217 {derived}]
      graph
      inputs:
        [t216 f32 [H=64 W=56 C=56] {derived},
         t217 f32 [D=128 H=64 W=3 C=3] {derived}]
      nodes:
        n57 {derived}: [t218 f32 [H=56 W=56 C=64] {derived}] =
          permute x=t216 {derived} perm=[H<-W, W<-C, C<-H]
        n58 {derived}: [t219 f32 [N=128 T=1 D=1 H=3 W=3 C=64] {derived}] =
          permute x=t217 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n59 {derived}: [t220 f32 [H=28 W=28 C=128] {derived}] =
          convolution
            x=t218 {derived}
            weight=t219 {derived}
            bias=none
            params={stride={h=2; w=2};
                   padding={h=1; w=1};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n60 {derived}: [t221 f32 [H=128 W=28 C=28] {derived}] =
          permute x=t220 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t221 f32 [H=128 W=28 C=28] {derived}]
    n65 {pt2=root[19] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_5)}: [t231 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_17}] =
      subgraph
        args=[t222 {pt2=root:convolution_5}
                -> t223 {derived},
              t16 {pt2=root:p_layer2_0_bn1_weight target=layer2.0.bn1.weight}
                -> t224 {derived},
              t17 {pt2=root:p_layer2_0_bn1_bias target=layer2.0.bn1.bias}
                -> t225 {derived},
              t77 {pt2=root:b_layer2_0_bn1_running_mean target=layer2.0.bn1.running_mean}
                -> t226 {derived},
              t78 {pt2=root:b_layer2_0_bn1_running_var target=layer2.0.bn1.running_var}
                -> t227 {derived}]
      graph
      inputs:
        [t223 f32 [H=128 W=28 C=28] {derived}, t224 f32 [C=128] {derived},
         t225 f32 [C=128] {derived}, t226 f32 [C=128] {derived},
         t227 f32 [C=128] {derived}]
      nodes:
        n62 {derived}: [t228 f32 [H=28 W=28 C=128] {derived}] =
          permute x=t223 {derived} perm=[H<-W, W<-C, C<-H]
        n63 {derived}: [t229 f32 [H=28 W=28 C=128] {derived}] =
          batch_norm
            x=t228 {derived}
            weight=t224 {derived}
            bias=t225 {derived}
            running_mean=t226 {derived}
            running_var=t227 {derived}
            params={channel=C; eps=1e-05}
        n64 {derived}: [t230 f32 [H=128 W=28 C=28] {derived}] =
          permute x=t229 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t230 f32 [H=128 W=28 C=28] {derived}]
    n66 {pt2=root[20] torch.ops.aten.relu.default (relu_5)}: [t232 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:relu_5}] =
      relu x=t231 {pt2=root:getitem_17}
    n71 {pt2=root[21] torch.ops.aten.convolution.default (convolution_6)}: [t239 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_6}] =
      subgraph
        args=[t232 {pt2=root:relu_5}
                -> t233 {derived},
              t18 {pt2=root:p_layer2_0_conv2_weight target=layer2.0.conv2.weight}
                -> t234 {derived}]
      graph
      inputs:
        [t233 f32 [H=128 W=28 C=28] {derived},
         t234 f32 [D=128 H=128 W=3 C=3] {derived}]
      nodes:
        n67 {derived}: [t235 f32 [H=28 W=28 C=128] {derived}] =
          permute x=t233 {derived} perm=[H<-W, W<-C, C<-H]
        n68 {derived}: [t236 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {derived}] =
          permute x=t234 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n69 {derived}: [t237 f32 [H=28 W=28 C=128] {derived}] =
          convolution
            x=t235 {derived}
            weight=t236 {derived}
            bias=none
            params={stride={h=1; w=1};
                   padding={h=1; w=1};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n70 {derived}: [t238 f32 [H=128 W=28 C=28] {derived}] =
          permute x=t237 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t238 f32 [H=128 W=28 C=28] {derived}]
    n75 {pt2=root[22] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_6)}: [t248 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_20}] =
      subgraph
        args=[t239 {pt2=root:convolution_6}
                -> t240 {derived},
              t19 {pt2=root:p_layer2_0_bn2_weight target=layer2.0.bn2.weight}
                -> t241 {derived},
              t20 {pt2=root:p_layer2_0_bn2_bias target=layer2.0.bn2.bias}
                -> t242 {derived},
              t80 {pt2=root:b_layer2_0_bn2_running_mean target=layer2.0.bn2.running_mean}
                -> t243 {derived},
              t81 {pt2=root:b_layer2_0_bn2_running_var target=layer2.0.bn2.running_var}
                -> t244 {derived}]
      graph
      inputs:
        [t240 f32 [H=128 W=28 C=28] {derived}, t241 f32 [C=128] {derived},
         t242 f32 [C=128] {derived}, t243 f32 [C=128] {derived},
         t244 f32 [C=128] {derived}]
      nodes:
        n72 {derived}: [t245 f32 [H=28 W=28 C=128] {derived}] =
          permute x=t240 {derived} perm=[H<-W, W<-C, C<-H]
        n73 {derived}: [t246 f32 [H=28 W=28 C=128] {derived}] =
          batch_norm
            x=t245 {derived}
            weight=t241 {derived}
            bias=t242 {derived}
            running_mean=t243 {derived}
            running_var=t244 {derived}
            params={channel=C; eps=1e-05}
        n74 {derived}: [t247 f32 [H=128 W=28 C=28] {derived}] =
          permute x=t246 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t247 f32 [H=128 W=28 C=28] {derived}]
    n80 {pt2=root[23] torch.ops.aten.convolution.default (convolution_7)}: [t255 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_7}] =
      subgraph
        args=[t215 {pt2=root:relu_4}
                -> t249 {derived},
              t21 {pt2=root:p_layer2_0_downsample_0_weight target=layer2.0.downsample.0.weight}
                -> t250 {derived}]
      graph
      inputs:
        [t249 f32 [H=64 W=56 C=56] {derived},
         t250 f32 [D=128 H=64 W=1 C=1] {derived}]
      nodes:
        n76 {derived}: [t251 f32 [H=56 W=56 C=64] {derived}] =
          permute x=t249 {derived} perm=[H<-W, W<-C, C<-H]
        n77 {derived}: [t252 f32 [N=128 T=1 D=1 H=1 W=1 C=64] {derived}] =
          permute x=t250 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n78 {derived}: [t253 f32 [H=28 W=28 C=128] {derived}] =
          convolution
            x=t251 {derived}
            weight=t252 {derived}
            bias=none
            params={stride={h=2; w=2};
                   padding={h=0; w=0};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n79 {derived}: [t254 f32 [H=128 W=28 C=28] {derived}] =
          permute x=t253 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t254 f32 [H=128 W=28 C=28] {derived}]
    n84 {pt2=root[24] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_7)}: [t264 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_23}] =
      subgraph
        args=[t255 {pt2=root:convolution_7}
                -> t256 {derived},
              t22 {pt2=root:p_layer2_0_downsample_1_weight target=layer2.0.downsample.1.weight}
                -> t257 {derived},
              t23 {pt2=root:p_layer2_0_downsample_1_bias target=layer2.0.downsample.1.bias}
                -> t258 {derived},
              t83 {pt2=root:b_layer2_0_downsample_1_running_mean target=layer2.0.downsample.1.running_mean}
                -> t259 {derived},
              t84 {pt2=root:b_layer2_0_downsample_1_running_var target=layer2.0.downsample.1.running_var}
                -> t260 {derived}]
      graph
      inputs:
        [t256 f32 [H=128 W=28 C=28] {derived}, t257 f32 [C=128] {derived},
         t258 f32 [C=128] {derived}, t259 f32 [C=128] {derived},
         t260 f32 [C=128] {derived}]
      nodes:
        n81 {derived}: [t261 f32 [H=28 W=28 C=128] {derived}] =
          permute x=t256 {derived} perm=[H<-W, W<-C, C<-H]
        n82 {derived}: [t262 f32 [H=28 W=28 C=128] {derived}] =
          batch_norm
            x=t261 {derived}
            weight=t257 {derived}
            bias=t258 {derived}
            running_mean=t259 {derived}
            running_var=t260 {derived}
            params={channel=C; eps=1e-05}
        n83 {derived}: [t263 f32 [H=128 W=28 C=28] {derived}] =
          permute x=t262 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t263 f32 [H=128 W=28 C=28] {derived}]
    n85 {pt2=root[25] torch.ops.aten.add.Tensor (add_2)}: [t265 f32 [H=128 W=28
                                                                     C=28] {pt2=root:add_2}] =
      add a=t248 {pt2=root:getitem_20} b=t264 {pt2=root:getitem_23}
    n86 {pt2=root[26] torch.ops.aten.relu.default (relu_6)}: [t266 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:relu_6}] =
      relu x=t265 {pt2=root:add_2}
    n91 {pt2=root[27] torch.ops.aten.convolution.default (convolution_8)}: [t273 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_8}] =
      subgraph
        args=[t266 {pt2=root:relu_6}
                -> t267 {derived},
              t24 {pt2=root:p_layer2_1_conv1_weight target=layer2.1.conv1.weight}
                -> t268 {derived}]
      graph
      inputs:
        [t267 f32 [H=128 W=28 C=28] {derived},
         t268 f32 [D=128 H=128 W=3 C=3] {derived}]
      nodes:
        n87 {derived}: [t269 f32 [H=28 W=28 C=128] {derived}] =
          permute x=t267 {derived} perm=[H<-W, W<-C, C<-H]
        n88 {derived}: [t270 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {derived}] =
          permute x=t268 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n89 {derived}: [t271 f32 [H=28 W=28 C=128] {derived}] =
          convolution
            x=t269 {derived}
            weight=t270 {derived}
            bias=none
            params={stride={h=1; w=1};
                   padding={h=1; w=1};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n90 {derived}: [t272 f32 [H=128 W=28 C=28] {derived}] =
          permute x=t271 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t272 f32 [H=128 W=28 C=28] {derived}]
    n95 {pt2=root[28] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_8)}: [t282 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_26}] =
      subgraph
        args=[t273 {pt2=root:convolution_8}
                -> t274 {derived},
              t25 {pt2=root:p_layer2_1_bn1_weight target=layer2.1.bn1.weight}
                -> t275 {derived},
              t26 {pt2=root:p_layer2_1_bn1_bias target=layer2.1.bn1.bias}
                -> t276 {derived},
              t86 {pt2=root:b_layer2_1_bn1_running_mean target=layer2.1.bn1.running_mean}
                -> t277 {derived},
              t87 {pt2=root:b_layer2_1_bn1_running_var target=layer2.1.bn1.running_var}
                -> t278 {derived}]
      graph
      inputs:
        [t274 f32 [H=128 W=28 C=28] {derived}, t275 f32 [C=128] {derived},
         t276 f32 [C=128] {derived}, t277 f32 [C=128] {derived},
         t278 f32 [C=128] {derived}]
      nodes:
        n92 {derived}: [t279 f32 [H=28 W=28 C=128] {derived}] =
          permute x=t274 {derived} perm=[H<-W, W<-C, C<-H]
        n93 {derived}: [t280 f32 [H=28 W=28 C=128] {derived}] =
          batch_norm
            x=t279 {derived}
            weight=t275 {derived}
            bias=t276 {derived}
            running_mean=t277 {derived}
            running_var=t278 {derived}
            params={channel=C; eps=1e-05}
        n94 {derived}: [t281 f32 [H=128 W=28 C=28] {derived}] =
          permute x=t280 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t281 f32 [H=128 W=28 C=28] {derived}]
    n96 {pt2=root[29] torch.ops.aten.relu.default (relu_7)}: [t283 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:relu_7}] =
      relu x=t282 {pt2=root:getitem_26}
    n101 {pt2=root[30] torch.ops.aten.convolution.default (convolution_9)}: [t290 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:convolution_9}] =
      subgraph
        args=[t283 {pt2=root:relu_7}
                -> t284 {derived},
              t27 {pt2=root:p_layer2_1_conv2_weight target=layer2.1.conv2.weight}
                -> t285 {derived}]
      graph
      inputs:
        [t284 f32 [H=128 W=28 C=28] {derived},
         t285 f32 [D=128 H=128 W=3 C=3] {derived}]
      nodes:
        n97 {derived}: [t286 f32 [H=28 W=28 C=128] {derived}] =
          permute x=t284 {derived} perm=[H<-W, W<-C, C<-H]
        n98 {derived}: [t287 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {derived}] =
          permute x=t285 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n99 {derived}: [t288 f32 [H=28 W=28 C=128] {derived}] =
          convolution
            x=t286 {derived}
            weight=t287 {derived}
            bias=none
            params={stride={h=1; w=1};
                   padding={h=1; w=1};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n100 {derived}: [t289 f32 [H=128 W=28 C=28] {derived}] =
          permute x=t288 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t289 f32 [H=128 W=28 C=28] {derived}]
    n105 {pt2=root[31] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_9)}: [t299 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_29}] =
      subgraph
        args=[t290 {pt2=root:convolution_9}
                -> t291 {derived},
              t28 {pt2=root:p_layer2_1_bn2_weight target=layer2.1.bn2.weight}
                -> t292 {derived},
              t29 {pt2=root:p_layer2_1_bn2_bias target=layer2.1.bn2.bias}
                -> t293 {derived},
              t89 {pt2=root:b_layer2_1_bn2_running_mean target=layer2.1.bn2.running_mean}
                -> t294 {derived},
              t90 {pt2=root:b_layer2_1_bn2_running_var target=layer2.1.bn2.running_var}
                -> t295 {derived}]
      graph
      inputs:
        [t291 f32 [H=128 W=28 C=28] {derived}, t292 f32 [C=128] {derived},
         t293 f32 [C=128] {derived}, t294 f32 [C=128] {derived},
         t295 f32 [C=128] {derived}]
      nodes:
        n102 {derived}: [t296 f32 [H=28 W=28 C=128] {derived}] =
          permute x=t291 {derived} perm=[H<-W, W<-C, C<-H]
        n103 {derived}: [t297 f32 [H=28 W=28 C=128] {derived}] =
          batch_norm
            x=t296 {derived}
            weight=t292 {derived}
            bias=t293 {derived}
            running_mean=t294 {derived}
            running_var=t295 {derived}
            params={channel=C; eps=1e-05}
        n104 {derived}: [t298 f32 [H=128 W=28 C=28] {derived}] =
          permute x=t297 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t298 f32 [H=128 W=28 C=28] {derived}]
    n106 {pt2=root[32] torch.ops.aten.add.Tensor (add_3)}: [t300 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:add_3}] =
      add a=t299 {pt2=root:getitem_29} b=t266 {pt2=root:relu_6}
    n107 {pt2=root[33] torch.ops.aten.relu.default (relu_8)}: [t301 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:relu_8}] =
      relu x=t300 {pt2=root:add_3}
    n112 {pt2=root[34] torch.ops.aten.convolution.default (convolution_10)}: [t308 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_10}] =
      subgraph
        args=[t301 {pt2=root:relu_8}
                -> t302 {derived},
              t30 {pt2=root:p_layer3_0_conv1_weight target=layer3.0.conv1.weight}
                -> t303 {derived}]
      graph
      inputs:
        [t302 f32 [H=128 W=28 C=28] {derived},
         t303 f32 [D=256 H=128 W=3 C=3] {derived}]
      nodes:
        n108 {derived}: [t304 f32 [H=28 W=28 C=128] {derived}] =
          permute x=t302 {derived} perm=[H<-W, W<-C, C<-H]
        n109 {derived}: [t305 f32 [N=256 T=1 D=1 H=3 W=3 C=128] {derived}] =
          permute x=t303 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n110 {derived}: [t306 f32 [H=14 W=14 C=256] {derived}] =
          convolution
            x=t304 {derived}
            weight=t305 {derived}
            bias=none
            params={stride={h=2; w=2};
                   padding={h=1; w=1};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n111 {derived}: [t307 f32 [H=256 W=14 C=14] {derived}] =
          permute x=t306 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t307 f32 [H=256 W=14 C=14] {derived}]
    n116 {pt2=root[35] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_10)}: [t317 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_32}] =
      subgraph
        args=[t308 {pt2=root:convolution_10}
                -> t309 {derived},
              t31 {pt2=root:p_layer3_0_bn1_weight target=layer3.0.bn1.weight}
                -> t310 {derived},
              t32 {pt2=root:p_layer3_0_bn1_bias target=layer3.0.bn1.bias}
                -> t311 {derived},
              t92 {pt2=root:b_layer3_0_bn1_running_mean target=layer3.0.bn1.running_mean}
                -> t312 {derived},
              t93 {pt2=root:b_layer3_0_bn1_running_var target=layer3.0.bn1.running_var}
                -> t313 {derived}]
      graph
      inputs:
        [t309 f32 [H=256 W=14 C=14] {derived}, t310 f32 [C=256] {derived},
         t311 f32 [C=256] {derived}, t312 f32 [C=256] {derived},
         t313 f32 [C=256] {derived}]
      nodes:
        n113 {derived}: [t314 f32 [H=14 W=14 C=256] {derived}] =
          permute x=t309 {derived} perm=[H<-W, W<-C, C<-H]
        n114 {derived}: [t315 f32 [H=14 W=14 C=256] {derived}] =
          batch_norm
            x=t314 {derived}
            weight=t310 {derived}
            bias=t311 {derived}
            running_mean=t312 {derived}
            running_var=t313 {derived}
            params={channel=C; eps=1e-05}
        n115 {derived}: [t316 f32 [H=256 W=14 C=14] {derived}] =
          permute x=t315 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t316 f32 [H=256 W=14 C=14] {derived}]
    n117 {pt2=root[36] torch.ops.aten.relu.default (relu_9)}: [t318 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:relu_9}] =
      relu x=t317 {pt2=root:getitem_32}
    n122 {pt2=root[37] torch.ops.aten.convolution.default (convolution_11)}: [t325 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_11}] =
      subgraph
        args=[t318 {pt2=root:relu_9}
                -> t319 {derived},
              t33 {pt2=root:p_layer3_0_conv2_weight target=layer3.0.conv2.weight}
                -> t320 {derived}]
      graph
      inputs:
        [t319 f32 [H=256 W=14 C=14] {derived},
         t320 f32 [D=256 H=256 W=3 C=3] {derived}]
      nodes:
        n118 {derived}: [t321 f32 [H=14 W=14 C=256] {derived}] =
          permute x=t319 {derived} perm=[H<-W, W<-C, C<-H]
        n119 {derived}: [t322 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {derived}] =
          permute x=t320 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n120 {derived}: [t323 f32 [H=14 W=14 C=256] {derived}] =
          convolution
            x=t321 {derived}
            weight=t322 {derived}
            bias=none
            params={stride={h=1; w=1};
                   padding={h=1; w=1};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n121 {derived}: [t324 f32 [H=256 W=14 C=14] {derived}] =
          permute x=t323 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t324 f32 [H=256 W=14 C=14] {derived}]
    n126 {pt2=root[38] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_11)}: [t334 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_35}] =
      subgraph
        args=[t325 {pt2=root:convolution_11}
                -> t326 {derived},
              t34 {pt2=root:p_layer3_0_bn2_weight target=layer3.0.bn2.weight}
                -> t327 {derived},
              t35 {pt2=root:p_layer3_0_bn2_bias target=layer3.0.bn2.bias}
                -> t328 {derived},
              t95 {pt2=root:b_layer3_0_bn2_running_mean target=layer3.0.bn2.running_mean}
                -> t329 {derived},
              t96 {pt2=root:b_layer3_0_bn2_running_var target=layer3.0.bn2.running_var}
                -> t330 {derived}]
      graph
      inputs:
        [t326 f32 [H=256 W=14 C=14] {derived}, t327 f32 [C=256] {derived},
         t328 f32 [C=256] {derived}, t329 f32 [C=256] {derived},
         t330 f32 [C=256] {derived}]
      nodes:
        n123 {derived}: [t331 f32 [H=14 W=14 C=256] {derived}] =
          permute x=t326 {derived} perm=[H<-W, W<-C, C<-H]
        n124 {derived}: [t332 f32 [H=14 W=14 C=256] {derived}] =
          batch_norm
            x=t331 {derived}
            weight=t327 {derived}
            bias=t328 {derived}
            running_mean=t329 {derived}
            running_var=t330 {derived}
            params={channel=C; eps=1e-05}
        n125 {derived}: [t333 f32 [H=256 W=14 C=14] {derived}] =
          permute x=t332 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t333 f32 [H=256 W=14 C=14] {derived}]
    n131 {pt2=root[39] torch.ops.aten.convolution.default (convolution_12)}: [t341 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_12}] =
      subgraph
        args=[t301 {pt2=root:relu_8}
                -> t335 {derived},
              t36 {pt2=root:p_layer3_0_downsample_0_weight target=layer3.0.downsample.0.weight}
                -> t336 {derived}]
      graph
      inputs:
        [t335 f32 [H=128 W=28 C=28] {derived},
         t336 f32 [D=256 H=128 W=1 C=1] {derived}]
      nodes:
        n127 {derived}: [t337 f32 [H=28 W=28 C=128] {derived}] =
          permute x=t335 {derived} perm=[H<-W, W<-C, C<-H]
        n128 {derived}: [t338 f32 [N=256 T=1 D=1 H=1 W=1 C=128] {derived}] =
          permute x=t336 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n129 {derived}: [t339 f32 [H=14 W=14 C=256] {derived}] =
          convolution
            x=t337 {derived}
            weight=t338 {derived}
            bias=none
            params={stride={h=2; w=2};
                   padding={h=0; w=0};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n130 {derived}: [t340 f32 [H=256 W=14 C=14] {derived}] =
          permute x=t339 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t340 f32 [H=256 W=14 C=14] {derived}]
    n135 {pt2=root[40] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_12)}: [t350 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_38}] =
      subgraph
        args=[t341 {pt2=root:convolution_12}
                -> t342 {derived},
              t37 {pt2=root:p_layer3_0_downsample_1_weight target=layer3.0.downsample.1.weight}
                -> t343 {derived},
              t38 {pt2=root:p_layer3_0_downsample_1_bias target=layer3.0.downsample.1.bias}
                -> t344 {derived},
              t98 {pt2=root:b_layer3_0_downsample_1_running_mean target=layer3.0.downsample.1.running_mean}
                -> t345 {derived},
              t99 {pt2=root:b_layer3_0_downsample_1_running_var target=layer3.0.downsample.1.running_var}
                -> t346 {derived}]
      graph
      inputs:
        [t342 f32 [H=256 W=14 C=14] {derived}, t343 f32 [C=256] {derived},
         t344 f32 [C=256] {derived}, t345 f32 [C=256] {derived},
         t346 f32 [C=256] {derived}]
      nodes:
        n132 {derived}: [t347 f32 [H=14 W=14 C=256] {derived}] =
          permute x=t342 {derived} perm=[H<-W, W<-C, C<-H]
        n133 {derived}: [t348 f32 [H=14 W=14 C=256] {derived}] =
          batch_norm
            x=t347 {derived}
            weight=t343 {derived}
            bias=t344 {derived}
            running_mean=t345 {derived}
            running_var=t346 {derived}
            params={channel=C; eps=1e-05}
        n134 {derived}: [t349 f32 [H=256 W=14 C=14] {derived}] =
          permute x=t348 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t349 f32 [H=256 W=14 C=14] {derived}]
    n136 {pt2=root[41] torch.ops.aten.add.Tensor (add_4)}: [t351 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:add_4}] =
      add a=t334 {pt2=root:getitem_35} b=t350 {pt2=root:getitem_38}
    n137 {pt2=root[42] torch.ops.aten.relu.default (relu_10)}: [t352 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:relu_10}] =
      relu x=t351 {pt2=root:add_4}
    n142 {pt2=root[43] torch.ops.aten.convolution.default (convolution_13)}: [t359 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_13}] =
      subgraph
        args=[t352 {pt2=root:relu_10}
                -> t353 {derived},
              t39 {pt2=root:p_layer3_1_conv1_weight target=layer3.1.conv1.weight}
                -> t354 {derived}]
      graph
      inputs:
        [t353 f32 [H=256 W=14 C=14] {derived},
         t354 f32 [D=256 H=256 W=3 C=3] {derived}]
      nodes:
        n138 {derived}: [t355 f32 [H=14 W=14 C=256] {derived}] =
          permute x=t353 {derived} perm=[H<-W, W<-C, C<-H]
        n139 {derived}: [t356 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {derived}] =
          permute x=t354 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n140 {derived}: [t357 f32 [H=14 W=14 C=256] {derived}] =
          convolution
            x=t355 {derived}
            weight=t356 {derived}
            bias=none
            params={stride={h=1; w=1};
                   padding={h=1; w=1};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n141 {derived}: [t358 f32 [H=256 W=14 C=14] {derived}] =
          permute x=t357 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t358 f32 [H=256 W=14 C=14] {derived}]
    n146 {pt2=root[44] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_13)}: [t368 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_41}] =
      subgraph
        args=[t359 {pt2=root:convolution_13}
                -> t360 {derived},
              t40 {pt2=root:p_layer3_1_bn1_weight target=layer3.1.bn1.weight}
                -> t361 {derived},
              t41 {pt2=root:p_layer3_1_bn1_bias target=layer3.1.bn1.bias}
                -> t362 {derived},
              t101 {pt2=root:b_layer3_1_bn1_running_mean target=layer3.1.bn1.running_mean}
                -> t363 {derived},
              t102 {pt2=root:b_layer3_1_bn1_running_var target=layer3.1.bn1.running_var}
                -> t364 {derived}]
      graph
      inputs:
        [t360 f32 [H=256 W=14 C=14] {derived}, t361 f32 [C=256] {derived},
         t362 f32 [C=256] {derived}, t363 f32 [C=256] {derived},
         t364 f32 [C=256] {derived}]
      nodes:
        n143 {derived}: [t365 f32 [H=14 W=14 C=256] {derived}] =
          permute x=t360 {derived} perm=[H<-W, W<-C, C<-H]
        n144 {derived}: [t366 f32 [H=14 W=14 C=256] {derived}] =
          batch_norm
            x=t365 {derived}
            weight=t361 {derived}
            bias=t362 {derived}
            running_mean=t363 {derived}
            running_var=t364 {derived}
            params={channel=C; eps=1e-05}
        n145 {derived}: [t367 f32 [H=256 W=14 C=14] {derived}] =
          permute x=t366 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t367 f32 [H=256 W=14 C=14] {derived}]
    n147 {pt2=root[45] torch.ops.aten.relu.default (relu_11)}: [t369 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:relu_11}] =
      relu x=t368 {pt2=root:getitem_41}
    n152 {pt2=root[46] torch.ops.aten.convolution.default (convolution_14)}: [t376 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:convolution_14}] =
      subgraph
        args=[t369 {pt2=root:relu_11}
                -> t370 {derived},
              t42 {pt2=root:p_layer3_1_conv2_weight target=layer3.1.conv2.weight}
                -> t371 {derived}]
      graph
      inputs:
        [t370 f32 [H=256 W=14 C=14] {derived},
         t371 f32 [D=256 H=256 W=3 C=3] {derived}]
      nodes:
        n148 {derived}: [t372 f32 [H=14 W=14 C=256] {derived}] =
          permute x=t370 {derived} perm=[H<-W, W<-C, C<-H]
        n149 {derived}: [t373 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {derived}] =
          permute x=t371 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n150 {derived}: [t374 f32 [H=14 W=14 C=256] {derived}] =
          convolution
            x=t372 {derived}
            weight=t373 {derived}
            bias=none
            params={stride={h=1; w=1};
                   padding={h=1; w=1};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n151 {derived}: [t375 f32 [H=256 W=14 C=14] {derived}] =
          permute x=t374 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t375 f32 [H=256 W=14 C=14] {derived}]
    n156 {pt2=root[47] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_14)}: [t385 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_44}] =
      subgraph
        args=[t376 {pt2=root:convolution_14}
                -> t377 {derived},
              t43 {pt2=root:p_layer3_1_bn2_weight target=layer3.1.bn2.weight}
                -> t378 {derived},
              t44 {pt2=root:p_layer3_1_bn2_bias target=layer3.1.bn2.bias}
                -> t379 {derived},
              t104 {pt2=root:b_layer3_1_bn2_running_mean target=layer3.1.bn2.running_mean}
                -> t380 {derived},
              t105 {pt2=root:b_layer3_1_bn2_running_var target=layer3.1.bn2.running_var}
                -> t381 {derived}]
      graph
      inputs:
        [t377 f32 [H=256 W=14 C=14] {derived}, t378 f32 [C=256] {derived},
         t379 f32 [C=256] {derived}, t380 f32 [C=256] {derived},
         t381 f32 [C=256] {derived}]
      nodes:
        n153 {derived}: [t382 f32 [H=14 W=14 C=256] {derived}] =
          permute x=t377 {derived} perm=[H<-W, W<-C, C<-H]
        n154 {derived}: [t383 f32 [H=14 W=14 C=256] {derived}] =
          batch_norm
            x=t382 {derived}
            weight=t378 {derived}
            bias=t379 {derived}
            running_mean=t380 {derived}
            running_var=t381 {derived}
            params={channel=C; eps=1e-05}
        n155 {derived}: [t384 f32 [H=256 W=14 C=14] {derived}] =
          permute x=t383 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t384 f32 [H=256 W=14 C=14] {derived}]
    n157 {pt2=root[48] torch.ops.aten.add.Tensor (add_5)}: [t386 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:add_5}] =
      add a=t385 {pt2=root:getitem_44} b=t352 {pt2=root:relu_10}
    n158 {pt2=root[49] torch.ops.aten.relu.default (relu_12)}: [t387 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:relu_12}] =
      relu x=t386 {pt2=root:add_5}
    n163 {pt2=root[50] torch.ops.aten.convolution.default (convolution_15)}: [t394 f32 [H=512
                                                                      W=7 C=7] {pt2=root:convolution_15}] =
      subgraph
        args=[t387 {pt2=root:relu_12}
                -> t388 {derived},
              t45 {pt2=root:p_layer4_0_conv1_weight target=layer4.0.conv1.weight}
                -> t389 {derived}]
      graph
      inputs:
        [t388 f32 [H=256 W=14 C=14] {derived},
         t389 f32 [D=512 H=256 W=3 C=3] {derived}]
      nodes:
        n159 {derived}: [t390 f32 [H=14 W=14 C=256] {derived}] =
          permute x=t388 {derived} perm=[H<-W, W<-C, C<-H]
        n160 {derived}: [t391 f32 [N=512 T=1 D=1 H=3 W=3 C=256] {derived}] =
          permute x=t389 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n161 {derived}: [t392 f32 [H=7 W=7 C=512] {derived}] =
          convolution
            x=t390 {derived}
            weight=t391 {derived}
            bias=none
            params={stride={h=2; w=2};
                   padding={h=1; w=1};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n162 {derived}: [t393 f32 [H=512 W=7 C=7] {derived}] =
          permute x=t392 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t393 f32 [H=512 W=7 C=7] {derived}]
    n167 {pt2=root[51] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_15)}: [t403 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_47}] =
      subgraph
        args=[t394 {pt2=root:convolution_15}
                -> t395 {derived},
              t46 {pt2=root:p_layer4_0_bn1_weight target=layer4.0.bn1.weight}
                -> t396 {derived},
              t47 {pt2=root:p_layer4_0_bn1_bias target=layer4.0.bn1.bias}
                -> t397 {derived},
              t107 {pt2=root:b_layer4_0_bn1_running_mean target=layer4.0.bn1.running_mean}
                -> t398 {derived},
              t108 {pt2=root:b_layer4_0_bn1_running_var target=layer4.0.bn1.running_var}
                -> t399 {derived}]
      graph
      inputs:
        [t395 f32 [H=512 W=7 C=7] {derived}, t396 f32 [C=512] {derived},
         t397 f32 [C=512] {derived}, t398 f32 [C=512] {derived},
         t399 f32 [C=512] {derived}]
      nodes:
        n164 {derived}: [t400 f32 [H=7 W=7 C=512] {derived}] =
          permute x=t395 {derived} perm=[H<-W, W<-C, C<-H]
        n165 {derived}: [t401 f32 [H=7 W=7 C=512] {derived}] =
          batch_norm
            x=t400 {derived}
            weight=t396 {derived}
            bias=t397 {derived}
            running_mean=t398 {derived}
            running_var=t399 {derived}
            params={channel=C; eps=1e-05}
        n166 {derived}: [t402 f32 [H=512 W=7 C=7] {derived}] =
          permute x=t401 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t402 f32 [H=512 W=7 C=7] {derived}]
    n168 {pt2=root[52] torch.ops.aten.relu.default (relu_13)}: [t404 f32 [H=512
                                                                      W=7 C=7] {pt2=root:relu_13}] =
      relu x=t403 {pt2=root:getitem_47}
    n173 {pt2=root[53] torch.ops.aten.convolution.default (convolution_16)}: [t411 f32 [H=512
                                                                      W=7 C=7] {pt2=root:convolution_16}] =
      subgraph
        args=[t404 {pt2=root:relu_13}
                -> t405 {derived},
              t48 {pt2=root:p_layer4_0_conv2_weight target=layer4.0.conv2.weight}
                -> t406 {derived}]
      graph
      inputs:
        [t405 f32 [H=512 W=7 C=7] {derived},
         t406 f32 [D=512 H=512 W=3 C=3] {derived}]
      nodes:
        n169 {derived}: [t407 f32 [H=7 W=7 C=512] {derived}] =
          permute x=t405 {derived} perm=[H<-W, W<-C, C<-H]
        n170 {derived}: [t408 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {derived}] =
          permute x=t406 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n171 {derived}: [t409 f32 [H=7 W=7 C=512] {derived}] =
          convolution
            x=t407 {derived}
            weight=t408 {derived}
            bias=none
            params={stride={h=1; w=1};
                   padding={h=1; w=1};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n172 {derived}: [t410 f32 [H=512 W=7 C=7] {derived}] =
          permute x=t409 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t410 f32 [H=512 W=7 C=7] {derived}]
    n177 {pt2=root[54] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_16)}: [t420 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_50}] =
      subgraph
        args=[t411 {pt2=root:convolution_16}
                -> t412 {derived},
              t49 {pt2=root:p_layer4_0_bn2_weight target=layer4.0.bn2.weight}
                -> t413 {derived},
              t50 {pt2=root:p_layer4_0_bn2_bias target=layer4.0.bn2.bias}
                -> t414 {derived},
              t110 {pt2=root:b_layer4_0_bn2_running_mean target=layer4.0.bn2.running_mean}
                -> t415 {derived},
              t111 {pt2=root:b_layer4_0_bn2_running_var target=layer4.0.bn2.running_var}
                -> t416 {derived}]
      graph
      inputs:
        [t412 f32 [H=512 W=7 C=7] {derived}, t413 f32 [C=512] {derived},
         t414 f32 [C=512] {derived}, t415 f32 [C=512] {derived},
         t416 f32 [C=512] {derived}]
      nodes:
        n174 {derived}: [t417 f32 [H=7 W=7 C=512] {derived}] =
          permute x=t412 {derived} perm=[H<-W, W<-C, C<-H]
        n175 {derived}: [t418 f32 [H=7 W=7 C=512] {derived}] =
          batch_norm
            x=t417 {derived}
            weight=t413 {derived}
            bias=t414 {derived}
            running_mean=t415 {derived}
            running_var=t416 {derived}
            params={channel=C; eps=1e-05}
        n176 {derived}: [t419 f32 [H=512 W=7 C=7] {derived}] =
          permute x=t418 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t419 f32 [H=512 W=7 C=7] {derived}]
    n182 {pt2=root[55] torch.ops.aten.convolution.default (convolution_17)}: [t427 f32 [H=512
                                                                      W=7 C=7] {pt2=root:convolution_17}] =
      subgraph
        args=[t387 {pt2=root:relu_12}
                -> t421 {derived},
              t51 {pt2=root:p_layer4_0_downsample_0_weight target=layer4.0.downsample.0.weight}
                -> t422 {derived}]
      graph
      inputs:
        [t421 f32 [H=256 W=14 C=14] {derived},
         t422 f32 [D=512 H=256 W=1 C=1] {derived}]
      nodes:
        n178 {derived}: [t423 f32 [H=14 W=14 C=256] {derived}] =
          permute x=t421 {derived} perm=[H<-W, W<-C, C<-H]
        n179 {derived}: [t424 f32 [N=512 T=1 D=1 H=1 W=1 C=256] {derived}] =
          permute x=t422 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n180 {derived}: [t425 f32 [H=7 W=7 C=512] {derived}] =
          convolution
            x=t423 {derived}
            weight=t424 {derived}
            bias=none
            params={stride={h=2; w=2};
                   padding={h=0; w=0};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n181 {derived}: [t426 f32 [H=512 W=7 C=7] {derived}] =
          permute x=t425 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t426 f32 [H=512 W=7 C=7] {derived}]
    n186 {pt2=root[56] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_17)}: [t436 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_53}] =
      subgraph
        args=[t427 {pt2=root:convolution_17}
                -> t428 {derived},
              t52 {pt2=root:p_layer4_0_downsample_1_weight target=layer4.0.downsample.1.weight}
                -> t429 {derived},
              t53 {pt2=root:p_layer4_0_downsample_1_bias target=layer4.0.downsample.1.bias}
                -> t430 {derived},
              t113 {pt2=root:b_layer4_0_downsample_1_running_mean target=layer4.0.downsample.1.running_mean}
                -> t431 {derived},
              t114 {pt2=root:b_layer4_0_downsample_1_running_var target=layer4.0.downsample.1.running_var}
                -> t432 {derived}]
      graph
      inputs:
        [t428 f32 [H=512 W=7 C=7] {derived}, t429 f32 [C=512] {derived},
         t430 f32 [C=512] {derived}, t431 f32 [C=512] {derived},
         t432 f32 [C=512] {derived}]
      nodes:
        n183 {derived}: [t433 f32 [H=7 W=7 C=512] {derived}] =
          permute x=t428 {derived} perm=[H<-W, W<-C, C<-H]
        n184 {derived}: [t434 f32 [H=7 W=7 C=512] {derived}] =
          batch_norm
            x=t433 {derived}
            weight=t429 {derived}
            bias=t430 {derived}
            running_mean=t431 {derived}
            running_var=t432 {derived}
            params={channel=C; eps=1e-05}
        n185 {derived}: [t435 f32 [H=512 W=7 C=7] {derived}] =
          permute x=t434 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t435 f32 [H=512 W=7 C=7] {derived}]
    n187 {pt2=root[57] torch.ops.aten.add.Tensor (add_6)}: [t437 f32 [H=512 W=7
                                                                      C=7] {pt2=root:add_6}] =
      add a=t420 {pt2=root:getitem_50} b=t436 {pt2=root:getitem_53}
    n188 {pt2=root[58] torch.ops.aten.relu.default (relu_14)}: [t438 f32 [H=512
                                                                      W=7 C=7] {pt2=root:relu_14}] =
      relu x=t437 {pt2=root:add_6}
    n193 {pt2=root[59] torch.ops.aten.convolution.default (convolution_18)}: [t445 f32 [H=512
                                                                      W=7 C=7] {pt2=root:convolution_18}] =
      subgraph
        args=[t438 {pt2=root:relu_14}
                -> t439 {derived},
              t54 {pt2=root:p_layer4_1_conv1_weight target=layer4.1.conv1.weight}
                -> t440 {derived}]
      graph
      inputs:
        [t439 f32 [H=512 W=7 C=7] {derived},
         t440 f32 [D=512 H=512 W=3 C=3] {derived}]
      nodes:
        n189 {derived}: [t441 f32 [H=7 W=7 C=512] {derived}] =
          permute x=t439 {derived} perm=[H<-W, W<-C, C<-H]
        n190 {derived}: [t442 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {derived}] =
          permute x=t440 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n191 {derived}: [t443 f32 [H=7 W=7 C=512] {derived}] =
          convolution
            x=t441 {derived}
            weight=t442 {derived}
            bias=none
            params={stride={h=1; w=1};
                   padding={h=1; w=1};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n192 {derived}: [t444 f32 [H=512 W=7 C=7] {derived}] =
          permute x=t443 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t444 f32 [H=512 W=7 C=7] {derived}]
    n197 {pt2=root[60] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_18)}: [t454 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_56}] =
      subgraph
        args=[t445 {pt2=root:convolution_18}
                -> t446 {derived},
              t55 {pt2=root:p_layer4_1_bn1_weight target=layer4.1.bn1.weight}
                -> t447 {derived},
              t56 {pt2=root:p_layer4_1_bn1_bias target=layer4.1.bn1.bias}
                -> t448 {derived},
              t116 {pt2=root:b_layer4_1_bn1_running_mean target=layer4.1.bn1.running_mean}
                -> t449 {derived},
              t117 {pt2=root:b_layer4_1_bn1_running_var target=layer4.1.bn1.running_var}
                -> t450 {derived}]
      graph
      inputs:
        [t446 f32 [H=512 W=7 C=7] {derived}, t447 f32 [C=512] {derived},
         t448 f32 [C=512] {derived}, t449 f32 [C=512] {derived},
         t450 f32 [C=512] {derived}]
      nodes:
        n194 {derived}: [t451 f32 [H=7 W=7 C=512] {derived}] =
          permute x=t446 {derived} perm=[H<-W, W<-C, C<-H]
        n195 {derived}: [t452 f32 [H=7 W=7 C=512] {derived}] =
          batch_norm
            x=t451 {derived}
            weight=t447 {derived}
            bias=t448 {derived}
            running_mean=t449 {derived}
            running_var=t450 {derived}
            params={channel=C; eps=1e-05}
        n196 {derived}: [t453 f32 [H=512 W=7 C=7] {derived}] =
          permute x=t452 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t453 f32 [H=512 W=7 C=7] {derived}]
    n198 {pt2=root[61] torch.ops.aten.relu.default (relu_15)}: [t455 f32 [H=512
                                                                      W=7 C=7] {pt2=root:relu_15}] =
      relu x=t454 {pt2=root:getitem_56}
    n203 {pt2=root[62] torch.ops.aten.convolution.default (convolution_19)}: [t462 f32 [H=512
                                                                      W=7 C=7] {pt2=root:convolution_19}] =
      subgraph
        args=[t455 {pt2=root:relu_15}
                -> t456 {derived},
              t57 {pt2=root:p_layer4_1_conv2_weight target=layer4.1.conv2.weight}
                -> t457 {derived}]
      graph
      inputs:
        [t456 f32 [H=512 W=7 C=7] {derived},
         t457 f32 [D=512 H=512 W=3 C=3] {derived}]
      nodes:
        n199 {derived}: [t458 f32 [H=7 W=7 C=512] {derived}] =
          permute x=t456 {derived} perm=[H<-W, W<-C, C<-H]
        n200 {derived}: [t459 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {derived}] =
          permute x=t457 {derived} perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n201 {derived}: [t460 f32 [H=7 W=7 C=512] {derived}] =
          convolution
            x=t458 {derived}
            weight=t459 {derived}
            bias=none
            params={stride={h=1; w=1};
                   padding={h=1; w=1};
                   dilation={h=1; w=1};
                   transposed=false;
                   output_padding={h=0; w=0};
                   groups=1}
        n202 {derived}: [t461 f32 [H=512 W=7 C=7] {derived}] =
          permute x=t460 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t461 f32 [H=512 W=7 C=7] {derived}]
    n207 {pt2=root[63] torch.ops.aten._native_batch_norm_legit_no_training.default (_native_batch_norm_legit_no_training_19)}: [t471 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_59}] =
      subgraph
        args=[t462 {pt2=root:convolution_19}
                -> t463 {derived},
              t58 {pt2=root:p_layer4_1_bn2_weight target=layer4.1.bn2.weight}
                -> t464 {derived},
              t59 {pt2=root:p_layer4_1_bn2_bias target=layer4.1.bn2.bias}
                -> t465 {derived},
              t119 {pt2=root:b_layer4_1_bn2_running_mean target=layer4.1.bn2.running_mean}
                -> t466 {derived},
              t120 {pt2=root:b_layer4_1_bn2_running_var target=layer4.1.bn2.running_var}
                -> t467 {derived}]
      graph
      inputs:
        [t463 f32 [H=512 W=7 C=7] {derived}, t464 f32 [C=512] {derived},
         t465 f32 [C=512] {derived}, t466 f32 [C=512] {derived},
         t467 f32 [C=512] {derived}]
      nodes:
        n204 {derived}: [t468 f32 [H=7 W=7 C=512] {derived}] =
          permute x=t463 {derived} perm=[H<-W, W<-C, C<-H]
        n205 {derived}: [t469 f32 [H=7 W=7 C=512] {derived}] =
          batch_norm
            x=t468 {derived}
            weight=t464 {derived}
            bias=t465 {derived}
            running_mean=t466 {derived}
            running_var=t467 {derived}
            params={channel=C; eps=1e-05}
        n206 {derived}: [t470 f32 [H=512 W=7 C=7] {derived}] =
          permute x=t469 {derived} perm=[H<-C, W<-H, C<-W]
      outputs: [t470 f32 [H=512 W=7 C=7] {derived}]
    n208 {pt2=root[64] torch.ops.aten.add.Tensor (add_7)}: [t472 f32 [H=512 W=7
                                                                      C=7] {pt2=root:add_7}] =
      add a=t471 {pt2=root:getitem_59} b=t438 {pt2=root:relu_14}
    n209 {pt2=root[65] torch.ops.aten.relu.default (relu_16)}: [t473 f32 [H=512
                                                                      W=7 C=7] {pt2=root:relu_16}] =
      relu x=t472 {pt2=root:add_7}
    n210 {pt2=root[66] torch.ops.aten.mean.dim (mean)}: [t474 f32 [H=512 W=1
                                                                   C=1] {pt2=root:mean}] =
      mean x=t473 {pt2=root:relu_16} params={dims=[C, W]; keepdim=true}
    n211 {pt2=root[67] torch.ops.aten.view.default (view)}: [t475 f32 [C=512] {pt2=root:view}] =
      reshape x=t474 {pt2=root:mean} params={shape=[C=512]}
    n212 {pt2=root[68] torch.ops.aten.permute.default (permute)}: [t476 f32 [W=512
                                                                      C=1000] {pt2=root:permute}] =
      permute x=t60 {pt2=root:p_fc_weight target=fc.weight} perm=[W<-C, C<-W]
    n215 {pt2=root[69] torch.ops.aten.addmm.default (addmm)}: [t482 f32 [C=1000] {pt2=root:addmm}] =
      subgraph
        args=[t61 {pt2=root:p_fc_bias target=fc.bias}
                -> t477 {derived},
              t475 {pt2=root:view}
                -> t478 {derived},
              t476 {pt2=root:permute}
                -> t479 {derived}]
      graph
      inputs:
        [t477 f32 [C=1000] {derived}, t478 f32 [C=512] {derived},
         t479 f32 [W=512 C=1000] {derived}]
      nodes:
        n213 {derived}: [t480 f32 [N=1000 T=1 D=1 H=1 W=1 C=512] {derived}] =
          permute x=t479 {derived} perm=[N<-C, W<-N, C<-W]
        n214 {derived}: [t481 f32 [C=1000] {derived}] =
          linear
            x=t478 {derived}
            weight=t480 {derived}
            bias=t477 {derived}
            params={in_features=512}
      outputs: [t481 f32 [C=1000] {derived}]
  outputs: [t482 f32 [C=1000] {pt2=root:addmm}]
