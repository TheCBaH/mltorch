Rewrite ResNet-18's imported graph with the transformation framework and print
the result, with PT2 provenance recovered for the transformed graph by walking
the composed source-to-destination map backwards rather than by rebuilding the
sidecar. See .ai/native_transform_design.md §10. Gated on PT2_DATA; run with
`make pt2.runtest` after `make pt2.download-cram`.

Structure only, deliberately: executing the result is `make
native-transform-verify`, because a full inference is slow and the residual it
reports is floating point, neither of which belongs in a golden.

Structural passes alone. Nothing is loaded, so no weight is materialised. The
node count falls because the relayout lowering emits an inverse permute pair at
each op boundary and the permute passes cancel them — one of the three cases
§1 was written for. Every id here is an origin id, so packing moves nothing and
every name resolves exactly as the importer recorded it.

  $ ../bin/native_graph.exe transform --pt2 "$PT2_DATA/resnet18/resnet18.pt2"
  nodes: 174 -> 133
  constants: 102, of which 0 folded
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
     t65 f32 [C=64] {pt2=root:b_layer1_0_bn1_running_mean target=layer1.0.bn1.running_mean} constant,
     t66 f32 [C=64] {pt2=root:b_layer1_0_bn1_running_var target=layer1.0.bn1.running_var} constant,
     t68 f32 [C=64] {pt2=root:b_layer1_0_bn2_running_mean target=layer1.0.bn2.running_mean} constant,
     t69 f32 [C=64] {pt2=root:b_layer1_0_bn2_running_var target=layer1.0.bn2.running_var} constant,
     t71 f32 [C=64] {pt2=root:b_layer1_1_bn1_running_mean target=layer1.1.bn1.running_mean} constant,
     t72 f32 [C=64] {pt2=root:b_layer1_1_bn1_running_var target=layer1.1.bn1.running_var} constant,
     t74 f32 [C=64] {pt2=root:b_layer1_1_bn2_running_mean target=layer1.1.bn2.running_mean} constant,
     t75 f32 [C=64] {pt2=root:b_layer1_1_bn2_running_var target=layer1.1.bn2.running_var} constant,
     t77 f32 [C=128] {pt2=root:b_layer2_0_bn1_running_mean target=layer2.0.bn1.running_mean} constant,
     t78 f32 [C=128] {pt2=root:b_layer2_0_bn1_running_var target=layer2.0.bn1.running_var} constant,
     t80 f32 [C=128] {pt2=root:b_layer2_0_bn2_running_mean target=layer2.0.bn2.running_mean} constant,
     t81 f32 [C=128] {pt2=root:b_layer2_0_bn2_running_var target=layer2.0.bn2.running_var} constant,
     t83 f32 [C=128] {pt2=root:b_layer2_0_downsample_1_running_mean target=layer2.0.downsample.1.running_mean} constant,
     t84 f32 [C=128] {pt2=root:b_layer2_0_downsample_1_running_var target=layer2.0.downsample.1.running_var} constant,
     t86 f32 [C=128] {pt2=root:b_layer2_1_bn1_running_mean target=layer2.1.bn1.running_mean} constant,
     t87 f32 [C=128] {pt2=root:b_layer2_1_bn1_running_var target=layer2.1.bn1.running_var} constant,
     t89 f32 [C=128] {pt2=root:b_layer2_1_bn2_running_mean target=layer2.1.bn2.running_mean} constant,
     t90 f32 [C=128] {pt2=root:b_layer2_1_bn2_running_var target=layer2.1.bn2.running_var} constant,
     t92 f32 [C=256] {pt2=root:b_layer3_0_bn1_running_mean target=layer3.0.bn1.running_mean} constant,
     t93 f32 [C=256] {pt2=root:b_layer3_0_bn1_running_var target=layer3.0.bn1.running_var} constant,
     t95 f32 [C=256] {pt2=root:b_layer3_0_bn2_running_mean target=layer3.0.bn2.running_mean} constant,
     t96 f32 [C=256] {pt2=root:b_layer3_0_bn2_running_var target=layer3.0.bn2.running_var} constant,
     t98 f32 [C=256] {pt2=root:b_layer3_0_downsample_1_running_mean target=layer3.0.downsample.1.running_mean} constant,
     t99 f32 [C=256] {pt2=root:b_layer3_0_downsample_1_running_var target=layer3.0.downsample.1.running_var} constant,
     t101 f32 [C=256] {pt2=root:b_layer3_1_bn1_running_mean target=layer3.1.bn1.running_mean} constant,
     t102 f32 [C=256] {pt2=root:b_layer3_1_bn1_running_var target=layer3.1.bn1.running_var} constant,
     t104 f32 [C=256] {pt2=root:b_layer3_1_bn2_running_mean target=layer3.1.bn2.running_mean} constant,
     t105 f32 [C=256] {pt2=root:b_layer3_1_bn2_running_var target=layer3.1.bn2.running_var} constant,
     t107 f32 [C=512] {pt2=root:b_layer4_0_bn1_running_mean target=layer4.0.bn1.running_mean} constant,
     t108 f32 [C=512] {pt2=root:b_layer4_0_bn1_running_var target=layer4.0.bn1.running_var} constant,
     t110 f32 [C=512] {pt2=root:b_layer4_0_bn2_running_mean target=layer4.0.bn2.running_mean} constant,
     t111 f32 [C=512] {pt2=root:b_layer4_0_bn2_running_var target=layer4.0.bn2.running_var} constant,
     t113 f32 [C=512] {pt2=root:b_layer4_0_downsample_1_running_mean target=layer4.0.downsample.1.running_mean} constant,
     t114 f32 [C=512] {pt2=root:b_layer4_0_downsample_1_running_var target=layer4.0.downsample.1.running_var} constant,
     t116 f32 [C=512] {pt2=root:b_layer4_1_bn1_running_mean target=layer4.1.bn1.running_mean} constant,
     t117 f32 [C=512] {pt2=root:b_layer4_1_bn1_running_var target=layer4.1.bn1.running_var} constant,
     t119 f32 [C=512] {pt2=root:b_layer4_1_bn2_running_mean target=layer4.1.bn2.running_mean} constant,
     t120 f32 [C=512] {pt2=root:b_layer4_1_bn2_running_var target=layer4.1.bn2.running_var} constant,
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
    group g4 torch.ops.aten.convolution.default:
      n13 {derived}: [t136 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {derived}] =
        permute
          x=t3 {pt2=root:p_layer1_0_conv1_weight target=layer1.0.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n12 {derived}: [t135 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t134 {pt2=root:getitem_3} perm=[H<-W, W<-C, C<-H]
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
    group g6 torch.ops.aten.convolution.default:
      n21 {derived}: [t144 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {derived}] =
        permute
          x=t6 {pt2=root:p_layer1_0_conv2_weight target=layer1.0.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n20 {derived}: [t143 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t142 {pt2=root:relu_1} perm=[H<-W, W<-C, C<-H]
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
    group g8 torch.ops.aten.convolution.default:
      n30 {derived}: [t153 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {derived}] =
        permute
          x=t9 {pt2=root:p_layer1_1_conv1_weight target=layer1.1.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n29 {derived}: [t152 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t151 {pt2=root:relu_2} perm=[H<-W, W<-C, C<-H]
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
    group g10 torch.ops.aten.convolution.default:
      n38 {derived}: [t161 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {derived}] =
        permute
          x=t12 {pt2=root:p_layer1_1_conv2_weight target=layer1.1.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n37 {derived}: [t160 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t159 {pt2=root:relu_3} perm=[H<-W, W<-C, C<-H]
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
    group g12 torch.ops.aten.convolution.default:
      n47 {derived}: [t170 f32 [N=128 T=1 D=1 H=3 W=3 C=64] {derived}] =
        permute
          x=t15 {pt2=root:p_layer2_0_conv1_weight target=layer2.0.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n46 {derived}: [t169 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t168 {pt2=root:relu_4} perm=[H<-W, W<-C, C<-H]
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
    group g14 torch.ops.aten.convolution.default:
      n55 {derived}: [t178 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {derived}] =
        permute
          x=t18 {pt2=root:p_layer2_0_conv2_weight target=layer2.0.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n54 {derived}: [t177 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t176 {pt2=root:relu_5} perm=[H<-W, W<-C, C<-H]
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
    group g16 torch.ops.aten.convolution.default:
      n62 {derived}: [t185 f32 [N=128 T=1 D=1 H=1 W=1 C=64] {derived}] =
        permute
          x=t21 {pt2=root:p_layer2_0_downsample_0_weight target=layer2.0.downsample.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n61 {derived}: [t184 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t168 {pt2=root:relu_4} perm=[H<-W, W<-C, C<-H]
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
    group g18 torch.ops.aten.convolution.default:
      n71 {derived}: [t194 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {derived}] =
        permute
          x=t24 {pt2=root:p_layer2_1_conv1_weight target=layer2.1.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n70 {derived}: [t193 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t192 {pt2=root:relu_6} perm=[H<-W, W<-C, C<-H]
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
    group g20 torch.ops.aten.convolution.default:
      n79 {derived}: [t202 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {derived}] =
        permute
          x=t27 {pt2=root:p_layer2_1_conv2_weight target=layer2.1.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n78 {derived}: [t201 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t200 {pt2=root:relu_7} perm=[H<-W, W<-C, C<-H]
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
    group g22 torch.ops.aten.convolution.default:
      n88 {derived}: [t211 f32 [N=256 T=1 D=1 H=3 W=3 C=128] {derived}] =
        permute
          x=t30 {pt2=root:p_layer3_0_conv1_weight target=layer3.0.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n87 {derived}: [t210 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t209 {pt2=root:relu_8} perm=[H<-W, W<-C, C<-H]
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
    group g24 torch.ops.aten.convolution.default:
      n96 {derived}: [t219 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {derived}] =
        permute
          x=t33 {pt2=root:p_layer3_0_conv2_weight target=layer3.0.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n95 {derived}: [t218 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t217 {pt2=root:relu_9} perm=[H<-W, W<-C, C<-H]
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
    group g26 torch.ops.aten.convolution.default:
      n103 {derived}: [t226 f32 [N=256 T=1 D=1 H=1 W=1 C=128] {derived}] =
        permute
          x=t36 {pt2=root:p_layer3_0_downsample_0_weight target=layer3.0.downsample.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n102 {derived}: [t225 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t209 {pt2=root:relu_8} perm=[H<-W, W<-C, C<-H]
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
    group g28 torch.ops.aten.convolution.default:
      n112 {derived}: [t235 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {derived}] =
        permute
          x=t39 {pt2=root:p_layer3_1_conv1_weight target=layer3.1.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n111 {derived}: [t234 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t233 {pt2=root:relu_10} perm=[H<-W, W<-C, C<-H]
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
    group g30 torch.ops.aten.convolution.default:
      n120 {derived}: [t243 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {derived}] =
        permute
          x=t42 {pt2=root:p_layer3_1_conv2_weight target=layer3.1.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n119 {derived}: [t242 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t241 {pt2=root:relu_11} perm=[H<-W, W<-C, C<-H]
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
    group g32 torch.ops.aten.convolution.default:
      n129 {derived}: [t252 f32 [N=512 T=1 D=1 H=3 W=3 C=256] {derived}] =
        permute
          x=t45 {pt2=root:p_layer4_0_conv1_weight target=layer4.0.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n128 {derived}: [t251 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t250 {pt2=root:relu_12} perm=[H<-W, W<-C, C<-H]
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
    group g34 torch.ops.aten.convolution.default:
      n137 {derived}: [t260 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {derived}] =
        permute
          x=t48 {pt2=root:p_layer4_0_conv2_weight target=layer4.0.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n136 {derived}: [t259 f32 [H=7 W=7 C=512] {derived}] =
        permute x=t258 {pt2=root:relu_13} perm=[H<-W, W<-C, C<-H]
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
    group g36 torch.ops.aten.convolution.default:
      n144 {derived}: [t267 f32 [N=512 T=1 D=1 H=1 W=1 C=256] {derived}] =
        permute
          x=t51 {pt2=root:p_layer4_0_downsample_0_weight target=layer4.0.downsample.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n143 {derived}: [t266 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t250 {pt2=root:relu_12} perm=[H<-W, W<-C, C<-H]
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
    group g38 torch.ops.aten.convolution.default:
      n153 {derived}: [t276 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {derived}] =
        permute
          x=t54 {pt2=root:p_layer4_1_conv1_weight target=layer4.1.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n152 {derived}: [t275 f32 [H=7 W=7 C=512] {derived}] =
        permute x=t274 {pt2=root:relu_14} perm=[H<-W, W<-C, C<-H]
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
    group g40 torch.ops.aten.convolution.default:
      n161 {derived}: [t284 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {derived}] =
        permute
          x=t57 {pt2=root:p_layer4_1_conv2_weight target=layer4.1.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n160 {derived}: [t283 f32 [H=7 W=7 C=512] {derived}] =
        permute x=t282 {pt2=root:relu_15} perm=[H<-W, W<-C, C<-H]
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
    n174 {pt2=root[68] torch.ops.aten.permute.default}: [t295 f32 [N=1000 T=1
                                                                   D=1 H=1 W=1
                                                                   C=512] {derived}] =
      permute x=t60 {pt2=root:p_fc_weight target=fc.weight} perm=[N<-W, W<-N]
    group g2 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n5 {derived}: [t128 f32 [H=112 W=112 C=64] {derived}] =
        batch_norm
          x=t125 {derived}
          weight=t1 {pt2=root:p_bn1_weight target=bn1.weight}
          bias=t2 {pt2=root:p_bn1_bias target=bn1.bias}
          running_mean=t62 {pt2=root:b_bn1_running_mean target=bn1.running_mean}
          running_var=t63 {pt2=root:b_bn1_running_var target=bn1.running_var}
          params={channel=C; eps=1e-05}
      n6 {pt2=root[1] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t129 f32 [H=64
                                                                      W=112
                                                                      C=112] {pt2=root:getitem}] =
        permute x=t128 {derived} perm=[H<-C, W<-H, C<-W]
    n7 {pt2=root[2] torch.ops.aten.relu.default}: [t130 f32 [H=64 W=112 C=112] {pt2=root:relu}] =
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
      n11 {pt2=root[3] torch.ops.aten.max_pool2d_with_indices.default}: [t134 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_3}] =
        permute x=t132 {derived} perm=[H<-C, W<-H, C<-W]
    group g5 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n17 {derived}: [t140 f32 [H=56 W=56 C=64] {derived}] =
        batch_norm
          x=t137 {derived}
          weight=t4 {pt2=root:p_layer1_0_bn1_weight target=layer1.0.bn1.weight}
          bias=t5 {pt2=root:p_layer1_0_bn1_bias target=layer1.0.bn1.bias}
          running_mean=t65 {pt2=root:b_layer1_0_bn1_running_mean target=layer1.0.bn1.running_mean}
          running_var=t66 {pt2=root:b_layer1_0_bn1_running_var target=layer1.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n18 {pt2=root[5] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t141 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_5}] =
        permute x=t140 {derived} perm=[H<-C, W<-H, C<-W]
    n19 {pt2=root[6] torch.ops.aten.relu.default}: [t142 f32 [H=64 W=56 C=56] {pt2=root:relu_1}] =
      relu x=t141 {pt2=root:getitem_5}
    group g7 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n25 {derived}: [t148 f32 [H=56 W=56 C=64] {derived}] =
        batch_norm
          x=t145 {derived}
          weight=t7 {pt2=root:p_layer1_0_bn2_weight target=layer1.0.bn2.weight}
          bias=t8 {pt2=root:p_layer1_0_bn2_bias target=layer1.0.bn2.bias}
          running_mean=t68 {pt2=root:b_layer1_0_bn2_running_mean target=layer1.0.bn2.running_mean}
          running_var=t69 {pt2=root:b_layer1_0_bn2_running_var target=layer1.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n26 {pt2=root[8] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t149 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_8}] =
        permute x=t148 {derived} perm=[H<-C, W<-H, C<-W]
    n27 {pt2=root[9] torch.ops.aten.add.Tensor}: [t150 f32 [H=64 W=56 C=56] {pt2=root:add}] =
      add a=t149 {pt2=root:getitem_8} b=t134 {pt2=root:getitem_3}
    n28 {pt2=root[10] torch.ops.aten.relu.default}: [t151 f32 [H=64 W=56 C=56] {pt2=root:relu_2}] =
      relu x=t150 {pt2=root:add}
    group g9 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n34 {derived}: [t157 f32 [H=56 W=56 C=64] {derived}] =
        batch_norm
          x=t154 {derived}
          weight=t10 {pt2=root:p_layer1_1_bn1_weight target=layer1.1.bn1.weight}
          bias=t11 {pt2=root:p_layer1_1_bn1_bias target=layer1.1.bn1.bias}
          running_mean=t71 {pt2=root:b_layer1_1_bn1_running_mean target=layer1.1.bn1.running_mean}
          running_var=t72 {pt2=root:b_layer1_1_bn1_running_var target=layer1.1.bn1.running_var}
          params={channel=C; eps=1e-05}
      n35 {pt2=root[12] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t158 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_11}] =
        permute x=t157 {derived} perm=[H<-C, W<-H, C<-W]
    n36 {pt2=root[13] torch.ops.aten.relu.default}: [t159 f32 [H=64 W=56 C=56] {pt2=root:relu_3}] =
      relu x=t158 {pt2=root:getitem_11}
    group g11 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n42 {derived}: [t165 f32 [H=56 W=56 C=64] {derived}] =
        batch_norm
          x=t162 {derived}
          weight=t13 {pt2=root:p_layer1_1_bn2_weight target=layer1.1.bn2.weight}
          bias=t14 {pt2=root:p_layer1_1_bn2_bias target=layer1.1.bn2.bias}
          running_mean=t74 {pt2=root:b_layer1_1_bn2_running_mean target=layer1.1.bn2.running_mean}
          running_var=t75 {pt2=root:b_layer1_1_bn2_running_var target=layer1.1.bn2.running_var}
          params={channel=C; eps=1e-05}
      n43 {pt2=root[15] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t166 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_14}] =
        permute x=t165 {derived} perm=[H<-C, W<-H, C<-W]
    n44 {pt2=root[16] torch.ops.aten.add.Tensor}: [t167 f32 [H=64 W=56 C=56] {pt2=root:add_1}] =
      add a=t166 {pt2=root:getitem_14} b=t151 {pt2=root:relu_2}
    n45 {pt2=root[17] torch.ops.aten.relu.default}: [t168 f32 [H=64 W=56 C=56] {pt2=root:relu_4}] =
      relu x=t167 {pt2=root:add_1}
    group g13 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n51 {derived}: [t174 f32 [H=28 W=28 C=128] {derived}] =
        batch_norm
          x=t171 {derived}
          weight=t16 {pt2=root:p_layer2_0_bn1_weight target=layer2.0.bn1.weight}
          bias=t17 {pt2=root:p_layer2_0_bn1_bias target=layer2.0.bn1.bias}
          running_mean=t77 {pt2=root:b_layer2_0_bn1_running_mean target=layer2.0.bn1.running_mean}
          running_var=t78 {pt2=root:b_layer2_0_bn1_running_var target=layer2.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n52 {pt2=root[19] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t175 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_17}] =
        permute x=t174 {derived} perm=[H<-C, W<-H, C<-W]
    group g17 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n66 {derived}: [t189 f32 [H=28 W=28 C=128] {derived}] =
        batch_norm
          x=t186 {derived}
          weight=t22 {pt2=root:p_layer2_0_downsample_1_weight target=layer2.0.downsample.1.weight}
          bias=t23 {pt2=root:p_layer2_0_downsample_1_bias target=layer2.0.downsample.1.bias}
          running_mean=t83 {pt2=root:b_layer2_0_downsample_1_running_mean target=layer2.0.downsample.1.running_mean}
          running_var=t84 {pt2=root:b_layer2_0_downsample_1_running_var target=layer2.0.downsample.1.running_var}
          params={channel=C; eps=1e-05}
      n67 {pt2=root[24] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t190 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_23}] =
        permute x=t189 {derived} perm=[H<-C, W<-H, C<-W]
    n53 {pt2=root[20] torch.ops.aten.relu.default}: [t176 f32 [H=128 W=28 C=28] {pt2=root:relu_5}] =
      relu x=t175 {pt2=root:getitem_17}
    group g15 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n59 {derived}: [t182 f32 [H=28 W=28 C=128] {derived}] =
        batch_norm
          x=t179 {derived}
          weight=t19 {pt2=root:p_layer2_0_bn2_weight target=layer2.0.bn2.weight}
          bias=t20 {pt2=root:p_layer2_0_bn2_bias target=layer2.0.bn2.bias}
          running_mean=t80 {pt2=root:b_layer2_0_bn2_running_mean target=layer2.0.bn2.running_mean}
          running_var=t81 {pt2=root:b_layer2_0_bn2_running_var target=layer2.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n60 {pt2=root[22] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t183 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_20}] =
        permute x=t182 {derived} perm=[H<-C, W<-H, C<-W]
    n68 {pt2=root[25] torch.ops.aten.add.Tensor}: [t191 f32 [H=128 W=28 C=28] {pt2=root:add_2}] =
      add a=t183 {pt2=root:getitem_20} b=t190 {pt2=root:getitem_23}
    n69 {pt2=root[26] torch.ops.aten.relu.default}: [t192 f32 [H=128 W=28 C=28] {pt2=root:relu_6}] =
      relu x=t191 {pt2=root:add_2}
    group g19 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n75 {derived}: [t198 f32 [H=28 W=28 C=128] {derived}] =
        batch_norm
          x=t195 {derived}
          weight=t25 {pt2=root:p_layer2_1_bn1_weight target=layer2.1.bn1.weight}
          bias=t26 {pt2=root:p_layer2_1_bn1_bias target=layer2.1.bn1.bias}
          running_mean=t86 {pt2=root:b_layer2_1_bn1_running_mean target=layer2.1.bn1.running_mean}
          running_var=t87 {pt2=root:b_layer2_1_bn1_running_var target=layer2.1.bn1.running_var}
          params={channel=C; eps=1e-05}
      n76 {pt2=root[28] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t199 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_26}] =
        permute x=t198 {derived} perm=[H<-C, W<-H, C<-W]
    n77 {pt2=root[29] torch.ops.aten.relu.default}: [t200 f32 [H=128 W=28 C=28] {pt2=root:relu_7}] =
      relu x=t199 {pt2=root:getitem_26}
    group g21 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n83 {derived}: [t206 f32 [H=28 W=28 C=128] {derived}] =
        batch_norm
          x=t203 {derived}
          weight=t28 {pt2=root:p_layer2_1_bn2_weight target=layer2.1.bn2.weight}
          bias=t29 {pt2=root:p_layer2_1_bn2_bias target=layer2.1.bn2.bias}
          running_mean=t89 {pt2=root:b_layer2_1_bn2_running_mean target=layer2.1.bn2.running_mean}
          running_var=t90 {pt2=root:b_layer2_1_bn2_running_var target=layer2.1.bn2.running_var}
          params={channel=C; eps=1e-05}
      n84 {pt2=root[31] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t207 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_29}] =
        permute x=t206 {derived} perm=[H<-C, W<-H, C<-W]
    n85 {pt2=root[32] torch.ops.aten.add.Tensor}: [t208 f32 [H=128 W=28 C=28] {pt2=root:add_3}] =
      add a=t207 {pt2=root:getitem_29} b=t192 {pt2=root:relu_6}
    n86 {pt2=root[33] torch.ops.aten.relu.default}: [t209 f32 [H=128 W=28 C=28] {pt2=root:relu_8}] =
      relu x=t208 {pt2=root:add_3}
    group g23 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n92 {derived}: [t215 f32 [H=14 W=14 C=256] {derived}] =
        batch_norm
          x=t212 {derived}
          weight=t31 {pt2=root:p_layer3_0_bn1_weight target=layer3.0.bn1.weight}
          bias=t32 {pt2=root:p_layer3_0_bn1_bias target=layer3.0.bn1.bias}
          running_mean=t92 {pt2=root:b_layer3_0_bn1_running_mean target=layer3.0.bn1.running_mean}
          running_var=t93 {pt2=root:b_layer3_0_bn1_running_var target=layer3.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n93 {pt2=root[35] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t216 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_32}] =
        permute x=t215 {derived} perm=[H<-C, W<-H, C<-W]
    group g27 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n107 {derived}: [t230 f32 [H=14 W=14 C=256] {derived}] =
        batch_norm
          x=t227 {derived}
          weight=t37 {pt2=root:p_layer3_0_downsample_1_weight target=layer3.0.downsample.1.weight}
          bias=t38 {pt2=root:p_layer3_0_downsample_1_bias target=layer3.0.downsample.1.bias}
          running_mean=t98 {pt2=root:b_layer3_0_downsample_1_running_mean target=layer3.0.downsample.1.running_mean}
          running_var=t99 {pt2=root:b_layer3_0_downsample_1_running_var target=layer3.0.downsample.1.running_var}
          params={channel=C; eps=1e-05}
      n108 {pt2=root[40] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t231 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_38}] =
        permute x=t230 {derived} perm=[H<-C, W<-H, C<-W]
    n94 {pt2=root[36] torch.ops.aten.relu.default}: [t217 f32 [H=256 W=14 C=14] {pt2=root:relu_9}] =
      relu x=t216 {pt2=root:getitem_32}
    group g25 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n100 {derived}: [t223 f32 [H=14 W=14 C=256] {derived}] =
        batch_norm
          x=t220 {derived}
          weight=t34 {pt2=root:p_layer3_0_bn2_weight target=layer3.0.bn2.weight}
          bias=t35 {pt2=root:p_layer3_0_bn2_bias target=layer3.0.bn2.bias}
          running_mean=t95 {pt2=root:b_layer3_0_bn2_running_mean target=layer3.0.bn2.running_mean}
          running_var=t96 {pt2=root:b_layer3_0_bn2_running_var target=layer3.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n101 {pt2=root[38] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t224 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_35}] =
        permute x=t223 {derived} perm=[H<-C, W<-H, C<-W]
    n109 {pt2=root[41] torch.ops.aten.add.Tensor}: [t232 f32 [H=256 W=14 C=14] {pt2=root:add_4}] =
      add a=t224 {pt2=root:getitem_35} b=t231 {pt2=root:getitem_38}
    n110 {pt2=root[42] torch.ops.aten.relu.default}: [t233 f32 [H=256 W=14
                                                                C=14] {pt2=root:relu_10}] =
      relu x=t232 {pt2=root:add_4}
    group g29 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n116 {derived}: [t239 f32 [H=14 W=14 C=256] {derived}] =
        batch_norm
          x=t236 {derived}
          weight=t40 {pt2=root:p_layer3_1_bn1_weight target=layer3.1.bn1.weight}
          bias=t41 {pt2=root:p_layer3_1_bn1_bias target=layer3.1.bn1.bias}
          running_mean=t101 {pt2=root:b_layer3_1_bn1_running_mean target=layer3.1.bn1.running_mean}
          running_var=t102 {pt2=root:b_layer3_1_bn1_running_var target=layer3.1.bn1.running_var}
          params={channel=C; eps=1e-05}
      n117 {pt2=root[44] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t240 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_41}] =
        permute x=t239 {derived} perm=[H<-C, W<-H, C<-W]
    n118 {pt2=root[45] torch.ops.aten.relu.default}: [t241 f32 [H=256 W=14
                                                                C=14] {pt2=root:relu_11}] =
      relu x=t240 {pt2=root:getitem_41}
    group g31 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n124 {derived}: [t247 f32 [H=14 W=14 C=256] {derived}] =
        batch_norm
          x=t244 {derived}
          weight=t43 {pt2=root:p_layer3_1_bn2_weight target=layer3.1.bn2.weight}
          bias=t44 {pt2=root:p_layer3_1_bn2_bias target=layer3.1.bn2.bias}
          running_mean=t104 {pt2=root:b_layer3_1_bn2_running_mean target=layer3.1.bn2.running_mean}
          running_var=t105 {pt2=root:b_layer3_1_bn2_running_var target=layer3.1.bn2.running_var}
          params={channel=C; eps=1e-05}
      n125 {pt2=root[47] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t248 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_44}] =
        permute x=t247 {derived} perm=[H<-C, W<-H, C<-W]
    n126 {pt2=root[48] torch.ops.aten.add.Tensor}: [t249 f32 [H=256 W=14 C=14] {pt2=root:add_5}] =
      add a=t248 {pt2=root:getitem_44} b=t233 {pt2=root:relu_10}
    n127 {pt2=root[49] torch.ops.aten.relu.default}: [t250 f32 [H=256 W=14
                                                                C=14] {pt2=root:relu_12}] =
      relu x=t249 {pt2=root:add_5}
    group g33 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n133 {derived}: [t256 f32 [H=7 W=7 C=512] {derived}] =
        batch_norm
          x=t253 {derived}
          weight=t46 {pt2=root:p_layer4_0_bn1_weight target=layer4.0.bn1.weight}
          bias=t47 {pt2=root:p_layer4_0_bn1_bias target=layer4.0.bn1.bias}
          running_mean=t107 {pt2=root:b_layer4_0_bn1_running_mean target=layer4.0.bn1.running_mean}
          running_var=t108 {pt2=root:b_layer4_0_bn1_running_var target=layer4.0.bn1.running_var}
          params={channel=C; eps=1e-05}
      n134 {pt2=root[51] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t257 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_47}] =
        permute x=t256 {derived} perm=[H<-C, W<-H, C<-W]
    group g37 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n148 {derived}: [t271 f32 [H=7 W=7 C=512] {derived}] =
        batch_norm
          x=t268 {derived}
          weight=t52 {pt2=root:p_layer4_0_downsample_1_weight target=layer4.0.downsample.1.weight}
          bias=t53 {pt2=root:p_layer4_0_downsample_1_bias target=layer4.0.downsample.1.bias}
          running_mean=t113 {pt2=root:b_layer4_0_downsample_1_running_mean target=layer4.0.downsample.1.running_mean}
          running_var=t114 {pt2=root:b_layer4_0_downsample_1_running_var target=layer4.0.downsample.1.running_var}
          params={channel=C; eps=1e-05}
      n149 {pt2=root[56] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t272 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_53}] =
        permute x=t271 {derived} perm=[H<-C, W<-H, C<-W]
    n135 {pt2=root[52] torch.ops.aten.relu.default}: [t258 f32 [H=512 W=7 C=7] {pt2=root:relu_13}] =
      relu x=t257 {pt2=root:getitem_47}
    group g35 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n141 {derived}: [t264 f32 [H=7 W=7 C=512] {derived}] =
        batch_norm
          x=t261 {derived}
          weight=t49 {pt2=root:p_layer4_0_bn2_weight target=layer4.0.bn2.weight}
          bias=t50 {pt2=root:p_layer4_0_bn2_bias target=layer4.0.bn2.bias}
          running_mean=t110 {pt2=root:b_layer4_0_bn2_running_mean target=layer4.0.bn2.running_mean}
          running_var=t111 {pt2=root:b_layer4_0_bn2_running_var target=layer4.0.bn2.running_var}
          params={channel=C; eps=1e-05}
      n142 {pt2=root[54] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t265 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_50}] =
        permute x=t264 {derived} perm=[H<-C, W<-H, C<-W]
    n150 {pt2=root[57] torch.ops.aten.add.Tensor}: [t273 f32 [H=512 W=7 C=7] {pt2=root:add_6}] =
      add a=t265 {pt2=root:getitem_50} b=t272 {pt2=root:getitem_53}
    n151 {pt2=root[58] torch.ops.aten.relu.default}: [t274 f32 [H=512 W=7 C=7] {pt2=root:relu_14}] =
      relu x=t273 {pt2=root:add_6}
    group g39 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n157 {derived}: [t280 f32 [H=7 W=7 C=512] {derived}] =
        batch_norm
          x=t277 {derived}
          weight=t55 {pt2=root:p_layer4_1_bn1_weight target=layer4.1.bn1.weight}
          bias=t56 {pt2=root:p_layer4_1_bn1_bias target=layer4.1.bn1.bias}
          running_mean=t116 {pt2=root:b_layer4_1_bn1_running_mean target=layer4.1.bn1.running_mean}
          running_var=t117 {pt2=root:b_layer4_1_bn1_running_var target=layer4.1.bn1.running_var}
          params={channel=C; eps=1e-05}
      n158 {pt2=root[60] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t281 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_56}] =
        permute x=t280 {derived} perm=[H<-C, W<-H, C<-W]
    n159 {pt2=root[61] torch.ops.aten.relu.default}: [t282 f32 [H=512 W=7 C=7] {pt2=root:relu_15}] =
      relu x=t281 {pt2=root:getitem_56}
    group g41 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n165 {derived}: [t288 f32 [H=7 W=7 C=512] {derived}] =
        batch_norm
          x=t285 {derived}
          weight=t58 {pt2=root:p_layer4_1_bn2_weight target=layer4.1.bn2.weight}
          bias=t59 {pt2=root:p_layer4_1_bn2_bias target=layer4.1.bn2.bias}
          running_mean=t119 {pt2=root:b_layer4_1_bn2_running_mean target=layer4.1.bn2.running_mean}
          running_var=t120 {pt2=root:b_layer4_1_bn2_running_var target=layer4.1.bn2.running_var}
          params={channel=C; eps=1e-05}
      n166 {pt2=root[63] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t289 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_59}] =
        permute x=t288 {derived} perm=[H<-C, W<-H, C<-W]
    n167 {pt2=root[64] torch.ops.aten.add.Tensor}: [t290 f32 [H=512 W=7 C=7] {pt2=root:add_7}] =
      add a=t289 {pt2=root:getitem_59} b=t274 {pt2=root:relu_14}
    n168 {pt2=root[65] torch.ops.aten.relu.default}: [t291 f32 [H=512 W=7 C=7] {pt2=root:relu_16}] =
      relu x=t290 {pt2=root:add_7}
    n169 {pt2=root[66] torch.ops.aten.mean.dim}: [t292 f32 [H=512 W=1 C=1] {pt2=root:mean}] =
      mean x=t291 {pt2=root:relu_16} params={dims=[C, W]; keepdim=true}
    n175 {pt2=root[67] torch.ops.aten.view.default}: [t293 f32 [C=512] {pt2=root:view}] =
      permute x=t292 {pt2=root:mean} perm=[H<-W, W<-C, C<-H]
    group g42 torch.ops.aten.addmm.default:
      n173 {pt2=root[69] torch.ops.aten.addmm.default}: [t296 f32 [C=1000] {pt2=root:addmm}] =
        linear
          x=t293 {pt2=root:view}
          weight=t295 {derived}
          bias=t61 {pt2=root:p_fc_bias target=fc.bias}
          params={in_features=512}
  outputs: [t296 f32 [C=1000] {pt2=root:addmm}]

With `--fold`, the weights are bound up front and constant folding hoists the
permuted conv weights to load time — the motivating case of the whole framework,
which today re-permutes every OIHW weight on every inference. That in turn makes
the weights constant, which is what lets batch-norm folding fire: all 20 norms
disappear into the convolutions feeding them, and the arithmetic they emit folds
away in the same pipeline.

The folded weights and biases have no PT2 name and no archive path, because the
captured bytes are the pre-fold values and handing them back would be corruption
rather than imprecision. What they were computed FROM is still recorded, and
prints as `folded from=[...]` — provenance answering the question value
correspondence must not.

  $ ../bin/native_graph.exe transform --fold --pt2 "$PT2_DATA/resnet18/resnet18.pt2"
  nodes: 174 -> 92
  constants: 42, of which 41 folded
  graph
  inputs:
    [t61 f32 [C=1000] {pt2=root:p_fc_bias target=fc.bias} constant,
     t122 f32 [H=3 W=224 C=224] {pt2=root:x},
     t295 f32 [N=1000 T=1 D=1 H=1 W=1 C=512] {folded from=[p_fc_weight]} constant,
     t297 f32 [N=64 T=1 D=1 H=7 W=7 C=3] {folded from=[p_conv1_weight,
                                                       p_bn1_weight,
                                                       b_bn1_running_var]} constant,
     t298 f32 [C=64] {folded from=[p_bn1_weight, p_bn1_bias,
                                   b_bn1_running_mean, b_bn1_running_var]} constant,
     t299 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_0_conv1_weight,
                                                        p_layer1_0_bn1_weight,
                                                        b_layer1_0_bn1_running_var]} constant,
     t300 f32 [C=64] {folded from=[p_layer1_0_bn1_weight, p_layer1_0_bn1_bias,
                                   b_layer1_0_bn1_running_mean,
                                   b_layer1_0_bn1_running_var]} constant,
     t301 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_0_conv2_weight,
                                                        p_layer1_0_bn2_weight,
                                                        b_layer1_0_bn2_running_var]} constant,
     t302 f32 [C=64] {folded from=[p_layer1_0_bn2_weight, p_layer1_0_bn2_bias,
                                   b_layer1_0_bn2_running_mean,
                                   b_layer1_0_bn2_running_var]} constant,
     t303 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_1_conv1_weight,
                                                        p_layer1_1_bn1_weight,
                                                        b_layer1_1_bn1_running_var]} constant,
     t304 f32 [C=64] {folded from=[p_layer1_1_bn1_weight, p_layer1_1_bn1_bias,
                                   b_layer1_1_bn1_running_mean,
                                   b_layer1_1_bn1_running_var]} constant,
     t305 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_1_conv2_weight,
                                                        p_layer1_1_bn2_weight,
                                                        b_layer1_1_bn2_running_var]} constant,
     t306 f32 [C=64] {folded from=[p_layer1_1_bn2_weight, p_layer1_1_bn2_bias,
                                   b_layer1_1_bn2_running_mean,
                                   b_layer1_1_bn2_running_var]} constant,
     t307 f32 [N=128 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer2_0_conv1_weight,
                                                         p_layer2_0_bn1_weight,
                                                         b_layer2_0_bn1_running_var]} constant,
     t308 f32 [C=128] {folded from=[p_layer2_0_bn1_weight, p_layer2_0_bn1_bias,
                                    b_layer2_0_bn1_running_mean,
                                    b_layer2_0_bn1_running_var]} constant,
     t309 f32 [N=128 T=1 D=1 H=1 W=1 C=64] {folded from=[p_layer2_0_downsample_0_weight,
                                                         p_layer2_0_downsample_1_weight,
                                                         b_layer2_0_downsample_1_running_var]} constant,
     t310 f32 [C=128] {folded from=[p_layer2_0_downsample_1_weight,
                                    p_layer2_0_downsample_1_bias,
                                    b_layer2_0_downsample_1_running_mean,
                                    b_layer2_0_downsample_1_running_var]} constant,
     t311 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer2_0_conv2_weight,
                                                          p_layer2_0_bn2_weight,
                                                          b_layer2_0_bn2_running_var]} constant,
     t312 f32 [C=128] {folded from=[p_layer2_0_bn2_weight, p_layer2_0_bn2_bias,
                                    b_layer2_0_bn2_running_mean,
                                    b_layer2_0_bn2_running_var]} constant,
     t313 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer2_1_conv1_weight,
                                                          p_layer2_1_bn1_weight,
                                                          b_layer2_1_bn1_running_var]} constant,
     t314 f32 [C=128] {folded from=[p_layer2_1_bn1_weight, p_layer2_1_bn1_bias,
                                    b_layer2_1_bn1_running_mean,
                                    b_layer2_1_bn1_running_var]} constant,
     t315 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer2_1_conv2_weight,
                                                          p_layer2_1_bn2_weight,
                                                          b_layer2_1_bn2_running_var]} constant,
     t316 f32 [C=128] {folded from=[p_layer2_1_bn2_weight, p_layer2_1_bn2_bias,
                                    b_layer2_1_bn2_running_mean,
                                    b_layer2_1_bn2_running_var]} constant,
     t317 f32 [N=256 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer3_0_conv1_weight,
                                                          p_layer3_0_bn1_weight,
                                                          b_layer3_0_bn1_running_var]} constant,
     t318 f32 [C=256] {folded from=[p_layer3_0_bn1_weight, p_layer3_0_bn1_bias,
                                    b_layer3_0_bn1_running_mean,
                                    b_layer3_0_bn1_running_var]} constant,
     t319 f32 [N=256 T=1 D=1 H=1 W=1 C=128] {folded from=[p_layer3_0_downsample_0_weight,
                                                          p_layer3_0_downsample_1_weight,
                                                          b_layer3_0_downsample_1_running_var]} constant,
     t320 f32 [C=256] {folded from=[p_layer3_0_downsample_1_weight,
                                    p_layer3_0_downsample_1_bias,
                                    b_layer3_0_downsample_1_running_mean,
                                    b_layer3_0_downsample_1_running_var]} constant,
     t321 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer3_0_conv2_weight,
                                                          p_layer3_0_bn2_weight,
                                                          b_layer3_0_bn2_running_var]} constant,
     t322 f32 [C=256] {folded from=[p_layer3_0_bn2_weight, p_layer3_0_bn2_bias,
                                    b_layer3_0_bn2_running_mean,
                                    b_layer3_0_bn2_running_var]} constant,
     t323 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer3_1_conv1_weight,
                                                          p_layer3_1_bn1_weight,
                                                          b_layer3_1_bn1_running_var]} constant,
     t324 f32 [C=256] {folded from=[p_layer3_1_bn1_weight, p_layer3_1_bn1_bias,
                                    b_layer3_1_bn1_running_mean,
                                    b_layer3_1_bn1_running_var]} constant,
     t325 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer3_1_conv2_weight,
                                                          p_layer3_1_bn2_weight,
                                                          b_layer3_1_bn2_running_var]} constant,
     t326 f32 [C=256] {folded from=[p_layer3_1_bn2_weight, p_layer3_1_bn2_bias,
                                    b_layer3_1_bn2_running_mean,
                                    b_layer3_1_bn2_running_var]} constant,
     t327 f32 [N=512 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer4_0_conv1_weight,
                                                          p_layer4_0_bn1_weight,
                                                          b_layer4_0_bn1_running_var]} constant,
     t328 f32 [C=512] {folded from=[p_layer4_0_bn1_weight, p_layer4_0_bn1_bias,
                                    b_layer4_0_bn1_running_mean,
                                    b_layer4_0_bn1_running_var]} constant,
     t329 f32 [N=512 T=1 D=1 H=1 W=1 C=256] {folded from=[p_layer4_0_downsample_0_weight,
                                                          p_layer4_0_downsample_1_weight,
                                                          b_layer4_0_downsample_1_running_var]} constant,
     t330 f32 [C=512] {folded from=[p_layer4_0_downsample_1_weight,
                                    p_layer4_0_downsample_1_bias,
                                    b_layer4_0_downsample_1_running_mean,
                                    b_layer4_0_downsample_1_running_var]} constant,
     t331 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {folded from=[p_layer4_0_conv2_weight,
                                                          p_layer4_0_bn2_weight,
                                                          b_layer4_0_bn2_running_var]} constant,
     t332 f32 [C=512] {folded from=[p_layer4_0_bn2_weight, p_layer4_0_bn2_bias,
                                    b_layer4_0_bn2_running_mean,
                                    b_layer4_0_bn2_running_var]} constant,
     t333 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {folded from=[p_layer4_1_conv1_weight,
                                                          p_layer4_1_bn1_weight,
                                                          b_layer4_1_bn1_running_var]} constant,
     t334 f32 [C=512] {folded from=[p_layer4_1_bn1_weight, p_layer4_1_bn1_bias,
                                    b_layer4_1_bn1_running_mean,
                                    b_layer4_1_bn1_running_var]} constant,
     t335 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {folded from=[p_layer4_1_conv2_weight,
                                                          p_layer4_1_bn2_weight,
                                                          b_layer4_1_bn2_running_var]} constant,
     t336 f32 [C=512] {folded from=[p_layer4_1_bn2_weight, p_layer4_1_bn2_bias,
                                    b_layer4_1_bn2_running_mean,
                                    b_layer4_1_bn2_running_var]} constant]
  nodes:
    group g1 torch.ops.aten.convolution.default:
      n0 {derived}: [t123 f32 [H=224 W=224 C=3] {derived}] =
        permute x=t122 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
    n174 {derived}: [t337 f32 [H=112 W=112 C=64] {derived}] =
      convolution
        x=t123 {derived}
        weight=t297 {folded from=[p_conv1_weight, p_bn1_weight,
                                  b_bn1_running_var]}
        bias=t298 {folded from=[p_bn1_weight, p_bn1_bias, b_bn1_running_mean,
                                b_bn1_running_var]}
        params={stride={h=2; w=2};
               padding={h=3; w=3};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g2 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n6 {pt2=root[1] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t129 f32 [H=64
                                                                      W=112
                                                                      C=112] {pt2=root:getitem}] =
        permute x=t337 {derived} perm=[H<-C, W<-H, C<-W]
    n7 {pt2=root[2] torch.ops.aten.relu.default}: [t130 f32 [H=64 W=112 C=112] {pt2=root:relu}] =
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
      n11 {pt2=root[3] torch.ops.aten.max_pool2d_with_indices.default}: [t134 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_3}] =
        permute x=t132 {derived} perm=[H<-C, W<-H, C<-W]
    group g4 torch.ops.aten.convolution.default:
      n12 {derived}: [t135 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t134 {pt2=root:getitem_3} perm=[H<-W, W<-C, C<-H]
    n175 {derived}: [t338 f32 [H=56 W=56 C=64] {derived}] =
      convolution
        x=t135 {derived}
        weight=t299 {folded from=[p_layer1_0_conv1_weight,
                                  p_layer1_0_bn1_weight,
                                  b_layer1_0_bn1_running_var]}
        bias=t300 {folded from=[p_layer1_0_bn1_weight, p_layer1_0_bn1_bias,
                                b_layer1_0_bn1_running_mean,
                                b_layer1_0_bn1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g5 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n18 {pt2=root[5] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t141 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_5}] =
        permute x=t338 {derived} perm=[H<-C, W<-H, C<-W]
    n19 {pt2=root[6] torch.ops.aten.relu.default}: [t142 f32 [H=64 W=56 C=56] {pt2=root:relu_1}] =
      relu x=t141 {pt2=root:getitem_5}
    group g6 torch.ops.aten.convolution.default:
      n20 {derived}: [t143 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t142 {pt2=root:relu_1} perm=[H<-W, W<-C, C<-H]
    n176 {derived}: [t339 f32 [H=56 W=56 C=64] {derived}] =
      convolution
        x=t143 {derived}
        weight=t301 {folded from=[p_layer1_0_conv2_weight,
                                  p_layer1_0_bn2_weight,
                                  b_layer1_0_bn2_running_var]}
        bias=t302 {folded from=[p_layer1_0_bn2_weight, p_layer1_0_bn2_bias,
                                b_layer1_0_bn2_running_mean,
                                b_layer1_0_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g7 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n26 {pt2=root[8] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t149 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_8}] =
        permute x=t339 {derived} perm=[H<-C, W<-H, C<-W]
    n27 {pt2=root[9] torch.ops.aten.add.Tensor}: [t150 f32 [H=64 W=56 C=56] {pt2=root:add}] =
      add a=t149 {pt2=root:getitem_8} b=t134 {pt2=root:getitem_3}
    n28 {pt2=root[10] torch.ops.aten.relu.default}: [t151 f32 [H=64 W=56 C=56] {pt2=root:relu_2}] =
      relu x=t150 {pt2=root:add}
    group g8 torch.ops.aten.convolution.default:
      n29 {derived}: [t152 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t151 {pt2=root:relu_2} perm=[H<-W, W<-C, C<-H]
    n177 {derived}: [t340 f32 [H=56 W=56 C=64] {derived}] =
      convolution
        x=t152 {derived}
        weight=t303 {folded from=[p_layer1_1_conv1_weight,
                                  p_layer1_1_bn1_weight,
                                  b_layer1_1_bn1_running_var]}
        bias=t304 {folded from=[p_layer1_1_bn1_weight, p_layer1_1_bn1_bias,
                                b_layer1_1_bn1_running_mean,
                                b_layer1_1_bn1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g9 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n35 {pt2=root[12] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t158 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_11}] =
        permute x=t340 {derived} perm=[H<-C, W<-H, C<-W]
    n36 {pt2=root[13] torch.ops.aten.relu.default}: [t159 f32 [H=64 W=56 C=56] {pt2=root:relu_3}] =
      relu x=t158 {pt2=root:getitem_11}
    group g10 torch.ops.aten.convolution.default:
      n37 {derived}: [t160 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t159 {pt2=root:relu_3} perm=[H<-W, W<-C, C<-H]
    n178 {derived}: [t341 f32 [H=56 W=56 C=64] {derived}] =
      convolution
        x=t160 {derived}
        weight=t305 {folded from=[p_layer1_1_conv2_weight,
                                  p_layer1_1_bn2_weight,
                                  b_layer1_1_bn2_running_var]}
        bias=t306 {folded from=[p_layer1_1_bn2_weight, p_layer1_1_bn2_bias,
                                b_layer1_1_bn2_running_mean,
                                b_layer1_1_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g11 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n43 {pt2=root[15] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t166 f32 [H=64
                                                                      W=56
                                                                      C=56] {pt2=root:getitem_14}] =
        permute x=t341 {derived} perm=[H<-C, W<-H, C<-W]
    n44 {pt2=root[16] torch.ops.aten.add.Tensor}: [t167 f32 [H=64 W=56 C=56] {pt2=root:add_1}] =
      add a=t166 {pt2=root:getitem_14} b=t151 {pt2=root:relu_2}
    n45 {pt2=root[17] torch.ops.aten.relu.default}: [t168 f32 [H=64 W=56 C=56] {pt2=root:relu_4}] =
      relu x=t167 {pt2=root:add_1}
    group g12 torch.ops.aten.convolution.default:
      n46 {derived}: [t169 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t168 {pt2=root:relu_4} perm=[H<-W, W<-C, C<-H]
    group g16 torch.ops.aten.convolution.default:
      n61 {derived}: [t184 f32 [H=56 W=56 C=64] {derived}] =
        permute x=t168 {pt2=root:relu_4} perm=[H<-W, W<-C, C<-H]
    n179 {derived}: [t342 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t169 {derived}
        weight=t307 {folded from=[p_layer2_0_conv1_weight,
                                  p_layer2_0_bn1_weight,
                                  b_layer2_0_bn1_running_var]}
        bias=t308 {folded from=[p_layer2_0_bn1_weight, p_layer2_0_bn1_bias,
                                b_layer2_0_bn1_running_mean,
                                b_layer2_0_bn1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n180 {derived}: [t343 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t184 {derived}
        weight=t309 {folded from=[p_layer2_0_downsample_0_weight,
                                  p_layer2_0_downsample_1_weight,
                                  b_layer2_0_downsample_1_running_var]}
        bias=t310 {folded from=[p_layer2_0_downsample_1_weight,
                                p_layer2_0_downsample_1_bias,
                                b_layer2_0_downsample_1_running_mean,
                                b_layer2_0_downsample_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g13 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n52 {pt2=root[19] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t175 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_17}] =
        permute x=t342 {derived} perm=[H<-C, W<-H, C<-W]
    group g17 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n67 {pt2=root[24] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t190 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_23}] =
        permute x=t343 {derived} perm=[H<-C, W<-H, C<-W]
    n53 {pt2=root[20] torch.ops.aten.relu.default}: [t176 f32 [H=128 W=28 C=28] {pt2=root:relu_5}] =
      relu x=t175 {pt2=root:getitem_17}
    group g14 torch.ops.aten.convolution.default:
      n54 {derived}: [t177 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t176 {pt2=root:relu_5} perm=[H<-W, W<-C, C<-H]
    n181 {derived}: [t344 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t177 {derived}
        weight=t311 {folded from=[p_layer2_0_conv2_weight,
                                  p_layer2_0_bn2_weight,
                                  b_layer2_0_bn2_running_var]}
        bias=t312 {folded from=[p_layer2_0_bn2_weight, p_layer2_0_bn2_bias,
                                b_layer2_0_bn2_running_mean,
                                b_layer2_0_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g15 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n60 {pt2=root[22] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t183 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_20}] =
        permute x=t344 {derived} perm=[H<-C, W<-H, C<-W]
    n68 {pt2=root[25] torch.ops.aten.add.Tensor}: [t191 f32 [H=128 W=28 C=28] {pt2=root:add_2}] =
      add a=t183 {pt2=root:getitem_20} b=t190 {pt2=root:getitem_23}
    n69 {pt2=root[26] torch.ops.aten.relu.default}: [t192 f32 [H=128 W=28 C=28] {pt2=root:relu_6}] =
      relu x=t191 {pt2=root:add_2}
    group g18 torch.ops.aten.convolution.default:
      n70 {derived}: [t193 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t192 {pt2=root:relu_6} perm=[H<-W, W<-C, C<-H]
    n182 {derived}: [t345 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t193 {derived}
        weight=t313 {folded from=[p_layer2_1_conv1_weight,
                                  p_layer2_1_bn1_weight,
                                  b_layer2_1_bn1_running_var]}
        bias=t314 {folded from=[p_layer2_1_bn1_weight, p_layer2_1_bn1_bias,
                                b_layer2_1_bn1_running_mean,
                                b_layer2_1_bn1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g19 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n76 {pt2=root[28] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t199 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_26}] =
        permute x=t345 {derived} perm=[H<-C, W<-H, C<-W]
    n77 {pt2=root[29] torch.ops.aten.relu.default}: [t200 f32 [H=128 W=28 C=28] {pt2=root:relu_7}] =
      relu x=t199 {pt2=root:getitem_26}
    group g20 torch.ops.aten.convolution.default:
      n78 {derived}: [t201 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t200 {pt2=root:relu_7} perm=[H<-W, W<-C, C<-H]
    n183 {derived}: [t346 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t201 {derived}
        weight=t315 {folded from=[p_layer2_1_conv2_weight,
                                  p_layer2_1_bn2_weight,
                                  b_layer2_1_bn2_running_var]}
        bias=t316 {folded from=[p_layer2_1_bn2_weight, p_layer2_1_bn2_bias,
                                b_layer2_1_bn2_running_mean,
                                b_layer2_1_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g21 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n84 {pt2=root[31] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t207 f32 [H=128
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_29}] =
        permute x=t346 {derived} perm=[H<-C, W<-H, C<-W]
    n85 {pt2=root[32] torch.ops.aten.add.Tensor}: [t208 f32 [H=128 W=28 C=28] {pt2=root:add_3}] =
      add a=t207 {pt2=root:getitem_29} b=t192 {pt2=root:relu_6}
    n86 {pt2=root[33] torch.ops.aten.relu.default}: [t209 f32 [H=128 W=28 C=28] {pt2=root:relu_8}] =
      relu x=t208 {pt2=root:add_3}
    group g22 torch.ops.aten.convolution.default:
      n87 {derived}: [t210 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t209 {pt2=root:relu_8} perm=[H<-W, W<-C, C<-H]
    group g26 torch.ops.aten.convolution.default:
      n102 {derived}: [t225 f32 [H=28 W=28 C=128] {derived}] =
        permute x=t209 {pt2=root:relu_8} perm=[H<-W, W<-C, C<-H]
    n184 {derived}: [t347 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t210 {derived}
        weight=t317 {folded from=[p_layer3_0_conv1_weight,
                                  p_layer3_0_bn1_weight,
                                  b_layer3_0_bn1_running_var]}
        bias=t318 {folded from=[p_layer3_0_bn1_weight, p_layer3_0_bn1_bias,
                                b_layer3_0_bn1_running_mean,
                                b_layer3_0_bn1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n185 {derived}: [t348 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t225 {derived}
        weight=t319 {folded from=[p_layer3_0_downsample_0_weight,
                                  p_layer3_0_downsample_1_weight,
                                  b_layer3_0_downsample_1_running_var]}
        bias=t320 {folded from=[p_layer3_0_downsample_1_weight,
                                p_layer3_0_downsample_1_bias,
                                b_layer3_0_downsample_1_running_mean,
                                b_layer3_0_downsample_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g23 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n93 {pt2=root[35] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t216 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_32}] =
        permute x=t347 {derived} perm=[H<-C, W<-H, C<-W]
    group g27 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n108 {pt2=root[40] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t231 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_38}] =
        permute x=t348 {derived} perm=[H<-C, W<-H, C<-W]
    n94 {pt2=root[36] torch.ops.aten.relu.default}: [t217 f32 [H=256 W=14 C=14] {pt2=root:relu_9}] =
      relu x=t216 {pt2=root:getitem_32}
    group g24 torch.ops.aten.convolution.default:
      n95 {derived}: [t218 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t217 {pt2=root:relu_9} perm=[H<-W, W<-C, C<-H]
    n186 {derived}: [t349 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t218 {derived}
        weight=t321 {folded from=[p_layer3_0_conv2_weight,
                                  p_layer3_0_bn2_weight,
                                  b_layer3_0_bn2_running_var]}
        bias=t322 {folded from=[p_layer3_0_bn2_weight, p_layer3_0_bn2_bias,
                                b_layer3_0_bn2_running_mean,
                                b_layer3_0_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g25 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n101 {pt2=root[38] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t224 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_35}] =
        permute x=t349 {derived} perm=[H<-C, W<-H, C<-W]
    n109 {pt2=root[41] torch.ops.aten.add.Tensor}: [t232 f32 [H=256 W=14 C=14] {pt2=root:add_4}] =
      add a=t224 {pt2=root:getitem_35} b=t231 {pt2=root:getitem_38}
    n110 {pt2=root[42] torch.ops.aten.relu.default}: [t233 f32 [H=256 W=14
                                                                C=14] {pt2=root:relu_10}] =
      relu x=t232 {pt2=root:add_4}
    group g28 torch.ops.aten.convolution.default:
      n111 {derived}: [t234 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t233 {pt2=root:relu_10} perm=[H<-W, W<-C, C<-H]
    n187 {derived}: [t350 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t234 {derived}
        weight=t323 {folded from=[p_layer3_1_conv1_weight,
                                  p_layer3_1_bn1_weight,
                                  b_layer3_1_bn1_running_var]}
        bias=t324 {folded from=[p_layer3_1_bn1_weight, p_layer3_1_bn1_bias,
                                b_layer3_1_bn1_running_mean,
                                b_layer3_1_bn1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g29 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n117 {pt2=root[44] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t240 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_41}] =
        permute x=t350 {derived} perm=[H<-C, W<-H, C<-W]
    n118 {pt2=root[45] torch.ops.aten.relu.default}: [t241 f32 [H=256 W=14
                                                                C=14] {pt2=root:relu_11}] =
      relu x=t240 {pt2=root:getitem_41}
    group g30 torch.ops.aten.convolution.default:
      n119 {derived}: [t242 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t241 {pt2=root:relu_11} perm=[H<-W, W<-C, C<-H]
    n188 {derived}: [t351 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t242 {derived}
        weight=t325 {folded from=[p_layer3_1_conv2_weight,
                                  p_layer3_1_bn2_weight,
                                  b_layer3_1_bn2_running_var]}
        bias=t326 {folded from=[p_layer3_1_bn2_weight, p_layer3_1_bn2_bias,
                                b_layer3_1_bn2_running_mean,
                                b_layer3_1_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g31 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n125 {pt2=root[47] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t248 f32 [H=256
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_44}] =
        permute x=t351 {derived} perm=[H<-C, W<-H, C<-W]
    n126 {pt2=root[48] torch.ops.aten.add.Tensor}: [t249 f32 [H=256 W=14 C=14] {pt2=root:add_5}] =
      add a=t248 {pt2=root:getitem_44} b=t233 {pt2=root:relu_10}
    n127 {pt2=root[49] torch.ops.aten.relu.default}: [t250 f32 [H=256 W=14
                                                                C=14] {pt2=root:relu_12}] =
      relu x=t249 {pt2=root:add_5}
    group g32 torch.ops.aten.convolution.default:
      n128 {derived}: [t251 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t250 {pt2=root:relu_12} perm=[H<-W, W<-C, C<-H]
    group g36 torch.ops.aten.convolution.default:
      n143 {derived}: [t266 f32 [H=14 W=14 C=256] {derived}] =
        permute x=t250 {pt2=root:relu_12} perm=[H<-W, W<-C, C<-H]
    n189 {derived}: [t352 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t251 {derived}
        weight=t327 {folded from=[p_layer4_0_conv1_weight,
                                  p_layer4_0_bn1_weight,
                                  b_layer4_0_bn1_running_var]}
        bias=t328 {folded from=[p_layer4_0_bn1_weight, p_layer4_0_bn1_bias,
                                b_layer4_0_bn1_running_mean,
                                b_layer4_0_bn1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n190 {derived}: [t353 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t266 {derived}
        weight=t329 {folded from=[p_layer4_0_downsample_0_weight,
                                  p_layer4_0_downsample_1_weight,
                                  b_layer4_0_downsample_1_running_var]}
        bias=t330 {folded from=[p_layer4_0_downsample_1_weight,
                                p_layer4_0_downsample_1_bias,
                                b_layer4_0_downsample_1_running_mean,
                                b_layer4_0_downsample_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g33 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n134 {pt2=root[51] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t257 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_47}] =
        permute x=t352 {derived} perm=[H<-C, W<-H, C<-W]
    group g37 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n149 {pt2=root[56] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t272 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_53}] =
        permute x=t353 {derived} perm=[H<-C, W<-H, C<-W]
    n135 {pt2=root[52] torch.ops.aten.relu.default}: [t258 f32 [H=512 W=7 C=7] {pt2=root:relu_13}] =
      relu x=t257 {pt2=root:getitem_47}
    group g34 torch.ops.aten.convolution.default:
      n136 {derived}: [t259 f32 [H=7 W=7 C=512] {derived}] =
        permute x=t258 {pt2=root:relu_13} perm=[H<-W, W<-C, C<-H]
    n191 {derived}: [t354 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t259 {derived}
        weight=t331 {folded from=[p_layer4_0_conv2_weight,
                                  p_layer4_0_bn2_weight,
                                  b_layer4_0_bn2_running_var]}
        bias=t332 {folded from=[p_layer4_0_bn2_weight, p_layer4_0_bn2_bias,
                                b_layer4_0_bn2_running_mean,
                                b_layer4_0_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g35 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n142 {pt2=root[54] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t265 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_50}] =
        permute x=t354 {derived} perm=[H<-C, W<-H, C<-W]
    n150 {pt2=root[57] torch.ops.aten.add.Tensor}: [t273 f32 [H=512 W=7 C=7] {pt2=root:add_6}] =
      add a=t265 {pt2=root:getitem_50} b=t272 {pt2=root:getitem_53}
    n151 {pt2=root[58] torch.ops.aten.relu.default}: [t274 f32 [H=512 W=7 C=7] {pt2=root:relu_14}] =
      relu x=t273 {pt2=root:add_6}
    group g38 torch.ops.aten.convolution.default:
      n152 {derived}: [t275 f32 [H=7 W=7 C=512] {derived}] =
        permute x=t274 {pt2=root:relu_14} perm=[H<-W, W<-C, C<-H]
    n192 {derived}: [t355 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t275 {derived}
        weight=t333 {folded from=[p_layer4_1_conv1_weight,
                                  p_layer4_1_bn1_weight,
                                  b_layer4_1_bn1_running_var]}
        bias=t334 {folded from=[p_layer4_1_bn1_weight, p_layer4_1_bn1_bias,
                                b_layer4_1_bn1_running_mean,
                                b_layer4_1_bn1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g39 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n158 {pt2=root[60] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t281 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_56}] =
        permute x=t355 {derived} perm=[H<-C, W<-H, C<-W]
    n159 {pt2=root[61] torch.ops.aten.relu.default}: [t282 f32 [H=512 W=7 C=7] {pt2=root:relu_15}] =
      relu x=t281 {pt2=root:getitem_56}
    group g40 torch.ops.aten.convolution.default:
      n160 {derived}: [t283 f32 [H=7 W=7 C=512] {derived}] =
        permute x=t282 {pt2=root:relu_15} perm=[H<-W, W<-C, C<-H]
    n193 {derived}: [t356 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t283 {derived}
        weight=t335 {folded from=[p_layer4_1_conv2_weight,
                                  p_layer4_1_bn2_weight,
                                  b_layer4_1_bn2_running_var]}
        bias=t336 {folded from=[p_layer4_1_bn2_weight, p_layer4_1_bn2_bias,
                                b_layer4_1_bn2_running_mean,
                                b_layer4_1_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    group g41 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n166 {pt2=root[63] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t289 f32 [H=512
                                                                      W=7 C=7] {pt2=root:getitem_59}] =
        permute x=t356 {derived} perm=[H<-C, W<-H, C<-W]
    n167 {pt2=root[64] torch.ops.aten.add.Tensor}: [t290 f32 [H=512 W=7 C=7] {pt2=root:add_7}] =
      add a=t289 {pt2=root:getitem_59} b=t274 {pt2=root:relu_14}
    n168 {pt2=root[65] torch.ops.aten.relu.default}: [t291 f32 [H=512 W=7 C=7] {pt2=root:relu_16}] =
      relu x=t290 {pt2=root:add_7}
    n169 {pt2=root[66] torch.ops.aten.mean.dim}: [t292 f32 [H=512 W=1 C=1] {pt2=root:mean}] =
      mean x=t291 {pt2=root:relu_16} params={dims=[C, W]; keepdim=true}
    n194 {pt2=root[67] torch.ops.aten.view.default}: [t293 f32 [C=512] {pt2=root:view}] =
      permute x=t292 {pt2=root:mean} perm=[H<-W, W<-C, C<-H]
    group g42 torch.ops.aten.addmm.default:
      n173 {pt2=root[69] torch.ops.aten.addmm.default}: [t296 f32 [C=1000] {pt2=root:addmm}] =
        linear
          x=t293 {pt2=root:view}
          weight=t295 {folded from=[p_fc_weight]}
          bias=t61 {pt2=root:p_fc_bias target=fc.bias}
          params={in_features=512}
  outputs: [t296 f32 [C=1000] {pt2=root:addmm}]
