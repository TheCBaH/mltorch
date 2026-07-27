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
  nodes: 174 -> 91
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
      n14 {derived}: [t137 f32 [H=56 W=56 C=64] {derived}] =
        convolution
          x=t132 {derived}
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
      n22 {derived}: [t145 f32 [H=56 W=56 C=64] {derived}] =
        convolution
          x=t298 {derived}
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
      n31 {derived}: [t154 f32 [H=56 W=56 C=64] {derived}] =
        convolution
          x=t300 {derived}
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
      n39 {derived}: [t162 f32 [H=56 W=56 C=64] {derived}] =
        convolution
          x=t301 {derived}
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
      n48 {derived}: [t171 f32 [H=28 W=28 C=128] {derived}] =
        convolution
          x=t303 {derived}
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
      n56 {derived}: [t179 f32 [H=28 W=28 C=128] {derived}] =
        convolution
          x=t304 {derived}
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
      n63 {derived}: [t186 f32 [H=28 W=28 C=128] {derived}] =
        convolution
          x=t303 {derived}
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
      n72 {derived}: [t195 f32 [H=28 W=28 C=128] {derived}] =
        convolution
          x=t306 {derived}
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
      n80 {derived}: [t203 f32 [H=28 W=28 C=128] {derived}] =
        convolution
          x=t307 {derived}
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
      n89 {derived}: [t212 f32 [H=14 W=14 C=256] {derived}] =
        convolution
          x=t309 {derived}
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
      n97 {derived}: [t220 f32 [H=14 W=14 C=256] {derived}] =
        convolution
          x=t310 {derived}
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
      n104 {derived}: [t227 f32 [H=14 W=14 C=256] {derived}] =
        convolution
          x=t309 {derived}
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
      n113 {derived}: [t236 f32 [H=14 W=14 C=256] {derived}] =
        convolution
          x=t312 {derived}
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
      n121 {derived}: [t244 f32 [H=14 W=14 C=256] {derived}] =
        convolution
          x=t313 {derived}
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
      n130 {derived}: [t253 f32 [H=7 W=7 C=512] {derived}] =
        convolution
          x=t315 {derived}
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
      n138 {derived}: [t261 f32 [H=7 W=7 C=512] {derived}] =
        convolution
          x=t316 {derived}
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
      n145 {derived}: [t268 f32 [H=7 W=7 C=512] {derived}] =
        convolution
          x=t315 {derived}
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
      n154 {derived}: [t277 f32 [H=7 W=7 C=512] {derived}] =
        convolution
          x=t318 {derived}
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
      n162 {derived}: [t285 f32 [H=7 W=7 C=512] {derived}] =
        convolution
          x=t319 {derived}
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
    n175 {pt2=root[2] torch.ops.aten.relu.default}: [t297 f32 [H=112 W=112
                                                               C=64] {derived}] =
      relu x=t128 {derived}
    group g3 torch.ops.aten.max_pool2d_with_indices.default:
      n9 {derived}: [t132 f32 [H=56 W=56 C=64] {derived},
                     t133 f32 [H=56 W=56 C=64] {derived}] =
        max_pool2d_with_indices
          x=t297 {derived}
          params={kernel={h=3; w=3}; stride={h=2; w=2}; pad={h=1; w=1}}
      n10 {derived}: [] = discard x=t133 {derived}
    group g5 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n17 {derived}: [t140 f32 [H=56 W=56 C=64] {derived}] =
        batch_norm
          x=t137 {derived}
          weight=t4 {pt2=root:p_layer1_0_bn1_weight target=layer1.0.bn1.weight}
          bias=t5 {pt2=root:p_layer1_0_bn1_bias target=layer1.0.bn1.bias}
          running_mean=t65 {pt2=root:b_layer1_0_bn1_running_mean target=layer1.0.bn1.running_mean}
          running_var=t66 {pt2=root:b_layer1_0_bn1_running_var target=layer1.0.bn1.running_var}
          params={channel=C; eps=1e-05}
    n176 {pt2=root[6] torch.ops.aten.relu.default}: [t298 f32 [H=56 W=56 C=64] {derived}] =
      relu x=t140 {derived}
    group g7 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n25 {derived}: [t148 f32 [H=56 W=56 C=64] {derived}] =
        batch_norm
          x=t145 {derived}
          weight=t7 {pt2=root:p_layer1_0_bn2_weight target=layer1.0.bn2.weight}
          bias=t8 {pt2=root:p_layer1_0_bn2_bias target=layer1.0.bn2.bias}
          running_mean=t68 {pt2=root:b_layer1_0_bn2_running_mean target=layer1.0.bn2.running_mean}
          running_var=t69 {pt2=root:b_layer1_0_bn2_running_var target=layer1.0.bn2.running_var}
          params={channel=C; eps=1e-05}
    n177 {pt2=root[9] torch.ops.aten.add.Tensor}: [t299 f32 [H=56 W=56 C=64] {derived}] =
      add a=t148 {derived} b=t132 {derived}
    n178 {pt2=root[10] torch.ops.aten.relu.default}: [t300 f32 [H=56 W=56 C=64] {derived}] =
      relu x=t299 {derived}
    group g9 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n34 {derived}: [t157 f32 [H=56 W=56 C=64] {derived}] =
        batch_norm
          x=t154 {derived}
          weight=t10 {pt2=root:p_layer1_1_bn1_weight target=layer1.1.bn1.weight}
          bias=t11 {pt2=root:p_layer1_1_bn1_bias target=layer1.1.bn1.bias}
          running_mean=t71 {pt2=root:b_layer1_1_bn1_running_mean target=layer1.1.bn1.running_mean}
          running_var=t72 {pt2=root:b_layer1_1_bn1_running_var target=layer1.1.bn1.running_var}
          params={channel=C; eps=1e-05}
    n179 {pt2=root[13] torch.ops.aten.relu.default}: [t301 f32 [H=56 W=56 C=64] {derived}] =
      relu x=t157 {derived}
    group g11 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n42 {derived}: [t165 f32 [H=56 W=56 C=64] {derived}] =
        batch_norm
          x=t162 {derived}
          weight=t13 {pt2=root:p_layer1_1_bn2_weight target=layer1.1.bn2.weight}
          bias=t14 {pt2=root:p_layer1_1_bn2_bias target=layer1.1.bn2.bias}
          running_mean=t74 {pt2=root:b_layer1_1_bn2_running_mean target=layer1.1.bn2.running_mean}
          running_var=t75 {pt2=root:b_layer1_1_bn2_running_var target=layer1.1.bn2.running_var}
          params={channel=C; eps=1e-05}
    n180 {pt2=root[16] torch.ops.aten.add.Tensor}: [t302 f32 [H=56 W=56 C=64] {derived}] =
      add a=t165 {derived} b=t300 {derived}
    n181 {pt2=root[17] torch.ops.aten.relu.default}: [t303 f32 [H=56 W=56 C=64] {derived}] =
      relu x=t302 {derived}
    group g13 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n51 {derived}: [t174 f32 [H=28 W=28 C=128] {derived}] =
        batch_norm
          x=t171 {derived}
          weight=t16 {pt2=root:p_layer2_0_bn1_weight target=layer2.0.bn1.weight}
          bias=t17 {pt2=root:p_layer2_0_bn1_bias target=layer2.0.bn1.bias}
          running_mean=t77 {pt2=root:b_layer2_0_bn1_running_mean target=layer2.0.bn1.running_mean}
          running_var=t78 {pt2=root:b_layer2_0_bn1_running_var target=layer2.0.bn1.running_var}
          params={channel=C; eps=1e-05}
    group g17 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n66 {derived}: [t189 f32 [H=28 W=28 C=128] {derived}] =
        batch_norm
          x=t186 {derived}
          weight=t22 {pt2=root:p_layer2_0_downsample_1_weight target=layer2.0.downsample.1.weight}
          bias=t23 {pt2=root:p_layer2_0_downsample_1_bias target=layer2.0.downsample.1.bias}
          running_mean=t83 {pt2=root:b_layer2_0_downsample_1_running_mean target=layer2.0.downsample.1.running_mean}
          running_var=t84 {pt2=root:b_layer2_0_downsample_1_running_var target=layer2.0.downsample.1.running_var}
          params={channel=C; eps=1e-05}
    n182 {pt2=root[20] torch.ops.aten.relu.default}: [t304 f32 [H=28 W=28
                                                                C=128] {derived}] =
      relu x=t174 {derived}
    group g15 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n59 {derived}: [t182 f32 [H=28 W=28 C=128] {derived}] =
        batch_norm
          x=t179 {derived}
          weight=t19 {pt2=root:p_layer2_0_bn2_weight target=layer2.0.bn2.weight}
          bias=t20 {pt2=root:p_layer2_0_bn2_bias target=layer2.0.bn2.bias}
          running_mean=t80 {pt2=root:b_layer2_0_bn2_running_mean target=layer2.0.bn2.running_mean}
          running_var=t81 {pt2=root:b_layer2_0_bn2_running_var target=layer2.0.bn2.running_var}
          params={channel=C; eps=1e-05}
    n183 {pt2=root[25] torch.ops.aten.add.Tensor}: [t305 f32 [H=28 W=28 C=128] {derived}] =
      add a=t182 {derived} b=t189 {derived}
    n184 {pt2=root[26] torch.ops.aten.relu.default}: [t306 f32 [H=28 W=28
                                                                C=128] {derived}] =
      relu x=t305 {derived}
    group g19 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n75 {derived}: [t198 f32 [H=28 W=28 C=128] {derived}] =
        batch_norm
          x=t195 {derived}
          weight=t25 {pt2=root:p_layer2_1_bn1_weight target=layer2.1.bn1.weight}
          bias=t26 {pt2=root:p_layer2_1_bn1_bias target=layer2.1.bn1.bias}
          running_mean=t86 {pt2=root:b_layer2_1_bn1_running_mean target=layer2.1.bn1.running_mean}
          running_var=t87 {pt2=root:b_layer2_1_bn1_running_var target=layer2.1.bn1.running_var}
          params={channel=C; eps=1e-05}
    n185 {pt2=root[29] torch.ops.aten.relu.default}: [t307 f32 [H=28 W=28
                                                                C=128] {derived}] =
      relu x=t198 {derived}
    group g21 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n83 {derived}: [t206 f32 [H=28 W=28 C=128] {derived}] =
        batch_norm
          x=t203 {derived}
          weight=t28 {pt2=root:p_layer2_1_bn2_weight target=layer2.1.bn2.weight}
          bias=t29 {pt2=root:p_layer2_1_bn2_bias target=layer2.1.bn2.bias}
          running_mean=t89 {pt2=root:b_layer2_1_bn2_running_mean target=layer2.1.bn2.running_mean}
          running_var=t90 {pt2=root:b_layer2_1_bn2_running_var target=layer2.1.bn2.running_var}
          params={channel=C; eps=1e-05}
    n186 {pt2=root[32] torch.ops.aten.add.Tensor}: [t308 f32 [H=28 W=28 C=128] {derived}] =
      add a=t206 {derived} b=t306 {derived}
    n187 {pt2=root[33] torch.ops.aten.relu.default}: [t309 f32 [H=28 W=28
                                                                C=128] {derived}] =
      relu x=t308 {derived}
    group g23 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n92 {derived}: [t215 f32 [H=14 W=14 C=256] {derived}] =
        batch_norm
          x=t212 {derived}
          weight=t31 {pt2=root:p_layer3_0_bn1_weight target=layer3.0.bn1.weight}
          bias=t32 {pt2=root:p_layer3_0_bn1_bias target=layer3.0.bn1.bias}
          running_mean=t92 {pt2=root:b_layer3_0_bn1_running_mean target=layer3.0.bn1.running_mean}
          running_var=t93 {pt2=root:b_layer3_0_bn1_running_var target=layer3.0.bn1.running_var}
          params={channel=C; eps=1e-05}
    group g27 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n107 {derived}: [t230 f32 [H=14 W=14 C=256] {derived}] =
        batch_norm
          x=t227 {derived}
          weight=t37 {pt2=root:p_layer3_0_downsample_1_weight target=layer3.0.downsample.1.weight}
          bias=t38 {pt2=root:p_layer3_0_downsample_1_bias target=layer3.0.downsample.1.bias}
          running_mean=t98 {pt2=root:b_layer3_0_downsample_1_running_mean target=layer3.0.downsample.1.running_mean}
          running_var=t99 {pt2=root:b_layer3_0_downsample_1_running_var target=layer3.0.downsample.1.running_var}
          params={channel=C; eps=1e-05}
    n188 {pt2=root[36] torch.ops.aten.relu.default}: [t310 f32 [H=14 W=14
                                                                C=256] {derived}] =
      relu x=t215 {derived}
    group g25 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n100 {derived}: [t223 f32 [H=14 W=14 C=256] {derived}] =
        batch_norm
          x=t220 {derived}
          weight=t34 {pt2=root:p_layer3_0_bn2_weight target=layer3.0.bn2.weight}
          bias=t35 {pt2=root:p_layer3_0_bn2_bias target=layer3.0.bn2.bias}
          running_mean=t95 {pt2=root:b_layer3_0_bn2_running_mean target=layer3.0.bn2.running_mean}
          running_var=t96 {pt2=root:b_layer3_0_bn2_running_var target=layer3.0.bn2.running_var}
          params={channel=C; eps=1e-05}
    n189 {pt2=root[41] torch.ops.aten.add.Tensor}: [t311 f32 [H=14 W=14 C=256] {derived}] =
      add a=t223 {derived} b=t230 {derived}
    n190 {pt2=root[42] torch.ops.aten.relu.default}: [t312 f32 [H=14 W=14
                                                                C=256] {derived}] =
      relu x=t311 {derived}
    group g29 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n116 {derived}: [t239 f32 [H=14 W=14 C=256] {derived}] =
        batch_norm
          x=t236 {derived}
          weight=t40 {pt2=root:p_layer3_1_bn1_weight target=layer3.1.bn1.weight}
          bias=t41 {pt2=root:p_layer3_1_bn1_bias target=layer3.1.bn1.bias}
          running_mean=t101 {pt2=root:b_layer3_1_bn1_running_mean target=layer3.1.bn1.running_mean}
          running_var=t102 {pt2=root:b_layer3_1_bn1_running_var target=layer3.1.bn1.running_var}
          params={channel=C; eps=1e-05}
    n191 {pt2=root[45] torch.ops.aten.relu.default}: [t313 f32 [H=14 W=14
                                                                C=256] {derived}] =
      relu x=t239 {derived}
    group g31 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n124 {derived}: [t247 f32 [H=14 W=14 C=256] {derived}] =
        batch_norm
          x=t244 {derived}
          weight=t43 {pt2=root:p_layer3_1_bn2_weight target=layer3.1.bn2.weight}
          bias=t44 {pt2=root:p_layer3_1_bn2_bias target=layer3.1.bn2.bias}
          running_mean=t104 {pt2=root:b_layer3_1_bn2_running_mean target=layer3.1.bn2.running_mean}
          running_var=t105 {pt2=root:b_layer3_1_bn2_running_var target=layer3.1.bn2.running_var}
          params={channel=C; eps=1e-05}
    n192 {pt2=root[48] torch.ops.aten.add.Tensor}: [t314 f32 [H=14 W=14 C=256] {derived}] =
      add a=t247 {derived} b=t312 {derived}
    n193 {pt2=root[49] torch.ops.aten.relu.default}: [t315 f32 [H=14 W=14
                                                                C=256] {derived}] =
      relu x=t314 {derived}
    group g33 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n133 {derived}: [t256 f32 [H=7 W=7 C=512] {derived}] =
        batch_norm
          x=t253 {derived}
          weight=t46 {pt2=root:p_layer4_0_bn1_weight target=layer4.0.bn1.weight}
          bias=t47 {pt2=root:p_layer4_0_bn1_bias target=layer4.0.bn1.bias}
          running_mean=t107 {pt2=root:b_layer4_0_bn1_running_mean target=layer4.0.bn1.running_mean}
          running_var=t108 {pt2=root:b_layer4_0_bn1_running_var target=layer4.0.bn1.running_var}
          params={channel=C; eps=1e-05}
    group g37 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n148 {derived}: [t271 f32 [H=7 W=7 C=512] {derived}] =
        batch_norm
          x=t268 {derived}
          weight=t52 {pt2=root:p_layer4_0_downsample_1_weight target=layer4.0.downsample.1.weight}
          bias=t53 {pt2=root:p_layer4_0_downsample_1_bias target=layer4.0.downsample.1.bias}
          running_mean=t113 {pt2=root:b_layer4_0_downsample_1_running_mean target=layer4.0.downsample.1.running_mean}
          running_var=t114 {pt2=root:b_layer4_0_downsample_1_running_var target=layer4.0.downsample.1.running_var}
          params={channel=C; eps=1e-05}
    n194 {pt2=root[52] torch.ops.aten.relu.default}: [t316 f32 [H=7 W=7 C=512] {derived}] =
      relu x=t256 {derived}
    group g35 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n141 {derived}: [t264 f32 [H=7 W=7 C=512] {derived}] =
        batch_norm
          x=t261 {derived}
          weight=t49 {pt2=root:p_layer4_0_bn2_weight target=layer4.0.bn2.weight}
          bias=t50 {pt2=root:p_layer4_0_bn2_bias target=layer4.0.bn2.bias}
          running_mean=t110 {pt2=root:b_layer4_0_bn2_running_mean target=layer4.0.bn2.running_mean}
          running_var=t111 {pt2=root:b_layer4_0_bn2_running_var target=layer4.0.bn2.running_var}
          params={channel=C; eps=1e-05}
    n195 {pt2=root[57] torch.ops.aten.add.Tensor}: [t317 f32 [H=7 W=7 C=512] {derived}] =
      add a=t264 {derived} b=t271 {derived}
    n196 {pt2=root[58] torch.ops.aten.relu.default}: [t318 f32 [H=7 W=7 C=512] {derived}] =
      relu x=t317 {derived}
    group g39 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n157 {derived}: [t280 f32 [H=7 W=7 C=512] {derived}] =
        batch_norm
          x=t277 {derived}
          weight=t55 {pt2=root:p_layer4_1_bn1_weight target=layer4.1.bn1.weight}
          bias=t56 {pt2=root:p_layer4_1_bn1_bias target=layer4.1.bn1.bias}
          running_mean=t116 {pt2=root:b_layer4_1_bn1_running_mean target=layer4.1.bn1.running_mean}
          running_var=t117 {pt2=root:b_layer4_1_bn1_running_var target=layer4.1.bn1.running_var}
          params={channel=C; eps=1e-05}
    n197 {pt2=root[61] torch.ops.aten.relu.default}: [t319 f32 [H=7 W=7 C=512] {derived}] =
      relu x=t280 {derived}
    group g41 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n165 {derived}: [t288 f32 [H=7 W=7 C=512] {derived}] =
        batch_norm
          x=t285 {derived}
          weight=t58 {pt2=root:p_layer4_1_bn2_weight target=layer4.1.bn2.weight}
          bias=t59 {pt2=root:p_layer4_1_bn2_bias target=layer4.1.bn2.bias}
          running_mean=t119 {pt2=root:b_layer4_1_bn2_running_mean target=layer4.1.bn2.running_mean}
          running_var=t120 {pt2=root:b_layer4_1_bn2_running_var target=layer4.1.bn2.running_var}
          params={channel=C; eps=1e-05}
    n198 {pt2=root[64] torch.ops.aten.add.Tensor}: [t320 f32 [H=7 W=7 C=512] {derived}] =
      add a=t288 {derived} b=t318 {derived}
    n199 {pt2=root[65] torch.ops.aten.relu.default}: [t321 f32 [H=7 W=7 C=512] {derived}] =
      relu x=t320 {derived}
    n200 {pt2=root[66] torch.ops.aten.mean.dim}: [t322 f32 [C=512] {pt2=root:view}] =
      mean x=t321 {derived} params={dims=[W, H]; keepdim=true}
    group g42 torch.ops.aten.addmm.default:
      n173 {pt2=root[69] torch.ops.aten.addmm.default}: [t296 f32 [C=1000] {pt2=root:addmm}] =
        linear
          x=t322 {pt2=root:view}
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
  nodes: 174 -> 50
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
    n175 {pt2=root[2] torch.ops.aten.relu.default}: [t338 f32 [H=112 W=112
                                                               C=64] {derived}] =
      relu x=t337 {derived}
    group g3 torch.ops.aten.max_pool2d_with_indices.default:
      n9 {derived}: [t132 f32 [H=56 W=56 C=64] {derived},
                     t133 f32 [H=56 W=56 C=64] {derived}] =
        max_pool2d_with_indices
          x=t338 {derived}
          params={kernel={h=3; w=3}; stride={h=2; w=2}; pad={h=1; w=1}}
      n10 {derived}: [] = discard x=t133 {derived}
    n176 {derived}: [t339 f32 [H=56 W=56 C=64] {derived}] =
      convolution
        x=t132 {derived}
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
    n177 {pt2=root[6] torch.ops.aten.relu.default}: [t340 f32 [H=56 W=56 C=64] {derived}] =
      relu x=t339 {derived}
    n178 {derived}: [t341 f32 [H=56 W=56 C=64] {derived}] =
      convolution
        x=t340 {derived}
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
    n179 {pt2=root[9] torch.ops.aten.add.Tensor}: [t342 f32 [H=56 W=56 C=64] {derived}] =
      add a=t341 {derived} b=t132 {derived}
    n180 {pt2=root[10] torch.ops.aten.relu.default}: [t343 f32 [H=56 W=56 C=64] {derived}] =
      relu x=t342 {derived}
    n181 {derived}: [t344 f32 [H=56 W=56 C=64] {derived}] =
      convolution
        x=t343 {derived}
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
    n182 {pt2=root[13] torch.ops.aten.relu.default}: [t345 f32 [H=56 W=56 C=64] {derived}] =
      relu x=t344 {derived}
    n183 {derived}: [t346 f32 [H=56 W=56 C=64] {derived}] =
      convolution
        x=t345 {derived}
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
    n184 {pt2=root[16] torch.ops.aten.add.Tensor}: [t347 f32 [H=56 W=56 C=64] {derived}] =
      add a=t346 {derived} b=t343 {derived}
    n185 {pt2=root[17] torch.ops.aten.relu.default}: [t348 f32 [H=56 W=56 C=64] {derived}] =
      relu x=t347 {derived}
    n186 {derived}: [t349 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t348 {derived}
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
    n187 {derived}: [t350 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t348 {derived}
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
    n188 {pt2=root[20] torch.ops.aten.relu.default}: [t351 f32 [H=28 W=28
                                                                C=128] {derived}] =
      relu x=t349 {derived}
    n189 {derived}: [t352 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t351 {derived}
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
    n190 {pt2=root[25] torch.ops.aten.add.Tensor}: [t353 f32 [H=28 W=28 C=128] {derived}] =
      add a=t352 {derived} b=t350 {derived}
    n191 {pt2=root[26] torch.ops.aten.relu.default}: [t354 f32 [H=28 W=28
                                                                C=128] {derived}] =
      relu x=t353 {derived}
    n192 {derived}: [t355 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t354 {derived}
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
    n193 {pt2=root[29] torch.ops.aten.relu.default}: [t356 f32 [H=28 W=28
                                                                C=128] {derived}] =
      relu x=t355 {derived}
    n194 {derived}: [t357 f32 [H=28 W=28 C=128] {derived}] =
      convolution
        x=t356 {derived}
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
    n195 {pt2=root[32] torch.ops.aten.add.Tensor}: [t358 f32 [H=28 W=28 C=128] {derived}] =
      add a=t357 {derived} b=t354 {derived}
    n196 {pt2=root[33] torch.ops.aten.relu.default}: [t359 f32 [H=28 W=28
                                                                C=128] {derived}] =
      relu x=t358 {derived}
    n197 {derived}: [t360 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t359 {derived}
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
    n198 {derived}: [t361 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t359 {derived}
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
    n199 {pt2=root[36] torch.ops.aten.relu.default}: [t362 f32 [H=14 W=14
                                                                C=256] {derived}] =
      relu x=t360 {derived}
    n200 {derived}: [t363 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t362 {derived}
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
    n201 {pt2=root[41] torch.ops.aten.add.Tensor}: [t364 f32 [H=14 W=14 C=256] {derived}] =
      add a=t363 {derived} b=t361 {derived}
    n202 {pt2=root[42] torch.ops.aten.relu.default}: [t365 f32 [H=14 W=14
                                                                C=256] {derived}] =
      relu x=t364 {derived}
    n203 {derived}: [t366 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t365 {derived}
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
    n204 {pt2=root[45] torch.ops.aten.relu.default}: [t367 f32 [H=14 W=14
                                                                C=256] {derived}] =
      relu x=t366 {derived}
    n205 {derived}: [t368 f32 [H=14 W=14 C=256] {derived}] =
      convolution
        x=t367 {derived}
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
    n206 {pt2=root[48] torch.ops.aten.add.Tensor}: [t369 f32 [H=14 W=14 C=256] {derived}] =
      add a=t368 {derived} b=t365 {derived}
    n207 {pt2=root[49] torch.ops.aten.relu.default}: [t370 f32 [H=14 W=14
                                                                C=256] {derived}] =
      relu x=t369 {derived}
    n208 {derived}: [t371 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t370 {derived}
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
    n209 {derived}: [t372 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t370 {derived}
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
    n210 {pt2=root[52] torch.ops.aten.relu.default}: [t373 f32 [H=7 W=7 C=512] {derived}] =
      relu x=t371 {derived}
    n211 {derived}: [t374 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t373 {derived}
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
    n212 {pt2=root[57] torch.ops.aten.add.Tensor}: [t375 f32 [H=7 W=7 C=512] {derived}] =
      add a=t374 {derived} b=t372 {derived}
    n213 {pt2=root[58] torch.ops.aten.relu.default}: [t376 f32 [H=7 W=7 C=512] {derived}] =
      relu x=t375 {derived}
    n214 {derived}: [t377 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t376 {derived}
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
    n215 {pt2=root[61] torch.ops.aten.relu.default}: [t378 f32 [H=7 W=7 C=512] {derived}] =
      relu x=t377 {derived}
    n216 {derived}: [t379 f32 [H=7 W=7 C=512] {derived}] =
      convolution
        x=t378 {derived}
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
    n217 {pt2=root[64] torch.ops.aten.add.Tensor}: [t380 f32 [H=7 W=7 C=512] {derived}] =
      add a=t379 {derived} b=t376 {derived}
    n218 {pt2=root[65] torch.ops.aten.relu.default}: [t381 f32 [H=7 W=7 C=512] {derived}] =
      relu x=t380 {derived}
    n219 {pt2=root[66] torch.ops.aten.mean.dim}: [t382 f32 [C=512] {pt2=root:view}] =
      mean x=t381 {derived} params={dims=[W, H]; keepdim=true}
    group g42 torch.ops.aten.addmm.default:
      n173 {pt2=root[69] torch.ops.aten.addmm.default}: [t296 f32 [C=1000] {pt2=root:addmm}] =
        linear
          x=t382 {pt2=root:view}
          weight=t295 {folded from=[p_fc_weight]}
          bias=t61 {pt2=root:p_fc_bias target=fc.bias}
          params={in_features=512}
  outputs: [t296 f32 [C=1000] {pt2=root:addmm}]

MobileNet-v3-small through the same structural pipeline. The graph is printed in
full, as ResNet-18's is above: the point of these goldens is to be able to read
the resulting layout — where the permutes ended up, which relayouts cancelled,
how the scalar and clamping pointwise ops sit between the convolutions — and a
count per op cannot show any of that.

  $ ../bin/native_graph.exe transform --pt2 "$PT2_DATA/mobilenet_v3_small/mobilenet_v3_small.pt2"
  nodes: 488 -> 370
  constants: 210, of which 0 folded
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
     t145 f32 [C=16] {pt2=root:b_features_1_block_0_1_running_mean target=features.1.block.0.1.running_mean} constant,
     t146 f32 [C=16] {pt2=root:b_features_1_block_0_1_running_var target=features.1.block.0.1.running_var} constant,
     t148 f32 [C=16] {pt2=root:b_features_1_block_2_1_running_mean target=features.1.block.2.1.running_mean} constant,
     t149 f32 [C=16] {pt2=root:b_features_1_block_2_1_running_var target=features.1.block.2.1.running_var} constant,
     t151 f32 [C=72] {pt2=root:b_features_2_block_0_1_running_mean target=features.2.block.0.1.running_mean} constant,
     t152 f32 [C=72] {pt2=root:b_features_2_block_0_1_running_var target=features.2.block.0.1.running_var} constant,
     t154 f32 [C=72] {pt2=root:b_features_2_block_1_1_running_mean target=features.2.block.1.1.running_mean} constant,
     t155 f32 [C=72] {pt2=root:b_features_2_block_1_1_running_var target=features.2.block.1.1.running_var} constant,
     t157 f32 [C=24] {pt2=root:b_features_2_block_2_1_running_mean target=features.2.block.2.1.running_mean} constant,
     t158 f32 [C=24] {pt2=root:b_features_2_block_2_1_running_var target=features.2.block.2.1.running_var} constant,
     t160 f32 [C=88] {pt2=root:b_features_3_block_0_1_running_mean target=features.3.block.0.1.running_mean} constant,
     t161 f32 [C=88] {pt2=root:b_features_3_block_0_1_running_var target=features.3.block.0.1.running_var} constant,
     t163 f32 [C=88] {pt2=root:b_features_3_block_1_1_running_mean target=features.3.block.1.1.running_mean} constant,
     t164 f32 [C=88] {pt2=root:b_features_3_block_1_1_running_var target=features.3.block.1.1.running_var} constant,
     t166 f32 [C=24] {pt2=root:b_features_3_block_2_1_running_mean target=features.3.block.2.1.running_mean} constant,
     t167 f32 [C=24] {pt2=root:b_features_3_block_2_1_running_var target=features.3.block.2.1.running_var} constant,
     t169 f32 [C=96] {pt2=root:b_features_4_block_0_1_running_mean target=features.4.block.0.1.running_mean} constant,
     t170 f32 [C=96] {pt2=root:b_features_4_block_0_1_running_var target=features.4.block.0.1.running_var} constant,
     t172 f32 [C=96] {pt2=root:b_features_4_block_1_1_running_mean target=features.4.block.1.1.running_mean} constant,
     t173 f32 [C=96] {pt2=root:b_features_4_block_1_1_running_var target=features.4.block.1.1.running_var} constant,
     t175 f32 [C=40] {pt2=root:b_features_4_block_3_1_running_mean target=features.4.block.3.1.running_mean} constant,
     t176 f32 [C=40] {pt2=root:b_features_4_block_3_1_running_var target=features.4.block.3.1.running_var} constant,
     t178 f32 [C=240] {pt2=root:b_features_5_block_0_1_running_mean target=features.5.block.0.1.running_mean} constant,
     t179 f32 [C=240] {pt2=root:b_features_5_block_0_1_running_var target=features.5.block.0.1.running_var} constant,
     t181 f32 [C=240] {pt2=root:b_features_5_block_1_1_running_mean target=features.5.block.1.1.running_mean} constant,
     t182 f32 [C=240] {pt2=root:b_features_5_block_1_1_running_var target=features.5.block.1.1.running_var} constant,
     t184 f32 [C=40] {pt2=root:b_features_5_block_3_1_running_mean target=features.5.block.3.1.running_mean} constant,
     t185 f32 [C=40] {pt2=root:b_features_5_block_3_1_running_var target=features.5.block.3.1.running_var} constant,
     t187 f32 [C=240] {pt2=root:b_features_6_block_0_1_running_mean target=features.6.block.0.1.running_mean} constant,
     t188 f32 [C=240] {pt2=root:b_features_6_block_0_1_running_var target=features.6.block.0.1.running_var} constant,
     t190 f32 [C=240] {pt2=root:b_features_6_block_1_1_running_mean target=features.6.block.1.1.running_mean} constant,
     t191 f32 [C=240] {pt2=root:b_features_6_block_1_1_running_var target=features.6.block.1.1.running_var} constant,
     t193 f32 [C=40] {pt2=root:b_features_6_block_3_1_running_mean target=features.6.block.3.1.running_mean} constant,
     t194 f32 [C=40] {pt2=root:b_features_6_block_3_1_running_var target=features.6.block.3.1.running_var} constant,
     t196 f32 [C=120] {pt2=root:b_features_7_block_0_1_running_mean target=features.7.block.0.1.running_mean} constant,
     t197 f32 [C=120] {pt2=root:b_features_7_block_0_1_running_var target=features.7.block.0.1.running_var} constant,
     t199 f32 [C=120] {pt2=root:b_features_7_block_1_1_running_mean target=features.7.block.1.1.running_mean} constant,
     t200 f32 [C=120] {pt2=root:b_features_7_block_1_1_running_var target=features.7.block.1.1.running_var} constant,
     t202 f32 [C=48] {pt2=root:b_features_7_block_3_1_running_mean target=features.7.block.3.1.running_mean} constant,
     t203 f32 [C=48] {pt2=root:b_features_7_block_3_1_running_var target=features.7.block.3.1.running_var} constant,
     t205 f32 [C=144] {pt2=root:b_features_8_block_0_1_running_mean target=features.8.block.0.1.running_mean} constant,
     t206 f32 [C=144] {pt2=root:b_features_8_block_0_1_running_var target=features.8.block.0.1.running_var} constant,
     t208 f32 [C=144] {pt2=root:b_features_8_block_1_1_running_mean target=features.8.block.1.1.running_mean} constant,
     t209 f32 [C=144] {pt2=root:b_features_8_block_1_1_running_var target=features.8.block.1.1.running_var} constant,
     t211 f32 [C=48] {pt2=root:b_features_8_block_3_1_running_mean target=features.8.block.3.1.running_mean} constant,
     t212 f32 [C=48] {pt2=root:b_features_8_block_3_1_running_var target=features.8.block.3.1.running_var} constant,
     t214 f32 [C=288] {pt2=root:b_features_9_block_0_1_running_mean target=features.9.block.0.1.running_mean} constant,
     t215 f32 [C=288] {pt2=root:b_features_9_block_0_1_running_var target=features.9.block.0.1.running_var} constant,
     t217 f32 [C=288] {pt2=root:b_features_9_block_1_1_running_mean target=features.9.block.1.1.running_mean} constant,
     t218 f32 [C=288] {pt2=root:b_features_9_block_1_1_running_var target=features.9.block.1.1.running_var} constant,
     t220 f32 [C=96] {pt2=root:b_features_9_block_3_1_running_mean target=features.9.block.3.1.running_mean} constant,
     t221 f32 [C=96] {pt2=root:b_features_9_block_3_1_running_var target=features.9.block.3.1.running_var} constant,
     t223 f32 [C=576] {pt2=root:b_features_10_block_0_1_running_mean target=features.10.block.0.1.running_mean} constant,
     t224 f32 [C=576] {pt2=root:b_features_10_block_0_1_running_var target=features.10.block.0.1.running_var} constant,
     t226 f32 [C=576] {pt2=root:b_features_10_block_1_1_running_mean target=features.10.block.1.1.running_mean} constant,
     t227 f32 [C=576] {pt2=root:b_features_10_block_1_1_running_var target=features.10.block.1.1.running_var} constant,
     t229 f32 [C=96] {pt2=root:b_features_10_block_3_1_running_mean target=features.10.block.3.1.running_mean} constant,
     t230 f32 [C=96] {pt2=root:b_features_10_block_3_1_running_var target=features.10.block.3.1.running_var} constant,
     t232 f32 [C=576] {pt2=root:b_features_11_block_0_1_running_mean target=features.11.block.0.1.running_mean} constant,
     t233 f32 [C=576] {pt2=root:b_features_11_block_0_1_running_var target=features.11.block.0.1.running_var} constant,
     t235 f32 [C=576] {pt2=root:b_features_11_block_1_1_running_mean target=features.11.block.1.1.running_mean} constant,
     t236 f32 [C=576] {pt2=root:b_features_11_block_1_1_running_var target=features.11.block.1.1.running_var} constant,
     t238 f32 [C=96] {pt2=root:b_features_11_block_3_1_running_mean target=features.11.block.3.1.running_mean} constant,
     t239 f32 [C=96] {pt2=root:b_features_11_block_3_1_running_var target=features.11.block.3.1.running_var} constant,
     t241 f32 [C=576] {pt2=root:b_features_12_1_running_mean target=features.12.1.running_mean} constant,
     t242 f32 [C=576] {pt2=root:b_features_12_1_running_var target=features.12.1.running_var} constant,
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
    group g3 torch.ops.aten.convolution.default:
      n13 {derived}: [t258 f32 [N=16 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t3 {pt2=root:p_features_1_block_0_0_weight target=features.1.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n12 {derived}: [t257 f32 [H=112 W=112 C=16] {derived}] =
        permute x=t256 {pt2=root:div} perm=[H<-W, W<-C, C<-H]
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
    group g5 torch.ops.aten.convolution.default:
      n22 {derived}: [t267 f32 [N=8 T=1 D=1 H=1 W=1 C=16] {derived}] =
        permute
          x=t6 {pt2=root:p_features_1_block_1_fc1_weight target=features.1.block.1.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n21 {derived}: [t266 f32 [C=16] {derived}] =
        permute x=t265 {pt2=root:mean} perm=[H<-W, W<-C, C<-H]
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
    group g6 torch.ops.aten.convolution.default:
      n27 {derived}: [t272 f32 [N=16 T=1 D=1 H=1 W=1 C=8] {derived}] =
        permute
          x=t8 {pt2=root:p_features_1_block_1_fc2_weight target=features.1.block.1.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n28 {derived}: [t273 f32 [C=16] {derived}] =
        convolution
          x=t734 {derived}
          weight=t272 {derived}
          bias=t9 {pt2=root:p_features_1_block_1_fc2_bias target=features.1.block.1.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g7 torch.ops.aten.convolution.default:
      n36 {derived}: [t281 f32 [N=16 T=1 D=1 H=1 W=1 C=16] {derived}] =
        permute
          x=t10 {pt2=root:p_features_1_block_2_0_weight target=features.1.block.2.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n35 {derived}: [t280 f32 [H=56 W=56 C=16] {derived}] =
        permute x=t279 {pt2=root:mul_1} perm=[H<-W, W<-C, C<-H]
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
    group g9 torch.ops.aten.convolution.default:
      n43 {derived}: [t288 f32 [N=72 T=1 D=1 H=1 W=1 C=16] {derived}] =
        permute
          x=t13 {pt2=root:p_features_2_block_0_0_weight target=features.2.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n44 {derived}: [t289 f32 [H=56 W=56 C=72] {derived}] =
        convolution
          x=t285 {derived}
          weight=t288 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g11 torch.ops.aten.convolution.default:
      n51 {derived}: [t296 f32 [N=72 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t16 {pt2=root:p_features_2_block_1_0_weight target=features.2.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n52 {derived}: [t297 f32 [H=28 W=28 C=72] {derived}] =
        convolution
          x=t739 {derived}
          weight=t296 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=72}
    group g13 torch.ops.aten.convolution.default:
      n59 {derived}: [t304 f32 [N=24 T=1 D=1 H=1 W=1 C=72] {derived}] =
        permute
          x=t19 {pt2=root:p_features_2_block_2_0_weight target=features.2.block.2.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n60 {derived}: [t305 f32 [H=28 W=28 C=24] {derived}] =
        convolution
          x=t740 {derived}
          weight=t304 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g15 torch.ops.aten.convolution.default:
      n66 {derived}: [t311 f32 [N=88 T=1 D=1 H=1 W=1 C=24] {derived}] =
        permute
          x=t22 {pt2=root:p_features_3_block_0_0_weight target=features.3.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n67 {derived}: [t312 f32 [H=28 W=28 C=88] {derived}] =
        convolution
          x=t308 {derived}
          weight=t311 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g17 torch.ops.aten.convolution.default:
      n74 {derived}: [t319 f32 [N=88 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t25 {pt2=root:p_features_3_block_1_0_weight target=features.3.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n75 {derived}: [t320 f32 [H=28 W=28 C=88] {derived}] =
        convolution
          x=t741 {derived}
          weight=t319 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=88}
    group g19 torch.ops.aten.convolution.default:
      n82 {derived}: [t327 f32 [N=24 T=1 D=1 H=1 W=1 C=88] {derived}] =
        permute
          x=t28 {pt2=root:p_features_3_block_2_0_weight target=features.3.block.2.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n83 {derived}: [t328 f32 [H=28 W=28 C=24] {derived}] =
        convolution
          x=t742 {derived}
          weight=t327 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g21 torch.ops.aten.convolution.default:
      n90 {derived}: [t335 f32 [N=96 T=1 D=1 H=1 W=1 C=24] {derived}] =
        permute
          x=t31 {pt2=root:p_features_4_block_0_0_weight target=features.4.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n91 {derived}: [t336 f32 [H=28 W=28 C=96] {derived}] =
        convolution
          x=t743 {derived}
          weight=t335 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g23 torch.ops.aten.convolution.default:
      n102 {derived}: [t347 f32 [N=96 T=1 D=1 H=5 W=5 C=1] {derived}] =
        permute
          x=t34 {pt2=root:p_features_4_block_1_0_weight target=features.4.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n101 {derived}: [t346 f32 [H=28 W=28 C=96] {derived}] =
        permute x=t345 {pt2=root:div_2} perm=[H<-W, W<-C, C<-H]
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
    group g25 torch.ops.aten.convolution.default:
      n115 {derived}: [t360 f32 [N=24 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t37 {pt2=root:p_features_4_block_2_fc1_weight target=features.4.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n114 {derived}: [t359 f32 [C=96] {derived}] =
        permute x=t358 {pt2=root:mean_1} perm=[H<-W, W<-C, C<-H]
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
    group g26 torch.ops.aten.convolution.default:
      n120 {derived}: [t365 f32 [N=96 T=1 D=1 H=1 W=1 C=24] {derived}] =
        permute
          x=t39 {pt2=root:p_features_4_block_2_fc2_weight target=features.4.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n121 {derived}: [t366 f32 [C=96] {derived}] =
        convolution
          x=t744 {derived}
          weight=t365 {derived}
          bias=t40 {pt2=root:p_features_4_block_2_fc2_bias target=features.4.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g27 torch.ops.aten.convolution.default:
      n129 {derived}: [t374 f32 [N=40 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t41 {pt2=root:p_features_4_block_3_0_weight target=features.4.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n128 {derived}: [t373 f32 [H=14 W=14 C=96] {derived}] =
        permute x=t372 {pt2=root:mul_4} perm=[H<-W, W<-C, C<-H]
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
    group g29 torch.ops.aten.convolution.default:
      n136 {derived}: [t381 f32 [N=240 T=1 D=1 H=1 W=1 C=40] {derived}] =
        permute
          x=t44 {pt2=root:p_features_5_block_0_0_weight target=features.5.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n137 {derived}: [t382 f32 [H=14 W=14 C=240] {derived}] =
        convolution
          x=t378 {derived}
          weight=t381 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g31 torch.ops.aten.convolution.default:
      n148 {derived}: [t393 f32 [N=240 T=1 D=1 H=5 W=5 C=1] {derived}] =
        permute
          x=t47 {pt2=root:p_features_5_block_1_0_weight target=features.5.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n147 {derived}: [t392 f32 [H=14 W=14 C=240] {derived}] =
        permute x=t391 {pt2=root:div_5} perm=[H<-W, W<-C, C<-H]
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
    group g33 torch.ops.aten.convolution.default:
      n161 {derived}: [t406 f32 [N=64 T=1 D=1 H=1 W=1 C=240] {derived}] =
        permute
          x=t50 {pt2=root:p_features_5_block_2_fc1_weight target=features.5.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n160 {derived}: [t405 f32 [C=240] {derived}] =
        permute x=t404 {pt2=root:mean_2} perm=[H<-W, W<-C, C<-H]
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
    group g34 torch.ops.aten.convolution.default:
      n166 {derived}: [t411 f32 [N=240 T=1 D=1 H=1 W=1 C=64] {derived}] =
        permute
          x=t52 {pt2=root:p_features_5_block_2_fc2_weight target=features.5.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n167 {derived}: [t412 f32 [C=240] {derived}] =
        convolution
          x=t749 {derived}
          weight=t411 {derived}
          bias=t53 {pt2=root:p_features_5_block_2_fc2_bias target=features.5.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g35 torch.ops.aten.convolution.default:
      n175 {derived}: [t420 f32 [N=40 T=1 D=1 H=1 W=1 C=240] {derived}] =
        permute
          x=t54 {pt2=root:p_features_5_block_3_0_weight target=features.5.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n174 {derived}: [t419 f32 [H=14 W=14 C=240] {derived}] =
        permute x=t418 {pt2=root:mul_7} perm=[H<-W, W<-C, C<-H]
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
    group g37 torch.ops.aten.convolution.default:
      n183 {derived}: [t428 f32 [N=240 T=1 D=1 H=1 W=1 C=40] {derived}] =
        permute
          x=t57 {pt2=root:p_features_6_block_0_0_weight target=features.6.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n184 {derived}: [t429 f32 [H=14 W=14 C=240] {derived}] =
        convolution
          x=t754 {derived}
          weight=t428 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g39 torch.ops.aten.convolution.default:
      n195 {derived}: [t440 f32 [N=240 T=1 D=1 H=5 W=5 C=1] {derived}] =
        permute
          x=t60 {pt2=root:p_features_6_block_1_0_weight target=features.6.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n194 {derived}: [t439 f32 [H=14 W=14 C=240] {derived}] =
        permute x=t438 {pt2=root:div_8} perm=[H<-W, W<-C, C<-H]
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
    group g41 torch.ops.aten.convolution.default:
      n208 {derived}: [t453 f32 [N=64 T=1 D=1 H=1 W=1 C=240] {derived}] =
        permute
          x=t63 {pt2=root:p_features_6_block_2_fc1_weight target=features.6.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n207 {derived}: [t452 f32 [C=240] {derived}] =
        permute x=t451 {pt2=root:mean_3} perm=[H<-W, W<-C, C<-H]
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
    group g42 torch.ops.aten.convolution.default:
      n213 {derived}: [t458 f32 [N=240 T=1 D=1 H=1 W=1 C=64] {derived}] =
        permute
          x=t65 {pt2=root:p_features_6_block_2_fc2_weight target=features.6.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n214 {derived}: [t459 f32 [C=240] {derived}] =
        convolution
          x=t755 {derived}
          weight=t458 {derived}
          bias=t66 {pt2=root:p_features_6_block_2_fc2_bias target=features.6.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g43 torch.ops.aten.convolution.default:
      n222 {derived}: [t467 f32 [N=40 T=1 D=1 H=1 W=1 C=240] {derived}] =
        permute
          x=t67 {pt2=root:p_features_6_block_3_0_weight target=features.6.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n221 {derived}: [t466 f32 [H=14 W=14 C=240] {derived}] =
        permute x=t465 {pt2=root:mul_10} perm=[H<-W, W<-C, C<-H]
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
    group g45 torch.ops.aten.convolution.default:
      n230 {derived}: [t475 f32 [N=120 T=1 D=1 H=1 W=1 C=40] {derived}] =
        permute
          x=t70 {pt2=root:p_features_7_block_0_0_weight target=features.7.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n231 {derived}: [t476 f32 [H=14 W=14 C=120] {derived}] =
        convolution
          x=t760 {derived}
          weight=t475 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g47 torch.ops.aten.convolution.default:
      n242 {derived}: [t487 f32 [N=120 T=1 D=1 H=5 W=5 C=1] {derived}] =
        permute
          x=t73 {pt2=root:p_features_7_block_1_0_weight target=features.7.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n241 {derived}: [t486 f32 [H=14 W=14 C=120] {derived}] =
        permute x=t485 {pt2=root:div_11} perm=[H<-W, W<-C, C<-H]
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
    group g49 torch.ops.aten.convolution.default:
      n255 {derived}: [t500 f32 [N=32 T=1 D=1 H=1 W=1 C=120] {derived}] =
        permute
          x=t76 {pt2=root:p_features_7_block_2_fc1_weight target=features.7.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n254 {derived}: [t499 f32 [C=120] {derived}] =
        permute x=t498 {pt2=root:mean_4} perm=[H<-W, W<-C, C<-H]
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
    group g50 torch.ops.aten.convolution.default:
      n260 {derived}: [t505 f32 [N=120 T=1 D=1 H=1 W=1 C=32] {derived}] =
        permute
          x=t78 {pt2=root:p_features_7_block_2_fc2_weight target=features.7.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n261 {derived}: [t506 f32 [C=120] {derived}] =
        convolution
          x=t761 {derived}
          weight=t505 {derived}
          bias=t79 {pt2=root:p_features_7_block_2_fc2_bias target=features.7.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g51 torch.ops.aten.convolution.default:
      n269 {derived}: [t514 f32 [N=48 T=1 D=1 H=1 W=1 C=120] {derived}] =
        permute
          x=t80 {pt2=root:p_features_7_block_3_0_weight target=features.7.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n268 {derived}: [t513 f32 [H=14 W=14 C=120] {derived}] =
        permute x=t512 {pt2=root:mul_13} perm=[H<-W, W<-C, C<-H]
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
    group g53 torch.ops.aten.convolution.default:
      n276 {derived}: [t521 f32 [N=144 T=1 D=1 H=1 W=1 C=48] {derived}] =
        permute
          x=t83 {pt2=root:p_features_8_block_0_0_weight target=features.8.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n277 {derived}: [t522 f32 [H=14 W=14 C=144] {derived}] =
        convolution
          x=t518 {derived}
          weight=t521 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g55 torch.ops.aten.convolution.default:
      n288 {derived}: [t533 f32 [N=144 T=1 D=1 H=5 W=5 C=1] {derived}] =
        permute
          x=t86 {pt2=root:p_features_8_block_1_0_weight target=features.8.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n287 {derived}: [t532 f32 [H=14 W=14 C=144] {derived}] =
        permute x=t531 {pt2=root:div_14} perm=[H<-W, W<-C, C<-H]
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
    group g57 torch.ops.aten.convolution.default:
      n301 {derived}: [t546 f32 [N=40 T=1 D=1 H=1 W=1 C=144] {derived}] =
        permute
          x=t89 {pt2=root:p_features_8_block_2_fc1_weight target=features.8.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n300 {derived}: [t545 f32 [C=144] {derived}] =
        permute x=t544 {pt2=root:mean_5} perm=[H<-W, W<-C, C<-H]
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
    group g58 torch.ops.aten.convolution.default:
      n306 {derived}: [t551 f32 [N=144 T=1 D=1 H=1 W=1 C=40] {derived}] =
        permute
          x=t91 {pt2=root:p_features_8_block_2_fc2_weight target=features.8.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n307 {derived}: [t552 f32 [C=144] {derived}] =
        convolution
          x=t766 {derived}
          weight=t551 {derived}
          bias=t92 {pt2=root:p_features_8_block_2_fc2_bias target=features.8.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g59 torch.ops.aten.convolution.default:
      n315 {derived}: [t560 f32 [N=48 T=1 D=1 H=1 W=1 C=144] {derived}] =
        permute
          x=t93 {pt2=root:p_features_8_block_3_0_weight target=features.8.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n314 {derived}: [t559 f32 [H=14 W=14 C=144] {derived}] =
        permute x=t558 {pt2=root:mul_16} perm=[H<-W, W<-C, C<-H]
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
    group g61 torch.ops.aten.convolution.default:
      n323 {derived}: [t568 f32 [N=288 T=1 D=1 H=1 W=1 C=48] {derived}] =
        permute
          x=t96 {pt2=root:p_features_9_block_0_0_weight target=features.9.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n324 {derived}: [t569 f32 [H=14 W=14 C=288] {derived}] =
        convolution
          x=t771 {derived}
          weight=t568 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g63 torch.ops.aten.convolution.default:
      n335 {derived}: [t580 f32 [N=288 T=1 D=1 H=5 W=5 C=1] {derived}] =
        permute
          x=t99 {pt2=root:p_features_9_block_1_0_weight target=features.9.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n334 {derived}: [t579 f32 [H=14 W=14 C=288] {derived}] =
        permute x=t578 {pt2=root:div_17} perm=[H<-W, W<-C, C<-H]
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
    group g65 torch.ops.aten.convolution.default:
      n348 {derived}: [t593 f32 [N=72 T=1 D=1 H=1 W=1 C=288] {derived}] =
        permute
          x=t102 {pt2=root:p_features_9_block_2_fc1_weight target=features.9.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n347 {derived}: [t592 f32 [C=288] {derived}] =
        permute x=t591 {pt2=root:mean_6} perm=[H<-W, W<-C, C<-H]
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
    group g66 torch.ops.aten.convolution.default:
      n353 {derived}: [t598 f32 [N=288 T=1 D=1 H=1 W=1 C=72] {derived}] =
        permute
          x=t104 {pt2=root:p_features_9_block_2_fc2_weight target=features.9.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n354 {derived}: [t599 f32 [C=288] {derived}] =
        convolution
          x=t772 {derived}
          weight=t598 {derived}
          bias=t105 {pt2=root:p_features_9_block_2_fc2_bias target=features.9.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g67 torch.ops.aten.convolution.default:
      n362 {derived}: [t607 f32 [N=96 T=1 D=1 H=1 W=1 C=288] {derived}] =
        permute
          x=t106 {pt2=root:p_features_9_block_3_0_weight target=features.9.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n361 {derived}: [t606 f32 [H=7 W=7 C=288] {derived}] =
        permute x=t605 {pt2=root:mul_19} perm=[H<-W, W<-C, C<-H]
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
    group g69 torch.ops.aten.convolution.default:
      n369 {derived}: [t614 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t109 {pt2=root:p_features_10_block_0_0_weight target=features.10.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n370 {derived}: [t615 f32 [H=7 W=7 C=576] {derived}] =
        convolution
          x=t611 {derived}
          weight=t614 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g71 torch.ops.aten.convolution.default:
      n381 {derived}: [t626 f32 [N=576 T=1 D=1 H=5 W=5 C=1] {derived}] =
        permute
          x=t112 {pt2=root:p_features_10_block_1_0_weight target=features.10.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n380 {derived}: [t625 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t624 {pt2=root:div_20} perm=[H<-W, W<-C, C<-H]
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
    group g73 torch.ops.aten.convolution.default:
      n394 {derived}: [t639 f32 [N=144 T=1 D=1 H=1 W=1 C=576] {derived}] =
        permute
          x=t115 {pt2=root:p_features_10_block_2_fc1_weight target=features.10.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n393 {derived}: [t638 f32 [C=576] {derived}] =
        permute x=t637 {pt2=root:mean_7} perm=[H<-W, W<-C, C<-H]
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
    group g74 torch.ops.aten.convolution.default:
      n399 {derived}: [t644 f32 [N=576 T=1 D=1 H=1 W=1 C=144] {derived}] =
        permute
          x=t117 {pt2=root:p_features_10_block_2_fc2_weight target=features.10.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n400 {derived}: [t645 f32 [C=576] {derived}] =
        convolution
          x=t777 {derived}
          weight=t644 {derived}
          bias=t118 {pt2=root:p_features_10_block_2_fc2_bias target=features.10.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g75 torch.ops.aten.convolution.default:
      n408 {derived}: [t653 f32 [N=96 T=1 D=1 H=1 W=1 C=576] {derived}] =
        permute
          x=t119 {pt2=root:p_features_10_block_3_0_weight target=features.10.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n407 {derived}: [t652 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t651 {pt2=root:mul_22} perm=[H<-W, W<-C, C<-H]
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
    group g77 torch.ops.aten.convolution.default:
      n416 {derived}: [t661 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t122 {pt2=root:p_features_11_block_0_0_weight target=features.11.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n417 {derived}: [t662 f32 [H=7 W=7 C=576] {derived}] =
        convolution
          x=t782 {derived}
          weight=t661 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g79 torch.ops.aten.convolution.default:
      n428 {derived}: [t673 f32 [N=576 T=1 D=1 H=5 W=5 C=1] {derived}] =
        permute
          x=t125 {pt2=root:p_features_11_block_1_0_weight target=features.11.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n427 {derived}: [t672 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t671 {pt2=root:div_23} perm=[H<-W, W<-C, C<-H]
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
    group g81 torch.ops.aten.convolution.default:
      n441 {derived}: [t686 f32 [N=144 T=1 D=1 H=1 W=1 C=576] {derived}] =
        permute
          x=t128 {pt2=root:p_features_11_block_2_fc1_weight target=features.11.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n440 {derived}: [t685 f32 [C=576] {derived}] =
        permute x=t684 {pt2=root:mean_8} perm=[H<-W, W<-C, C<-H]
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
    group g82 torch.ops.aten.convolution.default:
      n446 {derived}: [t691 f32 [N=576 T=1 D=1 H=1 W=1 C=144] {derived}] =
        permute
          x=t130 {pt2=root:p_features_11_block_2_fc2_weight target=features.11.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n447 {derived}: [t692 f32 [C=576] {derived}] =
        convolution
          x=t783 {derived}
          weight=t691 {derived}
          bias=t131 {pt2=root:p_features_11_block_2_fc2_bias target=features.11.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g83 torch.ops.aten.convolution.default:
      n455 {derived}: [t700 f32 [N=96 T=1 D=1 H=1 W=1 C=576] {derived}] =
        permute
          x=t132 {pt2=root:p_features_11_block_3_0_weight target=features.11.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n454 {derived}: [t699 f32 [H=7 W=7 C=576] {derived}] =
        permute x=t698 {pt2=root:mul_25} perm=[H<-W, W<-C, C<-H]
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
    group g85 torch.ops.aten.convolution.default:
      n463 {derived}: [t708 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t135 {pt2=root:p_features_12_0_weight target=features.12.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n464 {derived}: [t709 f32 [H=7 W=7 C=576] {derived}] =
        convolution
          x=t788 {derived}
          weight=t708 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n488 {pt2=root[252] torch.ops.aten.permute.default}: [t722 f32 [N=1024 T=1
                                                                    D=1 H=1 W=1
                                                                    C=576] {derived}] =
      permute
        x=t138 {pt2=root:p_classifier_0_weight target=classifier.0.weight}
        perm=[N<-W, W<-N]
    n489 {pt2=root[260] torch.ops.aten.permute.default}: [t731 f32 [N=1000 T=1
                                                                    D=1 H=1 W=1
                                                                    C=1024] {derived}] =
      permute
        x=t140 {pt2=root:p_classifier_3_weight target=classifier.3.weight}
        perm=[N<-W, W<-N]
    group g2 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n5 {derived}: [t250 f32 [H=112 W=112 C=16] {derived}] =
        batch_norm
          x=t247 {derived}
          weight=t1 {pt2=root:p_features_0_1_weight target=features.0.1.weight}
          bias=t2 {pt2=root:p_features_0_1_bias target=features.0.1.bias}
          running_mean=t142 {pt2=root:b_features_0_1_running_mean target=features.0.1.running_mean}
          running_var=t143 {pt2=root:b_features_0_1_running_var target=features.0.1.running_var}
          params={channel=C; eps=0.001}
      n6 {pt2=root[1] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t251 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:getitem}] =
        permute x=t250 {derived} perm=[H<-C, W<-H, C<-W]
    n7 {pt2=root[2] torch.ops.aten.add.Tensor}: [t252 f32 [H=16 W=112 C=112] {pt2=root:add}] =
      add_scalar x=t251 {pt2=root:getitem} scalar=3
    n8 {pt2=root[3] torch.ops.aten.clamp.default}: [t253 f32 [H=16 W=112 C=112] {pt2=root:clamp}] =
      clamp x=t252 {pt2=root:add} params={min=0; max=none}
    n9 {pt2=root[4] torch.ops.aten.clamp.default}: [t254 f32 [H=16 W=112 C=112] {pt2=root:clamp_1}] =
      clamp x=t253 {pt2=root:clamp} params={min=none; max=6}
    n10 {pt2=root[5] torch.ops.aten.mul.Tensor}: [t255 f32 [H=16 W=112 C=112] {pt2=root:mul}] =
      mul a=t251 {pt2=root:getitem} b=t254 {pt2=root:clamp_1}
    n11 {pt2=root[6] torch.ops.aten.div.Tensor}: [t256 f32 [H=16 W=112 C=112] {pt2=root:div}] =
      div_scalar x=t255 {pt2=root:mul} scalar=6
    group g4 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n17 {derived}: [t262 f32 [H=56 W=56 C=16] {derived}] =
        batch_norm
          x=t259 {derived}
          weight=t4 {pt2=root:p_features_1_block_0_1_weight target=features.1.block.0.1.weight}
          bias=t5 {pt2=root:p_features_1_block_0_1_bias target=features.1.block.0.1.bias}
          running_mean=t145 {pt2=root:b_features_1_block_0_1_running_mean target=features.1.block.0.1.running_mean}
          running_var=t146 {pt2=root:b_features_1_block_0_1_running_var target=features.1.block.0.1.running_var}
          params={channel=C; eps=0.001}
    n490 {pt2=root[9] torch.ops.aten.relu.default}: [t733 f32 [H=56 W=56 C=16] {derived}] =
      relu x=t262 {derived}
    n491 {pt2=root[8] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t264 f32 [H=16
                                                                      W=56
                                                                      C=56] {pt2=root:relu}] =
      permute x=t733 {derived} perm=[H<-C, W<-H, C<-W]
    n20 {pt2=root[10] torch.ops.aten.mean.dim}: [t265 f32 [H=16 W=1 C=1] {pt2=root:mean}] =
      mean x=t264 {pt2=root:relu} params={dims=[C, W]; keepdim=true}
    n492 {pt2=root[12] torch.ops.aten.relu.default}: [t734 f32 [C=8] {derived}] =
      relu x=t268 {derived}
    n493 {pt2=root[14] torch.ops.aten.add.Tensor}: [t735 f32 [C=16] {derived}] =
      add_scalar x=t273 {derived} scalar=3
    n494 {pt2=root[15] torch.ops.aten.clamp.default}: [t736 f32 [C=16] {derived}] =
      clamp x=t735 {derived} params={min=0; max=none}
    n495 {pt2=root[16] torch.ops.aten.clamp.default}: [t737 f32 [C=16] {derived}] =
      clamp x=t736 {derived} params={min=none; max=6}
    n496 {pt2=root[17] torch.ops.aten.div.Tensor}: [t738 f32 [C=16] {derived}] =
      div_scalar x=t737 {derived} scalar=6
    n497 {pt2=root[13] torch.ops.aten.convolution.default}: [t278 f32 [H=16 W=1
                                                                      C=1] {pt2=root:div_1}] =
      permute x=t738 {derived} perm=[H<-C, W<-H, C<-W]
    n34 {pt2=root[18] torch.ops.aten.mul.Tensor}: [t279 f32 [H=16 W=56 C=56] {pt2=root:mul_1}] =
      mul a=t278 {pt2=root:div_1} b=t264 {pt2=root:relu}
    group g8 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n40 {derived}: [t285 f32 [H=56 W=56 C=16] {derived}] =
        batch_norm
          x=t282 {derived}
          weight=t11 {pt2=root:p_features_1_block_2_1_weight target=features.1.block.2.1.weight}
          bias=t12 {pt2=root:p_features_1_block_2_1_bias target=features.1.block.2.1.bias}
          running_mean=t148 {pt2=root:b_features_1_block_2_1_running_mean target=features.1.block.2.1.running_mean}
          running_var=t149 {pt2=root:b_features_1_block_2_1_running_var target=features.1.block.2.1.running_var}
          params={channel=C; eps=0.001}
    group g10 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n47 {derived}: [t292 f32 [H=56 W=56 C=72] {derived}] =
        batch_norm
          x=t289 {derived}
          weight=t14 {pt2=root:p_features_2_block_0_1_weight target=features.2.block.0.1.weight}
          bias=t15 {pt2=root:p_features_2_block_0_1_bias target=features.2.block.0.1.bias}
          running_mean=t151 {pt2=root:b_features_2_block_0_1_running_mean target=features.2.block.0.1.running_mean}
          running_var=t152 {pt2=root:b_features_2_block_0_1_running_var target=features.2.block.0.1.running_var}
          params={channel=C; eps=0.001}
    n498 {pt2=root[23] torch.ops.aten.relu.default}: [t739 f32 [H=56 W=56 C=72] {derived}] =
      relu x=t292 {derived}
    group g12 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n55 {derived}: [t300 f32 [H=28 W=28 C=72] {derived}] =
        batch_norm
          x=t297 {derived}
          weight=t17 {pt2=root:p_features_2_block_1_1_weight target=features.2.block.1.1.weight}
          bias=t18 {pt2=root:p_features_2_block_1_1_bias target=features.2.block.1.1.bias}
          running_mean=t154 {pt2=root:b_features_2_block_1_1_running_mean target=features.2.block.1.1.running_mean}
          running_var=t155 {pt2=root:b_features_2_block_1_1_running_var target=features.2.block.1.1.running_var}
          params={channel=C; eps=0.001}
    n499 {pt2=root[26] torch.ops.aten.relu.default}: [t740 f32 [H=28 W=28 C=72] {derived}] =
      relu x=t300 {derived}
    group g14 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n63 {derived}: [t308 f32 [H=28 W=28 C=24] {derived}] =
        batch_norm
          x=t305 {derived}
          weight=t20 {pt2=root:p_features_2_block_2_1_weight target=features.2.block.2.1.weight}
          bias=t21 {pt2=root:p_features_2_block_2_1_bias target=features.2.block.2.1.bias}
          running_mean=t157 {pt2=root:b_features_2_block_2_1_running_mean target=features.2.block.2.1.running_mean}
          running_var=t158 {pt2=root:b_features_2_block_2_1_running_var target=features.2.block.2.1.running_var}
          params={channel=C; eps=0.001}
    group g16 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n70 {derived}: [t315 f32 [H=28 W=28 C=88] {derived}] =
        batch_norm
          x=t312 {derived}
          weight=t23 {pt2=root:p_features_3_block_0_1_weight target=features.3.block.0.1.weight}
          bias=t24 {pt2=root:p_features_3_block_0_1_bias target=features.3.block.0.1.bias}
          running_mean=t160 {pt2=root:b_features_3_block_0_1_running_mean target=features.3.block.0.1.running_mean}
          running_var=t161 {pt2=root:b_features_3_block_0_1_running_var target=features.3.block.0.1.running_var}
          params={channel=C; eps=0.001}
    n500 {pt2=root[31] torch.ops.aten.relu.default}: [t741 f32 [H=28 W=28 C=88] {derived}] =
      relu x=t315 {derived}
    group g18 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n78 {derived}: [t323 f32 [H=28 W=28 C=88] {derived}] =
        batch_norm
          x=t320 {derived}
          weight=t26 {pt2=root:p_features_3_block_1_1_weight target=features.3.block.1.1.weight}
          bias=t27 {pt2=root:p_features_3_block_1_1_bias target=features.3.block.1.1.bias}
          running_mean=t163 {pt2=root:b_features_3_block_1_1_running_mean target=features.3.block.1.1.running_mean}
          running_var=t164 {pt2=root:b_features_3_block_1_1_running_var target=features.3.block.1.1.running_var}
          params={channel=C; eps=0.001}
    n501 {pt2=root[34] torch.ops.aten.relu.default}: [t742 f32 [H=28 W=28 C=88] {derived}] =
      relu x=t323 {derived}
    group g20 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n86 {derived}: [t331 f32 [H=28 W=28 C=24] {derived}] =
        batch_norm
          x=t328 {derived}
          weight=t29 {pt2=root:p_features_3_block_2_1_weight target=features.3.block.2.1.weight}
          bias=t30 {pt2=root:p_features_3_block_2_1_bias target=features.3.block.2.1.bias}
          running_mean=t166 {pt2=root:b_features_3_block_2_1_running_mean target=features.3.block.2.1.running_mean}
          running_var=t167 {pt2=root:b_features_3_block_2_1_running_var target=features.3.block.2.1.running_var}
          params={channel=C; eps=0.001}
    n502 {pt2=root[37] torch.ops.aten.add.Tensor}: [t743 f32 [H=28 W=28 C=24] {derived}] =
      add a=t331 {derived} b=t308 {derived}
    group g22 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n94 {derived}: [t339 f32 [H=28 W=28 C=96] {derived}] =
        batch_norm
          x=t336 {derived}
          weight=t32 {pt2=root:p_features_4_block_0_1_weight target=features.4.block.0.1.weight}
          bias=t33 {pt2=root:p_features_4_block_0_1_bias target=features.4.block.0.1.bias}
          running_mean=t169 {pt2=root:b_features_4_block_0_1_running_mean target=features.4.block.0.1.running_mean}
          running_var=t170 {pt2=root:b_features_4_block_0_1_running_var target=features.4.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n95 {pt2=root[39] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t340 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_27}] =
        permute x=t339 {derived} perm=[H<-C, W<-H, C<-W]
    n96 {pt2=root[40] torch.ops.aten.add.Tensor}: [t341 f32 [H=96 W=28 C=28] {pt2=root:add_3}] =
      add_scalar x=t340 {pt2=root:getitem_27} scalar=3
    n97 {pt2=root[41] torch.ops.aten.clamp.default}: [t342 f32 [H=96 W=28 C=28] {pt2=root:clamp_4}] =
      clamp x=t341 {pt2=root:add_3} params={min=0; max=none}
    n98 {pt2=root[42] torch.ops.aten.clamp.default}: [t343 f32 [H=96 W=28 C=28] {pt2=root:clamp_5}] =
      clamp x=t342 {pt2=root:clamp_4} params={min=none; max=6}
    n99 {pt2=root[43] torch.ops.aten.mul.Tensor}: [t344 f32 [H=96 W=28 C=28] {pt2=root:mul_2}] =
      mul a=t340 {pt2=root:getitem_27} b=t343 {pt2=root:clamp_5}
    n100 {pt2=root[44] torch.ops.aten.div.Tensor}: [t345 f32 [H=96 W=28 C=28] {pt2=root:div_2}] =
      div_scalar x=t344 {pt2=root:mul_2} scalar=6
    group g24 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n106 {derived}: [t351 f32 [H=14 W=14 C=96] {derived}] =
        batch_norm
          x=t348 {derived}
          weight=t35 {pt2=root:p_features_4_block_1_1_weight target=features.4.block.1.1.weight}
          bias=t36 {pt2=root:p_features_4_block_1_1_bias target=features.4.block.1.1.bias}
          running_mean=t172 {pt2=root:b_features_4_block_1_1_running_mean target=features.4.block.1.1.running_mean}
          running_var=t173 {pt2=root:b_features_4_block_1_1_running_var target=features.4.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n107 {pt2=root[46] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t352 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_30}] =
        permute x=t351 {derived} perm=[H<-C, W<-H, C<-W]
    n108 {pt2=root[47] torch.ops.aten.add.Tensor}: [t353 f32 [H=96 W=14 C=14] {pt2=root:add_4}] =
      add_scalar x=t352 {pt2=root:getitem_30} scalar=3
    n109 {pt2=root[48] torch.ops.aten.clamp.default}: [t354 f32 [H=96 W=14
                                                                 C=14] {pt2=root:clamp_6}] =
      clamp x=t353 {pt2=root:add_4} params={min=0; max=none}
    n110 {pt2=root[49] torch.ops.aten.clamp.default}: [t355 f32 [H=96 W=14
                                                                 C=14] {pt2=root:clamp_7}] =
      clamp x=t354 {pt2=root:clamp_6} params={min=none; max=6}
    n111 {pt2=root[50] torch.ops.aten.mul.Tensor}: [t356 f32 [H=96 W=14 C=14] {pt2=root:mul_3}] =
      mul a=t352 {pt2=root:getitem_30} b=t355 {pt2=root:clamp_7}
    n112 {pt2=root[51] torch.ops.aten.div.Tensor}: [t357 f32 [H=96 W=14 C=14] {pt2=root:div_3}] =
      div_scalar x=t356 {pt2=root:mul_3} scalar=6
    n113 {pt2=root[52] torch.ops.aten.mean.dim}: [t358 f32 [H=96 W=1 C=1] {pt2=root:mean_1}] =
      mean x=t357 {pt2=root:div_3} params={dims=[C, W]; keepdim=true}
    n503 {pt2=root[54] torch.ops.aten.relu.default}: [t744 f32 [C=24] {derived}] =
      relu x=t361 {derived}
    n504 {pt2=root[56] torch.ops.aten.add.Tensor}: [t745 f32 [C=96] {derived}] =
      add_scalar x=t366 {derived} scalar=3
    n505 {pt2=root[57] torch.ops.aten.clamp.default}: [t746 f32 [C=96] {derived}] =
      clamp x=t745 {derived} params={min=0; max=none}
    n506 {pt2=root[58] torch.ops.aten.clamp.default}: [t747 f32 [C=96] {derived}] =
      clamp x=t746 {derived} params={min=none; max=6}
    n507 {pt2=root[59] torch.ops.aten.div.Tensor}: [t748 f32 [C=96] {derived}] =
      div_scalar x=t747 {derived} scalar=6
    n508 {pt2=root[55] torch.ops.aten.convolution.default}: [t371 f32 [H=96 W=1
                                                                      C=1] {pt2=root:div_4}] =
      permute x=t748 {derived} perm=[H<-C, W<-H, C<-W]
    n127 {pt2=root[60] torch.ops.aten.mul.Tensor}: [t372 f32 [H=96 W=14 C=14] {pt2=root:mul_4}] =
      mul a=t371 {pt2=root:div_4} b=t357 {pt2=root:div_3}
    group g28 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n133 {derived}: [t378 f32 [H=14 W=14 C=40] {derived}] =
        batch_norm
          x=t375 {derived}
          weight=t42 {pt2=root:p_features_4_block_3_1_weight target=features.4.block.3.1.weight}
          bias=t43 {pt2=root:p_features_4_block_3_1_bias target=features.4.block.3.1.bias}
          running_mean=t175 {pt2=root:b_features_4_block_3_1_running_mean target=features.4.block.3.1.running_mean}
          running_var=t176 {pt2=root:b_features_4_block_3_1_running_var target=features.4.block.3.1.running_var}
          params={channel=C; eps=0.001}
    group g30 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n140 {derived}: [t385 f32 [H=14 W=14 C=240] {derived}] =
        batch_norm
          x=t382 {derived}
          weight=t45 {pt2=root:p_features_5_block_0_1_weight target=features.5.block.0.1.weight}
          bias=t46 {pt2=root:p_features_5_block_0_1_bias target=features.5.block.0.1.bias}
          running_mean=t178 {pt2=root:b_features_5_block_0_1_running_mean target=features.5.block.0.1.running_mean}
          running_var=t179 {pt2=root:b_features_5_block_0_1_running_var target=features.5.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n141 {pt2=root[64] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t386 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_36}] =
        permute x=t385 {derived} perm=[H<-C, W<-H, C<-W]
    n142 {pt2=root[65] torch.ops.aten.add.Tensor}: [t387 f32 [H=240 W=14 C=14] {pt2=root:add_6}] =
      add_scalar x=t386 {pt2=root:getitem_36} scalar=3
    n143 {pt2=root[66] torch.ops.aten.clamp.default}: [t388 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_10}] =
      clamp x=t387 {pt2=root:add_6} params={min=0; max=none}
    n144 {pt2=root[67] torch.ops.aten.clamp.default}: [t389 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_11}] =
      clamp x=t388 {pt2=root:clamp_10} params={min=none; max=6}
    n145 {pt2=root[68] torch.ops.aten.mul.Tensor}: [t390 f32 [H=240 W=14 C=14] {pt2=root:mul_5}] =
      mul a=t386 {pt2=root:getitem_36} b=t389 {pt2=root:clamp_11}
    n146 {pt2=root[69] torch.ops.aten.div.Tensor}: [t391 f32 [H=240 W=14 C=14] {pt2=root:div_5}] =
      div_scalar x=t390 {pt2=root:mul_5} scalar=6
    group g32 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n152 {derived}: [t397 f32 [H=14 W=14 C=240] {derived}] =
        batch_norm
          x=t394 {derived}
          weight=t48 {pt2=root:p_features_5_block_1_1_weight target=features.5.block.1.1.weight}
          bias=t49 {pt2=root:p_features_5_block_1_1_bias target=features.5.block.1.1.bias}
          running_mean=t181 {pt2=root:b_features_5_block_1_1_running_mean target=features.5.block.1.1.running_mean}
          running_var=t182 {pt2=root:b_features_5_block_1_1_running_var target=features.5.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n153 {pt2=root[71] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t398 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_39}] =
        permute x=t397 {derived} perm=[H<-C, W<-H, C<-W]
    n154 {pt2=root[72] torch.ops.aten.add.Tensor}: [t399 f32 [H=240 W=14 C=14] {pt2=root:add_7}] =
      add_scalar x=t398 {pt2=root:getitem_39} scalar=3
    n155 {pt2=root[73] torch.ops.aten.clamp.default}: [t400 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_12}] =
      clamp x=t399 {pt2=root:add_7} params={min=0; max=none}
    n156 {pt2=root[74] torch.ops.aten.clamp.default}: [t401 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_13}] =
      clamp x=t400 {pt2=root:clamp_12} params={min=none; max=6}
    n157 {pt2=root[75] torch.ops.aten.mul.Tensor}: [t402 f32 [H=240 W=14 C=14] {pt2=root:mul_6}] =
      mul a=t398 {pt2=root:getitem_39} b=t401 {pt2=root:clamp_13}
    n158 {pt2=root[76] torch.ops.aten.div.Tensor}: [t403 f32 [H=240 W=14 C=14] {pt2=root:div_6}] =
      div_scalar x=t402 {pt2=root:mul_6} scalar=6
    n159 {pt2=root[77] torch.ops.aten.mean.dim}: [t404 f32 [H=240 W=1 C=1] {pt2=root:mean_2}] =
      mean x=t403 {pt2=root:div_6} params={dims=[C, W]; keepdim=true}
    n509 {pt2=root[79] torch.ops.aten.relu.default}: [t749 f32 [C=64] {derived}] =
      relu x=t407 {derived}
    n510 {pt2=root[81] torch.ops.aten.add.Tensor}: [t750 f32 [C=240] {derived}] =
      add_scalar x=t412 {derived} scalar=3
    n511 {pt2=root[82] torch.ops.aten.clamp.default}: [t751 f32 [C=240] {derived}] =
      clamp x=t750 {derived} params={min=0; max=none}
    n512 {pt2=root[83] torch.ops.aten.clamp.default}: [t752 f32 [C=240] {derived}] =
      clamp x=t751 {derived} params={min=none; max=6}
    n513 {pt2=root[84] torch.ops.aten.div.Tensor}: [t753 f32 [C=240] {derived}] =
      div_scalar x=t752 {derived} scalar=6
    n514 {pt2=root[80] torch.ops.aten.convolution.default}: [t417 f32 [H=240
                                                                      W=1 C=1] {pt2=root:div_7}] =
      permute x=t753 {derived} perm=[H<-C, W<-H, C<-W]
    n173 {pt2=root[85] torch.ops.aten.mul.Tensor}: [t418 f32 [H=240 W=14 C=14] {pt2=root:mul_7}] =
      mul a=t417 {pt2=root:div_7} b=t403 {pt2=root:div_6}
    group g36 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n179 {derived}: [t424 f32 [H=14 W=14 C=40] {derived}] =
        batch_norm
          x=t421 {derived}
          weight=t55 {pt2=root:p_features_5_block_3_1_weight target=features.5.block.3.1.weight}
          bias=t56 {pt2=root:p_features_5_block_3_1_bias target=features.5.block.3.1.bias}
          running_mean=t184 {pt2=root:b_features_5_block_3_1_running_mean target=features.5.block.3.1.running_mean}
          running_var=t185 {pt2=root:b_features_5_block_3_1_running_var target=features.5.block.3.1.running_var}
          params={channel=C; eps=0.001}
    n515 {pt2=root[88] torch.ops.aten.add.Tensor}: [t754 f32 [H=14 W=14 C=40] {derived}] =
      add a=t424 {derived} b=t378 {derived}
    group g38 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n187 {derived}: [t432 f32 [H=14 W=14 C=240] {derived}] =
        batch_norm
          x=t429 {derived}
          weight=t58 {pt2=root:p_features_6_block_0_1_weight target=features.6.block.0.1.weight}
          bias=t59 {pt2=root:p_features_6_block_0_1_bias target=features.6.block.0.1.bias}
          running_mean=t187 {pt2=root:b_features_6_block_0_1_running_mean target=features.6.block.0.1.running_mean}
          running_var=t188 {pt2=root:b_features_6_block_0_1_running_var target=features.6.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n188 {pt2=root[90] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t433 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_45}] =
        permute x=t432 {derived} perm=[H<-C, W<-H, C<-W]
    n189 {pt2=root[91] torch.ops.aten.add.Tensor}: [t434 f32 [H=240 W=14 C=14] {pt2=root:add_10}] =
      add_scalar x=t433 {pt2=root:getitem_45} scalar=3
    n190 {pt2=root[92] torch.ops.aten.clamp.default}: [t435 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_16}] =
      clamp x=t434 {pt2=root:add_10} params={min=0; max=none}
    n191 {pt2=root[93] torch.ops.aten.clamp.default}: [t436 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_17}] =
      clamp x=t435 {pt2=root:clamp_16} params={min=none; max=6}
    n192 {pt2=root[94] torch.ops.aten.mul.Tensor}: [t437 f32 [H=240 W=14 C=14] {pt2=root:mul_8}] =
      mul a=t433 {pt2=root:getitem_45} b=t436 {pt2=root:clamp_17}
    n193 {pt2=root[95] torch.ops.aten.div.Tensor}: [t438 f32 [H=240 W=14 C=14] {pt2=root:div_8}] =
      div_scalar x=t437 {pt2=root:mul_8} scalar=6
    group g40 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n199 {derived}: [t444 f32 [H=14 W=14 C=240] {derived}] =
        batch_norm
          x=t441 {derived}
          weight=t61 {pt2=root:p_features_6_block_1_1_weight target=features.6.block.1.1.weight}
          bias=t62 {pt2=root:p_features_6_block_1_1_bias target=features.6.block.1.1.bias}
          running_mean=t190 {pt2=root:b_features_6_block_1_1_running_mean target=features.6.block.1.1.running_mean}
          running_var=t191 {pt2=root:b_features_6_block_1_1_running_var target=features.6.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n200 {pt2=root[97] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t445 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_48}] =
        permute x=t444 {derived} perm=[H<-C, W<-H, C<-W]
    n201 {pt2=root[98] torch.ops.aten.add.Tensor}: [t446 f32 [H=240 W=14 C=14] {pt2=root:add_11}] =
      add_scalar x=t445 {pt2=root:getitem_48} scalar=3
    n202 {pt2=root[99] torch.ops.aten.clamp.default}: [t447 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_18}] =
      clamp x=t446 {pt2=root:add_11} params={min=0; max=none}
    n203 {pt2=root[100] torch.ops.aten.clamp.default}: [t448 f32 [H=240 W=14
                                                                  C=14] {pt2=root:clamp_19}] =
      clamp x=t447 {pt2=root:clamp_18} params={min=none; max=6}
    n204 {pt2=root[101] torch.ops.aten.mul.Tensor}: [t449 f32 [H=240 W=14 C=14] {pt2=root:mul_9}] =
      mul a=t445 {pt2=root:getitem_48} b=t448 {pt2=root:clamp_19}
    n205 {pt2=root[102] torch.ops.aten.div.Tensor}: [t450 f32 [H=240 W=14 C=14] {pt2=root:div_9}] =
      div_scalar x=t449 {pt2=root:mul_9} scalar=6
    n206 {pt2=root[103] torch.ops.aten.mean.dim}: [t451 f32 [H=240 W=1 C=1] {pt2=root:mean_3}] =
      mean x=t450 {pt2=root:div_9} params={dims=[C, W]; keepdim=true}
    n516 {pt2=root[105] torch.ops.aten.relu.default}: [t755 f32 [C=64] {derived}] =
      relu x=t454 {derived}
    n517 {pt2=root[107] torch.ops.aten.add.Tensor}: [t756 f32 [C=240] {derived}] =
      add_scalar x=t459 {derived} scalar=3
    n518 {pt2=root[108] torch.ops.aten.clamp.default}: [t757 f32 [C=240] {derived}] =
      clamp x=t756 {derived} params={min=0; max=none}
    n519 {pt2=root[109] torch.ops.aten.clamp.default}: [t758 f32 [C=240] {derived}] =
      clamp x=t757 {derived} params={min=none; max=6}
    n520 {pt2=root[110] torch.ops.aten.div.Tensor}: [t759 f32 [C=240] {derived}] =
      div_scalar x=t758 {derived} scalar=6
    n521 {pt2=root[106] torch.ops.aten.convolution.default}: [t464 f32 [H=240
                                                                      W=1 C=1] {pt2=root:div_10}] =
      permute x=t759 {derived} perm=[H<-C, W<-H, C<-W]
    n220 {pt2=root[111] torch.ops.aten.mul.Tensor}: [t465 f32 [H=240 W=14 C=14] {pt2=root:mul_10}] =
      mul a=t464 {pt2=root:div_10} b=t450 {pt2=root:div_9}
    group g44 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n226 {derived}: [t471 f32 [H=14 W=14 C=40] {derived}] =
        batch_norm
          x=t468 {derived}
          weight=t68 {pt2=root:p_features_6_block_3_1_weight target=features.6.block.3.1.weight}
          bias=t69 {pt2=root:p_features_6_block_3_1_bias target=features.6.block.3.1.bias}
          running_mean=t193 {pt2=root:b_features_6_block_3_1_running_mean target=features.6.block.3.1.running_mean}
          running_var=t194 {pt2=root:b_features_6_block_3_1_running_var target=features.6.block.3.1.running_var}
          params={channel=C; eps=0.001}
    n522 {pt2=root[114] torch.ops.aten.add.Tensor}: [t760 f32 [H=14 W=14 C=40] {derived}] =
      add a=t471 {derived} b=t754 {derived}
    group g46 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n234 {derived}: [t479 f32 [H=14 W=14 C=120] {derived}] =
        batch_norm
          x=t476 {derived}
          weight=t71 {pt2=root:p_features_7_block_0_1_weight target=features.7.block.0.1.weight}
          bias=t72 {pt2=root:p_features_7_block_0_1_bias target=features.7.block.0.1.bias}
          running_mean=t196 {pt2=root:b_features_7_block_0_1_running_mean target=features.7.block.0.1.running_mean}
          running_var=t197 {pt2=root:b_features_7_block_0_1_running_var target=features.7.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n235 {pt2=root[116] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t480 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_54}] =
        permute x=t479 {derived} perm=[H<-C, W<-H, C<-W]
    n236 {pt2=root[117] torch.ops.aten.add.Tensor}: [t481 f32 [H=120 W=14 C=14] {pt2=root:add_14}] =
      add_scalar x=t480 {pt2=root:getitem_54} scalar=3
    n237 {pt2=root[118] torch.ops.aten.clamp.default}: [t482 f32 [H=120 W=14
                                                                  C=14] {pt2=root:clamp_22}] =
      clamp x=t481 {pt2=root:add_14} params={min=0; max=none}
    n238 {pt2=root[119] torch.ops.aten.clamp.default}: [t483 f32 [H=120 W=14
                                                                  C=14] {pt2=root:clamp_23}] =
      clamp x=t482 {pt2=root:clamp_22} params={min=none; max=6}
    n239 {pt2=root[120] torch.ops.aten.mul.Tensor}: [t484 f32 [H=120 W=14 C=14] {pt2=root:mul_11}] =
      mul a=t480 {pt2=root:getitem_54} b=t483 {pt2=root:clamp_23}
    n240 {pt2=root[121] torch.ops.aten.div.Tensor}: [t485 f32 [H=120 W=14 C=14] {pt2=root:div_11}] =
      div_scalar x=t484 {pt2=root:mul_11} scalar=6
    group g48 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n246 {derived}: [t491 f32 [H=14 W=14 C=120] {derived}] =
        batch_norm
          x=t488 {derived}
          weight=t74 {pt2=root:p_features_7_block_1_1_weight target=features.7.block.1.1.weight}
          bias=t75 {pt2=root:p_features_7_block_1_1_bias target=features.7.block.1.1.bias}
          running_mean=t199 {pt2=root:b_features_7_block_1_1_running_mean target=features.7.block.1.1.running_mean}
          running_var=t200 {pt2=root:b_features_7_block_1_1_running_var target=features.7.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n247 {pt2=root[123] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t492 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_57}] =
        permute x=t491 {derived} perm=[H<-C, W<-H, C<-W]
    n248 {pt2=root[124] torch.ops.aten.add.Tensor}: [t493 f32 [H=120 W=14 C=14] {pt2=root:add_15}] =
      add_scalar x=t492 {pt2=root:getitem_57} scalar=3
    n249 {pt2=root[125] torch.ops.aten.clamp.default}: [t494 f32 [H=120 W=14
                                                                  C=14] {pt2=root:clamp_24}] =
      clamp x=t493 {pt2=root:add_15} params={min=0; max=none}
    n250 {pt2=root[126] torch.ops.aten.clamp.default}: [t495 f32 [H=120 W=14
                                                                  C=14] {pt2=root:clamp_25}] =
      clamp x=t494 {pt2=root:clamp_24} params={min=none; max=6}
    n251 {pt2=root[127] torch.ops.aten.mul.Tensor}: [t496 f32 [H=120 W=14 C=14] {pt2=root:mul_12}] =
      mul a=t492 {pt2=root:getitem_57} b=t495 {pt2=root:clamp_25}
    n252 {pt2=root[128] torch.ops.aten.div.Tensor}: [t497 f32 [H=120 W=14 C=14] {pt2=root:div_12}] =
      div_scalar x=t496 {pt2=root:mul_12} scalar=6
    n253 {pt2=root[129] torch.ops.aten.mean.dim}: [t498 f32 [H=120 W=1 C=1] {pt2=root:mean_4}] =
      mean x=t497 {pt2=root:div_12} params={dims=[C, W]; keepdim=true}
    n523 {pt2=root[131] torch.ops.aten.relu.default}: [t761 f32 [C=32] {derived}] =
      relu x=t501 {derived}
    n524 {pt2=root[133] torch.ops.aten.add.Tensor}: [t762 f32 [C=120] {derived}] =
      add_scalar x=t506 {derived} scalar=3
    n525 {pt2=root[134] torch.ops.aten.clamp.default}: [t763 f32 [C=120] {derived}] =
      clamp x=t762 {derived} params={min=0; max=none}
    n526 {pt2=root[135] torch.ops.aten.clamp.default}: [t764 f32 [C=120] {derived}] =
      clamp x=t763 {derived} params={min=none; max=6}
    n527 {pt2=root[136] torch.ops.aten.div.Tensor}: [t765 f32 [C=120] {derived}] =
      div_scalar x=t764 {derived} scalar=6
    n528 {pt2=root[132] torch.ops.aten.convolution.default}: [t511 f32 [H=120
                                                                      W=1 C=1] {pt2=root:div_13}] =
      permute x=t765 {derived} perm=[H<-C, W<-H, C<-W]
    n267 {pt2=root[137] torch.ops.aten.mul.Tensor}: [t512 f32 [H=120 W=14 C=14] {pt2=root:mul_13}] =
      mul a=t511 {pt2=root:div_13} b=t497 {pt2=root:div_12}
    group g52 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n273 {derived}: [t518 f32 [H=14 W=14 C=48] {derived}] =
        batch_norm
          x=t515 {derived}
          weight=t81 {pt2=root:p_features_7_block_3_1_weight target=features.7.block.3.1.weight}
          bias=t82 {pt2=root:p_features_7_block_3_1_bias target=features.7.block.3.1.bias}
          running_mean=t202 {pt2=root:b_features_7_block_3_1_running_mean target=features.7.block.3.1.running_mean}
          running_var=t203 {pt2=root:b_features_7_block_3_1_running_var target=features.7.block.3.1.running_var}
          params={channel=C; eps=0.001}
    group g54 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n280 {derived}: [t525 f32 [H=14 W=14 C=144] {derived}] =
        batch_norm
          x=t522 {derived}
          weight=t84 {pt2=root:p_features_8_block_0_1_weight target=features.8.block.0.1.weight}
          bias=t85 {pt2=root:p_features_8_block_0_1_bias target=features.8.block.0.1.bias}
          running_mean=t205 {pt2=root:b_features_8_block_0_1_running_mean target=features.8.block.0.1.running_mean}
          running_var=t206 {pt2=root:b_features_8_block_0_1_running_var target=features.8.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n281 {pt2=root[141] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t526 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_63}] =
        permute x=t525 {derived} perm=[H<-C, W<-H, C<-W]
    n282 {pt2=root[142] torch.ops.aten.add.Tensor}: [t527 f32 [H=144 W=14 C=14] {pt2=root:add_17}] =
      add_scalar x=t526 {pt2=root:getitem_63} scalar=3
    n283 {pt2=root[143] torch.ops.aten.clamp.default}: [t528 f32 [H=144 W=14
                                                                  C=14] {pt2=root:clamp_28}] =
      clamp x=t527 {pt2=root:add_17} params={min=0; max=none}
    n284 {pt2=root[144] torch.ops.aten.clamp.default}: [t529 f32 [H=144 W=14
                                                                  C=14] {pt2=root:clamp_29}] =
      clamp x=t528 {pt2=root:clamp_28} params={min=none; max=6}
    n285 {pt2=root[145] torch.ops.aten.mul.Tensor}: [t530 f32 [H=144 W=14 C=14] {pt2=root:mul_14}] =
      mul a=t526 {pt2=root:getitem_63} b=t529 {pt2=root:clamp_29}
    n286 {pt2=root[146] torch.ops.aten.div.Tensor}: [t531 f32 [H=144 W=14 C=14] {pt2=root:div_14}] =
      div_scalar x=t530 {pt2=root:mul_14} scalar=6
    group g56 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n292 {derived}: [t537 f32 [H=14 W=14 C=144] {derived}] =
        batch_norm
          x=t534 {derived}
          weight=t87 {pt2=root:p_features_8_block_1_1_weight target=features.8.block.1.1.weight}
          bias=t88 {pt2=root:p_features_8_block_1_1_bias target=features.8.block.1.1.bias}
          running_mean=t208 {pt2=root:b_features_8_block_1_1_running_mean target=features.8.block.1.1.running_mean}
          running_var=t209 {pt2=root:b_features_8_block_1_1_running_var target=features.8.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n293 {pt2=root[148] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t538 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_66}] =
        permute x=t537 {derived} perm=[H<-C, W<-H, C<-W]
    n294 {pt2=root[149] torch.ops.aten.add.Tensor}: [t539 f32 [H=144 W=14 C=14] {pt2=root:add_18}] =
      add_scalar x=t538 {pt2=root:getitem_66} scalar=3
    n295 {pt2=root[150] torch.ops.aten.clamp.default}: [t540 f32 [H=144 W=14
                                                                  C=14] {pt2=root:clamp_30}] =
      clamp x=t539 {pt2=root:add_18} params={min=0; max=none}
    n296 {pt2=root[151] torch.ops.aten.clamp.default}: [t541 f32 [H=144 W=14
                                                                  C=14] {pt2=root:clamp_31}] =
      clamp x=t540 {pt2=root:clamp_30} params={min=none; max=6}
    n297 {pt2=root[152] torch.ops.aten.mul.Tensor}: [t542 f32 [H=144 W=14 C=14] {pt2=root:mul_15}] =
      mul a=t538 {pt2=root:getitem_66} b=t541 {pt2=root:clamp_31}
    n298 {pt2=root[153] torch.ops.aten.div.Tensor}: [t543 f32 [H=144 W=14 C=14] {pt2=root:div_15}] =
      div_scalar x=t542 {pt2=root:mul_15} scalar=6
    n299 {pt2=root[154] torch.ops.aten.mean.dim}: [t544 f32 [H=144 W=1 C=1] {pt2=root:mean_5}] =
      mean x=t543 {pt2=root:div_15} params={dims=[C, W]; keepdim=true}
    n529 {pt2=root[156] torch.ops.aten.relu.default}: [t766 f32 [C=40] {derived}] =
      relu x=t547 {derived}
    n530 {pt2=root[158] torch.ops.aten.add.Tensor}: [t767 f32 [C=144] {derived}] =
      add_scalar x=t552 {derived} scalar=3
    n531 {pt2=root[159] torch.ops.aten.clamp.default}: [t768 f32 [C=144] {derived}] =
      clamp x=t767 {derived} params={min=0; max=none}
    n532 {pt2=root[160] torch.ops.aten.clamp.default}: [t769 f32 [C=144] {derived}] =
      clamp x=t768 {derived} params={min=none; max=6}
    n533 {pt2=root[161] torch.ops.aten.div.Tensor}: [t770 f32 [C=144] {derived}] =
      div_scalar x=t769 {derived} scalar=6
    n534 {pt2=root[157] torch.ops.aten.convolution.default}: [t557 f32 [H=144
                                                                      W=1 C=1] {pt2=root:div_16}] =
      permute x=t770 {derived} perm=[H<-C, W<-H, C<-W]
    n313 {pt2=root[162] torch.ops.aten.mul.Tensor}: [t558 f32 [H=144 W=14 C=14] {pt2=root:mul_16}] =
      mul a=t557 {pt2=root:div_16} b=t543 {pt2=root:div_15}
    group g60 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n319 {derived}: [t564 f32 [H=14 W=14 C=48] {derived}] =
        batch_norm
          x=t561 {derived}
          weight=t94 {pt2=root:p_features_8_block_3_1_weight target=features.8.block.3.1.weight}
          bias=t95 {pt2=root:p_features_8_block_3_1_bias target=features.8.block.3.1.bias}
          running_mean=t211 {pt2=root:b_features_8_block_3_1_running_mean target=features.8.block.3.1.running_mean}
          running_var=t212 {pt2=root:b_features_8_block_3_1_running_var target=features.8.block.3.1.running_var}
          params={channel=C; eps=0.001}
    n535 {pt2=root[165] torch.ops.aten.add.Tensor}: [t771 f32 [H=14 W=14 C=48] {derived}] =
      add a=t564 {derived} b=t518 {derived}
    group g62 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n327 {derived}: [t572 f32 [H=14 W=14 C=288] {derived}] =
        batch_norm
          x=t569 {derived}
          weight=t97 {pt2=root:p_features_9_block_0_1_weight target=features.9.block.0.1.weight}
          bias=t98 {pt2=root:p_features_9_block_0_1_bias target=features.9.block.0.1.bias}
          running_mean=t214 {pt2=root:b_features_9_block_0_1_running_mean target=features.9.block.0.1.running_mean}
          running_var=t215 {pt2=root:b_features_9_block_0_1_running_var target=features.9.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n328 {pt2=root[167] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t573 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_72}] =
        permute x=t572 {derived} perm=[H<-C, W<-H, C<-W]
    n329 {pt2=root[168] torch.ops.aten.add.Tensor}: [t574 f32 [H=288 W=14 C=14] {pt2=root:add_21}] =
      add_scalar x=t573 {pt2=root:getitem_72} scalar=3
    n330 {pt2=root[169] torch.ops.aten.clamp.default}: [t575 f32 [H=288 W=14
                                                                  C=14] {pt2=root:clamp_34}] =
      clamp x=t574 {pt2=root:add_21} params={min=0; max=none}
    n331 {pt2=root[170] torch.ops.aten.clamp.default}: [t576 f32 [H=288 W=14
                                                                  C=14] {pt2=root:clamp_35}] =
      clamp x=t575 {pt2=root:clamp_34} params={min=none; max=6}
    n332 {pt2=root[171] torch.ops.aten.mul.Tensor}: [t577 f32 [H=288 W=14 C=14] {pt2=root:mul_17}] =
      mul a=t573 {pt2=root:getitem_72} b=t576 {pt2=root:clamp_35}
    n333 {pt2=root[172] torch.ops.aten.div.Tensor}: [t578 f32 [H=288 W=14 C=14] {pt2=root:div_17}] =
      div_scalar x=t577 {pt2=root:mul_17} scalar=6
    group g64 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n339 {derived}: [t584 f32 [H=7 W=7 C=288] {derived}] =
        batch_norm
          x=t581 {derived}
          weight=t100 {pt2=root:p_features_9_block_1_1_weight target=features.9.block.1.1.weight}
          bias=t101 {pt2=root:p_features_9_block_1_1_bias target=features.9.block.1.1.bias}
          running_mean=t217 {pt2=root:b_features_9_block_1_1_running_mean target=features.9.block.1.1.running_mean}
          running_var=t218 {pt2=root:b_features_9_block_1_1_running_var target=features.9.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n340 {pt2=root[174] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t585 f32 [H=288
                                                                      W=7 C=7] {pt2=root:getitem_75}] =
        permute x=t584 {derived} perm=[H<-C, W<-H, C<-W]
    n341 {pt2=root[175] torch.ops.aten.add.Tensor}: [t586 f32 [H=288 W=7 C=7] {pt2=root:add_22}] =
      add_scalar x=t585 {pt2=root:getitem_75} scalar=3
    n342 {pt2=root[176] torch.ops.aten.clamp.default}: [t587 f32 [H=288 W=7
                                                                  C=7] {pt2=root:clamp_36}] =
      clamp x=t586 {pt2=root:add_22} params={min=0; max=none}
    n343 {pt2=root[177] torch.ops.aten.clamp.default}: [t588 f32 [H=288 W=7
                                                                  C=7] {pt2=root:clamp_37}] =
      clamp x=t587 {pt2=root:clamp_36} params={min=none; max=6}
    n344 {pt2=root[178] torch.ops.aten.mul.Tensor}: [t589 f32 [H=288 W=7 C=7] {pt2=root:mul_18}] =
      mul a=t585 {pt2=root:getitem_75} b=t588 {pt2=root:clamp_37}
    n345 {pt2=root[179] torch.ops.aten.div.Tensor}: [t590 f32 [H=288 W=7 C=7] {pt2=root:div_18}] =
      div_scalar x=t589 {pt2=root:mul_18} scalar=6
    n346 {pt2=root[180] torch.ops.aten.mean.dim}: [t591 f32 [H=288 W=1 C=1] {pt2=root:mean_6}] =
      mean x=t590 {pt2=root:div_18} params={dims=[C, W]; keepdim=true}
    n536 {pt2=root[182] torch.ops.aten.relu.default}: [t772 f32 [C=72] {derived}] =
      relu x=t594 {derived}
    n537 {pt2=root[184] torch.ops.aten.add.Tensor}: [t773 f32 [C=288] {derived}] =
      add_scalar x=t599 {derived} scalar=3
    n538 {pt2=root[185] torch.ops.aten.clamp.default}: [t774 f32 [C=288] {derived}] =
      clamp x=t773 {derived} params={min=0; max=none}
    n539 {pt2=root[186] torch.ops.aten.clamp.default}: [t775 f32 [C=288] {derived}] =
      clamp x=t774 {derived} params={min=none; max=6}
    n540 {pt2=root[187] torch.ops.aten.div.Tensor}: [t776 f32 [C=288] {derived}] =
      div_scalar x=t775 {derived} scalar=6
    n541 {pt2=root[183] torch.ops.aten.convolution.default}: [t604 f32 [H=288
                                                                      W=1 C=1] {pt2=root:div_19}] =
      permute x=t776 {derived} perm=[H<-C, W<-H, C<-W]
    n360 {pt2=root[188] torch.ops.aten.mul.Tensor}: [t605 f32 [H=288 W=7 C=7] {pt2=root:mul_19}] =
      mul a=t604 {pt2=root:div_19} b=t590 {pt2=root:div_18}
    group g68 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n366 {derived}: [t611 f32 [H=7 W=7 C=96] {derived}] =
        batch_norm
          x=t608 {derived}
          weight=t107 {pt2=root:p_features_9_block_3_1_weight target=features.9.block.3.1.weight}
          bias=t108 {pt2=root:p_features_9_block_3_1_bias target=features.9.block.3.1.bias}
          running_mean=t220 {pt2=root:b_features_9_block_3_1_running_mean target=features.9.block.3.1.running_mean}
          running_var=t221 {pt2=root:b_features_9_block_3_1_running_var target=features.9.block.3.1.running_var}
          params={channel=C; eps=0.001}
    group g70 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n373 {derived}: [t618 f32 [H=7 W=7 C=576] {derived}] =
        batch_norm
          x=t615 {derived}
          weight=t110 {pt2=root:p_features_10_block_0_1_weight target=features.10.block.0.1.weight}
          bias=t111 {pt2=root:p_features_10_block_0_1_bias target=features.10.block.0.1.bias}
          running_mean=t223 {pt2=root:b_features_10_block_0_1_running_mean target=features.10.block.0.1.running_mean}
          running_var=t224 {pt2=root:b_features_10_block_0_1_running_var target=features.10.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n374 {pt2=root[192] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t619 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_81}] =
        permute x=t618 {derived} perm=[H<-C, W<-H, C<-W]
    n375 {pt2=root[193] torch.ops.aten.add.Tensor}: [t620 f32 [H=576 W=7 C=7] {pt2=root:add_24}] =
      add_scalar x=t619 {pt2=root:getitem_81} scalar=3
    n376 {pt2=root[194] torch.ops.aten.clamp.default}: [t621 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_40}] =
      clamp x=t620 {pt2=root:add_24} params={min=0; max=none}
    n377 {pt2=root[195] torch.ops.aten.clamp.default}: [t622 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_41}] =
      clamp x=t621 {pt2=root:clamp_40} params={min=none; max=6}
    n378 {pt2=root[196] torch.ops.aten.mul.Tensor}: [t623 f32 [H=576 W=7 C=7] {pt2=root:mul_20}] =
      mul a=t619 {pt2=root:getitem_81} b=t622 {pt2=root:clamp_41}
    n379 {pt2=root[197] torch.ops.aten.div.Tensor}: [t624 f32 [H=576 W=7 C=7] {pt2=root:div_20}] =
      div_scalar x=t623 {pt2=root:mul_20} scalar=6
    group g72 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n385 {derived}: [t630 f32 [H=7 W=7 C=576] {derived}] =
        batch_norm
          x=t627 {derived}
          weight=t113 {pt2=root:p_features_10_block_1_1_weight target=features.10.block.1.1.weight}
          bias=t114 {pt2=root:p_features_10_block_1_1_bias target=features.10.block.1.1.bias}
          running_mean=t226 {pt2=root:b_features_10_block_1_1_running_mean target=features.10.block.1.1.running_mean}
          running_var=t227 {pt2=root:b_features_10_block_1_1_running_var target=features.10.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n386 {pt2=root[199] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t631 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_84}] =
        permute x=t630 {derived} perm=[H<-C, W<-H, C<-W]
    n387 {pt2=root[200] torch.ops.aten.add.Tensor}: [t632 f32 [H=576 W=7 C=7] {pt2=root:add_25}] =
      add_scalar x=t631 {pt2=root:getitem_84} scalar=3
    n388 {pt2=root[201] torch.ops.aten.clamp.default}: [t633 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_42}] =
      clamp x=t632 {pt2=root:add_25} params={min=0; max=none}
    n389 {pt2=root[202] torch.ops.aten.clamp.default}: [t634 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_43}] =
      clamp x=t633 {pt2=root:clamp_42} params={min=none; max=6}
    n390 {pt2=root[203] torch.ops.aten.mul.Tensor}: [t635 f32 [H=576 W=7 C=7] {pt2=root:mul_21}] =
      mul a=t631 {pt2=root:getitem_84} b=t634 {pt2=root:clamp_43}
    n391 {pt2=root[204] torch.ops.aten.div.Tensor}: [t636 f32 [H=576 W=7 C=7] {pt2=root:div_21}] =
      div_scalar x=t635 {pt2=root:mul_21} scalar=6
    n392 {pt2=root[205] torch.ops.aten.mean.dim}: [t637 f32 [H=576 W=1 C=1] {pt2=root:mean_7}] =
      mean x=t636 {pt2=root:div_21} params={dims=[C, W]; keepdim=true}
    n542 {pt2=root[207] torch.ops.aten.relu.default}: [t777 f32 [C=144] {derived}] =
      relu x=t640 {derived}
    n543 {pt2=root[209] torch.ops.aten.add.Tensor}: [t778 f32 [C=576] {derived}] =
      add_scalar x=t645 {derived} scalar=3
    n544 {pt2=root[210] torch.ops.aten.clamp.default}: [t779 f32 [C=576] {derived}] =
      clamp x=t778 {derived} params={min=0; max=none}
    n545 {pt2=root[211] torch.ops.aten.clamp.default}: [t780 f32 [C=576] {derived}] =
      clamp x=t779 {derived} params={min=none; max=6}
    n546 {pt2=root[212] torch.ops.aten.div.Tensor}: [t781 f32 [C=576] {derived}] =
      div_scalar x=t780 {derived} scalar=6
    n547 {pt2=root[208] torch.ops.aten.convolution.default}: [t650 f32 [H=576
                                                                      W=1 C=1] {pt2=root:div_22}] =
      permute x=t781 {derived} perm=[H<-C, W<-H, C<-W]
    n406 {pt2=root[213] torch.ops.aten.mul.Tensor}: [t651 f32 [H=576 W=7 C=7] {pt2=root:mul_22}] =
      mul a=t650 {pt2=root:div_22} b=t636 {pt2=root:div_21}
    group g76 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n412 {derived}: [t657 f32 [H=7 W=7 C=96] {derived}] =
        batch_norm
          x=t654 {derived}
          weight=t120 {pt2=root:p_features_10_block_3_1_weight target=features.10.block.3.1.weight}
          bias=t121 {pt2=root:p_features_10_block_3_1_bias target=features.10.block.3.1.bias}
          running_mean=t229 {pt2=root:b_features_10_block_3_1_running_mean target=features.10.block.3.1.running_mean}
          running_var=t230 {pt2=root:b_features_10_block_3_1_running_var target=features.10.block.3.1.running_var}
          params={channel=C; eps=0.001}
    n548 {pt2=root[216] torch.ops.aten.add.Tensor}: [t782 f32 [H=7 W=7 C=96] {derived}] =
      add a=t657 {derived} b=t611 {derived}
    group g78 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n420 {derived}: [t665 f32 [H=7 W=7 C=576] {derived}] =
        batch_norm
          x=t662 {derived}
          weight=t123 {pt2=root:p_features_11_block_0_1_weight target=features.11.block.0.1.weight}
          bias=t124 {pt2=root:p_features_11_block_0_1_bias target=features.11.block.0.1.bias}
          running_mean=t232 {pt2=root:b_features_11_block_0_1_running_mean target=features.11.block.0.1.running_mean}
          running_var=t233 {pt2=root:b_features_11_block_0_1_running_var target=features.11.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n421 {pt2=root[218] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t666 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_90}] =
        permute x=t665 {derived} perm=[H<-C, W<-H, C<-W]
    n422 {pt2=root[219] torch.ops.aten.add.Tensor}: [t667 f32 [H=576 W=7 C=7] {pt2=root:add_28}] =
      add_scalar x=t666 {pt2=root:getitem_90} scalar=3
    n423 {pt2=root[220] torch.ops.aten.clamp.default}: [t668 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_46}] =
      clamp x=t667 {pt2=root:add_28} params={min=0; max=none}
    n424 {pt2=root[221] torch.ops.aten.clamp.default}: [t669 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_47}] =
      clamp x=t668 {pt2=root:clamp_46} params={min=none; max=6}
    n425 {pt2=root[222] torch.ops.aten.mul.Tensor}: [t670 f32 [H=576 W=7 C=7] {pt2=root:mul_23}] =
      mul a=t666 {pt2=root:getitem_90} b=t669 {pt2=root:clamp_47}
    n426 {pt2=root[223] torch.ops.aten.div.Tensor}: [t671 f32 [H=576 W=7 C=7] {pt2=root:div_23}] =
      div_scalar x=t670 {pt2=root:mul_23} scalar=6
    group g80 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n432 {derived}: [t677 f32 [H=7 W=7 C=576] {derived}] =
        batch_norm
          x=t674 {derived}
          weight=t126 {pt2=root:p_features_11_block_1_1_weight target=features.11.block.1.1.weight}
          bias=t127 {pt2=root:p_features_11_block_1_1_bias target=features.11.block.1.1.bias}
          running_mean=t235 {pt2=root:b_features_11_block_1_1_running_mean target=features.11.block.1.1.running_mean}
          running_var=t236 {pt2=root:b_features_11_block_1_1_running_var target=features.11.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n433 {pt2=root[225] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t678 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_93}] =
        permute x=t677 {derived} perm=[H<-C, W<-H, C<-W]
    n434 {pt2=root[226] torch.ops.aten.add.Tensor}: [t679 f32 [H=576 W=7 C=7] {pt2=root:add_29}] =
      add_scalar x=t678 {pt2=root:getitem_93} scalar=3
    n435 {pt2=root[227] torch.ops.aten.clamp.default}: [t680 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_48}] =
      clamp x=t679 {pt2=root:add_29} params={min=0; max=none}
    n436 {pt2=root[228] torch.ops.aten.clamp.default}: [t681 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_49}] =
      clamp x=t680 {pt2=root:clamp_48} params={min=none; max=6}
    n437 {pt2=root[229] torch.ops.aten.mul.Tensor}: [t682 f32 [H=576 W=7 C=7] {pt2=root:mul_24}] =
      mul a=t678 {pt2=root:getitem_93} b=t681 {pt2=root:clamp_49}
    n438 {pt2=root[230] torch.ops.aten.div.Tensor}: [t683 f32 [H=576 W=7 C=7] {pt2=root:div_24}] =
      div_scalar x=t682 {pt2=root:mul_24} scalar=6
    n439 {pt2=root[231] torch.ops.aten.mean.dim}: [t684 f32 [H=576 W=1 C=1] {pt2=root:mean_8}] =
      mean x=t683 {pt2=root:div_24} params={dims=[C, W]; keepdim=true}
    n549 {pt2=root[233] torch.ops.aten.relu.default}: [t783 f32 [C=144] {derived}] =
      relu x=t687 {derived}
    n550 {pt2=root[235] torch.ops.aten.add.Tensor}: [t784 f32 [C=576] {derived}] =
      add_scalar x=t692 {derived} scalar=3
    n551 {pt2=root[236] torch.ops.aten.clamp.default}: [t785 f32 [C=576] {derived}] =
      clamp x=t784 {derived} params={min=0; max=none}
    n552 {pt2=root[237] torch.ops.aten.clamp.default}: [t786 f32 [C=576] {derived}] =
      clamp x=t785 {derived} params={min=none; max=6}
    n553 {pt2=root[238] torch.ops.aten.div.Tensor}: [t787 f32 [C=576] {derived}] =
      div_scalar x=t786 {derived} scalar=6
    n554 {pt2=root[234] torch.ops.aten.convolution.default}: [t697 f32 [H=576
                                                                      W=1 C=1] {pt2=root:div_25}] =
      permute x=t787 {derived} perm=[H<-C, W<-H, C<-W]
    n453 {pt2=root[239] torch.ops.aten.mul.Tensor}: [t698 f32 [H=576 W=7 C=7] {pt2=root:mul_25}] =
      mul a=t697 {pt2=root:div_25} b=t683 {pt2=root:div_24}
    group g84 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n459 {derived}: [t704 f32 [H=7 W=7 C=96] {derived}] =
        batch_norm
          x=t701 {derived}
          weight=t133 {pt2=root:p_features_11_block_3_1_weight target=features.11.block.3.1.weight}
          bias=t134 {pt2=root:p_features_11_block_3_1_bias target=features.11.block.3.1.bias}
          running_mean=t238 {pt2=root:b_features_11_block_3_1_running_mean target=features.11.block.3.1.running_mean}
          running_var=t239 {pt2=root:b_features_11_block_3_1_running_var target=features.11.block.3.1.running_var}
          params={channel=C; eps=0.001}
    n555 {pt2=root[242] torch.ops.aten.add.Tensor}: [t788 f32 [H=7 W=7 C=96] {derived}] =
      add a=t704 {derived} b=t782 {derived}
    group g86 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n467 {derived}: [t712 f32 [H=7 W=7 C=576] {derived}] =
        batch_norm
          x=t709 {derived}
          weight=t136 {pt2=root:p_features_12_1_weight target=features.12.1.weight}
          bias=t137 {pt2=root:p_features_12_1_bias target=features.12.1.bias}
          running_mean=t241 {pt2=root:b_features_12_1_running_mean target=features.12.1.running_mean}
          running_var=t242 {pt2=root:b_features_12_1_running_var target=features.12.1.running_var}
          params={channel=C; eps=0.001}
      n468 {pt2=root[244] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t713 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_99}] =
        permute x=t712 {derived} perm=[H<-C, W<-H, C<-W]
    n469 {pt2=root[245] torch.ops.aten.add.Tensor}: [t714 f32 [H=576 W=7 C=7] {pt2=root:add_32}] =
      add_scalar x=t713 {pt2=root:getitem_99} scalar=3
    n470 {pt2=root[246] torch.ops.aten.clamp.default}: [t715 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_52}] =
      clamp x=t714 {pt2=root:add_32} params={min=0; max=none}
    n471 {pt2=root[247] torch.ops.aten.clamp.default}: [t716 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_53}] =
      clamp x=t715 {pt2=root:clamp_52} params={min=none; max=6}
    n472 {pt2=root[248] torch.ops.aten.mul.Tensor}: [t717 f32 [H=576 W=7 C=7] {pt2=root:mul_26}] =
      mul a=t713 {pt2=root:getitem_99} b=t716 {pt2=root:clamp_53}
    n473 {pt2=root[249] torch.ops.aten.div.Tensor}: [t718 f32 [H=576 W=7 C=7] {pt2=root:div_26}] =
      div_scalar x=t717 {pt2=root:mul_26} scalar=6
    n474 {pt2=root[250] torch.ops.aten.mean.dim}: [t719 f32 [H=576 W=1 C=1] {pt2=root:mean_9}] =
      mean x=t718 {pt2=root:div_26} params={dims=[C, W]; keepdim=true}
    n556 {pt2=root[251] torch.ops.aten.view.default}: [t720 f32 [C=576] {pt2=root:view}] =
      permute x=t719 {pt2=root:mean_9} perm=[H<-W, W<-C, C<-H]
    group g87 torch.ops.aten.addmm.default:
      n478 {pt2=root[253] torch.ops.aten.addmm.default}: [t723 f32 [C=1024] {pt2=root:addmm}] =
        linear
          x=t720 {pt2=root:view}
          weight=t722 {derived}
          bias=t139 {pt2=root:p_classifier_0_bias target=classifier.0.bias}
          params={in_features=576}
    n479 {pt2=root[254] torch.ops.aten.add.Tensor}: [t724 f32 [C=1024] {pt2=root:add_33}] =
      add_scalar x=t723 {pt2=root:addmm} scalar=3
    n480 {pt2=root[255] torch.ops.aten.clamp.default}: [t725 f32 [C=1024] {pt2=root:clamp_54}] =
      clamp x=t724 {pt2=root:add_33} params={min=0; max=none}
    n481 {pt2=root[256] torch.ops.aten.clamp.default}: [t726 f32 [C=1024] {pt2=root:clamp_55}] =
      clamp x=t725 {pt2=root:clamp_54} params={min=none; max=6}
    n482 {pt2=root[257] torch.ops.aten.mul.Tensor}: [t727 f32 [C=1024] {pt2=root:mul_27}] =
      mul a=t723 {pt2=root:addmm} b=t726 {pt2=root:clamp_55}
    n483 {pt2=root[258] torch.ops.aten.div.Tensor}: [t728 f32 [C=1024] {pt2=root:div_27}] =
      div_scalar x=t727 {pt2=root:mul_27} scalar=6
    n484 {pt2=root[259] torch.ops.aten.clone.default}: [t729 f32 [C=1024] {pt2=root:clone}] =
      clone x=t728 {pt2=root:div_27}
    group g88 torch.ops.aten.addmm.default:
      n487 {pt2=root[261] torch.ops.aten.addmm.default}: [t732 f32 [C=1000] {pt2=root:addmm_1}] =
        linear
          x=t729 {pt2=root:clone}
          weight=t731 {derived}
          bias=t141 {pt2=root:p_classifier_3_bias target=classifier.3.bias}
          params={in_features=1024}
  outputs: [t732 f32 [C=1000] {pt2=root:addmm_1}]

MobileNet-v2 uses hardtanh (relu6) rather than the v3 hardswish chain, so it is
the variant that pins the [hardtanh] constructor through the pipeline.

  $ ../bin/native_graph.exe transform --pt2 "$PT2_DATA/mobilenet_v2/mobilenet_v2.pt2"
  nodes: 415 -> 206
  constants: 262, of which 0 folded
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
     t161 f32 [C=32] {pt2=root:b_features_1_conv_0_1_running_mean target=features.1.conv.0.1.running_mean} constant,
     t162 f32 [C=32] {pt2=root:b_features_1_conv_0_1_running_var target=features.1.conv.0.1.running_var} constant,
     t164 f32 [C=16] {pt2=root:b_features_1_conv_2_running_mean target=features.1.conv.2.running_mean} constant,
     t165 f32 [C=16] {pt2=root:b_features_1_conv_2_running_var target=features.1.conv.2.running_var} constant,
     t167 f32 [C=96] {pt2=root:b_features_2_conv_0_1_running_mean target=features.2.conv.0.1.running_mean} constant,
     t168 f32 [C=96] {pt2=root:b_features_2_conv_0_1_running_var target=features.2.conv.0.1.running_var} constant,
     t170 f32 [C=96] {pt2=root:b_features_2_conv_1_1_running_mean target=features.2.conv.1.1.running_mean} constant,
     t171 f32 [C=96] {pt2=root:b_features_2_conv_1_1_running_var target=features.2.conv.1.1.running_var} constant,
     t173 f32 [C=24] {pt2=root:b_features_2_conv_3_running_mean target=features.2.conv.3.running_mean} constant,
     t174 f32 [C=24] {pt2=root:b_features_2_conv_3_running_var target=features.2.conv.3.running_var} constant,
     t176 f32 [C=144] {pt2=root:b_features_3_conv_0_1_running_mean target=features.3.conv.0.1.running_mean} constant,
     t177 f32 [C=144] {pt2=root:b_features_3_conv_0_1_running_var target=features.3.conv.0.1.running_var} constant,
     t179 f32 [C=144] {pt2=root:b_features_3_conv_1_1_running_mean target=features.3.conv.1.1.running_mean} constant,
     t180 f32 [C=144] {pt2=root:b_features_3_conv_1_1_running_var target=features.3.conv.1.1.running_var} constant,
     t182 f32 [C=24] {pt2=root:b_features_3_conv_3_running_mean target=features.3.conv.3.running_mean} constant,
     t183 f32 [C=24] {pt2=root:b_features_3_conv_3_running_var target=features.3.conv.3.running_var} constant,
     t185 f32 [C=144] {pt2=root:b_features_4_conv_0_1_running_mean target=features.4.conv.0.1.running_mean} constant,
     t186 f32 [C=144] {pt2=root:b_features_4_conv_0_1_running_var target=features.4.conv.0.1.running_var} constant,
     t188 f32 [C=144] {pt2=root:b_features_4_conv_1_1_running_mean target=features.4.conv.1.1.running_mean} constant,
     t189 f32 [C=144] {pt2=root:b_features_4_conv_1_1_running_var target=features.4.conv.1.1.running_var} constant,
     t191 f32 [C=32] {pt2=root:b_features_4_conv_3_running_mean target=features.4.conv.3.running_mean} constant,
     t192 f32 [C=32] {pt2=root:b_features_4_conv_3_running_var target=features.4.conv.3.running_var} constant,
     t194 f32 [C=192] {pt2=root:b_features_5_conv_0_1_running_mean target=features.5.conv.0.1.running_mean} constant,
     t195 f32 [C=192] {pt2=root:b_features_5_conv_0_1_running_var target=features.5.conv.0.1.running_var} constant,
     t197 f32 [C=192] {pt2=root:b_features_5_conv_1_1_running_mean target=features.5.conv.1.1.running_mean} constant,
     t198 f32 [C=192] {pt2=root:b_features_5_conv_1_1_running_var target=features.5.conv.1.1.running_var} constant,
     t200 f32 [C=32] {pt2=root:b_features_5_conv_3_running_mean target=features.5.conv.3.running_mean} constant,
     t201 f32 [C=32] {pt2=root:b_features_5_conv_3_running_var target=features.5.conv.3.running_var} constant,
     t203 f32 [C=192] {pt2=root:b_features_6_conv_0_1_running_mean target=features.6.conv.0.1.running_mean} constant,
     t204 f32 [C=192] {pt2=root:b_features_6_conv_0_1_running_var target=features.6.conv.0.1.running_var} constant,
     t206 f32 [C=192] {pt2=root:b_features_6_conv_1_1_running_mean target=features.6.conv.1.1.running_mean} constant,
     t207 f32 [C=192] {pt2=root:b_features_6_conv_1_1_running_var target=features.6.conv.1.1.running_var} constant,
     t209 f32 [C=32] {pt2=root:b_features_6_conv_3_running_mean target=features.6.conv.3.running_mean} constant,
     t210 f32 [C=32] {pt2=root:b_features_6_conv_3_running_var target=features.6.conv.3.running_var} constant,
     t212 f32 [C=192] {pt2=root:b_features_7_conv_0_1_running_mean target=features.7.conv.0.1.running_mean} constant,
     t213 f32 [C=192] {pt2=root:b_features_7_conv_0_1_running_var target=features.7.conv.0.1.running_var} constant,
     t215 f32 [C=192] {pt2=root:b_features_7_conv_1_1_running_mean target=features.7.conv.1.1.running_mean} constant,
     t216 f32 [C=192] {pt2=root:b_features_7_conv_1_1_running_var target=features.7.conv.1.1.running_var} constant,
     t218 f32 [C=64] {pt2=root:b_features_7_conv_3_running_mean target=features.7.conv.3.running_mean} constant,
     t219 f32 [C=64] {pt2=root:b_features_7_conv_3_running_var target=features.7.conv.3.running_var} constant,
     t221 f32 [C=384] {pt2=root:b_features_8_conv_0_1_running_mean target=features.8.conv.0.1.running_mean} constant,
     t222 f32 [C=384] {pt2=root:b_features_8_conv_0_1_running_var target=features.8.conv.0.1.running_var} constant,
     t224 f32 [C=384] {pt2=root:b_features_8_conv_1_1_running_mean target=features.8.conv.1.1.running_mean} constant,
     t225 f32 [C=384] {pt2=root:b_features_8_conv_1_1_running_var target=features.8.conv.1.1.running_var} constant,
     t227 f32 [C=64] {pt2=root:b_features_8_conv_3_running_mean target=features.8.conv.3.running_mean} constant,
     t228 f32 [C=64] {pt2=root:b_features_8_conv_3_running_var target=features.8.conv.3.running_var} constant,
     t230 f32 [C=384] {pt2=root:b_features_9_conv_0_1_running_mean target=features.9.conv.0.1.running_mean} constant,
     t231 f32 [C=384] {pt2=root:b_features_9_conv_0_1_running_var target=features.9.conv.0.1.running_var} constant,
     t233 f32 [C=384] {pt2=root:b_features_9_conv_1_1_running_mean target=features.9.conv.1.1.running_mean} constant,
     t234 f32 [C=384] {pt2=root:b_features_9_conv_1_1_running_var target=features.9.conv.1.1.running_var} constant,
     t236 f32 [C=64] {pt2=root:b_features_9_conv_3_running_mean target=features.9.conv.3.running_mean} constant,
     t237 f32 [C=64] {pt2=root:b_features_9_conv_3_running_var target=features.9.conv.3.running_var} constant,
     t239 f32 [C=384] {pt2=root:b_features_10_conv_0_1_running_mean target=features.10.conv.0.1.running_mean} constant,
     t240 f32 [C=384] {pt2=root:b_features_10_conv_0_1_running_var target=features.10.conv.0.1.running_var} constant,
     t242 f32 [C=384] {pt2=root:b_features_10_conv_1_1_running_mean target=features.10.conv.1.1.running_mean} constant,
     t243 f32 [C=384] {pt2=root:b_features_10_conv_1_1_running_var target=features.10.conv.1.1.running_var} constant,
     t245 f32 [C=64] {pt2=root:b_features_10_conv_3_running_mean target=features.10.conv.3.running_mean} constant,
     t246 f32 [C=64] {pt2=root:b_features_10_conv_3_running_var target=features.10.conv.3.running_var} constant,
     t248 f32 [C=384] {pt2=root:b_features_11_conv_0_1_running_mean target=features.11.conv.0.1.running_mean} constant,
     t249 f32 [C=384] {pt2=root:b_features_11_conv_0_1_running_var target=features.11.conv.0.1.running_var} constant,
     t251 f32 [C=384] {pt2=root:b_features_11_conv_1_1_running_mean target=features.11.conv.1.1.running_mean} constant,
     t252 f32 [C=384] {pt2=root:b_features_11_conv_1_1_running_var target=features.11.conv.1.1.running_var} constant,
     t254 f32 [C=96] {pt2=root:b_features_11_conv_3_running_mean target=features.11.conv.3.running_mean} constant,
     t255 f32 [C=96] {pt2=root:b_features_11_conv_3_running_var target=features.11.conv.3.running_var} constant,
     t257 f32 [C=576] {pt2=root:b_features_12_conv_0_1_running_mean target=features.12.conv.0.1.running_mean} constant,
     t258 f32 [C=576] {pt2=root:b_features_12_conv_0_1_running_var target=features.12.conv.0.1.running_var} constant,
     t260 f32 [C=576] {pt2=root:b_features_12_conv_1_1_running_mean target=features.12.conv.1.1.running_mean} constant,
     t261 f32 [C=576] {pt2=root:b_features_12_conv_1_1_running_var target=features.12.conv.1.1.running_var} constant,
     t263 f32 [C=96] {pt2=root:b_features_12_conv_3_running_mean target=features.12.conv.3.running_mean} constant,
     t264 f32 [C=96] {pt2=root:b_features_12_conv_3_running_var target=features.12.conv.3.running_var} constant,
     t266 f32 [C=576] {pt2=root:b_features_13_conv_0_1_running_mean target=features.13.conv.0.1.running_mean} constant,
     t267 f32 [C=576] {pt2=root:b_features_13_conv_0_1_running_var target=features.13.conv.0.1.running_var} constant,
     t269 f32 [C=576] {pt2=root:b_features_13_conv_1_1_running_mean target=features.13.conv.1.1.running_mean} constant,
     t270 f32 [C=576] {pt2=root:b_features_13_conv_1_1_running_var target=features.13.conv.1.1.running_var} constant,
     t272 f32 [C=96] {pt2=root:b_features_13_conv_3_running_mean target=features.13.conv.3.running_mean} constant,
     t273 f32 [C=96] {pt2=root:b_features_13_conv_3_running_var target=features.13.conv.3.running_var} constant,
     t275 f32 [C=576] {pt2=root:b_features_14_conv_0_1_running_mean target=features.14.conv.0.1.running_mean} constant,
     t276 f32 [C=576] {pt2=root:b_features_14_conv_0_1_running_var target=features.14.conv.0.1.running_var} constant,
     t278 f32 [C=576] {pt2=root:b_features_14_conv_1_1_running_mean target=features.14.conv.1.1.running_mean} constant,
     t279 f32 [C=576] {pt2=root:b_features_14_conv_1_1_running_var target=features.14.conv.1.1.running_var} constant,
     t281 f32 [C=160] {pt2=root:b_features_14_conv_3_running_mean target=features.14.conv.3.running_mean} constant,
     t282 f32 [C=160] {pt2=root:b_features_14_conv_3_running_var target=features.14.conv.3.running_var} constant,
     t284 f32 [C=960] {pt2=root:b_features_15_conv_0_1_running_mean target=features.15.conv.0.1.running_mean} constant,
     t285 f32 [C=960] {pt2=root:b_features_15_conv_0_1_running_var target=features.15.conv.0.1.running_var} constant,
     t287 f32 [C=960] {pt2=root:b_features_15_conv_1_1_running_mean target=features.15.conv.1.1.running_mean} constant,
     t288 f32 [C=960] {pt2=root:b_features_15_conv_1_1_running_var target=features.15.conv.1.1.running_var} constant,
     t290 f32 [C=160] {pt2=root:b_features_15_conv_3_running_mean target=features.15.conv.3.running_mean} constant,
     t291 f32 [C=160] {pt2=root:b_features_15_conv_3_running_var target=features.15.conv.3.running_var} constant,
     t293 f32 [C=960] {pt2=root:b_features_16_conv_0_1_running_mean target=features.16.conv.0.1.running_mean} constant,
     t294 f32 [C=960] {pt2=root:b_features_16_conv_0_1_running_var target=features.16.conv.0.1.running_var} constant,
     t296 f32 [C=960] {pt2=root:b_features_16_conv_1_1_running_mean target=features.16.conv.1.1.running_mean} constant,
     t297 f32 [C=960] {pt2=root:b_features_16_conv_1_1_running_var target=features.16.conv.1.1.running_var} constant,
     t299 f32 [C=160] {pt2=root:b_features_16_conv_3_running_mean target=features.16.conv.3.running_mean} constant,
     t300 f32 [C=160] {pt2=root:b_features_16_conv_3_running_var target=features.16.conv.3.running_var} constant,
     t302 f32 [C=960] {pt2=root:b_features_17_conv_0_1_running_mean target=features.17.conv.0.1.running_mean} constant,
     t303 f32 [C=960] {pt2=root:b_features_17_conv_0_1_running_var target=features.17.conv.0.1.running_var} constant,
     t305 f32 [C=960] {pt2=root:b_features_17_conv_1_1_running_mean target=features.17.conv.1.1.running_mean} constant,
     t306 f32 [C=960] {pt2=root:b_features_17_conv_1_1_running_var target=features.17.conv.1.1.running_var} constant,
     t308 f32 [C=320] {pt2=root:b_features_17_conv_3_running_mean target=features.17.conv.3.running_mean} constant,
     t309 f32 [C=320] {pt2=root:b_features_17_conv_3_running_var target=features.17.conv.3.running_var} constant,
     t311 f32 [C=1280] {pt2=root:b_features_18_1_running_mean target=features.18.1.running_mean} constant,
     t312 f32 [C=1280] {pt2=root:b_features_18_1_running_var target=features.18.1.running_var} constant,
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
    group g3 torch.ops.aten.convolution.default:
      n9 {derived}: [t324 f32 [N=32 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t3 {pt2=root:p_features_1_conv_0_0_weight target=features.1.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n10 {derived}: [t325 f32 [H=112 W=112 C=32] {derived}] =
        convolution
          x=t730 {derived}
          weight=t324 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=32}
    group g5 torch.ops.aten.convolution.default:
      n17 {derived}: [t332 f32 [N=16 T=1 D=1 H=1 W=1 C=32] {derived}] =
        permute
          x=t6 {pt2=root:p_features_1_conv_1_weight target=features.1.conv.1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n18 {derived}: [t333 f32 [H=112 W=112 C=16] {derived}] =
        convolution
          x=t731 {derived}
          weight=t332 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g7 torch.ops.aten.convolution.default:
      n24 {derived}: [t339 f32 [N=96 T=1 D=1 H=1 W=1 C=16] {derived}] =
        permute
          x=t9 {pt2=root:p_features_2_conv_0_0_weight target=features.2.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n25 {derived}: [t340 f32 [H=112 W=112 C=96] {derived}] =
        convolution
          x=t336 {derived}
          weight=t339 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g9 torch.ops.aten.convolution.default:
      n32 {derived}: [t347 f32 [N=96 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t12 {pt2=root:p_features_2_conv_1_0_weight target=features.2.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n33 {derived}: [t348 f32 [H=56 W=56 C=96] {derived}] =
        convolution
          x=t732 {derived}
          weight=t347 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=96}
    group g11 torch.ops.aten.convolution.default:
      n40 {derived}: [t355 f32 [N=24 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t15 {pt2=root:p_features_2_conv_2_weight target=features.2.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n41 {derived}: [t356 f32 [H=56 W=56 C=24] {derived}] =
        convolution
          x=t733 {derived}
          weight=t355 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g13 torch.ops.aten.convolution.default:
      n47 {derived}: [t362 f32 [N=144 T=1 D=1 H=1 W=1 C=24] {derived}] =
        permute
          x=t18 {pt2=root:p_features_3_conv_0_0_weight target=features.3.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n48 {derived}: [t363 f32 [H=56 W=56 C=144] {derived}] =
        convolution
          x=t359 {derived}
          weight=t362 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g15 torch.ops.aten.convolution.default:
      n55 {derived}: [t370 f32 [N=144 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t21 {pt2=root:p_features_3_conv_1_0_weight target=features.3.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n56 {derived}: [t371 f32 [H=56 W=56 C=144] {derived}] =
        convolution
          x=t734 {derived}
          weight=t370 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=144}
    group g17 torch.ops.aten.convolution.default:
      n63 {derived}: [t378 f32 [N=24 T=1 D=1 H=1 W=1 C=144] {derived}] =
        permute
          x=t24 {pt2=root:p_features_3_conv_2_weight target=features.3.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n64 {derived}: [t379 f32 [H=56 W=56 C=24] {derived}] =
        convolution
          x=t735 {derived}
          weight=t378 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g19 torch.ops.aten.convolution.default:
      n71 {derived}: [t386 f32 [N=144 T=1 D=1 H=1 W=1 C=24] {derived}] =
        permute
          x=t27 {pt2=root:p_features_4_conv_0_0_weight target=features.4.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n72 {derived}: [t387 f32 [H=56 W=56 C=144] {derived}] =
        convolution
          x=t736 {derived}
          weight=t386 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g21 torch.ops.aten.convolution.default:
      n79 {derived}: [t394 f32 [N=144 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t30 {pt2=root:p_features_4_conv_1_0_weight target=features.4.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n80 {derived}: [t395 f32 [H=28 W=28 C=144] {derived}] =
        convolution
          x=t737 {derived}
          weight=t394 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=144}
    group g23 torch.ops.aten.convolution.default:
      n87 {derived}: [t402 f32 [N=32 T=1 D=1 H=1 W=1 C=144] {derived}] =
        permute
          x=t33 {pt2=root:p_features_4_conv_2_weight target=features.4.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n88 {derived}: [t403 f32 [H=28 W=28 C=32] {derived}] =
        convolution
          x=t738 {derived}
          weight=t402 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g25 torch.ops.aten.convolution.default:
      n94 {derived}: [t409 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {derived}] =
        permute
          x=t36 {pt2=root:p_features_5_conv_0_0_weight target=features.5.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n95 {derived}: [t410 f32 [H=28 W=28 C=192] {derived}] =
        convolution
          x=t406 {derived}
          weight=t409 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g27 torch.ops.aten.convolution.default:
      n102 {derived}: [t417 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t39 {pt2=root:p_features_5_conv_1_0_weight target=features.5.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n103 {derived}: [t418 f32 [H=28 W=28 C=192] {derived}] =
        convolution
          x=t739 {derived}
          weight=t417 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=192}
    group g29 torch.ops.aten.convolution.default:
      n110 {derived}: [t425 f32 [N=32 T=1 D=1 H=1 W=1 C=192] {derived}] =
        permute
          x=t42 {pt2=root:p_features_5_conv_2_weight target=features.5.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n111 {derived}: [t426 f32 [H=28 W=28 C=32] {derived}] =
        convolution
          x=t740 {derived}
          weight=t425 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g31 torch.ops.aten.convolution.default:
      n118 {derived}: [t433 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {derived}] =
        permute
          x=t45 {pt2=root:p_features_6_conv_0_0_weight target=features.6.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n119 {derived}: [t434 f32 [H=28 W=28 C=192] {derived}] =
        convolution
          x=t741 {derived}
          weight=t433 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g33 torch.ops.aten.convolution.default:
      n126 {derived}: [t441 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t48 {pt2=root:p_features_6_conv_1_0_weight target=features.6.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n127 {derived}: [t442 f32 [H=28 W=28 C=192] {derived}] =
        convolution
          x=t742 {derived}
          weight=t441 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=192}
    group g35 torch.ops.aten.convolution.default:
      n134 {derived}: [t449 f32 [N=32 T=1 D=1 H=1 W=1 C=192] {derived}] =
        permute
          x=t51 {pt2=root:p_features_6_conv_2_weight target=features.6.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n135 {derived}: [t450 f32 [H=28 W=28 C=32] {derived}] =
        convolution
          x=t743 {derived}
          weight=t449 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g37 torch.ops.aten.convolution.default:
      n142 {derived}: [t457 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {derived}] =
        permute
          x=t54 {pt2=root:p_features_7_conv_0_0_weight target=features.7.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n143 {derived}: [t458 f32 [H=28 W=28 C=192] {derived}] =
        convolution
          x=t744 {derived}
          weight=t457 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g39 torch.ops.aten.convolution.default:
      n150 {derived}: [t465 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t57 {pt2=root:p_features_7_conv_1_0_weight target=features.7.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n151 {derived}: [t466 f32 [H=14 W=14 C=192] {derived}] =
        convolution
          x=t745 {derived}
          weight=t465 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=192}
    group g41 torch.ops.aten.convolution.default:
      n158 {derived}: [t473 f32 [N=64 T=1 D=1 H=1 W=1 C=192] {derived}] =
        permute
          x=t60 {pt2=root:p_features_7_conv_2_weight target=features.7.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n159 {derived}: [t474 f32 [H=14 W=14 C=64] {derived}] =
        convolution
          x=t746 {derived}
          weight=t473 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g43 torch.ops.aten.convolution.default:
      n165 {derived}: [t480 f32 [N=384 T=1 D=1 H=1 W=1 C=64] {derived}] =
        permute
          x=t63 {pt2=root:p_features_8_conv_0_0_weight target=features.8.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n166 {derived}: [t481 f32 [H=14 W=14 C=384] {derived}] =
        convolution
          x=t477 {derived}
          weight=t480 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g45 torch.ops.aten.convolution.default:
      n173 {derived}: [t488 f32 [N=384 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t66 {pt2=root:p_features_8_conv_1_0_weight target=features.8.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n174 {derived}: [t489 f32 [H=14 W=14 C=384] {derived}] =
        convolution
          x=t747 {derived}
          weight=t488 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=384}
    group g47 torch.ops.aten.convolution.default:
      n181 {derived}: [t496 f32 [N=64 T=1 D=1 H=1 W=1 C=384] {derived}] =
        permute
          x=t69 {pt2=root:p_features_8_conv_2_weight target=features.8.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n182 {derived}: [t497 f32 [H=14 W=14 C=64] {derived}] =
        convolution
          x=t748 {derived}
          weight=t496 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g49 torch.ops.aten.convolution.default:
      n189 {derived}: [t504 f32 [N=384 T=1 D=1 H=1 W=1 C=64] {derived}] =
        permute
          x=t72 {pt2=root:p_features_9_conv_0_0_weight target=features.9.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n190 {derived}: [t505 f32 [H=14 W=14 C=384] {derived}] =
        convolution
          x=t749 {derived}
          weight=t504 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g51 torch.ops.aten.convolution.default:
      n197 {derived}: [t512 f32 [N=384 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t75 {pt2=root:p_features_9_conv_1_0_weight target=features.9.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n198 {derived}: [t513 f32 [H=14 W=14 C=384] {derived}] =
        convolution
          x=t750 {derived}
          weight=t512 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=384}
    group g53 torch.ops.aten.convolution.default:
      n205 {derived}: [t520 f32 [N=64 T=1 D=1 H=1 W=1 C=384] {derived}] =
        permute
          x=t78 {pt2=root:p_features_9_conv_2_weight target=features.9.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n206 {derived}: [t521 f32 [H=14 W=14 C=64] {derived}] =
        convolution
          x=t751 {derived}
          weight=t520 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g55 torch.ops.aten.convolution.default:
      n213 {derived}: [t528 f32 [N=384 T=1 D=1 H=1 W=1 C=64] {derived}] =
        permute
          x=t81 {pt2=root:p_features_10_conv_0_0_weight target=features.10.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n214 {derived}: [t529 f32 [H=14 W=14 C=384] {derived}] =
        convolution
          x=t752 {derived}
          weight=t528 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g57 torch.ops.aten.convolution.default:
      n221 {derived}: [t536 f32 [N=384 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t84 {pt2=root:p_features_10_conv_1_0_weight target=features.10.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n222 {derived}: [t537 f32 [H=14 W=14 C=384] {derived}] =
        convolution
          x=t753 {derived}
          weight=t536 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=384}
    group g59 torch.ops.aten.convolution.default:
      n229 {derived}: [t544 f32 [N=64 T=1 D=1 H=1 W=1 C=384] {derived}] =
        permute
          x=t87 {pt2=root:p_features_10_conv_2_weight target=features.10.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n230 {derived}: [t545 f32 [H=14 W=14 C=64] {derived}] =
        convolution
          x=t754 {derived}
          weight=t544 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g61 torch.ops.aten.convolution.default:
      n237 {derived}: [t552 f32 [N=384 T=1 D=1 H=1 W=1 C=64] {derived}] =
        permute
          x=t90 {pt2=root:p_features_11_conv_0_0_weight target=features.11.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n238 {derived}: [t553 f32 [H=14 W=14 C=384] {derived}] =
        convolution
          x=t755 {derived}
          weight=t552 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g63 torch.ops.aten.convolution.default:
      n245 {derived}: [t560 f32 [N=384 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t93 {pt2=root:p_features_11_conv_1_0_weight target=features.11.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n246 {derived}: [t561 f32 [H=14 W=14 C=384] {derived}] =
        convolution
          x=t756 {derived}
          weight=t560 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=384}
    group g65 torch.ops.aten.convolution.default:
      n253 {derived}: [t568 f32 [N=96 T=1 D=1 H=1 W=1 C=384] {derived}] =
        permute
          x=t96 {pt2=root:p_features_11_conv_2_weight target=features.11.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n254 {derived}: [t569 f32 [H=14 W=14 C=96] {derived}] =
        convolution
          x=t757 {derived}
          weight=t568 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g67 torch.ops.aten.convolution.default:
      n260 {derived}: [t575 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t99 {pt2=root:p_features_12_conv_0_0_weight target=features.12.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n261 {derived}: [t576 f32 [H=14 W=14 C=576] {derived}] =
        convolution
          x=t572 {derived}
          weight=t575 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g69 torch.ops.aten.convolution.default:
      n268 {derived}: [t583 f32 [N=576 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t102 {pt2=root:p_features_12_conv_1_0_weight target=features.12.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n269 {derived}: [t584 f32 [H=14 W=14 C=576] {derived}] =
        convolution
          x=t758 {derived}
          weight=t583 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=576}
    group g71 torch.ops.aten.convolution.default:
      n276 {derived}: [t591 f32 [N=96 T=1 D=1 H=1 W=1 C=576] {derived}] =
        permute
          x=t105 {pt2=root:p_features_12_conv_2_weight target=features.12.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n277 {derived}: [t592 f32 [H=14 W=14 C=96] {derived}] =
        convolution
          x=t759 {derived}
          weight=t591 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g73 torch.ops.aten.convolution.default:
      n284 {derived}: [t599 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t108 {pt2=root:p_features_13_conv_0_0_weight target=features.13.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n285 {derived}: [t600 f32 [H=14 W=14 C=576] {derived}] =
        convolution
          x=t760 {derived}
          weight=t599 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g75 torch.ops.aten.convolution.default:
      n292 {derived}: [t607 f32 [N=576 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t111 {pt2=root:p_features_13_conv_1_0_weight target=features.13.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n293 {derived}: [t608 f32 [H=14 W=14 C=576] {derived}] =
        convolution
          x=t761 {derived}
          weight=t607 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=576}
    group g77 torch.ops.aten.convolution.default:
      n300 {derived}: [t615 f32 [N=96 T=1 D=1 H=1 W=1 C=576] {derived}] =
        permute
          x=t114 {pt2=root:p_features_13_conv_2_weight target=features.13.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n301 {derived}: [t616 f32 [H=14 W=14 C=96] {derived}] =
        convolution
          x=t762 {derived}
          weight=t615 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g79 torch.ops.aten.convolution.default:
      n308 {derived}: [t623 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived}] =
        permute
          x=t117 {pt2=root:p_features_14_conv_0_0_weight target=features.14.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n309 {derived}: [t624 f32 [H=14 W=14 C=576] {derived}] =
        convolution
          x=t763 {derived}
          weight=t623 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g81 torch.ops.aten.convolution.default:
      n316 {derived}: [t631 f32 [N=576 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t120 {pt2=root:p_features_14_conv_1_0_weight target=features.14.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n317 {derived}: [t632 f32 [H=7 W=7 C=576] {derived}] =
        convolution
          x=t764 {derived}
          weight=t631 {derived}
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=576}
    group g83 torch.ops.aten.convolution.default:
      n324 {derived}: [t639 f32 [N=160 T=1 D=1 H=1 W=1 C=576] {derived}] =
        permute
          x=t123 {pt2=root:p_features_14_conv_2_weight target=features.14.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n325 {derived}: [t640 f32 [H=7 W=7 C=160] {derived}] =
        convolution
          x=t765 {derived}
          weight=t639 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g85 torch.ops.aten.convolution.default:
      n331 {derived}: [t646 f32 [N=960 T=1 D=1 H=1 W=1 C=160] {derived}] =
        permute
          x=t126 {pt2=root:p_features_15_conv_0_0_weight target=features.15.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n332 {derived}: [t647 f32 [H=7 W=7 C=960] {derived}] =
        convolution
          x=t643 {derived}
          weight=t646 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g87 torch.ops.aten.convolution.default:
      n339 {derived}: [t654 f32 [N=960 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t129 {pt2=root:p_features_15_conv_1_0_weight target=features.15.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n340 {derived}: [t655 f32 [H=7 W=7 C=960] {derived}] =
        convolution
          x=t766 {derived}
          weight=t654 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=960}
    group g89 torch.ops.aten.convolution.default:
      n347 {derived}: [t662 f32 [N=160 T=1 D=1 H=1 W=1 C=960] {derived}] =
        permute
          x=t132 {pt2=root:p_features_15_conv_2_weight target=features.15.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n348 {derived}: [t663 f32 [H=7 W=7 C=160] {derived}] =
        convolution
          x=t767 {derived}
          weight=t662 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g91 torch.ops.aten.convolution.default:
      n355 {derived}: [t670 f32 [N=960 T=1 D=1 H=1 W=1 C=160] {derived}] =
        permute
          x=t135 {pt2=root:p_features_16_conv_0_0_weight target=features.16.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n356 {derived}: [t671 f32 [H=7 W=7 C=960] {derived}] =
        convolution
          x=t768 {derived}
          weight=t670 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g93 torch.ops.aten.convolution.default:
      n363 {derived}: [t678 f32 [N=960 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t138 {pt2=root:p_features_16_conv_1_0_weight target=features.16.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n364 {derived}: [t679 f32 [H=7 W=7 C=960] {derived}] =
        convolution
          x=t769 {derived}
          weight=t678 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=960}
    group g95 torch.ops.aten.convolution.default:
      n371 {derived}: [t686 f32 [N=160 T=1 D=1 H=1 W=1 C=960] {derived}] =
        permute
          x=t141 {pt2=root:p_features_16_conv_2_weight target=features.16.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n372 {derived}: [t687 f32 [H=7 W=7 C=160] {derived}] =
        convolution
          x=t770 {derived}
          weight=t686 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g97 torch.ops.aten.convolution.default:
      n379 {derived}: [t694 f32 [N=960 T=1 D=1 H=1 W=1 C=160] {derived}] =
        permute
          x=t144 {pt2=root:p_features_17_conv_0_0_weight target=features.17.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n380 {derived}: [t695 f32 [H=7 W=7 C=960] {derived}] =
        convolution
          x=t771 {derived}
          weight=t694 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g99 torch.ops.aten.convolution.default:
      n387 {derived}: [t702 f32 [N=960 T=1 D=1 H=3 W=3 C=1] {derived}] =
        permute
          x=t147 {pt2=root:p_features_17_conv_1_0_weight target=features.17.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n388 {derived}: [t703 f32 [H=7 W=7 C=960] {derived}] =
        convolution
          x=t772 {derived}
          weight=t702 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=960}
    group g101 torch.ops.aten.convolution.default:
      n395 {derived}: [t710 f32 [N=320 T=1 D=1 H=1 W=1 C=960] {derived}] =
        permute
          x=t150 {pt2=root:p_features_17_conv_2_weight target=features.17.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n396 {derived}: [t711 f32 [H=7 W=7 C=320] {derived}] =
        convolution
          x=t773 {derived}
          weight=t710 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g103 torch.ops.aten.convolution.default:
      n402 {derived}: [t717 f32 [N=1280 T=1 D=1 H=1 W=1 C=320] {derived}] =
        permute
          x=t153 {pt2=root:p_features_18_0_weight target=features.18.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n403 {derived}: [t718 f32 [H=7 W=7 C=1280] {derived}] =
        convolution
          x=t714 {derived}
          weight=t717 {derived}
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n415 {pt2=root[152] torch.ops.aten.permute.default}: [t728 f32 [N=1000 T=1
                                                                    D=1 H=1 W=1
                                                                    C=1280] {derived}] =
      permute
        x=t156 {pt2=root:p_classifier_1_weight target=classifier.1.weight}
        perm=[N<-W, W<-N]
    group g2 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n5 {derived}: [t320 f32 [H=112 W=112 C=32] {derived}] =
        batch_norm
          x=t317 {derived}
          weight=t1 {pt2=root:p_features_0_1_weight target=features.0.1.weight}
          bias=t2 {pt2=root:p_features_0_1_bias target=features.0.1.bias}
          running_mean=t158 {pt2=root:b_features_0_1_running_mean target=features.0.1.running_mean}
          running_var=t159 {pt2=root:b_features_0_1_running_var target=features.0.1.running_var}
          params={channel=C; eps=1e-05}
    n416 {pt2=root[2] torch.ops.aten.hardtanh.default}: [t730 f32 [H=112 W=112
                                                                   C=32] {derived}] =
      hardtanh x=t320 {derived} params={min_val=0; max_val=6}
    group g4 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n13 {derived}: [t328 f32 [H=112 W=112 C=32] {derived}] =
        batch_norm
          x=t325 {derived}
          weight=t4 {pt2=root:p_features_1_conv_0_1_weight target=features.1.conv.0.1.weight}
          bias=t5 {pt2=root:p_features_1_conv_0_1_bias target=features.1.conv.0.1.bias}
          running_mean=t161 {pt2=root:b_features_1_conv_0_1_running_mean target=features.1.conv.0.1.running_mean}
          running_var=t162 {pt2=root:b_features_1_conv_0_1_running_var target=features.1.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n417 {pt2=root[5] torch.ops.aten.hardtanh.default}: [t731 f32 [H=112 W=112
                                                                   C=32] {derived}] =
      hardtanh x=t328 {derived} params={min_val=0; max_val=6}
    group g6 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n21 {derived}: [t336 f32 [H=112 W=112 C=16] {derived}] =
        batch_norm
          x=t333 {derived}
          weight=t7 {pt2=root:p_features_1_conv_2_weight target=features.1.conv.2.weight}
          bias=t8 {pt2=root:p_features_1_conv_2_bias target=features.1.conv.2.bias}
          running_mean=t164 {pt2=root:b_features_1_conv_2_running_mean target=features.1.conv.2.running_mean}
          running_var=t165 {pt2=root:b_features_1_conv_2_running_var target=features.1.conv.2.running_var}
          params={channel=C; eps=1e-05}
    group g8 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n28 {derived}: [t343 f32 [H=112 W=112 C=96] {derived}] =
        batch_norm
          x=t340 {derived}
          weight=t10 {pt2=root:p_features_2_conv_0_1_weight target=features.2.conv.0.1.weight}
          bias=t11 {pt2=root:p_features_2_conv_0_1_bias target=features.2.conv.0.1.bias}
          running_mean=t167 {pt2=root:b_features_2_conv_0_1_running_mean target=features.2.conv.0.1.running_mean}
          running_var=t168 {pt2=root:b_features_2_conv_0_1_running_var target=features.2.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n418 {pt2=root[10] torch.ops.aten.hardtanh.default}: [t732 f32 [H=112 W=112
                                                                    C=96] {derived}] =
      hardtanh x=t343 {derived} params={min_val=0; max_val=6}
    group g10 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n36 {derived}: [t351 f32 [H=56 W=56 C=96] {derived}] =
        batch_norm
          x=t348 {derived}
          weight=t13 {pt2=root:p_features_2_conv_1_1_weight target=features.2.conv.1.1.weight}
          bias=t14 {pt2=root:p_features_2_conv_1_1_bias target=features.2.conv.1.1.bias}
          running_mean=t170 {pt2=root:b_features_2_conv_1_1_running_mean target=features.2.conv.1.1.running_mean}
          running_var=t171 {pt2=root:b_features_2_conv_1_1_running_var target=features.2.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n419 {pt2=root[13] torch.ops.aten.hardtanh.default}: [t733 f32 [H=56 W=56
                                                                    C=96] {derived}] =
      hardtanh x=t351 {derived} params={min_val=0; max_val=6}
    group g12 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n44 {derived}: [t359 f32 [H=56 W=56 C=24] {derived}] =
        batch_norm
          x=t356 {derived}
          weight=t16 {pt2=root:p_features_2_conv_3_weight target=features.2.conv.3.weight}
          bias=t17 {pt2=root:p_features_2_conv_3_bias target=features.2.conv.3.bias}
          running_mean=t173 {pt2=root:b_features_2_conv_3_running_mean target=features.2.conv.3.running_mean}
          running_var=t174 {pt2=root:b_features_2_conv_3_running_var target=features.2.conv.3.running_var}
          params={channel=C; eps=1e-05}
    group g14 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n51 {derived}: [t366 f32 [H=56 W=56 C=144] {derived}] =
        batch_norm
          x=t363 {derived}
          weight=t19 {pt2=root:p_features_3_conv_0_1_weight target=features.3.conv.0.1.weight}
          bias=t20 {pt2=root:p_features_3_conv_0_1_bias target=features.3.conv.0.1.bias}
          running_mean=t176 {pt2=root:b_features_3_conv_0_1_running_mean target=features.3.conv.0.1.running_mean}
          running_var=t177 {pt2=root:b_features_3_conv_0_1_running_var target=features.3.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n420 {pt2=root[18] torch.ops.aten.hardtanh.default}: [t734 f32 [H=56 W=56
                                                                    C=144] {derived}] =
      hardtanh x=t366 {derived} params={min_val=0; max_val=6}
    group g16 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n59 {derived}: [t374 f32 [H=56 W=56 C=144] {derived}] =
        batch_norm
          x=t371 {derived}
          weight=t22 {pt2=root:p_features_3_conv_1_1_weight target=features.3.conv.1.1.weight}
          bias=t23 {pt2=root:p_features_3_conv_1_1_bias target=features.3.conv.1.1.bias}
          running_mean=t179 {pt2=root:b_features_3_conv_1_1_running_mean target=features.3.conv.1.1.running_mean}
          running_var=t180 {pt2=root:b_features_3_conv_1_1_running_var target=features.3.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n421 {pt2=root[21] torch.ops.aten.hardtanh.default}: [t735 f32 [H=56 W=56
                                                                    C=144] {derived}] =
      hardtanh x=t374 {derived} params={min_val=0; max_val=6}
    group g18 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n67 {derived}: [t382 f32 [H=56 W=56 C=24] {derived}] =
        batch_norm
          x=t379 {derived}
          weight=t25 {pt2=root:p_features_3_conv_3_weight target=features.3.conv.3.weight}
          bias=t26 {pt2=root:p_features_3_conv_3_bias target=features.3.conv.3.bias}
          running_mean=t182 {pt2=root:b_features_3_conv_3_running_mean target=features.3.conv.3.running_mean}
          running_var=t183 {pt2=root:b_features_3_conv_3_running_var target=features.3.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n422 {pt2=root[24] torch.ops.aten.add.Tensor}: [t736 f32 [H=56 W=56 C=24] {derived}] =
      add a=t359 {derived} b=t382 {derived}
    group g20 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n75 {derived}: [t390 f32 [H=56 W=56 C=144] {derived}] =
        batch_norm
          x=t387 {derived}
          weight=t28 {pt2=root:p_features_4_conv_0_1_weight target=features.4.conv.0.1.weight}
          bias=t29 {pt2=root:p_features_4_conv_0_1_bias target=features.4.conv.0.1.bias}
          running_mean=t185 {pt2=root:b_features_4_conv_0_1_running_mean target=features.4.conv.0.1.running_mean}
          running_var=t186 {pt2=root:b_features_4_conv_0_1_running_var target=features.4.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n423 {pt2=root[27] torch.ops.aten.hardtanh.default}: [t737 f32 [H=56 W=56
                                                                    C=144] {derived}] =
      hardtanh x=t390 {derived} params={min_val=0; max_val=6}
    group g22 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n83 {derived}: [t398 f32 [H=28 W=28 C=144] {derived}] =
        batch_norm
          x=t395 {derived}
          weight=t31 {pt2=root:p_features_4_conv_1_1_weight target=features.4.conv.1.1.weight}
          bias=t32 {pt2=root:p_features_4_conv_1_1_bias target=features.4.conv.1.1.bias}
          running_mean=t188 {pt2=root:b_features_4_conv_1_1_running_mean target=features.4.conv.1.1.running_mean}
          running_var=t189 {pt2=root:b_features_4_conv_1_1_running_var target=features.4.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n424 {pt2=root[30] torch.ops.aten.hardtanh.default}: [t738 f32 [H=28 W=28
                                                                    C=144] {derived}] =
      hardtanh x=t398 {derived} params={min_val=0; max_val=6}
    group g24 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n91 {derived}: [t406 f32 [H=28 W=28 C=32] {derived}] =
        batch_norm
          x=t403 {derived}
          weight=t34 {pt2=root:p_features_4_conv_3_weight target=features.4.conv.3.weight}
          bias=t35 {pt2=root:p_features_4_conv_3_bias target=features.4.conv.3.bias}
          running_mean=t191 {pt2=root:b_features_4_conv_3_running_mean target=features.4.conv.3.running_mean}
          running_var=t192 {pt2=root:b_features_4_conv_3_running_var target=features.4.conv.3.running_var}
          params={channel=C; eps=1e-05}
    group g26 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n98 {derived}: [t413 f32 [H=28 W=28 C=192] {derived}] =
        batch_norm
          x=t410 {derived}
          weight=t37 {pt2=root:p_features_5_conv_0_1_weight target=features.5.conv.0.1.weight}
          bias=t38 {pt2=root:p_features_5_conv_0_1_bias target=features.5.conv.0.1.bias}
          running_mean=t194 {pt2=root:b_features_5_conv_0_1_running_mean target=features.5.conv.0.1.running_mean}
          running_var=t195 {pt2=root:b_features_5_conv_0_1_running_var target=features.5.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n425 {pt2=root[35] torch.ops.aten.hardtanh.default}: [t739 f32 [H=28 W=28
                                                                    C=192] {derived}] =
      hardtanh x=t413 {derived} params={min_val=0; max_val=6}
    group g28 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n106 {derived}: [t421 f32 [H=28 W=28 C=192] {derived}] =
        batch_norm
          x=t418 {derived}
          weight=t40 {pt2=root:p_features_5_conv_1_1_weight target=features.5.conv.1.1.weight}
          bias=t41 {pt2=root:p_features_5_conv_1_1_bias target=features.5.conv.1.1.bias}
          running_mean=t197 {pt2=root:b_features_5_conv_1_1_running_mean target=features.5.conv.1.1.running_mean}
          running_var=t198 {pt2=root:b_features_5_conv_1_1_running_var target=features.5.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n426 {pt2=root[38] torch.ops.aten.hardtanh.default}: [t740 f32 [H=28 W=28
                                                                    C=192] {derived}] =
      hardtanh x=t421 {derived} params={min_val=0; max_val=6}
    group g30 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n114 {derived}: [t429 f32 [H=28 W=28 C=32] {derived}] =
        batch_norm
          x=t426 {derived}
          weight=t43 {pt2=root:p_features_5_conv_3_weight target=features.5.conv.3.weight}
          bias=t44 {pt2=root:p_features_5_conv_3_bias target=features.5.conv.3.bias}
          running_mean=t200 {pt2=root:b_features_5_conv_3_running_mean target=features.5.conv.3.running_mean}
          running_var=t201 {pt2=root:b_features_5_conv_3_running_var target=features.5.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n427 {pt2=root[41] torch.ops.aten.add.Tensor}: [t741 f32 [H=28 W=28 C=32] {derived}] =
      add a=t406 {derived} b=t429 {derived}
    group g32 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n122 {derived}: [t437 f32 [H=28 W=28 C=192] {derived}] =
        batch_norm
          x=t434 {derived}
          weight=t46 {pt2=root:p_features_6_conv_0_1_weight target=features.6.conv.0.1.weight}
          bias=t47 {pt2=root:p_features_6_conv_0_1_bias target=features.6.conv.0.1.bias}
          running_mean=t203 {pt2=root:b_features_6_conv_0_1_running_mean target=features.6.conv.0.1.running_mean}
          running_var=t204 {pt2=root:b_features_6_conv_0_1_running_var target=features.6.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n428 {pt2=root[44] torch.ops.aten.hardtanh.default}: [t742 f32 [H=28 W=28
                                                                    C=192] {derived}] =
      hardtanh x=t437 {derived} params={min_val=0; max_val=6}
    group g34 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n130 {derived}: [t445 f32 [H=28 W=28 C=192] {derived}] =
        batch_norm
          x=t442 {derived}
          weight=t49 {pt2=root:p_features_6_conv_1_1_weight target=features.6.conv.1.1.weight}
          bias=t50 {pt2=root:p_features_6_conv_1_1_bias target=features.6.conv.1.1.bias}
          running_mean=t206 {pt2=root:b_features_6_conv_1_1_running_mean target=features.6.conv.1.1.running_mean}
          running_var=t207 {pt2=root:b_features_6_conv_1_1_running_var target=features.6.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n429 {pt2=root[47] torch.ops.aten.hardtanh.default}: [t743 f32 [H=28 W=28
                                                                    C=192] {derived}] =
      hardtanh x=t445 {derived} params={min_val=0; max_val=6}
    group g36 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n138 {derived}: [t453 f32 [H=28 W=28 C=32] {derived}] =
        batch_norm
          x=t450 {derived}
          weight=t52 {pt2=root:p_features_6_conv_3_weight target=features.6.conv.3.weight}
          bias=t53 {pt2=root:p_features_6_conv_3_bias target=features.6.conv.3.bias}
          running_mean=t209 {pt2=root:b_features_6_conv_3_running_mean target=features.6.conv.3.running_mean}
          running_var=t210 {pt2=root:b_features_6_conv_3_running_var target=features.6.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n430 {pt2=root[50] torch.ops.aten.add.Tensor}: [t744 f32 [H=28 W=28 C=32] {derived}] =
      add a=t741 {derived} b=t453 {derived}
    group g38 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n146 {derived}: [t461 f32 [H=28 W=28 C=192] {derived}] =
        batch_norm
          x=t458 {derived}
          weight=t55 {pt2=root:p_features_7_conv_0_1_weight target=features.7.conv.0.1.weight}
          bias=t56 {pt2=root:p_features_7_conv_0_1_bias target=features.7.conv.0.1.bias}
          running_mean=t212 {pt2=root:b_features_7_conv_0_1_running_mean target=features.7.conv.0.1.running_mean}
          running_var=t213 {pt2=root:b_features_7_conv_0_1_running_var target=features.7.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n431 {pt2=root[53] torch.ops.aten.hardtanh.default}: [t745 f32 [H=28 W=28
                                                                    C=192] {derived}] =
      hardtanh x=t461 {derived} params={min_val=0; max_val=6}
    group g40 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n154 {derived}: [t469 f32 [H=14 W=14 C=192] {derived}] =
        batch_norm
          x=t466 {derived}
          weight=t58 {pt2=root:p_features_7_conv_1_1_weight target=features.7.conv.1.1.weight}
          bias=t59 {pt2=root:p_features_7_conv_1_1_bias target=features.7.conv.1.1.bias}
          running_mean=t215 {pt2=root:b_features_7_conv_1_1_running_mean target=features.7.conv.1.1.running_mean}
          running_var=t216 {pt2=root:b_features_7_conv_1_1_running_var target=features.7.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n432 {pt2=root[56] torch.ops.aten.hardtanh.default}: [t746 f32 [H=14 W=14
                                                                    C=192] {derived}] =
      hardtanh x=t469 {derived} params={min_val=0; max_val=6}
    group g42 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n162 {derived}: [t477 f32 [H=14 W=14 C=64] {derived}] =
        batch_norm
          x=t474 {derived}
          weight=t61 {pt2=root:p_features_7_conv_3_weight target=features.7.conv.3.weight}
          bias=t62 {pt2=root:p_features_7_conv_3_bias target=features.7.conv.3.bias}
          running_mean=t218 {pt2=root:b_features_7_conv_3_running_mean target=features.7.conv.3.running_mean}
          running_var=t219 {pt2=root:b_features_7_conv_3_running_var target=features.7.conv.3.running_var}
          params={channel=C; eps=1e-05}
    group g44 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n169 {derived}: [t484 f32 [H=14 W=14 C=384] {derived}] =
        batch_norm
          x=t481 {derived}
          weight=t64 {pt2=root:p_features_8_conv_0_1_weight target=features.8.conv.0.1.weight}
          bias=t65 {pt2=root:p_features_8_conv_0_1_bias target=features.8.conv.0.1.bias}
          running_mean=t221 {pt2=root:b_features_8_conv_0_1_running_mean target=features.8.conv.0.1.running_mean}
          running_var=t222 {pt2=root:b_features_8_conv_0_1_running_var target=features.8.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n433 {pt2=root[61] torch.ops.aten.hardtanh.default}: [t747 f32 [H=14 W=14
                                                                    C=384] {derived}] =
      hardtanh x=t484 {derived} params={min_val=0; max_val=6}
    group g46 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n177 {derived}: [t492 f32 [H=14 W=14 C=384] {derived}] =
        batch_norm
          x=t489 {derived}
          weight=t67 {pt2=root:p_features_8_conv_1_1_weight target=features.8.conv.1.1.weight}
          bias=t68 {pt2=root:p_features_8_conv_1_1_bias target=features.8.conv.1.1.bias}
          running_mean=t224 {pt2=root:b_features_8_conv_1_1_running_mean target=features.8.conv.1.1.running_mean}
          running_var=t225 {pt2=root:b_features_8_conv_1_1_running_var target=features.8.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n434 {pt2=root[64] torch.ops.aten.hardtanh.default}: [t748 f32 [H=14 W=14
                                                                    C=384] {derived}] =
      hardtanh x=t492 {derived} params={min_val=0; max_val=6}
    group g48 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n185 {derived}: [t500 f32 [H=14 W=14 C=64] {derived}] =
        batch_norm
          x=t497 {derived}
          weight=t70 {pt2=root:p_features_8_conv_3_weight target=features.8.conv.3.weight}
          bias=t71 {pt2=root:p_features_8_conv_3_bias target=features.8.conv.3.bias}
          running_mean=t227 {pt2=root:b_features_8_conv_3_running_mean target=features.8.conv.3.running_mean}
          running_var=t228 {pt2=root:b_features_8_conv_3_running_var target=features.8.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n435 {pt2=root[67] torch.ops.aten.add.Tensor}: [t749 f32 [H=14 W=14 C=64] {derived}] =
      add a=t477 {derived} b=t500 {derived}
    group g50 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n193 {derived}: [t508 f32 [H=14 W=14 C=384] {derived}] =
        batch_norm
          x=t505 {derived}
          weight=t73 {pt2=root:p_features_9_conv_0_1_weight target=features.9.conv.0.1.weight}
          bias=t74 {pt2=root:p_features_9_conv_0_1_bias target=features.9.conv.0.1.bias}
          running_mean=t230 {pt2=root:b_features_9_conv_0_1_running_mean target=features.9.conv.0.1.running_mean}
          running_var=t231 {pt2=root:b_features_9_conv_0_1_running_var target=features.9.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n436 {pt2=root[70] torch.ops.aten.hardtanh.default}: [t750 f32 [H=14 W=14
                                                                    C=384] {derived}] =
      hardtanh x=t508 {derived} params={min_val=0; max_val=6}
    group g52 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n201 {derived}: [t516 f32 [H=14 W=14 C=384] {derived}] =
        batch_norm
          x=t513 {derived}
          weight=t76 {pt2=root:p_features_9_conv_1_1_weight target=features.9.conv.1.1.weight}
          bias=t77 {pt2=root:p_features_9_conv_1_1_bias target=features.9.conv.1.1.bias}
          running_mean=t233 {pt2=root:b_features_9_conv_1_1_running_mean target=features.9.conv.1.1.running_mean}
          running_var=t234 {pt2=root:b_features_9_conv_1_1_running_var target=features.9.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n437 {pt2=root[73] torch.ops.aten.hardtanh.default}: [t751 f32 [H=14 W=14
                                                                    C=384] {derived}] =
      hardtanh x=t516 {derived} params={min_val=0; max_val=6}
    group g54 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n209 {derived}: [t524 f32 [H=14 W=14 C=64] {derived}] =
        batch_norm
          x=t521 {derived}
          weight=t79 {pt2=root:p_features_9_conv_3_weight target=features.9.conv.3.weight}
          bias=t80 {pt2=root:p_features_9_conv_3_bias target=features.9.conv.3.bias}
          running_mean=t236 {pt2=root:b_features_9_conv_3_running_mean target=features.9.conv.3.running_mean}
          running_var=t237 {pt2=root:b_features_9_conv_3_running_var target=features.9.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n438 {pt2=root[76] torch.ops.aten.add.Tensor}: [t752 f32 [H=14 W=14 C=64] {derived}] =
      add a=t749 {derived} b=t524 {derived}
    group g56 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n217 {derived}: [t532 f32 [H=14 W=14 C=384] {derived}] =
        batch_norm
          x=t529 {derived}
          weight=t82 {pt2=root:p_features_10_conv_0_1_weight target=features.10.conv.0.1.weight}
          bias=t83 {pt2=root:p_features_10_conv_0_1_bias target=features.10.conv.0.1.bias}
          running_mean=t239 {pt2=root:b_features_10_conv_0_1_running_mean target=features.10.conv.0.1.running_mean}
          running_var=t240 {pt2=root:b_features_10_conv_0_1_running_var target=features.10.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n439 {pt2=root[79] torch.ops.aten.hardtanh.default}: [t753 f32 [H=14 W=14
                                                                    C=384] {derived}] =
      hardtanh x=t532 {derived} params={min_val=0; max_val=6}
    group g58 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n225 {derived}: [t540 f32 [H=14 W=14 C=384] {derived}] =
        batch_norm
          x=t537 {derived}
          weight=t85 {pt2=root:p_features_10_conv_1_1_weight target=features.10.conv.1.1.weight}
          bias=t86 {pt2=root:p_features_10_conv_1_1_bias target=features.10.conv.1.1.bias}
          running_mean=t242 {pt2=root:b_features_10_conv_1_1_running_mean target=features.10.conv.1.1.running_mean}
          running_var=t243 {pt2=root:b_features_10_conv_1_1_running_var target=features.10.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n440 {pt2=root[82] torch.ops.aten.hardtanh.default}: [t754 f32 [H=14 W=14
                                                                    C=384] {derived}] =
      hardtanh x=t540 {derived} params={min_val=0; max_val=6}
    group g60 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n233 {derived}: [t548 f32 [H=14 W=14 C=64] {derived}] =
        batch_norm
          x=t545 {derived}
          weight=t88 {pt2=root:p_features_10_conv_3_weight target=features.10.conv.3.weight}
          bias=t89 {pt2=root:p_features_10_conv_3_bias target=features.10.conv.3.bias}
          running_mean=t245 {pt2=root:b_features_10_conv_3_running_mean target=features.10.conv.3.running_mean}
          running_var=t246 {pt2=root:b_features_10_conv_3_running_var target=features.10.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n441 {pt2=root[85] torch.ops.aten.add.Tensor}: [t755 f32 [H=14 W=14 C=64] {derived}] =
      add a=t752 {derived} b=t548 {derived}
    group g62 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n241 {derived}: [t556 f32 [H=14 W=14 C=384] {derived}] =
        batch_norm
          x=t553 {derived}
          weight=t91 {pt2=root:p_features_11_conv_0_1_weight target=features.11.conv.0.1.weight}
          bias=t92 {pt2=root:p_features_11_conv_0_1_bias target=features.11.conv.0.1.bias}
          running_mean=t248 {pt2=root:b_features_11_conv_0_1_running_mean target=features.11.conv.0.1.running_mean}
          running_var=t249 {pt2=root:b_features_11_conv_0_1_running_var target=features.11.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n442 {pt2=root[88] torch.ops.aten.hardtanh.default}: [t756 f32 [H=14 W=14
                                                                    C=384] {derived}] =
      hardtanh x=t556 {derived} params={min_val=0; max_val=6}
    group g64 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n249 {derived}: [t564 f32 [H=14 W=14 C=384] {derived}] =
        batch_norm
          x=t561 {derived}
          weight=t94 {pt2=root:p_features_11_conv_1_1_weight target=features.11.conv.1.1.weight}
          bias=t95 {pt2=root:p_features_11_conv_1_1_bias target=features.11.conv.1.1.bias}
          running_mean=t251 {pt2=root:b_features_11_conv_1_1_running_mean target=features.11.conv.1.1.running_mean}
          running_var=t252 {pt2=root:b_features_11_conv_1_1_running_var target=features.11.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n443 {pt2=root[91] torch.ops.aten.hardtanh.default}: [t757 f32 [H=14 W=14
                                                                    C=384] {derived}] =
      hardtanh x=t564 {derived} params={min_val=0; max_val=6}
    group g66 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n257 {derived}: [t572 f32 [H=14 W=14 C=96] {derived}] =
        batch_norm
          x=t569 {derived}
          weight=t97 {pt2=root:p_features_11_conv_3_weight target=features.11.conv.3.weight}
          bias=t98 {pt2=root:p_features_11_conv_3_bias target=features.11.conv.3.bias}
          running_mean=t254 {pt2=root:b_features_11_conv_3_running_mean target=features.11.conv.3.running_mean}
          running_var=t255 {pt2=root:b_features_11_conv_3_running_var target=features.11.conv.3.running_var}
          params={channel=C; eps=1e-05}
    group g68 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n264 {derived}: [t579 f32 [H=14 W=14 C=576] {derived}] =
        batch_norm
          x=t576 {derived}
          weight=t100 {pt2=root:p_features_12_conv_0_1_weight target=features.12.conv.0.1.weight}
          bias=t101 {pt2=root:p_features_12_conv_0_1_bias target=features.12.conv.0.1.bias}
          running_mean=t257 {pt2=root:b_features_12_conv_0_1_running_mean target=features.12.conv.0.1.running_mean}
          running_var=t258 {pt2=root:b_features_12_conv_0_1_running_var target=features.12.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n444 {pt2=root[96] torch.ops.aten.hardtanh.default}: [t758 f32 [H=14 W=14
                                                                    C=576] {derived}] =
      hardtanh x=t579 {derived} params={min_val=0; max_val=6}
    group g70 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n272 {derived}: [t587 f32 [H=14 W=14 C=576] {derived}] =
        batch_norm
          x=t584 {derived}
          weight=t103 {pt2=root:p_features_12_conv_1_1_weight target=features.12.conv.1.1.weight}
          bias=t104 {pt2=root:p_features_12_conv_1_1_bias target=features.12.conv.1.1.bias}
          running_mean=t260 {pt2=root:b_features_12_conv_1_1_running_mean target=features.12.conv.1.1.running_mean}
          running_var=t261 {pt2=root:b_features_12_conv_1_1_running_var target=features.12.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n445 {pt2=root[99] torch.ops.aten.hardtanh.default}: [t759 f32 [H=14 W=14
                                                                    C=576] {derived}] =
      hardtanh x=t587 {derived} params={min_val=0; max_val=6}
    group g72 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n280 {derived}: [t595 f32 [H=14 W=14 C=96] {derived}] =
        batch_norm
          x=t592 {derived}
          weight=t106 {pt2=root:p_features_12_conv_3_weight target=features.12.conv.3.weight}
          bias=t107 {pt2=root:p_features_12_conv_3_bias target=features.12.conv.3.bias}
          running_mean=t263 {pt2=root:b_features_12_conv_3_running_mean target=features.12.conv.3.running_mean}
          running_var=t264 {pt2=root:b_features_12_conv_3_running_var target=features.12.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n446 {pt2=root[102] torch.ops.aten.add.Tensor}: [t760 f32 [H=14 W=14 C=96] {derived}] =
      add a=t572 {derived} b=t595 {derived}
    group g74 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n288 {derived}: [t603 f32 [H=14 W=14 C=576] {derived}] =
        batch_norm
          x=t600 {derived}
          weight=t109 {pt2=root:p_features_13_conv_0_1_weight target=features.13.conv.0.1.weight}
          bias=t110 {pt2=root:p_features_13_conv_0_1_bias target=features.13.conv.0.1.bias}
          running_mean=t266 {pt2=root:b_features_13_conv_0_1_running_mean target=features.13.conv.0.1.running_mean}
          running_var=t267 {pt2=root:b_features_13_conv_0_1_running_var target=features.13.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n447 {pt2=root[105] torch.ops.aten.hardtanh.default}: [t761 f32 [H=14 W=14
                                                                     C=576] {derived}] =
      hardtanh x=t603 {derived} params={min_val=0; max_val=6}
    group g76 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n296 {derived}: [t611 f32 [H=14 W=14 C=576] {derived}] =
        batch_norm
          x=t608 {derived}
          weight=t112 {pt2=root:p_features_13_conv_1_1_weight target=features.13.conv.1.1.weight}
          bias=t113 {pt2=root:p_features_13_conv_1_1_bias target=features.13.conv.1.1.bias}
          running_mean=t269 {pt2=root:b_features_13_conv_1_1_running_mean target=features.13.conv.1.1.running_mean}
          running_var=t270 {pt2=root:b_features_13_conv_1_1_running_var target=features.13.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n448 {pt2=root[108] torch.ops.aten.hardtanh.default}: [t762 f32 [H=14 W=14
                                                                     C=576] {derived}] =
      hardtanh x=t611 {derived} params={min_val=0; max_val=6}
    group g78 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n304 {derived}: [t619 f32 [H=14 W=14 C=96] {derived}] =
        batch_norm
          x=t616 {derived}
          weight=t115 {pt2=root:p_features_13_conv_3_weight target=features.13.conv.3.weight}
          bias=t116 {pt2=root:p_features_13_conv_3_bias target=features.13.conv.3.bias}
          running_mean=t272 {pt2=root:b_features_13_conv_3_running_mean target=features.13.conv.3.running_mean}
          running_var=t273 {pt2=root:b_features_13_conv_3_running_var target=features.13.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n449 {pt2=root[111] torch.ops.aten.add.Tensor}: [t763 f32 [H=14 W=14 C=96] {derived}] =
      add a=t760 {derived} b=t619 {derived}
    group g80 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n312 {derived}: [t627 f32 [H=14 W=14 C=576] {derived}] =
        batch_norm
          x=t624 {derived}
          weight=t118 {pt2=root:p_features_14_conv_0_1_weight target=features.14.conv.0.1.weight}
          bias=t119 {pt2=root:p_features_14_conv_0_1_bias target=features.14.conv.0.1.bias}
          running_mean=t275 {pt2=root:b_features_14_conv_0_1_running_mean target=features.14.conv.0.1.running_mean}
          running_var=t276 {pt2=root:b_features_14_conv_0_1_running_var target=features.14.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n450 {pt2=root[114] torch.ops.aten.hardtanh.default}: [t764 f32 [H=14 W=14
                                                                     C=576] {derived}] =
      hardtanh x=t627 {derived} params={min_val=0; max_val=6}
    group g82 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n320 {derived}: [t635 f32 [H=7 W=7 C=576] {derived}] =
        batch_norm
          x=t632 {derived}
          weight=t121 {pt2=root:p_features_14_conv_1_1_weight target=features.14.conv.1.1.weight}
          bias=t122 {pt2=root:p_features_14_conv_1_1_bias target=features.14.conv.1.1.bias}
          running_mean=t278 {pt2=root:b_features_14_conv_1_1_running_mean target=features.14.conv.1.1.running_mean}
          running_var=t279 {pt2=root:b_features_14_conv_1_1_running_var target=features.14.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n451 {pt2=root[117] torch.ops.aten.hardtanh.default}: [t765 f32 [H=7 W=7
                                                                     C=576] {derived}] =
      hardtanh x=t635 {derived} params={min_val=0; max_val=6}
    group g84 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n328 {derived}: [t643 f32 [H=7 W=7 C=160] {derived}] =
        batch_norm
          x=t640 {derived}
          weight=t124 {pt2=root:p_features_14_conv_3_weight target=features.14.conv.3.weight}
          bias=t125 {pt2=root:p_features_14_conv_3_bias target=features.14.conv.3.bias}
          running_mean=t281 {pt2=root:b_features_14_conv_3_running_mean target=features.14.conv.3.running_mean}
          running_var=t282 {pt2=root:b_features_14_conv_3_running_var target=features.14.conv.3.running_var}
          params={channel=C; eps=1e-05}
    group g86 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n335 {derived}: [t650 f32 [H=7 W=7 C=960] {derived}] =
        batch_norm
          x=t647 {derived}
          weight=t127 {pt2=root:p_features_15_conv_0_1_weight target=features.15.conv.0.1.weight}
          bias=t128 {pt2=root:p_features_15_conv_0_1_bias target=features.15.conv.0.1.bias}
          running_mean=t284 {pt2=root:b_features_15_conv_0_1_running_mean target=features.15.conv.0.1.running_mean}
          running_var=t285 {pt2=root:b_features_15_conv_0_1_running_var target=features.15.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n452 {pt2=root[122] torch.ops.aten.hardtanh.default}: [t766 f32 [H=7 W=7
                                                                     C=960] {derived}] =
      hardtanh x=t650 {derived} params={min_val=0; max_val=6}
    group g88 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n343 {derived}: [t658 f32 [H=7 W=7 C=960] {derived}] =
        batch_norm
          x=t655 {derived}
          weight=t130 {pt2=root:p_features_15_conv_1_1_weight target=features.15.conv.1.1.weight}
          bias=t131 {pt2=root:p_features_15_conv_1_1_bias target=features.15.conv.1.1.bias}
          running_mean=t287 {pt2=root:b_features_15_conv_1_1_running_mean target=features.15.conv.1.1.running_mean}
          running_var=t288 {pt2=root:b_features_15_conv_1_1_running_var target=features.15.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n453 {pt2=root[125] torch.ops.aten.hardtanh.default}: [t767 f32 [H=7 W=7
                                                                     C=960] {derived}] =
      hardtanh x=t658 {derived} params={min_val=0; max_val=6}
    group g90 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n351 {derived}: [t666 f32 [H=7 W=7 C=160] {derived}] =
        batch_norm
          x=t663 {derived}
          weight=t133 {pt2=root:p_features_15_conv_3_weight target=features.15.conv.3.weight}
          bias=t134 {pt2=root:p_features_15_conv_3_bias target=features.15.conv.3.bias}
          running_mean=t290 {pt2=root:b_features_15_conv_3_running_mean target=features.15.conv.3.running_mean}
          running_var=t291 {pt2=root:b_features_15_conv_3_running_var target=features.15.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n454 {pt2=root[128] torch.ops.aten.add.Tensor}: [t768 f32 [H=7 W=7 C=160] {derived}] =
      add a=t643 {derived} b=t666 {derived}
    group g92 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n359 {derived}: [t674 f32 [H=7 W=7 C=960] {derived}] =
        batch_norm
          x=t671 {derived}
          weight=t136 {pt2=root:p_features_16_conv_0_1_weight target=features.16.conv.0.1.weight}
          bias=t137 {pt2=root:p_features_16_conv_0_1_bias target=features.16.conv.0.1.bias}
          running_mean=t293 {pt2=root:b_features_16_conv_0_1_running_mean target=features.16.conv.0.1.running_mean}
          running_var=t294 {pt2=root:b_features_16_conv_0_1_running_var target=features.16.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n455 {pt2=root[131] torch.ops.aten.hardtanh.default}: [t769 f32 [H=7 W=7
                                                                     C=960] {derived}] =
      hardtanh x=t674 {derived} params={min_val=0; max_val=6}
    group g94 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n367 {derived}: [t682 f32 [H=7 W=7 C=960] {derived}] =
        batch_norm
          x=t679 {derived}
          weight=t139 {pt2=root:p_features_16_conv_1_1_weight target=features.16.conv.1.1.weight}
          bias=t140 {pt2=root:p_features_16_conv_1_1_bias target=features.16.conv.1.1.bias}
          running_mean=t296 {pt2=root:b_features_16_conv_1_1_running_mean target=features.16.conv.1.1.running_mean}
          running_var=t297 {pt2=root:b_features_16_conv_1_1_running_var target=features.16.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n456 {pt2=root[134] torch.ops.aten.hardtanh.default}: [t770 f32 [H=7 W=7
                                                                     C=960] {derived}] =
      hardtanh x=t682 {derived} params={min_val=0; max_val=6}
    group g96 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n375 {derived}: [t690 f32 [H=7 W=7 C=160] {derived}] =
        batch_norm
          x=t687 {derived}
          weight=t142 {pt2=root:p_features_16_conv_3_weight target=features.16.conv.3.weight}
          bias=t143 {pt2=root:p_features_16_conv_3_bias target=features.16.conv.3.bias}
          running_mean=t299 {pt2=root:b_features_16_conv_3_running_mean target=features.16.conv.3.running_mean}
          running_var=t300 {pt2=root:b_features_16_conv_3_running_var target=features.16.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n457 {pt2=root[137] torch.ops.aten.add.Tensor}: [t771 f32 [H=7 W=7 C=160] {derived}] =
      add a=t768 {derived} b=t690 {derived}
    group g98 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n383 {derived}: [t698 f32 [H=7 W=7 C=960] {derived}] =
        batch_norm
          x=t695 {derived}
          weight=t145 {pt2=root:p_features_17_conv_0_1_weight target=features.17.conv.0.1.weight}
          bias=t146 {pt2=root:p_features_17_conv_0_1_bias target=features.17.conv.0.1.bias}
          running_mean=t302 {pt2=root:b_features_17_conv_0_1_running_mean target=features.17.conv.0.1.running_mean}
          running_var=t303 {pt2=root:b_features_17_conv_0_1_running_var target=features.17.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n458 {pt2=root[140] torch.ops.aten.hardtanh.default}: [t772 f32 [H=7 W=7
                                                                     C=960] {derived}] =
      hardtanh x=t698 {derived} params={min_val=0; max_val=6}
    group g100 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n391 {derived}: [t706 f32 [H=7 W=7 C=960] {derived}] =
        batch_norm
          x=t703 {derived}
          weight=t148 {pt2=root:p_features_17_conv_1_1_weight target=features.17.conv.1.1.weight}
          bias=t149 {pt2=root:p_features_17_conv_1_1_bias target=features.17.conv.1.1.bias}
          running_mean=t305 {pt2=root:b_features_17_conv_1_1_running_mean target=features.17.conv.1.1.running_mean}
          running_var=t306 {pt2=root:b_features_17_conv_1_1_running_var target=features.17.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n459 {pt2=root[143] torch.ops.aten.hardtanh.default}: [t773 f32 [H=7 W=7
                                                                     C=960] {derived}] =
      hardtanh x=t706 {derived} params={min_val=0; max_val=6}
    group g102 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n399 {derived}: [t714 f32 [H=7 W=7 C=320] {derived}] =
        batch_norm
          x=t711 {derived}
          weight=t151 {pt2=root:p_features_17_conv_3_weight target=features.17.conv.3.weight}
          bias=t152 {pt2=root:p_features_17_conv_3_bias target=features.17.conv.3.bias}
          running_mean=t308 {pt2=root:b_features_17_conv_3_running_mean target=features.17.conv.3.running_mean}
          running_var=t309 {pt2=root:b_features_17_conv_3_running_var target=features.17.conv.3.running_var}
          params={channel=C; eps=1e-05}
    group g104 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n406 {derived}: [t721 f32 [H=7 W=7 C=1280] {derived}] =
        batch_norm
          x=t718 {derived}
          weight=t154 {pt2=root:p_features_18_1_weight target=features.18.1.weight}
          bias=t155 {pt2=root:p_features_18_1_bias target=features.18.1.bias}
          running_mean=t311 {pt2=root:b_features_18_1_running_mean target=features.18.1.running_mean}
          running_var=t312 {pt2=root:b_features_18_1_running_var target=features.18.1.running_var}
          params={channel=C; eps=1e-05}
    n460 {pt2=root[148] torch.ops.aten.hardtanh.default}: [t774 f32 [H=7 W=7
                                                                     C=1280] {derived}] =
      hardtanh x=t721 {derived} params={min_val=0; max_val=6}
    n461 {pt2=root[149] torch.ops.aten.mean.dim}: [t775 f32 [C=1280] {derived}] =
      mean x=t774 {derived} params={dims=[W, H]; keepdim=true}
    n462 {pt2=root[151] torch.ops.aten.clone.default}: [t776 f32 [C=1280] {pt2=root:clone}] =
      clone x=t775 {derived}
    group g105 torch.ops.aten.addmm.default:
      n414 {pt2=root[153] torch.ops.aten.addmm.default}: [t729 f32 [C=1000] {pt2=root:addmm}] =
        linear
          x=t776 {pt2=root:clone}
          weight=t728 {derived}
          bias=t157 {pt2=root:p_classifier_1_bias target=classifier.1.bias}
          params={in_features=1280}
  outputs: [t729 f32 [C=1000] {pt2=root:addmm}]
