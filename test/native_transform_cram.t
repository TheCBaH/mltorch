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
    [t0 f32 [D=64 H=3 W=7 C=7] {pt2=root:p_conv1_weight target=conv1.weight} ->[n1] constant,
     t1 f32 [C=64] {pt2=root:p_bn1_weight target=bn1.weight} ->[n5] constant,
     t2 f32 [C=64] {pt2=root:p_bn1_bias target=bn1.bias} ->[n5] constant,
     t3 f32 [D=64 H=64 W=3 C=3] {pt2=root:p_layer1_0_conv1_weight target=layer1.0.conv1.weight} ->[n13] constant,
     t4 f32 [C=64] {pt2=root:p_layer1_0_bn1_weight target=layer1.0.bn1.weight} ->[n17] constant,
     t5 f32 [C=64] {pt2=root:p_layer1_0_bn1_bias target=layer1.0.bn1.bias} ->[n17] constant,
     t6 f32 [D=64 H=64 W=3 C=3] {pt2=root:p_layer1_0_conv2_weight target=layer1.0.conv2.weight} ->[n21] constant,
     t7 f32 [C=64] {pt2=root:p_layer1_0_bn2_weight target=layer1.0.bn2.weight} ->[n25] constant,
     t8 f32 [C=64] {pt2=root:p_layer1_0_bn2_bias target=layer1.0.bn2.bias} ->[n25] constant,
     t9 f32 [D=64 H=64 W=3 C=3] {pt2=root:p_layer1_1_conv1_weight target=layer1.1.conv1.weight} ->[n30] constant,
     t10 f32 [C=64] {pt2=root:p_layer1_1_bn1_weight target=layer1.1.bn1.weight} ->[n34] constant,
     t11 f32 [C=64] {pt2=root:p_layer1_1_bn1_bias target=layer1.1.bn1.bias} ->[n34] constant,
     t12 f32 [D=64 H=64 W=3 C=3] {pt2=root:p_layer1_1_conv2_weight target=layer1.1.conv2.weight} ->[n38] constant,
     t13 f32 [C=64] {pt2=root:p_layer1_1_bn2_weight target=layer1.1.bn2.weight} ->[n42] constant,
     t14 f32 [C=64] {pt2=root:p_layer1_1_bn2_bias target=layer1.1.bn2.bias} ->[n42] constant,
     t15 f32 [D=128 H=64 W=3 C=3] {pt2=root:p_layer2_0_conv1_weight target=layer2.0.conv1.weight} ->[n47] constant,
     t16 f32 [C=128] {pt2=root:p_layer2_0_bn1_weight target=layer2.0.bn1.weight} ->[n51] constant,
     t17 f32 [C=128] {pt2=root:p_layer2_0_bn1_bias target=layer2.0.bn1.bias} ->[n51] constant,
     t18 f32 [D=128 H=128 W=3 C=3] {pt2=root:p_layer2_0_conv2_weight target=layer2.0.conv2.weight} ->[n55] constant,
     t19 f32 [C=128] {pt2=root:p_layer2_0_bn2_weight target=layer2.0.bn2.weight} ->[n59] constant,
     t20 f32 [C=128] {pt2=root:p_layer2_0_bn2_bias target=layer2.0.bn2.bias} ->[n59] constant,
     t21 f32 [D=128 H=64 W=1 C=1] {pt2=root:p_layer2_0_downsample_0_weight target=layer2.0.downsample.0.weight} ->[n62] constant,
     t22 f32 [C=128] {pt2=root:p_layer2_0_downsample_1_weight target=layer2.0.downsample.1.weight} ->[n66] constant,
     t23 f32 [C=128] {pt2=root:p_layer2_0_downsample_1_bias target=layer2.0.downsample.1.bias} ->[n66] constant,
     t24 f32 [D=128 H=128 W=3 C=3] {pt2=root:p_layer2_1_conv1_weight target=layer2.1.conv1.weight} ->[n71] constant,
     t25 f32 [C=128] {pt2=root:p_layer2_1_bn1_weight target=layer2.1.bn1.weight} ->[n75] constant,
     t26 f32 [C=128] {pt2=root:p_layer2_1_bn1_bias target=layer2.1.bn1.bias} ->[n75] constant,
     t27 f32 [D=128 H=128 W=3 C=3] {pt2=root:p_layer2_1_conv2_weight target=layer2.1.conv2.weight} ->[n79] constant,
     t28 f32 [C=128] {pt2=root:p_layer2_1_bn2_weight target=layer2.1.bn2.weight} ->[n83] constant,
     t29 f32 [C=128] {pt2=root:p_layer2_1_bn2_bias target=layer2.1.bn2.bias} ->[n83] constant,
     t30 f32 [D=256 H=128 W=3 C=3] {pt2=root:p_layer3_0_conv1_weight target=layer3.0.conv1.weight} ->[n88] constant,
     t31 f32 [C=256] {pt2=root:p_layer3_0_bn1_weight target=layer3.0.bn1.weight} ->[n92] constant,
     t32 f32 [C=256] {pt2=root:p_layer3_0_bn1_bias target=layer3.0.bn1.bias} ->[n92] constant,
     t33 f32 [D=256 H=256 W=3 C=3] {pt2=root:p_layer3_0_conv2_weight target=layer3.0.conv2.weight} ->[n96] constant,
     t34 f32 [C=256] {pt2=root:p_layer3_0_bn2_weight target=layer3.0.bn2.weight} ->[n100] constant,
     t35 f32 [C=256] {pt2=root:p_layer3_0_bn2_bias target=layer3.0.bn2.bias} ->[n100] constant,
     t36 f32 [D=256 H=128 W=1 C=1] {pt2=root:p_layer3_0_downsample_0_weight target=layer3.0.downsample.0.weight} ->[n103] constant,
     t37 f32 [C=256] {pt2=root:p_layer3_0_downsample_1_weight target=layer3.0.downsample.1.weight} ->[n107] constant,
     t38 f32 [C=256] {pt2=root:p_layer3_0_downsample_1_bias target=layer3.0.downsample.1.bias} ->[n107] constant,
     t39 f32 [D=256 H=256 W=3 C=3] {pt2=root:p_layer3_1_conv1_weight target=layer3.1.conv1.weight} ->[n112] constant,
     t40 f32 [C=256] {pt2=root:p_layer3_1_bn1_weight target=layer3.1.bn1.weight} ->[n116] constant,
     t41 f32 [C=256] {pt2=root:p_layer3_1_bn1_bias target=layer3.1.bn1.bias} ->[n116] constant,
     t42 f32 [D=256 H=256 W=3 C=3] {pt2=root:p_layer3_1_conv2_weight target=layer3.1.conv2.weight} ->[n120] constant,
     t43 f32 [C=256] {pt2=root:p_layer3_1_bn2_weight target=layer3.1.bn2.weight} ->[n124] constant,
     t44 f32 [C=256] {pt2=root:p_layer3_1_bn2_bias target=layer3.1.bn2.bias} ->[n124] constant,
     t45 f32 [D=512 H=256 W=3 C=3] {pt2=root:p_layer4_0_conv1_weight target=layer4.0.conv1.weight} ->[n129] constant,
     t46 f32 [C=512] {pt2=root:p_layer4_0_bn1_weight target=layer4.0.bn1.weight} ->[n133] constant,
     t47 f32 [C=512] {pt2=root:p_layer4_0_bn1_bias target=layer4.0.bn1.bias} ->[n133] constant,
     t48 f32 [D=512 H=512 W=3 C=3] {pt2=root:p_layer4_0_conv2_weight target=layer4.0.conv2.weight} ->[n137] constant,
     t49 f32 [C=512] {pt2=root:p_layer4_0_bn2_weight target=layer4.0.bn2.weight} ->[n141] constant,
     t50 f32 [C=512] {pt2=root:p_layer4_0_bn2_bias target=layer4.0.bn2.bias} ->[n141] constant,
     t51 f32 [D=512 H=256 W=1 C=1] {pt2=root:p_layer4_0_downsample_0_weight target=layer4.0.downsample.0.weight} ->[n144] constant,
     t52 f32 [C=512] {pt2=root:p_layer4_0_downsample_1_weight target=layer4.0.downsample.1.weight} ->[n148] constant,
     t53 f32 [C=512] {pt2=root:p_layer4_0_downsample_1_bias target=layer4.0.downsample.1.bias} ->[n148] constant,
     t54 f32 [D=512 H=512 W=3 C=3] {pt2=root:p_layer4_1_conv1_weight target=layer4.1.conv1.weight} ->[n153] constant,
     t55 f32 [C=512] {pt2=root:p_layer4_1_bn1_weight target=layer4.1.bn1.weight} ->[n157] constant,
     t56 f32 [C=512] {pt2=root:p_layer4_1_bn1_bias target=layer4.1.bn1.bias} ->[n157] constant,
     t57 f32 [D=512 H=512 W=3 C=3] {pt2=root:p_layer4_1_conv2_weight target=layer4.1.conv2.weight} ->[n161] constant,
     t58 f32 [C=512] {pt2=root:p_layer4_1_bn2_weight target=layer4.1.bn2.weight} ->[n165] constant,
     t59 f32 [C=512] {pt2=root:p_layer4_1_bn2_bias target=layer4.1.bn2.bias} ->[n165] constant,
     t60 f32 [W=1000 C=512] {pt2=root:p_fc_weight target=fc.weight} ->[n174] constant,
     t61 f32 [C=1000] {pt2=root:p_fc_bias target=fc.bias} ->[n173] constant,
     t62 f32 [C=64] {pt2=root:b_bn1_running_mean target=bn1.running_mean} ->[n5] constant,
     t63 f32 [C=64] {pt2=root:b_bn1_running_var target=bn1.running_var} ->[n5] constant,
     t65 f32 [C=64] {pt2=root:b_layer1_0_bn1_running_mean target=layer1.0.bn1.running_mean} ->[n17] constant,
     t66 f32 [C=64] {pt2=root:b_layer1_0_bn1_running_var target=layer1.0.bn1.running_var} ->[n17] constant,
     t68 f32 [C=64] {pt2=root:b_layer1_0_bn2_running_mean target=layer1.0.bn2.running_mean} ->[n25] constant,
     t69 f32 [C=64] {pt2=root:b_layer1_0_bn2_running_var target=layer1.0.bn2.running_var} ->[n25] constant,
     t71 f32 [C=64] {pt2=root:b_layer1_1_bn1_running_mean target=layer1.1.bn1.running_mean} ->[n34] constant,
     t72 f32 [C=64] {pt2=root:b_layer1_1_bn1_running_var target=layer1.1.bn1.running_var} ->[n34] constant,
     t74 f32 [C=64] {pt2=root:b_layer1_1_bn2_running_mean target=layer1.1.bn2.running_mean} ->[n42] constant,
     t75 f32 [C=64] {pt2=root:b_layer1_1_bn2_running_var target=layer1.1.bn2.running_var} ->[n42] constant,
     t77 f32 [C=128] {pt2=root:b_layer2_0_bn1_running_mean target=layer2.0.bn1.running_mean} ->[n51] constant,
     t78 f32 [C=128] {pt2=root:b_layer2_0_bn1_running_var target=layer2.0.bn1.running_var} ->[n51] constant,
     t80 f32 [C=128] {pt2=root:b_layer2_0_bn2_running_mean target=layer2.0.bn2.running_mean} ->[n59] constant,
     t81 f32 [C=128] {pt2=root:b_layer2_0_bn2_running_var target=layer2.0.bn2.running_var} ->[n59] constant,
     t83 f32 [C=128] {pt2=root:b_layer2_0_downsample_1_running_mean target=layer2.0.downsample.1.running_mean} ->[n66] constant,
     t84 f32 [C=128] {pt2=root:b_layer2_0_downsample_1_running_var target=layer2.0.downsample.1.running_var} ->[n66] constant,
     t86 f32 [C=128] {pt2=root:b_layer2_1_bn1_running_mean target=layer2.1.bn1.running_mean} ->[n75] constant,
     t87 f32 [C=128] {pt2=root:b_layer2_1_bn1_running_var target=layer2.1.bn1.running_var} ->[n75] constant,
     t89 f32 [C=128] {pt2=root:b_layer2_1_bn2_running_mean target=layer2.1.bn2.running_mean} ->[n83] constant,
     t90 f32 [C=128] {pt2=root:b_layer2_1_bn2_running_var target=layer2.1.bn2.running_var} ->[n83] constant,
     t92 f32 [C=256] {pt2=root:b_layer3_0_bn1_running_mean target=layer3.0.bn1.running_mean} ->[n92] constant,
     t93 f32 [C=256] {pt2=root:b_layer3_0_bn1_running_var target=layer3.0.bn1.running_var} ->[n92] constant,
     t95 f32 [C=256] {pt2=root:b_layer3_0_bn2_running_mean target=layer3.0.bn2.running_mean} ->[n100] constant,
     t96 f32 [C=256] {pt2=root:b_layer3_0_bn2_running_var target=layer3.0.bn2.running_var} ->[n100] constant,
     t98 f32 [C=256] {pt2=root:b_layer3_0_downsample_1_running_mean target=layer3.0.downsample.1.running_mean} ->[n107] constant,
     t99 f32 [C=256] {pt2=root:b_layer3_0_downsample_1_running_var target=layer3.0.downsample.1.running_var} ->[n107] constant,
     t101 f32 [C=256] {pt2=root:b_layer3_1_bn1_running_mean target=layer3.1.bn1.running_mean} ->[n116] constant,
     t102 f32 [C=256] {pt2=root:b_layer3_1_bn1_running_var target=layer3.1.bn1.running_var} ->[n116] constant,
     t104 f32 [C=256] {pt2=root:b_layer3_1_bn2_running_mean target=layer3.1.bn2.running_mean} ->[n124] constant,
     t105 f32 [C=256] {pt2=root:b_layer3_1_bn2_running_var target=layer3.1.bn2.running_var} ->[n124] constant,
     t107 f32 [C=512] {pt2=root:b_layer4_0_bn1_running_mean target=layer4.0.bn1.running_mean} ->[n133] constant,
     t108 f32 [C=512] {pt2=root:b_layer4_0_bn1_running_var target=layer4.0.bn1.running_var} ->[n133] constant,
     t110 f32 [C=512] {pt2=root:b_layer4_0_bn2_running_mean target=layer4.0.bn2.running_mean} ->[n141] constant,
     t111 f32 [C=512] {pt2=root:b_layer4_0_bn2_running_var target=layer4.0.bn2.running_var} ->[n141] constant,
     t113 f32 [C=512] {pt2=root:b_layer4_0_downsample_1_running_mean target=layer4.0.downsample.1.running_mean} ->[n148] constant,
     t114 f32 [C=512] {pt2=root:b_layer4_0_downsample_1_running_var target=layer4.0.downsample.1.running_var} ->[n148] constant,
     t116 f32 [C=512] {pt2=root:b_layer4_1_bn1_running_mean target=layer4.1.bn1.running_mean} ->[n157] constant,
     t117 f32 [C=512] {pt2=root:b_layer4_1_bn1_running_var target=layer4.1.bn1.running_var} ->[n157] constant,
     t119 f32 [C=512] {pt2=root:b_layer4_1_bn2_running_mean target=layer4.1.bn2.running_mean} ->[n165] constant,
     t120 f32 [C=512] {pt2=root:b_layer4_1_bn2_running_var target=layer4.1.bn2.running_var} ->[n165] constant,
     t122 f32 [H=3 W=224 C=224] {pt2=root:x} ->[n0]]
  nodes:
    group g1 torch.ops.aten.convolution.default:
      n0 {derived}: [t123 f32 [H=224 W=224 C=3] {derived} ->[n2]] =
        permute x=t122 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
      n1 {derived}: [t124 f32 [N=64 T=1 D=1 H=7 W=7 C=3] {derived} ->[n2]] =
        permute
          x=t0 {pt2=root:p_conv1_weight target=conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2 {derived}: [t125 f32 [H=112 W=112 C=64] {derived} ->[n5]] =
        convolution
          x=t123 {derived} <-n0
          weight=t124 {derived} <-n1
          bias=none
          params={stride={h=2; w=2};
                 padding={h=3; w=3};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g4 torch.ops.aten.convolution.default:
      n13 {derived}: [t136 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {derived} ->[n14]] =
        permute
          x=t3 {pt2=root:p_layer1_0_conv1_weight target=layer1.0.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n14 {derived}: [t137 f32 [H=56 W=56 C=64] {derived} ->[n17]] =
        convolution
          x=t132 {derived} <-n9
          weight=t136 {derived} <-n13
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g6 torch.ops.aten.convolution.default:
      n21 {derived}: [t144 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {derived} ->[n22]] =
        permute
          x=t6 {pt2=root:p_layer1_0_conv2_weight target=layer1.0.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n22 {derived}: [t145 f32 [H=56 W=56 C=64] {derived} ->[n25]] =
        convolution
          x=t298 {derived} <-n176
          weight=t144 {derived} <-n21
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g8 torch.ops.aten.convolution.default:
      n30 {derived}: [t153 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {derived} ->[n31]] =
        permute
          x=t9 {pt2=root:p_layer1_1_conv1_weight target=layer1.1.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n31 {derived}: [t154 f32 [H=56 W=56 C=64] {derived} ->[n34]] =
        convolution
          x=t300 {derived} <-n178
          weight=t153 {derived} <-n30
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g10 torch.ops.aten.convolution.default:
      n38 {derived}: [t161 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {derived} ->[n39]] =
        permute
          x=t12 {pt2=root:p_layer1_1_conv2_weight target=layer1.1.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n39 {derived}: [t162 f32 [H=56 W=56 C=64] {derived} ->[n42]] =
        convolution
          x=t301 {derived} <-n179
          weight=t161 {derived} <-n38
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g12 torch.ops.aten.convolution.default:
      n47 {derived}: [t170 f32 [N=128 T=1 D=1 H=3 W=3 C=64] {derived} ->[n48]] =
        permute
          x=t15 {pt2=root:p_layer2_0_conv1_weight target=layer2.0.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n48 {derived}: [t171 f32 [H=28 W=28 C=128] {derived} ->[n51]] =
        convolution
          x=t303 {derived} <-n181
          weight=t170 {derived} <-n47
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g14 torch.ops.aten.convolution.default:
      n55 {derived}: [t178 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {derived} ->[n56]] =
        permute
          x=t18 {pt2=root:p_layer2_0_conv2_weight target=layer2.0.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n56 {derived}: [t179 f32 [H=28 W=28 C=128] {derived} ->[n59]] =
        convolution
          x=t304 {derived} <-n182
          weight=t178 {derived} <-n55
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g16 torch.ops.aten.convolution.default:
      n62 {derived}: [t185 f32 [N=128 T=1 D=1 H=1 W=1 C=64] {derived} ->[n63]] =
        permute
          x=t21 {pt2=root:p_layer2_0_downsample_0_weight target=layer2.0.downsample.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n63 {derived}: [t186 f32 [H=28 W=28 C=128] {derived} ->[n66]] =
        convolution
          x=t303 {derived} <-n181
          weight=t185 {derived} <-n62
          bias=none
          params={stride={h=2; w=2};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g18 torch.ops.aten.convolution.default:
      n71 {derived}: [t194 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {derived} ->[n72]] =
        permute
          x=t24 {pt2=root:p_layer2_1_conv1_weight target=layer2.1.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n72 {derived}: [t195 f32 [H=28 W=28 C=128] {derived} ->[n75]] =
        convolution
          x=t306 {derived} <-n184
          weight=t194 {derived} <-n71
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g20 torch.ops.aten.convolution.default:
      n79 {derived}: [t202 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {derived} ->[n80]] =
        permute
          x=t27 {pt2=root:p_layer2_1_conv2_weight target=layer2.1.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n80 {derived}: [t203 f32 [H=28 W=28 C=128] {derived} ->[n83]] =
        convolution
          x=t307 {derived} <-n185
          weight=t202 {derived} <-n79
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g22 torch.ops.aten.convolution.default:
      n88 {derived}: [t211 f32 [N=256 T=1 D=1 H=3 W=3 C=128] {derived} ->[n89]] =
        permute
          x=t30 {pt2=root:p_layer3_0_conv1_weight target=layer3.0.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n89 {derived}: [t212 f32 [H=14 W=14 C=256] {derived} ->[n92]] =
        convolution
          x=t309 {derived} <-n187
          weight=t211 {derived} <-n88
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g24 torch.ops.aten.convolution.default:
      n96 {derived}: [t219 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {derived} ->[n97]] =
        permute
          x=t33 {pt2=root:p_layer3_0_conv2_weight target=layer3.0.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n97 {derived}: [t220 f32 [H=14 W=14 C=256] {derived} ->[n100]] =
        convolution
          x=t310 {derived} <-n188
          weight=t219 {derived} <-n96
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g26 torch.ops.aten.convolution.default:
      n103 {derived}: [t226 f32 [N=256 T=1 D=1 H=1 W=1 C=128] {derived} ->[n104]] =
        permute
          x=t36 {pt2=root:p_layer3_0_downsample_0_weight target=layer3.0.downsample.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n104 {derived}: [t227 f32 [H=14 W=14 C=256] {derived} ->[n107]] =
        convolution
          x=t309 {derived} <-n187
          weight=t226 {derived} <-n103
          bias=none
          params={stride={h=2; w=2};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g28 torch.ops.aten.convolution.default:
      n112 {derived}: [t235 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {derived} ->[n113]] =
        permute
          x=t39 {pt2=root:p_layer3_1_conv1_weight target=layer3.1.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n113 {derived}: [t236 f32 [H=14 W=14 C=256] {derived} ->[n116]] =
        convolution
          x=t312 {derived} <-n190
          weight=t235 {derived} <-n112
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g30 torch.ops.aten.convolution.default:
      n120 {derived}: [t243 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {derived} ->[n121]] =
        permute
          x=t42 {pt2=root:p_layer3_1_conv2_weight target=layer3.1.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n121 {derived}: [t244 f32 [H=14 W=14 C=256] {derived} ->[n124]] =
        convolution
          x=t313 {derived} <-n191
          weight=t243 {derived} <-n120
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g32 torch.ops.aten.convolution.default:
      n129 {derived}: [t252 f32 [N=512 T=1 D=1 H=3 W=3 C=256] {derived} ->[n130]] =
        permute
          x=t45 {pt2=root:p_layer4_0_conv1_weight target=layer4.0.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n130 {derived}: [t253 f32 [H=7 W=7 C=512] {derived} ->[n133]] =
        convolution
          x=t315 {derived} <-n193
          weight=t252 {derived} <-n129
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g34 torch.ops.aten.convolution.default:
      n137 {derived}: [t260 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {derived} ->[n138]] =
        permute
          x=t48 {pt2=root:p_layer4_0_conv2_weight target=layer4.0.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n138 {derived}: [t261 f32 [H=7 W=7 C=512] {derived} ->[n141]] =
        convolution
          x=t316 {derived} <-n194
          weight=t260 {derived} <-n137
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g36 torch.ops.aten.convolution.default:
      n144 {derived}: [t267 f32 [N=512 T=1 D=1 H=1 W=1 C=256] {derived} ->[n145]] =
        permute
          x=t51 {pt2=root:p_layer4_0_downsample_0_weight target=layer4.0.downsample.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n145 {derived}: [t268 f32 [H=7 W=7 C=512] {derived} ->[n148]] =
        convolution
          x=t315 {derived} <-n193
          weight=t267 {derived} <-n144
          bias=none
          params={stride={h=2; w=2};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g38 torch.ops.aten.convolution.default:
      n153 {derived}: [t276 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {derived} ->[n154]] =
        permute
          x=t54 {pt2=root:p_layer4_1_conv1_weight target=layer4.1.conv1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n154 {derived}: [t277 f32 [H=7 W=7 C=512] {derived} ->[n157]] =
        convolution
          x=t318 {derived} <-n196
          weight=t276 {derived} <-n153
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g40 torch.ops.aten.convolution.default:
      n161 {derived}: [t284 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {derived} ->[n162]] =
        permute
          x=t57 {pt2=root:p_layer4_1_conv2_weight target=layer4.1.conv2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n162 {derived}: [t285 f32 [H=7 W=7 C=512] {derived} ->[n165]] =
        convolution
          x=t319 {derived} <-n197
          weight=t284 {derived} <-n161
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n174 {pt2=root[68] torch.ops.aten.permute.default}: [t295 f32 [N=1000 T=1
                                                                   D=1 H=1 W=1
                                                                   C=512] {derived} ->[n173]] =
      permute x=t60 {pt2=root:p_fc_weight target=fc.weight} perm=[N<-W, W<-N]
    group g2 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n5 {derived}: [t128 f32 [H=112 W=112 C=64] {derived} ->[n175]] =
        batch_norm
          x=t125 {derived} <-n2
          weight=t1 {pt2=root:p_bn1_weight target=bn1.weight}
          bias=t2 {pt2=root:p_bn1_bias target=bn1.bias}
          running_mean=t62 {pt2=root:b_bn1_running_mean target=bn1.running_mean}
          running_var=t63 {pt2=root:b_bn1_running_var target=bn1.running_var}
          params={channel=C; eps=1e-05}
    n175 {pt2=root[2] torch.ops.aten.relu.default}: [t297 f32 [H=112 W=112
                                                               C=64] {derived} ->[n9]] =
      relu x=t128 {derived} <-n5
    group g3 torch.ops.aten.max_pool2d_with_indices.default:
      n9 {derived}: [t132 f32 [H=56 W=56 C=64] {derived} ->[n14, n177],
                     t133 f32 [H=56 W=56 C=64] {derived} ->[n10]] =
        max_pool2d_with_indices
          x=t297 {derived} <-n175
          params={kernel={h=3; w=3}; stride={h=2; w=2}; pad={h=1; w=1}}
      n10 {derived}: [] = discard x=t133 {derived} <-n9
    group g5 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n17 {derived}: [t140 f32 [H=56 W=56 C=64] {derived} ->[n176]] =
        batch_norm
          x=t137 {derived} <-n14
          weight=t4 {pt2=root:p_layer1_0_bn1_weight target=layer1.0.bn1.weight}
          bias=t5 {pt2=root:p_layer1_0_bn1_bias target=layer1.0.bn1.bias}
          running_mean=t65 {pt2=root:b_layer1_0_bn1_running_mean target=layer1.0.bn1.running_mean}
          running_var=t66 {pt2=root:b_layer1_0_bn1_running_var target=layer1.0.bn1.running_var}
          params={channel=C; eps=1e-05}
    n176 {pt2=root[6] torch.ops.aten.relu.default}: [t298 f32 [H=56 W=56 C=64] {derived} ->[n22]] =
      relu x=t140 {derived} <-n17
    group g7 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n25 {derived}: [t148 f32 [H=56 W=56 C=64] {derived} ->[n177]] =
        batch_norm
          x=t145 {derived} <-n22
          weight=t7 {pt2=root:p_layer1_0_bn2_weight target=layer1.0.bn2.weight}
          bias=t8 {pt2=root:p_layer1_0_bn2_bias target=layer1.0.bn2.bias}
          running_mean=t68 {pt2=root:b_layer1_0_bn2_running_mean target=layer1.0.bn2.running_mean}
          running_var=t69 {pt2=root:b_layer1_0_bn2_running_var target=layer1.0.bn2.running_var}
          params={channel=C; eps=1e-05}
    n177 {pt2=root[9] torch.ops.aten.add.Tensor}: [t299 f32 [H=56 W=56 C=64] {derived} ->[n178]] =
      add a=t148 {derived} <-n25 b=t132 {derived} <-n9
    n178 {pt2=root[10] torch.ops.aten.relu.default}: [t300 f32 [H=56 W=56 C=64] {derived} ->[n31,
                                                                      n180]] =
      relu x=t299 {derived} <-n177
    group g9 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n34 {derived}: [t157 f32 [H=56 W=56 C=64] {derived} ->[n179]] =
        batch_norm
          x=t154 {derived} <-n31
          weight=t10 {pt2=root:p_layer1_1_bn1_weight target=layer1.1.bn1.weight}
          bias=t11 {pt2=root:p_layer1_1_bn1_bias target=layer1.1.bn1.bias}
          running_mean=t71 {pt2=root:b_layer1_1_bn1_running_mean target=layer1.1.bn1.running_mean}
          running_var=t72 {pt2=root:b_layer1_1_bn1_running_var target=layer1.1.bn1.running_var}
          params={channel=C; eps=1e-05}
    n179 {pt2=root[13] torch.ops.aten.relu.default}: [t301 f32 [H=56 W=56 C=64] {derived} ->[n39]] =
      relu x=t157 {derived} <-n34
    group g11 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n42 {derived}: [t165 f32 [H=56 W=56 C=64] {derived} ->[n180]] =
        batch_norm
          x=t162 {derived} <-n39
          weight=t13 {pt2=root:p_layer1_1_bn2_weight target=layer1.1.bn2.weight}
          bias=t14 {pt2=root:p_layer1_1_bn2_bias target=layer1.1.bn2.bias}
          running_mean=t74 {pt2=root:b_layer1_1_bn2_running_mean target=layer1.1.bn2.running_mean}
          running_var=t75 {pt2=root:b_layer1_1_bn2_running_var target=layer1.1.bn2.running_var}
          params={channel=C; eps=1e-05}
    n180 {pt2=root[16] torch.ops.aten.add.Tensor}: [t302 f32 [H=56 W=56 C=64] {derived} ->[n181]] =
      add a=t165 {derived} <-n42 b=t300 {derived} <-n178
    n181 {pt2=root[17] torch.ops.aten.relu.default}: [t303 f32 [H=56 W=56 C=64] {derived} ->[n48,
                                                                      n63]] =
      relu x=t302 {derived} <-n180
    group g13 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n51 {derived}: [t174 f32 [H=28 W=28 C=128] {derived} ->[n182]] =
        batch_norm
          x=t171 {derived} <-n48
          weight=t16 {pt2=root:p_layer2_0_bn1_weight target=layer2.0.bn1.weight}
          bias=t17 {pt2=root:p_layer2_0_bn1_bias target=layer2.0.bn1.bias}
          running_mean=t77 {pt2=root:b_layer2_0_bn1_running_mean target=layer2.0.bn1.running_mean}
          running_var=t78 {pt2=root:b_layer2_0_bn1_running_var target=layer2.0.bn1.running_var}
          params={channel=C; eps=1e-05}
    group g17 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n66 {derived}: [t189 f32 [H=28 W=28 C=128] {derived} ->[n183]] =
        batch_norm
          x=t186 {derived} <-n63
          weight=t22 {pt2=root:p_layer2_0_downsample_1_weight target=layer2.0.downsample.1.weight}
          bias=t23 {pt2=root:p_layer2_0_downsample_1_bias target=layer2.0.downsample.1.bias}
          running_mean=t83 {pt2=root:b_layer2_0_downsample_1_running_mean target=layer2.0.downsample.1.running_mean}
          running_var=t84 {pt2=root:b_layer2_0_downsample_1_running_var target=layer2.0.downsample.1.running_var}
          params={channel=C; eps=1e-05}
    n182 {pt2=root[20] torch.ops.aten.relu.default}: [t304 f32 [H=28 W=28
                                                                C=128] {derived} ->[n56]] =
      relu x=t174 {derived} <-n51
    group g15 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n59 {derived}: [t182 f32 [H=28 W=28 C=128] {derived} ->[n183]] =
        batch_norm
          x=t179 {derived} <-n56
          weight=t19 {pt2=root:p_layer2_0_bn2_weight target=layer2.0.bn2.weight}
          bias=t20 {pt2=root:p_layer2_0_bn2_bias target=layer2.0.bn2.bias}
          running_mean=t80 {pt2=root:b_layer2_0_bn2_running_mean target=layer2.0.bn2.running_mean}
          running_var=t81 {pt2=root:b_layer2_0_bn2_running_var target=layer2.0.bn2.running_var}
          params={channel=C; eps=1e-05}
    n183 {pt2=root[25] torch.ops.aten.add.Tensor}: [t305 f32 [H=28 W=28 C=128] {derived} ->[n184]] =
      add a=t182 {derived} <-n59 b=t189 {derived} <-n66
    n184 {pt2=root[26] torch.ops.aten.relu.default}: [t306 f32 [H=28 W=28
                                                                C=128] {derived} ->[n72,
                                                                      n186]] =
      relu x=t305 {derived} <-n183
    group g19 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n75 {derived}: [t198 f32 [H=28 W=28 C=128] {derived} ->[n185]] =
        batch_norm
          x=t195 {derived} <-n72
          weight=t25 {pt2=root:p_layer2_1_bn1_weight target=layer2.1.bn1.weight}
          bias=t26 {pt2=root:p_layer2_1_bn1_bias target=layer2.1.bn1.bias}
          running_mean=t86 {pt2=root:b_layer2_1_bn1_running_mean target=layer2.1.bn1.running_mean}
          running_var=t87 {pt2=root:b_layer2_1_bn1_running_var target=layer2.1.bn1.running_var}
          params={channel=C; eps=1e-05}
    n185 {pt2=root[29] torch.ops.aten.relu.default}: [t307 f32 [H=28 W=28
                                                                C=128] {derived} ->[n80]] =
      relu x=t198 {derived} <-n75
    group g21 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n83 {derived}: [t206 f32 [H=28 W=28 C=128] {derived} ->[n186]] =
        batch_norm
          x=t203 {derived} <-n80
          weight=t28 {pt2=root:p_layer2_1_bn2_weight target=layer2.1.bn2.weight}
          bias=t29 {pt2=root:p_layer2_1_bn2_bias target=layer2.1.bn2.bias}
          running_mean=t89 {pt2=root:b_layer2_1_bn2_running_mean target=layer2.1.bn2.running_mean}
          running_var=t90 {pt2=root:b_layer2_1_bn2_running_var target=layer2.1.bn2.running_var}
          params={channel=C; eps=1e-05}
    n186 {pt2=root[32] torch.ops.aten.add.Tensor}: [t308 f32 [H=28 W=28 C=128] {derived} ->[n187]] =
      add a=t206 {derived} <-n83 b=t306 {derived} <-n184
    n187 {pt2=root[33] torch.ops.aten.relu.default}: [t309 f32 [H=28 W=28
                                                                C=128] {derived} ->[n89,
                                                                      n104]] =
      relu x=t308 {derived} <-n186
    group g23 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n92 {derived}: [t215 f32 [H=14 W=14 C=256] {derived} ->[n188]] =
        batch_norm
          x=t212 {derived} <-n89
          weight=t31 {pt2=root:p_layer3_0_bn1_weight target=layer3.0.bn1.weight}
          bias=t32 {pt2=root:p_layer3_0_bn1_bias target=layer3.0.bn1.bias}
          running_mean=t92 {pt2=root:b_layer3_0_bn1_running_mean target=layer3.0.bn1.running_mean}
          running_var=t93 {pt2=root:b_layer3_0_bn1_running_var target=layer3.0.bn1.running_var}
          params={channel=C; eps=1e-05}
    group g27 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n107 {derived}: [t230 f32 [H=14 W=14 C=256] {derived} ->[n189]] =
        batch_norm
          x=t227 {derived} <-n104
          weight=t37 {pt2=root:p_layer3_0_downsample_1_weight target=layer3.0.downsample.1.weight}
          bias=t38 {pt2=root:p_layer3_0_downsample_1_bias target=layer3.0.downsample.1.bias}
          running_mean=t98 {pt2=root:b_layer3_0_downsample_1_running_mean target=layer3.0.downsample.1.running_mean}
          running_var=t99 {pt2=root:b_layer3_0_downsample_1_running_var target=layer3.0.downsample.1.running_var}
          params={channel=C; eps=1e-05}
    n188 {pt2=root[36] torch.ops.aten.relu.default}: [t310 f32 [H=14 W=14
                                                                C=256] {derived} ->[n97]] =
      relu x=t215 {derived} <-n92
    group g25 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n100 {derived}: [t223 f32 [H=14 W=14 C=256] {derived} ->[n189]] =
        batch_norm
          x=t220 {derived} <-n97
          weight=t34 {pt2=root:p_layer3_0_bn2_weight target=layer3.0.bn2.weight}
          bias=t35 {pt2=root:p_layer3_0_bn2_bias target=layer3.0.bn2.bias}
          running_mean=t95 {pt2=root:b_layer3_0_bn2_running_mean target=layer3.0.bn2.running_mean}
          running_var=t96 {pt2=root:b_layer3_0_bn2_running_var target=layer3.0.bn2.running_var}
          params={channel=C; eps=1e-05}
    n189 {pt2=root[41] torch.ops.aten.add.Tensor}: [t311 f32 [H=14 W=14 C=256] {derived} ->[n190]] =
      add a=t223 {derived} <-n100 b=t230 {derived} <-n107
    n190 {pt2=root[42] torch.ops.aten.relu.default}: [t312 f32 [H=14 W=14
                                                                C=256] {derived} ->[n113,
                                                                      n192]] =
      relu x=t311 {derived} <-n189
    group g29 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n116 {derived}: [t239 f32 [H=14 W=14 C=256] {derived} ->[n191]] =
        batch_norm
          x=t236 {derived} <-n113
          weight=t40 {pt2=root:p_layer3_1_bn1_weight target=layer3.1.bn1.weight}
          bias=t41 {pt2=root:p_layer3_1_bn1_bias target=layer3.1.bn1.bias}
          running_mean=t101 {pt2=root:b_layer3_1_bn1_running_mean target=layer3.1.bn1.running_mean}
          running_var=t102 {pt2=root:b_layer3_1_bn1_running_var target=layer3.1.bn1.running_var}
          params={channel=C; eps=1e-05}
    n191 {pt2=root[45] torch.ops.aten.relu.default}: [t313 f32 [H=14 W=14
                                                                C=256] {derived} ->[n121]] =
      relu x=t239 {derived} <-n116
    group g31 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n124 {derived}: [t247 f32 [H=14 W=14 C=256] {derived} ->[n192]] =
        batch_norm
          x=t244 {derived} <-n121
          weight=t43 {pt2=root:p_layer3_1_bn2_weight target=layer3.1.bn2.weight}
          bias=t44 {pt2=root:p_layer3_1_bn2_bias target=layer3.1.bn2.bias}
          running_mean=t104 {pt2=root:b_layer3_1_bn2_running_mean target=layer3.1.bn2.running_mean}
          running_var=t105 {pt2=root:b_layer3_1_bn2_running_var target=layer3.1.bn2.running_var}
          params={channel=C; eps=1e-05}
    n192 {pt2=root[48] torch.ops.aten.add.Tensor}: [t314 f32 [H=14 W=14 C=256] {derived} ->[n193]] =
      add a=t247 {derived} <-n124 b=t312 {derived} <-n190
    n193 {pt2=root[49] torch.ops.aten.relu.default}: [t315 f32 [H=14 W=14
                                                                C=256] {derived} ->[n130,
                                                                      n145]] =
      relu x=t314 {derived} <-n192
    group g33 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n133 {derived}: [t256 f32 [H=7 W=7 C=512] {derived} ->[n194]] =
        batch_norm
          x=t253 {derived} <-n130
          weight=t46 {pt2=root:p_layer4_0_bn1_weight target=layer4.0.bn1.weight}
          bias=t47 {pt2=root:p_layer4_0_bn1_bias target=layer4.0.bn1.bias}
          running_mean=t107 {pt2=root:b_layer4_0_bn1_running_mean target=layer4.0.bn1.running_mean}
          running_var=t108 {pt2=root:b_layer4_0_bn1_running_var target=layer4.0.bn1.running_var}
          params={channel=C; eps=1e-05}
    group g37 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n148 {derived}: [t271 f32 [H=7 W=7 C=512] {derived} ->[n195]] =
        batch_norm
          x=t268 {derived} <-n145
          weight=t52 {pt2=root:p_layer4_0_downsample_1_weight target=layer4.0.downsample.1.weight}
          bias=t53 {pt2=root:p_layer4_0_downsample_1_bias target=layer4.0.downsample.1.bias}
          running_mean=t113 {pt2=root:b_layer4_0_downsample_1_running_mean target=layer4.0.downsample.1.running_mean}
          running_var=t114 {pt2=root:b_layer4_0_downsample_1_running_var target=layer4.0.downsample.1.running_var}
          params={channel=C; eps=1e-05}
    n194 {pt2=root[52] torch.ops.aten.relu.default}: [t316 f32 [H=7 W=7 C=512] {derived} ->[n138]] =
      relu x=t256 {derived} <-n133
    group g35 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n141 {derived}: [t264 f32 [H=7 W=7 C=512] {derived} ->[n195]] =
        batch_norm
          x=t261 {derived} <-n138
          weight=t49 {pt2=root:p_layer4_0_bn2_weight target=layer4.0.bn2.weight}
          bias=t50 {pt2=root:p_layer4_0_bn2_bias target=layer4.0.bn2.bias}
          running_mean=t110 {pt2=root:b_layer4_0_bn2_running_mean target=layer4.0.bn2.running_mean}
          running_var=t111 {pt2=root:b_layer4_0_bn2_running_var target=layer4.0.bn2.running_var}
          params={channel=C; eps=1e-05}
    n195 {pt2=root[57] torch.ops.aten.add.Tensor}: [t317 f32 [H=7 W=7 C=512] {derived} ->[n196]] =
      add a=t264 {derived} <-n141 b=t271 {derived} <-n148
    n196 {pt2=root[58] torch.ops.aten.relu.default}: [t318 f32 [H=7 W=7 C=512] {derived} ->[n154,
                                                                      n198]] =
      relu x=t317 {derived} <-n195
    group g39 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n157 {derived}: [t280 f32 [H=7 W=7 C=512] {derived} ->[n197]] =
        batch_norm
          x=t277 {derived} <-n154
          weight=t55 {pt2=root:p_layer4_1_bn1_weight target=layer4.1.bn1.weight}
          bias=t56 {pt2=root:p_layer4_1_bn1_bias target=layer4.1.bn1.bias}
          running_mean=t116 {pt2=root:b_layer4_1_bn1_running_mean target=layer4.1.bn1.running_mean}
          running_var=t117 {pt2=root:b_layer4_1_bn1_running_var target=layer4.1.bn1.running_var}
          params={channel=C; eps=1e-05}
    n197 {pt2=root[61] torch.ops.aten.relu.default}: [t319 f32 [H=7 W=7 C=512] {derived} ->[n162]] =
      relu x=t280 {derived} <-n157
    group g41 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n165 {derived}: [t288 f32 [H=7 W=7 C=512] {derived} ->[n198]] =
        batch_norm
          x=t285 {derived} <-n162
          weight=t58 {pt2=root:p_layer4_1_bn2_weight target=layer4.1.bn2.weight}
          bias=t59 {pt2=root:p_layer4_1_bn2_bias target=layer4.1.bn2.bias}
          running_mean=t119 {pt2=root:b_layer4_1_bn2_running_mean target=layer4.1.bn2.running_mean}
          running_var=t120 {pt2=root:b_layer4_1_bn2_running_var target=layer4.1.bn2.running_var}
          params={channel=C; eps=1e-05}
    n198 {pt2=root[64] torch.ops.aten.add.Tensor}: [t320 f32 [H=7 W=7 C=512] {derived} ->[n199]] =
      add a=t288 {derived} <-n165 b=t318 {derived} <-n196
    n199 {pt2=root[65] torch.ops.aten.relu.default}: [t321 f32 [H=7 W=7 C=512] {derived} ->[n200]] =
      relu x=t320 {derived} <-n198
    n200 {pt2=root[66] torch.ops.aten.mean.dim}: [t322 f32 [C=512] {pt2=root:view} ->[n173]] =
      mean x=t321 {derived} <-n199 params={dims=[W, H]; keepdim=true}
    group g42 torch.ops.aten.addmm.default:
      n173 {pt2=root[69] torch.ops.aten.addmm.default}: [t296 f32 [C=1000] {pt2=root:addmm}] =
        linear
          x=t322 {pt2=root:view} <-n200
          weight=t295 {derived} <-n174
          bias=t61 {pt2=root:p_fc_bias target=fc.bias}
          params={in_features=512}
  outputs: [t296 f32 [C=1000] {pt2=root:addmm} <-n173]

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
    [t61 f32 [C=1000] {pt2=root:p_fc_bias target=fc.bias} ->[n173] constant,
     t122 f32 [H=3 W=224 C=224] {pt2=root:x} ->[n0],
     t295 f32 [N=1000 T=1 D=1 H=1 W=1 C=512] {folded from=[p_fc_weight]} ->[n173] constant,
     t297 f32 [N=64 T=1 D=1 H=7 W=7 C=3] {folded from=[p_conv1_weight,p_bn1_weight,b_bn1_running_var]} ->[n174] constant,
     t298 f32 [C=64] {folded from=[p_bn1_weight,p_bn1_bias,b_bn1_running_mean,b_bn1_running_var]} ->[n174] constant,
     t299 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_0_conv1_weight,p_layer1_0_bn1_weight,b_layer1_0_bn1_running_var]} ->[n176] constant,
     t300 f32 [C=64] {folded from=[p_layer1_0_bn1_weight,p_layer1_0_bn1_bias,b_layer1_0_bn1_running_mean,b_layer1_0_bn1_running_var]} ->[n176] constant,
     t301 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_0_conv2_weight,p_layer1_0_bn2_weight,b_layer1_0_bn2_running_var]} ->[n178] constant,
     t302 f32 [C=64] {folded from=[p_layer1_0_bn2_weight,p_layer1_0_bn2_bias,b_layer1_0_bn2_running_mean,b_layer1_0_bn2_running_var]} ->[n178] constant,
     t303 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_1_conv1_weight,p_layer1_1_bn1_weight,b_layer1_1_bn1_running_var]} ->[n181] constant,
     t304 f32 [C=64] {folded from=[p_layer1_1_bn1_weight,p_layer1_1_bn1_bias,b_layer1_1_bn1_running_mean,b_layer1_1_bn1_running_var]} ->[n181] constant,
     t305 f32 [N=64 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer1_1_conv2_weight,p_layer1_1_bn2_weight,b_layer1_1_bn2_running_var]} ->[n183] constant,
     t306 f32 [C=64] {folded from=[p_layer1_1_bn2_weight,p_layer1_1_bn2_bias,b_layer1_1_bn2_running_mean,b_layer1_1_bn2_running_var]} ->[n183] constant,
     t307 f32 [N=128 T=1 D=1 H=3 W=3 C=64] {folded from=[p_layer2_0_conv1_weight,p_layer2_0_bn1_weight,b_layer2_0_bn1_running_var]} ->[n186] constant,
     t308 f32 [C=128] {folded from=[p_layer2_0_bn1_weight,p_layer2_0_bn1_bias,b_layer2_0_bn1_running_mean,b_layer2_0_bn1_running_var]} ->[n186] constant,
     t309 f32 [N=128 T=1 D=1 H=1 W=1 C=64] {folded from=[p_layer2_0_downsample_0_weight,p_layer2_0_downsample_1_weight,b_layer2_0_downsample_1_running_var]} ->[n187] constant,
     t310 f32 [C=128] {folded from=[p_layer2_0_downsample_1_weight,p_layer2_0_downsample_1_bias,b_layer2_0_downsample_1_running_mean,b_layer2_0_downsample_1_running_var]} ->[n187] constant,
     t311 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer2_0_conv2_weight,p_layer2_0_bn2_weight,b_layer2_0_bn2_running_var]} ->[n189] constant,
     t312 f32 [C=128] {folded from=[p_layer2_0_bn2_weight,p_layer2_0_bn2_bias,b_layer2_0_bn2_running_mean,b_layer2_0_bn2_running_var]} ->[n189] constant,
     t313 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer2_1_conv1_weight,p_layer2_1_bn1_weight,b_layer2_1_bn1_running_var]} ->[n192] constant,
     t314 f32 [C=128] {folded from=[p_layer2_1_bn1_weight,p_layer2_1_bn1_bias,b_layer2_1_bn1_running_mean,b_layer2_1_bn1_running_var]} ->[n192] constant,
     t315 f32 [N=128 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer2_1_conv2_weight,p_layer2_1_bn2_weight,b_layer2_1_bn2_running_var]} ->[n194] constant,
     t316 f32 [C=128] {folded from=[p_layer2_1_bn2_weight,p_layer2_1_bn2_bias,b_layer2_1_bn2_running_mean,b_layer2_1_bn2_running_var]} ->[n194] constant,
     t317 f32 [N=256 T=1 D=1 H=3 W=3 C=128] {folded from=[p_layer3_0_conv1_weight,p_layer3_0_bn1_weight,b_layer3_0_bn1_running_var]} ->[n197] constant,
     t318 f32 [C=256] {folded from=[p_layer3_0_bn1_weight,p_layer3_0_bn1_bias,b_layer3_0_bn1_running_mean,b_layer3_0_bn1_running_var]} ->[n197] constant,
     t319 f32 [N=256 T=1 D=1 H=1 W=1 C=128] {folded from=[p_layer3_0_downsample_0_weight,p_layer3_0_downsample_1_weight,b_layer3_0_downsample_1_running_var]} ->[n198] constant,
     t320 f32 [C=256] {folded from=[p_layer3_0_downsample_1_weight,p_layer3_0_downsample_1_bias,b_layer3_0_downsample_1_running_mean,b_layer3_0_downsample_1_running_var]} ->[n198] constant,
     t321 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer3_0_conv2_weight,p_layer3_0_bn2_weight,b_layer3_0_bn2_running_var]} ->[n200] constant,
     t322 f32 [C=256] {folded from=[p_layer3_0_bn2_weight,p_layer3_0_bn2_bias,b_layer3_0_bn2_running_mean,b_layer3_0_bn2_running_var]} ->[n200] constant,
     t323 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer3_1_conv1_weight,p_layer3_1_bn1_weight,b_layer3_1_bn1_running_var]} ->[n203] constant,
     t324 f32 [C=256] {folded from=[p_layer3_1_bn1_weight,p_layer3_1_bn1_bias,b_layer3_1_bn1_running_mean,b_layer3_1_bn1_running_var]} ->[n203] constant,
     t325 f32 [N=256 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer3_1_conv2_weight,p_layer3_1_bn2_weight,b_layer3_1_bn2_running_var]} ->[n205] constant,
     t326 f32 [C=256] {folded from=[p_layer3_1_bn2_weight,p_layer3_1_bn2_bias,b_layer3_1_bn2_running_mean,b_layer3_1_bn2_running_var]} ->[n205] constant,
     t327 f32 [N=512 T=1 D=1 H=3 W=3 C=256] {folded from=[p_layer4_0_conv1_weight,p_layer4_0_bn1_weight,b_layer4_0_bn1_running_var]} ->[n208] constant,
     t328 f32 [C=512] {folded from=[p_layer4_0_bn1_weight,p_layer4_0_bn1_bias,b_layer4_0_bn1_running_mean,b_layer4_0_bn1_running_var]} ->[n208] constant,
     t329 f32 [N=512 T=1 D=1 H=1 W=1 C=256] {folded from=[p_layer4_0_downsample_0_weight,p_layer4_0_downsample_1_weight,b_layer4_0_downsample_1_running_var]} ->[n209] constant,
     t330 f32 [C=512] {folded from=[p_layer4_0_downsample_1_weight,p_layer4_0_downsample_1_bias,b_layer4_0_downsample_1_running_mean,b_layer4_0_downsample_1_running_var]} ->[n209] constant,
     t331 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {folded from=[p_layer4_0_conv2_weight,p_layer4_0_bn2_weight,b_layer4_0_bn2_running_var]} ->[n211] constant,
     t332 f32 [C=512] {folded from=[p_layer4_0_bn2_weight,p_layer4_0_bn2_bias,b_layer4_0_bn2_running_mean,b_layer4_0_bn2_running_var]} ->[n211] constant,
     t333 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {folded from=[p_layer4_1_conv1_weight,p_layer4_1_bn1_weight,b_layer4_1_bn1_running_var]} ->[n214] constant,
     t334 f32 [C=512] {folded from=[p_layer4_1_bn1_weight,p_layer4_1_bn1_bias,b_layer4_1_bn1_running_mean,b_layer4_1_bn1_running_var]} ->[n214] constant,
     t335 f32 [N=512 T=1 D=1 H=3 W=3 C=512] {folded from=[p_layer4_1_conv2_weight,p_layer4_1_bn2_weight,b_layer4_1_bn2_running_var]} ->[n216] constant,
     t336 f32 [C=512] {folded from=[p_layer4_1_bn2_weight,p_layer4_1_bn2_bias,b_layer4_1_bn2_running_mean,b_layer4_1_bn2_running_var]} ->[n216] constant]
  nodes:
    group g1 torch.ops.aten.convolution.default:
      n0 {derived}: [t123 f32 [H=224 W=224 C=3] {derived} ->[n174]] =
        permute x=t122 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
    n174 {derived}: [t337 f32 [H=112 W=112 C=64] {derived} ->[n175]] =
      convolution
        x=t123 {derived} <-n0
        weight=t297 {folded from=[p_conv1_weight,p_bn1_weight,b_bn1_running_var]}
        bias=t298 {folded from=[p_bn1_weight,p_bn1_bias,b_bn1_running_mean,b_bn1_running_var]}
        params={stride={h=2; w=2};
               padding={h=3; w=3};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n175 {pt2=root[2] torch.ops.aten.relu.default}: [t338 f32 [H=112 W=112
                                                               C=64] {derived} ->[n9]] =
      relu x=t337 {derived} <-n174
    group g3 torch.ops.aten.max_pool2d_with_indices.default:
      n9 {derived}: [t132 f32 [H=56 W=56 C=64] {derived} ->[n176, n179],
                     t133 f32 [H=56 W=56 C=64] {derived} ->[n10]] =
        max_pool2d_with_indices
          x=t338 {derived} <-n175
          params={kernel={h=3; w=3}; stride={h=2; w=2}; pad={h=1; w=1}}
      n10 {derived}: [] = discard x=t133 {derived} <-n9
    n176 {derived}: [t339 f32 [H=56 W=56 C=64] {derived} ->[n177]] =
      convolution
        x=t132 {derived} <-n9
        weight=t299 {folded from=[p_layer1_0_conv1_weight,p_layer1_0_bn1_weight,b_layer1_0_bn1_running_var]}
        bias=t300 {folded from=[p_layer1_0_bn1_weight,p_layer1_0_bn1_bias,b_layer1_0_bn1_running_mean,b_layer1_0_bn1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n177 {pt2=root[6] torch.ops.aten.relu.default}: [t340 f32 [H=56 W=56 C=64] {derived} ->[n178]] =
      relu x=t339 {derived} <-n176
    n178 {derived}: [t341 f32 [H=56 W=56 C=64] {derived} ->[n179]] =
      convolution
        x=t340 {derived} <-n177
        weight=t301 {folded from=[p_layer1_0_conv2_weight,p_layer1_0_bn2_weight,b_layer1_0_bn2_running_var]}
        bias=t302 {folded from=[p_layer1_0_bn2_weight,p_layer1_0_bn2_bias,b_layer1_0_bn2_running_mean,b_layer1_0_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n179 {pt2=root[9] torch.ops.aten.add.Tensor}: [t342 f32 [H=56 W=56 C=64] {derived} ->[n180]] =
      add a=t341 {derived} <-n178 b=t132 {derived} <-n9
    n180 {pt2=root[10] torch.ops.aten.relu.default}: [t343 f32 [H=56 W=56 C=64] {derived} ->[n181,
                                                                      n184]] =
      relu x=t342 {derived} <-n179
    n181 {derived}: [t344 f32 [H=56 W=56 C=64] {derived} ->[n182]] =
      convolution
        x=t343 {derived} <-n180
        weight=t303 {folded from=[p_layer1_1_conv1_weight,p_layer1_1_bn1_weight,b_layer1_1_bn1_running_var]}
        bias=t304 {folded from=[p_layer1_1_bn1_weight,p_layer1_1_bn1_bias,b_layer1_1_bn1_running_mean,b_layer1_1_bn1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n182 {pt2=root[13] torch.ops.aten.relu.default}: [t345 f32 [H=56 W=56 C=64] {derived} ->[n183]] =
      relu x=t344 {derived} <-n181
    n183 {derived}: [t346 f32 [H=56 W=56 C=64] {derived} ->[n184]] =
      convolution
        x=t345 {derived} <-n182
        weight=t305 {folded from=[p_layer1_1_conv2_weight,p_layer1_1_bn2_weight,b_layer1_1_bn2_running_var]}
        bias=t306 {folded from=[p_layer1_1_bn2_weight,p_layer1_1_bn2_bias,b_layer1_1_bn2_running_mean,b_layer1_1_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n184 {pt2=root[16] torch.ops.aten.add.Tensor}: [t347 f32 [H=56 W=56 C=64] {derived} ->[n185]] =
      add a=t346 {derived} <-n183 b=t343 {derived} <-n180
    n185 {pt2=root[17] torch.ops.aten.relu.default}: [t348 f32 [H=56 W=56 C=64] {derived} ->[n186,
                                                                      n187]] =
      relu x=t347 {derived} <-n184
    n186 {derived}: [t349 f32 [H=28 W=28 C=128] {derived} ->[n188]] =
      convolution
        x=t348 {derived} <-n185
        weight=t307 {folded from=[p_layer2_0_conv1_weight,p_layer2_0_bn1_weight,b_layer2_0_bn1_running_var]}
        bias=t308 {folded from=[p_layer2_0_bn1_weight,p_layer2_0_bn1_bias,b_layer2_0_bn1_running_mean,b_layer2_0_bn1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n187 {derived}: [t350 f32 [H=28 W=28 C=128] {derived} ->[n190]] =
      convolution
        x=t348 {derived} <-n185
        weight=t309 {folded from=[p_layer2_0_downsample_0_weight,p_layer2_0_downsample_1_weight,b_layer2_0_downsample_1_running_var]}
        bias=t310 {folded from=[p_layer2_0_downsample_1_weight,p_layer2_0_downsample_1_bias,b_layer2_0_downsample_1_running_mean,b_layer2_0_downsample_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n188 {pt2=root[20] torch.ops.aten.relu.default}: [t351 f32 [H=28 W=28
                                                                C=128] {derived} ->[n189]] =
      relu x=t349 {derived} <-n186
    n189 {derived}: [t352 f32 [H=28 W=28 C=128] {derived} ->[n190]] =
      convolution
        x=t351 {derived} <-n188
        weight=t311 {folded from=[p_layer2_0_conv2_weight,p_layer2_0_bn2_weight,b_layer2_0_bn2_running_var]}
        bias=t312 {folded from=[p_layer2_0_bn2_weight,p_layer2_0_bn2_bias,b_layer2_0_bn2_running_mean,b_layer2_0_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n190 {pt2=root[25] torch.ops.aten.add.Tensor}: [t353 f32 [H=28 W=28 C=128] {derived} ->[n191]] =
      add a=t352 {derived} <-n189 b=t350 {derived} <-n187
    n191 {pt2=root[26] torch.ops.aten.relu.default}: [t354 f32 [H=28 W=28
                                                                C=128] {derived} ->[n192,
                                                                      n195]] =
      relu x=t353 {derived} <-n190
    n192 {derived}: [t355 f32 [H=28 W=28 C=128] {derived} ->[n193]] =
      convolution
        x=t354 {derived} <-n191
        weight=t313 {folded from=[p_layer2_1_conv1_weight,p_layer2_1_bn1_weight,b_layer2_1_bn1_running_var]}
        bias=t314 {folded from=[p_layer2_1_bn1_weight,p_layer2_1_bn1_bias,b_layer2_1_bn1_running_mean,b_layer2_1_bn1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n193 {pt2=root[29] torch.ops.aten.relu.default}: [t356 f32 [H=28 W=28
                                                                C=128] {derived} ->[n194]] =
      relu x=t355 {derived} <-n192
    n194 {derived}: [t357 f32 [H=28 W=28 C=128] {derived} ->[n195]] =
      convolution
        x=t356 {derived} <-n193
        weight=t315 {folded from=[p_layer2_1_conv2_weight,p_layer2_1_bn2_weight,b_layer2_1_bn2_running_var]}
        bias=t316 {folded from=[p_layer2_1_bn2_weight,p_layer2_1_bn2_bias,b_layer2_1_bn2_running_mean,b_layer2_1_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n195 {pt2=root[32] torch.ops.aten.add.Tensor}: [t358 f32 [H=28 W=28 C=128] {derived} ->[n196]] =
      add a=t357 {derived} <-n194 b=t354 {derived} <-n191
    n196 {pt2=root[33] torch.ops.aten.relu.default}: [t359 f32 [H=28 W=28
                                                                C=128] {derived} ->[n197,
                                                                      n198]] =
      relu x=t358 {derived} <-n195
    n197 {derived}: [t360 f32 [H=14 W=14 C=256] {derived} ->[n199]] =
      convolution
        x=t359 {derived} <-n196
        weight=t317 {folded from=[p_layer3_0_conv1_weight,p_layer3_0_bn1_weight,b_layer3_0_bn1_running_var]}
        bias=t318 {folded from=[p_layer3_0_bn1_weight,p_layer3_0_bn1_bias,b_layer3_0_bn1_running_mean,b_layer3_0_bn1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n198 {derived}: [t361 f32 [H=14 W=14 C=256] {derived} ->[n201]] =
      convolution
        x=t359 {derived} <-n196
        weight=t319 {folded from=[p_layer3_0_downsample_0_weight,p_layer3_0_downsample_1_weight,b_layer3_0_downsample_1_running_var]}
        bias=t320 {folded from=[p_layer3_0_downsample_1_weight,p_layer3_0_downsample_1_bias,b_layer3_0_downsample_1_running_mean,b_layer3_0_downsample_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n199 {pt2=root[36] torch.ops.aten.relu.default}: [t362 f32 [H=14 W=14
                                                                C=256] {derived} ->[n200]] =
      relu x=t360 {derived} <-n197
    n200 {derived}: [t363 f32 [H=14 W=14 C=256] {derived} ->[n201]] =
      convolution
        x=t362 {derived} <-n199
        weight=t321 {folded from=[p_layer3_0_conv2_weight,p_layer3_0_bn2_weight,b_layer3_0_bn2_running_var]}
        bias=t322 {folded from=[p_layer3_0_bn2_weight,p_layer3_0_bn2_bias,b_layer3_0_bn2_running_mean,b_layer3_0_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n201 {pt2=root[41] torch.ops.aten.add.Tensor}: [t364 f32 [H=14 W=14 C=256] {derived} ->[n202]] =
      add a=t363 {derived} <-n200 b=t361 {derived} <-n198
    n202 {pt2=root[42] torch.ops.aten.relu.default}: [t365 f32 [H=14 W=14
                                                                C=256] {derived} ->[n203,
                                                                      n206]] =
      relu x=t364 {derived} <-n201
    n203 {derived}: [t366 f32 [H=14 W=14 C=256] {derived} ->[n204]] =
      convolution
        x=t365 {derived} <-n202
        weight=t323 {folded from=[p_layer3_1_conv1_weight,p_layer3_1_bn1_weight,b_layer3_1_bn1_running_var]}
        bias=t324 {folded from=[p_layer3_1_bn1_weight,p_layer3_1_bn1_bias,b_layer3_1_bn1_running_mean,b_layer3_1_bn1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n204 {pt2=root[45] torch.ops.aten.relu.default}: [t367 f32 [H=14 W=14
                                                                C=256] {derived} ->[n205]] =
      relu x=t366 {derived} <-n203
    n205 {derived}: [t368 f32 [H=14 W=14 C=256] {derived} ->[n206]] =
      convolution
        x=t367 {derived} <-n204
        weight=t325 {folded from=[p_layer3_1_conv2_weight,p_layer3_1_bn2_weight,b_layer3_1_bn2_running_var]}
        bias=t326 {folded from=[p_layer3_1_bn2_weight,p_layer3_1_bn2_bias,b_layer3_1_bn2_running_mean,b_layer3_1_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n206 {pt2=root[48] torch.ops.aten.add.Tensor}: [t369 f32 [H=14 W=14 C=256] {derived} ->[n207]] =
      add a=t368 {derived} <-n205 b=t365 {derived} <-n202
    n207 {pt2=root[49] torch.ops.aten.relu.default}: [t370 f32 [H=14 W=14
                                                                C=256] {derived} ->[n208,
                                                                      n209]] =
      relu x=t369 {derived} <-n206
    n208 {derived}: [t371 f32 [H=7 W=7 C=512] {derived} ->[n210]] =
      convolution
        x=t370 {derived} <-n207
        weight=t327 {folded from=[p_layer4_0_conv1_weight,p_layer4_0_bn1_weight,b_layer4_0_bn1_running_var]}
        bias=t328 {folded from=[p_layer4_0_bn1_weight,p_layer4_0_bn1_bias,b_layer4_0_bn1_running_mean,b_layer4_0_bn1_running_var]}
        params={stride={h=2; w=2};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n209 {derived}: [t372 f32 [H=7 W=7 C=512] {derived} ->[n212]] =
      convolution
        x=t370 {derived} <-n207
        weight=t329 {folded from=[p_layer4_0_downsample_0_weight,p_layer4_0_downsample_1_weight,b_layer4_0_downsample_1_running_var]}
        bias=t330 {folded from=[p_layer4_0_downsample_1_weight,p_layer4_0_downsample_1_bias,b_layer4_0_downsample_1_running_mean,b_layer4_0_downsample_1_running_var]}
        params={stride={h=2; w=2};
               padding={h=0; w=0};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n210 {pt2=root[52] torch.ops.aten.relu.default}: [t373 f32 [H=7 W=7 C=512] {derived} ->[n211]] =
      relu x=t371 {derived} <-n208
    n211 {derived}: [t374 f32 [H=7 W=7 C=512] {derived} ->[n212]] =
      convolution
        x=t373 {derived} <-n210
        weight=t331 {folded from=[p_layer4_0_conv2_weight,p_layer4_0_bn2_weight,b_layer4_0_bn2_running_var]}
        bias=t332 {folded from=[p_layer4_0_bn2_weight,p_layer4_0_bn2_bias,b_layer4_0_bn2_running_mean,b_layer4_0_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n212 {pt2=root[57] torch.ops.aten.add.Tensor}: [t375 f32 [H=7 W=7 C=512] {derived} ->[n213]] =
      add a=t374 {derived} <-n211 b=t372 {derived} <-n209
    n213 {pt2=root[58] torch.ops.aten.relu.default}: [t376 f32 [H=7 W=7 C=512] {derived} ->[n214,
                                                                      n217]] =
      relu x=t375 {derived} <-n212
    n214 {derived}: [t377 f32 [H=7 W=7 C=512] {derived} ->[n215]] =
      convolution
        x=t376 {derived} <-n213
        weight=t333 {folded from=[p_layer4_1_conv1_weight,p_layer4_1_bn1_weight,b_layer4_1_bn1_running_var]}
        bias=t334 {folded from=[p_layer4_1_bn1_weight,p_layer4_1_bn1_bias,b_layer4_1_bn1_running_mean,b_layer4_1_bn1_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n215 {pt2=root[61] torch.ops.aten.relu.default}: [t378 f32 [H=7 W=7 C=512] {derived} ->[n216]] =
      relu x=t377 {derived} <-n214
    n216 {derived}: [t379 f32 [H=7 W=7 C=512] {derived} ->[n217]] =
      convolution
        x=t378 {derived} <-n215
        weight=t335 {folded from=[p_layer4_1_conv2_weight,p_layer4_1_bn2_weight,b_layer4_1_bn2_running_var]}
        bias=t336 {folded from=[p_layer4_1_bn2_weight,p_layer4_1_bn2_bias,b_layer4_1_bn2_running_mean,b_layer4_1_bn2_running_var]}
        params={stride={h=1; w=1};
               padding={h=1; w=1};
               dilation={h=1; w=1};
               transposed=false;
               output_padding={h=0; w=0};
               groups=1}
    n217 {pt2=root[64] torch.ops.aten.add.Tensor}: [t380 f32 [H=7 W=7 C=512] {derived} ->[n218]] =
      add a=t379 {derived} <-n216 b=t376 {derived} <-n213
    n218 {pt2=root[65] torch.ops.aten.relu.default}: [t381 f32 [H=7 W=7 C=512] {derived} ->[n219]] =
      relu x=t380 {derived} <-n217
    n219 {pt2=root[66] torch.ops.aten.mean.dim}: [t382 f32 [C=512] {pt2=root:view} ->[n173]] =
      mean x=t381 {derived} <-n218 params={dims=[W, H]; keepdim=true}
    group g42 torch.ops.aten.addmm.default:
      n173 {pt2=root[69] torch.ops.aten.addmm.default}: [t296 f32 [C=1000] {pt2=root:addmm}] =
        linear
          x=t382 {pt2=root:view} <-n219
          weight=t295 {folded from=[p_fc_weight]}
          bias=t61 {pt2=root:p_fc_bias target=fc.bias}
          params={in_features=512}
  outputs: [t296 f32 [C=1000] {pt2=root:addmm} <-n173]

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
    [t0 f32 [D=16 H=3 W=3 C=3] {pt2=root:p_features_0_0_weight target=features.0.0.weight} ->[n1] constant,
     t1 f32 [C=16] {pt2=root:p_features_0_1_weight target=features.0.1.weight} ->[n5] constant,
     t2 f32 [C=16] {pt2=root:p_features_0_1_bias target=features.0.1.bias} ->[n5] constant,
     t3 f32 [D=16 H=1 W=3 C=3] {pt2=root:p_features_1_block_0_0_weight target=features.1.block.0.0.weight} ->[n13] constant,
     t4 f32 [C=16] {pt2=root:p_features_1_block_0_1_weight target=features.1.block.0.1.weight} ->[n17] constant,
     t5 f32 [C=16] {pt2=root:p_features_1_block_0_1_bias target=features.1.block.0.1.bias} ->[n17] constant,
     t6 f32 [D=8 H=16 W=1 C=1] {pt2=root:p_features_1_block_1_fc1_weight target=features.1.block.1.fc1.weight} ->[n22] constant,
     t7 f32 [C=8] {pt2=root:p_features_1_block_1_fc1_bias target=features.1.block.1.fc1.bias} ->[n23] constant,
     t8 f32 [D=16 H=8 W=1 C=1] {pt2=root:p_features_1_block_1_fc2_weight target=features.1.block.1.fc2.weight} ->[n27] constant,
     t9 f32 [C=16] {pt2=root:p_features_1_block_1_fc2_bias target=features.1.block.1.fc2.bias} ->[n28] constant,
     t10 f32 [D=16 H=16 W=1 C=1] {pt2=root:p_features_1_block_2_0_weight target=features.1.block.2.0.weight} ->[n36] constant,
     t11 f32 [C=16] {pt2=root:p_features_1_block_2_1_weight target=features.1.block.2.1.weight} ->[n40] constant,
     t12 f32 [C=16] {pt2=root:p_features_1_block_2_1_bias target=features.1.block.2.1.bias} ->[n40] constant,
     t13 f32 [D=72 H=16 W=1 C=1] {pt2=root:p_features_2_block_0_0_weight target=features.2.block.0.0.weight} ->[n43] constant,
     t14 f32 [C=72] {pt2=root:p_features_2_block_0_1_weight target=features.2.block.0.1.weight} ->[n47] constant,
     t15 f32 [C=72] {pt2=root:p_features_2_block_0_1_bias target=features.2.block.0.1.bias} ->[n47] constant,
     t16 f32 [D=72 H=1 W=3 C=3] {pt2=root:p_features_2_block_1_0_weight target=features.2.block.1.0.weight} ->[n51] constant,
     t17 f32 [C=72] {pt2=root:p_features_2_block_1_1_weight target=features.2.block.1.1.weight} ->[n55] constant,
     t18 f32 [C=72] {pt2=root:p_features_2_block_1_1_bias target=features.2.block.1.1.bias} ->[n55] constant,
     t19 f32 [D=24 H=72 W=1 C=1] {pt2=root:p_features_2_block_2_0_weight target=features.2.block.2.0.weight} ->[n59] constant,
     t20 f32 [C=24] {pt2=root:p_features_2_block_2_1_weight target=features.2.block.2.1.weight} ->[n63] constant,
     t21 f32 [C=24] {pt2=root:p_features_2_block_2_1_bias target=features.2.block.2.1.bias} ->[n63] constant,
     t22 f32 [D=88 H=24 W=1 C=1] {pt2=root:p_features_3_block_0_0_weight target=features.3.block.0.0.weight} ->[n66] constant,
     t23 f32 [C=88] {pt2=root:p_features_3_block_0_1_weight target=features.3.block.0.1.weight} ->[n70] constant,
     t24 f32 [C=88] {pt2=root:p_features_3_block_0_1_bias target=features.3.block.0.1.bias} ->[n70] constant,
     t25 f32 [D=88 H=1 W=3 C=3] {pt2=root:p_features_3_block_1_0_weight target=features.3.block.1.0.weight} ->[n74] constant,
     t26 f32 [C=88] {pt2=root:p_features_3_block_1_1_weight target=features.3.block.1.1.weight} ->[n78] constant,
     t27 f32 [C=88] {pt2=root:p_features_3_block_1_1_bias target=features.3.block.1.1.bias} ->[n78] constant,
     t28 f32 [D=24 H=88 W=1 C=1] {pt2=root:p_features_3_block_2_0_weight target=features.3.block.2.0.weight} ->[n82] constant,
     t29 f32 [C=24] {pt2=root:p_features_3_block_2_1_weight target=features.3.block.2.1.weight} ->[n86] constant,
     t30 f32 [C=24] {pt2=root:p_features_3_block_2_1_bias target=features.3.block.2.1.bias} ->[n86] constant,
     t31 f32 [D=96 H=24 W=1 C=1] {pt2=root:p_features_4_block_0_0_weight target=features.4.block.0.0.weight} ->[n90] constant,
     t32 f32 [C=96] {pt2=root:p_features_4_block_0_1_weight target=features.4.block.0.1.weight} ->[n94] constant,
     t33 f32 [C=96] {pt2=root:p_features_4_block_0_1_bias target=features.4.block.0.1.bias} ->[n94] constant,
     t34 f32 [D=96 H=1 W=5 C=5] {pt2=root:p_features_4_block_1_0_weight target=features.4.block.1.0.weight} ->[n102] constant,
     t35 f32 [C=96] {pt2=root:p_features_4_block_1_1_weight target=features.4.block.1.1.weight} ->[n106] constant,
     t36 f32 [C=96] {pt2=root:p_features_4_block_1_1_bias target=features.4.block.1.1.bias} ->[n106] constant,
     t37 f32 [D=24 H=96 W=1 C=1] {pt2=root:p_features_4_block_2_fc1_weight target=features.4.block.2.fc1.weight} ->[n115] constant,
     t38 f32 [C=24] {pt2=root:p_features_4_block_2_fc1_bias target=features.4.block.2.fc1.bias} ->[n116] constant,
     t39 f32 [D=96 H=24 W=1 C=1] {pt2=root:p_features_4_block_2_fc2_weight target=features.4.block.2.fc2.weight} ->[n120] constant,
     t40 f32 [C=96] {pt2=root:p_features_4_block_2_fc2_bias target=features.4.block.2.fc2.bias} ->[n121] constant,
     t41 f32 [D=40 H=96 W=1 C=1] {pt2=root:p_features_4_block_3_0_weight target=features.4.block.3.0.weight} ->[n129] constant,
     t42 f32 [C=40] {pt2=root:p_features_4_block_3_1_weight target=features.4.block.3.1.weight} ->[n133] constant,
     t43 f32 [C=40] {pt2=root:p_features_4_block_3_1_bias target=features.4.block.3.1.bias} ->[n133] constant,
     t44 f32 [D=240 H=40 W=1 C=1] {pt2=root:p_features_5_block_0_0_weight target=features.5.block.0.0.weight} ->[n136] constant,
     t45 f32 [C=240] {pt2=root:p_features_5_block_0_1_weight target=features.5.block.0.1.weight} ->[n140] constant,
     t46 f32 [C=240] {pt2=root:p_features_5_block_0_1_bias target=features.5.block.0.1.bias} ->[n140] constant,
     t47 f32 [D=240 H=1 W=5 C=5] {pt2=root:p_features_5_block_1_0_weight target=features.5.block.1.0.weight} ->[n148] constant,
     t48 f32 [C=240] {pt2=root:p_features_5_block_1_1_weight target=features.5.block.1.1.weight} ->[n152] constant,
     t49 f32 [C=240] {pt2=root:p_features_5_block_1_1_bias target=features.5.block.1.1.bias} ->[n152] constant,
     t50 f32 [D=64 H=240 W=1 C=1] {pt2=root:p_features_5_block_2_fc1_weight target=features.5.block.2.fc1.weight} ->[n161] constant,
     t51 f32 [C=64] {pt2=root:p_features_5_block_2_fc1_bias target=features.5.block.2.fc1.bias} ->[n162] constant,
     t52 f32 [D=240 H=64 W=1 C=1] {pt2=root:p_features_5_block_2_fc2_weight target=features.5.block.2.fc2.weight} ->[n166] constant,
     t53 f32 [C=240] {pt2=root:p_features_5_block_2_fc2_bias target=features.5.block.2.fc2.bias} ->[n167] constant,
     t54 f32 [D=40 H=240 W=1 C=1] {pt2=root:p_features_5_block_3_0_weight target=features.5.block.3.0.weight} ->[n175] constant,
     t55 f32 [C=40] {pt2=root:p_features_5_block_3_1_weight target=features.5.block.3.1.weight} ->[n179] constant,
     t56 f32 [C=40] {pt2=root:p_features_5_block_3_1_bias target=features.5.block.3.1.bias} ->[n179] constant,
     t57 f32 [D=240 H=40 W=1 C=1] {pt2=root:p_features_6_block_0_0_weight target=features.6.block.0.0.weight} ->[n183] constant,
     t58 f32 [C=240] {pt2=root:p_features_6_block_0_1_weight target=features.6.block.0.1.weight} ->[n187] constant,
     t59 f32 [C=240] {pt2=root:p_features_6_block_0_1_bias target=features.6.block.0.1.bias} ->[n187] constant,
     t60 f32 [D=240 H=1 W=5 C=5] {pt2=root:p_features_6_block_1_0_weight target=features.6.block.1.0.weight} ->[n195] constant,
     t61 f32 [C=240] {pt2=root:p_features_6_block_1_1_weight target=features.6.block.1.1.weight} ->[n199] constant,
     t62 f32 [C=240] {pt2=root:p_features_6_block_1_1_bias target=features.6.block.1.1.bias} ->[n199] constant,
     t63 f32 [D=64 H=240 W=1 C=1] {pt2=root:p_features_6_block_2_fc1_weight target=features.6.block.2.fc1.weight} ->[n208] constant,
     t64 f32 [C=64] {pt2=root:p_features_6_block_2_fc1_bias target=features.6.block.2.fc1.bias} ->[n209] constant,
     t65 f32 [D=240 H=64 W=1 C=1] {pt2=root:p_features_6_block_2_fc2_weight target=features.6.block.2.fc2.weight} ->[n213] constant,
     t66 f32 [C=240] {pt2=root:p_features_6_block_2_fc2_bias target=features.6.block.2.fc2.bias} ->[n214] constant,
     t67 f32 [D=40 H=240 W=1 C=1] {pt2=root:p_features_6_block_3_0_weight target=features.6.block.3.0.weight} ->[n222] constant,
     t68 f32 [C=40] {pt2=root:p_features_6_block_3_1_weight target=features.6.block.3.1.weight} ->[n226] constant,
     t69 f32 [C=40] {pt2=root:p_features_6_block_3_1_bias target=features.6.block.3.1.bias} ->[n226] constant,
     t70 f32 [D=120 H=40 W=1 C=1] {pt2=root:p_features_7_block_0_0_weight target=features.7.block.0.0.weight} ->[n230] constant,
     t71 f32 [C=120] {pt2=root:p_features_7_block_0_1_weight target=features.7.block.0.1.weight} ->[n234] constant,
     t72 f32 [C=120] {pt2=root:p_features_7_block_0_1_bias target=features.7.block.0.1.bias} ->[n234] constant,
     t73 f32 [D=120 H=1 W=5 C=5] {pt2=root:p_features_7_block_1_0_weight target=features.7.block.1.0.weight} ->[n242] constant,
     t74 f32 [C=120] {pt2=root:p_features_7_block_1_1_weight target=features.7.block.1.1.weight} ->[n246] constant,
     t75 f32 [C=120] {pt2=root:p_features_7_block_1_1_bias target=features.7.block.1.1.bias} ->[n246] constant,
     t76 f32 [D=32 H=120 W=1 C=1] {pt2=root:p_features_7_block_2_fc1_weight target=features.7.block.2.fc1.weight} ->[n255] constant,
     t77 f32 [C=32] {pt2=root:p_features_7_block_2_fc1_bias target=features.7.block.2.fc1.bias} ->[n256] constant,
     t78 f32 [D=120 H=32 W=1 C=1] {pt2=root:p_features_7_block_2_fc2_weight target=features.7.block.2.fc2.weight} ->[n260] constant,
     t79 f32 [C=120] {pt2=root:p_features_7_block_2_fc2_bias target=features.7.block.2.fc2.bias} ->[n261] constant,
     t80 f32 [D=48 H=120 W=1 C=1] {pt2=root:p_features_7_block_3_0_weight target=features.7.block.3.0.weight} ->[n269] constant,
     t81 f32 [C=48] {pt2=root:p_features_7_block_3_1_weight target=features.7.block.3.1.weight} ->[n273] constant,
     t82 f32 [C=48] {pt2=root:p_features_7_block_3_1_bias target=features.7.block.3.1.bias} ->[n273] constant,
     t83 f32 [D=144 H=48 W=1 C=1] {pt2=root:p_features_8_block_0_0_weight target=features.8.block.0.0.weight} ->[n276] constant,
     t84 f32 [C=144] {pt2=root:p_features_8_block_0_1_weight target=features.8.block.0.1.weight} ->[n280] constant,
     t85 f32 [C=144] {pt2=root:p_features_8_block_0_1_bias target=features.8.block.0.1.bias} ->[n280] constant,
     t86 f32 [D=144 H=1 W=5 C=5] {pt2=root:p_features_8_block_1_0_weight target=features.8.block.1.0.weight} ->[n288] constant,
     t87 f32 [C=144] {pt2=root:p_features_8_block_1_1_weight target=features.8.block.1.1.weight} ->[n292] constant,
     t88 f32 [C=144] {pt2=root:p_features_8_block_1_1_bias target=features.8.block.1.1.bias} ->[n292] constant,
     t89 f32 [D=40 H=144 W=1 C=1] {pt2=root:p_features_8_block_2_fc1_weight target=features.8.block.2.fc1.weight} ->[n301] constant,
     t90 f32 [C=40] {pt2=root:p_features_8_block_2_fc1_bias target=features.8.block.2.fc1.bias} ->[n302] constant,
     t91 f32 [D=144 H=40 W=1 C=1] {pt2=root:p_features_8_block_2_fc2_weight target=features.8.block.2.fc2.weight} ->[n306] constant,
     t92 f32 [C=144] {pt2=root:p_features_8_block_2_fc2_bias target=features.8.block.2.fc2.bias} ->[n307] constant,
     t93 f32 [D=48 H=144 W=1 C=1] {pt2=root:p_features_8_block_3_0_weight target=features.8.block.3.0.weight} ->[n315] constant,
     t94 f32 [C=48] {pt2=root:p_features_8_block_3_1_weight target=features.8.block.3.1.weight} ->[n319] constant,
     t95 f32 [C=48] {pt2=root:p_features_8_block_3_1_bias target=features.8.block.3.1.bias} ->[n319] constant,
     t96 f32 [D=288 H=48 W=1 C=1] {pt2=root:p_features_9_block_0_0_weight target=features.9.block.0.0.weight} ->[n323] constant,
     t97 f32 [C=288] {pt2=root:p_features_9_block_0_1_weight target=features.9.block.0.1.weight} ->[n327] constant,
     t98 f32 [C=288] {pt2=root:p_features_9_block_0_1_bias target=features.9.block.0.1.bias} ->[n327] constant,
     t99 f32 [D=288 H=1 W=5 C=5] {pt2=root:p_features_9_block_1_0_weight target=features.9.block.1.0.weight} ->[n335] constant,
     t100 f32 [C=288] {pt2=root:p_features_9_block_1_1_weight target=features.9.block.1.1.weight} ->[n339] constant,
     t101 f32 [C=288] {pt2=root:p_features_9_block_1_1_bias target=features.9.block.1.1.bias} ->[n339] constant,
     t102 f32 [D=72 H=288 W=1 C=1] {pt2=root:p_features_9_block_2_fc1_weight target=features.9.block.2.fc1.weight} ->[n348] constant,
     t103 f32 [C=72] {pt2=root:p_features_9_block_2_fc1_bias target=features.9.block.2.fc1.bias} ->[n349] constant,
     t104 f32 [D=288 H=72 W=1 C=1] {pt2=root:p_features_9_block_2_fc2_weight target=features.9.block.2.fc2.weight} ->[n353] constant,
     t105 f32 [C=288] {pt2=root:p_features_9_block_2_fc2_bias target=features.9.block.2.fc2.bias} ->[n354] constant,
     t106 f32 [D=96 H=288 W=1 C=1] {pt2=root:p_features_9_block_3_0_weight target=features.9.block.3.0.weight} ->[n362] constant,
     t107 f32 [C=96] {pt2=root:p_features_9_block_3_1_weight target=features.9.block.3.1.weight} ->[n366] constant,
     t108 f32 [C=96] {pt2=root:p_features_9_block_3_1_bias target=features.9.block.3.1.bias} ->[n366] constant,
     t109 f32 [D=576 H=96 W=1 C=1] {pt2=root:p_features_10_block_0_0_weight target=features.10.block.0.0.weight} ->[n369] constant,
     t110 f32 [C=576] {pt2=root:p_features_10_block_0_1_weight target=features.10.block.0.1.weight} ->[n373] constant,
     t111 f32 [C=576] {pt2=root:p_features_10_block_0_1_bias target=features.10.block.0.1.bias} ->[n373] constant,
     t112 f32 [D=576 H=1 W=5 C=5] {pt2=root:p_features_10_block_1_0_weight target=features.10.block.1.0.weight} ->[n381] constant,
     t113 f32 [C=576] {pt2=root:p_features_10_block_1_1_weight target=features.10.block.1.1.weight} ->[n385] constant,
     t114 f32 [C=576] {pt2=root:p_features_10_block_1_1_bias target=features.10.block.1.1.bias} ->[n385] constant,
     t115 f32 [D=144 H=576 W=1 C=1] {pt2=root:p_features_10_block_2_fc1_weight target=features.10.block.2.fc1.weight} ->[n394] constant,
     t116 f32 [C=144] {pt2=root:p_features_10_block_2_fc1_bias target=features.10.block.2.fc1.bias} ->[n395] constant,
     t117 f32 [D=576 H=144 W=1 C=1] {pt2=root:p_features_10_block_2_fc2_weight target=features.10.block.2.fc2.weight} ->[n399] constant,
     t118 f32 [C=576] {pt2=root:p_features_10_block_2_fc2_bias target=features.10.block.2.fc2.bias} ->[n400] constant,
     t119 f32 [D=96 H=576 W=1 C=1] {pt2=root:p_features_10_block_3_0_weight target=features.10.block.3.0.weight} ->[n408] constant,
     t120 f32 [C=96] {pt2=root:p_features_10_block_3_1_weight target=features.10.block.3.1.weight} ->[n412] constant,
     t121 f32 [C=96] {pt2=root:p_features_10_block_3_1_bias target=features.10.block.3.1.bias} ->[n412] constant,
     t122 f32 [D=576 H=96 W=1 C=1] {pt2=root:p_features_11_block_0_0_weight target=features.11.block.0.0.weight} ->[n416] constant,
     t123 f32 [C=576] {pt2=root:p_features_11_block_0_1_weight target=features.11.block.0.1.weight} ->[n420] constant,
     t124 f32 [C=576] {pt2=root:p_features_11_block_0_1_bias target=features.11.block.0.1.bias} ->[n420] constant,
     t125 f32 [D=576 H=1 W=5 C=5] {pt2=root:p_features_11_block_1_0_weight target=features.11.block.1.0.weight} ->[n428] constant,
     t126 f32 [C=576] {pt2=root:p_features_11_block_1_1_weight target=features.11.block.1.1.weight} ->[n432] constant,
     t127 f32 [C=576] {pt2=root:p_features_11_block_1_1_bias target=features.11.block.1.1.bias} ->[n432] constant,
     t128 f32 [D=144 H=576 W=1 C=1] {pt2=root:p_features_11_block_2_fc1_weight target=features.11.block.2.fc1.weight} ->[n441] constant,
     t129 f32 [C=144] {pt2=root:p_features_11_block_2_fc1_bias target=features.11.block.2.fc1.bias} ->[n442] constant,
     t130 f32 [D=576 H=144 W=1 C=1] {pt2=root:p_features_11_block_2_fc2_weight target=features.11.block.2.fc2.weight} ->[n446] constant,
     t131 f32 [C=576] {pt2=root:p_features_11_block_2_fc2_bias target=features.11.block.2.fc2.bias} ->[n447] constant,
     t132 f32 [D=96 H=576 W=1 C=1] {pt2=root:p_features_11_block_3_0_weight target=features.11.block.3.0.weight} ->[n455] constant,
     t133 f32 [C=96] {pt2=root:p_features_11_block_3_1_weight target=features.11.block.3.1.weight} ->[n459] constant,
     t134 f32 [C=96] {pt2=root:p_features_11_block_3_1_bias target=features.11.block.3.1.bias} ->[n459] constant,
     t135 f32 [D=576 H=96 W=1 C=1] {pt2=root:p_features_12_0_weight target=features.12.0.weight} ->[n463] constant,
     t136 f32 [C=576] {pt2=root:p_features_12_1_weight target=features.12.1.weight} ->[n467] constant,
     t137 f32 [C=576] {pt2=root:p_features_12_1_bias target=features.12.1.bias} ->[n467] constant,
     t138 f32 [W=1024 C=576] {pt2=root:p_classifier_0_weight target=classifier.0.weight} ->[n488] constant,
     t139 f32 [C=1024] {pt2=root:p_classifier_0_bias target=classifier.0.bias} ->[n478] constant,
     t140 f32 [W=1000 C=1024] {pt2=root:p_classifier_3_weight target=classifier.3.weight} ->[n489] constant,
     t141 f32 [C=1000] {pt2=root:p_classifier_3_bias target=classifier.3.bias} ->[n487] constant,
     t142 f32 [C=16] {pt2=root:b_features_0_1_running_mean target=features.0.1.running_mean} ->[n5] constant,
     t143 f32 [C=16] {pt2=root:b_features_0_1_running_var target=features.0.1.running_var} ->[n5] constant,
     t145 f32 [C=16] {pt2=root:b_features_1_block_0_1_running_mean target=features.1.block.0.1.running_mean} ->[n17] constant,
     t146 f32 [C=16] {pt2=root:b_features_1_block_0_1_running_var target=features.1.block.0.1.running_var} ->[n17] constant,
     t148 f32 [C=16] {pt2=root:b_features_1_block_2_1_running_mean target=features.1.block.2.1.running_mean} ->[n40] constant,
     t149 f32 [C=16] {pt2=root:b_features_1_block_2_1_running_var target=features.1.block.2.1.running_var} ->[n40] constant,
     t151 f32 [C=72] {pt2=root:b_features_2_block_0_1_running_mean target=features.2.block.0.1.running_mean} ->[n47] constant,
     t152 f32 [C=72] {pt2=root:b_features_2_block_0_1_running_var target=features.2.block.0.1.running_var} ->[n47] constant,
     t154 f32 [C=72] {pt2=root:b_features_2_block_1_1_running_mean target=features.2.block.1.1.running_mean} ->[n55] constant,
     t155 f32 [C=72] {pt2=root:b_features_2_block_1_1_running_var target=features.2.block.1.1.running_var} ->[n55] constant,
     t157 f32 [C=24] {pt2=root:b_features_2_block_2_1_running_mean target=features.2.block.2.1.running_mean} ->[n63] constant,
     t158 f32 [C=24] {pt2=root:b_features_2_block_2_1_running_var target=features.2.block.2.1.running_var} ->[n63] constant,
     t160 f32 [C=88] {pt2=root:b_features_3_block_0_1_running_mean target=features.3.block.0.1.running_mean} ->[n70] constant,
     t161 f32 [C=88] {pt2=root:b_features_3_block_0_1_running_var target=features.3.block.0.1.running_var} ->[n70] constant,
     t163 f32 [C=88] {pt2=root:b_features_3_block_1_1_running_mean target=features.3.block.1.1.running_mean} ->[n78] constant,
     t164 f32 [C=88] {pt2=root:b_features_3_block_1_1_running_var target=features.3.block.1.1.running_var} ->[n78] constant,
     t166 f32 [C=24] {pt2=root:b_features_3_block_2_1_running_mean target=features.3.block.2.1.running_mean} ->[n86] constant,
     t167 f32 [C=24] {pt2=root:b_features_3_block_2_1_running_var target=features.3.block.2.1.running_var} ->[n86] constant,
     t169 f32 [C=96] {pt2=root:b_features_4_block_0_1_running_mean target=features.4.block.0.1.running_mean} ->[n94] constant,
     t170 f32 [C=96] {pt2=root:b_features_4_block_0_1_running_var target=features.4.block.0.1.running_var} ->[n94] constant,
     t172 f32 [C=96] {pt2=root:b_features_4_block_1_1_running_mean target=features.4.block.1.1.running_mean} ->[n106] constant,
     t173 f32 [C=96] {pt2=root:b_features_4_block_1_1_running_var target=features.4.block.1.1.running_var} ->[n106] constant,
     t175 f32 [C=40] {pt2=root:b_features_4_block_3_1_running_mean target=features.4.block.3.1.running_mean} ->[n133] constant,
     t176 f32 [C=40] {pt2=root:b_features_4_block_3_1_running_var target=features.4.block.3.1.running_var} ->[n133] constant,
     t178 f32 [C=240] {pt2=root:b_features_5_block_0_1_running_mean target=features.5.block.0.1.running_mean} ->[n140] constant,
     t179 f32 [C=240] {pt2=root:b_features_5_block_0_1_running_var target=features.5.block.0.1.running_var} ->[n140] constant,
     t181 f32 [C=240] {pt2=root:b_features_5_block_1_1_running_mean target=features.5.block.1.1.running_mean} ->[n152] constant,
     t182 f32 [C=240] {pt2=root:b_features_5_block_1_1_running_var target=features.5.block.1.1.running_var} ->[n152] constant,
     t184 f32 [C=40] {pt2=root:b_features_5_block_3_1_running_mean target=features.5.block.3.1.running_mean} ->[n179] constant,
     t185 f32 [C=40] {pt2=root:b_features_5_block_3_1_running_var target=features.5.block.3.1.running_var} ->[n179] constant,
     t187 f32 [C=240] {pt2=root:b_features_6_block_0_1_running_mean target=features.6.block.0.1.running_mean} ->[n187] constant,
     t188 f32 [C=240] {pt2=root:b_features_6_block_0_1_running_var target=features.6.block.0.1.running_var} ->[n187] constant,
     t190 f32 [C=240] {pt2=root:b_features_6_block_1_1_running_mean target=features.6.block.1.1.running_mean} ->[n199] constant,
     t191 f32 [C=240] {pt2=root:b_features_6_block_1_1_running_var target=features.6.block.1.1.running_var} ->[n199] constant,
     t193 f32 [C=40] {pt2=root:b_features_6_block_3_1_running_mean target=features.6.block.3.1.running_mean} ->[n226] constant,
     t194 f32 [C=40] {pt2=root:b_features_6_block_3_1_running_var target=features.6.block.3.1.running_var} ->[n226] constant,
     t196 f32 [C=120] {pt2=root:b_features_7_block_0_1_running_mean target=features.7.block.0.1.running_mean} ->[n234] constant,
     t197 f32 [C=120] {pt2=root:b_features_7_block_0_1_running_var target=features.7.block.0.1.running_var} ->[n234] constant,
     t199 f32 [C=120] {pt2=root:b_features_7_block_1_1_running_mean target=features.7.block.1.1.running_mean} ->[n246] constant,
     t200 f32 [C=120] {pt2=root:b_features_7_block_1_1_running_var target=features.7.block.1.1.running_var} ->[n246] constant,
     t202 f32 [C=48] {pt2=root:b_features_7_block_3_1_running_mean target=features.7.block.3.1.running_mean} ->[n273] constant,
     t203 f32 [C=48] {pt2=root:b_features_7_block_3_1_running_var target=features.7.block.3.1.running_var} ->[n273] constant,
     t205 f32 [C=144] {pt2=root:b_features_8_block_0_1_running_mean target=features.8.block.0.1.running_mean} ->[n280] constant,
     t206 f32 [C=144] {pt2=root:b_features_8_block_0_1_running_var target=features.8.block.0.1.running_var} ->[n280] constant,
     t208 f32 [C=144] {pt2=root:b_features_8_block_1_1_running_mean target=features.8.block.1.1.running_mean} ->[n292] constant,
     t209 f32 [C=144] {pt2=root:b_features_8_block_1_1_running_var target=features.8.block.1.1.running_var} ->[n292] constant,
     t211 f32 [C=48] {pt2=root:b_features_8_block_3_1_running_mean target=features.8.block.3.1.running_mean} ->[n319] constant,
     t212 f32 [C=48] {pt2=root:b_features_8_block_3_1_running_var target=features.8.block.3.1.running_var} ->[n319] constant,
     t214 f32 [C=288] {pt2=root:b_features_9_block_0_1_running_mean target=features.9.block.0.1.running_mean} ->[n327] constant,
     t215 f32 [C=288] {pt2=root:b_features_9_block_0_1_running_var target=features.9.block.0.1.running_var} ->[n327] constant,
     t217 f32 [C=288] {pt2=root:b_features_9_block_1_1_running_mean target=features.9.block.1.1.running_mean} ->[n339] constant,
     t218 f32 [C=288] {pt2=root:b_features_9_block_1_1_running_var target=features.9.block.1.1.running_var} ->[n339] constant,
     t220 f32 [C=96] {pt2=root:b_features_9_block_3_1_running_mean target=features.9.block.3.1.running_mean} ->[n366] constant,
     t221 f32 [C=96] {pt2=root:b_features_9_block_3_1_running_var target=features.9.block.3.1.running_var} ->[n366] constant,
     t223 f32 [C=576] {pt2=root:b_features_10_block_0_1_running_mean target=features.10.block.0.1.running_mean} ->[n373] constant,
     t224 f32 [C=576] {pt2=root:b_features_10_block_0_1_running_var target=features.10.block.0.1.running_var} ->[n373] constant,
     t226 f32 [C=576] {pt2=root:b_features_10_block_1_1_running_mean target=features.10.block.1.1.running_mean} ->[n385] constant,
     t227 f32 [C=576] {pt2=root:b_features_10_block_1_1_running_var target=features.10.block.1.1.running_var} ->[n385] constant,
     t229 f32 [C=96] {pt2=root:b_features_10_block_3_1_running_mean target=features.10.block.3.1.running_mean} ->[n412] constant,
     t230 f32 [C=96] {pt2=root:b_features_10_block_3_1_running_var target=features.10.block.3.1.running_var} ->[n412] constant,
     t232 f32 [C=576] {pt2=root:b_features_11_block_0_1_running_mean target=features.11.block.0.1.running_mean} ->[n420] constant,
     t233 f32 [C=576] {pt2=root:b_features_11_block_0_1_running_var target=features.11.block.0.1.running_var} ->[n420] constant,
     t235 f32 [C=576] {pt2=root:b_features_11_block_1_1_running_mean target=features.11.block.1.1.running_mean} ->[n432] constant,
     t236 f32 [C=576] {pt2=root:b_features_11_block_1_1_running_var target=features.11.block.1.1.running_var} ->[n432] constant,
     t238 f32 [C=96] {pt2=root:b_features_11_block_3_1_running_mean target=features.11.block.3.1.running_mean} ->[n459] constant,
     t239 f32 [C=96] {pt2=root:b_features_11_block_3_1_running_var target=features.11.block.3.1.running_var} ->[n459] constant,
     t241 f32 [C=576] {pt2=root:b_features_12_1_running_mean target=features.12.1.running_mean} ->[n467] constant,
     t242 f32 [C=576] {pt2=root:b_features_12_1_running_var target=features.12.1.running_var} ->[n467] constant,
     t244 f32 [H=3 W=224 C=224] {pt2=root:x} ->[n0]]
  nodes:
    group g1 torch.ops.aten.convolution.default:
      n0 {derived}: [t245 f32 [H=224 W=224 C=3] {derived} ->[n2]] =
        permute x=t244 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
      n1 {derived}: [t246 f32 [N=16 T=1 D=1 H=3 W=3 C=3] {derived} ->[n2]] =
        permute
          x=t0 {pt2=root:p_features_0_0_weight target=features.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2 {derived}: [t247 f32 [H=112 W=112 C=16] {derived} ->[n5]] =
        convolution
          x=t245 {derived} <-n0
          weight=t246 {derived} <-n1
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g3 torch.ops.aten.convolution.default:
      n13 {derived}: [t258 f32 [N=16 T=1 D=1 H=3 W=3 C=1] {derived} ->[n14]] =
        permute
          x=t3 {pt2=root:p_features_1_block_0_0_weight target=features.1.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n12 {derived}: [t257 f32 [H=112 W=112 C=16] {derived} ->[n14]] =
        permute x=t256 {pt2=root:div} <-n11 perm=[H<-W, W<-C, C<-H]
      n14 {derived}: [t259 f32 [H=56 W=56 C=16] {derived} ->[n17]] =
        convolution
          x=t257 {derived} <-n12
          weight=t258 {derived} <-n13
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=16}
    group g5 torch.ops.aten.convolution.default:
      n22 {derived}: [t267 f32 [N=8 T=1 D=1 H=1 W=1 C=16] {derived} ->[n23]] =
        permute
          x=t6 {pt2=root:p_features_1_block_1_fc1_weight target=features.1.block.1.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n21 {derived}: [t266 f32 [C=16] {derived} ->[n23]] =
        permute x=t265 {pt2=root:mean} <-n20 perm=[H<-W, W<-C, C<-H]
      n23 {derived}: [t268 f32 [C=8] {derived} ->[n492]] =
        convolution
          x=t266 {derived} <-n21
          weight=t267 {derived} <-n22
          bias=t7 {pt2=root:p_features_1_block_1_fc1_bias target=features.1.block.1.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g6 torch.ops.aten.convolution.default:
      n27 {derived}: [t272 f32 [N=16 T=1 D=1 H=1 W=1 C=8] {derived} ->[n28]] =
        permute
          x=t8 {pt2=root:p_features_1_block_1_fc2_weight target=features.1.block.1.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n28 {derived}: [t273 f32 [C=16] {derived} ->[n493]] =
        convolution
          x=t734 {derived} <-n492
          weight=t272 {derived} <-n27
          bias=t9 {pt2=root:p_features_1_block_1_fc2_bias target=features.1.block.1.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g7 torch.ops.aten.convolution.default:
      n36 {derived}: [t281 f32 [N=16 T=1 D=1 H=1 W=1 C=16] {derived} ->[n37]] =
        permute
          x=t10 {pt2=root:p_features_1_block_2_0_weight target=features.1.block.2.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n35 {derived}: [t280 f32 [H=56 W=56 C=16] {derived} ->[n37]] =
        permute x=t279 {pt2=root:mul_1} <-n34 perm=[H<-W, W<-C, C<-H]
      n37 {derived}: [t282 f32 [H=56 W=56 C=16] {derived} ->[n40]] =
        convolution
          x=t280 {derived} <-n35
          weight=t281 {derived} <-n36
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g9 torch.ops.aten.convolution.default:
      n43 {derived}: [t288 f32 [N=72 T=1 D=1 H=1 W=1 C=16] {derived} ->[n44]] =
        permute
          x=t13 {pt2=root:p_features_2_block_0_0_weight target=features.2.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n44 {derived}: [t289 f32 [H=56 W=56 C=72] {derived} ->[n47]] =
        convolution
          x=t285 {derived} <-n40
          weight=t288 {derived} <-n43
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g11 torch.ops.aten.convolution.default:
      n51 {derived}: [t296 f32 [N=72 T=1 D=1 H=3 W=3 C=1] {derived} ->[n52]] =
        permute
          x=t16 {pt2=root:p_features_2_block_1_0_weight target=features.2.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n52 {derived}: [t297 f32 [H=28 W=28 C=72] {derived} ->[n55]] =
        convolution
          x=t739 {derived} <-n498
          weight=t296 {derived} <-n51
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=72}
    group g13 torch.ops.aten.convolution.default:
      n59 {derived}: [t304 f32 [N=24 T=1 D=1 H=1 W=1 C=72] {derived} ->[n60]] =
        permute
          x=t19 {pt2=root:p_features_2_block_2_0_weight target=features.2.block.2.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n60 {derived}: [t305 f32 [H=28 W=28 C=24] {derived} ->[n63]] =
        convolution
          x=t740 {derived} <-n499
          weight=t304 {derived} <-n59
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g15 torch.ops.aten.convolution.default:
      n66 {derived}: [t311 f32 [N=88 T=1 D=1 H=1 W=1 C=24] {derived} ->[n67]] =
        permute
          x=t22 {pt2=root:p_features_3_block_0_0_weight target=features.3.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n67 {derived}: [t312 f32 [H=28 W=28 C=88] {derived} ->[n70]] =
        convolution
          x=t308 {derived} <-n63
          weight=t311 {derived} <-n66
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g17 torch.ops.aten.convolution.default:
      n74 {derived}: [t319 f32 [N=88 T=1 D=1 H=3 W=3 C=1] {derived} ->[n75]] =
        permute
          x=t25 {pt2=root:p_features_3_block_1_0_weight target=features.3.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n75 {derived}: [t320 f32 [H=28 W=28 C=88] {derived} ->[n78]] =
        convolution
          x=t741 {derived} <-n500
          weight=t319 {derived} <-n74
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=88}
    group g19 torch.ops.aten.convolution.default:
      n82 {derived}: [t327 f32 [N=24 T=1 D=1 H=1 W=1 C=88] {derived} ->[n83]] =
        permute
          x=t28 {pt2=root:p_features_3_block_2_0_weight target=features.3.block.2.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n83 {derived}: [t328 f32 [H=28 W=28 C=24] {derived} ->[n86]] =
        convolution
          x=t742 {derived} <-n501
          weight=t327 {derived} <-n82
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g21 torch.ops.aten.convolution.default:
      n90 {derived}: [t335 f32 [N=96 T=1 D=1 H=1 W=1 C=24] {derived} ->[n91]] =
        permute
          x=t31 {pt2=root:p_features_4_block_0_0_weight target=features.4.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n91 {derived}: [t336 f32 [H=28 W=28 C=96] {derived} ->[n94]] =
        convolution
          x=t743 {derived} <-n502
          weight=t335 {derived} <-n90
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g23 torch.ops.aten.convolution.default:
      n102 {derived}: [t347 f32 [N=96 T=1 D=1 H=5 W=5 C=1] {derived} ->[n103]] =
        permute
          x=t34 {pt2=root:p_features_4_block_1_0_weight target=features.4.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n101 {derived}: [t346 f32 [H=28 W=28 C=96] {derived} ->[n103]] =
        permute x=t345 {pt2=root:div_2} <-n100 perm=[H<-W, W<-C, C<-H]
      n103 {derived}: [t348 f32 [H=14 W=14 C=96] {derived} ->[n106]] =
        convolution
          x=t346 {derived} <-n101
          weight=t347 {derived} <-n102
          bias=none
          params={stride={h=2; w=2};
                 padding={h=2; w=2};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=96}
    group g25 torch.ops.aten.convolution.default:
      n115 {derived}: [t360 f32 [N=24 T=1 D=1 H=1 W=1 C=96] {derived} ->[n116]] =
        permute
          x=t37 {pt2=root:p_features_4_block_2_fc1_weight target=features.4.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n114 {derived}: [t359 f32 [C=96] {derived} ->[n116]] =
        permute x=t358 {pt2=root:mean_1} <-n113 perm=[H<-W, W<-C, C<-H]
      n116 {derived}: [t361 f32 [C=24] {derived} ->[n503]] =
        convolution
          x=t359 {derived} <-n114
          weight=t360 {derived} <-n115
          bias=t38 {pt2=root:p_features_4_block_2_fc1_bias target=features.4.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g26 torch.ops.aten.convolution.default:
      n120 {derived}: [t365 f32 [N=96 T=1 D=1 H=1 W=1 C=24] {derived} ->[n121]] =
        permute
          x=t39 {pt2=root:p_features_4_block_2_fc2_weight target=features.4.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n121 {derived}: [t366 f32 [C=96] {derived} ->[n504]] =
        convolution
          x=t744 {derived} <-n503
          weight=t365 {derived} <-n120
          bias=t40 {pt2=root:p_features_4_block_2_fc2_bias target=features.4.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g27 torch.ops.aten.convolution.default:
      n129 {derived}: [t374 f32 [N=40 T=1 D=1 H=1 W=1 C=96] {derived} ->[n130]] =
        permute
          x=t41 {pt2=root:p_features_4_block_3_0_weight target=features.4.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n128 {derived}: [t373 f32 [H=14 W=14 C=96] {derived} ->[n130]] =
        permute x=t372 {pt2=root:mul_4} <-n127 perm=[H<-W, W<-C, C<-H]
      n130 {derived}: [t375 f32 [H=14 W=14 C=40] {derived} ->[n133]] =
        convolution
          x=t373 {derived} <-n128
          weight=t374 {derived} <-n129
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g29 torch.ops.aten.convolution.default:
      n136 {derived}: [t381 f32 [N=240 T=1 D=1 H=1 W=1 C=40] {derived} ->[n137]] =
        permute
          x=t44 {pt2=root:p_features_5_block_0_0_weight target=features.5.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n137 {derived}: [t382 f32 [H=14 W=14 C=240] {derived} ->[n140]] =
        convolution
          x=t378 {derived} <-n133
          weight=t381 {derived} <-n136
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g31 torch.ops.aten.convolution.default:
      n148 {derived}: [t393 f32 [N=240 T=1 D=1 H=5 W=5 C=1] {derived} ->[n149]] =
        permute
          x=t47 {pt2=root:p_features_5_block_1_0_weight target=features.5.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n147 {derived}: [t392 f32 [H=14 W=14 C=240] {derived} ->[n149]] =
        permute x=t391 {pt2=root:div_5} <-n146 perm=[H<-W, W<-C, C<-H]
      n149 {derived}: [t394 f32 [H=14 W=14 C=240] {derived} ->[n152]] =
        convolution
          x=t392 {derived} <-n147
          weight=t393 {derived} <-n148
          bias=none
          params={stride={h=1; w=1};
                 padding={h=2; w=2};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=240}
    group g33 torch.ops.aten.convolution.default:
      n161 {derived}: [t406 f32 [N=64 T=1 D=1 H=1 W=1 C=240] {derived} ->[n162]] =
        permute
          x=t50 {pt2=root:p_features_5_block_2_fc1_weight target=features.5.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n160 {derived}: [t405 f32 [C=240] {derived} ->[n162]] =
        permute x=t404 {pt2=root:mean_2} <-n159 perm=[H<-W, W<-C, C<-H]
      n162 {derived}: [t407 f32 [C=64] {derived} ->[n509]] =
        convolution
          x=t405 {derived} <-n160
          weight=t406 {derived} <-n161
          bias=t51 {pt2=root:p_features_5_block_2_fc1_bias target=features.5.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g34 torch.ops.aten.convolution.default:
      n166 {derived}: [t411 f32 [N=240 T=1 D=1 H=1 W=1 C=64] {derived} ->[n167]] =
        permute
          x=t52 {pt2=root:p_features_5_block_2_fc2_weight target=features.5.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n167 {derived}: [t412 f32 [C=240] {derived} ->[n510]] =
        convolution
          x=t749 {derived} <-n509
          weight=t411 {derived} <-n166
          bias=t53 {pt2=root:p_features_5_block_2_fc2_bias target=features.5.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g35 torch.ops.aten.convolution.default:
      n175 {derived}: [t420 f32 [N=40 T=1 D=1 H=1 W=1 C=240] {derived} ->[n176]] =
        permute
          x=t54 {pt2=root:p_features_5_block_3_0_weight target=features.5.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n174 {derived}: [t419 f32 [H=14 W=14 C=240] {derived} ->[n176]] =
        permute x=t418 {pt2=root:mul_7} <-n173 perm=[H<-W, W<-C, C<-H]
      n176 {derived}: [t421 f32 [H=14 W=14 C=40] {derived} ->[n179]] =
        convolution
          x=t419 {derived} <-n174
          weight=t420 {derived} <-n175
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g37 torch.ops.aten.convolution.default:
      n183 {derived}: [t428 f32 [N=240 T=1 D=1 H=1 W=1 C=40] {derived} ->[n184]] =
        permute
          x=t57 {pt2=root:p_features_6_block_0_0_weight target=features.6.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n184 {derived}: [t429 f32 [H=14 W=14 C=240] {derived} ->[n187]] =
        convolution
          x=t754 {derived} <-n515
          weight=t428 {derived} <-n183
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g39 torch.ops.aten.convolution.default:
      n195 {derived}: [t440 f32 [N=240 T=1 D=1 H=5 W=5 C=1] {derived} ->[n196]] =
        permute
          x=t60 {pt2=root:p_features_6_block_1_0_weight target=features.6.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n194 {derived}: [t439 f32 [H=14 W=14 C=240] {derived} ->[n196]] =
        permute x=t438 {pt2=root:div_8} <-n193 perm=[H<-W, W<-C, C<-H]
      n196 {derived}: [t441 f32 [H=14 W=14 C=240] {derived} ->[n199]] =
        convolution
          x=t439 {derived} <-n194
          weight=t440 {derived} <-n195
          bias=none
          params={stride={h=1; w=1};
                 padding={h=2; w=2};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=240}
    group g41 torch.ops.aten.convolution.default:
      n208 {derived}: [t453 f32 [N=64 T=1 D=1 H=1 W=1 C=240] {derived} ->[n209]] =
        permute
          x=t63 {pt2=root:p_features_6_block_2_fc1_weight target=features.6.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n207 {derived}: [t452 f32 [C=240] {derived} ->[n209]] =
        permute x=t451 {pt2=root:mean_3} <-n206 perm=[H<-W, W<-C, C<-H]
      n209 {derived}: [t454 f32 [C=64] {derived} ->[n516]] =
        convolution
          x=t452 {derived} <-n207
          weight=t453 {derived} <-n208
          bias=t64 {pt2=root:p_features_6_block_2_fc1_bias target=features.6.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g42 torch.ops.aten.convolution.default:
      n213 {derived}: [t458 f32 [N=240 T=1 D=1 H=1 W=1 C=64] {derived} ->[n214]] =
        permute
          x=t65 {pt2=root:p_features_6_block_2_fc2_weight target=features.6.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n214 {derived}: [t459 f32 [C=240] {derived} ->[n517]] =
        convolution
          x=t755 {derived} <-n516
          weight=t458 {derived} <-n213
          bias=t66 {pt2=root:p_features_6_block_2_fc2_bias target=features.6.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g43 torch.ops.aten.convolution.default:
      n222 {derived}: [t467 f32 [N=40 T=1 D=1 H=1 W=1 C=240] {derived} ->[n223]] =
        permute
          x=t67 {pt2=root:p_features_6_block_3_0_weight target=features.6.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n221 {derived}: [t466 f32 [H=14 W=14 C=240] {derived} ->[n223]] =
        permute x=t465 {pt2=root:mul_10} <-n220 perm=[H<-W, W<-C, C<-H]
      n223 {derived}: [t468 f32 [H=14 W=14 C=40] {derived} ->[n226]] =
        convolution
          x=t466 {derived} <-n221
          weight=t467 {derived} <-n222
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g45 torch.ops.aten.convolution.default:
      n230 {derived}: [t475 f32 [N=120 T=1 D=1 H=1 W=1 C=40] {derived} ->[n231]] =
        permute
          x=t70 {pt2=root:p_features_7_block_0_0_weight target=features.7.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n231 {derived}: [t476 f32 [H=14 W=14 C=120] {derived} ->[n234]] =
        convolution
          x=t760 {derived} <-n522
          weight=t475 {derived} <-n230
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g47 torch.ops.aten.convolution.default:
      n242 {derived}: [t487 f32 [N=120 T=1 D=1 H=5 W=5 C=1] {derived} ->[n243]] =
        permute
          x=t73 {pt2=root:p_features_7_block_1_0_weight target=features.7.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n241 {derived}: [t486 f32 [H=14 W=14 C=120] {derived} ->[n243]] =
        permute x=t485 {pt2=root:div_11} <-n240 perm=[H<-W, W<-C, C<-H]
      n243 {derived}: [t488 f32 [H=14 W=14 C=120] {derived} ->[n246]] =
        convolution
          x=t486 {derived} <-n241
          weight=t487 {derived} <-n242
          bias=none
          params={stride={h=1; w=1};
                 padding={h=2; w=2};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=120}
    group g49 torch.ops.aten.convolution.default:
      n255 {derived}: [t500 f32 [N=32 T=1 D=1 H=1 W=1 C=120] {derived} ->[n256]] =
        permute
          x=t76 {pt2=root:p_features_7_block_2_fc1_weight target=features.7.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n254 {derived}: [t499 f32 [C=120] {derived} ->[n256]] =
        permute x=t498 {pt2=root:mean_4} <-n253 perm=[H<-W, W<-C, C<-H]
      n256 {derived}: [t501 f32 [C=32] {derived} ->[n523]] =
        convolution
          x=t499 {derived} <-n254
          weight=t500 {derived} <-n255
          bias=t77 {pt2=root:p_features_7_block_2_fc1_bias target=features.7.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g50 torch.ops.aten.convolution.default:
      n260 {derived}: [t505 f32 [N=120 T=1 D=1 H=1 W=1 C=32] {derived} ->[n261]] =
        permute
          x=t78 {pt2=root:p_features_7_block_2_fc2_weight target=features.7.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n261 {derived}: [t506 f32 [C=120] {derived} ->[n524]] =
        convolution
          x=t761 {derived} <-n523
          weight=t505 {derived} <-n260
          bias=t79 {pt2=root:p_features_7_block_2_fc2_bias target=features.7.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g51 torch.ops.aten.convolution.default:
      n269 {derived}: [t514 f32 [N=48 T=1 D=1 H=1 W=1 C=120] {derived} ->[n270]] =
        permute
          x=t80 {pt2=root:p_features_7_block_3_0_weight target=features.7.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n268 {derived}: [t513 f32 [H=14 W=14 C=120] {derived} ->[n270]] =
        permute x=t512 {pt2=root:mul_13} <-n267 perm=[H<-W, W<-C, C<-H]
      n270 {derived}: [t515 f32 [H=14 W=14 C=48] {derived} ->[n273]] =
        convolution
          x=t513 {derived} <-n268
          weight=t514 {derived} <-n269
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g53 torch.ops.aten.convolution.default:
      n276 {derived}: [t521 f32 [N=144 T=1 D=1 H=1 W=1 C=48] {derived} ->[n277]] =
        permute
          x=t83 {pt2=root:p_features_8_block_0_0_weight target=features.8.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n277 {derived}: [t522 f32 [H=14 W=14 C=144] {derived} ->[n280]] =
        convolution
          x=t518 {derived} <-n273
          weight=t521 {derived} <-n276
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g55 torch.ops.aten.convolution.default:
      n288 {derived}: [t533 f32 [N=144 T=1 D=1 H=5 W=5 C=1] {derived} ->[n289]] =
        permute
          x=t86 {pt2=root:p_features_8_block_1_0_weight target=features.8.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n287 {derived}: [t532 f32 [H=14 W=14 C=144] {derived} ->[n289]] =
        permute x=t531 {pt2=root:div_14} <-n286 perm=[H<-W, W<-C, C<-H]
      n289 {derived}: [t534 f32 [H=14 W=14 C=144] {derived} ->[n292]] =
        convolution
          x=t532 {derived} <-n287
          weight=t533 {derived} <-n288
          bias=none
          params={stride={h=1; w=1};
                 padding={h=2; w=2};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=144}
    group g57 torch.ops.aten.convolution.default:
      n301 {derived}: [t546 f32 [N=40 T=1 D=1 H=1 W=1 C=144] {derived} ->[n302]] =
        permute
          x=t89 {pt2=root:p_features_8_block_2_fc1_weight target=features.8.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n300 {derived}: [t545 f32 [C=144] {derived} ->[n302]] =
        permute x=t544 {pt2=root:mean_5} <-n299 perm=[H<-W, W<-C, C<-H]
      n302 {derived}: [t547 f32 [C=40] {derived} ->[n529]] =
        convolution
          x=t545 {derived} <-n300
          weight=t546 {derived} <-n301
          bias=t90 {pt2=root:p_features_8_block_2_fc1_bias target=features.8.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g58 torch.ops.aten.convolution.default:
      n306 {derived}: [t551 f32 [N=144 T=1 D=1 H=1 W=1 C=40] {derived} ->[n307]] =
        permute
          x=t91 {pt2=root:p_features_8_block_2_fc2_weight target=features.8.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n307 {derived}: [t552 f32 [C=144] {derived} ->[n530]] =
        convolution
          x=t766 {derived} <-n529
          weight=t551 {derived} <-n306
          bias=t92 {pt2=root:p_features_8_block_2_fc2_bias target=features.8.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g59 torch.ops.aten.convolution.default:
      n315 {derived}: [t560 f32 [N=48 T=1 D=1 H=1 W=1 C=144] {derived} ->[n316]] =
        permute
          x=t93 {pt2=root:p_features_8_block_3_0_weight target=features.8.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n314 {derived}: [t559 f32 [H=14 W=14 C=144] {derived} ->[n316]] =
        permute x=t558 {pt2=root:mul_16} <-n313 perm=[H<-W, W<-C, C<-H]
      n316 {derived}: [t561 f32 [H=14 W=14 C=48] {derived} ->[n319]] =
        convolution
          x=t559 {derived} <-n314
          weight=t560 {derived} <-n315
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g61 torch.ops.aten.convolution.default:
      n323 {derived}: [t568 f32 [N=288 T=1 D=1 H=1 W=1 C=48] {derived} ->[n324]] =
        permute
          x=t96 {pt2=root:p_features_9_block_0_0_weight target=features.9.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n324 {derived}: [t569 f32 [H=14 W=14 C=288] {derived} ->[n327]] =
        convolution
          x=t771 {derived} <-n535
          weight=t568 {derived} <-n323
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g63 torch.ops.aten.convolution.default:
      n335 {derived}: [t580 f32 [N=288 T=1 D=1 H=5 W=5 C=1] {derived} ->[n336]] =
        permute
          x=t99 {pt2=root:p_features_9_block_1_0_weight target=features.9.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n334 {derived}: [t579 f32 [H=14 W=14 C=288] {derived} ->[n336]] =
        permute x=t578 {pt2=root:div_17} <-n333 perm=[H<-W, W<-C, C<-H]
      n336 {derived}: [t581 f32 [H=7 W=7 C=288] {derived} ->[n339]] =
        convolution
          x=t579 {derived} <-n334
          weight=t580 {derived} <-n335
          bias=none
          params={stride={h=2; w=2};
                 padding={h=2; w=2};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=288}
    group g65 torch.ops.aten.convolution.default:
      n348 {derived}: [t593 f32 [N=72 T=1 D=1 H=1 W=1 C=288] {derived} ->[n349]] =
        permute
          x=t102 {pt2=root:p_features_9_block_2_fc1_weight target=features.9.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n347 {derived}: [t592 f32 [C=288] {derived} ->[n349]] =
        permute x=t591 {pt2=root:mean_6} <-n346 perm=[H<-W, W<-C, C<-H]
      n349 {derived}: [t594 f32 [C=72] {derived} ->[n536]] =
        convolution
          x=t592 {derived} <-n347
          weight=t593 {derived} <-n348
          bias=t103 {pt2=root:p_features_9_block_2_fc1_bias target=features.9.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g66 torch.ops.aten.convolution.default:
      n353 {derived}: [t598 f32 [N=288 T=1 D=1 H=1 W=1 C=72] {derived} ->[n354]] =
        permute
          x=t104 {pt2=root:p_features_9_block_2_fc2_weight target=features.9.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n354 {derived}: [t599 f32 [C=288] {derived} ->[n537]] =
        convolution
          x=t772 {derived} <-n536
          weight=t598 {derived} <-n353
          bias=t105 {pt2=root:p_features_9_block_2_fc2_bias target=features.9.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g67 torch.ops.aten.convolution.default:
      n362 {derived}: [t607 f32 [N=96 T=1 D=1 H=1 W=1 C=288] {derived} ->[n363]] =
        permute
          x=t106 {pt2=root:p_features_9_block_3_0_weight target=features.9.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n361 {derived}: [t606 f32 [H=7 W=7 C=288] {derived} ->[n363]] =
        permute x=t605 {pt2=root:mul_19} <-n360 perm=[H<-W, W<-C, C<-H]
      n363 {derived}: [t608 f32 [H=7 W=7 C=96] {derived} ->[n366]] =
        convolution
          x=t606 {derived} <-n361
          weight=t607 {derived} <-n362
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g69 torch.ops.aten.convolution.default:
      n369 {derived}: [t614 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived} ->[n370]] =
        permute
          x=t109 {pt2=root:p_features_10_block_0_0_weight target=features.10.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n370 {derived}: [t615 f32 [H=7 W=7 C=576] {derived} ->[n373]] =
        convolution
          x=t611 {derived} <-n366
          weight=t614 {derived} <-n369
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g71 torch.ops.aten.convolution.default:
      n381 {derived}: [t626 f32 [N=576 T=1 D=1 H=5 W=5 C=1] {derived} ->[n382]] =
        permute
          x=t112 {pt2=root:p_features_10_block_1_0_weight target=features.10.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n380 {derived}: [t625 f32 [H=7 W=7 C=576] {derived} ->[n382]] =
        permute x=t624 {pt2=root:div_20} <-n379 perm=[H<-W, W<-C, C<-H]
      n382 {derived}: [t627 f32 [H=7 W=7 C=576] {derived} ->[n385]] =
        convolution
          x=t625 {derived} <-n380
          weight=t626 {derived} <-n381
          bias=none
          params={stride={h=1; w=1};
                 padding={h=2; w=2};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=576}
    group g73 torch.ops.aten.convolution.default:
      n394 {derived}: [t639 f32 [N=144 T=1 D=1 H=1 W=1 C=576] {derived} ->[n395]] =
        permute
          x=t115 {pt2=root:p_features_10_block_2_fc1_weight target=features.10.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n393 {derived}: [t638 f32 [C=576] {derived} ->[n395]] =
        permute x=t637 {pt2=root:mean_7} <-n392 perm=[H<-W, W<-C, C<-H]
      n395 {derived}: [t640 f32 [C=144] {derived} ->[n542]] =
        convolution
          x=t638 {derived} <-n393
          weight=t639 {derived} <-n394
          bias=t116 {pt2=root:p_features_10_block_2_fc1_bias target=features.10.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g74 torch.ops.aten.convolution.default:
      n399 {derived}: [t644 f32 [N=576 T=1 D=1 H=1 W=1 C=144] {derived} ->[n400]] =
        permute
          x=t117 {pt2=root:p_features_10_block_2_fc2_weight target=features.10.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n400 {derived}: [t645 f32 [C=576] {derived} ->[n543]] =
        convolution
          x=t777 {derived} <-n542
          weight=t644 {derived} <-n399
          bias=t118 {pt2=root:p_features_10_block_2_fc2_bias target=features.10.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g75 torch.ops.aten.convolution.default:
      n408 {derived}: [t653 f32 [N=96 T=1 D=1 H=1 W=1 C=576] {derived} ->[n409]] =
        permute
          x=t119 {pt2=root:p_features_10_block_3_0_weight target=features.10.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n407 {derived}: [t652 f32 [H=7 W=7 C=576] {derived} ->[n409]] =
        permute x=t651 {pt2=root:mul_22} <-n406 perm=[H<-W, W<-C, C<-H]
      n409 {derived}: [t654 f32 [H=7 W=7 C=96] {derived} ->[n412]] =
        convolution
          x=t652 {derived} <-n407
          weight=t653 {derived} <-n408
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g77 torch.ops.aten.convolution.default:
      n416 {derived}: [t661 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived} ->[n417]] =
        permute
          x=t122 {pt2=root:p_features_11_block_0_0_weight target=features.11.block.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n417 {derived}: [t662 f32 [H=7 W=7 C=576] {derived} ->[n420]] =
        convolution
          x=t782 {derived} <-n548
          weight=t661 {derived} <-n416
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g79 torch.ops.aten.convolution.default:
      n428 {derived}: [t673 f32 [N=576 T=1 D=1 H=5 W=5 C=1] {derived} ->[n429]] =
        permute
          x=t125 {pt2=root:p_features_11_block_1_0_weight target=features.11.block.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n427 {derived}: [t672 f32 [H=7 W=7 C=576] {derived} ->[n429]] =
        permute x=t671 {pt2=root:div_23} <-n426 perm=[H<-W, W<-C, C<-H]
      n429 {derived}: [t674 f32 [H=7 W=7 C=576] {derived} ->[n432]] =
        convolution
          x=t672 {derived} <-n427
          weight=t673 {derived} <-n428
          bias=none
          params={stride={h=1; w=1};
                 padding={h=2; w=2};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=576}
    group g81 torch.ops.aten.convolution.default:
      n441 {derived}: [t686 f32 [N=144 T=1 D=1 H=1 W=1 C=576] {derived} ->[n442]] =
        permute
          x=t128 {pt2=root:p_features_11_block_2_fc1_weight target=features.11.block.2.fc1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n440 {derived}: [t685 f32 [C=576] {derived} ->[n442]] =
        permute x=t684 {pt2=root:mean_8} <-n439 perm=[H<-W, W<-C, C<-H]
      n442 {derived}: [t687 f32 [C=144] {derived} ->[n549]] =
        convolution
          x=t685 {derived} <-n440
          weight=t686 {derived} <-n441
          bias=t129 {pt2=root:p_features_11_block_2_fc1_bias target=features.11.block.2.fc1.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g82 torch.ops.aten.convolution.default:
      n446 {derived}: [t691 f32 [N=576 T=1 D=1 H=1 W=1 C=144] {derived} ->[n447]] =
        permute
          x=t130 {pt2=root:p_features_11_block_2_fc2_weight target=features.11.block.2.fc2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n447 {derived}: [t692 f32 [C=576] {derived} ->[n550]] =
        convolution
          x=t783 {derived} <-n549
          weight=t691 {derived} <-n446
          bias=t131 {pt2=root:p_features_11_block_2_fc2_bias target=features.11.block.2.fc2.bias}
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g83 torch.ops.aten.convolution.default:
      n455 {derived}: [t700 f32 [N=96 T=1 D=1 H=1 W=1 C=576] {derived} ->[n456]] =
        permute
          x=t132 {pt2=root:p_features_11_block_3_0_weight target=features.11.block.3.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n454 {derived}: [t699 f32 [H=7 W=7 C=576] {derived} ->[n456]] =
        permute x=t698 {pt2=root:mul_25} <-n453 perm=[H<-W, W<-C, C<-H]
      n456 {derived}: [t701 f32 [H=7 W=7 C=96] {derived} ->[n459]] =
        convolution
          x=t699 {derived} <-n454
          weight=t700 {derived} <-n455
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g85 torch.ops.aten.convolution.default:
      n463 {derived}: [t708 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived} ->[n464]] =
        permute
          x=t135 {pt2=root:p_features_12_0_weight target=features.12.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n464 {derived}: [t709 f32 [H=7 W=7 C=576] {derived} ->[n467]] =
        convolution
          x=t788 {derived} <-n555
          weight=t708 {derived} <-n463
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n488 {pt2=root[252] torch.ops.aten.permute.default}: [t722 f32 [N=1024 T=1
                                                                    D=1 H=1 W=1
                                                                    C=576] {derived} ->[n478]] =
      permute
        x=t138 {pt2=root:p_classifier_0_weight target=classifier.0.weight}
        perm=[N<-W, W<-N]
    n489 {pt2=root[260] torch.ops.aten.permute.default}: [t731 f32 [N=1000 T=1
                                                                    D=1 H=1 W=1
                                                                    C=1024] {derived} ->[n487]] =
      permute
        x=t140 {pt2=root:p_classifier_3_weight target=classifier.3.weight}
        perm=[N<-W, W<-N]
    group g2 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n5 {derived}: [t250 f32 [H=112 W=112 C=16] {derived} ->[n6]] =
        batch_norm
          x=t247 {derived} <-n2
          weight=t1 {pt2=root:p_features_0_1_weight target=features.0.1.weight}
          bias=t2 {pt2=root:p_features_0_1_bias target=features.0.1.bias}
          running_mean=t142 {pt2=root:b_features_0_1_running_mean target=features.0.1.running_mean}
          running_var=t143 {pt2=root:b_features_0_1_running_var target=features.0.1.running_var}
          params={channel=C; eps=0.001}
      n6 {pt2=root[1] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t251 f32 [H=16
                                                                      W=112
                                                                      C=112] {pt2=root:getitem} ->[n7,
                                                                      n10]] =
        permute x=t250 {derived} <-n5 perm=[H<-C, W<-H, C<-W]
    n7 {pt2=root[2] torch.ops.aten.add.Tensor}: [t252 f32 [H=16 W=112 C=112] {pt2=root:add} ->[n8]] =
      add_scalar x=t251 {pt2=root:getitem} <-n6 scalar=3
    n8 {pt2=root[3] torch.ops.aten.clamp.default}: [t253 f32 [H=16 W=112 C=112] {pt2=root:clamp} ->[n9]] =
      clamp x=t252 {pt2=root:add} <-n7 params={min=0; max=none}
    n9 {pt2=root[4] torch.ops.aten.clamp.default}: [t254 f32 [H=16 W=112 C=112] {pt2=root:clamp_1} ->[n10]] =
      clamp x=t253 {pt2=root:clamp} <-n8 params={min=none; max=6}
    n10 {pt2=root[5] torch.ops.aten.mul.Tensor}: [t255 f32 [H=16 W=112 C=112] {pt2=root:mul} ->[n11]] =
      mul a=t251 {pt2=root:getitem} <-n6 b=t254 {pt2=root:clamp_1} <-n9
    n11 {pt2=root[6] torch.ops.aten.div.Tensor}: [t256 f32 [H=16 W=112 C=112] {pt2=root:div} ->[n12]] =
      div_scalar x=t255 {pt2=root:mul} <-n10 scalar=6
    group g4 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n17 {derived}: [t262 f32 [H=56 W=56 C=16] {derived} ->[n490]] =
        batch_norm
          x=t259 {derived} <-n14
          weight=t4 {pt2=root:p_features_1_block_0_1_weight target=features.1.block.0.1.weight}
          bias=t5 {pt2=root:p_features_1_block_0_1_bias target=features.1.block.0.1.bias}
          running_mean=t145 {pt2=root:b_features_1_block_0_1_running_mean target=features.1.block.0.1.running_mean}
          running_var=t146 {pt2=root:b_features_1_block_0_1_running_var target=features.1.block.0.1.running_var}
          params={channel=C; eps=0.001}
    n490 {pt2=root[9] torch.ops.aten.relu.default}: [t733 f32 [H=56 W=56 C=16] {derived} ->[n491]] =
      relu x=t262 {derived} <-n17
    n491 {pt2=root[8] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t264 f32 [H=16
                                                                      W=56
                                                                      C=56] {pt2=root:relu} ->[n20,
                                                                      n34]] =
      permute x=t733 {derived} <-n490 perm=[H<-C, W<-H, C<-W]
    n20 {pt2=root[10] torch.ops.aten.mean.dim}: [t265 f32 [H=16 W=1 C=1] {pt2=root:mean} ->[n21]] =
      mean x=t264 {pt2=root:relu} <-n491 params={dims=[C, W]; keepdim=true}
    n492 {pt2=root[12] torch.ops.aten.relu.default}: [t734 f32 [C=8] {derived} ->[n28]] =
      relu x=t268 {derived} <-n23
    n493 {pt2=root[14] torch.ops.aten.add.Tensor}: [t735 f32 [C=16] {derived} ->[n494]] =
      add_scalar x=t273 {derived} <-n28 scalar=3
    n494 {pt2=root[15] torch.ops.aten.clamp.default}: [t736 f32 [C=16] {derived} ->[n495]] =
      clamp x=t735 {derived} <-n493 params={min=0; max=none}
    n495 {pt2=root[16] torch.ops.aten.clamp.default}: [t737 f32 [C=16] {derived} ->[n496]] =
      clamp x=t736 {derived} <-n494 params={min=none; max=6}
    n496 {pt2=root[17] torch.ops.aten.div.Tensor}: [t738 f32 [C=16] {derived} ->[n497]] =
      div_scalar x=t737 {derived} <-n495 scalar=6
    n497 {pt2=root[13] torch.ops.aten.convolution.default}: [t278 f32 [H=16 W=1
                                                                      C=1] {pt2=root:div_1} ->[n34]] =
      permute x=t738 {derived} <-n496 perm=[H<-C, W<-H, C<-W]
    n34 {pt2=root[18] torch.ops.aten.mul.Tensor}: [t279 f32 [H=16 W=56 C=56] {pt2=root:mul_1} ->[n35]] =
      mul a=t278 {pt2=root:div_1} <-n497 b=t264 {pt2=root:relu} <-n491
    group g8 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n40 {derived}: [t285 f32 [H=56 W=56 C=16] {derived} ->[n44]] =
        batch_norm
          x=t282 {derived} <-n37
          weight=t11 {pt2=root:p_features_1_block_2_1_weight target=features.1.block.2.1.weight}
          bias=t12 {pt2=root:p_features_1_block_2_1_bias target=features.1.block.2.1.bias}
          running_mean=t148 {pt2=root:b_features_1_block_2_1_running_mean target=features.1.block.2.1.running_mean}
          running_var=t149 {pt2=root:b_features_1_block_2_1_running_var target=features.1.block.2.1.running_var}
          params={channel=C; eps=0.001}
    group g10 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n47 {derived}: [t292 f32 [H=56 W=56 C=72] {derived} ->[n498]] =
        batch_norm
          x=t289 {derived} <-n44
          weight=t14 {pt2=root:p_features_2_block_0_1_weight target=features.2.block.0.1.weight}
          bias=t15 {pt2=root:p_features_2_block_0_1_bias target=features.2.block.0.1.bias}
          running_mean=t151 {pt2=root:b_features_2_block_0_1_running_mean target=features.2.block.0.1.running_mean}
          running_var=t152 {pt2=root:b_features_2_block_0_1_running_var target=features.2.block.0.1.running_var}
          params={channel=C; eps=0.001}
    n498 {pt2=root[23] torch.ops.aten.relu.default}: [t739 f32 [H=56 W=56 C=72] {derived} ->[n52]] =
      relu x=t292 {derived} <-n47
    group g12 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n55 {derived}: [t300 f32 [H=28 W=28 C=72] {derived} ->[n499]] =
        batch_norm
          x=t297 {derived} <-n52
          weight=t17 {pt2=root:p_features_2_block_1_1_weight target=features.2.block.1.1.weight}
          bias=t18 {pt2=root:p_features_2_block_1_1_bias target=features.2.block.1.1.bias}
          running_mean=t154 {pt2=root:b_features_2_block_1_1_running_mean target=features.2.block.1.1.running_mean}
          running_var=t155 {pt2=root:b_features_2_block_1_1_running_var target=features.2.block.1.1.running_var}
          params={channel=C; eps=0.001}
    n499 {pt2=root[26] torch.ops.aten.relu.default}: [t740 f32 [H=28 W=28 C=72] {derived} ->[n60]] =
      relu x=t300 {derived} <-n55
    group g14 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n63 {derived}: [t308 f32 [H=28 W=28 C=24] {derived} ->[n67, n502]] =
        batch_norm
          x=t305 {derived} <-n60
          weight=t20 {pt2=root:p_features_2_block_2_1_weight target=features.2.block.2.1.weight}
          bias=t21 {pt2=root:p_features_2_block_2_1_bias target=features.2.block.2.1.bias}
          running_mean=t157 {pt2=root:b_features_2_block_2_1_running_mean target=features.2.block.2.1.running_mean}
          running_var=t158 {pt2=root:b_features_2_block_2_1_running_var target=features.2.block.2.1.running_var}
          params={channel=C; eps=0.001}
    group g16 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n70 {derived}: [t315 f32 [H=28 W=28 C=88] {derived} ->[n500]] =
        batch_norm
          x=t312 {derived} <-n67
          weight=t23 {pt2=root:p_features_3_block_0_1_weight target=features.3.block.0.1.weight}
          bias=t24 {pt2=root:p_features_3_block_0_1_bias target=features.3.block.0.1.bias}
          running_mean=t160 {pt2=root:b_features_3_block_0_1_running_mean target=features.3.block.0.1.running_mean}
          running_var=t161 {pt2=root:b_features_3_block_0_1_running_var target=features.3.block.0.1.running_var}
          params={channel=C; eps=0.001}
    n500 {pt2=root[31] torch.ops.aten.relu.default}: [t741 f32 [H=28 W=28 C=88] {derived} ->[n75]] =
      relu x=t315 {derived} <-n70
    group g18 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n78 {derived}: [t323 f32 [H=28 W=28 C=88] {derived} ->[n501]] =
        batch_norm
          x=t320 {derived} <-n75
          weight=t26 {pt2=root:p_features_3_block_1_1_weight target=features.3.block.1.1.weight}
          bias=t27 {pt2=root:p_features_3_block_1_1_bias target=features.3.block.1.1.bias}
          running_mean=t163 {pt2=root:b_features_3_block_1_1_running_mean target=features.3.block.1.1.running_mean}
          running_var=t164 {pt2=root:b_features_3_block_1_1_running_var target=features.3.block.1.1.running_var}
          params={channel=C; eps=0.001}
    n501 {pt2=root[34] torch.ops.aten.relu.default}: [t742 f32 [H=28 W=28 C=88] {derived} ->[n83]] =
      relu x=t323 {derived} <-n78
    group g20 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n86 {derived}: [t331 f32 [H=28 W=28 C=24] {derived} ->[n502]] =
        batch_norm
          x=t328 {derived} <-n83
          weight=t29 {pt2=root:p_features_3_block_2_1_weight target=features.3.block.2.1.weight}
          bias=t30 {pt2=root:p_features_3_block_2_1_bias target=features.3.block.2.1.bias}
          running_mean=t166 {pt2=root:b_features_3_block_2_1_running_mean target=features.3.block.2.1.running_mean}
          running_var=t167 {pt2=root:b_features_3_block_2_1_running_var target=features.3.block.2.1.running_var}
          params={channel=C; eps=0.001}
    n502 {pt2=root[37] torch.ops.aten.add.Tensor}: [t743 f32 [H=28 W=28 C=24] {derived} ->[n91]] =
      add a=t331 {derived} <-n86 b=t308 {derived} <-n63
    group g22 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n94 {derived}: [t339 f32 [H=28 W=28 C=96] {derived} ->[n95]] =
        batch_norm
          x=t336 {derived} <-n91
          weight=t32 {pt2=root:p_features_4_block_0_1_weight target=features.4.block.0.1.weight}
          bias=t33 {pt2=root:p_features_4_block_0_1_bias target=features.4.block.0.1.bias}
          running_mean=t169 {pt2=root:b_features_4_block_0_1_running_mean target=features.4.block.0.1.running_mean}
          running_var=t170 {pt2=root:b_features_4_block_0_1_running_var target=features.4.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n95 {pt2=root[39] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t340 f32 [H=96
                                                                      W=28
                                                                      C=28] {pt2=root:getitem_27} ->[n96,
                                                                      n99]] =
        permute x=t339 {derived} <-n94 perm=[H<-C, W<-H, C<-W]
    n96 {pt2=root[40] torch.ops.aten.add.Tensor}: [t341 f32 [H=96 W=28 C=28] {pt2=root:add_3} ->[n97]] =
      add_scalar x=t340 {pt2=root:getitem_27} <-n95 scalar=3
    n97 {pt2=root[41] torch.ops.aten.clamp.default}: [t342 f32 [H=96 W=28 C=28] {pt2=root:clamp_4} ->[n98]] =
      clamp x=t341 {pt2=root:add_3} <-n96 params={min=0; max=none}
    n98 {pt2=root[42] torch.ops.aten.clamp.default}: [t343 f32 [H=96 W=28 C=28] {pt2=root:clamp_5} ->[n99]] =
      clamp x=t342 {pt2=root:clamp_4} <-n97 params={min=none; max=6}
    n99 {pt2=root[43] torch.ops.aten.mul.Tensor}: [t344 f32 [H=96 W=28 C=28] {pt2=root:mul_2} ->[n100]] =
      mul a=t340 {pt2=root:getitem_27} <-n95 b=t343 {pt2=root:clamp_5} <-n98
    n100 {pt2=root[44] torch.ops.aten.div.Tensor}: [t345 f32 [H=96 W=28 C=28] {pt2=root:div_2} ->[n101]] =
      div_scalar x=t344 {pt2=root:mul_2} <-n99 scalar=6
    group g24 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n106 {derived}: [t351 f32 [H=14 W=14 C=96] {derived} ->[n107]] =
        batch_norm
          x=t348 {derived} <-n103
          weight=t35 {pt2=root:p_features_4_block_1_1_weight target=features.4.block.1.1.weight}
          bias=t36 {pt2=root:p_features_4_block_1_1_bias target=features.4.block.1.1.bias}
          running_mean=t172 {pt2=root:b_features_4_block_1_1_running_mean target=features.4.block.1.1.running_mean}
          running_var=t173 {pt2=root:b_features_4_block_1_1_running_var target=features.4.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n107 {pt2=root[46] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t352 f32 [H=96
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_30} ->[n108,
                                                                      n111]] =
        permute x=t351 {derived} <-n106 perm=[H<-C, W<-H, C<-W]
    n108 {pt2=root[47] torch.ops.aten.add.Tensor}: [t353 f32 [H=96 W=14 C=14] {pt2=root:add_4} ->[n109]] =
      add_scalar x=t352 {pt2=root:getitem_30} <-n107 scalar=3
    n109 {pt2=root[48] torch.ops.aten.clamp.default}: [t354 f32 [H=96 W=14
                                                                 C=14] {pt2=root:clamp_6} ->[n110]] =
      clamp x=t353 {pt2=root:add_4} <-n108 params={min=0; max=none}
    n110 {pt2=root[49] torch.ops.aten.clamp.default}: [t355 f32 [H=96 W=14
                                                                 C=14] {pt2=root:clamp_7} ->[n111]] =
      clamp x=t354 {pt2=root:clamp_6} <-n109 params={min=none; max=6}
    n111 {pt2=root[50] torch.ops.aten.mul.Tensor}: [t356 f32 [H=96 W=14 C=14] {pt2=root:mul_3} ->[n112]] =
      mul a=t352 {pt2=root:getitem_30} <-n107 b=t355 {pt2=root:clamp_7} <-n110
    n112 {pt2=root[51] torch.ops.aten.div.Tensor}: [t357 f32 [H=96 W=14 C=14] {pt2=root:div_3} ->[n113,
                                                                      n127]] =
      div_scalar x=t356 {pt2=root:mul_3} <-n111 scalar=6
    n113 {pt2=root[52] torch.ops.aten.mean.dim}: [t358 f32 [H=96 W=1 C=1] {pt2=root:mean_1} ->[n114]] =
      mean x=t357 {pt2=root:div_3} <-n112 params={dims=[C, W]; keepdim=true}
    n503 {pt2=root[54] torch.ops.aten.relu.default}: [t744 f32 [C=24] {derived} ->[n121]] =
      relu x=t361 {derived} <-n116
    n504 {pt2=root[56] torch.ops.aten.add.Tensor}: [t745 f32 [C=96] {derived} ->[n505]] =
      add_scalar x=t366 {derived} <-n121 scalar=3
    n505 {pt2=root[57] torch.ops.aten.clamp.default}: [t746 f32 [C=96] {derived} ->[n506]] =
      clamp x=t745 {derived} <-n504 params={min=0; max=none}
    n506 {pt2=root[58] torch.ops.aten.clamp.default}: [t747 f32 [C=96] {derived} ->[n507]] =
      clamp x=t746 {derived} <-n505 params={min=none; max=6}
    n507 {pt2=root[59] torch.ops.aten.div.Tensor}: [t748 f32 [C=96] {derived} ->[n508]] =
      div_scalar x=t747 {derived} <-n506 scalar=6
    n508 {pt2=root[55] torch.ops.aten.convolution.default}: [t371 f32 [H=96 W=1
                                                                      C=1] {pt2=root:div_4} ->[n127]] =
      permute x=t748 {derived} <-n507 perm=[H<-C, W<-H, C<-W]
    n127 {pt2=root[60] torch.ops.aten.mul.Tensor}: [t372 f32 [H=96 W=14 C=14] {pt2=root:mul_4} ->[n128]] =
      mul a=t371 {pt2=root:div_4} <-n508 b=t357 {pt2=root:div_3} <-n112
    group g28 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n133 {derived}: [t378 f32 [H=14 W=14 C=40] {derived} ->[n137, n515]] =
        batch_norm
          x=t375 {derived} <-n130
          weight=t42 {pt2=root:p_features_4_block_3_1_weight target=features.4.block.3.1.weight}
          bias=t43 {pt2=root:p_features_4_block_3_1_bias target=features.4.block.3.1.bias}
          running_mean=t175 {pt2=root:b_features_4_block_3_1_running_mean target=features.4.block.3.1.running_mean}
          running_var=t176 {pt2=root:b_features_4_block_3_1_running_var target=features.4.block.3.1.running_var}
          params={channel=C; eps=0.001}
    group g30 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n140 {derived}: [t385 f32 [H=14 W=14 C=240] {derived} ->[n141]] =
        batch_norm
          x=t382 {derived} <-n137
          weight=t45 {pt2=root:p_features_5_block_0_1_weight target=features.5.block.0.1.weight}
          bias=t46 {pt2=root:p_features_5_block_0_1_bias target=features.5.block.0.1.bias}
          running_mean=t178 {pt2=root:b_features_5_block_0_1_running_mean target=features.5.block.0.1.running_mean}
          running_var=t179 {pt2=root:b_features_5_block_0_1_running_var target=features.5.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n141 {pt2=root[64] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t386 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_36} ->[n142,
                                                                      n145]] =
        permute x=t385 {derived} <-n140 perm=[H<-C, W<-H, C<-W]
    n142 {pt2=root[65] torch.ops.aten.add.Tensor}: [t387 f32 [H=240 W=14 C=14] {pt2=root:add_6} ->[n143]] =
      add_scalar x=t386 {pt2=root:getitem_36} <-n141 scalar=3
    n143 {pt2=root[66] torch.ops.aten.clamp.default}: [t388 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_10} ->[n144]] =
      clamp x=t387 {pt2=root:add_6} <-n142 params={min=0; max=none}
    n144 {pt2=root[67] torch.ops.aten.clamp.default}: [t389 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_11} ->[n145]] =
      clamp x=t388 {pt2=root:clamp_10} <-n143 params={min=none; max=6}
    n145 {pt2=root[68] torch.ops.aten.mul.Tensor}: [t390 f32 [H=240 W=14 C=14] {pt2=root:mul_5} ->[n146]] =
      mul a=t386 {pt2=root:getitem_36} <-n141 b=t389 {pt2=root:clamp_11} <-n144
    n146 {pt2=root[69] torch.ops.aten.div.Tensor}: [t391 f32 [H=240 W=14 C=14] {pt2=root:div_5} ->[n147]] =
      div_scalar x=t390 {pt2=root:mul_5} <-n145 scalar=6
    group g32 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n152 {derived}: [t397 f32 [H=14 W=14 C=240] {derived} ->[n153]] =
        batch_norm
          x=t394 {derived} <-n149
          weight=t48 {pt2=root:p_features_5_block_1_1_weight target=features.5.block.1.1.weight}
          bias=t49 {pt2=root:p_features_5_block_1_1_bias target=features.5.block.1.1.bias}
          running_mean=t181 {pt2=root:b_features_5_block_1_1_running_mean target=features.5.block.1.1.running_mean}
          running_var=t182 {pt2=root:b_features_5_block_1_1_running_var target=features.5.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n153 {pt2=root[71] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t398 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_39} ->[n154,
                                                                      n157]] =
        permute x=t397 {derived} <-n152 perm=[H<-C, W<-H, C<-W]
    n154 {pt2=root[72] torch.ops.aten.add.Tensor}: [t399 f32 [H=240 W=14 C=14] {pt2=root:add_7} ->[n155]] =
      add_scalar x=t398 {pt2=root:getitem_39} <-n153 scalar=3
    n155 {pt2=root[73] torch.ops.aten.clamp.default}: [t400 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_12} ->[n156]] =
      clamp x=t399 {pt2=root:add_7} <-n154 params={min=0; max=none}
    n156 {pt2=root[74] torch.ops.aten.clamp.default}: [t401 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_13} ->[n157]] =
      clamp x=t400 {pt2=root:clamp_12} <-n155 params={min=none; max=6}
    n157 {pt2=root[75] torch.ops.aten.mul.Tensor}: [t402 f32 [H=240 W=14 C=14] {pt2=root:mul_6} ->[n158]] =
      mul a=t398 {pt2=root:getitem_39} <-n153 b=t401 {pt2=root:clamp_13} <-n156
    n158 {pt2=root[76] torch.ops.aten.div.Tensor}: [t403 f32 [H=240 W=14 C=14] {pt2=root:div_6} ->[n159,
                                                                      n173]] =
      div_scalar x=t402 {pt2=root:mul_6} <-n157 scalar=6
    n159 {pt2=root[77] torch.ops.aten.mean.dim}: [t404 f32 [H=240 W=1 C=1] {pt2=root:mean_2} ->[n160]] =
      mean x=t403 {pt2=root:div_6} <-n158 params={dims=[C, W]; keepdim=true}
    n509 {pt2=root[79] torch.ops.aten.relu.default}: [t749 f32 [C=64] {derived} ->[n167]] =
      relu x=t407 {derived} <-n162
    n510 {pt2=root[81] torch.ops.aten.add.Tensor}: [t750 f32 [C=240] {derived} ->[n511]] =
      add_scalar x=t412 {derived} <-n167 scalar=3
    n511 {pt2=root[82] torch.ops.aten.clamp.default}: [t751 f32 [C=240] {derived} ->[n512]] =
      clamp x=t750 {derived} <-n510 params={min=0; max=none}
    n512 {pt2=root[83] torch.ops.aten.clamp.default}: [t752 f32 [C=240] {derived} ->[n513]] =
      clamp x=t751 {derived} <-n511 params={min=none; max=6}
    n513 {pt2=root[84] torch.ops.aten.div.Tensor}: [t753 f32 [C=240] {derived} ->[n514]] =
      div_scalar x=t752 {derived} <-n512 scalar=6
    n514 {pt2=root[80] torch.ops.aten.convolution.default}: [t417 f32 [H=240
                                                                      W=1 C=1] {pt2=root:div_7} ->[n173]] =
      permute x=t753 {derived} <-n513 perm=[H<-C, W<-H, C<-W]
    n173 {pt2=root[85] torch.ops.aten.mul.Tensor}: [t418 f32 [H=240 W=14 C=14] {pt2=root:mul_7} ->[n174]] =
      mul a=t417 {pt2=root:div_7} <-n514 b=t403 {pt2=root:div_6} <-n158
    group g36 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n179 {derived}: [t424 f32 [H=14 W=14 C=40] {derived} ->[n515]] =
        batch_norm
          x=t421 {derived} <-n176
          weight=t55 {pt2=root:p_features_5_block_3_1_weight target=features.5.block.3.1.weight}
          bias=t56 {pt2=root:p_features_5_block_3_1_bias target=features.5.block.3.1.bias}
          running_mean=t184 {pt2=root:b_features_5_block_3_1_running_mean target=features.5.block.3.1.running_mean}
          running_var=t185 {pt2=root:b_features_5_block_3_1_running_var target=features.5.block.3.1.running_var}
          params={channel=C; eps=0.001}
    n515 {pt2=root[88] torch.ops.aten.add.Tensor}: [t754 f32 [H=14 W=14 C=40] {derived} ->[n184,
                                                                      n522]] =
      add a=t424 {derived} <-n179 b=t378 {derived} <-n133
    group g38 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n187 {derived}: [t432 f32 [H=14 W=14 C=240] {derived} ->[n188]] =
        batch_norm
          x=t429 {derived} <-n184
          weight=t58 {pt2=root:p_features_6_block_0_1_weight target=features.6.block.0.1.weight}
          bias=t59 {pt2=root:p_features_6_block_0_1_bias target=features.6.block.0.1.bias}
          running_mean=t187 {pt2=root:b_features_6_block_0_1_running_mean target=features.6.block.0.1.running_mean}
          running_var=t188 {pt2=root:b_features_6_block_0_1_running_var target=features.6.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n188 {pt2=root[90] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t433 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_45} ->[n189,
                                                                      n192]] =
        permute x=t432 {derived} <-n187 perm=[H<-C, W<-H, C<-W]
    n189 {pt2=root[91] torch.ops.aten.add.Tensor}: [t434 f32 [H=240 W=14 C=14] {pt2=root:add_10} ->[n190]] =
      add_scalar x=t433 {pt2=root:getitem_45} <-n188 scalar=3
    n190 {pt2=root[92] torch.ops.aten.clamp.default}: [t435 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_16} ->[n191]] =
      clamp x=t434 {pt2=root:add_10} <-n189 params={min=0; max=none}
    n191 {pt2=root[93] torch.ops.aten.clamp.default}: [t436 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_17} ->[n192]] =
      clamp x=t435 {pt2=root:clamp_16} <-n190 params={min=none; max=6}
    n192 {pt2=root[94] torch.ops.aten.mul.Tensor}: [t437 f32 [H=240 W=14 C=14] {pt2=root:mul_8} ->[n193]] =
      mul a=t433 {pt2=root:getitem_45} <-n188 b=t436 {pt2=root:clamp_17} <-n191
    n193 {pt2=root[95] torch.ops.aten.div.Tensor}: [t438 f32 [H=240 W=14 C=14] {pt2=root:div_8} ->[n194]] =
      div_scalar x=t437 {pt2=root:mul_8} <-n192 scalar=6
    group g40 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n199 {derived}: [t444 f32 [H=14 W=14 C=240] {derived} ->[n200]] =
        batch_norm
          x=t441 {derived} <-n196
          weight=t61 {pt2=root:p_features_6_block_1_1_weight target=features.6.block.1.1.weight}
          bias=t62 {pt2=root:p_features_6_block_1_1_bias target=features.6.block.1.1.bias}
          running_mean=t190 {pt2=root:b_features_6_block_1_1_running_mean target=features.6.block.1.1.running_mean}
          running_var=t191 {pt2=root:b_features_6_block_1_1_running_var target=features.6.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n200 {pt2=root[97] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t445 f32 [H=240
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_48} ->[n201,
                                                                      n204]] =
        permute x=t444 {derived} <-n199 perm=[H<-C, W<-H, C<-W]
    n201 {pt2=root[98] torch.ops.aten.add.Tensor}: [t446 f32 [H=240 W=14 C=14] {pt2=root:add_11} ->[n202]] =
      add_scalar x=t445 {pt2=root:getitem_48} <-n200 scalar=3
    n202 {pt2=root[99] torch.ops.aten.clamp.default}: [t447 f32 [H=240 W=14
                                                                 C=14] {pt2=root:clamp_18} ->[n203]] =
      clamp x=t446 {pt2=root:add_11} <-n201 params={min=0; max=none}
    n203 {pt2=root[100] torch.ops.aten.clamp.default}: [t448 f32 [H=240 W=14
                                                                  C=14] {pt2=root:clamp_19} ->[n204]] =
      clamp x=t447 {pt2=root:clamp_18} <-n202 params={min=none; max=6}
    n204 {pt2=root[101] torch.ops.aten.mul.Tensor}: [t449 f32 [H=240 W=14 C=14] {pt2=root:mul_9} ->[n205]] =
      mul a=t445 {pt2=root:getitem_48} <-n200 b=t448 {pt2=root:clamp_19} <-n203
    n205 {pt2=root[102] torch.ops.aten.div.Tensor}: [t450 f32 [H=240 W=14 C=14] {pt2=root:div_9} ->[n206,
                                                                      n220]] =
      div_scalar x=t449 {pt2=root:mul_9} <-n204 scalar=6
    n206 {pt2=root[103] torch.ops.aten.mean.dim}: [t451 f32 [H=240 W=1 C=1] {pt2=root:mean_3} ->[n207]] =
      mean x=t450 {pt2=root:div_9} <-n205 params={dims=[C, W]; keepdim=true}
    n516 {pt2=root[105] torch.ops.aten.relu.default}: [t755 f32 [C=64] {derived} ->[n214]] =
      relu x=t454 {derived} <-n209
    n517 {pt2=root[107] torch.ops.aten.add.Tensor}: [t756 f32 [C=240] {derived} ->[n518]] =
      add_scalar x=t459 {derived} <-n214 scalar=3
    n518 {pt2=root[108] torch.ops.aten.clamp.default}: [t757 f32 [C=240] {derived} ->[n519]] =
      clamp x=t756 {derived} <-n517 params={min=0; max=none}
    n519 {pt2=root[109] torch.ops.aten.clamp.default}: [t758 f32 [C=240] {derived} ->[n520]] =
      clamp x=t757 {derived} <-n518 params={min=none; max=6}
    n520 {pt2=root[110] torch.ops.aten.div.Tensor}: [t759 f32 [C=240] {derived} ->[n521]] =
      div_scalar x=t758 {derived} <-n519 scalar=6
    n521 {pt2=root[106] torch.ops.aten.convolution.default}: [t464 f32 [H=240
                                                                      W=1 C=1] {pt2=root:div_10} ->[n220]] =
      permute x=t759 {derived} <-n520 perm=[H<-C, W<-H, C<-W]
    n220 {pt2=root[111] torch.ops.aten.mul.Tensor}: [t465 f32 [H=240 W=14 C=14] {pt2=root:mul_10} ->[n221]] =
      mul a=t464 {pt2=root:div_10} <-n521 b=t450 {pt2=root:div_9} <-n205
    group g44 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n226 {derived}: [t471 f32 [H=14 W=14 C=40] {derived} ->[n522]] =
        batch_norm
          x=t468 {derived} <-n223
          weight=t68 {pt2=root:p_features_6_block_3_1_weight target=features.6.block.3.1.weight}
          bias=t69 {pt2=root:p_features_6_block_3_1_bias target=features.6.block.3.1.bias}
          running_mean=t193 {pt2=root:b_features_6_block_3_1_running_mean target=features.6.block.3.1.running_mean}
          running_var=t194 {pt2=root:b_features_6_block_3_1_running_var target=features.6.block.3.1.running_var}
          params={channel=C; eps=0.001}
    n522 {pt2=root[114] torch.ops.aten.add.Tensor}: [t760 f32 [H=14 W=14 C=40] {derived} ->[n231]] =
      add a=t471 {derived} <-n226 b=t754 {derived} <-n515
    group g46 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n234 {derived}: [t479 f32 [H=14 W=14 C=120] {derived} ->[n235]] =
        batch_norm
          x=t476 {derived} <-n231
          weight=t71 {pt2=root:p_features_7_block_0_1_weight target=features.7.block.0.1.weight}
          bias=t72 {pt2=root:p_features_7_block_0_1_bias target=features.7.block.0.1.bias}
          running_mean=t196 {pt2=root:b_features_7_block_0_1_running_mean target=features.7.block.0.1.running_mean}
          running_var=t197 {pt2=root:b_features_7_block_0_1_running_var target=features.7.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n235 {pt2=root[116] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t480 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_54} ->[n236,
                                                                      n239]] =
        permute x=t479 {derived} <-n234 perm=[H<-C, W<-H, C<-W]
    n236 {pt2=root[117] torch.ops.aten.add.Tensor}: [t481 f32 [H=120 W=14 C=14] {pt2=root:add_14} ->[n237]] =
      add_scalar x=t480 {pt2=root:getitem_54} <-n235 scalar=3
    n237 {pt2=root[118] torch.ops.aten.clamp.default}: [t482 f32 [H=120 W=14
                                                                  C=14] {pt2=root:clamp_22} ->[n238]] =
      clamp x=t481 {pt2=root:add_14} <-n236 params={min=0; max=none}
    n238 {pt2=root[119] torch.ops.aten.clamp.default}: [t483 f32 [H=120 W=14
                                                                  C=14] {pt2=root:clamp_23} ->[n239]] =
      clamp x=t482 {pt2=root:clamp_22} <-n237 params={min=none; max=6}
    n239 {pt2=root[120] torch.ops.aten.mul.Tensor}: [t484 f32 [H=120 W=14 C=14] {pt2=root:mul_11} ->[n240]] =
      mul a=t480 {pt2=root:getitem_54} <-n235 b=t483 {pt2=root:clamp_23} <-n238
    n240 {pt2=root[121] torch.ops.aten.div.Tensor}: [t485 f32 [H=120 W=14 C=14] {pt2=root:div_11} ->[n241]] =
      div_scalar x=t484 {pt2=root:mul_11} <-n239 scalar=6
    group g48 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n246 {derived}: [t491 f32 [H=14 W=14 C=120] {derived} ->[n247]] =
        batch_norm
          x=t488 {derived} <-n243
          weight=t74 {pt2=root:p_features_7_block_1_1_weight target=features.7.block.1.1.weight}
          bias=t75 {pt2=root:p_features_7_block_1_1_bias target=features.7.block.1.1.bias}
          running_mean=t199 {pt2=root:b_features_7_block_1_1_running_mean target=features.7.block.1.1.running_mean}
          running_var=t200 {pt2=root:b_features_7_block_1_1_running_var target=features.7.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n247 {pt2=root[123] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t492 f32 [H=120
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_57} ->[n248,
                                                                      n251]] =
        permute x=t491 {derived} <-n246 perm=[H<-C, W<-H, C<-W]
    n248 {pt2=root[124] torch.ops.aten.add.Tensor}: [t493 f32 [H=120 W=14 C=14] {pt2=root:add_15} ->[n249]] =
      add_scalar x=t492 {pt2=root:getitem_57} <-n247 scalar=3
    n249 {pt2=root[125] torch.ops.aten.clamp.default}: [t494 f32 [H=120 W=14
                                                                  C=14] {pt2=root:clamp_24} ->[n250]] =
      clamp x=t493 {pt2=root:add_15} <-n248 params={min=0; max=none}
    n250 {pt2=root[126] torch.ops.aten.clamp.default}: [t495 f32 [H=120 W=14
                                                                  C=14] {pt2=root:clamp_25} ->[n251]] =
      clamp x=t494 {pt2=root:clamp_24} <-n249 params={min=none; max=6}
    n251 {pt2=root[127] torch.ops.aten.mul.Tensor}: [t496 f32 [H=120 W=14 C=14] {pt2=root:mul_12} ->[n252]] =
      mul a=t492 {pt2=root:getitem_57} <-n247 b=t495 {pt2=root:clamp_25} <-n250
    n252 {pt2=root[128] torch.ops.aten.div.Tensor}: [t497 f32 [H=120 W=14 C=14] {pt2=root:div_12} ->[n253,
                                                                      n267]] =
      div_scalar x=t496 {pt2=root:mul_12} <-n251 scalar=6
    n253 {pt2=root[129] torch.ops.aten.mean.dim}: [t498 f32 [H=120 W=1 C=1] {pt2=root:mean_4} ->[n254]] =
      mean x=t497 {pt2=root:div_12} <-n252 params={dims=[C, W]; keepdim=true}
    n523 {pt2=root[131] torch.ops.aten.relu.default}: [t761 f32 [C=32] {derived} ->[n261]] =
      relu x=t501 {derived} <-n256
    n524 {pt2=root[133] torch.ops.aten.add.Tensor}: [t762 f32 [C=120] {derived} ->[n525]] =
      add_scalar x=t506 {derived} <-n261 scalar=3
    n525 {pt2=root[134] torch.ops.aten.clamp.default}: [t763 f32 [C=120] {derived} ->[n526]] =
      clamp x=t762 {derived} <-n524 params={min=0; max=none}
    n526 {pt2=root[135] torch.ops.aten.clamp.default}: [t764 f32 [C=120] {derived} ->[n527]] =
      clamp x=t763 {derived} <-n525 params={min=none; max=6}
    n527 {pt2=root[136] torch.ops.aten.div.Tensor}: [t765 f32 [C=120] {derived} ->[n528]] =
      div_scalar x=t764 {derived} <-n526 scalar=6
    n528 {pt2=root[132] torch.ops.aten.convolution.default}: [t511 f32 [H=120
                                                                      W=1 C=1] {pt2=root:div_13} ->[n267]] =
      permute x=t765 {derived} <-n527 perm=[H<-C, W<-H, C<-W]
    n267 {pt2=root[137] torch.ops.aten.mul.Tensor}: [t512 f32 [H=120 W=14 C=14] {pt2=root:mul_13} ->[n268]] =
      mul a=t511 {pt2=root:div_13} <-n528 b=t497 {pt2=root:div_12} <-n252
    group g52 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n273 {derived}: [t518 f32 [H=14 W=14 C=48] {derived} ->[n277, n535]] =
        batch_norm
          x=t515 {derived} <-n270
          weight=t81 {pt2=root:p_features_7_block_3_1_weight target=features.7.block.3.1.weight}
          bias=t82 {pt2=root:p_features_7_block_3_1_bias target=features.7.block.3.1.bias}
          running_mean=t202 {pt2=root:b_features_7_block_3_1_running_mean target=features.7.block.3.1.running_mean}
          running_var=t203 {pt2=root:b_features_7_block_3_1_running_var target=features.7.block.3.1.running_var}
          params={channel=C; eps=0.001}
    group g54 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n280 {derived}: [t525 f32 [H=14 W=14 C=144] {derived} ->[n281]] =
        batch_norm
          x=t522 {derived} <-n277
          weight=t84 {pt2=root:p_features_8_block_0_1_weight target=features.8.block.0.1.weight}
          bias=t85 {pt2=root:p_features_8_block_0_1_bias target=features.8.block.0.1.bias}
          running_mean=t205 {pt2=root:b_features_8_block_0_1_running_mean target=features.8.block.0.1.running_mean}
          running_var=t206 {pt2=root:b_features_8_block_0_1_running_var target=features.8.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n281 {pt2=root[141] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t526 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_63} ->[n282,
                                                                      n285]] =
        permute x=t525 {derived} <-n280 perm=[H<-C, W<-H, C<-W]
    n282 {pt2=root[142] torch.ops.aten.add.Tensor}: [t527 f32 [H=144 W=14 C=14] {pt2=root:add_17} ->[n283]] =
      add_scalar x=t526 {pt2=root:getitem_63} <-n281 scalar=3
    n283 {pt2=root[143] torch.ops.aten.clamp.default}: [t528 f32 [H=144 W=14
                                                                  C=14] {pt2=root:clamp_28} ->[n284]] =
      clamp x=t527 {pt2=root:add_17} <-n282 params={min=0; max=none}
    n284 {pt2=root[144] torch.ops.aten.clamp.default}: [t529 f32 [H=144 W=14
                                                                  C=14] {pt2=root:clamp_29} ->[n285]] =
      clamp x=t528 {pt2=root:clamp_28} <-n283 params={min=none; max=6}
    n285 {pt2=root[145] torch.ops.aten.mul.Tensor}: [t530 f32 [H=144 W=14 C=14] {pt2=root:mul_14} ->[n286]] =
      mul a=t526 {pt2=root:getitem_63} <-n281 b=t529 {pt2=root:clamp_29} <-n284
    n286 {pt2=root[146] torch.ops.aten.div.Tensor}: [t531 f32 [H=144 W=14 C=14] {pt2=root:div_14} ->[n287]] =
      div_scalar x=t530 {pt2=root:mul_14} <-n285 scalar=6
    group g56 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n292 {derived}: [t537 f32 [H=14 W=14 C=144] {derived} ->[n293]] =
        batch_norm
          x=t534 {derived} <-n289
          weight=t87 {pt2=root:p_features_8_block_1_1_weight target=features.8.block.1.1.weight}
          bias=t88 {pt2=root:p_features_8_block_1_1_bias target=features.8.block.1.1.bias}
          running_mean=t208 {pt2=root:b_features_8_block_1_1_running_mean target=features.8.block.1.1.running_mean}
          running_var=t209 {pt2=root:b_features_8_block_1_1_running_var target=features.8.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n293 {pt2=root[148] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t538 f32 [H=144
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_66} ->[n294,
                                                                      n297]] =
        permute x=t537 {derived} <-n292 perm=[H<-C, W<-H, C<-W]
    n294 {pt2=root[149] torch.ops.aten.add.Tensor}: [t539 f32 [H=144 W=14 C=14] {pt2=root:add_18} ->[n295]] =
      add_scalar x=t538 {pt2=root:getitem_66} <-n293 scalar=3
    n295 {pt2=root[150] torch.ops.aten.clamp.default}: [t540 f32 [H=144 W=14
                                                                  C=14] {pt2=root:clamp_30} ->[n296]] =
      clamp x=t539 {pt2=root:add_18} <-n294 params={min=0; max=none}
    n296 {pt2=root[151] torch.ops.aten.clamp.default}: [t541 f32 [H=144 W=14
                                                                  C=14] {pt2=root:clamp_31} ->[n297]] =
      clamp x=t540 {pt2=root:clamp_30} <-n295 params={min=none; max=6}
    n297 {pt2=root[152] torch.ops.aten.mul.Tensor}: [t542 f32 [H=144 W=14 C=14] {pt2=root:mul_15} ->[n298]] =
      mul a=t538 {pt2=root:getitem_66} <-n293 b=t541 {pt2=root:clamp_31} <-n296
    n298 {pt2=root[153] torch.ops.aten.div.Tensor}: [t543 f32 [H=144 W=14 C=14] {pt2=root:div_15} ->[n299,
                                                                      n313]] =
      div_scalar x=t542 {pt2=root:mul_15} <-n297 scalar=6
    n299 {pt2=root[154] torch.ops.aten.mean.dim}: [t544 f32 [H=144 W=1 C=1] {pt2=root:mean_5} ->[n300]] =
      mean x=t543 {pt2=root:div_15} <-n298 params={dims=[C, W]; keepdim=true}
    n529 {pt2=root[156] torch.ops.aten.relu.default}: [t766 f32 [C=40] {derived} ->[n307]] =
      relu x=t547 {derived} <-n302
    n530 {pt2=root[158] torch.ops.aten.add.Tensor}: [t767 f32 [C=144] {derived} ->[n531]] =
      add_scalar x=t552 {derived} <-n307 scalar=3
    n531 {pt2=root[159] torch.ops.aten.clamp.default}: [t768 f32 [C=144] {derived} ->[n532]] =
      clamp x=t767 {derived} <-n530 params={min=0; max=none}
    n532 {pt2=root[160] torch.ops.aten.clamp.default}: [t769 f32 [C=144] {derived} ->[n533]] =
      clamp x=t768 {derived} <-n531 params={min=none; max=6}
    n533 {pt2=root[161] torch.ops.aten.div.Tensor}: [t770 f32 [C=144] {derived} ->[n534]] =
      div_scalar x=t769 {derived} <-n532 scalar=6
    n534 {pt2=root[157] torch.ops.aten.convolution.default}: [t557 f32 [H=144
                                                                      W=1 C=1] {pt2=root:div_16} ->[n313]] =
      permute x=t770 {derived} <-n533 perm=[H<-C, W<-H, C<-W]
    n313 {pt2=root[162] torch.ops.aten.mul.Tensor}: [t558 f32 [H=144 W=14 C=14] {pt2=root:mul_16} ->[n314]] =
      mul a=t557 {pt2=root:div_16} <-n534 b=t543 {pt2=root:div_15} <-n298
    group g60 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n319 {derived}: [t564 f32 [H=14 W=14 C=48] {derived} ->[n535]] =
        batch_norm
          x=t561 {derived} <-n316
          weight=t94 {pt2=root:p_features_8_block_3_1_weight target=features.8.block.3.1.weight}
          bias=t95 {pt2=root:p_features_8_block_3_1_bias target=features.8.block.3.1.bias}
          running_mean=t211 {pt2=root:b_features_8_block_3_1_running_mean target=features.8.block.3.1.running_mean}
          running_var=t212 {pt2=root:b_features_8_block_3_1_running_var target=features.8.block.3.1.running_var}
          params={channel=C; eps=0.001}
    n535 {pt2=root[165] torch.ops.aten.add.Tensor}: [t771 f32 [H=14 W=14 C=48] {derived} ->[n324]] =
      add a=t564 {derived} <-n319 b=t518 {derived} <-n273
    group g62 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n327 {derived}: [t572 f32 [H=14 W=14 C=288] {derived} ->[n328]] =
        batch_norm
          x=t569 {derived} <-n324
          weight=t97 {pt2=root:p_features_9_block_0_1_weight target=features.9.block.0.1.weight}
          bias=t98 {pt2=root:p_features_9_block_0_1_bias target=features.9.block.0.1.bias}
          running_mean=t214 {pt2=root:b_features_9_block_0_1_running_mean target=features.9.block.0.1.running_mean}
          running_var=t215 {pt2=root:b_features_9_block_0_1_running_var target=features.9.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n328 {pt2=root[167] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t573 f32 [H=288
                                                                      W=14
                                                                      C=14] {pt2=root:getitem_72} ->[n329,
                                                                      n332]] =
        permute x=t572 {derived} <-n327 perm=[H<-C, W<-H, C<-W]
    n329 {pt2=root[168] torch.ops.aten.add.Tensor}: [t574 f32 [H=288 W=14 C=14] {pt2=root:add_21} ->[n330]] =
      add_scalar x=t573 {pt2=root:getitem_72} <-n328 scalar=3
    n330 {pt2=root[169] torch.ops.aten.clamp.default}: [t575 f32 [H=288 W=14
                                                                  C=14] {pt2=root:clamp_34} ->[n331]] =
      clamp x=t574 {pt2=root:add_21} <-n329 params={min=0; max=none}
    n331 {pt2=root[170] torch.ops.aten.clamp.default}: [t576 f32 [H=288 W=14
                                                                  C=14] {pt2=root:clamp_35} ->[n332]] =
      clamp x=t575 {pt2=root:clamp_34} <-n330 params={min=none; max=6}
    n332 {pt2=root[171] torch.ops.aten.mul.Tensor}: [t577 f32 [H=288 W=14 C=14] {pt2=root:mul_17} ->[n333]] =
      mul a=t573 {pt2=root:getitem_72} <-n328 b=t576 {pt2=root:clamp_35} <-n331
    n333 {pt2=root[172] torch.ops.aten.div.Tensor}: [t578 f32 [H=288 W=14 C=14] {pt2=root:div_17} ->[n334]] =
      div_scalar x=t577 {pt2=root:mul_17} <-n332 scalar=6
    group g64 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n339 {derived}: [t584 f32 [H=7 W=7 C=288] {derived} ->[n340]] =
        batch_norm
          x=t581 {derived} <-n336
          weight=t100 {pt2=root:p_features_9_block_1_1_weight target=features.9.block.1.1.weight}
          bias=t101 {pt2=root:p_features_9_block_1_1_bias target=features.9.block.1.1.bias}
          running_mean=t217 {pt2=root:b_features_9_block_1_1_running_mean target=features.9.block.1.1.running_mean}
          running_var=t218 {pt2=root:b_features_9_block_1_1_running_var target=features.9.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n340 {pt2=root[174] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t585 f32 [H=288
                                                                      W=7 C=7] {pt2=root:getitem_75} ->[n341,
                                                                      n344]] =
        permute x=t584 {derived} <-n339 perm=[H<-C, W<-H, C<-W]
    n341 {pt2=root[175] torch.ops.aten.add.Tensor}: [t586 f32 [H=288 W=7 C=7] {pt2=root:add_22} ->[n342]] =
      add_scalar x=t585 {pt2=root:getitem_75} <-n340 scalar=3
    n342 {pt2=root[176] torch.ops.aten.clamp.default}: [t587 f32 [H=288 W=7
                                                                  C=7] {pt2=root:clamp_36} ->[n343]] =
      clamp x=t586 {pt2=root:add_22} <-n341 params={min=0; max=none}
    n343 {pt2=root[177] torch.ops.aten.clamp.default}: [t588 f32 [H=288 W=7
                                                                  C=7] {pt2=root:clamp_37} ->[n344]] =
      clamp x=t587 {pt2=root:clamp_36} <-n342 params={min=none; max=6}
    n344 {pt2=root[178] torch.ops.aten.mul.Tensor}: [t589 f32 [H=288 W=7 C=7] {pt2=root:mul_18} ->[n345]] =
      mul a=t585 {pt2=root:getitem_75} <-n340 b=t588 {pt2=root:clamp_37} <-n343
    n345 {pt2=root[179] torch.ops.aten.div.Tensor}: [t590 f32 [H=288 W=7 C=7] {pt2=root:div_18} ->[n346,
                                                                      n360]] =
      div_scalar x=t589 {pt2=root:mul_18} <-n344 scalar=6
    n346 {pt2=root[180] torch.ops.aten.mean.dim}: [t591 f32 [H=288 W=1 C=1] {pt2=root:mean_6} ->[n347]] =
      mean x=t590 {pt2=root:div_18} <-n345 params={dims=[C, W]; keepdim=true}
    n536 {pt2=root[182] torch.ops.aten.relu.default}: [t772 f32 [C=72] {derived} ->[n354]] =
      relu x=t594 {derived} <-n349
    n537 {pt2=root[184] torch.ops.aten.add.Tensor}: [t773 f32 [C=288] {derived} ->[n538]] =
      add_scalar x=t599 {derived} <-n354 scalar=3
    n538 {pt2=root[185] torch.ops.aten.clamp.default}: [t774 f32 [C=288] {derived} ->[n539]] =
      clamp x=t773 {derived} <-n537 params={min=0; max=none}
    n539 {pt2=root[186] torch.ops.aten.clamp.default}: [t775 f32 [C=288] {derived} ->[n540]] =
      clamp x=t774 {derived} <-n538 params={min=none; max=6}
    n540 {pt2=root[187] torch.ops.aten.div.Tensor}: [t776 f32 [C=288] {derived} ->[n541]] =
      div_scalar x=t775 {derived} <-n539 scalar=6
    n541 {pt2=root[183] torch.ops.aten.convolution.default}: [t604 f32 [H=288
                                                                      W=1 C=1] {pt2=root:div_19} ->[n360]] =
      permute x=t776 {derived} <-n540 perm=[H<-C, W<-H, C<-W]
    n360 {pt2=root[188] torch.ops.aten.mul.Tensor}: [t605 f32 [H=288 W=7 C=7] {pt2=root:mul_19} ->[n361]] =
      mul a=t604 {pt2=root:div_19} <-n541 b=t590 {pt2=root:div_18} <-n345
    group g68 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n366 {derived}: [t611 f32 [H=7 W=7 C=96] {derived} ->[n370, n548]] =
        batch_norm
          x=t608 {derived} <-n363
          weight=t107 {pt2=root:p_features_9_block_3_1_weight target=features.9.block.3.1.weight}
          bias=t108 {pt2=root:p_features_9_block_3_1_bias target=features.9.block.3.1.bias}
          running_mean=t220 {pt2=root:b_features_9_block_3_1_running_mean target=features.9.block.3.1.running_mean}
          running_var=t221 {pt2=root:b_features_9_block_3_1_running_var target=features.9.block.3.1.running_var}
          params={channel=C; eps=0.001}
    group g70 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n373 {derived}: [t618 f32 [H=7 W=7 C=576] {derived} ->[n374]] =
        batch_norm
          x=t615 {derived} <-n370
          weight=t110 {pt2=root:p_features_10_block_0_1_weight target=features.10.block.0.1.weight}
          bias=t111 {pt2=root:p_features_10_block_0_1_bias target=features.10.block.0.1.bias}
          running_mean=t223 {pt2=root:b_features_10_block_0_1_running_mean target=features.10.block.0.1.running_mean}
          running_var=t224 {pt2=root:b_features_10_block_0_1_running_var target=features.10.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n374 {pt2=root[192] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t619 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_81} ->[n375,
                                                                      n378]] =
        permute x=t618 {derived} <-n373 perm=[H<-C, W<-H, C<-W]
    n375 {pt2=root[193] torch.ops.aten.add.Tensor}: [t620 f32 [H=576 W=7 C=7] {pt2=root:add_24} ->[n376]] =
      add_scalar x=t619 {pt2=root:getitem_81} <-n374 scalar=3
    n376 {pt2=root[194] torch.ops.aten.clamp.default}: [t621 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_40} ->[n377]] =
      clamp x=t620 {pt2=root:add_24} <-n375 params={min=0; max=none}
    n377 {pt2=root[195] torch.ops.aten.clamp.default}: [t622 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_41} ->[n378]] =
      clamp x=t621 {pt2=root:clamp_40} <-n376 params={min=none; max=6}
    n378 {pt2=root[196] torch.ops.aten.mul.Tensor}: [t623 f32 [H=576 W=7 C=7] {pt2=root:mul_20} ->[n379]] =
      mul a=t619 {pt2=root:getitem_81} <-n374 b=t622 {pt2=root:clamp_41} <-n377
    n379 {pt2=root[197] torch.ops.aten.div.Tensor}: [t624 f32 [H=576 W=7 C=7] {pt2=root:div_20} ->[n380]] =
      div_scalar x=t623 {pt2=root:mul_20} <-n378 scalar=6
    group g72 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n385 {derived}: [t630 f32 [H=7 W=7 C=576] {derived} ->[n386]] =
        batch_norm
          x=t627 {derived} <-n382
          weight=t113 {pt2=root:p_features_10_block_1_1_weight target=features.10.block.1.1.weight}
          bias=t114 {pt2=root:p_features_10_block_1_1_bias target=features.10.block.1.1.bias}
          running_mean=t226 {pt2=root:b_features_10_block_1_1_running_mean target=features.10.block.1.1.running_mean}
          running_var=t227 {pt2=root:b_features_10_block_1_1_running_var target=features.10.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n386 {pt2=root[199] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t631 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_84} ->[n387,
                                                                      n390]] =
        permute x=t630 {derived} <-n385 perm=[H<-C, W<-H, C<-W]
    n387 {pt2=root[200] torch.ops.aten.add.Tensor}: [t632 f32 [H=576 W=7 C=7] {pt2=root:add_25} ->[n388]] =
      add_scalar x=t631 {pt2=root:getitem_84} <-n386 scalar=3
    n388 {pt2=root[201] torch.ops.aten.clamp.default}: [t633 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_42} ->[n389]] =
      clamp x=t632 {pt2=root:add_25} <-n387 params={min=0; max=none}
    n389 {pt2=root[202] torch.ops.aten.clamp.default}: [t634 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_43} ->[n390]] =
      clamp x=t633 {pt2=root:clamp_42} <-n388 params={min=none; max=6}
    n390 {pt2=root[203] torch.ops.aten.mul.Tensor}: [t635 f32 [H=576 W=7 C=7] {pt2=root:mul_21} ->[n391]] =
      mul a=t631 {pt2=root:getitem_84} <-n386 b=t634 {pt2=root:clamp_43} <-n389
    n391 {pt2=root[204] torch.ops.aten.div.Tensor}: [t636 f32 [H=576 W=7 C=7] {pt2=root:div_21} ->[n392,
                                                                      n406]] =
      div_scalar x=t635 {pt2=root:mul_21} <-n390 scalar=6
    n392 {pt2=root[205] torch.ops.aten.mean.dim}: [t637 f32 [H=576 W=1 C=1] {pt2=root:mean_7} ->[n393]] =
      mean x=t636 {pt2=root:div_21} <-n391 params={dims=[C, W]; keepdim=true}
    n542 {pt2=root[207] torch.ops.aten.relu.default}: [t777 f32 [C=144] {derived} ->[n400]] =
      relu x=t640 {derived} <-n395
    n543 {pt2=root[209] torch.ops.aten.add.Tensor}: [t778 f32 [C=576] {derived} ->[n544]] =
      add_scalar x=t645 {derived} <-n400 scalar=3
    n544 {pt2=root[210] torch.ops.aten.clamp.default}: [t779 f32 [C=576] {derived} ->[n545]] =
      clamp x=t778 {derived} <-n543 params={min=0; max=none}
    n545 {pt2=root[211] torch.ops.aten.clamp.default}: [t780 f32 [C=576] {derived} ->[n546]] =
      clamp x=t779 {derived} <-n544 params={min=none; max=6}
    n546 {pt2=root[212] torch.ops.aten.div.Tensor}: [t781 f32 [C=576] {derived} ->[n547]] =
      div_scalar x=t780 {derived} <-n545 scalar=6
    n547 {pt2=root[208] torch.ops.aten.convolution.default}: [t650 f32 [H=576
                                                                      W=1 C=1] {pt2=root:div_22} ->[n406]] =
      permute x=t781 {derived} <-n546 perm=[H<-C, W<-H, C<-W]
    n406 {pt2=root[213] torch.ops.aten.mul.Tensor}: [t651 f32 [H=576 W=7 C=7] {pt2=root:mul_22} ->[n407]] =
      mul a=t650 {pt2=root:div_22} <-n547 b=t636 {pt2=root:div_21} <-n391
    group g76 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n412 {derived}: [t657 f32 [H=7 W=7 C=96] {derived} ->[n548]] =
        batch_norm
          x=t654 {derived} <-n409
          weight=t120 {pt2=root:p_features_10_block_3_1_weight target=features.10.block.3.1.weight}
          bias=t121 {pt2=root:p_features_10_block_3_1_bias target=features.10.block.3.1.bias}
          running_mean=t229 {pt2=root:b_features_10_block_3_1_running_mean target=features.10.block.3.1.running_mean}
          running_var=t230 {pt2=root:b_features_10_block_3_1_running_var target=features.10.block.3.1.running_var}
          params={channel=C; eps=0.001}
    n548 {pt2=root[216] torch.ops.aten.add.Tensor}: [t782 f32 [H=7 W=7 C=96] {derived} ->[n417,
                                                                      n555]] =
      add a=t657 {derived} <-n412 b=t611 {derived} <-n366
    group g78 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n420 {derived}: [t665 f32 [H=7 W=7 C=576] {derived} ->[n421]] =
        batch_norm
          x=t662 {derived} <-n417
          weight=t123 {pt2=root:p_features_11_block_0_1_weight target=features.11.block.0.1.weight}
          bias=t124 {pt2=root:p_features_11_block_0_1_bias target=features.11.block.0.1.bias}
          running_mean=t232 {pt2=root:b_features_11_block_0_1_running_mean target=features.11.block.0.1.running_mean}
          running_var=t233 {pt2=root:b_features_11_block_0_1_running_var target=features.11.block.0.1.running_var}
          params={channel=C; eps=0.001}
      n421 {pt2=root[218] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t666 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_90} ->[n422,
                                                                      n425]] =
        permute x=t665 {derived} <-n420 perm=[H<-C, W<-H, C<-W]
    n422 {pt2=root[219] torch.ops.aten.add.Tensor}: [t667 f32 [H=576 W=7 C=7] {pt2=root:add_28} ->[n423]] =
      add_scalar x=t666 {pt2=root:getitem_90} <-n421 scalar=3
    n423 {pt2=root[220] torch.ops.aten.clamp.default}: [t668 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_46} ->[n424]] =
      clamp x=t667 {pt2=root:add_28} <-n422 params={min=0; max=none}
    n424 {pt2=root[221] torch.ops.aten.clamp.default}: [t669 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_47} ->[n425]] =
      clamp x=t668 {pt2=root:clamp_46} <-n423 params={min=none; max=6}
    n425 {pt2=root[222] torch.ops.aten.mul.Tensor}: [t670 f32 [H=576 W=7 C=7] {pt2=root:mul_23} ->[n426]] =
      mul a=t666 {pt2=root:getitem_90} <-n421 b=t669 {pt2=root:clamp_47} <-n424
    n426 {pt2=root[223] torch.ops.aten.div.Tensor}: [t671 f32 [H=576 W=7 C=7] {pt2=root:div_23} ->[n427]] =
      div_scalar x=t670 {pt2=root:mul_23} <-n425 scalar=6
    group g80 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n432 {derived}: [t677 f32 [H=7 W=7 C=576] {derived} ->[n433]] =
        batch_norm
          x=t674 {derived} <-n429
          weight=t126 {pt2=root:p_features_11_block_1_1_weight target=features.11.block.1.1.weight}
          bias=t127 {pt2=root:p_features_11_block_1_1_bias target=features.11.block.1.1.bias}
          running_mean=t235 {pt2=root:b_features_11_block_1_1_running_mean target=features.11.block.1.1.running_mean}
          running_var=t236 {pt2=root:b_features_11_block_1_1_running_var target=features.11.block.1.1.running_var}
          params={channel=C; eps=0.001}
      n433 {pt2=root[225] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t678 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_93} ->[n434,
                                                                      n437]] =
        permute x=t677 {derived} <-n432 perm=[H<-C, W<-H, C<-W]
    n434 {pt2=root[226] torch.ops.aten.add.Tensor}: [t679 f32 [H=576 W=7 C=7] {pt2=root:add_29} ->[n435]] =
      add_scalar x=t678 {pt2=root:getitem_93} <-n433 scalar=3
    n435 {pt2=root[227] torch.ops.aten.clamp.default}: [t680 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_48} ->[n436]] =
      clamp x=t679 {pt2=root:add_29} <-n434 params={min=0; max=none}
    n436 {pt2=root[228] torch.ops.aten.clamp.default}: [t681 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_49} ->[n437]] =
      clamp x=t680 {pt2=root:clamp_48} <-n435 params={min=none; max=6}
    n437 {pt2=root[229] torch.ops.aten.mul.Tensor}: [t682 f32 [H=576 W=7 C=7] {pt2=root:mul_24} ->[n438]] =
      mul a=t678 {pt2=root:getitem_93} <-n433 b=t681 {pt2=root:clamp_49} <-n436
    n438 {pt2=root[230] torch.ops.aten.div.Tensor}: [t683 f32 [H=576 W=7 C=7] {pt2=root:div_24} ->[n439,
                                                                      n453]] =
      div_scalar x=t682 {pt2=root:mul_24} <-n437 scalar=6
    n439 {pt2=root[231] torch.ops.aten.mean.dim}: [t684 f32 [H=576 W=1 C=1] {pt2=root:mean_8} ->[n440]] =
      mean x=t683 {pt2=root:div_24} <-n438 params={dims=[C, W]; keepdim=true}
    n549 {pt2=root[233] torch.ops.aten.relu.default}: [t783 f32 [C=144] {derived} ->[n447]] =
      relu x=t687 {derived} <-n442
    n550 {pt2=root[235] torch.ops.aten.add.Tensor}: [t784 f32 [C=576] {derived} ->[n551]] =
      add_scalar x=t692 {derived} <-n447 scalar=3
    n551 {pt2=root[236] torch.ops.aten.clamp.default}: [t785 f32 [C=576] {derived} ->[n552]] =
      clamp x=t784 {derived} <-n550 params={min=0; max=none}
    n552 {pt2=root[237] torch.ops.aten.clamp.default}: [t786 f32 [C=576] {derived} ->[n553]] =
      clamp x=t785 {derived} <-n551 params={min=none; max=6}
    n553 {pt2=root[238] torch.ops.aten.div.Tensor}: [t787 f32 [C=576] {derived} ->[n554]] =
      div_scalar x=t786 {derived} <-n552 scalar=6
    n554 {pt2=root[234] torch.ops.aten.convolution.default}: [t697 f32 [H=576
                                                                      W=1 C=1] {pt2=root:div_25} ->[n453]] =
      permute x=t787 {derived} <-n553 perm=[H<-C, W<-H, C<-W]
    n453 {pt2=root[239] torch.ops.aten.mul.Tensor}: [t698 f32 [H=576 W=7 C=7] {pt2=root:mul_25} ->[n454]] =
      mul a=t697 {pt2=root:div_25} <-n554 b=t683 {pt2=root:div_24} <-n438
    group g84 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n459 {derived}: [t704 f32 [H=7 W=7 C=96] {derived} ->[n555]] =
        batch_norm
          x=t701 {derived} <-n456
          weight=t133 {pt2=root:p_features_11_block_3_1_weight target=features.11.block.3.1.weight}
          bias=t134 {pt2=root:p_features_11_block_3_1_bias target=features.11.block.3.1.bias}
          running_mean=t238 {pt2=root:b_features_11_block_3_1_running_mean target=features.11.block.3.1.running_mean}
          running_var=t239 {pt2=root:b_features_11_block_3_1_running_var target=features.11.block.3.1.running_var}
          params={channel=C; eps=0.001}
    n555 {pt2=root[242] torch.ops.aten.add.Tensor}: [t788 f32 [H=7 W=7 C=96] {derived} ->[n464]] =
      add a=t704 {derived} <-n459 b=t782 {derived} <-n548
    group g86 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n467 {derived}: [t712 f32 [H=7 W=7 C=576] {derived} ->[n468]] =
        batch_norm
          x=t709 {derived} <-n464
          weight=t136 {pt2=root:p_features_12_1_weight target=features.12.1.weight}
          bias=t137 {pt2=root:p_features_12_1_bias target=features.12.1.bias}
          running_mean=t241 {pt2=root:b_features_12_1_running_mean target=features.12.1.running_mean}
          running_var=t242 {pt2=root:b_features_12_1_running_var target=features.12.1.running_var}
          params={channel=C; eps=0.001}
      n468 {pt2=root[244] torch.ops.aten._native_batch_norm_legit_no_training.default}: [t713 f32 [H=576
                                                                      W=7 C=7] {pt2=root:getitem_99} ->[n469,
                                                                      n472]] =
        permute x=t712 {derived} <-n467 perm=[H<-C, W<-H, C<-W]
    n469 {pt2=root[245] torch.ops.aten.add.Tensor}: [t714 f32 [H=576 W=7 C=7] {pt2=root:add_32} ->[n470]] =
      add_scalar x=t713 {pt2=root:getitem_99} <-n468 scalar=3
    n470 {pt2=root[246] torch.ops.aten.clamp.default}: [t715 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_52} ->[n471]] =
      clamp x=t714 {pt2=root:add_32} <-n469 params={min=0; max=none}
    n471 {pt2=root[247] torch.ops.aten.clamp.default}: [t716 f32 [H=576 W=7
                                                                  C=7] {pt2=root:clamp_53} ->[n472]] =
      clamp x=t715 {pt2=root:clamp_52} <-n470 params={min=none; max=6}
    n472 {pt2=root[248] torch.ops.aten.mul.Tensor}: [t717 f32 [H=576 W=7 C=7] {pt2=root:mul_26} ->[n473]] =
      mul a=t713 {pt2=root:getitem_99} <-n468 b=t716 {pt2=root:clamp_53} <-n471
    n473 {pt2=root[249] torch.ops.aten.div.Tensor}: [t718 f32 [H=576 W=7 C=7] {pt2=root:div_26} ->[n474]] =
      div_scalar x=t717 {pt2=root:mul_26} <-n472 scalar=6
    n474 {pt2=root[250] torch.ops.aten.mean.dim}: [t719 f32 [H=576 W=1 C=1] {pt2=root:mean_9} ->[n556]] =
      mean x=t718 {pt2=root:div_26} <-n473 params={dims=[C, W]; keepdim=true}
    n556 {pt2=root[251] torch.ops.aten.view.default}: [t720 f32 [C=576] {pt2=root:view} ->[n478]] =
      permute x=t719 {pt2=root:mean_9} <-n474 perm=[H<-W, W<-C, C<-H]
    group g87 torch.ops.aten.addmm.default:
      n478 {pt2=root[253] torch.ops.aten.addmm.default}: [t723 f32 [C=1024] {pt2=root:addmm} ->[n479,
                                                                      n482]] =
        linear
          x=t720 {pt2=root:view} <-n556
          weight=t722 {derived} <-n488
          bias=t139 {pt2=root:p_classifier_0_bias target=classifier.0.bias}
          params={in_features=576}
    n479 {pt2=root[254] torch.ops.aten.add.Tensor}: [t724 f32 [C=1024] {pt2=root:add_33} ->[n480]] =
      add_scalar x=t723 {pt2=root:addmm} <-n478 scalar=3
    n480 {pt2=root[255] torch.ops.aten.clamp.default}: [t725 f32 [C=1024] {pt2=root:clamp_54} ->[n481]] =
      clamp x=t724 {pt2=root:add_33} <-n479 params={min=0; max=none}
    n481 {pt2=root[256] torch.ops.aten.clamp.default}: [t726 f32 [C=1024] {pt2=root:clamp_55} ->[n482]] =
      clamp x=t725 {pt2=root:clamp_54} <-n480 params={min=none; max=6}
    n482 {pt2=root[257] torch.ops.aten.mul.Tensor}: [t727 f32 [C=1024] {pt2=root:mul_27} ->[n483]] =
      mul a=t723 {pt2=root:addmm} <-n478 b=t726 {pt2=root:clamp_55} <-n481
    n483 {pt2=root[258] torch.ops.aten.div.Tensor}: [t728 f32 [C=1024] {pt2=root:div_27} ->[n484]] =
      div_scalar x=t727 {pt2=root:mul_27} <-n482 scalar=6
    n484 {pt2=root[259] torch.ops.aten.clone.default}: [t729 f32 [C=1024] {pt2=root:clone} ->[n487]] =
      clone x=t728 {pt2=root:div_27} <-n483
    group g88 torch.ops.aten.addmm.default:
      n487 {pt2=root[261] torch.ops.aten.addmm.default}: [t732 f32 [C=1000] {pt2=root:addmm_1}] =
        linear
          x=t729 {pt2=root:clone} <-n484
          weight=t731 {derived} <-n489
          bias=t141 {pt2=root:p_classifier_3_bias target=classifier.3.bias}
          params={in_features=1024}
  outputs: [t732 f32 [C=1000] {pt2=root:addmm_1} <-n487]

MobileNet-v2 uses hardtanh (relu6) rather than the v3 hardswish chain, so it is
the variant that pins the [hardtanh] constructor through the pipeline.

  $ ../bin/native_graph.exe transform --pt2 "$PT2_DATA/mobilenet_v2/mobilenet_v2.pt2"
  nodes: 415 -> 206
  constants: 262, of which 0 folded
  graph
  inputs:
    [t0 f32 [D=32 H=3 W=3 C=3] {pt2=root:p_features_0_0_weight target=features.0.0.weight} ->[n1] constant,
     t1 f32 [C=32] {pt2=root:p_features_0_1_weight target=features.0.1.weight} ->[n5] constant,
     t2 f32 [C=32] {pt2=root:p_features_0_1_bias target=features.0.1.bias} ->[n5] constant,
     t3 f32 [D=32 H=1 W=3 C=3] {pt2=root:p_features_1_conv_0_0_weight target=features.1.conv.0.0.weight} ->[n9] constant,
     t4 f32 [C=32] {pt2=root:p_features_1_conv_0_1_weight target=features.1.conv.0.1.weight} ->[n13] constant,
     t5 f32 [C=32] {pt2=root:p_features_1_conv_0_1_bias target=features.1.conv.0.1.bias} ->[n13] constant,
     t6 f32 [D=16 H=32 W=1 C=1] {pt2=root:p_features_1_conv_1_weight target=features.1.conv.1.weight} ->[n17] constant,
     t7 f32 [C=16] {pt2=root:p_features_1_conv_2_weight target=features.1.conv.2.weight} ->[n21] constant,
     t8 f32 [C=16] {pt2=root:p_features_1_conv_2_bias target=features.1.conv.2.bias} ->[n21] constant,
     t9 f32 [D=96 H=16 W=1 C=1] {pt2=root:p_features_2_conv_0_0_weight target=features.2.conv.0.0.weight} ->[n24] constant,
     t10 f32 [C=96] {pt2=root:p_features_2_conv_0_1_weight target=features.2.conv.0.1.weight} ->[n28] constant,
     t11 f32 [C=96] {pt2=root:p_features_2_conv_0_1_bias target=features.2.conv.0.1.bias} ->[n28] constant,
     t12 f32 [D=96 H=1 W=3 C=3] {pt2=root:p_features_2_conv_1_0_weight target=features.2.conv.1.0.weight} ->[n32] constant,
     t13 f32 [C=96] {pt2=root:p_features_2_conv_1_1_weight target=features.2.conv.1.1.weight} ->[n36] constant,
     t14 f32 [C=96] {pt2=root:p_features_2_conv_1_1_bias target=features.2.conv.1.1.bias} ->[n36] constant,
     t15 f32 [D=24 H=96 W=1 C=1] {pt2=root:p_features_2_conv_2_weight target=features.2.conv.2.weight} ->[n40] constant,
     t16 f32 [C=24] {pt2=root:p_features_2_conv_3_weight target=features.2.conv.3.weight} ->[n44] constant,
     t17 f32 [C=24] {pt2=root:p_features_2_conv_3_bias target=features.2.conv.3.bias} ->[n44] constant,
     t18 f32 [D=144 H=24 W=1 C=1] {pt2=root:p_features_3_conv_0_0_weight target=features.3.conv.0.0.weight} ->[n47] constant,
     t19 f32 [C=144] {pt2=root:p_features_3_conv_0_1_weight target=features.3.conv.0.1.weight} ->[n51] constant,
     t20 f32 [C=144] {pt2=root:p_features_3_conv_0_1_bias target=features.3.conv.0.1.bias} ->[n51] constant,
     t21 f32 [D=144 H=1 W=3 C=3] {pt2=root:p_features_3_conv_1_0_weight target=features.3.conv.1.0.weight} ->[n55] constant,
     t22 f32 [C=144] {pt2=root:p_features_3_conv_1_1_weight target=features.3.conv.1.1.weight} ->[n59] constant,
     t23 f32 [C=144] {pt2=root:p_features_3_conv_1_1_bias target=features.3.conv.1.1.bias} ->[n59] constant,
     t24 f32 [D=24 H=144 W=1 C=1] {pt2=root:p_features_3_conv_2_weight target=features.3.conv.2.weight} ->[n63] constant,
     t25 f32 [C=24] {pt2=root:p_features_3_conv_3_weight target=features.3.conv.3.weight} ->[n67] constant,
     t26 f32 [C=24] {pt2=root:p_features_3_conv_3_bias target=features.3.conv.3.bias} ->[n67] constant,
     t27 f32 [D=144 H=24 W=1 C=1] {pt2=root:p_features_4_conv_0_0_weight target=features.4.conv.0.0.weight} ->[n71] constant,
     t28 f32 [C=144] {pt2=root:p_features_4_conv_0_1_weight target=features.4.conv.0.1.weight} ->[n75] constant,
     t29 f32 [C=144] {pt2=root:p_features_4_conv_0_1_bias target=features.4.conv.0.1.bias} ->[n75] constant,
     t30 f32 [D=144 H=1 W=3 C=3] {pt2=root:p_features_4_conv_1_0_weight target=features.4.conv.1.0.weight} ->[n79] constant,
     t31 f32 [C=144] {pt2=root:p_features_4_conv_1_1_weight target=features.4.conv.1.1.weight} ->[n83] constant,
     t32 f32 [C=144] {pt2=root:p_features_4_conv_1_1_bias target=features.4.conv.1.1.bias} ->[n83] constant,
     t33 f32 [D=32 H=144 W=1 C=1] {pt2=root:p_features_4_conv_2_weight target=features.4.conv.2.weight} ->[n87] constant,
     t34 f32 [C=32] {pt2=root:p_features_4_conv_3_weight target=features.4.conv.3.weight} ->[n91] constant,
     t35 f32 [C=32] {pt2=root:p_features_4_conv_3_bias target=features.4.conv.3.bias} ->[n91] constant,
     t36 f32 [D=192 H=32 W=1 C=1] {pt2=root:p_features_5_conv_0_0_weight target=features.5.conv.0.0.weight} ->[n94] constant,
     t37 f32 [C=192] {pt2=root:p_features_5_conv_0_1_weight target=features.5.conv.0.1.weight} ->[n98] constant,
     t38 f32 [C=192] {pt2=root:p_features_5_conv_0_1_bias target=features.5.conv.0.1.bias} ->[n98] constant,
     t39 f32 [D=192 H=1 W=3 C=3] {pt2=root:p_features_5_conv_1_0_weight target=features.5.conv.1.0.weight} ->[n102] constant,
     t40 f32 [C=192] {pt2=root:p_features_5_conv_1_1_weight target=features.5.conv.1.1.weight} ->[n106] constant,
     t41 f32 [C=192] {pt2=root:p_features_5_conv_1_1_bias target=features.5.conv.1.1.bias} ->[n106] constant,
     t42 f32 [D=32 H=192 W=1 C=1] {pt2=root:p_features_5_conv_2_weight target=features.5.conv.2.weight} ->[n110] constant,
     t43 f32 [C=32] {pt2=root:p_features_5_conv_3_weight target=features.5.conv.3.weight} ->[n114] constant,
     t44 f32 [C=32] {pt2=root:p_features_5_conv_3_bias target=features.5.conv.3.bias} ->[n114] constant,
     t45 f32 [D=192 H=32 W=1 C=1] {pt2=root:p_features_6_conv_0_0_weight target=features.6.conv.0.0.weight} ->[n118] constant,
     t46 f32 [C=192] {pt2=root:p_features_6_conv_0_1_weight target=features.6.conv.0.1.weight} ->[n122] constant,
     t47 f32 [C=192] {pt2=root:p_features_6_conv_0_1_bias target=features.6.conv.0.1.bias} ->[n122] constant,
     t48 f32 [D=192 H=1 W=3 C=3] {pt2=root:p_features_6_conv_1_0_weight target=features.6.conv.1.0.weight} ->[n126] constant,
     t49 f32 [C=192] {pt2=root:p_features_6_conv_1_1_weight target=features.6.conv.1.1.weight} ->[n130] constant,
     t50 f32 [C=192] {pt2=root:p_features_6_conv_1_1_bias target=features.6.conv.1.1.bias} ->[n130] constant,
     t51 f32 [D=32 H=192 W=1 C=1] {pt2=root:p_features_6_conv_2_weight target=features.6.conv.2.weight} ->[n134] constant,
     t52 f32 [C=32] {pt2=root:p_features_6_conv_3_weight target=features.6.conv.3.weight} ->[n138] constant,
     t53 f32 [C=32] {pt2=root:p_features_6_conv_3_bias target=features.6.conv.3.bias} ->[n138] constant,
     t54 f32 [D=192 H=32 W=1 C=1] {pt2=root:p_features_7_conv_0_0_weight target=features.7.conv.0.0.weight} ->[n142] constant,
     t55 f32 [C=192] {pt2=root:p_features_7_conv_0_1_weight target=features.7.conv.0.1.weight} ->[n146] constant,
     t56 f32 [C=192] {pt2=root:p_features_7_conv_0_1_bias target=features.7.conv.0.1.bias} ->[n146] constant,
     t57 f32 [D=192 H=1 W=3 C=3] {pt2=root:p_features_7_conv_1_0_weight target=features.7.conv.1.0.weight} ->[n150] constant,
     t58 f32 [C=192] {pt2=root:p_features_7_conv_1_1_weight target=features.7.conv.1.1.weight} ->[n154] constant,
     t59 f32 [C=192] {pt2=root:p_features_7_conv_1_1_bias target=features.7.conv.1.1.bias} ->[n154] constant,
     t60 f32 [D=64 H=192 W=1 C=1] {pt2=root:p_features_7_conv_2_weight target=features.7.conv.2.weight} ->[n158] constant,
     t61 f32 [C=64] {pt2=root:p_features_7_conv_3_weight target=features.7.conv.3.weight} ->[n162] constant,
     t62 f32 [C=64] {pt2=root:p_features_7_conv_3_bias target=features.7.conv.3.bias} ->[n162] constant,
     t63 f32 [D=384 H=64 W=1 C=1] {pt2=root:p_features_8_conv_0_0_weight target=features.8.conv.0.0.weight} ->[n165] constant,
     t64 f32 [C=384] {pt2=root:p_features_8_conv_0_1_weight target=features.8.conv.0.1.weight} ->[n169] constant,
     t65 f32 [C=384] {pt2=root:p_features_8_conv_0_1_bias target=features.8.conv.0.1.bias} ->[n169] constant,
     t66 f32 [D=384 H=1 W=3 C=3] {pt2=root:p_features_8_conv_1_0_weight target=features.8.conv.1.0.weight} ->[n173] constant,
     t67 f32 [C=384] {pt2=root:p_features_8_conv_1_1_weight target=features.8.conv.1.1.weight} ->[n177] constant,
     t68 f32 [C=384] {pt2=root:p_features_8_conv_1_1_bias target=features.8.conv.1.1.bias} ->[n177] constant,
     t69 f32 [D=64 H=384 W=1 C=1] {pt2=root:p_features_8_conv_2_weight target=features.8.conv.2.weight} ->[n181] constant,
     t70 f32 [C=64] {pt2=root:p_features_8_conv_3_weight target=features.8.conv.3.weight} ->[n185] constant,
     t71 f32 [C=64] {pt2=root:p_features_8_conv_3_bias target=features.8.conv.3.bias} ->[n185] constant,
     t72 f32 [D=384 H=64 W=1 C=1] {pt2=root:p_features_9_conv_0_0_weight target=features.9.conv.0.0.weight} ->[n189] constant,
     t73 f32 [C=384] {pt2=root:p_features_9_conv_0_1_weight target=features.9.conv.0.1.weight} ->[n193] constant,
     t74 f32 [C=384] {pt2=root:p_features_9_conv_0_1_bias target=features.9.conv.0.1.bias} ->[n193] constant,
     t75 f32 [D=384 H=1 W=3 C=3] {pt2=root:p_features_9_conv_1_0_weight target=features.9.conv.1.0.weight} ->[n197] constant,
     t76 f32 [C=384] {pt2=root:p_features_9_conv_1_1_weight target=features.9.conv.1.1.weight} ->[n201] constant,
     t77 f32 [C=384] {pt2=root:p_features_9_conv_1_1_bias target=features.9.conv.1.1.bias} ->[n201] constant,
     t78 f32 [D=64 H=384 W=1 C=1] {pt2=root:p_features_9_conv_2_weight target=features.9.conv.2.weight} ->[n205] constant,
     t79 f32 [C=64] {pt2=root:p_features_9_conv_3_weight target=features.9.conv.3.weight} ->[n209] constant,
     t80 f32 [C=64] {pt2=root:p_features_9_conv_3_bias target=features.9.conv.3.bias} ->[n209] constant,
     t81 f32 [D=384 H=64 W=1 C=1] {pt2=root:p_features_10_conv_0_0_weight target=features.10.conv.0.0.weight} ->[n213] constant,
     t82 f32 [C=384] {pt2=root:p_features_10_conv_0_1_weight target=features.10.conv.0.1.weight} ->[n217] constant,
     t83 f32 [C=384] {pt2=root:p_features_10_conv_0_1_bias target=features.10.conv.0.1.bias} ->[n217] constant,
     t84 f32 [D=384 H=1 W=3 C=3] {pt2=root:p_features_10_conv_1_0_weight target=features.10.conv.1.0.weight} ->[n221] constant,
     t85 f32 [C=384] {pt2=root:p_features_10_conv_1_1_weight target=features.10.conv.1.1.weight} ->[n225] constant,
     t86 f32 [C=384] {pt2=root:p_features_10_conv_1_1_bias target=features.10.conv.1.1.bias} ->[n225] constant,
     t87 f32 [D=64 H=384 W=1 C=1] {pt2=root:p_features_10_conv_2_weight target=features.10.conv.2.weight} ->[n229] constant,
     t88 f32 [C=64] {pt2=root:p_features_10_conv_3_weight target=features.10.conv.3.weight} ->[n233] constant,
     t89 f32 [C=64] {pt2=root:p_features_10_conv_3_bias target=features.10.conv.3.bias} ->[n233] constant,
     t90 f32 [D=384 H=64 W=1 C=1] {pt2=root:p_features_11_conv_0_0_weight target=features.11.conv.0.0.weight} ->[n237] constant,
     t91 f32 [C=384] {pt2=root:p_features_11_conv_0_1_weight target=features.11.conv.0.1.weight} ->[n241] constant,
     t92 f32 [C=384] {pt2=root:p_features_11_conv_0_1_bias target=features.11.conv.0.1.bias} ->[n241] constant,
     t93 f32 [D=384 H=1 W=3 C=3] {pt2=root:p_features_11_conv_1_0_weight target=features.11.conv.1.0.weight} ->[n245] constant,
     t94 f32 [C=384] {pt2=root:p_features_11_conv_1_1_weight target=features.11.conv.1.1.weight} ->[n249] constant,
     t95 f32 [C=384] {pt2=root:p_features_11_conv_1_1_bias target=features.11.conv.1.1.bias} ->[n249] constant,
     t96 f32 [D=96 H=384 W=1 C=1] {pt2=root:p_features_11_conv_2_weight target=features.11.conv.2.weight} ->[n253] constant,
     t97 f32 [C=96] {pt2=root:p_features_11_conv_3_weight target=features.11.conv.3.weight} ->[n257] constant,
     t98 f32 [C=96] {pt2=root:p_features_11_conv_3_bias target=features.11.conv.3.bias} ->[n257] constant,
     t99 f32 [D=576 H=96 W=1 C=1] {pt2=root:p_features_12_conv_0_0_weight target=features.12.conv.0.0.weight} ->[n260] constant,
     t100 f32 [C=576] {pt2=root:p_features_12_conv_0_1_weight target=features.12.conv.0.1.weight} ->[n264] constant,
     t101 f32 [C=576] {pt2=root:p_features_12_conv_0_1_bias target=features.12.conv.0.1.bias} ->[n264] constant,
     t102 f32 [D=576 H=1 W=3 C=3] {pt2=root:p_features_12_conv_1_0_weight target=features.12.conv.1.0.weight} ->[n268] constant,
     t103 f32 [C=576] {pt2=root:p_features_12_conv_1_1_weight target=features.12.conv.1.1.weight} ->[n272] constant,
     t104 f32 [C=576] {pt2=root:p_features_12_conv_1_1_bias target=features.12.conv.1.1.bias} ->[n272] constant,
     t105 f32 [D=96 H=576 W=1 C=1] {pt2=root:p_features_12_conv_2_weight target=features.12.conv.2.weight} ->[n276] constant,
     t106 f32 [C=96] {pt2=root:p_features_12_conv_3_weight target=features.12.conv.3.weight} ->[n280] constant,
     t107 f32 [C=96] {pt2=root:p_features_12_conv_3_bias target=features.12.conv.3.bias} ->[n280] constant,
     t108 f32 [D=576 H=96 W=1 C=1] {pt2=root:p_features_13_conv_0_0_weight target=features.13.conv.0.0.weight} ->[n284] constant,
     t109 f32 [C=576] {pt2=root:p_features_13_conv_0_1_weight target=features.13.conv.0.1.weight} ->[n288] constant,
     t110 f32 [C=576] {pt2=root:p_features_13_conv_0_1_bias target=features.13.conv.0.1.bias} ->[n288] constant,
     t111 f32 [D=576 H=1 W=3 C=3] {pt2=root:p_features_13_conv_1_0_weight target=features.13.conv.1.0.weight} ->[n292] constant,
     t112 f32 [C=576] {pt2=root:p_features_13_conv_1_1_weight target=features.13.conv.1.1.weight} ->[n296] constant,
     t113 f32 [C=576] {pt2=root:p_features_13_conv_1_1_bias target=features.13.conv.1.1.bias} ->[n296] constant,
     t114 f32 [D=96 H=576 W=1 C=1] {pt2=root:p_features_13_conv_2_weight target=features.13.conv.2.weight} ->[n300] constant,
     t115 f32 [C=96] {pt2=root:p_features_13_conv_3_weight target=features.13.conv.3.weight} ->[n304] constant,
     t116 f32 [C=96] {pt2=root:p_features_13_conv_3_bias target=features.13.conv.3.bias} ->[n304] constant,
     t117 f32 [D=576 H=96 W=1 C=1] {pt2=root:p_features_14_conv_0_0_weight target=features.14.conv.0.0.weight} ->[n308] constant,
     t118 f32 [C=576] {pt2=root:p_features_14_conv_0_1_weight target=features.14.conv.0.1.weight} ->[n312] constant,
     t119 f32 [C=576] {pt2=root:p_features_14_conv_0_1_bias target=features.14.conv.0.1.bias} ->[n312] constant,
     t120 f32 [D=576 H=1 W=3 C=3] {pt2=root:p_features_14_conv_1_0_weight target=features.14.conv.1.0.weight} ->[n316] constant,
     t121 f32 [C=576] {pt2=root:p_features_14_conv_1_1_weight target=features.14.conv.1.1.weight} ->[n320] constant,
     t122 f32 [C=576] {pt2=root:p_features_14_conv_1_1_bias target=features.14.conv.1.1.bias} ->[n320] constant,
     t123 f32 [D=160 H=576 W=1 C=1] {pt2=root:p_features_14_conv_2_weight target=features.14.conv.2.weight} ->[n324] constant,
     t124 f32 [C=160] {pt2=root:p_features_14_conv_3_weight target=features.14.conv.3.weight} ->[n328] constant,
     t125 f32 [C=160] {pt2=root:p_features_14_conv_3_bias target=features.14.conv.3.bias} ->[n328] constant,
     t126 f32 [D=960 H=160 W=1 C=1] {pt2=root:p_features_15_conv_0_0_weight target=features.15.conv.0.0.weight} ->[n331] constant,
     t127 f32 [C=960] {pt2=root:p_features_15_conv_0_1_weight target=features.15.conv.0.1.weight} ->[n335] constant,
     t128 f32 [C=960] {pt2=root:p_features_15_conv_0_1_bias target=features.15.conv.0.1.bias} ->[n335] constant,
     t129 f32 [D=960 H=1 W=3 C=3] {pt2=root:p_features_15_conv_1_0_weight target=features.15.conv.1.0.weight} ->[n339] constant,
     t130 f32 [C=960] {pt2=root:p_features_15_conv_1_1_weight target=features.15.conv.1.1.weight} ->[n343] constant,
     t131 f32 [C=960] {pt2=root:p_features_15_conv_1_1_bias target=features.15.conv.1.1.bias} ->[n343] constant,
     t132 f32 [D=160 H=960 W=1 C=1] {pt2=root:p_features_15_conv_2_weight target=features.15.conv.2.weight} ->[n347] constant,
     t133 f32 [C=160] {pt2=root:p_features_15_conv_3_weight target=features.15.conv.3.weight} ->[n351] constant,
     t134 f32 [C=160] {pt2=root:p_features_15_conv_3_bias target=features.15.conv.3.bias} ->[n351] constant,
     t135 f32 [D=960 H=160 W=1 C=1] {pt2=root:p_features_16_conv_0_0_weight target=features.16.conv.0.0.weight} ->[n355] constant,
     t136 f32 [C=960] {pt2=root:p_features_16_conv_0_1_weight target=features.16.conv.0.1.weight} ->[n359] constant,
     t137 f32 [C=960] {pt2=root:p_features_16_conv_0_1_bias target=features.16.conv.0.1.bias} ->[n359] constant,
     t138 f32 [D=960 H=1 W=3 C=3] {pt2=root:p_features_16_conv_1_0_weight target=features.16.conv.1.0.weight} ->[n363] constant,
     t139 f32 [C=960] {pt2=root:p_features_16_conv_1_1_weight target=features.16.conv.1.1.weight} ->[n367] constant,
     t140 f32 [C=960] {pt2=root:p_features_16_conv_1_1_bias target=features.16.conv.1.1.bias} ->[n367] constant,
     t141 f32 [D=160 H=960 W=1 C=1] {pt2=root:p_features_16_conv_2_weight target=features.16.conv.2.weight} ->[n371] constant,
     t142 f32 [C=160] {pt2=root:p_features_16_conv_3_weight target=features.16.conv.3.weight} ->[n375] constant,
     t143 f32 [C=160] {pt2=root:p_features_16_conv_3_bias target=features.16.conv.3.bias} ->[n375] constant,
     t144 f32 [D=960 H=160 W=1 C=1] {pt2=root:p_features_17_conv_0_0_weight target=features.17.conv.0.0.weight} ->[n379] constant,
     t145 f32 [C=960] {pt2=root:p_features_17_conv_0_1_weight target=features.17.conv.0.1.weight} ->[n383] constant,
     t146 f32 [C=960] {pt2=root:p_features_17_conv_0_1_bias target=features.17.conv.0.1.bias} ->[n383] constant,
     t147 f32 [D=960 H=1 W=3 C=3] {pt2=root:p_features_17_conv_1_0_weight target=features.17.conv.1.0.weight} ->[n387] constant,
     t148 f32 [C=960] {pt2=root:p_features_17_conv_1_1_weight target=features.17.conv.1.1.weight} ->[n391] constant,
     t149 f32 [C=960] {pt2=root:p_features_17_conv_1_1_bias target=features.17.conv.1.1.bias} ->[n391] constant,
     t150 f32 [D=320 H=960 W=1 C=1] {pt2=root:p_features_17_conv_2_weight target=features.17.conv.2.weight} ->[n395] constant,
     t151 f32 [C=320] {pt2=root:p_features_17_conv_3_weight target=features.17.conv.3.weight} ->[n399] constant,
     t152 f32 [C=320] {pt2=root:p_features_17_conv_3_bias target=features.17.conv.3.bias} ->[n399] constant,
     t153 f32 [D=1280 H=320 W=1 C=1] {pt2=root:p_features_18_0_weight target=features.18.0.weight} ->[n402] constant,
     t154 f32 [C=1280] {pt2=root:p_features_18_1_weight target=features.18.1.weight} ->[n406] constant,
     t155 f32 [C=1280] {pt2=root:p_features_18_1_bias target=features.18.1.bias} ->[n406] constant,
     t156 f32 [W=1000 C=1280] {pt2=root:p_classifier_1_weight target=classifier.1.weight} ->[n415] constant,
     t157 f32 [C=1000] {pt2=root:p_classifier_1_bias target=classifier.1.bias} ->[n414] constant,
     t158 f32 [C=32] {pt2=root:b_features_0_1_running_mean target=features.0.1.running_mean} ->[n5] constant,
     t159 f32 [C=32] {pt2=root:b_features_0_1_running_var target=features.0.1.running_var} ->[n5] constant,
     t161 f32 [C=32] {pt2=root:b_features_1_conv_0_1_running_mean target=features.1.conv.0.1.running_mean} ->[n13] constant,
     t162 f32 [C=32] {pt2=root:b_features_1_conv_0_1_running_var target=features.1.conv.0.1.running_var} ->[n13] constant,
     t164 f32 [C=16] {pt2=root:b_features_1_conv_2_running_mean target=features.1.conv.2.running_mean} ->[n21] constant,
     t165 f32 [C=16] {pt2=root:b_features_1_conv_2_running_var target=features.1.conv.2.running_var} ->[n21] constant,
     t167 f32 [C=96] {pt2=root:b_features_2_conv_0_1_running_mean target=features.2.conv.0.1.running_mean} ->[n28] constant,
     t168 f32 [C=96] {pt2=root:b_features_2_conv_0_1_running_var target=features.2.conv.0.1.running_var} ->[n28] constant,
     t170 f32 [C=96] {pt2=root:b_features_2_conv_1_1_running_mean target=features.2.conv.1.1.running_mean} ->[n36] constant,
     t171 f32 [C=96] {pt2=root:b_features_2_conv_1_1_running_var target=features.2.conv.1.1.running_var} ->[n36] constant,
     t173 f32 [C=24] {pt2=root:b_features_2_conv_3_running_mean target=features.2.conv.3.running_mean} ->[n44] constant,
     t174 f32 [C=24] {pt2=root:b_features_2_conv_3_running_var target=features.2.conv.3.running_var} ->[n44] constant,
     t176 f32 [C=144] {pt2=root:b_features_3_conv_0_1_running_mean target=features.3.conv.0.1.running_mean} ->[n51] constant,
     t177 f32 [C=144] {pt2=root:b_features_3_conv_0_1_running_var target=features.3.conv.0.1.running_var} ->[n51] constant,
     t179 f32 [C=144] {pt2=root:b_features_3_conv_1_1_running_mean target=features.3.conv.1.1.running_mean} ->[n59] constant,
     t180 f32 [C=144] {pt2=root:b_features_3_conv_1_1_running_var target=features.3.conv.1.1.running_var} ->[n59] constant,
     t182 f32 [C=24] {pt2=root:b_features_3_conv_3_running_mean target=features.3.conv.3.running_mean} ->[n67] constant,
     t183 f32 [C=24] {pt2=root:b_features_3_conv_3_running_var target=features.3.conv.3.running_var} ->[n67] constant,
     t185 f32 [C=144] {pt2=root:b_features_4_conv_0_1_running_mean target=features.4.conv.0.1.running_mean} ->[n75] constant,
     t186 f32 [C=144] {pt2=root:b_features_4_conv_0_1_running_var target=features.4.conv.0.1.running_var} ->[n75] constant,
     t188 f32 [C=144] {pt2=root:b_features_4_conv_1_1_running_mean target=features.4.conv.1.1.running_mean} ->[n83] constant,
     t189 f32 [C=144] {pt2=root:b_features_4_conv_1_1_running_var target=features.4.conv.1.1.running_var} ->[n83] constant,
     t191 f32 [C=32] {pt2=root:b_features_4_conv_3_running_mean target=features.4.conv.3.running_mean} ->[n91] constant,
     t192 f32 [C=32] {pt2=root:b_features_4_conv_3_running_var target=features.4.conv.3.running_var} ->[n91] constant,
     t194 f32 [C=192] {pt2=root:b_features_5_conv_0_1_running_mean target=features.5.conv.0.1.running_mean} ->[n98] constant,
     t195 f32 [C=192] {pt2=root:b_features_5_conv_0_1_running_var target=features.5.conv.0.1.running_var} ->[n98] constant,
     t197 f32 [C=192] {pt2=root:b_features_5_conv_1_1_running_mean target=features.5.conv.1.1.running_mean} ->[n106] constant,
     t198 f32 [C=192] {pt2=root:b_features_5_conv_1_1_running_var target=features.5.conv.1.1.running_var} ->[n106] constant,
     t200 f32 [C=32] {pt2=root:b_features_5_conv_3_running_mean target=features.5.conv.3.running_mean} ->[n114] constant,
     t201 f32 [C=32] {pt2=root:b_features_5_conv_3_running_var target=features.5.conv.3.running_var} ->[n114] constant,
     t203 f32 [C=192] {pt2=root:b_features_6_conv_0_1_running_mean target=features.6.conv.0.1.running_mean} ->[n122] constant,
     t204 f32 [C=192] {pt2=root:b_features_6_conv_0_1_running_var target=features.6.conv.0.1.running_var} ->[n122] constant,
     t206 f32 [C=192] {pt2=root:b_features_6_conv_1_1_running_mean target=features.6.conv.1.1.running_mean} ->[n130] constant,
     t207 f32 [C=192] {pt2=root:b_features_6_conv_1_1_running_var target=features.6.conv.1.1.running_var} ->[n130] constant,
     t209 f32 [C=32] {pt2=root:b_features_6_conv_3_running_mean target=features.6.conv.3.running_mean} ->[n138] constant,
     t210 f32 [C=32] {pt2=root:b_features_6_conv_3_running_var target=features.6.conv.3.running_var} ->[n138] constant,
     t212 f32 [C=192] {pt2=root:b_features_7_conv_0_1_running_mean target=features.7.conv.0.1.running_mean} ->[n146] constant,
     t213 f32 [C=192] {pt2=root:b_features_7_conv_0_1_running_var target=features.7.conv.0.1.running_var} ->[n146] constant,
     t215 f32 [C=192] {pt2=root:b_features_7_conv_1_1_running_mean target=features.7.conv.1.1.running_mean} ->[n154] constant,
     t216 f32 [C=192] {pt2=root:b_features_7_conv_1_1_running_var target=features.7.conv.1.1.running_var} ->[n154] constant,
     t218 f32 [C=64] {pt2=root:b_features_7_conv_3_running_mean target=features.7.conv.3.running_mean} ->[n162] constant,
     t219 f32 [C=64] {pt2=root:b_features_7_conv_3_running_var target=features.7.conv.3.running_var} ->[n162] constant,
     t221 f32 [C=384] {pt2=root:b_features_8_conv_0_1_running_mean target=features.8.conv.0.1.running_mean} ->[n169] constant,
     t222 f32 [C=384] {pt2=root:b_features_8_conv_0_1_running_var target=features.8.conv.0.1.running_var} ->[n169] constant,
     t224 f32 [C=384] {pt2=root:b_features_8_conv_1_1_running_mean target=features.8.conv.1.1.running_mean} ->[n177] constant,
     t225 f32 [C=384] {pt2=root:b_features_8_conv_1_1_running_var target=features.8.conv.1.1.running_var} ->[n177] constant,
     t227 f32 [C=64] {pt2=root:b_features_8_conv_3_running_mean target=features.8.conv.3.running_mean} ->[n185] constant,
     t228 f32 [C=64] {pt2=root:b_features_8_conv_3_running_var target=features.8.conv.3.running_var} ->[n185] constant,
     t230 f32 [C=384] {pt2=root:b_features_9_conv_0_1_running_mean target=features.9.conv.0.1.running_mean} ->[n193] constant,
     t231 f32 [C=384] {pt2=root:b_features_9_conv_0_1_running_var target=features.9.conv.0.1.running_var} ->[n193] constant,
     t233 f32 [C=384] {pt2=root:b_features_9_conv_1_1_running_mean target=features.9.conv.1.1.running_mean} ->[n201] constant,
     t234 f32 [C=384] {pt2=root:b_features_9_conv_1_1_running_var target=features.9.conv.1.1.running_var} ->[n201] constant,
     t236 f32 [C=64] {pt2=root:b_features_9_conv_3_running_mean target=features.9.conv.3.running_mean} ->[n209] constant,
     t237 f32 [C=64] {pt2=root:b_features_9_conv_3_running_var target=features.9.conv.3.running_var} ->[n209] constant,
     t239 f32 [C=384] {pt2=root:b_features_10_conv_0_1_running_mean target=features.10.conv.0.1.running_mean} ->[n217] constant,
     t240 f32 [C=384] {pt2=root:b_features_10_conv_0_1_running_var target=features.10.conv.0.1.running_var} ->[n217] constant,
     t242 f32 [C=384] {pt2=root:b_features_10_conv_1_1_running_mean target=features.10.conv.1.1.running_mean} ->[n225] constant,
     t243 f32 [C=384] {pt2=root:b_features_10_conv_1_1_running_var target=features.10.conv.1.1.running_var} ->[n225] constant,
     t245 f32 [C=64] {pt2=root:b_features_10_conv_3_running_mean target=features.10.conv.3.running_mean} ->[n233] constant,
     t246 f32 [C=64] {pt2=root:b_features_10_conv_3_running_var target=features.10.conv.3.running_var} ->[n233] constant,
     t248 f32 [C=384] {pt2=root:b_features_11_conv_0_1_running_mean target=features.11.conv.0.1.running_mean} ->[n241] constant,
     t249 f32 [C=384] {pt2=root:b_features_11_conv_0_1_running_var target=features.11.conv.0.1.running_var} ->[n241] constant,
     t251 f32 [C=384] {pt2=root:b_features_11_conv_1_1_running_mean target=features.11.conv.1.1.running_mean} ->[n249] constant,
     t252 f32 [C=384] {pt2=root:b_features_11_conv_1_1_running_var target=features.11.conv.1.1.running_var} ->[n249] constant,
     t254 f32 [C=96] {pt2=root:b_features_11_conv_3_running_mean target=features.11.conv.3.running_mean} ->[n257] constant,
     t255 f32 [C=96] {pt2=root:b_features_11_conv_3_running_var target=features.11.conv.3.running_var} ->[n257] constant,
     t257 f32 [C=576] {pt2=root:b_features_12_conv_0_1_running_mean target=features.12.conv.0.1.running_mean} ->[n264] constant,
     t258 f32 [C=576] {pt2=root:b_features_12_conv_0_1_running_var target=features.12.conv.0.1.running_var} ->[n264] constant,
     t260 f32 [C=576] {pt2=root:b_features_12_conv_1_1_running_mean target=features.12.conv.1.1.running_mean} ->[n272] constant,
     t261 f32 [C=576] {pt2=root:b_features_12_conv_1_1_running_var target=features.12.conv.1.1.running_var} ->[n272] constant,
     t263 f32 [C=96] {pt2=root:b_features_12_conv_3_running_mean target=features.12.conv.3.running_mean} ->[n280] constant,
     t264 f32 [C=96] {pt2=root:b_features_12_conv_3_running_var target=features.12.conv.3.running_var} ->[n280] constant,
     t266 f32 [C=576] {pt2=root:b_features_13_conv_0_1_running_mean target=features.13.conv.0.1.running_mean} ->[n288] constant,
     t267 f32 [C=576] {pt2=root:b_features_13_conv_0_1_running_var target=features.13.conv.0.1.running_var} ->[n288] constant,
     t269 f32 [C=576] {pt2=root:b_features_13_conv_1_1_running_mean target=features.13.conv.1.1.running_mean} ->[n296] constant,
     t270 f32 [C=576] {pt2=root:b_features_13_conv_1_1_running_var target=features.13.conv.1.1.running_var} ->[n296] constant,
     t272 f32 [C=96] {pt2=root:b_features_13_conv_3_running_mean target=features.13.conv.3.running_mean} ->[n304] constant,
     t273 f32 [C=96] {pt2=root:b_features_13_conv_3_running_var target=features.13.conv.3.running_var} ->[n304] constant,
     t275 f32 [C=576] {pt2=root:b_features_14_conv_0_1_running_mean target=features.14.conv.0.1.running_mean} ->[n312] constant,
     t276 f32 [C=576] {pt2=root:b_features_14_conv_0_1_running_var target=features.14.conv.0.1.running_var} ->[n312] constant,
     t278 f32 [C=576] {pt2=root:b_features_14_conv_1_1_running_mean target=features.14.conv.1.1.running_mean} ->[n320] constant,
     t279 f32 [C=576] {pt2=root:b_features_14_conv_1_1_running_var target=features.14.conv.1.1.running_var} ->[n320] constant,
     t281 f32 [C=160] {pt2=root:b_features_14_conv_3_running_mean target=features.14.conv.3.running_mean} ->[n328] constant,
     t282 f32 [C=160] {pt2=root:b_features_14_conv_3_running_var target=features.14.conv.3.running_var} ->[n328] constant,
     t284 f32 [C=960] {pt2=root:b_features_15_conv_0_1_running_mean target=features.15.conv.0.1.running_mean} ->[n335] constant,
     t285 f32 [C=960] {pt2=root:b_features_15_conv_0_1_running_var target=features.15.conv.0.1.running_var} ->[n335] constant,
     t287 f32 [C=960] {pt2=root:b_features_15_conv_1_1_running_mean target=features.15.conv.1.1.running_mean} ->[n343] constant,
     t288 f32 [C=960] {pt2=root:b_features_15_conv_1_1_running_var target=features.15.conv.1.1.running_var} ->[n343] constant,
     t290 f32 [C=160] {pt2=root:b_features_15_conv_3_running_mean target=features.15.conv.3.running_mean} ->[n351] constant,
     t291 f32 [C=160] {pt2=root:b_features_15_conv_3_running_var target=features.15.conv.3.running_var} ->[n351] constant,
     t293 f32 [C=960] {pt2=root:b_features_16_conv_0_1_running_mean target=features.16.conv.0.1.running_mean} ->[n359] constant,
     t294 f32 [C=960] {pt2=root:b_features_16_conv_0_1_running_var target=features.16.conv.0.1.running_var} ->[n359] constant,
     t296 f32 [C=960] {pt2=root:b_features_16_conv_1_1_running_mean target=features.16.conv.1.1.running_mean} ->[n367] constant,
     t297 f32 [C=960] {pt2=root:b_features_16_conv_1_1_running_var target=features.16.conv.1.1.running_var} ->[n367] constant,
     t299 f32 [C=160] {pt2=root:b_features_16_conv_3_running_mean target=features.16.conv.3.running_mean} ->[n375] constant,
     t300 f32 [C=160] {pt2=root:b_features_16_conv_3_running_var target=features.16.conv.3.running_var} ->[n375] constant,
     t302 f32 [C=960] {pt2=root:b_features_17_conv_0_1_running_mean target=features.17.conv.0.1.running_mean} ->[n383] constant,
     t303 f32 [C=960] {pt2=root:b_features_17_conv_0_1_running_var target=features.17.conv.0.1.running_var} ->[n383] constant,
     t305 f32 [C=960] {pt2=root:b_features_17_conv_1_1_running_mean target=features.17.conv.1.1.running_mean} ->[n391] constant,
     t306 f32 [C=960] {pt2=root:b_features_17_conv_1_1_running_var target=features.17.conv.1.1.running_var} ->[n391] constant,
     t308 f32 [C=320] {pt2=root:b_features_17_conv_3_running_mean target=features.17.conv.3.running_mean} ->[n399] constant,
     t309 f32 [C=320] {pt2=root:b_features_17_conv_3_running_var target=features.17.conv.3.running_var} ->[n399] constant,
     t311 f32 [C=1280] {pt2=root:b_features_18_1_running_mean target=features.18.1.running_mean} ->[n406] constant,
     t312 f32 [C=1280] {pt2=root:b_features_18_1_running_var target=features.18.1.running_var} ->[n406] constant,
     t314 f32 [H=3 W=224 C=224] {pt2=root:x} ->[n0]]
  nodes:
    group g1 torch.ops.aten.convolution.default:
      n0 {derived}: [t315 f32 [H=224 W=224 C=3] {derived} ->[n2]] =
        permute x=t314 {pt2=root:x} perm=[H<-W, W<-C, C<-H]
      n1 {derived}: [t316 f32 [N=32 T=1 D=1 H=3 W=3 C=3] {derived} ->[n2]] =
        permute
          x=t0 {pt2=root:p_features_0_0_weight target=features.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2 {derived}: [t317 f32 [H=112 W=112 C=32] {derived} ->[n5]] =
        convolution
          x=t315 {derived} <-n0
          weight=t316 {derived} <-n1
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g3 torch.ops.aten.convolution.default:
      n9 {derived}: [t324 f32 [N=32 T=1 D=1 H=3 W=3 C=1] {derived} ->[n10]] =
        permute
          x=t3 {pt2=root:p_features_1_conv_0_0_weight target=features.1.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n10 {derived}: [t325 f32 [H=112 W=112 C=32] {derived} ->[n13]] =
        convolution
          x=t730 {derived} <-n416
          weight=t324 {derived} <-n9
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=32}
    group g5 torch.ops.aten.convolution.default:
      n17 {derived}: [t332 f32 [N=16 T=1 D=1 H=1 W=1 C=32] {derived} ->[n18]] =
        permute
          x=t6 {pt2=root:p_features_1_conv_1_weight target=features.1.conv.1.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n18 {derived}: [t333 f32 [H=112 W=112 C=16] {derived} ->[n21]] =
        convolution
          x=t731 {derived} <-n417
          weight=t332 {derived} <-n17
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g7 torch.ops.aten.convolution.default:
      n24 {derived}: [t339 f32 [N=96 T=1 D=1 H=1 W=1 C=16] {derived} ->[n25]] =
        permute
          x=t9 {pt2=root:p_features_2_conv_0_0_weight target=features.2.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n25 {derived}: [t340 f32 [H=112 W=112 C=96] {derived} ->[n28]] =
        convolution
          x=t336 {derived} <-n21
          weight=t339 {derived} <-n24
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g9 torch.ops.aten.convolution.default:
      n32 {derived}: [t347 f32 [N=96 T=1 D=1 H=3 W=3 C=1] {derived} ->[n33]] =
        permute
          x=t12 {pt2=root:p_features_2_conv_1_0_weight target=features.2.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n33 {derived}: [t348 f32 [H=56 W=56 C=96] {derived} ->[n36]] =
        convolution
          x=t732 {derived} <-n418
          weight=t347 {derived} <-n32
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=96}
    group g11 torch.ops.aten.convolution.default:
      n40 {derived}: [t355 f32 [N=24 T=1 D=1 H=1 W=1 C=96] {derived} ->[n41]] =
        permute
          x=t15 {pt2=root:p_features_2_conv_2_weight target=features.2.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n41 {derived}: [t356 f32 [H=56 W=56 C=24] {derived} ->[n44]] =
        convolution
          x=t733 {derived} <-n419
          weight=t355 {derived} <-n40
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g13 torch.ops.aten.convolution.default:
      n47 {derived}: [t362 f32 [N=144 T=1 D=1 H=1 W=1 C=24] {derived} ->[n48]] =
        permute
          x=t18 {pt2=root:p_features_3_conv_0_0_weight target=features.3.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n48 {derived}: [t363 f32 [H=56 W=56 C=144] {derived} ->[n51]] =
        convolution
          x=t359 {derived} <-n44
          weight=t362 {derived} <-n47
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g15 torch.ops.aten.convolution.default:
      n55 {derived}: [t370 f32 [N=144 T=1 D=1 H=3 W=3 C=1] {derived} ->[n56]] =
        permute
          x=t21 {pt2=root:p_features_3_conv_1_0_weight target=features.3.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n56 {derived}: [t371 f32 [H=56 W=56 C=144] {derived} ->[n59]] =
        convolution
          x=t734 {derived} <-n420
          weight=t370 {derived} <-n55
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=144}
    group g17 torch.ops.aten.convolution.default:
      n63 {derived}: [t378 f32 [N=24 T=1 D=1 H=1 W=1 C=144] {derived} ->[n64]] =
        permute
          x=t24 {pt2=root:p_features_3_conv_2_weight target=features.3.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n64 {derived}: [t379 f32 [H=56 W=56 C=24] {derived} ->[n67]] =
        convolution
          x=t735 {derived} <-n421
          weight=t378 {derived} <-n63
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g19 torch.ops.aten.convolution.default:
      n71 {derived}: [t386 f32 [N=144 T=1 D=1 H=1 W=1 C=24] {derived} ->[n72]] =
        permute
          x=t27 {pt2=root:p_features_4_conv_0_0_weight target=features.4.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n72 {derived}: [t387 f32 [H=56 W=56 C=144] {derived} ->[n75]] =
        convolution
          x=t736 {derived} <-n422
          weight=t386 {derived} <-n71
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g21 torch.ops.aten.convolution.default:
      n79 {derived}: [t394 f32 [N=144 T=1 D=1 H=3 W=3 C=1] {derived} ->[n80]] =
        permute
          x=t30 {pt2=root:p_features_4_conv_1_0_weight target=features.4.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n80 {derived}: [t395 f32 [H=28 W=28 C=144] {derived} ->[n83]] =
        convolution
          x=t737 {derived} <-n423
          weight=t394 {derived} <-n79
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=144}
    group g23 torch.ops.aten.convolution.default:
      n87 {derived}: [t402 f32 [N=32 T=1 D=1 H=1 W=1 C=144] {derived} ->[n88]] =
        permute
          x=t33 {pt2=root:p_features_4_conv_2_weight target=features.4.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n88 {derived}: [t403 f32 [H=28 W=28 C=32] {derived} ->[n91]] =
        convolution
          x=t738 {derived} <-n424
          weight=t402 {derived} <-n87
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g25 torch.ops.aten.convolution.default:
      n94 {derived}: [t409 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {derived} ->[n95]] =
        permute
          x=t36 {pt2=root:p_features_5_conv_0_0_weight target=features.5.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n95 {derived}: [t410 f32 [H=28 W=28 C=192] {derived} ->[n98]] =
        convolution
          x=t406 {derived} <-n91
          weight=t409 {derived} <-n94
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g27 torch.ops.aten.convolution.default:
      n102 {derived}: [t417 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {derived} ->[n103]] =
        permute
          x=t39 {pt2=root:p_features_5_conv_1_0_weight target=features.5.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n103 {derived}: [t418 f32 [H=28 W=28 C=192] {derived} ->[n106]] =
        convolution
          x=t739 {derived} <-n425
          weight=t417 {derived} <-n102
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=192}
    group g29 torch.ops.aten.convolution.default:
      n110 {derived}: [t425 f32 [N=32 T=1 D=1 H=1 W=1 C=192] {derived} ->[n111]] =
        permute
          x=t42 {pt2=root:p_features_5_conv_2_weight target=features.5.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n111 {derived}: [t426 f32 [H=28 W=28 C=32] {derived} ->[n114]] =
        convolution
          x=t740 {derived} <-n426
          weight=t425 {derived} <-n110
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g31 torch.ops.aten.convolution.default:
      n118 {derived}: [t433 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {derived} ->[n119]] =
        permute
          x=t45 {pt2=root:p_features_6_conv_0_0_weight target=features.6.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n119 {derived}: [t434 f32 [H=28 W=28 C=192] {derived} ->[n122]] =
        convolution
          x=t741 {derived} <-n427
          weight=t433 {derived} <-n118
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g33 torch.ops.aten.convolution.default:
      n126 {derived}: [t441 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {derived} ->[n127]] =
        permute
          x=t48 {pt2=root:p_features_6_conv_1_0_weight target=features.6.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n127 {derived}: [t442 f32 [H=28 W=28 C=192] {derived} ->[n130]] =
        convolution
          x=t742 {derived} <-n428
          weight=t441 {derived} <-n126
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=192}
    group g35 torch.ops.aten.convolution.default:
      n134 {derived}: [t449 f32 [N=32 T=1 D=1 H=1 W=1 C=192] {derived} ->[n135]] =
        permute
          x=t51 {pt2=root:p_features_6_conv_2_weight target=features.6.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n135 {derived}: [t450 f32 [H=28 W=28 C=32] {derived} ->[n138]] =
        convolution
          x=t743 {derived} <-n429
          weight=t449 {derived} <-n134
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g37 torch.ops.aten.convolution.default:
      n142 {derived}: [t457 f32 [N=192 T=1 D=1 H=1 W=1 C=32] {derived} ->[n143]] =
        permute
          x=t54 {pt2=root:p_features_7_conv_0_0_weight target=features.7.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n143 {derived}: [t458 f32 [H=28 W=28 C=192] {derived} ->[n146]] =
        convolution
          x=t744 {derived} <-n430
          weight=t457 {derived} <-n142
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g39 torch.ops.aten.convolution.default:
      n150 {derived}: [t465 f32 [N=192 T=1 D=1 H=3 W=3 C=1] {derived} ->[n151]] =
        permute
          x=t57 {pt2=root:p_features_7_conv_1_0_weight target=features.7.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n151 {derived}: [t466 f32 [H=14 W=14 C=192] {derived} ->[n154]] =
        convolution
          x=t745 {derived} <-n431
          weight=t465 {derived} <-n150
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=192}
    group g41 torch.ops.aten.convolution.default:
      n158 {derived}: [t473 f32 [N=64 T=1 D=1 H=1 W=1 C=192] {derived} ->[n159]] =
        permute
          x=t60 {pt2=root:p_features_7_conv_2_weight target=features.7.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n159 {derived}: [t474 f32 [H=14 W=14 C=64] {derived} ->[n162]] =
        convolution
          x=t746 {derived} <-n432
          weight=t473 {derived} <-n158
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g43 torch.ops.aten.convolution.default:
      n165 {derived}: [t480 f32 [N=384 T=1 D=1 H=1 W=1 C=64] {derived} ->[n166]] =
        permute
          x=t63 {pt2=root:p_features_8_conv_0_0_weight target=features.8.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n166 {derived}: [t481 f32 [H=14 W=14 C=384] {derived} ->[n169]] =
        convolution
          x=t477 {derived} <-n162
          weight=t480 {derived} <-n165
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g45 torch.ops.aten.convolution.default:
      n173 {derived}: [t488 f32 [N=384 T=1 D=1 H=3 W=3 C=1] {derived} ->[n174]] =
        permute
          x=t66 {pt2=root:p_features_8_conv_1_0_weight target=features.8.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n174 {derived}: [t489 f32 [H=14 W=14 C=384] {derived} ->[n177]] =
        convolution
          x=t747 {derived} <-n433
          weight=t488 {derived} <-n173
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=384}
    group g47 torch.ops.aten.convolution.default:
      n181 {derived}: [t496 f32 [N=64 T=1 D=1 H=1 W=1 C=384] {derived} ->[n182]] =
        permute
          x=t69 {pt2=root:p_features_8_conv_2_weight target=features.8.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n182 {derived}: [t497 f32 [H=14 W=14 C=64] {derived} ->[n185]] =
        convolution
          x=t748 {derived} <-n434
          weight=t496 {derived} <-n181
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g49 torch.ops.aten.convolution.default:
      n189 {derived}: [t504 f32 [N=384 T=1 D=1 H=1 W=1 C=64] {derived} ->[n190]] =
        permute
          x=t72 {pt2=root:p_features_9_conv_0_0_weight target=features.9.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n190 {derived}: [t505 f32 [H=14 W=14 C=384] {derived} ->[n193]] =
        convolution
          x=t749 {derived} <-n435
          weight=t504 {derived} <-n189
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g51 torch.ops.aten.convolution.default:
      n197 {derived}: [t512 f32 [N=384 T=1 D=1 H=3 W=3 C=1] {derived} ->[n198]] =
        permute
          x=t75 {pt2=root:p_features_9_conv_1_0_weight target=features.9.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n198 {derived}: [t513 f32 [H=14 W=14 C=384] {derived} ->[n201]] =
        convolution
          x=t750 {derived} <-n436
          weight=t512 {derived} <-n197
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=384}
    group g53 torch.ops.aten.convolution.default:
      n205 {derived}: [t520 f32 [N=64 T=1 D=1 H=1 W=1 C=384] {derived} ->[n206]] =
        permute
          x=t78 {pt2=root:p_features_9_conv_2_weight target=features.9.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n206 {derived}: [t521 f32 [H=14 W=14 C=64] {derived} ->[n209]] =
        convolution
          x=t751 {derived} <-n437
          weight=t520 {derived} <-n205
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g55 torch.ops.aten.convolution.default:
      n213 {derived}: [t528 f32 [N=384 T=1 D=1 H=1 W=1 C=64] {derived} ->[n214]] =
        permute
          x=t81 {pt2=root:p_features_10_conv_0_0_weight target=features.10.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n214 {derived}: [t529 f32 [H=14 W=14 C=384] {derived} ->[n217]] =
        convolution
          x=t752 {derived} <-n438
          weight=t528 {derived} <-n213
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g57 torch.ops.aten.convolution.default:
      n221 {derived}: [t536 f32 [N=384 T=1 D=1 H=3 W=3 C=1] {derived} ->[n222]] =
        permute
          x=t84 {pt2=root:p_features_10_conv_1_0_weight target=features.10.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n222 {derived}: [t537 f32 [H=14 W=14 C=384] {derived} ->[n225]] =
        convolution
          x=t753 {derived} <-n439
          weight=t536 {derived} <-n221
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=384}
    group g59 torch.ops.aten.convolution.default:
      n229 {derived}: [t544 f32 [N=64 T=1 D=1 H=1 W=1 C=384] {derived} ->[n230]] =
        permute
          x=t87 {pt2=root:p_features_10_conv_2_weight target=features.10.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n230 {derived}: [t545 f32 [H=14 W=14 C=64] {derived} ->[n233]] =
        convolution
          x=t754 {derived} <-n440
          weight=t544 {derived} <-n229
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g61 torch.ops.aten.convolution.default:
      n237 {derived}: [t552 f32 [N=384 T=1 D=1 H=1 W=1 C=64] {derived} ->[n238]] =
        permute
          x=t90 {pt2=root:p_features_11_conv_0_0_weight target=features.11.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n238 {derived}: [t553 f32 [H=14 W=14 C=384] {derived} ->[n241]] =
        convolution
          x=t755 {derived} <-n441
          weight=t552 {derived} <-n237
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g63 torch.ops.aten.convolution.default:
      n245 {derived}: [t560 f32 [N=384 T=1 D=1 H=3 W=3 C=1] {derived} ->[n246]] =
        permute
          x=t93 {pt2=root:p_features_11_conv_1_0_weight target=features.11.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n246 {derived}: [t561 f32 [H=14 W=14 C=384] {derived} ->[n249]] =
        convolution
          x=t756 {derived} <-n442
          weight=t560 {derived} <-n245
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=384}
    group g65 torch.ops.aten.convolution.default:
      n253 {derived}: [t568 f32 [N=96 T=1 D=1 H=1 W=1 C=384] {derived} ->[n254]] =
        permute
          x=t96 {pt2=root:p_features_11_conv_2_weight target=features.11.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n254 {derived}: [t569 f32 [H=14 W=14 C=96] {derived} ->[n257]] =
        convolution
          x=t757 {derived} <-n443
          weight=t568 {derived} <-n253
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g67 torch.ops.aten.convolution.default:
      n260 {derived}: [t575 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived} ->[n261]] =
        permute
          x=t99 {pt2=root:p_features_12_conv_0_0_weight target=features.12.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n261 {derived}: [t576 f32 [H=14 W=14 C=576] {derived} ->[n264]] =
        convolution
          x=t572 {derived} <-n257
          weight=t575 {derived} <-n260
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g69 torch.ops.aten.convolution.default:
      n268 {derived}: [t583 f32 [N=576 T=1 D=1 H=3 W=3 C=1] {derived} ->[n269]] =
        permute
          x=t102 {pt2=root:p_features_12_conv_1_0_weight target=features.12.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n269 {derived}: [t584 f32 [H=14 W=14 C=576] {derived} ->[n272]] =
        convolution
          x=t758 {derived} <-n444
          weight=t583 {derived} <-n268
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=576}
    group g71 torch.ops.aten.convolution.default:
      n276 {derived}: [t591 f32 [N=96 T=1 D=1 H=1 W=1 C=576] {derived} ->[n277]] =
        permute
          x=t105 {pt2=root:p_features_12_conv_2_weight target=features.12.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n277 {derived}: [t592 f32 [H=14 W=14 C=96] {derived} ->[n280]] =
        convolution
          x=t759 {derived} <-n445
          weight=t591 {derived} <-n276
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g73 torch.ops.aten.convolution.default:
      n284 {derived}: [t599 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived} ->[n285]] =
        permute
          x=t108 {pt2=root:p_features_13_conv_0_0_weight target=features.13.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n285 {derived}: [t600 f32 [H=14 W=14 C=576] {derived} ->[n288]] =
        convolution
          x=t760 {derived} <-n446
          weight=t599 {derived} <-n284
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g75 torch.ops.aten.convolution.default:
      n292 {derived}: [t607 f32 [N=576 T=1 D=1 H=3 W=3 C=1] {derived} ->[n293]] =
        permute
          x=t111 {pt2=root:p_features_13_conv_1_0_weight target=features.13.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n293 {derived}: [t608 f32 [H=14 W=14 C=576] {derived} ->[n296]] =
        convolution
          x=t761 {derived} <-n447
          weight=t607 {derived} <-n292
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=576}
    group g77 torch.ops.aten.convolution.default:
      n300 {derived}: [t615 f32 [N=96 T=1 D=1 H=1 W=1 C=576] {derived} ->[n301]] =
        permute
          x=t114 {pt2=root:p_features_13_conv_2_weight target=features.13.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n301 {derived}: [t616 f32 [H=14 W=14 C=96] {derived} ->[n304]] =
        convolution
          x=t762 {derived} <-n448
          weight=t615 {derived} <-n300
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g79 torch.ops.aten.convolution.default:
      n308 {derived}: [t623 f32 [N=576 T=1 D=1 H=1 W=1 C=96] {derived} ->[n309]] =
        permute
          x=t117 {pt2=root:p_features_14_conv_0_0_weight target=features.14.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n309 {derived}: [t624 f32 [H=14 W=14 C=576] {derived} ->[n312]] =
        convolution
          x=t763 {derived} <-n449
          weight=t623 {derived} <-n308
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g81 torch.ops.aten.convolution.default:
      n316 {derived}: [t631 f32 [N=576 T=1 D=1 H=3 W=3 C=1] {derived} ->[n317]] =
        permute
          x=t120 {pt2=root:p_features_14_conv_1_0_weight target=features.14.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n317 {derived}: [t632 f32 [H=7 W=7 C=576] {derived} ->[n320]] =
        convolution
          x=t764 {derived} <-n450
          weight=t631 {derived} <-n316
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=576}
    group g83 torch.ops.aten.convolution.default:
      n324 {derived}: [t639 f32 [N=160 T=1 D=1 H=1 W=1 C=576] {derived} ->[n325]] =
        permute
          x=t123 {pt2=root:p_features_14_conv_2_weight target=features.14.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n325 {derived}: [t640 f32 [H=7 W=7 C=160] {derived} ->[n328]] =
        convolution
          x=t765 {derived} <-n451
          weight=t639 {derived} <-n324
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g85 torch.ops.aten.convolution.default:
      n331 {derived}: [t646 f32 [N=960 T=1 D=1 H=1 W=1 C=160] {derived} ->[n332]] =
        permute
          x=t126 {pt2=root:p_features_15_conv_0_0_weight target=features.15.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n332 {derived}: [t647 f32 [H=7 W=7 C=960] {derived} ->[n335]] =
        convolution
          x=t643 {derived} <-n328
          weight=t646 {derived} <-n331
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g87 torch.ops.aten.convolution.default:
      n339 {derived}: [t654 f32 [N=960 T=1 D=1 H=3 W=3 C=1] {derived} ->[n340]] =
        permute
          x=t129 {pt2=root:p_features_15_conv_1_0_weight target=features.15.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n340 {derived}: [t655 f32 [H=7 W=7 C=960] {derived} ->[n343]] =
        convolution
          x=t766 {derived} <-n452
          weight=t654 {derived} <-n339
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=960}
    group g89 torch.ops.aten.convolution.default:
      n347 {derived}: [t662 f32 [N=160 T=1 D=1 H=1 W=1 C=960] {derived} ->[n348]] =
        permute
          x=t132 {pt2=root:p_features_15_conv_2_weight target=features.15.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n348 {derived}: [t663 f32 [H=7 W=7 C=160] {derived} ->[n351]] =
        convolution
          x=t767 {derived} <-n453
          weight=t662 {derived} <-n347
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g91 torch.ops.aten.convolution.default:
      n355 {derived}: [t670 f32 [N=960 T=1 D=1 H=1 W=1 C=160] {derived} ->[n356]] =
        permute
          x=t135 {pt2=root:p_features_16_conv_0_0_weight target=features.16.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n356 {derived}: [t671 f32 [H=7 W=7 C=960] {derived} ->[n359]] =
        convolution
          x=t768 {derived} <-n454
          weight=t670 {derived} <-n355
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g93 torch.ops.aten.convolution.default:
      n363 {derived}: [t678 f32 [N=960 T=1 D=1 H=3 W=3 C=1] {derived} ->[n364]] =
        permute
          x=t138 {pt2=root:p_features_16_conv_1_0_weight target=features.16.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n364 {derived}: [t679 f32 [H=7 W=7 C=960] {derived} ->[n367]] =
        convolution
          x=t769 {derived} <-n455
          weight=t678 {derived} <-n363
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=960}
    group g95 torch.ops.aten.convolution.default:
      n371 {derived}: [t686 f32 [N=160 T=1 D=1 H=1 W=1 C=960] {derived} ->[n372]] =
        permute
          x=t141 {pt2=root:p_features_16_conv_2_weight target=features.16.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n372 {derived}: [t687 f32 [H=7 W=7 C=160] {derived} ->[n375]] =
        convolution
          x=t770 {derived} <-n456
          weight=t686 {derived} <-n371
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g97 torch.ops.aten.convolution.default:
      n379 {derived}: [t694 f32 [N=960 T=1 D=1 H=1 W=1 C=160] {derived} ->[n380]] =
        permute
          x=t144 {pt2=root:p_features_17_conv_0_0_weight target=features.17.conv.0.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n380 {derived}: [t695 f32 [H=7 W=7 C=960] {derived} ->[n383]] =
        convolution
          x=t771 {derived} <-n457
          weight=t694 {derived} <-n379
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g99 torch.ops.aten.convolution.default:
      n387 {derived}: [t702 f32 [N=960 T=1 D=1 H=3 W=3 C=1] {derived} ->[n388]] =
        permute
          x=t147 {pt2=root:p_features_17_conv_1_0_weight target=features.17.conv.1.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n388 {derived}: [t703 f32 [H=7 W=7 C=960] {derived} ->[n391]] =
        convolution
          x=t772 {derived} <-n458
          weight=t702 {derived} <-n387
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=960}
    group g101 torch.ops.aten.convolution.default:
      n395 {derived}: [t710 f32 [N=320 T=1 D=1 H=1 W=1 C=960] {derived} ->[n396]] =
        permute
          x=t150 {pt2=root:p_features_17_conv_2_weight target=features.17.conv.2.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n396 {derived}: [t711 f32 [H=7 W=7 C=320] {derived} ->[n399]] =
        convolution
          x=t773 {derived} <-n459
          weight=t710 {derived} <-n395
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g103 torch.ops.aten.convolution.default:
      n402 {derived}: [t717 f32 [N=1280 T=1 D=1 H=1 W=1 C=320] {derived} ->[n403]] =
        permute
          x=t153 {pt2=root:p_features_18_0_weight target=features.18.0.weight}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n403 {derived}: [t718 f32 [H=7 W=7 C=1280] {derived} ->[n406]] =
        convolution
          x=t714 {derived} <-n399
          weight=t717 {derived} <-n402
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n415 {pt2=root[152] torch.ops.aten.permute.default}: [t728 f32 [N=1000 T=1
                                                                    D=1 H=1 W=1
                                                                    C=1280] {derived} ->[n414]] =
      permute
        x=t156 {pt2=root:p_classifier_1_weight target=classifier.1.weight}
        perm=[N<-W, W<-N]
    group g2 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n5 {derived}: [t320 f32 [H=112 W=112 C=32] {derived} ->[n416]] =
        batch_norm
          x=t317 {derived} <-n2
          weight=t1 {pt2=root:p_features_0_1_weight target=features.0.1.weight}
          bias=t2 {pt2=root:p_features_0_1_bias target=features.0.1.bias}
          running_mean=t158 {pt2=root:b_features_0_1_running_mean target=features.0.1.running_mean}
          running_var=t159 {pt2=root:b_features_0_1_running_var target=features.0.1.running_var}
          params={channel=C; eps=1e-05}
    n416 {pt2=root[2] torch.ops.aten.hardtanh.default}: [t730 f32 [H=112 W=112
                                                                   C=32] {derived} ->[n10]] =
      hardtanh x=t320 {derived} <-n5 params={min_val=0; max_val=6}
    group g4 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n13 {derived}: [t328 f32 [H=112 W=112 C=32] {derived} ->[n417]] =
        batch_norm
          x=t325 {derived} <-n10
          weight=t4 {pt2=root:p_features_1_conv_0_1_weight target=features.1.conv.0.1.weight}
          bias=t5 {pt2=root:p_features_1_conv_0_1_bias target=features.1.conv.0.1.bias}
          running_mean=t161 {pt2=root:b_features_1_conv_0_1_running_mean target=features.1.conv.0.1.running_mean}
          running_var=t162 {pt2=root:b_features_1_conv_0_1_running_var target=features.1.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n417 {pt2=root[5] torch.ops.aten.hardtanh.default}: [t731 f32 [H=112 W=112
                                                                   C=32] {derived} ->[n18]] =
      hardtanh x=t328 {derived} <-n13 params={min_val=0; max_val=6}
    group g6 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n21 {derived}: [t336 f32 [H=112 W=112 C=16] {derived} ->[n25]] =
        batch_norm
          x=t333 {derived} <-n18
          weight=t7 {pt2=root:p_features_1_conv_2_weight target=features.1.conv.2.weight}
          bias=t8 {pt2=root:p_features_1_conv_2_bias target=features.1.conv.2.bias}
          running_mean=t164 {pt2=root:b_features_1_conv_2_running_mean target=features.1.conv.2.running_mean}
          running_var=t165 {pt2=root:b_features_1_conv_2_running_var target=features.1.conv.2.running_var}
          params={channel=C; eps=1e-05}
    group g8 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n28 {derived}: [t343 f32 [H=112 W=112 C=96] {derived} ->[n418]] =
        batch_norm
          x=t340 {derived} <-n25
          weight=t10 {pt2=root:p_features_2_conv_0_1_weight target=features.2.conv.0.1.weight}
          bias=t11 {pt2=root:p_features_2_conv_0_1_bias target=features.2.conv.0.1.bias}
          running_mean=t167 {pt2=root:b_features_2_conv_0_1_running_mean target=features.2.conv.0.1.running_mean}
          running_var=t168 {pt2=root:b_features_2_conv_0_1_running_var target=features.2.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n418 {pt2=root[10] torch.ops.aten.hardtanh.default}: [t732 f32 [H=112 W=112
                                                                    C=96] {derived} ->[n33]] =
      hardtanh x=t343 {derived} <-n28 params={min_val=0; max_val=6}
    group g10 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n36 {derived}: [t351 f32 [H=56 W=56 C=96] {derived} ->[n419]] =
        batch_norm
          x=t348 {derived} <-n33
          weight=t13 {pt2=root:p_features_2_conv_1_1_weight target=features.2.conv.1.1.weight}
          bias=t14 {pt2=root:p_features_2_conv_1_1_bias target=features.2.conv.1.1.bias}
          running_mean=t170 {pt2=root:b_features_2_conv_1_1_running_mean target=features.2.conv.1.1.running_mean}
          running_var=t171 {pt2=root:b_features_2_conv_1_1_running_var target=features.2.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n419 {pt2=root[13] torch.ops.aten.hardtanh.default}: [t733 f32 [H=56 W=56
                                                                    C=96] {derived} ->[n41]] =
      hardtanh x=t351 {derived} <-n36 params={min_val=0; max_val=6}
    group g12 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n44 {derived}: [t359 f32 [H=56 W=56 C=24] {derived} ->[n48, n422]] =
        batch_norm
          x=t356 {derived} <-n41
          weight=t16 {pt2=root:p_features_2_conv_3_weight target=features.2.conv.3.weight}
          bias=t17 {pt2=root:p_features_2_conv_3_bias target=features.2.conv.3.bias}
          running_mean=t173 {pt2=root:b_features_2_conv_3_running_mean target=features.2.conv.3.running_mean}
          running_var=t174 {pt2=root:b_features_2_conv_3_running_var target=features.2.conv.3.running_var}
          params={channel=C; eps=1e-05}
    group g14 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n51 {derived}: [t366 f32 [H=56 W=56 C=144] {derived} ->[n420]] =
        batch_norm
          x=t363 {derived} <-n48
          weight=t19 {pt2=root:p_features_3_conv_0_1_weight target=features.3.conv.0.1.weight}
          bias=t20 {pt2=root:p_features_3_conv_0_1_bias target=features.3.conv.0.1.bias}
          running_mean=t176 {pt2=root:b_features_3_conv_0_1_running_mean target=features.3.conv.0.1.running_mean}
          running_var=t177 {pt2=root:b_features_3_conv_0_1_running_var target=features.3.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n420 {pt2=root[18] torch.ops.aten.hardtanh.default}: [t734 f32 [H=56 W=56
                                                                    C=144] {derived} ->[n56]] =
      hardtanh x=t366 {derived} <-n51 params={min_val=0; max_val=6}
    group g16 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n59 {derived}: [t374 f32 [H=56 W=56 C=144] {derived} ->[n421]] =
        batch_norm
          x=t371 {derived} <-n56
          weight=t22 {pt2=root:p_features_3_conv_1_1_weight target=features.3.conv.1.1.weight}
          bias=t23 {pt2=root:p_features_3_conv_1_1_bias target=features.3.conv.1.1.bias}
          running_mean=t179 {pt2=root:b_features_3_conv_1_1_running_mean target=features.3.conv.1.1.running_mean}
          running_var=t180 {pt2=root:b_features_3_conv_1_1_running_var target=features.3.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n421 {pt2=root[21] torch.ops.aten.hardtanh.default}: [t735 f32 [H=56 W=56
                                                                    C=144] {derived} ->[n64]] =
      hardtanh x=t374 {derived} <-n59 params={min_val=0; max_val=6}
    group g18 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n67 {derived}: [t382 f32 [H=56 W=56 C=24] {derived} ->[n422]] =
        batch_norm
          x=t379 {derived} <-n64
          weight=t25 {pt2=root:p_features_3_conv_3_weight target=features.3.conv.3.weight}
          bias=t26 {pt2=root:p_features_3_conv_3_bias target=features.3.conv.3.bias}
          running_mean=t182 {pt2=root:b_features_3_conv_3_running_mean target=features.3.conv.3.running_mean}
          running_var=t183 {pt2=root:b_features_3_conv_3_running_var target=features.3.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n422 {pt2=root[24] torch.ops.aten.add.Tensor}: [t736 f32 [H=56 W=56 C=24] {derived} ->[n72]] =
      add a=t359 {derived} <-n44 b=t382 {derived} <-n67
    group g20 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n75 {derived}: [t390 f32 [H=56 W=56 C=144] {derived} ->[n423]] =
        batch_norm
          x=t387 {derived} <-n72
          weight=t28 {pt2=root:p_features_4_conv_0_1_weight target=features.4.conv.0.1.weight}
          bias=t29 {pt2=root:p_features_4_conv_0_1_bias target=features.4.conv.0.1.bias}
          running_mean=t185 {pt2=root:b_features_4_conv_0_1_running_mean target=features.4.conv.0.1.running_mean}
          running_var=t186 {pt2=root:b_features_4_conv_0_1_running_var target=features.4.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n423 {pt2=root[27] torch.ops.aten.hardtanh.default}: [t737 f32 [H=56 W=56
                                                                    C=144] {derived} ->[n80]] =
      hardtanh x=t390 {derived} <-n75 params={min_val=0; max_val=6}
    group g22 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n83 {derived}: [t398 f32 [H=28 W=28 C=144] {derived} ->[n424]] =
        batch_norm
          x=t395 {derived} <-n80
          weight=t31 {pt2=root:p_features_4_conv_1_1_weight target=features.4.conv.1.1.weight}
          bias=t32 {pt2=root:p_features_4_conv_1_1_bias target=features.4.conv.1.1.bias}
          running_mean=t188 {pt2=root:b_features_4_conv_1_1_running_mean target=features.4.conv.1.1.running_mean}
          running_var=t189 {pt2=root:b_features_4_conv_1_1_running_var target=features.4.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n424 {pt2=root[30] torch.ops.aten.hardtanh.default}: [t738 f32 [H=28 W=28
                                                                    C=144] {derived} ->[n88]] =
      hardtanh x=t398 {derived} <-n83 params={min_val=0; max_val=6}
    group g24 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n91 {derived}: [t406 f32 [H=28 W=28 C=32] {derived} ->[n95, n427]] =
        batch_norm
          x=t403 {derived} <-n88
          weight=t34 {pt2=root:p_features_4_conv_3_weight target=features.4.conv.3.weight}
          bias=t35 {pt2=root:p_features_4_conv_3_bias target=features.4.conv.3.bias}
          running_mean=t191 {pt2=root:b_features_4_conv_3_running_mean target=features.4.conv.3.running_mean}
          running_var=t192 {pt2=root:b_features_4_conv_3_running_var target=features.4.conv.3.running_var}
          params={channel=C; eps=1e-05}
    group g26 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n98 {derived}: [t413 f32 [H=28 W=28 C=192] {derived} ->[n425]] =
        batch_norm
          x=t410 {derived} <-n95
          weight=t37 {pt2=root:p_features_5_conv_0_1_weight target=features.5.conv.0.1.weight}
          bias=t38 {pt2=root:p_features_5_conv_0_1_bias target=features.5.conv.0.1.bias}
          running_mean=t194 {pt2=root:b_features_5_conv_0_1_running_mean target=features.5.conv.0.1.running_mean}
          running_var=t195 {pt2=root:b_features_5_conv_0_1_running_var target=features.5.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n425 {pt2=root[35] torch.ops.aten.hardtanh.default}: [t739 f32 [H=28 W=28
                                                                    C=192] {derived} ->[n103]] =
      hardtanh x=t413 {derived} <-n98 params={min_val=0; max_val=6}
    group g28 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n106 {derived}: [t421 f32 [H=28 W=28 C=192] {derived} ->[n426]] =
        batch_norm
          x=t418 {derived} <-n103
          weight=t40 {pt2=root:p_features_5_conv_1_1_weight target=features.5.conv.1.1.weight}
          bias=t41 {pt2=root:p_features_5_conv_1_1_bias target=features.5.conv.1.1.bias}
          running_mean=t197 {pt2=root:b_features_5_conv_1_1_running_mean target=features.5.conv.1.1.running_mean}
          running_var=t198 {pt2=root:b_features_5_conv_1_1_running_var target=features.5.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n426 {pt2=root[38] torch.ops.aten.hardtanh.default}: [t740 f32 [H=28 W=28
                                                                    C=192] {derived} ->[n111]] =
      hardtanh x=t421 {derived} <-n106 params={min_val=0; max_val=6}
    group g30 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n114 {derived}: [t429 f32 [H=28 W=28 C=32] {derived} ->[n427]] =
        batch_norm
          x=t426 {derived} <-n111
          weight=t43 {pt2=root:p_features_5_conv_3_weight target=features.5.conv.3.weight}
          bias=t44 {pt2=root:p_features_5_conv_3_bias target=features.5.conv.3.bias}
          running_mean=t200 {pt2=root:b_features_5_conv_3_running_mean target=features.5.conv.3.running_mean}
          running_var=t201 {pt2=root:b_features_5_conv_3_running_var target=features.5.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n427 {pt2=root[41] torch.ops.aten.add.Tensor}: [t741 f32 [H=28 W=28 C=32] {derived} ->[n119,
                                                                      n430]] =
      add a=t406 {derived} <-n91 b=t429 {derived} <-n114
    group g32 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n122 {derived}: [t437 f32 [H=28 W=28 C=192] {derived} ->[n428]] =
        batch_norm
          x=t434 {derived} <-n119
          weight=t46 {pt2=root:p_features_6_conv_0_1_weight target=features.6.conv.0.1.weight}
          bias=t47 {pt2=root:p_features_6_conv_0_1_bias target=features.6.conv.0.1.bias}
          running_mean=t203 {pt2=root:b_features_6_conv_0_1_running_mean target=features.6.conv.0.1.running_mean}
          running_var=t204 {pt2=root:b_features_6_conv_0_1_running_var target=features.6.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n428 {pt2=root[44] torch.ops.aten.hardtanh.default}: [t742 f32 [H=28 W=28
                                                                    C=192] {derived} ->[n127]] =
      hardtanh x=t437 {derived} <-n122 params={min_val=0; max_val=6}
    group g34 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n130 {derived}: [t445 f32 [H=28 W=28 C=192] {derived} ->[n429]] =
        batch_norm
          x=t442 {derived} <-n127
          weight=t49 {pt2=root:p_features_6_conv_1_1_weight target=features.6.conv.1.1.weight}
          bias=t50 {pt2=root:p_features_6_conv_1_1_bias target=features.6.conv.1.1.bias}
          running_mean=t206 {pt2=root:b_features_6_conv_1_1_running_mean target=features.6.conv.1.1.running_mean}
          running_var=t207 {pt2=root:b_features_6_conv_1_1_running_var target=features.6.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n429 {pt2=root[47] torch.ops.aten.hardtanh.default}: [t743 f32 [H=28 W=28
                                                                    C=192] {derived} ->[n135]] =
      hardtanh x=t445 {derived} <-n130 params={min_val=0; max_val=6}
    group g36 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n138 {derived}: [t453 f32 [H=28 W=28 C=32] {derived} ->[n430]] =
        batch_norm
          x=t450 {derived} <-n135
          weight=t52 {pt2=root:p_features_6_conv_3_weight target=features.6.conv.3.weight}
          bias=t53 {pt2=root:p_features_6_conv_3_bias target=features.6.conv.3.bias}
          running_mean=t209 {pt2=root:b_features_6_conv_3_running_mean target=features.6.conv.3.running_mean}
          running_var=t210 {pt2=root:b_features_6_conv_3_running_var target=features.6.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n430 {pt2=root[50] torch.ops.aten.add.Tensor}: [t744 f32 [H=28 W=28 C=32] {derived} ->[n143]] =
      add a=t741 {derived} <-n427 b=t453 {derived} <-n138
    group g38 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n146 {derived}: [t461 f32 [H=28 W=28 C=192] {derived} ->[n431]] =
        batch_norm
          x=t458 {derived} <-n143
          weight=t55 {pt2=root:p_features_7_conv_0_1_weight target=features.7.conv.0.1.weight}
          bias=t56 {pt2=root:p_features_7_conv_0_1_bias target=features.7.conv.0.1.bias}
          running_mean=t212 {pt2=root:b_features_7_conv_0_1_running_mean target=features.7.conv.0.1.running_mean}
          running_var=t213 {pt2=root:b_features_7_conv_0_1_running_var target=features.7.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n431 {pt2=root[53] torch.ops.aten.hardtanh.default}: [t745 f32 [H=28 W=28
                                                                    C=192] {derived} ->[n151]] =
      hardtanh x=t461 {derived} <-n146 params={min_val=0; max_val=6}
    group g40 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n154 {derived}: [t469 f32 [H=14 W=14 C=192] {derived} ->[n432]] =
        batch_norm
          x=t466 {derived} <-n151
          weight=t58 {pt2=root:p_features_7_conv_1_1_weight target=features.7.conv.1.1.weight}
          bias=t59 {pt2=root:p_features_7_conv_1_1_bias target=features.7.conv.1.1.bias}
          running_mean=t215 {pt2=root:b_features_7_conv_1_1_running_mean target=features.7.conv.1.1.running_mean}
          running_var=t216 {pt2=root:b_features_7_conv_1_1_running_var target=features.7.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n432 {pt2=root[56] torch.ops.aten.hardtanh.default}: [t746 f32 [H=14 W=14
                                                                    C=192] {derived} ->[n159]] =
      hardtanh x=t469 {derived} <-n154 params={min_val=0; max_val=6}
    group g42 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n162 {derived}: [t477 f32 [H=14 W=14 C=64] {derived} ->[n166, n435]] =
        batch_norm
          x=t474 {derived} <-n159
          weight=t61 {pt2=root:p_features_7_conv_3_weight target=features.7.conv.3.weight}
          bias=t62 {pt2=root:p_features_7_conv_3_bias target=features.7.conv.3.bias}
          running_mean=t218 {pt2=root:b_features_7_conv_3_running_mean target=features.7.conv.3.running_mean}
          running_var=t219 {pt2=root:b_features_7_conv_3_running_var target=features.7.conv.3.running_var}
          params={channel=C; eps=1e-05}
    group g44 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n169 {derived}: [t484 f32 [H=14 W=14 C=384] {derived} ->[n433]] =
        batch_norm
          x=t481 {derived} <-n166
          weight=t64 {pt2=root:p_features_8_conv_0_1_weight target=features.8.conv.0.1.weight}
          bias=t65 {pt2=root:p_features_8_conv_0_1_bias target=features.8.conv.0.1.bias}
          running_mean=t221 {pt2=root:b_features_8_conv_0_1_running_mean target=features.8.conv.0.1.running_mean}
          running_var=t222 {pt2=root:b_features_8_conv_0_1_running_var target=features.8.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n433 {pt2=root[61] torch.ops.aten.hardtanh.default}: [t747 f32 [H=14 W=14
                                                                    C=384] {derived} ->[n174]] =
      hardtanh x=t484 {derived} <-n169 params={min_val=0; max_val=6}
    group g46 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n177 {derived}: [t492 f32 [H=14 W=14 C=384] {derived} ->[n434]] =
        batch_norm
          x=t489 {derived} <-n174
          weight=t67 {pt2=root:p_features_8_conv_1_1_weight target=features.8.conv.1.1.weight}
          bias=t68 {pt2=root:p_features_8_conv_1_1_bias target=features.8.conv.1.1.bias}
          running_mean=t224 {pt2=root:b_features_8_conv_1_1_running_mean target=features.8.conv.1.1.running_mean}
          running_var=t225 {pt2=root:b_features_8_conv_1_1_running_var target=features.8.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n434 {pt2=root[64] torch.ops.aten.hardtanh.default}: [t748 f32 [H=14 W=14
                                                                    C=384] {derived} ->[n182]] =
      hardtanh x=t492 {derived} <-n177 params={min_val=0; max_val=6}
    group g48 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n185 {derived}: [t500 f32 [H=14 W=14 C=64] {derived} ->[n435]] =
        batch_norm
          x=t497 {derived} <-n182
          weight=t70 {pt2=root:p_features_8_conv_3_weight target=features.8.conv.3.weight}
          bias=t71 {pt2=root:p_features_8_conv_3_bias target=features.8.conv.3.bias}
          running_mean=t227 {pt2=root:b_features_8_conv_3_running_mean target=features.8.conv.3.running_mean}
          running_var=t228 {pt2=root:b_features_8_conv_3_running_var target=features.8.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n435 {pt2=root[67] torch.ops.aten.add.Tensor}: [t749 f32 [H=14 W=14 C=64] {derived} ->[n190,
                                                                      n438]] =
      add a=t477 {derived} <-n162 b=t500 {derived} <-n185
    group g50 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n193 {derived}: [t508 f32 [H=14 W=14 C=384] {derived} ->[n436]] =
        batch_norm
          x=t505 {derived} <-n190
          weight=t73 {pt2=root:p_features_9_conv_0_1_weight target=features.9.conv.0.1.weight}
          bias=t74 {pt2=root:p_features_9_conv_0_1_bias target=features.9.conv.0.1.bias}
          running_mean=t230 {pt2=root:b_features_9_conv_0_1_running_mean target=features.9.conv.0.1.running_mean}
          running_var=t231 {pt2=root:b_features_9_conv_0_1_running_var target=features.9.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n436 {pt2=root[70] torch.ops.aten.hardtanh.default}: [t750 f32 [H=14 W=14
                                                                    C=384] {derived} ->[n198]] =
      hardtanh x=t508 {derived} <-n193 params={min_val=0; max_val=6}
    group g52 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n201 {derived}: [t516 f32 [H=14 W=14 C=384] {derived} ->[n437]] =
        batch_norm
          x=t513 {derived} <-n198
          weight=t76 {pt2=root:p_features_9_conv_1_1_weight target=features.9.conv.1.1.weight}
          bias=t77 {pt2=root:p_features_9_conv_1_1_bias target=features.9.conv.1.1.bias}
          running_mean=t233 {pt2=root:b_features_9_conv_1_1_running_mean target=features.9.conv.1.1.running_mean}
          running_var=t234 {pt2=root:b_features_9_conv_1_1_running_var target=features.9.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n437 {pt2=root[73] torch.ops.aten.hardtanh.default}: [t751 f32 [H=14 W=14
                                                                    C=384] {derived} ->[n206]] =
      hardtanh x=t516 {derived} <-n201 params={min_val=0; max_val=6}
    group g54 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n209 {derived}: [t524 f32 [H=14 W=14 C=64] {derived} ->[n438]] =
        batch_norm
          x=t521 {derived} <-n206
          weight=t79 {pt2=root:p_features_9_conv_3_weight target=features.9.conv.3.weight}
          bias=t80 {pt2=root:p_features_9_conv_3_bias target=features.9.conv.3.bias}
          running_mean=t236 {pt2=root:b_features_9_conv_3_running_mean target=features.9.conv.3.running_mean}
          running_var=t237 {pt2=root:b_features_9_conv_3_running_var target=features.9.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n438 {pt2=root[76] torch.ops.aten.add.Tensor}: [t752 f32 [H=14 W=14 C=64] {derived} ->[n214,
                                                                      n441]] =
      add a=t749 {derived} <-n435 b=t524 {derived} <-n209
    group g56 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n217 {derived}: [t532 f32 [H=14 W=14 C=384] {derived} ->[n439]] =
        batch_norm
          x=t529 {derived} <-n214
          weight=t82 {pt2=root:p_features_10_conv_0_1_weight target=features.10.conv.0.1.weight}
          bias=t83 {pt2=root:p_features_10_conv_0_1_bias target=features.10.conv.0.1.bias}
          running_mean=t239 {pt2=root:b_features_10_conv_0_1_running_mean target=features.10.conv.0.1.running_mean}
          running_var=t240 {pt2=root:b_features_10_conv_0_1_running_var target=features.10.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n439 {pt2=root[79] torch.ops.aten.hardtanh.default}: [t753 f32 [H=14 W=14
                                                                    C=384] {derived} ->[n222]] =
      hardtanh x=t532 {derived} <-n217 params={min_val=0; max_val=6}
    group g58 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n225 {derived}: [t540 f32 [H=14 W=14 C=384] {derived} ->[n440]] =
        batch_norm
          x=t537 {derived} <-n222
          weight=t85 {pt2=root:p_features_10_conv_1_1_weight target=features.10.conv.1.1.weight}
          bias=t86 {pt2=root:p_features_10_conv_1_1_bias target=features.10.conv.1.1.bias}
          running_mean=t242 {pt2=root:b_features_10_conv_1_1_running_mean target=features.10.conv.1.1.running_mean}
          running_var=t243 {pt2=root:b_features_10_conv_1_1_running_var target=features.10.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n440 {pt2=root[82] torch.ops.aten.hardtanh.default}: [t754 f32 [H=14 W=14
                                                                    C=384] {derived} ->[n230]] =
      hardtanh x=t540 {derived} <-n225 params={min_val=0; max_val=6}
    group g60 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n233 {derived}: [t548 f32 [H=14 W=14 C=64] {derived} ->[n441]] =
        batch_norm
          x=t545 {derived} <-n230
          weight=t88 {pt2=root:p_features_10_conv_3_weight target=features.10.conv.3.weight}
          bias=t89 {pt2=root:p_features_10_conv_3_bias target=features.10.conv.3.bias}
          running_mean=t245 {pt2=root:b_features_10_conv_3_running_mean target=features.10.conv.3.running_mean}
          running_var=t246 {pt2=root:b_features_10_conv_3_running_var target=features.10.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n441 {pt2=root[85] torch.ops.aten.add.Tensor}: [t755 f32 [H=14 W=14 C=64] {derived} ->[n238]] =
      add a=t752 {derived} <-n438 b=t548 {derived} <-n233
    group g62 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n241 {derived}: [t556 f32 [H=14 W=14 C=384] {derived} ->[n442]] =
        batch_norm
          x=t553 {derived} <-n238
          weight=t91 {pt2=root:p_features_11_conv_0_1_weight target=features.11.conv.0.1.weight}
          bias=t92 {pt2=root:p_features_11_conv_0_1_bias target=features.11.conv.0.1.bias}
          running_mean=t248 {pt2=root:b_features_11_conv_0_1_running_mean target=features.11.conv.0.1.running_mean}
          running_var=t249 {pt2=root:b_features_11_conv_0_1_running_var target=features.11.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n442 {pt2=root[88] torch.ops.aten.hardtanh.default}: [t756 f32 [H=14 W=14
                                                                    C=384] {derived} ->[n246]] =
      hardtanh x=t556 {derived} <-n241 params={min_val=0; max_val=6}
    group g64 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n249 {derived}: [t564 f32 [H=14 W=14 C=384] {derived} ->[n443]] =
        batch_norm
          x=t561 {derived} <-n246
          weight=t94 {pt2=root:p_features_11_conv_1_1_weight target=features.11.conv.1.1.weight}
          bias=t95 {pt2=root:p_features_11_conv_1_1_bias target=features.11.conv.1.1.bias}
          running_mean=t251 {pt2=root:b_features_11_conv_1_1_running_mean target=features.11.conv.1.1.running_mean}
          running_var=t252 {pt2=root:b_features_11_conv_1_1_running_var target=features.11.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n443 {pt2=root[91] torch.ops.aten.hardtanh.default}: [t757 f32 [H=14 W=14
                                                                    C=384] {derived} ->[n254]] =
      hardtanh x=t564 {derived} <-n249 params={min_val=0; max_val=6}
    group g66 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n257 {derived}: [t572 f32 [H=14 W=14 C=96] {derived} ->[n261, n446]] =
        batch_norm
          x=t569 {derived} <-n254
          weight=t97 {pt2=root:p_features_11_conv_3_weight target=features.11.conv.3.weight}
          bias=t98 {pt2=root:p_features_11_conv_3_bias target=features.11.conv.3.bias}
          running_mean=t254 {pt2=root:b_features_11_conv_3_running_mean target=features.11.conv.3.running_mean}
          running_var=t255 {pt2=root:b_features_11_conv_3_running_var target=features.11.conv.3.running_var}
          params={channel=C; eps=1e-05}
    group g68 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n264 {derived}: [t579 f32 [H=14 W=14 C=576] {derived} ->[n444]] =
        batch_norm
          x=t576 {derived} <-n261
          weight=t100 {pt2=root:p_features_12_conv_0_1_weight target=features.12.conv.0.1.weight}
          bias=t101 {pt2=root:p_features_12_conv_0_1_bias target=features.12.conv.0.1.bias}
          running_mean=t257 {pt2=root:b_features_12_conv_0_1_running_mean target=features.12.conv.0.1.running_mean}
          running_var=t258 {pt2=root:b_features_12_conv_0_1_running_var target=features.12.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n444 {pt2=root[96] torch.ops.aten.hardtanh.default}: [t758 f32 [H=14 W=14
                                                                    C=576] {derived} ->[n269]] =
      hardtanh x=t579 {derived} <-n264 params={min_val=0; max_val=6}
    group g70 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n272 {derived}: [t587 f32 [H=14 W=14 C=576] {derived} ->[n445]] =
        batch_norm
          x=t584 {derived} <-n269
          weight=t103 {pt2=root:p_features_12_conv_1_1_weight target=features.12.conv.1.1.weight}
          bias=t104 {pt2=root:p_features_12_conv_1_1_bias target=features.12.conv.1.1.bias}
          running_mean=t260 {pt2=root:b_features_12_conv_1_1_running_mean target=features.12.conv.1.1.running_mean}
          running_var=t261 {pt2=root:b_features_12_conv_1_1_running_var target=features.12.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n445 {pt2=root[99] torch.ops.aten.hardtanh.default}: [t759 f32 [H=14 W=14
                                                                    C=576] {derived} ->[n277]] =
      hardtanh x=t587 {derived} <-n272 params={min_val=0; max_val=6}
    group g72 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n280 {derived}: [t595 f32 [H=14 W=14 C=96] {derived} ->[n446]] =
        batch_norm
          x=t592 {derived} <-n277
          weight=t106 {pt2=root:p_features_12_conv_3_weight target=features.12.conv.3.weight}
          bias=t107 {pt2=root:p_features_12_conv_3_bias target=features.12.conv.3.bias}
          running_mean=t263 {pt2=root:b_features_12_conv_3_running_mean target=features.12.conv.3.running_mean}
          running_var=t264 {pt2=root:b_features_12_conv_3_running_var target=features.12.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n446 {pt2=root[102] torch.ops.aten.add.Tensor}: [t760 f32 [H=14 W=14 C=96] {derived} ->[n285,
                                                                      n449]] =
      add a=t572 {derived} <-n257 b=t595 {derived} <-n280
    group g74 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n288 {derived}: [t603 f32 [H=14 W=14 C=576] {derived} ->[n447]] =
        batch_norm
          x=t600 {derived} <-n285
          weight=t109 {pt2=root:p_features_13_conv_0_1_weight target=features.13.conv.0.1.weight}
          bias=t110 {pt2=root:p_features_13_conv_0_1_bias target=features.13.conv.0.1.bias}
          running_mean=t266 {pt2=root:b_features_13_conv_0_1_running_mean target=features.13.conv.0.1.running_mean}
          running_var=t267 {pt2=root:b_features_13_conv_0_1_running_var target=features.13.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n447 {pt2=root[105] torch.ops.aten.hardtanh.default}: [t761 f32 [H=14 W=14
                                                                     C=576] {derived} ->[n293]] =
      hardtanh x=t603 {derived} <-n288 params={min_val=0; max_val=6}
    group g76 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n296 {derived}: [t611 f32 [H=14 W=14 C=576] {derived} ->[n448]] =
        batch_norm
          x=t608 {derived} <-n293
          weight=t112 {pt2=root:p_features_13_conv_1_1_weight target=features.13.conv.1.1.weight}
          bias=t113 {pt2=root:p_features_13_conv_1_1_bias target=features.13.conv.1.1.bias}
          running_mean=t269 {pt2=root:b_features_13_conv_1_1_running_mean target=features.13.conv.1.1.running_mean}
          running_var=t270 {pt2=root:b_features_13_conv_1_1_running_var target=features.13.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n448 {pt2=root[108] torch.ops.aten.hardtanh.default}: [t762 f32 [H=14 W=14
                                                                     C=576] {derived} ->[n301]] =
      hardtanh x=t611 {derived} <-n296 params={min_val=0; max_val=6}
    group g78 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n304 {derived}: [t619 f32 [H=14 W=14 C=96] {derived} ->[n449]] =
        batch_norm
          x=t616 {derived} <-n301
          weight=t115 {pt2=root:p_features_13_conv_3_weight target=features.13.conv.3.weight}
          bias=t116 {pt2=root:p_features_13_conv_3_bias target=features.13.conv.3.bias}
          running_mean=t272 {pt2=root:b_features_13_conv_3_running_mean target=features.13.conv.3.running_mean}
          running_var=t273 {pt2=root:b_features_13_conv_3_running_var target=features.13.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n449 {pt2=root[111] torch.ops.aten.add.Tensor}: [t763 f32 [H=14 W=14 C=96] {derived} ->[n309]] =
      add a=t760 {derived} <-n446 b=t619 {derived} <-n304
    group g80 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n312 {derived}: [t627 f32 [H=14 W=14 C=576] {derived} ->[n450]] =
        batch_norm
          x=t624 {derived} <-n309
          weight=t118 {pt2=root:p_features_14_conv_0_1_weight target=features.14.conv.0.1.weight}
          bias=t119 {pt2=root:p_features_14_conv_0_1_bias target=features.14.conv.0.1.bias}
          running_mean=t275 {pt2=root:b_features_14_conv_0_1_running_mean target=features.14.conv.0.1.running_mean}
          running_var=t276 {pt2=root:b_features_14_conv_0_1_running_var target=features.14.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n450 {pt2=root[114] torch.ops.aten.hardtanh.default}: [t764 f32 [H=14 W=14
                                                                     C=576] {derived} ->[n317]] =
      hardtanh x=t627 {derived} <-n312 params={min_val=0; max_val=6}
    group g82 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n320 {derived}: [t635 f32 [H=7 W=7 C=576] {derived} ->[n451]] =
        batch_norm
          x=t632 {derived} <-n317
          weight=t121 {pt2=root:p_features_14_conv_1_1_weight target=features.14.conv.1.1.weight}
          bias=t122 {pt2=root:p_features_14_conv_1_1_bias target=features.14.conv.1.1.bias}
          running_mean=t278 {pt2=root:b_features_14_conv_1_1_running_mean target=features.14.conv.1.1.running_mean}
          running_var=t279 {pt2=root:b_features_14_conv_1_1_running_var target=features.14.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n451 {pt2=root[117] torch.ops.aten.hardtanh.default}: [t765 f32 [H=7 W=7
                                                                     C=576] {derived} ->[n325]] =
      hardtanh x=t635 {derived} <-n320 params={min_val=0; max_val=6}
    group g84 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n328 {derived}: [t643 f32 [H=7 W=7 C=160] {derived} ->[n332, n454]] =
        batch_norm
          x=t640 {derived} <-n325
          weight=t124 {pt2=root:p_features_14_conv_3_weight target=features.14.conv.3.weight}
          bias=t125 {pt2=root:p_features_14_conv_3_bias target=features.14.conv.3.bias}
          running_mean=t281 {pt2=root:b_features_14_conv_3_running_mean target=features.14.conv.3.running_mean}
          running_var=t282 {pt2=root:b_features_14_conv_3_running_var target=features.14.conv.3.running_var}
          params={channel=C; eps=1e-05}
    group g86 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n335 {derived}: [t650 f32 [H=7 W=7 C=960] {derived} ->[n452]] =
        batch_norm
          x=t647 {derived} <-n332
          weight=t127 {pt2=root:p_features_15_conv_0_1_weight target=features.15.conv.0.1.weight}
          bias=t128 {pt2=root:p_features_15_conv_0_1_bias target=features.15.conv.0.1.bias}
          running_mean=t284 {pt2=root:b_features_15_conv_0_1_running_mean target=features.15.conv.0.1.running_mean}
          running_var=t285 {pt2=root:b_features_15_conv_0_1_running_var target=features.15.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n452 {pt2=root[122] torch.ops.aten.hardtanh.default}: [t766 f32 [H=7 W=7
                                                                     C=960] {derived} ->[n340]] =
      hardtanh x=t650 {derived} <-n335 params={min_val=0; max_val=6}
    group g88 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n343 {derived}: [t658 f32 [H=7 W=7 C=960] {derived} ->[n453]] =
        batch_norm
          x=t655 {derived} <-n340
          weight=t130 {pt2=root:p_features_15_conv_1_1_weight target=features.15.conv.1.1.weight}
          bias=t131 {pt2=root:p_features_15_conv_1_1_bias target=features.15.conv.1.1.bias}
          running_mean=t287 {pt2=root:b_features_15_conv_1_1_running_mean target=features.15.conv.1.1.running_mean}
          running_var=t288 {pt2=root:b_features_15_conv_1_1_running_var target=features.15.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n453 {pt2=root[125] torch.ops.aten.hardtanh.default}: [t767 f32 [H=7 W=7
                                                                     C=960] {derived} ->[n348]] =
      hardtanh x=t658 {derived} <-n343 params={min_val=0; max_val=6}
    group g90 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n351 {derived}: [t666 f32 [H=7 W=7 C=160] {derived} ->[n454]] =
        batch_norm
          x=t663 {derived} <-n348
          weight=t133 {pt2=root:p_features_15_conv_3_weight target=features.15.conv.3.weight}
          bias=t134 {pt2=root:p_features_15_conv_3_bias target=features.15.conv.3.bias}
          running_mean=t290 {pt2=root:b_features_15_conv_3_running_mean target=features.15.conv.3.running_mean}
          running_var=t291 {pt2=root:b_features_15_conv_3_running_var target=features.15.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n454 {pt2=root[128] torch.ops.aten.add.Tensor}: [t768 f32 [H=7 W=7 C=160] {derived} ->[n356,
                                                                      n457]] =
      add a=t643 {derived} <-n328 b=t666 {derived} <-n351
    group g92 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n359 {derived}: [t674 f32 [H=7 W=7 C=960] {derived} ->[n455]] =
        batch_norm
          x=t671 {derived} <-n356
          weight=t136 {pt2=root:p_features_16_conv_0_1_weight target=features.16.conv.0.1.weight}
          bias=t137 {pt2=root:p_features_16_conv_0_1_bias target=features.16.conv.0.1.bias}
          running_mean=t293 {pt2=root:b_features_16_conv_0_1_running_mean target=features.16.conv.0.1.running_mean}
          running_var=t294 {pt2=root:b_features_16_conv_0_1_running_var target=features.16.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n455 {pt2=root[131] torch.ops.aten.hardtanh.default}: [t769 f32 [H=7 W=7
                                                                     C=960] {derived} ->[n364]] =
      hardtanh x=t674 {derived} <-n359 params={min_val=0; max_val=6}
    group g94 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n367 {derived}: [t682 f32 [H=7 W=7 C=960] {derived} ->[n456]] =
        batch_norm
          x=t679 {derived} <-n364
          weight=t139 {pt2=root:p_features_16_conv_1_1_weight target=features.16.conv.1.1.weight}
          bias=t140 {pt2=root:p_features_16_conv_1_1_bias target=features.16.conv.1.1.bias}
          running_mean=t296 {pt2=root:b_features_16_conv_1_1_running_mean target=features.16.conv.1.1.running_mean}
          running_var=t297 {pt2=root:b_features_16_conv_1_1_running_var target=features.16.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n456 {pt2=root[134] torch.ops.aten.hardtanh.default}: [t770 f32 [H=7 W=7
                                                                     C=960] {derived} ->[n372]] =
      hardtanh x=t682 {derived} <-n367 params={min_val=0; max_val=6}
    group g96 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n375 {derived}: [t690 f32 [H=7 W=7 C=160] {derived} ->[n457]] =
        batch_norm
          x=t687 {derived} <-n372
          weight=t142 {pt2=root:p_features_16_conv_3_weight target=features.16.conv.3.weight}
          bias=t143 {pt2=root:p_features_16_conv_3_bias target=features.16.conv.3.bias}
          running_mean=t299 {pt2=root:b_features_16_conv_3_running_mean target=features.16.conv.3.running_mean}
          running_var=t300 {pt2=root:b_features_16_conv_3_running_var target=features.16.conv.3.running_var}
          params={channel=C; eps=1e-05}
    n457 {pt2=root[137] torch.ops.aten.add.Tensor}: [t771 f32 [H=7 W=7 C=160] {derived} ->[n380]] =
      add a=t768 {derived} <-n454 b=t690 {derived} <-n375
    group g98 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n383 {derived}: [t698 f32 [H=7 W=7 C=960] {derived} ->[n458]] =
        batch_norm
          x=t695 {derived} <-n380
          weight=t145 {pt2=root:p_features_17_conv_0_1_weight target=features.17.conv.0.1.weight}
          bias=t146 {pt2=root:p_features_17_conv_0_1_bias target=features.17.conv.0.1.bias}
          running_mean=t302 {pt2=root:b_features_17_conv_0_1_running_mean target=features.17.conv.0.1.running_mean}
          running_var=t303 {pt2=root:b_features_17_conv_0_1_running_var target=features.17.conv.0.1.running_var}
          params={channel=C; eps=1e-05}
    n458 {pt2=root[140] torch.ops.aten.hardtanh.default}: [t772 f32 [H=7 W=7
                                                                     C=960] {derived} ->[n388]] =
      hardtanh x=t698 {derived} <-n383 params={min_val=0; max_val=6}
    group g100 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n391 {derived}: [t706 f32 [H=7 W=7 C=960] {derived} ->[n459]] =
        batch_norm
          x=t703 {derived} <-n388
          weight=t148 {pt2=root:p_features_17_conv_1_1_weight target=features.17.conv.1.1.weight}
          bias=t149 {pt2=root:p_features_17_conv_1_1_bias target=features.17.conv.1.1.bias}
          running_mean=t305 {pt2=root:b_features_17_conv_1_1_running_mean target=features.17.conv.1.1.running_mean}
          running_var=t306 {pt2=root:b_features_17_conv_1_1_running_var target=features.17.conv.1.1.running_var}
          params={channel=C; eps=1e-05}
    n459 {pt2=root[143] torch.ops.aten.hardtanh.default}: [t773 f32 [H=7 W=7
                                                                     C=960] {derived} ->[n396]] =
      hardtanh x=t706 {derived} <-n391 params={min_val=0; max_val=6}
    group g102 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n399 {derived}: [t714 f32 [H=7 W=7 C=320] {derived} ->[n403]] =
        batch_norm
          x=t711 {derived} <-n396
          weight=t151 {pt2=root:p_features_17_conv_3_weight target=features.17.conv.3.weight}
          bias=t152 {pt2=root:p_features_17_conv_3_bias target=features.17.conv.3.bias}
          running_mean=t308 {pt2=root:b_features_17_conv_3_running_mean target=features.17.conv.3.running_mean}
          running_var=t309 {pt2=root:b_features_17_conv_3_running_var target=features.17.conv.3.running_var}
          params={channel=C; eps=1e-05}
    group g104 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n406 {derived}: [t721 f32 [H=7 W=7 C=1280] {derived} ->[n460]] =
        batch_norm
          x=t718 {derived} <-n403
          weight=t154 {pt2=root:p_features_18_1_weight target=features.18.1.weight}
          bias=t155 {pt2=root:p_features_18_1_bias target=features.18.1.bias}
          running_mean=t311 {pt2=root:b_features_18_1_running_mean target=features.18.1.running_mean}
          running_var=t312 {pt2=root:b_features_18_1_running_var target=features.18.1.running_var}
          params={channel=C; eps=1e-05}
    n460 {pt2=root[148] torch.ops.aten.hardtanh.default}: [t774 f32 [H=7 W=7
                                                                     C=1280] {derived} ->[n461]] =
      hardtanh x=t721 {derived} <-n406 params={min_val=0; max_val=6}
    n461 {pt2=root[149] torch.ops.aten.mean.dim}: [t775 f32 [C=1280] {derived} ->[n462]] =
      mean x=t774 {derived} <-n460 params={dims=[W, H]; keepdim=true}
    n462 {pt2=root[151] torch.ops.aten.clone.default}: [t776 f32 [C=1280] {pt2=root:clone} ->[n414]] =
      clone x=t775 {derived} <-n461
    group g105 torch.ops.aten.addmm.default:
      n414 {pt2=root[153] torch.ops.aten.addmm.default}: [t729 f32 [C=1000] {pt2=root:addmm}] =
        linear
          x=t776 {pt2=root:clone} <-n462
          weight=t728 {derived} <-n415
          bias=t157 {pt2=root:p_classifier_1_bias target=classifier.1.bias}
          params={in_features=1280}
  outputs: [t729 f32 [C=1000] {pt2=root:addmm} <-n414]
