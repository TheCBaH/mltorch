Symbolically verify ResNet-18's structural rewrite: every pass's mapping is
checked against the state it came from, WITHOUT payloads for the graph inputs,
so a proved cluster is a statement about every input rather than about the one
tensor a numeric run happens to use. See .ai/native_transform_verify.md. Gated
on PT2_DATA; run with `make pt2.runtest` after `make pt2.download-cram`.

This is the companion to native_transform_cram.t, which pins the resulting
graph. Here the graph is dropped and only the verification is kept — the two
would otherwise duplicate a 300-line dump.

Results are per pass and per group, because a flat count over ~1600 clusters
says nothing about which part of the model was covered. The groups are the PT2
call sites the importer recorded, so they name what a reader recognises.

Read `unproved (too large)` as "not looked at": a whole activation tensor is
refused by the coordinate budget before any expansion, which is what stops the
verifier being pointed at a real model and hanging. The clusters that DO get
checked here are the layout-shaped ones the permute passes actually rewrite,
which is exactly where these passes could be wrong.

Every tensor and node carries its own claim inline, so the graph shows WHICH
nodes were verified rather than only how many. `verify=` on a node is the
weakest verdict over its outputs, since a node is only as verified as its
least-verified result. `origins=n` marks an edge that several origin edges
collapsed into — that is a node more than one pass rewrote — and `origins=0` an
edge a pass created outright.

The claims are the composed origin-to-final ones, not per-pass: those speak in
intermediate id spaces that no longer exist here, whereas the composed map's
destination ids ARE this graph's. Reading a node therefore gives what the whole
pipeline established about it, not a pile of intermediate steps.

The group tree the dump already prints is what turns this into per-group
results: clusters are attributed to those same groups in the summaries above,
so a group reads the same way in both.

  $ ../bin/native_graph.exe transform --verify-symbolic quick \
  >   --pt2 "$PT2_DATA/resnet18/resnet18.pt2"
  nodes: 174 -> 91
  constants: 102, of which 0 folded
  symbolic verification: reshape_to_permute
    (root)
         83  proved (structural) [sampled 4]
         48  unproved (too large)
         20  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         60  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
          1  unproved (too large)
    torch.ops.aten.convolution.default
         80  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: chain_permute
    (root)
         83  proved (structural) [sampled 4]
         68  unproved (too large)
         21  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         40  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: trim_permute
    (root)
         83  proved (structural) [sampled 4]
         48  unproved (too large)
    torch.ops.aten._native_batch_norm_legit_no_training.default
         40  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: sink_permute
    (root)
         83  proved (structural) [sampled 4]
         48  unproved (too large)
         27  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         25  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: sink_permute
    (root)
         83  proved (structural) [sampled 4]
         57  unproved (too large)
          6  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         25  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: reuse_permute
    (root)
         83  proved (structural) [sampled 4]
         60  unproved (too large)
         10  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: sink_permute
    (root)
         83  proved (structural) [sampled 4]
         60  unproved (too large)
         10  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: bypass_permute
    (root)
         83  proved (structural) [sampled 4]
         49  unproved (too large)
         17  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         41  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          2  unproved (too large)
  symbolic verification: sink_permute_mean
    (root)
         83  proved (structural) [sampled 4]
         48  unproved (too large)
          2  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         41  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          2  unproved (too large)
  symbolic verification: bypass_permute
    (root)
         81  proved (structural) [sampled 4]
          1  unproved (over max_rounds) [sampled 4]
         48  unproved (too large)
          1  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  unproved (over max_nodes) [sampled 4]
    torch.ops.aten.convolution.default
         41  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          2  unproved (too large)
  symbolic verification: total
      837  proved (structural) [sampled 4]
        1  unproved (over max_nodes) [sampled 4]
        1  unproved (over max_rounds) [sampled 4]
     1422  unproved (too large)
      114  vacuous
  graph
  inputs:
    [t0 f32 [D=64 H=3 W=7 C=7] {pt2=root:p_conv1_weight target=conv1.weight verify=unproved (too large)} ->[n1] constant,
     t1 f32 [C=64] {pt2=root:p_bn1_weight target=bn1.weight verify=proved (structural) [sampled 4]} ->[n5] constant,
     t2 f32 [C=64] {pt2=root:p_bn1_bias target=bn1.bias verify=proved (structural) [sampled 4]} ->[n5] constant,
     t3 f32 [D=64 H=64 W=3 C=3] {pt2=root:p_layer1_0_conv1_weight target=layer1.0.conv1.weight verify=unproved (too large)} ->[n13] constant,
     t4 f32 [C=64] {pt2=root:p_layer1_0_bn1_weight target=layer1.0.bn1.weight verify=proved (structural) [sampled 4]} ->[n17] constant,
     t5 f32 [C=64] {pt2=root:p_layer1_0_bn1_bias target=layer1.0.bn1.bias verify=proved (structural) [sampled 4]} ->[n17] constant,
     t6 f32 [D=64 H=64 W=3 C=3] {pt2=root:p_layer1_0_conv2_weight target=layer1.0.conv2.weight verify=unproved (too large)} ->[n21] constant,
     t7 f32 [C=64] {pt2=root:p_layer1_0_bn2_weight target=layer1.0.bn2.weight verify=proved (structural) [sampled 4]} ->[n25] constant,
     t8 f32 [C=64] {pt2=root:p_layer1_0_bn2_bias target=layer1.0.bn2.bias verify=proved (structural) [sampled 4]} ->[n25] constant,
     t9 f32 [D=64 H=64 W=3 C=3] {pt2=root:p_layer1_1_conv1_weight target=layer1.1.conv1.weight verify=unproved (too large)} ->[n30] constant,
     t10 f32 [C=64] {pt2=root:p_layer1_1_bn1_weight target=layer1.1.bn1.weight verify=proved (structural) [sampled 4]} ->[n34] constant,
     t11 f32 [C=64] {pt2=root:p_layer1_1_bn1_bias target=layer1.1.bn1.bias verify=proved (structural) [sampled 4]} ->[n34] constant,
     t12 f32 [D=64 H=64 W=3 C=3] {pt2=root:p_layer1_1_conv2_weight target=layer1.1.conv2.weight verify=unproved (too large)} ->[n38] constant,
     t13 f32 [C=64] {pt2=root:p_layer1_1_bn2_weight target=layer1.1.bn2.weight verify=proved (structural) [sampled 4]} ->[n42] constant,
     t14 f32 [C=64] {pt2=root:p_layer1_1_bn2_bias target=layer1.1.bn2.bias verify=proved (structural) [sampled 4]} ->[n42] constant,
     t15 f32 [D=128 H=64 W=3 C=3] {pt2=root:p_layer2_0_conv1_weight target=layer2.0.conv1.weight verify=unproved (too large)} ->[n47] constant,
     t16 f32 [C=128] {pt2=root:p_layer2_0_bn1_weight target=layer2.0.bn1.weight verify=proved (structural) [sampled 4]} ->[n51] constant,
     t17 f32 [C=128] {pt2=root:p_layer2_0_bn1_bias target=layer2.0.bn1.bias verify=proved (structural) [sampled 4]} ->[n51] constant,
     t18 f32 [D=128 H=128 W=3 C=3] {pt2=root:p_layer2_0_conv2_weight target=layer2.0.conv2.weight verify=unproved (too large)} ->[n55] constant,
     t19 f32 [C=128] {pt2=root:p_layer2_0_bn2_weight target=layer2.0.bn2.weight verify=proved (structural) [sampled 4]} ->[n59] constant,
     t20 f32 [C=128] {pt2=root:p_layer2_0_bn2_bias target=layer2.0.bn2.bias verify=proved (structural) [sampled 4]} ->[n59] constant,
     t21 f32 [D=128 H=64 W=1 C=1] {pt2=root:p_layer2_0_downsample_0_weight target=layer2.0.downsample.0.weight verify=unproved (too large)} ->[n62] constant,
     t22 f32 [C=128] {pt2=root:p_layer2_0_downsample_1_weight target=layer2.0.downsample.1.weight verify=proved (structural) [sampled 4]} ->[n66] constant,
     t23 f32 [C=128] {pt2=root:p_layer2_0_downsample_1_bias target=layer2.0.downsample.1.bias verify=proved (structural) [sampled 4]} ->[n66] constant,
     t24 f32 [D=128 H=128 W=3 C=3] {pt2=root:p_layer2_1_conv1_weight target=layer2.1.conv1.weight verify=unproved (too large)} ->[n71] constant,
     t25 f32 [C=128] {pt2=root:p_layer2_1_bn1_weight target=layer2.1.bn1.weight verify=proved (structural) [sampled 4]} ->[n75] constant,
     t26 f32 [C=128] {pt2=root:p_layer2_1_bn1_bias target=layer2.1.bn1.bias verify=proved (structural) [sampled 4]} ->[n75] constant,
     t27 f32 [D=128 H=128 W=3 C=3] {pt2=root:p_layer2_1_conv2_weight target=layer2.1.conv2.weight verify=unproved (too large)} ->[n79] constant,
     t28 f32 [C=128] {pt2=root:p_layer2_1_bn2_weight target=layer2.1.bn2.weight verify=proved (structural) [sampled 4]} ->[n83] constant,
     t29 f32 [C=128] {pt2=root:p_layer2_1_bn2_bias target=layer2.1.bn2.bias verify=proved (structural) [sampled 4]} ->[n83] constant,
     t30 f32 [D=256 H=128 W=3 C=3] {pt2=root:p_layer3_0_conv1_weight target=layer3.0.conv1.weight verify=unproved (too large)} ->[n88] constant,
     t31 f32 [C=256] {pt2=root:p_layer3_0_bn1_weight target=layer3.0.bn1.weight verify=proved (structural) [sampled 4]} ->[n92] constant,
     t32 f32 [C=256] {pt2=root:p_layer3_0_bn1_bias target=layer3.0.bn1.bias verify=proved (structural) [sampled 4]} ->[n92] constant,
     t33 f32 [D=256 H=256 W=3 C=3] {pt2=root:p_layer3_0_conv2_weight target=layer3.0.conv2.weight verify=unproved (too large)} ->[n96] constant,
     t34 f32 [C=256] {pt2=root:p_layer3_0_bn2_weight target=layer3.0.bn2.weight verify=proved (structural) [sampled 4]} ->[n100] constant,
     t35 f32 [C=256] {pt2=root:p_layer3_0_bn2_bias target=layer3.0.bn2.bias verify=proved (structural) [sampled 4]} ->[n100] constant,
     t36 f32 [D=256 H=128 W=1 C=1] {pt2=root:p_layer3_0_downsample_0_weight target=layer3.0.downsample.0.weight verify=unproved (too large)} ->[n103] constant,
     t37 f32 [C=256] {pt2=root:p_layer3_0_downsample_1_weight target=layer3.0.downsample.1.weight verify=proved (structural) [sampled 4]} ->[n107] constant,
     t38 f32 [C=256] {pt2=root:p_layer3_0_downsample_1_bias target=layer3.0.downsample.1.bias verify=proved (structural) [sampled 4]} ->[n107] constant,
     t39 f32 [D=256 H=256 W=3 C=3] {pt2=root:p_layer3_1_conv1_weight target=layer3.1.conv1.weight verify=unproved (too large)} ->[n112] constant,
     t40 f32 [C=256] {pt2=root:p_layer3_1_bn1_weight target=layer3.1.bn1.weight verify=proved (structural) [sampled 4]} ->[n116] constant,
     t41 f32 [C=256] {pt2=root:p_layer3_1_bn1_bias target=layer3.1.bn1.bias verify=proved (structural) [sampled 4]} ->[n116] constant,
     t42 f32 [D=256 H=256 W=3 C=3] {pt2=root:p_layer3_1_conv2_weight target=layer3.1.conv2.weight verify=unproved (too large)} ->[n120] constant,
     t43 f32 [C=256] {pt2=root:p_layer3_1_bn2_weight target=layer3.1.bn2.weight verify=proved (structural) [sampled 4]} ->[n124] constant,
     t44 f32 [C=256] {pt2=root:p_layer3_1_bn2_bias target=layer3.1.bn2.bias verify=proved (structural) [sampled 4]} ->[n124] constant,
     t45 f32 [D=512 H=256 W=3 C=3] {pt2=root:p_layer4_0_conv1_weight target=layer4.0.conv1.weight verify=unproved (too large)} ->[n129] constant,
     t46 f32 [C=512] {pt2=root:p_layer4_0_bn1_weight target=layer4.0.bn1.weight verify=proved (structural) [sampled 4]} ->[n133] constant,
     t47 f32 [C=512] {pt2=root:p_layer4_0_bn1_bias target=layer4.0.bn1.bias verify=proved (structural) [sampled 4]} ->[n133] constant,
     t48 f32 [D=512 H=512 W=3 C=3] {pt2=root:p_layer4_0_conv2_weight target=layer4.0.conv2.weight verify=unproved (too large)} ->[n137] constant,
     t49 f32 [C=512] {pt2=root:p_layer4_0_bn2_weight target=layer4.0.bn2.weight verify=proved (structural) [sampled 4]} ->[n141] constant,
     t50 f32 [C=512] {pt2=root:p_layer4_0_bn2_bias target=layer4.0.bn2.bias verify=proved (structural) [sampled 4]} ->[n141] constant,
     t51 f32 [D=512 H=256 W=1 C=1] {pt2=root:p_layer4_0_downsample_0_weight target=layer4.0.downsample.0.weight verify=unproved (too large)} ->[n144] constant,
     t52 f32 [C=512] {pt2=root:p_layer4_0_downsample_1_weight target=layer4.0.downsample.1.weight verify=proved (structural) [sampled 4]} ->[n148] constant,
     t53 f32 [C=512] {pt2=root:p_layer4_0_downsample_1_bias target=layer4.0.downsample.1.bias verify=proved (structural) [sampled 4]} ->[n148] constant,
     t54 f32 [D=512 H=512 W=3 C=3] {pt2=root:p_layer4_1_conv1_weight target=layer4.1.conv1.weight verify=unproved (too large)} ->[n153] constant,
     t55 f32 [C=512] {pt2=root:p_layer4_1_bn1_weight target=layer4.1.bn1.weight verify=proved (structural) [sampled 4]} ->[n157] constant,
     t56 f32 [C=512] {pt2=root:p_layer4_1_bn1_bias target=layer4.1.bn1.bias verify=proved (structural) [sampled 4]} ->[n157] constant,
     t57 f32 [D=512 H=512 W=3 C=3] {pt2=root:p_layer4_1_conv2_weight target=layer4.1.conv2.weight verify=unproved (too large)} ->[n161] constant,
     t58 f32 [C=512] {pt2=root:p_layer4_1_bn2_weight target=layer4.1.bn2.weight verify=proved (structural) [sampled 4]} ->[n165] constant,
     t59 f32 [C=512] {pt2=root:p_layer4_1_bn2_bias target=layer4.1.bn2.bias verify=proved (structural) [sampled 4]} ->[n165] constant,
     t60 f32 [W=1000 C=512] {pt2=root:p_fc_weight target=fc.weight verify=unproved (too large)} ->[n174] constant,
     t61 f32 [C=1000] {pt2=root:p_fc_bias target=fc.bias verify=proved (structural) [sampled 4]} ->[n173] constant,
     t62 f32 [C=64] {pt2=root:b_bn1_running_mean target=bn1.running_mean verify=proved (structural) [sampled 4]} ->[n5] constant,
     t63 f32 [C=64] {pt2=root:b_bn1_running_var target=bn1.running_var verify=proved (structural) [sampled 4]} ->[n5] constant,
     t65 f32 [C=64] {pt2=root:b_layer1_0_bn1_running_mean target=layer1.0.bn1.running_mean verify=proved (structural) [sampled 4]} ->[n17] constant,
     t66 f32 [C=64] {pt2=root:b_layer1_0_bn1_running_var target=layer1.0.bn1.running_var verify=proved (structural) [sampled 4]} ->[n17] constant,
     t68 f32 [C=64] {pt2=root:b_layer1_0_bn2_running_mean target=layer1.0.bn2.running_mean verify=proved (structural) [sampled 4]} ->[n25] constant,
     t69 f32 [C=64] {pt2=root:b_layer1_0_bn2_running_var target=layer1.0.bn2.running_var verify=proved (structural) [sampled 4]} ->[n25] constant,
     t71 f32 [C=64] {pt2=root:b_layer1_1_bn1_running_mean target=layer1.1.bn1.running_mean verify=proved (structural) [sampled 4]} ->[n34] constant,
     t72 f32 [C=64] {pt2=root:b_layer1_1_bn1_running_var target=layer1.1.bn1.running_var verify=proved (structural) [sampled 4]} ->[n34] constant,
     t74 f32 [C=64] {pt2=root:b_layer1_1_bn2_running_mean target=layer1.1.bn2.running_mean verify=proved (structural) [sampled 4]} ->[n42] constant,
     t75 f32 [C=64] {pt2=root:b_layer1_1_bn2_running_var target=layer1.1.bn2.running_var verify=proved (structural) [sampled 4]} ->[n42] constant,
     t77 f32 [C=128] {pt2=root:b_layer2_0_bn1_running_mean target=layer2.0.bn1.running_mean verify=proved (structural) [sampled 4]} ->[n51] constant,
     t78 f32 [C=128] {pt2=root:b_layer2_0_bn1_running_var target=layer2.0.bn1.running_var verify=proved (structural) [sampled 4]} ->[n51] constant,
     t80 f32 [C=128] {pt2=root:b_layer2_0_bn2_running_mean target=layer2.0.bn2.running_mean verify=proved (structural) [sampled 4]} ->[n59] constant,
     t81 f32 [C=128] {pt2=root:b_layer2_0_bn2_running_var target=layer2.0.bn2.running_var verify=proved (structural) [sampled 4]} ->[n59] constant,
     t83 f32 [C=128] {pt2=root:b_layer2_0_downsample_1_running_mean target=layer2.0.downsample.1.running_mean verify=proved (structural) [sampled 4]} ->[n66] constant,
     t84 f32 [C=128] {pt2=root:b_layer2_0_downsample_1_running_var target=layer2.0.downsample.1.running_var verify=proved (structural) [sampled 4]} ->[n66] constant,
     t86 f32 [C=128] {pt2=root:b_layer2_1_bn1_running_mean target=layer2.1.bn1.running_mean verify=proved (structural) [sampled 4]} ->[n75] constant,
     t87 f32 [C=128] {pt2=root:b_layer2_1_bn1_running_var target=layer2.1.bn1.running_var verify=proved (structural) [sampled 4]} ->[n75] constant,
     t89 f32 [C=128] {pt2=root:b_layer2_1_bn2_running_mean target=layer2.1.bn2.running_mean verify=proved (structural) [sampled 4]} ->[n83] constant,
     t90 f32 [C=128] {pt2=root:b_layer2_1_bn2_running_var target=layer2.1.bn2.running_var verify=proved (structural) [sampled 4]} ->[n83] constant,
     t92 f32 [C=256] {pt2=root:b_layer3_0_bn1_running_mean target=layer3.0.bn1.running_mean verify=proved (structural) [sampled 4]} ->[n92] constant,
     t93 f32 [C=256] {pt2=root:b_layer3_0_bn1_running_var target=layer3.0.bn1.running_var verify=proved (structural) [sampled 4]} ->[n92] constant,
     t95 f32 [C=256] {pt2=root:b_layer3_0_bn2_running_mean target=layer3.0.bn2.running_mean verify=proved (structural) [sampled 4]} ->[n100] constant,
     t96 f32 [C=256] {pt2=root:b_layer3_0_bn2_running_var target=layer3.0.bn2.running_var verify=proved (structural) [sampled 4]} ->[n100] constant,
     t98 f32 [C=256] {pt2=root:b_layer3_0_downsample_1_running_mean target=layer3.0.downsample.1.running_mean verify=proved (structural) [sampled 4]} ->[n107] constant,
     t99 f32 [C=256] {pt2=root:b_layer3_0_downsample_1_running_var target=layer3.0.downsample.1.running_var verify=proved (structural) [sampled 4]} ->[n107] constant,
     t101 f32 [C=256] {pt2=root:b_layer3_1_bn1_running_mean target=layer3.1.bn1.running_mean verify=proved (structural) [sampled 4]} ->[n116] constant,
     t102 f32 [C=256] {pt2=root:b_layer3_1_bn1_running_var target=layer3.1.bn1.running_var verify=proved (structural) [sampled 4]} ->[n116] constant,
     t104 f32 [C=256] {pt2=root:b_layer3_1_bn2_running_mean target=layer3.1.bn2.running_mean verify=proved (structural) [sampled 4]} ->[n124] constant,
     t105 f32 [C=256] {pt2=root:b_layer3_1_bn2_running_var target=layer3.1.bn2.running_var verify=proved (structural) [sampled 4]} ->[n124] constant,
     t107 f32 [C=512] {pt2=root:b_layer4_0_bn1_running_mean target=layer4.0.bn1.running_mean verify=proved (structural) [sampled 4]} ->[n133] constant,
     t108 f32 [C=512] {pt2=root:b_layer4_0_bn1_running_var target=layer4.0.bn1.running_var verify=proved (structural) [sampled 4]} ->[n133] constant,
     t110 f32 [C=512] {pt2=root:b_layer4_0_bn2_running_mean target=layer4.0.bn2.running_mean verify=proved (structural) [sampled 4]} ->[n141] constant,
     t111 f32 [C=512] {pt2=root:b_layer4_0_bn2_running_var target=layer4.0.bn2.running_var verify=proved (structural) [sampled 4]} ->[n141] constant,
     t113 f32 [C=512] {pt2=root:b_layer4_0_downsample_1_running_mean target=layer4.0.downsample.1.running_mean verify=proved (structural) [sampled 4]} ->[n148] constant,
     t114 f32 [C=512] {pt2=root:b_layer4_0_downsample_1_running_var target=layer4.0.downsample.1.running_var verify=proved (structural) [sampled 4]} ->[n148] constant,
     t116 f32 [C=512] {pt2=root:b_layer4_1_bn1_running_mean target=layer4.1.bn1.running_mean verify=proved (structural) [sampled 4]} ->[n157] constant,
     t117 f32 [C=512] {pt2=root:b_layer4_1_bn1_running_var target=layer4.1.bn1.running_var verify=proved (structural) [sampled 4]} ->[n157] constant,
     t119 f32 [C=512] {pt2=root:b_layer4_1_bn2_running_mean target=layer4.1.bn2.running_mean verify=proved (structural) [sampled 4]} ->[n165] constant,
     t120 f32 [C=512] {pt2=root:b_layer4_1_bn2_running_var target=layer4.1.bn2.running_var verify=proved (structural) [sampled 4]} ->[n165] constant,
     t122 f32 [H=3 W=224 C=224] {pt2=root:x verify=unproved (too large)} ->[n0]]
  nodes:
    group g1 torch.ops.aten.convolution.default:
      n0 {derived verify=unproved (too large)}: [t123 f32 [H=224 W=224 C=3] {derived verify=unproved (too large)} ->[n2]] =
        permute
          x=t122 {pt2=root:x verify=unproved (too large)}
          perm=[H<-W, W<-C, C<-H]
      n1 {derived verify=unproved (too large)}: [t124 f32 [N=64 T=1 D=1 H=7 W=7
                                                           C=3] {derived verify=unproved (too large)} ->[n2]] =
        permute
          x=t0 {pt2=root:p_conv1_weight target=conv1.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2 {derived verify=unproved (too large)}: [t125 f32 [H=112 W=112 C=64] {derived verify=unproved (too large) origins=2} ->[n5]] =
        convolution
          x=t123 {derived verify=unproved (too large)} <-n0
          weight=t124 {derived verify=unproved (too large)} <-n1
          bias=none
          params={stride={h=2; w=2};
                 padding={h=3; w=3};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g4 torch.ops.aten.convolution.default:
      n13 {derived verify=unproved (too large)}: [t136 f32 [N=64 T=1 D=1 H=3
                                                            W=3 C=64] {derived verify=unproved (too large)} ->[n14]] =
        permute
          x=t3 {pt2=root:p_layer1_0_conv1_weight target=layer1.0.conv1.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n14 {derived verify=unproved (too large)}: [t137 f32 [H=56 W=56 C=64] {derived verify=unproved (too large) origins=2} ->[n17]] =
        convolution
          x=t132 {derived verify=unproved (too large) origins=2} <-n9
          weight=t136 {derived verify=unproved (too large)} <-n13
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g6 torch.ops.aten.convolution.default:
      n21 {derived verify=unproved (too large)}: [t144 f32 [N=64 T=1 D=1 H=3
                                                            W=3 C=64] {derived verify=unproved (too large)} ->[n22]] =
        permute
          x=t6 {pt2=root:p_layer1_0_conv2_weight target=layer1.0.conv2.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n22 {derived verify=unproved (too large)}: [t145 f32 [H=56 W=56 C=64] {derived verify=unproved (too large) origins=2} ->[n25]] =
        convolution
          x=t298 {derived verify=unproved (too large)} <-n176
          weight=t144 {derived verify=unproved (too large)} <-n21
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g8 torch.ops.aten.convolution.default:
      n30 {derived verify=unproved (too large)}: [t153 f32 [N=64 T=1 D=1 H=3
                                                            W=3 C=64] {derived verify=unproved (too large)} ->[n31]] =
        permute
          x=t9 {pt2=root:p_layer1_1_conv1_weight target=layer1.1.conv1.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n31 {derived verify=unproved (too large)}: [t154 f32 [H=56 W=56 C=64] {derived verify=unproved (too large) origins=2} ->[n34]] =
        convolution
          x=t300 {derived verify=unproved (too large)} <-n178
          weight=t153 {derived verify=unproved (too large)} <-n30
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g10 torch.ops.aten.convolution.default:
      n38 {derived verify=unproved (too large)}: [t161 f32 [N=64 T=1 D=1 H=3
                                                            W=3 C=64] {derived verify=unproved (too large)} ->[n39]] =
        permute
          x=t12 {pt2=root:p_layer1_1_conv2_weight target=layer1.1.conv2.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n39 {derived verify=unproved (too large)}: [t162 f32 [H=56 W=56 C=64] {derived verify=unproved (too large) origins=2} ->[n42]] =
        convolution
          x=t301 {derived verify=unproved (too large)} <-n179
          weight=t161 {derived verify=unproved (too large)} <-n38
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g12 torch.ops.aten.convolution.default:
      n47 {derived verify=unproved (too large)}: [t170 f32 [N=128 T=1 D=1 H=3
                                                            W=3 C=64] {derived verify=unproved (too large)} ->[n48]] =
        permute
          x=t15 {pt2=root:p_layer2_0_conv1_weight target=layer2.0.conv1.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n48 {derived verify=unproved (too large)}: [t171 f32 [H=28 W=28 C=128] {derived verify=unproved (too large) origins=2} ->[n51]] =
        convolution
          x=t303 {derived verify=unproved (too large) origins=2} <-n181
          weight=t170 {derived verify=unproved (too large)} <-n47
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g14 torch.ops.aten.convolution.default:
      n55 {derived verify=unproved (too large)}: [t178 f32 [N=128 T=1 D=1 H=3
                                                            W=3 C=128] {derived verify=unproved (too large)} ->[n56]] =
        permute
          x=t18 {pt2=root:p_layer2_0_conv2_weight target=layer2.0.conv2.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n56 {derived verify=unproved (too large)}: [t179 f32 [H=28 W=28 C=128] {derived verify=unproved (too large) origins=2} ->[n59]] =
        convolution
          x=t304 {derived verify=unproved (too large)} <-n182
          weight=t178 {derived verify=unproved (too large)} <-n55
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g16 torch.ops.aten.convolution.default:
      n62 {derived verify=unproved (too large)}: [t185 f32 [N=128 T=1 D=1 H=1
                                                            W=1 C=64] {derived verify=unproved (too large)} ->[n63]] =
        permute
          x=t21 {pt2=root:p_layer2_0_downsample_0_weight target=layer2.0.downsample.0.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n63 {derived verify=unproved (too large)}: [t186 f32 [H=28 W=28 C=128] {derived verify=unproved (too large) origins=2} ->[n66]] =
        convolution
          x=t303 {derived verify=unproved (too large) origins=2} <-n181
          weight=t185 {derived verify=unproved (too large)} <-n62
          bias=none
          params={stride={h=2; w=2};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g18 torch.ops.aten.convolution.default:
      n71 {derived verify=unproved (too large)}: [t194 f32 [N=128 T=1 D=1 H=3
                                                            W=3 C=128] {derived verify=unproved (too large)} ->[n72]] =
        permute
          x=t24 {pt2=root:p_layer2_1_conv1_weight target=layer2.1.conv1.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n72 {derived verify=unproved (too large)}: [t195 f32 [H=28 W=28 C=128] {derived verify=unproved (too large) origins=2} ->[n75]] =
        convolution
          x=t306 {derived verify=unproved (too large)} <-n184
          weight=t194 {derived verify=unproved (too large)} <-n71
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g20 torch.ops.aten.convolution.default:
      n79 {derived verify=unproved (too large)}: [t202 f32 [N=128 T=1 D=1 H=3
                                                            W=3 C=128] {derived verify=unproved (too large)} ->[n80]] =
        permute
          x=t27 {pt2=root:p_layer2_1_conv2_weight target=layer2.1.conv2.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n80 {derived verify=unproved (too large)}: [t203 f32 [H=28 W=28 C=128] {derived verify=unproved (too large) origins=2} ->[n83]] =
        convolution
          x=t307 {derived verify=unproved (too large)} <-n185
          weight=t202 {derived verify=unproved (too large)} <-n79
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g22 torch.ops.aten.convolution.default:
      n88 {derived verify=unproved (too large)}: [t211 f32 [N=256 T=1 D=1 H=3
                                                            W=3 C=128] {derived verify=unproved (too large)} ->[n89]] =
        permute
          x=t30 {pt2=root:p_layer3_0_conv1_weight target=layer3.0.conv1.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n89 {derived verify=unproved (too large)}: [t212 f32 [H=14 W=14 C=256] {derived verify=unproved (too large) origins=2} ->[n92]] =
        convolution
          x=t309 {derived verify=unproved (too large) origins=2} <-n187
          weight=t211 {derived verify=unproved (too large)} <-n88
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g24 torch.ops.aten.convolution.default:
      n96 {derived verify=unproved (too large)}: [t219 f32 [N=256 T=1 D=1 H=3
                                                            W=3 C=256] {derived verify=unproved (too large)} ->[n97]] =
        permute
          x=t33 {pt2=root:p_layer3_0_conv2_weight target=layer3.0.conv2.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n97 {derived verify=unproved (too large)}: [t220 f32 [H=14 W=14 C=256] {derived verify=unproved (too large) origins=2} ->[n100]] =
        convolution
          x=t310 {derived verify=unproved (too large)} <-n188
          weight=t219 {derived verify=unproved (too large)} <-n96
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g26 torch.ops.aten.convolution.default:
      n103 {derived verify=unproved (too large)}: [t226 f32 [N=256 T=1 D=1 H=1
                                                             W=1 C=128] {derived verify=unproved (too large)} ->[n104]] =
        permute
          x=t36 {pt2=root:p_layer3_0_downsample_0_weight target=layer3.0.downsample.0.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n104 {derived verify=unproved (too large)}: [t227 f32 [H=14 W=14 C=256] {derived verify=unproved (too large) origins=2} ->[n107]] =
        convolution
          x=t309 {derived verify=unproved (too large) origins=2} <-n187
          weight=t226 {derived verify=unproved (too large)} <-n103
          bias=none
          params={stride={h=2; w=2};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g28 torch.ops.aten.convolution.default:
      n112 {derived verify=unproved (too large)}: [t235 f32 [N=256 T=1 D=1 H=3
                                                             W=3 C=256] {derived verify=unproved (too large)} ->[n113]] =
        permute
          x=t39 {pt2=root:p_layer3_1_conv1_weight target=layer3.1.conv1.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n113 {derived verify=unproved (too large)}: [t236 f32 [H=14 W=14 C=256] {derived verify=unproved (too large) origins=2} ->[n116]] =
        convolution
          x=t312 {derived verify=unproved (too large)} <-n190
          weight=t235 {derived verify=unproved (too large)} <-n112
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g30 torch.ops.aten.convolution.default:
      n120 {derived verify=unproved (too large)}: [t243 f32 [N=256 T=1 D=1 H=3
                                                             W=3 C=256] {derived verify=unproved (too large)} ->[n121]] =
        permute
          x=t42 {pt2=root:p_layer3_1_conv2_weight target=layer3.1.conv2.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n121 {derived verify=unproved (too large)}: [t244 f32 [H=14 W=14 C=256] {derived verify=unproved (too large) origins=2} ->[n124]] =
        convolution
          x=t313 {derived verify=unproved (too large)} <-n191
          weight=t243 {derived verify=unproved (too large)} <-n120
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g32 torch.ops.aten.convolution.default:
      n129 {derived verify=unproved (too large)}: [t252 f32 [N=512 T=1 D=1 H=3
                                                             W=3 C=256] {derived verify=unproved (too large)} ->[n130]] =
        permute
          x=t45 {pt2=root:p_layer4_0_conv1_weight target=layer4.0.conv1.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n130 {derived verify=unproved (too large)}: [t253 f32 [H=7 W=7 C=512] {derived verify=unproved (too large) origins=2} ->[n133]] =
        convolution
          x=t315 {derived verify=unproved (too large) origins=2} <-n193
          weight=t252 {derived verify=unproved (too large)} <-n129
          bias=none
          params={stride={h=2; w=2};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g34 torch.ops.aten.convolution.default:
      n137 {derived verify=unproved (too large)}: [t260 f32 [N=512 T=1 D=1 H=3
                                                             W=3 C=512] {derived verify=unproved (too large)} ->[n138]] =
        permute
          x=t48 {pt2=root:p_layer4_0_conv2_weight target=layer4.0.conv2.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n138 {derived verify=unproved (too large)}: [t261 f32 [H=7 W=7 C=512] {derived verify=unproved (too large) origins=2} ->[n141]] =
        convolution
          x=t316 {derived verify=unproved (too large)} <-n194
          weight=t260 {derived verify=unproved (too large)} <-n137
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g36 torch.ops.aten.convolution.default:
      n144 {derived verify=unproved (too large)}: [t267 f32 [N=512 T=1 D=1 H=1
                                                             W=1 C=256] {derived verify=unproved (too large)} ->[n145]] =
        permute
          x=t51 {pt2=root:p_layer4_0_downsample_0_weight target=layer4.0.downsample.0.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n145 {derived verify=unproved (too large)}: [t268 f32 [H=7 W=7 C=512] {derived verify=unproved (too large) origins=2} ->[n148]] =
        convolution
          x=t315 {derived verify=unproved (too large) origins=2} <-n193
          weight=t267 {derived verify=unproved (too large)} <-n144
          bias=none
          params={stride={h=2; w=2};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g38 torch.ops.aten.convolution.default:
      n153 {derived verify=unproved (too large)}: [t276 f32 [N=512 T=1 D=1 H=3
                                                             W=3 C=512] {derived verify=unproved (too large)} ->[n154]] =
        permute
          x=t54 {pt2=root:p_layer4_1_conv1_weight target=layer4.1.conv1.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n154 {derived verify=unproved (too large)}: [t277 f32 [H=7 W=7 C=512] {derived verify=unproved (too large) origins=2} ->[n157]] =
        convolution
          x=t318 {derived verify=unproved (too large)} <-n196
          weight=t276 {derived verify=unproved (too large)} <-n153
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    group g40 torch.ops.aten.convolution.default:
      n161 {derived verify=unproved (too large)}: [t284 f32 [N=512 T=1 D=1 H=3
                                                             W=3 C=512] {derived verify=unproved (too large)} ->[n162]] =
        permute
          x=t57 {pt2=root:p_layer4_1_conv2_weight target=layer4.1.conv2.weight verify=unproved (too large)}
          perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n162 {derived verify=unproved (too large)}: [t285 f32 [H=7 W=7 C=512] {derived verify=unproved (too large) origins=2} ->[n165]] =
        convolution
          x=t319 {derived verify=unproved (too large)} <-n197
          weight=t284 {derived verify=unproved (too large)} <-n161
          bias=none
          params={stride={h=1; w=1};
                 padding={h=1; w=1};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
    n174 {pt2=root[68] torch.ops.aten.permute.default verify=unproved (too large)}: [t295 f32 [N=1000
                                                                      T=1 D=1
                                                                      H=1 W=1
                                                                      C=512] {derived verify=unproved (too large)} ->[n173]] =
      permute
        x=t60 {pt2=root:p_fc_weight target=fc.weight verify=unproved (too large)}
        perm=[N<-W, W<-N]
    group g2 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n5 {derived verify=unproved (too large)}: [t128 f32 [H=112 W=112 C=64] {derived verify=unproved (too large)} ->[n175]] =
        batch_norm
          x=t125 {derived verify=unproved (too large) origins=2} <-n2
          weight=t1 {pt2=root:p_bn1_weight target=bn1.weight verify=proved (structural) [sampled 4]}
          bias=t2 {pt2=root:p_bn1_bias target=bn1.bias verify=proved (structural) [sampled 4]}
          running_mean=t62 {pt2=root:b_bn1_running_mean target=bn1.running_mean verify=proved (structural) [sampled 4]}
          running_var=t63 {pt2=root:b_bn1_running_var target=bn1.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    n175 {pt2=root[2] torch.ops.aten.relu.default verify=unproved (too large)}: [t297 f32 [H=112
                                                                      W=112
                                                                      C=64] {derived verify=unproved (too large)} ->[n9]] =
      relu x=t128 {derived verify=unproved (too large)} <-n5
    group g3 torch.ops.aten.max_pool2d_with_indices.default:
      n9 {derived verify=unproved (too large)}: [t132 f32 [H=56 W=56 C=64] {derived verify=unproved (too large) origins=2} ->[n14,
                                                                      n177],
                                                 t133 f32 [H=56 W=56 C=64] {derived verify=unproved (too large)} ->[n10]] =
        max_pool2d_with_indices
          x=t297 {derived verify=unproved (too large)} <-n175
          params={kernel={h=3; w=3}; stride={h=2; w=2}; pad={h=1; w=1}}
      n10 {derived}: [] =
        discard x=t133 {derived verify=unproved (too large)} <-n9
    group g5 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n17 {derived verify=unproved (too large)}: [t140 f32 [H=56 W=56 C=64] {derived verify=unproved (too large)} ->[n176]] =
        batch_norm
          x=t137 {derived verify=unproved (too large) origins=2} <-n14
          weight=t4 {pt2=root:p_layer1_0_bn1_weight target=layer1.0.bn1.weight verify=proved (structural) [sampled 4]}
          bias=t5 {pt2=root:p_layer1_0_bn1_bias target=layer1.0.bn1.bias verify=proved (structural) [sampled 4]}
          running_mean=t65 {pt2=root:b_layer1_0_bn1_running_mean target=layer1.0.bn1.running_mean verify=proved (structural) [sampled 4]}
          running_var=t66 {pt2=root:b_layer1_0_bn1_running_var target=layer1.0.bn1.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    n176 {pt2=root[6] torch.ops.aten.relu.default verify=unproved (too large)}: [t298 f32 [H=56
                                                                      W=56
                                                                      C=64] {derived verify=unproved (too large)} ->[n22]] =
      relu x=t140 {derived verify=unproved (too large)} <-n17
    group g7 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n25 {derived verify=unproved (too large)}: [t148 f32 [H=56 W=56 C=64] {derived verify=unproved (too large)} ->[n177]] =
        batch_norm
          x=t145 {derived verify=unproved (too large) origins=2} <-n22
          weight=t7 {pt2=root:p_layer1_0_bn2_weight target=layer1.0.bn2.weight verify=proved (structural) [sampled 4]}
          bias=t8 {pt2=root:p_layer1_0_bn2_bias target=layer1.0.bn2.bias verify=proved (structural) [sampled 4]}
          running_mean=t68 {pt2=root:b_layer1_0_bn2_running_mean target=layer1.0.bn2.running_mean verify=proved (structural) [sampled 4]}
          running_var=t69 {pt2=root:b_layer1_0_bn2_running_var target=layer1.0.bn2.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    n177 {pt2=root[9] torch.ops.aten.add.Tensor verify=vacuous}: [t299 f32 [H=56
                                                                      W=56
                                                                      C=64] {derived verify=vacuous origins=0} ->[n178]] =
      add
        a=t148 {derived verify=unproved (too large)} <-n25
        b=t132 {derived verify=unproved (too large) origins=2} <-n9
    n178 {pt2=root[10] torch.ops.aten.relu.default verify=unproved (too large)}: [t300 f32 [H=56
                                                                      W=56
                                                                      C=64] {derived verify=unproved (too large)} ->[n31,
                                                                      n180]] =
      relu x=t299 {derived verify=vacuous origins=0} <-n177
    group g9 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n34 {derived verify=unproved (too large)}: [t157 f32 [H=56 W=56 C=64] {derived verify=unproved (too large)} ->[n179]] =
        batch_norm
          x=t154 {derived verify=unproved (too large) origins=2} <-n31
          weight=t10 {pt2=root:p_layer1_1_bn1_weight target=layer1.1.bn1.weight verify=proved (structural) [sampled 4]}
          bias=t11 {pt2=root:p_layer1_1_bn1_bias target=layer1.1.bn1.bias verify=proved (structural) [sampled 4]}
          running_mean=t71 {pt2=root:b_layer1_1_bn1_running_mean target=layer1.1.bn1.running_mean verify=proved (structural) [sampled 4]}
          running_var=t72 {pt2=root:b_layer1_1_bn1_running_var target=layer1.1.bn1.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    n179 {pt2=root[13] torch.ops.aten.relu.default verify=unproved (too large)}: [t301 f32 [H=56
                                                                      W=56
                                                                      C=64] {derived verify=unproved (too large)} ->[n39]] =
      relu x=t157 {derived verify=unproved (too large)} <-n34
    group g11 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n42 {derived verify=unproved (too large)}: [t165 f32 [H=56 W=56 C=64] {derived verify=unproved (too large)} ->[n180]] =
        batch_norm
          x=t162 {derived verify=unproved (too large) origins=2} <-n39
          weight=t13 {pt2=root:p_layer1_1_bn2_weight target=layer1.1.bn2.weight verify=proved (structural) [sampled 4]}
          bias=t14 {pt2=root:p_layer1_1_bn2_bias target=layer1.1.bn2.bias verify=proved (structural) [sampled 4]}
          running_mean=t74 {pt2=root:b_layer1_1_bn2_running_mean target=layer1.1.bn2.running_mean verify=proved (structural) [sampled 4]}
          running_var=t75 {pt2=root:b_layer1_1_bn2_running_var target=layer1.1.bn2.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    n180 {pt2=root[16] torch.ops.aten.add.Tensor verify=vacuous}: [t302 f32 [H=56
                                                                      W=56
                                                                      C=64] {derived verify=vacuous origins=0} ->[n181]] =
      add
        a=t165 {derived verify=unproved (too large)} <-n42
        b=t300 {derived verify=unproved (too large)} <-n178
    n181 {pt2=root[17] torch.ops.aten.relu.default verify=unproved (too large)}: [t303 f32 [H=56
                                                                      W=56
                                                                      C=64] {derived verify=unproved (too large) origins=2} ->[n48,
                                                                      n63]] =
      relu x=t302 {derived verify=vacuous origins=0} <-n180
    group g13 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n51 {derived verify=unproved (too large)}: [t174 f32 [H=28 W=28 C=128] {derived verify=unproved (too large)} ->[n182]] =
        batch_norm
          x=t171 {derived verify=unproved (too large) origins=2} <-n48
          weight=t16 {pt2=root:p_layer2_0_bn1_weight target=layer2.0.bn1.weight verify=proved (structural) [sampled 4]}
          bias=t17 {pt2=root:p_layer2_0_bn1_bias target=layer2.0.bn1.bias verify=proved (structural) [sampled 4]}
          running_mean=t77 {pt2=root:b_layer2_0_bn1_running_mean target=layer2.0.bn1.running_mean verify=proved (structural) [sampled 4]}
          running_var=t78 {pt2=root:b_layer2_0_bn1_running_var target=layer2.0.bn1.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    group g17 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n66 {derived verify=unproved (too large)}: [t189 f32 [H=28 W=28 C=128] {derived verify=unproved (too large)} ->[n183]] =
        batch_norm
          x=t186 {derived verify=unproved (too large) origins=2} <-n63
          weight=t22 {pt2=root:p_layer2_0_downsample_1_weight target=layer2.0.downsample.1.weight verify=proved (structural) [sampled 4]}
          bias=t23 {pt2=root:p_layer2_0_downsample_1_bias target=layer2.0.downsample.1.bias verify=proved (structural) [sampled 4]}
          running_mean=t83 {pt2=root:b_layer2_0_downsample_1_running_mean target=layer2.0.downsample.1.running_mean verify=proved (structural) [sampled 4]}
          running_var=t84 {pt2=root:b_layer2_0_downsample_1_running_var target=layer2.0.downsample.1.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    n182 {pt2=root[20] torch.ops.aten.relu.default verify=unproved (too large)}: [t304 f32 [H=28
                                                                      W=28
                                                                      C=128] {derived verify=unproved (too large)} ->[n56]] =
      relu x=t174 {derived verify=unproved (too large)} <-n51
    group g15 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n59 {derived verify=unproved (too large)}: [t182 f32 [H=28 W=28 C=128] {derived verify=unproved (too large)} ->[n183]] =
        batch_norm
          x=t179 {derived verify=unproved (too large) origins=2} <-n56
          weight=t19 {pt2=root:p_layer2_0_bn2_weight target=layer2.0.bn2.weight verify=proved (structural) [sampled 4]}
          bias=t20 {pt2=root:p_layer2_0_bn2_bias target=layer2.0.bn2.bias verify=proved (structural) [sampled 4]}
          running_mean=t80 {pt2=root:b_layer2_0_bn2_running_mean target=layer2.0.bn2.running_mean verify=proved (structural) [sampled 4]}
          running_var=t81 {pt2=root:b_layer2_0_bn2_running_var target=layer2.0.bn2.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    n183 {pt2=root[25] torch.ops.aten.add.Tensor verify=vacuous}: [t305 f32 [H=28
                                                                      W=28
                                                                      C=128] {derived verify=vacuous origins=0} ->[n184]] =
      add
        a=t182 {derived verify=unproved (too large)} <-n59
        b=t189 {derived verify=unproved (too large)} <-n66
    n184 {pt2=root[26] torch.ops.aten.relu.default verify=unproved (too large)}: [t306 f32 [H=28
                                                                      W=28
                                                                      C=128] {derived verify=unproved (too large)} ->[n72,
                                                                      n186]] =
      relu x=t305 {derived verify=vacuous origins=0} <-n183
    group g19 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n75 {derived verify=unproved (too large)}: [t198 f32 [H=28 W=28 C=128] {derived verify=unproved (too large)} ->[n185]] =
        batch_norm
          x=t195 {derived verify=unproved (too large) origins=2} <-n72
          weight=t25 {pt2=root:p_layer2_1_bn1_weight target=layer2.1.bn1.weight verify=proved (structural) [sampled 4]}
          bias=t26 {pt2=root:p_layer2_1_bn1_bias target=layer2.1.bn1.bias verify=proved (structural) [sampled 4]}
          running_mean=t86 {pt2=root:b_layer2_1_bn1_running_mean target=layer2.1.bn1.running_mean verify=proved (structural) [sampled 4]}
          running_var=t87 {pt2=root:b_layer2_1_bn1_running_var target=layer2.1.bn1.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    n185 {pt2=root[29] torch.ops.aten.relu.default verify=unproved (too large)}: [t307 f32 [H=28
                                                                      W=28
                                                                      C=128] {derived verify=unproved (too large)} ->[n80]] =
      relu x=t198 {derived verify=unproved (too large)} <-n75
    group g21 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n83 {derived verify=unproved (too large)}: [t206 f32 [H=28 W=28 C=128] {derived verify=unproved (too large)} ->[n186]] =
        batch_norm
          x=t203 {derived verify=unproved (too large) origins=2} <-n80
          weight=t28 {pt2=root:p_layer2_1_bn2_weight target=layer2.1.bn2.weight verify=proved (structural) [sampled 4]}
          bias=t29 {pt2=root:p_layer2_1_bn2_bias target=layer2.1.bn2.bias verify=proved (structural) [sampled 4]}
          running_mean=t89 {pt2=root:b_layer2_1_bn2_running_mean target=layer2.1.bn2.running_mean verify=proved (structural) [sampled 4]}
          running_var=t90 {pt2=root:b_layer2_1_bn2_running_var target=layer2.1.bn2.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    n186 {pt2=root[32] torch.ops.aten.add.Tensor verify=vacuous}: [t308 f32 [H=28
                                                                      W=28
                                                                      C=128] {derived verify=vacuous origins=0} ->[n187]] =
      add
        a=t206 {derived verify=unproved (too large)} <-n83
        b=t306 {derived verify=unproved (too large)} <-n184
    n187 {pt2=root[33] torch.ops.aten.relu.default verify=unproved (too large)}: [t309 f32 [H=28
                                                                      W=28
                                                                      C=128] {derived verify=unproved (too large) origins=2} ->[n89,
                                                                      n104]] =
      relu x=t308 {derived verify=vacuous origins=0} <-n186
    group g23 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n92 {derived verify=unproved (too large)}: [t215 f32 [H=14 W=14 C=256] {derived verify=unproved (too large)} ->[n188]] =
        batch_norm
          x=t212 {derived verify=unproved (too large) origins=2} <-n89
          weight=t31 {pt2=root:p_layer3_0_bn1_weight target=layer3.0.bn1.weight verify=proved (structural) [sampled 4]}
          bias=t32 {pt2=root:p_layer3_0_bn1_bias target=layer3.0.bn1.bias verify=proved (structural) [sampled 4]}
          running_mean=t92 {pt2=root:b_layer3_0_bn1_running_mean target=layer3.0.bn1.running_mean verify=proved (structural) [sampled 4]}
          running_var=t93 {pt2=root:b_layer3_0_bn1_running_var target=layer3.0.bn1.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    group g27 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n107 {derived verify=unproved (too large)}: [t230 f32 [H=14 W=14 C=256] {derived verify=unproved (too large)} ->[n189]] =
        batch_norm
          x=t227 {derived verify=unproved (too large) origins=2} <-n104
          weight=t37 {pt2=root:p_layer3_0_downsample_1_weight target=layer3.0.downsample.1.weight verify=proved (structural) [sampled 4]}
          bias=t38 {pt2=root:p_layer3_0_downsample_1_bias target=layer3.0.downsample.1.bias verify=proved (structural) [sampled 4]}
          running_mean=t98 {pt2=root:b_layer3_0_downsample_1_running_mean target=layer3.0.downsample.1.running_mean verify=proved (structural) [sampled 4]}
          running_var=t99 {pt2=root:b_layer3_0_downsample_1_running_var target=layer3.0.downsample.1.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    n188 {pt2=root[36] torch.ops.aten.relu.default verify=unproved (too large)}: [t310 f32 [H=14
                                                                      W=14
                                                                      C=256] {derived verify=unproved (too large)} ->[n97]] =
      relu x=t215 {derived verify=unproved (too large)} <-n92
    group g25 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n100 {derived verify=unproved (too large)}: [t223 f32 [H=14 W=14 C=256] {derived verify=unproved (too large)} ->[n189]] =
        batch_norm
          x=t220 {derived verify=unproved (too large) origins=2} <-n97
          weight=t34 {pt2=root:p_layer3_0_bn2_weight target=layer3.0.bn2.weight verify=proved (structural) [sampled 4]}
          bias=t35 {pt2=root:p_layer3_0_bn2_bias target=layer3.0.bn2.bias verify=proved (structural) [sampled 4]}
          running_mean=t95 {pt2=root:b_layer3_0_bn2_running_mean target=layer3.0.bn2.running_mean verify=proved (structural) [sampled 4]}
          running_var=t96 {pt2=root:b_layer3_0_bn2_running_var target=layer3.0.bn2.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    n189 {pt2=root[41] torch.ops.aten.add.Tensor verify=vacuous}: [t311 f32 [H=14
                                                                      W=14
                                                                      C=256] {derived verify=vacuous origins=0} ->[n190]] =
      add
        a=t223 {derived verify=unproved (too large)} <-n100
        b=t230 {derived verify=unproved (too large)} <-n107
    n190 {pt2=root[42] torch.ops.aten.relu.default verify=unproved (too large)}: [t312 f32 [H=14
                                                                      W=14
                                                                      C=256] {derived verify=unproved (too large)} ->[n113,
                                                                      n192]] =
      relu x=t311 {derived verify=vacuous origins=0} <-n189
    group g29 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n116 {derived verify=unproved (too large)}: [t239 f32 [H=14 W=14 C=256] {derived verify=unproved (too large)} ->[n191]] =
        batch_norm
          x=t236 {derived verify=unproved (too large) origins=2} <-n113
          weight=t40 {pt2=root:p_layer3_1_bn1_weight target=layer3.1.bn1.weight verify=proved (structural) [sampled 4]}
          bias=t41 {pt2=root:p_layer3_1_bn1_bias target=layer3.1.bn1.bias verify=proved (structural) [sampled 4]}
          running_mean=t101 {pt2=root:b_layer3_1_bn1_running_mean target=layer3.1.bn1.running_mean verify=proved (structural) [sampled 4]}
          running_var=t102 {pt2=root:b_layer3_1_bn1_running_var target=layer3.1.bn1.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    n191 {pt2=root[45] torch.ops.aten.relu.default verify=unproved (too large)}: [t313 f32 [H=14
                                                                      W=14
                                                                      C=256] {derived verify=unproved (too large)} ->[n121]] =
      relu x=t239 {derived verify=unproved (too large)} <-n116
    group g31 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n124 {derived verify=unproved (too large)}: [t247 f32 [H=14 W=14 C=256] {derived verify=unproved (too large)} ->[n192]] =
        batch_norm
          x=t244 {derived verify=unproved (too large) origins=2} <-n121
          weight=t43 {pt2=root:p_layer3_1_bn2_weight target=layer3.1.bn2.weight verify=proved (structural) [sampled 4]}
          bias=t44 {pt2=root:p_layer3_1_bn2_bias target=layer3.1.bn2.bias verify=proved (structural) [sampled 4]}
          running_mean=t104 {pt2=root:b_layer3_1_bn2_running_mean target=layer3.1.bn2.running_mean verify=proved (structural) [sampled 4]}
          running_var=t105 {pt2=root:b_layer3_1_bn2_running_var target=layer3.1.bn2.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    n192 {pt2=root[48] torch.ops.aten.add.Tensor verify=vacuous}: [t314 f32 [H=14
                                                                      W=14
                                                                      C=256] {derived verify=vacuous origins=0} ->[n193]] =
      add
        a=t247 {derived verify=unproved (too large)} <-n124
        b=t312 {derived verify=unproved (too large)} <-n190
    n193 {pt2=root[49] torch.ops.aten.relu.default verify=unproved (too large)}: [t315 f32 [H=14
                                                                      W=14
                                                                      C=256] {derived verify=unproved (too large) origins=2} ->[n130,
                                                                      n145]] =
      relu x=t314 {derived verify=vacuous origins=0} <-n192
    group g33 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n133 {derived verify=unproved (too large)}: [t256 f32 [H=7 W=7 C=512] {derived verify=unproved (too large)} ->[n194]] =
        batch_norm
          x=t253 {derived verify=unproved (too large) origins=2} <-n130
          weight=t46 {pt2=root:p_layer4_0_bn1_weight target=layer4.0.bn1.weight verify=proved (structural) [sampled 4]}
          bias=t47 {pt2=root:p_layer4_0_bn1_bias target=layer4.0.bn1.bias verify=proved (structural) [sampled 4]}
          running_mean=t107 {pt2=root:b_layer4_0_bn1_running_mean target=layer4.0.bn1.running_mean verify=proved (structural) [sampled 4]}
          running_var=t108 {pt2=root:b_layer4_0_bn1_running_var target=layer4.0.bn1.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    group g37 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n148 {derived verify=unproved (too large)}: [t271 f32 [H=7 W=7 C=512] {derived verify=unproved (too large)} ->[n195]] =
        batch_norm
          x=t268 {derived verify=unproved (too large) origins=2} <-n145
          weight=t52 {pt2=root:p_layer4_0_downsample_1_weight target=layer4.0.downsample.1.weight verify=proved (structural) [sampled 4]}
          bias=t53 {pt2=root:p_layer4_0_downsample_1_bias target=layer4.0.downsample.1.bias verify=proved (structural) [sampled 4]}
          running_mean=t113 {pt2=root:b_layer4_0_downsample_1_running_mean target=layer4.0.downsample.1.running_mean verify=proved (structural) [sampled 4]}
          running_var=t114 {pt2=root:b_layer4_0_downsample_1_running_var target=layer4.0.downsample.1.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    n194 {pt2=root[52] torch.ops.aten.relu.default verify=unproved (too large)}: [t316 f32 [H=7
                                                                      W=7
                                                                      C=512] {derived verify=unproved (too large)} ->[n138]] =
      relu x=t256 {derived verify=unproved (too large)} <-n133
    group g35 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n141 {derived verify=unproved (too large)}: [t264 f32 [H=7 W=7 C=512] {derived verify=unproved (too large)} ->[n195]] =
        batch_norm
          x=t261 {derived verify=unproved (too large) origins=2} <-n138
          weight=t49 {pt2=root:p_layer4_0_bn2_weight target=layer4.0.bn2.weight verify=proved (structural) [sampled 4]}
          bias=t50 {pt2=root:p_layer4_0_bn2_bias target=layer4.0.bn2.bias verify=proved (structural) [sampled 4]}
          running_mean=t110 {pt2=root:b_layer4_0_bn2_running_mean target=layer4.0.bn2.running_mean verify=proved (structural) [sampled 4]}
          running_var=t111 {pt2=root:b_layer4_0_bn2_running_var target=layer4.0.bn2.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    n195 {pt2=root[57] torch.ops.aten.add.Tensor verify=vacuous}: [t317 f32 [H=7
                                                                      W=7
                                                                      C=512] {derived verify=vacuous origins=0} ->[n196]] =
      add
        a=t264 {derived verify=unproved (too large)} <-n141
        b=t271 {derived verify=unproved (too large)} <-n148
    n196 {pt2=root[58] torch.ops.aten.relu.default verify=unproved (too large)}: [t318 f32 [H=7
                                                                      W=7
                                                                      C=512] {derived verify=unproved (too large)} ->[n154,
                                                                      n198]] =
      relu x=t317 {derived verify=vacuous origins=0} <-n195
    group g39 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n157 {derived verify=unproved (too large)}: [t280 f32 [H=7 W=7 C=512] {derived verify=unproved (too large)} ->[n197]] =
        batch_norm
          x=t277 {derived verify=unproved (too large) origins=2} <-n154
          weight=t55 {pt2=root:p_layer4_1_bn1_weight target=layer4.1.bn1.weight verify=proved (structural) [sampled 4]}
          bias=t56 {pt2=root:p_layer4_1_bn1_bias target=layer4.1.bn1.bias verify=proved (structural) [sampled 4]}
          running_mean=t116 {pt2=root:b_layer4_1_bn1_running_mean target=layer4.1.bn1.running_mean verify=proved (structural) [sampled 4]}
          running_var=t117 {pt2=root:b_layer4_1_bn1_running_var target=layer4.1.bn1.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    n197 {pt2=root[61] torch.ops.aten.relu.default verify=unproved (too large)}: [t319 f32 [H=7
                                                                      W=7
                                                                      C=512] {derived verify=unproved (too large)} ->[n162]] =
      relu x=t280 {derived verify=unproved (too large)} <-n157
    group g41 torch.ops.aten._native_batch_norm_legit_no_training.default:
      n165 {derived verify=unproved (too large)}: [t288 f32 [H=7 W=7 C=512] {derived verify=unproved (too large)} ->[n198]] =
        batch_norm
          x=t285 {derived verify=unproved (too large) origins=2} <-n162
          weight=t58 {pt2=root:p_layer4_1_bn2_weight target=layer4.1.bn2.weight verify=proved (structural) [sampled 4]}
          bias=t59 {pt2=root:p_layer4_1_bn2_bias target=layer4.1.bn2.bias verify=proved (structural) [sampled 4]}
          running_mean=t119 {pt2=root:b_layer4_1_bn2_running_mean target=layer4.1.bn2.running_mean verify=proved (structural) [sampled 4]}
          running_var=t120 {pt2=root:b_layer4_1_bn2_running_var target=layer4.1.bn2.running_var verify=proved (structural) [sampled 4]}
          params={channel=C; eps=1e-05}
    n198 {pt2=root[64] torch.ops.aten.add.Tensor verify=vacuous}: [t320 f32 [H=7
                                                                      W=7
                                                                      C=512] {derived verify=vacuous origins=0} ->[n199]] =
      add
        a=t288 {derived verify=unproved (too large)} <-n165
        b=t318 {derived verify=unproved (too large)} <-n196
    n199 {pt2=root[65] torch.ops.aten.relu.default verify=vacuous}: [t321 f32 [H=7
                                                                      W=7
                                                                      C=512] {derived verify=vacuous origins=0} ->[n200]] =
      relu x=t320 {derived verify=vacuous origins=0} <-n198
    n200 {pt2=root[66] torch.ops.aten.mean.dim verify=unproved (over max_rounds) [sampled 4]}: [t322 f32 [C=512] {pt2=root:view verify=unproved (over max_rounds) [sampled 4]} ->[n173]] =
      mean
        x=t321 {derived verify=vacuous origins=0} <-n199
        params={dims=[W, H]; keepdim=true}
    group g42 torch.ops.aten.addmm.default:
      n173 {pt2=root[69] torch.ops.aten.addmm.default verify=unproved (over max_nodes) [sampled 4]}: [t296 f32 [C=1000] {pt2=root:addmm verify=unproved (over max_nodes) [sampled 4]}] =
        linear
          x=t322 {pt2=root:view verify=unproved (over max_rounds) [sampled 4]} <-n200
          weight=t295 {derived verify=unproved (too large)} <-n174
          bias=t61 {pt2=root:p_fc_bias target=fc.bias verify=proved (structural) [sampled 4]}
          params={in_features=512}
  outputs:
    [t296 f32 [C=1000] {pt2=root:addmm verify=unproved (over max_nodes) [sampled 4]} <-n173]
