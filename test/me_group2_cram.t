The five Group-2 functional overloads through the payload-free export path.

UNGATED, and hand-built rather than downloaded, for a reason that is specific to
this group: NO model this repository can fetch serialises any of these targets.
resnet18, mobilenet_v2/v3_small, efficientnet_b0 and vit_b_32 are all exported
post-decomposition and carry `convolution.default`, `addmm.default` and
`max_pool2d_with_indices.default` instead. So `me_visualize_json_cram.t` reaches
none of these arms, and neither does any `interp_*_cram.t`. This is the only
place they reach Model Explorer at all.

Every parameter below is NON-DEFAULT. A round trip over default parameters would
prove nothing: it cannot distinguish a projection that carries a field from one
that drops it and re-derives the default.

  $ ./cram_probe.exe fixture group2

  $ ../bin/native_graph.exe visualize --model group2.json --output session.json

The complete capability vector. Native4D is available: every weight here is
already rank 4, so none needs the Const-SSA folding resnet18 needs (and this
payload-free file does not have); the first conv's `groups=2` used to be
Native4D's own rejection (neither 1 nor depthwise), and now legalizes to
`GroupedConv2D` (`.ai/native4d_design.md` §7.2/§8).

  $ ./cram_probe.exe caps session.json
  stage:source                 available graph
  stage:initial_native         available graph
  stage:canonical              available graph
  stage:native4d               available graph
  stage:stage_program          available graph
  stage:kernel                 available graph
  stage:fusion                 available graph
  feature:flow                 available graph
  feature:verification         not_requested
  feature:pass_audits          not_requested
  feature:fold                 unavailable requires_payloads
  feature:expression_detail    available present
  feature:generated_js         not_requested
  feature:loop_ir              unavailable not_implemented
  feature:codegen              unavailable not_implemented

The SOURCE view is the exported program's own graph: one node per serialized
target, with the namespace taken off nn_module_stack.

  $ ./cram_probe.exe source 44 session.json
  torch.ops.aten.conv2d.default                ns=conv   in=3
  torch.ops.aten.conv2d.padding                ns=same   in=2
  torch.ops.aten.max_pool2d.default            ns=pool   in=1
  torch.ops.aten.rms_norm.default              ns=norm   in=2
  torch.ops.aten.linear.default                ns=fc     in=3

The IMPORTED native graph, with every op's parameters. This is the assertion
that matters: a relayout dropped, an H/W pair transposed, a padding mode resolved
early or a channel count taken from the wrong axis all show up HERE, and in
neither the capability vector nor any node count.

  $ ./cram_probe.exe params 16 session.json
  n0  Permute          permute x=t0 perm=[H<-W, W<-C, C<-H]
  n1  Permute          permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
  n2  Conv2d           conv2d
    x=t7
    weight=t8
    bias=t2
    params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
           w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=2};
           in_channels=4;
           groups=2}
  n3  Permute          permute x=t9 perm=[H<-C, W<-H, C<-W]
  n4  Permute          permute x=t10 perm=[H<-W, W<-C, C<-H]
  n5  Permute          permute x=t3 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
  n6  Conv2d_padding   conv2d_padding
    x=t11
    weight=t12
    bias=none
    params={stride={h=1; w=1}; padding=same; dilation={h=1; w=1}; groups=1}
  n7  Permute          permute x=t13 perm=[H<-C, W<-H, C<-W]
  n8  Permute          permute x=t14 perm=[H<-W, W<-C, C<-H]
  n9  Max_pool2d       max_pool2d
    x=t15
    params={kernel={h=3; w=2};
           stride={h=2; w=1};
           pad={h=1; w=0};
           ceil_mode=false}
  n10 Permute          permute x=t16 perm=[H<-C, W<-H, C<-W]
  n11 Rms_norm         rms_norm x=t17 weight=t4 params={dims=[W, C]; eps=1e-05}
  n12 Permute          permute x=t5 perm=[N<-W, W<-N]
  n13 Linear           linear x=t18 weight=t19 bias=t6 params={in_features=5}

Provenance and output metadata. Each native node's namespace is the SERIALIZED
TARGET that produced it, group-qualified -- so every relayout permute is
attributable to the node that caused it rather than floating between two of them.
The shape beside it is what a reader checks a parameter against: the conv's
[H=4 W=6] is stride 2x1 with pad 1x0 and dilation 1x2 over an 8x8 input, and no
other reading of those arguments produces it.

  $ ./cram_probe.exe shapes-ns 16 6 session.json
  n0  Permute          ns=torch.ops.aten.conv2d.default#g1 [H=8 W=8 C=4]
  n1  Permute          ns=torch.ops.aten.conv2d.default#g1 [N=8 T=1 D=1 H=3 W=2 C=2]
  n2  Conv2d           ns=torch.ops.aten.conv2d.default#g1 [H=4 W=6 C=8]
  n3  Permute          ns=torch.ops.aten.conv2d.default#g1 [H=8 W=4 C=6]
  n4  Permute          ns=torch.ops.aten.conv2d.padding#g2 [H=4 W=6 C=8]
  n5  Permute          ns=torch.ops.aten.conv2d.padding#g2 [N=8 T=1 D=1 H=3 W=3 C=8]
  n6  Conv2d_padding   ns=torch.ops.aten.conv2d.padding#g2 [H=4 W=6 C=8]
  n7  Permute          ns=torch.ops.aten.conv2d.padding#g2 [H=8 W=4 C=6]
  n8  Permute          ns=torch.ops.aten.max_pool2d.default#g3 [H=4 W=6 C=8]
  n9  Max_pool2d       ns=torch.ops.aten.max_pool2d.default#g3 [H=2 W=5 C=8]
  n10 Permute          ns=torch.ops.aten.max_pool2d.default#g3 [H=8 W=2 C=5]
  n11 Rms_norm         ns=torch.ops.aten.rms_norm.default#g4 [H=8 W=2 C=5]
  n12 Permute          ns=torch.ops.aten.linear.default#g5 [N=3 T=1 D=1 H=1 W=1 C=5]
  n13 Linear           ns=torch.ops.aten.linear.default#g5 [H=8 W=2 C=3]

Stable slot ids: every incoming edge names its source node, the OUTPUT SLOT it
reads, and the input position it feeds. A single-output op makes slot 0 the only
answer and proves nothing, so what this pins is that the chain is wired in the
order the serialized graph declared -- conv's bias is operand 2, not operand 1.

  $ ./cram_probe.exe edges 16 session.json
  n0  Permute          [('in:t0', '0', 't0')]
  n1  Permute          [('const:t1', '0', 't1')]
  n2  Conv2d           [('n0', '0', 't7'), ('n1', '0', 't8'), ('const:t2', '0', 't2')]
  n3  Permute          [('n2', '0', 't9')]
  n4  Permute          [('n3', '0', 't10')]
  n5  Permute          [('const:t3', '0', 't3')]
  n6  Conv2d_padding   [('n4', '0', 't11'), ('n5', '0', 't12')]
  n7  Permute          [('n6', '0', 't13')]
  n8  Permute          [('n7', '0', 't14')]
  n9  Max_pool2d       [('n8', '0', 't15')]
  n10 Permute          [('n9', '0', 't16')]
  n11 Rms_norm         [('n10', '0', 't17'), ('const:t4', '0', 't4')]
  n12 Permute          [('const:t5', '0', 't5')]
  n13 Linear           [('n11', '0', 't18'), ('n12', '0', 't19'), ('const:t6', '0', 't6')]
