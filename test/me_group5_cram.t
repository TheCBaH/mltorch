Group 5 (op5.md): `silu.default`, `hardsigmoid.default` and `hardswish.default`
through the payload-free export path.

UNGATED and hand-built, same reason as `me_group2_cram.t`/`me_group3_cram.t`:
no model this repository can download serializes any of the three functional
targets (op5-impl F1) -- efficientnet_b0-b5 serialize `silu_.default` instead,
and mobilenet_v3_small exports `hardswish.default` pre-decomposed into
`mul(x, div_scalar(clamp(add_scalar(x,3),0,6), 6))`. This is the only place
these three targets reach Model Explorer at all.

  $ ./cram_probe.exe fixture group5

  $ ../bin/native_graph.exe visualize --model group5.json --output session.json

The complete capability vector: every stage available, Native4D included --
none of the three ops names an axis or carries a shape, so nothing here needs a
payload to relay.

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

The SOURCE view: one node per serialized target, with the namespace taken off
`nn_module_stack`.

  $ ./cram_probe.exe source 40 session.json
  torch.ops.aten.silu.default              ns=act1   in=1
  torch.ops.aten.hardsigmoid.default       ns=act2   in=1
  torch.ops.aten.hardswish.default         ns=act3   in=1

The IMPORTED native graph, with every op's parameters: no permutes at all,
unlike Group 2/3's conv/pool/norm/linear chains, since none of these three ops
touches layout.

  $ ./cram_probe.exe params 14 session.json
  n0  Silu           silu x=t0
  n1  Hardsigmoid    hardsigmoid x=t1
  n2  Hardswish      hardswish x=t2

Provenance and output metadata. Unlike Group 2's conv/pool/norm/linear, none of
these three ops needs a relayout permute, so the bridge arm builds them
directly with no `Graph_builder.group` wrapper -- the namespace is empty
(matching `sub.Tensor`'s ungrouped arm in Group 3, which is why that cram never
bothers checking it) rather than group-qualified. The shape stays
`[H=4 W=4 C=4]` throughout, since all three ops are shape-preserving.

  $ ./cram_probe.exe shapes-ns 14 38 session.json
  n0  Silu           ns=                                       [H=4 W=4 C=4]
  n1  Hardsigmoid    ns=                                       [H=4 W=4 C=4]
  n2  Hardswish      ns=                                       [H=4 W=4 C=4]

Stable slot ids: every incoming edge names its source node, the output slot it
reads, and the input position it feeds -- the chain is wired source -> silu ->
hardsigmoid -> hardswish, each reading the previous node's sole output at slot 0.

  $ ./cram_probe.exe edges 14 session.json
  n0  Silu           [('in:t0', '0', 't0')]
  n1  Hardsigmoid    [('n0', '0', 't1')]
  n2  Hardswish      [('n1', '0', 't2')]
