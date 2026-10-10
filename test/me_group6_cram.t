Group 6 (op6.md): `pad.default` and `slice.Tensor` through the payload-free
export path.

UNGATED and hand-built, same reason as `me_group2_cram.t`/`me_group3_cram.t`/
`me_group5_cram.t`: no model this repository can download serializes either
target -- not the 23 core-ATen graphs in `modules/devcontainer.pytorch-image-models`, not the
five downloadable release models (op6-impl F1). This is the only place either
target reaches Model Explorer at all.

  $ ./cram_probe.exe fixture group6

  $ ../bin/native_graph.exe visualize --model group6.json --output session.json

The complete capability vector: every stage available, Native4D included. Both
ops name axes, and on a rank-4 input the used axes are the innermost four --
`D, H, W, C` -- so the ones these two name (`W` and `C`) are all inside the
dialect. The refusal below is the contrast.

  $ caps() {
  >   ./cram_probe.exe caps-diags "$1"
  > }
  $ caps session.json
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

The SOURCE view: one node per serialized target, namespace off
`nn_module_stack`.

  $ ./cram_probe.exe source 40 session.json
  torch.ops.aten.pad.default               ns=pad    in=1
  torch.ops.aten.slice.Tensor              ns=sel    in=1

The IMPORTED native graph, with every op's parameters. This is the whole
Group-6 lowering, readable in two lines.

The serialized pad list `[1, 2, -1, 0]` is INNERMOST-FIRST, and a rank-4 input
right-aligns onto `D, H, W, C` -- so the first pair belongs to `C` and the
second to `W`. A reversal would print `W:1,2 C:-1,0`, and a decoder that read
the pairs as unsigned would lose the crop entirely.

The slice's `dim=-1` names `C`; its `start` arrived spelled `as_sym_int` and
resolved to a value rather than being refused; and its `end=99` CLAMPED to the
extent, which is ATen's own rule. What is stored is the canonical `[1, 7)`,
which is the point of resolving at import rather than at evaluation.

  $ ./cram_probe.exe params 8 session.json
  n0  Pad      pad x=t0 params={pads=[W:-1,0, C:1,2] mode=constant(0.5)}
  n1  Slice    slice x=t1 params={axis=C start=1 stop=7 step=2}

Output metadata: the shapes the two ops derive rather than read. Pad crops `W`
from 4 to 3 and grows `C` from 4 to 7; the slice then takes `[1, 7)` of `C` at
step 2, which is 3 BY THE CEILING -- a floor would print `C=2` here, and the
span is 6 so the two disagree only because of the offset start.

  $ ./cram_probe.exe shapes-ns 8 10 session.json
  n0  Pad      ns=           [H=4 W=3 C=7]
  n1  Slice    ns=           [H=4 W=3 C=3]

Stable slot ids: source -> pad -> slice, each reading the previous node's sole
output at slot 0.

  $ ./cram_probe.exe edges 8 session.json
  n0  Pad      [('in:t0', '0', 't0')]
  n1  Slice    [('n0', '0', 't1')]

An axis the dialect cannot name. `slice` along `dim=0` of a rank-5 tensor is
the frame's `T` under right-alignment, so Native imports it and Native4D
refuses it BY NAME -- the actionable diagnostic, rather than a consequence like
"some tensor has extent on T". The same contrast `me_group3_cram.t` draws for
`transpose.int`.

  $ ./cram_probe.exe fixture outside6

  $ ../bin/native_graph.exe visualize --model outside.json --output outside.session.json

  $ caps outside.session.json | grep -E 'stage:(source|initial_native|native4d)|diagnostic'
  stage:source                 available graph
  stage:initial_native         available graph
  stage:native4d               unavailable outside_dialect_domain
    diagnostic: outside_dialect_domain | node n0: axis T is outside the N/H/W/C dialect

A configuration that is legal upstream and outside the Native domain: a crop
that empties an axis. ATen returns a size-0 tensor and the engine has no empty
extent, so the import fails outright -- there is no session to inspect, unlike
the axis case above. That difference is the point: an axis outside the dialect
still has a Native graph to show, and a shape with no Native form does not.

  $ ./cram_probe.exe fixture empty-pad

  $ ../bin/native_graph.exe visualize --model empty.json --output empty.session.json
  native_graph: pad of axis W by (-1, -2) over extent 2 leaves -1 elements; the engine has no empty extent
  [123]

The same boundary reached through `slice` instead, so the two structural ops
are shown refusing for one reason rather than two.

  $ ./cram_probe.exe fixture empty-slice

  $ ../bin/native_graph.exe visualize --model empty-slice.json --output empty-slice.session.json
  native_graph: slice of axis C [2, 2) step 1 over extent 4 selects 0 elements; the engine has no empty extent
  [123]
