Group 7 (op7.md): `layer_norm.default` and `native_layer_norm.default` through
the payload-free export path.

UNGATED and hand-built, and here the reason SPLITS -- unlike
`me_group2_cram.t`/`me_group3_cram.t`/`me_group5_cram.t`/`me_group6_cram.t`,
where no reachable model serializes the target at all.

`layer_norm.default` is that case: every ExportedProgram in the corpus lowers it
before writing model.json, so this is the only place it reaches Model Explorer.
`native_layer_norm.default` is NOT: four ViT models serialize it 148 times, and
`vit_b_32` is downloadable. It is here anyway, because a payload-free fixture
pins the rendering deterministically and because its two DEAD outputs are the
thing to see -- a real model would show them too, but only after a download and
a 839-node graph.

  $ ./cram_probe.exe fixture group7

  $ ../bin/native_graph.exe visualize --model group7.json --output session.json

The complete capability vector: every stage available, Native4D included. A
rank-3 input right-aligns onto `H, W, C`, so the single normalized axis is `C`
and is inside the dialect. The refusal at the end of this file is the contrast.

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
`nn_module_stack`. Three incoming edges each -- the input and both affine
operands, which is the state all 148 corpus nodes are in.

  $ ./cram_probe.exe source 46 session.json
  torch.ops.aten.layer_norm.default              ns=ln1    in=3
  torch.ops.aten.native_layer_norm.default       ns=ln2    in=3

The IMPORTED native graph. Both targets become the SAME `Layer_norm` node --
they differ in their argument list and their return arity, not in this
arithmetic -- and the only visible difference is the epsilon each carried.

`params` is the op's own `pp` verbatim, so what is pinned here is the operand
ORDER (`x`, then `weight`, then `bias`) and the normalized axis. Reading the
trailing extents as leading ones would print `dims=[H]`, and a swapped affine
pair would print the operands the other way round -- both shape-preserving, so
this is where they show.

  $ ./cram_probe.exe params 12 session.json
  n0  Layer_norm   layer_norm x=t0 weight=t1 bias=t2 params={dims=[C]; eps=1e-05}
  n1  Layer_norm   layer_norm x=t3 weight=t1 bias=t2 params={dims=[C]; eps=1e-06}

THE DEAD OUTPUTS. `native_layer_norm` declares three and `n1` has ONE -- the
`mean` and `rstd` edges do not exist in the native graph at all, so there is no
node to render them from and nothing downstream could read one. That is sound
only because nothing does: a graph reading either is refused at import with
`Live_layer_norm_stats`, which is why this rendering is a fact about the graph
rather than a hope about it.

Layer norm rescales, so both outputs keep the input's full shape.

  $ ./cram_probe.exe outputs 12 session.json
  n0  Layer_norm   outputs=1 ['[W=4 C=8]']
  n1  Layer_norm   outputs=1 ['[W=4 C=8]']

Stable slot ids, and the affine operands SHARED between the two nodes: both
read `t1` and `t2` -- the two `parameter` input specs, which is why the source
prefix is `const:` and not `in:` -- at slots 1 and 2. An importer that
materialized a ones/zeros tensor per node would show four extra constants here
instead, and the two arms would build structurally different graphs for the same
node.

  $ ./cram_probe.exe edges 12 session.json
  n0  Layer_norm   [('in:t0', '0', 't0'), ('const:t1', '0', 't1'), ('const:t2', '0', 't2')]
  n1  Layer_norm   [('n0', '0', 't3'), ('const:t1', '0', 't1'), ('const:t2', '0', 't2')]

An axis the dialect cannot name. A rank-4 input right-aligns onto `D, H, W, C`,
so normalizing over all four names `D` -- Native imports it and Native4D
refuses it BY NAME, the actionable diagnostic rather than a consequence like
"some tensor has extent on D". The same contrast `me_group6_cram.t` draws for
`slice.Tensor`.

  $ ./cram_probe.exe fixture outside7

  $ ../bin/native_graph.exe visualize --model outside.json --output outside.session.json

  $ caps outside.session.json | grep -E 'stage:(source|initial_native|native4d)|diagnostic'
  stage:source                 available graph
  stage:initial_native         available graph
  stage:native4d               unavailable outside_dialect_domain
    diagnostic: outside_dialect_domain | node n0: axis D is outside the N/H/W/C dialect
