Group 8 (op8.md): `scaled_dot_product_attention.default` through the
payload-free export path.

UNGATED and hand-built for a stronger reason than
`me_group2_cram.t`/`me_group3_cram.t`/`me_group5_cram.t`/`me_group6_cram.t`'s:
those targets are simply absent from every downloadable model's graph, but
this one's NAME appears 600-1200 times inside the four `vit_*` graphs'
`from_node` provenance and its actual TARGET zero times -- every exporter
decomposes it into twelve primitives (view/expand/permute/mul.Scalar/bmm/
logical_not/_softmax/eq.Scalar/any.dim/full_like/where.self/clone) before
writing model.json (op8-impl.md F1). This fixture is therefore the row's
only Model Explorer evidence, and will remain so until the decomposition
(out of scope here) is separately supported.

  $ ./cram_probe.exe fixture group8

  $ ../bin/native_graph.exe visualize --model group8.json --output session.json

The complete capability vector: every stage available, Native4D included.
This graph's batch is D = 1 -- SDPA's batch axis genuinely is D (heads are on
H), but at extent 1 that is admissible, the same way `me_group7_cram.t`'s
layer_norm graph is admissible at D = 1. `test/native4d/domain_test.ml`'s
"domain: sdpa batch extent" pins the D > 1 rejection this row does not
exercise.

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
`nn_module_stack`. `attn1` has three incoming edges (no mask); `attn2` has
four (query, key, value, mask) -- the state every possible corpus node would
be in, if any model reached this target directly instead of decomposing it.

  $ ./cram_probe.exe source 46 session.json
  torch.ops.aten.scaled_dot_product_attention.default ns=attn1  in=3
  torch.ops.aten.scaled_dot_product_attention.default ns=attn2  in=4

The IMPORTED native graph. `params` is the op's own `pp` verbatim, so what is
pinned here is `n0`'s default scale and `n1`'s explicit one.

  $ ./cram_probe.exe params 5 session.json
  n0  Sdpa  sdpa query=t0 key=t1 value=t2 mask=none params={scale=default}
  n1  Sdpa  sdpa query=t4 key=t1 value=t2 mask=t3 params={scale=explicit(0.1)}

THE OPERAND ORDER, pinned by the edge list: query, key, value, THEN mask when
present -- `n1`'s four incoming edges in exactly that order, at stable slot
ids. `n0`'s mask stays absent rather than a materialized ones-shaped tensor:
[Graph_ir]'s `Sdpa` carries `mask : Tensor_ref.t option`, and a fourth
constant edge here would mean this importer and `Native_interp` built
structurally different graphs for the same absent-mask node.

  $ ./cram_probe.exe edges 5 session.json
  n0  Sdpa  [('in:t0', '0', 't0'), ('const:t1', '0', 't1'), ('const:t2', '0', 't2')]
  n1  Sdpa  [('n0', '0', 't4'), ('const:t1', '0', 't1'), ('const:t2', '0', 't2'), ('const:t3', '0', 't3')]

Output shape: `Sdpa`'s output is exactly `query_shape` (Ev = E, the
flash-oracle constraint that keeps the shape well-defined), so both nodes
keep query's `[Wq=2, C=4]`.

  $ ./cram_probe.exe outputs 5 session.json
  n0  Sdpa  outputs=1 ['[W=2 C=4]']
  n1  Sdpa  outputs=1 ['[W=2 C=4]']

NO NATIVE4D DIAGNOSTIC remains for either node -- confirming the admission
above is a real one, not a coincidence of the capability vector: at D = 1
neither `check_sdpa` nor `check_shapes` has anything left to reject.

  $ ./cram_probe.exe sdpa-diagnostics session.json
