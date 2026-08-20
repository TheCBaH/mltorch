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

  $ python3 -c "
  > import json
  > def tm(sizes):
  >     return {'dtype': 7, 'sizes': [{'as_int': s} for s in sizes],
  >             'requires_grad': False, 'device': {'type': 'cpu'},
  >             'strides': [{'as_int': 1}], 'storage_offset': {'as_int': 0},
  >             'layout': 7}
  > def t(n): return {'as_tensor': {'name': n}}
  > def arg(name, a): return {'name': name, 'arg': a, 'kind': 1}
  > # x [1,4,8] -layer_norm(w,b,eps=1e-5)-> y1 -native_layer_norm(w,b,eps=1e-6)-> y
  > # native_layer_norm's mean/rstd are declared and read by nothing.
  > shapes = {'x': [1, 4, 8], 'w': [8], 'b': [8], 'y1': [1, 4, 8], 'y': [1, 4, 8]}
  > nodes = [
  >   {'target': 'torch.ops.aten.layer_norm.default',
  >    'inputs': [arg('input', t('x')), arg('normalized_shape', {'as_ints': [8]}),
  >               arg('weight', t('w')), arg('bias', t('b')),
  >               arg('eps', {'as_float': 1e-05}),
  >               arg('cudnn_enable', {'as_bool': True})],
  >    'outputs': [t('y1')],
  >    'metadata': {'nn_module_stack': 'L__self__,,M;L__self__ln1,ln1,LN'}},
  >   {'target': 'torch.ops.aten.native_layer_norm.default',
  >    'inputs': [arg('input', t('y1')), arg('normalized_shape', {'as_ints': [8]}),
  >               arg('weight', t('w')), arg('bias', t('b')),
  >               arg('eps', {'as_float': 1e-06})],
  >    'outputs': [t('y'), t('mean'), t('rstd')],
  >    'metadata': {'nn_module_stack': 'L__self__,,M;L__self__ln2,ln2,LN'}},
  > ]
  > prog = {'graph_module': {
  >           'graph': {'inputs': [t('x'), t('w'), t('b')], 'outputs': [t('y')],
  >                     'nodes': nodes,
  >                     'tensor_values': {k: tm(v) for k, v in shapes.items()},
  >                     'sym_int_values': {}, 'sym_bool_values': {},
  >                     'is_single_tensor_return': True},
  >           'signature': {'input_specs': [
  >                           {'user_input': {'arg': t('x')}},
  >                           {'parameter': {'arg': {'name': 'w'},
  >                                          'parameter_name': 'ln.weight'}},
  >                           {'parameter': {'arg': {'name': 'b'},
  >                                          'parameter_name': 'ln.bias'}}],
  >                         'output_specs': [{'user_output': {'arg': t('y')}}]},
  >           'module_call_graph': []},
  >         'opset_version': {'aten': 15}, 'range_constraints': {},
  >         'schema_version': {'major': 8, 'minor': 5}}
  > json.dump(prog, open('group7.json', 'w'))"

  $ ../bin/native_graph.exe visualize --model group7.json --output session.json

The complete capability vector: every stage available, Native4D included. A
rank-3 input right-aligns onto `H, W, C`, so the single normalized axis is `C`
and is inside the dialect. The refusal at the end of this file is the contrast.

  $ caps() {
  >   python3 -c "
  > import json
  > s = json.load(open('$1'))
  > for c in s['capabilities']:
  >     st = c['status']
  >     detail = st['state']
  >     if st['state'] == 'available': detail += ' ' + st['payload']['kind']
  >     elif st['state'] == 'unavailable': detail += ' ' + st['reason']
  >     print('%-28s %s' % (c['key'], detail))
  > for d in s['diagnostics']:
  >     if d['code'] != 'unsupported_graph_shape': print('  diagnostic:', d['code'], '|', d['message'])"
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
  feature:loop_ir              unavailable not_implemented
  feature:codegen              unavailable not_implemented

The SOURCE view: one node per serialized target, namespace off
`nn_module_stack`. Three incoming edges each -- the input and both affine
operands, which is the state all 148 corpus nodes are in.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['pt2/root']
  > for n in g['nodes']:
  >     if n['label'] in ('input', 'constant', 'output'): continue
  >     print('%-46s ns=%-6s in=%d' % (n['label'], n['namespace'], len(n.get('incomingEdges', []))))"
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

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['g/native/000']
  > for n in g['nodes']:
  >     attrs = {a['key']: a['value'] for a in n.get('attrs', [])}
  >     params = attrs.get('params')
  >     if params is None: continue
  >     print('%-3s %-12s %s' % (n['id'], n['label'], params))"
  n0  Layer_norm   layer_norm x=t0 weight=t1 bias=t2 params={dims=[C]; eps=1e-05}
  n1  Layer_norm   layer_norm x=t3 weight=t1 bias=t2 params={dims=[C]; eps=1e-06}

THE DEAD OUTPUTS. `native_layer_norm` declares three and `n1` has ONE -- the
`mean` and `rstd` edges do not exist in the native graph at all, so there is no
node to render them from and nothing downstream could read one. That is sound
only because nothing does: a graph reading either is refused at import with
`Live_layer_norm_stats`, which is why this rendering is a fact about the graph
rather than a hope about it.

Layer norm rescales, so both outputs keep the input's full shape.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['g/native/000']
  > for n in g['nodes']:
  >     if n['label'] in ('input', 'constant', 'output'): continue
  >     shapes = []
  >     for m in n.get('outputsMetadata', []):
  >         for a in m.get('attrs', []):
  >             if a['key'] == 'shape': shapes.append(a['value'])
  >     print('%-3s %-12s outputs=%d %s' % (n['id'], n['label'], len(shapes), shapes))"
  n0  Layer_norm   outputs=1 ['[W=4 C=8]']
  n1  Layer_norm   outputs=1 ['[W=4 C=8]']

Stable slot ids, and the affine operands SHARED between the two nodes: both
read `t1` and `t2` -- the two `parameter` input specs, which is why the source
prefix is `const:` and not `in:` -- at slots 1 and 2. An importer that
materialized a ones/zeros tensor per node would show four extra constants here
instead, and the two arms would build structurally different graphs for the same
node.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['g/native/000']
  > for n in g['nodes']:
  >     if n['label'] in ('input', 'constant', 'output'): continue
  >     e = [(x['sourceNodeId'], x['sourceNodeOutputId'], x['targetNodeInputId'])
  >          for x in n.get('incomingEdges', [])]
  >     print('%-3s %-12s %s' % (n['id'], n['label'], e))"
  n0  Layer_norm   [('in:t0', '0', 't0'), ('const:t1', '0', 't1'), ('const:t2', '0', 't2')]
  n1  Layer_norm   [('n0', '0', 't3'), ('const:t1', '0', 't1'), ('const:t2', '0', 't2')]

An axis the dialect cannot name. A rank-4 input right-aligns onto `D, H, W, C`,
so normalizing over all four names `D` -- Native imports it and Native4D
refuses it BY NAME, the actionable diagnostic rather than a consequence like
"some tensor has extent on D". The same contrast `me_group6_cram.t` draws for
`slice.Tensor`.

  $ python3 -c "
  > import json
  > def tm(sizes):
  >     return {'dtype': 7, 'sizes': [{'as_int': s} for s in sizes],
  >             'requires_grad': False, 'device': {'type': 'cpu'},
  >             'strides': [{'as_int': 1}], 'storage_offset': {'as_int': 0},
  >             'layout': 7}
  > def t(n): return {'as_tensor': {'name': n}}
  > def arg(name, a): return {'name': name, 'arg': a, 'kind': 1}
  > shapes = {'x': [2, 3, 4, 5], 'y': [2, 3, 4, 5]}
  > nodes = [{'target': 'torch.ops.aten.layer_norm.default',
  >           'inputs': [arg('input', t('x')),
  >                      arg('normalized_shape', {'as_ints': [2, 3, 4, 5]}),
  >                      arg('eps', {'as_float': 1e-05})],
  >           'outputs': [t('y')], 'metadata': {}}]
  > prog = {'graph_module': {
  >           'graph': {'inputs': [t('x')], 'outputs': [t('y')], 'nodes': nodes,
  >                     'tensor_values': {k: tm(v) for k, v in shapes.items()},
  >                     'sym_int_values': {}, 'sym_bool_values': {},
  >                     'is_single_tensor_return': True},
  >           'signature': {'input_specs': [{'user_input': {'arg': t('x')}}],
  >                         'output_specs': [{'user_output': {'arg': t('y')}}]},
  >           'module_call_graph': []},
  >         'opset_version': {'aten': 15}, 'range_constraints': {},
  >         'schema_version': {'major': 8, 'minor': 5}}
  > import json; json.dump(prog, open('outside.json', 'w'))"

  $ ../bin/native_graph.exe visualize --model outside.json --output outside.session.json

  $ caps outside.session.json | grep -E 'stage:(source|initial_native|native4d)|diagnostic'
  stage:source                 available graph
  stage:initial_native         available graph
  stage:native4d               unavailable outside_dialect_domain
    diagnostic: outside_dialect_domain | node n0: axis D is outside the N/H/W/C dialect
