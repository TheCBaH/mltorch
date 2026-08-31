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

  $ python3 -c "
  > import json
  > def tm(sizes):
  >     return {'dtype': 7, 'sizes': [{'as_int': s} for s in sizes],
  >             'requires_grad': False, 'device': {'type': 'cpu'},
  >             'strides': [{'as_int': 1}], 'storage_offset': {'as_int': 0},
  >             'layout': 7}
  > def t(n): return {'as_tensor': {'name': n}}
  > def arg(name, a): return {'name': name, 'arg': a, 'kind': 1}
  > # x [1,1,2,4] (D=1,H=1,Wq=2,C=4) -sdpa(k,v)-> y1 [1,1,2,4], no mask, default scale
  > # y1        -sdpa(k,v,mask,scale=0.1)-> y [1,1,2,4], mask present, explicit scale
  > shapes = {'x': [1, 1, 2, 4], 'k': [1, 1, 3, 4], 'v': [1, 1, 3, 4],
  >           'm': [2, 3], 'y1': [1, 1, 2, 4], 'y': [1, 1, 2, 4]}
  > nodes = [
  >   {'target': 'torch.ops.aten.scaled_dot_product_attention.default',
  >    'inputs': [arg('query', t('x')), arg('key', t('k')), arg('value', t('v'))],
  >    'outputs': [t('y1')],
  >    'metadata': {'nn_module_stack': 'L__self__,,M;L__self__attn1,attn1,SDPA'}},
  >   {'target': 'torch.ops.aten.scaled_dot_product_attention.default',
  >    'inputs': [arg('query', t('y1')), arg('key', t('k')), arg('value', t('v')),
  >               arg('attn_mask', t('m')), arg('dropout_p', {'as_float': 0.0}),
  >               arg('is_causal', {'as_bool': False}),
  >               arg('scale', {'as_float': 0.1})],
  >    'outputs': [t('y')],
  >    'metadata': {'nn_module_stack': 'L__self__,,M;L__self__attn2,attn2,SDPA'}},
  > ]
  > prog = {'graph_module': {
  >           'graph': {'inputs': [t('x'), t('k'), t('v'), t('m')], 'outputs': [t('y')],
  >                     'nodes': nodes,
  >                     'tensor_values': {kk: tm(v) for kk, v in shapes.items()},
  >                     'sym_int_values': {}, 'sym_bool_values': {},
  >                     'is_single_tensor_return': True},
  >           'signature': {'input_specs': [
  >                           {'user_input': {'arg': t('x')}},
  >                           {'parameter': {'arg': {'name': 'k'},
  >                                          'parameter_name': 'attn.key'}},
  >                           {'parameter': {'arg': {'name': 'v'},
  >                                          'parameter_name': 'attn.value'}},
  >                           {'parameter': {'arg': {'name': 'm'},
  >                                          'parameter_name': 'attn.mask'}}],
  >                         'output_specs': [{'user_output': {'arg': t('y')}}]},
  >           'module_call_graph': []},
  >         'opset_version': {'aten': 15}, 'range_constraints': {},
  >         'schema_version': {'major': 8, 'minor': 5}}
  > json.dump(prog, open('group8.json', 'w'))"

  $ ../bin/native_graph.exe visualize --model group8.json --output session.json

The complete capability vector: every stage available EXCEPT Native4D, which
is the whole point of this row (F8) -- SDPA names D as its batch axis, which
the N/H/W/C dialect has no name for, and there is no legalization to fall
back to (Native4D's Bmm legalization admits only a single batch).

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
  stage:native4d               unavailable outside_dialect_domain
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
    diagnostic: outside_dialect_domain | node n0: scaled-dot-product attention's batch axis is D, which the N/H/W/C dialect has no name for; no legalization is available (Native4D's Bmm legalization admits only a single batch)

The SOURCE view: one node per serialized target, namespace off
`nn_module_stack`. `attn1` has three incoming edges (no mask); `attn2` has
four (query, key, value, mask) -- the state every possible corpus node would
be in, if any model reached this target directly instead of decomposing it.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['pt2/root']
  > for n in g['nodes']:
  >     if n['label'] in ('input', 'constant', 'output'): continue
  >     print('%-46s ns=%-6s in=%d' % (n['label'], n['namespace'], len(n.get('incomingEdges', []))))"
  torch.ops.aten.scaled_dot_product_attention.default ns=attn1  in=3
  torch.ops.aten.scaled_dot_product_attention.default ns=attn2  in=4

The IMPORTED native graph. `params` is the op's own `pp` verbatim, so what is
pinned here is `n0`'s default scale and `n1`'s explicit one.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['g/native/000']
  > for n in g['nodes']:
  >     attrs = {a['key']: a['value'] for a in n.get('attrs', [])}
  >     params = attrs.get('params')
  >     if params is None: continue
  >     print('%-3s %-5s %s' % (n['id'], n['label'], params))"
  n0  Sdpa  sdpa query=t0 key=t1 value=t2 mask=none params={scale=default}
  n1  Sdpa  sdpa query=t4 key=t1 value=t2 mask=t3 params={scale=explicit(0.1)}

THE OPERAND ORDER, pinned by the edge list: query, key, value, THEN mask when
present -- `n1`'s four incoming edges in exactly that order, at stable slot
ids. `n0`'s mask stays absent rather than a materialized ones-shaped tensor:
[Graph_ir]'s `Sdpa` carries `mask : Tensor_ref.t option`, and a fourth
constant edge here would mean this importer and `Native_interp` built
structurally different graphs for the same absent-mask node.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['g/native/000']
  > for n in g['nodes']:
  >     if n['label'] in ('input', 'constant', 'output'): continue
  >     e = [(x['sourceNodeId'], x['sourceNodeOutputId'], x['targetNodeInputId'])
  >          for x in n.get('incomingEdges', [])]
  >     print('%-3s %-5s %s' % (n['id'], n['label'], e))"
  n0  Sdpa  [('in:t0', '0', 't0'), ('const:t1', '0', 't1'), ('const:t2', '0', 't2')]
  n1  Sdpa  [('n0', '0', 't4'), ('const:t1', '0', 't1'), ('const:t2', '0', 't2'), ('const:t3', '0', 't3')]

Output shape: `Sdpa`'s output is exactly `query_shape` (Ev = E, the
flash-oracle constraint that keeps the shape well-defined), so both nodes
keep query's `[Wq=2, C=4]`.

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
  >     print('%-3s %-5s outputs=%d %s' % (n['id'], n['label'], len(shapes), shapes))"
  n0  Sdpa  outputs=1 ['[W=2 C=4]']
  n1  Sdpa  outputs=1 ['[W=2 C=4]']

Native4D BY NAME: the diagnostic says which axis and why, the actionable
answer rather than a consequence like "some tensor has extent on D" -- the
same contrast `me_group6_cram.t` and `me_group7_cram.t` draw for
`slice.Tensor` and `layer_norm`'s D-normalization case.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > for d in s['diagnostics']:
  >     if 'sdpa' in d['message'] or 'batch axis' in d['message']:
  >         print(d['code'], '|', d['message'])"
  outside_dialect_domain | node n0: scaled-dot-product attention's batch axis is D, which the N/H/W/C dialect has no name for; no legalization is available (Native4D's Bmm legalization admits only a single batch)
