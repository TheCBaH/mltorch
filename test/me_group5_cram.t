Group 5 (op5.md): `silu.default`, `hardsigmoid.default` and `hardswish.default`
through the payload-free export path.

UNGATED and hand-built, same reason as `me_group2_cram.t`/`me_group3_cram.t`:
no model this repository can download serializes any of the three functional
targets (op5-impl F1) -- efficientnet_b0-b5 serialize `silu_.default` instead,
and mobilenet_v3_small exports `hardswish.default` pre-decomposed into
`mul(x, div_scalar(clamp(add_scalar(x,3),0,6), 6))`. This is the only place
these three targets reach Model Explorer at all.

  $ python3 -c "
  > import json
  > def tm(sizes):
  >     return {'dtype': 7, 'sizes': [{'as_int': s} for s in sizes],
  >             'requires_grad': False, 'device': {'type': 'cpu'},
  >             'strides': [{'as_int': 1}], 'storage_offset': {'as_int': 0},
  >             'layout': 7}
  > def t(n): return {'as_tensor': {'name': n}}
  > def arg(name, a): return {'name': name, 'arg': a, 'kind': 1}
  > def node(target, ins, out, stack):
  >     return {'target': target, 'inputs': ins, 'outputs': [t(out)],
  >             'metadata': {'nn_module_stack': stack}}
  > # x [1,4,4,4] -silu-> y1 -hardsigmoid-> y2 -hardswish-> y
  > shapes = {'x': [1, 4, 4, 4], 'y1': [1, 4, 4, 4], 'y2': [1, 4, 4, 4],
  >           'y': [1, 4, 4, 4]}
  > nodes = [
  >   node('torch.ops.aten.silu.default',
  >        [arg('self', t('x'))], 'y1', 'L__self__,,M;L__self__act1,act1,S'),
  >   node('torch.ops.aten.hardsigmoid.default',
  >        [arg('self', t('y1'))], 'y2', 'L__self__,,M;L__self__act2,act2,H'),
  >   node('torch.ops.aten.hardswish.default',
  >        [arg('self', t('y2'))], 'y', 'L__self__,,M;L__self__act3,act3,W'),
  > ]
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
  > json.dump(prog, open('group5.json', 'w'))"

  $ ../bin/native_graph.exe visualize --model group5.json --output session.json

The complete capability vector: every stage available, Native4D included --
none of the three ops names an axis or carries a shape, so nothing here needs a
payload to relay.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > for c in s['capabilities']:
  >     st = c['status']
  >     detail = st['state']
  >     if st['state'] == 'available':
  >         detail += ' ' + st['payload']['kind']
  >     elif st['state'] == 'unavailable':
  >         detail += ' ' + st['reason']
  >     print('%-28s %s' % (c['key'], detail))"
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

The SOURCE view: one node per serialized target, with the namespace taken off
`nn_module_stack`.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['pt2/root']
  > for n in g['nodes']:
  >     if n['label'] in ('input', 'constant', 'output'): continue
  >     print('%-40s ns=%-6s in=%d' % (n['label'], n['namespace'], len(n.get('incomingEdges', []))))"
  torch.ops.aten.silu.default              ns=act1   in=1
  torch.ops.aten.hardsigmoid.default       ns=act2   in=1
  torch.ops.aten.hardswish.default         ns=act3   in=1

The IMPORTED native graph, with every op's parameters: no permutes at all,
unlike Group 2/3's conv/pool/norm/linear chains, since none of these three ops
touches layout.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['g/native/000']
  > for n in g['nodes']:
  >     attrs = {a['key']: a['value'] for a in n.get('attrs', [])}
  >     params = attrs.get('params')
  >     if params is None: continue
  >     print('%-3s %-14s %s' % (n['id'], n['label'], params))"
  n0  Silu           silu x=t0
  n1  Hardsigmoid    hardsigmoid x=t1
  n2  Hardswish      hardswish x=t2

Provenance and output metadata. Unlike Group 2's conv/pool/norm/linear, none of
these three ops needs a relayout permute, so the bridge arm builds them
directly with no `Graph_builder.group` wrapper -- the namespace is empty
(matching `sub.Tensor`'s ungrouped arm in Group 3, which is why that cram never
bothers checking it) rather than group-qualified. The shape stays
`[H=4 W=4 C=4]` throughout, since all three ops are shape-preserving.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['g/native/000']
  > for n in g['nodes']:
  >     if n['label'] in ('input', 'constant', 'output'): continue
  >     shape = ''
  >     for m in n.get('outputsMetadata', []):
  >         for a in m.get('attrs', []):
  >             if a['key'] == 'shape': shape = a['value']
  >     print('%-3s %-14s ns=%-38s %s' % (n['id'], n['label'], n['namespace'], shape))"
  n0  Silu           ns=                                       [H=4 W=4 C=4]
  n1  Hardsigmoid    ns=                                       [H=4 W=4 C=4]
  n2  Hardswish      ns=                                       [H=4 W=4 C=4]

Stable slot ids: every incoming edge names its source node, the output slot it
reads, and the input position it feeds -- the chain is wired source -> silu ->
hardsigmoid -> hardswish, each reading the previous node's sole output at slot 0.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['g/native/000']
  > for n in g['nodes']:
  >     if n['label'] in ('input', 'constant', 'output'): continue
  >     e = [(x['sourceNodeId'], x['sourceNodeOutputId'], x['targetNodeInputId'])
  >          for x in n.get('incomingEdges', [])]
  >     print('%-3s %-14s %s' % (n['id'], n['label'], e))"
  n0  Silu           [('in:t0', '0', 't0')]
  n1  Hardsigmoid    [('n0', '0', 't1')]
  n2  Hardswish      [('n1', '0', 't2')]
