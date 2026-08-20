Group 6 (op6.md): `pad.default` and `slice.Tensor` through the payload-free
export path.

UNGATED and hand-built, same reason as `me_group2_cram.t`/`me_group3_cram.t`/
`me_group5_cram.t`: no model this repository can download serializes either
target -- not the 23 core-ATen graphs in `modules/pytorch.models.pt2`, not the
five downloadable release models (op6-impl F1). This is the only place either
target reaches Model Explorer at all.

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
  > # x [1,4,4,4] -pad(W by 1,2; H by -1,0)-> y1 [1,4,3,7] -slice(W, 1..7 step 2)-> y [1,4,3,3]
  > shapes = {'x': [1, 4, 4, 4], 'y1': [1, 4, 3, 7], 'y': [1, 4, 3, 3]}
  > nodes = [
  >   node('torch.ops.aten.pad.default',
  >        [arg('self', t('x')), arg('pad', {'as_ints': [1, 2, -1, 0]}),
  >         arg('mode', {'as_string': 'constant'}), arg('value', {'as_float': 0.5})],
  >        'y1', 'L__self__,,M;L__self__pad,pad,P'),
  >   node('torch.ops.aten.slice.Tensor',
  >        [arg('self', t('y1')), arg('dim', {'as_int': -1}),
  >         arg('start', {'as_sym_int': {'as_int': 1}}),
  >         arg('end', {'as_int': 99}), arg('step', {'as_int': 2})],
  >        'y', 'L__self__,,M;L__self__sel,sel,S'),
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
  > json.dump(prog, open('group6.json', 'w'))"

  $ ../bin/native_graph.exe visualize --model group6.json --output session.json

The complete capability vector: every stage available, Native4D included. Both
ops name axes, and on a rank-4 input the used axes are the innermost four --
`D, H, W, C` -- so the ones these two name (`W` and `C`) are all inside the
dialect. The refusal below is the contrast.

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
`nn_module_stack`.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['pt2/root']
  > for n in g['nodes']:
  >     if n['label'] in ('input', 'constant', 'output'): continue
  >     print('%-40s ns=%-6s in=%d' % (n['label'], n['namespace'], len(n.get('incomingEdges', []))))"
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

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['g/native/000']
  > for n in g['nodes']:
  >     attrs = {a['key']: a['value'] for a in n.get('attrs', [])}
  >     params = attrs.get('params')
  >     if params is None: continue
  >     print('%-3s %-8s %s' % (n['id'], n['label'], params))"
  n0  Pad      pad x=t0 params={pads=[W:-1,0, C:1,2] mode=constant(0.5)}
  n1  Slice    slice x=t1 params={axis=C start=1 stop=7 step=2}

Output metadata: the shapes the two ops derive rather than read. Pad crops `W`
from 4 to 3 and grows `C` from 4 to 7; the slice then takes `[1, 7)` of `C` at
step 2, which is 3 BY THE CEILING -- a floor would print `C=2` here, and the
span is 6 so the two disagree only because of the offset start.

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
  >     print('%-3s %-8s ns=%-10s %s' % (n['id'], n['label'], n['namespace'], shape))"
  n0  Pad      ns=           [H=4 W=3 C=7]
  n1  Slice    ns=           [H=4 W=3 C=3]

Stable slot ids: source -> pad -> slice, each reading the previous node's sole
output at slot 0.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['g/native/000']
  > for n in g['nodes']:
  >     if n['label'] in ('input', 'constant', 'output'): continue
  >     e = [(x['sourceNodeId'], x['sourceNodeOutputId'], x['targetNodeInputId'])
  >          for x in n.get('incomingEdges', [])]
  >     print('%-3s %-8s %s' % (n['id'], n['label'], e))"
  n0  Pad      [('in:t0', '0', 't0')]
  n1  Slice    [('n0', '0', 't1')]

An axis the dialect cannot name. `slice` along `dim=0` of a rank-5 tensor is
the frame's `T` under right-alignment, so Native imports it and Native4D
refuses it BY NAME -- the actionable diagnostic, rather than a consequence like
"some tensor has extent on T". The same contrast `me_group3_cram.t` draws for
`transpose.int`.

  $ python3 -c "
  > import json
  > def tm(sizes):
  >     return {'dtype': 7, 'sizes': [{'as_int': s} for s in sizes],
  >             'requires_grad': False, 'device': {'type': 'cpu'},
  >             'strides': [{'as_int': 1}], 'storage_offset': {'as_int': 0},
  >             'layout': 7}
  > def t(n): return {'as_tensor': {'name': n}}
  > def arg(name, a): return {'name': name, 'arg': a, 'kind': 1}
  > shapes = {'x': [3, 1, 2, 2, 2], 'y': [2, 1, 2, 2, 2]}
  > nodes = [{'target': 'torch.ops.aten.slice.Tensor',
  >           'inputs': [arg('self', t('x')), arg('dim', {'as_int': 0}),
  >                      arg('end', {'as_int': 2})],
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
    diagnostic: outside_dialect_domain | node n0: axis T is outside the N/H/W/C dialect

A configuration that is legal upstream and outside the Native domain: a crop
that empties an axis. ATen returns a size-0 tensor and the engine has no empty
extent, so the import fails outright -- there is no session to inspect, unlike
the axis case above. That difference is the point: an axis outside the dialect
still has a Native graph to show, and a shape with no Native form does not.

  $ python3 -c "
  > import json
  > def tm(sizes):
  >     return {'dtype': 7, 'sizes': [{'as_int': s} for s in sizes],
  >             'requires_grad': False, 'device': {'type': 'cpu'},
  >             'strides': [{'as_int': 1}], 'storage_offset': {'as_int': 0},
  >             'layout': 7}
  > def t(n): return {'as_tensor': {'name': n}}
  > def arg(name, a): return {'name': name, 'arg': a, 'kind': 1}
  > shapes = {'x': [1, 4, 2, 4], 'y': [1, 4, 2, 4]}
  > nodes = [{'target': 'torch.ops.aten.pad.default',
  >           'inputs': [arg('self', t('x')), arg('pad', {'as_ints': [0, 0, -1, -2]})],
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
  > import json; json.dump(prog, open('empty.json', 'w'))"

  $ ../bin/native_graph.exe visualize --model empty.json --output empty.session.json
  native_graph: pad of axis W by (-1, -2) over extent 2 leaves -1 elements; the engine has no empty extent
  [123]

The same boundary reached through `slice` instead, so the two structural ops
are shown refusing for one reason rather than two.

  $ python3 -c "
  > import json
  > def tm(sizes):
  >     return {'dtype': 7, 'sizes': [{'as_int': s} for s in sizes],
  >             'requires_grad': False, 'device': {'type': 'cpu'},
  >             'strides': [{'as_int': 1}], 'storage_offset': {'as_int': 0},
  >             'layout': 7}
  > def t(n): return {'as_tensor': {'name': n}}
  > def arg(name, a): return {'name': name, 'arg': a, 'kind': 1}
  > shapes = {'x': [1, 4, 2, 4], 'y': [1, 4, 2, 4]}
  > nodes = [{'target': 'torch.ops.aten.slice.Tensor',
  >           'inputs': [arg('self', t('x')), arg('dim', {'as_int': 3}),
  >                      arg('start', {'as_int': 2}), arg('end', {'as_int': 2})],
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
  > import json; json.dump(prog, open('empty-slice.json', 'w'))"

  $ ../bin/native_graph.exe visualize --model empty-slice.json --output empty-slice.session.json
  native_graph: slice of axis C [2, 2) step 1 over extent 4 selects 0 elements; the engine has no empty extent
  [123]
