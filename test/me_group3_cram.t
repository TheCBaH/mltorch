Group 3 (op3.md): `sub.Tensor`, `_unsafe_view.default` and `transpose.int`
through the payload-free export path.

UNGATED and hand-built, for the same reason as `me_group2_cram.t`: no model
this repository can download serializes any of these three targets (every
release model is exported post-decomposition), so this is the only place they
reach Model Explorer at all.

Three small models, following `unbind_native_cram.t`'s per-case style rather
than `me_group2_cram.t`'s single chained pipeline: Group 3's targets do not
compose into a realistic layer sequence the way conv/pool/norm/linear do.

`group3.json` chains a broadcast `sub.Tensor`, a scalar-spelled `sub.Tensor`,
and a rank-changing `_unsafe_view.default` carrying a `-1` -- every node stays
inside the four-axis dialect, so this is the ACCEPTED case.

`group3-transpose.json` and `group3-transpose-refused.json` isolate
`transpose.int`'s two questions separately, per op3-impl.md's round-1 review
response: both use `[1,h,w,c]` with DISTINCT, non-unit `h`/`w`/`c` (a
conventional `[2,3,4,5]` would put a real batch on `D` and answer the
shape-domain question instead of the axis-domain one). `dims=(1,2)` stays
inside the dialect; `dims=(0,1)` names axis `D` and is refused by the axis
check specifically, not by the shape check.

  $ python3 -c "
  > import json
  > def tm(sizes):
  >     return {'dtype': 7, 'sizes': [{'as_int': s} for s in sizes],
  >             'requires_grad': False, 'device': {'type': 'cpu'},
  >             'strides': [{'as_int': 1}], 'storage_offset': {'as_int': 0},
  >             'layout': 7}
  > def t(n): return {'as_tensor': {'name': n}}
  > def arg(name, a): return {'name': name, 'arg': a, 'kind': 1}
  > def node(target, ins, out):
  >     return {'target': target, 'inputs': ins, 'outputs': [t(out)], 'metadata': {}}
  > def w(name, prog): json.dump(prog, open(name + '.json', 'w'))
  > def program(nodes, tv, params, out):
  >     specs = [{'user_input': {'arg': t('x')}}] + [
  >         {'parameter': {'arg': {'name': p}, 'parameter_name': p}} for p in params]
  >     return {'graph_module': {
  >               'graph': {'inputs': [t('x')] + [t(p) for p in params],
  >                         'outputs': [t(out)], 'nodes': nodes,
  >                         'tensor_values': tv, 'sym_int_values': {},
  >                         'sym_bool_values': {}, 'is_single_tensor_return': True},
  >               'signature': {'input_specs': specs,
  >                             'output_specs': [{'user_output': {'arg': t(out)}}]},
  >               'module_call_graph': []},
  >             'opset_version': {'aten': 15}, 'range_constraints': {},
  >             'schema_version': {'major': 8, 'minor': 5}}
  > # x [1,4,8,8] (D=1,H=4,W=8,C=8) -sub(broadcast, other=[8])-> y1 [1,4,8,8]
  > #   -sub(scalar=3)-> y2 [1,4,8,8] -_unsafe_view([-1,32])-> y [8,32]
  > nodes = [
  >   node('torch.ops.aten.sub.Tensor',
  >        [arg('self', t('x')), arg('other', t('other'))], 'y1'),
  >   node('torch.ops.aten.sub.Tensor',
  >        [arg('self', t('y1')), arg('other', {'as_int': 3})], 'y2'),
  >   node('torch.ops.aten._unsafe_view.default',
  >        [arg('self', t('y2')), arg('size', {'as_ints': [-1, 32]})], 'y'),
  > ]
  > tv = {'x': tm([1, 4, 8, 8]), 'other': tm([8]), 'y1': tm([1, 4, 8, 8]),
  >       'y2': tm([1, 4, 8, 8]), 'y': tm([8, 32])}
  > w('group3', program(nodes, tv, ['other'], 'y'))
  > tr_nodes = lambda d0, d1: [node('torch.ops.aten.transpose.int',
  >     [arg('self', t('x')), arg('dim0', {'as_int': d0}), arg('dim1', {'as_int': d1})], 'y')]
  > tr_tv = {'x': tm([1, 3, 4, 5]), 'y': tm([1, 4, 3, 5])}
  > w('group3-transpose', program(tr_nodes(1, 2), tr_tv, [], 'y'))
  > tr_tv0 = {'x': tm([1, 3, 4, 5]), 'y': tm([3, 1, 4, 5])}
  > w('group3-transpose-refused', program(tr_nodes(0, 1), tr_tv0, [], 'y'))
  > # A malformed _unsafe_view target (two -1s, op3-impl.md F1) -- Me_classify
  > # needs no Group-3 change (it matches #Native_interp.malformed wholesale),
  > # confirmed below rather than assumed.
  > bad_nodes = [node('torch.ops.aten._unsafe_view.default',
  >     [arg('self', t('x')), arg('size', {'as_ints': [-1, -1]})], 'y')]
  > w('group3-malformed', program(bad_nodes, {'x': tm([1, 4, 8, 8]), 'y': tm([1, 4, 8, 8])}, [], 'y'))"

Each model's outcome, and the full capability vector for `group3.json`: every
stage available, Native4D included -- the accept case op3-impl.md's exit
condition asks for. `group3-malformed` confirms, rather than assumes,
op3-impl.md's claim that Me_classify needs no Group-3 change: it matches
`#Native_interp.malformed` wholesale, so commit 1's new `` `Bad_view `` row is
classified already, and reads exactly like every other malformed row here --
refused before a session is built at all, which is what a decoder that
accepted a graph it should not have is supposed to do.

  $ for m in group3 group3-transpose group3-transpose-refused group3-malformed; do
  >   printf '%-28s ' "$m"
  >   if ../bin/native_graph.exe visualize --model $m.json --output $m.session.json >/dev/null 2>err.txt; then
  >     echo lowered
  >   else
  >     head -1 err.txt
  >   fi
  > done
  group3                       lowered
  group3-transpose             lowered
  group3-transpose-refused     lowered
  group3-malformed             native_graph: malformed PT2 graph: view size [-1, -1]: view size [-1, -1] has more than one inferred (-1) dimension

  $ caps() {
  >   python3 -c "
  > import json
  > s = json.load(open('$1'))
  > for c in s['capabilities']:
  >     st = c['status']
  >     detail = st['state']
  >     if st['state'] == 'available': detail += ' ' + st['payload']['kind']
  >     elif st['state'] == 'unavailable': detail += ' ' + st['reason']
  >     if c['key'] in ('stage:source', 'stage:initial_native', 'stage:native4d'):
  >         print('%-24s %s' % (c['key'], detail))
  > for d in s['diagnostics']:
  >     if d['code'] != 'unsupported_graph_shape': print('  diagnostic:', d['code'], '|', d['message'])"
  > }
  $ caps group3.session.json
  stage:source             available graph
  stage:initial_native     available graph
  stage:native4d           available graph
  $ caps group3-transpose.session.json
  stage:source             available graph
  stage:initial_native     available graph
  stage:native4d           available graph
  $ caps group3-transpose-refused.session.json
  stage:source             available graph
  stage:initial_native     available graph
  stage:native4d           unavailable outside_dialect_domain
    diagnostic: outside_dialect_domain | node n0: axis D is outside the N/H/W/C dialect

The IMPORTED native graph for `group3.json`, with every op's parameters: the
broadcast operand's own shape, the scalar sub's negated value, and the
`_unsafe_view` target -- an importer that dropped the broadcast, used `+3`
instead of `-3`, or resolved the `-1` from the wrong product would show up
here.

  $ python3 -c "
  > import json
  > s = json.load(open('group3.session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['g/native/000']
  > for n in g['nodes']:
  >     attrs = {a['key']: a['value'] for a in n.get('attrs', [])}
  >     params = attrs.get('params')
  >     if params is None: continue
  >     print('%-3s %-8s %s' % (n['id'], n['label'], params))"
  n0  Sub      sub a=t0 b=t1
  n1  Add_scalar add_scalar x=t2 scalar=-3
  n2  Reshape  reshape x=t3 params={shape=[W=8 C=32]}

Output metadata carries the shape a reader checks each parameter against --
the broadcast sub's output stays at `x`'s shape (leading `D=1` trimmed by
`Vec6.pp_shape`, not carried away), the scalar sub leaves it unchanged, and
the `_unsafe_view` target is exactly what `[-1, 32]` resolves to against 256
elements.

  $ python3 -c "
  > import json
  > s = json.load(open('group3.session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['g/native/000']
  > for n in g['nodes']:
  >     if n['label'] in ('input', 'constant', 'output'): continue
  >     shape = ''
  >     for m in n.get('outputsMetadata', []):
  >         for a in m.get('attrs', []):
  >             if a['key'] == 'shape': shape = a['value']
  >     print('%-3s %-10s %s' % (n['id'], n['label'], shape))"
  n0  Sub        [H=4 W=8 C=8]
  n1  Add_scalar [H=4 W=8 C=8]
  n2  Reshape    [W=8 C=32]

Stable slot ids and wiring order: `other` is a captured parameter (the
broadcast weight a real model would serialize this way), so its edge sources
from a CONSTANT input, not the graph's user input.

  $ python3 -c "
  > import json
  > s = json.load(open('group3.session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['g/native/000']
  > for n in g['nodes']:
  >     if n['label'] in ('input', 'constant', 'output'): continue
  >     e = [(x['sourceNodeId'], x['sourceNodeOutputId'], x['targetNodeInputId'])
  >          for x in n.get('incomingEdges', [])]
  >     print('%-3s %-10s %s' % (n['id'], n['label'], e))"
  n0  Sub        [('in:t0', '0', 't0'), ('const:t1', '0', 't1')]
  n1  Add_scalar [('n0', '0', 't2')]
  n2  Reshape    [('n1', '0', 't3')]

The `transpose.int` permutation itself, for the accepted case: `dims=(1,2)`
swaps `H` and `W`.

  $ python3 -c "
  > import json
  > s = json.load(open('group3-transpose.session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['g/native/000']
  > for n in g['nodes']:
  >     attrs = {a['key']: a['value'] for a in n.get('attrs', [])}
  >     params = attrs.get('params')
  >     if params is None: continue
  >     print('%-3s %-8s %s' % (n['id'], n['label'], params))"
  n0  Permute  permute x=t0 perm=[H<-W, W<-H]
