Native lowering of a serialized `torch.ops.aten.unbind.int` node.

UNGATED and hand-built, for the same reason as me_visualize_unsupported_cram: what
this has to exercise is a node shape no released model in `data/pt2/` contains in a
form small enough to read, and — for the malformed variants — one no exporter emits
at all.

`unbind.int` returns `Tensor[]`, which the PyTorch serializer represents as ONE node
output of kind `as_tensors` holding every result name in order. That is a different
shape from a fixed tuple, whose elements are separate entries in `Node.outputs`, so
`Native_interp.output_names` has to flatten it rather than reject it — which is what
it used to do, uniformly, for every fixture here.

  $ python3 -c "
  > import json
  > def tm(sizes):
  >     return {'dtype': 7, 'sizes': [{'as_int': i} for i in sizes],
  >             'requires_grad': False, 'device': {'type': 'cpu'},
  >             'strides': [{'as_int': 1}], 'storage_offset': {'as_int': 0},
  >             'layout': 7}
  > def program(nodes, tv, outs=[{'as_tensor': {'name': 'u0'}}]):
  >     return {'graph_module': {
  >               'graph': {'inputs': [{'as_tensor': {'name': 'x'}}],
  >                         'outputs': outs, 'nodes': nodes, 'tensor_values': tv,
  >                         'sym_int_values': {}, 'sym_bool_values': {},
  >                         'is_single_tensor_return': True},
  >               'signature': {'input_specs': [{'user_input': {'arg': {'as_tensor': {'name': 'x'}}}}],
  >                             'output_specs': [{'user_output': {'arg': o}} for o in outs]},
  >               'module_call_graph': []},
  >             'opset_version': {'aten': 15}, 'range_constraints': {},
  >             'schema_version': {'major': 8, 'minor': 5}}
  > def unbind(dim, names):
  >     ins = [{'name': 'self', 'arg': {'as_tensor': {'name': 'x'}}, 'kind': 1}]
  >     if dim is not None:
  >         ins.append({'name': 'dim', 'arg': {'as_int': dim}, 'kind': 1})
  >     return {'target': 'torch.ops.aten.unbind.int', 'inputs': ins,
  >             'outputs': [{'as_tensors': [{'name': n} for n in names]}],
  >             'metadata': {}}
  > def w(name, p): json.dump(p, open(name + '.json', 'w'))
  > def case(name, shape, dim, count, out_shape):
  >     names = ['u%d' % i for i in range(count)]
  >     tv = {'x': tm(shape)}
  >     for n in names: tv[n] = tm(out_shape)
  >     w(name, program([unbind(dim, names)], tv))
  > # dim absent: the serializer omits a defaulted argument, so this is the real
  > # exported spelling and the schema default 0 has to supply it.
  > case('u-dim-absent', [2, 3, 4], None, 2, [3, 4])
  > case('u-dim-pos',    [2, 3, 4], 1,    3, [2, 4])
  > case('u-dim-neg',    [2, 3, 4], -1,   4, [2, 3])
  > # The rank-five ViT case: [3,1,3,101,32] unbound at the default dim 0. Native
  > # can represent it; the four-axis dialect cannot, since dim 0 of a rank-five
  > # tensor is the frame's T axis.
  > case('u-vit-rank5', [3, 1, 3, 101, 32], None, 3, [1, 3, 101, 32])
  > # Serialized name count disagrees with the extent at the selected axis: three
  > # names, but x has extent 2 at dim 0. Model data, not a defect here.
  > case('u-count-mismatch', [2, 3, 4], 0, 3, [3, 4])
  > # Output cardinality is model data, so it is an untrusted aggregate: 4096 is
  > # [Kernel.Limits.Hard.outputs], the existing cross-backend ceiling, and the
  > # rule is exclusive, so this must be refused.
  > case('u-over-limit', [4096, 2], 0, 4096, [2])
  > # A list-valued GRAPH output, the second place a Tensor[] has to flatten. The
  > # node here is an ordinary relu, so this isolates the graph-output site.
  > relu = {'target': 'torch.ops.aten.relu.default',
  >         'inputs': [{'name': 'self', 'arg': {'as_tensor': {'name': 'x'}}, 'kind': 1}],
  >         'outputs': [{'as_tensor': {'name': 'y'}}], 'metadata': {}}
  > flat = tm([2, 3])
  > w('u-graph-list', program([relu], {'x': flat, 'y': flat},
  >                           outs=[{'as_tensors': [{'name': 'y'}]}]))"

Each fixture now reaches its own outcome. The four well-formed ones lower, including
the rank-five ViT case — Native has no four-axis restriction, so `dim=0` on a rank-five
tensor is an ordinary `T`-axis unbind here; it is the Native4D conversion that has to
refuse it, and this file is not where that shows.

`count-mismatch` is the one shape only this op can have: the node's single `as_tensors`
output declares three names while `x`'s extent at `dim=0` is two. Both numbers are model
data, so it is a malformed graph rather than a defect, and it is checked before the
builder allocates anything.

`over-limit` is refused by Model Explorer's OWN per-node ceiling
(`max_outputs_metadata_per_node` = 1024), not by the lowering bound, which sits higher
at 4096. That is the honest result for this driver and worth recording: the projection
is bounded before the lowerer gets the chance. The lowering bound protects every caller
that has no Model Explorer limits — the `.pt2` interpreter and `native_graph`'s
print/eval/transform/to4d — and is driven directly in
`test/native_interp/output_limit_test.ml`.

  $ for m in dim-absent dim-pos dim-neg vit-rank5 count-mismatch over-limit graph-list; do
  >   printf '%-16s ' "$m"
  >   if ../bin/native_graph.exe visualize --model u-$m.json >/dev/null 2>err.txt; then
  >     echo lowered
  >   else
  >     head -1 err.txt
  >   fi
  > done
  dim-absent       lowered
  dim-pos          lowered
  dim-neg          lowered
  vit-rank5        lowered
  count-mismatch   native_graph: malformed PT2 graph: torch.ops.aten.unbind.int declares 3 outputs but produces 2
  over-limit       native_graph: session outputsMetadataPerNode = 4096 is over the ceiling
  graph-list       lowered

The whole projection, over the two fixtures that differ only in what the four-axis
dialect can represent. Nothing in Model Explorer needed a special case for a
multi-output node: the source projection reads `Argument.Tensors` through
`Me_source.tensor_names`, and every later stage works from the registries and the
generic output lists.

`u-dim-neg` unbinds C on a rank-three input, so every slice stays four-axis and the
graph converts the whole way down. `u-vit-rank5` is the motivating ViT node: Native
represents it happily — `dim=0` on a rank-five tensor is an ordinary `T`-axis
unbind — and only the Native4D row goes unavailable.

  $ caps() {
  >   ../bin/native_graph.exe visualize --model "$1" --output "$2" 2>/dev/null
  >   python3 -c "
  > import json, sys
  > s = json.load(open('$2'))
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
  $ caps u-dim-neg.json neg.json
  stage:source             available graph
  stage:initial_native     available graph
  stage:native4d           available graph
  $ caps u-vit-rank5.json vit.json
  stage:source             available graph
  stage:initial_native     available graph
  stage:native4d           unavailable outside_dialect_domain
    diagnostic: outside_dialect_domain | node n0: axis T is outside the N/H/W/C dialect

The source node carries ONE `Tensor[]` output, and it renders as its ordered SSA
elements rather than as a single edge or as none. Filtering for `Argument.Tensor`
alone would print an empty list here — a node with no outputs at all, which reads
as a graph that simply does not use its result.

  $ python3 -c "
  > import json
  > s = json.load(open('neg.json'))
  > g = [g for g in s['graphCollections'][0]['graphs'] if g['id'] == 'pt2/root'][0]
  > for n in g['nodes']:
  >     if 'unbind' in n.get('label', ''):
  >         print('label:', n['label'])
  >         for m in n.get('outputsMetadata', []):
  >             ssa = [a['value'] for a in m['attrs'] if a['key'] == 'ssa']
  >             print('  slot', m['id'], 'ssa', ssa)"
  label: torch.ops.aten.unbind.int
    slot 0 ssa ['u0']
    slot 1 ssa ['u1']
    slot 2 ssa ['u2']
    slot 3 ssa ['u3']

One Native node with every output edge, and the ids stay distinct — a projection
that collapsed a multi-output node's slots would show fewer, or repeat one.

  $ python3 -c "
  > import json
  > s = json.load(open('neg.json'))
  > g = [g for g in s['graphCollections'][0]['graphs'] if g['id'] == 'g/native/000'][0]
  > for n in g['nodes']:
  >     if 'unbind' in n.get('label', '').lower():
  >         slots = [m['id'] for m in n.get('outputsMetadata', [])]
  >         print('label:', n['label'], 'slots:', slots, 'unique:', len(set(slots)) == len(slots))"
  label: Unbind slots: ['0', '1', '2', '3'] unique: True
