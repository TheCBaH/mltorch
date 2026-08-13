A model this repository cannot lower is a SUCCESSFUL session, not a usage error.

UNGATED, and hand-built rather than downloaded: what it has to exercise is a
graph the lowerer rejects, and no released model is one. The exported program
decoded, which is the whole of what stage:source claims, so the session carries
a source view and a capability vector saying what is missing and why. The
browser shell cannot surface a usage error, and the same code path serves both.

  $ python3 -c "
  > import json
  > tm = {'dtype': 7, 'sizes': [{'as_int': 1}, {'as_int': 4}],
  >       'requires_grad': False, 'device': {'type': 'cpu'},
  >       'strides': [{'as_int': 4}, {'as_int': 1}],
  >       'storage_offset': {'as_int': 0}, 'layout': 7}
  > def program(nodes, specs):
  >     return {'graph_module': {
  >               'graph': {'inputs': [{'as_tensor': {'name': 'x'}}],
  >                         'outputs': [{'as_tensor': {'name': 'y'}}],
  >                         'nodes': nodes,
  >                         'tensor_values': {'x': tm, 'y': tm},
  >                         'sym_int_values': {}, 'sym_bool_values': {},
  >                         'is_single_tensor_return': True},
  >               'signature': {'input_specs': specs,
  >                             'output_specs': [{'user_output': {'arg': {'as_tensor': {'name': 'y'}}}}]},
  >               'module_call_graph': []},
  >             'opset_version': {'aten': 15}, 'range_constraints': {},
  >             'schema_version': {'major': 8, 'minor': 5}}
  > node = {'target': 'torch.ops.aten.bogus_operator.default',
  >         'inputs': [{'name': 'self', 'arg': {'as_tensor': {'name': 'x'}}, 'kind': 1}],
  >         'outputs': [{'as_tensor': {'name': 'y'}}],
  >         'metadata': {'nn_module_stack': 'L__self__,,M;L__self__blk,blk,B'}}
  > user = [{'user_input': {'arg': {'as_tensor': {'name': 'x'}}}}]
  > json.dump(program([node], user), open('operator.json', 'w'))
  > relu = dict(node, target='torch.ops.aten.relu.default')
  > json.dump(program([relu], user + [{'token': {'arg': {'name': 'tok'}}}]),
  >           open('input.json', 'w'))"

  $ ../bin/native_graph.exe visualize --model operator.json --output session.json

The COMPLETE vector, not the failing key: a test checking only initial_native
would let every downstream row drift, and the whole point of propagating a
prerequisite is what happens to the rows the failure did not touch.

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
  stage:initial_native         unavailable unsupported_operator
  stage:canonical              unavailable prerequisite_unavailable
  stage:native4d               unavailable prerequisite_unavailable
  stage:stage_program          unavailable prerequisite_unavailable
  stage:kernel                 unavailable prerequisite_unavailable
  stage:fusion                 unavailable prerequisite_unavailable
  feature:flow                 unavailable prerequisite_unavailable
  feature:verification         unavailable prerequisite_unavailable
  feature:pass_audits          unavailable prerequisite_unavailable
  feature:fold                 unavailable prerequisite_unavailable
  feature:expression_detail    available present
  feature:loop_ir              unavailable not_implemented
  feature:codegen              unavailable not_implemented

initial_native carries the reason lowering actually gave, not
prerequisite_unavailable -- "its prerequisite is unavailable" would be circular
for the row that IS the lowering, and the classified reason is the actionable
one. The free text lands in a diagnostic, which is the bounded type that crosses
every boundary of this design.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > for d in s['diagnostics']:
  >     print(d['code'], '|', d['message'], '|', d.get('graph'), '| truncated', d['truncated'])"
  unsupported_operator | unsupported PT2 operator: torch.ops.aten.bogus_operator.default | pt2/root | truncated False

One graph, one view, and NO flow: with no Native state the spine would hold
s/pt2/000 and no transitions, and a one-node flow graph asserts a navigability
that does not exist.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > print('graphs', [g['id'] for g in s['graphCollections'][0]['graphs']])
  > print('views', [v['id'] for v in s['views']], 'default', s['defaultView'])
  > print('comparisons', s['comparisons'], 'flow', s.get('flow'))
  > print('opTargets', s['model']['opTargets'])"
  graphs ['pt2/root']
  views ['v/source'] default v/source
  comparisons [] flow None
  opTargets 1

The other recoverable row. An input spec the lowerer does not handle is
Unsupported_input, and it reaches the same shape through the same classifier --
which is what makes the two rows one code path rather than two.

  $ ../bin/native_graph.exe visualize --model input.json --output input-session.json
  $ python3 -c "
  > import json
  > s = json.load(open('input-session.json'))
  > c = [c for c in s['capabilities'] if c['key'] == 'stage:initial_native'][0]
  > print(c['status']['state'], c['status']['reason'])
  > print(s['diagnostics'][0]['code'], '|', s['diagnostics'][0]['message'])"
  unavailable unsupported_input
  unsupported_input | unsupported PT2 input: not a tensor

A defect is NOT downgraded to a capability. Reporting an internal invariant
failure as "this model is outside what we support" tells the user to change
their model to work around our bug, so those rows still exit -- here a graph
whose output names nothing, which the lowerer rejects as malformed.

  $ python3 -c "
  > import json
  > p = json.load(open('operator.json'))
  > p['graph_module']['graph']['nodes'] = []
  > json.dump(p, open('malformed.json', 'w'))"
  $ ../bin/native_graph.exe visualize --model malformed.json
  native_graph: malformed PT2 graph: SSA tensor "y" is not defined
  [123]

Every OTHER row of [Native_interp.malformed], each with its own witness.

The rows were one `Malformed_graph of string` carrying a ksprintf sentence from
32 sites, so no test could distinguish them and none tried. They are now twelve
typed cases, and a case no input reaches is indistinguishable from one that
cannot be constructed -- so each is driven here, from the smallest mutation of a
valid program that produces it.

The two most recent are the ones a `Tensor[]` return forced. `zero-dim` is its
own `dim_fault` rather than a negative one: the engine has no empty tensor at
all, and unlike a negative size this does arrive from real models, since ATen's
unbind of a zero-length dim returns an empty list. `output-arity` is the shape
only a list-returning node can have -- one `as_tensors` output whose name count
is model data, checked against a count derived from the operand's extent.

  $ python3 -c "
  > import json
  > def tm(sizes):
  >     return {'dtype': 7, 'sizes': sizes, 'requires_grad': False,
  >             'device': {'type': 'cpu'}, 'strides': [{'as_int': 1}],
  >             'storage_offset': {'as_int': 0}, 'layout': 7}
  > def ints(xs): return [{'as_int': i} for i in xs]
  > def program(nodes, tv, names=['x'], outs=[{'as_tensor': {'name': 'y'}}]):
  >     return {'graph_module': {
  >               'graph': {'inputs': [{'as_tensor': {'name': n}} for n in names],
  >                         'outputs': outs, 'nodes': nodes, 'tensor_values': tv,
  >                         'sym_int_values': {}, 'sym_bool_values': {},
  >                         'is_single_tensor_return': True},
  >               'signature': {'input_specs': [{'user_input': {'arg': {'as_tensor': {'name': n}}}} for n in names],
  >                             'output_specs': [{'user_output': {'arg': {'as_tensor': {'name': 'y'}}}}]},
  >               'module_call_graph': []},
  >             'opset_version': {'aten': 15}, 'range_constraints': {},
  >             'schema_version': {'major': 8, 'minor': 5}}
  > flat = tm(ints([1, 4]))
  > relu = {'target': 'torch.ops.aten.relu.default',
  >         'inputs': [{'name': 'self', 'arg': {'as_tensor': {'name': 'x'}}, 'kind': 1}],
  >         'outputs': [{'as_tensor': {'name': 'y'}}], 'metadata': {}}
  > both = {'x': flat, 'y': flat}
  > def w(name, p): json.dump(p, open(name + '.json', 'w'))
  > w('m-missing-arg',  program([dict(relu, inputs=[])], both))
  > w('m-wrong-kind',   program([dict(relu, inputs=[{'name': 'self', 'arg': {'as_int': 3}, 'kind': 1}])], both))
  > w('m-node-output',  program([dict(relu, outputs=[{'as_int': 1}])], both))
  > w('m-graph-output', program([relu], both, outs=[{'as_int': 1}]))
  > w('m-no-metadata',  program([relu], {'y': flat}))
  > w('m-negative',     program([relu], {'x': tm(ints([-1])), 'y': flat}))
  > w('m-symbolic',     program([relu], {'x': tm([{'as_expr': {'expr_str': 's0', 'hint': None}}]), 'y': flat}))
  > w('m-rank-seven',   program([relu], {'x': tm(ints([1] * 7)), 'y': flat}))
  > mean = {'target': 'torch.ops.aten.mean.dim',
  >         'inputs': [{'name': 'self', 'arg': {'as_tensor': {'name': 'x'}}, 'kind': 1},
  >                    {'name': 'dim', 'arg': {'as_ints': [9]}, 'kind': 1}],
  >         'outputs': [{'as_tensor': {'name': 'y'}}], 'metadata': {}}
  > w('m-axis', program([mean], both))
  > add = {'target': 'torch.ops.aten.add.Tensor',
  >        'inputs': [{'name': 'self', 'arg': {'as_tensor': {'name': 'x'}}, 'kind': 1},
  >                   {'name': 'other', 'arg': {'as_tensor': {'name': 'x'}}, 'kind': 1},
  >                   {'name': 'alpha', 'arg': {'as_int': 2}, 'kind': 1}],
  >        'outputs': [{'as_tensor': {'name': 'y'}}], 'metadata': {}}
  > w('m-alpha', program([add], both))
  > clone = {'target': 'torch.ops.aten.clone.default',
  >          'inputs': [{'name': 'self', 'arg': {'as_tensor': {'name': 'x'}}, 'kind': 1},
  >                     {'name': 'memory_format', 'arg': {'as_memory_format': 2}, 'kind': 1}],
  >          'outputs': [{'as_tensor': {'name': 'y'}}], 'metadata': {}}
  > w('m-memory-format', program([clone], both))
  > def conv(stride):
  >     return {'target': 'torch.ops.aten.convolution.default',
  >             'inputs': [{'name': 'input', 'arg': {'as_tensor': {'name': 'x'}}, 'kind': 1},
  >                        {'name': 'weight', 'arg': {'as_tensor': {'name': 'w'}}, 'kind': 1},
  >                        {'name': 'bias', 'arg': {'as_none': True}, 'kind': 1},
  >                        {'name': 'stride', 'arg': {'as_ints': stride}, 'kind': 1},
  >                        {'name': 'padding', 'arg': {'as_ints': [0, 0]}, 'kind': 1},
  >                        {'name': 'dilation', 'arg': {'as_ints': [1, 1]}, 'kind': 1},
  >                        {'name': 'transposed', 'arg': {'as_bool': False}, 'kind': 1},
  >                        {'name': 'output_padding', 'arg': {'as_ints': [0, 0]}, 'kind': 1},
  >                        {'name': 'groups', 'arg': {'as_int': 1}, 'kind': 1}],
  >             'outputs': [{'as_tensor': {'name': 'y'}}], 'metadata': {}}
  > img, out = tm(ints([1, 3, 8, 8])), tm(ints([1, 2, 8, 8]))
  > w('m-arity',     program([conv([1, 1, 1])], {'x': img, 'w': tm(ints([2, 3, 1, 1])), 'y': out}, ['x', 'w']))
  > w('m-conv-rank', program([conv([1, 1])],    {'x': img, 'w': tm(ints([2, 3])),       'y': out}, ['x', 'w']))
  > # A zero extent is its OWN fault, not a negative one: the engine has no empty
  > # tensor at all ([Dim.extent] is >= 1), and unlike a negative size this does
  > # arrive from real models -- ATen's unbind of a zero-length dim returns an
  > # empty list. Without its own arm it reached [Dim.extent] and escaped as an
  > # uncaught [Invalid_argument].
  > w('m-zero-dim', program([relu], {'x': tm(ints([0, 4])), 'y': flat}))
  > # The only arity a fixed-tuple op cannot have: one as_tensors output whose
  > # name count is model data, against a count derived from the operand's extent.
  > unbind = {'target': 'torch.ops.aten.unbind.int',
  >           'inputs': [{'name': 'self', 'arg': {'as_tensor': {'name': 'x'}}, 'kind': 1}],
  >           'outputs': [{'as_tensors': [{'name': 'y'}, {'name': 'z'}, {'name': 'w'}]}],
  >           'metadata': {}}
  > w('m-output-arity', program([unbind], {'x': tm(ints([2, 4])), 'y': flat}))"

  $ for m in missing-arg wrong-kind node-output graph-output no-metadata \
  >          negative zero-dim symbolic rank-seven axis alpha memory-format arity \
  >          conv-rank output-arity; do
  >   ../bin/native_graph.exe visualize --model m-$m.json 2>&1 >/dev/null | head -1
  > done
  native_graph: malformed PT2 graph: torch.ops.aten.relu.default: missing argument "self"
  native_graph: malformed PT2 graph: torch.ops.aten.relu.default.self is not a tensor
  native_graph: malformed PT2 graph: torch.ops.aten.relu.default has a non-tensor output
  native_graph: malformed PT2 graph: non-tensor graph output
  native_graph: malformed PT2 graph: no tensor metadata for "x"
  native_graph: malformed PT2 graph: x has negative dimension -1
  native_graph: malformed PT2 graph: x has a zero-length dimension
  native_graph: malformed PT2 graph: x has a symbolic dimension
  native_graph: malformed PT2 graph: x has rank greater than six
  native_graph: malformed PT2 graph: invalid dimension 9 for rank 2
  native_graph: malformed PT2 graph: torch.ops.aten.add.Tensor: alpha=2 is not supported (only 1)
  native_graph: malformed PT2 graph: torch.ops.aten.clone.default: memory_format is not supported
  native_graph: malformed PT2 graph: stride must have one or two values, got 3
  native_graph: malformed PT2 graph: w is rank 2, expected four
  native_graph: malformed PT2 graph: torch.ops.aten.unbind.int declares 3 outputs but produces 2

[`Output_not_evaluated] has no witness here and is marked as such rather than
left to look covered. It fires when [Eval_direct] returns an environment missing
a graph output -- a graph declaring an output no node produces and that is
neither an input nor a constant.

No model can reach it. [Native_interp.lower]'s body resolves every output
through [env_find], which returns ids [add_env] bound from node outputs, so a
lowered graph's outputs are produced by construction.

It is kept anyway, and that is a different judgement from the one that deleted
[`Dangling_edge] in the same migration. [`Dangling_edge] had no construction
site at all -- nothing in the tree could raise it, and its condition was already
caught by [`Unknown_node]. This one IS raised, at both run sites, and NOTHING
checks it earlier: [Graph_builder.build] does not verify that outputs are
produced, and [Map_verify] runs only under verification. What stands between a
pass that drops a producer and a silent wrong answer is this row.
