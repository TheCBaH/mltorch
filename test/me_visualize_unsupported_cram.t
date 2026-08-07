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
  unsupported_input | unsupported PT2 input: non-tensor input

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
