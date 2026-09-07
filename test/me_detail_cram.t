Export one kernel value's expression as a detail delta.

UNGATED, over the committed mobilenetv2_050 model.json. On demand is the point: a
kernel value's expression can be far larger than the graph node that produced
it, so a session that carried every one of them would pay for every expression
to show one.

A convolution value, which is the shape worth looking at -- a triple reduction
over a product of two loads, plus the bias.

  $ ../bin/native_graph.exe detail --model model.json --graph g/kernel/000 --value 834 --output delta.json
  $ python3 -c "
  > import json
  > d = json.load(open('delta.json'))
  > print(d['collection'], d['graph']['id'], d['view']['id'] == d['graph']['id'])
  > n = {n['id']: n for n in d['graph']['nodes']}
  > for i in ('e0', 'e1', 'e4', 'e55', 'e67'):
  >     x = n[i]
  >     a = {a['key']: a['value'] for a in x['attrs']}
  >     edge = x.get('incomingEdges', [])
  >     role = edge[0].get('metadata', {}).get('role', 'root') if edge else 'root'
  >     print(' ', i, x['label'], a['language'], a['constructor'], role)"
  mltorch:model expr/g/kernel/000/t834/t834 True
    e0 round_f32 presentation result root
    e1 region region region computation
    e4 sum value reduce lhs
    e55 load t315 value load lhs
    e67 reduce index reduce operand

The graph and the view carry the KEY's id, and the key's last two components are
the same tensor id -- that is what "one identity, not two" looks like once the
parent node stopped being a field the caller supplies.

The complete structure is emitted as typed decomposition nodes. The selected
sample shows the result and Region presentation nodes, a value reduction, a
load, and an index reducer reference. The final column is the incoming edge's
role, so the graph distinguishes containment from execution order without
overloading graph slots.

A key naming no value the model produces is a valid request about something
ABSENT, which carries its own code rather than reading as a malformed one.

  $ ../bin/native_graph.exe detail --model model.json --graph g/kernel/000 --value 9999
  native_graph: the key names no value this model produces
  [123]

The ceiling counts every emitted presentation, value, Boolean, index, and
binder node, so it tracks the graph the browser must process rather than a
smaller proxy expression.

  $ ../bin/native_graph.exe detail --model model.json --graph g/kernel/000 --value 834 --limits small --output small.json
  $ python3 -c "
  > import json
  > print(len(json.load(open('small.json'))['graph']['nodes']), 'nodes')"
  94 nodes

The key is checked against the REQUEST's profile, not the one it was built
under: a parent graph inside untrusted's 4096-byte id ceiling is past small's
256. The parent check fires first here, so the derived-id one is not what this
row shows -- the suite has that case.

  $ ../bin/native_graph.exe detail --model model.json --value 834 --limits small --graph "$(python3 -c "print('g' * 300)")"
  native_graph: detail key parent graph is too long
  [123]
