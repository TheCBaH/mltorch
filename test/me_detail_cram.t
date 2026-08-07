Export one kernel value's expression as a detail delta.

UNGATED, over the committed resnet18 model.json. On demand is the point: a
kernel value's expression can be far larger than the graph node that produced
it, so a session that carried every one of them would pay for every expression
to show one.

A convolution value, which is the shape worth looking at -- a triple reduction
over a product of two loads, plus the bias.

  $ ../bin/native_graph.exe detail --model model.json --graph g/kernel/000 --value 125 --output delta.json
  $ python3 -c "
  > import json
  > d = json.load(open('delta.json'))
  > print(d['collection'], d['graph']['id'], d['view']['id'] == d['graph']['id'])
  > for n in d['graph']['nodes']:
  >     print(' ', n['id'], n['label'], '<-', [e['sourceNodeId'] for e in n.get('incomingEdges', [])])"
  mltorch:model expr/g/kernel/000/t125/t125 True
    e0 + <- []
    e1 sum <- ['e0']
    e2 sum <- ['e1']
    e3 sum <- ['e2']
    e4 * <- ['e3']
    e5 t123 <- ['e4']
    e6 t124 <- ['e4']
    e7 t323 <- ['e0']

The graph and the view carry the KEY's id, and the key's last two components are
the same tensor id -- that is what "one identity, not two" looks like once the
parent node stopped being a field the caller supplies.

Every node carries the subtree rooted there, rendered and bounded. One attribute
rather than per-constructor ones for the index-level parts: Expr.Pp.index takes
a names function precisely so its output cannot depend on allocation history,
and this walk holds no scoped naming environment to give it.

Which is visible below: a reducer BOUND at this node prints as r1, and one bound
above it prints as ?#0. That is the honest rendering of a subtree read on its
own, and the reason the attribute is a subtree rather than a whole expression
repeated at every depth.

  $ python3 -c "
  > import json
  > d = json.load(open('delta.json'))
  > n = {n['id']: n for n in d['graph']['nodes']}
  > for i in ('e0', 'e4', 'e5'):
  >     print(i, [a['value'] for a in n[i]['attrs'] if a['key'] == 'expr'][0][:70])"
  e0 (sum(r1=0..3: sum(r2=max(0,-1*2*H+-3)..min(7,224+-1+-1*2*H+-3+1): sum(
  e4 (t123[N,T,D,2*H+-3+?#1,2*W+-3+?#2,?#0] * t124[C,0,0,?#1,?#2,?#0])
  e5 t123[N,T,D,2*H+-3+?#1,2*W+-3+?#2,?#0]

A key naming no value the model produces is a valid request about something
ABSENT, which carries its own code rather than reading as a malformed one.

  $ ../bin/native_graph.exe detail --model model.json --graph g/kernel/000 --value 9999
  native_graph: the key names no value this model produces
  [123]

The ceiling is measured with Expr.Fold.size, which counts index trees too -- so
it exceeds the node count this produces (88 against 8 for that convolution), and
the rejection is named for what was measured rather than for what would have
been built.

  $ ../bin/native_graph.exe detail --model model.json --graph g/kernel/000 --value 125 --limits small --output small.json
  $ python3 -c "
  > import json
  > print(len(json.load(open('small.json'))['graph']['nodes']), 'nodes')"
  8 nodes

The key is checked against the REQUEST's profile, not the one it was built
under: a parent graph inside untrusted's 4096-byte id ceiling is past small's
256. The parent check fires first here, so the derived-id one is not what this
row shows -- the suite has that case.

  $ ../bin/native_graph.exe detail --model model.json --value 125 --limits small --graph "$(python3 -c "print('g' * 300)")"
  native_graph: detail key parent graph is too long
  [123]
