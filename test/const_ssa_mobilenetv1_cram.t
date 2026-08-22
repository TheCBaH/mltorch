Payload-free MobileNetV1-class conversion through Const-SSA.

The source is a committed `model.json`, not a `.pt2` archive: this invocation
has no parameter bytes to preload. Canonicalization must still fold the weight
relayouts and inference batch norms symbolically, then Native4D must carry the
same Const-SSA exports. `quick` verification is allowed to be inconclusive on
large activation clusters, but it must report no refutation.

  $ ../bin/native_graph.exe visualize --model mobilenetv1_100_model.json --limits small --verify-symbolic quick --output session.json

  $ python3 -c "
  > import collections, json
  > s = json.load(open('session.json'))
  > caps = {c['key']: c['status'] for c in s['capabilities']}
  > print('native4d:', caps['stage:native4d']['state'])
  > print('verification:', caps['feature:verification']['state'])
  > graphs = {g['id']: g for g in s['graphCollections'][0]['graphs']}
  > for ident in ('g/native/001', 'g/native4d/000'):
  >     counts = collections.Counter(n['label'] for n in graphs[ident]['nodes'])
  >     print('%s nodes=%d batch_norm=%d add=%d sqrt=%d' %
  >           (ident, len(graphs[ident]['nodes']), counts['Batch_norm'], counts['Add'], counts['Sqrt']))
  > rows = caps['feature:verification']['payload']['verificationSummary']
  > print('refuted:', any('refuted' in r['label'] for r in rows))"
  native4d: available
  verification: available
  g/native/001 nodes=115 batch_norm=0 add=0 sqrt=0
  g/native4d/000 nodes=115 batch_norm=0 add=0 sqrt=0
  refuted: False
