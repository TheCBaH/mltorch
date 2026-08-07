Export a Model Explorer session from a real .pt2 archive.

GATED on PT2_DATA, since it needs downloaded release weights. What it adds over
the ungated model.json case is the half that file cannot reach: the archive
path through the five ordered checkpoints, and the capabilities that need
payloads.

  $ ../bin/native_graph.exe visualize --model "$PT2_DATA/resnet18/resnet18.pt2" --output session.json
  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > m = s['model']
  > print('sourceKind', m['sourceKind'])
  > print('sha256', m.get('sourceSha256', 'absent'))
  > print('views=%d comparisons=%d graphs=%d' % (
  >     len(s['views']), len(s['comparisons']),
  >     len(s['graphCollections'][0]['graphs'])))"
  sourceKind pt2
  sha256 absent
  views=5 comparisons=2 graphs=5

The digest is absent because no expected one was supplied: a locally chosen
file has nothing to verify against, and recording a digest we computed
ourselves would assert a provenance nobody checked. The catalog path is where
Some appears.

--fold has payloads to load here, so the capability is available rather than
carrying Requires_payloads -- which is the one difference from the model.json
run, and the reason both fixtures exist.

  $ ../bin/native_graph.exe visualize --model "$PT2_DATA/resnet18/resnet18.pt2" --fold --output folded.json
  $ python3 -c "
  > import json
  > c = [c for c in json.load(open('folded.json'))['capabilities']
  >      if c['key'] == 'feature:fold'][0]
  > print(c['status']['state'], c['status'].get('reason', ''))"
  available 

--fold on an archive actually folds, and that is what puts the graph inside the
Native4D dialect: the importer emits every conv weight right-aligned onto
D/H/W/C, so an unfolded graph has non-unit D on every weight. Canonical drops
from 194 nodes to 93, and the four-axis branch of the flow spine appears.

A structural session reaches transform_lowered with no constants, and Fold_const
declines every node that has no payload -- so before this the archive path was
running a fold that folded nothing while reporting the capability available.

  $ ../bin/native_graph.exe visualize --model "$PT2_DATA/resnet18/resnet18.pt2" --fold --output n4.json
  $ python3 -c "
  > import json
  > s = json.load(open('n4.json'))
  > print('graphs', [(g['id'], len(g['nodes'])) for g in s['graphCollections'][0]['graphs']])
  > print('views', [v['id'] for v in s['views']])
  > print('states', [x['id'] for x in s['flow']['states']])
  > print('transitions', [(t['id'], t['kind']['kind']) for t in s['flow']['transitions']])
  > print('native4d', [c['status']['state'] for c in s['capabilities']
  >                    if c['key'] == 'stage:native4d'])"
  graphs [('pt2/root', 194), ('g/native/000', 298), ('g/native/001', 93), ('g/symbolic/000', 93), ('g/native4d/000', 93), ('g/kernel/000', 93)]
  views ['v/canonical', 'v/initial', 'v/source', 'v/native4d', 'v/stage_program', 'v/kernel']
  states ['s/pt2/000', 's/native/000', 's/native/001', 's/native4d/000', 's/symbolic/000', 's/kernel/000']
  transitions [('t/native/000', 'import'), ('t/native/001', 'pack'), ('t/native4d/000', 'cross_dialect'), ('t/symbolic/000', 'adapt'), ('t/kernel/000', 'adapt')]
  native4d ['available']

Without --fold the same archive is outside the dialect, which is the fact the
matrix calls Native4D conditional for.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > c = [c for c in s['capabilities'] if c['key'] == 'stage:native4d'][0]
  > print(c['status']['state'], c['status'].get('reason'))"
  unavailable outside_dialect_domain

Detection reads the archive's magic bytes, not its extension.

  $ cp "$PT2_DATA/resnet18/resnet18.pt2" mislabelled.json
  $ ../bin/native_graph.exe visualize --model mislabelled.json --output m.json
  $ python3 -c "
  > import json
  > print(json.load(open('m.json'))['model']['sourceKind'])"
  pt2
