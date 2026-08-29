Export a Model Explorer session from a real .pt2 archive.

GATED on PT2_DATA, since it needs downloaded release weights. What it adds over
the ungated model.json case is the half that file cannot reach: the archive
path through the five ordered checkpoints, and the capabilities that need
payloads. MobileNetV2-050 stands in for the retired resnet18 role model here
— it is one of the models that fully converts to Native4D (unlike regnetx_002,
see test/native4d_to4d_cram.t).

  $ ../bin/native_graph.exe visualize --model "$PT2_DATA/mobilenetv2_050/mobilenetv2_050.pt2" --output session.json
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
  views=7 comparisons=2 graphs=7

The digest is absent because no expected one was supplied: a locally chosen
file has nothing to verify against, and recording a digest we computed
ourselves would assert a provenance nobody checked. The catalog path is where
Some appears.

--fold has payloads to load here, so the capability is available rather than
carrying Requires_payloads -- which is the one difference from the model.json
run, and the reason both fixtures exist.

  $ ../bin/native_graph.exe visualize --model "$PT2_DATA/mobilenetv2_050/mobilenetv2_050.pt2" --fold --output folded.json
  $ python3 -c "
  > import json
  > c = [c for c in json.load(open('folded.json'))['capabilities']
  >      if c['key'] == 'feature:fold'][0]
  > print(c['status']['state'], c['status'].get('reason', ''))"
  available 

The canonical pipeline (`lib/native/transform/pipeline.ml`) now runs constant
folding and batch-norm folding unconditionally, so `--fold` no longer decides
whether the graph reaches the Native4D dialect — compare `native4d` here with
the no-`--fold` capability check below: both report `available`. `--fold`'s
remaining effect is `feature:fold` above (materializing real payload bytes),
not the structural relayout.

A structural session reaches transform_lowered with no constants, and Fold_const
declines every node that has no payload -- so before this the archive path was
running a fold that folded nothing while reporting the capability available.

  $ ../bin/native_graph.exe visualize --model "$PT2_DATA/mobilenetv2_050/mobilenetv2_050.pt2" --fold --output n4.json
  $ python3 -c "
  > import json
  > s = json.load(open('n4.json'))
  > print('graphs', [(g['id'], len(g['nodes'])) for g in s['graphCollections'][0]['graphs']])
  > print('views', [v['id'] for v in s['views']])
  > print('states', [x['id'] for x in s['flow']['states']])
  > print('transitions', [(t['id'], t['kind']['kind']) for t in s['flow']['transitions']])
  > print('native4d', [c['status']['state'] for c in s['capabilities']
  >                    if c['key'] == 'stage:native4d'])"
  graphs [('pt2/root', 468), ('g/native/000', 731), ('g/native/001', 208), ('g/symbolic/000', 208), ('g/native4d/000', 208), ('g/kernel/000', 208), ('g/flow', 11)]
  views ['v/canonical', 'v/initial', 'v/source', 'v/native4d', 'v/stage_program', 'v/kernel', 'v/flow']
  states ['s/pt2/000', 's/native/000', 's/native/001', 's/native4d/000', 's/symbolic/000', 's/kernel/000']
  transitions [('t/native/000', 'import'), ('t/native/001', 'pack'), ('t/native4d/000', 'cross_dialect'), ('t/symbolic/000', 'adapt'), ('t/kernel/000', 'adapt')]
  native4d ['available']

The same `session.json`, built without `--fold`, already reports `native4d`
`available` too — confirming `--fold` is not the gate (see above); `stage:
native4d` is conditional on the graph fitting the four-axis dialect, not on
whether payloads were preloaded.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > c = [c for c in s['capabilities'] if c['key'] == 'stage:native4d'][0]
  > print(c['status']['state'], c['status'].get('reason'))"
  available None

Detection reads the archive's magic bytes, not its extension.

  $ cp "$PT2_DATA/mobilenetv2_050/mobilenetv2_050.pt2" mislabelled.json
  $ ../bin/native_graph.exe visualize --model mislabelled.json --output m.json
  $ python3 -c "
  > import json
  > print(json.load(open('m.json'))['model']['sourceKind'])"
  pt2
