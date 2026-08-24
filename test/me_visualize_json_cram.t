Export a Model Explorer session from the committed mobilenetv2_050 model.json.

UNGATED: this model is a submodule file, not a downloaded weight, so the whole
canonical path -- lower, transform, project, validate -- runs in every build.
Its payload-free nature is the point rather than a limitation: it is the
concrete case for the capabilities that need bytes.

The session is summarised rather than printed. A golden holding a megabyte of
node JSON would change on every unrelated projection tweak and be re-promoted
without being read, which is the opposite of what a golden is for; what is
pinned here is the shape, the counts, and the COMPLETE capability vector.

  $ ../bin/native_graph.exe visualize --model model.json --output session.json
  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > print('views=%d comparisons=%d capabilities=%d graphs=%d' % (
  >     len(s['views']), len(s['comparisons']), len(s['capabilities']),
  >     len(s['graphCollections'][0]['graphs'])))"
  views=7 comparisons=2 capabilities=14 graphs=7

The complete capability vector, which is what a drifting downstream row would
show up in. A test asserting only the interesting key would let the rest move.

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

The source view is the exported program's OWN graph, so its node count is the
model.json's -- 152 nodes plus one boundary per graph input and output, and the
315 inputs are what a signature-driven input/constant split is for.

The two comparisons are opposites, and the pairing of entry count with
matchNodeIdFallback is what says so -- neither number alone does. Import carries
every correspondence explicitly and declares equal ids meaningless, because the
two panes speak different id languages. Canonical carries none and declares
equal ids a claim, because stable ids already pair what the pass did not touch.
An empty entry list with the fallback OFF would be a third thing entirely: a
comparison in which nothing corresponds to anything, which the renderer draws as
every node changed.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['pt2/root']
  > kinds = {}
  > for n in g['nodes']:
  >     kinds[n['label'] if n['label'] in ('input','constant','output') else 'op'] = \
  >         kinds.get(n['label'] if n['label'] in ('input','constant','output') else 'op', 0) + 1
  > print('source', sorted(kinds.items()))
  > c = {c['id']: c for c in s['comparisons']}
  > for i in ('c/import', 'c/canonical'):
  >     sync = c[i]['sync']
  >     print(i, 'entries', len(sync['entries']),
  >           'matchNodeIdFallback', sync['matchNodeIdFallback'],
  >           'showDiffHighlights', sync['showDiffHighlights'])"
  source [('constant', 314), ('input', 1), ('op', 152), ('output', 1)]
  c/import entries 152 matchNodeIdFallback False showDiffHighlights False
  c/canonical entries 0 matchNodeIdFallback True showDiffHighlights True

Namespaces come off nn_module_stack, one level RELATIVE to its parent -- the
naive join would repeat the whole dotted path at every depth.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['pt2/root']
  > seen = []
  > for n in g['nodes']:
  >     ns = n['namespace']
  >     if ns and ns not in seen: seen.append(ns)
  > print(seen[:6])"
  ['conv_stem', 'bn1', 'bn1/act', 'blocks/0/0/conv_dw', 'blocks/0/0/bn1', 'blocks/0/0/bn1/act']

Native4D is the OTHER conditional row, and mobilenetv2_050's canonical graph
converts fully -- payload-free, no folding needed -- into 36 Conv2D, 17
DepthwiseConv2D, 35 Hardtanh, 10 Add, one Adaptive_avg_pool2d and one Permute4
(the classifier's Linear legalizes to a 1x1 Conv2D, so it doesn't add a
distinct kind). That is a property of THIS model, not a general guarantee --
Requires_payloads and Outside_dialect_domain both remain live capability
states the matrix admits, for a model whose weights genuinely need folding to
fit, or don't fit the four-axis dialect at all.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > print([d['message'] for d in s['diagnostics'] if d['code'] == 'outside_dialect_domain'])
  > print('graphs', [g['id'] for g in s['graphCollections'][0]['graphs']])
  > print('states', [x['id'] for x in s['flow']['states']])"
  []
  graphs ['pt2/root', 'g/native/000', 'g/native/001', 'g/symbolic/000', 'g/native4d/000', 'g/kernel/000', 'g/flow']
  states ['s/pt2/000', 's/native/000', 's/native/001', 's/native4d/000', 's/symbolic/000', 's/kernel/000']

The Native4D graph and its flow state ARE present here, which is the same rule
the unlowered case follows in reverse: a spine node opens exactly the graphs the
classifier actually produced, never fewer and never a placeholder for one it
didn't.

--fold on a payload-free model.json is a SUCCESSFUL structural session carrying
a Requires_payloads capability, not a usage error: the browser cannot surface
one, and the same code path serves both shells.

  $ ../bin/native_graph.exe visualize --model model.json --fold --output folded.json
  $ python3 -c "
  > import json
  > s = json.load(open('folded.json'))
  > c = [c for c in s['capabilities'] if c['key'] == 'feature:fold'][0]
  > print(c['status']['state'], c['status'].get('reason'))"
  unavailable requires_payloads

The flow spine reaches the document, because Transition.comparison exists
nowhere else and a spine encoded only as a rendered graph leaves every
transition node resolving to nothing.

  $ python3 -c "
  > import json
  > f = json.load(open('session.json'))['flow']
  > print('states', [s['id'] for s in f['states']])
  > print('transitions', [(t['id'], t['kind']['kind'], t.get('comparison')) for t in f['transitions']])"
  states ['s/pt2/000', 's/native/000', 's/native/001', 's/native4d/000', 's/symbolic/000', 's/kernel/000']
  transitions [('t/native/000', 'import', 'c/import'), ('t/native/001', 'pack', 'c/canonical'), ('t/native4d/000', 'cross_dialect', None), ('t/symbolic/000', 'adapt', None), ('t/kernel/000', 'adapt', None)]

The flow destination: the spine's graph is in the collection, exactly one
View.Flow opens it, and feature:flow offers that same graph. Session.validate
binds all three, so any one of them drifting is a validation failure rather
than something the browser discovers.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > flow = [v for v in s['views'] if v['kind'] == 'flow']
  > cap = [c for c in s['capabilities'] if c['key'] == 'feature:flow'][0]
  > g = {x['id']: x for x in s['graphCollections'][0]['graphs']}[s['flow']['graph']]
  > print('view', [(v['id'], v['graph']) for v in flow])
  > print('capability', cap['status']['payload']['graph'])
  > print('nodes', len(g['nodes']), 'states+transitions',
  >       len(s['flow']['states']) + len(s['flow']['transitions']))
  > edges = sum(len(n.get('incomingEdges', [])) for n in g['nodes'])
  > print('edges', edges, '2*transitions', 2 * len(s['flow']['transitions']))
  > print('every node has one slot',
  >       all(len(n.get('outputsMetadata', [])) == 1 for n in g['nodes']))"
  view [('v/flow', 'g/flow')]
  capability g/flow
  nodes 11 states+transitions 11
  edges 10 2*transitions 10
  every node has one slot True

v/flow is never the default: the browser stays source-first and the CLI keeps
canonical Native.

  $ python3 -c "
  > import json
  > print(json.load(open('session.json'))['defaultView'])"
  v/canonical

Each state names the stage view it opens, EXPLICITLY. Two stage views may name
one graph without breaking any rule, so a graph-to-view lookup is ambiguous by
contract; this pins the pairing the exporter actually emits, and that every
named view is a declared stage view over that state's own graph.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > views = {v['id']: v for v in s['views']}
  > for st in s['flow']['states']:
  >     v = views.get(st['view'])
  >     print(st['id'], '->', st['view'], v['kind'], 'graph-agrees', v['graph'] == st['graph'])"
  s/pt2/000 -> v/source stage:source graph-agrees True
  s/native/000 -> v/initial stage:initial_native graph-agrees True
  s/native/001 -> v/canonical stage:canonical graph-agrees True
  s/native4d/000 -> v/native4d stage:native4d graph-agrees True
  s/symbolic/000 -> v/stage_program stage:stage_program graph-agrees True
  s/kernel/000 -> v/kernel stage:kernel graph-agrees True

--verify-symbolic turns the two verification capabilities on. They are two keys
rather than one because the composed report survives per-pass audit truncation:
mapping that truncation to a single Verification -> Over_limit would hide a
result that is still available, while Available alone would conceal that the
per-pass detail is incomplete.

mobilenetv2_050 is big enough (152 op nodes against resnet18's 70) that even
`quick` exhausts the global verification budget almost immediately -- and
`standard` hits the same ceiling, checked by hand (a few more vacuous
verdicts squeeze in before it does, but the budget itself doesn't move): it
is a fixed aggregate ceiling, not something effort raises. That is the same
kind of real, deliberate
limit as [small]'s rejection above, not a gap in this fixture; what is pinned
here is that an exhausted budget still reaches the document as its own
verdict, not silence or an error.

  $ ../bin/native_graph.exe visualize --model model.json --verify-symbolic quick --output verified.json
  $ python3 -c "
  > import json
  > s = json.load(open('verified.json'))
  > for c in s['capabilities']:
  >     if c['key'] in ('feature:verification', 'feature:pass_audits'):
  >         p = c['status']['payload']
  >         if p['kind'] == 'verification_summary':
  >             print(c['key'], [(b['label'], b['count']) for b in p['verificationSummary']])
  >         else:
  >             a = p['passAuditStatus']
  >             print(c['key'], a['retainedReports'], a['omittedReports'], a['omittedCounts'])"
  feature:verification [('unproved (global verification budget exhausted)', '1'), ('vacuous', '1')]
  feature:pass_audits 0 13 []

Verdicts reach the document twice, saying different things. The node data set is
NODE-keyed, which is the only form that can colour a node; groupNodeAttributes is
NAMESPACE-keyed, which is the only part of the wire format that can attach a fact
to a group at all.

Those namespaces are ID-QUALIFIED, which is why the exporter places clusters
itself: Map_verify's own Group_path appends a group's label alone, and the
importer labels every group with the PT2 target -- so same-named groups
below would have collapsed into one. The exhausted budget above means almost
nothing gets a per-node verdict at all here (unlike resnet18, where enough of
the budget survived to reach real batch-norm groups) -- which is itself worth
pinning: an empty verdict set is still a valid one node/group data set, not a
missing one.

  $ python3 -c "
  > import json
  > s = json.load(open('verified.json'))
  > d = s['nodeDataSets'][0]
  > print('nodeData', d['name'], 'over', d['graph'], len(d['results']), 'nodes')
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['g/native/001']
  > gna = g['groupNodeAttributes']
  > print('groups', len(gna))
  > print('root  ', sorted(gna[''].items()))
  > bn = sorted(k for k in gna if 'batch_norm' in k)
  > print('batch_norm groups', len(bn))"
  nodeData verification over g/native/001 0 nodes
  groups 1
  root   [('unproved (global verification budget exhausted)', '1'), ('vacuous', '1')]
  batch_norm groups 0

Without --verify-symbolic there is no report, so both keys are not_requested and
neither the verification node data nor the group attributes appear. A capability
nobody asked for was not blocked by anything. Fusion's node data is there either
way: it is a fact about the kernel, not about a verification run.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > print([c['status']['state'] for c in s['capabilities']
  >        if c['key'] in ('feature:verification', 'feature:pass_audits')])
  > print('nodeDataSets', [d['name'] for d in s['nodeDataSets']])
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['g/native/001']
  > print('groupNodeAttributes', g.get('groupNodeAttributes'))"
  ['not_requested', 'not_requested']
  nodeDataSets ['fusion']
  groupNodeAttributes None

Fusion is an OVERLAY and a decision list over the unchanged kernel, never a
fabricated rewrite -- so the capability names the KERNEL graph rather than one
of its own, and the overlay rides on that graph rather than on a comparison,
because it is a fact about one plan and not about two panes.

Placement is TWO facts. Which dependency edges are virtual, and which values
need stores: an externally live producer is both, and one enum per value cannot
say that. So the edges are the overlay and the values are the node data.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['g/kernel/000']
  > o = g['tasksData']['edgeOverlaysDataListLeftPane'][0]
  > print('overlay', o['name'], 'over', o['graphName'],
  >       [(ov['name'], len(ov['edges'])) for ov in o['overlays']])
  > d = [d for d in s['nodeDataSets'] if d['name'] == 'fusion'][0]
  > by = {}
  > for r in d['results']:
  >     k = r['value']['label'].split(' (')[0]
  >     by[k] = by.get(k, 0) + 1
  > print('placement', sorted(by.items()))
  > print([m['message'] for m in s['diagnostics'] if 'virtual' in m['message']])"
  overlay fusion over g/kernel/000 [('virtual dependencies', 10)]
  placement [('stored', 90), ('virtual', 10)]
  ['10 virtual edges, 89 producers not fused']

A rejection is a fact about a VALUE, so it is on that value's own datum -- one
diagnostic per rejection would be seventy on this model alone, against a
max_diagnostics of 64, which makes it a list the ceiling truncates rather than a
report. What reaches the diagnostics is one summary.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > d = [d for d in s['nodeDataSets'] if d['name'] == 'fusion'][0]
  > for r in d['results']:
  >     if '>= 2' in r['value']['label']:
  >         print(r['nodeId'], r['value']['label']); break"
  v834 stored (t834 has >= 2 uses)

That count is rendered '>= 2' and never as a figure: the planner's counter
saturates at two because the cross-body total is an int, and the per-value limits
admit a mathematical aggregate past 2^31 -- which wraps negative under
js_of_ocaml and would read as unique-use.

--format collections is lossy and says so on stderr rather than silently
dropping half the document.

  $ ../bin/native_graph.exe visualize --model model.json --format collections --output c.json
  warning: --format collections is lossy; comparisons, capabilities and the flow are discarded
  $ python3 -c "
  > import json
  > c = json.load(open('c.json'))
  > print(type(c).__name__, len(c), c[0]['label'], len(c[0]['graphs']))"
  list 1 mltorch:model 7

Format detection is CONTENT, never the extension. The worker also carries a
declared format and checks the two agree; the CLI has only the bytes.

  $ printf 'not a model' > bogus.json
  $ ../bin/native_graph.exe visualize --model bogus.json
  native_graph: not a zip archive and not a JSON object
  [123]

Only the two wire-selectable profiles are offered, which is what makes
Wire_limits a guarantee rather than a UI convention -- [large] and [trusted]
exist for callers holding data they produced and cannot be named here.
mobilenetv2_050's full session is genuinely bigger than [small]'s
max_session_bytes (0x8_0000 = 512 KiB): its [untrusted] session alone runs to
roughly 750 KB before [small]'s tighter trace/diagnostic/audit ceilings even
apply. [small] rejecting it, cleanly, is the fact worth pinning -- a ceiling
that could never be exceeded would not be a guarantee, just an unenforced
number.

  $ ../bin/native_graph.exe visualize --model model.json --limits small --output small.json
  native_graph: the encoded document is over the ceiling
  [123]

NO_COLOR, because this is the one assertion in the suite that pins CMDLINER's
own text. Cmdliner styles by default and quotes only when styling is off, and
cram strips ANSI escapes from output -- so a styled run reaches the golden as
"option --limits", unquoted, with no visible sign that styling was ever
involved. Dune scrubs the cram environment, so the variable has to be set here
rather than by the caller.

  $ NO_COLOR=1 ../bin/native_graph.exe visualize --model model.json --limits large
  Usage: native_graph visualize [--help] [OPTION]…
  native_graph: option '--limits': invalid value 'large', expected either
                'untrusted' or 'small'
  [124]
