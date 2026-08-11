Export a Model Explorer session from the committed resnet18 model.json.

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
  views=5 comparisons=2 capabilities=14 graphs=5

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
  stage:native4d               unavailable outside_dialect_domain
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
model.json's -- 70 nodes plus one boundary per graph input and output, and the
123 inputs are what a signature-driven input/constant split is for. The import
comparison is the one that must carry explicit entries: the two panes speak
different id languages, so MATCH_NODE_ID pairs nothing across them.

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
  > print('import entries', len(c['c/import']['sync']['entries']),
  >       'canonical entries', len(c['c/canonical']['sync']['entries']))"
  source [('constant', 122), ('input', 1), ('op', 70), ('output', 1)]
  import entries 70 canonical entries 0

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
  ['conv1', 'bn1', 'relu', 'maxpool', 'layer1/0/conv1', 'layer1/0/bn1']

Native4D is the OTHER conditional row, and this is the payload-free half of it.
The importer emits every conv weight right-aligned onto D/H/W/C, so an unfolded
graph has non-unit D on every weight -- and folding is what relays them, which
needs payloads this file does not have. Outside_dialect_domain rather than
Requires_payloads because that is the first node that actually fails; the matrix
admits either, and the classifier reports which.

  $ python3 -c "
  > import json
  > s = json.load(open('session.json'))
  > print([d['message'] for d in s['diagnostics'] if d['code'] == 'outside_dialect_domain'])
  > print('graphs', [g['id'] for g in s['graphCollections'][0]['graphs']])
  > print('states', [x['id'] for x in s['flow']['states']])"
  ['node n1: axis D is outside the N/H/W/C dialect']
  graphs ['pt2/root', 'g/native/000', 'g/native/001', 'g/symbolic/000', 'g/kernel/000']
  states ['s/pt2/000', 's/native/000', 's/native/001', 's/symbolic/000', 's/kernel/000']

There is no Native4D graph and no Native4D flow state, which is the same rule the
unlowered session follows: a spine node for a graph nobody can open asserts a
navigability that is not there.

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
  states ['s/pt2/000', 's/native/000', 's/native/001', 's/symbolic/000', 's/kernel/000']
  transitions [('t/native/000', 'import', 'c/import'), ('t/native/001', 'pack', 'c/canonical'), ('t/symbolic/000', 'adapt', None), ('t/kernel/000', 'adapt', None)]

--verify-symbolic turns the two verification capabilities on. They are two keys
rather than one because the composed report survives per-pass audit truncation:
mapping that truncation to a single Verification -> Over_limit would hide a
result that is still available, while Available alone would conceal that the
per-pass detail is incomplete.

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
  feature:verification [('proved (structural) [sampled 4]', '1'), ('unproved (over max_rounds) [sampled 4]', '1'), ('unproved (too large)', '101'), ('unproved (unbound constant) [sampled 4]', '81'), ('vacuous', '98')]
  feature:pass_audits 12 0 []

Verdicts reach the document twice, saying different things. The node data set is
NODE-keyed, which is the only form that can colour a node; groupNodeAttributes is
NAMESPACE-keyed, which is the only part of the wire format that can attach a fact
to a group at all.

Those namespaces are ID-QUALIFIED, which is why the exporter places clusters
itself: Map_verify's own Group_path appends a group's label alone, and the
importer labels every group with the PT2 target -- so the eight batch-norm groups
below would have collapsed into one.

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
  > print('batch_norm groups', len(bn))
  > print('first ', bn[0], sorted(gna[bn[0]].items()))"
  nodeData verification over g/native/001 90 nodes
  groups 43
  root   [('unproved (over max_rounds) [sampled 4]', '1'), ('unproved (too large)', '39'), ('unproved (unbound constant) [sampled 4]', '81'), ('vacuous', '98')]
  batch_norm groups 20
  first  torch.ops.aten._native_batch_norm_legit_no_training.default#g11 [('unproved (too large)', '1')]

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
  overlay fusion over g/kernel/000 [('virtual dependencies', 20)]
  placement [('stored', 70), ('virtual', 20)]
  ['20 virtual edges, 69 producers not fused']

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
  v128 stored (t128 has >= 2 uses)

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
  list 1 mltorch:model 5

Format detection is CONTENT, never the extension. The worker also carries a
declared format and checks the two agree; the CLI has only the bytes.

  $ printf 'not a model' > bogus.json
  $ ../bin/native_graph.exe visualize --model bogus.json
  native_graph: not a zip archive and not a JSON object
  [123]

Only the two wire-selectable profiles are offered, which is what makes
Wire_limits a guarantee rather than a UI convention -- [large] and [trusted]
exist for callers holding data they produced and cannot be named here. [small]
accepts a real vision model, which is the fact worth pinning: a profile the
browser may select and that no real input drives past.

  $ ../bin/native_graph.exe visualize --model model.json --limits small --output small.json
  $ python3 -c "
  > import json
  > print(len(json.load(open('small.json'))['graphCollections'][0]['graphs']), 'graphs')"
  5 graphs

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
