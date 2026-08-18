import assert from 'node:assert/strict';
import test from 'node:test';
import {
  ALL_STAGES, BACKBONE_STAGES, OPTIONAL_STAGES, RANK_BUCKETS,
  bucketOfRank, buildIndex, capabilityWording, comparisonPresentation,
  controlsFromOptions, decodeUrl, defaultPresentation, encodeUrl,
  optionsFromControls, optionsFromUrl, preferredViews, requestKey,
  flowPresentation, resolvePresentation, samePresentation, singlePresentation,
  staleNotice, staleComparisonNotice, staleFlowNotice,
} from '../app/presentation.js';

const view = (id, kind, graph) => ({ id, label: id, kind, collection: 'c', graph });
const capability = (key, status) => ({ key, status });

/* A session shaped like the real exporter's output: a source view, two Native
 * views, one unavailable stage carrying its reason, and the flow capability
 * that names a graph no collection holds. */
const SESSION = {
  model: { name: 'resnet18' },
  defaultView: 'v/canonical',
  graphCollections: [{ label: 'mltorch:model', graphs: [
    { id: 'pt2/root', nodes: [{ id: 'n0' }] },
    { id: 'g/native/000', nodes: [{ id: 'n0' }] },
    { id: 'g/native/001', nodes: [{ id: 'n0' }] },
  ] }],
  views: [
    view('v/canonical', 'stage:canonical', 'g/native/001'),
    view('v/initial', 'stage:initial_native', 'g/native/000'),
    view('v/source', 'stage:source', 'pt2/root'),
  ],
  comparisons: [],
  nodeDataSets: [{ name: 'verification', graph: 'g/native/001', results: [] }],
  capabilities: [
    capability('stage:source', { state: 'available', payload: { kind: 'graph', graph: 'pt2/root' } }),
    capability('stage:native4d', { state: 'unavailable', reason: 'outside_dialect_domain', detail: 'node n1: axis D' }),
    capability('stage:kernel', { state: 'not_requested' }),
    capability('feature:flow', { state: 'available', payload: { kind: 'graph', graph: 'g/flow' } }),
    capability('feature:fold', { state: 'unavailable', reason: 'requires_payloads' }),
    capability('feature:loop_ir', { state: 'unavailable', reason: 'not_implemented' }),
    capability('feature:verification', { state: 'available', payload: {
      kind: 'verification_summary',
      verificationSummary: [{ label: 'proved (structural) [sampled 4]', count: '1' }],
    } }),
    capability('feature:pass_audits', { state: 'available', payload: {
      kind: 'pass_audit_status',
      passAuditStatus: { retainedReports: '12', omittedReports: '0', omittedCounts: [] },
    } }),
  ],
  diagnostics: [{ code: 'over_limit', message: 'too big', graph: 'g/native/001', truncated: false }],
};

const session = (overrides) => JSON.stringify({ ...SESSION, ...overrides });

/* ---------------------------------------------------------------- the index */

test('the view list is exactly the stage views', () => {
  const index = buildIndex(session({
    views: [...SESSION.views, view('v/flow', 'flow', 'g/flow'), view('v/cmp', 'compare', 'g/native/000')],
  }));
  assert.deepEqual(index.views.map((v) => v.id), ['v/canonical', 'v/initial', 'v/source']);
  assert.equal(index.viewById.has('v/flow'), false);
  assert.equal(index.viewById.has('v/cmp'), false);
});

/* `feature:flow` is `available` and names `g/flow`, which no collection holds
 * and no view points at. A capability says whether something exists; only a
 * View declares somewhere the page can go. */
test('an available capability is never mistaken for a destination', () => {
  const index = buildIndex(session());
  assert.equal(index.capabilityByKey.get('feature:flow').status.payload.graph, 'g/flow');
  assert.equal(index.views.some((v) => v.graph === 'g/flow'), false);
  const graphs = SESSION.graphCollections[0].graphs.map((g) => g.id);
  assert.equal(graphs.includes('g/flow'), false);
});

test('the index survives a document missing every optional field', () => {
  const index = buildIndex('{}');
  assert.deepEqual(index.views, []);
  assert.deepEqual(index.diagnostics, []);
  assert.equal(index.verification, null);
  assert.equal(index.passAudits, null);
  assert.equal(index.defaultView, null);
});

/* Reading a payload off a row that is not `available` would yield `undefined`
 * and render as "no findings", which is the one wrong answer here. */
test('an unavailable verification row yields no summary', () => {
  const index = buildIndex(session({
    capabilities: [capability('feature:verification', { state: 'not_requested' })],
  }));
  assert.equal(index.verification, null);
});

test('counts stay decimal strings, never numbers', () => {
  const index = buildIndex(session());
  assert.equal(typeof index.verification[0].count, 'string');
  assert.equal(typeof index.passAudits.retainedReports, 'string');
});

/* --------------------------------------------------------- rank buckets */

test('every rank lands in a named bucket, and nothing else does', () => {
  const seen = [0, 1, 2, 3, 4, 5, 6].map((r) => bucketOfRank(r).name);
  assert.deepEqual(seen, ['vacuous', 'proved', 'proved', 'tested', 'tested', 'unproved', 'refuted']);
  for (const bad of [-1, 7, 1.5, NaN, undefined, null, '1']) {
    assert.equal(bucketOfRank(bad), null, `rank ${String(bad)}`);
  }
});

/* Vacuous is neutral because a creation or deletion claims nothing; refuted is
 * the strongest contrast because a counterexample was found. */
test('vacuous is neutral and refuted is the strongest contrast', () => {
  assert.equal(bucketOfRank(0).title, 'Vacuous');
  assert.equal(bucketOfRank(0).bg, RANK_BUCKETS[0].bg);
  assert.notEqual(bucketOfRank(0).bg, bucketOfRank(1).bg);
  assert.equal(bucketOfRank(6).title, 'Refuted');
  assert.equal(new Set(RANK_BUCKETS.map((b) => b.bg)).size, RANK_BUCKETS.length);
});

/* ------------------------------------------------------ capability wording */

test('not-requested, unavailable and unimplemented read as three different things', () => {
  const index = buildIndex(session());
  const of = (key) => capabilityWording(index.capabilityByKey.get(key));

  assert.equal(of('stage:kernel').state, 'not_requested');
  assert.match(of('stage:kernel').detail, /left out of the request/);

  const native4d = of('stage:native4d');
  assert.equal(native4d.state, 'unavailable');
  assert.match(native4d.detail, /outside that dialect’s domain/);
  assert.match(native4d.detail, /node n1: axis D/);

  /* Never `Available` and never `Not_requested`: there is nothing to ask for,
   * so "not requested" would read as "you did not ask". */
  const loop = of('feature:loop_ir');
  assert.equal(loop.state, 'unimplemented');
  assert.match(loop.detail, /cannot be requested/);
});

test('fold reads as unavailable with the exporter’s own reason', () => {
  const index = buildIndex(session());
  const fold = capabilityWording(index.capabilityByKey.get('feature:fold'));
  assert.equal(fold.state, 'unavailable');
  assert.match(fold.detail, /no constant payloads/);
});

/* ------------------------------------------------------ controls to request */

test('the backbones are always requested and can never be switched off', () => {
  for (const optional of [[], ['kernel'], [...OPTIONAL_STAGES]]) {
    const options = optionsFromControls({ optional });
    for (const backbone of BACKBONE_STAGES) assert.ok(options.stages.includes(backbone));
    assert.ok(options.stages.length >= BACKBONE_STAGES.length);
  }
  /* `Options.create` normalises by filtering `all_stages`; it never ADDS, so
   * an empty request here would become `Invalid_options` at the bridge. */
  assert.deepEqual(optionsFromControls({ optional: [] }).stages, [...BACKBONE_STAGES]);
});

test('the request is already in all_stages order, whatever the controls say', () => {
  const options = optionsFromControls({ optional: ['fusion', 'native4d'] });
  assert.deepEqual(options.stages, ['source', 'initial_native', 'canonical', 'native4d', 'fusion']);
  assert.deepEqual(optionsFromControls({ optional: [...OPTIONAL_STAGES] }).stages, [...ALL_STAGES]);
});

/* Fusion forces the kernel computation but names the kernel graph, so it adds
 * no view: there is no `v/fusion` and a fabricated one would be a rendering
 * presented as the thing it renders. */
test('fusion is a stage but never a view', () => {
  assert.ok(optionsFromControls({ optional: ['fusion'] }).stages.includes('fusion'));
  const index = buildIndex(session());
  assert.equal(index.views.some((v) => v.id === 'v/fusion'), false);
});

test('every effort crosses, and anything else is off', () => {
  for (const effort of ['quick', 'standard', 'thorough']) {
    assert.equal(optionsFromControls({ effort }).verifySymbolic, effort);
  }
  for (const bad of [null, undefined, 'deep', '', 7]) {
    assert.equal(optionsFromControls({ effort: bad }).verifySymbolic, null);
  }
});

test('folding is never requested, because no accepted source can provide it', () => {
  assert.equal(optionsFromControls({ optional: [...OPTIONAL_STAGES], effort: 'quick' }).fold, false);
});

/* ------------------------------------------------------------- request key */

test('a differently spelled but equal selection compares equal', () => {
  const a = requestKey('m', { stages: ['kernel', 'source'], fold: false, verifySymbolic: null });
  const b = requestKey('m', { stages: ['source', 'kernel', 'source'], fold: false, verifySymbolic: null });
  assert.equal(a, b);
});

test('the key separates model, stages and effort', () => {
  const base = { stages: [...ALL_STAGES], fold: false, verifySymbolic: null };
  assert.notEqual(requestKey('a', base), requestKey('b', base));
  assert.notEqual(requestKey('a', base), requestKey('a', { ...base, stages: BACKBONE_STAGES }));
  assert.notEqual(requestKey('a', base), requestKey('a', { ...base, verifySymbolic: 'quick' }));
});

/* -------------------------------------------------------------- selection */

test('the preference is the URL view then source', () => {
  assert.deepEqual(preferredViews('v/kernel'), ['v/kernel', 'v/source']);
  assert.deepEqual(preferredViews(null), ['v/source']);
  assert.deepEqual(preferredViews('v/source'), ['v/source']);
});

/* Staleness is two strings compared, not a flag: "a view was asked for and a
 * different one is showing". */
test('a notice appears exactly when a requested view is not the one shown', () => {
  assert.match(staleNotice('v/kernel', 'v/source'), /no view “v\/kernel”/);
  assert.equal(staleNotice('v/kernel', 'v/kernel'), '');
  assert.equal(staleNotice(null, 'v/source'), '');
  assert.equal(staleNotice(undefined, 'v/source'), '');
  assert.match(staleNotice('v/kernel', null), /the default view/);
});

/* -------------------------------------------------------------------- URL */

test('the URL round-trips a catalogue selection', () => {
  const options = optionsFromControls({ optional: ['kernel'], effort: 'standard' });
  const search = encodeUrl({ model: 'resnet18', options, presentation: singlePresentation('v/kernel') });
  const decoded = decodeUrl(search);
  assert.equal(decoded.model, 'resnet18');
  assert.deepEqual(decoded.presentation, singlePresentation('v/kernel'));
  assert.equal(decoded.verify, 'standard');
  assert.deepEqual(optionsFromUrl(decoded).stages, options.stages);
  assert.equal(requestKey('resnet18', optionsFromUrl(decoded)), requestKey('resnet18', options));
});

test('an unusable value is treated as absent, never as a request for nothing', () => {
  assert.equal(decodeUrl('?verify=deep').verify, null);
  assert.equal(decodeUrl('?stages=nope,alsonope').stages, null);
  assert.deepEqual(decodeUrl('?stages=kernel,nope').stages, ['kernel']);
  assert.deepEqual(optionsFromUrl(decodeUrl('?stages=nope')).stages, [...ALL_STAGES]);
});

/* `fold` is not in this UI's vocabulary. Carrying a decoded `fold=1` through
 * would break both promises the URL makes: `optionsFromControls` returns
 * `fold: false` unconditionally, so the next reload would rebuild a different
 * request, and `requestKey` would never match the controls, so every popstate
 * would reload instead of resolving to a no-op. */
test('a fold parameter is ignored, and never written back', () => {
  const decoded = decodeUrl('?model=m&fold=1');
  assert.equal(optionsFromUrl(decoded).fold, false);
  assert.equal(
    requestKey('m', optionsFromUrl(decoded)),
    requestKey('m', optionsFromUrl(decodeUrl('?model=m'))),
  );
  const written = encodeUrl({ model: 'm', options: optionsFromUrl(decoded), presentation: singlePresentation('v/source') });
  assert.equal(written.includes('fold'), false);
  /* And a following reload builds the identical request. */
  assert.equal(
    requestKey('m', optionsFromControls(controlsFromOptions(optionsFromUrl(decoded)))),
    requestKey('m', optionsFromUrl(decoded)),
  );
});

test('a local source claims no reproducible URL', () => {
  assert.equal(encodeUrl({ model: null, options: optionsFromControls({}), presentation: singlePresentation('v/source') }).includes('model'), false);
  assert.equal(encodeUrl({}), '');
});

/* ------------------------------------------------- options back to controls */

test('the controls follow the normalised options', () => {
  const controls = controlsFromOptions({ stages: ['source', 'canonical', 'kernel'], fold: false, verifySymbolic: 'thorough' });
  assert.deepEqual(controls.optional, ['kernel']);
  assert.equal(controls.effort, 'thorough');
  /* Round trip: what the controls produce, restored, produces the same request. */
  const again = optionsFromControls(controls);
  assert.deepEqual(again.stages, ['source', 'initial_native', 'canonical', 'kernel']);
});

/* ------------------------------------------------------- comparisons */

const comparison = (id, left, right, extra = {}) => ({
  id,
  label: id,
  left: { collection: 'mltorch:model', graph: left },
  right: { collection: 'mltorch:model', graph: right },
  sync: { entries: [], showDiffHighlights: false, matchNodeIdFallback: false },
  overlaysLeft: [],
  overlaysRight: [],
  ...extra,
});

const IMPORT = comparison('c/import', 'pt2/root', 'g/native/000');
const CANONICAL = comparison('c/canonical', 'g/native/000', 'g/native/001');

test('comparisons keep session order and are keyed by their declared id', () => {
  const index = buildIndex(session({ comparisons: [IMPORT, CANONICAL] }));
  assert.deepEqual(index.comparisons.map((c) => c.id), ['c/import', 'c/canonical']);
  assert.equal(index.comparisonById.get('c/canonical').left.graph, 'g/native/000');
  assert.equal(index.comparisonById.has('c/nope'), false);
});

/* None of these can arrive from the bridge -- `Session.validate` rejects a
 * duplicate id and resolves both panes. Dropping them anyway is the difference
 * between a selector entry that does not appear and a pane pair nobody asked
 * for, if that boundary ever changes. */
test('a malformed or duplicate comparison is dropped, never silently chosen', () => {
  const index = buildIndex(session({
    comparisons: [
      IMPORT,
      { ...CANONICAL, id: 'c/import' },              // duplicate id
      { ...comparison('c/nostring', 'a', 'b'), id: 7 },
      { ...comparison('c/nopane', 'a', 'b'), left: null },
      { ...comparison('c/badpane', 'a', 'b'), right: { collection: 'c' } },
      CANONICAL,
    ],
  }));
  assert.deepEqual(index.comparisons.map((c) => c.id), ['c/import', 'c/canonical']);
  // The FIRST declaration of a repeated id survives, not the last.
  assert.equal(index.comparisonById.get('c/import').right.graph, 'g/native/000');
  assert.equal(index.comparisonById.size, 2);
});

test('the index survives a session that declares no comparisons at all', () => {
  const index = buildIndex(session({ comparisons: undefined }));
  assert.deepEqual(index.comparisons, []);
  assert.equal(index.comparisonById.size, 0);
});

/* A comparison names only `{collection, graph}` per pane and labels itself in
 * prose, so a heading is looked up from the view that declares the same graph.
 * First declaration wins, and a graph no view names has no heading to offer. */
test('pane headings come from the view that declares the graph', () => {
  const index = buildIndex(session({}));
  assert.equal(index.viewByGraph.get('pt2/root').label, 'v/source');
  assert.equal(index.viewByGraph.get('g/native/001').label, 'v/canonical');
  assert.equal(index.viewByGraph.has('g/flow'), false);
});

/* ------------------------------------------------ presentation descriptors */

test('a presentation resolves only against what the session declares', () => {
  const index = buildIndex(session({ comparisons: [IMPORT] }));
  assert.deepEqual(resolvePresentation(index, singlePresentation('v/source')),
    singlePresentation('v/source'));
  assert.deepEqual(resolvePresentation(index, comparisonPresentation('c/import')),
    comparisonPresentation('c/import'));
  assert.equal(resolvePresentation(index, singlePresentation('v/nope')), null);
  assert.equal(resolvePresentation(index, comparisonPresentation('c/canonical')), null);
  assert.equal(resolvePresentation(index, null), null);
  // A view id is not a comparison id, and neither crosses into the other branch.
  assert.equal(resolvePresentation(index, comparisonPresentation('v/source')), null);
  assert.equal(resolvePresentation(index, singlePresentation('c/import')), null);
});

test('a non-stage view can never resolve as a single presentation', () => {
  const index = buildIndex(session({
    views: [...SESSION.views, view('v/cmp', 'compare', 'g/native/000')],
  }));
  assert.equal(resolvePresentation(index, singlePresentation('v/cmp')), null);
});

test('two descriptors are the same only within one branch', () => {
  assert.equal(samePresentation(singlePresentation('v/a'), singlePresentation('v/a')), true);
  assert.equal(samePresentation(singlePresentation('v/a'), singlePresentation('v/b')), false);
  assert.equal(samePresentation(comparisonPresentation('c'), comparisonPresentation('c')), true);
  assert.equal(samePresentation(singlePresentation('x'), comparisonPresentation('x')), false);
  assert.equal(samePresentation(defaultPresentation(), singlePresentation('v/source')), true);
});

/* --------------------------------------------------------- the URL branch */

test('the URL carries exactly one presentation branch', () => {
  const options = optionsFromControls({});
  const single = encodeUrl({ model: 'm', options, presentation: singlePresentation('v/kernel') });
  assert.equal(new URLSearchParams(single).get('view'), 'v/kernel');
  assert.equal(new URLSearchParams(single).has('comparison'), false);

  const compare = encodeUrl({ model: 'm', options, presentation: comparisonPresentation('c/import') });
  assert.equal(new URLSearchParams(compare).get('comparison'), 'c/import');
  assert.equal(new URLSearchParams(compare).has('view'), false);

  assert.deepEqual(decodeUrl(compare).presentation, comparisonPresentation('c/import'));
});

/* Both keys present names NEITHER: any tie-break would open a presentation the
 * link did not unambiguously ask for. */
test('a URL naming both a view and a comparison names no presentation', () => {
  assert.equal(decodeUrl('?view=v/source&comparison=c/import').presentation, null);
  assert.equal(decodeUrl('?model=m').presentation, null);
});

test('a stale comparison says it fell out of compare mode, not into another one', () => {
  assert.equal(staleComparisonNotice('c/nope', null).includes('c/nope'), true);
  assert.equal(staleComparisonNotice('c/nope', null).includes('single view'), true);
  // Resolved, or never asked for: nothing to say.
  assert.equal(staleComparisonNotice('c/import', comparisonPresentation('c/import')), '');
  assert.equal(staleComparisonNotice(null, null), '');
});

/* ------------------------------------------------------------------- flow */

const FLOW_STATES = [
  { id: 's/pt2/000', graph: 'pt2/root', view: 'v/source', layer: 'pt2', label: 'exported program' },
  { id: 's/native/000', graph: 'g/native/000', view: 'v/initial', layer: 'native', label: 'initial', producedBy: 't/native/000' },
  { id: 's/native/001', graph: 'g/native/001', view: 'v/canonical', layer: 'native', label: 'canonical', producedBy: 't/native/001' },
];
const FLOW_TRANSITIONS = [
  { id: 't/native/000', before: 's/pt2/000', after: 's/native/000', kind: { kind: 'import' }, comparison: 'c/import' },
  { id: 't/native/001', before: 's/native/000', after: 's/native/001', kind: { kind: 'pack' } },
];

const FLOW_VIEW = view('v/flow', 'flow', 'g/flow');
const flowed = (overrides = {}) => session({
  views: [...SESSION.views, FLOW_VIEW],
  comparisons: [IMPORT],
  flow: { states: FLOW_STATES, transitions: FLOW_TRANSITIONS, graph: 'g/flow' },
  ...overrides,
});

test('the flow is indexed only when its whole destination resolves', () => {
  const index = buildIndex(flowed());
  assert.equal(index.flowView.id, 'v/flow');
  assert.deepEqual([...index.stateById.keys()], ['s/pt2/000', 's/native/000', 's/native/001']);
  assert.deepEqual([...index.transitionById.keys()], ['t/native/000', 't/native/001']);
});

/* `feature:flow` is `available` in this fixture and names `g/flow`, exactly as
 * the real exporter emits -- yet with no flow view there is nothing to open. A
 * capability is never a destination, which is the rule this whole module
 * exists to keep. */
test('an available flow capability never makes a destination on its own', () => {
  const index = buildIndex(session({}));
  assert.equal(index.flowView, null);
  assert.equal(index.flow, null);
  assert.equal(index.stateById.size, 0);
  assert.equal(
    index.capabilityByKey.get('feature:flow').status.state, 'available',
    'the fixture must keep the capability available, or this proves nothing',
  );
});

test('a flow view naming another graph than the spine is not a destination', () => {
  const index = buildIndex(flowed({
    views: [...SESSION.views, view('v/flow', 'flow', 'g/native/000')],
  }));
  assert.equal(index.flowView, null);
  assert.equal(index.flow, null);
});

test('two flow views resolve to none rather than to a guess', () => {
  const index = buildIndex(flowed({
    views: [...SESSION.views, FLOW_VIEW, view('v/flow2', 'flow', 'g/flow')],
  }));
  assert.equal(index.flowView, null);
});

/* A broken link makes ONE node non-actionable; it never falls back to routing
 * by graph, label or order. */
test('a state whose view is missing or shows another graph is dropped', () => {
  const index = buildIndex(flowed({
    flow: {
      graph: 'g/flow',
      states: [
        { ...FLOW_STATES[0], view: 'v/nope' },
        { ...FLOW_STATES[1], view: 'v/canonical' },   // a stage view, wrong graph
        FLOW_STATES[2],
      ],
      transitions: FLOW_TRANSITIONS,
    },
  }));
  assert.deepEqual([...index.stateById.keys()], ['s/native/001']);
});

test('a transition naming a comparison that is gone is dropped, an unpaired one is not', () => {
  const index = buildIndex(flowed({
    flow: {
      graph: 'g/flow',
      states: FLOW_STATES,
      transitions: [
        { ...FLOW_TRANSITIONS[0], comparison: 'c/nope' },
        FLOW_TRANSITIONS[1],
      ],
    },
  }));
  // The unpaired one survives: honest absence is not a broken link.
  assert.deepEqual([...index.transitionById.keys()], ['t/native/001']);
  assert.equal(index.transitionById.get('t/native/001').comparison, undefined);
});

test('a flow presentation resolves only against the one flow view', () => {
  const index = buildIndex(flowed());
  assert.deepEqual(resolvePresentation(index, flowPresentation('v/flow')), flowPresentation('v/flow'));
  assert.equal(resolvePresentation(index, flowPresentation('v/source')), null);
  assert.equal(resolvePresentation(index, flowPresentation('v/nope')), null);
});

/* The strict-route rule, in both directions. A flow view reachable through
 * `{kind:'single'}` would make the renderer's stage-only route reachable by a
 * typo, which is the one thing it exists to prevent. */
test('a flow view is not a single view, and a stage view is not a flow', () => {
  const index = buildIndex(flowed());
  assert.equal(resolvePresentation(index, singlePresentation('v/flow')), null);
  assert.equal(index.viewById.has('v/flow'), false);
  assert.equal(resolvePresentation(index, flowPresentation('v/canonical')), null);
});

test('the URL carries a flow branch, exclusive with the other two', () => {
  const options = optionsFromControls({});
  const written = encodeUrl({ model: 'm', options, presentation: flowPresentation('v/flow') });
  assert.equal(new URLSearchParams(written).get('flow'), 'v/flow');
  assert.equal(new URLSearchParams(written).has('view'), false);
  assert.equal(new URLSearchParams(written).has('comparison'), false);
  assert.deepEqual(decodeUrl(written).presentation, flowPresentation('v/flow'));
});

test('any two presentation keys in one URL name none', () => {
  assert.equal(decodeUrl('?flow=v/flow&view=v/source').presentation, null);
  assert.equal(decodeUrl('?flow=v/flow&comparison=c/import').presentation, null);
  assert.equal(decodeUrl('?view=v/source&comparison=c/import').presentation, null);
  assert.equal(decodeUrl('?view=v/source&comparison=c/import&flow=v/flow').presentation, null);
  // And exactly one still decodes.
  assert.deepEqual(decodeUrl('?flow=v/flow').presentation, flowPresentation('v/flow'));
});

test('a stale flow says it fell out of flow mode', () => {
  assert.equal(staleFlowNotice('v/flow', null).includes('v/flow'), true);
  assert.equal(staleFlowNotice('v/flow', null).includes('single view'), true);
  assert.equal(staleFlowNotice('v/flow', flowPresentation('v/flow')), '');
  assert.equal(staleFlowNotice(null, null), '');
});
