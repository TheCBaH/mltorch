import assert from 'node:assert/strict';
import test from 'node:test';
import {
  ALL_STAGES, BACKBONE_STAGES, OPTIONAL_STAGES, RANK_BUCKETS,
  bucketOfRank, buildIndex, capabilityWording, controlsFromOptions, decodeUrl,
  encodeUrl, optionsFromControls, optionsFromUrl, preferredViews, requestKey,
  staleNotice,
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
  const search = encodeUrl({ model: 'resnet18', options, view: 'v/kernel' });
  const decoded = decodeUrl(search);
  assert.equal(decoded.model, 'resnet18');
  assert.equal(decoded.view, 'v/kernel');
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
  const written = encodeUrl({ model: 'm', options: optionsFromUrl(decoded), view: 'v/source' });
  assert.equal(written.includes('fold'), false);
  /* And a following reload builds the identical request. */
  assert.equal(
    requestKey('m', optionsFromControls(controlsFromOptions(optionsFromUrl(decoded)))),
    requestKey('m', optionsFromUrl(decoded)),
  );
});

test('a local source claims no reproducible URL', () => {
  assert.equal(encodeUrl({ model: null, options: optionsFromControls({}), view: 'v/source' }).includes('model'), false);
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
