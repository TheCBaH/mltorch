/* What the page knows about a session, and nothing it does not.
 *
 * Pure: no DOM, no fetch, no renderer. Everything here is a function of a
 * bridge-validated session document plus the browser's own control state, which
 * is what makes it testable under `node --test` without a fake DOM.
 *
 * The one rule this module exists to enforce: a *navigable graph* comes from a
 * `View`, never from a capability. `feature:flow` is `available` today and names
 * `g/flow`, a graph no collection holds and no view points at -- so a page that
 * treated a capability's graph id as somewhere it could go would offer a
 * destination that does not exist.
 */

/* The stage vocabulary, in `Me_session.Capability.all_stages` order. Source,
 * Initial Native and Canonical are export backbones rather than choices: the
 * exporter builds them for every session, every comparison rests on them, and
 * `Me_request.Options.create` normalises by filtering this order without ever
 * ADDING to it -- so if the page did not send them, nothing would. */
export const BACKBONE_STAGES = Object.freeze(['source', 'initial_native', 'canonical']);
export const OPTIONAL_STAGES = Object.freeze(['native4d', 'stage_program', 'kernel', 'fusion']);
export const ALL_STAGES = Object.freeze([...BACKBONE_STAGES, ...OPTIONAL_STAGES]);
export const EFFORTS = Object.freeze(['quick', 'standard', 'thorough']);

const SOURCE_VIEW = 'v/source';
const STAGE_KIND = 'stage:';

const canonicalStages = (stages) => {
  const wanted = new Set(Array.isArray(stages) ? stages : []);
  return ALL_STAGES.filter((stage) => wanted.has(stage));
};

/* ------------------------------------------------------------ rank buckets */

/* `Map_verify.Verdict.rank` is HOW WEAK a verdict is: 0 vacuous, 1-2 proved,
 * 3-4 tested, 5 unproved, 6 refuted. Bucketing keys on that number and never on
 * the label text -- the label carries strength and coverage (`proved
 * (structural) [sampled 4]`) and is displayed verbatim, so parsing it back into
 * a category would be re-deriving something the rank already states exactly.
 *
 * Vacuous is neutral, not positive: a creation or a deletion claims nothing.
 * It sits below proved only so a claimless cluster cannot mask a real verdict
 * when a node's outputs are joined. */
export const RANK_BUCKETS = Object.freeze([
  { name: 'vacuous', ranks: [0], bg: '#e0e3e7', title: 'Vacuous', meaning: 'creation or deletion; no equivalence claim' },
  { name: 'proved', ranks: [1, 2], bg: '#a5d6a7', title: 'Proved', meaning: 'symbolic equality at the stated strength and coverage' },
  { name: 'tested', ranks: [3, 4], bg: '#90caf9', title: 'Tested', meaning: 'deterministic probe evidence; not a proof' },
  { name: 'unproved', ranks: [5], bg: '#ffcc80', title: 'Unproved', meaning: 'no claim established, often a budget or an unsupported tier' },
  { name: 'refuted', ranks: [6], bg: '#e53935', title: 'Refuted', meaning: 'a counterexample was found' },
]);

/** The bucket for a rank, or `null` for a value outside the known scale. */
export function bucketOfRank(rank) {
  if (!Number.isInteger(rank)) return null;
  return RANK_BUCKETS.find((bucket) => bucket.ranks.includes(rank)) ?? null;
}

/* ------------------------------------------------------------ capabilities */

const KEY_NAMES = Object.freeze({
  'stage:source': 'Exported Program',
  'stage:initial_native': 'Initial Native',
  'stage:canonical': 'Canonical Native',
  'stage:native4d': 'Native4D',
  'stage:stage_program': 'Stage Program',
  'stage:kernel': 'Kernel',
  'stage:fusion': 'Kernel fusion',
  'feature:flow': 'Export flow',
  'feature:verification': 'Symbolic verification',
  'feature:pass_audits': 'Per-pass audits',
  'feature:fold': 'Constant/payload folding',
  'feature:expression_detail': 'Expression detail',
  'feature:loop_ir': 'Loop IR',
  'feature:codegen': 'Code generation',
});

/* `Capability.reason`, spelled for a reader. A reason the exporter adds later
 * falls through to its own wire tag rather than being described wrongly. */
const REASONS = Object.freeze({
  unsupported_operator: 'an operator this build does not support',
  unsupported_input: 'an input this build does not support',
  unsupported_graph_shape: 'a graph shape this build does not support',
  outside_dialect_domain: 'the graph is outside that dialect’s domain',
  over_limit: 'a resource limit was reached',
  requires_payloads: 'this source carries no constant payloads',
  prerequisite_unavailable: 'a prerequisite output is unavailable',
  not_implemented: 'not implemented',
});

/* Never `Available`, and never `Not_requested` either -- there is nothing to
 * ask for. Saying "not requested" here would read as "you did not ask". */
const PERMANENTLY_ABSENT = new Set(['feature:loop_ir', 'feature:codegen']);

/**
 * How one capability row should read. Four states, deliberately distinct:
 * `available`, `not_requested` (you can ask for this), `unavailable` (you did,
 * and here is the exporter's own reason), `unimplemented` (it does not exist).
 */
export function capabilityWording(capability) {
  const key = capability?.key ?? '';
  const name = KEY_NAMES[key] ?? key;
  const status = capability?.status ?? {};
  if (PERMANENTLY_ABSENT.has(key)) {
    return { key, name, state: 'unimplemented', title: 'Not implemented', detail: 'this output does not exist yet, so it cannot be requested' };
  }
  if (status.state === 'available') return { key, name, state: 'available', title: 'Available', detail: '' };
  if (status.state === 'not_requested') {
    return { key, name, state: 'not_requested', title: 'Not requested', detail: 'this output was left out of the request' };
  }
  const reason = REASONS[status.reason] ?? status.reason ?? 'no reason given';
  const detail = status.detail ? `${reason} — ${status.detail}` : reason;
  return { key, name, state: 'unavailable', title: 'Unavailable', detail };
}

/* --------------------------------------------------------------- the index */

const isStageView = (view) => typeof view?.kind === 'string' && view.kind.startsWith(STAGE_KIND);

/* Only when the row is `available` AND carries the payload its key requires.
 * `Capability.compatible` guarantees that pairing on the producing side; this
 * re-checks it because reading `.verificationSummary` off an `unavailable` row
 * would otherwise quietly yield `undefined` and render as "no findings". */
function payload(capability, kind) {
  const status = capability?.status;
  if (status?.state !== 'available') return null;
  return status.payload?.kind === kind ? status.payload : null;
}

/* Declared comparisons, in session order, keyed by their own id.
 *
 * `Session.validate` already guarantees the ids are unique and both panes
 * resolve, so this re-reads rather than re-validates -- but it DROPS a record
 * whose id is absent, non-string or repeated instead of letting one silently
 * win. That cannot arrive from the bridge today; if the boundary ever changes,
 * an absent comparison is a selector entry that does not appear, while a
 * silently chosen one is a pane pair nobody asked for.
 *
 * A pane's shape is checked here too, for the same reason: the selector must
 * not offer a comparison the renderer will then refuse before connection.
 */
const pane = (p) =>
  p && typeof p.collection === 'string' && typeof p.graph === 'string';

function indexComparisons(raw) {
  const comparisons = [];
  const comparisonById = new Map();
  for (const c of Array.isArray(raw) ? raw : []) {
    if (!c || typeof c.id !== 'string' || c.id === '') continue;
    if (comparisonById.has(c.id)) continue;
    if (!pane(c.left) || !pane(c.right)) continue;
    comparisonById.set(c.id, c);
    comparisons.push(c);
  }
  return { comparisons, comparisonById };
}

/**
 * The defensive index. The document has already passed
 * `Me_session.Session.validate` in the bridge, so this re-reads rather than
 * re-validates: a missing or malformed field becomes absent, never a throw,
 * because a panel that cannot render is not a reason to lose the graph.
 */
export function buildIndex(sessionText) {
  const session = typeof sessionText === 'string' ? JSON.parse(sessionText) : sessionText;
  const capabilities = Array.isArray(session?.capabilities) ? session.capabilities : [];
  const capabilityByKey = new Map(capabilities.map((c) => [c?.key, c]));
  /* `flow` and `compare` kinds are excluded: neither is a single-graph
   * destination this delivery can open. */
  const views = (Array.isArray(session?.views) ? session.views : []).filter(isStageView);
  const verification = payload(capabilityByKey.get('feature:verification'), 'verification_summary');
  const audits = payload(capabilityByKey.get('feature:pass_audits'), 'pass_audit_status');
  const { comparisons, comparisonById } = indexComparisons(session?.comparisons);
  return {
    model: session?.model ?? null,
    defaultView: typeof session?.defaultView === 'string' ? session.defaultView : null,
    views,
    viewById: new Map(views.map((v) => [v.id, v])),
    /* A comparison names only `{collection, graph}` per pane, and its own label
     * is prose ("exported program -> initial"). Pane HEADINGS come from the
     * stage view that declares the same graph, which is a lookup rather than a
     * derivation -- no graph id is ever built from a stage name. First
     * declaration wins; a graph no view names falls back to its own id. */
    viewByGraph: new Map(views.map((v) => [v.graph, v]).reverse()),
    capabilities,
    capabilityByKey,
    diagnostics: Array.isArray(session?.diagnostics) ? session.diagnostics : [],
    comparisons,
    comparisonById,
    nodeDataSets: Array.isArray(session?.nodeDataSets) ? session.nodeDataSets : [],
    /* Arrays of `{label, count}` where `count` is a DECIMAL STRING
     * (`int64_as_string`): displayed verbatim, never parsed through `Number`,
     * which loses exactness past 2^53. */
    verification: verification?.verificationSummary ?? null,
    passAudits: audits?.passAuditStatus ?? null,
  };
}

/* ------------------------------------------------------------- selection */

/**
 * The ordered view preference for a load. The renderer resolves it against the
 * arriving document and enforces stage-only at every rung; this only says what
 * the page would like. Source first is a browser default for reading order --
 * the exporter keeps Canonical as its own `defaultView` for the CLI.
 */
export function preferredViews(urlView) {
  return urlView && urlView !== SOURCE_VIEW ? [urlView, SOURCE_VIEW] : [SOURCE_VIEW];
}

/* ---------------------------------------------------------- presentation */

/**
 * What the page is showing, as a closed sum with exactly one branch active.
 *
 * This replaces "the current view id" at the app boundary. A comparison is not
 * a view and cannot be encoded as one: it names two graphs, and `View` names
 * one. Phase 4 adds a third branch (`flow`), which is why every consumer
 * switches on `kind` rather than testing for the presence of a field.
 */
export const singlePresentation = (view) => ({ kind: 'single', view });
export const comparisonPresentation = (comparison) => ({ kind: 'comparison', comparison });

/** The browser's own default: source first, for reading order. */
export const defaultPresentation = () => singlePresentation(SOURCE_VIEW);

/**
 * A descriptor resolved against a session, or `null` if it names nothing the
 * document declares. `null` is the whole error vocabulary on purpose: every
 * caller's response is the same -- fall back and say so -- and a richer result
 * would invite a caller to render a destination that does not exist.
 */
export function resolvePresentation(index, presentation) {
  if (!index || !presentation) return null;
  if (presentation.kind === 'single') {
    return index.viewById.has(presentation.view) ? presentation : null;
  }
  if (presentation.kind === 'comparison') {
    return index.comparisonById.has(presentation.comparison) ? presentation : null;
  }
  return null;
}

/** True when two descriptors name the same thing; used to skip no-op changes. */
export const samePresentation = (a, b) =>
  a?.kind === b?.kind
  && (a?.kind === 'comparison' ? a.comparison === b.comparison : a?.view === b?.view);

/**
 * Non-fatal text when a URL asked for a comparison this session does not
 * declare. Separate from `staleNotice` rather than a generalisation of it: the
 * two say different things -- a stale view falls back to another view, a stale
 * comparison falls back out of compare mode entirely -- and merging them would
 * make one sentence describe two different retreats.
 */
export function staleComparisonNotice(urlComparison, resolved) {
  if (!urlComparison || resolved) return '';
  return `This model has no comparison “${urlComparison}”; showing a single view instead.`;
}

/**
 * Non-fatal text when a URL asked for a view it did not get. No flag is needed
 * and none exists: staleness is exactly "a view was requested and a different
 * one is showing", which is two strings the caller already holds.
 */
export function staleNotice(urlView, viewId) {
  if (!urlView || urlView === viewId) return '';
  return `This model has no view “${urlView}”; showing ${viewId ? `“${viewId}”` : 'the default view'} instead.`;
}

/* --------------------------------------------------- controls and requests */

/**
 * The one place control state becomes a request. The backbones are appended
 * unconditionally, so the list is never empty and they can never be switched
 * off; optional stages are filtered through `OPTIONAL_STAGES` so the result is
 * already in `all_stages` order.
 *
 * `fold` is always false: no source this UI accepts can make folding available
 * (`feature:fold` is `unavailable requires_payloads` for every `model.json`),
 * so the control is inert and sending anything else would request a
 * configuration the page cannot display or reproduce.
 */
export function optionsFromControls({ optional = [], effort = null } = {}) {
  const wanted = new Set(optional);
  return {
    stages: [...BACKBONE_STAGES, ...OPTIONAL_STAGES.filter((s) => wanted.has(s))],
    fold: false,
    verifySymbolic: EFFORTS.includes(effort) ? effort : null,
  };
}

/**
 * The identity of a request, for deciding load-versus-present on a history
 * transition. Stages are canonicalised so a differently-spelled but equal
 * selection compares equal, exactly as `Options.create` treats it.
 */
export function requestKey(model, options) {
  return JSON.stringify([
    model ?? null,
    canonicalStages(options?.stages),
    options?.fold === true,
    EFFORTS.includes(options?.verifySymbolic) ? options.verifySymbolic : null,
  ]);
}

/* --------------------------------------------------------------- the URL */

/**
 * Catalogue sources only. `fold` is deliberately absent from this vocabulary:
 * the control is inert, so a decoded `fold=1` could not survive the next
 * `optionsFromControls()` and would make both the reproducible URL and
 * `requestKey` unstable. It joins when PT2 sources make the control real.
 *
 * An unusable value is treated as absent rather than as an error: an unknown
 * stage name is dropped, and a `stages` that names nothing known falls back to
 * the default rather than to a request for nothing.
 */
export function decodeUrl(search) {
  const query = new URLSearchParams(search);
  const rawStages = query.get('stages');
  const stages = rawStages === null ? null : canonicalStages(rawStages.split(','));
  const verify = query.get('verify');
  const view = query.get('view');
  const comparison = query.get('comparison');
  return {
    model: query.get('model'),
    /* `view` and `comparison` are branches of one closed choice, so a URL
     * carrying both names NEITHER: there is no rule for picking a winner that
     * is not arbitrary, and guessing would open a presentation the link did not
     * unambiguously ask for. Phase 4's `flow` joins the same exclusion. */
    presentation: view && comparison ? null
      : comparison ? comparisonPresentation(comparison)
      : view ? singlePresentation(view)
      : null,
    stages: stages && stages.length > 0 ? stages : null,
    verify: EFFORTS.includes(verify) ? verify : null,
  };
}

/**
 * The query string for a reproducible catalogue selection, leading `?` included.
 *
 * Exactly one presentation key is ever written, and it is written from the
 * BRANCH rather than from two optional arguments -- which is what makes the
 * exclusivity above structural instead of a rule each caller has to remember.
 */
export function encodeUrl({ model, options, presentation } = {}) {
  const query = new URLSearchParams();
  if (model) query.set('model', model);
  const stages = canonicalStages(options?.stages);
  if (stages.length > 0) query.set('stages', stages.join(','));
  if (EFFORTS.includes(options?.verifySymbolic)) query.set('verify', options.verifySymbolic);
  if (presentation?.kind === 'comparison' && presentation.comparison) {
    query.set('comparison', presentation.comparison);
  } else if (presentation?.kind === 'single' && presentation.view) {
    query.set('view', presentation.view);
  }
  const text = query.toString();
  return text ? `?${text}` : '';
}

/** Decoded URL options as the object `load()` takes. */
export function optionsFromUrl(decoded) {
  return {
    stages: decoded?.stages ?? [...ALL_STAGES],
    fold: false,
    verifySymbolic: decoded?.verify ?? null,
  };
}

/**
 * The control state an options object implies. Used both to restore the
 * checkboxes from a URL and to make them follow the bridge's NORMALISED echo
 * after a load, so what is ticked is always what was actually requested.
 */
export function controlsFromOptions(options) {
  const stages = options?.stages ?? [...ALL_STAGES];
  return {
    optional: OPTIONAL_STAGES.filter((stage) => stages.includes(stage)),
    effort: EFFORTS.includes(options?.verifySymbolic) ? options.verifySymbolic : null,
  };
}

/** The control state a decoded URL implies. */
export const controlsFromUrl = (decoded) => controlsFromOptions(optionsFromUrl(decoded));
