/* Path B install cancellation.
 *
 * The pinned `ai-edge-model-explorer-visualizer@0.1.2` has no abort or dispose
 * API, and disconnecting an element that is still processing throws Angular's
 * NG0953 ("emit for destroyed OutputRef"). So no install can actually be
 * stopped. What a cancelled install loses is *authority* -- the right to become
 * visible, to be committed, or to report status -- never its DOM connection.
 *
 * Every candidate therefore gets its own stable, connected, `visibility:hidden`
 * slot (see `.visualizer-slot` in app.css; never `display:none`, which stops
 * the element processing). A candidate is physically removed only once its
 * expected `modelGraphProcessed` event has proved removal safe. Elements whose
 * event has not arrived are *quarantined*, and quarantine is bounded: the
 * ceiling comes from `Me_limits.Hard.max_quarantined_elements`, against which
 * `response_live_bytes` budgets the browser's retained elements.
 */

import { bucketOfRank } from './presentation.js';

const HEARTBEAT_MS = 5_000;
const SLOT_CLASS = 'visualizer-slot';
const CURRENT_CLASS = 'visualizer-slot--current';
/* How many times one slot's removal may be re-attempted before the renderer
 * stops promising to try again. */
const MAX_DETACH_ATTEMPTS = 3;

const text = (error) => (error && error.message) || String(error);
/* What a candidate is for, for a log line. A comparison names two graphs, so
 * "graph g/native/000" would describe half of one. */
const describe = (d) => (d.kind === 'comparison'
  ? `comparison ${d.id} (${d.left.graph} | ${d.right.graph})`
  : d.kind === 'flow' ? `flow ${d.viewId} (${d.graphId})`
  : `graph ${d.graphId}`);
/* Safe, still-connected elements the renderer failed to remove. They are not
 * inconsistency -- the DOM is coherent -- but they hold quarantine slots. */
const isDebt = (state) => state === 'cleanup_failed' || state === 'cleanup_abandoned';

/* `View.kind` is `stage:<name>`, `flow`, or `compare`. Only a stage view names
 * a graph this shell can open as a single view, so every selection rung below
 * filters on this and not merely on "the id resolves". */
const isStage = (view) => typeof view?.kind === 'string' && view.kind.startsWith('stage:');

/* The one graph-addressed set the shell interprets rather than merely relays.
 * `Me_verify` names it, and `Me_fusion`'s "fusion" is the only other. */
const VERIFICATION_SET = 'verification';

/* A `Node_data_set` result is `{ nodeId, value: { value, label? } }` -- an
 * OBJECT, not a pair. Reading it as a pair throws "object is not iterable",
 * which for a long time nothing noticed: the only set addressed to the default
 * view is the verification one, and the browser never asked for verification. */
const entries = (set) => (set.results || []).map((r) => [r.nodeId, r.value ?? {}]);

/* Everything except verification, on the original anonymous gradient. */
function gradientData(set) {
  return {
    results: Object.fromEntries(entries(set).map(([id, v]) => [id, { value: v.value }])),
    gradient: [{ stop: 0, bgColor: '#e8f5e9' }, { stop: 1, bgColor: '#ffcdd2' }],
  };
}

/* Verification, as named rank buckets carrying the exporter's own label.
 *
 * The pinned element derives the text it shows (`strValue`) from `value`, and
 * reads NO `label` key anywhere -- so the supplied
 * "proved (structural) [sampled 4]" can only reach the screen AS the value,
 * which is why these are strings; `typeof value === 'string'` is explicitly
 * handled in `processNodeDataProviderDataForGraph`. A per-result `bgColor`
 * takes precedence over any gradient there, so the buckets need no other API,
 * and `textColor` is omitted because the element derives a contrasting one by
 * luminance.
 *
 * Bucketing keys on the numeric rank, never on the label text: the rank already
 * states the category exactly, and re-deriving it from prose would be the one
 * place a shortened "proved" could creep in. */
function verificationData(set) {
  const results = {};
  for (const [nodeId, value] of entries(set)) {
    const bucket = bucketOfRank(value.value);
    results[nodeId] = {
      value: value.label ?? String(value.value),
      ...(bucket ? { bgColor: bucket.bg } : {}),
    };
  }
  return { results };
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

/** The renderer's classified failures. `kind` replaces message matching. */
export class RenderFailure extends Error {
  constructor(kind, message) {
    super(message);
    this.name = 'RenderFailure';
    this.kind = kind;
  }
}

const integer = (value, name, minimum) => {
  if (!Number.isInteger(value) || value < minimum) {
    throw new Error(`${name} must be an integer >= ${minimum}`);
  }
  return value;
};

export class Renderer {
  #mount;
  #doc;
  #timers;
  #now;
  #maxQuarantined;
  #onFlowSelect;
  #active = null;
  #quarantined = new Set();
  /* handle -> entry, or a tombstone `{consumedBy}` once the entry has consumed
   * it. A tombstone is what keeps a handle the coordinator legitimately still
   * holds distinguishable from a forged or twice-used one. */
  #handles = new WeakMap();
  #inconsistent = null;

  constructor({
    mount,
    hardMaxQuarantined,
    maxQuarantined,
    timeout = 90_000,
    capacityTimeout = 10_000,
    doc = globalThis.document,
    onFlowSelect = () => {},
    timers = globalThis,
    now = () => globalThis.performance?.now?.() ?? Date.now(),
  }) {
    // `hardMaxQuarantined` is mandatory: a caller that omits it must fail here
    // rather than silently pick a literal that the OCaml budget does not know
    // about. A requested policy may tighten the ceiling, never widen it.
    integer(hardMaxQuarantined, 'hardMaxQuarantined', 1);
    if (maxQuarantined !== undefined) integer(maxQuarantined, 'maxQuarantined', 0);
    this.#mount = mount;
    this.#doc = doc;
    this.#onFlowSelect = onFlowSelect;
    this.#timers = timers;
    this.#now = now;
    this.#maxQuarantined = Math.min(maxQuarantined ?? hardMaxQuarantined, hardMaxQuarantined);
    this.timeout = timeout;
    this.capacityTimeout = capacityTimeout;
    this.current = null;
  }

  get maxQuarantined() { return this.#maxQuarantined; }
  get quarantined() { return this.#quarantined.size; }
  get inconsistent() { return this.#inconsistent; }

  /* ---------------------------------------------------------------- install */

  /* `selection` is `{ view }` (exact -- an unresolvable or non-stage id fails),
   * `{ prefer: [ids] }` (first stage view that resolves, then `defaultView`,
   * then any stage view), or omitted (the original behaviour). It is resolved
   * inside `#parse`, so a bad selection is refused BEFORE a candidate exists. */
  async install(renderText, selection) {
    if (this.#inconsistent) throw new RenderFailure('inconsistent', this.#inconsistent);
    // Parsing precedes everything fallible, and allocates no entry, no deferred
    // and no DOM: an invalid document is refused without the renderer having
    // mutated anything on its behalf.
    const descriptor = this.#parse(renderText, selection);
    // Both remaining preflight steps can mutate the DOM, and either can discover
    // inconsistency. `markInconsistent` settles whatever entry is active at that
    // moment -- never the newcomer -- so the newcomer must not exist yet, or it
    // would be orphaned with its deferred pending forever.
    this.#sweepDebt();
    if (this.#inconsistent) throw new RenderFailure('inconsistent', this.#inconsistent);
    this.#supersede();
    if (this.#inconsistent) throw new RenderFailure('inconsistent', this.#inconsistent);
    const entry = this.#createEntry(descriptor);
    this.#active = entry;
    this.#start(entry);
    return entry.promise;
  }

  /* Stage-only at EVERY rung, not merely for the ids a caller asked for.
   * `Session.validate` requires `defaultView` to name *some* declared view and
   * says nothing about its kind, so a bridge-valid session may default to a
   * `flow` or `compare` view -- and a chain that filtered the requested ids but
   * then fell through to `defaultView` would re-open the excluded view by the
   * back door. */
  #select(state, selection) {
    const views = Array.isArray(state.views) ? state.views : [];
    const stage = (id) => {
      const view = views.find((v) => v.id === id);
      return view && isStage(view) ? view : null;
    };
    if (selection?.view != null) {
      const view = stage(selection.view);
      if (!view) {
        throw new RenderFailure('invalid', `session has no stage view "${selection.view}"`);
      }
      return { graphId: view.graph, viewId: view.id };
    }
    if (Array.isArray(selection?.prefer)) {
      for (const id of selection.prefer) {
        const view = stage(id);
        if (view) return { graphId: view.graph, viewId: view.id };
      }
      const fallback = stage(state.defaultView) ?? views.find(isStage);
      if (!fallback) throw new RenderFailure('invalid', 'session declares no stage view');
      return { graphId: fallback.graph, viewId: fallback.id };
    }
    // No descriptor: the original behaviour, kind filter and all, so the
    // callers and tests that predate selection keep their contract.
    const view = views.find((v) => v.id === state.defaultView);
    return {
      graphId: view?.graph || state.graphCollections?.[0]?.graphs?.[0]?.id,
      viewId: view?.id ?? null,
    };
  }

  /* One pane, resolved against the collection that DECLARES it.
   *
   * `Pane_state` names a collection as well as a graph, and the pair is what
   * `Session.validate` checks -- a graph id that exists in some OTHER
   * collection is `Wrong_collection` there, and must be a refusal here too
   * rather than a lookup that happens to find something. */
  #pane(collections, side, label) {
    const holder = collections.find((c) => c.label === side?.collection);
    if (!holder) {
      throw new RenderFailure('invalid', `${label}: session has no collection "${side?.collection}"`);
    }
    const graph = (holder.graphs ?? []).find((g) => g.id === side.graph);
    if (!graph) {
      throw new RenderFailure('invalid',
        `${label}: collection "${side.collection}" has no graph "${side.graph}"`);
    }
    const firstNodeId = graph.nodes?.[0]?.id;
    // A graph with no node cannot be navigated to: `selectNode` is the only way
    // to reach a pane, and it takes a node. Refusing here keeps that a
    // pre-connection failure rather than a candidate that never becomes ready.
    if (typeof firstNodeId !== 'string') {
      throw new RenderFailure('invalid', `${label}: graph "${side.graph}" has no node to select`);
    }
    return { graph: side.graph, collectionLabel: holder.label, firstNodeId };
  }

  /* Exactly one [View.Flow], resolved before anything is allocated. Kept apart
   * from [#select] on purpose: the stage-only route stays strict, so a flow
   * view can never enter it and a stage view can never enter this one. A typo
   * is a refusal, never a silently different presentation. */
  #selectFlow(state, viewId) {
    const views = Array.isArray(state.views) ? state.views : [];
    const found = views.filter((v) => v?.id === viewId);
    if (found.length === 0) {
      throw new RenderFailure('invalid', `session has no view "${viewId}"`);
    }
    if (found.length > 1) {
      throw new RenderFailure('invalid', `session declares view "${viewId}" more than once`);
    }
    const view = found[0];
    if (view.kind !== 'flow') {
      throw new RenderFailure('invalid', `view "${viewId}" is not a flow view`);
    }
    const collections = state.graphCollections ?? [];
    const pane = this.#pane(collections, { collection: view.collection, graph: view.graph },
      `flow view "${viewId}"`);
    return {
      kind: 'flow',
      viewId,
      graphId: pane.graph,
      collectionLabel: pane.collectionLabel,
      targetFirstNodeId: pane.firstNodeId,
    };
  }

  /* Every field the candidate needs, resolved BEFORE anything is allocated.
   * `Session.validate` has already accepted this document, so these are
   * re-reads -- but an exact request that does not resolve is a refusal, never
   * a fall-through to some other comparison or to `defaultView`. */
  #selectComparison(state, id) {
    const declared = Array.isArray(state.comparisons) ? state.comparisons : [];
    const found = declared.filter((c) => c?.id === id);
    if (found.length === 0) {
      throw new RenderFailure('invalid', `session has no comparison "${id}"`);
    }
    if (found.length > 1) {
      throw new RenderFailure('invalid', `session declares comparison "${id}" more than once`);
    }
    const comparison = found[0];
    const collections = state.graphCollections ?? [];
    const sync = comparison.sync ?? {};
    const entries = Array.isArray(sync.entries) ? sync.entries : [];
    const mappingEntries = entries.map((entry, index) => {
      const left = entry?.left;
      const right = entry?.right;
      if (!Array.isArray(left) || !Array.isArray(right)) {
        throw new RenderFailure('invalid', `comparison "${id}": mapping entry ${index} is malformed`);
      }
      // The two arrays cross as ARRAYS. Model Explorer's `mappingEntries` is
      // the 1:N/N:1/N:M representation; expanding one into pairs would turn a
      // single correspondence component into a Cartesian product of claims.
      return { leftNodeIds: [...left], rightNodeIds: [...right] };
    });
    const overlays = (list, side) => {
      if (list === undefined || list === null) return [];
      if (!Array.isArray(list)) {
        throw new RenderFailure('invalid', `comparison "${id}": ${side} overlays are malformed`);
      }
      return list;
    };
    return {
      kind: 'comparison',
      id: comparison.id,
      label: typeof comparison.label === 'string' ? comparison.label : comparison.id,
      left: this.#pane(collections, comparison.left, `comparison "${id}" left pane`),
      right: this.#pane(collections, comparison.right, `comparison "${id}" right pane`),
      mappingEntries,
      /* Translated, never inferred. An empty `entries` means "the ids already
       * pair what a pass did not touch" for one comparison and "no mapping was
       * computed" for another; only the producer knows which, and
       * `matchNodeIdFallback` is where it says so. Absent reads as `false`,
       * which is the conservative half: equal ids claim nothing. */
      disableMappingFallback: sync.matchNodeIdFallback !== true,
      showDiffHighlights: sync.showDiffHighlights === true,
      overlaysLeft: overlays(comparison.overlaysLeft, 'left'),
      overlaysRight: overlays(comparison.overlaysRight, 'right'),
    };
  }

  #parse(renderText, selection) {
    let state;
    try {
      state = JSON.parse(renderText);
    } catch (error) {
      throw new RenderFailure('invalid', `session is not JSON: ${text(error)}`);
    }
    const collections = state.graphCollections ?? [];
    const nodeCounts = Object.fromEntries(
      collections.flatMap((c) => c.graphs ?? []).map((g) => [g.id, g.nodes?.length ?? 0]),
    );
    const shared = {
      graphCollections: state.graphCollections,
      nodeDataSets: state.nodeDataSets || [],
      modelName: state.model?.name ?? '(unknown)',
      nodeCounts,
    };
    // Exactly one branch. `{ view, comparison }` together is not a request this
    // renderer can honour, and picking one would open a presentation the caller
    // did not unambiguously ask for.
    const named = ['view', 'comparison', 'flow'].filter((k) => selection?.[k] != null);
    if (named.length > 1) {
      throw new RenderFailure('invalid', `selection names ${named.join(' and ')}`);
    }
    if (selection?.comparison != null) {
      return { ...shared, ...this.#selectComparison(state, selection.comparison) };
    }
    if (selection?.flow != null) {
      return { ...shared, ...this.#selectFlow(state, selection.flow) };
    }
    const { graphId, viewId } = this.#select(state, selection);
    if (!graphId) throw new RenderFailure('invalid', 'session has no renderable graph');
    // The collection that actually HOLDS the target, not collection zero:
    // `selectNode` is addressed by collection label, and a view may name a
    // graph in any of them.
    const holder = collections.find((c) => (c.graphs ?? []).some((g) => g.id === graphId));
    return {
      ...shared,
      kind: 'single',
      graphId,
      viewId,
      collectionLabel: holder?.label,
      targetFirstNodeId: (holder?.graphs ?? []).find((g) => g.id === graphId)?.nodes?.[0]?.id,
    };
  }

  #createEntry(descriptor) {
    const entry = {
      descriptor,
      state: 'waiting',
      cancelled: false,
      safe: false,
      attached: false,
      selected: false,
      cleaningUp: false,
      detachAttempts: 0,
      /* Which pane a comparison is still waiting for; always `left` for a
       * single view, which has only the one. */
      phase: 'left',
      /* [D2] The setup selection this renderer asked for and has not yet seen
       * come back. Identity, never a count: one `selectNode` emits TWO events
       * -- a clear then the target -- so a budget would be spent on the clear
       * and the target would escape. Measured in `web/test/flow.spec.ts`. */
      expected: null,
      observer: null,
      router: null,
      seen: new Set(),
      startedAt: 0,
      handle: null,
      slot: null,
      element: null,
      listener: null,
      timer: null,
      heartbeat: null,
      capacityWait: null,
      capacityTimer: null,
      settle: null,
      promise: null,
    };
    entry.promise = new Promise((resolve, reject) => {
      entry.settle = { resolve, reject, done: false };
    });
    return entry;
  }

  async #start(entry) {
    try {
      const wait = this.#acquireCapacity(entry);
      if (wait) await wait;
      // Authority is rechecked after the await: a cancel or a supersession while
      // this entry was parked has already settled and parked it, and connecting
      // now would add an element nobody is waiting for.
      if (this.#inconsistent) throw new RenderFailure('inconsistent', this.#inconsistent);
      if (entry !== this.#active || entry.cancelled) return;
      // Capacity is NOT rechecked, and that is sound rather than an oversight:
      // the only thing that can refill the set between the wake and this
      // continuation is the rest of a debt sweep, which runs inside `install`
      // and is always followed by `#supersede` -- so this entry would be
      // cancelled and returned above rather than reaching `#connect`.
      this.#connect(entry);
    } catch (error) {
      this.#startFailed(entry, error);
    }
  }

  /* Resolves at once while there is room for this entry to *become* quarantined;
   * otherwise parks a waiter that a release wakes, or `capacityTimeout` fails. */
  #acquireCapacity(entry) {
    if (this.#quarantined.size < this.#maxQuarantined) return null;
    const waiter = deferred();
    entry.capacityWait = waiter;
    entry.capacityTimer = this.#timers.setTimeout(
      () => this.#guard(entry, 'capacity timeout', () => this.#onCapacityTimeout(entry)),
      this.capacityTimeout,
    );
    return waiter.promise;
  }

  #onCapacityTimeout(entry) {
    this.#settleCapacityWait(entry, new RenderFailure(
      'exhausted',
      `${this.#quarantined.size} unfinished views are still held; reload the page to continue`,
    ));
  }

  #settleCapacityWait(entry, failure) {
    if (entry.capacityTimer !== null) {
      this.#timers.clearTimeout(entry.capacityTimer);
      entry.capacityTimer = null;
    }
    const waiter = entry.capacityWait;
    if (!waiter) return;
    entry.capacityWait = null;
    if (failure) waiter.reject(failure);
    else waiter.resolve();
  }

  #wakeCapacity() {
    const entry = this.#active;
    if (!entry?.capacityWait) return;
    if (this.#quarantined.size >= this.#maxQuarantined) return;
    this.#settleCapacityWait(entry, null);
  }

  /* The whole of what this shell tells the visualizer, built fresh per
   * candidate -- assigning `config` on a live element does not take, which is
   * why every presentation change is a new element rather than a re-point.
   *
   * `defaultGraphId` is not a real VisualizerConfig field (it appears nowhere
   * in the library's types or compiled bundle) -- kept in case a future version
   * adds it, but nothing depends on it. The library auto-selects some graph on
   * its own; `selectNode` is what navigates to the one we want. */
  #config(d) {
    if (d.kind !== 'comparison') return { defaultGraphId: d.graphId };
    return {
      defaultGraphId: d.left.graph,
      syncNavigationData: {
        type: 'sync_navigation',
        mappingEntries: d.mappingEntries,
        disableMappingFallback: d.disableMappingFallback,
        ...(d.showDiffHighlights ? { showDiffHighlights: true } : {}),
      },
      /* `Comparison.overlaysLeft` is a bare `EdgeOverlay[]`, but the config
       * field takes `EdgeOverlaysData[]` -- `{type, name, graphName, overlays}`
       * -- so the list is wrapped rather than passed through. The key is
       * omitted entirely for an empty list, which is every session produced
       * today: the one real overlay rides on the kernel graph's own
       * `tasksData`, not on a comparison. Each pane gets only its own. */
      ...(d.overlaysLeft.length
        ? { edgeOverlaysDataListLeftPane: [this.#overlays(d, d.overlaysLeft, d.left.graph)] }
        : {}),
      ...(d.overlaysRight.length
        ? { edgeOverlaysDataListRightPane: [this.#overlays(d, d.overlaysRight, d.right.graph)] }
        : {}),
    };
  }

  #overlays(d, list, graphName) {
    return { type: 'edge_overlays', name: d.label, graphName, overlays: list };
  }

  #connect(entry) {
    const d = entry.descriptor;
    const slot = this.#doc.createElement('div');
    slot.classList.add(SLOT_CLASS);
    slot.setAttribute('aria-hidden', 'true');
    const element = this.#doc.createElement('model-explorer-visualizer');
    element.graphCollections = d.graphCollections;
    element.config = this.#config(d);
    entry.slot = slot;
    entry.element = element;
    entry.listener = (event) =>
      this.#guard(entry, 'modelGraphProcessed', () => this.#onProcessed(entry, event));
    element.addEventListener('modelGraphProcessed', entry.listener);
    slot.appendChild(element);
    this.#mount.appendChild(slot);
    entry.attached = true;
    entry.state = 'connected';
    // The clock starts here, once the candidate is live in the DOM and has a
    // real chance to process.
    entry.startedAt = this.#now();
    entry.timer = this.#timers.setTimeout(
      () => this.#guard(entry, 'timeout', () => this.#onTimeout(entry)),
      this.timeout,
    );
    entry.heartbeat = this.#timers.setInterval(
      () => this.#guard(entry, 'heartbeat', () => this.#onHeartbeat(entry)),
      HEARTBEAT_MS,
    );
    console.debug(`[renderer] model=${d.modelName} connected hidden, waiting for ${describe(d)}`);
  }

  /* A comparison's second pane is a second processing round, asked for only
   * once the first has finished, so it gets its own budget rather than serving
   * the remainder of the first one's. `startedAt` is deliberately NOT reset:
   * the heartbeat keeps reporting total elapsed, which is what a reader wants
   * when a candidate is slow. */
  #restartTimeout(entry) {
    if (entry.timer !== null) this.#timers.clearTimeout(entry.timer);
    entry.timer = this.#timers.setTimeout(
      () => this.#guard(entry, 'timeout', () => this.#onTimeout(entry)),
      this.timeout,
    );
  }

  #startFailed(entry, error) {
    this.#reject(entry, error);
    if (this.#inconsistent) return;                 // rule 0: no DOM mutation after terminal
    this.#park(entry, 'a failed install');
  }

  /* The one place that decides where a settled entry goes: an element that is
   * connected without its expected event cannot be removed (NG0953), so it is
   * quarantined; anything else is released. */
  #park(entry, what) {
    try {
      if (entry.attached && !entry.safe) this.#quarantine(entry);
      else this.#release(entry);
    } catch (nested) {
      this.markInconsistent(`the renderer could not clean up ${what}: ${text(nested)}`);
    }
  }

  /* --------------------------------------------------------- guarded entries */

  /* Every entry point that is not a direct `await` -- the event listener, the
   * processing timeout, the heartbeat -- comes through here, and nothing
   * rethrows outward. */
  #guard(entry, name, fn) {
    if (this.#inconsistent) return;
    try {
      fn();
    } catch (error) {
      this.#guardFailed(entry, name, error);
    }
  }

  #guardFailed(entry, name, error) {
    console.warn(`[renderer] ${name} failed`, error);
    // A throw from inside cleanup never re-enters cleanup: `#detach` inspects
    // its own postcondition and decides there.
    if (entry.cleaningUp) return;
    // Already settled and no longer authoritative: swallow it, keep the
    // quarantine, and say nothing to the user about a request they abandoned.
    if (entry.settle.done && entry !== this.#active) return;
    this.#reject(entry, error);
    this.#park(entry, `a ${name} failure`);
  }

  #onProcessed(entry, event) {
    // Retired: a `ready`, `current` or `released` entry ignores late events even
    // if `removeEventListener` did not take.
    if (entry.state !== 'connected' && entry.state !== 'quarantined') return;
    const d = entry.descriptor;
    const detail = event?.detail;
    const seen = detail?.modelGraph?.id;
    /* The pane is part of the identity of an event, not decoration.
     * `Session.validate` permits a comparison whose two panes name the SAME
     * graph, and opening the second pane re-lays-out the first -- so a wait
     * keyed on the graph id alone can be satisfied by the wrong pane, and the
     * candidate would be finalized with a pane that never processed.
     * `web/test/compare.spec.ts` covers exactly that. */
    const pane = typeof detail?.paneIndex === 'number' ? detail.paneIndex : 0;
    entry.seen.add(`${pane}:${seen}`);

    const target = this.#expected(entry);
    if (seen !== target.graph || pane !== target.pane) {
      // The library settled on some other graph as its own default; navigate to
      // the one we asked for, the way a manual click does. Only ever for the
      // first pane: the second is reached by an explicit `selectNode` below.
      if (entry.selected || target.pane !== 0 || !target.firstNodeId) return;
      entry.selected = true;
      // A single view passes three arguments, exactly as it always has: the
      // pane index defaults to 0 in the element, and a shell that has never
      // had a second pane should not start naming the first one.
      if (d.kind === 'comparison') {
        entry.element.selectNode(target.firstNodeId, target.graph, target.collectionLabel, 0);
      } else {
        /* [D2] A flow candidate arms its expectation and its OBSERVER before
         * the call, not after: the emission can land in either order relative
         * to finalize, and only an observation -- never an assumption about
         * timing -- can tell which happened. */
        if (d.kind === 'flow') this.#observe(entry, target);
        entry.element.selectNode(target.firstNodeId, target.graph, target.collectionLabel);
      }
      return;
    }

    /* The expected event. `safe` means "nothing this renderer asked for is
     * still outstanding", which at this instant is true -- it is the
     * precondition `#release` asserts and `#park` branches on. For a comparison
     * it becomes false again the moment the second pane is requested. */
    entry.safe = true;

    if (d.kind === 'comparison' && entry.phase === 'left') {
      // Authority is rechecked BEFORE more work is requested: a cancelled entry
      // is idle right now and can simply be released, whereas asking for a
      // second pane would make it unsafe again and cost a quarantine slot for
      // an event nobody is waiting for.
      if (entry !== this.#active || entry.cancelled || entry.state === 'quarantined') {
        this.#clearTimers(entry);
        this.#release(entry);
        return;
      }
      entry.phase = 'right';
      entry.safe = false;
      this.#restartTimeout(entry);
      const right = d.right;
      entry.element.selectNode(right.firstNodeId, right.graph, right.collectionLabel, 1);
      return;
    }

    this.#clearTimers(entry);
    if (entry !== this.#active || entry.cancelled || entry.state === 'quarantined') {
      this.#release(entry);
      return;
    }
    this.#installNodeData(entry);
    // Retired before the handle is exposed: a later graph event must not
    // reinstall node data, nor throw and let the guard release an entry the
    // coordinator is about to commit.
    this.#retireListener(entry);
    entry.state = 'ready';
    /* The presentation rides the handle so the caller learns what was actually
     * shown without re-running the resolution above -- two selection algorithms
     * would be free to drift apart. */
    const handle = Object.freeze(d.kind === 'comparison'
      ? {
        graph: { left: d.left.graph, right: d.right.graph },
        viewId: null,
        presentation: { kind: 'comparison', comparison: d.id },
      }
      : d.kind === 'flow'
        ? {
          graph: d.graphId,
          // A flow is not a single view, so nothing that reads `viewId` may
          // treat it as one.
          viewId: null,
          presentation: { kind: 'flow', view: d.viewId },
        }
        : {
          graph: d.graphId,
          viewId: d.viewId,
          presentation: d.viewId ? { kind: 'single', view: d.viewId } : null,
        });
    this.#handles.set(handle, entry);
    entry.handle = handle;
    this.#resolve(entry, handle);
  }

  /* What this entry is waiting for right now: one graph, on one pane. */
  #expected(entry) {
    const d = entry.descriptor;
    if (d.kind !== 'comparison') {
      return {
        graph: d.graphId,
        pane: 0,
        firstNodeId: d.targetFirstNodeId,
        collectionLabel: d.collectionLabel,
      };
    }
    const side = entry.phase === 'right' ? d.right : d.left;
    return {
      graph: side.graph,
      pane: entry.phase === 'right' ? 1 : 0,
      firstNodeId: side.firstNodeId,
      collectionLabel: side.collectionLabel,
    };
  }

  /* Graph-addressed, exactly as for a single view -- a set is installed in the
   * pane whose graph OWNS it, and a set addressed to neither pane is not
   * installed at all. */
  #installNodeData(entry) {
    const d = entry.descriptor;
    const panes = d.kind === 'comparison'
      ? [{ graph: d.left.graph, index: 0 }, { graph: d.right.graph, index: 1 }]
      : [{ graph: d.graphId, index: 0 }];
    for (const set of d.nodeDataSets) {
      for (const pane of panes) {
        if (set.graph !== pane.graph) continue;
        const data = set.name === VERIFICATION_SET ? verificationData(set) : gradientData(set);
        // The pane index is omitted for a single view, so the call the shell has
        // always made stays byte-for-byte what it was.
        if (d.kind === 'comparison') {
          entry.element.addNodeDataProviderData(set.name, data, pane.index);
        } else {
          entry.element.addNodeDataProviderData(set.name, data);
        }
      }
    }
  }

  #onTimeout(entry) {
    if (entry.state !== 'connected') return;
    const d = entry.descriptor;
    const target = this.#expected(entry);
    console.warn(
      `[renderer] model=${d.modelName} timed out waiting for ${describe(d)}`
      + ` at pane ${target.pane} (${target.graph});`
      + ` received=${JSON.stringify([...entry.seen])}`,
    );
    if (entry === this.#active) this.#active = null;
    this.#reject(entry, new RenderFailure('timeout', 'renderer timed out'));
    // Nothing is removed and `current` stays visible: the element is presumed
    // still processing, and its later expected event is what authorises removal.
    this.#quarantine(entry);
  }

  #onHeartbeat(entry) {
    const d = entry.descriptor;
    const elapsed = Math.round(this.#now() - entry.startedAt);
    console.debug(
      `[renderer] model=${d.modelName} still waiting at ${elapsed}ms; received so far=${JSON.stringify([...entry.seen])}`,
    );
  }

  /* ------------------------------------------------------- flow selection

     [D2]. `NodeInfo` carries no origin bit, and one `selectNode` emits TWO
     events -- a clear (`nodeId: ""`) then the target -- so a user's click can
     only be told from the shell's own setup selection by IDENTITY and ORDER.
     `web/test/flow.spec.ts` measures both against the real element.

     Two listeners with disjoint jobs. The observer exists from before the setup
     call until finalize and routes nothing; the router exists only while the
     entry is `current`. Whichever side of finalize the setup emission lands on,
     exactly one of them sees it. */

  #observe(entry, target) {
    entry.expected = { nodeId: target.firstNodeId, graphId: target.graph };
    if (entry.observer) return;
    entry.observer = (event) =>
      this.#guard(entry, 'selectedNodeChanged (setup)', () => {
        const seen = event?.detail;
        // The clear is not the target and never satisfies the expectation --
        // consuming it here is precisely the defect that sank the counted
        // design.
        if (typeof seen?.nodeId !== 'string' || seen.nodeId === '') return;
        if (entry.expected
            && entry.expected.nodeId === seen.nodeId
            && entry.expected.graphId === seen.graphId) {
          entry.expected = null;
        }
      });
    entry.element.addEventListener('selectedNodeChanged', entry.observer);
  }

  #retireObserver(entry) {
    if (!entry.observer) return;
    const observer = entry.observer;
    entry.observer = null;
    entry.element.removeEventListener('selectedNodeChanged', observer);
  }

  /* Attached by `finalize` to the entry it promotes, because a hidden candidate
   * must never navigate: its slot is `pointer-events: none`, so every event it
   * could see is one this renderer caused. */
  #attachRouter(entry) {
    if (entry.descriptor.kind !== 'flow' || entry.router) return;
    entry.router = (event) =>
      this.#guard(entry, 'selectedNodeChanged', () => this.#onSelected(entry, event));
    entry.element.addEventListener('selectedNodeChanged', entry.router);
  }

  #retireRouter(entry) {
    if (!entry.router) return;
    const router = entry.router;
    entry.router = null;
    entry.element.removeEventListener('selectedNodeChanged', router);
  }

  #onSelected(entry, event) {
    // Authority: a quarantined, superseded or replaced element still emits --
    // `web/test/flow.spec.ts` proves it does -- so this is the guard, not the
    // element going quiet.
    if (entry !== this.current) return;
    const d = entry.descriptor;
    const seen = event?.detail;
    // An absent detail is a real possibility, and is not a selection.
    if (typeof seen?.nodeId !== 'string') return;
    // Scoped on the event's own public fields; nothing private is consulted.
    if (seen.graphId !== d.graphId || seen.collectionLabel !== d.collectionLabel) return;
    if (seen.nodeId === '') {
      // A cleared selection is not a node. It does NOT clear the expectation:
      // setup emits the clear BEFORE its target, so consuming it here would let
      // the target through as though a user had chosen it.
      this.#onFlowSelect(null);
      return;
    }
    if (entry.expected
        && entry.expected.nodeId === seen.nodeId
        && entry.expected.graphId === seen.graphId) {
      entry.expected = null;
      return;
    }
    // Once anything has routed, setup is definitively over.
    entry.expected = null;
    this.#onFlowSelect({
      nodeId: seen.nodeId,
      graphId: seen.graphId,
      collectionLabel: seen.collectionLabel,
    });
  }

  /* ------------------------------------------------------------ settlement */

  #resolve(entry, value) {
    if (entry.settle.done) return;
    entry.settle.done = true;
    entry.settle.resolve(value);
  }

  #reject(entry, error) {
    if (entry.settle.done) return;
    entry.settle.done = true;
    entry.settle.reject(error);
  }

  /* -------------------------------------------------------------- lifecycle */

  cancel() {
    this.#abandonActive(new RenderFailure('cancelled', 'render cancelled'));
  }

  #supersede() {
    this.#abandonActive(new RenderFailure('cancelled', 'superseded by a newer request'));
    if (this.#active !== null) throw new Error('renderer defect: two active entries');
  }

  #abandonActive(failure) {
    const entry = this.#active;
    if (!entry) return;
    this.#active = null;
    entry.cancelled = true;
    this.#reject(entry, failure);
    // A parked entry is woken rather than rejected: `#start` rechecks authority
    // after the await and returns, so it never connects an element for a caller
    // that is already settled.
    this.#settleCapacityWait(entry, null);
    if (this.#inconsistent) return;
    // A ready entry's handle may already be in the coordinator's hands with its
    // continuation queued; the tombstone is what lets `abortReady` tell that
    // good-faith call apart from misuse.
    if (entry.safe) this.#consume(entry, 'cancel');
    this.#park(entry, 'a cancelled install');
  }

  #quarantine(entry) {
    if (entry === this.#active) this.#active = null;
    if (entry.state === 'quarantined') return;
    entry.state = 'quarantined';
    this.#quarantined.add(entry);
    // The graph listener stays -- its late event is what authorises removal --
    // but a quarantined element must never navigate, so selection authority is
    // dropped here rather than resting on the `current` check alone.
    this.#retireObserver(entry);
    this.#retireRouter(entry);
    // The listener stays: that late expected event is the only thing that
    // authorises removal.
    this.#clearTimers(entry);
  }

  #release(entry) {
    if (entry === this.#active) this.#active = null;
    if (entry.state === 'released') return;
    if (entry.attached && !entry.safe) {
      throw new Error('renderer defect: release of an unsafe connected element');
    }
    this.#detach(entry);
  }

  /* The single cleanup-debt retry point. Debt is safe but has no future
   * `modelGraphProcessed` event to ride on, so retries are attached to work the
   * renderer is doing anyway: exactly one attempt per debt entry per install,
   * which is what makes the attempt budget mean anything. No timer, no
   * microtask loop -- debt is retried when it is in the way. */
  #sweepDebt() {
    for (const entry of [...this.#quarantined]) {
      if (entry.state !== 'cleanup_failed') continue;
      this.#detach(entry);
      if (this.#inconsistent) return;
    }
  }

  /* One `slot.remove()` per attempt, never recursive: `cleaningUp` routes a
   * throw from inside here to the postcondition inspection below rather than
   * back into cleanup. */
  #detach(entry) {
    if (entry.state === 'released' || entry.state === 'cleanup_abandoned') return;
    if (!entry.slot) {
      this.#finishRelease(entry);
      return;
    }
    entry.detachAttempts += 1;
    entry.cleaningUp = true;
    try {
      entry.slot.remove();
    } catch (error) {
      console.warn('[renderer] slot removal failed', error);
    } finally {
      entry.cleaningUp = false;
    }
    let connected;
    try {
      connected = entry.slot.isConnected;
    } catch (error) {
      this.markInconsistent(`the renderer could not determine whether a view was removed: ${text(error)}`);
      return;
    }
    if (!connected) {
      this.#finishRelease(entry);
      return;
    }
    // Safe, still connected: debt that holds one quarantine slot, which is
    // exactly what the budget counts. Once the attempt budget is spent the
    // renderer stops promising a retry -- the DOM is still coherent, so this is
    // not inconsistency, but it permanently lowers capacity.
    entry.state = entry.detachAttempts >= MAX_DETACH_ATTEMPTS ? 'cleanup_abandoned' : 'cleanup_failed';
    this.#quarantined.add(entry);
  }

  #finishRelease(entry) {
    this.#clearTimers(entry);
    this.#retireListener(entry);
    this.#retireObserver(entry);
    this.#retireRouter(entry);
    const held = this.#quarantined.delete(entry);
    entry.attached = false;
    entry.state = 'released';
    if (entry === this.current) this.current = null;
    // Only once the entry has actually left the set.
    if (held) this.#wakeCapacity();
  }

  #clearTimers(entry) {
    if (entry.timer !== null) { this.#timers.clearTimeout(entry.timer); entry.timer = null; }
    if (entry.heartbeat !== null) { this.#timers.clearInterval(entry.heartbeat); entry.heartbeat = null; }
  }

  #retireListener(entry) {
    if (!entry.listener) return;
    const listener = entry.listener;
    entry.listener = null;
    entry.element.removeEventListener('modelGraphProcessed', listener);
  }

  /* ----------------------------------------------------------- handles */

  #resolveHandle(handle) {
    if (handle === null || (typeof handle !== 'object' && typeof handle !== 'function')) {
      return { kind: 'misuse' };
    }
    const found = this.#handles.get(handle);
    if (found === undefined) return { kind: 'misuse' };
    if (found.consumedBy) return { kind: 'consumed', by: found.consumedBy };
    return { kind: 'entry', entry: found };
  }

  /** Consumes a handle exactly once, into `current`, `released` or a tombstone. */
  #consume(entry, tombstone) {
    const handle = entry.handle;
    if (!handle) return;
    entry.handle = null;
    if (tombstone) this.#handles.set(handle, { consumedBy: tombstone });
    else this.#handles.delete(handle);
  }

  #ready(handle) {
    if (this.#inconsistent) {
      return { failure: { ok: false, state: 'inconsistent', error: this.#inconsistent } };
    }
    const found = this.#resolveHandle(handle);
    if (found.kind === 'consumed') return { consumed: found.by };
    if (found.kind !== 'entry') {
      return { failure: { ok: false, state: 'misuse', error: 'unknown render handle' } };
    }
    const entry = found.entry;
    if (entry.state !== 'ready' || entry !== this.#active) {
      return { failure: { ok: false, state: 'misuse', error: `render handle is ${entry.state}, not ready` } };
    }
    return { entry };
  }

  /* --------------------------------------------------- terminal operations */

  /** Reveals a ready candidate and retires the previous one. Never throws. */
  finalize(handle) {
    const resolved = this.#ready(handle);
    if (resolved.consumed) {
      // Reachable only after a successful commit, so the bridge holds a session
      // nothing is showing. The coordinator treats it as fatal.
      if (resolved.consumed !== 'cancel') {
        return { ok: false, state: 'inconsistent', error: this.#inconsistent ?? 'the view is in an unknown state' };
      }
      return { ok: false, state: 'cancelled', error: 'the prepared view was discarded before it was shown' };
    }
    if (resolved.failure) return resolved.failure;
    const entry = resolved.entry;
    const previous = this.current;

    // 1. Promote the newcomer first, so no instant has nothing visible.
    try {
      entry.slot.classList.add(CURRENT_CLASS);
      entry.slot.removeAttribute('aria-hidden');
    } catch (error) {
      let undone = false;
      try {
        entry.slot.classList.remove(CURRENT_CLASS);
        entry.slot.setAttribute('aria-hidden', 'true');
        undone = !entry.slot.classList.contains(CURRENT_CLASS);
      } catch { undone = false; }
      if (!undone) {
        this.markInconsistent(`the new view could not be revealed or hidden again: ${text(error)}`);
        return { ok: false, state: 'inconsistent', error: text(error) };
      }
      this.#consume(entry, null);
      this.#release(entry);
      if (this.#inconsistent) return { ok: false, state: 'inconsistent', error: this.#inconsistent };
      return { ok: false, state: 'known_old', error: text(error) };
    }

    // 2. Commit the state. Pure JavaScript; it cannot throw.
    this.#consume(entry, null);
    entry.state = 'current';
    this.current = entry;
    this.#active = null;
    // [D2] Only now: a hidden candidate must not navigate, and the observer's
    // window closes exactly where the router's opens, so the setup emission is
    // seen by one of them and never by both.
    this.#retireObserver(entry);
    this.#attachRouter(entry);

    // 3. Demote the predecessor. Two visible slots is not a state to paper over.
    if (previous) {
      let demoted = false;
      let failure = null;
      for (let attempt = 0; attempt < 2 && !demoted; attempt += 1) {
        try {
          previous.slot.classList.remove(CURRENT_CLASS);
          previous.slot.setAttribute('aria-hidden', 'true');
          demoted = !previous.slot.classList.contains(CURRENT_CLASS);
        } catch (error) {
          failure = error;
          demoted = false;
        }
      }
      if (!demoted) {
        const why = failure ? text(failure) : 'the previous view is still presentation-current';
        this.markInconsistent(`the previous view could not be hidden: ${why}`);
        return { ok: false, state: 'inconsistent', error: why };
      }
      // 4. Remove it. The newcomer is already authoritative, so a failure here
      //    is cleanup debt, not a failed finalization.
      this.#release(previous);
      if (this.#inconsistent) {
        return { ok: false, state: 'inconsistent', error: this.#inconsistent };
      }
    }
    return { ok: true };
  }

  /** Discards a ready candidate the coordinator decided not to show. */
  abortReady(handle) {
    const resolved = this.#ready(handle);
    if (resolved.consumed) {
      if (resolved.consumed === 'cancel') return { ok: true, already: 'cancelled' };
      return { ok: false, state: 'inconsistent', error: this.#inconsistent ?? 'the view is in an unknown state' };
    }
    if (resolved.failure) return resolved.failure;
    const entry = resolved.entry;
    this.#consume(entry, null);
    this.#release(entry);
    if (this.#inconsistent) return { ok: false, state: 'inconsistent', error: this.#inconsistent };
    if (isDebt(entry.state)) {
      return { ok: false, state: 'cleanup_failed', error: 'the abandoned view could not be removed' };
    }
    return { ok: true };
  }

  /* The renderer's own terminal operation: the flag lands before anything
   * fallible, so every pending callback is inert from that instant. */
  markInconsistent(reason) {
    const why = String(reason);
    if (this.#inconsistent) return { ok: true };
    this.#inconsistent = why;
    console.error(`[renderer] inconsistent: ${why}`);
    const errors = [];
    const step = (fn) => { try { fn(); } catch (error) { errors.push(text(error)); } };

    // Settle the active caller in whatever phase it is in. Inconsistency can be
    // discovered long before any handle exists, and the guard's rule 0 would
    // otherwise leave that `install()` promise pending forever.
    const victim = this.#active;
    if (victim) {
      step(() => this.#reject(victim, new RenderFailure('inconsistent', why)));
      step(() => this.#consume(victim, 'inconsistent'));
      this.#active = null;
    }
    const all = new Set(this.#quarantined);
    if (victim) all.add(victim);
    if (this.current) all.add(this.current);
    for (const entry of all) step(() => this.#clearTimers(entry));
    // Nothing may navigate once the renderer has declared itself terminal.
    for (const entry of all) {
      step(() => this.#retireObserver(entry));
      step(() => this.#retireRouter(entry));
    }
    // A parked caller is settled too; `#startFailed` sees the flag and mutates
    // no DOM.
    if (victim) step(() => this.#settleCapacityWait(victim, new RenderFailure('inconsistent', why)));
    // Every connected element stays in place, the settled one included: bridge
    // state is unknown, so removing anything would assert a rollback nobody can
    // confirm.
    return errors.length === 0 ? { ok: true } : { ok: false, error: errors.join('; ') };
  }
}
