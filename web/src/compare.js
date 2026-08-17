/* The Phase 3 comparison API gate.
 *
 * `web-ui-3.md` makes a two-pane comparison a hard prerequisite: the renderer
 * adapter may use only visualizer behaviour proven here, in Chromium, against
 * the bundle we actually ship. Its siblings settle different questions -- the
 * 0A spike asks what the renderer can do, the Stage 1 gate asks whether what we
 * export is something it accepts, the Path B gate asks whether the cancellation
 * lifecycle is real. This one asks whether the COMPARISON surface is real:
 * two panes, a supplied correspondence, per-pane overlays and node data.
 *
 * Hand-authored and deliberately tiny, so every observation is about the
 * visualizer rather than about resnet18. The two graphs speak different id
 * vocabularies on purpose -- that is the `c/import` case, where matching equal
 * ids would pair nothing -- with exactly one id, `shared`, deliberately present
 * in BOTH. That node is what makes `disableMappingFallback` observable at all:
 * it is the only node an id-identity fallback could pair, and it appears in no
 * mapping entry, so pairing it can only have come from the fallback.
 *
 * What is OBSERVABLE, and what is not, decided the shape of this file.
 * `modelGraphProcessed` carries `{modelGraph, paneIndex}` and `uiStateChanged`
 * carries `paneStates[i].selectedNodeId`, so pane creation and cross-pane
 * navigation are public API. Diff highlights and overlay edges are drawn in
 * WebGL, and the element renders into a shadow root, so "it is highlighted" has
 * no public observable -- those are recorded as accepted-without-error and the
 * spec says exactly that rather than claiming more.
 */

const MOUNT = document.getElementById('mount');
const result = { steps: {}, errors: [] };
window.__compare = result;

const LABEL = 'mltorch:compare';
const LEFT = 'g/left';
const RIGHT = 'g/right';

const note = (key, value) => { result.steps[key] = value; };

/* A node with no incoming edges is a root; `outputsMetadata` gives it a slot so
 * an edge into it is expressible. This is the minimum the element accepts. */
function node(id, inputs = []) {
  return {
    id,
    label: id,
    namespace: '',
    subgraphIds: [],
    incomingEdges: inputs.map((source, index) => ({
      sourceNodeId: source,
      sourceNodeOutputId: '0',
      targetNodeInputId: String(index),
    })),
    outputsMetadata: [{ id: '0', attrs: [] }],
  };
}

/* Left ids are `L*`, right ids are `r-*`: no accidental overlap, so `shared` is
 * the ONLY id an id-identity fallback can pair. */
const collections = [{
  label: LABEL,
  graphs: [
    { id: LEFT, nodes: [node('L1'), node('L2', ['L1']), node('L3', ['L2']), node('L4', ['L3']), node('shared', ['L4'])] },
    { id: RIGHT, nodes: [node('r-alpha'), node('r-beta', ['r-alpha']), node('r-gamma', ['r-beta']), node('r-delta', ['r-gamma']), node('shared', ['r-delta'])] },
  ],
}];

/* 1:N and N:M, the two shapes `Me_session.Mapping_entry` admits that a naive
 * pair-wise mapping cannot express. `L4` is deliberately unmapped, and `shared`
 * deliberately absent from every entry. */
const MAPPING_ENTRIES = [
  { leftNodeIds: ['L1'], rightNodeIds: ['r-alpha', 'r-beta'] },
  { leftNodeIds: ['L2', 'L3'], rightNodeIds: ['r-gamma', 'r-delta'] },
];

const overlays = (name, edges) => ([{
  type: 'edge_overlays',
  name,
  overlays: [{ name: `${name} overlay`, edges, edgeColor: '#ff00be', edgeWidth: 3 }],
}]);

const LEFT_OVERLAYS = overlays('left-only', [{ sourceNodeId: 'L1', targetNodeId: 'L4', label: 'skip' }]);
const RIGHT_OVERLAYS = overlays('right-only', [{ sourceNodeId: 'r-alpha', targetNodeId: 'r-delta', label: 'skip' }]);

const firstNode = (graphId) =>
  collections[0].graphs.find((g) => g.id === graphId).nodes[0].id;

/* The application's own wait, in miniature, and filtered on BOTH keys.
 *
 * `paneIndex` is not decoration: `openGraphInSplitPane` re-lays-out pane 0
 * (`app_service.ts`), so the left graph emits again while the right pane is
 * being built, and a listener filtered on the graph id alone would resolve the
 * right-pane wait on a left-pane event. Equally, if the two panes ever showed
 * the same graph, the id alone could not tell them apart. */
function awaitProcessed(element, graphId, paneIndex, ms) {
  return new Promise((resolve) => {
    let settled = false;
    /* `seenPane` is what the RESOLVING event actually carried, recorded
     * separately from the pane that was asked for. Without it this wait is not
     * checkable: dropping the pane filter still reports `fired: true`, because
     * the wrong event resolves the promise just as well as the right one. With
     * it, a wait satisfied by the other pane's event is visible as a
     * mismatch -- which is what makes the assertion in the spec able to fail. */
    let seenPane = null;
    const finish = (fired) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      element.removeEventListener('modelGraphProcessed', onEvent);
      resolve({ fired, graphId, paneIndex, seenPane });
    };
    const timer = setTimeout(() => finish(false), ms);
    const onEvent = (event) => {
      const seen = event.detail ?? {};
      if (seen.modelGraph?.id === graphId && seen.paneIndex === paneIndex) {
        seenPane = seen.paneIndex;
        finish(true);
      }
    };
    element.addEventListener('modelGraphProcessed', onEvent);
  });
}

/* The most recent `uiStateChanged`, which is how a selection in one pane is
 * read back in the OTHER. `selectedNodeChanged` carries only an opaque
 * `paneId`; `paneStates` is positional, so it answers "what is selected in pane
 * 1" directly. */
function watchUiState(candidate) {
  candidate.uiState = null;
  candidate.element.addEventListener('uiStateChanged', (event) => {
    candidate.uiState = event.detail;
  });
}

const selectedIn = (candidate, paneIndex) =>
  candidate.uiState?.paneStates?.[paneIndex]?.selectedNodeId ?? '';

/* One slot, one element, connected exactly once -- the shape `Renderer.#connect`
 * has. Configured while detached, so the element is connected by the same single
 * operation that makes its slot a child of the mount. */
function connectHidden(config) {
  const slot = document.createElement('div');
  slot.className = 'slot';
  slot.setAttribute('aria-hidden', 'true');
  const element = document.createElement('model-explorer-visualizer');
  element.graphCollections = collections;
  element.config = config;
  slot.appendChild(element);
  MOUNT.appendChild(slot);
  const candidate = { slot, element };
  watchUiState(candidate);
  return candidate;
}

function reveal(slot) {
  slot.classList.add('slot--current');
  slot.removeAttribute('aria-hidden');
}

const syncConfig = (extra = {}) => ({
  syncNavigationData: {
    type: 'sync_navigation',
    mappingEntries: MAPPING_ENTRIES,
    ...extra,
  },
});

/* The two-phase open the adapter will perform: left graph into pane 0, then the
 * right graph into pane 1 through the public `selectNode`. Returns once BOTH
 * pane events have arrived -- one event cannot prove two panes. */
async function openComparison(config, { leftOverlays, rightOverlays } = {}) {
  const full = {
    ...config,
    ...(leftOverlays ? { edgeOverlaysDataListLeftPane: leftOverlays } : {}),
    ...(rightOverlays ? { edgeOverlaysDataListRightPane: rightOverlays } : {}),
  };
  const candidate = connectHidden(full);
  const leftReady = awaitProcessed(candidate.element, LEFT, 0, 60000);
  /* The library auto-selects a graph of its own -- `defaultGraphId` is inert
   * (`renderer.js`) -- so pane 0 is navigated the way a click does. */
  candidate.element.selectNode(firstNode(LEFT), LEFT, LABEL, 0);
  const left = await leftReady;

  const rightReady = awaitProcessed(candidate.element, RIGHT, 1, 60000);
  candidate.element.selectNode(firstNode(RIGHT), RIGHT, LABEL, 1);
  const right = await rightReady;
  return { candidate, left, right };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitFor(read, want, ms = 5000) {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline && read() !== want) await sleep(50);
  return read() === want;
}

/* Resolves as soon as `read()` leaves `from`, and otherwise runs the window out
 * and reports that it did not. "Nothing was paired" is a legitimate outcome, so
 * this cannot wait for a value to appear -- the timeout IS the answer in that
 * case, which is why the window is short. */
async function waitChange(read, from, ms = 3000) {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    const now = read();
    if (now !== from) return now;
    await sleep(50);
  }
  return read();
}

/* One pairing observation, and the reason it is this elaborate: reading the
 * other pane's CURRENT selection cannot distinguish "the correspondence put it
 * there" from "it was already there". Both panes carry a selection from the
 * moment the comparison opened, so a naive read reports a pass for a mapping
 * that never fired.
 *
 * So the other pane is first parked on a value that is deliberately NOT the
 * expected answer, and the observation is the CHANGE away from it. `paired` is
 * then a fact about this selection rather than about the pane's history.
 * Priming syncs back into the source pane, which is harmless: the source
 * selection is made explicitly afterwards and waited for. */
/* Sync navigation is NOT live when the two pane events have arrived.
 *
 * Found here, and it is the kind of thing only the real element can say: the
 * component that reads `config.syncNavigationData` and selects
 * `VISUALIZER_CONFIG` mode is `sync-navigation-button`, and
 * `split_panes_container.ng.html` constructs it under
 * `@if (hasSplitPane && allPanesLoaded())`. So the mapping is installed a beat
 * AFTER `modelGraphProcessed` fires for pane 1, and the first probe run
 * immediately on that event reports "nothing paired" for a pairing that is
 * about to work.
 *
 * That is a fact about the fixture's timing rather than about the adapter --
 * nobody clicks a node in the same tick the pane appears -- but a gate that did
 * not know it would record a false negative and pin it as the contract. Every
 * candidate is therefore armed on a pairing KNOWN to exist before any
 * observation is trusted. Where no such pairing exists by construction (an
 * empty mapping with the fallback off) there is nothing to arm on, and the
 * settle below is the only option; it is generous for that reason. */
async function armSync(candidate, { prime, source, sourcePane, expect }, ms = 15000) {
  const started = Date.now();
  let attempts = 0;
  while (Date.now() - started < ms) {
    attempts += 1;
    const outcome = await probe(candidate, { prime, source, sourcePane });
    if (outcome.after === expect) {
      return { armed: true, attempts, ms: Date.now() - started };
    }
    await sleep(200);
  }
  return { armed: false, attempts, ms: Date.now() - started };
}

async function probe(candidate, { prime, source, sourcePane }) {
  const otherPane = sourcePane === 0 ? 1 : 0;
  const graph = (pane) => (pane === 0 ? LEFT : RIGHT);

  candidate.element.selectNode(prime, graph(otherPane), LABEL, otherPane);
  const primed = await waitFor(() => selectedIn(candidate, otherPane), prime);
  const before = selectedIn(candidate, otherPane);

  candidate.element.selectNode(source, graph(sourcePane), LABEL, sourcePane);
  const took = await waitFor(() => selectedIn(candidate, sourcePane), source);
  const after = await waitChange(() => selectedIn(candidate, otherPane), before);
  return { source, primed, took, before, after, paired: after !== before };
}

async function run() {
  /* 1 + 2. Two panes, each proved by its OWN event. */
  const supplied = await openComparison(syncConfig({ disableMappingFallback: true }),
    { leftOverlays: LEFT_OVERLAYS, rightOverlays: RIGHT_OVERLAYS });
  reveal(supplied.candidate.slot);
  note('twoPanes', {
    left: supplied.left,
    right: supplied.right,
    panes: supplied.candidate.uiState?.paneStates?.length ?? 0,
    leftGraph: supplied.candidate.uiState?.paneStates?.[0]?.selectedGraphId ?? null,
    rightGraph: supplied.candidate.uiState?.paneStates?.[1]?.selectedGraphId ?? null,
    /* This candidate carried per-pane overlays through the same config. Both
     * panes opening is the whole of what can be claimed: `SplitPane` provides
     * its OWN `EdgeOverlaysService` and loads the matching list in `ngOnInit`,
     * so a malformed payload throws there -- but the result is drawn in WebGL
     * inside a shadow root, so which pane got which list has no public
     * observable. Recorded, not asserted beyond acceptance. */
    overlaysSupplied: {
      left: LEFT_OVERLAYS[0].name,
      right: RIGHT_OVERLAYS[0].name,
    },
  });

  /* 3a. The supplied correspondence is followed, including 1:N and N:M, and the
   * fallback is genuinely off: `shared` exists on both sides and is in no entry,
   * so pairing it could only be id-identity. */
  const armed = await armSync(supplied.candidate,
    { prime: 'shared', source: 'L3', sourcePane: 0, expect: 'r-gamma' });
  note('syncArming', armed);

  note('mappingFollowed', {
    // 1:N -- `shared` primes, because it is in no entry and so can never be the
    // answer; with the fallback off it moves nothing either.
    oneToMany: await probe(supplied.candidate, { prime: 'shared', source: 'L1', sourcePane: 0 }),
    // N:M, and from the member that is second in its entry.
    manyToMany: await probe(supplied.candidate, { prime: 'shared', source: 'L3', sourcePane: 0 }),
    // The same relation read right-to-left, which is a separate code path.
    reverse: await probe(supplied.candidate, { prime: 'shared', source: 'r-delta', sourcePane: 1 }),
    // The fallback is genuinely OFF: `shared` exists on both sides and is in no
    // entry, so a pairing here could only be id-identity.
    fallbackOff: await probe(supplied.candidate, { prime: 'r-gamma', source: 'shared', sourcePane: 0 }),
    // An unmapped id with no twin pairs nothing at all.
    unmapped: await probe(supplied.candidate, { prime: 'r-gamma', source: 'L4', sourcePane: 0 }),
  });

  /* 5. Pane-addressed node data, at the point the adapter installs it: after
   * BOTH pane events. Recorded as accepted-or-threw. */
  const nodeData = { accepted: [], errors: [] };
  for (const [paneIndex, ids] of [[0, ['L1', 'L2']], [1, ['r-alpha', 'r-beta']]]) {
    try {
      supplied.candidate.element.addNodeDataProviderData(
        `pane-${paneIndex}`,
        {
          results: Object.fromEntries(ids.map((id, i) => [id, { value: i }])),
          gradient: [{ stop: 0, bgColor: '#e8f5e9' }, { stop: 1, bgColor: '#ffcdd2' }],
        },
        paneIndex,
      );
      nodeData.accepted.push(paneIndex);
    } catch (error) {
      nodeData.errors.push(`pane ${paneIndex}: ${String(error)}`);
    }
  }
  note('paneNodeData', nodeData);

  /* 3b. The SAME document with the fallback left unset. `shared` must now pair,
   * and nothing else about the comparison may change. This is the pair of
   * observations `web-ui-3.md`'s `disableMappingFallback` rule rests on: one
   * direction alone cannot tell an honoured flag from a mapping that never
   * loaded. */
  const fallback = await openComparison(syncConfig());
  note('fallbackOn', {
    left: fallback.left,
    right: fallback.right,
    armed: await armSync(fallback.candidate,
      { prime: 'shared', source: 'L3', sourcePane: 0, expect: 'r-gamma' }),
    // The same selection that paired NOTHING above now pairs its twin.
    shared: await probe(fallback.candidate, { prime: 'r-gamma', source: 'shared', sourcePane: 0 }),
    // ...and the supplied entries still win where they exist.
    stillMapped: await probe(fallback.candidate, { prime: 'shared', source: 'L1', sourcePane: 0 }),
  });

  /* 6. `showDiffHighlights` and the per-pane overlays are ACCEPTED. Both are
   * drawn in WebGL inside a shadow root, so this records that a config carrying
   * them opens both panes without error -- not that anything was drawn. The
   * empty-mapping case is the one `[C2]` exists to prevent: with the fallback
   * off and no entries, every node's mapped set is empty, which upstream's
   * `renderDiffHighlights` treats as "all mapped nodes missing". */
  const diff = await openComparison({
    syncNavigationData: {
      type: 'sync_navigation',
      mappingEntries: MAPPING_ENTRIES,
      disableMappingFallback: true,
      showDiffHighlights: true,
      deletedNodesBorderColor: '#e53935',
      newNodesBorderColor: '#43a047',
    },
  });
  note('diffHighlights', {
    left: diff.left,
    right: diff.right,
    armed: await armSync(diff.candidate,
      { prime: 'shared', source: 'L3', sourcePane: 0, expect: 'r-gamma' }),
    unmapped: await probe(diff.candidate, { prime: 'r-gamma', source: 'L4', sourcePane: 0 }),
  });

  const empty = await openComparison({
    syncNavigationData: {
      type: 'sync_navigation',
      mappingEntries: [],
      disableMappingFallback: true,
      showDiffHighlights: true,
    },
  });
  /* Nothing here can pair, so there is no pairing to arm on; the settle is the
   * substitute, and is deliberately far longer than the arming latency
   * `syncArming` measures. */
  await sleep(5000);
  note('emptyMappingWithFallbackOff', {
    left: empty.left,
    right: empty.right,
    // Not one pairing exists in this configuration -- not even for the id that
    // appears verbatim on both sides. That is `[C2]`'s whole point: an empty
    // exported mapping plus a disabled fallback is a comparison in which
    // nothing corresponds to anything.
    shared: await probe(empty.candidate, { prime: 'r-gamma', source: 'shared', sourcePane: 0 }),
  });

  /* The case that makes the `paneIndex` filter load-bearing rather than
   * belt-and-braces: BOTH panes on the same graph. `Me_session.Session.validate`
   * permits it -- nothing requires a comparison's two panes to differ -- and
   * when it happens `detail.modelGraph.id` is identical for both, so a wait
   * filtered on the graph id alone resolves the right-pane wait on the LEFT
   * pane's event and the adapter finalizes a comparison whose second pane never
   * processed. Only the pane index separates them. */
  const same = connectHidden(syncConfig({ disableMappingFallback: true }));
  const sameLeft = awaitProcessed(same.element, LEFT, 0, 60000);
  same.element.selectNode(firstNode(LEFT), LEFT, LABEL, 0);
  const sameLeftOutcome = await sameLeft;
  const sameRight = awaitProcessed(same.element, LEFT, 1, 60000);
  same.element.selectNode(firstNode(LEFT), LEFT, LABEL, 1);
  const sameRightOutcome = await sameRight;
  note('samePaneGraph', {
    left: sameLeftOutcome,
    right: sameRightOutcome,
    panes: same.uiState?.paneStates?.length ?? 0,
  });

  /* 7. Replacement is the Path B lifecycle, across TWO waits. The visible
   * candidate stays current until both pane events of its replacement have
   * arrived, and a candidate abandoned between the two never becomes visible. */
  const replacement = connectHidden(syncConfig({ disableMappingFallback: true }));
  const replacementLeft = awaitProcessed(replacement.element, LEFT, 0, 60000);
  replacement.element.selectNode(firstNode(LEFT), LEFT, LABEL, 0);
  const replacementLeftOutcome = await replacementLeft;
  /* Abandoned HERE -- after its left pane, before its right. This is the window
   * the adapter must not finalize in, and the DOM state it leaves is exactly
   * what a quarantined entry looks like. */
  const abandonedVisible = replacement.slot.classList.contains('slot--current');
  note('abandonedBetweenPanes', {
    leftFired: replacementLeftOutcome.fired,
    abandonedVisible,
    abandonedStillConnected: replacement.element.isConnected,
    originalStillCurrent: supplied.candidate.slot.classList.contains('slot--current'),
    connectedElements: MOUNT.querySelectorAll('model-explorer-visualizer').length,
  });

  result.done = true;
}

run().catch((error) => {
  result.error = String(error && error.stack ? error.stack : error);
  result.done = true;
});
