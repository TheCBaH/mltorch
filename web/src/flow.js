/* Phase 4 Prerequisite: what the visualizer's SELECTED-NODE event actually
 * does, measured rather than assumed.
 *
 * The production adapter routes a flow-node selection to a declared stage view
 * or comparison, so it has to tell a user's click from the shell's own
 * `selectNode` during setup. `NodeInfo` carries no origin bit
 * (`{nodeId, graphId, collectionLabel, node?, paneId?}`), so the only thing
 * that can distinguish them is the ORDER and IDENTITY of the events. Three
 * designs for that suppression have already been wrong; this fixture exists so
 * the fourth rests on a measurement.
 *
 * Records into `window.__flow`. Asserted by `web/test/flow.spec.ts`.
 */

const MOUNT = document.getElementById('mount');
const result = { steps: {}, errors: [] };
window.__flow = result;

const LABEL = 'mltorch:flow';
const FLOW = 'g/flow';
const OTHER = 'g/other';

const note = (key, value) => { result.steps[key] = value; };

window.addEventListener('error', (e) => result.errors.push(String(e.message)));
window.addEventListener('unhandledrejection', (e) => result.errors.push(String(e.reason)));

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

/* The bipartite shape `Me_flow_graph` emits: state -> transition -> state.
 * State and transition ids are deliberately NOT distinguishable by label or
 * position -- only by which declared list they appear in, which is the rule the
 * adapter has to obey. */
const STATES = ['s/pt2/000', 's/native/000', 's/native/001'];
const TRANSITIONS = ['t/native/000', 't/native/001'];

const collections = [{
  label: LABEL,
  graphs: [
    {
      id: FLOW,
      nodes: [
        node('s/pt2/000'),
        node('t/native/000', ['s/pt2/000']),
        node('s/native/000', ['t/native/000']),
        node('t/native/001', ['s/native/000']),
        node('s/native/001', ['t/native/001']),
      ],
    },
    { id: OTHER, nodes: [node('x0'), node('x1', ['x0'])] },
  ],
}];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function awaitProcessed(element, graphId, ms) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (fired) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      element.removeEventListener('modelGraphProcessed', onEvent);
      resolve(fired);
    };
    const timer = setTimeout(() => finish(false), ms);
    const onEvent = (event) => {
      if (event.detail?.modelGraph?.id === graphId) finish(true);
    };
    element.addEventListener('modelGraphProcessed', onEvent);
  });
}

/* EVERY selected-node event, in order, with a monotonic sequence number and a
 * caller-supplied phase marker. The whole point is the sequence: a count of
 * wrapper calls is not the same as a count of events, which is exactly what
 * broke the previous design. */
function recordSelections(candidate) {
  candidate.selections = [];
  candidate.phase = 'setup';
  candidate.element.addEventListener('selectedNodeChanged', (event) => {
    const d = event.detail;
    candidate.selections.push({
      seq: candidate.selections.length,
      phase: candidate.phase,
      // `undefined` detail is a real possibility the adapter must survive; it
      // is recorded as such rather than being allowed to throw.
      nodeId: d === undefined ? '<no detail>' : (d?.nodeId ?? null),
      graphId: d?.graphId ?? null,
      collectionLabel: d?.collectionLabel ?? null,
    });
  });
}

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
  recordSelections(candidate);
  return candidate;
}

const reveal = (slot) => {
  slot.classList.add('slot--current');
  slot.removeAttribute('aria-hidden');
};

/* Opens the flow graph the way `Renderer` does: connect hidden, wait for the
 * graph event, correct a mismatched auto-selection with `selectNode`. Returns
 * before revealing, so the caller can inspect exactly what setup emitted. */
async function openFlow({ select = true } = {}) {
  const candidate = connectHidden({ defaultGraphId: FLOW });
  const processed = await awaitProcessed(candidate.element, FLOW, 20000);
  candidate.processed = processed;
  if (select) {
    candidate.requested = STATES[0];
    candidate.element.selectNode(STATES[0], FLOW, LABEL);
  }
  return candidate;
}

const settleSelections = async (candidate, ms = 3000) => {
  const deadline = Date.now() + ms;
  let last = -1;
  while (Date.now() < deadline) {
    if (candidate.selections.length === last && candidate.selections.length > 0) break;
    last = candidate.selections.length;
    await sleep(250);
  }
  return candidate.selections.slice();
};

async function run() {
  try {
    /* --- 1/2/3. the graph opens, and setup's own emissions are enumerated --- */
    const a = await openFlow();
    note('processed', a.processed);
    /* Everything up to here is programmatic: the slot is `visibility:hidden;
     * pointer-events:none`, so no user event can exist yet. This is the
     * sequence the adapter's suppression must model. */
    const setup = await settleSelections(a);
    note('setupSelections', setup);
    note('setupRequested', a.requested);
    /* The load itself, BEFORE any `selectNode`: does the element auto-select?
     * A residual case the adapter cannot account for if it happens. */
    const auto = await openFlow({ select: false });
    note('autoSelectOnLoad', await settleSelections(auto, 2500));

    /* --- 4. clear-before / target-after: is the target event separable? --- */
    note('setupShape', {
      total: setup.length,
      clears: setup.filter((s) => s.nodeId === '').length,
      matchingRequested: setup.filter((s) => s.nodeId === a.requested).length,
      // Every setup event names the flow graph and collection, or the adapter's
      // graph/collection guard would not be able to scope anything.
      allOnFlowGraph: setup.every((s) => s.graphId === FLOW && s.collectionLabel === LABEL),
    });

    /* --- 5. A -> B -> A, the sequence that killed the first design --- */
    reveal(a.slot);
    a.phase = 'user';
    const before = a.selections.length;
    // Driven through the same public method a click reaches: `AppService`'s
    // `selectNode` is the single path for both, so the events are the same
    // events. The fixture cannot synthesise a real click into a WebGL canvas.
    a.element.selectNode(TRANSITIONS[0], FLOW, LABEL);
    await settleSelections(a);
    a.element.selectNode(STATES[0], FLOW, LABEL);
    await settleSelections(a);
    const user = a.selections.slice(before);
    note('userSelections', user);
    note('reselectRoutes', {
      // The final A must be present as a distinct event; a design that swallows
      // it on a stale expectation would show it missing here.
      sawB: user.some((s) => s.nodeId === TRANSITIONS[0]),
      sawAAgain: user.some((s) => s.nodeId === STATES[0]),
    });

    /* --- 6. states and transitions are told apart only by declared id --- */
    note('identity', {
      states: STATES,
      transitions: TRANSITIONS,
      // No label or namespace distinguishes them in the graph itself.
      labelsDistinguish: false,
      selectedAreKnown: user
        .filter((s) => s.nodeId)
        .every((s) => STATES.includes(s.nodeId) || TRANSITIONS.includes(s.nodeId)),
    });

    /* --- an event from another graph is ignorable on its own fields --- */
    const other = connectHidden({ defaultGraphId: OTHER });
    await awaitProcessed(other.element, OTHER, 20000);
    other.element.selectNode('x0', OTHER, LABEL);
    const foreign = await settleSelections(other, 2500);
    note('foreignGraph', {
      any: foreign.length > 0,
      // The adapter scopes on graphId, so a foreign selection must be
      // distinguishable without inspecting anything private.
      allNameOtherGraph: foreign.filter((s) => s.nodeId).every((s) => s.graphId === OTHER),
    });

    /* --- 7. a replaced candidate's late event carries no authority --- */
    const late = await openFlow();
    await settleSelections(late);
    late.phase = 'after-replacement';
    const beforeLate = late.selections.length;
    // The production renderer removes its listener here; the fixture keeps it
    // to prove the element still emits, which is WHY removal is required.
    late.element.selectNode(STATES[1], FLOW, LABEL);
    await settleSelections(late);
    note('lateEventStillEmitted', late.selections.length > beforeLate);
    note('listenerRemovalStops', (() => {
      const seen = [];
      const handler = (e) => seen.push(e.detail?.nodeId ?? null);
      late.element.addEventListener('selectedNodeChanged', handler);
      late.element.removeEventListener('selectedNodeChanged', handler);
      late.element.selectNode(TRANSITIONS[1], FLOW, LABEL);
      return seen.length;
    })());

    note('done', true);
  } catch (error) {
    result.errors.push(String(error && error.message ? error.message : error));
    note('done', false);
  }
}

run();
