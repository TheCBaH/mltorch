/* The Path B lifecycle gate.
 *
 * `cancel.md` chooses to keep a cancelled Model Explorer install CONNECTED and
 * merely hidden, because the pinned element has no abort or dispose API and
 * disconnecting one mid-flight is what throws Angular's NG0953. That decision
 * rests on three properties of the real element which the application cannot
 * assume and a fake DOM cannot check:
 *
 *   1. a candidate in a connected `visibility:hidden` slot still lays out and
 *      still emits `modelGraphProcessed`;
 *   2. two connected candidates process concurrently, so an abandoned one does
 *      not serialise the next install behind itself;
 *   3. removing a candidate AFTER its expected event is safe, and the element's
 *      parent never changes between connection and that removal.
 *
 * This driver records those, the same way the 0A spike and the Stage 1 gate
 * record theirs; `web/test/lifecycle.spec.ts` is what turns them into
 * assertions. It drives the raw element rather than the application, so it is
 * green against the tree as it stands -- that is the point of committing it
 * first. If it ever fails, the design is wrong, not the assertion.
 */

const MOUNT = document.getElementById('mount');
const result = { steps: {} };
window.__lifecycle = result;

function note(key, value) {
  result.steps[key] = value;
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

const ORIGIN = performance.now();
const now = () => Math.round(performance.now() - ORIGIN);

/* The application's own wait, in miniature (`web/app/renderer.js:49-68`).
 *
 * Filtered by graph id, for the reason session.js documents at length: an
 * element that already processed one graph still carries queued events from
 * that first graph's own work, so an unfiltered listener can resolve on one of
 * those instead of on the graph actually asked for.
 *
 * And when the id does not match, ONE `selectNode` -- because `defaultGraphId`
 * is inert (`renderer.js:15-18`) and the library picks a graph of its own, so a
 * candidate asked for any other graph would otherwise never emit for it. Using
 * the real mechanism here is deliberate: a gate that drove the element
 * differently from the application would be pinning a lifecycle nobody runs. */
function awaitProcessed(candidate, ms) {
  const { element, graphId, firstNodeId, label } = candidate;
  const startedAt = now();
  return new Promise((resolve) => {
    let settled = false;
    let selected = false;
    const finish = (fired) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      element.removeEventListener('modelGraphProcessed', onEvent);
      resolve({ fired, startedAt, firedAt: now(), ms: now() - startedAt });
    };
    const timer = setTimeout(() => finish(false), ms);
    const onEvent = (event) => {
      const seen = event.detail?.modelGraph?.id;
      if (seen === graphId) return finish(true);
      if (selected || !firstNodeId) return;
      selected = true;
      element.selectNode(firstNodeId, graphId, label);
    };
    element.addEventListener('modelGraphProcessed', onEvent);
  });
}

/* One slot, one element, connected exactly once -- the shape the renderer's
 * `#connect` will have. The element is configured while detached and the slot
 * is appended to the mount as a whole, so the element is connected by the same
 * single operation that makes its slot a child of the mount. */
function connectHidden(collections, graphId) {
  const slot = document.createElement('div');
  slot.className = 'slot';
  slot.setAttribute('aria-hidden', 'true');
  const element = document.createElement('model-explorer-visualizer');
  element.graphCollections = collections;
  element.config = { defaultGraphId: graphId };
  slot.appendChild(element);
  MOUNT.appendChild(slot);
  return {
    slot,
    element,
    graphId,
    label: collections[0]?.label,
    firstNodeId: collections[0]?.graphs?.find((g) => g.id === graphId)?.nodes?.[0]?.id,
  };
}

function reveal(slot) {
  slot.classList.add('slot--current');
  slot.removeAttribute('aria-hidden');
}

function box(element) {
  const rect = element.getBoundingClientRect();
  return { width: Math.round(rect.width), height: Math.round(rect.height) };
}

async function run() {
  const collections = await fetchJson('collection.json');
  const session = await fetchJson('session.json');

  /* 1. A hidden connected candidate lays out and processes.
   *
   * Both halves matter and they fail differently. A zero-sized box would mean
   * the renderer had handed Model Explorer no room to lay out in, and the
   * design's promise that a hidden candidate is doing real work would be
   * hollow even if the event still fired. */
  const first = connectHidden(collections, 'g/flow');
  const hiddenBox = box(first.element);
  const hiddenVisibility = getComputedStyle(first.element).visibility;
  const firstProcessed = await awaitProcessed(first, 30000);
  note('hiddenSlotProcesses', {
    ...firstProcessed,
    ...hiddenBox,
    visibility: hiddenVisibility,
    connected: first.element.isConnected,
  });

  /* The parent an element is connected under must be the parent it is removed
   * from: Path B never reparents, and a lifecycle that moved the element
   * between containers would be relying on Angular Elements' delayed-destroy
   * behaviour instead of avoiding the question. Sampled here, again after
   * processing, and once more just before removal. */
  const parentAtConnection = first.element.parentNode === first.slot;

  /* 2. A visible current and a hidden candidate process CONCURRENTLY.
   *
   * This is the property that removes the 90-second stall. If the pinned
   * element serialised its layout worker across instances, a cancelled install
   * would still block its replacement and Path B would buy nothing. The second
   * candidate carries the real exported MobileNet session, so this is not a
   * question answered only for a toy graph. */
  reveal(first.slot);
  const second = connectHidden(session.graphCollections, 'g/native/000');
  const third = connectHidden(collections, 'g/native/003');
  const [secondProcessed, thirdProcessed] = await Promise.all([
    awaitProcessed(second, 60000),
    awaitProcessed(third, 60000),
  ]);
  note('concurrentProcessing', {
    visibleCurrentStillConnected: first.element.isConnected,
    real: secondProcessed,
    small: thirdProcessed,
  });

  /* 3. An abandoned candidate does not delay the next one.
   *
   * `abandoned` is what a quarantined entry looks like from the DOM's side:
   * connected, hidden, nobody waiting on it. A fresh candidate is started
   * immediately afterwards and timed. The application's own 90-second timeout
   * is not involved -- that is the point, since under the old queue this
   * candidate could not even have started. */
  const abandoned = connectHidden(session.graphCollections, 'g/symbolic/000');
  let abandonedFinished = false;
  const abandonedWatch = awaitProcessed(abandoned, 90000)
    .then((outcome) => { abandonedFinished = outcome.fired; return outcome; });
  const next = connectHidden(collections, 'g/native/000');
  const nextProcessed = await awaitProcessed(next, 60000);
  note('abandonedDoesNotBlock', {
    ...nextProcessed,
    abandonedFinishedFirst: abandonedFinished,
    abandonedStillConnected: abandoned.element.isConnected,
    connectedElements: MOUNT.querySelectorAll('model-explorer-visualizer').length,
  });

  /* 4. Removal AFTER the expected event, which is the only point at which the
   * design permits it. The element's parent is checked one last time first, so
   * the three samples together say it never moved. */
  const parentBeforeRemoval = first.element.parentNode === first.slot;
  const removedElement = first.element;
  first.slot.remove();
  note('safeRemoval', {
    parentAtConnection,
    parentBeforeRemoval,
    parentStable: parentAtConnection && parentBeforeRemoval,
    disconnected: !removedElement.isConnected,
    slotDetached: first.slot.parentNode === null,
  });

  /* Angular reports a destroyed-OutputRef emit asynchronously, so a removal
   * that was going to be unsafe has to be given time to say so. The spec fails
   * on any NG0953 in a page error; this is the window in which one would
   * arrive. */
  const abandonedOutcome = await Promise.race([
    abandonedWatch,
    new Promise((resolve) => setTimeout(() => resolve({ stillPending: true }), 3000)),
  ]);
  note('settled', {
    connectedElements: MOUNT.querySelectorAll('model-explorer-visualizer').length,
    abandonedOutcome,
  });

  result.done = true;
}

run().catch((error) => {
  result.error = String(error && error.stack ? error.stack : error);
  result.done = true;
});
