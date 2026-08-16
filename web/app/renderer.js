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

const HEARTBEAT_MS = 5_000;
const SLOT_CLASS = 'visualizer-slot';
const CURRENT_CLASS = 'visualizer-slot--current';
/* How many times one slot's removal may be re-attempted before the renderer
 * stops promising to try again. */
const MAX_DETACH_ATTEMPTS = 3;

const text = (error) => (error && error.message) || String(error);
/* Safe, still-connected elements the renderer failed to remove. They are not
 * inconsistency -- the DOM is coherent -- but they hold quarantine slots. */
const isDebt = (state) => state === 'cleanup_failed' || state === 'cleanup_abandoned';

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

  async install(renderText) {
    if (this.#inconsistent) throw new RenderFailure('inconsistent', this.#inconsistent);
    // Parsing precedes everything fallible, and allocates no entry, no deferred
    // and no DOM: an invalid document is refused without the renderer having
    // mutated anything on its behalf.
    const descriptor = this.#parse(renderText);
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

  #parse(renderText) {
    let state;
    try {
      state = JSON.parse(renderText);
    } catch (error) {
      throw new RenderFailure('invalid', `session is not JSON: ${text(error)}`);
    }
    const view = state.views?.find((v) => v.id === state.defaultView);
    const graphId = view?.graph || state.graphCollections?.[0]?.graphs?.[0]?.id;
    if (!graphId) throw new RenderFailure('invalid', 'session has no renderable graph');
    const nodeCounts = Object.fromEntries(
      (state.graphCollections ?? []).flatMap((c) => c.graphs ?? []).map((g) => [g.id, g.nodes?.length ?? 0]),
    );
    return {
      graphId,
      graphCollections: state.graphCollections,
      nodeDataSets: state.nodeDataSets || [],
      modelName: state.model?.name ?? '(unknown)',
      collectionLabel: state.graphCollections?.[0]?.label,
      targetFirstNodeId: state.graphCollections?.[0]?.graphs?.find((g) => g.id === graphId)?.nodes?.[0]?.id,
      nodeCounts,
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

  #connect(entry) {
    const d = entry.descriptor;
    const slot = this.#doc.createElement('div');
    slot.classList.add(SLOT_CLASS);
    slot.setAttribute('aria-hidden', 'true');
    const element = this.#doc.createElement('model-explorer-visualizer');
    element.graphCollections = d.graphCollections;
    // `defaultGraphId` is not a real VisualizerConfig field (it appears nowhere
    // in the library's types or compiled bundle) -- kept in case a future
    // version adds it, but nothing depends on it. The library auto-selects some
    // graph on its own; `selectNode` below is what navigates to the one we want.
    element.config = { defaultGraphId: d.graphId };
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
    console.debug(`[renderer] model=${d.modelName} connected hidden, waiting for graph ${d.graphId}`);
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
    const seen = event?.detail?.modelGraph?.id;
    entry.seen.add(seen);
    if (seen !== d.graphId) {
      // The library settled on some other graph as its own default; navigate to
      // the one we asked for, the way a manual click does.
      if (entry.selected || !d.targetFirstNodeId) return;
      entry.selected = true;
      entry.element.selectNode(d.targetFirstNodeId, d.graphId, d.collectionLabel);
      return;
    }
    // The expected event: this element may now be removed safely.
    entry.safe = true;
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
    const handle = Object.freeze({ graph: d.graphId });
    this.#handles.set(handle, entry);
    entry.handle = handle;
    this.#resolve(entry, handle);
  }

  #installNodeData(entry) {
    const d = entry.descriptor;
    for (const set of d.nodeDataSets) {
      if (set.graph !== d.graphId) continue;
      const results = Object.fromEntries(
        (set.results || []).map(([nodeId, value]) => [nodeId, { value: value.value }]),
      );
      entry.element.addNodeDataProviderData(set.name, {
        results,
        gradient: [{ stop: 0, bgColor: '#e8f5e9' }, { stop: 1, bgColor: '#ffcdd2' }],
      });
    }
  }

  #onTimeout(entry) {
    if (entry.state !== 'connected') return;
    const d = entry.descriptor;
    console.warn(
      `[renderer] model=${d.modelName} timed out waiting for graph ${d.graphId}; received=${JSON.stringify([...entry.seen])}`,
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
    // A parked caller is settled too; `#startFailed` sees the flag and mutates
    // no DOM.
    if (victim) step(() => this.#settleCapacityWait(victim, new RenderFailure('inconsistent', why)));
    // Every connected element stays in place, the settled one included: bridge
    // state is unknown, so removing anything would assert a rollback nobody can
    // confirm.
    return errors.length === 0 ? { ok: true } : { ok: false, error: errors.join('; ') };
  }
}
