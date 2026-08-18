/* Doubles for the two host facilities `Renderer` takes by injection: a
 * document and a clock.
 *
 * Deliberately NOT a DOM emulation. It models parent/child links, class and
 * attribute sets, listeners and connectedness -- the things the renderer's
 * state machine reads and writes -- and nothing about layout, custom element
 * upgrades or Angular. Everything this cannot say is said instead by
 * `web/test/lifecycle.spec.ts`, which drives the real element in Chromium.
 *
 * The one feature here that is not obvious is `rig`. Every mutation can be made
 * to fail in TWO ways, because they leave different wreckage and the renderer
 * has to distinguish them:
 *
 *   'throw'      -- fail without mutating
 *   'mutateThrow' -- apply the mutation, THEN fail
 *
 * A fake that only offered the first could not prove any of the postconditions
 * the design states for a partly-applied `finalize`, or the difference between
 * a `slot.remove()` that left the slot attached and one that detached it and
 * then threw.
 */

class FakeClassList {
  #node;
  #names = new Set();

  constructor(node) { this.#node = node; }

  add(name) {
    this.#node._fail('classList.add', () => this.#names.add(name));
  }

  remove(name) {
    this.#node._fail('classList.remove', () => this.#names.delete(name));
  }

  contains(name) { return this.#names.has(name); }
  get value() { return [...this.#names].join(' '); }
}

class FakeNode {
  constructor(tagName, document) {
    this.tagName = tagName;
    this.ownerDocument = document;
    this.parentNode = null;
    this.children = [];
    this.attributes = new Map();
    this.listeners = new Map();
    this.classList = new FakeClassList(this);
    this.rigs = new Map();
    this.calls = [];
    this.nodeData = [];
    this.selected = null;
    this.selections = [];
  }

  /* The two visualizer methods the renderer calls on a candidate. Riggable like
   * every other mutation; the real ones come from the Angular custom element.
   *
   * `paneIndex` is recorded rather than defaulted, so a test can tell "asked
   * for pane 0" from "did not say" -- the single-view path omits the argument
   * entirely and must keep doing so. Every call is kept, not just the last:
   * opening a comparison makes two, and which pane each addressed is the point.
   */
  selectNode(nodeId, graphId, label, paneIndex) {
    this._fail('selectNode', () => {
      // Absent when the caller did not pass one, rather than present-and-
      // undefined: "did not name a pane" and "named pane 0" are different
      // calls, and only the single-view path makes the former.
      this.selected = { nodeId, graphId, label, ...(paneIndex === undefined ? {} : { paneIndex }) };
      this.selections.push(this.selected);
    });
  }

  addNodeDataProviderData(name, data, paneIndex) {
    this._fail('addNodeDataProviderData', () => {
      this.nodeData.push({ name, data, ...(paneIndex === undefined ? {} : { paneIndex }) });
    });
  }

  /* The single choke point for "this operation may be rigged to fail". Records
   * the call either way, so a test can assert that `slot.remove()` was
   * attempted exactly once even when it threw. */
  _fail(name, apply) {
    this.calls.push(name);
    const rig = this.rigs.get(name);
    if (rig === undefined) return apply();
    rig.times -= 1;
    if (rig.times <= 0) this.rigs.delete(name);
    if (rig.mode === 'throw') throw new Error(`rigged: ${name}`);
    const value = apply();
    throw new Error(`rigged after mutation: ${name}`);
  }

  /** Make `name` fail on its next `times` calls; once by default. */
  rig(name, mode = 'throw', { times = 1 } = {}) {
    this.rigs.set(name, { mode, times });
    return this;
  }

  unrig(name) { this.rigs.delete(name); }

  callCount(name) { return this.calls.filter((call) => call === name).length; }

  /* Riggable, because the renderer's cleanup treats "I cannot tell whether the
   * removal took" as a different -- and terminal -- outcome from "it did not". */
  get isConnected() {
    return this._fail('isConnected', () => {
      let node = this;
      while (node.parentNode) node = node.parentNode;
      return node === this.ownerDocument.root;
    });
  }

  appendChild(child) {
    return this._fail('appendChild', () => {
      if (child.parentNode) child.parentNode._detach(child);
      child.parentNode = this;
      this.children.push(child);
      return child;
    });
  }

  _detach(child) {
    const index = this.children.indexOf(child);
    if (index >= 0) this.children.splice(index, 1);
    child.parentNode = null;
  }

  remove() {
    return this._fail('remove', () => {
      if (this.parentNode) this.parentNode._detach(this);
    });
  }

  setAttribute(name, value) {
    this._fail('setAttribute', () => this.attributes.set(name, String(value)));
  }

  removeAttribute(name) {
    this._fail('removeAttribute', () => this.attributes.delete(name));
  }

  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }

  addEventListener(type, handler) {
    this._fail('addEventListener', () => {
      if (!this.listeners.has(type)) this.listeners.set(type, new Set());
      this.listeners.get(type).add(handler);
    });
  }

  removeEventListener(type, handler) {
    this._fail('removeEventListener', () => this.listeners.get(type)?.delete(handler));
  }

  listenerCount(type) { return this.listeners.get(type)?.size ?? 0; }

  /* Synchronous, like a real dispatch. A handler that throws must not stop the
   * others, and must not propagate to the dispatcher -- matching how the
   * browser reports a listener exception rather than rethrowing it. */
  dispatchEvent(event) {
    const handlers = [...(this.listeners.get(event.type) ?? [])];
    for (const handler of handlers) handler(event);
    return true;
  }

  /* The expected event, in the shape the renderer reads it.
   *
   * `paneIndex` is part of the event's identity, not a detail: a comparison
   * waits on `(graph, pane)` because `Session.validate` permits both panes to
   * name the same graph. It defaults to 0 so every single-view test that
   * predates comparisons keeps meaning what it meant. */
  emitProcessed(graphId, paneIndex = 0) {
    this.dispatchEvent({
      type: 'modelGraphProcessed',
      detail: { modelGraph: { id: graphId }, paneIndex },
    });
  }
}

export function createDocument() {
  const document = {
    root: null,
    createElement(tagName) { return new FakeNode(tagName, document); },
  };
  document.root = new FakeNode('#document', document);
  document.mount = document.createElement('section');
  document.root.appendChild(document.mount);
  return document;
}

/* A clock the renderer takes in place of the global timer functions. Nothing
 * here waits in real time: `advance` fires whatever is due, in due order, and
 * re-entrant scheduling from inside a callback is handled by looping. */
export function createClock() {
  let currentTime = 0;
  let nextId = 1;
  const scheduled = new Map();

  const clock = {
    setTimeout(handler, delay) {
      const id = nextId++;
      scheduled.set(id, { at: currentTime + (delay || 0), handler, repeat: null });
      return id;
    },
    clearTimeout(id) { scheduled.delete(id); },
    setInterval(handler, delay) {
      const id = nextId++;
      scheduled.set(id, { at: currentTime + (delay || 0), handler, repeat: delay || 1 });
      return id;
    },
    clearInterval(id) { scheduled.delete(id); },

    get pending() { return scheduled.size; },
    get time() { return currentTime; },
    now() { return currentTime; },

    advance(ms) {
      const until = currentTime + ms;
      for (;;) {
        let due = null;
        for (const [id, entry] of scheduled) {
          if (entry.at <= until && (due === null || entry.at < due[1].at)) due = [id, entry];
        }
        if (!due) break;
        const [id, entry] = due;
        currentTime = entry.at;
        if (entry.repeat === null) scheduled.delete(id);
        else entry.at = currentTime + entry.repeat;
        entry.handler();
      }
      currentTime = until;
    },
  };
  return clock;
}

/** Lets a test await a promise's settlement without knowing when it settles. */
export function settled(promise) {
  return promise.then(
    (value) => ({ status: 'resolved', value }),
    (error) => ({ status: 'rejected', error }),
  );
}

/** Drains the microtask queue so a settled promise's continuations have run. */
export async function flush(times = 4) {
  for (let index = 0; index < times; index += 1) await Promise.resolve();
}
