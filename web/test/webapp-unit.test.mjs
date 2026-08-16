import assert from 'node:assert/strict';
import test from 'node:test';
import { validateCatalog } from '../app/source_store.js';
import { Coordinator } from '../app/coordinator.js';
import { RenderFailure } from '../app/renderer.js';

for (const level of ['debug', 'warn', 'error']) console[level] = () => {};

const digest = 'a'.repeat(64);
const catalogue = { schemaVersion: 1, sourceRef: 'a'.repeat(40), defaultModel: 'a', models: [
  { id: 'a', displayName: 'A', bytes: '1', sha256: digest, url: 'models/a.json' },
] };

test('catalogue validation requires sorted, digested entries', () => {
  assert.equal(validateCatalog(catalogue).models[0].bytes, 1n);
  assert.throws(() => validateCatalog({ ...catalogue, models: [{ ...catalogue.models[0], sha256: 'A'.repeat(64) }] }));
  assert.throws(() => validateCatalog({ ...catalogue, defaultModel: 'missing' }));
});

test('coordinator transfers fresh request and source bytes exactly once', async () => {
  const worker = { postMessage(...args) { this.args = args; }, terminate() {} };
  const bridge = { request: { buildSession: () => ({ ok: true, request: new ArrayBuffer(2) }) }, session: {} };
  const coordinator = new Coordinator({ workerFactory: () => worker, bridge, render: {}, hard: {}, onStatus() {}, onError() {} });
  const bytes = new ArrayBuffer(3);
  await coordinator.load({ name: 'fixture.json', bytes: 3n, kind: 'local', readBytes: async () => bytes });
  assert.deepEqual(worker.args[0], { request: worker.args[0].request, bytes: worker.args[0].bytes });
  assert.equal(worker.args[1][0], worker.args[0].request);
  assert.equal(worker.args[1][1], bytes);
});

/* ------------------------------------------------------------------ fixtures */

const HARD = { maxResponseMetaBytes: 65_536, maxResponseDocumentBytes: 1 << 20 };
const source = () => ({ name: 'f.json', bytes: 3n, kind: 'local', readBytes: async () => new ArrayBuffer(3) });
/* A poisoned coordinator refuses with one fixed text; the diagnostic naming
 * what went wrong was delivered once, through onError. */
const POISONED = 'the application is in an unknown state. Reload the page to continue.';

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

/* One scenario: a coordinator wired to recording fakes, plus the two verbs the
 * protocol needs -- start a request, and deliver a session frame for it. */
function scenario({ session = {}, render = {} } = {}) {
  const workers = [];
  const errors = [];
  const statuses = [];
  const calls = { prepare: [], commit: [], abort: [], install: [], finalize: [], abortReady: [], markInconsistent: [], cancel: 0 };
  const handle = Object.freeze({ handle: true });
  let lastId = null;

  const bridge = {
    request: { buildSession: ({ id }) => { lastId = id; return { ok: true, request: new ArrayBuffer(2) }; } },
    session: {
      prepare: (id) => { calls.prepare.push(id); return (session.prepare ?? (() => ({ ok: true, token: id, render: '{}' })))(id); },
      commit: (token) => { calls.commit.push(token); return (session.commit ?? (() => ({ ok: true })))(token); },
      abort: (token) => { calls.abort.push(token); return (session.abort ?? (() => ({ ok: true })))(token); },
    },
  };
  const view = {
    install: (text) => { calls.install.push(text); return (render.install ?? (async () => handle))(text); },
    finalize: (h) => { calls.finalize.push(h); return (render.finalize ?? (() => ({ ok: true })))(h); },
    abortReady: (h) => { calls.abortReady.push(h); return (render.abortReady ?? (() => ({ ok: true })))(h); },
    markInconsistent: (r) => { calls.markInconsistent.push(r); return (render.markInconsistent ?? (() => ({ ok: true })))(r); },
    cancel: () => { calls.cancel += 1; return (render.cancel ?? (() => {}))(); },
  };
  const coordinator = new Coordinator({
    workerFactory: () => {
      const worker = { posted: [], terminated: 0, postMessage(...a) { this.posted.push(a); }, terminate() { this.terminated += 1; } };
      workers.push(worker);
      return worker;
    },
    bridge, render: view, hard: HARD,
    onStatus: (t) => statuses.push(t),
    onError: (t) => errors.push(t),
  });

  const start = async () => {
    await coordinator.load(source());
    return { worker: workers.at(-1), id: lastId };
  };
  const deliver = (request, overrides = {}) => {
    const payload = new TextEncoder().encode('{"session":true}').buffer;
    const meta = JSON.stringify({ kind: 'session', id: request.id, bytes: payload.byteLength, ...overrides });
    return request.worker.onmessage({ data: { meta, payload } });
  };
  const inFlight = () => coordinator.load(source()).then(() => 'accepted', (error) => error.message);

  return { coordinator, bridge, view, calls, errors, statuses, handle, workers, start, deliver, inFlight };
}

/** The next load() is refused, and nothing is constructed or posted for it. */
async function refused(s, label) {
  const workers = s.workers.length;
  assert.equal(await s.inFlight(), POISONED, label);
  assert.equal(s.workers.length, workers, label);
}

/* -------------------------------------------------------------- cancellation */

test('cancellation terminates the worker and admits no later message', async () => {
  const s = scenario();
  const a = await s.start();
  s.coordinator.cancel();
  assert.equal(a.worker.terminated, 1);
  await s.deliver(a);
  assert.deepEqual(s.calls.prepare, []);
  assert.deepEqual(s.errors, []);
});

test('cancelling during install aborts the token exactly once and says nothing', async () => {
  const install = deferred();
  const s = scenario({ render: { install: () => install.promise } });
  const a = await s.start();
  const done = s.deliver(a);
  assert.deepEqual(s.calls.prepare, [a.id]);
  s.coordinator.cancel();
  assert.deepEqual(s.calls.abort, [a.id]);
  install.resolve(s.handle);
  await done;
  // The stale install is still consumed -- and only once.
  assert.equal(s.calls.abortReady.length, 1);
  assert.deepEqual(s.calls.abort, [a.id]);
  assert.deepEqual(s.calls.commit, []);
  assert.deepEqual(s.errors, []);
  assert.ok(!s.statuses.includes('Model loaded'));
});

test('a cancelled install that later resolves cannot disturb a newer request', async () => {
  const install = deferred();
  const s = scenario({ render: { install: () => install.promise } });
  const a = await s.start();
  const done = s.deliver(a);
  s.coordinator.cancel();
  const b = await s.start();
  install.resolve(s.handle);
  await done;
  assert.deepEqual(s.calls.commit, []);
  assert.deepEqual(s.errors, []);
  assert.ok(!s.statuses.includes('Model loaded'));
  assert.equal(await s.inFlight(), 'a request is already in flight');
  assert.equal(b.worker.terminated, 0);
});

test('a valid cancel during the resolution continuation is benign', async () => {
  // install() resolves, a re-entrant cancel() releases the ready entry and
  // consumes its handle, then the queued continuation calls abortReady on it.
  const install = deferred();
  const s = scenario({
    render: { install: () => install.promise, abortReady: () => ({ ok: true, already: 'cancelled' }) },
  });
  const a = await s.start();
  const done = s.deliver(a);
  install.resolve(s.handle);
  s.coordinator.cancel();
  await done;
  assert.equal(s.calls.abortReady.length, 1);
  assert.deepEqual(s.errors, []);
  assert.equal(await s.inFlight(), 'accepted');   // not poisoned
});

/* ------------------------------------------------------ renderer rejections */

test('a stale renderer failure is silent; a current one reports exactly one error', async () => {
  const stale = deferred();
  const a = scenario({ render: { install: () => stale.promise } });
  const first = await a.start();
  const doneA = a.deliver(first);
  a.coordinator.cancel();
  stale.reject(new Error('boom'));
  await doneA;
  assert.deepEqual(a.errors, []);

  const b = scenario({ render: { install: async () => { throw new Error('boom'); } } });
  await b.deliver(await b.start());
  assert.deepEqual(b.errors, ['boom']);
});

test('a renderer cancellation of a still-current request is a reported protocol mismatch', async () => {
  // The expected shape: authority is already lost, so nothing is said.
  const stale = deferred();
  const a = scenario({ render: { install: () => stale.promise } });
  const first = await a.start();
  const doneA = a.deliver(first);
  a.coordinator.cancel();
  stale.reject(new RenderFailure('cancelled', 'render cancelled'));
  await doneA;
  assert.deepEqual(a.errors, []);

  // The protocol mismatch: a `cancelled` for a record that never lost
  // authority. Suppressing it on the kind alone would strand the record in
  // #pending and hang every later load().
  const b = scenario({ render: { install: async () => { throw new RenderFailure('cancelled', 'render cancelled'); } } });
  const request = await b.start();
  await b.deliver(request);
  assert.equal(b.errors.length, 1);
  assert.match(b.errors[0], /still current/);
  assert.equal(request.worker.terminated, 1);
  assert.equal(await b.inFlight(), 'accepted');
});

test('a terminal renderer is fatal whoever discovers it', async () => {
  // (a) the rejected record is current.
  const a = scenario({ render: { install: async () => { throw new RenderFailure('inconsistent', 'the view is broken'); } } });
  await a.deliver(await a.start());
  assert.equal(a.errors.length, 1);
  assert.match(a.errors[0], /the view is broken\. Reload the page/);
  await refused(a);
  assert.equal(a.workers.length, 1);            // the refused load posted nothing

  // (b) the rejected record is stale, and the newer one has not yet entered
  //     render.install -- so the renderer has no caller of its own to reject.
  const install = deferred();
  const b = scenario({ render: { install: () => install.promise } });
  const first = await b.start();
  const doneB = b.deliver(first);
  b.coordinator.cancel();
  const second = await b.start();
  install.reject(new RenderFailure('inconsistent', 'the view is broken'));
  await doneB;
  assert.equal(b.errors.length, 1);
  assert.match(b.errors[0], /Reload the page/);
  assert.equal(second.worker.terminated, 1);    // the current victim is settled
  await refused(b);
});

/* ------------------------------------------------------------------- commit */

test('a rejected commit keeps the old view and reports once', async () => {
  const s = scenario({ session: { commit: () => ({ ok: false, error: 'invalid session token' }) } });
  await s.deliver(await s.start());
  assert.equal(s.calls.abortReady.length, 1);
  assert.deepEqual(s.calls.finalize, []);
  assert.deepEqual(s.errors, ['invalid session token']);
  assert.equal(await s.inFlight(), 'accepted');   // a rejection is not fatal
});

test('a rejected commit whose rollback leaves debt says so without poisoning', async () => {
  const s = scenario({
    session: { commit: () => ({ ok: false, error: 'invalid session token' }) },
    render: { abortReady: () => ({ ok: false, state: 'cleanup_failed', error: 'stuck' }) },
  });
  await s.deliver(await s.start());
  assert.equal(s.errors.length, 1);
  assert.match(s.errors[0], /invalid session token \(the abandoned view could not be removed\)/);
  assert.equal(await s.inFlight(), 'accepted');
});

test('a commit that throws leaves the state unknown and poisons the page', async () => {
  for (const mutate of [false, true]) {
    const s = scenario({
      session: { commit: () => { if (mutate) s.calls.commit.push('applied'); throw new Error('bridge exploded'); } },
    });
    const request = await s.start();
    await s.deliver(request);
    assert.equal(s.errors.length, 1, String(mutate));
    assert.match(s.errors[0], /may or may not have been installed/);
    assert.match(s.errors[0], /Reload the page/);
    // No rollback is claimed: nothing is aborted and nothing is discarded.
    assert.deepEqual(s.calls.abortReady, [], String(mutate));
    assert.deepEqual(s.calls.markInconsistent, ['session state unknown after commit']);
    assert.equal(request.worker.terminated, 1);
    await refused(s);
    assert.equal(s.workers.length, 1);          // the refused load constructed nothing
  }
});

test('the unknown-commit diagnostic carries the renderer cleanup failure', async () => {
  const s = scenario({
    session: { commit: () => { throw new Error('bridge exploded'); } },
    render: { markInconsistent: () => ({ ok: false, error: 'timers survived' }) },
  });
  await s.deliver(await s.start());
  assert.match(s.errors[0], /\(timers survived\)/);
});

/* ----------------------------------------------------------------- finalize */

test('every post-commit finalize failure is fatal and none claims a rollback', async () => {
  const seen = new Set();
  for (const state of ['known_old', 'cancelled', 'inconsistent', 'misuse']) {
    const s = scenario({ render: { finalize: () => ({ ok: false, state, error: 'why' }) } });
    await s.deliver(await s.start());
    assert.equal(s.errors.length, 1, state);
    assert.match(s.errors[0], /Reload the page/, state);
    assert.doesNotMatch(s.errors[0], /roll(ed)? back/i, state);
    assert.match(s.errors[0], /installed/, state);
    assert.ok(!seen.has(s.errors[0]), `duplicate diagnostic for ${state}`);
    seen.add(s.errors[0]);
    await refused(s);
  }
});

test('losing authority between a confirmed commit and finalize is fatal, not silent', async () => {
  const s = scenario({ session: { commit: () => { s.coordinator.cancel(); return { ok: true }; } } });
  await s.deliver(await s.start());
  assert.deepEqual(s.calls.finalize, []);
  assert.equal(s.errors.length, 1);
  assert.match(s.errors[0], /superseded/);
  assert.match(s.errors[0], /Reload the page/);
});

/* -------------------------------------------------------- abortReady results */

test('every abortReady result is consumed', async () => {
  const cases = [
    { result: { ok: true }, fatal: false },
    { result: { ok: true, already: 'cancelled' }, fatal: false },
    { result: { ok: false, state: 'cleanup_failed', error: 'stuck' }, fatal: false },
    { result: { ok: false, state: 'inconsistent', error: 'unknown' }, fatal: true },
    { result: { ok: false, state: 'misuse', error: 'forged' }, fatal: true },
  ];
  for (const { result, fatal } of cases) {
    const install = deferred();
    const s = scenario({ render: { install: () => install.promise, abortReady: () => result } });
    const a = await s.start();
    const done = s.deliver(a);
    s.coordinator.cancel();
    install.resolve(s.handle);
    await done;
    const label = JSON.stringify(result);
    if (fatal) {
      assert.equal(s.errors.length, 1, label);
      assert.match(s.errors[0], /Reload the page/, label);
      await refused(s, label);
    } else {
      // Stale work never speaks for itself.
      assert.deepEqual(s.errors, [], label);
      assert.equal(await s.inFlight(), 'accepted', label);
    }
  }
});

test('a stale record poisoning the page settles the current one, not itself', async () => {
  for (const state of ['inconsistent', 'misuse']) {
    const install = deferred();
    const s = scenario({
      render: { install: () => install.promise, abortReady: () => ({ ok: false, state, error: 'why' }) },
    });
    const a = await s.start();
    const doneA = s.deliver(a);
    s.coordinator.cancel();
    const b = await s.start();
    const doneB = s.deliver(b);                 // B is past prepare, so it holds a token
    install.resolve(s.handle);
    await Promise.all([doneA, doneB]);
    assert.equal(s.errors.length, 1, state);
    assert.equal(b.worker.terminated, 1, state);
    assert.equal(s.calls.abort.filter((t) => t === b.id).length, 1, state);
    assert.deepEqual(s.calls.commit, [], state);
  }
});

/* ------------------------------------------------------------ re-entrancy */

test('fatal termination survives re-entrancy and speaks exactly once', async () => {
  // (a) bridge.commit cancels A and starts B, then reports success.
  const a = scenario({ session: { commit: () => { a.coordinator.cancel(); a.coordinator.load(source()); return { ok: true }; } } });
  const first = await a.start();
  await a.deliver(first);
  const b = a.workers[1];
  assert.equal(a.errors.length, 1);
  assert.match(a.errors[0], /Reload the page/);
  assert.equal(b.terminated, 1);                // the CURRENT victim is settled
  assert.deepEqual(a.calls.finalize, []);
  // B never reached prepare, so it has a worker and no token: nothing to abort.
  assert.deepEqual(a.calls.abort, []);
  await refused(a);

  // (b) finalize cancels A and starts B, then returns each fatal result.
  for (const state of ['known_old', 'cancelled', 'inconsistent', 'misuse']) {
    const s = scenario({ render: { finalize: () => { s.coordinator.cancel(); s.coordinator.load(source()); return { ok: false, state }; } } });
    await s.deliver(await s.start());
    assert.equal(s.errors.length, 1, state);
    assert.equal(s.workers[1].terminated, 1, state);
    assert.deepEqual(s.calls.abort, [], state);
    await refused(s, state);
  }

  // (c) #cancelRenderer re-enters Coordinator.cancel() from inside #fatal.
  const c = scenario({
    render: { finalize: () => ({ ok: false, state: 'misuse' }), cancel: () => { c.coordinator.cancel(); } },
  });
  await c.deliver(await c.start());
  assert.equal(c.errors.length, 1);
  assert.match(c.errors[0], /Reload the page/);
});

test('re-entrant success does not announce a model the user moved on from', async () => {
  // (a) finalize cancels and starts a newer request: ownership is already gone.
  const a = scenario({ render: { finalize: () => { a.coordinator.cancel(); a.coordinator.load(source()); return { ok: true }; } } });
  await a.deliver(await a.start());
  assert.ok(!a.statuses.includes('Model loaded'));
  assert.deepEqual(a.errors, []);
  assert.equal(await a.inFlight(), 'a request is already in flight');

  // (b) worker.terminate cancels: ownership held at entry, but the epoch moved
  //     while the worker was being retired. A single pre-retirement flag cannot
  //     tell this from (a).
  const b = scenario();
  const request = await b.start();
  request.worker.terminate = function terminate() {
    this.terminated = (this.terminated ?? 0) + 1;
    b.coordinator.cancel();
    b.coordinator.load(source());
  };
  await b.deliver(request);
  assert.ok(!b.statuses.includes('Model loaded'));
  assert.deepEqual(b.errors, []);
  assert.equal(request.worker.terminated, 1);   // exactly once, despite the re-entrant cancel
  assert.equal(await b.inFlight(), 'a request is already in flight');
});

/* ------------------------------------------------------ contained failures */

test('throwing teardown never escapes and never settles twice', async () => {
  const thrower = () => { throw new Error('nope'); };

  const a = scenario();
  const first = await a.start();
  first.worker.terminate = thrower;
  await a.deliver(first);                       // success path with a throwing terminate
  assert.deepEqual(a.statuses.at(-1), 'Model loaded');

  const install = deferred();
  const b = scenario({ render: { install: () => install.promise, cancel: thrower }, session: { abort: thrower } });
  const second = await b.start();
  const done = b.deliver(second);
  b.coordinator.cancel();                       // render.cancel and bridge.abort both throw
  assert.deepEqual(b.calls.abort, [second.id]); // attempted exactly once, despite the throw
  install.resolve(b.handle);
  await done;
  assert.deepEqual(b.errors, []);
  assert.equal(b.calls.abort.length, 1);
});

test('the worker is terminated exactly once on success', async () => {
  const s = scenario();
  const request = await s.start();
  await s.deliver(request);
  assert.equal(request.worker.terminated, 1);
  assert.deepEqual(s.calls.commit, [request.id]);
  assert.equal(s.calls.finalize.length, 1);
  assert.equal(s.statuses.at(-1), 'Model loaded');
  assert.equal(await s.inFlight(), 'accepted');
});
