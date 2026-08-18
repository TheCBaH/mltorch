import assert from 'node:assert/strict';
import test from 'node:test';
import { Renderer } from '../app/renderer.js';
import { createClock, createDocument, flush, settled } from './fake_dom.mjs';

const CURRENT = 'visualizer-slot--current';

/* The renderer logs progress the way the application does. This suite asserts on
 * state, so keep the TAP stream readable. */
for (const level of ['debug', 'warn', 'error']) console[level] = () => {};

const session = (graphId, nodeDataSets = []) => JSON.stringify({
  model: { name: graphId },
  defaultView: 'v',
  views: [{ id: 'v', graph: graphId }],
  graphCollections: [{ label: 'c', graphs: [{ id: graphId, nodes: [{ id: `${graphId}-n0` }] }] }],
  nodeDataSets,
});

function harness({ hardMaxQuarantined = 2, maxQuarantined } = {}) {
  const doc = createDocument();
  const clock = createClock();
  const renderer = new Renderer({
    mount: doc.mount,
    hardMaxQuarantined,
    maxQuarantined,
    doc,
    timers: clock,
    now: () => clock.now(),
  });
  const slots = () => doc.mount.children;
  const last = () => doc.mount.children.at(-1);
  const element = (slot) => slot.children[0];
  const emit = (slot, graphId) => element(slot).emitProcessed(graphId);
  return { doc, clock, renderer, mount: doc.mount, slots, last, element, emit };
}

/** Installs and drives a candidate all the way to `ready`, returning its handle. */
async function ready(h, graphId, nodeDataSets = []) {
  const promise = h.renderer.install(session(graphId, nodeDataSets));
  const slot = h.last();
  h.emit(slot, graphId);
  return { slot, handle: await promise };
}

const isKind = (kind) => (error) => error.name === 'RenderFailure' && error.kind === kind;

/* ------------------------------------------------------------- construction */

test('the hard quarantine ceiling is mandatory and can only be tightened', () => {
  const doc = createDocument();
  const base = { mount: doc.mount, doc, timers: createClock() };
  assert.throws(() => new Renderer({ ...base }), /hardMaxQuarantined/);
  assert.throws(() => new Renderer({ ...base, hardMaxQuarantined: 0 }), /hardMaxQuarantined/);
  assert.throws(() => new Renderer({ ...base, hardMaxQuarantined: 2.5 }), /hardMaxQuarantined/);
  assert.equal(new Renderer({ ...base, hardMaxQuarantined: 3 }).maxQuarantined, 3);
  // A requested policy tightens...
  assert.equal(new Renderer({ ...base, hardMaxQuarantined: 3, maxQuarantined: 1 }).maxQuarantined, 1);
  // ...and never widens.
  assert.equal(new Renderer({ ...base, hardMaxQuarantined: 3, maxQuarantined: 9 }).maxQuarantined, 3);
});

/* ------------------------------------------------------------------ install */

test('a candidate processes in a connected but hidden slot', async () => {
  const h = harness();
  const promise = h.renderer.install(session('g1'));
  assert.equal(h.slots().length, 1);
  const slot = h.last();
  assert.ok(slot.isConnected);
  assert.ok(slot.classList.contains('visualizer-slot'));
  assert.ok(!slot.classList.contains(CURRENT));
  assert.equal(slot.getAttribute('aria-hidden'), 'true');
  h.emit(slot, 'g1');
  const handle = await promise;
  assert.ok(handle);
  // install() alone reveals nothing.
  assert.ok(!slot.classList.contains(CURRENT));
  assert.equal(h.renderer.finalize(handle).ok, true);
  assert.ok(slot.classList.contains(CURRENT));
  assert.equal(slot.getAttribute('aria-hidden'), null);
});

test('an invalid document is refused without touching the DOM', async () => {
  const h = harness();
  await assert.rejects(h.renderer.install('{'), isKind('invalid'));
  await assert.rejects(h.renderer.install('{"graphCollections":[]}'), isKind('invalid'));
  assert.equal(h.slots().length, 0);
});

test('node data is installed for the target graph before the handle exists', async () => {
  const h = harness();
  const sets = [
    /* The wire shape, an object per result -- the fixture used to carry a pair,
       which is the same misreading the installer had, so the two agreed with
       each other and neither with the exporter. */
    { graph: 'g1', name: 'metric', results: [{ nodeId: 'n', value: { value: 1 } }] },
    { graph: 'other', name: 'ignored', results: [] },
  ];
  const { slot } = await ready(h, 'g1', sets);
  assert.deepEqual(h.element(slot).nodeData.map((d) => d.name), ['metric']);
  assert.deepEqual(h.element(slot).nodeData[0].data.results, { n: { value: 1 } });
});

test('a non-target event selects the target graph exactly once', async () => {
  const h = harness();
  const promise = h.renderer.install(session('g1'));
  const slot = h.last();
  h.emit(slot, 'other');
  h.emit(slot, 'another');
  assert.deepEqual(h.element(slot).selected, { nodeId: 'g1-n0', graphId: 'g1', label: 'c' });
  assert.equal(h.element(slot).callCount('selectNode'), 1);
  h.emit(slot, 'g1');
  await promise;
});

/* ----------------------------------------------------------------- finalize */

test('finalize reveals the newcomer and removes the predecessor', async () => {
  const h = harness();
  const first = await ready(h, 'g1');
  h.renderer.finalize(first.handle);
  const second = await ready(h, 'g2');
  // Both are connected while the second is ready but not finalized, and only
  // the first is visible.
  assert.equal(h.slots().length, 2);
  assert.ok(first.slot.classList.contains(CURRENT));
  assert.ok(!second.slot.classList.contains(CURRENT));
  assert.deepEqual(h.renderer.finalize(second.handle), { ok: true });
  assert.equal(h.slots().length, 1);
  assert.equal(h.slots()[0], second.slot);
  assert.ok(second.slot.classList.contains(CURRENT));
});

test('the previous view stays visible and connected while a candidate is active', async () => {
  const h = harness();
  const first = await ready(h, 'g1');
  h.renderer.finalize(first.handle);
  const promise = h.renderer.install(session('g2'));
  assert.ok(first.slot.isConnected);
  assert.ok(first.slot.classList.contains(CURRENT));
  // ...and while it is quarantined by a timeout.
  h.clock.advance(90_000);
  await assert.rejects(promise, isKind('timeout'));
  assert.ok(first.slot.classList.contains(CURRENT));
  assert.equal(h.slots().length, 2);
});

/* --------------------------------------------------------------- finalize matrix */

test('finalize step 1: a reveal that fails and is verifiably undone reports known_old', async () => {
  for (const mode of ['throw', 'mutateThrow']) {
    const h = harness();
    const first = await ready(h, 'g1');
    h.renderer.finalize(first.handle);
    const second = await ready(h, 'g2');
    second.slot.rig('classList.add', mode);
    const result = h.renderer.finalize(second.handle);
    assert.equal(result.ok, false, mode);
    assert.equal(result.state, 'known_old', mode);
    // The old view is still the current one, and the newcomer is gone.
    assert.ok(first.slot.classList.contains(CURRENT), mode);
    assert.equal(h.slots().length, 1, mode);
    assert.equal(h.slots()[0], first.slot, mode);
  }
});

test('finalize step 1: a reveal that cannot be undone is terminal', async () => {
  const h = harness();
  const first = await ready(h, 'g1');
  h.renderer.finalize(first.handle);
  const second = await ready(h, 'g2');
  second.slot.rig('classList.add', 'mutateThrow');
  second.slot.rig('classList.remove', 'throw');
  const result = h.renderer.finalize(second.handle);
  assert.equal(result.state, 'inconsistent');
  assert.ok(h.renderer.inconsistent);
  // Nothing was removed: bridge state is unknown, so no rollback is claimed.
  assert.equal(h.slots().length, 2);
});

test('finalize step 3: one bounded retry is enough for a transient demotion failure', async () => {
  const h = harness();
  const first = await ready(h, 'g1');
  h.renderer.finalize(first.handle);
  const second = await ready(h, 'g2');
  first.slot.rig('setAttribute', 'throw');
  assert.deepEqual(h.renderer.finalize(second.handle), { ok: true });
  assert.ok(second.slot.classList.contains(CURRENT));
  assert.equal(h.slots().length, 1);
});

test('finalize step 3: two visible slots is terminal, not papered over', async () => {
  const h = harness();
  const first = await ready(h, 'g1');
  h.renderer.finalize(first.handle);
  const second = await ready(h, 'g2');
  first.slot.rig('classList.remove', 'throw', { times: 4 });
  const result = h.renderer.finalize(second.handle);
  assert.equal(result.state, 'inconsistent');
  // The newcomer IS presentation-current; the state commit already happened.
  assert.ok(second.slot.classList.contains(CURRENT));
  assert.equal(first.slot.callCount('classList.remove'), 2);
});

test('finalize step 4: a removal that fails is cleanup debt, not a failed finalize', async () => {
  const h = harness();
  const first = await ready(h, 'g1');
  h.renderer.finalize(first.handle);
  const second = await ready(h, 'g2');
  first.slot.rig('remove', 'throw');
  assert.deepEqual(h.renderer.finalize(second.handle), { ok: true });
  assert.ok(first.slot.isConnected);
  assert.equal(first.slot.callCount('remove'), 1);
  // The debt holds one quarantine slot, which is exactly what the budget counts.
  assert.equal(h.renderer.quarantined, 1);
});

test('finalize step 4: a removal that detaches and then throws is a clean release', async () => {
  const h = harness();
  const first = await ready(h, 'g1');
  h.renderer.finalize(first.handle);
  const second = await ready(h, 'g2');
  first.slot.rig('remove', 'mutateThrow');
  assert.deepEqual(h.renderer.finalize(second.handle), { ok: true });
  assert.equal(h.slots().length, 1);
  assert.equal(h.renderer.quarantined, 0);
});

test('cleanup that cannot be inspected is terminal', async () => {
  const h = harness();
  const { handle, slot } = await ready(h, 'g1');
  slot.rig('remove', 'throw');
  slot.rig('isConnected', 'throw');
  const result = h.renderer.abortReady(handle);
  assert.equal(result.state, 'inconsistent');
  assert.ok(h.renderer.inconsistent);
  assert.equal(slot.callCount('remove'), 1);
});

/* ------------------------------------------------------------------- cancel */

test('cancel settles the caller at once and leaves the element connected', async () => {
  const h = harness();
  const promise = h.renderer.install(session('g1'));
  const slot = h.last();
  h.renderer.cancel();
  await assert.rejects(promise, isKind('cancelled'));
  assert.ok(slot.isConnected);
  assert.equal(h.renderer.quarantined, 1);
  h.renderer.cancel();                      // idempotent
  assert.equal(h.renderer.quarantined, 1);
  // The late expected event is what authorises removal.
  h.emit(slot, 'g1');
  assert.equal(h.slots().length, 0);
  assert.equal(h.renderer.quarantined, 0);
});

test('a superseded candidate is quarantined and never revealed', async () => {
  const h = harness();
  const first = h.renderer.install(session('g1'));
  const firstSlot = h.last();
  const second = h.renderer.install(session('g2'));
  await assert.rejects(first, isKind('cancelled'));
  assert.equal(h.slots().length, 2);
  h.emit(firstSlot, 'g1');                  // stale: releases, reveals nothing
  assert.equal(h.slots().length, 1);
  assert.ok(!h.last().classList.contains(CURRENT));
  h.emit(h.last(), 'g2');
  h.renderer.finalize(await second);
  assert.ok(h.last().classList.contains(CURRENT));
});

test('cancelling a ready candidate consumes its handle as a tombstone', async () => {
  const h = harness();
  const { handle, slot } = await ready(h, 'g1');
  h.renderer.cancel();
  assert.equal(slot.isConnected, false);
  // The coordinator may still hold that handle with its continuation queued.
  assert.deepEqual(h.renderer.abortReady(handle), { ok: true, already: 'cancelled' });
  assert.deepEqual(h.renderer.abortReady(handle), { ok: true, already: 'cancelled' });
  // finalize is a different question: it can only be reached after a commit.
  assert.equal(h.renderer.finalize(handle).state, 'cancelled');
  assert.equal(h.renderer.finalize({}).state, 'misuse');
});

/* ------------------------------------------------------------------ handles */

test('handle misuse is contained and mutates nothing', async () => {
  const h = harness();
  const other = harness();
  const first = await ready(h, 'g1');
  const foreign = (await ready(other, 'gx')).handle;

  for (const bad of [null, undefined, {}, 'handle', 42, foreign]) {
    assert.equal(h.renderer.finalize(bad).state, 'misuse', String(bad));
    assert.equal(h.renderer.abortReady(bad).state, 'misuse', String(bad));
  }
  assert.equal(h.slots().length, 1);
  assert.ok(!first.slot.classList.contains(CURRENT));

  assert.deepEqual(h.renderer.finalize(first.handle), { ok: true });
  assert.equal(h.renderer.finalize(first.handle).state, 'misuse');
  assert.equal(h.renderer.abortReady(first.handle).state, 'misuse');

  const second = await ready(h, 'g2');
  assert.deepEqual(h.renderer.abortReady(second.handle), { ok: true });
  assert.equal(h.renderer.abortReady(second.handle).state, 'misuse');
  assert.equal(h.renderer.finalize(second.handle).state, 'misuse');
});

test('abortReady discards the candidate and keeps the current view', async () => {
  const h = harness();
  const first = await ready(h, 'g1');
  h.renderer.finalize(first.handle);
  const second = await ready(h, 'g2');
  assert.deepEqual(h.renderer.abortReady(second.handle), { ok: true });
  assert.equal(h.slots().length, 1);
  assert.equal(h.slots()[0], first.slot);
  assert.ok(first.slot.classList.contains(CURRENT));
});

test('abortReady reports cleanup debt without poisoning the renderer', async () => {
  const h = harness();
  const { handle, slot } = await ready(h, 'g1');
  slot.rig('remove', 'throw');
  const result = h.renderer.abortReady(handle);
  assert.equal(result.ok, false);
  assert.equal(result.state, 'cleanup_failed');
  assert.equal(h.renderer.inconsistent, null);
  assert.equal(h.renderer.quarantined, 1);
});

/* -------------------------------------------------------- listener retirement */

test('a ready candidate ignores later graph events', async () => {
  const h = harness();
  const { handle, slot } = await ready(h, 'g1', [{ graph: 'g1', name: 'metric', results: [] }]);
  assert.equal(h.element(slot).listenerCount('modelGraphProcessed'), 0);
  h.emit(slot, 'g1');                       // even if removeEventListener had not taken
  h.emit(slot, 'other');
  assert.equal(h.element(slot).nodeData.length, 1);
  assert.equal(h.element(slot).callCount('selectNode'), 0);
  assert.deepEqual(h.renderer.finalize(handle), { ok: true });
});

/* -------------------------------------------------------- capacity waiting */

/** Fills the quarantine to the ceiling, returning the retained slots. */
async function saturate(h, count) {
  const held = [];
  for (let index = 0; index < count; index += 1) {
    const promise = h.renderer.install(session(`q${index}`));
    held.push(h.last());
    h.renderer.cancel();
    await assert.rejects(promise, isKind('cancelled'));
  }
  assert.equal(h.renderer.quarantined, count);
  return held;
}

test('a candidate waits for capacity rather than exceeding the ceiling', async () => {
  const h = harness({ hardMaxQuarantined: 2 });
  const held = await saturate(h, 2);
  const waiting = h.renderer.install(session('g1'));
  // Parked: no element exists for it yet.
  assert.equal(h.slots().length, 2);
  // A safe release admits the waiter.
  h.emit(held[0], 'q0');
  await Promise.resolve();
  assert.equal(h.slots().length, 2);
  h.emit(h.last(), 'g1');
  assert.ok(await waiting);
});

test('a parked candidate that times out reports exhausted and removes nothing', async () => {
  const h = harness({ hardMaxQuarantined: 1 });
  await saturate(h, 1);
  const waiting = h.renderer.install(session('g1'));
  h.clock.advance(10_000);
  await assert.rejects(waiting, isKind('exhausted'));
  assert.equal(h.slots().length, 1);
  assert.equal(h.clock.pending, 0);
});

test('a cancelled or superseded waiter never connects an element', async () => {
  const h = harness({ hardMaxQuarantined: 1 });
  const held = await saturate(h, 1);

  const cancelled = h.renderer.install(session('g1'));
  assert.equal(h.clock.pending, 1);                 // its capacity timer
  h.renderer.cancel();
  await assert.rejects(cancelled, isKind('cancelled'));
  assert.equal(h.slots().length, 1);
  // The wait is settled, not merely abandoned: nothing is left to fire later.
  assert.equal(h.clock.pending, 0);

  const superseded = h.renderer.install(session('g2'));
  const newcomer = h.renderer.install(session('g3'));
  await assert.rejects(superseded, isKind('cancelled'));
  assert.equal(h.slots().length, 1);

  // Only the surviving waiter connects when room appears.
  h.emit(held[0], 'q0');
  await Promise.resolve();
  assert.equal(h.slots().length, 1);
  h.emit(h.last(), 'g3');
  assert.ok(await newcomer);
});

test('markInconsistent settles a caller parked for capacity', async () => {
  const h = harness({ hardMaxQuarantined: 1 });
  await saturate(h, 1);
  const waiting = h.renderer.install(session('g1'));
  h.renderer.markInconsistent('bridge state unknown');
  await assert.rejects(waiting, isKind('inconsistent'));
  assert.equal(h.slots().length, 1);
  assert.equal(h.clock.pending, 0);
});

/* ------------------------------------------------------------ cleanup debt */

/** Leaves one un-removable safe element behind, holding a quarantine slot. */
async function withDebt(h) {
  const { handle, slot } = await ready(h, 'debt');
  slot.rig('remove', 'throw', { times: 99 });
  assert.equal(h.renderer.abortReady(handle).state, 'cleanup_failed');
  assert.equal(h.renderer.quarantined, 1);
  return slot;
}

test('cleanup debt is retried once per install and by nothing else', async () => {
  const h = harness({ hardMaxQuarantined: 2 });
  const debt = await withDebt(h);
  assert.equal(debt.callCount('remove'), 1);

  // An idle renderer, a bare cancel, and a supersession schedule no retry.
  h.clock.advance(600_000);
  h.renderer.cancel();
  assert.equal(debt.callCount('remove'), 1);

  const first = h.renderer.install(session('g1'));   // preflight sweep: attempt 2
  assert.equal(debt.callCount('remove'), 2);
  h.renderer.install(session('g2')).catch(() => {}); // supersedes, and sweeps: attempt 3
  await assert.rejects(first, isKind('cancelled'));
  assert.equal(debt.callCount('remove'), 3);
});

test('debt beyond the attempt budget is abandoned, not retried forever', async () => {
  const h = harness({ hardMaxQuarantined: 2 });
  const debt = await withDebt(h);
  for (let index = 0; index < 4; index += 1) {
    const promise = h.renderer.install(session(`g${index}`));
    h.emit(h.last(), `g${index}`);
    h.renderer.finalize(await promise);
  }
  assert.equal(debt.callCount('remove'), 3);         // MAX_DETACH_ATTEMPTS
  assert.ok(debt.isConnected);
  assert.equal(h.renderer.quarantined, 1);           // it still costs one slot
  assert.equal(h.renderer.inconsistent, null);       // the DOM is coherent
});

test('abandoned debt lowers capacity until installs are refused', async () => {
  const h = harness({ hardMaxQuarantined: 1 });
  const debt = await withDebt(h);
  for (let index = 0; index < 3; index += 1) {
    await assert.rejects(
      (() => { const p = h.renderer.install(session(`g${index}`)); h.clock.advance(10_000); return p; })(),
      isKind('exhausted'),
    );
  }
  assert.equal(debt.callCount('remove'), 3);
  await assert.rejects(
    (() => { const p = h.renderer.install(session('gx')); h.clock.advance(10_000); return p; })(),
    isKind('exhausted'),
  );
  assert.equal(debt.callCount('remove'), 3);         // abandoned: no further attempt
});

test('a debt sweep that cannot inspect its own removal is terminal', async () => {
  const h = harness({ hardMaxQuarantined: 2 });
  const debt = await withDebt(h);
  debt.rig('isConnected', 'throw', { times: 99 });
  await assert.rejects(h.renderer.install(session('g1')), isKind('inconsistent'));
  assert.equal(h.slots().length, 1);                 // no element for the newcomer
  assert.ok(h.renderer.inconsistent);
});

/* --------------------------------------------------------- injected failures */

test('a node-data failure settles the caller and releases the safe candidate', async () => {
  const h = harness();
  const promise = h.renderer.install(session('g1', [{ graph: 'g1', name: 'metric', results: [] }]));
  const slot = h.last();
  h.element(slot).rig('addNodeDataProviderData', 'throw');
  h.emit(slot, 'g1');
  await assert.rejects(promise, /rigged/);
  assert.equal(h.slots().length, 0);        // the expected event arrived, so removal is safe
});

test('a selectNode failure quarantines an active candidate rather than removing it', async () => {
  const h = harness();
  const promise = h.renderer.install(session('g1'));
  const slot = h.last();
  h.element(slot).rig('selectNode', 'throw');
  h.emit(slot, 'other');
  await assert.rejects(promise, /rigged/);
  assert.ok(slot.isConnected);
  assert.equal(h.renderer.quarantined, 1);
});

test('a cancelled candidate whose selectNode throws produces no second settlement', async () => {
  const h = harness();
  const promise = h.renderer.install(session('g1'));
  const slot = h.last();
  h.renderer.cancel();
  const outcome = await settled(promise);
  h.element(slot).rig('selectNode', 'throw');
  h.emit(slot, 'other');                    // swallowed: already settled, not authoritative
  assert.equal(outcome.status, 'rejected');
  assert.equal(outcome.error.kind, 'cancelled');
  assert.ok(slot.isConnected);
  assert.equal(h.renderer.inconsistent, null);
});

test('the heartbeat stops when the candidate settles', async () => {
  const h = harness();
  const promise = h.renderer.install(session('g1'));
  h.clock.advance(15_000);
  assert.ok(h.element(h.last()));
  h.emit(h.last(), 'g1');
  await promise;
  assert.equal(h.clock.pending, 0);
});

/* -------------------------------------------------------- markInconsistent */

test('markInconsistent settles a connected caller and freezes the renderer', async () => {
  const h = harness();
  const promise = h.renderer.install(session('g1'));
  const slot = h.last();
  assert.deepEqual(h.renderer.markInconsistent('bridge state unknown'), { ok: true });
  await assert.rejects(promise, isKind('inconsistent'));
  // Nothing is removed and every timer is off.
  assert.equal(h.slots().length, 1);
  assert.equal(h.clock.pending, 0);
  assert.deepEqual(h.renderer.markInconsistent('again'), { ok: true });
  assert.match(h.renderer.inconsistent, /bridge state unknown/);

  await assert.rejects(h.renderer.install(session('g2')), isKind('inconsistent'));
  assert.equal(h.slots().length, 1);
  // A late event mutates nothing.
  h.emit(slot, 'g1');
  assert.equal(h.slots().length, 1);
});

test('markInconsistent settles a ready caller and consumes its handle', async () => {
  const h = harness();
  const { handle, slot } = await ready(h, 'g1');
  h.renderer.markInconsistent('bridge state unknown');
  assert.equal(h.renderer.finalize(handle).state, 'inconsistent');
  assert.equal(h.renderer.abortReady(handle).state, 'inconsistent');
  assert.ok(slot.isConnected);
  assert.ok(!slot.classList.contains(CURRENT));
});

test('markInconsistent with no active caller is still terminal', () => {
  const h = harness();
  assert.deepEqual(h.renderer.markInconsistent('nothing in flight'), { ok: true });
  assert.match(h.renderer.inconsistent, /nothing in flight/);
});

test('supersession that discovers inconsistency orphans no caller', async () => {
  const h = harness();
  const first = await ready(h, 'g1');
  first.slot.rig('remove', 'throw');
  first.slot.rig('isConnected', 'throw');
  await assert.rejects(h.renderer.install(session('g2')), isKind('inconsistent'));
  // No entry, no deferred and no element were created for the newcomer.
  assert.equal(h.slots().length, 1);
  assert.ok(h.renderer.inconsistent);
});

/* ---------------------------------------------------------------- selection */

/* A session with kinds, which the fixture above deliberately lacks: the legacy
 * no-descriptor path applies no kind filter, so only these cases can exercise
 * the stage-only rule. `flow` and `compare` views are hand-authored here
 * because the exporter emits neither today -- which is exactly why the guard
 * lives in the renderer rather than in a caller that might forget it. */
const kinded = ({ defaultView = 'v/canonical', views, nodeDataSets = [] }) => JSON.stringify({
  model: { name: 'm' },
  defaultView,
  views,
  graphCollections: [{ label: 'c', graphs: [
    { id: 'pt2/root', nodes: [{ id: 's0' }] },
    { id: 'g/native/001', nodes: [{ id: 'c0' }] },
    { id: 'g/flow', nodes: [{ id: 'f0' }] },
  ] }],
  nodeDataSets,
});

const STAGE_VIEWS = [
  { id: 'v/canonical', label: 'Canonical', kind: 'stage:canonical', collection: 'c', graph: 'g/native/001' },
  { id: 'v/source', label: 'Source', kind: 'stage:source', collection: 'c', graph: 'pt2/root' },
];
const FLOW_VIEW = { id: 'v/flow', label: 'Flow', kind: 'flow', collection: 'c', graph: 'g/flow' };

async function enter(h, text, selection, graphId) {
  const promise = h.renderer.install(text, selection);
  h.emit(h.last(), graphId);
  return promise;
}

test('an exact selection enters that view and reports it', async () => {
  const h = harness();
  const handle = await enter(h, kinded({ views: STAGE_VIEWS }), { view: 'v/source' }, 'pt2/root');
  assert.equal(handle.graph, 'pt2/root');
  assert.equal(handle.viewId, 'v/source');
});

test('an unknown exact selection fails before any element is connected', async () => {
  const h = harness();
  await assert.rejects(
    h.renderer.install(kinded({ views: STAGE_VIEWS }), { view: 'v/nope' }),
    isKind('invalid'));
  assert.equal(h.slots().length, 0);
});

/* The delivery boundary, enforced where it cannot be bypassed. */
test('a non-stage view cannot be selected exactly', async () => {
  const h = harness();
  await assert.rejects(
    h.renderer.install(kinded({ views: [...STAGE_VIEWS, FLOW_VIEW] }), { view: 'v/flow' }),
    isKind('invalid'));
  assert.equal(h.slots().length, 0);
});

test('a preference skips a non-stage id and lands on the next stage view', async () => {
  const h = harness();
  const handle = await enter(h, kinded({ views: [...STAGE_VIEWS, FLOW_VIEW] }),
    { prefer: ['v/flow', 'v/source'] }, 'pt2/root');
  assert.equal(handle.viewId, 'v/source');
});

/* `Session.validate` checks `defaultView` with a membership test alone and says
 * nothing about its kind, so a bridge-valid session may default to a flow view.
 * A chain that filtered only the requested ids would re-open it from here. */
test('a non-stage defaultView is skipped too', async () => {
  const h = harness();
  const handle = await enter(h,
    kinded({ defaultView: 'v/flow', views: [...STAGE_VIEWS, FLOW_VIEW] }),
    { prefer: ['v/missing'] }, 'g/native/001');
  assert.equal(handle.viewId, 'v/canonical');
});

test('a preference falls back through defaultView', async () => {
  const h = harness();
  const handle = await enter(h, kinded({ views: STAGE_VIEWS }), { prefer: ['v/missing'] }, 'g/native/001');
  assert.equal(handle.viewId, 'v/canonical');
});

test('a session with no stage view at all is refused rather than rendered', async () => {
  const h = harness();
  await assert.rejects(
    h.renderer.install(kinded({ defaultView: 'v/flow', views: [FLOW_VIEW] }), { prefer: ['v/source'] }),
    isKind('invalid'));
  assert.equal(h.slots().length, 0);
});

/* ---------------------------------------------------------------- node data */

const verificationSet = (graph) => ({
  name: 'verification', graph,
  results: [
    { nodeId: 'c0', value: { value: 1, label: 'proved (structural) [sampled 4]' } },
    { nodeId: 'c1', value: { value: 6, label: 'refuted (counterexample)' } },
    { nodeId: 'c2', value: { value: 42, label: 'from the future' } },
  ],
});

/* The element renders `strValue` from `value` and reads no `label` key at all,
 * so the supplied label can only reach the screen AS the value. A per-result
 * `bgColor` overrides any gradient, which is what makes the buckets named
 * rather than a position on an anonymous ramp. */
test('verification data carries the verbatim label and a bucket colour', async () => {
  const h = harness();
  await enter(h, kinded({ views: STAGE_VIEWS, nodeDataSets: [verificationSet('g/native/001')] }),
    { view: 'v/canonical' }, 'g/native/001');
  const [installed] = h.element(h.last()).nodeData;
  assert.equal(installed.name, 'verification');
  assert.equal(installed.data.gradient, undefined);
  assert.equal(installed.data.results.c0.value, 'proved (structural) [sampled 4]');
  assert.ok(installed.data.results.c0.bgColor);
  assert.notEqual(installed.data.results.c1.bgColor, installed.data.results.c0.bgColor);
  // A rank outside the known scale keeps its label and takes no colour rather
  // than borrowing a neighbouring bucket's meaning.
  assert.equal(installed.data.results.c2.value, 'from the future');
  assert.equal(installed.data.results.c2.bgColor, undefined);
});

/* The wire shape is `{ nodeId, value }`, an object -- reading it as a pair
 * throws "object is not iterable". Nothing noticed for a long time because the
 * only set addressed to the default view is the verification one and the
 * browser never asked for verification. */
test('a non-verification set keeps its gradient and its numeric value', async () => {
  const h = harness();
  const fusion = { name: 'fusion', graph: 'g/native/001', results: [{ nodeId: 'c0', value: { value: 3 } }] };
  await enter(h, kinded({ views: STAGE_VIEWS, nodeDataSets: [fusion] }), { view: 'v/canonical' }, 'g/native/001');
  const [installed] = h.element(h.last()).nodeData;
  assert.equal(installed.data.results.c0.value, 3);
  assert.ok(Array.isArray(installed.data.gradient));
});

test('only the selected graph’s node data is installed', async () => {
  const h = harness();
  const sets = [verificationSet('g/native/001'), { name: 'fusion', graph: 'pt2/root', results: [] }];
  await enter(h, kinded({ views: STAGE_VIEWS, nodeDataSets: sets }), { view: 'v/canonical' }, 'g/native/001');
  assert.deepEqual(h.element(h.last()).nodeData.map((d) => d.name), ['verification']);
});

/* ================================================================ comparisons

   A comparison is a two-pane presentation resolved entirely before a slot
   exists, then driven through TWO processing rounds. What follows pins both
   halves: the refusals that must never mutate the DOM, and the transitions of
   the two-phase candidate -- above all the `safe` invariant, which decides
   whether a cancelled entry costs a quarantine slot or none. */

const ENTRIES = [
  { left: ['l0'], right: ['r0', 'r1'] },              // 1:N
  { left: ['l1', 'l2'], right: ['r2', 'r3'] },        // N:M
];

const compared = ({
  sync = { entries: ENTRIES },
  left = { collection: 'cL', graph: 'gL' },
  right = { collection: 'cR', graph: 'gR' },
  comparisons,
  collections,
  nodeDataSets = [],
  extra = {},
} = {}) => JSON.stringify({
  model: { name: 'm' },
  defaultView: 'v/l',
  views: [
    { id: 'v/l', label: 'Left', kind: 'stage:source', collection: 'cL', graph: 'gL' },
    { id: 'v/r', label: 'Right', kind: 'stage:canonical', collection: 'cR', graph: 'gR' },
  ],
  graphCollections: collections ?? [
    { label: 'cL', graphs: [{ id: 'gL', nodes: [{ id: 'l0' }] }] },
    { label: 'cR', graphs: [{ id: 'gR', nodes: [{ id: 'r0' }] }] },
  ],
  comparisons: comparisons ?? [{ id: 'c/x', label: 'left → right', left, right, sync, ...extra }],
  nodeDataSets,
});

/** Installs a comparison and drives it to `ready` through both pane events. */
async function readyComparison(h, text = compared(), id = 'c/x') {
  const promise = h.renderer.install(text, { comparison: id });
  const slot = h.last();
  h.element(slot).emitProcessed('gL', 0);
  h.element(slot).emitProcessed('gR', 1);
  return { slot, handle: await promise };
}

const refuses = async (h, text, selection = { comparison: 'c/x' }) => {
  await assert.rejects(h.renderer.install(text, selection), isKind('invalid'));
  assert.equal(h.slots().length, 0, 'a refusal must not leave a slot behind');
};

/* ------------------------------------------------------------- resolution */

test('a comparison that does not resolve is refused before any DOM exists', async () => {
  const h = harness();
  await refuses(h, compared(), { comparison: 'c/nope' });
  await refuses(h, compared({ comparisons: [
    { id: 'c/x', label: 'a', left: { collection: 'cL', graph: 'gL' }, right: { collection: 'cR', graph: 'gR' }, sync: {} },
    { id: 'c/x', label: 'b', left: { collection: 'cL', graph: 'gL' }, right: { collection: 'cR', graph: 'gR' }, sync: {} },
  ] }));
  // A graph that exists, but not in the collection the pane names: exactly
  // `Session.validate`'s `Wrong_collection`, and a lookup that "found" it would
  // open a pane the document never declared.
  await refuses(h, compared({ right: { collection: 'cL', graph: 'gR' } }));
  await refuses(h, compared({ left: { collection: 'nope', graph: 'gL' } }));
  // A graph with no node cannot be navigated to: `selectNode` is the only way
  // into a pane and it takes a node.
  await refuses(h, compared({ collections: [
    { label: 'cL', graphs: [{ id: 'gL', nodes: [] }] },
    { label: 'cR', graphs: [{ id: 'gR', nodes: [{ id: 'r0' }] }] },
  ] }));
  await refuses(h, compared({ sync: { entries: [{ left: ['l0'], right: 'r0' }] } }));
  await refuses(h, compared({ extra: { overlaysLeft: 'not a list' } }));
});

test('a selection naming both a view and a comparison is refused', async () => {
  const h = harness();
  await refuses(h, compared(), { comparison: 'c/x', view: 'v/l' });
});

/* --------------------------------------------------------------- the config */

const configOf = (h, slot) => h.element(slot).config;

test('mapping entries cross as arrays, never as a Cartesian product', async () => {
  const h = harness();
  const { slot } = await readyComparison(h);
  assert.deepEqual(configOf(h, slot).syncNavigationData.mappingEntries, [
    { leftNodeIds: ['l0'], rightNodeIds: ['r0', 'r1'] },
    { leftNodeIds: ['l1', 'l2'], rightNodeIds: ['r2', 'r3'] },
  ]);
  assert.equal(configOf(h, slot).syncNavigationData.type, 'sync_navigation');
});

test('an empty exported mapping produces no synthetic entry', async () => {
  const h = harness();
  const { slot } = await readyComparison(h, compared({ sync: { entries: [], matchNodeIdFallback: true } }));
  assert.deepEqual(configOf(h, slot).syncNavigationData.mappingEntries, []);
});

/* `matchNodeIdFallback` is the wire field that says whether equal ids are a
 * CORRESPONDENCE CLAIM. It is translated, never inferred: with the fallback off
 * and no entries, `renderDiffHighlights` treats every node's empty mapped set as
 * "all counterparts missing" and outlines the whole of both panes. */
test('the navigation fallback is translated from the wire in both directions', async () => {
  const h = harness();
  const off = await readyComparison(h, compared({ sync: { entries: [], matchNodeIdFallback: true } }));
  assert.equal(configOf(h, off.slot).syncNavigationData.disableMappingFallback, false);
  h.renderer.finalize(off.handle);

  const on = await readyComparison(h, compared({ sync: { entries: [], matchNodeIdFallback: false } }));
  assert.equal(configOf(h, on.slot).syncNavigationData.disableMappingFallback, true);
  h.renderer.finalize(on.handle);

  // Absent reads as `false`, which is the conservative half.
  const absent = await readyComparison(h, compared({ sync: { entries: [] } }));
  assert.equal(configOf(h, absent.slot).syncNavigationData.disableMappingFallback, true);
});

test('showDiffHighlights is present only when the session declares it', async () => {
  const h = harness();
  const off = await readyComparison(h);
  assert.ok(!('showDiffHighlights' in configOf(h, off.slot).syncNavigationData));
  h.renderer.finalize(off.handle);

  const on = await readyComparison(h,
    compared({ sync: { entries: ENTRIES, showDiffHighlights: true } }));
  assert.equal(configOf(h, on.slot).syncNavigationData.showDiffHighlights, true);
});

/* `Comparison.overlaysLeft` is a bare `EdgeOverlay[]`; the config field takes
 * `EdgeOverlaysData[]`. The wrapper is the adapter's, and each pane gets only
 * its own list. */
test('overlays are wrapped, side-correct, and omitted when empty', async () => {
  const h = harness();
  const bare = await readyComparison(h);
  assert.ok(!('edgeOverlaysDataListLeftPane' in configOf(h, bare.slot)));
  assert.ok(!('edgeOverlaysDataListRightPane' in configOf(h, bare.slot)));
  h.renderer.finalize(bare.handle);

  const overlay = { id: 'o', name: 'fusion', edges: [] };
  const { slot } = await readyComparison(h, compared({ extra: { overlaysRight: [overlay] } }));
  assert.ok(!('edgeOverlaysDataListLeftPane' in configOf(h, slot)));
  assert.deepEqual(configOf(h, slot).edgeOverlaysDataListRightPane, [
    { type: 'edge_overlays', name: 'left → right', graphName: 'gR', overlays: [overlay] },
  ]);
});

/* ------------------------------------------------------------ the two phases */

test('the second pane is requested only once the first has processed', async () => {
  const h = harness();
  const promise = h.renderer.install(compared(), { comparison: 'c/x' });
  const element = h.element(h.last());
  assert.deepEqual(element.selections, [], 'nothing is navigated before an event');

  element.emitProcessed('gL', 0);
  assert.deepEqual(element.selections, [{ nodeId: 'r0', graphId: 'gR', label: 'cR', paneIndex: 1 }]);

  element.emitProcessed('gR', 1);
  const handle = await promise;
  assert.deepEqual(handle.graph, { left: 'gL', right: 'gR' });
  assert.equal(handle.viewId, null);
  assert.deepEqual(handle.presentation, { kind: 'comparison', comparison: 'c/x' });
});

test('a mismatched auto-selection is corrected on pane 0, naming the pane', async () => {
  const h = harness();
  const promise = h.renderer.install(compared(), { comparison: 'c/x' });
  const element = h.element(h.last());
  element.emitProcessed('gOther', 0);
  assert.deepEqual(element.selections, [{ nodeId: 'l0', graphId: 'gL', label: 'cL', paneIndex: 0 }]);
  element.emitProcessed('gL', 0);
  element.emitProcessed('gR', 1);
  await promise;
});

/* The pane is part of an event's IDENTITY. `Session.validate` permits both
 * panes to name the same graph, and opening the second pane re-lays-out the
 * first -- so a wait keyed on the graph id alone is satisfied by the wrong
 * pane, and the candidate is finalized with a pane that never processed.
 *
 * Deleting the `pane !== target.pane` term in `#onProcessed` makes this test
 * fail: the stray pane-0 event resolves the promise with the right pane never
 * having been laid out. */
test('a pane-0 re-layout does not satisfy the wait for pane 1', async () => {
  const h = harness();
  const same = compared({ right: { collection: 'cL', graph: 'gL' } });
  const promise = h.renderer.install(same, { comparison: 'c/x' });
  const element = h.element(h.last());
  element.emitProcessed('gL', 0);
  assert.equal(element.selections.length, 1);

  let settledEarly = false;
  promise.then(() => { settledEarly = true; }, () => { settledEarly = true; });
  element.emitProcessed('gL', 0);                 // the re-layout of pane 0
  // `install` is an async function, so its promise ADOPTS the entry's rather
  // than being it: a single microtask tick is not enough to observe a
  // settlement, and a check that flushed once would pass without the filter.
  await flush();
  assert.equal(settledEarly, false, 'the pane-1 wait was satisfied by a pane-0 event');

  element.emitProcessed('gL', 1);
  assert.ok(await promise);
});

test('node data is installed in the pane whose graph owns it', async () => {
  const h = harness();
  const sets = [
    { graph: 'gL', name: 'left-metric', results: [{ nodeId: 'l0', value: { value: 1 } }] },
    { graph: 'gR', name: 'right-metric', results: [{ nodeId: 'r0', value: { value: 2 } }] },
    { graph: 'gNeither', name: 'ignored', results: [] },
  ];
  const { slot } = await readyComparison(h, compared({ nodeDataSets: sets }));
  assert.deepEqual(h.element(slot).nodeData.map((d) => ({ name: d.name, paneIndex: d.paneIndex })), [
    { name: 'left-metric', paneIndex: 0 },
    { name: 'right-metric', paneIndex: 1 },
  ]);
});

/* ------------------------------------------------------- cancellation windows

   `entry.safe` means "no processing this renderer asked for is outstanding".
   It is true at the pane-0 event and FALSE AGAIN the instant the pane-1
   `selectNode` is issued, and those two windows must behave differently. */

test('cancelling before the first pane event quarantines, and the late event releases it', async () => {
  const h = harness();
  const promise = h.renderer.install(compared(), { comparison: 'c/x' });
  const slot = h.last();
  const element = h.element(slot);
  h.renderer.cancel();
  assert.equal((await settled(promise)).status, 'rejected');
  assert.equal(h.renderer.quarantined, 1, 'an element mid-processing cannot be removed');

  element.emitProcessed('gL', 0);
  // Cancelled: the second pane is never asked for, because asking would make
  // the entry unsafe again for an event nobody is waiting for.
  assert.deepEqual(element.selections, []);
  assert.equal(h.renderer.quarantined, 0);
  assert.ok(!slot.isConnected);
});

test('cancelling between the two pane events quarantines until the second arrives', async () => {
  const h = harness();
  const promise = h.renderer.install(compared(), { comparison: 'c/x' });
  const slot = h.last();
  const element = h.element(slot);
  element.emitProcessed('gL', 0);
  assert.equal(element.selections.length, 1, 'pane 1 was requested, so the entry is unsafe again');

  h.renderer.cancel();
  assert.equal((await settled(promise)).status, 'rejected');
  assert.equal(h.renderer.quarantined, 1);
  assert.ok(slot.isConnected, 'removing an element still processing throws NG0953');

  element.emitProcessed('gR', 1);
  assert.equal(h.renderer.quarantined, 0);
  assert.ok(!slot.isConnected);
});

/* The vacuity check the rule above earns: if `entry.safe` were left TRUE when
 * the pane-1 request is issued, the cancel in the previous test would take the
 * `#release` branch and assert its way out, or remove an element that is still
 * processing. Removing `entry.safe = false` from `#onProcessed` makes that test
 * fail with a quarantine count of 0 and a detached slot. This one states the
 * invariant directly. */
test('a comparison is unsafe again while the second pane is outstanding', async () => {
  const h = harness();
  const promise = h.renderer.install(compared(), { comparison: 'c/x' });
  const element = h.element(h.last());
  element.emitProcessed('gL', 0);
  h.renderer.cancel();
  await settled(promise);
  // Safe would mean "released"; unsafe means "quarantined".
  assert.equal(h.renderer.quarantined, 1);
});

test('no partial comparison is ever finalized', async () => {
  const h = harness();
  const first = await ready(h, 'g1');
  assert.equal(h.renderer.finalize(first.handle).ok, true);

  const promise = h.renderer.install(compared(), { comparison: 'c/x' });
  h.element(h.last()).emitProcessed('gL', 0);
  // The left pane is up, but the candidate is not ready and there is no handle
  // to finalize -- the previous view is still the only visible slot.
  const current = h.slots().filter((s) => s.classList.contains(CURRENT));
  assert.deepEqual(current, [first.slot]);
  h.element(h.last()).emitProcessed('gR', 1);
  assert.equal(h.renderer.finalize(await promise).ok, true);
});

/* Each pane is its own layout round, so each gets its own budget rather than
 * the remainder of the first one's. */
test('the processing deadline restarts for the second pane', async () => {
  const h = harness();
  const promise = h.renderer.install(compared(), { comparison: 'c/x' });
  const element = h.element(h.last());
  h.clock.advance(h.renderer.timeout - 1);
  element.emitProcessed('gL', 0);
  h.clock.advance(h.renderer.timeout - 1);            // past the FIRST deadline
  element.emitProcessed('gR', 1);
  assert.ok(await promise);
});

test('a stall in either phase times out and quarantines', async () => {
  for (const phase of ['left', 'right']) {
    const h = harness();
    const promise = h.renderer.install(compared(), { comparison: 'c/x' });
    if (phase === 'right') h.element(h.last()).emitProcessed('gL', 0);
    h.clock.advance(h.renderer.timeout + 1);
    const outcome = await settled(promise);
    assert.equal(outcome.status, 'rejected');
    assert.equal(outcome.error.kind, 'timeout');
    assert.equal(h.renderer.quarantined, 1);
  }
});

test('a comparison superseded by a newer request settles as cancelled', async () => {
  const h = harness();
  const first = h.renderer.install(compared(), { comparison: 'c/x' });
  const firstElement = h.element(h.last());
  const second = h.renderer.install(compared(), { comparison: 'c/x' });
  const outcome = await settled(first);
  assert.equal(outcome.status, 'rejected');
  assert.equal(outcome.error.kind, 'cancelled');
  firstElement.emitProcessed('gL', 0);                // releases the superseded one
  assert.equal(h.renderer.quarantined, 0);
  const element = h.element(h.last());
  element.emitProcessed('gL', 0);
  element.emitProcessed('gR', 1);
  assert.ok(await second);
});
