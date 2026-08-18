import { test, expect } from '@playwright/test';

/* The Phase 4 selected-node gate (`web-ui-4.md`, "Browser integration
 * prerequisite").
 *
 * The flow adapter routes a node selection to a declared stage view or
 * comparison, so it must tell a user's click from the shell's own `selectNode`
 * during setup. `NodeInfo` carries no origin bit, so identity and ORDER are the
 * only things that can separate them -- and three designs for that suppression
 * were wrong before this fixture existed:
 *
 *   1. a barrier armed at finalize on the setup node id: unsound, because a
 *      setup event delivered BEFORE finalize leaves it armed, so
 *      `setup A -> user B -> user A` discards the user's A;
 *   2. a COUNT of issued `selectNode` calls: unsound, because one wrapper call
 *      emits twice -- `selectNode(paneId, undefined)` clears first, then
 *      `revealNode` selects the target -- so the count is spent on the clear
 *      and the target escapes;
 *   3. anything that assumes a fixed sequence without measuring it.
 *
 * So this suite ENUMERATES the setup sequence rather than counting it, and the
 * adapter may use only what is proven here. If an expectation fails, the answer
 * is to change the design, not to soften the assertion.
 *
 * What it does NOT claim. A real pointer click into a WebGL canvas inside a
 * shadow root is not synthesizable, so user selections are driven through the
 * element's public `selectNode`. That is sound because `AppService.selectNode`
 * is the single path BOTH reach -- the events are literally the same events --
 * and it is the reason an origin bit does not exist to be read.
 */

type Selection = {
  seq: number; phase: string;
  nodeId: string | null; graphId: string | null; collectionLabel: string | null;
};

test('the selected-node event is identifiable, ordered, and scoped', async ({ page }) => {
  const pageErrors: string[] = [];
  const consoleErrors: string[] = [];
  const missing: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(String(error)));
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
  page.on('response', (r) => { if (r.status() === 404) missing.push(new URL(r.url()).pathname); });

  await page.goto('/flow.html');
  await page.waitForFunction(() => (window as any).__flow?.steps?.done !== undefined, null,
    { timeout: 240_000 });

  const observed = await page.evaluate(() => (window as any).__flow);
  const s = observed.steps;
  console.log('--- selected-node contract ---');
  console.log(JSON.stringify(s, null, 2));
  if (observed.errors?.length) console.log('driver errors:\n' + observed.errors.join('\n'));
  if (consoleErrors.length) console.log('console errors:\n' + consoleErrors.slice(0, 5).join('\n'));

  expect(observed.errors, 'the flow driver threw').toEqual([]);
  expect(s.done, 'the driver did not finish').toBe(true);

  /* 1. A flow graph is an ordinary single-pane candidate. */
  expect(s.processed, 'the flow graph never processed').toBe(true);

  /* 2 + 3. Setup emits, and what it emits is enumerable. This is the
   * measurement the whole design rests on: the adapter models THIS sequence. */
  const setup: Selection[] = s.setupSelections;
  expect(setup.length, 'setup produced no selected-node event at all').toBeGreaterThan(0);
  expect(s.setupShape.allOnFlowGraph,
    'a setup event named another graph or collection, so scoping cannot work').toBe(true);

  /* The target the shell asked for is present and identifiable BY ID. Without
   * this the adapter has nothing to suppress on, and Stage 5's design is dead. */
  expect(s.setupShape.matchingRequested,
    `no setup event named the requested node ${s.setupRequested}`).toBeGreaterThan(0);

  /* And the clear is separable from it. An empty `nodeId` is never a valid
   * node, which is what lets the adapter drop it unconditionally rather than
   * spending a suppression budget on it -- the defect that sank design 2. */
  expect(setup.every((e) => e.nodeId !== null),
    'a setup event carried no nodeId field at all').toBe(true);
  const clears = setup.filter((e) => e.nodeId === '');
  const targets = setup.filter((e) => e.nodeId === s.setupRequested);
  console.log(`setup: ${setup.length} events, ${clears.length} clear(s), ${targets.length} target(s)`);
  expect(clears.length + targets.length,
    'setup emitted an event that is neither a clear nor the requested target').toBe(setup.length);

  /* 4. Ordering: every clear precedes the target it belongs to. The adapter
   * relies on the target being the LAST thing setup does, so that a later event
   * is a user's. */
  if (clears.length > 0 && targets.length > 0) {
    expect(Math.max(...clears.map((e) => e.seq)),
      'a clear arrived after the target, so setup is not finished at the target')
      .toBeLessThan(Math.max(...targets.map((e) => e.seq)));
  }

  /* 5. A -> B -> A. The sequence that killed design 1: the second A must arrive
   * as its own event, or a stale expectation would swallow a real selection. */
  expect(s.reselectRoutes.sawB, 'selecting the transition emitted nothing').toBe(true);
  expect(s.reselectRoutes.sawAAgain,
    're-selecting the first state emitted nothing, so A->B->A cannot be routed').toBe(true);

  /* 6. A state and a transition are distinguishable only by declared id. */
  expect(s.identity.labelsDistinguish,
    'the fixture must not let labels stand in for declared ids').toBe(false);
  expect(s.identity.selectedAreKnown,
    'a selection named an id in neither declared list').toBe(true);

  /* A foreign graph's selection is refusable on the event's own public fields,
   * with nothing private inspected. */
  expect(s.foreignGraph.allNameOtherGraph,
    'a foreign selection did not identify its own graph').toBe(true);

  /* 7. The element keeps emitting for a candidate the app has moved on from --
   * which is precisely WHY the renderer must remove the listener rather than
   * rely on the element going quiet. */
  expect(s.lateEventStillEmitted,
    'a replaced candidate went quiet on its own; the removal rule needs re-deriving').toBe(true);
  expect(s.listenerRemovalStops,
    'removeEventListener did not stop delivery').toBe(0);

  /* 8. Residual case, measured rather than assumed: does loading a flow graph
   * auto-select anything the renderer never asked for? Recorded either way --
   * under select-then-act it costs a stale action row, never a navigation. */
  const auto: Selection[] = s.autoSelectOnLoad;
  console.log(`auto-select on load: ${auto.length} event(s)` +
    (auto.length ? ` -> ${JSON.stringify(auto.map((e) => e.nodeId))}` : ''));

  expect(pageErrors, 'the page threw').toEqual([]);
  expect(pageErrors.join('\n')).not.toContain('NG0953');
  /* Only the bundle's own assets. `/api/v1/get_extensions` 404s against any
   * static server -- the element probes for an extension host that does not
   * exist here -- and asserting on it would pin a third party's optional
   * request as a failure. A missing worker or static file would break the
   * element outright, so that is what this checks. */
  if (missing.length) console.log('404s:\n' + [...new Set(missing)].sort().join('\n'));
  expect(missing.filter((p) => p.startsWith('/vendor/')), 'bundle assets 404ed').toEqual([]);
});
