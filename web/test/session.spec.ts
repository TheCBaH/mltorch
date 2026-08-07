import { test, expect } from '@playwright/test';

/* §5.8's gate, and the only thing in this repository that checks our export
 * against the real renderer rather than against our own reading of its schema.
 *
 * `Me_session.validate` is that reading. A projection can satisfy it and still
 * be rejected by the element -- a graph id it cannot address, a namespace it
 * splits somewhere we did not expect, node data keyed by an id no node carries.
 * So this loads the session `native_graph visualize` actually produced from the
 * committed resnet18 model.json and enters every graph in it.
 *
 * One hard assertion, and it is about US: every graph finishes processing. The
 * rest is recorded, like the 0A spike, because a gate that fails when an
 * observation is inconvenient teaches nothing.
 */

test('the exported session renders in the pinned element', async ({ page }) => {
  const missing: string[] = [];
  page.on('response', (r) => { if (r.status() === 404) missing.push(new URL(r.url()).pathname); });
  const consoleErrors: string[] = [];
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
  page.on('pageerror', (e) => consoleErrors.push(String(e)));

  await page.goto('/session.html');
  await page.waitForFunction(() => (window as any).__gate?.done === true, null,
    { timeout: 180_000 });

  const gate = await page.evaluate(() => (window as any).__gate);
  console.log('--- Stage 1 gate ---');
  console.log(JSON.stringify({
    model: gate.model,
    collectionLabel: gate.collectionLabel,
    defaultView: gate.defaultView,
    views: gate.views,
    graphs: gate.graphs,
    detail: gate.detail,
    nodeData: gate.nodeData,
    groupAttributes: gate.groupAttributes,
  }, null, 2));
  console.log('capabilities:\n' + (gate.capabilities ?? []).join('\n'));
  if (gate.error) console.log('driver error: ' + gate.error);
  if (missing.length) console.log('404s:\n' + [...new Set(missing)].sort().join('\n'));
  if (consoleErrors.length) console.log('console errors:\n' + consoleErrors.slice(0, 3).join('\n'));

  expect(gate.error, 'the gate driver threw').toBeUndefined();

  // The session holds the source graph, the initial Native graph and the
  // canonical one. Fewer than three would mean the exporter changed shape and
  // this gate stopped covering what it claims to.
  expect(gate.graphs.length, 'graphs entered').toBeGreaterThanOrEqual(3);

  // The one hard assertion. A graph that never finishes processing is a defect
  // in what we emitted however the renderer chooses to display it.
  for (const g of gate.graphs) {
    expect(g.fired, `${g.id} (${g.nodes} nodes) never finished processing`).toBe(true);
  }

  // The detail delta, installed LATE and linked from a node already on screen.
  // Same rule: whether the renderer chooses to show it is its business, but a
  // session that never finishes processing with it merged in is ours.
  expect(gate.detail, 'no detail delta was exported').toBeDefined();
  expect(gate.detail.linkedFrom, 'the delta named no node in its parent graph').not.toBeNull();
  expect(gate.detail.fired, 'the merged session never finished processing').toBe(true);

  // The actual detail-graph entry, not merely finding the link on screen.
  // `selectNode` into the parent graph cannot by itself prove the subgraph is
  // reachable -- it never asks the renderer to activate it. Entering it does,
  // through the second `modelGraphProcessed` event only a newly-activated
  // graph can fire.
  expect(gate.detail.selectedInParent, 'selecting the linking node in its parent graph threw').toBe(true);
  expect(gate.detail.entered, 'selecting the detail graph itself threw').toBe(true);
  expect(gate.detail.enteredFired, 'the detail graph never finished processing once entered').toBe(true);
});
