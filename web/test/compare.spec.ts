import { test, expect } from '@playwright/test';

/* The Phase 3 comparison gate (`web-ui-3.md`, "Prerequisite 0").
 *
 * The renderer adapter may use only visualizer behaviour proven here, against
 * the bundle we actually ship. Unlike the 0A spike next door this asserts
 * nearly everything it records, for the reason `lifecycle.spec.ts` gives: these
 * are not observations about a third party we must accommodate, they are the
 * premises the two-pane adapter is built on. If one fails, the answer is to
 * change `web-ui-3.md`, not to soften the assertion.
 *
 * What it does NOT claim. Diff highlights and overlay edges are drawn in WebGL
 * inside a shadow root, so "the node is outlined" and "the overlay is in the
 * left pane" have no public observable. Those configs are asserted ACCEPTED --
 * both panes open, nothing throws, and the code path that consumes them is live
 * -- and nothing more. Claiming more would mean scraping internals the adapter
 * has no business depending on, which is exactly what the brief forbids.
 */

type Probe = {
  source: string; primed: boolean; took: boolean;
  before: string; after: string; paired: boolean;
};

/* A probe's verdict is only meaningful if the probe actually ran: `paired:
 * false` from a selection that never landed is a false negative that looks
 * exactly like a correct refusal. */
function ran(probe: Probe, what: string) {
  expect(probe.primed, `${what}: the other pane was never primed`).toBe(true);
  expect(probe.took, `${what}: the source selection never landed`).toBe(true);
}

function paired(probe: Probe, expected: string, what: string) {
  ran(probe, what);
  expect(probe.after, `${what}: expected ${expected}, got ${probe.after}`).toBe(expected);
  expect(probe.paired, `${what}: the other pane never moved off ${probe.before}`).toBe(true);
}

function unpaired(probe: Probe, what: string) {
  ran(probe, what);
  expect(probe.paired, `${what}: paired to ${probe.after}, which no entry declares`).toBe(false);
}

test('the comparison API supports two panes, a supplied mapping and per-pane data', async ({ page }) => {
  const pageErrors: string[] = [];
  const consoleErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(String(error)));
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });

  await page.goto('/compare.html');
  await page.waitForFunction(() => (window as any).__compare?.done === true, null,
    { timeout: 240_000 });

  const observed = await page.evaluate(() => (window as any).__compare);
  console.log('--- comparison contract ---');
  console.log(JSON.stringify(observed.steps, null, 2));
  if (observed.error) console.log('driver error: ' + observed.error);
  if (consoleErrors.length) console.log('console errors:\n' + consoleErrors.slice(0, 5).join('\n'));

  expect(observed.error, 'the comparison driver threw').toBeUndefined();
  const s = observed.steps;

  /* 1 + 2. Two panes, each proved by its OWN event. One `modelGraphProcessed`
   * cannot prove two panes, and the left graph emits a second time while the
   * split is being created -- so both filters matter. */
  expect(s.twoPanes.left.fired, 'the left graph never processed in pane 0').toBe(true);
  expect(s.twoPanes.right.fired, 'the right graph never processed in pane 1').toBe(true);
  expect(s.twoPanes.panes, 'a second pane was never created').toBe(2);
  expect(s.twoPanes.leftGraph).toBe('g/left');
  expect(s.twoPanes.rightGraph).toBe('g/right');

  /* Sync navigation is armed a beat AFTER the second pane event, because the
   * component that reads `config.syncNavigationData` is constructed under
   * `hasSplitPane && allPanesLoaded()`. The adapter does not care -- nobody
   * clicks in the same tick -- but a gate that probed immediately would record
   * a false negative and pin it as the contract. */
  expect(s.syncArming.armed, 'sync navigation never became active').toBe(true);

  /* 3. The supplied correspondence is followed, in both shapes that a
   * pair-wise mapping cannot express, and in both directions. */
  paired(s.mappingFollowed.oneToMany, 'r-alpha', '1:N');
  paired(s.mappingFollowed.manyToMany, 'r-gamma', 'N:M');
  paired(s.mappingFollowed.reverse, 'L2', 'N:M right-to-left');

  /* `disableMappingFallback: true` is honoured. `shared` is the only id present
   * in both graphs and appears in no entry, so pairing it could only ever be
   * id-identity -- which is why the fixture has it at all. */
  unpaired(s.mappingFollowed.fallbackOff, 'shared with the fallback off');
  unpaired(s.mappingFollowed.unmapped, 'an unmapped id with no twin');

  /* ...and the same document with the flag unset pairs it. Both directions,
   * because one alone cannot tell an honoured flag from a mapping that never
   * loaded -- and `[C2]`'s wire field decides exactly this per comparison. */
  expect(s.fallbackOn.armed.armed, 'sync never became active for the fallback case').toBe(true);
  paired(s.fallbackOn.shared, 'shared', 'shared with the fallback on');
  paired(s.fallbackOn.stillMapped, 'r-alpha', 'a declared entry still wins over the fallback');

  /* 5. Pane-addressed node data, installed where the adapter installs it: after
   * both pane events. */
  expect(s.paneNodeData.errors, 'pane-addressed node data threw').toEqual([]);
  expect(s.paneNodeData.accepted).toEqual([0, 1]);

  /* 4 + 6. ACCEPTANCE only, and the header says why that is all. A config
   * carrying `showDiffHighlights` with its two border colours, and one carrying
   * per-pane overlays, both open two panes without throwing. */
  expect(s.twoPanes.overlaysSupplied.left, 'no left-pane overlay was supplied').toBe('left-only');
  expect(s.twoPanes.overlaysSupplied.right, 'no right-pane overlay was supplied').toBe('right-only');
  expect(s.diffHighlights.left.fired, 'showDiffHighlights broke the left pane').toBe(true);
  expect(s.diffHighlights.right.fired, 'showDiffHighlights broke the right pane').toBe(true);
  expect(s.diffHighlights.armed.armed).toBe(true);
  unpaired(s.diffHighlights.unmapped, 'an unmapped id under showDiffHighlights');

  /* The configuration `[C2]` exists to prevent, stated as a fact rather than a
   * worry: an empty exported mapping with the fallback disabled is a comparison
   * in which NOTHING corresponds to anything -- not even the id that appears
   * verbatim on both sides. Upstream's `renderDiffHighlights` reads that same
   * empty mapped set as "all mapped nodes are missing", which is why
   * `c/canonical` must declare its fallback rather than have one assumed. */
  expect(s.emptyMappingWithFallbackOff.left.fired).toBe(true);
  expect(s.emptyMappingWithFallbackOff.right.fired).toBe(true);
  unpaired(s.emptyMappingWithFallbackOff.shared, 'an empty mapping with the fallback off');

  /* The case that makes the pane filter necessary rather than merely careful.
   * `Session.validate` permits a comparison whose two panes name the same
   * graph, and then `detail.modelGraph.id` is identical for both events -- so a
   * wait keyed on the graph id alone resolves the right-pane wait on the left
   * pane's event, and the adapter finalizes a comparison whose second pane
   * never processed. Both panes here carry `g/left`, and both waits still
   * resolve on their own pane. */
  expect(s.samePaneGraph.left.fired, 'pane 0 never processed the shared graph').toBe(true);
  expect(s.samePaneGraph.right.fired, 'pane 1 never processed the shared graph').toBe(true);
  expect(s.samePaneGraph.panes, 'a comparison of one graph with itself opened one pane').toBe(2);
  /* `fired` alone cannot police this: an event from the WRONG pane resolves the
   * promise just as well, so a wait keyed on the graph id alone still reports
   * success. `seenPane` is what the resolving event actually carried, so a
   * dropped pane filter surfaces here as a mismatch instead of a green run.
   * Verified by deleting the filter and watching this go red. */
  expect(s.samePaneGraph.left.seenPane, 'the pane-0 wait resolved on another pane').toBe(0);
  expect(s.samePaneGraph.right.seenPane, 'the pane-1 wait resolved on another pane').toBe(1);
  expect(s.twoPanes.left.seenPane).toBe(0);
  expect(s.twoPanes.right.seenPane).toBe(1);

  /* 7. The Path B lifecycle, across TWO waits. A candidate abandoned after its
   * left pane and before its right is exactly the window the adapter must never
   * finalize in: it stays connected (tearing it out is what throws NG0953), it
   * never becomes visible, and the comparison already on screen keeps the
   * screen. */
  expect(s.abandonedBetweenPanes.leftFired, 'the replacement never processed its left pane').toBe(true);
  expect(s.abandonedBetweenPanes.abandonedVisible,
    'a one-pane comparison became visible').toBe(false);
  expect(s.abandonedBetweenPanes.abandonedStillConnected,
    'the abandoned candidate was torn out mid-flight').toBe(true);
  expect(s.abandonedBetweenPanes.originalStillCurrent,
    'the visible comparison was taken off screen for a replacement that was not ready').toBe(true);

  // The specific failure the whole connected-hidden design exists to avoid.
  const destroyed = [...pageErrors, ...consoleErrors].filter((t) => t.includes('NG0953'));
  expect(destroyed, 'a destroyed-OutputRef emit escaped').toEqual([]);
  expect(pageErrors, 'the page raised an error').toEqual([]);
});
