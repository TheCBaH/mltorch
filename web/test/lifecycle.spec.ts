import { test, expect } from '@playwright/test';

/* The gate for Path B's three load-bearing DOM claims (`cancel.md`).
 *
 * Distinct from the two specs next door in what it is about. The 0A spike asks
 * what the renderer can do; the Stage 1 gate asks whether what we export is
 * something the renderer accepts. This one asks whether the LIFECYCLE the
 * cancellation design assumes is real: that a hidden-but-connected candidate
 * still works, that candidates do not serialise behind each other, and that
 * removal after the expected event is safe.
 *
 * Unlike those two it asserts nearly everything it records, because these are
 * not observations about a third party we must accommodate -- they are the
 * premises the design is built on. If one fails, the answer is to change
 * `cancel.md`, not to soften the assertion.
 */

test('the pinned element supports connected-hidden slots and concurrent candidates', async ({ page }) => {
  const pageErrors: string[] = [];
  const consoleErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(String(error)));
  page.on('console', (message) => { if (message.type() === 'error') consoleErrors.push(message.text()); });

  await page.goto('/lifecycle.html');
  await page.waitForFunction(() => (window as any).__lifecycle?.done === true, null,
    { timeout: 240_000 });

  const lifecycle = await page.evaluate(() => (window as any).__lifecycle);
  console.log('--- Path B lifecycle ---');
  console.log(JSON.stringify(lifecycle.steps, null, 2));
  if (lifecycle.error) console.log('driver error: ' + lifecycle.error);
  if (consoleErrors.length) console.log('console errors:\n' + consoleErrors.slice(0, 5).join('\n'));

  expect(lifecycle.error, 'the lifecycle driver threw').toBeUndefined();

  // 1. Connected and hidden is a working state, not merely a rendered-nothing
  // state. A candidate with a zero-sized box would have been given no room to
  // lay out in, so the size is asserted alongside the event.
  const hidden = lifecycle.steps.hiddenSlotProcesses;
  expect(hidden.fired, 'a hidden connected candidate never finished processing').toBe(true);
  expect(hidden.connected).toBe(true);
  expect(hidden.visibility, 'the slot must hide with visibility, not display').toBe('hidden');
  expect(hidden.width, 'hidden candidate width').toBeGreaterThan(0);
  expect(hidden.height, 'hidden candidate height').toBeGreaterThan(0);

  // 2. Concurrency is what removes the 90-second stall: if the pinned element
  // serialised layout across instances, a cancelled install would still block
  // its replacement and Path B would buy nothing.
  const concurrent = lifecycle.steps.concurrentProcessing;
  expect(concurrent.visibleCurrentStillConnected).toBe(true);
  expect(concurrent.real.fired, 'the real session did not process alongside a visible current').toBe(true);
  expect(concurrent.small.fired, 'the second hidden candidate did not process concurrently').toBe(true);

  // 3. An abandoned candidate is exactly what a quarantined entry looks like
  // to the DOM. A new candidate must not wait behind it -- under the old
  // single-queue renderer this one could not have started at all.
  const abandoned = lifecycle.steps.abandonedDoesNotBlock;
  expect(abandoned.fired, 'a candidate started behind an abandoned one never processed').toBe(true);
  expect(abandoned.abandonedStillConnected).toBe(true);
  expect(abandoned.connectedElements, 'every candidate stays connected').toBeGreaterThanOrEqual(4);
  // The margin, not just the outcome: the application's processing timeout is
  // 90 s, and the whole point of Path B is that a replacement does not serve
  // any part of it behind an abandoned candidate. Observed at ~76 ms, so a
  // 30 s bound fails only if candidates have started serialising again.
  // `abandonedFinishedFirst` is recorded rather than asserted -- which of two
  // concurrent candidates wins is a race, and asserting it would be flaky.
  expect(abandoned.ms, 'the replacement waited behind the abandoned candidate').toBeLessThan(30_000);

  // 4. The element is never reparented, and removal happens only after the
  // expected event.
  const removal = lifecycle.steps.safeRemoval;
  expect(removal.parentStable, 'the element changed parent between connection and removal').toBe(true);
  expect(removal.disconnected).toBe(true);
  expect(removal.slotDetached).toBe(true);

  // NG0953 is the specific failure this whole design exists to avoid: an emit
  // from a component destroyed while its background work was still running.
  const destroyed = [...pageErrors, ...consoleErrors].filter((text) => text.includes('NG0953'));
  expect(destroyed, 'a destroyed-OutputRef emit escaped').toEqual([]);
  expect(pageErrors, 'the page raised an error').toEqual([]);
});
