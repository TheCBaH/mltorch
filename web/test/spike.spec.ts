import { test, expect } from '@playwright/test';

/* Stage 0A: the contract spike.
 *
 * This does not assert that the answers are the ones the design hoped for. It
 * RECORDS them, and fails only if the page could not run at all -- because the
 * point of a gate is to settle a question, and a gate that fails when the
 * answer is inconvenient teaches nothing. The design doc records the outcome;
 * §6.1.1 then either keeps its off-DOM staging or takes the specified
 * snapshot/restore fallback.
 */

test('the pinned renderer answers the coordinator design questions', async ({ page }) => {
  const missing: string[] = [];
  page.on('response', (r) => { if (r.status() === 404) missing.push(new URL(r.url()).pathname); });
  const consoleErrors: string[] = [];
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
  page.on('pageerror', (e) => consoleErrors.push(String(e)));

  await page.goto('/index.html');
  await page.waitForFunction(() => (window as any).__spike?.done === true, null,
    { timeout: 90_000 });

  const spike = await page.evaluate(() => (window as any).__spike);
  console.log('--- Stage 0A answers ---');
  console.log(JSON.stringify(spike.answers, null, 2));
  if (spike.error) console.log('driver error: ' + spike.error);
  if (missing.length) console.log('404s:\n' + [...new Set(missing)].sort().join('\n'));
  if (consoleErrors.length) console.log('console errors:\n' + consoleErrors.slice(0, 3).join('\n'));

  // The one hard assertion: the driver ran to completion. Everything else is
  // an observation this gate exists to make.
  expect(spike.error, 'the spike driver threw').toBeUndefined();

  // The custom element has to exist and have processed at least one graph,
  // or none of the recorded answers mean anything.
  expect(spike.answers.offDomBuildThenAttach, 'no first-load answer').toBeDefined();
});
