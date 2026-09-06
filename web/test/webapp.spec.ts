import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';

/* Two instruments, both installed before the application script runs.
 *
 * 1. Worker traffic, counted for the APP's worker only. The pinned visualizer
 *    runs a layout worker of its own (`vendor/worker.js`), and a view switch
 *    deliberately builds a new visualizer candidate -- so a global counter
 *    would rise on exactly the presentation-only change it is meant to prove
 *    quiet. Filtering by basename would not do either: the visualizer's own
 *    default path is `worker.js`, the same basename the app uses, and only the
 *    page's explicit override separates them. So the full resolved href is
 *    compared, against the same expression `app.js` uses.
 *
 * 2. Every mutation of the mount, rather than a sample of it. Path B keeps each
 *    candidate connected, so what the user must never see is a second slot
 *    competing for the screen, the pointer or a screen reader. Sampling on a
 *    timer can miss that window -- and does, now that a small graph can finish
 *    in one processing round -- while a MutationObserver sees every transition
 *    the DOM actually went through.
 */
const INSTRUMENT = () => {
  const w = window as any;
  w.__appWorkerPosts = 0;
  const tracked = new WeakSet<Worker>();
  const Native = window.Worker;
  const app = () => new URL('./worker.js', document.baseURI).href;
  // Resolved at CONSTRUCTION time: `document.baseURI` is not necessarily final
  // when this script runs.
  window.Worker = class extends Native {
    constructor(url: string | URL, options?: WorkerOptions) {
      super(url as any, options);
      if (new URL(url as any, document.baseURI).href === app()) tracked.add(this as any);
    }
    postMessage(...args: any[]) {
      if (tracked.has(this as any)) w.__appWorkerPosts += 1;
      return (Native.prototype.postMessage as any).apply(this, args);
    }
  } as any;

  w.__slotSamples = [];
  const sample = () => {
    w.__slotSamples.push([...document.querySelectorAll('#visualizer .visualizer-slot')].map((el) => {
      const style = getComputedStyle(el);
      return {
        current: el.classList.contains('visualizer-slot--current'),
        ariaHidden: el.getAttribute('aria-hidden'),
        visibility: style.visibility,
        pointerEvents: style.pointerEvents,
        width: el.getBoundingClientRect().width,
      };
    }));
  };
  const start = () => {
    const mount = document.getElementById('visualizer');
    if (!mount) { requestAnimationFrame(start); return; }
    new MutationObserver(sample).observe(mount, {
      childList: true, subtree: true, attributes: true,
      attributeFilter: ['class', 'aria-hidden'],
    });
    sample();
  };
  start();
};

type Slot = { current: boolean; ariaHidden: string | null; visibility: string; pointerEvents: string; width: number };

test.beforeEach(async ({ page }) => { await page.addInitScript(INSTRUMENT); });

const posts = (page: Page): Promise<number> => page.evaluate(() => (window as any).__appWorkerPosts);
const samples = (page: Page): Promise<Slot[][]> => page.evaluate(() => (window as any).__slotSamples);
const slots = (page: Page): Promise<Slot[]> =>
  page.evaluate(() => (window as any).__slotSamples.at(-1) ?? []);
const loaded = (page: Page) => expect(page.locator('#status')).toHaveText('Model loaded', { timeout: 90_000 });

function exactlyOneOnScreen(observed: Slot[], when: string) {
  const current = observed.filter((slot) => slot.current);
  expect(current, when).toHaveLength(1);
  expect(current[0].visibility, when).toBe('visible');
  expect(current[0].pointerEvents, when).toBe('auto');
  expect(current[0].ariaHidden, when).toBeNull();
  expect(current[0].width, when).toBeGreaterThan(100);
  for (const hidden of observed.filter((slot) => !slot.current)) {
    // Retained, but out of reach of the eye, the pointer and the accessibility
    // tree alike.
    expect(hidden.visibility, when).toBe('hidden');
    expect(hidden.pointerEvents, when).toBe('none');
    expect(hidden.ariaHidden, when).toBe('true');
  }
}

/* A completion signal for any replacement -- a reload, an option change, a view
 * switch -- that does not depend on the status text.
 *
 * `#status` is not usable for this: it still reads "Model loaded" (or "View
 * changed") from the PREVIOUS operation, so asserting on it passes instantly
 * and the test measures the state before its own action. Every replacement
 * instead connects a candidate and then removes its predecessor, so the mount's
 * own mutation history says when one has been through and settled.
 */
async function replaced(page: Page, from: number, timeout = 90_000) {
  await expect.poll(async () => {
    const history = await samples(page);
    return history.length > from
      && history.slice(from).some((observed) => observed.length > 1)
      && history.at(-1)!.length === 1;
  }, { timeout, message: 'no replacement completed' }).toBe(true);
}

/* The invariant over the WHOLE recorded history, plus the proof that the
 * history contains the window it is about: a run in which two slots were never
 * connected at once would satisfy "never two on screen" vacuously. */
async function neverTwoOnScreen(page: Page, label: string) {
  const history = await samples(page);
  let sawTwo = false;
  for (const [index, observed] of history.entries()) {
    const current = observed.filter((slot) => slot.current);
    expect(current.length, `${label} sample ${index}: ${JSON.stringify(observed)}`).toBeLessThanOrEqual(1);
    if (observed.length > 1) {
      sawTwo = true;
      // While two are connected, the hidden one is out of reach on every axis.
      for (const hidden of observed.filter((slot) => !slot.current)) {
        expect(hidden.visibility, `${label} sample ${index}`).toBe('hidden');
        expect(hidden.pointerEvents, `${label} sample ${index}`).toBe('none');
        expect(hidden.ariaHidden, `${label} sample ${index}`).toBe('true');
      }
    }
  }
  return sawTwo;
}

/* ------------------------------------------------------------------ assets */

test('assembled app is prefix-safe and loads its application assets', async ({ page }) => {
  const failures: string[] = [];
  page.on('response', (response) => { if (!response.ok()) failures.push(`${response.status()} ${response.url()}`); });
  await page.goto('/index.html');
  await expect(page.locator('h1')).toHaveText('MLTorch Model Explorer');
  await expect(page.locator('#catalogue')).toBeVisible();
  await expect(page.locator('#status')).not.toHaveText('Starting…');
  expect(failures).toEqual([]);
});

test('assembled app starts loading the default catalogue model without error', async ({ page }) => {
  await page.goto('/index.html');
  await expect(page.locator('#status')).not.toHaveText('Starting…');
  await expect(page.locator('#error')).toBeHidden();
});

/* ------------------------------------------------------------- replacement */

test('a replacement load never takes the current model off screen', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  await page.goto('/index.html');
  await loaded(page);
  exactlyOneOnScreen(await slots(page), 'first load');

  const from = (await samples(page)).length;
  await page.locator('#reload').click();
  await replaced(page, from);

  // The replacement is the only thing left; the predecessor was removed only
  // after the replacement's own processing event made the swap safe.
  await expect(page.locator('#visualizer .visualizer-slot')).toHaveCount(1);
  exactlyOneOnScreen(await slots(page), 'after replacement');
  const sawTwo = await neverTwoOnScreen(page, 'replacement');
  expect(sawTwo, 'the two-slot window never occurred, so the invariant was vacuous').toBe(true);
  await expect(page.locator('#error')).toBeHidden();
  expect(errors.join('\n')).not.toContain('NG0953');
  expect(errors).toEqual([]);
});

test('cancelling mid-load leaves the previous model usable and says nothing stale', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  await page.goto('/index.html');
  await loaded(page);

  // This test needs to click #reload a SECOND time while the first candidate
  // is still connected but not yet processed -- and for a small graph that
  // window can close in a single event-loop tick (see the instrumentation
  // note above `INSTRUMENT`), too narrow for a live DOM poll to reliably
  // land in. CPU-throttling the page widens it without changing what either
  // click does.
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setCPUThrottlingRate', { rate: 8 });
  try {
    // Reload cancels whatever is in flight before starting, so the second click
    // supersedes the first candidate while it is still processing.
    await page.locator('#reload').click();
    await expect(page.locator('#visualizer .visualizer-slot')).toHaveCount(2, { timeout: 90_000 });
    await page.locator('#reload').click();
    // Cancellation removes authority, not the DOM connection: tearing a still
    // processing element out is precisely what throws NG0953.
    expect(await page.locator('#visualizer .visualizer-slot').count(),
      'the cancelled candidate was torn out').toBe(2);
  } finally {
    await cdp.send('Emulation.setCPUThrottlingRate', { rate: 1 });
  }

  await loaded(page);
  await expect(page.locator('#visualizer .visualizer-slot')).toHaveCount(1, { timeout: 90_000 });
  exactlyOneOnScreen(await slots(page), 'after cancellation settled');
  await neverTwoOnScreen(page, 'cancellation');
  // A cancelled load reports nothing at all -- neither a status nor an error.
  await expect(page.locator('#error')).toBeHidden();
  expect(errors.join('\n')).not.toContain('NG0953');
  expect(errors).toEqual([]);
});

/* ------------------------------------------------------- view navigation */

/* Source first is the browser's own reading-order default; the exporter keeps
 * Canonical as its `defaultView` for the CLI, and neither had to change for
 * the other. */
test('the first view is Source, and every stage view is offered', async ({ page }) => {
  await page.goto('/index.html');
  await loaded(page);
  await expect(page.locator('#view')).toHaveValue('v/source');
  const offered = await page.locator('#view option').evaluateAll((os) => os.map((o) => (o as HTMLOptionElement).value));
  expect(offered).toContain('v/source');
  expect(offered).toContain('v/initial');
  expect(offered).toContain('v/canonical');
  // Only stage views: the flow capability names `g/flow`, which no collection
  // holds and no view points at.
  expect(offered.some((id) => id.includes('flow'))).toBe(false);
});

test('switching views sends no worker message', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  await page.goto('/index.html');
  await loaded(page);
  const before = await posts(page);
  expect(before).toBeGreaterThan(0);

  const offered = await page.locator('#view option').evaluateAll((os) => os.map((o) => (o as HTMLOptionElement).value));
  for (const view of offered.filter((id) => id !== 'v/source')) {
    await page.locator('#view').selectOption(view);
    /* The URL is written after `present()` resolves, so it is a completion
     * signal. The status text is not: it still reads "View changed" from the
     * previous switch, so asserting on it would pass before this one ran. */
    await expect.poll(() => new URL(page.url()).searchParams.get('view'),
      { timeout: 90_000, message: `switching to ${view}` }).toBe(view);
    expect(await posts(page), `switching to ${view} posted to the app worker`).toBe(before);
  }
  await expect(page.locator('#error')).toBeHidden();
  await neverTwoOnScreen(page, 'view switching');
  expect(errors.join('\n')).not.toContain('NG0953');
});

test('back and forward across a view-only change sends no worker message', async ({ page }) => {
  await page.goto('/index.html');
  await loaded(page);
  const before = await posts(page);

  await page.locator('#view').selectOption('v/canonical');
  await expect.poll(() => new URL(page.url()).searchParams.get('view'), { timeout: 90_000 })
    .toBe('v/canonical');

  /* On a history move the URL changes before anything is rendered, so it is
   * not a completion signal here; the selector is, because the page resets it
   * from the coordinator's resolved view only after `present()` resolves. */
  await page.goBack();
  await expect(page.locator('#view')).toHaveValue('v/source', { timeout: 90_000 });
  expect(new URL(page.url()).searchParams.get('view')).toBe('v/source');
  expect(await posts(page), 'going back re-ran the worker').toBe(before);

  await page.goForward();
  await expect(page.locator('#view')).toHaveValue('v/canonical', { timeout: 90_000 });
  expect(new URL(page.url()).searchParams.get('view')).toBe('v/canonical');
  expect(await posts(page), 'going forward re-ran the worker').toBe(before);
});

/* A URL naming a view this model does not have falls back rather than failing,
 * says so, and rewrites itself to what is actually on screen. */
test('a stale view in the URL falls back to Source with a notice', async ({ page }) => {
  await page.goto('/index.html?model=mobilenetv2_050&view=v%2Fnot-a-view');
  await loaded(page);
  await expect(page.locator('#view')).toHaveValue('v/source');
  await expect(page.locator('#notice')).toBeVisible();
  await expect(page.locator('#notice')).toContainText('v/not-a-view');
  await expect(page.locator('#error')).toBeHidden();
  expect(new URL(page.url()).searchParams.get('view')).toBe('v/source');
});

/* ------------------------------------------------- constants: grouped/explicit */

/* No worker call, and every other candidate-lifecycle guarantee a view switch
 * already has -- `Coordinator.setConstants` reaches the bridge's own
 * `groupConstants`, never `request.buildSession`, and replays the current
 * presentation through the same `present()` a view switch uses. */
test('toggling constants sends no worker message and keeps the same view', async ({ page }) => {
  await page.goto('/index.html');
  await loaded(page);
  expect(await page.locator('#constants').inputValue()).toBe('grouped');
  // The default is never written, so a plain load names nothing.
  expect(new URL(page.url()).searchParams.get('constants')).toBeNull();
  const before = await posts(page);
  const from = (await samples(page)).length;

  await page.locator('#constants').selectOption('explicit');
  await expect.poll(() => new URL(page.url()).searchParams.get('constants'),
    { timeout: 90_000, message: 'switching to explicit' }).toBe('explicit');
  expect(await posts(page), 'switching to explicit posted to the app worker').toBe(before);
  await expect(page.locator('#view')).toHaveValue('v/source');

  await page.locator('#constants').selectOption('grouped');
  // Back at the default, so the URL drops the parameter again.
  await expect.poll(() => new URL(page.url()).searchParams.get('constants'),
    { timeout: 90_000, message: 'switching back to grouped' }).toBeNull();
  expect(await posts(page), 'switching back to grouped posted to the app worker').toBe(before);
  await expect(page.locator('#view')).toHaveValue('v/source');

  await expect(page.locator('#error')).toBeHidden();
  await neverTwoOnScreen(page, 'constants toggling');
  expect((await samples(page)).length).toBeGreaterThan(from);
});

test('back and forward across a constants-only change sends no worker message', async ({ page }) => {
  await page.goto('/index.html');
  await loaded(page);
  const before = await posts(page);

  await page.locator('#constants').selectOption('explicit');
  await expect.poll(() => new URL(page.url()).searchParams.get('constants'), { timeout: 90_000 })
    .toBe('explicit');

  await page.goBack();
  await expect(page.locator('#constants')).toHaveValue('grouped', { timeout: 90_000 });
  expect(new URL(page.url()).searchParams.get('constants')).toBeNull();
  expect(await posts(page), 'going back re-ran the worker').toBe(before);

  await page.goForward();
  await expect(page.locator('#constants')).toHaveValue('explicit', { timeout: 90_000 });
  expect(new URL(page.url()).searchParams.get('constants')).toBe('explicit');
  expect(await posts(page), 'going forward re-ran the worker').toBe(before);
});

/* Unlike a stale view/comparison/flow -- which name a destination that does
 * not exist and so earn a notice -- an unusable `constants` is the same kind
 * of unusable as an unknown `stages`/`verify` value: silently treated as
 * absent, never a request for nothing and never worth alarming the reader
 * about. */
test('an unknown constants value in the URL is silently treated as the default', async ({ page }) => {
  await page.goto('/index.html?model=mobilenetv2_050&constants=bogus');
  await loaded(page);
  await expect(page.locator('#constants')).toHaveValue('grouped');
  await expect(page.locator('#notice')).toBeHidden();
  await expect(page.locator('#error')).toBeHidden();
});

test('reload preserves an explicit constants selection', async ({ page }) => {
  await page.goto('/index.html');
  await loaded(page);
  await page.locator('#constants').selectOption('explicit');
  await expect.poll(() => new URL(page.url()).searchParams.get('constants'), { timeout: 90_000 })
    .toBe('explicit');

  const before = await posts(page);
  const from = (await samples(page)).length;
  await page.locator('#reload').click();
  await expect.poll(() => posts(page), { timeout: 90_000 }).toBe(before + 1);
  await replaced(page, from, 240_000);
  await expect(page.locator('#constants')).toHaveValue('explicit');
  expect(new URL(page.url()).searchParams.get('constants')).toBe('explicit');
});

/* --------------------------------------------------- exported comparisons */

/* The two the exporter declares: `c/import` (source -> initial, with its
 * explicit correspondence components) and `c/canonical` (initial ->
 * canonical, deliberately with none). `Session.comparisons` is the sole
 * authority for what is offered -- no capability is consulted. */
const comparisonOptions = (page: Page) =>
  page.locator('#comparison option').evaluateAll((os) => os.map((o) => (o as HTMLOptionElement).value));

/* The URL is written after `present()` resolves, so it is the completion
 * signal; `#status` still reads "View changed" from the previous operation. */
const openedComparison = (page: Page, id: string) =>
  expect.poll(() => new URL(page.url()).searchParams.get('comparison'),
    { timeout: 90_000, message: `opening ${id}` }).toBe(id);

test('the exported comparisons are offered, and only those', async ({ page }) => {
  await page.goto('/index.html');
  await loaded(page);
  const offered = await comparisonOptions(page);
  // The empty leading entry is the way back to a single view; everything else
  // is a declared comparison.
  expect(offered[0]).toBe('');
  expect(offered.slice(1)).toEqual(['c/import', 'c/canonical']);
  await expect(page.locator('#comparison')).toHaveValue('');
  await expect(page.locator('#pane-labels')).toBeHidden();
});

test('opening a comparison sends no worker message and keeps one model on screen', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  await page.goto('/index.html');
  await loaded(page);
  const before = await posts(page);
  expect(before).toBeGreaterThan(0);

  for (const id of ['c/import', 'c/canonical']) {
    await page.locator('#comparison').selectOption(id);
    await openedComparison(page, id);
    expect(await posts(page), `opening ${id} posted to the app worker`).toBe(before);
    exactlyOneOnScreen(await slots(page), `after opening ${id}`);
    // A comparison is not a view, so the view key must not survive alongside it.
    expect(new URL(page.url()).searchParams.get('view')).toBeNull();
  }

  // And back out again, to the single view the selector still shows.
  await page.locator('#comparison').selectOption('');
  await expect.poll(() => new URL(page.url()).searchParams.get('view'), { timeout: 90_000 })
    .toBe('v/source');
  expect(new URL(page.url()).searchParams.get('comparison')).toBeNull();
  expect(await posts(page)).toBe(before);

  await expect(page.locator('#error')).toBeHidden();
  await neverTwoOnScreen(page, 'comparison switching');
  expect(errors.join('\n')).not.toContain('NG0953');
});

/* Everything the chrome says about a comparison comes from the session: the
 * pane headings from the stage views naming the same graphs, the mapping count
 * verbatim, and -- for `c/canonical` -- the limitation stated rather than
 * papered over. */
test('a comparison names its panes and states what its mapping claims', async ({ page }) => {
  await page.goto('/index.html');
  await loaded(page);

  await page.locator('#comparison').selectOption('c/import');
  await openedComparison(page, 'c/import');
  await expect(page.locator('#pane-labels')).toBeVisible();
  await expect(page.locator('#pane-labels')).toContainText('Exported Program');
  await expect(page.locator('#pane-labels')).toContainText('Initial Native');
  await expect(page.locator('#presentation-note')).toContainText('exported correspondence components');

  await page.locator('#comparison').selectOption('c/canonical');
  await openedComparison(page, 'c/canonical');
  await expect(page.locator('#pane-labels')).toContainText('Initial Native');
  await expect(page.locator('#pane-labels')).toContainText('Canonical Native');
  await expect(page.locator('#presentation-note'))
    .toContainText('No explicit changed-node mapping was exported.');
  // `matchNodeIdFallback` is true for this one, so the empty list is a claim
  // about identifiers rather than a missing computation -- and the copy has to
  // say which, because the two look identical on the wire without it.
  await expect(page.locator('#presentation-note')).toContainText('sharing an identifier');

  // Leaving compare mode takes the headings with it.
  await page.locator('#comparison').selectOption('');
  await expect(page.locator('#pane-labels')).toBeHidden({ timeout: 90_000 });
});

test('a comparison the session does not declare falls back without inventing one', async ({ page }) => {
  await page.goto('/index.html?model=mobilenetv2_050&comparison=c%2Fnope');
  await loaded(page);
  await expect(page.locator('#comparison')).toHaveValue('');
  await expect(page.locator('#pane-labels')).toBeHidden();
  await expect(page.locator('#notice')).toContainText('c/nope');
  await expect(page.locator('#error')).toBeHidden();
  // Rewritten to what is actually on screen.
  const url = new URL(page.url());
  expect(url.searchParams.get('comparison')).toBeNull();
  expect(url.searchParams.get('view')).toBe('v/source');
});

test('back and forward between a view and a comparison sends no worker message', async ({ page }) => {
  await page.goto('/index.html');
  await loaded(page);
  const before = await posts(page);

  await page.locator('#comparison').selectOption('c/import');
  await openedComparison(page, 'c/import');

  await page.goBack();
  await expect(page.locator('#comparison')).toHaveValue('', { timeout: 90_000 });
  expect(new URL(page.url()).searchParams.get('view')).toBe('v/source');
  expect(await posts(page), 'going back re-ran the worker').toBe(before);

  await page.goForward();
  await expect(page.locator('#comparison')).toHaveValue('c/import', { timeout: 90_000 });
  expect(new URL(page.url()).searchParams.get('comparison')).toBe('c/import');
  expect(await posts(page), 'going forward re-ran the worker').toBe(before);
});

/* A comparison is two processing rounds, so there is a window in which its left
 * pane is up and its right is not. Cancelling inside that window must expose
 * neither a one-pane comparison nor a second visible slot -- and must not tear
 * out an element that is still processing. */
test('cancelling while a comparison is being built exposes no partial comparison', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  await page.goto('/index.html');
  await loaded(page);

  await page.locator('#comparison').selectOption('c/import');
  // The candidate is connected and processing; the previous view is still the
  // only visible slot.
  await expect(page.locator('#visualizer .visualizer-slot')).toHaveCount(2, { timeout: 90_000 });
  exactlyOneOnScreen(await slots(page), 'while the comparison is being built');

  // Reload cancels whatever is in flight, superseding the half-built comparison.
  await page.locator('#reload').click();
  await loaded(page);
  await expect(page.locator('#visualizer .visualizer-slot')).toHaveCount(1, { timeout: 90_000 });
  exactlyOneOnScreen(await slots(page), 'after cancellation settled');
  await neverTwoOnScreen(page, 'comparison cancellation');
  await expect(page.locator('#error')).toBeHidden();
  expect(errors.join('\n')).not.toContain('NG0953');
});

/* -------------------------------------------------------- the export flow */

/* The coarse spine: states and transitions as nodes, opened through declared
 * destinations only. Selection and action are two steps -- the visualizer
 * reports an ordinary single click, so routing on selection alone would make a
 * mis-click destructive. */

const openedFlow = (page: Page, id: string) =>
  expect.poll(() => new URL(page.url()).searchParams.get('flow'),
    { timeout: 90_000, message: `opening ${id}` }).toBe(id);

test('the export flow is offered only when the session carries one', async ({ page }) => {
  await page.goto('/index.html');
  await loaded(page);
  await expect(page.locator('#flow-open')).toBeVisible();
  await expect(page.locator('#flow-open')).toHaveAttribute('aria-pressed', 'false');
  // Nothing about the flow is shown until it is open.
  await expect(page.locator('#flow-note')).toBeHidden();
  await expect(page.locator('#flow-selection')).toBeHidden();
});

test('opening the flow sends no worker message and explains what it is', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  await page.goto('/index.html');
  await loaded(page);
  const before = await posts(page);
  expect(before).toBeGreaterThan(0);

  await page.locator('#flow-open').click();
  await openedFlow(page, 'v/flow');
  expect(await posts(page), 'opening the flow posted to the app worker').toBe(before);
  exactlyOneOnScreen(await slots(page), 'after opening the flow');

  await expect(page.locator('#flow-note')).toContainText('Coarse export flow');
  await expect(page.locator('#flow-note')).toContainText('not individual pass execution');
  await expect(page.locator('#flow-legend')).toContainText('Representations');
  await expect(page.locator('#flow-legend')).toContainText('Transformations');
  // A flow is not a view or a comparison, so neither key survives beside it.
  expect(new URL(page.url()).searchParams.get('view')).toBeNull();
  expect(new URL(page.url()).searchParams.get('comparison')).toBeNull();

  // And back out again, to the single view the selector still shows.
  await page.locator('#flow-open').click();
  await expect.poll(() => new URL(page.url()).searchParams.get('view'), { timeout: 90_000 })
    .toBe('v/source');
  expect(new URL(page.url()).searchParams.get('flow')).toBeNull();
  expect(await posts(page)).toBe(before);
  await expect(page.locator('#flow-selection')).toBeHidden();

  await expect(page.locator('#error')).toBeHidden();
  await neverTwoOnScreen(page, 'flow switching');
  expect(errors.join('\n')).not.toContain('NG0953');
});

/* Selecting a node offers ONE declared action and navigates nothing until it is
 * taken. Driven through the element's public `selectNode`, which is the same
 * `AppService.selectNode` a click reaches -- a pointer event into a WebGL canvas
 * inside a shadow root is not synthesizable, and `web/test/flow.spec.ts` is
 * where that equivalence is established. */
async function selectFlowNode(page: Page, nodeId: string) {
  await page.evaluate((id) => {
    const slot = document.querySelector('#visualizer .visualizer-slot--current');
    const element = slot?.querySelector('model-explorer-visualizer') as any;
    /* No collection label: the element then searches every collection and
     * fills the real one into the event. The label is derived from the model
     * name, so spelling it here would pin a fixture
     * detail this test is not about. */
    element.selectNode(id, 'g/flow');
  }, nodeId);
  await expect(page.locator('#flow-selection')).toBeVisible({ timeout: 90_000 });
}

test('a state offers its declared stage view, and taking it opens exactly that', async ({ page }) => {
  await page.goto('/index.html');
  await loaded(page);
  const before = await posts(page);
  await page.locator('#flow-open').click();
  await openedFlow(page, 'v/flow');

  await selectFlowNode(page, 's/native/001');
  await expect(page.locator('#flow-selection')).toContainText('representation');
  await expect(page.locator('#flow-selection .flow-action')).toHaveText('Open Canonical Native');
  // Selection alone navigates nothing.
  expect(new URL(page.url()).searchParams.get('flow')).toBe('v/flow');

  await page.locator('#flow-selection .flow-action').click();
  await expect.poll(() => new URL(page.url()).searchParams.get('view'), { timeout: 90_000 })
    .toBe('v/canonical');
  expect(await posts(page), 'taking a state action posted to the app worker').toBe(before);
  await expect(page.locator('#error')).toBeHidden();
});

test('a paired transition offers only its declared comparison, an unpaired one none', async ({ page }) => {
  await page.goto('/index.html');
  await loaded(page);
  const before = await posts(page);
  await page.locator('#flow-open').click();
  await openedFlow(page, 'v/flow');

  await selectFlowNode(page, 't/native/000');
  await expect(page.locator('#flow-selection')).toContainText('import');
  await expect(page.locator('#flow-selection .flow-action'))
    .toHaveText('Open comparison exported program -> initial');

  /* The unpaired case is a visible STATEMENT, not a click that does nothing --
   * `t/symbolic/000` is an adaptation the exporter declares with no comparison,
   * and inferring one from its endpoints is exactly what the safeguards
   * forbid. */
  await selectFlowNode(page, 't/symbolic/000');
  await expect(page.locator('#flow-selection')).toContainText('adapt');
  await expect(page.locator('#flow-selection'))
    .toContainText('No exported comparison is available for this transition.');
  await expect(page.locator('#flow-selection .flow-action')).toHaveCount(0);

  // Back to the paired one, and take it.
  await selectFlowNode(page, 't/native/001');
  await page.locator('#flow-selection .flow-action').click();
  await expect.poll(() => new URL(page.url()).searchParams.get('comparison'), { timeout: 90_000 })
    .toBe('c/canonical');
  expect(await posts(page), 'taking a transition action posted to the app worker').toBe(before);
  await expect(page.locator('#error')).toBeHidden();
});

/* --- the app.js translation boundary, which clicking the control never reaches --- */

/* A load resolves a stage VIEW, so `?flow=` has to be opened on top of the
 * document once it arrives. Before this was wired the page loaded, resolved,
 * and then left the source stage on screen with a flow URL. */
test('a direct flow URL ends with the flow on screen, not the source stage', async ({ page }) => {
  await page.goto('/index.html?model=mobilenetv2_050&flow=v%2Fflow');
  await loaded(page);
  await expect(page.locator('#flow-open')).toHaveAttribute('aria-pressed', 'true', { timeout: 90_000 });
  await expect(page.locator('#flow-note')).toContainText('Coarse export flow');
  expect(new URL(page.url()).searchParams.get('flow')).toBe('v/flow');
  expect(new URL(page.url()).searchParams.get('view')).toBeNull();
  await expect(page.locator('#error')).toBeHidden();
});

test('back and forward across a flow sends no worker message', async ({ page }) => {
  await page.goto('/index.html');
  await loaded(page);
  const before = await posts(page);

  await page.locator('#flow-open').click();
  await openedFlow(page, 'v/flow');

  await page.goBack();
  await expect(page.locator('#flow-open')).toHaveAttribute('aria-pressed', 'false', { timeout: 90_000 });
  expect(new URL(page.url()).searchParams.get('view')).toBe('v/source');
  expect(await posts(page), 'going back re-ran the worker').toBe(before);

  await page.goForward();
  await expect(page.locator('#flow-open')).toHaveAttribute('aria-pressed', 'true', { timeout: 90_000 });
  expect(new URL(page.url()).searchParams.get('flow')).toBe('v/flow');
  expect(await posts(page), 'going forward re-ran the worker').toBe(before);
});

test('a flow the session does not declare falls back without inventing one', async ({ page }) => {
  await page.goto('/index.html?model=mobilenetv2_050&flow=v%2Fnope');
  await loaded(page);
  await expect(page.locator('#flow-open')).toHaveAttribute('aria-pressed', 'false');
  await expect(page.locator('#notice')).toContainText('v/nope');
  await expect(page.locator('#error')).toBeHidden();
  const url = new URL(page.url());
  expect(url.searchParams.get('flow')).toBeNull();
  expect(url.searchParams.get('view')).toBe('v/source');
});

/* ------------------------------------------------------ construction options */

test('capabilities are shown, and not-requested differs from unavailable', async ({ page }) => {
  await page.goto('/index.html');
  await loaded(page);
  const disclosure = page.locator('#unavailable');
  await disclosure.locator('summary').click();

  // These choices remain distinct in the current MobileNet source: verification
  // was omitted from the request, while folding cannot run without payloads.
  await expect(disclosure).toContainText('Symbolic verification');
  await expect(disclosure).toContainText('Not requested');
  await expect(disclosure).toContainText('Unavailable');
  // Loop IR and code generation stay visibly unimplemented rather than reading
  // as "you did not ask".
  await expect(disclosure).toContainText('Loop IR');
  await expect(disclosure).toContainText('Not implemented');
  // Folding is inert for a bare model.json, with the capability's own reason.
  await expect(disclosure).toContainText('folding');
  await expect(page.locator('#fold')).toBeDisabled();
  await expect(page.locator('#fold-note')).toContainText('payload');

});

/* Off installs no verification provider at all: an absent claim is said, never
 * rendered as a neutral or green one. */
test('with verification off there is no validation evidence', async ({ page }) => {
  await page.goto('/index.html');
  await loaded(page);
  await page.locator('#validation summary').click();
  await expect(page.locator('#validation summary')).toContainText('none');
  await expect(page.locator('#validation-body')).toContainText('Not requested');
  await expect(page.locator('#legend')).toBeHidden();
});

test('a verification effort starts a new export and shows its evidence verbatim', async ({ page }) => {
  await page.goto('/index.html');
  await loaded(page);
  const before = await posts(page);

  const from = (await samples(page)).length;
  await page.locator('#effort-controls input[value="quick"]').check();
  await expect.poll(() => posts(page),
    { timeout: 90_000, message: 'changing the effort did not start an export' }).toBe(before + 1);
  await replaced(page, from);
  await expect.poll(() => new URL(page.url()).searchParams.get('verify'), { timeout: 90_000 }).toBe('quick');
  await expect(page.locator('#effective-options')).toContainText('symbolic verification quick');

  await page.locator('#validation summary').click();
  const body = page.locator('#validation-body');
  await expect(body).toContainText('verifier clusters');
  // Strength and coverage ride in the supplied label and are never shortened.
  await expect(body).toContainText('unproved (');
  await expect(body).toContainText('vacuous');
  await expect(body).toContainText('audit reports');

  // The legend names the buckets beside the mount.
  await expect(page.locator('#legend')).toBeVisible();
  for (const bucket of ['Vacuous', 'Proved', 'Tested', 'Unproved', 'Refuted']) {
    await expect(page.locator('#legend')).toContainText(bucket);
  }
});

/* The controls are browser state now, so a reload must rebuild the
 * configuration on screen rather than the bridge's defaults. */
test('reload preserves the selected stages and effort', async ({ page }) => {
  // Two full "standard"-effort symbolic verification passes back to back --
  // this test is the only one that pays that cost twice -- run well inside
  // budget locally but not on CI's slower, shared runners, where the standard
  // pass alone can exceed the default 90s poll.
  test.setTimeout(600_000);
  await page.goto('/index.html');
  await loaded(page);

  /* Asserted on the URL's stage list rather than the prose readout: the
   * readout names "Kernel fusion" too, so a substring check for "Kernel"
   * cannot tell a deselected Kernel stage from the Fusion one that is still
   * requested. The URL carries the bridge's normalised value verbatim.
   *
   * Fusion surviving is correct: it forces the kernel computation internally
   * but names the kernel graph, so it is a stage without a view. */
  const stages = () => new URL(page.url()).searchParams.get('stages');
  const WITHOUT_KERNEL = 'source,initial_native,canonical,native4d,stage_program';
  let from = (await samples(page)).length;
  /* Fusion goes too, and it has to: it forces the kernel computation as its
   * prerequisite, so leaving it selected keeps the Kernel graph built and
   * `v/kernel` legitimately declared -- a view for a stage whose capability
   * reads `not_requested`. Deselecting only Kernel would therefore prove
   * nothing about the selector. */
  await page.locator('#stage-controls input#stage-kernel').uncheck();
  // Same shared, slower CI runners as the standard-effort passes below: an
  // export at default effort has hit the 90s default here too.
  await replaced(page, from, 240_000);
  from = (await samples(page)).length;
  await page.locator('#stage-controls input#stage-fusion').uncheck();
  await replaced(page, from, 240_000);
  await expect.poll(stages, { timeout: 90_000 }).toBe(WITHOUT_KERNEL);

  from = (await samples(page)).length;
  await page.locator('#effort-controls input[value="standard"]').check();
  await replaced(page, from, 240_000);

  const before = await posts(page);
  from = (await samples(page)).length;
  await page.locator('#reload').click();
  await expect.poll(() => posts(page), { timeout: 90_000 }).toBe(before + 1);
  await replaced(page, from, 240_000);
  await expect(page.locator('#effective-options')).toContainText('symbolic verification standard');
  await expect.poll(stages, { timeout: 90_000 }).toBe(WITHOUT_KERNEL);
  await expect(page.locator('#stage-controls input#stage-kernel')).not.toBeChecked();
  await expect(page.locator('#effort-controls input[value="standard"]')).toBeChecked();
  // A deselected stage leaves the selector too.
  const offered = await page.locator('#view option').evaluateAll((os) => os.map((o) => (o as HTMLOptionElement).value));
  expect(offered).not.toContain('v/kernel');
});

/* The backbones are stated, never offered: `Options.create` normalises by
 * filtering `all_stages` and never adds to it, so a page that did not send
 * them would not get them. */
test('the backbone stages are labelled, not offered as checkboxes', async ({ page }) => {
  await page.goto('/index.html');
  await loaded(page);
  await expect(page.locator('#backbone-note')).toContainText('Always built');
  const boxes = await page.locator('#stage-controls input[type=checkbox]')
    .evaluateAll((es) => es.map((e) => (e as HTMLInputElement).id));
  expect(boxes.sort()).toEqual(['stage-fusion', 'stage-kernel', 'stage-native4d', 'stage-stage_program']);
});
