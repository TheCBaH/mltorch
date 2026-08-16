import { expect, test } from '@playwright/test';

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

/* Path B keeps every candidate connected, so what the user must never see is a
 * second slot competing for the screen, for pointer input, or for a screen
 * reader. These two cases sample that invariant while a replacement is in
 * flight -- the window the old serialised renderer never had, because it
 * refused to start one. */

type Slot = { current: boolean; ariaHidden: string | null; visibility: string; pointerEvents: string; width: number };

const slots = (page: import('@playwright/test').Page): Promise<Slot[]> => page.evaluate(() =>
  [...document.querySelectorAll('#visualizer .visualizer-slot')].map((el) => {
    const style = getComputedStyle(el);
    return {
      current: el.classList.contains('visualizer-slot--current'),
      ariaHidden: el.getAttribute('aria-hidden'),
      visibility: style.visibility,
      pointerEvents: style.pointerEvents,
      width: el.getBoundingClientRect().width,
    };
  }));

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

/* Waiting for the extra slot rather than sampling on a timer is what makes
 * these cases load-bearing: the candidate is connected and processing at that
 * instant, which is exactly the window the invariant is about, and a blind
 * timer misses it -- the fetch and export in front of it take longer than any
 * sampling budget worth spending. */
const connected = (page: import('@playwright/test').Page) =>
  page.locator('#visualizer .visualizer-slot');

async function whileCandidateIsConnected(
  page: import('@playwright/test').Page, count: number, label: string,
) {
  await expect(connected(page)).toHaveCount(count, { timeout: 90_000 });
  for (let index = 0; index < 4; index += 1) {
    if (await connected(page).count() !== count) return;   // it finished; done
    exactlyOneOnScreen(await slots(page), `${label} sample ${index}`);
    await page.waitForTimeout(100);
  }
}

test('a replacement load never takes the current model off screen', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  await page.goto('/index.html');
  await expect(page.locator('#status')).toHaveText('Model loaded', { timeout: 90_000 });
  exactlyOneOnScreen(await slots(page), 'first load');

  await page.locator('#reload').click();
  await whileCandidateIsConnected(page, 2, 'replacement connected');

  await expect(page.locator('#status')).toHaveText('Model loaded', { timeout: 90_000 });
  // The replacement is the only thing left; the predecessor was removed only
  // after the replacement's own processing event made the swap safe.
  await expect(connected(page)).toHaveCount(1);
  exactlyOneOnScreen(await slots(page), 'after replacement');
  await expect(page.locator('#error')).toBeHidden();
  expect(errors.join('\n')).not.toContain('NG0953');
  expect(errors).toEqual([]);
});

test('cancelling mid-load leaves the previous model usable and says nothing stale', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  await page.goto('/index.html');
  await expect(page.locator('#status')).toHaveText('Model loaded', { timeout: 90_000 });

  // Reload cancels whatever is in flight before starting, so the second click
  // supersedes the first candidate while it is still processing.
  await page.locator('#reload').click();
  await expect(connected(page)).toHaveCount(2, { timeout: 90_000 });
  await page.locator('#reload').click();
  // Cancellation removes authority, not the DOM connection: tearing a still
  // processing element out is precisely what throws NG0953. So it is still
  // there immediately afterwards -- and still off screen.
  expect(await connected(page).count(), 'the cancelled candidate was torn out').toBe(2);
  for (let index = 0; index < 6; index += 1) {
    exactlyOneOnScreen(await slots(page), `cancelled candidate retained ${index}`);
    await page.waitForTimeout(100);
  }

  await expect(page.locator('#status')).toHaveText('Model loaded', { timeout: 90_000 });
  await expect(connected(page)).toHaveCount(1, { timeout: 90_000 });
  exactlyOneOnScreen(await slots(page), 'after cancellation settled');
  // A cancelled load reports nothing at all -- neither a status nor an error.
  await expect(page.locator('#error')).toBeHidden();
  expect(errors.join('\n')).not.toContain('NG0953');
  expect(errors).toEqual([]);
});
