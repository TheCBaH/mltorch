import { expect, test } from '@playwright/test';

test('assembled app is prefix-safe and loads its application assets', async ({ page }) => {
  const responses: string[] = [];
  page.on('response', (response) => responses.push(response.url()));
  await page.goto('/index.html');
  await expect(page.locator('h1')).toHaveText('MLTorch Model Explorer');
  await expect(page.locator('#catalogue')).toBeVisible();
  expect(responses.some((url) => url.endsWith('/worker.js'))).toBe(false);
});
