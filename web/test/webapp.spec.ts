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
