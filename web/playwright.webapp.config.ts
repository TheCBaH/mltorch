import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './test',
  testMatch: 'webapp.spec.ts',
  timeout: 120_000,
  reporter: [['list']],
  use: { baseURL: 'http://127.0.0.1:8124' },
  webServer: {
    command: 'npm run serve:webapp',
    url: 'http://127.0.0.1:8124/index.html',
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
