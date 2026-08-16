import { defineConfig } from '@playwright/test';

// The spike serves web/src over http rather than file://: the visualizer loads
// a worker and font textures, and both are blocked under file:// in a way that
// would make every answer below "no" for the wrong reason.
export default defineConfig({
  testDir: './test',
  testMatch: /(?:spike|session|lifecycle)\.spec\.ts/,
  timeout: 120_000,
  reporter: [['list']],
  use: { baseURL: 'http://127.0.0.1:8123' },
  webServer: {
    command: 'npm run serve',
    url: 'http://127.0.0.1:8123/index.html',
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
