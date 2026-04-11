// Copied into the ComfyUI_frontend checkout by tests-frontend/run-tests.sh so
// that node_modules and tsconfig path aliases (@e2e/*) resolve correctly.
import { defineConfig, devices } from '@playwright/test'
import path from 'path'

const specsDir =
  process.env.BACKEND_SPECS_DIR ??
  path.join(__dirname, '..', 'specs')

export default defineConfig({
  testDir: specsDir,
  tsconfig: './tsconfig.json',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: 'html',
  use: { trace: 'on-first-retry' },
  globalSetup: './browser_tests/globalSetup.ts',
  globalTeardown: './browser_tests/globalTeardown.ts',
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
      timeout: 30_000
    }
  ]
})
