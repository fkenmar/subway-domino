/**
 * Visual verification (dev only): builds the app against the protocol v1 mock
 * and captures screenshots at desktop and phone widths (test/e2e).
 *
 *   npm run screenshots                       # mock on :8765, app on :3100
 *   MOCK_GEO=/path/to/geo.json npm run screenshots
 *
 * The mock must stream trains on the same geometry the page renders: the
 * committed public/data/network_geo.json when WP1 has built it, otherwise
 * MOCK_GEO (served to the page through the backend-copy fallback) or the
 * mini fixture.
 */
import { existsSync } from "node:fs";
import path from "node:path";
import { defineConfig } from "@playwright/test";

const MOCK_PORT = Number(process.env.MOCK_PORT ?? 8765);
const WEB_PORT = Number(process.env.WEB_PORT ?? 3100);
const PUBLIC_GEO = path.join(__dirname, "public", "data", "network_geo.json");
const FIXTURE_GEO = path.join(__dirname, "test", "fixtures", "mini_geo.json");
const MOCK_GEO = existsSync(PUBLIC_GEO) ? PUBLIC_GEO : (process.env.MOCK_GEO ?? FIXTURE_GEO);
const WS_URL = `ws://127.0.0.1:${MOCK_PORT}/ws`;

export default defineConfig({
  testDir: "test/e2e",
  outputDir: "test-results",
  timeout: 180_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${WEB_PORT}`,
    colorScheme: "dark",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "desktop",
      use: { browserName: "chromium", viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 },
    },
    {
      name: "laptop",
      use: { browserName: "chromium", viewport: { width: 1280, height: 760 }, deviceScaleFactor: 2 },
    },
    {
      name: "mobile",
      use: {
        browserName: "chromium",
        viewport: { width: 390, height: 844 },
        deviceScaleFactor: 3,
        isMobile: true,
        hasTouch: true,
      },
    },
  ],
  webServer: [
    {
      command: `node scripts/mock-server.mjs --port ${MOCK_PORT} --geo "${MOCK_GEO}" --seed 1 --quiet`,
      url: `http://127.0.0.1:${MOCK_PORT}/health`,
      reuseExistingServer: false,
      timeout: 30_000,
    },
    {
      // Production build: NEXT_PUBLIC_WS_URL is inlined at build time.
      command: `npm run build && npx next start -p ${WEB_PORT} -H 127.0.0.1`,
      url: `http://127.0.0.1:${WEB_PORT}/subway`,
      env: { NEXT_PUBLIC_WS_URL: WS_URL },
      reuseExistingServer: false,
      timeout: 300_000,
    },
  ],
});
