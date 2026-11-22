/**
 * Visual verification tour against the protocol v1 mock (see playwright.config.ts).
 * Each project (desktop / laptop / mobile) walks the first-use flow — land on the
 * live map, break the city with a preset, hand control to the AI, zoom into
 * Midtown, open a disruption popover and the info modal — saving screenshots to
 * test/screenshots/ and asserting the contract-level UX along the way.
 */
import path from "node:path";
import { expect, type Page, test } from "@playwright/test";

const SHOTS = path.join(__dirname, "..", "screenshots");
const WS_PATTERN = /\/ws$/;

/** Wait for running finite CSS animations (entrances, fades) — not the infinite pulses. */
async function settle(page: Page): Promise<void> {
  await page.evaluate(async () => {
    const finite = document.getAnimations().filter((a) => {
      const timing = a.effect?.getComputedTiming();
      return a.playState === "running" && timing !== undefined && timing.iterations !== Number.POSITIVE_INFINITY;
    });
    await Promise.race([
      Promise.all(finite.map((a) => a.finished.catch(() => undefined))),
      new Promise((resolve) => setTimeout(resolve, 2000)),
    ]);
  });
}

async function shot(page: Page, name: string): Promise<void> {
  await settle(page);
  await page.screenshot({ path: path.join(SHOTS, `${test.info().project.name}-${name}.png`) });
}

/** The static map is painted and the first live frame has arrived. */
async function waitForLive(page: Page): Promise<void> {
  await expect(page.locator('[data-ready="true"]')).toBeAttached();
  await expect(page.getByLabel("Simulation time")).not.toContainText("--:--");
}

/** Centre of the map area the HUD leaves uncovered (mirrors App.useMeasuredInsets + clampInsets). */
async function freeAreaCenter(page: Page): Promise<{ x: number; y: number }> {
  return page.evaluate(() => {
    const W = window.innerWidth;
    const H = window.innerHeight;
    const wide = W >= 700;
    let top = 0;
    let bottom = 0;
    let right = 0;
    for (const node of document.querySelectorAll<HTMLElement>("[data-inset]")) {
      const r = node.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      const role = node.dataset.inset;
      if (role === "top" || (role === "side" && !wide)) top = Math.max(top, r.bottom);
      else if (role === "bottom") bottom = Math.max(bottom, H - r.top);
      else if (role === "side") right = Math.max(right, W - r.left);
    }
    const fix = (a: number, b: number, size: number): [number, number] => {
      const free = Math.min(size, Math.max(160, size * 0.35));
      const excess = a + b - (size - free);
      if (excess <= 0) return [a, b];
      const k = (size - free) / Math.max(1, a + b);
      return [a * k, b * k];
    };
    const [l, r] = fix(wide ? 16 : 6, right + (wide ? 12 : 6), W);
    const [t, b] = fix(top + 8, bottom + 8, H);
    return { x: l + (W - l - r) / 2, y: t + (H - t - b) / 2 };
  });
}

test.describe("Subway Domino tour", () => {
  test("first use: preset meltdown, AI takeover, Midtown detail", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (err) => errors.push(String(err)));
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    const geoRequests: string[] = [];
    page.on("request", (req) => {
      if (req.url().includes("network_geo.json")) geoRequests.push(new URL(req.url()).pathname);
    });

    await page.goto("/subway");
    await waitForLive(page);
    // Static asset URLs go through asset(): the committed geo file loads under the base path
    // (no fallback to the backend's copy).
    expect(geoRequests).toEqual(["/subway/data/network_geo.json"]);
    await expect(page.locator('a[href="https://kenmar.dev"]')).toBeVisible();
    await page.waitForTimeout(1500);
    await shot(page, "01-citywide-first-visit");

    // 1 — Break the city: one menu holds every preset; pick the first.
    const breakButton = page.getByRole("button", { name: /Break the city/ });
    await breakButton.click();
    const presets = page.getByRole("dialog", { name: "Break the city" }).getByRole("navigation", { name: "Meltdown presets" });
    expect(await presets.getByRole("button").count()).toBeGreaterThan(3);
    await presets.getByRole("button").first().click();
    // The same menu lists the active disruptions with clear buttons.
    await breakButton.click();
    await expect(page.getByRole("region", { name: "Active disruptions" }).getByRole("button", { name: /^Clear/ }).first()).toBeAttached();
    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog", { name: "Break the city" })).toBeHidden();
    await expect(page.getByRole("radiogroup", { name: "Who dispatches the trains" })).toHaveAttribute("data-pulse", "true");
    await page.waitForTimeout(6000);
    await shot(page, "02-preset-normal");

    // 2 — Hand control to the AI: twin comparison + callouts.
    await page.getByRole("radio", { name: /AI Dispatcher/ }).click();
    const hud = page.getByRole("region", { name: "Live metrics" });
    await expect(hud).toHaveAttribute("data-mode", "ai");
    await expect(hud).toContainText(/since AI took over/);
    await page.waitForTimeout(7000);
    await shot(page, "03-preset-ai");

    // 3 — Dense Midtown: search flies to Times Sq (3 m/px), zoom in once (1.5 m/px).
    await page.getByRole("button", { name: "Search stations and lines" }).click();
    const search = page.getByRole("combobox");
    await search.fill("Times Sq");
    await expect(page.getByRole("option").first()).toContainText("Times Sq");
    await search.press("Enter");
    await page.waitForTimeout(2200);
    const c = await freeAreaCenter(page);
    const zoomIn = page.getByRole("button", { name: "Zoom in" });
    if (await zoomIn.isVisible()) {
      await zoomIn.click();
    } else {
      // Phones pinch instead of using buttons; a wheel step at the centre zooms 2× about it.
      await page.mouse.move(c.x, c.y);
      await page.mouse.wheel(0, -315);
      // Leave the canvas so no mouse-hover tooltip shows (real phones have no hover).
      await page.getByRole("region", { name: "Live metrics" }).hover();
    }
    await page.waitForTimeout(1800);
    await shot(page, "04-midtown-ai");

    // 4 — Click (tap) the station under the centre: the popover offers only valid kinds.
    if (test.info().project.use.hasTouch) await page.touchscreen.tap(c.x, c.y);
    else await page.mouse.click(c.x, c.y);
    const popover = page.getByRole("dialog", { name: /Times Sq/ });
    await expect(popover).toBeVisible();
    await expect(popover.getByRole("radiogroup", { name: "Duration" }).getByRole("radio")).toHaveText([
      "Until cleared",
      "10 min",
      "20 min",
      "30 min",
    ]);
    await expect(popover).not.toContainText("Signal problems"); // not valid on a station (§3.5)
    await page.waitForTimeout(500);
    await shot(page, "05-disruption-popover");
    await popover.getByRole("button", { name: "Close", exact: true }).click();

    // 5 — Methodology modal: datasets with links, reward terms, actions and masks.
    await page.getByRole("button", { name: "How it works" }).click();
    const modal = page.getByRole("dialog", { name: "Subway Domino" });
    await expect(modal).toBeVisible();
    await page.waitForTimeout(400);
    await shot(page, "06-info-modal");
    for (const id of ["fgm6-ccue", "5f5g-n3cz", "5wq4-mkjj"]) {
      if ((await modal.locator(`a[href*="${id}"]`).count()) === 0) {
        // Datasets may live on another tab.
        for (const tab of await modal.getByRole("tab").all()) {
          await tab.click();
          if ((await modal.locator(`a[href*="${id}"]`).count()) > 0) break;
        }
      }
      await expect(modal.locator(`a[href*="${id}"]`).first()).toBeAttached();
    }

    expect(errors, errors.join("\n")).toEqual([]);
  });
});

test.describe("connection lifecycle banners", () => {
  test.skip(({ isMobile }) => isMobile, "covered at desktop width");

  test("connecting, at capacity, idle pause, reconnecting", async ({ context }) => {
    /** A fresh tab whose WebSocket is driven by `handler` (each tab has its own route and session storage). */
    const tab = async (handler: Parameters<Page["routeWebSocket"]>[1]): Promise<Page> => {
      const p = await context.newPage();
      await p.routeWebSocket(WS_PATTERN, handler);
      await p.goto("/subway");
      await expect(p.locator('[data-ready="true"]')).toBeAttached();
      return p;
    };

    // Connecting: the socket opens but the server never welcomes.
    const connecting = await tab(() => {});
    await expect(connecting.getByRole("status").filter({ hasText: "Connecting to the live simulation" })).toBeVisible();
    await connecting.waitForTimeout(300);
    await shot(connecting, "07-connecting");
    await connecting.close();

    // At capacity: friendly card with a jittered 30-60 s automatic retry.
    const capacity = await tab((ws) => {
      ws.send(JSON.stringify({ type: "at_capacity", message: "Subway Domino is at capacity right now — please try again in a minute." }));
      ws.close({ code: 1013, reason: "at capacity" });
    });
    const full = capacity.getByRole("alert").filter({ hasText: "The control room is full" });
    await expect(full).toBeVisible();
    await expect(full).toContainText(/Trying again automatically in (3\d|4\d|5\d|60) s/);
    await shot(capacity, "08-at-capacity");
    await capacity.close();

    // Idle timeout: a real mock session, then the server's idle_timeout.
    const server: { idleTimeout?: () => void } = {};
    const idle = await tab((ws) => {
      ws.connectToServer();
      server.idleTimeout = () => {
        ws.send(JSON.stringify({ type: "idle_timeout" }));
        ws.close({ code: 1000, reason: "idle timeout" });
      };
    });
    await waitForLive(idle);
    await idle.waitForTimeout(1500);
    expect(server.idleTimeout).toBeDefined();
    server.idleTimeout?.();
    const paused = idle.getByRole("button", { name: /Paused — click to resume/ });
    await expect(paused).toBeVisible();
    await expect(paused).toContainText("5 min");
    await shot(idle, "09-idle-paused");
    await paused.click(); // resumes through the same route (forwarded to the mock)
    await expect(paused).toBeHidden();
    await waitForLive(idle);
    await idle.close();

    // Reconnecting: the server drops the connection, then refuses to welcome.
    let connections = 0;
    const lost = await tab((ws) => {
      connections++;
      if (connections === 1) {
        ws.connectToServer();
        setTimeout(() => ws.close({ code: 1011, reason: "server restart" }), 2000);
      } else {
        ws.close({ code: 1011, reason: "still restarting" });
      }
    });
    await waitForLive(lost);
    await expect(lost.getByRole("status").filter({ hasText: /Connection lost/ })).toBeVisible({ timeout: 15_000 });
    await shot(lost, "10-reconnecting");
    await lost.close();
  });
});
