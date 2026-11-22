/** Home view (WP3b polish): the whole network centred on the visible map, clear of the floating scoreboard. */
import { describe, expect, it } from "vitest";
import type { Bbox } from "@/lib/geometry";
import { type Insets, fitWithInsets, homeView, type ScreenRect } from "@/lib/layout";

const W = 1440;
const H = 900;
const INSETS: Insets = { top: 60, right: 324, bottom: 80, left: 16 };
const SCOREBOARD: ScreenRect = { x0: 1128, y0: 56, x1: 1424, y1: 260 };

function screenX(v: { cx: number; mPerPx: number }, x: number): number {
  return (x - v.cx) / v.mPerPx + W / 2;
}

function screenY(v: { cy: number; mPerPx: number }, y: number): number {
  return (v.cy - y) / v.mPerPx + H / 2;
}

describe("homeView", () => {
  it("centres a tall network across the full width when the scoreboard covers nothing", () => {
    const bbox: Bbox = [0, 0, 1000, 2000];
    const points = [
      { x: 0, y: 0 },
      { x: 500, y: 1000 },
      { x: 1000, y: 2000 },
    ];
    const v = homeView(bbox, points, W, H, INSETS, [SCOREBOARD]);
    expect(screenX(v, 500)).toBeCloseTo((INSETS.left + W - INSETS.left) / 2, 5);
    // Same scale as the inset-aware fit: the network is height-limited.
    expect(v.mPerPx).toBeCloseTo(fitWithInsets(bbox, W, H, INSETS).mPerPx, 9);
  });

  it("slides toward the inset fit just far enough to clear the scoreboard", () => {
    const bbox: Bbox = [0, 0, 2400, 2000];
    const corner = { x: 2400, y: 2000 };
    const points = [{ x: 0, y: 0 }, corner];
    const v = homeView(bbox, points, W, H, INSETS, [SCOREBOARD]);
    const sx = screenX(v, corner.x);
    const sy = screenY(v, corner.y);
    const under = sx > SCOREBOARD.x0 - 12 && sx < SCOREBOARD.x1 + 12 && sy > SCOREBOARD.y0 - 12 && sy < SCOREBOARD.y1 + 12;
    expect(under).toBe(false);
    const inset = fitWithInsets(bbox, W, H, INSETS);
    const full = fitWithInsets(bbox, W, H, { ...INSETS, right: INSETS.left });
    // Strictly between the full-width centre and the inset fit.
    expect(v.cx).toBeGreaterThan(Math.min(inset.cx, full.cx));
    expect(v.cx).toBeLessThan(Math.max(inset.cx, full.cx));
  });

  it("falls back to the inset fit before the HUD is measured, and on phones", () => {
    const bbox: Bbox = [0, 0, 1000, 2000];
    const points = [{ x: 500, y: 1000 }];
    expect(homeView(bbox, points, W, H, INSETS, [])).toEqual(fitWithInsets(bbox, W, H, INSETS));
    const phone: Insets = { top: 140, right: 6, bottom: 200, left: 6 };
    expect(homeView(bbox, points, 390, 844, phone, [SCOREBOARD])).toEqual(fitWithInsets(bbox, 390, 844, phone));
  });
});
