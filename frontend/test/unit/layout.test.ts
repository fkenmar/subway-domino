/** HUD-aware camera framing and popover placement. */
import { describe, expect, it } from "vitest";
import { Camera } from "@/lib/camera";
import { centerWithInsets, clampInsets, fitWithInsets, type Insets, mapInsets } from "@/lib/layout";
import { placePopover } from "@/lib/popover";

const W = 1440;
const H = 900;
const insets: Insets = { top: 120, right: 380, bottom: 90, left: 20 };

function cameraAt(state: { cx: number; cy: number; mPerPx: number }): Camera {
  const c = new Camera();
  c.setViewport(W, H);
  c.set(state);
  return c;
}

describe("framing inside the uncovered map area", () => {
  it("centres a point in the area the HUD leaves free", () => {
    const c = cameraAt(centerWithInsets(500, -200, 4, insets));
    expect(c.toScreenX(500)).toBeCloseTo(insets.left + (W - insets.left - insets.right) / 2, 6);
    expect(c.toScreenY(-200)).toBeCloseTo(insets.top + (H - insets.top - insets.bottom) / 2, 6);
  });

  it("fits a bbox inside the free area", () => {
    const bbox: [number, number, number, number] = [-5000, -3000, 9000, 6000];
    const c = cameraAt(fitWithInsets(bbox, W, H, insets, 12));
    expect(c.toScreenX(bbox[0])).toBeGreaterThanOrEqual(insets.left + 11.9);
    expect(c.toScreenX(bbox[2])).toBeLessThanOrEqual(W - insets.right - 11.9);
    expect(c.toScreenY(bbox[3])).toBeGreaterThanOrEqual(insets.top + 11.9);
    expect(c.toScreenY(bbox[1])).toBeLessThanOrEqual(H - insets.bottom - 11.9);
  });

  it("never lets the HUD cover the whole map", () => {
    const tight = clampInsets({ top: 500, right: 300, bottom: 500, left: 300 }, 390, 844);
    expect(390 - tight.left - tight.right).toBeGreaterThanOrEqual(160 - 1e-9);
    expect(844 - tight.top - tight.bottom).toBeGreaterThanOrEqual(0.35 * 844 - 1e-9);
    const roomy = clampInsets(insets, W, H);
    expect(roomy).toEqual(insets);
  });

  it("estimates insets per breakpoint before the HUD is measured", () => {
    expect(mapInsets(1440, 900).right).toBeGreaterThan(300); // side column on desktop
    expect(mapInsets(390, 844).bottom).toBeGreaterThan(mapInsets(1440, 900).bottom); // bottom dock on phones
  });
});

describe("placePopover", () => {
  const size = { w: 300, h: 220 };
  const viewport = { w: W, h: H };

  it("prefers above the anchor, with the arrow pointing at it", () => {
    const p = placePopover({ anchor: { x: 700, y: 600 }, size, viewport, insets });
    expect(p.arrow.side).toBe("bottom");
    expect(p.top + size.h).toBeLessThanOrEqual(600);
    expect(p.left + p.arrow.offset).toBeCloseTo(700);
  });

  it("goes below when there is no room above", () => {
    const p = placePopover({ anchor: { x: 700, y: 160 }, size, viewport, insets });
    expect(p.arrow.side).toBe("top");
    expect(p.top).toBeGreaterThan(160);
  });

  it("goes beside the anchor when neither fits, and stays on screen", () => {
    const small = { w: 390, h: 500 };
    const p = placePopover({ anchor: { x: 40, y: 250 }, size: { w: 300, h: 420 }, viewport: small, insets: mapInsets(390, 500) });
    expect(["left", "right"]).toContain(p.arrow.side);
    expect(p.left).toBeGreaterThanOrEqual(8);
    expect(p.left + 300).toBeLessThanOrEqual(390 - 8 + 1e-9);
  });

  it("clamps horizontally near the viewport edge but keeps the arrow on the anchor", () => {
    const p = placePopover({ anchor: { x: W - 20, y: 600 }, size, viewport, insets });
    expect(p.left + size.w).toBeLessThanOrEqual(W - 8);
    expect(p.arrow.offset).toBeLessThanOrEqual(size.w - 16);
  });
});
