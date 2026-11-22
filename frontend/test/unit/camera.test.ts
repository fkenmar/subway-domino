/** Map camera (§6): pan/zoom about the cursor, zoom range, fly-to. */
import { describe, expect, it } from "vitest";
import { Camera, CONTENT_REACH, easeInOutCubic, MAX_M_PER_PX_FLOOR, MIN_M_PER_PX, zoomPath } from "@/lib/camera";

function camera(): Camera {
  const c = new Camera();
  c.setViewport(1200, 800);
  c.set({ cx: 1000, cy: -500, mPerPx: 20 });
  return c;
}

describe("Camera", () => {
  it("maps world ↔ screen consistently (y up in the world, down on screen)", () => {
    const c = camera();
    expect(c.toScreenX(1000)).toBe(600);
    expect(c.toScreenY(-500)).toBe(400);
    expect(c.toScreenY(-500 + 20)).toBe(399); // north is up
    for (const [sx, sy] of [
      [0, 0],
      [37.5, 791],
      [1200, 800],
    ]) {
      expect(c.toScreenX(c.toWorldX(sx))).toBeCloseTo(sx, 9);
      expect(c.toScreenY(c.toWorldY(sy))).toBeCloseTo(sy, 9);
    }
  });

  it("zooms about the cursor: the world point under the pointer stays put", () => {
    for (const [sx, sy, factor] of [
      [100, 120, 1.25],
      [1100, 700, 0.6],
      [600, 400, 3],
      [0, 799, 1.01],
    ]) {
      const c = camera();
      const wx = c.toWorldX(sx);
      const wy = c.toWorldY(sy);
      c.zoomAt(sx, sy, factor);
      expect(c.mPerPx).toBeCloseTo(20 / factor, 9);
      expect(c.toScreenX(wx)).toBeCloseTo(sx, 6);
      expect(c.toScreenY(wy)).toBeCloseTo(sy, 6);
    }
  });

  it("clamps zoom from Midtown detail to the whole city", () => {
    const c = camera();
    c.zoomAt(600, 400, 1e6);
    expect(c.mPerPx).toBe(MIN_M_PER_PX);
    expect(MIN_M_PER_PX).toBeLessThanOrEqual(1.5); // dense Midtown (≈ 1.5 m/px) is reachable
    c.zoomAt(600, 400, 1e-6);
    expect(c.mPerPx).toBe(MAX_M_PER_PX_FLOOR);
    expect(MAX_M_PER_PX_FLOOR).toBeGreaterThanOrEqual(60); // full NYC (≈ 60 m/px) is reachable
    const v = c.version;
    c.zoomAt(600, 400, 1e-6); // already at the limit: no change, no redraw
    expect(c.version).toBe(v);
  });

  it("widens the zoom range so the whole extent fits on small screens", () => {
    const c = new Camera();
    c.setViewport(390, 700);
    const nyc: [number, number, number, number] = [-17000, -21000, 18000, 21000];
    c.configureForExtent(nyc);
    expect(c.maxMPerPx).toBeGreaterThanOrEqual(c.fitScale(nyc, 8));
    c.fitBbox(nyc, 8);
    expect(c.toScreenX(nyc[0])).toBeGreaterThanOrEqual(0);
    expect(c.toScreenX(nyc[2])).toBeLessThanOrEqual(390);
  });

  it("pans with the pointer (drag moves the content, not the camera)", () => {
    const c = camera();
    const wx = c.toWorldX(300);
    const wy = c.toWorldY(200);
    c.panBy(50, -30);
    expect(c.toScreenX(wx)).toBeCloseTo(350, 9);
    expect(c.toScreenY(wy)).toBeCloseTo(170, 9);
  });

  it("keeps the centre near the network bounds", () => {
    const c = camera();
    c.setBounds([0, 0, 10_000, 10_000]);
    c.panBy(1e7, -1e7);
    const [x0, , x1] = c.visibleWorld();
    expect(x1).toBeGreaterThan(-20_000);
    expect(x0).toBeLessThan(30_000);
  });

  it("flies to a target and lands exactly on it", () => {
    const c = camera();
    const target = { cx: -3164, cy: 4722, mPerPx: 2 };
    c.flyTo(target, 0, { maxDurationMs: 1000 });
    expect(c.animating).toBe(true);
    let t = 0;
    while (c.step(t)) {
      expect(c.mPerPx).toBeGreaterThanOrEqual(c.minMPerPx);
      expect(c.mPerPx).toBeLessThanOrEqual(c.maxMPerPx);
      t += 16;
      if (t > 5000) throw new Error("flight never finished");
    }
    expect(c.animating).toBe(false);
    expect(c.cx).toBeCloseTo(target.cx, 3);
    expect(c.cy).toBeCloseTo(target.cy, 3);
    expect(c.mPerPx).toBeCloseTo(target.mPerPx, 6);
  });

  it("cancels a flight on user input", () => {
    const c = camera();
    c.flyTo({ cx: 0, cy: 0, mPerPx: 5 }, 0);
    c.panBy(10, 0);
    expect(c.animating).toBe(false);
  });
});

describe("zoomPath (van Wijk & Nuij)", () => {
  it("starts and ends at the requested views", () => {
    const path = zoomPath([0, 0, 1000], [5000, -2000, 200]);
    const [x0, y0, w0] = path.at(0);
    const [x1, y1, w1] = path.at(1);
    expect(x0).toBeCloseTo(0, 9);
    expect(y0).toBeCloseTo(0, 9);
    expect(w0).toBeCloseTo(1000, 9);
    expect(x1).toBeCloseTo(5000, 6);
    expect(y1).toBeCloseTo(-2000, 6);
    expect(w1).toBeCloseTo(200, 6);
    expect(path.durationMs).toBeGreaterThan(0);
  });

  it("zooms out mid-flight for long pans", () => {
    const path = zoomPath([0, 0, 1000], [50_000, 0, 1000]);
    expect(path.at(0.5)[2]).toBeGreaterThan(1000);
  });

  it("handles a pure zoom", () => {
    const path = zoomPath([10, 10, 1000], [10, 10, 250]);
    expect(path.at(1)[2]).toBeCloseTo(250, 6);
    expect(path.at(0.5)[0]).toBe(10);
  });

  it("eases symmetrically", () => {
    expect(easeInOutCubic(0)).toBe(0);
    expect(easeInOutCubic(1)).toBe(1);
    expect(easeInOutCubic(0.5)).toBeCloseTo(0.5);
    expect(easeInOutCubic(0.25) + easeInOutCubic(0.75)).toBeCloseTo(1);
  });
});

describe("content clamp (the screen never shows an empty map)", () => {
  // One straight line along y = 0 from x = 0 to x = 10 km.
  const probe = (x: number, y: number) => {
    const qx = Math.min(10_000, Math.max(0, x));
    return { x: qx, y: 0, d: Math.hypot(x - qx, y) };
  };
  const make = () => {
    const cam = new Camera();
    cam.setViewport(1000, 800);
    cam.setContentProbe(probe);
    return cam;
  };

  it("stops zooming in once the line would leave the screen", () => {
    const cam = make();
    cam.set({ cx: 5000, cy: 3000, mPerPx: 20 }); // line 3 km below the centre: on screen at 20 m/px
    for (let i = 0; i < 40; i++) cam.zoomAt(500, 400, 1.5);
    const reach = CONTENT_REACH * 800 * cam.mPerPx;
    expect(Math.abs(cam.cy)).toBeLessThanOrEqual(reach + 1);
    expect(cam.mPerPx).toBeGreaterThan(MIN_M_PER_PX);
  });

  it("still zooms all the way in on top of the network", () => {
    const cam = make();
    cam.set({ cx: 5000, cy: 0, mPerPx: 20 });
    for (let i = 0; i < 40; i++) cam.zoomAt(500, 400, 1.5);
    expect(cam.mPerPx).toBeCloseTo(MIN_M_PER_PX);
  });

  it("pulls a pan back towards the network", () => {
    const cam = make();
    cam.set({ cx: 5000, cy: 0, mPerPx: 2 });
    cam.panBy(0, 5000); // drag the map far down: the centre heads north, away from the line
    expect(Math.abs(cam.cy)).toBeLessThanOrEqual(CONTENT_REACH * 800 * 2 + 1e-6);
  });
});
