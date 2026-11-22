/**
 * Map styling rules (§6): zoom tiers, bundle geometry, the crowding ramp, label
 * candidates, and hit-testing of stations / segments / interlockings / chips.
 */
import { describe, expect, it } from "vitest";
import { Camera } from "@/lib/camera";
import { HitTester } from "@/map/hitTest";
import { interlockingOffset, labelCandidates } from "@/map/renderBase";
import {
  bundleOffset,
  bundleWidth,
  CROWD_ALERT,
  CROWD_CALM,
  crowdColor,
  PALETTE,
  tierFor,
  ZOOM_TIERS,
} from "@/map/style";
import { miniNetwork } from "../helpers/geo";

const net = miniNetwork();

describe("zoom tiers", () => {
  it("cover full NYC (~60 m/px) to dense Midtown (~1.5 m/px), widest first", () => {
    expect(tierFor(60).name).toBe("city");
    expect(tierFor(1.5).name).toBe("street");
    for (let i = 1; i < ZOOM_TIERS.length; i++) {
      expect(ZOOM_TIERS[i].minMPerPx).toBeLessThan(ZOOM_TIERS[i - 1].minMPerPx);
    }
    expect(tierFor(0.1)).toBe(ZOOM_TIERS.at(-1));
  });

  it("only grow marker and line sizes (screen pixels) as the map zooms in", () => {
    for (let i = 1; i < ZOOM_TIERS.length; i++) {
      const [a, b] = [ZOOM_TIERS[i - 1], ZOOM_TIERS[i]];
      expect(b.lineWidth).toBeGreaterThanOrEqual(a.lineWidth);
      expect(b.stationRadius).toBeGreaterThanOrEqual(a.stationRadius);
      expect(b.trainLength).toBeGreaterThanOrEqual(a.trainLength);
      expect(b.labelMaxRank).toBeGreaterThanOrEqual(a.labelMaxRank);
    }
  });

  it("shows interlockings and yards from mid zoom, route letters on trains at high zoom", () => {
    expect(tierFor(60).showInterlockings).toBe(false);
    expect(tierFor(10).showInterlockings).toBe(true);
    expect(tierFor(10).showYards).toBe(true);
    expect(tierFor(10).showTrainLabels).toBe(false);
    expect(tierFor(2).showTrainLabels).toBe(true);
  });
});

describe("segment bundles", () => {
  it("lays route-colour strokes side by side, centred on the track", () => {
    const tier = tierFor(5);
    expect(bundleWidth(tier, 1)).toBe(tier.lineWidth);
    expect(bundleWidth(tier, 3)).toBeCloseTo(3 * tier.lineWidth + 2 * tier.lineGap);
    const offsets = [0, 1, 2].map((i) => bundleOffset(tier, i, 3));
    expect(offsets[1]).toBe(0);
    expect(offsets[0]).toBeCloseTo(-offsets[2]);
    expect(offsets[2] - offsets[1]).toBeCloseTo(tier.lineWidth + tier.lineGap);
  });
});

describe("crowding ramp", () => {
  it("is calm below 40 %, amber → red up to 80 %, red at and above", () => {
    expect(crowdColor(0)).toBeNull();
    expect(crowdColor(CROWD_CALM - 1)).toBeNull();
    expect(crowdColor(CROWD_CALM)).toBe("rgb(255, 196, 32)");
    expect(crowdColor(CROWD_ALERT)).toBe(PALETTE.red);
    expect(crowdColor(254)).toBe(PALETTE.red);
    const mid = crowdColor((CROWD_CALM + CROWD_ALERT) / 2) ?? "";
    const green = Number(/rgb\(255, (\d+)/.exec(mid)?.[1]);
    expect(green).toBeLessThan(196);
    expect(green).toBeGreaterThan(96);
  });
});

describe("label candidates", () => {
  it("labels each complex once on wide tiers", () => {
    const names = labelCandidates(net, { labelAllStations: false }).map((s) => s.name);
    expect(names).toHaveLength(net.complexes.length);
    expect(names).not.toContain("42 St-Port Authority Bus Terminal");
  });

  it("adds differently named stations of a complex on dense tiers, once per name", () => {
    const dense = labelCandidates(net, { labelAllStations: true });
    const names = dense.map((s) => s.name);
    expect(names).toContain("42 St-Port Authority Bus Terminal");
    expect(names.filter((n) => n === "Times Sq-42 St")).toHaveLength(1);
    expect(names).toContain("8 Av"); // 14 St / 8 Av complex: "14 St" + "8 Av"
    expect(new Set(dense.map((s) => s.idx)).size).toBe(dense.length);
  });
});

describe("hit-testing", () => {
  const hit = new HitTester(net);

  function cameraAt(x: number, y: number, mPerPx: number): Camera {
    const c = new Camera();
    c.setViewport(1000, 800);
    c.set({ cx: x, cy: y, mPerPx });
    return c;
  }

  it("indexes stations and polyline pieces for nearby queries", () => {
    const ts = net.stations[0];
    expect(hit.nearestStation(ts.x + 3, ts.y - 4, 20)?.idx).toBe(0);
    expect(hit.nearestStation(ts.x + 300, ts.y, 20)).toBeNull();
    const seg = net.segments[5]; // Wall St – Clark St (river crossing)
    const [mx, my] = seg.pts[1];
    expect(hit.nearestSegment(mx + 5, my, 30)?.idx).toBe(5);
  });

  it("resolves stations, river crossings and nothing at all", () => {
    const ts = net.stations[0];
    const tier = tierFor(6);
    const cam = cameraAt(ts.x, ts.y, 6);
    expect(hit.hit(cam, tier, 500, 400, { coarse: false, chips: [] })).toEqual({ type: "station", idx: 0 });

    const [mx, my] = net.segments[5].pts[1];
    const cam2 = cameraAt(mx, my, 6);
    expect(hit.hit(cam2, tier, 500, 400, { coarse: false, chips: [] })).toEqual({ type: "segment", idx: 5 });
    expect(hit.hit(cam2, tier, 500 + 200, 400 - 200, { coarse: false, chips: [] })).toBeNull();
  });

  it("finds interlocking diamonds drawn beside their station from mid zoom", () => {
    const il = net.interlockings[0];
    const tier = tierFor(6);
    const cam = cameraAt(il.x, il.y, 6);
    const off = interlockingOffset(tier);
    expect(hit.hit(cam, tier, 500 + off.dx, 400 + off.dy, { coarse: false, chips: [] })).toEqual({
      type: "interlocking",
      idx: 0,
    });
    // A tap on the station's own marker is the station, even with a finger-sized tolerance
    // that also reaches the diamond beside it.
    const street = tierFor(1.5);
    const close = cameraAt(il.x, il.y, 1.5);
    expect(hit.hit(close, street, 500, 400, { coarse: true, chips: [] })).toEqual({ type: "station", idx: il.station });
    const offStreet = interlockingOffset(street);
    expect(hit.hit(close, street, 500 + offStreet.dx, 400 + offStreet.dy, { coarse: true, chips: [] })).toEqual({
      type: "interlocking",
      idx: 0,
    });
    // At city zoom interlockings are hidden, so the same spot is the station.
    const city = tierFor(60);
    expect(hit.hit(cameraAt(il.x, il.y, 60), city, 500, 400, { coarse: false, chips: [] })?.type).toBe("station");
  });

  it("gives countdown chips priority, with touch-sized tolerance for coarse pointers", () => {
    const ts = net.stations[0];
    const tier = tierFor(6);
    const cam = cameraAt(ts.x, ts.y, 6);
    const chips = [{ id: 9, x0: 480, y0: 380, x1: 540, y1: 396 }];
    expect(hit.hit(cam, tier, 500, 390, { coarse: false, chips })).toEqual({ type: "disruption", id: 9 });

    // The most isolated station without an interlocking: only the touch tolerance reaches it.
    const isolated = net.stations
      .filter((s) => net.stationInterlocking[s.idx] < 0)
      .map((s) => ({
        s,
        gap: Math.min(...net.stations.filter((o) => o !== s).map((o) => Math.hypot(o.x - s.x, o.y - s.y))),
      }))
      .sort((a, b) => b.gap - a.gap)[0].s;
    const cam2 = cameraAt(isolated.x, isolated.y, 6);
    const sx = 500 + tier.hubRadius + 12;
    expect(hit.hit(cam2, tier, sx, 400, { coarse: false, chips: [] })?.type).not.toBe("station");
    expect(hit.hit(cam2, tier, sx, 400, { coarse: true, chips: [] })).toEqual({ type: "station", idx: isolated.idx });
  });
});
