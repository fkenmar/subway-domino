/**
 * The committed WP1 geo file (public/data/network_geo.json, §2.7) through the
 * same code paths the page uses: validation and lookup tables, the positioning
 * rule for every edge, preset targets, and hit-testing precision on the dense
 * real geometry. Skipped when the file has not been built.
 */
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { Camera } from "@/lib/camera";
import { isValidDisruption, targetAnchor } from "@/lib/disruptions";
import { makePolyline, type PointOnLine, pointAt } from "@/lib/geometry";
import { buildNetwork, EDGE_YARD_LEAD, type Network, validateGeo } from "@/lib/network";
import { locate, makePose } from "@/lib/positioning";
import { HitTester } from "@/map/hitTest";
import { tierFor } from "@/map/style";

const GEO_PATH = fileURLToPath(new URL("../../public/data/network_geo.json", import.meta.url));
const HAVE_GEO = existsSync(GEO_PATH);

describe.skipIf(!HAVE_GEO)("real network_geo.json (WP1)", () => {
  const net: Network = HAVE_GEO ? buildNetwork(validateGeo(JSON.parse(readFileSync(GEO_PATH, "utf8")))) : (null as never);

  it("matches the §2.7 scope: 28 routes, 475 stations, 424 complexes", () => {
    expect(net.routes).toHaveLength(28);
    expect(net.stations).toHaveLength(475);
    expect(net.complexes).toHaveLength(424);
    expect(net.segments.length).toBeGreaterThan(400);
    expect(net.interlockings.length).toBeGreaterThan(0);
    expect(net.yards.length).toBeGreaterThan(0);
  });

  it("positions every edge at its start, middle and end (the §2.7 rule)", () => {
    const pose = makePose();
    const [x0, y0, x1, y1] = net.bbox;
    const pad = 2000;
    const bad: number[] = [];
    for (let e = 0; e < net.edges.count; e++) {
      for (const f of [0, 0.5, 1]) {
        const ok = locate(net, e, f, pose);
        const inside = pose.x >= x0 - pad && pose.x <= x1 + pad && pose.y >= y0 - pad && pose.y <= y1 + pad;
        if (!ok || !Number.isFinite(pose.x) || !Number.isFinite(pose.y) || !inside) bad.push(e);
      }
    }
    expect(bad).toEqual([]);
  });

  it("anchors yard leads on their lead station (accepted §2.7 change)", () => {
    const E = net.edges;
    let n = 0;
    for (let e = 0; e < E.count; e++) {
      if (E.kind[e] !== EDGE_YARD_LEAD) continue;
      n++;
      expect(net.stations[E.station[e]], `edge ${e}`).toBeDefined();
      expect(net.yards[E.yard[e]], `edge ${e}`).toBeDefined();
      expect([-1, 1]).toContain(E.segDir[e]);
    }
    expect(n).toBeGreaterThan(0);
  });

  it("ships presets whose disruptions are valid for their targets and whose focus is near them", () => {
    expect(net.presets.length).toBeGreaterThanOrEqual(3);
    for (const p of net.presets) {
      for (const d of p.disruptions) {
        // The server serves presets as blockages: signal problems become a local block on the same target.
        const served = d.kind === "signal_delay" ? "local_block" : d.kind;
        expect(isValidDisruption(net, served, d.target), `${p.id}: ${served}`).toBe(true);
        const a = targetAnchor(net, d.target);
        expect(a, p.id).not.toBeNull();
        expect(p.focus, p.id).toBeDefined();
        if (!a || !p.focus) continue;
        // The camera lands on the meltdown: the target is within ~half a phone screen of focus.
        const reachM = 200 * p.focus.zoom_m_per_px;
        expect(Math.hypot(a.x - p.focus.x, a.y - p.focus.y), p.id).toBeLessThan(reachM);
      }
    }
  });

  describe("hit-testing on dense real geometry", () => {
    const hit = new HitTester(net);
    const cam = new Camera();
    cam.setViewport(1200, 800);

    it("clicking a station marker selects that station (or a co-located one)", () => {
      const tier = tierFor(3);
      const misses: string[] = [];
      for (const st of net.stations) {
        cam.set({ cx: st.x, cy: st.y, mPerPx: 3 });
        const t = hit.hit(cam, tier, 600, 400, { coarse: false, chips: [] });
        const got = t?.type === "station" ? net.stations[t.idx] : undefined;
        if (!got || Math.hypot(got.x - st.x, got.y - st.y) > 3 * 2) misses.push(`${st.idx} ${st.name} → ${JSON.stringify(t)}`);
      }
      expect(misses).toEqual([]);
    });

    it("clicking a segment's centre line selects the nearest drawn segment", () => {
      // At Midtown-dense zoom, the click at the midpoint of each segment's centre line
      // must resolve to that segment, unless another segment's line runs within 1 px
      // of it (overlapping alignments) — then either is a correct answer.
      const mPerPx = 2;
      const tier = tierFor(mPerPx);
      const wrong: string[] = [];
      const pt: PointOnLine = { x: 0, y: 0, tx: 1, ty: 0 };
      for (const seg of net.segments) {
        const line = makePolyline(seg.pts);
        pointAt(line, 0.5, pt);
        cam.set({ cx: pt.x, cy: pt.y, mPerPx });
        const t = hit.hit(cam, tier, 600, 400, { coarse: false, chips: [] });
        if (t?.type === "segment" && t.idx === seg.idx) continue;
        if (t?.type === "station" || t?.type === "interlocking") continue; // short segment: marker wins
        if (t?.type === "segment") {
          const other = hit.nearestSegment(pt.x, pt.y, mPerPx);
          if (other && other.idx === t.idx) continue; // overlapping alignment within 1 px
        }
        wrong.push(`${seg.idx} ${seg.name} → ${JSON.stringify(t)}`);
      }
      expect(wrong).toEqual([]);
    });
  });
});
