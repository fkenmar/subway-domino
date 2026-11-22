/**
 * Hit-testing for clicks and hover: disruption chips, trains, stations,
 * interlockings and segments (river crossings are segments). A uniform world
 * grid indexes station points and polyline pieces so queries stay O(nearby).
 */
import type { Camera } from "@/lib/camera";
import { distSqToSegment } from "@/lib/geometry";
import type { Network } from "@/lib/network";
import { interlockingOffset } from "./renderBase";
import type { ChipRect, HoverTarget } from "./renderDynamic";
import { bundleWidth, type ZoomTier } from "./style";

const CELL_M = 400;

export interface HitOptions {
  /** Larger tolerances for coarse pointers (touch). */
  coarse: boolean;
  chips: readonly ChipRect[];
  /** Train screen positions from the last dynamic render: [x0,y0,x1,y1,…]. */
  trainXY?: Float32Array;
  trainIds?: Uint16Array;
  trainCount?: number;
  /** Include trains (hover tooltips); clicks target infrastructure. */
  includeTrains?: boolean;
}

interface Piece {
  seg: number;
  i: number;
}

export class HitTester {
  private readonly stationCells = new Map<number, number[]>();
  private readonly pieceCells = new Map<number, Piece[]>();

  constructor(private readonly net: Network) {
    const key = HitTester.key;
    net.stations.forEach((s) => {
      const k = key(Math.floor(s.x / CELL_M), Math.floor(s.y / CELL_M));
      const list = this.stationCells.get(k);
      if (list) list.push(s.idx);
      else this.stationCells.set(k, [s.idx]);
    });
    for (const seg of net.segments) {
      const { xs, ys } = seg.line;
      for (let i = 0; i < xs.length - 1; i++) {
        const gx0 = Math.floor(Math.min(xs[i], xs[i + 1]) / CELL_M);
        const gx1 = Math.floor(Math.max(xs[i], xs[i + 1]) / CELL_M);
        const gy0 = Math.floor(Math.min(ys[i], ys[i + 1]) / CELL_M);
        const gy1 = Math.floor(Math.max(ys[i], ys[i + 1]) / CELL_M);
        for (let gx = gx0; gx <= gx1; gx++) {
          for (let gy = gy0; gy <= gy1; gy++) {
            const k = key(gx, gy);
            const list = this.pieceCells.get(k);
            const piece = { seg: seg.idx, i };
            if (list) list.push(piece);
            else this.pieceCells.set(k, [piece]);
          }
        }
      }
    }
  }

  private static key(gx: number, gy: number): number {
    return gx * 1_000_003 + gy;
  }

  /** Nearest station within `radiusM` of a world point. */
  nearestStation(wx: number, wy: number, radiusM: number): { idx: number; dist: number } | null {
    let best: { idx: number; dist: number } | null = null;
    const g0x = Math.floor((wx - radiusM) / CELL_M);
    const g1x = Math.floor((wx + radiusM) / CELL_M);
    const g0y = Math.floor((wy - radiusM) / CELL_M);
    const g1y = Math.floor((wy + radiusM) / CELL_M);
    for (let gx = g0x; gx <= g1x; gx++) {
      for (let gy = g0y; gy <= g1y; gy++) {
        for (const s of this.stationCells.get(HitTester.key(gx, gy)) ?? []) {
          const st = this.net.stations[s];
          const d = Math.hypot(st.x - wx, st.y - wy);
          if (d <= radiusM && (!best || d < best.dist)) best = { idx: s, dist: d };
        }
      }
    }
    return best;
  }

  /** Nearest segment within `radiusM` of a world point. */
  nearestSegment(wx: number, wy: number, radiusM: number): { idx: number; dist: number } | null {
    let bestSeg = -1;
    let bestD2 = radiusM * radiusM;
    const g0x = Math.floor((wx - radiusM) / CELL_M);
    const g1x = Math.floor((wx + radiusM) / CELL_M);
    const g0y = Math.floor((wy - radiusM) / CELL_M);
    const g1y = Math.floor((wy + radiusM) / CELL_M);
    for (let gx = g0x; gx <= g1x; gx++) {
      for (let gy = g0y; gy <= g1y; gy++) {
        for (const p of this.pieceCells.get(HitTester.key(gx, gy)) ?? []) {
          const { xs, ys } = this.net.segments[p.seg].line;
          const d2 = distSqToSegment(wx, wy, xs[p.i], ys[p.i], xs[p.i + 1], ys[p.i + 1]);
          if (d2 <= bestD2) {
            bestD2 = d2;
            bestSeg = p.seg;
          }
        }
      }
    }
    return bestSeg >= 0 ? { idx: bestSeg, dist: Math.sqrt(bestD2) } : null;
  }

  /** Resolve the target under a screen point (CSS px). */
  hit(camera: Camera, tier: ZoomTier, sx: number, sy: number, opts: HitOptions): HoverTarget | null {
    const tol = opts.coarse ? 16 : 8;

    for (const c of opts.chips) {
      if (sx >= c.x0 - 2 && sx <= c.x1 + 2 && sy >= c.y0 - 2 && sy <= c.y1 + 2) return { type: "disruption", id: c.id };
    }

    const m = camera.mPerPx;
    const wx = camera.toWorldX(sx);
    const wy = camera.toWorldY(sy);
    const stationTolPx = tier.hubRadius + tol;
    const st = this.nearestStation(wx, wy, stationTolPx * m);

    // A pointer on a station's own marker always means the station — even for a
    // finger-sized touch tolerance that also reaches the interlocking diamond beside it.
    if (st && st.dist / m <= tier.hubRadius + 2) return { type: "station", idx: st.idx };

    // Interlocking diamonds (drawn offset from their station at mid zoom and closer).
    if (tier.showInterlockings) {
      const off = interlockingOffset(tier);
      let best = -1;
      let bestD = tol + 2;
      for (const il of this.net.interlockings) {
        const st = this.net.stations[il.station];
        const atStation = st ? Math.hypot(st.x - il.x, st.y - il.y) < 25 : false;
        const x = camera.toScreenX(il.x) + (atStation ? off.dx : 0);
        const y = camera.toScreenY(il.y) + (atStation ? off.dy : 0);
        const d = Math.abs(x - sx) + Math.abs(y - sy); // diamond (L1) distance
        if (d < bestD) {
          bestD = d;
          best = il.idx;
        }
      }
      if (best >= 0) return { type: "interlocking", idx: best };
    }

    if (opts.includeTrains && opts.trainXY && opts.trainIds && opts.trainCount && tier.capsules) {
      const r = Math.max(6, tier.trainLength / 2 + 2);
      let best = -1;
      let bestD = r;
      for (let k = 0; k < opts.trainCount; k++) {
        const d = Math.hypot(opts.trainXY[2 * k] - sx, opts.trainXY[2 * k + 1] - sy);
        if (d < bestD) {
          bestD = d;
          best = k;
        }
      }
      if (best >= 0) return { type: "train", trainId: opts.trainIds[best] };
    }

    const seg = this.nearestSegment(wx, wy, (bundleWidth(tier, 3) / 2 + tol) * m);
    if (st && (!seg || st.dist / m <= tier.hubRadius + 6 || st.dist <= seg.dist + 2 * m)) {
      return { type: "station", idx: st.idx };
    }
    if (seg) return { type: "segment", idx: seg.idx };
    return null;
  }
}
