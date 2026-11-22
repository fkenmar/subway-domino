import { describe, expect, it } from "vitest";
import {
  easeDistance,
  type EdgeAdjacency,
  type EdgeGraph,
  edgesContinue,
  fadeCurve,
  FrameBuffer,
  InterpolatedTrains,
  interpolateTrain,
  MAX_PATH_HOPS,
  PathFinder,
  plausibleDistanceM,
  RESET_CLOCK_JUMP_S,
  RESET_FADE_MS,
} from "@/lib/interp";
import { EDGE_CROSSOVER, EDGE_RUN } from "@/lib/network";
import { decodeFrame, type Frame } from "@/lib/protocol";
import { encodeFrame, type FrameInput, type TrainRow } from "../helpers/frameEncoder";

/** Edges: 0: S0→S1 (100 m), 1: S1→S2 (300 m), 2: S5→S6 (200 m, unrelated). */
const ADJ: EdgeAdjacency = {
  count: 3,
  fromStation: Int32Array.from([0, 1, 5]),
  toStation: Int32Array.from([1, 2, 6]),
  lengthM: Float64Array.from([100, 300, 200]),
};

const out = () => ({ edge: -1, frac: -1 });

describe("interpolateTrain", () => {
  it("lerps frac on the same edge", () => {
    expect(interpolateTrain(0, 0.2, 0, 0.6, 0, ADJ, out())).toEqual({ edge: 0, frac: 0.2 });
    expect(interpolateTrain(0, 0.2, 0, 0.6, 0.5, ADJ, out()).frac).toBeCloseTo(0.4);
    expect(interpolateTrain(0, 0.2, 0, 0.6, 1, ADJ, out())).toEqual({ edge: 0, frac: 0.6 });
  });

  it("clamps alpha to [0, 1]", () => {
    expect(interpolateTrain(0, 0.2, 0, 0.6, -3, ADJ, out()).frac).toBeCloseTo(0.2);
    expect(interpolateTrain(0, 0.2, 0, 0.6, 7, ADJ, out()).frac).toBeCloseTo(0.6);
  });

  it("snaps when frac moves backwards on the same edge (slot reused)", () => {
    expect(interpolateTrain(0, 0.9, 0, 0.1, 0.5, ADJ, out())).toEqual({ edge: 0, frac: 0.1 });
  });

  it("snaps to the new frame when the edge changed to an unrelated edge (contract)", () => {
    expect(interpolateTrain(0, 0.5, 2, 0.3, 0.25, ADJ, out())).toEqual({ edge: 2, frac: 0.3 });
    expect(interpolateTrain(0, 0.5, 1, 0.3, 0.25, null, out())).toEqual({ edge: 1, frac: 0.3 });
  });

  it("moves at constant speed across a directly continuing edge", () => {
    // 20 m left on edge 0 + 60 m into edge 1 = 80 m in one frame interval.
    const a = interpolateTrain(0, 0.8, 1, 0.2, 0.125, ADJ, out()); // 10 m
    expect(a.edge).toBe(0);
    expect(a.frac).toBeCloseTo(0.9);
    const b = interpolateTrain(0, 0.8, 1, 0.2, 0.5, ADJ, out()); // 40 m → 20 m into edge 1
    expect(b.edge).toBe(1);
    expect(b.frac).toBeCloseTo(20 / 300);
    expect(interpolateTrain(0, 0.8, 1, 0.2, 1, ADJ, out())).toEqual({ edge: 1, frac: 0.2 });
  });

  it("recognises continuing edges from the geo topology", () => {
    expect(edgesContinue(ADJ, 0, 1)).toBe(true);
    expect(edgesContinue(ADJ, 1, 0)).toBe(false);
    expect(edgesContinue(ADJ, 0, 2)).toBe(false);
    expect(edgesContinue(ADJ, -1, 1)).toBe(false);
    expect(edgesContinue(ADJ, 0, 99)).toBe(false);
  });
});

function frame(seq: number, clockS: number, trains: TrainRow[], crowd: number[] = [10, 255]): Frame {
  const input: FrameInput = {
    seq,
    clockS,
    mode: "normal",
    speed: "1x",
    trains,
    stationCrowd: crowd,
    stationWait: crowd.map(() => 0),
  };
  return decodeFrame(encodeFrame(input));
}

const t = (id: number, edge: number, fracNorm: number, route = 1): TrainRow => ({
  id,
  edge,
  frac: Math.round(fracNorm * 65535),
  route,
  status: 0,
  load: 50,
});

describe("FrameBuffer", () => {
  it("interpolates between the last two frames by arrival time", () => {
    const fb = new FrameBuffer(4); // 250 ms interval
    fb.push(frame(1, 100, [t(7, 0, 0.2)]), 1000);
    fb.push(frame(2, 115, [t(7, 0, 0.6)]), 1250);
    const trains = new InterpolatedTrains(4);
    expect(fb.sample(1250, ADJ, trains)).toBe(1);
    expect(trains.frac[0]).toBeCloseTo(0.2, 3);
    fb.sample(1375, ADJ, trains);
    expect(trains.frac[0]).toBeCloseTo(0.4, 3);
    fb.sample(1600, ADJ, trains);
    expect(trains.frac[0]).toBeCloseTo(0.6, 3);
    expect(fb.clockAt(1375)).toBeCloseTo(107.5);
  });

  it("matches trains by train_id, not array position", () => {
    const fb = new FrameBuffer(4);
    fb.push(frame(1, 100, [t(3, 0, 0.1), t(9, 1, 0.5)]), 0);
    fb.push(frame(2, 115, [t(9, 1, 0.7)]), 250);
    const trains = new InterpolatedTrains(1);
    fb.sample(375, ADJ, trains);
    expect(trains.count).toBe(1);
    expect(trains.trainId[0]).toBe(9);
    expect(trains.frac[0]).toBeCloseTo(0.6, 3);
  });

  it("places new trains at their current position and snaps on route change", () => {
    const fb = new FrameBuffer(4);
    fb.push(frame(1, 100, [t(4, 0, 0.1, 2)]), 0);
    fb.push(frame(2, 115, [t(4, 0, 0.9, 5), t(8, 1, 0.3)]), 250);
    const trains = new InterpolatedTrains(1);
    fb.sample(375, ADJ, trains);
    expect(trains.frac[0]).toBeCloseTo(0.9, 3); // slot reused by another route → snap
    expect(trains.frac[1]).toBeCloseTo(0.3, 3); // new train
    expect(trains.route[0]).toBe(5);
  });

  it("restarts instead of interpolating across a reset or an impossible clock jump", () => {
    for (const [seq, clock] of [
      [1, 50], // clock went backwards
      [0, 120], // sequence went backwards
      [3, 100 + RESET_CLOCK_JUMP_S + 1], // jump larger than any speed produces
    ] as const) {
      const fb = new FrameBuffer(4);
      fb.push(frame(2, 100, [t(1, 0, 0.1)]), 0);
      fb.push(frame(seq, clock, [t(1, 0, 0.9)]), 250);
      expect(fb.previous).toBeNull();
      const trains = new InterpolatedTrains(1);
      fb.sample(260, ADJ, trains);
      expect(trains.frac[0]).toBeCloseTo(0.9, 3);
    }
  });

  it("ignores duplicate frames", () => {
    const fb = new FrameBuffer(4);
    const a = frame(5, 100, [t(1, 0, 0.1)]);
    fb.push(a, 0);
    fb.push(frame(5, 100, [t(1, 0, 0.1)]), 100);
    expect(fb.previous).toBeNull();
    expect(fb.latestArrival).toBe(0);
  });

  it("interpolates station crowding but never blends a closure (255)", () => {
    const fb = new FrameBuffer(4);
    fb.push(frame(1, 100, [], [10, 40]), 0);
    fb.push(frame(2, 115, [], [30, 255]), 250);
    const crowd = new Float32Array(2);
    expect(fb.sampleStationCrowd(375, crowd)).toBe(2);
    expect(crowd[0]).toBeCloseTo(20);
    expect(crowd[1]).toBe(255);
  });

  it("adapts its interval to the observed frame rate", () => {
    const fb = new FrameBuffer(4);
    fb.setFrameHz(2);
    expect(fb.frameIntervalMs).toBe(500);
    fb.push(frame(1, 100, []), 0);
    fb.push(frame(2, 115, []), 300);
    expect(fb.frameIntervalMs).toBeCloseTo(0.8 * 500 + 0.2 * 300);
    expect(fb.alphaAt(300)).toBe(0);
  });

  it("returns nothing before the first frame", () => {
    const fb = new FrameBuffer();
    const trains = new InterpolatedTrains(1);
    expect(fb.sample(0, ADJ, trains)).toBe(0);
    expect(fb.clockAt(0)).toBeNull();
    expect(fb.sampleStationCrowd(0, new Float32Array(3))).toBe(0);
  });
});

/**
 * A small line with a parallel express track and a crossover:
 *
 *   S0 ─e0→ S1 ─e1→ S2 ─e3 (route 0)─→ S3 ─e5→ S4        S5 ─e2→ S6
 *      ←e6─            ─e4 (route 1)─→    ←e7─
 *                      e8: crossover @S2
 */
function graph(): EdgeGraph {
  const routes: number[][] = [[0, 1], [0, 1], [0, 1], [0], [1], [0, 1], [0, 1], [0], [0, 1]];
  const words = 1;
  const bits = new Uint32Array(routes.length * words);
  routes.forEach((rs, e) => rs.forEach((r) => (bits[e] |= 1 << r)));
  return {
    count: 9,
    fromStation: Int32Array.from([0, 1, 5, 2, 2, 3, 1, 3, 2]),
    toStation: Int32Array.from([1, 2, 6, 3, 3, 4, 0, 2, 2]),
    lengthM: Float64Array.from([100, 300, 200, 200, 200, 100, 100, 200, 0]),
    kind: Uint8Array.from([EDGE_RUN, EDGE_RUN, EDGE_RUN, EDGE_RUN, EDGE_RUN, EDGE_RUN, EDGE_RUN, EDGE_RUN, EDGE_CROSSOVER]),
    segment: Int32Array.from([0, 1, 2, 3, 4, 5, 0, 3, -1]),
    segDir: Int8Array.from([1, 1, 1, 1, 1, 1, -1, -1, 0]),
    routeBits: bits,
    routeWords: words,
  };
}

/** A straight chain S0 → S1 → … → S(n), 100 m per edge. */
function chain(n: number): EdgeAdjacency {
  return {
    count: n,
    fromStation: Int32Array.from({ length: n }, (_, i) => i),
    toStation: Int32Array.from({ length: n }, (_, i) => i + 1),
    lengthM: new Float64Array(n).fill(100),
  };
}

describe("PathFinder", () => {
  it("finds a chain of continuing edges and caches it", () => {
    const pf = new PathFinder(graph());
    const p = pf.find(0, 5, 0);
    expect(p && Array.from(p.via)).toEqual([1, 3]);
    expect(p?.viaLengthM).toBe(500);
    expect(pf.find(0, 5, 0)).toBe(p);
    expect(pf.searches).toBe(1);
  });

  it("prefers the edges the train's route uses", () => {
    const pf = new PathFinder(graph());
    expect(Array.from(pf.find(1, 5, 0)?.via ?? [])).toEqual([3]);
    expect(Array.from(pf.find(1, 5, 1)?.via ?? [])).toEqual([4]);
  });

  it("goes straight through a station rather than over its crossover", () => {
    expect(new PathFinder(graph()).find(1, 3, 0)?.via.length).toBe(0);
  });

  it("allows a reversal at a terminal (the train relays)", () => {
    expect(new PathFinder(graph()).find(0, 6, 0)?.via.length).toBe(0);
  });

  it("gives up beyond the depth limit and on unrelated edges", () => {
    const pf = new PathFinder(chain(10));
    expect(pf.find(0, MAX_PATH_HOPS, -1)?.via.length).toBe(MAX_PATH_HOPS - 1);
    expect(pf.find(0, MAX_PATH_HOPS + 1, -1)).toBeNull();
    expect(new PathFinder(graph()).find(0, 2, 0)).toBeNull();
    expect(new PathFinder(graph()).find(5, 0, 0)).toBeNull(); // backwards along the line
  });
});

describe("interpolateTrain across several edges", () => {
  it("moves at constant speed along the path", () => {
    const g = graph();
    // 50 m left on e0 + 300 (e1) + 200 (e3) + 50 m into e5 = 600 m.
    const a = interpolateTrain(0, 0.5, 5, 0.5, 0.25, g, out(), 0); // 150 m
    expect(a.edge).toBe(1);
    expect(a.frac).toBeCloseTo(100 / 300);
    const b = interpolateTrain(0, 0.5, 5, 0.5, 0.75, g, out(), 0); // 450 m
    expect(b.edge).toBe(3);
    expect(b.frac).toBeCloseTo(100 / 200);
    expect(interpolateTrain(0, 0.5, 5, 0.5, 1, g, out(), 0)).toEqual({ edge: 5, frac: 0.5 });
  });
});

describe("easeDistance", () => {
  it("is exact at the frame times and linear when the speed is steady", () => {
    expect(easeDistance(0, 300, 100)).toBe(0);
    expect(easeDistance(1, 300, 100)).toBeCloseTo(100);
    expect(easeDistance(0.3, 100, 100)).toBeCloseTo(30);
    expect(easeDistance(0.5, 50, 0)).toBe(0);
  });

  it("starts at the previous speed and never runs backwards or overshoots", () => {
    for (const v0 of [0, 20, 100, 250, 1000]) {
      let last = 0;
      for (let k = 1; k <= 100; k++) {
        const s = easeDistance(k / 100, v0, 100);
        expect(s).toBeGreaterThanOrEqual(last - 1e-9);
        expect(s).toBeLessThanOrEqual(100);
        last = s;
      }
    }
    // Decelerating (previous interval covered 250 m): ahead of linear early on.
    expect(easeDistance(0.1, 250, 100)).toBeGreaterThan(10);
    // Accelerating from a stop: behind linear early on.
    expect(easeDistance(0.1, 0, 100)).toBeLessThan(10);
  });
});

describe("FrameBuffer motion", () => {
  it("glides across several edges between two frames", () => {
    const fb = new FrameBuffer(4);
    fb.push(frame(1, 100, [t(7, 0, 0.5, 0)]), 0);
    fb.push(frame(2, 175, [t(7, 5, 0.5, 0)]), 250);
    const trains = new InterpolatedTrains(2);
    fb.sample(250 + 62.5, graph(), trains); // alpha 0.25 → 150 m
    expect(trains.edge[0]).toBe(1);
    expect(trains.frac[0]).toBeCloseTo(100 / 300, 3);
    expect(trains.fade[0]).toBe(1);
    expect(fb.stats).toMatchObject({ updates: 1, path: 1, snap: 0, newTrip: 0 });
  });

  it("cross-fades a slot reused by another route instead of sliding", () => {
    const fb = new FrameBuffer(4);
    fb.push(frame(1, 100, [t(4, 0, 0.1, 2)]), 0);
    fb.push(frame(2, 115, [t(4, 5, 0.3, 5)]), 250);
    const trains = new InterpolatedTrains(1);
    const ghosts = new InterpolatedTrains(1);
    fb.sample(375, graph(), trains, ghosts);
    expect(trains.edge[0]).toBe(5);
    expect(trains.frac[0]).toBeCloseTo(0.3, 3);
    expect(trains.fade[0]).toBeCloseTo(fadeCurve(0.5));
    expect(ghosts.count).toBe(1);
    expect(ghosts.edge[0]).toBe(0);
    expect(ghosts.frac[0]).toBeCloseTo(0.1, 3);
    expect(ghosts.route[0]).toBe(2);
    expect(ghosts.fade[0]).toBeCloseTo(1 - fadeCurve(0.5));
    expect(fb.stats).toMatchObject({ updates: 1, newTrip: 1, snap: 0 });
    fb.sample(600, graph(), trains, ghosts);
    expect(ghosts.count).toBe(0);
    expect(trains.fade[0]).toBe(1);
  });

  it("treats a move too long for the clock advance as a discontinuity", () => {
    const fb = new FrameBuffer(4);
    // 700 m in 5 sim-s is far beyond V_MAX.
    expect(700).toBeGreaterThan(plausibleDistanceM(5));
    fb.push(frame(1, 100, [t(3, 0, 0, 0)]), 0);
    fb.push(frame(2, 105, [t(3, 5, 1, 0)]), 250);
    const trains = new InterpolatedTrains(1);
    const ghosts = new InterpolatedTrains(1);
    fb.sample(300, graph(), trains, ghosts);
    expect(trains.edge[0]).toBe(5);
    expect(trains.fade[0]).toBeLessThan(1);
    expect(ghosts.count).toBe(1);
    expect(fb.stats).toMatchObject({ updates: 1, snap: 1 });
  });

  it("fades out trains that left and fades in trains that appeared", () => {
    const fb = new FrameBuffer(4);
    fb.push(frame(1, 100, [t(1, 0, 0.2, 0)]), 0);
    fb.push(frame(2, 115, [t(2, 1, 0.4, 0)]), 250);
    const trains = new InterpolatedTrains(1);
    const ghosts = new InterpolatedTrains(1);
    fb.sample(250, graph(), trains, ghosts);
    expect(trains.fade[0]).toBe(0);
    expect(ghosts.fade[0]).toBe(1);
    expect(fb.stats).toMatchObject({ updates: 0, appeared: 1, vanished: 1 });
  });

  it("carries the previous interval's speed into the next one", () => {
    const fb = new FrameBuffer(4);
    const trains = new InterpolatedTrains(1);
    fb.push(frame(1, 100, [t(9, 1, 0, 0)]), 0);
    fb.push(frame(2, 115, [t(9, 1, 0.5, 0)]), 250); // 150 m
    fb.sample(260, graph(), trains);
    fb.push(frame(3, 130, [t(9, 1, 0.6, 0)]), 500); // 30 m: braking
    fb.sample(500, graph(), trains);
    expect(trains.frac[0]).toBeCloseTo(0.5, 3);
    fb.sample(525, graph(), trains); // alpha 0.1: still rolling faster than the 30 m mean
    expect(trains.frac[0]).toBeGreaterThan(0.5 + 0.1 * 0.1 + 1e-3);
    let last = 0.5;
    for (let ms = 525; ms <= 750; ms += 25) {
      fb.sample(ms, graph(), trains);
      expect(trains.frac[0]).toBeGreaterThanOrEqual(last - 1e-6);
      last = trains.frac[0];
    }
    expect(last).toBeCloseTo(0.6, 3);
  });

  it("fades the whole fleet across a reset", () => {
    const fb = new FrameBuffer(4);
    fb.push(frame(1, 100, [t(1, 0, 0.2, 0)]), 0);
    fb.push(frame(2, 115, [t(1, 0, 0.4, 0), t(2, 1, 0.5, 0)]), 250);
    fb.push(frame(3, 100 + RESET_CLOCK_JUMP_S + 200, [t(5, 3, 0.5, 0)]), 500); // preset: sim restarted
    const trains = new InterpolatedTrains(1);
    const ghosts = new InterpolatedTrains(1);
    fb.sample(500 + RESET_FADE_MS / 2, graph(), trains, ghosts);
    expect(trains.count).toBe(1);
    expect(trains.fade[0]).toBeCloseTo(0.5);
    expect(ghosts.count).toBe(2);
    expect(ghosts.fade[0]).toBeCloseTo(0.5);
    fb.sample(500 + RESET_FADE_MS + 1, graph(), trains, ghosts);
    expect(ghosts.count).toBe(0);
    expect(trains.fade[0]).toBe(1);
  });
});
