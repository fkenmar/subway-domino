import { describe, expect, it } from "vitest";
import {
  type EdgeAdjacency,
  edgesContinue,
  FrameBuffer,
  InterpolatedTrains,
  interpolateTrain,
  RESET_CLOCK_JUMP_S,
} from "@/lib/interp";
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
