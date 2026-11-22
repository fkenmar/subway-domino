/**
 * §2.7 positioning rule (CONTRACT, shared by server and client):
 * run edge → arc-length point on its segment polyline (reversed for seg_dir −1);
 * crossover / turnback → the station point; yard lead → straight yard↔station line.
 */
import { describe, expect, it } from "vitest";
import { EDGE_RUN } from "@/lib/network";
import { FRAC_SCALE, fracFromWire, laneFor, locate, makePose, type Pose } from "@/lib/positioning";
import { miniNetwork, syntheticNetwork } from "../helpers/geo";

const net = syntheticNetwork();

function at(edge: number, frac: number): Pose {
  const pose = makePose();
  expect(locate(net, edge, frac, pose), `locate(${edge}, ${frac})`).toBe(true);
  return pose;
}

function expectPoint(p: Pose, x: number, y: number): void {
  expect(p.x).toBeCloseTo(x, 6);
  expect(p.y).toBeCloseTo(y, 6);
}

function expectDir(p: Pose, dx: number, dy: number): void {
  expect(p.dx).toBeCloseTo(dx, 6);
  expect(p.dy).toBeCloseTo(dy, 6);
}

describe("run edges: polyline(segment).point_at(seg_dir == 1 ? frac : 1 − frac)", () => {
  it("seg_dir +1 runs a → b, arc-length parameterised over unequal pieces", () => {
    // Segment 0 is 400 m: a 300 m piece east, then a 100 m piece north.
    expectPoint(at(0, 0), 0, 0);
    expectPoint(at(0, 0.5), 200, 0); // 200 m along — still on the first piece
    expectPoint(at(0, 0.75), 300, 0); // exactly the corner
    expectPoint(at(0, 0.875), 300, 50); // 350 m along
    expectPoint(at(0, 1), 300, 100);
    expectDir(at(0, 0.25), 1, 0);
    expectDir(at(0, 0.9), 0, 1);
  });

  it("seg_dir −1 runs b → a: the same frac measured from the other end, heading reversed", () => {
    expectPoint(at(1, 0), 300, 100);
    expectPoint(at(1, 0.125), 300, 50);
    expectPoint(at(1, 0.5), 200, 0);
    expectPoint(at(1, 1), 0, 0);
    expectDir(at(1, 0.125), 0, -1);
    expectDir(at(1, 0.5), -1, 0);
  });

  it("clamps frac to [0, 1]", () => {
    expectPoint(at(0, -0.4), 0, 0);
    expectPoint(at(0, 1.7), 300, 100);
  });

  it("puts express tracks inside and local tracks outside on 4-track segments", () => {
    expect(at(2, 0.5).lane).toBe(laneFor(1, true));
    expect(at(0, 0.5).lane).toBe(laneFor(0, true));
    expect(at(3, 0.5).lane).toBe(laneFor(0, false)); // 2-track segment: inner lane
    expect(laneFor(1, true)).toBeLessThan(laneFor(0, true));
    expect(laneFor(0, false)).toBe(laneFor(1, true));
  });
});

describe("crossover and turnback edges: at the station point", () => {
  it("places a crossover at its station, with no heading or lane", () => {
    const p = at(4, 0.6);
    expectPoint(p, 0, 0);
    expect([p.dx, p.dy, p.lane]).toEqual([0, 0, 0]);
  });

  it("places a turnback at its station regardless of frac", () => {
    expectPoint(at(5, 0), 300, 400);
    expectPoint(at(5, 1), 300, 400);
  });
});

describe("yard leads: the straight line between the yard and the lead station", () => {
  it("pulls out yard → station (seg_dir +1)", () => {
    expectPoint(at(6, 0), 600, 400);
    expectPoint(at(6, 0.5), 450, 400);
    expectPoint(at(6, 1), 300, 400);
    expectDir(at(6, 0.5), -1, 0);
  });

  it("puts in station → yard (seg_dir −1)", () => {
    expectPoint(at(7, 0), 300, 400);
    expectPoint(at(7, 1), 600, 400);
    expectDir(at(7, 0.5), 1, 0);
  });

  it("falls back to the nearest same-division station when the lead station is not given", () => {
    expect(net.edges.station[8]).toBe(2);
    expectPoint(at(8, 1), 300, 400);
  });
});

describe("invalid input", () => {
  it("returns false and leaves the pose untouched for unknown or unplaceable edges", () => {
    for (const edge of [-1, 10, 65535, 9]) {
      const pose = makePose();
      pose.x = 42;
      expect(locate(net, edge, 0.5, pose)).toBe(false);
      expect(pose.x).toBe(42);
    }
  });
});

describe("wire frac", () => {
  it("maps the u16 range onto [0, 1]", () => {
    expect(FRAC_SCALE).toBe(65535);
    expect(fracFromWire(0)).toBe(0);
    expect(fracFromWire(65535)).toBe(1);
    expect(fracFromWire(Math.round(0.25 * 65535))).toBeCloseTo(0.25, 4);
  });
});

describe("mini_geo fixture: positions are continuous along the network", () => {
  const mini = miniNetwork();

  it("every run edge starts at its from-station and ends at its to-station", () => {
    let checked = 0;
    const pose = makePose();
    for (let e = 0; e < mini.edges.count; e++) {
      if (mini.edges.kind[e] !== EDGE_RUN) continue;
      const from = mini.stations[mini.edges.fromStation[e]];
      const to = mini.stations[mini.edges.toStation[e]];
      expect(locate(mini, e, 0, pose)).toBe(true);
      expect(Math.hypot(pose.x - from.x, pose.y - from.y), `edge ${e} start`).toBeLessThan(1);
      expect(locate(mini, e, 1, pose)).toBe(true);
      expect(Math.hypot(pose.x - to.x, pose.y - to.y), `edge ${e} end`).toBeLessThan(1);
      checked++;
    }
    expect(checked).toBeGreaterThan(100);
  });

  it("moves monotonically along the arc as frac grows", () => {
    const pose = makePose();
    for (let e = 0; e < mini.edges.count; e++) {
      if (mini.edges.kind[e] !== EDGE_RUN) continue;
      const len = mini.edges.lengthM[e];
      const from = mini.stations[mini.edges.fromStation[e]];
      let travelled = 0;
      let px = from.x;
      let py = from.y;
      for (let k = 1; k <= 20; k++) {
        locate(mini, e, k / 20, pose);
        travelled += Math.hypot(pose.x - px, pose.y - py);
        px = pose.x;
        py = pose.y;
        // Arc length covered so far never exceeds frac · length (chords ≤ arcs).
        expect(travelled).toBeLessThanOrEqual((k / 20) * len + 1e-6);
      }
      expect(travelled).toBeGreaterThan(0.95 * len);
    }
  });
});
