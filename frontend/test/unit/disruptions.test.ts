/** Disruption targeting (§3.5 allowed targets; §6 UX flow 5 popover choices). */
import { describe, expect, it } from "vitest";
import {
  disruptionStations,
  DURATION_CHOICES,
  isValidDisruption,
  remainingS,
  targetAnchor,
  targetKindLabel,
  targetLabel,
  validKinds,
} from "@/lib/disruptions";
import type { DisruptionInfo, DisruptionTarget, TargetType } from "@/lib/protocol";
import { miniNetwork } from "../helpers/geo";

const net = miniNetwork();

function allTargets(): DisruptionTarget[] {
  const out: DisruptionTarget[] = [];
  const add = (type: TargetType, n: number) => {
    for (let idx = 0; idx < n; idx++) out.push({ type, idx });
  };
  add("station", net.stations.length);
  add("complex", net.complexes.length);
  add("segment", net.segments.length);
  add("interlocking", net.interlockings.length);
  return out;
}

describe("validKinds — only kinds the server accepts for the target (§3.5)", () => {
  it("never offers a kind outside its allowed target types", () => {
    const allowed: Record<string, TargetType[]> = {
      local_block: ["segment", "station", "interlocking"],
      express_block: ["segment", "station", "interlocking"],
      full_station_closure: ["station", "complex"],
      signal_delay: ["segment", "interlocking"],
    };
    for (const target of allTargets()) {
      const kinds = validKinds(net, target);
      expect(kinds.length, `${target.type} ${target.idx}`).toBeGreaterThan(0);
      for (const k of kinds) expect(allowed[k]).toContain(target.type);
    }
  });

  it("offers an express block only where there is an express track", () => {
    for (const seg of net.segments) {
      const hasX = seg.has_express || net.segmentHasXEdge[seg.idx] === 1;
      expect(validKinds(net, { type: "segment", idx: seg.idx }).includes("express_block")).toBe(hasX);
    }
    const local = net.stations.find((s) => net.stationHasExpress[s.idx] === 0);
    const express = net.stations.find((s) => net.stationHasExpress[s.idx] === 1);
    expect(local && express).toBeTruthy();
    expect(validKinds(net, { type: "station", idx: local?.idx ?? -1 })).toEqual(["full_station_closure", "local_block"]);
    expect(validKinds(net, { type: "station", idx: express?.idx ?? -1 })).toEqual([
      "full_station_closure",
      "local_block",
      "express_block",
    ]);
  });

  it("offers closure for complexes and signal problems for segments and interlockings", () => {
    expect(validKinds(net, { type: "complex", idx: 0 })).toEqual(["full_station_closure"]);
    expect(validKinds(net, { type: "segment", idx: 0 })).toContain("signal_delay");
    expect(validKinds(net, { type: "interlocking", idx: 0 })).toContain("signal_delay");
    expect(validKinds(net, { type: "station", idx: 0 })).not.toContain("signal_delay");
  });

  it("offers nothing for unknown targets", () => {
    for (const type of ["station", "complex", "segment", "interlocking"] as const) {
      expect(validKinds(net, { type, idx: 10_000 })).toEqual([]);
    }
    expect(isValidDisruption(net, "local_block", { type: "segment", idx: -1 })).toBe(false);
    expect(isValidDisruption(net, "local_block", { type: "segment", idx: 0 })).toBe(true);
  });
});

describe("labels and anchors", () => {
  it("names targets in plain English", () => {
    expect(targetLabel(net, { type: "station", idx: 0 })).toBe("Times Sq-42 St");
    expect(targetLabel(net, { type: "segment", idx: 0 })).toBe("Times Sq-42 St ↔ 34 St-Penn Station");
    expect(targetLabel(net, { type: "interlocking", idx: 0 })).toBe("Times Sq-42 St interlocking");
    const river = net.segments.find((s) => s.river);
    expect(targetKindLabel(net, { type: "segment", idx: river?.idx ?? -1 })).toBe("River crossing");
    expect(targetKindLabel(net, { type: "segment", idx: 0 })).toBe("Track segment");
  });

  it("anchors overlays on the target", () => {
    expect(targetAnchor(net, { type: "station", idx: 0 })).toEqual({ x: -3164, y: 4722 });
    const mid = targetAnchor(net, { type: "segment", idx: 0 });
    const seg = net.segments[0];
    expect(mid).not.toBeNull();
    // The anchor lies on the segment polyline, between its two stations.
    const minX = Math.min(...seg.line.xs);
    const maxX = Math.max(...seg.line.xs);
    expect(mid?.x).toBeGreaterThanOrEqual(minX);
    expect(mid?.x).toBeLessThanOrEqual(maxX);
    expect(targetAnchor(net, { type: "complex", idx: 99_999 })).toBeNull();
  });

  it("lists the stations a disruption touches", () => {
    const d = (target: DisruptionTarget): DisruptionInfo => ({
      id: 1,
      kind: "local_block",
      target,
      label: "",
      start_t: 0,
      end_t: null,
    });
    expect(disruptionStations(net, d({ type: "segment", idx: 0 }))).toEqual([0, 1]);
    expect(disruptionStations(net, d({ type: "complex", idx: 0 }))).toEqual(net.complexes[0].stations);
  });
});

describe("durations", () => {
  it("offers until-cleared, 10, 20 and 30 minutes", () => {
    expect(DURATION_CHOICES.map((c) => c.seconds)).toEqual([null, 600, 1200, 1800]);
  });

  it("counts down in sim seconds and never goes negative", () => {
    const d: DisruptionInfo = { id: 1, kind: "signal_delay", target: { type: "segment", idx: 0 }, label: "", start_t: 100, end_t: 1900 };
    expect(remainingS(d, 1000)).toBe(900);
    expect(remainingS(d, 5000)).toBe(0);
    expect(remainingS(d, null)).toBe(1800);
    expect(remainingS({ ...d, end_t: null }, 1000)).toBeNull();
  });
});
