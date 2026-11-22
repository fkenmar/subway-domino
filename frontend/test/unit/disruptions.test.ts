/** Disruption targeting (§3.5 allowed targets; §6 UX flow 5 popover choices). */
import { describe, expect, it } from "vitest";
import {
  blockedWord,
  chipText,
  disruptionStations,
  isValidDisruption,
  remainingS,
  restoredText,
  sinceText,
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

describe("validKinds — only blockage kinds the server accepts for the target (§3.5)", () => {
  it("never offers a kind outside its allowed target types", () => {
    const allowed: Record<string, TargetType[]> = {
      local_block: ["segment", "station", "interlocking"],
      express_block: ["segment", "station", "interlocking"],
      full_station_closure: ["station", "complex"],
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

  it("offers closure for complexes and track blocks for segments and interlockings, never signal problems", () => {
    expect(validKinds(net, { type: "complex", idx: 0 })).toEqual(["full_station_closure"]);
    expect(validKinds(net, { type: "segment", idx: 0 })[0]).toBe("local_block");
    expect(validKinds(net, { type: "interlocking", idx: 0 })[0]).toBe("local_block");
    for (const target of allTargets()) expect(validKinds(net, target)).not.toContain("signal_delay");
    expect(isValidDisruption(net, "signal_delay", { type: "segment", idx: 0 })).toBe(false);
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

describe("blockages last until cleared", () => {
  const block = (kind: DisruptionInfo["kind"], start_t = 29_520): DisruptionInfo => ({
    id: 1,
    kind,
    target: { type: "segment", idx: 0 },
    label: "",
    start_t,
    end_t: null,
  });

  it("labels chips CLOSED / BLOCKED with no countdown", () => {
    expect(chipText(block("full_station_closure"), 0)).toEqual({ label: "CLOSED", since: "" });
    expect(chipText(block("local_block"), 0)).toEqual({ label: "BLOCKED", since: "" });
    expect(chipText(block("express_block"), 1)).toEqual({ label: "EXPRESS BLOCKED", since: "" });
    expect(chipText(block("local_block"), 2)).toEqual({ label: "LOCAL BLOCKED", since: "since 8:12 AM" });
    expect(chipText(block("full_station_closure"), 4)).toEqual({ label: "CLOSED", since: "since 8:12 AM" });
    for (const kind of ["full_station_closure", "local_block", "express_block", "signal_delay"] as const) {
      for (let tier = 0; tier <= 4; tier++) {
        const { label, since } = chipText({ ...block(kind), end_t: 31_320 } as DisruptionInfo, tier);
        // Even a timed disruption (an old restored session) shows no countdown or "∞".
        expect(`${label} ${since}`).not.toMatch(/\d+:\d\d(?! [AP]M)|∞|left/);
      }
    }
    expect(blockedWord("full_station_closure")).toBe("CLOSED");
    expect(blockedWord("express_block")).toBe("BLOCKED");
  });

  it("says since when, not how long is left", () => {
    expect(sinceText(block("full_station_closure"))).toBe("Closed since 8:12 AM");
    expect(sinceText(block("local_block", 61_200))).toBe("Blocked since 5:00 PM");
  });

  it("words the service-restored notice", () => {
    const st = { ...block("full_station_closure"), target: { type: "station" as const, idx: 0 } };
    expect(restoredText(net, [st])).toBe("Times Sq-42 St reopened");
    expect(restoredText(net, [block("express_block")])).toBe("Times Sq-42 St ↔ 34 St-Penn Station: express track clear again");
    expect(restoredText(null, [{ ...block("local_block"), label: "Canarsie Tube" }])).toBe("Canarsie Tube: local track clear again");
    expect(restoredText(net, [st, block("local_block"), block("express_block")])).toBe("Service restored at 3 places");
  });

  it("keeps the remaining time only for a session restore of a timed disruption", () => {
    const d: DisruptionInfo = { ...block("local_block", 100), end_t: 1900 };
    expect(remainingS(d, 1000)).toBe(900);
    expect(remainingS(d, 5000)).toBe(0);
    expect(remainingS(d, null)).toBe(1800);
    expect(remainingS({ ...d, end_t: null }, 1000)).toBeNull();
  });
});
