/** AI-vs-Normal scoreboard and AI callout pacing (WP3b critique fixes). */
import { describe, expect, it } from "vitest";
import { accumulateScore, type AiScoreState, scoreLines, verdictText } from "@/lib/aiScore";
import { CALLOUT_GAP_MS, CalloutPacer, calloutHeadline, calloutReason } from "@/lib/calloutPacer";
import { HUD_METRICS } from "@/lib/copy";
import type { CalloutEvent, MetricsMessage, MetricValues } from "@/lib/protocol";
import { mergeWindows } from "@/lib/store";
import { twinDelta } from "@/components/MetricsHUD";
import { windowBands } from "@/components/Sparkline";
import { desaturate } from "@/map/style";

const vals = (p: Partial<MetricValues> = {}): MetricValues => ({
  avg_wait_min: 3,
  overcrowded_platforms: 0,
  tunnel_stuck_trains: 0,
  riders_served: 1000,
  riders_waiting: 1000,
  trains_active: 400,
  ...p,
});

const msg = (t: number, live: MetricValues, twin: MetricValues | null, since: number | null = 0): MetricsMessage => ({
  type: "metrics",
  t,
  mode: twin ? "ai" : "normal",
  effective_speed: 1,
  live,
  twin,
  ai_since_t: since,
  actions: {
    HOLD_IN_STATION: 0,
    RUN_EXPRESS: 0,
    RUN_LOCAL: 0,
    SKIP_STOP: 0,
    SHORT_TURN_RELAY: 0,
    REROUTE_VIA_COMPATIBLE_TRUNK: 0,
    DISPATCH_YARD_GAP_TRAIN: 0,
  },
});

describe("cumulative AI vs Normal score", () => {
  it("integrates riders waiting over sim time (trapezoid) into rider-minutes", () => {
    let s: AiScoreState | null = null;
    s = accumulateScore(s, msg(0, vals({ riders_waiting: 900 }), vals({ riders_waiting: 1000 })));
    s = accumulateScore(s, msg(60, vals({ riders_waiting: 900 }), vals({ riders_waiting: 1000 })));
    s = accumulateScore(s, msg(120, vals({ riders_waiting: 900 }), vals({ riders_waiting: 1000 })));
    const wait = scoreLines(s as AiScoreState)[0];
    expect(wait.key).toBe("wait");
    expect(wait.ai).toBeCloseTo(1800);
    expect(wait.normal).toBeCloseTo(2000);
    expect(wait.pct).toBeCloseTo(-10);
    expect(wait.verdict).toBe("ahead");
  });

  it("is stable against instantaneous swings: riders-served noise inside the dead band reads even", () => {
    let s: AiScoreState | null = null;
    const served = [0, 707, 692, 265, -239, 585];
    served.forEach((d, i) => {
      s = accumulateScore(s, msg(i * 60, vals({ riders_served: 100_000 + i * 20_000 + d }), vals({ riders_served: 100_000 + i * 20_000 })));
    });
    const line = scoreLines(s as unknown as AiScoreState).find((l) => l.key === "served");
    expect(line?.verdict).toBe("even");
  });

  it("restarts on a new takeover or a reset, and is null without a twin", () => {
    const a = accumulateScore(null, msg(100, vals(), vals(), 100));
    const b = accumulateScore(a, msg(160, vals({ riders_waiting: 0 }), vals(), 100));
    expect(b?.sums.wait[0]).toBeGreaterThan(0);
    const c = accumulateScore(b, msg(500, vals(), vals(), 500));
    expect(c?.sums.wait[0]).toBe(0);
    expect(accumulateScore(c, msg(560, vals(), null, null))).toBeNull();
  });

  it("verdict counts metrics ahead / behind beyond the dead band", () => {
    let s: AiScoreState | null = null;
    for (let t = 0; t <= 600; t += 60) {
      s = accumulateScore(s, msg(t, vals({ riders_waiting: 800, tunnel_stuck_trains: 4 }), vals({ riders_waiting: 1000, tunnel_stuck_trains: 1 })));
    }
    const v = verdictText(scoreLines(s as unknown as AiScoreState));
    expect(v.text).toBe("AI ahead on 1 of 4 · behind on 1");
    expect(verdictText([]).text).toBe("Even so far");
  });
});

describe("twinDelta dead band", () => {
  const wait = HUD_METRICS.find((m) => m.key === "avg_wait_min");
  const stuck = HUD_METRICS.find((m) => m.key === "tunnel_stuck_trains");
  const served = HUD_METRICS.find((m) => m.key === "riders_served");
  if (!wait || !stuck || !served) throw new Error("metrics missing");

  it("greys out small gaps (< 3 % or ±1)", () => {
    expect(twinDelta(wait, 3.05, 3.0).tone).toBe("flat");
    expect(twinDelta(stuck, 1, 0).tone).toBe("flat");
    expect(twinDelta(served, 746_577, 746_000).tone).toBe("flat");
  });

  it("colours real gaps", () => {
    expect(twinDelta(wait, 2.5, 3.0)).toEqual({ text: "−17%", tone: "good" });
    expect(twinDelta(stuck, 4, 1).tone).toBe("bad");
  });
});

const callout = (id: number, station: number): CalloutEvent => ({
  kind: "callout",
  id,
  t: id,
  train_id: id,
  route: "4",
  action: "RUN_LOCAL",
  station_idx: station,
  text: "Moving a 4 train to the local track at 125 St — trains are stopped in the tunnel ahead near 86 St. It makes all local stops until 59 St.",
});

describe("CalloutPacer", () => {
  it("shows at most one callout per gap and prefers one in view", () => {
    const p = new CalloutPacer();
    const inView = (s: number) => s === 7;
    p.offer(callout(1, 1), 0);
    expect(p.next(0, inView)?.event.id).toBe(1);
    p.offer(callout(2, 7), 100);
    p.offer(callout(3, 9), 200);
    expect(p.next(1_000, inView)).toBeNull();
    const n = p.next(CALLOUT_GAP_MS + 10, inView);
    expect(n?.event.id).toBe(2);
    expect(n?.recent).toBe(3);
  });

  it("drops stale and cleared callouts", () => {
    const p = new CalloutPacer();
    p.offer(callout(1, 1), 0);
    p.clear();
    expect(p.next(10, () => true)).toBeNull();
    p.offer(callout(2, 1), 0);
    expect(p.next(60_000, () => true)).toBeNull();
  });

  it("writes a one-line headline and extracts the reason clause", () => {
    expect(calloutHeadline("RUN_LOCAL", "125 St")).toBe("To local at 125 St");
    expect(calloutReason(callout(1, 1).text)).toBe("Trains are stopped in the tunnel ahead near 86 St");
    expect(calloutReason("Holding a 6 train at 23 St with doors open for 30s.")).toBeNull();
    expect(calloutReason("Holding a 6 train at 23 St with doors open for 30s — the train ahead is only 45s ahead, with service every 4m.")).toBe(
      "The train ahead is only 45s ahead",
    );
  });
});

describe("disruption windows", () => {
  const d = { id: 1, kind: "full_station_closure" as const, target: { type: "station" as const, idx: 3 }, label: "x", start_t: 100, end_t: 1800 };

  it("keeps a window after the disruption ends and shades it on the sparkline", () => {
    const w = mergeWindows([], [d]);
    expect(w).toEqual([{ id: 1, start_t: 100, end_t: 1800, ended: false }]);
    const ended = [{ ...w[0], end_t: 1700, ended: true }];
    expect(mergeWindows(ended, [])).toEqual(ended);
    const bands = windowBands(ended, 0, 2000, (t) => t / 10);
    expect(bands).toEqual([{ x0: 10, x1: 170, ended: true }]);
    expect(windowBands(ended, 1800, 2000, (t) => t)).toEqual([]);
  });
});

describe("desaturate", () => {
  it("pulls a colour towards its grey", () => {
    expect(desaturate("#ff0000", 0)).toBe("#ff0000");
    expect(desaturate("#ff0000", 1)).toBe("#4c4c4c");
  });
});
