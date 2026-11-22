/** zustand store reducers that carry product rules (sparkline window, toasts, feed). */
import { beforeEach, describe, expect, it } from "vitest";
import type { MetricsMessage, MetricValues } from "@/lib/protocol";
import { appendHistory, appendTrend, FEED_LIMIT, HISTORY_WINDOW_S, MAX_TOASTS, TREND_POINTS, useStore } from "@/lib/store";

const values = (w: number): MetricValues => ({
  avg_wait_min: w,
  overcrowded_platforms: 0,
  tunnel_stuck_trains: 0,
  riders_served: 0,
  riders_waiting: 0,
  trains_active: 0,
});

const metrics = (t: number, twin: MetricValues | null = null): MetricsMessage => ({
  type: "metrics",
  t,
  mode: twin ? "ai" : "normal",
  effective_speed: 1,
  live: values(t / 100),
  twin,
  ai_since_t: twin ? 0 : null,
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

describe("appendHistory (sparklines over the last 10 sim-min)", () => {
  it("keeps exactly the last HISTORY_WINDOW_S sim seconds", () => {
    expect(HISTORY_WINDOW_S).toBe(600);
    let h = appendHistory([], metrics(30_000));
    for (let t = 30_060; t <= 31_200; t += 60) h = appendHistory(h, metrics(t, values(1)));
    expect(h[0].t).toBe(31_200 - 600);
    expect(h.at(-1)?.t).toBe(31_200);
    expect(h.at(-1)?.twin).toEqual(values(1));
  });

  it("starts a fresh series when the clock goes backwards (reset, preset, profile change)", () => {
    const h = appendHistory(appendHistory([], metrics(40_000)), metrics(40_060));
    const after = appendHistory(h, metrics(28_800));
    expect(after.map((p) => p.t)).toEqual([28_800]);
  });
});

describe("store", () => {
  beforeEach(() => {
    useStore.setState({ toasts: [], feed: [], history: [], metrics: null });
  });

  it("shows at most MAX_TOASTS toasts and collapses repeats of the same callout", () => {
    const { pushToast } = useStore.getState();
    pushToast({ kind: "callout", text: "a", dedupeKey: "HOLD:4:12" });
    pushToast({ kind: "callout", text: "b", dedupeKey: "HOLD:4:12" });
    expect(useStore.getState().toasts.map((t) => t.text)).toEqual(["b"]);
    pushToast({ kind: "notice", text: "c" });
    pushToast({ kind: "error", text: "d" });
    expect(useStore.getState().toasts).toHaveLength(MAX_TOASTS);
    expect(useStore.getState().toasts.map((t) => t.text)).toEqual(["c", "d"]);
    pushToast({ id: "fixed", kind: "notice", text: "e" });
    pushToast({ id: "fixed", kind: "notice", text: "f" });
    expect(useStore.getState().toasts.filter((t) => t.id === "fixed").map((t) => t.text)).toEqual(["e"]);
  });

  it("lists feed items newest first, without duplicates, capped", () => {
    const { addFeed } = useStore.getState();
    const callout = (id: number) => ({
      id: `c${id}`,
      kind: "callout" as const,
      t: id,
      event: {
        kind: "callout" as const,
        id,
        t: id,
        train_id: 1,
        route: "4",
        action: "HOLD_IN_STATION" as const,
        station_idx: 0,
        text: `#${id}`,
      },
    });
    addFeed([callout(1), callout(2)]);
    addFeed([callout(2), callout(3)]);
    expect(useStore.getState().feed.map((f) => f.id)).toEqual(["c3", "c2", "c1"]);
    addFeed(Array.from({ length: FEED_LIMIT + 20 }, (_, i) => callout(100 + i)));
    expect(useStore.getState().feed).toHaveLength(FEED_LIMIT);
    expect(useStore.getState().feed[0].id).toBe(`c${100 + FEED_LIMIT + 19}`);
  });

  it("records metrics into the history series", () => {
    useStore.getState().pushMetrics(metrics(30_000));
    useStore.getState().pushMetrics(metrics(30_060, values(2)));
    const s = useStore.getState();
    expect(s.metrics?.t).toBe(30_060);
    expect(s.history).toHaveLength(2);
    s.clearHistory();
    expect(useStore.getState().history).toEqual([]);
    expect(useStore.getState().metrics).toBeNull();
  });
});

describe("appendTrend (sparkline series)", () => {
  it("keeps the last TREND_POINTS points at any sim speed and restarts on a reset", () => {
    let tr = appendTrend([], metrics(30_000));
    for (let k = 1; k < TREND_POINTS + 15; k++) tr = appendTrend(tr, metrics(30_000 + 300 * k)); // 5x: 300 sim-s apart
    expect(tr).toHaveLength(TREND_POINTS);
    expect(tr.at(-1)?.t).toBe(30_000 + 300 * (TREND_POINTS + 14));
    expect(appendTrend(tr, metrics(1_000))).toHaveLength(1);
  });

  it("is filled by pushMetrics and cleared with the history", () => {
    useStore.getState().clearHistory();
    useStore.getState().pushMetrics(metrics(50_000));
    expect(useStore.getState().trend).toHaveLength(1);
    useStore.getState().clearHistory();
    expect(useStore.getState().trend).toHaveLength(0);
  });
});
