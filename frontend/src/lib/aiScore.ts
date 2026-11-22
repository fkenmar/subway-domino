/**
 * Cumulative AI-vs-Normal comparison since the AI took over, computed only from
 * the server's live and twin metrics (§5.3 `metrics`): time integrals of riders
 * waiting, tunnel-stuck trains and overcrowded platforms (trapezoid rule over sim
 * time), plus the riders-served gap (the twin is forked at takeover, so the gap of
 * the cumulative counters is already "since takeover").
 *
 * Instantaneous deltas swing with every train arrival; these totals only move
 * slowly, so the scoreboard's headline stays stable.
 */
import type { MetricsMessage, MetricValues } from "./protocol";

export type ScoreKey = "wait" | "stuck" | "crowded" | "served";
export type Verdict = "ahead" | "behind" | "even";

export interface ScoreLine {
  key: ScoreKey;
  label: string;
  /** Compact label for tight rows ("Rider-min waited"). */
  short: string;
  unit: string;
  better: "lower" | "higher";
  ai: number;
  normal: number;
  /** ai − normal (signed). */
  diff: number;
  /** Relative difference in % of the Normal total (null when the baseline is ~0). */
  pct: number | null;
  verdict: Verdict;
}

export interface AiScoreState {
  /** `ai_since_t` the totals belong to. */
  since: number;
  /** Sim time the integration started (later than `since` after a resume mid-takeover). */
  from: number;
  lastT: number;
  last: { live: MetricValues; twin: MetricValues };
  /** Integrals in unit·sim-seconds. */
  sums: { wait: [number, number]; stuck: [number, number]; crowded: [number, number] };
  served: [number, number];
  /** Last verdict per line (hysteresis: a line must fall well inside the dead band to go back to "even"). */
  verdicts: Partial<Record<ScoreKey, Verdict>>;
  /** Riders served when the integration started (both counters are counted from here). */
  servedBase: number;
}

/** Below this relative gap (%) a total reads as "even". */
export const DEAD_BAND_PCT = 3;
/** …or below this absolute gap, in the line's own unit (minutes / riders). */
const DEAD_BAND_ABS: Record<ScoreKey, number> = { wait: 1, stuck: 1, crowded: 1, served: 1 };
/** Gaps between metrics messages longer than this (sim s) are not integrated (paused socket, resume). */
const MAX_STEP_S = 900;

const finite = (v: number | undefined): v is number => typeof v === "number" && Number.isFinite(v);

/** Fold one metrics message into the running totals (returns null while no twin exists). */
export function accumulateScore(state: AiScoreState | null, m: MetricsMessage): AiScoreState | null {
  if (!m.twin || m.ai_since_t == null) return null;
  const live = m.live;
  const twin = m.twin;
  const fresh = !state || state.since !== m.ai_since_t || m.t < state.lastT;
  if (fresh) {
    return {
      since: m.ai_since_t,
      from: Math.max(m.ai_since_t, m.t),
      lastT: m.t,
      last: { live, twin },
      sums: { wait: [0, 0], stuck: [0, 0], crowded: [0, 0] },
      served: [live.riders_served, twin.riders_served],
      verdicts: {},
      servedBase: Math.min(live.riders_served, twin.riders_served),
    };
  }
  const dt = m.t - state.lastT;
  if (dt <= 0) return state;
  const sums = { wait: [...state.sums.wait], stuck: [...state.sums.stuck], crowded: [...state.sums.crowded] } as AiScoreState["sums"];
  if (dt <= MAX_STEP_S) {
    const trap = (k: keyof MetricValues, a: MetricValues, b: MetricValues) => {
      const x = a[k];
      const y = b[k];
      return finite(x) && finite(y) ? ((x + y) / 2) * dt : 0;
    };
    sums.wait[0] += trap("riders_waiting", state.last.live, live);
    sums.wait[1] += trap("riders_waiting", state.last.twin, twin);
    sums.stuck[0] += trap("tunnel_stuck_trains", state.last.live, live);
    sums.stuck[1] += trap("tunnel_stuck_trains", state.last.twin, twin);
    sums.crowded[0] += trap("overcrowded_platforms", state.last.live, live);
    sums.crowded[1] += trap("overcrowded_platforms", state.last.twin, twin);
  }
  const next: AiScoreState = {
    ...state,
    lastT: m.t,
    last: { live, twin },
    sums,
    served: [finite(live.riders_served) ? live.riders_served : state.served[0], finite(twin.riders_served) ? twin.riders_served : state.served[1]],
  };
  next.verdicts = Object.fromEntries(scoreLines(next).map((l) => [l.key, l.verdict]));
  return next;
}

function line(
  key: ScoreKey,
  label: string,
  short: string,
  unit: string,
  better: "lower" | "higher",
  ai: number,
  normal: number,
  prev: Verdict | undefined,
): ScoreLine {
  const diff = ai - normal;
  const base = Math.max(Math.abs(normal), Math.abs(ai));
  const pct = Math.abs(normal) > 1e-9 ? (diff / Math.abs(normal)) * 100 : null;
  const good = better === "lower" ? diff < 0 : diff > 0;
  const raw: Verdict = good ? "ahead" : "behind";
  // Hysteresis: entering "ahead"/"behind" needs the full dead band; leaving it back
  // to "even" needs the gap to shrink to half of it, so a total hovering at the edge
  // doesn't flip the verdict every second.
  const k = prev === raw ? 0.5 : 1;
  const small = Math.abs(diff) < DEAD_BAND_ABS[key] * k || (base > 0 && (Math.abs(diff) / base) * 100 < DEAD_BAND_PCT * k);
  return { key, label, short, unit, better, ai, normal, diff, pct, verdict: small ? "even" : raw };
}

/** The four since-takeover comparisons, headline (rider-minutes waited) first. */
export function scoreLines(s: AiScoreState): ScoreLine[] {
  const min = (v: number) => v / 60;
  const p = s.verdicts ?? {};
  return [
    line("wait", "Rider-minutes waited", "Rider-min waited", "rider-min", "lower", min(s.sums.wait[0]), min(s.sums.wait[1]), p.wait),
    line("stuck", "Train-minutes stuck in tunnels", "Train-min stuck", "train-min", "lower", min(s.sums.stuck[0]), min(s.sums.stuck[1]), p.stuck),
    line("crowded", "Platform-minutes overcrowded", "Crowded platform-min", "platform-min", "lower", min(s.sums.crowded[0]), min(s.sums.crowded[1]), p.crowded),
    line("served", "Riders served", "Riders served", "riders", "higher", s.served[0] - s.servedBase, s.served[1] - s.servedBase, p.served),
  ];
}

/** "AI ahead on 2 of 4" / "AI behind on 1 of 4" / "Even so far". */
export function verdictText(lines: readonly ScoreLine[]): { text: string; tone: "good" | "bad" | "flat" } {
  const ahead = lines.filter((l) => l.verdict === "ahead").length;
  const behind = lines.filter((l) => l.verdict === "behind").length;
  const n = lines.length;
  if (ahead === 0 && behind === 0) return { text: "Even so far", tone: "flat" };
  if (ahead >= behind) {
    const tail = behind ? ` · behind on ${behind}` : "";
    return { text: `AI ahead on ${ahead} of ${n}${tail}`, tone: ahead > behind ? "good" : "flat" };
  }
  const tail = ahead ? ` · ahead on ${ahead}` : "";
  return { text: `AI behind on ${behind} of ${n}${tail}`, tone: "bad" };
}
