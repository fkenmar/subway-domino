import { useMemo } from "react";
import type { MetricValues } from "@/lib/protocol";
import type { DisruptionWindow, MetricsPoint } from "@/lib/store";
import styles from "./MetricsHUD.module.css";

interface SparklineProps {
  points: readonly MetricsPoint[];
  metric: keyof MetricValues;
  width?: number;
  height?: number;
  /** Colour of the live (AI or Normal) series. */
  color: string;
  label: string;
  /** Disruption windows (sim time) to shade behind the lines. */
  windows?: readonly DisruptionWindow[];
}

/**
 * Smallest vertical span per metric, so a flat series doesn't wobble across the
 * whole box; waits and counts are drawn from zero (a consistent, honest scale).
 */
const MIN_SPAN: Partial<Record<keyof MetricValues, number>> = {
  avg_wait_min: 4,
  tunnel_stuck_trains: 4,
  overcrowded_platforms: 4,
};

/** Centred moving average (half-window `w`). */
export function movingAverage(values: readonly number[], w = 2): number[] {
  return values.map((_, i) => {
    let sum = 0;
    let n = 0;
    for (let k = Math.max(0, i - w); k <= Math.min(values.length - 1, i + w); k++) {
      sum += values[k];
      n++;
    }
    return sum / n;
  });
}

/** SVG path of a monotone cubic through the points (Fritsch–Carlson): smooth, never overshoots. */
export function monotonePath(xs: readonly number[], ys: readonly number[]): string {
  const n = xs.length;
  if (n === 0) return "";
  const f = (v: number) => v.toFixed(1);
  if (n === 1) return `M${f(xs[0])},${f(ys[0])}`;
  const m: number[] = [];
  for (let i = 0; i < n - 1; i++) m.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i] || 1));
  const t: number[] = [m[0]];
  for (let i = 1; i < n - 1; i++) {
    const a = m[i - 1];
    const b = m[i];
    if (a * b <= 0) t.push(0);
    else {
      const h0 = xs[i] - xs[i - 1];
      const h1 = xs[i + 1] - xs[i];
      t.push((3 * (h0 + h1)) / ((2 * h1 + h0) / a + (h1 + 2 * h0) / b));
    }
  }
  t.push(m[n - 2]);
  let d = `M${f(xs[0])},${f(ys[0])}`;
  for (let i = 0; i < n - 1; i++) {
    const h = (xs[i + 1] - xs[i]) / 3;
    d += `C${f(xs[i] + h)},${f(ys[i] + t[i] * h)} ${f(xs[i + 1] - h)},${f(ys[i + 1] - t[i + 1] * h)} ${f(xs[i + 1])},${f(ys[i + 1])}`;
  }
  return d;
}

/** Shaded x-ranges (px) of the disruption windows that overlap [t0, t1]. */
export function windowBands(
  windows: readonly DisruptionWindow[],
  t0: number,
  t1: number,
  x: (t: number) => number,
): { x0: number; x1: number; ended: boolean }[] {
  const out: { x0: number; x1: number; ended: boolean }[] = [];
  for (const w of windows) {
    const end = w.ended && w.end_t != null ? w.end_t : t1;
    if (end < t0 || w.start_t > t1) continue;
    const a = x(Math.max(t0, w.start_t));
    const b = x(Math.min(t1, end));
    if (b - a >= 0.5) out.push({ x0: a, x1: b, ended: w.ended && end <= t1 });
  }
  return out;
}

/** Recent trend of one metric, live vs the Normal twin (solid grey), smoothed; disruptions as a red baseline. */
export default function Sparkline({ points, metric, width = 56, height = 20, color, label, windows }: SparklineProps) {
  const paths = useMemo(() => {
    if (points.length < 2) return null;
    const t0 = points[0].t;
    const t1 = points[points.length - 1].t;
    const span = Math.max(1, t1 - t0);
    const live: { t: number; v: number }[] = [];
    const twin: { t: number; v: number }[] = [];
    for (const p of points) {
      const a = p.live[metric];
      if (typeof a === "number" && Number.isFinite(a)) live.push({ t: p.t, v: a });
      const b = p.twin?.[metric];
      if (typeof b === "number" && Number.isFinite(b)) twin.push({ t: p.t, v: b });
    }
    if (live.length < 2) return null;
    const all = [...live, ...twin].map((p) => p.v);
    const minSpan = MIN_SPAN[metric];
    let lo: number;
    let hi: number;
    if (minSpan !== undefined) {
      lo = 0;
      hi = Math.max(minSpan, Math.max(...all) * 1.12);
    } else {
      lo = Math.min(...all);
      hi = Math.max(...all);
      if (hi - lo < 1e-6) {
        hi += 1;
        lo -= 1;
      }
    }
    const pad = 2;
    const x = (t: number) => pad + ((t - t0) / span) * (width - 2 * pad);
    const y = (v: number) => height - pad - ((v - lo) / (hi - lo)) * (height - 2 * pad);
    const build = (series: { t: number; v: number }[]) => {
      if (series.length < 2) return "";
      const ys = movingAverage(series.map((p) => p.v)).map(y);
      return monotonePath(
        series.map((p) => x(p.t)),
        ys,
      );
    };
    const smoothLive = movingAverage(live.map((p) => p.v));
    const last = live[live.length - 1];
    return {
      live: build(live),
      twin: build(twin),
      dot: { cx: x(last.t), cy: y(smoothLive[smoothLive.length - 1]) },
      bands: windows ? windowBands(windows, t0, t1, x) : [],
    };
  }, [points, metric, width, height, windows]);

  return (
    <svg className={styles.spark} width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
      {/* Disruption window: a thin red baseline segment with a start tick (and an end tick once over). */}
      {paths?.bands.map((b, i) => (
        <g key={i} className={styles.sparkBand}>
          <rect x={b.x0} y={height - 1.6} width={b.x1 - b.x0} height={1.6} />
          <rect x={b.x0} y={height - 5} width={1} height={5} />
          {b.ended && <rect x={b.x1 - 1} y={height - 5} width={1} height={5} />}
        </g>
      ))}
      {paths?.twin && <path d={paths.twin} className={styles.sparkTwin} />}
      {paths?.live && <path d={paths.live} stroke={color} className={styles.sparkLive} />}
      {paths?.dot && <circle cx={paths.dot.cx} cy={paths.dot.cy} r={2} fill={color} />}
      {!paths && <line x1={2} x2={width - 2} y1={height - 2} y2={height - 2} className={styles.sparkEmpty} />}
    </svg>
  );
}
