import { useMemo } from "react";
import type { MetricValues } from "@/lib/protocol";
import { HISTORY_WINDOW_S, type MetricsPoint } from "@/lib/store";
import styles from "./MetricsHUD.module.css";

interface SparklineProps {
  points: readonly MetricsPoint[];
  metric: keyof MetricValues;
  width?: number;
  height?: number;
  /** Colour of the live (AI or Normal) series. */
  color: string;
  label: string;
}

/** Live vs twin over the last 10 sim-minutes (twin dashed). */
export default function Sparkline({ points, metric, width = 104, height = 30, color, label }: SparklineProps) {
  const paths = useMemo(() => {
    if (points.length < 2) return null;
    const tMax = points[points.length - 1].t;
    const tMin = tMax - HISTORY_WINDOW_S;
    let lo = Infinity;
    let hi = -Infinity;
    for (const p of points) {
      for (const v of [p.live[metric], p.twin?.[metric]]) {
        if (typeof v === "number" && Number.isFinite(v)) {
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      }
    }
    if (!Number.isFinite(lo)) return null;
    if (hi - lo < 1e-6) {
      hi += 1;
      lo -= 1;
    }
    const pad = 2.5;
    const x = (t: number) => pad + ((t - tMin) / HISTORY_WINDOW_S) * (width - 2 * pad);
    const y = (v: number) => height - pad - ((v - lo) / (hi - lo)) * (height - 2 * pad);
    const build = (get: (p: MetricsPoint) => number | undefined) => {
      let d = "";
      for (const p of points) {
        const v = get(p);
        if (typeof v !== "number" || !Number.isFinite(v)) continue;
        d += `${d ? "L" : "M"}${x(p.t).toFixed(1)},${y(v).toFixed(1)}`;
      }
      return d;
    };
    const live = build((p) => p.live[metric]);
    const twin = build((p) => p.twin?.[metric]);
    const last = points[points.length - 1];
    const lastV = last.live[metric];
    const dot = typeof lastV === "number" && Number.isFinite(lastV) ? { cx: x(last.t), cy: y(lastV) } : null;
    return { live, twin, dot };
  }, [points, metric, width, height]);

  return (
    <svg className={styles.spark} width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
      {paths?.twin && <path d={paths.twin} className={styles.sparkTwin} />}
      {paths?.live && <path d={paths.live} stroke={color} className={styles.sparkLive} />}
      {paths?.dot && <circle cx={paths.dot.cx} cy={paths.dot.cy} r={2.2} fill={color} />}
      {!paths && <line x1={3} x2={width - 3} y1={height / 2} y2={height / 2} className={styles.sparkEmpty} />}
    </svg>
  );
}
