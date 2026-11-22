"use client";

import { HUD_METRICS, type MetricInfo } from "@/lib/copy";
import { formatCompact, formatInt, formatMinutes, formatSigned, formatSpan, percentChange } from "@/lib/format";
import type { MetricValues } from "@/lib/protocol";
import { useStore } from "@/lib/store";
import styles from "./MetricsHUD.module.css";
import Sparkline from "./Sparkline";

const AI_COLOR = "#5ac8fa";
const NORMAL_COLOR = "#c9d3e1";

function fmtValue(info: MetricInfo, v: number | undefined, compact = false): string {
  if (v == null || !Number.isFinite(v)) return "—";
  if (info.key === "avg_wait_min") return formatMinutes(v);
  return compact ? formatCompact(v) : formatInt(v);
}

/** Full value for roomy layouts plus a compact twin (e.g. "579k") that CSS swaps in on phones. */
function Value({ info, v }: { info: MetricInfo; v: number | undefined }) {
  const full = fmtValue(info, v);
  const compact = fmtValue(info, v, true);
  if (full === compact) return <>{full}</>;
  return (
    <>
      <span className={styles.full}>{full}</span>
      <span className={styles.compact}>{compact}</span>
    </>
  );
}

function Delta({ info, live, twin }: { info: MetricInfo; live: number; twin: number }) {
  const diff = live - twin;
  const fmt = (x: number) => fmtValue(info, x, true);
  const eps = info.key === "avg_wait_min" ? 0.05 : 0.5;
  const better = info.better === "lower" ? diff < -eps : diff > eps;
  const worse = info.better === "lower" ? diff > eps : diff < -eps;
  const pct = percentChange(live, twin);
  const tone = better ? styles.good : worse ? styles.bad : styles.flat;
  // A flat delta ("±0.0 min") never carries a percentage that would contradict it.
  const pctText =
    (better || worse) && pct !== null && Math.abs(pct) >= 1 && info.key !== "riders_served"
      ? `${pct > 0 ? "+" : "−"}${Math.abs(pct).toFixed(0)}%`
      : null;
  return (
    <span className={`${styles.delta} ${tone}`}>
      {formatSigned(diff, fmt)}
      {info.unit && ` ${info.unit}`}
      {pctText && <span className={styles.pct}>{pctText}</span>}
    </span>
  );
}

/** Live metrics; in AI mode the AI values sit beside "Normal would be" twin values. */
export default function MetricsHUD() {
  const metrics = useStore((s) => s.metrics);
  const history = useStore((s) => s.history);
  const mode = useStore((s) => s.settings.mode);
  const twinOn = metrics?.twin != null;
  const live: MetricValues | null = metrics?.live ?? null;
  const twin: MetricValues | null = metrics?.twin ?? null;
  const aiFor = metrics?.ai_since_t != null ? metrics.t - metrics.ai_since_t : null;
  const color = mode === "ai" ? AI_COLOR : NORMAL_COLOR;

  return (
    <section className={`panel ${styles.hud}`} aria-label="Live metrics" data-mode={mode}>
      <header className={styles.head}>
        <span className={styles.kicker}>Rider outcomes</span>
        <span className={styles.modeTag} data-mode={mode}>
          {twinOn ? "AI vs Normal twin" : mode === "ai" ? "AI Dispatcher" : "Normal Schedule"}
        </span>
      </header>

      <ul className={styles.list}>
        {HUD_METRICS.map((info) => {
          const v = live?.[info.key];
          const t = twin?.[info.key];
          return (
            <li key={info.key} className={styles.row} title={info.help}>
              <div className={styles.rowTop}>
                <span className={styles.label}>
                  <span className={styles.full}>{info.label}</span>
                  <span className={styles.compact}>{info.short}</span>
                </span>
                <Sparkline
                  points={history}
                  metric={info.key}
                  color={color}
                  label={`${info.label}, last 10 sim-minutes${twinOn ? " (dashed: Normal twin)" : ""}`}
                />
              </div>
              <div className={styles.values}>
                <span className={styles.value}>
                  <Value info={info} v={v} />
                  {info.unit && <small>{info.unit}</small>}
                </span>
                {twinOn && t != null && v != null && Number.isFinite(v) && Number.isFinite(t) && (
                  <span className={styles.compare}>
                    <span className={styles.twin}>
                      <span className={styles.full}>Normal would be </span>
                      <span className={styles.compact}>vs </span>
                      <b>
                        <Value info={info} v={t} />
                      </b>
                    </span>
                    <Delta info={info} live={v} twin={t} />
                  </span>
                )}
              </div>
            </li>
          );
        })}
      </ul>

      <footer className={styles.foot}>
        {live ? (
          <>
            <span>{formatInt(live.trains_active)} trains</span>
            <span aria-hidden="true">·</span>
            <span>{formatCompact(live.riders_waiting)} waiting</span>
            {aiFor != null && aiFor >= 0 && (
              <>
                <span aria-hidden="true">·</span>
                <span className={styles.aiFor}>AI in control {formatSpan(aiFor)}</span>
              </>
            )}
          </>
        ) : (
          <span>Waiting for the live simulation…</span>
        )}
      </footer>
      {metrics && metrics.effective_speed < 0.97 && (
        <p className={styles.governor}>
          Server busy — running at {Math.round(metrics.effective_speed * 100)}% speed for everyone.
        </p>
      )}
      {!twinOn && mode === "normal" && live && (
        <p className={styles.hint}>Turn on the AI Dispatcher to race it against a Normal-Schedule twin of this exact city.</p>
      )}
    </section>
  );
}
