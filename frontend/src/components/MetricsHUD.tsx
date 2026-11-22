"use client";

import { type ReactNode, useMemo, useState } from "react";
import { type ScoreLine, scoreLines, verdictText } from "@/lib/aiScore";
import { HUD_METRICS, type MetricInfo, type MetricKey } from "@/lib/copy";
import { formatCompact, formatDuration, formatInt, percentChange } from "@/lib/format";
import { useTweenedText } from "@/lib/motion";
import type { MetricValues } from "@/lib/protocol";
import { useStore } from "@/lib/store";
import styles from "./MetricsHUD.module.css";
import Sparkline from "./Sparkline";

/** Sparkline colours on the black board: amber LED for the schedule, signage blue for the AI. */
const AI_COLOR = "#4aa8ff";
const NORMAL_COLOR = "#ffb100";

const LABEL: Record<MetricKey, string> = {
  avg_wait_min: "Avg wait",
  tunnel_stuck_trains: "Tunnel-stuck",
  overcrowded_platforms: "Overcrowded",
  riders_served: "Riders served",
};

const FORMAT: Record<MetricKey, (v: number) => string> = {
  avg_wait_min: (v) => v.toFixed(1),
  tunnel_stuck_trains: (v) => formatInt(v),
  overcrowded_platforms: (v) => formatInt(v),
  // Past a million the full count no longer fits beside its label: "1.39M".
  riders_served: (v) => (Math.abs(v) >= 1e6 ? formatCompact(v) : formatInt(v)),
};

type Tone = "good" | "bad" | "flat";

/** Deltas inside this band read neutral (grey): |Δ| < 3 % of the twin, or ±1 for counts. */
const DEAD_BAND_PCT = 3;

/**
 * AI value against the Normal-Schedule twin right now: −12 % wait, −2 stuck
 * trains, +1.2k riders. Small gaps (inside the dead band) are coloured neutral so
 * noise doesn't read as a result.
 */
export function twinDelta(info: MetricInfo, live: number, twin: number): { text: string; tone: Tone } {
  const diff = live - twin;
  const rel = Math.abs(twin) > 1e-9 ? (Math.abs(diff) / Math.abs(twin)) * 100 : Number.POSITIVE_INFINITY;
  const small = (info.key === "avg_wait_min" ? Math.abs(diff) < 0.05 : Math.abs(diff) <= 1) || rel < DEAD_BAND_PCT;
  const better = !small && (info.better === "lower" ? diff < 0 : diff > 0);
  const worse = !small && !better;
  const tone: Tone = better ? "good" : worse ? "bad" : "flat";
  const sign = diff > 0 ? "+" : "−";
  if (info.key === "avg_wait_min") {
    const pct = percentChange(live, twin);
    if (Math.abs(diff) < 0.05 || pct === null || Math.abs(pct) < 0.5) return { text: "±0%", tone };
    return { text: `${sign}${Math.abs(pct).toFixed(0)}%`, tone };
  }
  if (Math.abs(diff) < 0.5) return { text: "±0", tone };
  if (info.key === "riders_served") return { text: `${sign}${formatCompact(Math.abs(diff))}`, tone };
  return { text: `${sign}${formatInt(Math.abs(diff))}`, tone };
}

const VERDICT_TONE = { ahead: "good", behind: "bad", even: "flat" } as const;

function pct(l: ScoreLine): string {
  if (l.pct === null) return l.diff === 0 ? "±0" : `${l.diff > 0 ? "+" : "−"}${formatCompact(Math.abs(l.diff))}`;
  if (Math.abs(l.pct) < 0.5) return "±0%";
  return `${l.pct > 0 ? "+" : "−"}${Math.abs(l.pct).toFixed(Math.abs(l.pct) < 10 ? 1 : 0)}%`;
}

/**
 * The stable headline in AI mode: cumulative totals since the AI took over (live
 * vs the Normal twin), with a one-line verdict. Hover for all four comparisons.
 */
function Verdict() {
  const score = useStore((s) => s.score);
  const lines = useMemo(() => (score ? scoreLines(score) : null), [score]);
  if (!score || !lines) return null;
  const covered = score.lastT - score.from;
  const partial = score.from - score.since > 90;
  const v = verdictText(lines);
  const head = lines[0];
  const scope = partial ? `last ${formatDuration(covered)}` : "since takeover";
  const detail = lines
    .map((l) => `${l.label}: AI ${formatCompact(l.ai)} · Normal ${formatCompact(l.normal)} (${pct(l)}) — ${l.verdict === "even" ? "even" : l.verdict === "ahead" ? "AI ahead" : "AI behind"}`)
    .join("\n");
  return (
    <div className={styles.verdict} data-tone={v.tone} title={`Totals ${scope}, AI vs the Normal Schedule twin:\n${detail}`}>
      <div className={styles.verdictTop}>
        <span className={styles.verdictText}>{covered < 1 ? "Comparing…" : v.text}</span>
        <span className={styles.verdictScope}>{scope}</span>
      </div>
      <div className={styles.verdictLine}>
        <span className={styles.verdictLabel}>{head.short}</span>
        <span className={styles.verdictNums}>
          <span className={styles.ai}>
            AI <b className="led">{formatCompact(head.ai)}</b>
          </span>
          <span className={styles.vsN}>
            vs <b className="led">{formatCompact(head.normal)}</b>
          </span>
        </span>
        <span className={styles.delta} data-tone={VERDICT_TONE[head.verdict]}>
          {pct(head)}
        </span>
      </div>
      <ul className={styles.verdictDots} aria-label="Per-metric verdicts since takeover">
        {lines.map((l) => (
          <li key={l.key} data-tone={VERDICT_TONE[l.verdict]} title={`${l.label}: AI ${formatCompact(l.ai)} vs Normal ${formatCompact(l.normal)} (${pct(l)})`}>
            {SHORT[l.key]}
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * Row bullets for the arrival board: white-rimmed circles with a pictogram (never a letter,
 * so they can't be mistaken for a real route bullet).
 */
function MetricBullet({ k }: { k: MetricKey }) {
  const glyph: Record<MetricKey, ReactNode> = {
    avg_wait_min: (
      <>
        <circle cx="10" cy="10" r="4.6" fill="none" stroke="currentColor" strokeWidth="1.5" />
        <path d="M10 7.4V10l1.7 1.2" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </>
    ),
    tunnel_stuck_trains: (
      <>
        <path d="M5 14.6V10a5 5 0 0 1 10 0v4.6" fill="none" stroke="currentColor" strokeWidth="1.5" />
        <rect x="7.6" y="9.4" width="4.8" height="4.6" rx="1.2" fill="currentColor" />
      </>
    ),
    overcrowded_platforms: (
      <>
        <circle cx="6.4" cy="7" r="1.35" fill="currentColor" />
        <circle cx="10" cy="7" r="1.35" fill="currentColor" />
        <circle cx="13.6" cy="7" r="1.35" fill="currentColor" />
        <path d="M6.4 9.4v4.4M10 9.4v4.4M13.6 9.4v4.4" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" />
      </>
    ),
    riders_served: <path d="M6.2 10.3l2.6 2.6 5-5.4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />,
  };
  return (
    <svg className={styles.bullet} viewBox="0 0 20 20" aria-hidden="true">
      <circle cx="10" cy="10" r="9" fill="#ffffff" />
      <g color="#111111">{glyph[k]}</g>
    </svg>
  );
}

const SHORT: Record<ScoreLine["key"], string> = { wait: "Wait", stuck: "Stuck", crowded: "Crowding", served: "Served" };

function Value({ info, v }: { info: MetricInfo; v: number | undefined }) {
  const ref = useTweenedText<HTMLSpanElement>(v, FORMAT[info.key]);
  return (
    <span className={styles.value}>
      <span ref={ref} className={`led ${styles.num}`} />
      {info.unit && <small>{info.unit}</small>}
    </span>
  );
}

/**
 * The compact floating scoreboard, styled as a station countdown board (black panel,
 * bullet-led rows, amber dot-matrix digits). Normal mode: four live metrics with smoothed
 * sparklines. AI mode: "AI vs Normal" — a stable verdict from cumulative totals
 * since the AI took over, then each metric right now with its delta against the
 * Normal-Schedule twin (green better, red worse, grey inside the dead band) and
 * the twin's own line on every sparkline. Every number comes from the server's
 * live and twin metrics.
 */
export default function MetricsHUD() {
  const metrics = useStore((s) => s.metrics);
  const trend = useStore((s) => s.trend);
  const windows = useStore((s) => s.disruptionWindows);
  const score = useStore((s) => s.score);
  const mode = useStore((s) => s.settings.mode);
  const [expanded, setExpanded] = useState(false);
  const twinOn = metrics?.twin != null;
  const live: MetricValues | null = metrics?.live ?? null;
  const twin: MetricValues | null = metrics?.twin ?? null;
  const aiFor = metrics?.ai_since_t != null ? metrics.t - metrics.ai_since_t : null;
  const color = mode === "ai" ? AI_COLOR : NORMAL_COLOR;
  const headline = twinOn ? "AI vs Normal" : mode === "ai" ? "AI Dispatcher" : "Normal Schedule";

  // Phones: a pill with the headline metric that expands into the full card.
  const pillVerdict = useMemo(() => (twinOn && score ? verdictText(scoreLines(score)) : null), [twinOn, score]);

  return (
    <section className={`glass sign ${styles.hud}`} aria-label="Live metrics" data-mode={mode} data-twin={twinOn} data-expanded={expanded}>
      <button
        type="button"
        className={styles.pill}
        aria-expanded={expanded}
        aria-label={expanded ? "Hide live metrics" : "Show live metrics"}
        onClick={() => setExpanded((e) => !e)}
      >
        <span className={styles.dot} aria-hidden="true" />
        {expanded ? (
          // Expanded, the pill is the board header (the card's own header hides on phones).
          <>
            <span className={styles.pillHead}>{headline}</span>
            {twinOn && aiFor != null && aiFor >= 0 && (
              <span className={styles.since}>
                since takeover <b className="led">{formatDuration(aiFor)}</b>
              </span>
            )}
          </>
        ) : pillVerdict ? (
          <>
            <span className={styles.pillLabel}>vs Normal</span>
            <span className={styles.delta} data-tone={pillVerdict.tone}>
              {pillVerdict.text.replace(/^AI /, "").replace(/ · .*$/, "")}
            </span>
          </>
        ) : (
          <>
            <span className={styles.pillLabel}>Wait</span>
            <span className={styles.pillValue}>{live ? live.avg_wait_min.toFixed(1) : "—"}</span>
            {live && <span className={styles.pillUnit}>min</span>}
          </>
        )}
        <svg className={styles.chev} viewBox="0 0 10 10" aria-hidden="true">
          <path d="M2 3.5l3 3 3-3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      </button>

      <div className={styles.card}>
        <header className={styles.head}>
          <span className={styles.kicker}>
            <span className={styles.dot} aria-hidden="true" />
            {headline}
          </span>
          {twinOn && aiFor != null && aiFor >= 0 ? (
            <span className={styles.since}>
              since AI took over <b className="led">{formatDuration(aiFor)}</b>
            </span>
          ) : null}
        </header>

        {twinOn && <Verdict />}
        {twinOn && <p className={styles.nowCap}>Right now · vs Normal twin</p>}

        <ul className={styles.list}>
          {HUD_METRICS.map((info) => {
            const v = live?.[info.key];
            const t = twin?.[info.key];
            const d = twinOn && v != null && t != null && Number.isFinite(v) && Number.isFinite(t) ? twinDelta(info, v, t) : null;
            return (
              <li key={info.key} className={styles.row} title={info.help}>
                <MetricBullet k={info.key} />
                <span className={styles.label}>{LABEL[info.key]}</span>
                <Value info={info} v={v} />
                {d ? (
                  <span
                    className={styles.delta}
                    data-tone={d.tone}
                    aria-label={`${d.text} vs Normal right now`}
                    title={`Right now: AI ${FORMAT[info.key](v as number)} · Normal Schedule twin ${FORMAT[info.key](t as number)}${info.unit ? ` ${info.unit}` : ""}`}
                  >
                    {d.text}
                  </span>
                ) : (
                  <span className={styles.deltaEmpty} aria-hidden="true" />
                )}
                <Sparkline
                  width={twinOn ? 44 : 80}
                  points={trend}
                  metric={info.key}
                  color={color}
                  windows={windows}
                  label={`${info.label}, recent trend${twinOn ? " (grey: Normal twin)" : ""}; red baseline: disruption active`}
                />
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
              {windows.length > 0 && <span className={styles.bandKey}>disruption</span>}
            </>
          ) : (
            <span>Waiting for the live simulation…</span>
          )}
        </footer>
        <p className={styles.affil}>Fan-made simulator · Not affiliated with the MTA</p>
        {metrics && metrics.effective_speed < 0.97 && (
          <p className={styles.governor}>Server busy: running at {Math.round(metrics.effective_speed * 100)}% speed.</p>
        )}
      </div>
    </section>
  );
}
