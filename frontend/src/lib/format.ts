/** Display formatting (pure, locale-stable: en-US like MTA signage). */

const intFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const oneDecimal = new Intl.NumberFormat("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

export const DASH = "—";

/** Sim clock → wall-clock time of day, e.g. 30120 → "8:22 AM". */
export function formatClock(clockS: number | null | undefined, opts: { seconds?: boolean } = {}): string {
  if (clockS == null || !Number.isFinite(clockS)) return opts.seconds ? "--:--:--" : "--:--";
  const day = ((Math.floor(clockS) % 86400) + 86400) % 86400;
  const h24 = Math.floor(day / 3600);
  const m = Math.floor((day % 3600) / 60);
  const h12 = h24 % 12 === 0 ? 12 : h24 % 12;
  const sec = opts.seconds ? `:${String(day % 60).padStart(2, "0")}` : "";
  return `${h12}:${String(m).padStart(2, "0")}${sec} ${h24 < 12 ? "AM" : "PM"}`;
}

/** Seconds → "m:ss" (or "h:mm:ss"). Negative values clamp to 0. */
export function formatCountdown(seconds: number): string {
  const s = Math.max(0, Math.ceil(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  return `${m}:${String(sec).padStart(2, "0")}`;
}

/** Seconds → "12 min" / "45 s" / "1 h 05 min" for prose. */
export function formatSpan(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} s`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min`;
  return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, "0")} min`;
}

/** Seconds → a duration that can't be mistaken for a time of day: "42s", "8m 40s", "1h 05m". */
export function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
}

/** Split the server's aggregated-callout suffix " (+N more dispatch actions)" off a callout sentence. */
export function splitMoreSuffix(text: string): { text: string; more: number } {
  const m = /\s*\(\+(\d+) more dispatch actions?\)\s*$/.exec(text);
  return m ? { text: text.slice(0, m.index), more: Number(m[1]) } : { text, more: 0 };
}

export function formatInt(n: number | null | undefined): string {
  return n == null || !Number.isFinite(n) ? DASH : intFmt.format(Math.round(n));
}

export function formatMinutes(n: number | null | undefined): string {
  return n == null || !Number.isFinite(n) ? DASH : `${oneDecimal.format(n)}`;
}

/** Compact counts for tight spaces: 1234 → "1.2k", 1_234_567 → "1.23M". */
export function formatCompact(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return DASH;
  const a = Math.abs(n);
  if (a >= 1e6) return `${(n / 1e6).toFixed(a >= 1e7 ? 1 : 2)}M`;
  if (a >= 1e4) return `${(n / 1e3).toFixed(0)}k`;
  if (a >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return intFmt.format(Math.round(n));
}

/**
 * Signed delta with the metric's own formatter, e.g. "+3", "−1.2". A delta that
 * rounds to zero at the formatter's precision reads "±0" / "±0.0", never "−0.0".
 */
export function formatSigned(n: number, fmt: (v: number) => string): string {
  if (!Number.isFinite(n)) return DASH;
  const magnitude = fmt(Math.abs(n));
  if (Math.abs(n) < 1e-9 || /^0(?:\.0+)?$/.test(magnitude)) return `±${magnitude === DASH ? "0" : magnitude}`;
  return `${n > 0 ? "+" : "−"}${magnitude}`;
}

/** Relative change in percent, guarded for tiny baselines. */
export function percentChange(value: number, baseline: number): number | null {
  if (!Number.isFinite(value) || !Number.isFinite(baseline) || Math.abs(baseline) < 1e-9) return null;
  return ((value - baseline) / Math.abs(baseline)) * 100;
}

export function pluralize(n: number, one: string, many = `${one}s`): string {
  return `${formatInt(n)} ${n === 1 ? one : many}`;
}
