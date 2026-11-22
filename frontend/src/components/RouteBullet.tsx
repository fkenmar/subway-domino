"use client";

import { routeBullet } from "@/lib/network";
import { useStore } from "@/lib/store";
import styles from "./RouteBullet.module.css";

/** Fallback colours (official MTA palette) used before the geo file has loaded. */
const FALLBACK: Record<string, [string, string]> = {
  "1": ["#D82233", "#FFFFFF"],
  "2": ["#D82233", "#FFFFFF"],
  "3": ["#D82233", "#FFFFFF"],
  "4": ["#009952", "#FFFFFF"],
  "5": ["#009952", "#FFFFFF"],
  "6": ["#009952", "#FFFFFF"],
  "6X": ["#009952", "#FFFFFF"],
  "7": ["#9A38A1", "#FFFFFF"],
  "7X": ["#9A38A1", "#FFFFFF"],
  A: ["#0062CF", "#FFFFFF"],
  C: ["#0062CF", "#FFFFFF"],
  E: ["#0062CF", "#FFFFFF"],
  B: ["#EB6800", "#FFFFFF"],
  D: ["#EB6800", "#FFFFFF"],
  F: ["#EB6800", "#FFFFFF"],
  FX: ["#EB6800", "#FFFFFF"],
  M: ["#EB6800", "#FFFFFF"],
  G: ["#799534", "#FFFFFF"],
  J: ["#8E5C33", "#FFFFFF"],
  Z: ["#8E5C33", "#FFFFFF"],
  L: ["#7C858C", "#FFFFFF"],
  N: ["#F6BC26", "#000000"],
  Q: ["#F6BC26", "#000000"],
  R: ["#F6BC26", "#000000"],
  W: ["#F6BC26", "#000000"],
  GS: ["#7C858C", "#FFFFFF"],
  FS: ["#7C858C", "#FFFFFF"],
  H: ["#7C858C", "#FFFFFF"],
};

export interface RouteBulletProps {
  /** Route id ("4", "A", "6X"…) or route index into the geo file. */
  id?: string;
  routeIdx?: number;
  size?: number;
}

/** An MTA-style route bullet (diamond for express variants). */
export default function RouteBullet({ id, routeIdx, size = 18 }: RouteBulletProps) {
  const net = useStore((s) => s.network);
  const idx = routeIdx ?? (id != null ? net?.routeById.get(id) : undefined);
  const route = idx != null ? net?.routes[idx] : undefined;
  const rid = route?.id ?? id ?? "?";
  const shape = route ? { text: route.bullet, diamond: route.diamond } : routeBullet(rid);
  const [bg, fg] = route ? [route.color, route.textColor] : (FALLBACK[rid] ?? ["#5b6678", "#FFFFFF"]);
  return (
    <span
      className={shape.diamond ? styles.diamond : styles.bullet}
      style={{ width: size, height: size, fontSize: Math.round(size * 0.6), background: bg, color: fg }}
      role="img"
      aria-label={`${rid} train`}
      title={route?.name ?? `${rid} train`}
    >
      <span className={styles.text}>{shape.text}</span>
    </span>
  );
}

/** Render a list of route indices as bullets, de-duplicated by id. */
export function RouteBullets({ routes, size = 16, max = 8 }: { routes: readonly number[]; size?: number; max?: number }) {
  const unique = [...new Set(routes)].slice(0, max);
  return (
    <span className={styles.row}>
      {unique.map((r) => (
        <RouteBullet key={r} routeIdx={r} size={size} />
      ))}
    </span>
  );
}

/** "4 5 6 · Lexington Av" → bullets for the leading route ids, then the text. */
export function SubtitleWithBullets({ text, size = 15 }: { text: string; size?: number }) {
  const [head, ...rest] = text.split(" · ");
  const tokens = (head ?? "").trim().split(/\s+/);
  const allRoutes = tokens.length > 0 && tokens.every((t) => t in FALLBACK);
  if (!allRoutes) return <span className={styles.subtitle}>{text}</span>;
  return (
    <span className={styles.subtitle}>
      <span className={styles.row}>
        {tokens.map((t) => (
          <RouteBullet key={t} id={t} size={size} />
        ))}
      </span>
      {rest.length > 0 && <span className={styles.subtitleText}>{rest.join(" · ")}</span>}
    </span>
  );
}
