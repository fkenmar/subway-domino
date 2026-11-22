"use client";

import { sinceText, targetAnchor } from "@/lib/disruptions";
import { formatClock } from "@/lib/format";
import type { DisruptionInfo } from "@/lib/protocol";
import { getSession } from "@/lib/session";
import { useStore } from "@/lib/store";
import styles from "./Panels.module.css";

export const KIND_DOT: Record<DisruptionInfo["kind"], string> = {
  full_station_closure: "var(--bad)",
  local_block: "var(--bad)",
  express_block: "var(--bad)",
  signal_delay: "var(--warn)",
};

/**
 * Active blockages (inside "Break the city"): each one stays until cleared, so every
 * row says since when, can be located on the map, and has its own "Restore" button;
 * "Clear all" restores every line at once. No countdowns.
 */
export default function DisruptionList({ onLocate }: { onLocate?: () => void }) {
  const disruptions = useStore((s) => s.disruptions);
  const net = useStore((s) => s.network);
  const requestCamera = useStore((s) => s.requestCamera);
  const select = useStore((s) => s.select);

  if (disruptions.length === 0) return null;

  const locate = (d: DisruptionInfo) => {
    if (!net) return;
    const a = targetAnchor(net, d.target);
    if (!a) return;
    requestCamera({ kind: "view", view: { cx: a.x, cy: a.y, mPerPx: 4 } });
    select({ target: d.target, wx: a.x, wy: a.y, disruptionId: d.id });
    onLocate?.();
  };

  return (
    <section className={styles.active} aria-label="Active blockages">
      <header className={styles.head}>
        <span className={styles.kicker}>
          Blocked now <span className={styles.count}>{disruptions.length}</span>
        </span>
        <button type="button" className={styles.linkButton} onClick={() => getSession().clearAll()}>
          Clear all
        </button>
      </header>
      <ul className={styles.dlist}>
        {disruptions.map((d) => (
          <li key={d.id} className={styles.ditem}>
            <button type="button" className={styles.dlocate} onClick={() => locate(d)} title="Show on map">
              <span className={styles.ddot} style={{ background: KIND_DOT[d.kind] }} aria-hidden="true" />
              <span className={styles.dtext}>
                <span className={styles.dlabel}>{d.label || d.kind}</span>
                <span className={styles.dtime} title={sinceText(d)}>
                  since {formatClock(d.start_t)}
                </span>
              </span>
            </button>
            <button
              type="button"
              className={styles.drestore}
              onClick={() => getSession().clearDisruption(d.id)}
              aria-label={`Restore service: ${d.label || d.kind}`}
              title="Restore service"
            >
              Restore
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
