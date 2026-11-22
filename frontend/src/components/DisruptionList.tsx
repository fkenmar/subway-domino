"use client";

import { remainingS, targetAnchor } from "@/lib/disruptions";
import { formatCountdown } from "@/lib/format";
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

/** Active disruptions with countdowns, locate and clear actions (inside "Break the city"). */
export default function DisruptionList({ onLocate }: { onLocate?: () => void }) {
  const disruptions = useStore((s) => s.disruptions);
  const clockS = useStore((s) => s.clockS);
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
    <section className={styles.active} aria-label="Active disruptions">
      <header className={styles.head}>
        <span className={styles.kicker}>
          Active now <span className={styles.count}>{disruptions.length}</span>
        </span>
        {disruptions.length > 1 && (
          <button type="button" className={styles.linkButton} onClick={() => getSession().clearAll()}>
            Clear all
          </button>
        )}
      </header>
      <ul className={styles.dlist}>
        {disruptions.map((d) => {
          const left = remainingS(d, clockS);
          return (
            <li key={d.id} className={styles.ditem}>
              <button type="button" className={styles.dlocate} onClick={() => locate(d)} title="Show on map">
                <span className={styles.ddot} style={{ background: KIND_DOT[d.kind] }} aria-hidden="true" />
                <span className={styles.dlabel}>{d.label || d.kind}</span>
              </button>
              <span className={styles.dtime} aria-label={left == null ? "until cleared" : `${formatCountdown(left)} left`}>
                {left == null ? "∞" : formatCountdown(left)}
              </span>
              <button
                type="button"
                className={styles.dclear}
                onClick={() => getSession().clearDisruption(d.id)}
                aria-label={`Clear: ${d.label}`}
                title="Clear"
              >
                <svg viewBox="0 0 12 12" aria-hidden="true">
                  <path d="M3 3l6 6M9 3l-6 6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                </svg>
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
