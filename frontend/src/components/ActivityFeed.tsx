"use client";

import { ACTION_TITLE } from "@/lib/copy";
import { targetAnchor } from "@/lib/disruptions";
import { formatClock } from "@/lib/format";
import { DISPATCH_ACTION_NAMES } from "@/lib/protocol";
import { type FeedItem, useStore } from "@/lib/store";
import { KIND_DOT } from "./DisruptionList";
import styles from "./Panels.module.css";
import RouteBullet from "./RouteBullet";

const EVENT_VERB = {
  disruption_started: "Started",
  disruption_cleared: "Cleared",
  disruption_expired: "Ended",
} as const;

/** Every AI dispatch action (and disruption event) of this session, newest first. */
export default function ActivityFeed() {
  const feed = useStore((s) => s.feed);
  const open = useStore((s) => s.feedOpen);
  const setOpen = useStore((s) => s.setFeedOpen);
  const actions = useStore((s) => s.metrics?.actions ?? null);
  const net = useStore((s) => s.network);
  const requestCamera = useStore((s) => s.requestCamera);
  const setHighlight = useStore((s) => s.setHighlight);
  const total = actions ? DISPATCH_ACTION_NAMES.reduce((a, n) => a + (actions[n] ?? 0), 0) : 0;

  const focus = (item: FeedItem) => {
    if (!net) return;
    if (item.kind === "callout") {
      const st = net.stations[item.event.station_idx];
      if (!st) return;
      requestCamera({ kind: "view", view: { cx: st.x, cy: st.y, mPerPx: 3.5 } });
      setHighlight(st.idx);
    } else {
      const a = targetAnchor(net, item.event.disruption.target);
      if (a) requestCamera({ kind: "view", view: { cx: a.x, cy: a.y, mPerPx: 4 } });
    }
  };

  return (
    <section className={`panel ${styles.card} ${styles.feedCard}`} aria-label="Dispatch log" data-open={open}>
      <header className={styles.head}>
        <button type="button" className={styles.disclosure} aria-expanded={open} onClick={() => setOpen(!open)}>
          <span className={styles.kicker}>
            Dispatch log <span className={styles.count}>{total}</span>
          </span>
          <svg className={styles.chev} viewBox="0 0 10 10" aria-hidden="true">
            <path d="M2 3.5l3 3 3-3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </button>
      </header>
      {open && (
        <ol className={styles.feed}>
          {feed.length === 0 && (
            <li className={styles.empty}>
              No dispatch actions yet. Break something, then hand control to the AI Dispatcher — every move it makes
              shows up here.
            </li>
          )}
          {feed.map((item) => (
            <li key={item.id}>
              <button type="button" className={styles.fitem} onClick={() => focus(item)}>
                <time className={styles.ftime}>{formatClock(item.t)}</time>
                {item.kind === "callout" ? (
                  <span className={styles.fbody}>
                    <span className={styles.fhead}>
                      {item.event.route && <RouteBullet id={item.event.route} size={15} />}
                      <span className={styles.faction}>{ACTION_TITLE[item.event.action] ?? item.event.action}</span>
                    </span>
                    <span className={styles.ftext}>{item.event.text}</span>
                  </span>
                ) : (
                  <span className={styles.fbody}>
                    <span className={styles.fhead}>
                      <span
                        className={styles.ddot}
                        style={{ background: KIND_DOT[item.event.disruption.kind] }}
                        aria-hidden="true"
                      />
                      <span
                        className={styles.faction}
                        data-tone={item.event.kind === "disruption_started" ? "bad" : "muted"}
                      >
                        {EVENT_VERB[item.event.kind]}
                      </span>
                    </span>
                    <span className={styles.ftext}>{item.event.disruption.label}</span>
                  </span>
                )}
              </button>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

/** "Log" button for narrow layouts, where the dispatch log opens as a sheet above the dock. */
export function ActivityFeedToggle() {
  const open = useStore((s) => s.feedOpen);
  const setOpen = useStore((s) => s.setFeedOpen);
  const total = useStore((s) => {
    const a = s.metrics?.actions;
    return a ? DISPATCH_ACTION_NAMES.reduce((sum, n) => sum + (a[n] ?? 0), 0) : 0;
  });
  return (
    <button
      type="button"
      className={styles.logToggle}
      aria-expanded={open}
      aria-label={`Dispatch log (${total} actions)`}
      onClick={() => setOpen(!open)}
    >
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <path d="M3 4h10M3 8h10M3 12h6" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
      </svg>
      <span className={styles.logLabel}>Log</span>
      {total > 0 && <span className={styles.count}>{total}</span>}
    </button>
  );
}
