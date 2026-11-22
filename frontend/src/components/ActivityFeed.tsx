"use client";

import { useRef } from "react";
import { posterCopy } from "@/lib/calloutPacer";
import { ACTION_TITLE } from "@/lib/copy";
import { targetAnchor } from "@/lib/disruptions";
import { formatClock, splitMoreSuffix } from "@/lib/format";
import { useDisclosure, useDismiss } from "@/lib/motion";
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

/**
 * Every AI dispatch action (and disruption event) of this session, newest first:
 * a small "Dispatch log · N" chip that opens a drawer, closed by default.
 */
export default function ActivityFeed() {
  const feed = useStore((s) => s.feed);
  const setOpen = useStore((s) => s.setFeedOpen);
  const actions = useStore((s) => s.metrics?.actions ?? null);
  const net = useStore((s) => s.network);
  const requestCamera = useStore((s) => s.requestCamera);
  const setHighlight = useStore((s) => s.setHighlight);
  const total = actions ? DISPATCH_ACTION_NAMES.reduce((a, n) => a + (actions[n] ?? 0), 0) : 0;
  const drawer = useDisclosure(setOpen);
  const isOpen = drawer.mounted && !drawer.closing;
  const chipRef = useRef<HTMLButtonElement>(null);
  const drawerRef = useRef<HTMLElement>(null);
  useDismiss(isOpen, [chipRef, drawerRef], drawer.close);

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
    <>
      <button
        ref={chipRef}
        type="button"
        className={`glass ${styles.chip}`}
        aria-expanded={isOpen}
        aria-controls="dispatch-log"
        data-occlude
        onClick={drawer.toggle}
      >
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <path d="M3 4h10M3 8h10M3 12h6" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
        Dispatch log
        <span className={styles.count} data-zero={total === 0}>
          {total}
        </span>
      </button>
      {drawer.mounted && (
        <section
          ref={drawerRef}
          id="dispatch-log"
          className={`glass ${styles.drawer}`}
          data-state={drawer.closing ? "closing" : "open"}
          aria-label="Dispatch log"
        >
          <header className={styles.drawerHead}>
            <span className={styles.kicker}>
              Dispatch log <span className={styles.count}>{total}</span>
            </span>
            <button type="button" className={styles.close} onClick={drawer.close} aria-label="Close the dispatch log">
              <svg viewBox="0 0 12 12" aria-hidden="true">
                <path d="M3 3l6 6M9 3l-6 6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
              </svg>
            </button>
          </header>
          <ol className={styles.feed}>
            {feed.length === 0 && (
              <li className={styles.empty}>
                No dispatch actions yet. Break something, then hand control to the AI Dispatcher: every move it makes
                shows up here.
              </li>
            )}
            {feed.map((item) => {
              const copy =
                item.kind === "callout"
                  ? posterCopy(item.event.action, net?.stations[item.event.station_idx]?.name, item.event.text)
                  : null;
              const kind = item.kind === "callout" ? "change" : item.event.kind === "disruption_started" ? "alert" : "restored";
              return (
                <li key={item.id}>
                  <button
                    type="button"
                    className={styles.fitem}
                    onClick={() => focus(item)}
                    data-kind={kind}
                    data-major={copy ? copy.major : true}
                  >
                    <span className={styles.ftag}>
                      <span>{copy ? copy.header : kind === "alert" ? "Service alert" : "Service restored"}</span>
                      <time className={styles.ftime}>{formatClock(item.t)}</time>
                    </span>
                    {item.kind === "callout" && copy ? (
                      <span className={styles.fbody}>
                        {item.event.route ? <RouteBullet id={item.event.route} size={26} /> : <span />}
                        <span className={styles.fcopy}>
                          <span className={styles.faction}>{copy.headline}</span>
                          {copy.reason && !item.event.text.toLowerCase().includes(copy.reason.toLowerCase()) && (
                            // The why, unless the server text below already says it word for word.
                            <span className={styles.freason}>{copy.reason}</span>
                          )}
                          <span className={styles.ftext}>
                            {splitMoreSuffix(item.event.text).text}
                            {splitMoreSuffix(item.event.text).more > 0 && (
                              <span className={styles.fmore} title="Similar dispatch actions grouped into this entry">
                                +{splitMoreSuffix(item.event.text).more} similar
                              </span>
                            )}
                          </span>
                          <span className={styles.fby}>
                            AI Dispatcher · {ACTION_TITLE[item.event.action] ?? item.event.action}
                          </span>
                        </span>
                      </span>
                    ) : item.kind !== "callout" ? (
                      <span className={styles.fbody}>
                        <span
                          className={styles.ddot}
                          style={{ background: KIND_DOT[item.event.disruption.kind] }}
                          aria-hidden="true"
                        />
                        <span className={styles.fcopy}>
                          <span className={styles.faction}>{EVENT_VERB[item.event.kind]}</span>
                          <span className={styles.ftext}>{item.event.disruption.label}</span>
                        </span>
                      </span>
                    ) : null}
                  </button>
                </li>
              );
            })}
          </ol>
        </section>
      )}
    </>
  );
}
