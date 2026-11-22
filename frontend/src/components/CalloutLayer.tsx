"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { posterCopy } from "@/lib/calloutPacer";
import { ACTION_TITLE } from "@/lib/copy";
import { EXIT_MS, prefersReducedMotion } from "@/lib/motion";
import { getSession } from "@/lib/session";
import { type ActiveCallout, type Toast, useStore } from "@/lib/store";
import styles from "./CalloutLayer.module.css";
import RouteBullet from "./RouteBullet";

/** How long each toast stays up (ms of unpaused visibility). */
const LIFETIME_MS: Record<Toast["kind"], number> = {
  callout: 6_500,
  notice: 7_000,
  error: 7_000,
};
/** The AI callout leaves after this long without a newer one. */
const CALLOUT_LIFETIME_MS = 7_000;
/** "N actions in the last 10 s" shows from this many. */
const BUSY_COUNT = 3;

/** Unpaused lifetime timer: `paused` freezes it; `onDone` fires once it runs out. */
function useLifetime(totalMs: number, paused: boolean, onDone: () => void): void {
  const remaining = useRef(totalMs);
  useEffect(() => {
    if (paused) return;
    const startedAt = Date.now();
    const handle = setTimeout(onDone, remaining.current);
    return () => {
      clearTimeout(handle);
      remaining.current = Math.max(1_500, remaining.current - (Date.now() - startedAt));
    };
  }, [paused, onDone]);
}

function CloseButton({ onClick, label = "Dismiss" }: { onClick: () => void; label?: string }) {
  return (
    <button type="button" className={styles.close} onClick={onClick} aria-label={label}>
      <svg viewBox="0 0 12 12" aria-hidden="true">
        <path d="M3 3l6 6M9 3l-6 6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
    </button>
  );
}

function ToastCard({ toast }: { toast: Toast }) {
  const dismiss = useStore((s) => s.dismissToast);
  const [paused, setPaused] = useState(false);
  const [leaving, setLeaving] = useState(false);

  // Fade out first, then leave the store.
  const leave = useCallback(() => {
    setLeaving(true);
    setTimeout(() => dismiss(toast.id), prefersReducedMotion() ? 0 : EXIT_MS + 40);
  }, [dismiss, toast.id]);
  useLifetime(LIFETIME_MS[toast.kind], paused || leaving, leave);

  return (
    <div
      className={styles.toast}
      data-kind={toast.kind}
      data-state={leaving ? "leaving" : "shown"}
      role={toast.kind === "error" ? "alert" : "status"}
      onPointerEnter={() => setPaused(true)}
      onPointerLeave={() => setPaused(false)}
      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
    >
      <span className={styles.kicker}>{toast.kind === "error" ? "Not applied" : "Notice"}</span>
      <p className={styles.text}>{toast.text}</p>
      {toast.cta && (
        <button
          type="button"
          className={styles.cta}
          onClick={() => {
            if (toast.cta) getSession().setSpeed(toast.cta.speed);
            leave();
          }}
        >
          {toast.cta.label}
        </button>
      )}
      <CloseButton onClick={leave} />
    </div>
  );
}

/**
 * The single, paced AI callout as a service-change poster: a header bar ("SERVICE CHANGE",
 * "EXTRA SERVICE" in black; "HOLDING", "SKIP STOP" in a lighter grey),
 * the route bullet large, a plain-language instruction and why, and an "AI Dispatcher" byline.
 */
function CalloutCard({ callout, leaving, onDone }: { callout: ActiveCallout; leaving: boolean; onDone: () => void }) {
  const net = useStore((s) => s.network);
  const requestCamera = useStore((s) => s.requestCamera);
  const setHighlight = useStore((s) => s.setHighlight);
  const [paused, setPaused] = useState(false);
  const ev = callout.event;
  const station = ev.station_idx >= 0 ? net?.stations[ev.station_idx] : undefined;
  const copy = posterCopy(ev.action, station?.name, ev.text);
  useLifetime(CALLOUT_LIFETIME_MS, paused || leaving, onDone);

  const hold = (on: boolean) => {
    setPaused(on);
    getSession().holdCallout(on);
  };
  useEffect(() => () => getSession().holdCallout(false), []);

  const locate = () => {
    if (!station) return;
    requestCamera({ kind: "view", view: { cx: station.x, cy: station.y, mPerPx: 3.5 } });
    setHighlight(station.idx);
  };

  return (
    <div
      className={`${styles.toast} ${styles.callout}`}
      data-kind="callout"
      data-state={leaving ? "leaving" : "shown"}
      role="status"
      title={ev.text}
      onPointerEnter={() => hold(true)}
      onPointerLeave={() => hold(false)}
      onFocus={() => hold(true)}
      onBlur={() => hold(false)}
    >
      <header className={styles.posterHead} data-major={copy.major}>
        <span className={styles.posterTitle}>{copy.header}</span>
        <span className={styles.byline} aria-label={`AI dispatcher: ${ACTION_TITLE[ev.action] ?? ev.action}`}>
          <span className={styles.spark} aria-hidden="true" />
          AI Dispatcher
        </span>
      </header>
      <div className={styles.posterBody}>
        {ev.route && <RouteBullet id={ev.route} size={34} />}
        <p className={styles.line}>
          <b>{copy.headline}</b>
          {copy.reason && <span className={styles.reason}>{copy.reason}</span>}
        </p>
        <span className={styles.posterSide}>
          {callout.recent >= BUSY_COUNT && (
            <span className={styles.rate} title={`${callout.recent} AI actions in the last 10 s: every one is in the dispatch log`}>
              +{callout.recent - 1} more
            </span>
          )}
          {station && (
            <button type="button" className={styles.locate} onClick={locate} aria-label={`Show ${station.name} on the map`} title={`Show ${station.name}`}>
              <svg viewBox="0 0 12 12" aria-hidden="true">
                <path d="M6 11s3.6-3.4 3.6-6A3.6 3.6 0 0 0 2.4 5c0 2.6 3.6 6 3.6 6z" fill="none" stroke="currentColor" strokeWidth="1.3" />
                <circle cx="6" cy="5" r="1.2" fill="currentColor" />
              </svg>
            </button>
          )}
        </span>
      </div>
    </div>
  );
}

/**
 * Notices and the AI callout, docked above the control pill. At most one AI
 * callout at a time (paced in the session controller); it leaves as soon as the
 * mode goes back to Normal. Steps aside while the dispatch log or the map key is open.
 */
export default function CalloutLayer() {
  const toasts = useStore((s) => s.toasts);
  const callout = useStore((s) => s.callout);
  const showCallout = useStore((s) => s.showCallout);
  const logOpen = useStore((s) => s.feedOpen);
  const keyOpen = useStore((s) => s.keyOpen);
  // Keep the last callout mounted while it fades out (state derived during render).
  const [last, setLast] = useState<ActiveCallout | null>(null);
  if (callout && callout !== last) setLast(callout);
  const leaving = !callout && last !== null;
  useEffect(() => {
    if (!leaving) return;
    const t = setTimeout(() => setLast(null), prefersReducedMotion() ? 0 : EXIT_MS + 40);
    return () => clearTimeout(t);
  }, [leaving]);
  const expire = useCallback(() => {
    const cur = useStore.getState().callout;
    if (cur && cur.id === last?.id) showCallout(null);
  }, [last, showCallout]);

  return (
    <div className={styles.layer} aria-live="polite" aria-relevant="additions" data-hidden={logOpen || keyOpen}>
      {toasts
        .filter((t) => t.kind !== "callout")
        .map((t) => (
          <ToastCard key={t.id} toast={t} />
        ))}
      {last && <CalloutCard key={last.id} callout={last} leaving={leaving} onDone={expire} />}
    </div>
  );
}
