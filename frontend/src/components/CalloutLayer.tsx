"use client";

import { useEffect, useRef, useState } from "react";
import { ACTION_TITLE } from "@/lib/copy";
import { type Toast, useStore } from "@/lib/store";
import styles from "./CalloutLayer.module.css";
import RouteBullet from "./RouteBullet";

/** How long each toast stays up (ms of unpaused visibility). */
const LIFETIME_MS: Record<Toast["kind"], number> = {
  callout: 9_000,
  notice: 7_000,
  error: 7_000,
};

function ToastCard({ toast }: { toast: Toast }) {
  const dismiss = useStore((s) => s.dismissToast);
  const net = useStore((s) => s.network);
  const requestCamera = useStore((s) => s.requestCamera);
  const setHighlight = useStore((s) => s.setHighlight);
  const [paused, setPaused] = useState(false);
  // Remaining lifetime survives pauses (hover / focus) without restarting from scratch.
  const remainingMs = useRef(LIFETIME_MS[toast.kind]);

  useEffect(() => {
    if (paused) return;
    const startedAt = Date.now();
    const handle = setTimeout(() => dismiss(toast.id), remainingMs.current);
    return () => {
      clearTimeout(handle);
      remainingMs.current = Math.max(1_500, remainingMs.current - (Date.now() - startedAt));
    };
  }, [paused, dismiss, toast.id]);

  const station = toast.stationIdx != null && toast.stationIdx >= 0 ? net?.stations[toast.stationIdx] : undefined;
  const locate = () => {
    if (!station) return;
    requestCamera({ kind: "view", view: { cx: station.x, cy: station.y, mPerPx: 3.5 } });
    setHighlight(station.idx);
  };

  return (
    <div
      className={styles.toast}
      data-kind={toast.kind}
      role={toast.kind === "error" ? "alert" : "status"}
      onPointerEnter={() => setPaused(true)}
      onPointerLeave={() => setPaused(false)}
      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
    >
      <div className={styles.body}>
        {toast.kind === "callout" ? (
          <div className={styles.head}>
            <span className={styles.kicker}>
              <span className={styles.spark} aria-hidden="true" />
              AI Dispatcher
            </span>
            {toast.route && <RouteBullet id={toast.route} size={16} />}
            {toast.action && <span className={styles.action}>{ACTION_TITLE[toast.action]}</span>}
          </div>
        ) : (
          <div className={styles.head}>
            <span className={styles.kicker}>{toast.kind === "error" ? "Not applied" : "Note"}</span>
          </div>
        )}
        <p className={styles.text}>{toast.text}</p>
        {station && (
          <button type="button" className={styles.locate} onClick={locate}>
            <svg viewBox="0 0 12 12" aria-hidden="true">
              <circle cx="6" cy="6" r="3.6" fill="none" stroke="currentColor" strokeWidth="1.5" />
              <circle cx="6" cy="6" r="1.2" fill="currentColor" />
            </svg>
            <span className={styles.locateFull}>Show {station.name}</span>
            <span className={styles.locateShort}>Show on map</span>
          </button>
        )}
      </div>
      <button type="button" className={styles.close} onClick={() => dismiss(toast.id)} aria-label="Dismiss">
        <svg viewBox="0 0 12 12" aria-hidden="true">
          <path d="M3 3l6 6M9 3l-6 6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      </button>
    </div>
  );
}

/** Plain-English AI dispatch callouts and notices, stacked over the map (newest last). */
export default function CalloutLayer() {
  const toasts = useStore((s) => s.toasts);
  return (
    <div className={styles.layer} aria-live="polite" aria-relevant="additions">
      {toasts.map((t) => (
        <ToastCard key={t.id} toast={t} />
      ))}
    </div>
  );
}
