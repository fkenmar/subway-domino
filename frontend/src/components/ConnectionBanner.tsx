"use client";

import { useEffect, useState } from "react";
import { formatSpan } from "@/lib/format";
import { getSession } from "@/lib/session";
import { type ConnectionState, useStore } from "@/lib/store";
import styles from "./ConnectionBanner.module.css";

/** Pills appear only after this delay, so fast connects never flash a banner. */
const SHOW_DELAY_MS = 450;

/** Seconds until `retryAt` (a performance.now() timestamp), re-rendered every second. */
function useSecondsUntil(retryAt: number | null): number | null {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (retryAt == null) return;
    const id = setInterval(() => setNow(performance.now()), 1_000);
    return () => clearInterval(id);
  }, [retryAt]);
  return retryAt == null ? null : Math.max(0, Math.ceil((retryAt - now) / 1000));
}

function useDelayedFlag(on: boolean, delayMs: number): boolean {
  const [shown, setShown] = useState(false);
  useEffect(() => {
    if (!on) return;
    const id = setTimeout(() => setShown(true), delayMs);
    return () => {
      clearTimeout(id);
      setShown(false);
    };
  }, [on, delayMs]);
  return on && shown;
}

function Pill({ tone, children }: { tone: "info" | "warn" | "good"; children: React.ReactNode }) {
  return (
    <div className={styles.pill} data-tone={tone} role="status">
      {children}
    </div>
  );
}

function Spinner() {
  return <span className={styles.spinner} aria-hidden="true" />;
}

function Transient({ connection }: { connection: ConnectionState }) {
  const { status, attempt, retryAt, justReconnected } = connection;
  const waiting = status === "connecting" || status === "reconnecting";
  const show = useDelayedFlag(waiting, SHOW_DELAY_MS);
  const seconds = useSecondsUntil(waiting ? retryAt : null);

  if (status === "open" && justReconnected) {
    return (
      <Pill tone="good">
        <svg className={styles.check} viewBox="0 0 14 14" aria-hidden="true">
          <path d="M3 7.4l2.6 2.6L11 4.6" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" />
        </svg>
        Reconnected — picking up where you left off
      </Pill>
    );
  }
  if (!show) return null;
  if (status === "connecting" && attempt === 0) {
    return (
      <Pill tone="info">
        <Spinner />
        Connecting to the live simulation…
      </Pill>
    );
  }
  const label = status === "connecting" ? "Can’t reach the live simulation" : "Connection lost";
  return (
    <Pill tone="warn">
      <Spinner />
      <span>
        {label}
        {seconds != null && seconds > 0 ? ` — retrying in ${seconds} s` : " — reconnecting…"}
      </span>
      {seconds != null && seconds > 1 && (
        <button type="button" className={styles.pillButton} onClick={() => getSession().resume()}>
          Retry now
        </button>
      )}
    </Pill>
  );
}

function AtCapacity({ connection }: { connection: ConnectionState }) {
  const seconds = useSecondsUntil(connection.retryAt);
  return (
    <div className={styles.cardWrap}>
      <div className={styles.card} role="alert">
        <span className={styles.cardIcon} data-tone="warn" aria-hidden="true">
          <svg viewBox="0 0 24 24">
            <circle cx="7" cy="9" r="2.6" fill="currentColor" />
            <circle cx="12" cy="7.5" r="2.6" fill="currentColor" />
            <circle cx="17" cy="9" r="2.6" fill="currentColor" />
            <path d="M3 18c.6-3 2.2-4.6 4-4.6s3 1 3.6 2M21 18c-.6-3-2.2-4.6-4-4.6s-3 1-3.6 2M8.5 19c.5-3.2 1.8-4.8 3.5-4.8s3 1.6 3.5 4.8" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </span>
        <h2 className={styles.cardTitle}>The control room is full</h2>
        <p className={styles.cardText}>
          {connection.message ?? "Subway Domino is at capacity right now — please try again in a minute."}
        </p>
        <p className={styles.cardMeta}>
          {seconds != null && seconds > 0 ? `Trying again automatically in ${seconds} s.` : "Trying again…"} You can
          explore the static map meanwhile.
        </p>
        <button type="button" className={styles.cardButton} onClick={() => getSession().resume()}>
          Try again now
        </button>
      </div>
    </div>
  );
}

function IdlePaused() {
  // The server announces its idle timeout in `welcome.constants` (300 s in production).
  const idleS = useStore((s) => s.session?.constants.idle_timeout_s ?? 300);
  // The inner alert announces the pause; the button gets an explicit name because an
  // alert-role descendant does not contribute to a name computed from content.
  return (
    <button
      type="button"
      className={styles.idle}
      onClick={() => getSession().resume()}
      aria-label="Paused — click to resume"
      aria-describedby="idle-paused-text"
    >
      <span className={styles.card} role="alert">
        <span className={styles.cardIcon} data-tone="info" aria-hidden="true">
          <svg viewBox="0 0 24 24">
            <rect x="7" y="5" width="3.6" height="14" rx="1.2" fill="currentColor" />
            <rect x="13.4" y="5" width="3.6" height="14" rx="1.2" fill="currentColor" />
          </svg>
        </span>
        <span className={styles.cardTitle}>Paused — click to resume</span>
        <span className={styles.cardText} id="idle-paused-text">
          The simulation paused after {formatSpan(idleS)} without activity, to keep the server free for other visitors.
        </span>
        <span className={styles.cardButton}>Resume the simulation</span>
      </span>
    </button>
  );
}

function Unconfigured({ message }: { message: string | null }) {
  return (
    <Pill tone="warn">
      <span>{message ?? "The live simulation isn’t configured for this build."} The static map still works.</span>
    </Pill>
  );
}

/** Connection lifecycle (§6 UX flow 7): connecting / reconnecting / at capacity / idle / unconfigured. */
export default function ConnectionBanner() {
  const connection = useStore((s) => s.connection);
  switch (connection.status) {
    case "at_capacity":
      return <AtCapacity connection={connection} />;
    case "idle_timeout":
      return <IdlePaused />;
    case "unconfigured":
      return <Unconfigured message={connection.message} />;
    case "hidden":
      return (
        <Pill tone="info">
          <span>Paused while this tab is in the background.</span>
        </Pill>
      );
    default:
      return <Transient connection={connection} />;
  }
}
