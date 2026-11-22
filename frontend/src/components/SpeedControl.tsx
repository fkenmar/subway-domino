"use client";

import { useEffect, useRef } from "react";
import { SPEED_INFO } from "@/lib/copy";
import type { Speed } from "@/lib/protocol";
import { getSession } from "@/lib/session";
import { useStore } from "@/lib/store";
import styles from "./Controls.module.css";

const RATES = ["1x", "2x", "5x"] as const;

/** Play / pause, then 1 · 2 · 5 sim-minutes per second (a segmented control with a sliding indicator). */
export default function SpeedControl() {
  const speed = useStore((s) => s.settings.speed);
  const resumeAt = useRef<Speed>("1x");
  useEffect(() => {
    if (speed !== "pause") resumeAt.current = speed;
  }, [speed]);
  const paused = speed === "pause";
  const index = RATES.indexOf(speed as (typeof RATES)[number]);

  return (
    <div className={styles.speed}>
      <button
        type="button"
        className={styles.play}
        aria-label={paused ? "Play" : "Pause"}
        title={paused ? "Resume the simulation" : SPEED_INFO.pause.hint}
        aria-pressed={!paused}
        onClick={() => getSession().setSpeed(paused ? resumeAt.current : "pause")}
      >
        {paused ? (
          <svg viewBox="0 0 14 14" aria-hidden="true">
            <path d="M4 2.6v8.8a.6.6 0 0 0 .9.5l7-4.4a.6.6 0 0 0 0-1L4.9 2.1a.6.6 0 0 0-.9.5z" fill="currentColor" />
          </svg>
        ) : (
          <svg viewBox="0 0 14 14" aria-hidden="true">
            <rect x="3" y="2.4" width="2.8" height="9.2" rx="0.9" fill="currentColor" />
            <rect x="8.2" y="2.4" width="2.8" height="9.2" rx="0.9" fill="currentColor" />
          </svg>
        )}
      </button>
      <div className={styles.rates} role="radiogroup" aria-label="Simulation speed" data-paused={paused}>
        <span
          className={styles.rateThumb}
          style={{ transform: `translateX(${Math.max(0, index) * 100}%)` }}
          data-hidden={index < 0}
          aria-hidden="true"
        />
        {RATES.map((sp) => (
          <button
            key={sp}
            type="button"
            role="radio"
            aria-checked={speed === sp}
            aria-label={SPEED_INFO[sp].hint}
            title={SPEED_INFO[sp].hint}
            className={styles.rate}
            onClick={() => getSession().setSpeed(sp)}
          >
            {sp.slice(0, 1)}
          </button>
        ))}
      </div>
      <span className={styles.unit}>min/s</span>
    </div>
  );
}
