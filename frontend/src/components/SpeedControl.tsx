"use client";

import { SPEED_INFO } from "@/lib/copy";
import { SPEEDS } from "@/lib/protocol";
import { getSession } from "@/lib/session";
import { useStore } from "@/lib/store";
import styles from "./Controls.module.css";

/** Pause / 1x / 2x / 5x playback. */
export default function SpeedControl() {
  const speed = useStore((s) => s.settings.speed);
  return (
    <div className={styles.segmented} role="radiogroup" aria-label="Simulation speed">
      {SPEEDS.map((sp) => (
        <button
          key={sp}
          type="button"
          role="radio"
          aria-checked={speed === sp}
          aria-label={SPEED_INFO[sp].hint}
          title={SPEED_INFO[sp].hint}
          className={styles.segment}
          onClick={() => getSession().setSpeed(sp)}
        >
          {sp === "pause" ? (
            <svg viewBox="0 0 14 14" aria-hidden="true">
              <rect x="2.5" y="2" width="3.2" height="10" rx="1" fill="currentColor" />
              <rect x="8.3" y="2" width="3.2" height="10" rx="1" fill="currentColor" />
            </svg>
          ) : (
            SPEED_INFO[sp].label
          )}
        </button>
      ))}
    </div>
  );
}
