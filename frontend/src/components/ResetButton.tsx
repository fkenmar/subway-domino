"use client";

import { getSession } from "@/lib/session";
import styles from "./Controls.module.css";

/** One-click clean reset: same profile, no disruptions, fresh timetable. */
export default function ResetButton() {
  return (
    <button
      type="button"
      className={styles.iconButton}
      onClick={() => getSession().reset()}
      title="Reset the simulation (clears every disruption)"
      aria-label="Reset the simulation"
    >
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <path d="M3.2 8a4.8 4.8 0 1 0 1.5-3.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        <path d="M3 2.2v3h3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </button>
  );
}
