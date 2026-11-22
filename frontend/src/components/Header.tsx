"use client";

import { SPEED_INFO } from "@/lib/copy";
import { formatClock } from "@/lib/format";
import { useStore } from "@/lib/store";
import styles from "./Header.module.css";
import SearchBox from "./SearchBox";

function Logo() {
  return (
    <svg className={styles.logo} viewBox="0 0 64 64" aria-hidden="true">
      <rect x="12" y="10" width="18" height="34" rx="4" fill="#f4f6fa" transform="rotate(-14 21 27)" />
      <rect x="30" y="18" width="18" height="34" rx="4" fill="var(--ai)" transform="rotate(10 39 35)" />
      <circle cx="21" cy="20" r="3" fill="#0a0f17" transform="rotate(-14 21 27)" />
      <circle cx="21" cy="33" r="3" fill="#0a0f17" transform="rotate(-14 21 27)" />
      <circle cx="39" cy="28" r="3" fill="#0a0f17" transform="rotate(10 39 35)" />
      <circle cx="39" cy="42" r="3" fill="#0a0f17" transform="rotate(10 39 35)" />
    </svg>
  );
}

export default function Header() {
  const clockS = useStore((s) => s.clockS);
  const speed = useStore((s) => s.settings.speed);
  const setInfoOpen = useStore((s) => s.setInfoOpen);

  return (
    <header className={styles.bar}>
      <div className={styles.brand}>
        <Logo />
        <h1 className={styles.title}>Subway Domino</h1>
        <a className={styles.back} href="https://kenmar.dev" aria-label="Back to kenmar.dev">
          <span aria-hidden="true">←</span>
          <span className={styles.backText}>kenmar.dev</span>
        </a>
      </div>

      <div className={styles.right}>
        <div className={styles.clock} aria-label="Simulation time">
          <span className={styles.clockTime}>{formatClock(clockS, { seconds: true })}</span>
          <span className={styles.clockMeta}>
            {speed === "pause" ? "Paused" : SPEED_INFO[speed].label}
          </span>
        </div>
        <SearchBox />
        <button
          type="button"
          className={styles.iconButton}
          onClick={() => setInfoOpen(true)}
          aria-label="How it works"
          title="How it works: data, simulation, reward and the AI's actions"
        >
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <circle cx="10" cy="10" r="7.6" fill="none" stroke="currentColor" strokeWidth="1.5" />
            <path d="M10 9v4.6" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
            <circle cx="10" cy="6.4" r="1.05" fill="currentColor" />
          </svg>
        </button>
      </div>
    </header>
  );
}
