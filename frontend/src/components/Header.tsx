"use client";

import { PROFILE_INFO } from "@/lib/copy";
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
  const profile = useStore((s) => s.settings.profile);
  const speed = useStore((s) => s.settings.speed);
  const setInfoOpen = useStore((s) => s.setInfoOpen);

  return (
    <header className={styles.header}>
      <div className={styles.brand}>
        <a className={styles.back} href="https://kenmar.dev" aria-label="Back to kenmar.dev">
          <span aria-hidden="true">←</span> kenmar.dev
        </a>
        <div className={styles.titleRow}>
          <Logo />
          <div>
            <h1 className={styles.title}>Subway Domino</h1>
            <p className={styles.tagline}>Human vs. AI · NYC subway control room</p>
          </div>
        </div>
      </div>

      <div className={styles.search}>
        <SearchBox />
      </div>

      <div className={styles.right}>
        <div className={styles.clock} aria-label="Simulation time">
          <span className={styles.clockTime}>{formatClock(clockS)}</span>
          <span className={styles.clockMeta}>
            {PROFILE_INFO[profile].label}
            {speed === "pause" ? " · paused" : ` · ${speed}`}
          </span>
        </div>
        <button
          type="button"
          className={styles.info}
          onClick={() => setInfoOpen(true)}
          aria-label="How it works"
          title="How it works: data, simulation, reward and the AI's actions"
        >
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <circle cx="10" cy="10" r="8" fill="none" stroke="currentColor" strokeWidth="1.7" />
            <path d="M10 9v5" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" />
            <circle cx="10" cy="6.2" r="1.15" fill="currentColor" />
          </svg>
          <span className={styles.infoLabel}>How it works</span>
        </button>
      </div>
    </header>
  );
}
