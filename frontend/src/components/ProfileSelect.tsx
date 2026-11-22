"use client";

import { useEffect, useRef, useState } from "react";
import { PROFILE_INFO } from "@/lib/copy";
import { PROFILES, type Profile } from "@/lib/protocol";
import { getSession } from "@/lib/session";
import { useStore } from "@/lib/store";
import styles from "./Controls.module.css";

/** Time-of-day demand profile. Changing it restarts the simulation (the menu says so). */
export default function ProfileSelect() {
  const profile = useStore((s) => s.settings.profile);
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const choose = (p: Profile) => {
    setOpen(false);
    if (p !== profile) getSession().setProfile(p);
  };

  return (
    <div className={styles.menuWrap} ref={wrapRef}>
      <button
        type="button"
        className={styles.button}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`Time of day: ${PROFILE_INFO[profile].label}`}
        onClick={() => setOpen((o) => !o)}
      >
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" strokeWidth="1.6" />
          <path d="M8 4.6V8l2.4 1.6" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
        <span className={styles.label}>{PROFILE_INFO[profile].label}</span>
        <svg className={styles.caret} viewBox="0 0 10 10" aria-hidden="true">
          <path d="M2 6.5L5 3.5l3 3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      </button>
      {open && (
        <div className={styles.menu} role="menu" aria-label="Time of day">
          <p className={styles.menuNote}>
            <span aria-hidden="true">⚠︎</span>
            <span>Switching the time of day restarts the simulation and clears all disruptions.</span>
          </p>
          {PROFILES.map((p) => (
            <button
              key={p}
              type="button"
              role="menuitemradio"
              aria-checked={p === profile}
              className={styles.menuItem}
              onClick={() => choose(p)}
            >
              <span>{PROFILE_INFO[p].label}</span>
              <small>{PROFILE_INFO[p].hours}</small>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
