"use client";

import { useRef } from "react";
import { PROFILE_INFO } from "@/lib/copy";
import { useDisclosure, useDismiss } from "@/lib/motion";
import { PROFILES, type Profile } from "@/lib/protocol";
import { getSession } from "@/lib/session";
import { useStore } from "@/lib/store";
import styles from "./Controls.module.css";

/** Time-of-day demand profile. Changing it restarts the simulation (the menu says so). */
export default function ProfileSelect() {
  const profile = useStore((s) => s.settings.profile);
  const wrapRef = useRef<HTMLDivElement>(null);
  const menu = useDisclosure();
  const open = menu.mounted && !menu.closing;
  useDismiss(open, [wrapRef], menu.close);

  const choose = (p: Profile) => {
    menu.close();
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
        onClick={menu.toggle}
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
      {menu.mounted && (
        <div className={`glass ${styles.menu}`} role="menu" aria-label="Time of day" data-state={menu.closing ? "closing" : "open"}>
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
