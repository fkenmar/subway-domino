"use client";

import { useEffect, useRef, useState } from "react";
import type { GeoPreset } from "@/lib/network";
import { getSession } from "@/lib/session";
import { useStore } from "@/lib/store";
import styles from "./PresetBar.module.css";
import { SubtitleWithBullets } from "./RouteBullet";

const VISIBLE = 3;

/** Curated meltdowns: 3 prominent presets plus "More…" (§6 UX flow 2). */
export default function PresetBar() {
  const presets = useStore((s) => s.network?.presets ?? null);
  const active = useStore((s) => s.activePreset);
  const [moreOpen, setMoreOpen] = useState(false);
  const moreRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!moreOpen) return;
    const onDown = (e: PointerEvent) => {
      if (moreRef.current && !moreRef.current.contains(e.target as Node)) setMoreOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setMoreOpen(false);
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [moreOpen]);

  const trigger = (p: GeoPreset) => {
    getSession().triggerPreset(p);
    setMoreOpen(false);
  };

  const visible = presets?.slice(0, VISIBLE) ?? [];
  const more = presets?.slice(VISIBLE) ?? [];

  return (
    <nav className={styles.bar} aria-label="Meltdown presets" data-onboarding="presets">
      <span className={styles.label}>
        <span className={styles.labelDot} aria-hidden="true" />
        Break the city
      </span>
      <div className={styles.scroller}>
        {presets === null
          ? Array.from({ length: VISIBLE }, (_, i) => <span key={i} className={styles.skeleton} aria-hidden="true" />)
          : visible.map((p) => (
              <button
                key={p.id}
                type="button"
                className={p.id === active ? styles.presetActive : styles.preset}
                onClick={() => trigger(p)}
                title={`${p.title} — ${p.subtitle}`}
              >
                <span className={styles.title}>{p.title}</span>
                <SubtitleWithBullets text={p.subtitle} size={14} />
              </button>
            ))}
        {more.length > 0 && (
          <div className={styles.moreWrap} ref={moreRef}>
            <button
              type="button"
              className={styles.more}
              aria-haspopup="menu"
              aria-expanded={moreOpen}
              onClick={() => setMoreOpen((o) => !o)}
            >
              More…
            </button>
            {moreOpen && (
              <div className={styles.menu} role="menu">
                {more.map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    role="menuitem"
                    className={p.id === active ? styles.menuItemActive : styles.menuItem}
                    onClick={() => trigger(p)}
                  >
                    <span className={styles.title}>{p.title}</span>
                    <SubtitleWithBullets text={p.subtitle} size={14} />
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </nav>
  );
}
