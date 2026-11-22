"use client";

import { useRef } from "react";
import { presetSubtitle } from "@/lib/copy";
import { useDisclosure, useDismiss } from "@/lib/motion";
import type { GeoPreset } from "@/lib/network";
import { getSession } from "@/lib/session";
import { useStore } from "@/lib/store";
import DisruptionList from "./DisruptionList";
import styles from "./PresetBar.module.css";
import { SubtitleWithBullets } from "./RouteBullet";

/**
 * "Break the city ▾": every curated meltdown in one menu (route bullets and a
 * one-line description each), a reminder that anything on the map can be
 * broken by clicking it, and the active disruptions with clear / clear all.
 */
export default function PresetBar() {
  const presets = useStore((s) => s.network?.presets ?? null);
  const active = useStore((s) => s.activePreset);
  const count = useStore((s) => s.disruptions.length);
  const touch = typeof window !== "undefined" && window.matchMedia?.("(pointer: coarse)").matches === true;
  const wrapRef = useRef<HTMLDivElement>(null);
  const menu = useDisclosure((open) => {
    if (open && useStore.getState().onboarding === "presets") useStore.getState().setOnboarding("toggle");
  });
  const isOpen = menu.mounted && !menu.closing;
  useDismiss(isOpen, [wrapRef], menu.close);

  const trigger = (p: GeoPreset) => {
    getSession().triggerPreset(p);
    menu.close();
  };

  return (
    <div className={styles.wrap} ref={wrapRef}>
      {menu.mounted && (
        <div className={`glass sign ${styles.menu}`} data-state={menu.closing ? "closing" : "open"} role="dialog" aria-label="Break the city">
          <header className={styles.menuHead}>
            <p className={styles.menuTitle}>Break the city</p>
            <p className={styles.menuNote}>
              Pick a meltdown, or {touch ? "tap" : "click"} any station, track or interlocking on the map to choose what fails.
            </p>
          </header>
          <nav aria-label="Meltdown presets">
            <ul className={styles.list}>
              {presets === null
                ? Array.from({ length: 4 }, (_, i) => <li key={i} className={styles.skeleton} aria-hidden="true" />)
                : presets.map((p) => (
                    <li key={p.id}>
                      <button
                        type="button"
                        className={styles.item}
                        data-active={p.id === active}
                        onClick={() => trigger(p)}
                        title={`${p.title} — ${presetSubtitle(p.subtitle)}`}
                      >
                        <span className={styles.itemTitle}>{p.title}</span>
                        <SubtitleWithBullets text={presetSubtitle(p.subtitle)} size={15} />
                      </button>
                    </li>
                  ))}
            </ul>
          </nav>
          <DisruptionList onLocate={menu.close} />
        </div>
      )}
      <button
        type="button"
        className={`glass ${styles.trigger}`}
        aria-haspopup="dialog"
        aria-expanded={isOpen}
        data-open={isOpen}
        data-onboarding="presets"
        onClick={menu.toggle}
      >
        <span className={styles.alarm} aria-hidden="true" />
        <span className={styles.triggerLabel}>Break the city</span>
        {count > 0 && (
          <span className={styles.badge} aria-label={`${count} active`}>
            {count}
          </span>
        )}
        <svg className={styles.caret} viewBox="0 0 10 10" aria-hidden="true">
          <path d="M2 6.5L5 3.5l3 3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      </button>
    </div>
  );
}
