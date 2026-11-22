"use client";

import { useEffect, useRef, useState } from "react";
import { useStore } from "@/lib/store";
import styles from "./MapControls.module.css";

type SwatchKind =
  | "train"
  | "held"
  | "stuck"
  | "shortTurn"
  | "gap"
  | "rerouted"
  | "calm"
  | "busy"
  | "crowded"
  | "closed"
  | "block"
  | "signal"
  | "interlocking"
  | "yard"
  | "river";

/** Small SVG samples drawn to match the canvas renderer (renderDynamic / renderBase). */
function Swatch({ kind }: { kind: SwatchKind }) {
  const capsule = (extra?: React.ReactNode, fill = "#00933C") => (
    <svg viewBox="0 0 36 20" aria-hidden="true">
      {extra}
      <rect x="7" y="5.5" width="22" height="9" rx="4.5" fill={fill} stroke="#05080c" strokeWidth="1.6" />
      <text x="18" y="12.9" textAnchor="middle" fontSize="7.5" fontWeight="800" fill="#fff">
        6
      </text>
    </svg>
  );
  switch (kind) {
    case "train":
      return capsule();
    case "held":
      return capsule(<ellipse cx="18" cy="10" rx="14" ry="8" fill="none" stroke="#ffb020" strokeWidth="1.8" />);
    case "stuck":
      return capsule(
        <>
          <defs>
            <radialGradient id="mk-stuck">
              <stop offset="0" stopColor="rgba(255,69,58,0.8)" />
              <stop offset="1" stopColor="rgba(255,69,58,0)" />
            </radialGradient>
          </defs>
          <ellipse cx="18" cy="10" rx="17" ry="10" fill="url(#mk-stuck)" />
        </>,
      );
    case "shortTurn":
    case "gap":
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          <rect x="7" y="9" width="22" height="9" rx="4.5" fill="#0062CF" stroke="#05080c" strokeWidth="1.6" />
          <text x="18" y="8" textAnchor="middle" fontSize="9" fontWeight="800" fill={kind === "gap" ? "#34c759" : "#5ac8fa"}>
            {kind === "gap" ? "✚" : "↺"}
          </text>
        </svg>
      );
    case "rerouted":
      return capsule(<ellipse cx="18" cy="10" rx="15" ry="8.5" fill="none" stroke="#fff" strokeWidth="1.2" strokeDasharray="2.5 2" />, "#EB6800");
    case "calm":
    case "busy":
    case "crowded":
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          {kind === "crowded" && <circle cx="18" cy="10" r="8.5" fill="none" stroke="#ff453a" strokeWidth="1.4" opacity="0.6" />}
          <circle
            cx="18"
            cy="10"
            r="5"
            fill={kind === "calm" ? "#f4f6fa" : kind === "busy" ? "rgb(255,160,38)" : "#ff453a"}
            stroke="#0a0f17"
            strokeWidth="1.6"
          />
        </svg>
      );
    case "closed":
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          <circle cx="18" cy="10" r="7" fill="#05070b" stroke="#ff453a" strokeWidth="1.2" />
          <path d="M14.5 6.5l7 7M21.5 6.5l-7 7" stroke="#ff453a" strokeWidth="1.8" strokeLinecap="round" />
        </svg>
      );
    case "block":
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          <defs>
            <pattern id="mk-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)">
              <rect width="5" height="5" fill="rgba(255,69,58,0.25)" />
              <rect width="2" height="5" fill="#ff453a" />
            </pattern>
          </defs>
          <rect x="2" y="5" width="32" height="10" rx="2" fill="url(#mk-hatch)" />
        </svg>
      );
    case "signal":
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          <path d="M3 10h30" stroke="#ffb020" strokeWidth="5" strokeDasharray="6 4" />
        </svg>
      );
    case "interlocking":
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          <path d="M18 4l6 6-6 6-6-6z" fill="#8fb3ff" stroke="#0a0f17" strokeWidth="1.4" />
        </svg>
      );
    case "yard":
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          <rect x="13" y="5" width="10" height="10" fill="#9aa8bc" stroke="#0a0f17" strokeWidth="1.4" />
        </svg>
      );
    case "river":
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          <path d="M3 6h30M3 14h30" stroke="rgba(120,180,230,0.7)" strokeWidth="1" strokeDasharray="3 3" />
          <path d="M3 10h30" stroke="#0062CF" strokeWidth="3" />
        </svg>
      );
  }
}

const KEY: readonly { group: string; items: readonly [SwatchKind, string][] }[] = [
  {
    group: "Trains",
    items: [
      ["train", "Train, in its route colour"],
      ["held", "Held at a platform, doors open"],
      ["stuck", "Stuck in a tunnel"],
      ["shortTurn", "Short-turning"],
      ["gap", "Gap train from a yard"],
      ["rerouted", "Rerouted via another line"],
    ],
  },
  {
    group: "Stations",
    items: [
      ["calm", "Normal platform"],
      ["busy", "Crowding up"],
      ["crowded", "Over 80 % of safe capacity"],
      ["closed", "Closed"],
    ],
  },
  {
    group: "Track",
    items: [
      ["block", "Blocked track"],
      ["signal", "Signal problems"],
      ["interlocking", "Interlocking (switches)"],
      ["yard", "Rail yard"],
      ["river", "River crossing"],
    ],
  },
];

function MapKey({ onClose }: { onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    const onDown = (e: PointerEvent) => {
      const target = e.target as HTMLElement | null;
      if (ref.current && target && !ref.current.contains(target) && !target.closest("[data-map-key-toggle]")) onClose();
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onDown);
    };
  }, [onClose]);
  return (
    <div ref={ref} className={styles.key} role="dialog" aria-label="Map key">
      {KEY.map((g) => (
        <section key={g.group} className={styles.keyGroup}>
          <h3 className={styles.keyTitle}>{g.group}</h3>
          <ul className={styles.keyList}>
            {g.items.map(([kind, label]) => (
              <li key={kind}>
                <span className={styles.swatch}>
                  <Swatch kind={kind} />
                </span>
                {label}
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}

/** Zoom / whole-city buttons and the map key, floating at the edge of the map area. */
export default function MapControls() {
  const requestCamera = useStore((s) => s.requestCamera);
  const ready = useStore((s) => s.network !== null);
  const [keyOpen, setKeyOpen] = useState(false);
  if (!ready) return null;
  return (
    <div className={styles.root}>
      {keyOpen && <MapKey onClose={() => setKeyOpen(false)} />}
      <button
        type="button"
        className={styles.keyButton}
        aria-expanded={keyOpen}
        data-map-key-toggle
        data-occlude
        onClick={() => setKeyOpen((o) => !o)}
      >
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <circle cx="4" cy="4.5" r="2" fill="currentColor" />
          <path d="M8 4.5h6M8 11.5h6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          <rect x="2" y="9.5" width="4" height="4" rx="1" fill="currentColor" />
        </svg>
        Map key
      </button>
      <div className={styles.stack} role="group" aria-label="Map view" data-occlude>
        <button type="button" className={styles.zoom} aria-label="Zoom in" title="Zoom in" onClick={() => requestCamera({ kind: "zoom", factor: 2 })}>
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </button>
        <button type="button" className={styles.zoom} aria-label="Zoom out" title="Zoom out" onClick={() => requestCamera({ kind: "zoom", factor: 0.5 })}>
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <path d="M3 8h10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </button>
        <button
          type="button"
          className={styles.zoom}
          aria-label="Show the whole city"
          title="Show the whole city"
          onClick={() => requestCamera({ kind: "home" })}
        >
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <path d="M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>
    </div>
  );
}
