"use client";

import { useEffect, useRef } from "react";
import { useDisclosure } from "@/lib/motion";
import { useStore } from "@/lib/store";
import styles from "./MapControls.module.css";

type SwatchKind =
  | "train"
  | "held"
  | "stuck"
  | "aiAction"
  | "shortTurn"
  | "gap"
  | "skip"
  | "switchTrack"
  | "rerouted"
  | "station"
  | "transfer"
  | "busy"
  | "crowded"
  | "closed"
  | "block"
  | "signal"
  | "interlocking"
  | "yard"
  | "river";

/** Small SVG samples drawn to match the canvas renderer (renderDynamic / renderBase), on a land-coloured tile. */
function Swatch({ kind }: { kind: SwatchKind }) {
  const capsule = (extra?: React.ReactNode, fill = "#00933C", under?: React.ReactNode) => (
    <svg viewBox="0 0 36 20" aria-hidden="true">
      {under}
      <rect x="6.5" y="6.6" width="23" height="9" rx="4.5" fill="rgba(40,30,12,0.16)" />
      <rect x="6.2" y="5.2" width="23.6" height="9.6" rx="4.8" fill="#111" />
      <rect x="7" y="6" width="22" height="8" rx="4" fill={fill} />
      <text x="18" y="12.4" textAnchor="middle" fontSize="7" fontWeight="800" fill="#fff">
        6
      </text>
      {extra}
    </svg>
  );
  switch (kind) {
    case "train":
      return capsule();
    case "held":
      return capsule(
        <>
          <rect x="4.2" y="3.2" width="27.6" height="13.6" rx="6.8" fill="none" stroke="#111" strokeWidth="3" />
          <rect x="4.2" y="3.2" width="27.6" height="13.6" rx="6.8" fill="none" stroke="#fff" strokeWidth="1.4" />
        </>,
      );
    case "stuck":
      return capsule(
        <>
          <rect x="4.8" y="3.8" width="26.4" height="12.4" rx="6.2" fill="none" stroke="#fff" strokeWidth="1.8" />
          <rect x="3.2" y="2.2" width="29.6" height="15.6" rx="7.8" fill="none" stroke="#d7261e" strokeWidth="1.6" />
        </>,
        "#00933C",
        <>
          <defs>
            <radialGradient id="mk-stuck">
              <stop offset="0" stopColor="rgba(215,38,30,0.7)" />
              <stop offset="1" stopColor="rgba(215,38,30,0)" />
            </radialGradient>
          </defs>
          <ellipse cx="18" cy="10" rx="17" ry="10" fill="url(#mk-stuck)" />
        </>,
      );
    case "aiAction":
      return capsule(
        <>
          <rect x="2.6" y="1.8" width="30.8" height="16.4" rx="8.2" fill="none" stroke="#fff" strokeWidth="3.4" />
          <rect x="2.6" y="1.8" width="30.8" height="16.4" rx="8.2" fill="none" stroke="#111" strokeWidth="1.6" />
        </>,
      );
    case "shortTurn":
    case "gap":
    case "skip":
    case "switchTrack": {
      const glyph = { shortTurn: "↺", gap: "✚", skip: "»", switchTrack: "⇄" }[kind];
      const color = kind === "gap" ? "#14853b" : kind === "shortTurn" ? "#0b5cc4" : "#111111";
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          <rect x="7" y="10" width="22" height="9" rx="4.5" fill="#00933C" stroke="#111" strokeWidth="1.2" />
          <text
            x="18"
            y="8"
            textAnchor="middle"
            fontSize="10"
            fontWeight="800"
            fill={color}
            stroke="rgba(250,247,240,0.94)"
            strokeWidth="2.5"
            paintOrder="stroke"
          >
            {glyph}
          </text>
        </svg>
      );
    }
    case "rerouted":
      return capsule(<rect x="4.2" y="3.2" width="27.6" height="13.6" rx="6.8" fill="none" stroke="#111" strokeWidth="1.1" strokeDasharray="2.5 2" />, "#FF6319");
    case "station":
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          <path d="M3 10h30" stroke="#EE352E" strokeWidth="4" />
          <circle cx="18" cy="10" r="2.6" fill="#151515" stroke="#fff" strokeWidth="0.8" />
        </svg>
      );
    case "transfer":
      // Two white circles joined by one white connector, one black outline (as drawn on the map).
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          <path d="M3 6h30" stroke="#EE352E" strokeWidth="4" />
          <path d="M3 14h30" stroke="#0039A6" strokeWidth="4" />
          <path d="M18 6v8" stroke="#151515" strokeWidth="9.6" strokeLinecap="round" />
          <path d="M18 6v8" stroke="#fff" strokeWidth="7" strokeLinecap="round" />
        </svg>
      );
    case "busy":
    case "crowded":
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          {kind === "crowded" && <circle cx="18" cy="10" r="8.5" fill="none" stroke="#d7261e" strokeWidth="1.4" opacity="0.7" />}
          <circle cx="18" cy="10" r={4.2} fill={kind === "busy" ? "rgb(255,160,38)" : "#d7261e"} />
        </svg>
      );
    case "closed":
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          <circle cx="18" cy="10" r="6.4" fill="#111" />
          <path d="M15.4 7.4l5.2 5.2M20.6 7.4l-5.2 5.2" stroke="#fff" strokeWidth="1.7" strokeLinecap="round" />
        </svg>
      );
    case "block":
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          <defs>
            <pattern id="mk-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)">
              <rect width="5" height="5" fill="rgba(215,38,30,0.2)" />
              <rect width="2" height="5" fill="#d7261e" />
            </pattern>
          </defs>
          <rect x="2" y="5" width="32" height="10" rx="2" fill="url(#mk-hatch)" />
        </svg>
      );
    case "signal":
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          <path d="M3 10h30" stroke="#e89a00" strokeWidth="5" strokeDasharray="6 4" />
        </svg>
      );
    case "interlocking":
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          <path d="M18 4l6 6-6 6-6-6z" fill="#3e5578" stroke="#fff" strokeWidth="1.4" />
        </svg>
      );
    case "yard":
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          <rect x="13" y="5" width="10" height="10" fill="#857d6e" stroke="#fff" strokeWidth="1.4" />
        </svg>
      );
    case "river":
      return (
        <svg viewBox="0 0 36 20" aria-hidden="true">
          <rect x="0" y="0" width="36" height="20" fill="#bfdaea" />
          <path d="M3 5.5h30M3 14.5h30" stroke="rgba(52,104,140,0.5)" strokeWidth="1" strokeDasharray="3 3" />
          <path d="M3 10h30" stroke="#fcfaf5" strokeWidth="5.4" />
          <path d="M3 10h30" stroke="#0062CF" strokeWidth="3.4" />
        </svg>
      );
  }
}

const KEY: readonly { group: string; items: readonly [SwatchKind, string][] }[] = [
  {
    group: "Trains",
    items: [
      ["train", "Train in its route color (citywide: a white bead)"],
      ["held", "Held at a platform, doors open"],
      ["stuck", "Stuck in a tunnel"],
      ["aiAction", "AI just acted on this train"],
      ["shortTurn", "Short-turning"],
      ["gap", "Gap train from a yard"],
      ["skip", "Skipping its next stop"],
      ["switchTrack", "Switching track (express ⇄ local)"],
      ["rerouted", "Rerouted via another line"],
    ],
  },
  {
    group: "Stations",
    items: [
      ["station", "Station"],
      ["transfer", "Transfer between stations"],
      ["busy", "Crowding up"],
      ["crowded", "Over 80 % of safe capacity"],
      ["closed", "Closed (badge counts down)"],
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

function MapKey({ onClose, closing }: { onClose: () => void; closing: boolean }) {
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
    <div ref={ref} className={styles.key} role="dialog" aria-label="Map key" data-state={closing ? "closing" : "open"}>
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
  const setKeyOpen = useStore((s) => s.setKeyOpen);
  const key = useDisclosure(setKeyOpen);
  const keyOpen = key.mounted && !key.closing;
  if (!ready) return null;
  return (
    <div className={styles.root}>
      {key.mounted && <MapKey onClose={key.close} closing={key.closing} />}
      <div className={`glass ${styles.pill}`} role="group" aria-label="Map view" data-occlude>
        <button
          type="button"
          className={styles.tool}
          aria-label="Map key"
          title="Map key"
          aria-expanded={keyOpen}
          data-map-key-toggle
          onClick={key.toggle}
        >
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <circle cx="4" cy="4.5" r="1.8" fill="currentColor" />
            <path d="M8 4.5h5.5M8 11.5h5.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            <rect x="2.3" y="9.7" width="3.6" height="3.6" rx="1" fill="currentColor" />
          </svg>
        </button>
        <span className={styles.sep} aria-hidden="true" />
        <button type="button" className={`${styles.tool} ${styles.zoom}`} aria-label="Zoom out" title="Zoom out" onClick={() => requestCamera({ kind: "zoom", factor: 0.5 })}>
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <path d="M3.5 8h9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </button>
        <button type="button" className={`${styles.tool} ${styles.zoom}`} aria-label="Zoom in" title="Zoom in" onClick={() => requestCamera({ kind: "zoom", factor: 2 })}>
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <path d="M8 3.5v9M3.5 8h9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </button>
        <button
          type="button"
          className={`${styles.tool} ${styles.zoom}`}
          aria-label="Show the whole city"
          title="Show the whole city"
          onClick={() => requestCamera({ kind: "home" })}
        >
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <path d="M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>
    </div>
  );
}
