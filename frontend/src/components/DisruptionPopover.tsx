"use client";

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { DISRUPTION_INFO } from "@/lib/copy";
import {
  DURATION_CHOICES,
  remainingS,
  targetKindLabel,
  targetLabel,
  validKinds,
} from "@/lib/disruptions";
import { formatCountdown } from "@/lib/format";
import { BREAKPOINT_MOBILE, currentInsets } from "@/lib/layout";
import type { Network } from "@/lib/network";
import { placePopover, type PopoverPlacement } from "@/lib/popover";
import type { DisruptionInfo, DisruptionKind, DisruptionTarget } from "@/lib/protocol";
import { getSession } from "@/lib/session";
import { type Selection, useStore } from "@/lib/store";
import { mapView, useMapViewVersion } from "@/map/mapView";
import styles from "./DisruptionPopover.module.css";
import { RouteBullets } from "./RouteBullet";

/** A choice in the popover: a disruption kind on a concrete target. */
interface Choice {
  key: string;
  kind: DisruptionKind;
  target: DisruptionTarget;
  title: string;
  effect: string;
}

const DEFAULT_DURATION_S = 1200;

function KindIcon({ kind }: { kind: DisruptionKind }) {
  switch (kind) {
    case "full_station_closure":
      return (
        <svg viewBox="0 0 20 20" aria-hidden="true">
          <circle cx="10" cy="10" r="7" fill="none" stroke="currentColor" strokeWidth="1.8" />
          <path d="M7 7l6 6M13 7l-6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </svg>
      );
    case "signal_delay":
      return (
        <svg viewBox="0 0 20 20" aria-hidden="true">
          <rect x="6.5" y="2.5" width="7" height="15" rx="3.5" fill="none" stroke="currentColor" strokeWidth="1.7" />
          <circle cx="10" cy="6.4" r="1.6" fill="currentColor" />
          <circle cx="10" cy="10" r="1.6" fill="currentColor" opacity="0.45" />
          <circle cx="10" cy="13.6" r="1.6" fill="currentColor" opacity="0.45" />
        </svg>
      );
    default:
      return (
        <svg viewBox="0 0 20 20" aria-hidden="true">
          <path d="M3 6h14M3 14h14" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
          <path
            d={kind === "express_block" ? "M10 3.5v13" : "M6 3.5v13M14 3.5v13"}
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            opacity="0.55"
          />
          <rect x="7.5" y="7.5" width="5" height="5" rx="1" fill="currentColor" />
        </svg>
      );
  }
}

/** Valid choices for a selected target, most dramatic first (§3.5 targeting rules). */
export function choicesFor(net: Network, target: DisruptionTarget): Choice[] {
  const out: Choice[] = validKinds(net, target).map((kind) => ({
    key: `${kind}:${target.type}:${target.idx}`,
    kind,
    target,
    title: kind === "full_station_closure" && target.type === "station" ? "Close this station" : DISRUPTION_INFO[kind].verb,
    effect: DISRUPTION_INFO[kind].effect,
  }));
  if (target.type === "station") {
    const st = net.stations[target.idx];
    const complex = st ? net.complexes[st.complex] : undefined;
    if (complex && complex.stations.length > 1) {
      out.splice(1, 0, {
        key: `full_station_closure:complex:${complex.idx}`,
        kind: "full_station_closure",
        target: { type: "complex", idx: complex.idx },
        title: `Close the whole complex (${complex.stations.length} stations)`,
        effect: `Every platform of ${complex.name} closes, including the transfers.`,
      });
    }
  }
  return out;
}

/** Disruptions that apply to the target (exact matches, plus a complex closure covering a station). */
function activeOn(net: Network, target: DisruptionTarget, disruptions: readonly DisruptionInfo[]): DisruptionInfo[] {
  const complexIdx = target.type === "station" ? net.stations[target.idx]?.complex : undefined;
  return disruptions.filter(
    (d) =>
      (d.target.type === target.type && d.target.idx === target.idx) ||
      (complexIdx !== undefined && d.target.type === "complex" && d.target.idx === complexIdx),
  );
}

function routesOf(net: Network, target: DisruptionTarget): number[] {
  switch (target.type) {
    case "station":
      return net.stations[target.idx]?.routes ?? [];
    case "complex":
      return [...new Set((net.complexes[target.idx]?.stations ?? []).flatMap((s) => net.stations[s]?.routes ?? []))];
    case "segment":
      return net.segments[target.idx]?.routes ?? [];
    case "interlocking": {
      const il = net.interlockings[target.idx];
      return il ? (net.stations[il.station]?.routes ?? []) : [];
    }
    default:
      return [];
  }
}

function PopoverBody({ net, selection }: { net: Network; selection: Selection }) {
  useMapViewVersion();
  const disruptions = useStore((s) => s.disruptions);
  const clockS = useStore((s) => s.clockS);
  const select = useStore((s) => s.select);
  const onboarding = useStore((s) => s.onboarding);
  const setOnboarding = useStore((s) => s.setOnboarding);
  const rootRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 320, h: 260 });
  const [viewport, setViewport] = useState(() =>
    typeof window === "undefined" ? { w: 1280, h: 800 } : { w: window.innerWidth, h: window.innerHeight },
  );

  const { target } = selection;
  const choices = useMemo(() => choicesFor(net, target), [net, target]);
  const active = activeOn(net, target, disruptions);
  const [choiceKey, setChoiceKey] = useState(() => choices[0]?.key ?? "");
  const [durationS, setDurationS] = useState<number | null>(DEFAULT_DURATION_S);
  const [formOpen, setFormOpen] = useState(() => selection.disruptionId == null);
  const choice = choices.find((c) => c.key === choiceKey) ?? choices[0];
  const sheet = viewport.w < BREAKPOINT_MOBILE;

  useLayoutEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.offsetWidth, h: el.offsetHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const onResize = () => setViewport({ w: window.innerWidth, h: window.innerHeight });
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") select(null);
    };
    window.addEventListener("resize", onResize);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("resize", onResize);
      window.removeEventListener("keydown", onKey);
    };
  }, [select]);

  // Move focus into the dialog so keyboard users land on its first control.
  useEffect(() => {
    rootRef.current?.querySelector<HTMLElement>("input, button:not([data-close])")?.focus({ preventScroll: true });
  }, []);

  // Placement: above the anchor, else below, else beside it (see lib/popover.ts).
  const anchor = mapView.project(selection.wx, selection.wy);
  let placement: PopoverPlacement | null = null;
  let offscreen = false;
  if (!sheet && anchor) {
    offscreen = anchor.x < 0 || anchor.y < 0 || anchor.x > viewport.w || anchor.y > viewport.h;
    placement = placePopover({ anchor, size, viewport, insets: currentInsets(viewport.w, viewport.h) });
  }
  const arrowSide = placement?.arrow.side;
  const arrowStyle: React.CSSProperties | undefined =
    placement && (arrowSide === "top" || arrowSide === "bottom")
      ? { left: placement.arrow.offset }
      : placement
        ? { top: placement.arrow.offset }
        : undefined;

  const start = () => {
    if (!choice) return;
    getSession().addDisruption(choice.kind, choice.target, durationS);
    if (onboarding === "presets") setOnboarding("toggle");
    select(null);
  };

  const kindLabel = targetKindLabel(net, target);
  const title = targetLabel(net, target);
  const routes = routesOf(net, target);

  return (
    <div
      ref={rootRef}
      className={sheet ? styles.sheet : styles.popover}
      style={placement ? { left: placement.left, top: placement.top } : undefined}
      data-offscreen={offscreen ? "true" : "false"}
      role="dialog"
      aria-label={`${kindLabel}: ${title}`}
    >
      {placement && <span className={styles.arrow} data-side={arrowSide} style={arrowStyle} aria-hidden="true" />}
      <header className={styles.head}>
        <div className={styles.headText}>
          <span className={styles.kicker}>{kindLabel}</span>
          <h2 className={styles.title}>{title}</h2>
          {routes.length > 0 && <RouteBullets routes={routes} size={16} max={10} />}
        </div>
        <button type="button" className={styles.close} data-close onClick={() => select(null)} aria-label="Close">
          <svg viewBox="0 0 12 12" aria-hidden="true">
            <path d="M3 3l6 6M9 3l-6 6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </button>
      </header>

      {active.length > 0 && (
        <ul className={styles.active} aria-label="Active here">
          {active.map((d) => {
            const left = remainingS(d, clockS);
            return (
              <li key={d.id} className={styles.activeItem} data-highlight={d.id === selection.disruptionId}>
                <span className={styles.activeIcon} data-kind={d.kind}>
                  <KindIcon kind={d.kind} />
                </span>
                <span className={styles.activeText}>
                  <span className={styles.activeLabel}>{d.label || DISRUPTION_INFO[d.kind].title}</span>
                  <span className={styles.activeTime}>{left == null ? "Until cleared" : `${formatCountdown(left)} left`}</span>
                </span>
                <button type="button" className={styles.clear} onClick={() => getSession().clearDisruption(d.id)}>
                  Clear
                </button>
              </li>
            );
          })}
        </ul>
      )}

      {choices.length === 0 ? (
        <p className={styles.empty}>Nothing here can be disrupted.</p>
      ) : !formOpen ? (
        <button type="button" className={styles.more} onClick={() => setFormOpen(true)}>
          Add another disruption here…
        </button>
      ) : (
        <>
          <fieldset className={styles.kinds}>
            <legend className={styles.legend}>What breaks?</legend>
            {choices.map((c) => (
              <label key={c.key} className={styles.kind} data-kind={c.kind}>
                <input
                  type="radio"
                  name="disruption-kind"
                  value={c.key}
                  checked={c.key === choice?.key}
                  onChange={() => setChoiceKey(c.key)}
                />
                <span className={styles.kindIcon}>
                  <KindIcon kind={c.kind} />
                </span>
                <span className={styles.kindText}>
                  <span className={styles.kindTitle}>{c.title}</span>
                  <span className={styles.kindEffect}>{c.effect}</span>
                </span>
              </label>
            ))}
          </fieldset>

          <fieldset className={styles.durations}>
            <legend className={styles.legend}>For how long?</legend>
            <div className={styles.segmented} role="radiogroup" aria-label="Duration">
              {DURATION_CHOICES.map((d) => (
                <button
                  key={d.label}
                  type="button"
                  role="radio"
                  aria-checked={durationS === d.seconds}
                  className={styles.segment}
                  onClick={() => setDurationS(d.seconds)}
                >
                  {d.label}
                </button>
              ))}
            </div>
          </fieldset>

          <button type="button" className={styles.go} onClick={start} disabled={!choice}>
            <KindIcon kind={choice?.kind ?? "local_block"} />
            {choice ? `${choice.title}${durationS == null ? "" : ` · ${Math.round(durationS / 60)} min`}` : "Choose a disruption"}
          </button>
        </>
      )}
    </div>
  );
}

/** Opens on any map target: lists only the disruption kinds valid there (§6 UX flow 5). */
export default function DisruptionPopover() {
  const selection = useStore((s) => s.selection);
  const net = useStore((s) => s.network);
  if (!selection || !net) return null;
  const key = `${selection.target.type}:${selection.target.idx}:${selection.disruptionId ?? ""}`;
  return <PopoverBody key={key} net={net} selection={selection} />;
}
