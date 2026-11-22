"use client";

import { useEffect, useLayoutEffect, useState, useSyncExternalStore } from "react";
import { useStore } from "@/lib/store";
import styles from "./Onboarding.module.css";

const STORAGE_KEY = "subway-domino.onboarding.v1";
const SHOW_DELAY_MS = 1200;
const BUBBLE_W = 264;
const EDGE = 10;

interface Placement {
  left: number;
  top: number;
  arrowLeft: number;
  /** The bubble sits below its target (arrow on top) or above it (arrow below). */
  below: boolean;
}

function readDone(): boolean {
  try {
    return globalThis.localStorage?.getItem(STORAGE_KEY) === "done";
  } catch {
    return false;
  }
}

function writeDone(): void {
  try {
    globalThis.localStorage?.setItem(STORAGE_KEY, "done");
  } catch {
    /* storage unavailable: the coach-mark just shows again next visit */
  }
}

/** Above the "Break the city" button, left-aligned with it: never over the map centre. */
function place(target: DOMRect, bubbleH: number): Placement {
  const W = window.innerWidth;
  const left = Math.min(Math.max(EDGE, target.left), W - BUBBLE_W - EDGE);
  const top = target.top - 12 - bubbleH;
  const anchorX = target.left + Math.min(28, target.width / 2);
  return { left, top, below: false, arrowLeft: Math.min(Math.max(16, anchorX - left), BUBBLE_W - 16) };
}

/** Hint copy; touch screens get "tap" wording. */
function copyFor(touch: boolean): { title: string; body: string } {
  return {
    title: "Start here: break the city",
    body: touch
      ? "Pick a meltdown, or tap any station, track or interlocking."
      : "Pick a meltdown, or click any station, track or interlocking on the map.",
  };
}

const COARSE_POINTER = "(pointer: coarse)";

function subscribeCoarse(onChange: () => void): () => void {
  const mq = window.matchMedia(COARSE_POINTER);
  mq.addEventListener("change", onChange);
  return () => mq.removeEventListener("change", onChange);
}

/** True on touch-first devices (no hover, finger-sized targets). */
function useCoarsePointer(): boolean {
  return useSyncExternalStore(
    subscribeCoarse,
    () => window.matchMedia(COARSE_POINTER).matches,
    () => false,
  );
}

/**
 * One small, dismissible hint anchored to "Break the city ▾" (§6 UX flow 8). It
 * goes away for good on "Got it", when the menu opens or once a preset runs.
 */
export default function Onboarding() {
  const step = useStore((s) => s.onboarding);
  const setStep = useStore((s) => s.setOnboarding);
  const ready = useStore((s) => s.network !== null);
  // Also stand aside while a full-screen connection card (at capacity / idle pause) is up.
  const blocked = useStore(
    (s) =>
      s.selection !== null ||
      s.infoOpen ||
      s.connection.status === "at_capacity" ||
      s.connection.status === "idle_timeout",
  );
  const touch = useCoarsePointer();
  const [visible, setVisible] = useState(false);
  const [placement, setPlacement] = useState<Placement | null>(null);
  const [bubble, setBubble] = useState<HTMLDivElement | null>(null);

  // Returning visitors skip the hint.
  useEffect(() => {
    if (readDone()) setStep("done");
  }, [setStep]);

  useEffect(() => {
    if (step === "done") writeDone();
  }, [step]);

  useEffect(() => {
    if (!ready || step !== "presets") return;
    const id = setTimeout(() => setVisible(true), SHOW_DELAY_MS);
    return () => clearTimeout(id);
  }, [ready, step]);

  useLayoutEffect(() => {
    if (!visible || step !== "presets") return;
    const target = document.querySelector<HTMLElement>('[data-onboarding="presets"]');
    if (!target) return;
    const update = () => setPlacement(place(target.getBoundingClientRect(), bubble?.offsetHeight ?? 96));
    update();
    const ro = new ResizeObserver(update);
    ro.observe(target);
    if (bubble) ro.observe(bubble);
    window.addEventListener("resize", update);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", update);
    };
  }, [visible, step, bubble]);

  if (step !== "presets" || !visible || blocked) return null;
  const copy = copyFor(touch);

  return (
    <div
      ref={setBubble}
      className={styles.bubble}
      role="dialog"
      aria-live="polite"
      aria-label={copy.title}
      data-below={placement?.below ?? false}
      style={placement ? { left: placement.left, top: placement.top, width: BUBBLE_W } : { visibility: "hidden" }}
    >
      {placement && <span className={styles.arrow} style={{ left: placement.arrowLeft }} aria-hidden="true" />}
      <p className={styles.title}>{copy.title}</p>
      <p className={styles.body}>{copy.body}</p>
      <button type="button" className={styles.dismiss} onClick={() => setStep("done")}>
        Got it
      </button>
    </div>
  );
}
