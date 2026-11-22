"use client";

import { useEffect, useLayoutEffect, useState, useSyncExternalStore } from "react";
import { type OnboardingStep, useStore } from "@/lib/store";
import styles from "./Onboarding.module.css";

const STORAGE_KEY = "subway-domino.onboarding.v1";
const SHOW_DELAY_MS = 900;
const BUBBLE_W = 300;
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

function place(target: DOMRect, step: Exclude<OnboardingStep, "done">, bubbleH: number): Placement {
  const W = window.innerWidth;
  // Presets: point at the first preset button; toggle: at the AI half of the switch.
  const anchorX = step === "presets" ? Math.min(target.left + 110, target.right - 30) : target.left + target.width * 0.75;
  const below = step === "presets";
  const left = Math.min(Math.max(EDGE, anchorX - BUBBLE_W / 2), W - BUBBLE_W - EDGE);
  const top = below ? target.bottom + 12 : target.top - 12 - bubbleH;
  return { left, top, below, arrowLeft: Math.min(Math.max(18, anchorX - left), BUBBLE_W - 18) };
}

type Step = Exclude<OnboardingStep, "done">;

/** Coach-mark copy; touch screens get "tap" wording (and a shorter first step). */
function copyFor(step: Step, touch: boolean): { title: string; body: string } {
  if (step === "presets") {
    return {
      title: "Start here: break the city",
      body: touch
        ? "Tap a meltdown preset — or any station, track or interlocking — to choose what fails."
        : "Pick a meltdown preset — or click any station, track or interlocking on the map to choose what fails.",
    };
  }
  return {
    title: "Now hand it to the AI",
    body: "Flip to AI Dispatcher. A Normal-Schedule twin keeps running, so you can see exactly what the AI changed.",
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

/** A single dismissible coach-mark: first the PresetBar, then the mode toggle (§6 UX flow 8). */
export default function Onboarding() {
  const step = useStore((s) => s.onboarding);
  const setStep = useStore((s) => s.setOnboarding);
  const ready = useStore((s) => s.network !== null);
  const blocked = useStore((s) => s.selection !== null || s.infoOpen);
  const aiUnavailable = useStore((s) => s.session !== null && !s.session.aiAvailable);
  const touch = useCoarsePointer();
  const [visible, setVisible] = useState(false);
  const [placement, setPlacement] = useState<Placement | null>(null);
  const [bubble, setBubble] = useState<HTMLDivElement | null>(null);

  // Returning visitors skip the tour.
  useEffect(() => {
    if (readDone()) setStep("done");
  }, [setStep]);

  useEffect(() => {
    if (step === "done") writeDone();
  }, [step]);

  useEffect(() => {
    if (!ready || step === "done") return;
    const id = setTimeout(() => setVisible(true), SHOW_DELAY_MS);
    return () => clearTimeout(id);
  }, [ready, step]);

  useLayoutEffect(() => {
    if (!visible || step === "done") return;
    const target = document.querySelector<HTMLElement>(`[data-onboarding="${step}"]`);
    if (!target) return;
    const update = () => setPlacement(place(target.getBoundingClientRect(), step, bubble?.offsetHeight ?? 110));
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

  if (step === "done" || !visible || blocked) return null;
  const copy =
    step === "toggle" && aiUnavailable
      ? {
          title: "The AI Dispatcher is offline",
          body: "The trained model isn’t loaded right now, so Normal Schedule keeps running. Try again later!",
        }
      : copyFor(step, touch);

  return (
    <div
      ref={setBubble}
      className={styles.bubble}
      role="dialog"
      aria-live="polite"
      aria-label={copy.title}
      data-below={placement?.below ?? true}
      style={placement ? { left: placement.left, top: placement.top, width: BUBBLE_W } : { visibility: "hidden" }}
    >
      {placement && <span className={styles.arrow} style={{ left: placement.arrowLeft }} aria-hidden="true" />}
      <p className={styles.step}>{step === "presets" ? "1 of 2" : "2 of 2"}</p>
      <p className={styles.title}>{copy.title}</p>
      <p className={styles.body}>{copy.body}</p>
      <div className={styles.actions}>
        <button type="button" className={styles.dismiss} onClick={() => setStep("done")}>
          Got it
        </button>
      </div>
    </div>
  );
}
