"use client";
/**
 * Motion helpers for the HUD: panels that animate out before they unmount,
 * numbers that tween to new values, and the reduced-motion preference.
 */
import { useCallback, useEffect, useRef, useState } from "react";

/** Exit animations last this long (keep in sync with the CSS `*-out` keyframes). */
export const EXIT_MS = 170;

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;
}

export interface Disclosure {
  /** Rendered (true while opening, open, or animating out). */
  mounted: boolean;
  /** Animating out: render with `data-state="closing"`. */
  closing: boolean;
  open(): void;
  close(): void;
  toggle(): void;
}

/**
 * Open/close state for a popover, menu or drawer whose exit is animated: `close()`
 * keeps it mounted with `closing = true` for {@link EXIT_MS}, then unmounts it.
 */
export function useDisclosure(onChange?: (open: boolean) => void): Disclosure {
  const [mounted, setMounted] = useState(false);
  const [closing, setClosing] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const state = useRef({ mounted: false, closing: false });
  const notify = useRef(onChange);
  useEffect(() => {
    notify.current = onChange;
  }, [onChange]);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  const open = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
    state.current = { mounted: true, closing: false };
    setMounted(true);
    setClosing(false);
    notify.current?.(true);
  }, []);

  const close = useCallback(() => {
    if (!state.current.mounted || state.current.closing) return;
    state.current = { mounted: true, closing: true };
    setClosing(true);
    notify.current?.(false);
    const done = () => {
      timer.current = null;
      state.current = { mounted: false, closing: false };
      setMounted(false);
      setClosing(false);
    };
    if (prefersReducedMotion()) done();
    else timer.current = setTimeout(done, EXIT_MS);
  }, []);

  const toggle = useCallback(() => {
    if (state.current.mounted && !state.current.closing) close();
    else open();
  }, [open, close]);

  return { mounted, closing, open, close, toggle };
}

/** Close on Escape and on pointer-down outside `refs` while `active`. */
export function useDismiss(active: boolean, refs: readonly React.RefObject<HTMLElement | null>[], onDismiss: () => void): void {
  const cb = useRef(onDismiss);
  const inside = useRef(refs);
  useEffect(() => {
    cb.current = onDismiss;
    inside.current = refs;
  }, [onDismiss, refs]);
  useEffect(() => {
    if (!active) return;
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node | null;
      if (t && inside.current.some((r) => r.current?.contains(t))) return;
      cb.current();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") cb.current();
    };
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [active]);
}

/**
 * A number that eases (ease-out cubic, `ms`) to each new value. Writes the
 * formatted text straight into the returned element ref, so tweening never
 * re-renders React. `format` must be stable (module-level or memoised).
 */
export function useTweenedText<T extends HTMLElement>(
  value: number | null | undefined,
  format: (v: number) => string,
  ms = 320,
): React.RefObject<T | null> {
  const ref = useRef<T | null>(null);
  const shown = useRef<number | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (value == null || !Number.isFinite(value)) {
      shown.current = null;
      el.textContent = "—";
      return;
    }
    const from = shown.current;
    if (from == null || from === value || prefersReducedMotion()) {
      shown.current = value;
      el.textContent = format(value);
      return;
    }
    let raf = 0;
    const t0 = performance.now();
    const step = (now: number) => {
      const k = Math.min(1, (now - t0) / ms);
      const v = from + (value - from) * (1 - (1 - k) ** 3);
      shown.current = v;
      el.textContent = format(k >= 1 ? value : v);
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [value, format, ms]);
  return ref;
}
