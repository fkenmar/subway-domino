"use client";

import { useEffect, useState } from "react";
import type { Mode } from "@/lib/protocol";
import { useStore } from "@/lib/store";
import styles from "./ModeCue.module.css";

/**
 * A brief map-wide cue when control changes hands: a cyan vignette breathes in
 * from the edges as the AI takes over (a neutral one when it hands back).
 */
export default function ModeCue() {
  const [cue, setCue] = useState<{ n: number; mode: Mode } | null>(null);
  useEffect(
    () =>
      useStore.subscribe((s, prev) => {
        if (s.settings.mode !== prev.settings.mode) setCue((c) => ({ n: (c?.n ?? 0) + 1, mode: s.settings.mode }));
      }),
    [],
  );
  if (!cue) return null;
  return <div key={cue.n} className={styles.cue} data-mode={cue.mode} aria-hidden="true" />;
}
