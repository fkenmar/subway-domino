"use client";

import { getSession } from "@/lib/session";
import { useStore } from "@/lib/store";
import styles from "./ModeToggle.module.css";

const UNAVAILABLE =
  "AI Dispatcher unavailable — the trained model isn’t loaded on the server right now. Normal Schedule keeps running; there is no heuristic stand-in.";

/** The big "Normal Schedule | AI Dispatcher" switch (§6 UX flow 3). */
export default function ModeToggle() {
  const mode = useStore((s) => s.settings.mode);
  const aiUnavailable = useStore((s) => s.session !== null && !s.session.aiAvailable);
  const pulse = useStore((s) => s.modePulse && s.settings.mode === "normal" && s.disruptions.length > 0);
  const pushToast = useStore((s) => s.pushToast);

  const chooseAi = () => {
    if (aiUnavailable) {
      pushToast({ id: "ai-unavailable", kind: "notice", text: UNAVAILABLE });
      return;
    }
    getSession().setMode("ai");
  };

  return (
    <div
      className={styles.toggle}
      role="radiogroup"
      aria-label="Who dispatches the trains"
      data-onboarding="toggle"
      data-mode={mode}
      data-pulse={pulse ? "true" : "false"}
    >
      <span className={styles.thumb} aria-hidden="true" />
      <button
        type="button"
        role="radio"
        aria-checked={mode === "normal"}
        className={styles.option}
        onClick={() => getSession().setMode("normal")}
      >
        <span className={styles.name}>Normal Schedule</span>
        <span className={styles.sub}>Timetable &amp; signals only</span>
      </button>
      <span className={styles.aiWrap}>
        <button
          type="button"
          role="radio"
          aria-checked={mode === "ai"}
          aria-disabled={aiUnavailable}
          aria-describedby={aiUnavailable ? "ai-unavailable-tip" : undefined}
          className={styles.option}
          data-ai="true"
          onClick={chooseAi}
        >
          <span className={styles.name}>
            <span className={styles.spark} aria-hidden="true" />
            AI Dispatcher
          </span>
          <span className={styles.sub}>{aiUnavailable ? "Unavailable" : "Trained RL policy"}</span>
        </button>
        {aiUnavailable && (
          <span id="ai-unavailable-tip" role="tooltip" className={styles.tip}>
            {UNAVAILABLE}
          </span>
        )}
      </span>
    </div>
  );
}
