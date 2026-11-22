"use client";

import { useEffect, useId, useRef, useState } from "react";
import { ACTION_INFO, DATASETS, datasetUrl, REWARD_TERMS } from "@/lib/copy";
import { formatInt } from "@/lib/format";
import { DISPATCH_ACTION_NAMES, type DispatchActionName } from "@/lib/protocol";
import { useStore } from "@/lib/store";
import styles from "./InfoModal.module.css";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "ai", label: "The AI" },
  { id: "actions", label: "Actions & masks" },
  { id: "data", label: "Data" },
] as const;
type TabId = (typeof TABS)[number]["id"];

function AiStatus() {
  const session = useStore((s) => s.session);
  const status = useStore((s) => s.connection.status);
  if (!session) {
    return (
      <p className={styles.status} data-tone="muted">
        {status === "unconfigured" ? "The live simulation isn’t configured for this build." : "Connecting to the live simulation…"}
      </p>
    );
  }
  if (!session.aiAvailable) {
    return (
      <p className={styles.status} data-tone="warn">
        <strong>AI Dispatcher unavailable.</strong> The trained model isn’t loaded on the server right now, so the toggle
        is disabled. Normal Schedule keeps running — there is deliberately no heuristic stand-in.
      </p>
    );
  }
  return (
    <p className={styles.status} data-tone="good">
      <strong>AI Dispatcher online.</strong> The trained policy is loaded on the server (map data version{" "}
      <code>{session.graphVersion || "unknown"}</code>).
    </p>
  );
}

function Overview() {
  return (
    <>
      <p className={styles.lede}>
        A live simulation of every subway line in Manhattan, Brooklyn, Queens and the Bronx — 28 routes, 475 stations,
        424 station complexes — built from MTA open data. Break a station or a track, watch the delays cascade under
        the normal timetable, then hand control to a reinforcement-learning dispatcher and compare.
      </p>
      <ol className={styles.steps}>
        <li>
          <b>Break the city.</b> Pick a meltdown preset, or click any station, track segment, interlocking (◆) or river
          crossing and choose what fails. It stays closed or blocked until you restore service.
        </li>
        <li>
          <b>Watch Normal Schedule.</b> Trains keep to the timetable and the signals: they queue at red signals in
          tunnels, crowded platforms stretch dwell times, and gaps cascade down the line.
        </li>
        <li>
          <b>Switch on the AI Dispatcher.</b> A twin of this exact city keeps running Normal Schedule from that moment,
          so the panel shows what the AI changed: <i>AI</i> next to <i>Normal would be</i>.
        </li>
      </ol>
      <h3 className={styles.h3}>How the simulation works</h3>
      <ul className={styles.bullets}>
        <li>Every train moves in 10-second ticks along the real track layout, with express and local tracks, crossovers, pocket tracks and yards.</li>
        <li>
          <b>Fixed-block signals</b> (250 m blocks) on most lines; <b>CBTC</b> on the L, the 7, Queens Blvd and 8 Av lets
          trains run closer and recover faster.
        </li>
        <li>A-division (numbered) and B-division (lettered) trains never share track — their cars are different sizes.</li>
        <li>Riders arrive per station by hour from real ridership. Crowded platforms slow boarding, which bunches trains.</li>
        <li>Short-turns take 3–5 minutes (crew walks to the other cab and tests the brakes); yards hold few spares.</li>
      </ul>
      <div className={styles.callout}>
        <h3 className={styles.h3}>No tricks</h3>
        <p>
          <b>Normal Schedule</b> answers every decision with “run as scheduled”. <b>AI Dispatcher</b> is the trained
          neural-network policy, choosing only among the moves the physical rules allow. There is no hand-written
          dispatch logic anywhere — if the model isn’t loaded, AI mode is simply unavailable.
        </p>
      </div>
      <AiStatus />
      <h3 className={styles.h3}>Controls</h3>
      <dl className={styles.keys}>
        <dt>Drag · scroll · pinch</dt>
        <dd>Pan and zoom the map (zoom follows the cursor)</dd>
        <dt>Click</dt>
        <dd>Disrupt a station, track, interlocking or river crossing; click a red chip to clear it</dd>
        <dt>
          <kbd>/</kbd>
        </dt>
        <dd>Search a station or line</dd>
        <dt>
          <kbd>←</kbd> <kbd>→</kbd> <kbd>↑</kbd> <kbd>↓</kbd> · <kbd>+</kbd> <kbd>−</kbd> · <kbd>0</kbd>
        </dt>
        <dd>Pan · zoom · whole city (with the map focused)</dd>
      </dl>
    </>
  );
}

function TheAi() {
  return (
    <>
      <p className={styles.lede}>
        The dispatcher is a <b>MaskablePPO</b> policy (two 256-unit layers) trained in Google Colab on this same
        simulator, exported to ONNX and run on the server’s CPU with ONNX Runtime — the exact physics code is shared by
        training and the live site.
      </p>
      <h3 className={styles.h3}>When does it decide?</h3>
      <p>
        Only when a train is ready to leave a stop <i>and</i> at least one maneuver other than “run as scheduled” is
        physically allowed. The policy then sees 216 numbers about that train: its corridor (3 stations behind, 6 ahead —
        crowds, wait times, blocks, signal problems, crossovers, pocket tracks), its own load, headways and lateness, and
        which actions the rules allow. Calm service needs almost no decisions.
      </p>
      <h3 className={styles.h3}>What it is trained to minimise</h3>
      <ul className={styles.terms}>
        {REWARD_TERMS.map((t) => (
          <li key={t.name}>
            <span className={styles.termName}>{t.name}</span>
            <code className={styles.formula}>{t.formula}</code>
            <span className={styles.termWhy}>{t.why}</span>
          </li>
        ))}
      </ul>
      <h3 className={styles.h3}>Credit where it’s due</h3>
      <p>
        During training a Normal-Schedule twin runs in lockstep, and the reward is the difference in cost between the
        two. Parts of the city the AI didn’t touch contribute exactly zero, so the score literally means “how much better
        than Normal Schedule” — the same comparison you see in the panel.
      </p>
    </>
  );
}

function Actions() {
  const actions = useStore((s) => s.metrics?.actions ?? null);
  const total = actions ? DISPATCH_ACTION_NAMES.reduce((sum, n) => sum + (actions[n] ?? 0), 0) : 0;
  const max = actions ? Math.max(1, ...DISPATCH_ACTION_NAMES.map((n) => actions[n] ?? 0)) : 1;
  return (
    <>
      <p className={styles.lede}>
        Eight real Rail Control Center commands. Each one is masked by physical and operational rules, checked again at
        the moment it is applied — the AI can only choose among moves that are actually possible.
      </p>
      <div className={styles.breakdownHead}>
        <span>This session</span>
        <span className={styles.breakdownTotal}>
          {formatInt(total)} dispatch action{total === 1 ? "" : "s"}
        </span>
      </div>
      <ol className={styles.actions}>
        {ACTION_INFO.map((a) => {
          const count = a.name === "CONTINUE_SCHEDULED" || !actions ? null : (actions[a.name as DispatchActionName] ?? 0);
          return (
            <li key={a.name} className={styles.action}>
              <div className={styles.actionHead}>
                <span className={styles.actionId}>{a.id}</span>
                <span className={styles.actionTitle}>{a.title}</span>
                <code className={styles.actionName}>{a.name}</code>
                {count !== null && (
                  <span className={styles.actionCount} title="Applied this session">
                    <span className={styles.bar} style={{ width: `${Math.round((count / max) * 100)}%` }} aria-hidden="true" />
                    <span className={styles.countValue}>{formatInt(count)}</span>
                  </span>
                )}
              </div>
              <p className={styles.actionEffect}>{a.effect}</p>
              <p className={styles.actionMask}>
                <span className={styles.maskLabel}>Allowed when</span> {a.allowedWhen}
              </p>
            </li>
          );
        })}
      </ol>
    </>
  );
}

function Data() {
  return (
    <>
      <p className={styles.lede}>Everything comes from official MTA datasets on NY Open Data, rebuilt by one script.</p>
      <ul className={styles.datasets}>
        {DATASETS.map((d) => (
          <li key={d.id}>
            <a href={datasetUrl(d.id)} target="_blank" rel="noreferrer" className={styles.dataset}>
              <span className={styles.datasetName}>{d.name}</span>
              <code className={styles.datasetId}>{d.id}</code>
              <span className={styles.datasetUse}>{d.use}</span>
            </a>
          </li>
        ))}
      </ul>
      <h3 className={styles.h3}>From timetable to track layout</h3>
      <ul className={styles.bullets}>
        <li>Weekday GTFS stop patterns give every route’s stations; express tracks are inferred from timetable overtakes.</li>
        <li>Crossovers, pocket tracks, CBTC corridors and the 10 yards come from the timetable plus a small curated overlay.</li>
        <li>
          Ridership is one server-side SoQL aggregation (entries by complex × hour, recent weekdays) — never the 48M raw
          rows — split into AM peak, midday, PM rush and late-night demand.
        </li>
        <li>Staten Island Railway is out of scope: it shares no track with the rest of the system.</li>
      </ul>
      <p className={styles.footnote}>
        A <a href="https://kenmar.dev">kenmar.dev</a> project. Not affiliated with or endorsed by the MTA.
      </p>
    </>
  );
}

/** Methodology / info modal (§6 UX flow 6). */
export default function InfoModal() {
  const open = useStore((s) => s.infoOpen);
  const setOpen = useStore((s) => s.setInfoOpen);
  const ref = useRef<HTMLDialogElement>(null);
  const [tab, setTab] = useState<TabId>("overview");
  const baseId = useId();

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    else if (!open && d.open) d.close();
  }, [open]);

  const onTabKey = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    const i = TABS.findIndex((t) => t.id === tab);
    const next = TABS[(i + (e.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length];
    setTab(next.id);
    document.getElementById(`${baseId}-tab-${next.id}`)?.focus();
  };

  return (
    <dialog
      ref={ref}
      className={styles.dialog}
      aria-labelledby={`${baseId}-title`}
      onClose={() => setOpen(false)}
      onClick={(e) => {
        // A click on the backdrop lands on the <dialog> element itself.
        if (e.target === ref.current) setOpen(false);
      }}
    >
      {open && (
        <div className={styles.frame}>
          <header className={styles.head}>
            <div>
              <p className={styles.kicker}>How it works</p>
              <h2 id={`${baseId}-title`} className={styles.title}>
                Subway Domino
              </h2>
              <p className={styles.unaffiliated}>An independent fan-made simulator. Not affiliated with or endorsed by the MTA.</p>
            </div>
            <button type="button" className={styles.close} onClick={() => setOpen(false)} aria-label="Close">
              <svg viewBox="0 0 12 12" aria-hidden="true">
                <path d="M3 3l6 6M9 3l-6 6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
              </svg>
            </button>
          </header>
          <div className={styles.tabs} role="tablist" aria-label="Sections" onKeyDown={onTabKey}>
            {TABS.map((t) => (
              <button
                key={t.id}
                id={`${baseId}-tab-${t.id}`}
                type="button"
                role="tab"
                aria-selected={tab === t.id}
                aria-controls={`${baseId}-panel`}
                tabIndex={tab === t.id ? 0 : -1}
                className={styles.tab}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>
          <div
            id={`${baseId}-panel`}
            className={styles.panel}
            role="tabpanel"
            aria-labelledby={`${baseId}-tab-${tab}`}
            tabIndex={0}
          >
            {tab === "overview" && <Overview />}
            {tab === "ai" && <TheAi />}
            {tab === "actions" && <Actions />}
            {tab === "data" && <Data />}
          </div>
        </div>
      )}
    </dialog>
  );
}
