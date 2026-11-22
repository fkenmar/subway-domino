/**
 * Application state (zustand). High-frequency train frames do NOT live here —
 * they go to the non-reactive FrameBuffer (see session.ts) and are read by the
 * canvas render loop directly. The store holds what React renders.
 */
import { create } from "zustand";
import { type AiScoreState, accumulateScore } from "./aiScore";
import type { CameraState } from "./camera";
import type { Bbox } from "./geometry";
import type { Network } from "./network";
import type {
  ActionName,
  CalloutEvent,
  DisruptionEvent,
  DisruptionInfo,
  DisruptionTarget,
  MetricsMessage,
  MetricValues,
  Settings,
  Speed,
  WelcomeConstants,
} from "./protocol";

export type ConnectionStatus =
  | "idle"
  | "connecting"
  | "open"
  | "reconnecting"
  | "at_capacity"
  | "idle_timeout"
  | "hidden"
  | "unconfigured";

export interface ConnectionState {
  status: ConnectionStatus;
  /** Consecutive failed attempts (0 when open). */
  attempt: number;
  /** performance.now() timestamp of the next scheduled retry, if any. */
  retryAt: number | null;
  message: string | null;
  /** Briefly true after a seamless reconnect (drives the "Reconnected" banner). */
  justReconnected: boolean;
  rttMs: number | null;
}

export interface SessionInfo {
  sessionId: string;
  graphVersion: string;
  modelLoaded: boolean;
  aiAvailable: boolean;
  resumed: boolean;
  constants: WelcomeConstants;
}

export interface MetricsPoint {
  t: number;
  live: MetricValues;
  twin: MetricValues | null;
}

export type FeedItem =
  | { id: string; kind: "callout"; t: number; event: CalloutEvent }
  | { id: string; kind: "disruption"; t: number; event: DisruptionEvent };

export interface Toast {
  id: string;
  /** Toasts with the same key replace each other (e.g. repeated holds of one train). */
  dedupeKey?: string;
  kind: "callout" | "notice" | "error";
  text: string;
  route?: string;
  stationIdx?: number;
  /** Dispatch action of a callout toast. */
  action?: ActionName;
  /** Wall-clock ms (Date.now()) when the toast was created. */
  createdAt: number;
  /** Optional one-click follow-up (e.g. "Slow to 1 min/s"). */
  cta?: { label: string; speed: Speed };
}

/** The single AI callout on screen (paced: see calloutPacer.ts). */
export interface ActiveCallout {
  id: string;
  event: CalloutEvent;
  /** AI actions received in the last 10 s of wall time. */
  recent: number;
}

/** A disruption's sim-time window, kept after it ends so sparklines can shade it. */
export interface DisruptionWindow {
  id: number;
  start_t: number;
  /** Planned end while active (null = until cleared); the actual end once it has ended. */
  end_t: number | null;
  ended: boolean;
}

export interface Selection {
  target: DisruptionTarget;
  /** World anchor (metres) the popover points at; follows the camera. */
  wx: number;
  wy: number;
  /** Disruption the user clicked (chip), if any. */
  disruptionId?: number;
}

export type CameraRequest =
  | { nonce: number; kind: "view"; view: CameraState }
  | { nonce: number; kind: "bbox"; bbox: Bbox }
  /** Animated zoom about the centre of the visible map area (factor > 1 zooms in). */
  | { nonce: number; kind: "zoom"; factor: number }
  | { nonce: number; kind: "home" };

/** `Omit` that distributes over a union (plain `Omit` collapses it to the common keys). */
type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;
export type CameraRequestInput = DistributiveOmit<CameraRequest, "nonce">;

export type OnboardingStep = "presets" | "toggle" | "done";

/** Sparkline window: the last 10 sim-minutes. */
export const HISTORY_WINDOW_S = 600;
/** Sparkline series length: the last N metrics messages (≈ N s of wall time at METRICS_HZ = 1), at any speed. */
export const TREND_POINTS = 40;
export const FEED_LIMIT = 200;
export const MAX_TOASTS = 2;

export interface AppState {
  network: Network | null;
  networkError: string | null;
  connection: ConnectionState;
  session: SessionInfo | null;
  settings: Settings;
  /** Sim clock of the latest frame (updated at frame rate). */
  clockS: number | null;
  hasFrame: boolean;
  twinActive: boolean;
  metrics: MetricsMessage | null;
  history: MetricsPoint[];
  /** The last {@link TREND_POINTS} metrics points (sparklines), however fast the sim runs. */
  trend: MetricsPoint[];
  disruptions: DisruptionInfo[];
  feed: FeedItem[];
  toasts: Toast[];
  selection: Selection | null;
  cameraRequest: CameraRequest | null;
  infoOpen: boolean;
  feedOpen: boolean;
  /** ModeToggle pulses after a disruption starts while in Normal mode. */
  modePulse: boolean;
  onboarding: OnboardingStep;
  /** Station highlighted by the search box / feed (pulsing ring on the map). */
  highlight: { stationIdx: number; until: number } | null;
  /** Preset most recently triggered (cleared by reset / profile change / clear all). */
  activePreset: string | null;
  /** The map key popover is open (toasts step aside). */
  keyOpen: boolean;
  callout: ActiveCallout | null;
  /** Cumulative AI-vs-Normal totals since the AI took over (null without a twin). */
  score: AiScoreState | null;
  disruptionWindows: DisruptionWindow[];
}

export interface AppActions {
  setNetwork(net: Network): void;
  setNetworkError(msg: string): void;
  setConnection(patch: Partial<ConnectionState>): void;
  setSession(info: SessionInfo): void;
  patchSettings(patch: Partial<Settings>): void;
  setFrameInfo(clockS: number, twinActive: boolean): void;
  pushMetrics(m: MetricsMessage): void;
  clearHistory(): void;
  setDisruptions(items: DisruptionInfo[]): void;
  addFeed(items: FeedItem[]): void;
  pushToast(t: Omit<Toast, "id" | "createdAt"> & { id?: string }): void;
  dismissToast(id: string): void;
  select(sel: Selection | null): void;
  requestCamera(req: CameraRequestInput): void;
  setInfoOpen(open: boolean): void;
  setFeedOpen(open: boolean): void;
  setModePulse(on: boolean): void;
  setOnboarding(step: OnboardingStep): void;
  setHighlight(stationIdx: number | null, durationMs?: number): void;
  setActivePreset(id: string | null): void;
  setKeyOpen(open: boolean): void;
  showCallout(c: ActiveCallout | null): void;
  /** Record that a disruption ended at sim time `t`. */
  endDisruptionWindow(id: number, t: number): void;
}

export type Store = AppState & AppActions;

export const DEFAULT_SETTINGS: Settings = { mode: "normal", speed: "1x", profile: "am_peak" };

const initialState: AppState = {
  network: null,
  networkError: null,
  connection: { status: "idle", attempt: 0, retryAt: null, message: null, justReconnected: false, rttMs: null },
  session: null,
  settings: DEFAULT_SETTINGS,
  clockS: null,
  hasFrame: false,
  twinActive: false,
  metrics: null,
  history: [],
  trend: [],
  disruptions: [],
  feed: [],
  toasts: [],
  selection: null,
  cameraRequest: null,
  infoOpen: false,
  feedOpen: false,
  modePulse: false,
  onboarding: "presets",
  highlight: null,
  activePreset: null,
  keyOpen: false,
  callout: null,
  score: null,
  disruptionWindows: [],
};

/** Merge the active disruption list into the kept windows (sparkline shading). */
export function mergeWindows(windows: readonly DisruptionWindow[], active: readonly DisruptionInfo[]): DisruptionWindow[] {
  const out = windows.map((w) => ({ ...w }));
  for (const d of active) {
    const w = out.find((x) => x.id === d.id);
    if (w) {
      if (!w.ended) w.end_t = d.end_t;
    } else out.push({ id: d.id, start_t: d.start_t, end_t: d.end_t, ended: false });
  }
  return out.slice(-24);
}

let cameraNonce = 0;
let toastSeq = 0;

/** Append a metrics point, dropping points older than the window or from a previous run. */
export function appendHistory(history: MetricsPoint[], m: MetricsMessage, windowS = HISTORY_WINDOW_S): MetricsPoint[] {
  const last = history[history.length - 1];
  // Clock moved backwards → the sim was reset: start a fresh series.
  const base = last && m.t < last.t ? [] : history;
  const cutoff = m.t - windowS;
  const kept = base.filter((p) => p.t >= cutoff);
  kept.push({ t: m.t, live: m.live, twin: m.twin });
  return kept;
}

/** Append a metrics point to the sparkline trend (last {@link TREND_POINTS} points; restarts on a sim reset). */
export function appendTrend(trend: MetricsPoint[], m: MetricsMessage, maxPoints = TREND_POINTS): MetricsPoint[] {
  const last = trend[trend.length - 1];
  const base = last && m.t < last.t ? [] : trend;
  const next = base.length >= maxPoints ? base.slice(base.length - maxPoints + 1) : base.slice();
  next.push({ t: m.t, live: m.live, twin: m.twin });
  return next;
}

export const useStore = create<Store>()((set) => ({
  ...initialState,
  setNetwork: (network) => set({ network, networkError: null }),
  setNetworkError: (networkError) => set({ networkError }),
  setConnection: (patch) => set((s) => ({ connection: { ...s.connection, ...patch } })),
  setSession: (session) => set({ session }),
  patchSettings: (patch) => set((s) => ({ settings: { ...s.settings, ...patch } })),
  setFrameInfo: (clockS, twinActive) =>
    set((s) => (s.clockS === clockS && s.hasFrame && s.twinActive === twinActive ? s : { clockS, twinActive, hasFrame: true })),
  pushMetrics: (m) =>
    set((s) => ({
      metrics: m,
      history: appendHistory(s.history, m),
      trend: appendTrend(s.trend, m),
      score: accumulateScore(s.score, m),
      // A sim reset (clock ran backwards) forgets the old disruption windows.
      disruptionWindows: s.metrics && m.t < s.metrics.t ? [] : s.disruptionWindows,
    })),
  clearHistory: () => set({ history: [], trend: [], metrics: null, score: null, disruptionWindows: [] }),
  setDisruptions: (disruptions) => set((s) => ({ disruptions, disruptionWindows: mergeWindows(s.disruptionWindows, disruptions) })),
  addFeed: (items) =>
    set((s) => {
      if (items.length === 0) return s;
      const known = new Set(s.feed.map((f) => f.id));
      const fresh = items.filter((i) => !known.has(i.id));
      return { feed: [...fresh.reverse(), ...s.feed].slice(0, FEED_LIMIT) };
    }),
  pushToast: (t) =>
    set((s) => {
      const id = t.id ?? `toast-${++toastSeq}`;
      if (s.toasts.some((x) => x.id === id)) return s;
      const toast: Toast = { ...t, id, createdAt: Date.now() };
      const kept = t.dedupeKey ? s.toasts.filter((x) => x.dedupeKey !== t.dedupeKey) : s.toasts;
      return { toasts: [...kept, toast].slice(-MAX_TOASTS) };
    }),
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
  select: (selection) => set({ selection }),
  requestCamera: (req) => set({ cameraRequest: { ...req, nonce: ++cameraNonce } }),
  setInfoOpen: (infoOpen) => set({ infoOpen }),
  setFeedOpen: (feedOpen) => set({ feedOpen }),
  setModePulse: (modePulse) => set({ modePulse }),
  setOnboarding: (onboarding) => set({ onboarding }),
  setHighlight: (stationIdx, durationMs = 2600) =>
    set({ highlight: stationIdx == null ? null : { stationIdx, until: performance.now() + durationMs } }),
  setActivePreset: (activePreset) => set({ activePreset }),
  setKeyOpen: (keyOpen) => set({ keyOpen }),
  showCallout: (callout) => set({ callout }),
  endDisruptionWindow: (id, t) =>
    set((s) => ({
      disruptionWindows: s.disruptionWindows.map((w) => (w.id === id && !w.ended ? { ...w, end_t: Math.max(w.start_t, t), ended: true } : w)),
    })),
}));
