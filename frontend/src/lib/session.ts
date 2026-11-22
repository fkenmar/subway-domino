/**
 * Session controller: glues the WebSocket client, the zustand store and the
 * non-reactive FrameBuffer, and exposes the user commands. One per page.
 */
import { WS_URL } from "./config";
import { remainingS } from "./disruptions";
import { FrameBuffer } from "./interp";
import type { GeoPreset } from "./network";
import type {
  ClientMessage,
  DisruptionKind,
  DisruptionTarget,
  Frame,
  Mode,
  Profile,
  RestorePayload,
  ServerMessage,
  Settings,
  Speed,
} from "./protocol";
import { type FeedItem, useStore } from "./store";
import { documentVisibility, type HelloState, WsClient, type WsStatus, type WsStatusInfo } from "./wsClient";

const STORAGE_KEY = "subway-domino.session.v1";
/** Matches the server's DETACH_GRACE_S: a resume token is useless after this. */
const RESUME_GRACE_MS = 90_000;
/** Ignore conflicting server echoes of a setting for this long after a local change. */
const OPTIMISTIC_HOLD_MS = 2_000;
const OUTBOX_LIMIT = 16;
const RECONNECTED_BANNER_MS = 2_500;

interface Persisted {
  resumeToken: string | null;
  restore: RestorePayload | null;
  savedAt: number;
}

function readPersisted(): Persisted | null {
  try {
    const raw = globalThis.sessionStorage?.getItem(STORAGE_KEY);
    if (!raw) return null;
    const p = JSON.parse(raw) as Persisted;
    if (typeof p.savedAt !== "number" || Date.now() - p.savedAt > RESUME_GRACE_MS) return null;
    return p;
  } catch {
    return null;
  }
}

function writePersisted(p: Persisted): void {
  try {
    globalThis.sessionStorage?.setItem(STORAGE_KEY, JSON.stringify(p));
  } catch {
    /* storage unavailable (private mode, quota) — resume still works in-memory */
  }
}

export class SessionController {
  readonly frames = new FrameBuffer();
  private client: WsClient | null = null;
  private resumeToken: string | null = null;
  private pending: Partial<Record<keyof Settings, { value: string; until: number }>> = {};
  private outbox: ClientMessage[] = [];
  private reconnectedTimer: ReturnType<typeof setTimeout> | null = null;
  private warnedGraphMismatch = false;
  private persistTimer: ReturnType<typeof setInterval> | null = null;

  start(): void {
    if (this.client) return;
    const store = useStore.getState();
    if (!WS_URL) {
      store.setConnection({
        status: "unconfigured",
        message: "The live simulation isn’t configured for this build (NEXT_PUBLIC_WS_URL is missing).",
      });
      return;
    }
    const saved = readPersisted();
    if (saved) {
      this.resumeToken = saved.resumeToken;
      if (saved.restore) {
        store.patchSettings({ mode: saved.restore.mode, speed: saved.restore.speed, profile: saved.restore.profile });
      }
    }
    this.client = new WsClient({
      url: WS_URL,
      getHello: () => this.hello(saved),
      onMessage: (m) => this.onMessage(m),
      onFrame: (f, t) => this.onFrame(f, t),
      onStatus: (s, i) => this.onStatus(s, i),
      visibility: typeof document !== "undefined" ? documentVisibility(document) : null,
      log: (msg, err) => console.warn(`[subway-domino] ${msg}`, err ?? ""),
    });
    this.client.start();
    // Keep the persisted restore payload fresh so a reload within the grace window resumes.
    this.persistTimer = setInterval(() => this.persist(), 5_000);
    if (typeof window !== "undefined") window.addEventListener("pagehide", this.persist);
  }

  stop(): void {
    this.client?.stop();
    this.client = null;
    if (this.persistTimer) clearInterval(this.persistTimer);
    this.persistTimer = null;
    if (typeof window !== "undefined") window.removeEventListener("pagehide", this.persist);
  }

  // ─── commands ───────────────────────────────────────────────────────────

  setMode(mode: Mode): void {
    const s = useStore.getState();
    if (mode === "ai" && s.session && !s.session.aiAvailable) return;
    this.optimistic("mode", mode);
    s.patchSettings({ mode });
    if (mode === "ai") {
      s.setModePulse(false);
      if (s.onboarding !== "done") s.setOnboarding("done");
    }
    this.send({ type: "set_mode", mode });
  }

  setSpeed(speed: Speed): void {
    this.optimistic("speed", speed);
    useStore.getState().patchSettings({ speed });
    this.send({ type: "set_speed", speed });
  }

  setProfile(profile: Profile): void {
    const s = useStore.getState();
    this.optimistic("profile", profile);
    s.patchSettings({ profile });
    s.clearHistory();
    s.setModePulse(false);
    s.setActivePreset(null);
    this.send({ type: "set_profile", profile });
  }

  addDisruption(kind: DisruptionKind, target: DisruptionTarget, durationS: number | null): void {
    this.send({ type: "add_disruption", kind, target, duration_s: durationS });
  }

  clearDisruption(id: number): void {
    this.send({ type: "clear_disruption", id });
  }

  clearAll(): void {
    useStore.getState().setActivePreset(null);
    this.send({ type: "clear_all" });
  }

  triggerPreset(preset: GeoPreset): void {
    const s = useStore.getState();
    this.optimistic("profile", preset.profile);
    s.patchSettings({ profile: preset.profile });
    s.clearHistory();
    s.select(null);
    if (preset.focus) {
      s.requestCamera({
        kind: "view",
        view: { cx: preset.focus.x, cy: preset.focus.y, mPerPx: preset.focus.zoom_m_per_px },
      });
    }
    if (s.onboarding === "presets") s.setOnboarding("toggle");
    s.setActivePreset(preset.id);
    this.send({ type: "preset", id: preset.id });
  }

  reset(): void {
    const s = useStore.getState();
    s.clearHistory();
    s.setModePulse(false);
    s.select(null);
    s.setActivePreset(null);
    this.send({ type: "reset" });
  }

  resume(): void {
    this.client?.resume();
  }

  /** Called on user interaction (pointer, wheel, keys). */
  activity(): void {
    this.client?.noteActivity();
  }

  // ─── internals ──────────────────────────────────────────────────────────

  private send(msg: ClientMessage): void {
    if (this.client?.send(msg)) return;
    // Not connected yet (or reconnecting): deliver after the next welcome.
    // Settings messages supersede earlier ones of the same type.
    const supersedes = msg.type === "set_mode" || msg.type === "set_speed" || msg.type === "set_profile";
    if (supersedes) this.outbox = this.outbox.filter((m) => m.type !== msg.type);
    this.outbox.push(msg);
    if (this.outbox.length > OUTBOX_LIMIT) this.outbox.splice(0, this.outbox.length - OUTBOX_LIMIT);
  }

  private optimistic(key: keyof Settings, value: string): void {
    this.pending[key] = { value, until: performance.now() + OPTIMISTIC_HOLD_MS };
  }

  /** Accept a server-reported setting unless a local change is still settling. */
  private reconcile(key: keyof Settings, value: string): void {
    const p = this.pending[key];
    if (p && performance.now() < p.until && p.value !== value) return;
    delete this.pending[key];
    const s = useStore.getState();
    if (s.settings[key] !== value) s.patchSettings({ [key]: value } as Partial<Settings>);
  }

  private restorePayload(): RestorePayload {
    const s = useStore.getState();
    return {
      mode: s.settings.mode,
      speed: s.settings.speed,
      profile: s.settings.profile,
      disruptions: s.disruptions.map((d) => {
        const left = remainingS(d, s.clockS);
        return { kind: d.kind, target: d.target, duration_s: left == null ? null : Math.max(60, Math.round(left)) };
      }),
    };
  }

  private hello(saved: Persisted | null): HelloState {
    const s = useStore.getState();
    // Fresh page load without saved state: let the server choose defaults.
    if (!s.session && !this.resumeToken && !saved?.restore) return { resumeToken: null, restore: null };
    const restore = !s.session && saved?.restore ? saved.restore : this.restorePayload();
    return { resumeToken: this.resumeToken, restore };
  }

  private persist = (): void => {
    const s = useStore.getState();
    if (!s.session) return;
    writePersisted({ resumeToken: this.resumeToken, restore: this.restorePayload(), savedAt: Date.now() });
  };

  private onStatus(status: WsStatus, info: WsStatusInfo): void {
    const s = useStore.getState();
    const mapped = status === "stopped" ? "idle" : status;
    const justReconnected = status === "open" && info.reconnected === true;
    s.setConnection({
      status: mapped,
      attempt: info.attempt,
      retryAt: info.retryAt,
      message: info.message,
      justReconnected,
    });
    if (justReconnected) {
      if (this.reconnectedTimer) clearTimeout(this.reconnectedTimer);
      this.reconnectedTimer = setTimeout(() => useStore.getState().setConnection({ justReconnected: false }), RECONNECTED_BANNER_MS);
    }
  }

  private onFrame(frame: Frame, receivedAt: number): void {
    this.frames.push(frame, receivedAt);
    useStore.getState().setFrameInfo(frame.clockS, frame.twinActive);
    this.reconcile("mode", frame.mode);
    this.reconcile("speed", frame.speed);
  }

  private onMessage(msg: ServerMessage): void {
    const s = useStore.getState();
    switch (msg.type) {
      case "welcome": {
        this.resumeToken = msg.resume_token;
        this.frames.setFrameHz(msg.constants.frame_hz);
        if (!msg.resumed) {
          this.frames.reset();
          s.clearHistory();
        }
        s.setSession({
          sessionId: msg.session_id,
          graphVersion: msg.graph_version,
          modelLoaded: msg.model_loaded,
          aiAvailable: msg.ai_available,
          resumed: msg.resumed,
          constants: msg.constants,
        });
        this.pending = {};
        s.patchSettings(msg.settings);
        if (!msg.ai_available && msg.settings.mode === "ai") s.patchSettings({ mode: "normal" });
        const net = s.network;
        if (net && msg.graph_version && net.graphVersion !== msg.graph_version && !this.warnedGraphMismatch) {
          this.warnedGraphMismatch = true;
          console.warn(
            `[subway-domino] graph_version mismatch: map ${net.graphVersion}, server ${msg.graph_version}`,
          );
          s.pushToast({
            kind: "notice",
            text: "The map data and the live simulation are different versions — reload the page if trains look misplaced.",
          });
        }
        const queued = this.outbox;
        this.outbox = [];
        for (const m of queued) this.client?.send(m);
        this.persist();
        return;
      }
      case "metrics":
        s.pushMetrics(msg);
        this.reconcile("mode", msg.mode);
        return;
      case "disruptions":
        s.setDisruptions(msg.items);
        return;
      case "events": {
        const feed: FeedItem[] = [];
        for (const ev of msg.items) {
          if (ev.kind === "callout") {
            feed.push({ id: `c${ev.id}`, kind: "callout", t: ev.t, event: ev });
            s.pushToast({
              id: `callout-${ev.id}`,
              dedupeKey: `${ev.action}:${ev.route}:${ev.station_idx}`,
              kind: "callout",
              text: ev.text,
              route: ev.route,
              stationIdx: ev.station_idx,
              action: ev.action,
            });
          } else {
            const t = ev.kind === "disruption_started" ? ev.disruption.start_t : (s.clockS ?? ev.disruption.start_t);
            feed.push({ id: `${ev.kind}-${ev.disruption.id}`, kind: "disruption", t, event: ev });
            if (ev.kind === "disruption_started" && useStore.getState().settings.mode === "normal") {
              s.setModePulse(true);
            }
          }
        }
        s.addFeed(feed);
        return;
      }
      case "error":
        if (msg.code === "ai_unavailable") {
          s.patchSettings({ mode: "normal" });
          delete this.pending.mode;
          if (s.session) s.setSession({ ...s.session, aiAvailable: false });
        }
        s.pushToast({ kind: "error", text: msg.message || "The server rejected that request." });
        return;
      case "pong":
        s.setConnection({ rttMs: Math.max(0, performance.now() - msg.t) });
        return;
      case "at_capacity":
      case "idle_timeout":
        return; // reflected through onStatus
    }
  }
}

let controller: SessionController | null = null;

/** The page-wide session controller (client-side singleton). */
export function getSession(): SessionController {
  controller ??= new SessionController();
  return controller;
}
