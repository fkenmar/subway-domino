/**
 * WebSocket transport and connection lifecycle (§5.2, §5.3, §6 UX flow 7).
 *
 * - On open: `hello {protocol, resume_token, restore}`; the session is "open" on `welcome`.
 * - Unexpected close: reconnect with exponential backoff + jitter.
 * - `at_capacity` (or close 1013): retry after 30–60 s with jitter.
 * - `idle_timeout`: stay paused until the user clicks resume.
 * - Tab hidden > 60 s: close the socket; reconnect (resume token + restore) on return.
 * - User activity: throttled `{"type":"activity"}` heartbeat (≤ 1 per 30 s).
 *
 * Every side-effect source (socket factory, clock, timers, randomness,
 * visibility) is injectable so the state machine is unit-testable.
 */
import {
  type ClientMessage,
  decodeFrame,
  type Frame,
  parseServerMessage,
  PROTOCOL_VERSION,
  type RestorePayload,
  type ServerMessage,
} from "./protocol";

export const HIDDEN_DISCONNECT_MS = 60_000;
export const ACTIVITY_THROTTLE_MS = 30_000;
export const PING_INTERVAL_MS = 20_000;
export const WELCOME_TIMEOUT_MS = 10_000;
export const BACKOFF_BASE_MS = 600;
export const BACKOFF_MAX_MS = 15_000;
export const CAPACITY_RETRY_MIN_MS = 30_000;
export const CAPACITY_RETRY_MAX_MS = 60_000;
/** Server close code for "try again later" (at capacity). */
export const CLOSE_TRY_AGAIN_LATER = 1013;

export type WsStatus = "connecting" | "open" | "reconnecting" | "at_capacity" | "idle_timeout" | "hidden" | "stopped";

export interface WsStatusInfo {
  attempt: number;
  /** `now()` timestamp of the next automatic retry, or null. */
  retryAt: number | null;
  message: string | null;
  /** On "open": the server reattached the previous session state (resume token). */
  resumed?: boolean;
  /** On "open": this is a re-connection after an earlier successful session. */
  reconnected?: boolean;
}

/** Minimal surface of the browser WebSocket we rely on. */
export interface WebSocketLike {
  binaryType: string;
  readonly readyState: number;
  onopen: ((ev: unknown) => void) | null;
  onmessage: ((ev: { data: unknown }) => void) | null;
  onclose: ((ev: { code: number; reason: string }) => void) | null;
  onerror: ((ev: unknown) => void) | null;
  send(data: string): void;
  close(code?: number, reason?: string): void;
}

export interface Timers {
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
}

export interface VisibilitySource {
  isHidden(): boolean;
  subscribe(listener: () => void): () => void;
}

export interface HelloState {
  resumeToken: string | null;
  restore: RestorePayload | null;
}

export interface WsClientOptions {
  url: string;
  getHello: () => HelloState;
  onMessage: (msg: ServerMessage) => void;
  onFrame: (frame: Frame, receivedAt: number) => void;
  onStatus: (status: WsStatus, info: WsStatusInfo) => void;
  createSocket?: (url: string) => WebSocketLike;
  now?: () => number;
  random?: () => number;
  timers?: Timers;
  visibility?: VisibilitySource | null;
  log?: (message: string, err?: unknown) => void;
}

type CloseIntent = "stop" | "hidden" | "capacity" | "idle" | "restart" | null;

const WS_OPEN = 1;

/** Exponential backoff with ±50 % jitter: attempt 1 → ~0.6 s … capped at 15 s. */
export function backoffDelay(attempt: number, random: () => number): number {
  const base = Math.min(BACKOFF_MAX_MS, BACKOFF_BASE_MS * 2 ** Math.max(0, attempt - 1));
  return Math.round(base * (0.5 + random() * 0.5));
}

/** Uniform 30–60 s jittered retry after `at_capacity`. */
export function capacityDelay(random: () => number): number {
  return Math.round(CAPACITY_RETRY_MIN_MS + random() * (CAPACITY_RETRY_MAX_MS - CAPACITY_RETRY_MIN_MS));
}

export function documentVisibility(doc: Document): VisibilitySource {
  return {
    isHidden: () => doc.visibilityState === "hidden",
    subscribe: (listener) => {
      doc.addEventListener("visibilitychange", listener);
      return () => doc.removeEventListener("visibilitychange", listener);
    },
  };
}

export class WsClient {
  private readonly opts: WsClientOptions;
  private readonly now: () => number;
  private readonly random: () => number;
  private readonly timers: Timers;
  private readonly createSocket: (url: string) => WebSocketLike;

  private socket: WebSocketLike | null = null;
  private status: WsStatus = "stopped";
  private attempt = 0;
  private everOpened = false;
  private closeIntent: CloseIntent = null;
  private retryTimer: unknown = null;
  private hiddenTimer: unknown = null;
  private pingTimer: unknown = null;
  private welcomeTimer: unknown = null;
  private lastActivitySentAt = Number.NEGATIVE_INFINITY;
  private unsubscribeVisibility: (() => void) | null = null;
  private loggedDecodeError = false;

  constructor(opts: WsClientOptions) {
    this.opts = opts;
    this.now = opts.now ?? (() => performance.now());
    this.random = opts.random ?? Math.random;
    this.timers = opts.timers ?? {
      setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms),
      clearTimeout: (h) => globalThis.clearTimeout(h as ReturnType<typeof setTimeout>),
    };
    this.createSocket = opts.createSocket ?? ((url) => new WebSocket(url) as unknown as WebSocketLike);
  }

  get currentStatus(): WsStatus {
    return this.status;
  }

  get isOpen(): boolean {
    return this.status === "open" && this.socket !== null && this.socket.readyState === WS_OPEN;
  }

  start(): void {
    if (this.status !== "stopped") return;
    if (this.opts.visibility && !this.unsubscribeVisibility) {
      this.unsubscribeVisibility = this.opts.visibility.subscribe(() => this.onVisibilityChange());
    }
    this.connect();
  }

  stop(): void {
    this.clearTimer("retryTimer");
    this.clearTimer("hiddenTimer");
    this.clearTimer("pingTimer");
    this.clearTimer("welcomeTimer");
    this.unsubscribeVisibility?.();
    this.unsubscribeVisibility = null;
    const ws = this.socket;
    this.socket = null;
    this.closeIntent = null;
    if (ws) {
      ws.onopen = ws.onmessage = ws.onclose = ws.onerror = null;
      try {
        ws.close(1000, "client stopped");
      } catch {
        /* already closed */
      }
    }
    this.setStatus("stopped", { message: null, retryAt: null });
  }

  /** User asked to resume (idle pause, capacity banner "retry now", hidden tab). */
  resume(): void {
    if (this.status === "open" || this.status === "stopped") return;
    if (this.socket) return; // a connection attempt is already in flight
    this.clearTimer("retryTimer");
    this.connect();
  }

  /** Send a client message. Returns false (and drops it) when not connected. */
  send(msg: ClientMessage): boolean {
    const ws = this.socket;
    if (!ws || this.status !== "open" || ws.readyState !== WS_OPEN) return false;
    try {
      ws.send(JSON.stringify(msg));
    } catch (err) {
      this.opts.log?.("send failed", err);
      return false;
    }
    // Any message except ping resets the server's idle timer.
    if (msg.type !== "ping") this.lastActivitySentAt = this.now();
    return true;
  }

  /** User interacted with the page: heartbeat at most once per 30 s. */
  noteActivity(): void {
    if (this.status !== "open") return;
    if (this.now() - this.lastActivitySentAt < ACTIVITY_THROTTLE_MS) return;
    this.send({ type: "activity" });
  }

  // ─── internals ────────────────────────────────────────────────────────────

  private connect(): void {
    this.clearTimer("retryTimer");
    this.closeIntent = null;
    this.setStatus(this.everOpened ? "reconnecting" : "connecting", { retryAt: null, message: null });
    let ws: WebSocketLike;
    try {
      ws = this.createSocket(this.opts.url);
    } catch (err) {
      this.opts.log?.("socket construction failed", err);
      this.scheduleReconnect();
      return;
    }
    ws.binaryType = "arraybuffer";
    this.socket = ws;

    ws.onopen = () => {
      if (this.socket !== ws) return;
      const hello = this.opts.getHello();
      try {
        ws.send(
          JSON.stringify({
            type: "hello",
            protocol: PROTOCOL_VERSION,
            resume_token: hello.resumeToken,
            restore: hello.restore,
          } satisfies ClientMessage),
        );
      } catch (err) {
        this.opts.log?.("hello failed", err);
      }
      this.lastActivitySentAt = this.now();
      this.welcomeTimer = this.timers.setTimeout(() => {
        this.welcomeTimer = null;
        if (this.socket === ws && this.status !== "open") {
          this.closeIntent = "restart";
          ws.close(4000, "no welcome");
        }
      }, WELCOME_TIMEOUT_MS);
    };

    ws.onmessage = (ev) => {
      if (this.socket !== ws) return;
      this.handleData(ev.data);
    };

    ws.onerror = () => {
      /* a close event always follows */
    };

    ws.onclose = (ev) => {
      if (this.socket !== ws) return;
      this.socket = null;
      this.clearTimer("pingTimer");
      this.clearTimer("welcomeTimer");
      const intent = this.closeIntent;
      this.closeIntent = null;
      switch (intent) {
        case "stop":
          return;
        case "hidden":
          this.setStatus("hidden", { retryAt: null, message: null });
          return;
        case "capacity":
        case "idle":
          return; // status and any retry were set when the message arrived
        default:
          if (ev.code === CLOSE_TRY_AGAIN_LATER) {
            this.scheduleCapacityRetry(null);
            return;
          }
          this.scheduleReconnect();
      }
    };
  }

  private handleData(data: unknown): void {
    if (typeof data === "string") {
      const msg = parseServerMessage(data);
      if (!msg) {
        this.opts.log?.(`ignored server message: ${data.slice(0, 120)}`);
        return;
      }
      this.handleMessage(msg);
      return;
    }
    if (data instanceof ArrayBuffer || ArrayBuffer.isView(data)) {
      try {
        this.opts.onFrame(decodeFrame(data), this.now());
      } catch (err) {
        if (!this.loggedDecodeError) {
          this.loggedDecodeError = true;
          this.opts.log?.("dropping undecodable frame", err);
        }
      }
      return;
    }
    if (typeof Blob !== "undefined" && data instanceof Blob) {
      void data.arrayBuffer().then((buf) => this.handleData(buf));
    }
  }

  private handleMessage(msg: ServerMessage): void {
    switch (msg.type) {
      case "welcome": {
        this.clearTimer("welcomeTimer");
        const reconnected = this.everOpened;
        this.everOpened = true;
        this.attempt = 0;
        this.setStatus("open", { retryAt: null, message: null, resumed: msg.resumed, reconnected });
        this.schedulePing();
        this.opts.onMessage(msg);
        return;
      }
      case "at_capacity":
        this.scheduleCapacityRetry(msg.message);
        this.opts.onMessage(msg);
        return;
      case "idle_timeout":
        this.closeIntent = "idle";
        this.clearTimer("retryTimer");
        this.setStatus("idle_timeout", { retryAt: null, message: null });
        this.opts.onMessage(msg);
        return;
      default:
        this.opts.onMessage(msg);
    }
  }

  private scheduleReconnect(): void {
    this.attempt += 1;
    const delay = backoffDelay(this.attempt, this.random);
    const retryAt = this.now() + delay;
    this.setStatus(this.everOpened ? "reconnecting" : "connecting", { retryAt, message: null });
    this.retryTimer = this.timers.setTimeout(() => {
      this.retryTimer = null;
      this.connect();
    }, delay);
  }

  private scheduleCapacityRetry(message: string | null): void {
    this.closeIntent = "capacity";
    this.clearTimer("retryTimer");
    const delay = capacityDelay(this.random);
    const retryAt = this.now() + delay;
    this.setStatus("at_capacity", {
      retryAt,
      message: message ?? "Subway Domino is at capacity right now — please try again in a minute.",
    });
    this.retryTimer = this.timers.setTimeout(() => {
      this.retryTimer = null;
      this.connect();
    }, delay);
  }

  private schedulePing(): void {
    this.clearTimer("pingTimer");
    this.pingTimer = this.timers.setTimeout(() => {
      this.pingTimer = null;
      if (this.isOpen) {
        this.send({ type: "ping", t: this.now() });
        this.schedulePing();
      }
    }, PING_INTERVAL_MS);
  }

  private onVisibilityChange(): void {
    const vis = this.opts.visibility;
    if (!vis) return;
    if (vis.isHidden()) {
      if (this.hiddenTimer === null && this.status !== "stopped") {
        this.hiddenTimer = this.timers.setTimeout(() => {
          this.hiddenTimer = null;
          if (vis.isHidden()) this.disconnectForHidden();
        }, HIDDEN_DISCONNECT_MS);
      }
      return;
    }
    // Visible again.
    this.clearTimer("hiddenTimer");
    if (this.status === "hidden") {
      this.connect();
    } else if ((this.status === "reconnecting" || this.status === "connecting") && !this.socket) {
      // Waiting out a backoff while hidden: retry right away.
      this.connect();
    }
  }

  private disconnectForHidden(): void {
    this.clearTimer("retryTimer");
    const ws = this.socket;
    if (ws) {
      this.closeIntent = "hidden";
      try {
        ws.close(1000, "tab hidden");
      } catch {
        this.socket = null;
        this.setStatus("hidden", { retryAt: null, message: null });
      }
    } else if (this.status !== "idle_timeout" && this.status !== "stopped") {
      this.setStatus("hidden", { retryAt: null, message: null });
    }
  }

  private setStatus(status: WsStatus, info: Partial<WsStatusInfo>): void {
    this.status = status;
    this.opts.onStatus(status, {
      attempt: this.attempt,
      retryAt: info.retryAt ?? null,
      message: info.message ?? null,
      resumed: info.resumed,
      reconnected: info.reconnected,
    });
  }

  private clearTimer(name: "retryTimer" | "hiddenTimer" | "pingTimer" | "welcomeTimer"): void {
    const h = this[name];
    if (h !== null) {
      this.timers.clearTimeout(h);
      this[name] = null;
    }
  }
}
