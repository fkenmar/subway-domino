/**
 * Connection lifecycle (§5.2 resume, §5.3 protocol, §6 UX flow 7): hello/restore,
 * reconnect with backoff, at-capacity retry with jitter, idle pause, hidden-tab
 * disconnect, activity heartbeat. Every side effect is injected and simulated.
 */
import { describe, expect, it, vi } from "vitest";
import type { RestorePayload } from "@/lib/protocol";
import {
  ACTIVITY_THROTTLE_MS,
  BACKOFF_MAX_MS,
  backoffDelay,
  CAPACITY_RETRY_MAX_MS,
  CAPACITY_RETRY_MIN_MS,
  capacityDelay,
  HIDDEN_DISCONNECT_MS,
  PING_INTERVAL_MS,
  WELCOME_TIMEOUT_MS,
  type WebSocketLike,
  WsClient,
  type WsStatus,
  type WsStatusInfo,
} from "@/lib/wsClient";
import { encodeFrame, sampleFrame } from "../helpers/frameEncoder";

// ─── fakes ───────────────────────────────────────────────────────────────────

class ManualClock {
  now = 0;
  private seq = 0;
  private readonly tasks = new Map<number, { at: number; fn: () => void }>();

  readonly timers = {
    setTimeout: (fn: () => void, ms: number): unknown => {
      const id = ++this.seq;
      this.tasks.set(id, { at: this.now + ms, fn });
      return id;
    },
    clearTimeout: (handle: unknown): void => {
      this.tasks.delete(handle as number);
    },
  };

  /** Advance time, running due timers in order (timers may schedule timers). */
  advance(ms: number): void {
    const end = this.now + ms;
    for (;;) {
      let nextId = -1;
      let nextAt = Number.POSITIVE_INFINITY;
      for (const [id, t] of this.tasks) {
        if (t.at <= end && t.at < nextAt) {
          nextAt = t.at;
          nextId = id;
        }
      }
      if (nextId < 0) break;
      const task = this.tasks.get(nextId);
      this.tasks.delete(nextId);
      this.now = nextAt;
      task?.fn();
    }
    this.now = end;
  }

  get pendingCount(): number {
    return this.tasks.size;
  }
}

class FakeSocket implements WebSocketLike {
  binaryType = "blob";
  readyState = 0;
  readonly sent: string[] = [];
  closedWith: { code?: number; reason?: string } | null = null;
  onopen: ((ev: unknown) => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code: number; reason: string }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;

  constructor(readonly url: string) {}

  send(data: string): void {
    if (this.readyState !== 1) throw new Error("socket not open");
    this.sent.push(data);
  }

  /** Client-initiated close: the browser reports the close back. */
  close(code?: number, reason?: string): void {
    this.closedWith = { code, reason };
    this.readyState = 3;
    this.onclose?.({ code: code ?? 1005, reason: reason ?? "" });
  }

  // Server / network side.
  open(): void {
    this.readyState = 1;
    this.onopen?.({});
  }
  receive(data: unknown): void {
    this.onmessage?.({ data });
  }
  receiveJson(msg: unknown): void {
    this.receive(JSON.stringify(msg));
  }
  drop(code = 1006): void {
    this.readyState = 3;
    this.onclose?.({ code, reason: "" });
  }
  json(): Record<string, unknown>[] {
    return this.sent.map((s) => JSON.parse(s) as Record<string, unknown>);
  }
}

class FakeVisibility {
  hidden = false;
  private listeners = new Set<() => void>();
  isHidden = (): boolean => this.hidden;
  subscribe = (fn: () => void): (() => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };
  set(hidden: boolean): void {
    this.hidden = hidden;
    this.listeners.forEach((fn) => fn());
  }
}

const WELCOME = {
  type: "welcome",
  protocol: 1,
  session_id: "s1",
  resume_token: "tok-1",
  resumed: false,
  graph_version: "g",
  model_loaded: true,
  ai_available: true,
  settings: { mode: "normal", speed: "1x", profile: "am_peak" },
  constants: { dt_s: 10, frame_hz: 4, speeds: { pause: 0, "1x": 60, "2x": 120, "5x": 300 }, idle_timeout_s: 300 },
};

const RESTORE: RestorePayload = {
  mode: "ai",
  speed: "2x",
  profile: "pm_rush",
  disruptions: [{ kind: "local_block", target: { type: "segment", idx: 12 }, duration_s: null }],
};

function setup(opts: { random?: () => number; visibility?: boolean } = {}) {
  const clock = new ManualClock();
  const sockets: FakeSocket[] = [];
  const statuses: [WsStatus, WsStatusInfo][] = [];
  const messages: unknown[] = [];
  const frames: { seq: number; at: number }[] = [];
  const visibility = opts.visibility ? new FakeVisibility() : null;
  const log = vi.fn();
  let hello: { resumeToken: string | null; restore: RestorePayload | null } = { resumeToken: null, restore: null };
  const client = new WsClient({
    url: "ws://test/ws",
    getHello: () => hello,
    onMessage: (m) => messages.push(m),
    onFrame: (f, at) => frames.push({ seq: f.seq, at }),
    onStatus: (s, i) => statuses.push([s, i]),
    createSocket: (url) => {
      const s = new FakeSocket(url);
      sockets.push(s);
      return s;
    },
    now: () => clock.now,
    random: opts.random ?? (() => 0.5),
    timers: clock.timers,
    visibility,
    log,
  });
  const last = () => sockets[sockets.length - 1];
  /** Open the latest socket and complete the welcome handshake. */
  const handshake = (welcome: Record<string, unknown> = WELCOME) => {
    last().open();
    last().receiveJson(welcome);
  };
  return {
    clock,
    sockets,
    statuses,
    messages,
    frames,
    visibility,
    log,
    client,
    last,
    handshake,
    setHello: (h: typeof hello) => {
      hello = h;
    },
    status: () => client.currentStatus,
  };
}

// ─── tests ───────────────────────────────────────────────────────────────────

describe("handshake", () => {
  it("sends hello {protocol 1, resume_token, restore} as the first message", () => {
    const t = setup();
    t.setHello({ resumeToken: "tok-0", restore: RESTORE });
    t.client.start();
    expect(t.status()).toBe("connecting");
    expect(t.sockets).toHaveLength(1);
    expect(t.last().url).toBe("ws://test/ws");
    t.last().open();
    expect(t.last().binaryType).toBe("arraybuffer");
    expect(t.last().json()).toEqual([{ type: "hello", protocol: 1, resume_token: "tok-0", restore: RESTORE }]);
  });

  it("is open only after welcome; sends are dropped before that", () => {
    const t = setup();
    t.client.start();
    t.last().open();
    expect(t.status()).toBe("connecting");
    expect(t.client.send({ type: "set_speed", speed: "2x" })).toBe(false);
    t.last().receiveJson(WELCOME);
    expect(t.status()).toBe("open");
    expect(t.client.isOpen).toBe(true);
    expect(t.messages[0]).toMatchObject({ type: "welcome", resume_token: "tok-1" });
    expect(t.client.send({ type: "set_speed", speed: "2x" })).toBe(true);
    expect(t.last().json().at(-1)).toEqual({ type: "set_speed", speed: "2x" });
  });

  it("restarts the connection when no welcome arrives", () => {
    const t = setup();
    t.client.start();
    t.last().open();
    t.clock.advance(WELCOME_TIMEOUT_MS);
    expect(t.sockets[0].closedWith?.code).toBe(4000);
    expect(t.status()).toBe("connecting");
    t.clock.advance(BACKOFF_MAX_MS);
    expect(t.sockets).toHaveLength(2);
  });
});

describe("reconnect with backoff", () => {
  it("reconnects after an unexpected close with exponential, jittered delays", () => {
    const t = setup({ random: () => 1 });
    t.client.start();
    t.handshake();
    t.last().drop(1006);
    expect(t.status()).toBe("reconnecting");
    const first = t.statuses.at(-1)?.[1];
    expect(first?.retryAt).toBe(t.clock.now + backoffDelay(1, () => 1));

    t.clock.advance(backoffDelay(1, () => 1) - 1);
    expect(t.sockets).toHaveLength(1);
    t.clock.advance(1);
    expect(t.sockets).toHaveLength(2);

    // The second attempt fails before opening: the delay doubles.
    t.last().drop(1006);
    const second = t.statuses.at(-1)?.[1];
    expect(second?.attempt).toBe(2);
    expect((second?.retryAt ?? 0) - t.clock.now).toBe(backoffDelay(2, () => 1));
    expect(backoffDelay(2, () => 1)).toBe(2 * backoffDelay(1, () => 1));
  });

  it("re-sends hello with the resume token and reports a seamless resume", () => {
    const t = setup();
    t.client.start();
    t.handshake();
    t.setHello({ resumeToken: "tok-1", restore: RESTORE });
    t.last().drop(1006);
    t.clock.advance(BACKOFF_MAX_MS);
    t.last().open();
    expect(t.last().json()[0]).toEqual({ type: "hello", protocol: 1, resume_token: "tok-1", restore: RESTORE });
    t.last().receiveJson({ ...WELCOME, resumed: true });
    const [status, info] = t.statuses.at(-1) ?? [];
    expect(status).toBe("open");
    expect(info).toMatchObject({ resumed: true, reconnected: true, attempt: 0 });
  });

  it("ignores events from a superseded socket", () => {
    const t = setup();
    t.client.start();
    t.handshake();
    const stale = t.last();
    stale.drop(1006);
    t.clock.advance(BACKOFF_MAX_MS);
    stale.receiveJson(WELCOME);
    expect(t.status()).toBe("reconnecting");
  });

  it("caps the backoff and keeps ±50 % jitter", () => {
    for (let attempt = 1; attempt <= 12; attempt++) {
      const lo = backoffDelay(attempt, () => 0);
      const hi = backoffDelay(attempt, () => 1);
      expect(lo).toBeCloseTo(hi / 2, 0);
      expect(hi).toBeLessThanOrEqual(BACKOFF_MAX_MS);
    }
    expect(backoffDelay(30, () => 1)).toBe(BACKOFF_MAX_MS);
  });
});

describe("at capacity", () => {
  it("shows the server's message and retries after 30–60 s with jitter", () => {
    const t = setup({ random: () => 0.25 });
    t.client.start();
    t.last().open();
    t.last().receiveJson({ type: "at_capacity", message: "Subway Domino is at capacity right now." });
    t.last().drop(1013);
    expect(t.status()).toBe("at_capacity");
    const info = t.statuses.at(-1)?.[1];
    expect(info?.message).toBe("Subway Domino is at capacity right now.");
    const delay = (info?.retryAt ?? 0) - t.clock.now;
    expect(delay).toBe(capacityDelay(() => 0.25));
    expect(delay).toBeGreaterThanOrEqual(CAPACITY_RETRY_MIN_MS);
    expect(delay).toBeLessThanOrEqual(CAPACITY_RETRY_MAX_MS);

    t.clock.advance(delay - 1);
    expect(t.sockets).toHaveLength(1); // no fast reconnect hammering a full server
    t.clock.advance(1);
    expect(t.sockets).toHaveLength(2);
  });

  it("treats close code 1013 without a message as at capacity", () => {
    const t = setup();
    t.client.start();
    t.last().open();
    t.last().drop(1013);
    expect(t.status()).toBe("at_capacity");
    expect(t.statuses.at(-1)?.[1].message).toMatch(/capacity/);
  });

  it("spreads retries over the whole 30–60 s window", () => {
    expect(capacityDelay(() => 0)).toBe(CAPACITY_RETRY_MIN_MS);
    expect(capacityDelay(() => 1)).toBe(CAPACITY_RETRY_MAX_MS);
    expect(capacityDelay(() => 0.5)).toBe(45_000);
  });
});

describe("idle timeout", () => {
  it("pauses without reconnecting until the user resumes", () => {
    const t = setup();
    t.client.start();
    t.handshake();
    t.last().receiveJson({ type: "idle_timeout" });
    t.last().drop(1000);
    expect(t.status()).toBe("idle_timeout");
    t.clock.advance(10 * 60_000);
    expect(t.sockets).toHaveLength(1);

    t.setHello({ resumeToken: "tok-1", restore: RESTORE });
    t.client.resume();
    expect(t.sockets).toHaveLength(2);
    expect(t.status()).toBe("reconnecting");
    t.last().open();
    expect(t.last().json()[0]).toMatchObject({ type: "hello", resume_token: "tok-1", restore: RESTORE });
  });
});

describe("hidden tab", () => {
  it("disconnects after more than 60 s hidden and reconnects on return", () => {
    const t = setup({ visibility: true });
    t.client.start();
    t.handshake();
    t.visibility?.set(true);
    t.clock.advance(HIDDEN_DISCONNECT_MS - 1);
    expect(t.last().closedWith).toBeNull();
    t.clock.advance(1);
    expect(t.last().closedWith?.code).toBe(1000);
    expect(t.status()).toBe("hidden");
    t.clock.advance(5 * 60_000);
    expect(t.sockets).toHaveLength(1); // stays disconnected while hidden

    t.visibility?.set(false);
    expect(t.sockets).toHaveLength(2);
    expect(t.status()).toBe("reconnecting");
    t.last().open();
    expect(t.last().json()[0]).toMatchObject({ type: "hello", protocol: 1 });
  });

  it("keeps the connection through a short hide", () => {
    const t = setup({ visibility: true });
    t.client.start();
    t.handshake();
    t.visibility?.set(true);
    t.clock.advance(30_000);
    t.visibility?.set(false);
    t.clock.advance(HIDDEN_DISCONNECT_MS * 2);
    expect(t.sockets).toHaveLength(1);
    expect(t.last().closedWith).toBeNull();
    expect(t.status()).toBe("open");
  });

  it("retries immediately on return while waiting out a backoff", () => {
    const t = setup({ visibility: true });
    t.client.start();
    t.handshake();
    t.visibility?.set(true);
    t.last().drop(1006); // network blip while hidden
    expect(t.status()).toBe("reconnecting");
    t.visibility?.set(false);
    expect(t.sockets).toHaveLength(2);
  });
});

describe("activity heartbeat and keep-alive", () => {
  it("sends at most one activity message per 30 s, counting other messages", () => {
    const t = setup();
    t.client.start();
    t.handshake(); // hello counts as activity at t = 0
    const activity = () => t.last().json().filter((m) => m.type === "activity").length;

    t.clock.advance(1_000);
    t.client.noteActivity();
    expect(activity()).toBe(0);

    t.clock.advance(ACTIVITY_THROTTLE_MS);
    t.client.noteActivity();
    t.client.noteActivity();
    expect(activity()).toBe(1);

    t.clock.advance(5_000);
    t.client.send({ type: "set_mode", mode: "ai" }); // any non-ping message resets the server's idle timer
    t.clock.advance(ACTIVITY_THROTTLE_MS - 1);
    t.client.noteActivity();
    expect(activity()).toBe(1);
    t.clock.advance(1);
    t.client.noteActivity();
    expect(activity()).toBe(2);
  });

  it("pings while open without counting pings as activity", () => {
    const t = setup();
    t.client.start();
    t.handshake();
    t.clock.advance(PING_INTERVAL_MS * 2);
    const pings = t.last().json().filter((m) => m.type === "ping");
    expect(pings).toHaveLength(2);
    t.client.noteActivity(); // 40 s since hello: pings did not reset the throttle
    expect(t.last().json().filter((m) => m.type === "activity")).toHaveLength(1);
  });

  it("does nothing while not connected", () => {
    const t = setup();
    t.client.noteActivity();
    expect(t.client.send({ type: "reset" })).toBe(false);
    expect(t.sockets).toHaveLength(0);
  });
});

describe("data handling", () => {
  it("decodes binary frames with their arrival time and drops malformed ones (logging once)", () => {
    const t = setup();
    t.client.start();
    t.handshake();
    t.clock.advance(250);
    const bytes = encodeFrame(sampleFrame({ seq: 77 }));
    t.last().receive(bytes.buffer);
    expect(t.frames).toEqual([{ seq: 77, at: 250 }]);

    t.last().receive(new ArrayBuffer(5));
    t.last().receive(new Uint8Array(7));
    expect(t.frames).toHaveLength(1);
    expect(t.log).toHaveBeenCalledTimes(1);
  });

  it("ignores malformed or unknown JSON", () => {
    const t = setup();
    t.client.start();
    t.handshake();
    t.last().receive("{not json");
    t.last().receiveJson({ type: "from_the_future" });
    expect(t.messages).toHaveLength(1); // the welcome only
    expect(t.status()).toBe("open");
  });
});

describe("stop()", () => {
  it("closes the socket and never reconnects", () => {
    const t = setup({ visibility: true });
    t.client.start();
    t.handshake();
    t.client.stop();
    expect(t.status()).toBe("stopped");
    expect(t.sockets[0].closedWith?.code).toBe(1000);
    t.clock.advance(10 * 60_000);
    t.visibility?.set(true);
    t.visibility?.set(false);
    expect(t.sockets).toHaveLength(1);
    expect(t.clock.pendingCount).toBe(0);
  });
});
