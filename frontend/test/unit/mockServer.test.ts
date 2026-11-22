/**
 * The dev mock (scripts/mock-server.mjs) must speak protocol v1 (§5.3) exactly
 * as the frontend decodes it — otherwise UI work against it proves nothing.
 * Boots the mock in-process on an ephemeral port and talks to it over a real socket.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import WebSocket from "ws";
import { locate, makePose } from "@/lib/positioning";
import {
  type ClientMessage,
  decodeFrame,
  DISPATCH_ACTION_NAMES,
  type Frame,
  parseServerMessage,
  type ServerMessage,
} from "@/lib/protocol";
import { type MockServerHandle, startMockServer } from "../../scripts/mock-server.mjs";
import { miniNetwork } from "../helpers/geo";

const GEO_PATH = fileURLToPath(new URL("../fixtures/mini_geo.json", import.meta.url));
const net = miniNetwork();

type Incoming = string | Buffer;

class TestClient {
  private readonly ws: WebSocket;
  private readonly inbox: Incoming[] = [];
  private wake: (() => void)[] = [];
  readonly opened: Promise<void>;
  readonly closed: Promise<{ code: number; reason: string }>;

  constructor(url: string) {
    this.ws = new WebSocket(url);
    this.ws.on("message", (data: Buffer, isBinary: boolean) => {
      this.inbox.push(isBinary ? data : data.toString("utf8"));
      for (const w of this.wake.splice(0)) w();
    });
    this.opened = new Promise((resolve, reject) => {
      this.ws.once("open", () => resolve());
      this.ws.once("error", reject);
    });
    this.closed = new Promise((resolve) => {
      this.ws.once("close", (code: number, reason: Buffer) => resolve({ code, reason: reason.toString("utf8") }));
    });
  }

  send(msg: ClientMessage): void {
    this.ws.send(JSON.stringify(msg));
  }

  hello(resumeToken: string | null = null): void {
    this.send({ type: "hello", protocol: 1, resume_token: resumeToken, restore: null });
  }

  /** First queued message matching `pick` (consumed), waiting up to `timeoutMs`. */
  async next<T>(pick: (m: Incoming) => T | null, timeoutMs = 4000): Promise<T> {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      for (let i = 0; i < this.inbox.length; i++) {
        const v = pick(this.inbox[i]);
        if (v !== null) {
          this.inbox.splice(i, 1);
          return v;
        }
      }
      const left = deadline - Date.now();
      if (left <= 0) throw new Error("timed out waiting for a message");
      await new Promise<void>((resolve) => {
        const timer = setTimeout(resolve, left);
        this.wake.push(() => {
          clearTimeout(timer);
          resolve();
        });
      });
    }
  }

  json<K extends ServerMessage["type"]>(type: K, where: (m: Extract<ServerMessage, { type: K }>) => boolean = () => true) {
    return this.next((m) => {
      if (typeof m !== "string") return null;
      const msg = parseServerMessage(m);
      return msg?.type === type && where(msg as Extract<ServerMessage, { type: K }>)
        ? (msg as Extract<ServerMessage, { type: K }>)
        : null;
    });
  }

  frame(where: (f: Frame) => boolean = () => true): Promise<Frame> {
    return this.next((m) => {
      if (typeof m === "string") return null;
      const f = decodeFrame(m);
      return where(f) ? f : null;
    });
  }

  close(): void {
    this.ws.close();
  }
}

async function connect(server: MockServerHandle): Promise<TestClient> {
  const c = new TestClient(server.wsUrl);
  await c.opened;
  return c;
}

describe("mock server speaks protocol v1", () => {
  let server: MockServerHandle;

  beforeAll(async () => {
    server = await startMockServer({ port: 0, geoPath: GEO_PATH, frameHz: 20, quiet: true, seed: 7 });
  });

  afterAll(async () => {
    await server.close();
  });

  it("serves /health exactly and the geo file", async () => {
    const health = await fetch(`${server.url}/health`);
    expect(await health.text()).toBe('{"status":"ok","model_loaded":true}');
    const geo = await fetch(`${server.url}/network_geo.json`);
    expect(await geo.text()).toBe(readFileSync(GEO_PATH, "utf8"));
  });

  it("welcomes, then streams frames whose trains sit on geo-file edges", async () => {
    const c = await connect(server);
    c.hello();
    const welcome = await c.json("welcome");
    expect(welcome).toMatchObject({
      protocol: 1,
      resumed: false,
      ai_available: true,
      settings: { mode: "normal", speed: "1x", profile: "am_peak" },
      constants: { dt_s: 10, frame_hz: 20, speeds: { pause: 0, "1x": 60, "2x": 120, "5x": 300 } },
    });
    expect(welcome.graph_version).toBe(net.graphVersion);
    expect(welcome.resume_token).not.toBe("");

    const frames: Frame[] = [];
    for (let i = 0; i < 6; i++) frames.push(await c.frame());
    const pose = makePose();
    for (const f of frames) {
      expect(f.nStations).toBe(net.stations.length);
      expect(f.nTrains).toBeGreaterThan(0);
      expect(f.mode).toBe("normal");
      for (let i = 0; i < f.nTrains; i++) {
        if (i > 0) expect(f.trainId[i]).toBeGreaterThan(f.trainId[i - 1]); // ascending slot order
        expect(f.route[i]).toBeLessThan(net.routes.length);
        expect(locate(net, f.edge[i], f.frac[i] / 65535, pose), `train ${f.trainId[i]} edge ${f.edge[i]}`).toBe(true);
      }
    }
    for (let i = 1; i < frames.length; i++) {
      expect(frames[i].seq).toBeGreaterThan(frames[i - 1].seq);
      expect(frames[i].clockS).toBeGreaterThanOrEqual(frames[i - 1].clockS);
    }
    const first = frames[0];
    const last = frames.at(-1) as Frame;
    const moved = Array.from(first.trainId).some((id, i) => {
      const j = Array.from(last.trainId).indexOf(id);
      return j >= 0 && (last.edge[j] !== first.edge[i] || last.frac[j] !== first.frac[i]);
    });
    expect(moved).toBe(true);
    c.close();
  });

  it("reports AI metrics beside the Normal twin, with the twin flag on frames", async () => {
    const c = await connect(server);
    c.hello();
    await c.json("welcome");
    c.send({ type: "set_mode", mode: "ai" });
    const m = await c.json("metrics", (x) => x.mode === "ai");
    expect(m.twin).not.toBeNull();
    expect(m.ai_since_t).not.toBeNull();
    expect(Object.keys(m.actions).sort()).toEqual([...DISPATCH_ACTION_NAMES].sort());
    const f = await c.frame((x) => x.mode === "ai");
    expect(f.twinActive).toBe(true);
    c.close();
  });

  it("starts a preset meltdown with disruption events and list", async () => {
    const c = await connect(server);
    c.hello();
    await c.json("welcome");
    const preset = net.presets[0];
    c.send({ type: "preset", id: preset.id });
    const started = await c.json("events", (e) => e.items.some((i) => i.kind === "disruption_started"));
    const item = started.items.find((i) => i.kind === "disruption_started");
    expect(item && "disruption" in item ? item.disruption.kind : null).toBe(preset.disruptions[0].kind);
    const list = await c.json("disruptions", (d) => d.items.length > 0);
    expect(list.items[0]).toMatchObject({ kind: preset.disruptions[0].kind, target: preset.disruptions[0].target });
    expect(list.items[0].label).not.toBe("");

    c.send({ type: "clear_all" });
    await c.json("disruptions", (d) => d.items.length === 0);
    c.close();
  });

  it("rejects disruptions on targets that cannot take them", async () => {
    const c = await connect(server);
    c.hello();
    await c.json("welcome");
    c.send({ type: "add_disruption", kind: "signal_delay", target: { type: "station", idx: 0 }, duration_s: 600 });
    const err = await c.json("error");
    expect(err.code).toBe("invalid_target");
    c.close();
  });

  it("answers pings and reattaches a session by resume token", async () => {
    const a = await connect(server);
    a.hello();
    const first = await a.json("welcome");
    a.send({ type: "ping", t: 1234.5 });
    expect(await a.json("pong")).toEqual({ type: "pong", t: 1234.5 });
    a.close();
    await a.closed;
    await new Promise((r) => setTimeout(r, 100));

    const b = await connect(server);
    b.hello(first.resume_token);
    const again = await b.json("welcome");
    expect(again.resumed).toBe(true);
    expect(again.session_id).toBe(first.session_id);
    b.close();
  });
});

describe("mock server failure modes", () => {
  it("turns sessions away when at capacity (at_capacity + close 1013)", async () => {
    const server = await startMockServer({ port: 0, geoPath: GEO_PATH, atCapacity: true, quiet: true });
    try {
      const c = new TestClient(server.wsUrl);
      const msg = await c.json("at_capacity");
      expect(msg.message).toMatch(/capacity/);
      expect((await c.closed).code).toBe(1013);
    } finally {
      await server.close();
    }
  });

  it("times out idle sessions (idle_timeout + close 1000)", async () => {
    const server = await startMockServer({ port: 0, geoPath: GEO_PATH, idleTimeoutS: 0.5, quiet: true });
    try {
      const c = await connect(server);
      c.hello();
      await c.json("welcome");
      await c.json("idle_timeout");
      expect((await c.closed).code).toBe(1000);
    } finally {
      await server.close();
    }
  });

  it("reports AI as unavailable with --no-ai", async () => {
    const server = await startMockServer({ port: 0, geoPath: GEO_PATH, noAi: true, quiet: true });
    try {
      const c = await connect(server);
      c.hello();
      expect((await c.json("welcome")).ai_available).toBe(false);
      c.send({ type: "set_mode", mode: "ai" });
      expect((await c.json("error")).code).toBe("ai_unavailable");
      c.close();
    } finally {
      await server.close();
    }
  });
});
