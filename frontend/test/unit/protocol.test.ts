import { describe, expect, it } from "vitest";
import {
  decodeFrame,
  DISPATCH_ACTION_NAMES,
  frameByteLength,
  parseServerMessage,
  ProtocolError,
  STATION_CLOSED,
  STATUS,
} from "@/lib/protocol";
import { encodeFrame, sampleFrame } from "../helpers/frameEncoder";

describe("binary frame decoder (§5.3)", () => {
  it("round-trips every header field and array", () => {
    const input = sampleFrame();
    const f = decodeFrame(encodeFrame(input));
    expect(f).toMatchObject({ seq: 41, clockS: 30120.5, mode: "ai", speed: "2x", flags: 1, twinActive: true });
    expect(f.nTrains).toBe(3);
    expect(f.nStations).toBe(5);
    expect([...f.trainId]).toEqual(input.trains.map((t) => t.id));
    expect([...f.edge]).toEqual(input.trains.map((t) => t.edge));
    expect([...f.frac]).toEqual(input.trains.map((t) => t.frac));
    expect([...f.route]).toEqual(input.trains.map((t) => t.route));
    expect([...f.status]).toEqual(input.trains.map((t) => t.status));
    expect([...f.load]).toEqual(input.trains.map((t) => t.load));
    expect([...f.stationCrowd]).toEqual(input.stationCrowd);
    expect([...f.stationWait]).toEqual(input.stationWait);
    expect(f.stationCrowd[4]).toBe(STATION_CLOSED);
    expect(f.status[1] & STATUS.TUNNEL_STUCK).toBeTruthy();
    expect(f.status[1] & STATUS.HELD).toBeTruthy();
  });

  it("decodes every mode and speed code", () => {
    for (const mode of ["normal", "ai"] as const) {
      for (const speed of ["pause", "1x", "2x", "5x"] as const) {
        const f = decodeFrame(encodeFrame(sampleFrame({ mode, speed, flags: 0 })));
        expect(f.mode).toBe(mode);
        expect(f.speed).toBe(speed);
        expect(f.twinActive).toBe(false);
      }
    }
  });

  it("matches the documented size: 24 + 9n + 2m (≈ 5.6 KB at n=510, m=475)", () => {
    expect(frameByteLength(0, 0)).toBe(24);
    expect(frameByteLength(510, 475)).toBe(5564);
    const input = sampleFrame();
    expect(encodeFrame(input).byteLength).toBe(frameByteLength(3, 5));
  });

  it("accepts ArrayBuffer, typed-array views and unaligned offsets", () => {
    const input = sampleFrame();
    const aligned = encodeFrame(input);
    const buffer = aligned.buffer.slice(aligned.byteOffset, aligned.byteOffset + aligned.byteLength);
    expect(decodeFrame(buffer).seq).toBe(41);
    for (const byteOffset of [1, 3, 7]) {
      const view = encodeFrame(input, { byteOffset });
      expect(view.byteOffset).toBe(byteOffset);
      const f = decodeFrame(view);
      expect([...f.trainId]).toEqual([3, 17, 699]);
      expect(f.clockS).toBe(30120.5);
    }
  });

  it("handles empty frames and maximum field values", () => {
    const empty = decodeFrame(encodeFrame(sampleFrame({ trains: [], stationCrowd: [], stationWait: [] })));
    expect(empty.nTrains).toBe(0);
    expect(empty.trainId.length).toBe(0);
    expect(empty.stationCrowd.length).toBe(0);
    const max = decodeFrame(encodeFrame(sampleFrame({ seq: 0xffffffff, clockS: 1e9 + 0.25, flags: 0xffff })));
    expect(max.seq).toBe(0xffffffff);
    expect(max.clockS).toBe(1e9 + 0.25);
    expect(max.twinActive).toBe(true);
  });

  it("ignores trailing bytes after the arrays", () => {
    const f = decodeFrame(encodeFrame(sampleFrame(), { extraBytes: 9 }));
    expect([...f.stationWait]).toEqual([0, 4, 60, 120, 255]);
  });

  it("copies arrays out of the socket buffer (no aliasing)", () => {
    const bytes = encodeFrame(sampleFrame());
    const f = decodeFrame(bytes);
    bytes.fill(0);
    expect(f.trainId[1]).toBe(17);
    expect(f.load[1]).toBe(102);
    expect(f.stationCrowd[4]).toBe(255);
  });

  it.each([
    ["too short", () => new Uint8Array(23), /too short/],
    ["bad magic", () => encodeFrame(sampleFrame({ magic: 0x4453 })), /magic/],
    ["unknown version", () => encodeFrame(sampleFrame({ version: 2 })), /version/],
    ["unknown kind", () => encodeFrame(sampleFrame({ kind: 2 })), /kind/],
    ["bad mode code", () => encodeFrame(sampleFrame({ modeCode: 2 })), /mode/],
    ["bad speed code", () => encodeFrame(sampleFrame({ speedCode: 4 })), /speed/],
    ["truncated arrays", () => encodeFrame(sampleFrame({ nOverride: 40 })), /truncated/],
  ])("rejects %s", (_name, make, message) => {
    expect(() => decodeFrame(make())).toThrow(ProtocolError);
    expect(() => decodeFrame(make())).toThrow(message);
  });
});

describe("server JSON messages (§5.3)", () => {
  const welcome = {
    type: "welcome",
    protocol: 1,
    session_id: "s-1",
    resume_token: "tok",
    resumed: false,
    graph_version: "abc123def456",
    model_loaded: true,
    ai_available: true,
    settings: { mode: "normal", speed: "1x", profile: "am_peak" },
    constants: { dt_s: 10, frame_hz: 4, speeds: { pause: 0, "1x": 60, "2x": 120, "5x": 300 }, idle_timeout_s: 300 },
  };

  it("parses welcome exactly as documented", () => {
    const m = parseServerMessage(JSON.stringify(welcome));
    expect(m).toEqual(welcome);
  });

  it("fills safe defaults for a sparse welcome and rejects one without a resume token", () => {
    const sparse = parseServerMessage(
      JSON.stringify({ type: "welcome", resume_token: "t", settings: { mode: "warp" }, constants: {} }),
    );
    expect(sparse).toMatchObject({
      type: "welcome",
      ai_available: false,
      settings: { mode: "normal", speed: "1x", profile: "am_peak" },
      constants: { dt_s: 10, frame_hz: 4, idle_timeout_s: 300 },
    });
    expect(parseServerMessage(JSON.stringify({ ...welcome, resume_token: undefined }))).toBeNull();
  });

  it("parses metrics with and without the Normal twin", () => {
    const live = {
      avg_wait_min: 4.2,
      overcrowded_platforms: 3,
      tunnel_stuck_trains: 2,
      riders_served: 123456,
      riders_waiting: 23000,
      trains_active: 498,
    };
    const m = parseServerMessage(
      JSON.stringify({
        type: "metrics",
        t: 30120,
        mode: "ai",
        effective_speed: 1,
        live,
        twin: { ...live, tunnel_stuck_trains: 9 },
        ai_since_t: 29400,
        actions: { HOLD_IN_STATION: 12, SHORT_TURN_RELAY: 2 },
      }),
    );
    expect(m?.type).toBe("metrics");
    if (m?.type !== "metrics") throw new Error("unreachable");
    expect(m.live).toEqual(live);
    expect(m.twin?.tunnel_stuck_trains).toBe(9);
    expect(m.ai_since_t).toBe(29400);
    // All seven dispatch actions are present, missing ones default to 0.
    expect(Object.keys(m.actions).sort()).toEqual([...DISPATCH_ACTION_NAMES].sort());
    expect(m.actions.HOLD_IN_STATION).toBe(12);
    expect(m.actions.DISPATCH_YARD_GAP_TRAIN).toBe(0);

    const normal = parseServerMessage(JSON.stringify({ type: "metrics", t: 1, mode: "normal", live, twin: null, ai_since_t: null }));
    expect(normal).toMatchObject({ twin: null, ai_since_t: null, effective_speed: 1 });
  });

  it("parses events and drops malformed items", () => {
    const disruption = {
      id: 3,
      kind: "local_block",
      target: { type: "segment", idx: 12 },
      label: "Local track blocked: 59 St ↔ 51 St",
      start_t: 29000,
      end_t: null,
    };
    const m = parseServerMessage(
      JSON.stringify({
        type: "events",
        items: [
          {
            kind: "callout",
            id: 17,
            t: 30110,
            train_id: 12,
            route: "4",
            action: "SHORT_TURN_RELAY",
            station_idx: 123,
            text: "Short-turning a 4 at Grand Central-42 St…",
          },
          { kind: "disruption_started", disruption },
          { kind: "disruption_started", disruption: { ...disruption, kind: "meteor" } },
          { kind: "mystery" },
        ],
      }),
    );
    expect(m?.type).toBe("events");
    if (m?.type !== "events") throw new Error("unreachable");
    expect(m.items).toHaveLength(2);
    expect(m.items[0]).toMatchObject({ kind: "callout", action: "SHORT_TURN_RELAY", route: "4", station_idx: 123 });
    expect(m.items[1]).toEqual({ kind: "disruption_started", disruption });
  });

  it("parses the disruptions list, errors, pong, idle and capacity messages", () => {
    expect(parseServerMessage(JSON.stringify({ type: "disruptions", items: [] }))).toEqual({ type: "disruptions", items: [] });
    expect(parseServerMessage('{"type":"error","code":"ai_unavailable","message":"no model"}')).toEqual({
      type: "error",
      code: "ai_unavailable",
      message: "no model",
    });
    expect(parseServerMessage('{"type":"pong","t":1234.5}')).toEqual({ type: "pong", t: 1234.5 });
    expect(parseServerMessage('{"type":"idle_timeout"}')).toEqual({ type: "idle_timeout" });
    expect(parseServerMessage('{"type":"at_capacity","message":"full"}')).toEqual({ type: "at_capacity", message: "full" });
  });

  it.each(["", "not json", "[]", "42", '{"no":"type"}', '{"type":"from_the_future"}', '{"type":"pong"}'])(
    "returns null for %j",
    (text) => {
      expect(parseServerMessage(text)).toBeNull();
    },
  );
});
