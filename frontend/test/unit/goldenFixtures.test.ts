/**
 * WP4 golden wire fixtures (§5.3): `backend/tests/fixtures/protocol/frame_v1.bin`
 * + `frame_v1.json` (its decoded values) and `messages_v1.json` (one example of
 * every JSON message). Both sides test against the same files; this suite is
 * skipped until WP4 has written them.
 */
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { decodeFrame, type Frame, MODES, parseServerMessage, SPEEDS } from "@/lib/protocol";

const DIR = fileURLToPath(new URL("../../../backend/tests/fixtures/protocol", import.meta.url));
const FRAME_BIN = path.join(DIR, "frame_v1.bin");
const FRAME_JSON = path.join(DIR, "frame_v1.json");
const MESSAGES_JSON = path.join(DIR, "messages_v1.json");

const SERVER_TYPES = new Set(["welcome", "metrics", "events", "disruptions", "error", "pong", "idle_timeout", "at_capacity"]);
const CLIENT_TYPES = new Set([
  "hello",
  "set_mode",
  "set_speed",
  "set_profile",
  "add_disruption",
  "clear_disruption",
  "clear_all",
  "preset",
  "reset",
  "activity",
  "ping",
]);

type Json = Record<string, unknown>;

/** Look a field up under its snake_case or camelCase spelling. */
function field(obj: Json, ...names: string[]): unknown {
  for (const n of names) if (n in obj) return obj[n];
  return undefined;
}

/** Normalise an enum given either as its wire code or as its name. */
function enumName(v: unknown, names: readonly string[]): unknown {
  return typeof v === "number" ? names[v] : v;
}

function asNumbers(v: unknown): number[] | undefined {
  return Array.isArray(v) ? v.map(Number) : undefined;
}

/** Every JSON message in the fixture, whether it is an array, a map by type, or a map of arrays. */
function collectMessages(raw: unknown): Json[] {
  const out: Json[] = [];
  const visit = (v: unknown): void => {
    if (Array.isArray(v)) v.forEach(visit);
    else if (v && typeof v === "object") {
      const o = v as Json;
      if (typeof o.type === "string") out.push(o);
      else Object.values(o).forEach(visit);
    }
  };
  visit(raw);
  return out;
}

describe.skipIf(!existsSync(FRAME_BIN) || !existsSync(FRAME_JSON))("golden frame_v1 (WP4 fixture)", () => {
  it("decodes frame_v1.bin to the values in frame_v1.json", () => {
    const frame: Frame = decodeFrame(new Uint8Array(readFileSync(FRAME_BIN)));
    const expected = JSON.parse(readFileSync(FRAME_JSON, "utf8")) as Json;

    const scalars: [keyof Frame, string[]][] = [
      ["seq", ["seq"]],
      ["clockS", ["clock_s", "clockS", "clock"]],
      ["nTrains", ["n_trains", "nTrains", "n"]],
      ["nStations", ["n_stations", "nStations", "m"]],
      ["flags", ["flags"]],
    ];
    for (const [key, names] of scalars) {
      const want = field(expected, ...names);
      if (want !== undefined) expect(frame[key], String(key)).toBe(Number(want));
    }
    const mode = enumName(field(expected, "mode"), MODES);
    if (mode !== undefined) expect(frame.mode).toBe(mode);
    const speed = enumName(field(expected, "speed"), SPEEDS);
    if (speed !== undefined) expect(frame.speed).toBe(speed);

    const arrays: [keyof Frame, string[]][] = [
      ["trainId", ["train_id", "trainId"]],
      ["edge", ["edge"]],
      ["frac", ["frac"]],
      ["route", ["route"]],
      ["status", ["status"]],
      ["load", ["load"]],
      ["stationCrowd", ["station_crowd", "stationCrowd"]],
      ["stationWait", ["station_wait", "stationWait"]],
    ];
    let compared = 0;
    for (const [key, names] of arrays) {
      const want = asNumbers(field(expected, ...names));
      if (!want) continue;
      compared++;
      expect(Array.from(frame[key] as ArrayLike<number>), String(key)).toEqual(want);
    }
    expect(compared, "frame_v1.json should list the wire arrays").toBeGreaterThan(0);
  });
});

describe.skipIf(!existsSync(MESSAGES_JSON))("golden messages_v1 (WP4 fixture)", () => {
  const messages = existsSync(MESSAGES_JSON) ? collectMessages(JSON.parse(readFileSync(MESSAGES_JSON, "utf8"))) : [];

  it("contains every server message type", () => {
    const seen = new Set(messages.map((m) => m.type));
    for (const t of SERVER_TYPES) expect(seen.has(t), `missing ${t}`).toBe(true);
  });

  it("parses every server → client example", () => {
    for (const m of messages.filter((x) => SERVER_TYPES.has(String(x.type)))) {
      const parsed = parseServerMessage(JSON.stringify(m));
      expect(parsed, `failed to parse ${String(m.type)}`).not.toBeNull();
      expect(parsed?.type).toBe(m.type);
      if (parsed?.type === "events" && Array.isArray(m.items)) expect(parsed.items).toHaveLength(m.items.length);
      if (parsed?.type === "disruptions" && Array.isArray(m.items)) expect(parsed.items).toHaveLength(m.items.length);
    }
  });

  it("uses only known client → server message types with the documented fields", () => {
    for (const m of messages.filter((x) => !SERVER_TYPES.has(String(x.type)))) {
      expect(CLIENT_TYPES.has(String(m.type)), `unknown message type ${String(m.type)}`).toBe(true);
      if (m.type === "hello") expect(m).toHaveProperty("protocol", 1);
      if (m.type === "add_disruption") {
        expect(m).toHaveProperty("kind");
        expect(m).toHaveProperty("target.type");
        expect(m).toHaveProperty("target.idx");
        expect(m).toHaveProperty("duration_s");
      }
    }
  });
});
