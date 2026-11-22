/**
 * WebSocket protocol v1 (ARCHITECTURE §5.3 — CONTRACT).
 *
 * - Client → server: JSON text messages ({@link ClientMessage}).
 * - Server → client: JSON text messages ({@link ServerMessage}) and little-endian
 *   binary train frames ({@link Frame}, decoded by {@link decodeFrame}).
 */

export const PROTOCOL_VERSION = 1;

// ─── Domain enums ────────────────────────────────────────────────────────────

export const MODES = ["normal", "ai"] as const;
export type Mode = (typeof MODES)[number];

/** Wire order matters: the binary frame encodes speed as an index into this list. */
export const SPEEDS = ["pause", "1x", "2x", "5x"] as const;
export type Speed = (typeof SPEEDS)[number];

export const PROFILES = ["am_peak", "midday", "pm_rush", "late_night"] as const;
export type Profile = (typeof PROFILES)[number];

export const DISRUPTION_KINDS = ["local_block", "express_block", "full_station_closure", "signal_delay"] as const;
export type DisruptionKind = (typeof DISRUPTION_KINDS)[number];

export const TARGET_TYPES = ["station", "complex", "segment", "interlocking"] as const;
export type TargetType = (typeof TARGET_TYPES)[number];

/** §4.2 action ids are positions in this list. */
export const ACTION_NAMES = [
  "CONTINUE_SCHEDULED",
  "HOLD_IN_STATION",
  "RUN_EXPRESS",
  "RUN_LOCAL",
  "SKIP_STOP",
  "SHORT_TURN_RELAY",
  "REROUTE_VIA_COMPATIBLE_TRUNK",
  "DISPATCH_YARD_GAP_TRAIN",
] as const;
export type ActionName = (typeof ACTION_NAMES)[number];
/** The seven dispatch actions reported in `metrics.actions`. */
export type DispatchActionName = Exclude<ActionName, "CONTINUE_SCHEDULED">;
export const DISPATCH_ACTION_NAMES = ACTION_NAMES.slice(1) as readonly DispatchActionName[];

/** Train status bits (§3.9). */
export const STATUS = {
  AT_PLATFORM: 1,
  TUNNEL_STUCK: 2,
  HELD: 4,
  SHORT_TURNING: 8,
  TRACK_SWITCH: 16,
  REROUTED: 32,
  GAP_TRAIN: 64,
  SKIPPING_NEXT: 128,
} as const;

/** `station_crowd` value meaning the station is closed (§3.9). */
export const STATION_CLOSED = 255;

export interface DisruptionTarget {
  type: TargetType;
  idx: number;
}

// ─── Binary frame (kind = 1) ─────────────────────────────────────────────────

export const FRAME_MAGIC = 0x5344;
export const FRAME_VERSION = 1;
export const FRAME_KIND_TRAINS = 1;
export const FRAME_HEADER_BYTES = 24;
/** bit0 of the header `flags` field. */
export const FRAME_FLAG_TWIN_ACTIVE = 1;

export interface Frame {
  seq: number;
  /** Sim seconds since 00:00 of sim day 0 (keeps increasing). */
  clockS: number;
  mode: Mode;
  speed: Speed;
  flags: number;
  twinActive: boolean;
  nTrains: number;
  nStations: number;
  /** Per train (ascending slot order, active trains only). */
  trainId: Uint16Array;
  edge: Uint16Array;
  /** `round(offset / length · 65535)`; 0 on zero-length edges. */
  frac: Uint16Array;
  route: Uint8Array;
  status: Uint8Array;
  load: Uint8Array;
  /** Per station: `100 · max waiting/capacity`, 255 = closed. */
  stationCrowd: Uint8Array;
  /** Per station: `4 · mean wait in minutes`. */
  stationWait: Uint8Array;
}

export class ProtocolError extends Error {
  override name = "ProtocolError";
}

/** Exact byte length of a v1 frame with `n` trains and `m` stations. */
export function frameByteLength(n: number, m: number): number {
  return FRAME_HEADER_BYTES + 9 * n + 2 * m;
}

function toDataView(data: ArrayBuffer | ArrayBufferView): DataView {
  if (data instanceof ArrayBuffer) return new DataView(data);
  if (ArrayBuffer.isView(data)) return new DataView(data.buffer, data.byteOffset, data.byteLength);
  throw new ProtocolError("frame must be an ArrayBuffer or a typed-array view");
}

function readU16Array(view: DataView, offset: number, n: number): Uint16Array {
  const out = new Uint16Array(n);
  for (let i = 0; i < n; i++) out[i] = view.getUint16(offset + 2 * i, true);
  return out;
}

function copyU8(view: DataView, offset: number, n: number): Uint8Array {
  // slice() copies, so the frame never aliases the (possibly pooled) socket buffer.
  return new Uint8Array(view.buffer, view.byteOffset + offset, n).slice();
}

/**
 * Decode a binary train frame. Throws {@link ProtocolError} on a bad magic,
 * version, kind, enum value or length. Accepts unaligned views.
 */
export function decodeFrame(data: ArrayBuffer | ArrayBufferView): Frame {
  const view = toDataView(data);
  if (view.byteLength < FRAME_HEADER_BYTES) {
    throw new ProtocolError(`frame too short: ${view.byteLength} bytes`);
  }
  const magic = view.getUint16(0, true);
  if (magic !== FRAME_MAGIC) throw new ProtocolError(`bad magic 0x${magic.toString(16)}`);
  const version = view.getUint8(2);
  if (version !== FRAME_VERSION) throw new ProtocolError(`unsupported frame version ${version}`);
  const kind = view.getUint8(3);
  if (kind !== FRAME_KIND_TRAINS) throw new ProtocolError(`unsupported frame kind ${kind}`);

  const seq = view.getUint32(4, true);
  const clockS = view.getFloat64(8, true);
  const n = view.getUint16(16, true);
  const m = view.getUint16(18, true);
  const modeCode = view.getUint8(20);
  const speedCode = view.getUint8(21);
  const flags = view.getUint16(22, true);

  const mode = MODES[modeCode];
  if (mode === undefined) throw new ProtocolError(`bad mode code ${modeCode}`);
  const speed = SPEEDS[speedCode];
  if (speed === undefined) throw new ProtocolError(`bad speed code ${speedCode}`);

  const expected = frameByteLength(n, m);
  if (view.byteLength < expected) {
    throw new ProtocolError(`truncated frame: ${view.byteLength} < ${expected} bytes (n=${n}, m=${m})`);
  }

  const o = FRAME_HEADER_BYTES;
  return {
    seq,
    clockS,
    mode,
    speed,
    flags,
    twinActive: (flags & FRAME_FLAG_TWIN_ACTIVE) !== 0,
    nTrains: n,
    nStations: m,
    trainId: readU16Array(view, o, n),
    edge: readU16Array(view, o + 2 * n, n),
    frac: readU16Array(view, o + 4 * n, n),
    route: copyU8(view, o + 6 * n, n),
    status: copyU8(view, o + 7 * n, n),
    load: copyU8(view, o + 8 * n, n),
    stationCrowd: copyU8(view, o + 9 * n, m),
    stationWait: copyU8(view, o + 9 * n + m, m),
  };
}

// ─── JSON messages ───────────────────────────────────────────────────────────

export interface Settings {
  mode: Mode;
  speed: Speed;
  profile: Profile;
}

export interface RestoreDisruption {
  kind: DisruptionKind;
  target: DisruptionTarget;
  duration_s: number | null;
}

export interface RestorePayload extends Settings {
  disruptions: RestoreDisruption[];
}

export type ClientMessage =
  | { type: "hello"; protocol: number; resume_token: string | null; restore: RestorePayload | null }
  | { type: "set_mode"; mode: Mode }
  | { type: "set_speed"; speed: Speed }
  | { type: "set_profile"; profile: Profile }
  | {
      type: "add_disruption";
      kind: DisruptionKind;
      target: DisruptionTarget;
      duration_s: number | null;
    }
  | { type: "clear_disruption"; id: number }
  | { type: "clear_all" }
  | { type: "preset"; id: string }
  | { type: "reset" }
  | { type: "activity" }
  | { type: "ping"; t: number };

export interface WelcomeConstants {
  dt_s: number;
  frame_hz: number;
  speeds: Record<Speed, number>;
  idle_timeout_s: number;
}

export interface WelcomeMessage {
  type: "welcome";
  protocol: number;
  session_id: string;
  resume_token: string;
  resumed: boolean;
  graph_version: string;
  model_loaded: boolean;
  ai_available: boolean;
  settings: Settings;
  constants: WelcomeConstants;
}

export interface MetricValues {
  avg_wait_min: number;
  overcrowded_platforms: number;
  tunnel_stuck_trains: number;
  riders_served: number;
  riders_waiting: number;
  trains_active: number;
  riders_onboard?: number;
}

export const METRIC_KEYS = [
  "avg_wait_min",
  "overcrowded_platforms",
  "tunnel_stuck_trains",
  "riders_served",
  "riders_waiting",
  "trains_active",
] as const satisfies readonly (keyof MetricValues)[];

export interface MetricsMessage {
  type: "metrics";
  t: number;
  mode: Mode;
  effective_speed: number;
  live: MetricValues;
  twin: MetricValues | null;
  ai_since_t: number | null;
  actions: Record<DispatchActionName, number>;
}

export interface DisruptionInfo {
  id: number;
  kind: DisruptionKind;
  target: DisruptionTarget;
  label: string;
  start_t: number;
  end_t: number | null;
}

export interface CalloutEvent {
  kind: "callout";
  id: number;
  t: number;
  train_id: number;
  route: string;
  action: ActionName;
  station_idx: number;
  text: string;
}

export interface DisruptionEvent {
  kind: "disruption_started" | "disruption_cleared" | "disruption_expired";
  disruption: DisruptionInfo;
}

export type SimEvent = CalloutEvent | DisruptionEvent;

export interface EventsMessage {
  type: "events";
  items: SimEvent[];
}

export interface DisruptionsMessage {
  type: "disruptions";
  items: DisruptionInfo[];
}

export type ErrorCode = "bad_request" | "invalid_target" | "ai_unavailable" | "rate_limited";

export interface ErrorMessage {
  type: "error";
  code: ErrorCode | (string & {});
  message: string;
}

export interface PongMessage {
  type: "pong";
  t: number;
}

export interface IdleTimeoutMessage {
  type: "idle_timeout";
}

export interface AtCapacityMessage {
  type: "at_capacity";
  message: string;
}

export type ServerMessage =
  | WelcomeMessage
  | MetricsMessage
  | EventsMessage
  | DisruptionsMessage
  | ErrorMessage
  | PongMessage
  | IdleTimeoutMessage
  | AtCapacityMessage;

// ─── Runtime validation of server JSON ───────────────────────────────────────

type Json = Record<string, unknown>;

const isObj = (v: unknown): v is Json => typeof v === "object" && v !== null && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isStr = (v: unknown): v is string => typeof v === "string";
const oneOf = <T extends string>(list: readonly T[], v: unknown): v is T =>
  typeof v === "string" && (list as readonly string[]).includes(v);
const numOrNull = (v: unknown): number | null => (isNum(v) ? v : null);

function parseTarget(v: unknown): DisruptionTarget | null {
  if (!isObj(v) || !oneOf(TARGET_TYPES, v.type) || !isNum(v.idx)) return null;
  return { type: v.type, idx: v.idx };
}

export function parseDisruption(v: unknown): DisruptionInfo | null {
  if (!isObj(v) || !isNum(v.id) || !oneOf(DISRUPTION_KINDS, v.kind)) return null;
  const target = parseTarget(v.target);
  if (!target) return null;
  return {
    id: v.id,
    kind: v.kind,
    target,
    label: isStr(v.label) ? v.label : "",
    start_t: isNum(v.start_t) ? v.start_t : 0,
    end_t: numOrNull(v.end_t),
  };
}

function parseMetricValues(v: unknown): MetricValues | null {
  if (!isObj(v)) return null;
  const num = (k: string): number => (isNum(v[k]) ? (v[k] as number) : Number.NaN);
  const out: MetricValues = {
    avg_wait_min: num("avg_wait_min"),
    overcrowded_platforms: num("overcrowded_platforms"),
    tunnel_stuck_trains: num("tunnel_stuck_trains"),
    riders_served: num("riders_served"),
    riders_waiting: num("riders_waiting"),
    trains_active: num("trains_active"),
  };
  if (isNum(v.riders_onboard)) out.riders_onboard = v.riders_onboard;
  return out;
}

function parseActions(v: unknown): Record<DispatchActionName, number> {
  const out = {} as Record<DispatchActionName, number>;
  for (const name of DISPATCH_ACTION_NAMES) {
    const value = isObj(v) ? v[name] : undefined;
    out[name] = isNum(value) ? value : 0;
  }
  return out;
}

function parseEvent(v: unknown): SimEvent | null {
  if (!isObj(v)) return null;
  if (v.kind === "callout") {
    if (!isNum(v.id) || !isStr(v.text)) return null;
    return {
      kind: "callout",
      id: v.id,
      t: isNum(v.t) ? v.t : 0,
      train_id: isNum(v.train_id) ? v.train_id : -1,
      route: isStr(v.route) ? v.route : "",
      action: oneOf(ACTION_NAMES, v.action) ? v.action : "CONTINUE_SCHEDULED",
      station_idx: isNum(v.station_idx) ? v.station_idx : -1,
      text: v.text,
    };
  }
  if (v.kind === "disruption_started" || v.kind === "disruption_cleared" || v.kind === "disruption_expired") {
    const disruption = parseDisruption(v.disruption);
    return disruption ? { kind: v.kind, disruption } : null;
  }
  return null;
}

/**
 * Parse and validate a server JSON message. Returns `null` for malformed input
 * or unknown message types (forward compatible: newer servers may add types).
 */
export function parseServerMessage(text: string): ServerMessage | null {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isObj(raw) || !isStr(raw.type)) return null;

  switch (raw.type) {
    case "welcome": {
      const s = raw.settings;
      const c = raw.constants;
      if (!isStr(raw.resume_token) || !isObj(s) || !isObj(c)) return null;
      const speeds = isObj(c.speeds) ? c.speeds : {};
      return {
        type: "welcome",
        protocol: isNum(raw.protocol) ? raw.protocol : PROTOCOL_VERSION,
        session_id: isStr(raw.session_id) ? raw.session_id : "",
        resume_token: raw.resume_token,
        resumed: raw.resumed === true,
        graph_version: isStr(raw.graph_version) ? raw.graph_version : "",
        model_loaded: raw.model_loaded === true,
        ai_available: raw.ai_available === true,
        settings: {
          mode: oneOf(MODES, s.mode) ? s.mode : "normal",
          speed: oneOf(SPEEDS, s.speed) ? s.speed : "1x",
          profile: oneOf(PROFILES, s.profile) ? s.profile : "am_peak",
        },
        constants: {
          dt_s: isNum(c.dt_s) ? c.dt_s : 10,
          frame_hz: isNum(c.frame_hz) && c.frame_hz > 0 ? c.frame_hz : 4,
          speeds: {
            pause: isNum(speeds.pause) ? speeds.pause : 0,
            "1x": isNum(speeds["1x"]) ? speeds["1x"] : 60,
            "2x": isNum(speeds["2x"]) ? speeds["2x"] : 120,
            "5x": isNum(speeds["5x"]) ? speeds["5x"] : 300,
          },
          idle_timeout_s: isNum(c.idle_timeout_s) ? c.idle_timeout_s : 300,
        },
      };
    }
    case "metrics": {
      const live = parseMetricValues(raw.live);
      if (!live || !isNum(raw.t)) return null;
      return {
        type: "metrics",
        t: raw.t,
        mode: oneOf(MODES, raw.mode) ? raw.mode : "normal",
        effective_speed: isNum(raw.effective_speed) ? raw.effective_speed : 1,
        live,
        twin: parseMetricValues(raw.twin),
        ai_since_t: numOrNull(raw.ai_since_t),
        actions: parseActions(raw.actions),
      };
    }
    case "events": {
      if (!Array.isArray(raw.items)) return null;
      const items = raw.items.map(parseEvent).filter((e): e is SimEvent => e !== null);
      return { type: "events", items };
    }
    case "disruptions": {
      if (!Array.isArray(raw.items)) return null;
      const items = raw.items.map(parseDisruption).filter((d): d is DisruptionInfo => d !== null);
      return { type: "disruptions", items };
    }
    case "error":
      return { type: "error", code: isStr(raw.code) ? raw.code : "bad_request", message: isStr(raw.message) ? raw.message : "" };
    case "pong":
      return isNum(raw.t) ? { type: "pong", t: raw.t } : null;
    case "idle_timeout":
      return { type: "idle_timeout" };
    case "at_capacity":
      return {
        type: "at_capacity",
        message: isStr(raw.message) ? raw.message : "Subway Domino is at capacity right now — please try again in a minute.",
      };
    default:
      return null;
  }
}

export function isMode(v: unknown): v is Mode {
  return oneOf(MODES, v);
}
export function isSpeed(v: unknown): v is Speed {
  return oneOf(SPEEDS, v);
}
export function isProfile(v: unknown): v is Profile {
  return oneOf(PROFILES, v);
}
