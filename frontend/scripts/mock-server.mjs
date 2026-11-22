#!/usr/bin/env node
/**
 * DEV-ONLY mock of the Subway Domino backend — WebSocket protocol v1
 * (docs/ARCHITECTURE.md §5.3) — so the frontend can be exercised before the
 * real FastAPI server exists.
 *
 * It is NOT the simulator. Trains random-walk along their route's segments of
 * the geo file (dwell, leader spacing, queues behind blocks, signal crawl), and
 * "AI mode" is a scripted stand-in that holds trains at platforms, switches
 * tracks, short-turns and sends gap trains so every UI path (callouts, twin
 * metrics, status icons) can be seen. Production never runs this file.
 *
 *   node scripts/mock-server.mjs [--port 8000] [--host 127.0.0.1] [--geo path]
 *        [--max-sessions 40] [--at-capacity] [--idle-timeout 300] [--no-ai]
 *        [--frame-hz 4] [--governor 1] [--seed 1]
 *
 * HTTP: GET /health · GET /stats · GET /network_geo.json (CORS *, like §5.1).
 * WS:   /ws — hello/welcome, binary frames at FRAME_HZ, metrics at 1 Hz,
 *       events / disruptions on change, resume tokens (90 s grace), idle timeout.
 */
import { randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { WebSocketServer } from "ws";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = path.resolve(HERE, "..");

// ─── Protocol constants (§3.2, §5.2, §5.3) ───────────────────────────────────

export const PROTOCOL = 1;
const DT_S = 10;
const SPEEDS = /** @type {const} */ (["pause", "1x", "2x", "5x"]);
const SPEED_SIM_PER_S = { pause: 0, "1x": 60, "2x": 120, "5x": 300 };
const MODES = /** @type {const} */ (["normal", "ai"]);
const PROFILES = /** @type {const} */ (["am_peak", "midday", "pm_rush", "late_night"]);
const PROFILE_START_S = { am_peak: 28800, midday: 43200, pm_rush: 61200, late_night: 7200 };
const PROFILE_DENSITY = { am_peak: 1, midday: 0.75, pm_rush: 1, late_night: 0.35 };
const PROFILE_CROWD = { am_peak: 1, midday: 0.62, pm_rush: 1.05, late_night: 0.25 };
const DISRUPTION_KINDS = ["local_block", "express_block", "full_station_closure", "signal_delay"];
const TARGET_TYPES = ["station", "complex", "segment", "interlocking"];
const ACTION_NAMES = [
  "CONTINUE_SCHEDULED",
  "HOLD_IN_STATION",
  "RUN_EXPRESS",
  "RUN_LOCAL",
  "SKIP_STOP",
  "SHORT_TURN_RELAY",
  "REROUTE_VIA_COMPATIBLE_TRUNK",
  "DISPATCH_YARD_GAP_TRAIN",
];
const DISPATCH_ACTIONS = ACTION_NAMES.slice(1);
const MAX_TICKS_PER_FRAME = 8;
const DETACH_GRACE_S = 90;
const CALLOUT_MIN_INTERVAL_MS = 1500;

// Status bits (§3.9).
const AT_PLATFORM = 1;
const TUNNEL_STUCK = 2;
const HELD = 4;
const SHORT_TURNING = 8;
const TRACK_SWITCH = 16;
const REROUTED = 32;
const GAP_TRAIN = 64;

// Mock kinematics.
const V_LOCAL = 12.5;
const V_EXPRESS = 17;
const V_SIGNAL = 4.5;
const SPACING_M = 190;
const DWELL_S = 25;
const LAYOVER_S = 90;
const KEYBY_S = 45;
const MAX_HOLD_S = 180;
const HOLD_STEP_S = 30;
const TUNNEL_STUCK_S = 60;
const YARD_COOLDOWN_S = 600;

// ─── Small utilities ─────────────────────────────────────────────────────────

/** Deterministic PRNG (mulberry32). */
function rng(seed) {
  let a = seed >>> 0 || 1;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  next.state = () => a;
  next.restore = (s) => {
    a = s >>> 0;
  };
  return next;
}

const pick = (r, list) => list[Math.floor(r() * list.length) % list.length];
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

function polylineLength(pts) {
  let len = 0;
  for (let i = 1; i < pts.length; i++) len += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
  return len;
}

function parseArgs(argv) {
  const opts = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const [key, inline] = a.slice(2).split("=", 2);
    const next = argv[i + 1];
    const value = inline ?? (next !== undefined && !next.startsWith("--") ? (i++, next) : "true");
    opts[key.replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = value;
  }
  return opts;
}

// ─── Network model built from network_geo.json (§2.7) ────────────────────────

export function defaultGeoPath() {
  const real = path.join(FRONTEND, "public", "data", "network_geo.json");
  return existsSync(real) ? real : path.join(FRONTEND, "test", "fixtures", "mini_geo.json");
}

/** Lookup tables the mock sim needs, derived from the geo file. */
export function buildMockNetwork(geo) {
  const nS = geo.stations.length;
  const segLen = geo.segments.map((s) => Math.max(1, polylineLength(s.pts)));
  const routeExpress = geo.routes.map((r) => /express/i.test(r.name ?? ""));
  const edge = geo.edges.map((e) => ({
    idx: e.idx,
    kind: e.kind,
    segment: e.segment,
    segDir: e.seg_dir > 0 ? 1 : e.seg_dir < 0 ? -1 : 0,
    track: e.track === "X" ? "X" : "L",
    station: typeof e.station === "number" ? e.station : -1,
    yard: typeof e.yard === "number" ? e.yard : -1,
    from: -1,
    to: -1,
    len: 0,
  }));
  const outRun = Array.from({ length: nS }, () => []);
  const turnbacks = Array.from({ length: nS }, () => []);
  const runByKey = new Map();
  const segRoutes = geo.segments.map((s) => new Set(s.routes ?? []));
  const segHasX = new Uint8Array(geo.segments.length);
  for (const e of edge) {
    if (e.kind === "run" && e.segment >= 0 && e.segment < geo.segments.length && e.segDir !== 0) {
      const seg = geo.segments[e.segment];
      e.from = e.segDir === 1 ? seg.a : seg.b;
      e.to = e.segDir === 1 ? seg.b : seg.a;
      e.len = segLen[e.segment];
      if (e.from >= 0 && e.from < nS) outRun[e.from].push(e.idx);
      runByKey.set(`${e.segment}:${e.segDir}:${e.track}`, e.idx);
      if (e.track === "X") segHasX[e.segment] = 1;
    } else if (e.kind === "turnback" && e.station >= 0 && e.station < nS) {
      turnbacks[e.station].push(e.idx);
      e.from = e.to = e.station;
    } else if (e.kind === "crossover" && e.station >= 0) {
      e.from = e.to = e.station;
    }
  }
  const stationSegs = Array.from({ length: nS }, () => []);
  geo.segments.forEach((s, i) => {
    if (s.a >= 0 && s.a < nS) stationSegs[s.a].push(i);
    if (s.b >= 0 && s.b < nS && s.b !== s.a) stationSegs[s.b].push(i);
  });
  const stationHasX = geo.stations.map((_, s) => stationSegs[s].some((i) => segHasX[i] || geo.segments[i].has_express));
  // Yard leads: the lead station is `station` when present (see WP3 contract notes), else the nearest station.
  const yardLeads = [];
  for (const e of edge) {
    if (e.kind !== "yard_lead" || e.yard < 0 || e.yard >= geo.yards.length) continue;
    const yard = geo.yards[e.yard];
    let st = e.station;
    if (!(st >= 0 && st < nS)) {
      let best = Infinity;
      geo.stations.forEach((s, i) => {
        const d = Math.hypot(s.x - yard.x, s.y - yard.y);
        if (d < best) {
          best = d;
          st = i;
        }
      });
    }
    e.station = st;
    e.len = Math.max(50, Math.hypot(geo.stations[st].x - yard.x, geo.stations[st].y - yard.y));
    const pullOut = e.segDir !== -1;
    e.from = pullOut ? -1 : st;
    e.to = pullOut ? st : -1;
    if (pullOut) yardLeads.push({ edge: e.idx, yard: e.yard, station: st });
  }
  const base = geo.stations.map((s) => 12 + 42 * Math.exp(-(s.label_rank ?? 200) / 70));
  return { geo, nS, edge, outRun, turnbacks, runByKey, segRoutes, segHasX, stationSegs, stationHasX, yardLeads, routeExpress, base };
}

// ─── Disruption rules (§3.5) ─────────────────────────────────────────────────

export function validKinds(net, target) {
  const { geo } = net;
  switch (target.type) {
    case "station":
      if (!geo.stations[target.idx]) return [];
      return net.stationHasX[target.idx]
        ? ["full_station_closure", "local_block", "express_block"]
        : ["full_station_closure", "local_block"];
    case "complex":
      return geo.complexes[target.idx] ? ["full_station_closure"] : [];
    case "segment": {
      const seg = geo.segments[target.idx];
      if (!seg) return [];
      return seg.has_express || net.segHasX[target.idx]
        ? ["local_block", "express_block", "signal_delay"]
        : ["local_block", "signal_delay"];
    }
    case "interlocking": {
      const il = geo.interlockings[target.idx];
      if (!il) return [];
      return net.stationHasX[il.station] ? ["local_block", "express_block", "signal_delay"] : ["local_block", "signal_delay"];
    }
    default:
      return [];
  }
}

function segmentName(net, i) {
  const seg = net.geo.segments[i];
  const a = net.geo.stations[seg.a]?.name;
  const b = net.geo.stations[seg.b]?.name;
  return a && b ? `${a} ↔ ${b}` : seg.name;
}

function disruptionLabel(net, kind, target) {
  const { geo } = net;
  const where =
    target.type === "station"
      ? geo.stations[target.idx].name
      : target.type === "complex"
        ? geo.complexes[target.idx].name
        : target.type === "interlocking"
          ? geo.stations[geo.interlockings[target.idx].station].name
          : segmentName(net, target.idx);
  const onSegment = target.type === "segment";
  switch (kind) {
    case "full_station_closure":
      return target.type === "complex" ? `Station complex closed: ${where}` : `Station closed: ${where}`;
    case "local_block":
      return onSegment ? `Local track blocked: ${where}` : `Local track blocked at ${where}`;
    case "express_block":
      return onSegment ? `Express track blocked: ${where}` : `Express track blocked at ${where}`;
    default:
      return onSegment ? `Signal problems: ${where}` : `Signal problems at ${where}`;
  }
}

/** Blocked edges, slow (signal-delay) edges and closed stations for a disruption list. */
function disruptionEffects(net, disruptions) {
  const blocked = new Set();
  const slow = new Set();
  const closed = new Set();
  const blockTrack = (track, segs, station) => {
    for (const si of segs) {
      for (const dir of [1, -1]) {
        const e = net.runByKey.get(`${si}:${dir}:${track}`);
        if (e === undefined) continue;
        const ed = net.edge[e];
        if (station === undefined || ed.from === station || ed.to === station) blocked.add(e);
      }
    }
  };
  for (const d of disruptions) {
    const t = d.target;
    const station = t.type === "station" ? t.idx : t.type === "interlocking" ? net.geo.interlockings[t.idx].station : undefined;
    if (d.kind === "full_station_closure") {
      if (t.type === "complex") for (const s of net.geo.complexes[t.idx].stations) closed.add(s);
      else closed.add(t.idx);
    } else if (d.kind === "local_block" || d.kind === "express_block") {
      const track = d.kind === "local_block" ? "L" : "X";
      if (t.type === "segment") blockTrack(track, [t.idx]);
      else blockTrack(track, net.stationSegs[station], station);
    } else {
      const segs = t.type === "segment" ? [t.idx] : net.stationSegs[station];
      for (const si of segs) {
        for (const dir of [1, -1]) {
          for (const track of ["L", "X"]) {
            const e = net.runByKey.get(`${si}:${dir}:${track}`);
            if (e !== undefined) slow.add(e);
          }
        }
      }
    }
  }
  return { blocked, slow, closed };
}

// ─── The mock sim ────────────────────────────────────────────────────────────

const AI_CALLOUT = {
  HOLD_IN_STATION: (c) =>
    `Holding a ${c.route} at ${c.station} with doors open — the track ahead${c.ahead ? ` at ${c.ahead}` : ""} is ${c.jam ? "jammed with stopped trains" : "blocked"}, so ${c.riders} riders wait on a platform instead of a stalled tunnel.`,
  RUN_EXPRESS: (c) => `Switching a ${c.route} to the express track at ${c.station} to get around the blocked local track.`,
  RUN_LOCAL: (c) => `Switching a ${c.route} to the local track at ${c.station} — the express track ahead is blocked.`,
  SHORT_TURN_RELAY: (c) =>
    `Short-turning a ${c.route} at ${c.station}: ${c.riders} riders step off for the next train, and it heads back to fill the gap behind the blockage.`,
  REROUTE_VIA_COMPATIBLE_TRUNK: (c) => `Rerouting a ${c.route} at ${c.station} via ${c.via} — its usual line ahead is blocked.`,
  DISPATCH_YARD_GAP_TRAIN: (c) =>
    `Sending a gap train from ${c.yard} onto the ${c.route} at ${c.station} — riders there haven’t seen a train in ${c.minutes} minutes.`,
};

export class MockSim {
  /** @param {ReturnType<typeof buildMockNetwork>} net */
  constructor(net, profile, seed) {
    this.net = net;
    this.profile = profile;
    this.r = rng(seed);
    this.clock = PROFILE_START_S[profile];
    this.trains = [];
    this.nextId = 0;
    this.disruptions = [];
    this.effects = { blocked: new Set(), slow: new Set(), closed: new Set() };
    this.crowd = new Float32Array(net.nS);
    this.lastArrival = new Float64Array(net.nS);
    this.served = 0;
    this.actions = Object.fromEntries(DISPATCH_ACTIONS.map((a) => [a, 0]));
    this.yardStock = net.geo.yards.map(() => 4);
    this.yardCooldown = net.geo.yards.map(() => -Infinity);
    /** Callout facts produced by the last step (consumed by the session). */
    this.pendingCallouts = [];
    this.populate();
  }

  copy() {
    const c = Object.create(MockSim.prototype);
    Object.assign(c, this);
    c.r = rng(1);
    c.r.restore(this.r.state());
    c.trains = this.trains.map((t) => ({ ...t }));
    c.disruptions = this.disruptions.map((d) => ({ ...d }));
    c.effects = disruptionEffects(this.net, c.disruptions);
    c.crowd = this.crowd.slice();
    c.lastArrival = this.lastArrival.slice();
    c.actions = { ...this.actions };
    c.yardStock = [...this.yardStock];
    c.yardCooldown = [...this.yardCooldown];
    c.pendingCallouts = [];
    return c;
  }

  routeRunEdges(route) {
    const out = [];
    for (const e of this.net.edge) {
      if (e.kind === "run" && e.from >= 0 && this.net.segRoutes[e.segment].has(route)) out.push(e.idx);
    }
    return out;
  }

  populate() {
    const { net } = this;
    const density = PROFILE_DENSITY[this.profile];
    net.geo.routes.forEach((route) => {
      const edges = this.routeRunEdges(route.idx);
      if (!edges.length) return;
      const preferX = net.routeExpress[route.idx];
      const pool = edges.filter((e) => (net.edge[e].track === "X") === preferX || !net.segHasX[net.edge[e].segment]);
      const candidates = pool.length ? pool : edges;
      const count = Math.max(1, Math.round((candidates.length / 4.2) * density));
      const used = new Set();
      for (let k = 0; k < count; k++) {
        const e = pick(this.r, candidates);
        if (used.has(e)) continue;
        used.add(e);
        this.spawn(route.idx, e, this.r() * net.edge[e].len * 0.9);
      }
    });
    const scale = PROFILE_CROWD[this.profile];
    for (let s = 0; s < net.nS; s++) {
      this.crowd[s] = net.base[s] * scale * (0.8 + this.r() * 0.4);
      this.lastArrival[s] = this.clock - this.r() * 240;
    }
  }

  spawn(route, edgeIdx, off, extra = {}) {
    const t = {
      id: this.nextId++,
      route,
      edge: edgeIdx,
      off,
      v: 0,
      dwell: 0,
      dwellAtEnd: false,
      stopped: 0,
      holdTotal: 0,
      heldBySim: false,
      keyby: 0,
      status: 0,
      flags: 0,
      flagUntil: 0,
      load: 25 + Math.floor(this.r() * 55),
      next: -1,
      turnLeft: 0,
      turnTo: -1,
      ...extra,
    };
    t.next = this.planNext(t, edgeIdx);
    this.trains.push(t);
    return t;
  }

  /** Next run edge after `edgeIdx` for this train's route (random at junctions; turn back at terminals). */
  planNext(t, edgeIdx) {
    const { net } = this;
    const e = net.edge[edgeIdx];
    const at = e.to;
    if (at < 0) return -1;
    const preferX = net.routeExpress[t.route];
    const options = net.outRun[at].filter((o) => {
      const oe = net.edge[o];
      return oe.segment !== e.segment && net.segRoutes[oe.segment].has(t.route);
    });
    if (options.length) {
      const sameClass = options.filter((o) => {
        const oe = net.edge[o];
        return net.segHasX[oe.segment] ? (oe.track === "X") === preferX : true;
      });
      return pick(this.r, sameClass.length ? sameClass : options);
    }
    // Terminal: turn back on the same segment.
    if (e.kind === "run") {
      const back = net.runByKey.get(`${e.segment}:${-e.segDir}:${e.track}`) ?? net.runByKey.get(`${e.segment}:${-e.segDir}:L`);
      if (back !== undefined) return back;
    }
    return -1;
  }

  setDisruptions(list) {
    this.disruptions = list.map((d) => ({ ...d }));
    this.effects = disruptionEffects(this.net, this.disruptions);
  }

  /**
   * Is the edge enterable right now (signal at its entry clear, room at its entry)? An edge into
   * a closed station is enterable: the train stops in the tunnel at the station's home signal.
   */
  canEnter(edgeIdx, self) {
    const { effects } = this;
    if (edgeIdx < 0 || effects.blocked.has(edgeIdx)) return false;
    for (const o of this.trains) {
      if (o === self || o.edge !== edgeIdx) continue;
      if (o.off < SPACING_M) return false;
      if (effects.slow.has(edgeIdx)) return false; // single-train occupancy under signal problems
    }
    return true;
  }

  /** Walk the planned path a few edges ahead looking for a disruption (§4.2 blocked_ahead). */
  blockedAhead(t) {
    const { net, effects } = this;
    let cur = t.next;
    let prev = t.edge;
    for (let k = 0; k < 4 && cur >= 0; k++) {
      const e = net.edge[cur];
      if (effects.blocked.has(cur) || (e.to >= 0 && effects.closed.has(e.to))) {
        return { blocked: true, jam: false, station: e.to >= 0 ? e.to : e.from, depth: k };
      }
      const stuck = this.trains.filter((o) => o.edge === cur && o.status & TUNNEL_STUCK).length;
      if (stuck >= 1 && k <= 1) return { blocked: true, jam: true, station: e.to, depth: k };
      const nxt = this.peekNext(t.route, cur, prev);
      prev = cur;
      cur = nxt;
    }
    return { blocked: false, jam: false, station: -1, depth: -1 };
  }

  peekNext(route, edgeIdx, prevIdx) {
    const { net } = this;
    const e = net.edge[edgeIdx];
    if (e.to < 0) return -1;
    const pe = prevIdx >= 0 ? net.edge[prevIdx] : null;
    return (
      net.outRun[e.to].find((o) => {
        const oe = net.edge[o];
        return oe.segment !== e.segment && (!pe || oe.segment !== pe.segment) && net.segRoutes[oe.segment].has(route) && oe.track === e.track;
      }) ?? -1
    );
  }

  step(dt, ai) {
    const { net, effects } = this;
    this.clock += dt;
    const byEdge = new Map();
    for (const t of this.trains) {
      const list = byEdge.get(t.edge);
      if (list) list.push(t);
      else byEdge.set(t.edge, [t]);
    }
    for (const list of byEdge.values()) list.sort((a, b) => b.off - a.off);

    for (const t of this.trains) {
      if (t.flags && this.clock >= t.flagUntil) t.flags = 0;
      const e = net.edge[t.edge];

      // Turnback in progress (§3.2 TURNBACK_S): sits on the turnback edge at the station point.
      if (t.turnLeft > 0) {
        t.turnLeft -= dt;
        t.off = e.len > 0 ? e.len * (1 - Math.max(0, t.turnLeft) / 240) : 0;
        if (t.turnLeft <= 0 && t.turnTo >= 0 && this.canEnter(t.turnTo, t)) {
          t.edge = t.turnTo;
          t.off = 0;
          t.turnTo = -1;
          t.next = this.planNext(t, t.edge);
        }
        t.status = SHORT_TURNING;
        continue;
      }

      // Dwelling at the end of the edge (at a platform).
      if (t.dwellAtEnd) {
        const station = e.to;
        if (t.dwell > 0) {
          t.dwell -= dt;
          t.status = AT_PLATFORM | (t.heldBySim ? HELD : 0) | t.flags;
          continue;
        }
        if (effects.closed.has(station)) {
          t.status = AT_PLATFORM | HELD;
          continue;
        }
        if (ai && this.decide(t, station)) continue; // decide() sets the status
        if (t.next < 0) t.next = this.planNext(t, t.edge);
        if (this.canEnter(t.next, t)) {
          t.edge = t.next;
          t.off = 0;
          t.dwellAtEnd = false;
          t.heldBySim = false;
          t.holdTotal = 0;
          t.keyby = effects.slow.has(t.edge) ? KEYBY_S : 0;
          t.next = this.planNext(t, t.edge);
          t.status = t.flags;
        } else {
          t.status = AT_PLATFORM | t.flags; // waiting at a red signal, doors closed
        }
        continue;
      }

      // Moving along the edge.
      if (t.keyby > 0) {
        t.keyby -= dt;
        t.v = 0;
      } else {
        let limit = e.len;
        if (effects.blocked.has(t.edge)) limit = t.off; // red signal: stop where you are
        if (e.to >= 0 && effects.closed.has(e.to)) limit = Math.min(limit, e.len - 40);
        const list = byEdge.get(t.edge) ?? [];
        const i = list.indexOf(t);
        if (i > 0) limit = Math.min(limit, list[i - 1].off - SPACING_M);
        const vmax = effects.slow.has(t.edge) ? V_SIGNAL : e.track === "X" ? V_EXPRESS : V_LOCAL;
        const target = Math.max(t.off, limit);
        const step = Math.min(vmax * dt, target - t.off);
        t.v = step / dt;
        t.off += step;
      }
      const moving = t.v > 0.5;
      t.stopped = moving ? 0 : t.stopped + dt;
      if (t.off >= e.len - 0.01 && e.to >= 0 && !effects.closed.has(e.to)) {
        this.arrive(t, e.to);
      } else {
        t.status = (t.stopped >= TUNNEL_STUCK_S ? TUNNEL_STUCK : 0) | t.flags;
      }
    }

    this.updateCrowding(dt);
    if (ai) this.maybeGapTrain();
  }

  arrive(t, station) {
    t.off = this.net.edge[t.edge].len;
    t.dwellAtEnd = true;
    t.stopped = 0;
    const terminal = t.next >= 0 && this.net.edge[t.next].segment === this.net.edge[t.edge].segment;
    t.dwell = terminal ? LAYOVER_S : DWELL_S + Math.floor(this.r() * 15);
    t.status = AT_PLATFORM | t.flags;
    const cap = this.net.geo.routes[t.route]?.division === "A" ? 1100 : 1450;
    const alight = Math.round((t.load / 100) * cap * (terminal ? 1 : 0.28));
    this.served += alight;
    const boarding = this.crowd[station] * 0.25;
    t.load = clamp(Math.round(t.load * (terminal ? 0.1 : 0.72) + boarding + this.r() * 8), 5, 125);
    this.lastArrival[station] = this.clock;
    this.crowd[station] *= 0.72;
  }

  /** Scripted stand-in for the AI at a departure decision. Returns true if the train stays. */
  decide(t, station) {
    const { net, effects } = this;
    const ahead = this.blockedAhead(t);
    if (!ahead.blocked) return false;
    const routeId = net.geo.routes[t.route]?.id ?? "?";
    const stName = net.geo.stations[station]?.name ?? "the station";
    const facts = {
      route: routeId,
      station: stName,
      ahead: ahead.station >= 0 ? net.geo.stations[ahead.station]?.name : "",
      jam: ahead.jam,
      riders: Math.round((t.load / 100) * 1300),
    };
    const nextBlocked = t.next < 0 || effects.blocked.has(t.next);
    // 1) Change track class on the same segment (RUN_EXPRESS / RUN_LOCAL).
    if (nextBlocked && t.next >= 0) {
      const ne = net.edge[t.next];
      const altTrack = ne.track === "L" ? "X" : "L";
      const alt = net.runByKey.get(`${ne.segment}:${ne.segDir}:${altTrack}`);
      if (alt !== undefined && this.canEnter(alt, t)) {
        const action = altTrack === "X" ? "RUN_EXPRESS" : "RUN_LOCAL";
        this.applied(t, action, station, facts);
        t.next = alt;
        t.flags = TRACK_SWITCH;
        t.flagUntil = this.clock + 240;
        return false;
      }
    }
    // 2) Reroute at a junction onto another segment carrying a same-division route.
    if (nextBlocked && ahead.depth === 0) {
      const e = net.edge[t.edge];
      const division = net.geo.routes[t.route]?.division;
      const alts = net.outRun[station].filter((o) => {
        const oe = net.edge[o];
        if (oe.segment === e.segment || o === t.next || effects.blocked.has(o)) return false;
        return [...net.segRoutes[oe.segment]].some((r) => net.geo.routes[r]?.division === division);
      });
      const alt = alts.find((o) => this.canEnter(o, t));
      if (alt !== undefined && net.outRun[station].length >= 3) {
        const via = net.geo.stations[net.edge[alt].to]?.name ?? "another line";
        this.applied(t, "REROUTE_VIA_COMPATIBLE_TRUNK", station, { ...facts, via });
        t.next = alt;
        t.flags = REROUTED;
        t.flagUntil = this.clock + 600;
        return false;
      }
    }
    // 3) Short-turn at a pocket track after holding a while.
    const back = (() => {
      const e = net.edge[t.edge];
      return net.runByKey.get(`${e.segment}:${-e.segDir}:${e.track}`) ?? net.runByKey.get(`${e.segment}:${-e.segDir}:L`);
    })();
    if (net.turnbacks[station].length && t.holdTotal >= 60 && back !== undefined) {
      this.applied(t, "SHORT_TURN_RELAY", station, facts);
      t.load = 0;
      t.edge = net.turnbacks[station][0];
      t.off = 0;
      t.turnLeft = 180 + this.r() * 120;
      t.turnTo = back;
      t.dwellAtEnd = false;
      t.heldBySim = false;
      t.holdTotal = 0;
      t.status = SHORT_TURNING;
      return true;
    }
    // 4) Hold at the platform with doors open.
    if (t.holdTotal < MAX_HOLD_S) {
      if (t.holdTotal === 0) this.applied(t, "HOLD_IN_STATION", station, facts);
      else this.actions.HOLD_IN_STATION++;
      t.holdTotal += HOLD_STEP_S;
      t.dwell = HOLD_STEP_S;
      t.heldBySim = true;
      t.status = AT_PLATFORM | HELD | t.flags;
      return true;
    }
    return false;
  }

  applied(t, action, station, facts) {
    this.actions[action]++;
    this.pendingCallouts.push({
      action,
      trainId: t.id,
      route: this.net.geo.routes[t.route]?.id ?? "",
      stationIdx: station,
      text: AI_CALLOUT[action](facts),
    });
  }

  maybeGapTrain() {
    const { net } = this;
    if (Math.floor(this.clock / DT_S) % 6 !== 0) return;
    for (const lead of net.yardLeads) {
      if (this.yardStock[lead.yard] <= 0 || this.clock - this.yardCooldown[lead.yard] < YARD_COOLDOWN_S) continue;
      const gap = this.clock - this.lastArrival[lead.station];
      if (gap < 480 && this.crowd[lead.station] < 70) continue;
      const routes = net.geo.stations[lead.station].routes ?? [];
      if (!routes.length || !this.disruptions.length) continue;
      const route = routes[0];
      this.yardStock[lead.yard]--;
      this.yardCooldown[lead.yard] = this.clock;
      const t = this.spawn(route, lead.edge, 0, { flags: GAP_TRAIN, load: 0 });
      t.flagUntil = this.clock + 1200;
      const exits = net.outRun[lead.station].filter((o) => net.segRoutes[net.edge[o].segment].has(route));
      t.next = exits.length ? pick(this.r, exits) : -1;
      this.applied(t, "DISPATCH_YARD_GAP_TRAIN", lead.station, {
        route: net.geo.routes[route]?.id ?? "?",
        station: net.geo.stations[lead.station].name,
        yard: net.geo.yards[lead.yard].name,
        minutes: Math.max(1, Math.round(gap / 60)),
      });
    }
  }

  updateCrowding(dt) {
    const { net } = this;
    const scale = PROFILE_CROWD[this.profile];
    const k = 1 - Math.exp(-dt / 90);
    for (let s = 0; s < net.nS; s++) {
      const gap = this.clock - this.lastArrival[s];
      const starving = Math.max(0, gap - 300) / 60;
      const target = net.base[s] * scale + starving * 7;
      this.crowd[s] += (target - this.crowd[s]) * k;
      if (this.crowd[s] > 170) this.crowd[s] = 170;
    }
  }

  waitMin(s) {
    return clamp(1.2 + (this.clock - this.lastArrival[s]) / 110, 0, 30);
  }

  metrics() {
    const { net, effects } = this;
    let wNum = 0;
    let wDen = 0;
    let crowded = 0;
    let waiting = 0;
    for (let s = 0; s < net.nS; s++) {
      if (effects.closed.has(s)) continue;
      const riders = this.crowd[s] * 9;
      waiting += riders;
      wNum += riders * this.waitMin(s);
      wDen += riders;
      if (this.crowd[s] >= 80) crowded++;
    }
    let stuck = 0;
    let onboard = 0;
    for (const t of this.trains) {
      if (t.status & TUNNEL_STUCK) stuck++;
      onboard += (t.load / 100) * 1300;
    }
    return {
      clock_s: this.clock,
      avg_wait_min: Math.round((wDen > 0 ? wNum / wDen : 0) * 100) / 100,
      overcrowded_platforms: crowded,
      tunnel_stuck_trains: stuck,
      riders_served: Math.round(this.served),
      riders_waiting: Math.round(waiting),
      riders_onboard: Math.round(onboard),
      trains_active: this.trains.length,
    };
  }

  frameArrays() {
    const { net, effects } = this;
    const trains = [...this.trains].sort((a, b) => a.id - b.id);
    const crowd = new Uint8Array(net.nS);
    const wait = new Uint8Array(net.nS);
    for (let s = 0; s < net.nS; s++) {
      crowd[s] = effects.closed.has(s) ? 255 : clamp(Math.round(this.crowd[s]), 0, 254);
      wait[s] = clamp(Math.round(4 * this.waitMin(s)), 0, 255);
    }
    return {
      trainId: trains.map((t) => t.id),
      edge: trains.map((t) => t.edge),
      frac: trains.map((t) => {
        const len = net.edge[t.edge].len;
        return len > 0 ? clamp(Math.round((t.off / len) * 65535), 0, 65535) : 0;
      }),
      route: trains.map((t) => t.route),
      status: trains.map((t) => t.status & 255),
      load: trains.map((t) => clamp(Math.round(t.load), 0, 255)),
      stationCrowd: crowd,
      stationWait: wait,
    };
  }
}

// ─── Binary frame (§5.3) ─────────────────────────────────────────────────────

/** Encode a v1 train frame (little-endian, kind = 1). */
export function encodeFrame({ seq, clockS, mode, speed, flags, arrays }) {
  const n = arrays.trainId.length;
  const m = arrays.stationCrowd.length;
  const buf = Buffer.alloc(24 + 9 * n + 2 * m);
  buf.writeUInt16LE(0x5344, 0);
  buf.writeUInt8(1, 2);
  buf.writeUInt8(1, 3);
  buf.writeUInt32LE(seq >>> 0, 4);
  buf.writeDoubleLE(clockS, 8);
  buf.writeUInt16LE(n, 16);
  buf.writeUInt16LE(m, 18);
  buf.writeUInt8(MODES.indexOf(mode), 20);
  buf.writeUInt8(SPEEDS.indexOf(speed), 21);
  buf.writeUInt16LE(flags, 22);
  let o = 24;
  for (const key of ["trainId", "edge", "frac"]) {
    for (let i = 0; i < n; i++) buf.writeUInt16LE(arrays[key][i], o + 2 * i);
    o += 2 * n;
  }
  for (const key of ["route", "status", "load"]) {
    for (let i = 0; i < n; i++) buf.writeUInt8(arrays[key][i], o + i);
    o += n;
  }
  buf.set(arrays.stationCrowd, o);
  buf.set(arrays.stationWait, o + m);
  return buf;
}

// ─── Sessions ────────────────────────────────────────────────────────────────

class Session {
  constructor(server, settings, seed) {
    this.server = server;
    this.id = randomUUID();
    this.token = randomUUID();
    this.settings = { ...settings };
    this.seed = seed;
    this.sim = new MockSim(server.net, settings.profile, seed);
    this.twin = null;
    this.aiSinceT = null;
    this.disruptions = [];
    this.nextDisruptionId = 1;
    this.seq = 0;
    this.acc = 0;
    this.calloutId = 1;
    this.lastCalloutAt = 0;
    this.suppressed = 0;
    this.ws = null;
    this.timers = [];
    this.detachTimer = null;
    this.idleTimer = null;
  }

  attach(ws) {
    this.ws = ws;
    if (this.detachTimer) clearTimeout(this.detachTimer);
    this.detachTimer = null;
    const hz = this.server.opts.frameHz;
    this.timers.push(setInterval(() => this.frameTick(), 1000 / hz));
    this.timers.push(setInterval(() => this.sendMetrics(), 1000));
    this.touch();
  }

  detach(keep) {
    for (const t of this.timers) clearInterval(t);
    this.timers = [];
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.idleTimer = null;
    this.ws = null;
    if (!keep) {
      this.server.sessions.delete(this.token);
      return;
    }
    this.detachTimer = setTimeout(() => this.server.sessions.delete(this.token), DETACH_GRACE_S * 1000);
  }

  touch() {
    if (this.idleTimer) clearTimeout(this.idleTimer);
    const ms = this.server.opts.idleTimeoutS * 1000;
    this.idleTimer = setTimeout(() => {
      this.send({ type: "idle_timeout" });
      const ws = this.ws;
      this.detach(true);
      ws?.close(1000, "idle timeout");
    }, ms);
  }

  send(obj) {
    if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(obj));
  }

  welcome(resumed) {
    const ai = this.server.opts.aiAvailable;
    this.send({
      type: "welcome",
      protocol: PROTOCOL,
      session_id: this.id,
      resume_token: this.token,
      resumed,
      graph_version: this.server.net.geo.graph_version,
      model_loaded: ai,
      ai_available: ai,
      settings: this.settings,
      constants: { dt_s: DT_S, frame_hz: this.server.opts.frameHz, speeds: SPEED_SIM_PER_S, idle_timeout_s: this.server.opts.idleTimeoutS },
    });
    this.sendDisruptions();
    this.sendMetrics();
    this.frameTick(true);
  }

  // ─── simulation control ───────────────────────────────────────────────

  resetSim(profile) {
    this.settings.profile = profile;
    this.sim = new MockSim(this.server.net, profile, ++this.seed);
    this.twin = null;
    this.aiSinceT = null;
    this.disruptions = [];
    this.acc = 0;
    if (this.settings.mode === "ai") this.startTwin();
    this.sendDisruptions();
  }

  startTwin() {
    this.twin = this.sim.copy();
    this.aiSinceT = this.sim.clock;
  }

  addDisruption(kind, target, durationS, emit = true) {
    const net = this.server.net;
    if (!DISRUPTION_KINDS.includes(kind) || !target || !TARGET_TYPES.includes(target.type) || !Number.isInteger(target.idx)) {
      return this.error("bad_request", "Malformed disruption.");
    }
    if (!validKinds(net, target).includes(kind)) {
      return this.error("invalid_target", `A ${kind.replace(/_/g, " ")} can’t be placed on that ${target.type}.`);
    }
    const start = this.sim.clock;
    const d = {
      id: this.nextDisruptionId++,
      kind,
      target: { type: target.type, idx: target.idx },
      label: disruptionLabel(net, kind, target),
      start_t: start,
      end_t: durationS == null ? null : start + Math.max(60, Number(durationS)),
    };
    this.disruptions.push(d);
    this.syncDisruptions();
    if (emit) {
      this.send({ type: "events", items: [{ kind: "disruption_started", disruption: d }] });
      this.sendDisruptions();
    }
    return d;
  }

  removeDisruption(id, kind) {
    const i = this.disruptions.findIndex((d) => d.id === id);
    if (i < 0) return false;
    const [d] = this.disruptions.splice(i, 1);
    this.syncDisruptions();
    this.send({ type: "events", items: [{ kind, disruption: d }] });
    this.sendDisruptions();
    return true;
  }

  syncDisruptions() {
    this.sim.setDisruptions(this.disruptions);
    this.twin?.setDisruptions(this.disruptions);
  }

  sendDisruptions() {
    this.send({ type: "disruptions", items: this.disruptions });
  }

  error(code, message) {
    this.send({ type: "error", code, message });
    return null;
  }

  handle(msg) {
    if (msg.type !== "ping") this.touch();
    switch (msg.type) {
      case "set_mode":
        if (!MODES.includes(msg.mode)) return this.error("bad_request", "Unknown mode.");
        if (msg.mode === "ai" && !this.server.opts.aiAvailable) {
          return this.error("ai_unavailable", "The AI Dispatcher is unavailable right now (no model loaded).");
        }
        if (msg.mode === this.settings.mode) return;
        this.settings.mode = msg.mode;
        if (msg.mode === "ai") this.startTwin();
        else {
          this.twin = null;
          this.aiSinceT = null;
        }
        this.sendMetrics();
        return;
      case "set_speed":
        if (!SPEEDS.includes(msg.speed)) return this.error("bad_request", "Unknown speed.");
        this.settings.speed = msg.speed;
        return;
      case "set_profile":
        if (!PROFILES.includes(msg.profile)) return this.error("bad_request", "Unknown profile.");
        this.resetSim(msg.profile);
        this.sendMetrics();
        return;
      case "add_disruption":
        this.addDisruption(msg.kind, msg.target, msg.duration_s ?? null);
        return;
      case "clear_disruption":
        if (!this.removeDisruption(msg.id, "disruption_cleared")) this.error("bad_request", "No such disruption.");
        return;
      case "clear_all":
        for (const d of [...this.disruptions]) this.removeDisruption(d.id, "disruption_cleared");
        return;
      case "preset": {
        const preset = this.server.net.geo.presets.find((p) => p.id === msg.id);
        if (!preset) return this.error("bad_request", `Unknown preset ${msg.id}.`);
        this.resetSim(PROFILES.includes(preset.profile) ? preset.profile : this.settings.profile);
        // Like the server's _on_preset: every preset disruption is a blockage until cleared,
        // and signal problems become a local block on the same target.
        for (const d of preset.disruptions) {
          this.addDisruption(d.kind === "signal_delay" ? "local_block" : d.kind, d.target, null);
        }
        this.sendMetrics();
        return;
      }
      case "reset":
        this.resetSim(this.settings.profile);
        this.sendMetrics();
        return;
      case "activity":
        return;
      case "ping":
        this.send({ type: "pong", t: msg.t });
        return;
      case "hello":
        return this.error("bad_request", "Already said hello.");
      default:
        return this.error("bad_request", `Unknown message type ${String(msg.type)}.`);
    }
  }

  frameTick(force = false) {
    const ws = this.ws;
    if (!ws || ws.readyState !== 1) return;
    const speed = SPEED_SIM_PER_S[this.settings.speed] * this.server.opts.governor;
    this.acc += speed / this.server.opts.frameHz / DT_S;
    const ticks = Math.min(MAX_TICKS_PER_FRAME, Math.floor(this.acc));
    this.acc -= ticks;
    if (this.acc > MAX_TICKS_PER_FRAME) this.acc = 0;
    const ai = this.settings.mode === "ai";
    for (let i = 0; i < ticks; i++) {
      this.sim.step(DT_S, ai);
      this.twin?.step(DT_S, false);
      this.expireDisruptions();
    }
    this.flushCallouts();
    if (ws.bufferedAmount > 256 * 1024 && !force) return; // backpressure: frames are idempotent keyframes
    const frame = encodeFrame({
      seq: this.seq++,
      clockS: this.sim.clock,
      mode: this.settings.mode,
      speed: this.settings.speed,
      flags: this.twin ? 1 : 0,
      arrays: this.sim.frameArrays(),
    });
    ws.send(frame, { binary: true });
  }

  expireDisruptions() {
    for (const d of [...this.disruptions]) {
      if (d.end_t != null && d.end_t <= this.sim.clock) this.removeDisruption(d.id, "disruption_expired");
    }
  }

  flushCallouts() {
    const facts = this.sim.pendingCallouts;
    if (!facts.length) return;
    this.sim.pendingCallouts = [];
    const now = Date.now();
    if (now - this.lastCalloutAt < CALLOUT_MIN_INTERVAL_MS) {
      this.suppressed += facts.length;
      return;
    }
    const f = facts[0];
    const more = this.suppressed + facts.length - 1;
    this.suppressed = 0;
    this.lastCalloutAt = now;
    const text = more > 0 ? `${f.text} (+${more} more dispatch action${more === 1 ? "" : "s"})` : f.text;
    this.send({
      type: "events",
      items: [
        {
          kind: "callout",
          id: this.calloutId++,
          t: this.sim.clock,
          train_id: f.trainId,
          route: f.route,
          action: f.action,
          station_idx: f.stationIdx,
          text,
        },
      ],
    });
  }

  sendMetrics() {
    if (!this.ws) return;
    const live = this.sim.metrics();
    const twin = this.twin ? this.twin.metrics() : null;
    for (const m of [live, twin]) if (m) delete m.clock_s;
    this.send({
      type: "metrics",
      t: this.sim.clock,
      mode: this.settings.mode,
      effective_speed: this.server.opts.governor,
      live,
      twin,
      ai_since_t: this.aiSinceT,
      actions: { ...this.sim.actions },
    });
  }
}

// ─── Server ──────────────────────────────────────────────────────────────────

function normalizeOptions(raw = {}) {
  const num = (v, d) => (v === undefined || v === "" || Number.isNaN(Number(v)) ? d : Number(v));
  const bool = (v) => v === true || v === "true" || v === "1";
  const atCapacity = bool(raw.atCapacity);
  return {
    port: num(raw.port, 8000),
    host: raw.host ?? "127.0.0.1",
    geoPath: raw.geo ? path.resolve(String(raw.geo)) : (raw.geoPath ?? defaultGeoPath()),
    maxSessions: atCapacity ? 0 : num(raw.maxSessions, 40),
    idleTimeoutS: num(raw.idleTimeout ?? raw.idleTimeoutS, 300),
    aiAvailable: !bool(raw.noAi) && raw.aiAvailable !== false,
    frameHz: clamp(num(raw.frameHz, 4), 0.5, 30),
    governor: clamp(num(raw.governor, 1), 0.05, 1),
    seed: num(raw.seed, 1),
    quiet: bool(raw.quiet),
  };
}

/**
 * Start the mock server. Resolves once listening.
 * @returns {Promise<{ port: number, url: string, wsUrl: string, close: () => Promise<void> }>}
 */
export async function startMockServer(rawOptions = {}) {
  const opts = normalizeOptions(rawOptions);
  const geoText = readFileSync(opts.geoPath, "utf8");
  const geo = JSON.parse(geoText);
  const net = buildMockNetwork(geo);
  const log = (...a) => {
    if (!opts.quiet) console.log("[mock]", ...a);
  };
  const state = { opts, net, sessions: new Map(), seed: opts.seed };
  const started = Date.now();
  const activeCount = () => [...state.sessions.values()].filter((s) => s.ws).length;

  const http = createServer((req, res) => {
    const url = new URL(req.url ?? "/", "http://localhost");
    const cors = { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "*" };
    if (req.method === "OPTIONS") {
      res.writeHead(204, cors).end();
      return;
    }
    if (url.pathname === "/health") {
      res.writeHead(200, { ...cors, "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "ok", model_loaded: opts.aiAvailable }));
    } else if (url.pathname === "/stats") {
      res.writeHead(200, { ...cors, "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          sessions: activeCount(),
          detached: state.sessions.size - activeCount(),
          governor_load: 1 - opts.governor,
          uptime_s: Math.round((Date.now() - started) / 1000),
          graph_version: geo.graph_version,
          mock: true,
        }),
      );
    } else if (url.pathname === "/network_geo.json") {
      res.writeHead(200, { ...cors, "Content-Type": "application/json", "Cache-Control": "no-cache" });
      res.end(geoText);
    } else {
      res.writeHead(404, { ...cors, "Content-Type": "text/plain" }).end("not found");
    }
  });

  const wss = new WebSocketServer({ noServer: true, perMessageDeflate: { threshold: 1024 } });
  http.on("upgrade", (req, socket, head) => {
    const url = new URL(req.url ?? "/", "http://localhost");
    if (url.pathname !== "/ws") {
      socket.destroy();
      return;
    }
    wss.handleUpgrade(req, socket, head, (ws) => wss.emit("connection", ws, req));
  });

  wss.on("connection", (ws) => {
    if (activeCount() >= opts.maxSessions) {
      ws.send(JSON.stringify({ type: "at_capacity", message: "Subway Domino is at capacity right now — please try again in a minute." }));
      ws.close(1013, "at capacity");
      log("rejected: at capacity");
      return;
    }
    /** @type {Session | null} */
    let session = null;
    ws.on("message", (data, isBinary) => {
      if (isBinary) return;
      let msg;
      try {
        msg = JSON.parse(String(data));
      } catch {
        ws.send(JSON.stringify({ type: "error", code: "bad_request", message: "Invalid JSON." }));
        return;
      }
      if (!session) {
        if (msg.type !== "hello" || msg.protocol !== PROTOCOL) {
          ws.send(JSON.stringify({ type: "error", code: "bad_request", message: "Send hello (protocol 1) first." }));
          ws.close(1008, "hello required");
          return;
        }
        const resumable = msg.resume_token ? state.sessions.get(msg.resume_token) : undefined;
        if (resumable && !resumable.ws) {
          session = resumable;
          session.attach(ws);
          session.welcome(true);
          log(`resumed ${session.id.slice(0, 8)}`);
        } else {
          const r = msg.restore ?? {};
          const settings = {
            mode: MODES.includes(r.mode) && (r.mode !== "ai" || opts.aiAvailable) ? r.mode : "normal",
            speed: SPEEDS.includes(r.speed) ? r.speed : "1x",
            profile: PROFILES.includes(r.profile) ? r.profile : "am_peak",
          };
          session = new Session(state, settings, ++state.seed);
          state.sessions.set(session.token, session);
          session.attach(ws);
          if (Array.isArray(r.disruptions)) {
            for (const d of r.disruptions) session.addDisruption(d.kind, d.target, d.duration_s ?? null, false);
          }
          if (settings.mode === "ai") session.startTwin();
          session.welcome(false);
          log(`new session ${session.id.slice(0, 8)} (${settings.profile}, ${settings.mode})`);
        }
        return;
      }
      session.handle(msg);
    });
    ws.on("close", (code) => {
      if (!session) return;
      if (session.ws === ws) session.detach(true);
      log(`detached ${session.id.slice(0, 8)} (code ${code})`);
    });
    ws.on("error", () => {});
  });

  await new Promise((resolve, reject) => {
    http.once("error", reject);
    http.listen(opts.port, opts.host, () => resolve(undefined));
  });
  const address = http.address();
  const port = typeof address === "object" && address ? address.port : opts.port;
  log(`listening on http://${opts.host}:${port} (ws: /ws) — geo: ${path.relative(process.cwd(), opts.geoPath)}`);
  log(`flags: maxSessions=${opts.maxSessions} idleTimeoutS=${opts.idleTimeoutS} ai=${opts.aiAvailable} governor=${opts.governor}`);

  return {
    port,
    url: `http://${opts.host}:${port}`,
    wsUrl: `ws://${opts.host}:${port}/ws`,
    close: async () => {
      for (const s of state.sessions.values()) {
        s.detach(false);
        if (s.detachTimer) clearTimeout(s.detachTimer);
      }
      for (const c of wss.clients) c.terminate();
      await new Promise((resolve) => wss.close(() => resolve(undefined)));
      await new Promise((resolve) => http.close(() => resolve(undefined)));
    },
  };
}

const invokedDirectly = process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href;
if (invokedDirectly) {
  const server = await startMockServer(parseArgs(process.argv.slice(2)));
  const shutdown = () => {
    void server.close().then(() => process.exit(0));
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}
