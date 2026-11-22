/**
 * Frame interpolation (§6 rendering — CONTRACT behaviour).
 *
 * Frames arrive at FRAME_HZ (4 Hz); the map renders at 60 fps one frame behind:
 * positions are interpolated between the last two frames with
 * `alpha = (now − arrival(curr)) / interval`, clamped to [0, 1].
 *
 * Per train (matched by train_id and route):
 * - same edge, `frac` not decreasing → move along the edge;
 * - different edge → move at constant speed along the cheapest chain of
 *   *continuing* edges (each edge's to-station is the next one's from-station),
 *   at most {@link MAX_PATH_HOPS} edges past the old one. Edges the train's
 *   route uses are preferred; crossovers, turnbacks and reversals cost extra.
 *   Paths are cached per (prev edge, next edge, route).
 * - anything else is a real discontinuity — a new trip in a reused slot (the
 *   route changed, or `frac` ran backwards on the same edge), or no physically
 *   plausible path within the depth limit. The old train fades out where it was
 *   and the new one fades in where it is; nothing slides across the map.
 *
 * Speed is eased across frame boundaries: the distance covered within an
 * interval follows a monotone cubic Hermite whose start speed is the previous
 * interval's end speed. Positions at frame times stay exact (no added latency).
 */
import { EDGE_CROSSOVER, EDGE_RUN, EDGE_TURNBACK } from "./network";
import { fracFromWire } from "./positioning";
import type { Frame } from "./protocol";

/** Topology needed to recognise continuing edges (from `Network.edges`). */
export interface EdgeAdjacency {
  count: number;
  fromStation: Int32Array;
  toStation: Int32Array;
  lengthM: Float64Array;
}

/** Optional hints that rank candidate paths (all present on `Network.edges`). */
export interface EdgeGraph extends EdgeAdjacency {
  kind?: Uint8Array;
  segment?: Int32Array;
  segDir?: Int8Array;
  /** Routes per edge as bitsets: bit `r & 31` of word `e · routeWords + (r >> 5)`. */
  routeBits?: Uint32Array;
  routeWords?: number;
}

export interface TrainSample {
  edge: number;
  frac: number;
}

/** A resolved chain between two edges: the edges strictly between them, in travel order. */
export interface TrainPath {
  via: Int32Array;
  viaLengthM: number;
}

const MIN_INTERVAL_MS = 100;
const MAX_INTERVAL_MS = 1000;
/**
 * Largest plausible sim-clock advance between consecutive frames: 5x speed is
 * 300 sim-s/s → 75 sim-s per 4 Hz frame, capped at 8 ticks × 10 s = 80 s by the
 * server. Anything beyond this is a reset/jump, not motion.
 */
export const RESET_CLOCK_JUMP_S = 240;
/** Path search depth: edges after the previous one, the current edge included. */
export const MAX_PATH_HOPS = 6;
/** `sim/constants.py` V_MAX_CAP (m/s); with 25 % headroom bounds a plausible move. */
const V_MAX_M_S = 24.6;
const PLAUSIBLE_SLACK_M = 150;
/** Path costs (metres of track) on top of each traversed edge's length. */
const OFF_ROUTE_PENALTY_M = 3000;
const SWITCH_PENALTY_M = 150;
const REVERSAL_PENALTY_M = 1500;
const PATH_CACHE_MAX = 50_000;
/** Fade of the whole fleet after a reset (preset, profile change, new session). */
export const RESET_FADE_MS = 220;
/** Start speed ≤ this × the interval's mean speed keeps the Hermite monotone (Fritsch–Carlson). */
const MAX_EASE_RATIO = 2.8;

const DIRECT_PATH: TrainPath = { via: new Int32Array(0), viaLengthM: 0 };

const clamp01 = (v: number): number => (v <= 0 ? 0 : v >= 1 ? 1 : v);

/** Smoothstep used for fades. */
export function fadeCurve(t: number): number {
  const x = clamp01(t);
  return x * x * (3 - 2 * x);
}

/** Largest distance (m) a train can plausibly cover while the sim clock advances `dtS`. */
export function plausibleDistanceM(dtS: number): number {
  return Math.max(0, dtS) * V_MAX_M_S * 1.25 + PLAUSIBLE_SLACK_M;
}

/** True if `next` starts at the station where `prev` ends (a physical continuation). */
export function edgesContinue(adj: EdgeAdjacency, prev: number, next: number): boolean {
  if (prev < 0 || next < 0 || prev >= adj.count || next >= adj.count) return false;
  const at = adj.toStation[prev];
  return at >= 0 && at === adj.fromStation[next];
}

/**
 * Cheapest chains of continuing edges between two edges. A layered search (exact
 * for the depth limit) over out-edges by station; results are cached.
 */
export class PathFinder {
  /** Number of uncached searches run (diagnostics / tests). */
  searches = 0;
  private readonly outStart: Int32Array;
  private readonly outEdges: Int32Array;
  private readonly cache = new Map<number, TrainPath | null>();
  private readonly best: Float64Array;
  private readonly bestGen: Uint32Array;
  private gen = 0;
  private stEdge = new Int32Array(256);
  private stCost = new Float64Array(256);
  private stParent = new Int32Array(256);

  constructor(readonly graph: EdgeGraph) {
    const { count, fromStation, toStation } = graph;
    let nSt = 0;
    for (let e = 0; e < count; e++) nSt = Math.max(nSt, fromStation[e] + 1, toStation[e] + 1);
    const start = new Int32Array(nSt + 1);
    for (let e = 0; e < count; e++) if (fromStation[e] >= 0) start[fromStation[e] + 1]++;
    for (let s = 0; s < nSt; s++) start[s + 1] += start[s];
    const fill = start.slice(0, nSt);
    const out = new Int32Array(start[nSt]);
    for (let e = 0; e < count; e++) if (fromStation[e] >= 0) out[fill[fromStation[e]]++] = e;
    this.outStart = start;
    this.outEdges = out;
    this.best = new Float64Array(count);
    this.bestGen = new Uint32Array(count);
  }

  /** Cheapest path from `prev` to `next` for `route` (−1 = unknown), or null if none within the depth limit. */
  find(prev: number, next: number, route = -1): TrainPath | null {
    const count = this.graph.count;
    if (prev < 0 || next < 0 || prev >= count || next >= count || prev === next) return null;
    const key = (prev * count + next) * 257 + (route < 0 || route > 255 ? 0 : route + 1);
    const hit = this.cache.get(key);
    if (hit !== undefined) return hit;
    const found = this.search(prev, next, route);
    if (this.cache.size >= PATH_CACHE_MAX) this.cache.clear();
    this.cache.set(key, found);
    return found;
  }

  private carries(edge: number, route: number): boolean {
    const { routeBits, routeWords = 0 } = this.graph;
    if (!routeBits || route < 0 || route >> 5 >= routeWords) return true;
    return (routeBits[edge * routeWords + (route >> 5)] & (1 << (route & 31))) !== 0;
  }

  private stepCost(from: number, to: number, route: number, final: boolean): number {
    const g = this.graph;
    let c = final ? 0 : g.lengthM[to];
    const kind = g.kind ? g.kind[to] : EDGE_RUN;
    if (kind === EDGE_TURNBACK) c += REVERSAL_PENALTY_M;
    else if (kind === EDGE_CROSSOVER) c += SWITCH_PENALTY_M;
    if (
      g.kind &&
      g.segment &&
      g.segDir &&
      g.kind[from] === EDGE_RUN &&
      kind === EDGE_RUN &&
      g.segment[from] === g.segment[to] &&
      g.segDir[from] !== g.segDir[to]
    ) {
      c += REVERSAL_PENALTY_M;
    }
    if (!final && !this.carries(to, route)) c += OFF_ROUTE_PENALTY_M;
    return c;
  }

  private push(n: number, edge: number, cost: number, parent: number): number {
    if (n >= this.stEdge.length) {
      const grow = <T extends Int32Array | Float64Array>(a: T, make: (len: number) => T): T => {
        const b = make(a.length * 2);
        b.set(a);
        return b;
      };
      this.stEdge = grow(this.stEdge, (l) => new Int32Array(l));
      this.stCost = grow(this.stCost, (l) => new Float64Array(l));
      this.stParent = grow(this.stParent, (l) => new Int32Array(l));
    }
    this.stEdge[n] = edge;
    this.stCost[n] = cost;
    this.stParent[n] = parent;
    return n + 1;
  }

  private search(prev: number, next: number, route: number): TrainPath | null {
    this.searches++;
    const g = this.graph;
    if (++this.gen >= 0xffffffff) {
      this.bestGen.fill(0);
      this.gen = 1;
    }
    const gen = this.gen;
    const nSt = this.outStart.length - 1;
    let n = this.push(0, prev, 0, -1);
    let lo = 0;
    let hi = n;
    let foundCost = Number.POSITIVE_INFINITY;
    let foundParent = -1;
    for (let hop = 0; hop < MAX_PATH_HOPS && lo < hi; hop++) {
      for (let si = lo; si < hi; si++) {
        const e = this.stEdge[si];
        const cost = this.stCost[si];
        const at = g.toStation[e];
        if (at < 0 || at >= nSt) continue;
        for (let k = this.outStart[at]; k < this.outStart[at + 1]; k++) {
          const nx = this.outEdges[k];
          if (nx === e || nx === prev) continue;
          const final = nx === next;
          const nc = cost + this.stepCost(e, nx, route, final);
          if (nc >= foundCost) continue;
          if (final) {
            foundCost = nc;
            foundParent = si;
            continue;
          }
          if (this.bestGen[nx] === gen && this.best[nx] <= nc) continue;
          this.bestGen[nx] = gen;
          this.best[nx] = nc;
          n = this.push(n, nx, nc, si);
        }
      }
      lo = hi;
      hi = n;
    }
    if (foundParent < 0) return null;
    let len = 0;
    for (let si = foundParent; si > 0; si = this.stParent[si]) len++;
    if (len === 0) return DIRECT_PATH;
    const via = new Int32Array(len);
    let viaLengthM = 0;
    for (let si = foundParent, k = len - 1; si > 0; si = this.stParent[si], k--) {
      via[k] = this.stEdge[si];
      viaLengthM += g.lengthM[via[k]];
    }
    return { via, viaLengthM };
  }
}

const finders = new WeakMap<EdgeAdjacency, PathFinder>();

/** The (cached) path finder for an edge table. */
export function pathFinderFor(adj: EdgeAdjacency): PathFinder {
  let f = finders.get(adj);
  if (!f) {
    f = new PathFinder(adj);
    finders.set(adj, f);
  }
  return f;
}

/** Total length (m) of the move prev@pf → path → curr@cf. */
export function pathDistanceM(
  lengthM: Float64Array,
  prevEdge: number,
  prevFrac: number,
  path: TrainPath,
  currEdge: number,
  currFrac: number,
): number {
  return (1 - prevFrac) * lengthM[prevEdge] + path.viaLengthM + currFrac * lengthM[currEdge];
}

/** The position `s` metres into the move prev@pf → path → curr@cf. Writes into `out`. */
export function walkPath(
  lengthM: Float64Array,
  prevEdge: number,
  prevFrac: number,
  path: TrainPath,
  currEdge: number,
  currFrac: number,
  s: number,
  out: TrainSample,
): TrainSample {
  const l0 = lengthM[prevEdge];
  const d0 = (1 - prevFrac) * l0;
  let rest = s;
  if (rest < d0 && l0 > 0) {
    out.edge = prevEdge;
    out.frac = prevFrac + rest / l0;
    return out;
  }
  rest -= d0;
  const via = path.via;
  for (let k = 0; k < via.length; k++) {
    const l = lengthM[via[k]];
    if (rest < l) {
      out.edge = via[k];
      out.frac = rest / l;
      return out;
    }
    rest -= l;
  }
  const l2 = lengthM[currEdge];
  out.edge = currEdge;
  out.frac = l2 > 0 ? Math.min(currFrac, Math.max(0, rest / l2)) : currFrac;
  return out;
}

/**
 * Distance covered after `alpha` of an interval that covers `d`, starting at
 * speed `v0` and ending at the interval's mean speed `d` (units per interval).
 * A monotone cubic Hermite: never runs backwards or past `d`; `v0 = d` is linear.
 */
export function easeDistance(alpha: number, v0: number, d: number): number {
  if (!(d > 0)) return 0;
  const a = clamp01(alpha);
  const v = v0 > 0 ? Math.min(v0, MAX_EASE_RATIO * d) : 0;
  const a2 = a * a;
  const a3 = a2 * a;
  const s = (a3 - 2 * a2 + a) * v + (2 * a2 - a3) * d;
  return s <= 0 ? 0 : s >= d ? d : s;
}

/**
 * Interpolate one train between two frames. `prevFrac` / `currFrac` are
 * normalised (0..1); `alpha` ∈ [0,1]; `route` (optional) ranks candidate paths.
 * Pure (apart from the path cache); writes into `out`.
 */
export function interpolateTrain(
  prevEdge: number,
  prevFrac: number,
  currEdge: number,
  currFrac: number,
  alpha: number,
  adj: EdgeAdjacency | null,
  out: TrainSample,
  route = -1,
): TrainSample {
  const a = clamp01(alpha);
  if (prevEdge === currEdge) {
    // Directed edges never run backwards; a backwards jump means a new trip in a reused slot.
    out.edge = currEdge;
    out.frac = currFrac + 1e-6 < prevFrac ? currFrac : prevFrac + (currFrac - prevFrac) * a;
    return out;
  }
  if (adj && a < 1) {
    const path = pathFinderFor(adj).find(prevEdge, currEdge, route);
    if (path) {
      const total = pathDistanceM(adj.lengthM, prevEdge, prevFrac, path, currEdge, currFrac);
      return walkPath(adj.lengthM, prevEdge, prevFrac, path, currEdge, currFrac, a * total, out);
    }
  }
  out.edge = currEdge;
  out.frac = currFrac;
  return out;
}

/** Struct-of-arrays output of {@link FrameBuffer.sample}, reused across frames. */
export class InterpolatedTrains {
  count = 0;
  trainId: Uint16Array;
  edge: Int32Array;
  frac: Float32Array;
  route: Uint8Array;
  status: Uint8Array;
  load: Uint8Array;
  /** Opacity 0..1: < 1 while a train fades in (new trip) or out (ghost). */
  fade: Float32Array;

  constructor(capacity = 1024) {
    this.trainId = new Uint16Array(capacity);
    this.edge = new Int32Array(capacity);
    this.frac = new Float32Array(capacity);
    this.route = new Uint8Array(capacity);
    this.status = new Uint8Array(capacity);
    this.load = new Uint8Array(capacity);
    this.fade = new Float32Array(capacity);
  }

  /** Grow to at least `capacity`, keeping the first `count` entries. */
  ensure(capacity: number): void {
    if (capacity <= this.trainId.length) return;
    const cap = Math.max(capacity, this.trainId.length * 2);
    const keep = this.count;
    const grow = <T extends Uint16Array | Int32Array | Float32Array | Uint8Array>(a: T, b: T): T => {
      b.set(a.subarray(0, keep));
      return b;
    };
    this.trainId = grow(this.trainId, new Uint16Array(cap));
    this.edge = grow(this.edge, new Int32Array(cap));
    this.frac = grow(this.frac, new Float32Array(cap));
    this.route = grow(this.route, new Uint8Array(cap));
    this.status = grow(this.status, new Uint8Array(cap));
    this.load = grow(this.load, new Uint8Array(cap));
    this.fade = grow(this.fade, new Float32Array(cap));
  }
}

/** Running counts of how trains moved between consecutive frames. */
export interface MotionStats {
  /** Trains present in both frames (same train_id). */
  updates: number;
  /** …that moved along the same edge. */
  sameEdge: number;
  /** …that moved along a chain of continuing edges. */
  path: number;
  /** …whose slot was reused by another route (a new trip): cross-faded. */
  newTrip: number;
  /** …same route but no plausible path, or `frac` ran backwards: cross-faded (a "snap"). */
  snap: number;
  /** Trains not in the previous frame (faded in). */
  appeared: number;
  /** Trains of the previous frame missing from the new one (faded out). */
  vanished: number;
}

const MOTION_HOLD = 0; // no previous frame: shown where it is
const MOTION_MOVE = 1; // same train: moves along its resolved path
const MOTION_ENTER = 2; // new train or discontinuity: fades in where it is

/**
 * Holds the last two frames and their arrival times; samples interpolated
 * positions for any render time. Not reactive (lives outside React state).
 */
export class FrameBuffer {
  readonly stats: MotionStats = { updates: 0, sameEdge: 0, path: 0, newTrip: 0, snap: 0, appeared: 0, vanished: 0 };
  private prev: Frame | null = null;
  private curr: Frame | null = null;
  private currAt = 0;
  private intervalMs: number;
  /** prevIndex[train_id] = index in `prev`, −1 if absent. */
  private prevIndex = new Int32Array(1024).fill(-1);
  private readonly sampleOut: TrainSample = { edge: 0, frac: 0 };
  // Motion for the interval prev → curr, resolved lazily on the first sample.
  private resolvedFor: Frame | null = null;
  private resolvedAdj: EdgeAdjacency | null = null;
  private motion = new Uint8Array(0);
  private prevOf = new Int32Array(0);
  private paths: (TrainPath | null)[] = [];
  private dist = new Float64Array(0);
  private v0 = new Float64Array(0);
  /** Per prev train: 0 = gone, 1 = matched but discontinuous, 2 = continued. */
  private continued = new Uint8Array(0);
  private ghostIdx = new Int32Array(0);
  private ghostCount = 0;
  // Per train_id speed memory, so speed is continuous across frame boundaries.
  private memSeq = new Float64Array(1024).fill(-1);
  private memDist = new Float64Array(1024);
  private memV0 = new Float64Array(1024);
  // After a reset the last frame before it fades out and the first after it fades in.
  private retired: Frame | null = null;
  private retiredAt = 0;
  private enterAt = Number.NEGATIVE_INFINITY;

  constructor(frameHz = 4) {
    this.intervalMs = 1000 / frameHz;
  }

  /** Nominal frame rate from `welcome.constants.frame_hz` (resets the interval estimate). */
  setFrameHz(hz: number): void {
    if (hz > 0) this.intervalMs = Math.min(MAX_INTERVAL_MS, Math.max(MIN_INTERVAL_MS, 1000 / hz));
  }

  get latest(): Frame | null {
    return this.curr;
  }

  get previous(): Frame | null {
    return this.prev;
  }

  get frameIntervalMs(): number {
    return this.intervalMs;
  }

  /** Arrival time (performance.now() ms) of the latest frame. */
  get latestArrival(): number {
    return this.currAt;
  }

  /**
   * Forget both frames. With `now`, the trains on screen fade out over
   * {@link RESET_FADE_MS} while the next frame's trains fade in.
   */
  reset(now?: number): void {
    if (this.prev) this.clearIndex(this.prev);
    if (now !== undefined && this.curr) {
      this.retired = this.curr;
      this.retiredAt = now;
    }
    this.prev = null;
    this.curr = null;
    this.currAt = 0;
    this.resolvedFor = null;
  }

  /**
   * Add a decoded frame received at `now` (ms). Frames arrive in order (TCP), so
   * a sequence or clock that moves backwards — or a clock jump larger than any
   * playback speed can produce — means the sim was reset (preset, profile
   * change, new session): the buffer restarts instead of interpolating across it.
   */
  push(frame: Frame, now: number): void {
    const curr = this.curr;
    if (curr) {
      if (frame.seq === curr.seq && frame.clockS === curr.clockS) return; // duplicate
      const jump = frame.clockS - curr.clockS;
      if (frame.seq < curr.seq || jump < 0 || jump > RESET_CLOCK_JUMP_S) {
        this.reset(now);
      } else {
        const dt = now - this.currAt;
        if (dt > 0 && dt < 4 * MAX_INTERVAL_MS) {
          const clamped = Math.min(MAX_INTERVAL_MS, Math.max(MIN_INTERVAL_MS, dt));
          this.intervalMs = this.intervalMs * 0.8 + clamped * 0.2;
        }
      }
    }
    if (!this.curr) this.enterAt = now;
    if (this.prev) this.clearIndex(this.prev);
    this.prev = this.curr;
    this.curr = frame;
    this.currAt = now;
    if (this.prev) this.buildIndex(this.prev);
  }

  /** Sim clock interpolated for render time `now` (smooth countdowns), or null before any frame. */
  clockAt(now: number): number | null {
    const curr = this.curr;
    if (!curr) return null;
    const prev = this.prev;
    if (!prev) return curr.clockS;
    return prev.clockS + (curr.clockS - prev.clockS) * this.alphaAt(now);
  }

  /** Interpolation parameter for render time `now`. */
  alphaAt(now: number): number {
    if (!this.prev) return 1;
    const a = (now - this.currAt) / this.intervalMs;
    return a <= 0 ? 0 : a >= 1 ? 1 : a;
  }

  /**
   * Fill `out` with interpolated trains for render time `now`; returns the train
   * count. `ghosts`, when given, receives trains that are fading out.
   */
  sample(now: number, adj: EdgeAdjacency | null, out: InterpolatedTrains, ghosts?: InterpolatedTrains): number {
    const curr = this.curr;
    if (ghosts) ghosts.count = 0;
    if (!curr) {
      out.count = 0;
      if (ghosts) this.sampleRetired(now, ghosts);
      return 0;
    }
    this.resolve(adj);
    const prev = this.prev;
    const n = curr.nTrains;
    out.ensure(n);
    const alpha = this.alphaAt(now);
    const enterFade = fadeCurve(alpha);
    const holdFade = fadeCurve((now - this.enterAt) / RESET_FADE_MS);
    const s = this.sampleOut;
    for (let i = 0; i < n; i++) {
      const m = this.motion[i];
      const cf = fracFromWire(curr.frac[i]);
      const ce = curr.edge[i];
      if (m === MOTION_MOVE && prev) {
        const j = this.prevOf[i];
        const pe = prev.edge[j];
        const pf = fracFromWire(prev.frac[j]);
        const d = this.dist[i];
        const r = d > 0 ? easeDistance(alpha, this.v0[i], d) / d : alpha;
        const path = this.paths[i];
        if (path && adj) {
          walkPath(adj.lengthM, pe, pf, path, ce, cf, r * d, s);
          out.edge[i] = s.edge;
          out.frac[i] = s.frac;
        } else {
          out.edge[i] = ce;
          out.frac[i] = pf + (cf - pf) * r;
        }
        out.fade[i] = 1;
      } else {
        out.edge[i] = ce;
        out.frac[i] = cf;
        out.fade[i] = m === MOTION_ENTER ? enterFade : holdFade;
      }
      out.trainId[i] = curr.trainId[i];
      out.route[i] = curr.route[i];
      out.status[i] = curr.status[i];
      out.load[i] = curr.load[i];
    }
    out.count = n;
    if (ghosts) {
      if (prev && enterFade < 1) {
        ghosts.ensure(this.ghostCount);
        const fade = 1 - enterFade;
        for (let k = 0; k < this.ghostCount; k++) copyTrain(prev, this.ghostIdx[k], ghosts, ghosts.count++, fade);
      }
      this.sampleRetired(now, ghosts);
    }
    return n;
  }

  /**
   * Station crowding for render time `now`, interpolated between frames so the
   * crowding ramp animates smoothly. Closed stations (255) never blend.
   */
  sampleStationCrowd(now: number, out: Float32Array): number {
    const curr = this.curr;
    if (!curr) return 0;
    const m = Math.min(out.length, curr.nStations);
    const prev = this.prev;
    const alpha = this.alphaAt(now);
    const blend = prev !== null && prev.nStations === curr.nStations;
    for (let i = 0; i < m; i++) {
      const c = curr.stationCrowd[i];
      if (!blend || c === 255 || prev.stationCrowd[i] === 255) out[i] = c;
      else out[i] = prev.stationCrowd[i] + (c - prev.stationCrowd[i]) * alpha;
    }
    return m;
  }

  private sampleRetired(now: number, ghosts: InterpolatedTrains): void {
    const old = this.retired;
    if (!old) return;
    const t = (now - this.retiredAt) / RESET_FADE_MS;
    if (t >= 1) {
      this.retired = null;
      return;
    }
    const fade = 1 - fadeCurve(t);
    ghosts.ensure(ghosts.count + old.nTrains);
    for (let j = 0; j < old.nTrains; j++) copyTrain(old, j, ghosts, ghosts.count++, fade);
  }

  /** Classify every train of the interval prev → curr once (paths, distances, start speeds). */
  private resolve(adj: EdgeAdjacency | null): void {
    const curr = this.curr;
    if (!curr || (this.resolvedFor === curr && this.resolvedAdj === adj)) return;
    const counting = this.resolvedFor !== curr;
    this.resolvedFor = curr;
    this.resolvedAdj = adj;
    const n = curr.nTrains;
    if (this.motion.length < n) {
      const cap = Math.max(n, this.motion.length * 2, 64);
      this.motion = new Uint8Array(cap);
      this.prevOf = new Int32Array(cap);
      this.dist = new Float64Array(cap);
      this.v0 = new Float64Array(cap);
    }
    this.paths.length = n;
    this.ghostCount = 0;
    const prev = this.prev;
    if (!prev) {
      this.motion.fill(MOTION_HOLD, 0, n);
      this.paths.fill(null);
      return;
    }
    if (this.continued.length < prev.nTrains) this.continued = new Uint8Array(Math.max(prev.nTrains, 64));
    else this.continued.fill(0, 0, prev.nTrains);
    const finder = adj ? pathFinderFor(adj) : null;
    const lengthM = adj ? adj.lengthM : null;
    const maxDist = plausibleDistanceM(curr.clockS - prev.clockS);
    const st = this.stats;
    for (let i = 0; i < n; i++) {
      const id = curr.trainId[i];
      const j = id < this.prevIndex.length ? this.prevIndex[id] : -1;
      this.prevOf[i] = j;
      this.paths[i] = null;
      this.motion[i] = MOTION_ENTER;
      if (j < 0) {
        if (counting) st.appeared++;
        this.forgetSpeed(id);
        continue;
      }
      this.continued[j] = 1;
      if (counting) st.updates++;
      if (prev.route[j] !== curr.route[i]) {
        if (counting) st.newTrip++;
        this.forgetSpeed(id);
        continue;
      }
      const pe = prev.edge[j];
      const ce = curr.edge[i];
      const pf = fracFromWire(prev.frac[j]);
      const cf = fracFromWire(curr.frac[i]);
      let d = -1;
      let path: TrainPath | null = null;
      if (pe === ce) {
        if (cf + 1e-6 >= pf) d = lengthM && pe < lengthM.length ? (cf - pf) * lengthM[pe] : 0;
      } else if (finder && lengthM) {
        path = finder.find(pe, ce, curr.route[i]);
        if (path) d = pathDistanceM(lengthM, pe, pf, path, ce, cf);
      }
      if (d < 0 || d > maxDist) {
        if (counting) st.snap++;
        this.forgetSpeed(id);
        continue;
      }
      if (counting) {
        if (pe === ce) st.sameEdge++;
        else st.path++;
      }
      this.motion[i] = MOTION_MOVE;
      this.paths[i] = path;
      this.dist[i] = d;
      this.continued[j] = 2;
      this.v0[i] = this.startSpeed(id, prev.seq, curr.seq, d);
    }
    if (this.ghostIdx.length < prev.nTrains) this.ghostIdx = new Int32Array(Math.max(prev.nTrains, 64));
    for (let j = 0; j < prev.nTrains; j++) {
      const c = this.continued[j];
      if (c === 2) continue;
      this.ghostIdx[this.ghostCount++] = j;
      if (counting && c === 0) st.vanished++;
    }
  }

  /** Start speed for this interval: the previous interval's mean speed if the train kept moving. */
  private startSpeed(id: number, prevSeq: number, currSeq: number, d: number): number {
    if (id >= this.memSeq.length) {
      const cap = Math.max(id + 1, this.memSeq.length * 2);
      const seq = new Float64Array(cap).fill(-1);
      seq.set(this.memSeq);
      const dist = new Float64Array(cap);
      dist.set(this.memDist);
      const v0 = new Float64Array(cap);
      v0.set(this.memV0);
      this.memSeq = seq;
      this.memDist = dist;
      this.memV0 = v0;
    }
    let v = d;
    if (this.memSeq[id] === prevSeq) v = this.memDist[id];
    else if (this.memSeq[id] === currSeq) v = this.memV0[id]; // same interval resolved again
    this.memSeq[id] = currSeq;
    this.memDist[id] = d;
    this.memV0[id] = v;
    return v;
  }

  private forgetSpeed(id: number): void {
    if (id < this.memSeq.length) this.memSeq[id] = -1;
  }

  private buildIndex(frame: Frame): void {
    let maxId = 0;
    for (let i = 0; i < frame.nTrains; i++) if (frame.trainId[i] > maxId) maxId = frame.trainId[i];
    if (maxId >= this.prevIndex.length) {
      const next = new Int32Array(Math.max(maxId + 1, this.prevIndex.length * 2)).fill(-1);
      this.prevIndex = next;
    }
    for (let i = 0; i < frame.nTrains; i++) this.prevIndex[frame.trainId[i]] = i;
  }

  private clearIndex(frame: Frame): void {
    for (let i = 0; i < frame.nTrains; i++) {
      const id = frame.trainId[i];
      if (id < this.prevIndex.length) this.prevIndex[id] = -1;
    }
  }
}

function copyTrain(src: Frame, j: number, dst: InterpolatedTrains, k: number, fade: number): void {
  dst.trainId[k] = src.trainId[j];
  dst.edge[k] = src.edge[j];
  dst.frac[k] = fracFromWire(src.frac[j]);
  dst.route[k] = src.route[j];
  dst.status[k] = src.status[j];
  dst.load[k] = src.load[j];
  dst.fade[k] = fade;
}
