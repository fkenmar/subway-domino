/**
 * Frame interpolation (§6 rendering — CONTRACT behaviour).
 *
 * Frames arrive at FRAME_HZ (4 Hz); the map renders at 60 fps one frame behind:
 * positions are interpolated between the last two frames with
 * `alpha = (now − arrival(curr)) / interval`, clamped to [0, 1].
 *
 * Per train (matched by train_id):
 * - same edge → lerp `frac` (contract);
 * - different edge → snap to the new frame (contract) — except when the new edge
 *   *directly continues* the old one (the old edge's to-station is the new edge's
 *   from-station, derived from the geo file). Then the train moves along that
 *   two-edge path at constant speed instead of jumping (see WP3 progress notes:
 *   contract clarification request). Unrelated edges always snap.
 */
import type { Frame } from "./protocol";
import { fracFromWire } from "./positioning";

/** Topology needed to recognise a continuing edge pair (from `Network.edges`). */
export interface EdgeAdjacency {
  count: number;
  fromStation: Int32Array;
  toStation: Int32Array;
  lengthM: Float64Array;
}

export interface TrainSample {
  edge: number;
  frac: number;
}

const MIN_INTERVAL_MS = 100;
const MAX_INTERVAL_MS = 1000;
/**
 * Largest plausible sim-clock advance between consecutive frames: 5x speed is
 * 300 sim-s/s → 75 sim-s per 4 Hz frame, capped at 8 ticks × 10 s = 80 s by the
 * server. Anything beyond this is a reset/jump, not motion.
 */
export const RESET_CLOCK_JUMP_S = 240;

/** True if `next` starts at the station where `prev` ends (a physical continuation). */
export function edgesContinue(adj: EdgeAdjacency, prev: number, next: number): boolean {
  if (prev < 0 || next < 0 || prev >= adj.count || next >= adj.count) return false;
  const at = adj.toStation[prev];
  return at >= 0 && at === adj.fromStation[next];
}

/**
 * Interpolate one train between two frames. `prevFrac` / `currFrac` are
 * normalised (0..1); `alpha` ∈ [0,1]. Pure; writes into `out`.
 */
export function interpolateTrain(
  prevEdge: number,
  prevFrac: number,
  currEdge: number,
  currFrac: number,
  alpha: number,
  adj: EdgeAdjacency | null,
  out: TrainSample,
): TrainSample {
  const a = alpha <= 0 ? 0 : alpha >= 1 ? 1 : alpha;
  if (prevEdge === currEdge) {
    // Directed edges never run backwards; a backwards jump means a new trip in a reused slot.
    if (currFrac + 1e-6 < prevFrac) {
      out.edge = currEdge;
      out.frac = currFrac;
      return out;
    }
    out.edge = currEdge;
    out.frac = prevFrac + (currFrac - prevFrac) * a;
    return out;
  }
  if (adj && a < 1 && edgesContinue(adj, prevEdge, currEdge)) {
    const l1 = adj.lengthM[prevEdge];
    const l2 = adj.lengthM[currEdge];
    const d1 = (1 - prevFrac) * l1;
    const d2 = currFrac * l2;
    const total = d1 + d2;
    if (total > 0) {
      const s = a * total;
      if (s < d1 && l1 > 0) {
        out.edge = prevEdge;
        out.frac = prevFrac + s / l1;
        return out;
      }
      if (l2 > 0) {
        out.edge = currEdge;
        out.frac = (s - d1) / l2;
        return out;
      }
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

  constructor(capacity = 1024) {
    this.trainId = new Uint16Array(capacity);
    this.edge = new Int32Array(capacity);
    this.frac = new Float32Array(capacity);
    this.route = new Uint8Array(capacity);
    this.status = new Uint8Array(capacity);
    this.load = new Uint8Array(capacity);
  }

  ensure(capacity: number): void {
    if (capacity <= this.trainId.length) return;
    const cap = Math.max(capacity, this.trainId.length * 2);
    this.trainId = new Uint16Array(cap);
    this.edge = new Int32Array(cap);
    this.frac = new Float32Array(cap);
    this.route = new Uint8Array(cap);
    this.status = new Uint8Array(cap);
    this.load = new Uint8Array(cap);
  }
}

/**
 * Holds the last two frames and their arrival times; samples interpolated
 * positions for any render time. Not reactive (lives outside React state).
 */
export class FrameBuffer {
  private prev: Frame | null = null;
  private curr: Frame | null = null;
  private currAt = 0;
  private intervalMs: number;
  /** prevIndex[train_id] = index in `prev`, −1 if absent. */
  private prevIndex = new Int32Array(1024).fill(-1);
  private readonly sampleOut: TrainSample = { edge: 0, frac: 0 };

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

  reset(): void {
    if (this.prev) this.clearIndex(this.prev);
    this.prev = null;
    this.curr = null;
    this.currAt = 0;
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
        this.reset();
      } else {
        const dt = now - this.currAt;
        if (dt > 0 && dt < 4 * MAX_INTERVAL_MS) {
          const clamped = Math.min(MAX_INTERVAL_MS, Math.max(MIN_INTERVAL_MS, dt));
          this.intervalMs = this.intervalMs * 0.8 + clamped * 0.2;
        }
      }
    }
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

  /** Fill `out` with interpolated trains for render time `now`; returns the train count. */
  sample(now: number, adj: EdgeAdjacency | null, out: InterpolatedTrains): number {
    const curr = this.curr;
    if (!curr) {
      out.count = 0;
      return 0;
    }
    const prev = this.prev;
    const n = curr.nTrains;
    out.ensure(n);
    const alpha = this.alphaAt(now);
    const s = this.sampleOut;
    for (let i = 0; i < n; i++) {
      const id = curr.trainId[i];
      const edge = curr.edge[i];
      const frac = fracFromWire(curr.frac[i]);
      const j = prev && id < this.prevIndex.length ? this.prevIndex[id] : -1;
      if (prev && j >= 0 && prev.route[j] === curr.route[i]) {
        interpolateTrain(prev.edge[j], fracFromWire(prev.frac[j]), edge, frac, alpha, adj, s);
        out.edge[i] = s.edge;
        out.frac[i] = s.frac;
      } else {
        out.edge[i] = edge;
        out.frac[i] = frac;
      }
      out.trainId[i] = id;
      out.route[i] = curr.route[i];
      out.status[i] = curr.status[i];
      out.load[i] = curr.load[i];
    }
    out.count = n;
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
