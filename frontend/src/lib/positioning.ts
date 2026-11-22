/**
 * §2.7 positioning rule (CONTRACT, shared by server and client). A train on edge
 * `e` at `frac ∈ [0,1]` is located:
 *
 * - on a `run` edge at `polyline(segment).point_at(seg_dir == 1 ? frac : 1 − frac)`
 *   (arc-length parameterised), plus a lateral screen-space offset by (seg_dir, track);
 * - on a crossover or turnback edge at the station point;
 * - on a yard lead on the straight line from the yard to the lead station.
 *
 * The world-space part lives here; the screen-space lateral offset is applied by
 * the renderer from {@link Pose.lane} (so it stays constant in pixels).
 */
import { pointAt, type PointOnLine } from "./geometry";
import {
  EDGE_CROSSOVER,
  EDGE_RUN,
  EDGE_TURNBACK,
  EDGE_YARD_LEAD,
  type Network,
  TRACK_X,
} from "./network";

export const FRAC_SCALE = 65535;

export interface Pose {
  /** World position (metres, §2.6 projection). */
  x: number;
  y: number;
  /** Unit direction of travel in world space; (0,0) when undefined (crossover/turnback). */
  dx: number;
  dy: number;
  /**
   * Lateral lane, in units of the renderer's lane width, to the right of the
   * direction of travel (NYC runs on the right). 0 = on the centre line.
   * Express (X) tracks sit inside, local (L) tracks outside.
   */
  lane: number;
}

export function makePose(): Pose {
  return { x: 0, y: 0, dx: 0, dy: 0, lane: 0 };
}

/**
 * Lateral lane by (track, has_express): 4-track trunks put the express (X)
 * tracks inside and the local (L) tracks outside; 2-track lines use the inner lane.
 */
export function laneFor(track: number, segmentHasExpress: boolean): number {
  if (track === TRACK_X) return 0.6;
  return segmentHasExpress ? 1.8 : 0.6;
}

const scratch: PointOnLine = { x: 0, y: 0, tx: 0, ty: 0 };

/**
 * Locate a train on `edge` at normalised `frac` (0..1). Returns false (and leaves
 * `out` untouched) if the edge is unknown or cannot be placed.
 */
export function locate(net: Network, edge: number, frac: number, out: Pose): boolean {
  const E = net.edges;
  if (edge < 0 || edge >= E.count) return false;
  const kind = E.kind[edge];
  const f = frac <= 0 ? 0 : frac >= 1 ? 1 : frac;

  if (kind === EDGE_RUN) {
    const seg = net.segments[E.segment[edge]];
    if (!seg) return false;
    const dir = E.segDir[edge];
    pointAt(seg.line, dir === 1 ? f : 1 - f, scratch);
    out.x = scratch.x;
    out.y = scratch.y;
    out.dx = scratch.tx * dir;
    out.dy = scratch.ty * dir;
    out.lane = laneFor(E.track[edge], seg.has_express || net.segmentHasXEdge[seg.idx] === 1);
    return true;
  }

  if (kind === EDGE_CROSSOVER || kind === EDGE_TURNBACK) {
    const st = net.stations[E.station[edge]];
    if (!st) return false;
    out.x = st.x;
    out.y = st.y;
    out.dx = 0;
    out.dy = 0;
    out.lane = 0;
    return true;
  }

  if (kind === EDGE_YARD_LEAD) {
    const yard = net.yards[E.yard[edge]];
    if (!yard) return false;
    const st = net.stations[E.station[edge]];
    if (!st) {
      // Lead station unknown: the best we can do is the yard itself.
      out.x = yard.x;
      out.y = yard.y;
      out.dx = 0;
      out.dy = 0;
      out.lane = 0;
      return true;
    }
    // Straight line yard → station (pull-out) or station → yard (put-in, seg_dir −1).
    const putIn = E.segDir[edge] === -1;
    const ax = putIn ? st.x : yard.x;
    const ay = putIn ? st.y : yard.y;
    const bx = putIn ? yard.x : st.x;
    const by = putIn ? yard.y : st.y;
    const len = Math.hypot(bx - ax, by - ay);
    out.x = ax + (bx - ax) * f;
    out.y = ay + (by - ay) * f;
    out.dx = len > 0 ? (bx - ax) / len : 0;
    out.dy = len > 0 ? (by - ay) / len : 0;
    out.lane = 0;
    return true;
  }
  return false;
}

/** Wire `frac` (u16) → normalised fraction. */
export function fracFromWire(v: number): number {
  return v / FRAC_SCALE;
}
