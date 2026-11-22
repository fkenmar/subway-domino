/**
 * Planar geometry helpers in the local metric projection (§2.6: x east, y north,
 * metres). Polylines are arc-length parameterised for the §2.7 positioning rule.
 */

export interface Polyline {
  xs: Float64Array;
  ys: Float64Array;
  /** cum[i] = arc length from the first vertex to vertex i (cum[0] = 0). */
  cum: Float64Array;
  length: number;
}

export interface PointOnLine {
  x: number;
  y: number;
  /** Unit tangent in the polyline's own (first → last vertex) direction. (0,0) if degenerate. */
  tx: number;
  ty: number;
}

export function makePolyline(pts: ReadonlyArray<readonly [number, number] | readonly number[]>): Polyline {
  const n = pts.length;
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  const cum = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const p = pts[i];
    xs[i] = Number(p[0]);
    ys[i] = Number(p[1]);
    if (i > 0) cum[i] = cum[i - 1] + Math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]);
  }
  return { xs, ys, cum, length: n > 0 ? cum[n - 1] : 0 };
}

/** Index i of the polyline piece [i, i+1] containing arc length s (binary search). */
export function pieceAt(cum: Float64Array, s: number): number {
  const n = cum.length;
  if (n < 2) return 0;
  let lo = 0;
  let hi = n - 2;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (cum[mid] <= s) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/**
 * Point at normalised arc length t ∈ [0,1] (clamped), with the local unit tangent.
 * Zero-length pieces are skipped so the tangent is always meaningful when the
 * polyline has any length.
 */
export function pointAt(line: Polyline, t: number, out: PointOnLine): PointOnLine {
  const n = line.xs.length;
  if (n === 0) {
    out.x = 0;
    out.y = 0;
    out.tx = 0;
    out.ty = 0;
    return out;
  }
  if (n === 1 || line.length <= 0) {
    out.x = line.xs[0];
    out.y = line.ys[0];
    out.tx = 0;
    out.ty = 0;
    return out;
  }
  const tc = t <= 0 ? 0 : t >= 1 ? 1 : t;
  const s = tc * line.length;
  let i = pieceAt(line.cum, s);
  // Skip degenerate (zero-length) pieces for a stable tangent.
  while (i < n - 2 && line.cum[i + 1] - line.cum[i] <= 1e-9) i++;
  while (i > 0 && line.cum[i + 1] - line.cum[i] <= 1e-9) i--;
  const x0 = line.xs[i];
  const y0 = line.ys[i];
  const x1 = line.xs[i + 1];
  const y1 = line.ys[i + 1];
  const segLen = line.cum[i + 1] - line.cum[i];
  const u = segLen > 0 ? Math.min(1, Math.max(0, (s - line.cum[i]) / segLen)) : 0;
  out.x = x0 + (x1 - x0) * u;
  out.y = y0 + (y1 - y0) * u;
  if (segLen > 0) {
    out.tx = (x1 - x0) / segLen;
    out.ty = (y1 - y0) / segLen;
  } else {
    out.tx = 0;
    out.ty = 0;
  }
  return out;
}

/** Squared distance from point p to segment [a, b]. */
export function distSqToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax;
  const dy = by - ay;
  const len2 = dx * dx + dy * dy;
  let u = len2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
  u = u < 0 ? 0 : u > 1 ? 1 : u;
  const qx = ax + dx * u - px;
  const qy = ay + dy * u - py;
  return qx * qx + qy * qy;
}

/** Minimum distance from a point to a polyline, with the arc-length fraction of the closest point. */
export function nearestOnPolyline(line: Polyline, px: number, py: number): { dist: number; t: number } {
  const n = line.xs.length;
  if (n === 0) return { dist: Number.POSITIVE_INFINITY, t: 0 };
  if (n === 1) return { dist: Math.hypot(px - line.xs[0], py - line.ys[0]), t: 0 };
  let best = Number.POSITIVE_INFINITY;
  let bestS = 0;
  for (let i = 0; i < n - 1; i++) {
    const ax = line.xs[i];
    const ay = line.ys[i];
    const bx = line.xs[i + 1];
    const by = line.ys[i + 1];
    const d2 = distSqToSegment(px, py, ax, ay, bx, by);
    if (d2 < best) {
      best = d2;
      const dx = bx - ax;
      const dy = by - ay;
      const len2 = dx * dx + dy * dy;
      const u = len2 > 0 ? Math.min(1, Math.max(0, ((px - ax) * dx + (py - ay) * dy) / len2)) : 0;
      bestS = line.cum[i] + u * Math.sqrt(len2);
    }
  }
  return { dist: Math.sqrt(best), t: line.length > 0 ? bestS / line.length : 0 };
}

export type Bbox = readonly [number, number, number, number];

export function bboxOfPoints(xs: ArrayLike<number>, ys: ArrayLike<number>): [number, number, number, number] {
  let x0 = Number.POSITIVE_INFINITY;
  let y0 = Number.POSITIVE_INFINITY;
  let x1 = Number.NEGATIVE_INFINITY;
  let y1 = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i];
    const y = ys[i];
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  return [x0, y0, x1, y1];
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
