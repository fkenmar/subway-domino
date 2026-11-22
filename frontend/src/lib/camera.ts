/**
 * 2D map camera over the §2.6 metric projection.
 *
 * World: metres, x east, y north. Screen: CSS pixels, y down. The camera is the
 * world point at the viewport centre plus a scale in metres per CSS pixel.
 * Zoom runs from full NYC (≈ 60 m/px) to dense Midtown (≈ 1.5 m/px).
 */
import { type Bbox, clamp } from "./geometry";

export interface CameraState {
  cx: number;
  cy: number;
  /** Metres per CSS pixel (larger = zoomed out). */
  mPerPx: number;
}

/** Closest zoom (a little past the 1.5 m/px Midtown target, for pinch slack). */
export const MIN_M_PER_PX = 0.9;
/** Hard floor for the widest zoom (the fit-to-city scale may require more on phones). */
export const MAX_M_PER_PX_FLOOR = 70;

const RHO = Math.SQRT2;

/** Nearest drawn content (world metres) to a point, searched within `radiusM` (then anywhere). */
export type ContentProbe = (x: number, y: number, radiusM: number) => { x: number; y: number; d: number } | null;

/** Some network must stay within this fraction of the smaller viewport side from the centre. */
export const CONTENT_REACH = 0.4;

/**
 * van Wijk & Nuij "smooth and efficient zooming and panning" path between two
 * views (the algorithm behind d3.interpolateZoom). Views are [cx, cy, w] with
 * `w` the viewport width in world units. Returns the interpolator and its
 * natural duration in ms.
 */
export function zoomPath(
  from: readonly [number, number, number],
  to: readonly [number, number, number],
): { at: (t: number) => [number, number, number]; durationMs: number } {
  const [ux0, uy0, w0] = from;
  const [ux1, uy1, w1] = to;
  const dx = ux1 - ux0;
  const dy = uy1 - uy0;
  const d2 = dx * dx + dy * dy;
  if (d2 < 1e-12) {
    const S = Math.log(w1 / w0) / RHO;
    return {
      at: (t) => [ux0 + t * dx, uy0 + t * dy, w0 * Math.exp(RHO * t * S)],
      durationMs: Math.abs(S) * 1000,
    };
  }
  const d1 = Math.sqrt(d2);
  const b0 = (w1 * w1 - w0 * w0 + RHO ** 4 * d2) / (2 * w0 * RHO ** 2 * d1);
  const b1 = (w1 * w1 - w0 * w0 - RHO ** 4 * d2) / (2 * w1 * RHO ** 2 * d1);
  const r0 = Math.log(Math.sqrt(b0 * b0 + 1) - b0);
  const r1 = Math.log(Math.sqrt(b1 * b1 + 1) - b1);
  const S = (r1 - r0) / RHO;
  const coshR0 = Math.cosh(r0);
  const sinhR0 = Math.sinh(r0);
  return {
    at: (t) => {
      const s = t * S;
      const u = (w0 / (RHO ** 2 * d1)) * (coshR0 * Math.tanh(RHO * s + r0) - sinhR0);
      return [ux0 + u * dx, uy0 + u * dy, (w0 * coshR0) / Math.cosh(RHO * s + r0)];
    },
    durationMs: Math.abs(S) * 1000,
  };
}

export function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
}

interface Flight {
  start: number;
  duration: number;
  at: (t: number) => [number, number, number];
  width: number;
}

export class Camera implements CameraState {
  cx = 0;
  cy = 0;
  mPerPx = 40;
  width = 1;
  height = 1;
  minMPerPx = MIN_M_PER_PX;
  maxMPerPx = MAX_M_PER_PX_FLOOR;
  /** Increments on every change; consumers compare it to detect motion. */
  version = 0;
  private flight: Flight | null = null;
  private bounds: Bbox | null = null;
  private probe: ContentProbe | null = null;

  /** Keep part of the network on screen at every zoom (see {@link CONTENT_REACH}). */
  setContentProbe(probe: ContentProbe | null): void {
    this.probe = probe;
    this.constrain();
  }

  /** Viewport size in CSS pixels. */
  setViewport(width: number, height: number): void {
    const w = Math.max(1, width);
    const h = Math.max(1, height);
    if (w === this.width && h === this.height) return;
    this.width = w;
    this.height = h;
    this.version++;
  }

  /** Constrain the centre to `bbox` (padded) so the network can't be lost off-screen. */
  setBounds(bbox: Bbox | null): void {
    this.bounds = bbox;
    this.constrain();
  }

  get state(): CameraState {
    return { cx: this.cx, cy: this.cy, mPerPx: this.mPerPx };
  }

  get animating(): boolean {
    return this.flight !== null;
  }

  toScreenX(x: number): number {
    return (x - this.cx) / this.mPerPx + this.width / 2;
  }

  toScreenY(y: number): number {
    return (this.cy - y) / this.mPerPx + this.height / 2;
  }

  toWorldX(sx: number): number {
    return this.cx + (sx - this.width / 2) * this.mPerPx;
  }

  toWorldY(sy: number): number {
    return this.cy - (sy - this.height / 2) * this.mPerPx;
  }

  /** World bbox currently visible (optionally padded by `padPx` CSS px). */
  visibleWorld(padPx = 0): [number, number, number, number] {
    return [
      this.toWorldX(-padPx),
      this.toWorldY(this.height + padPx),
      this.toWorldX(this.width + padPx),
      this.toWorldY(-padPx),
    ];
  }

  set(state: CameraState): void {
    this.cx = state.cx;
    this.cy = state.cy;
    this.mPerPx = clamp(state.mPerPx, this.minMPerPx, this.maxMPerPx);
    this.constrain();
    this.version++;
  }

  /** Pan by a screen-space delta (drag): the content follows the pointer. */
  panBy(dxPx: number, dyPx: number): void {
    if (dxPx === 0 && dyPx === 0) return;
    this.flight = null;
    this.cx -= dxPx * this.mPerPx;
    this.cy += dyPx * this.mPerPx;
    this.constrain();
    this.version++;
  }

  /** Zoom by `factor` (> 1 zooms in) keeping the world point under (sx, sy) fixed. */
  zoomAt(sx: number, sy: number, factor: number): void {
    if (!(factor > 0) || factor === 1) return;
    this.flight = null;
    const wx = this.toWorldX(sx);
    const wy = this.toWorldY(sy);
    let next = clamp(this.mPerPx / factor, this.minMPerPx, this.maxMPerPx);
    if (factor > 1) next = this.limitZoomIn(wx, wy, sx, sy, next);
    if (next === this.mPerPx) return;
    this.mPerPx = next;
    // Re-anchor: the same world point must map back to (sx, sy).
    this.cx = wx - (sx - this.width / 2) * this.mPerPx;
    this.cy = wy + (sy - this.height / 2) * this.mPerPx;
    this.constrain();
    this.version++;
  }

  /** Scale that fits `bbox` inside the viewport with `padPx` padding. */
  fitScale(bbox: Bbox, padPx = 24): number {
    const w = Math.max(1, this.width - 2 * padPx);
    const h = Math.max(1, this.height - 2 * padPx);
    return Math.max((bbox[2] - bbox[0]) / w, (bbox[3] - bbox[1]) / h);
  }

  /** Configure zoom limits for a network extent: the widest zoom shows the whole city. */
  configureForExtent(bbox: Bbox): void {
    this.maxMPerPx = Math.max(MAX_M_PER_PX_FLOOR, this.fitScale(bbox, 8) * 1.35);
    this.mPerPx = clamp(this.mPerPx, this.minMPerPx, this.maxMPerPx);
    this.version++;
  }

  fitBbox(bbox: Bbox, padPx = 24): void {
    this.set({
      cx: (bbox[0] + bbox[2]) / 2,
      cy: (bbox[1] + bbox[3]) / 2,
      mPerPx: this.fitScale(bbox, padPx),
    });
  }

  /** Animate to `target` along a smooth zoom-pan path. */
  flyTo(target: CameraState, now: number, opts: { maxDurationMs?: number; minDurationMs?: number } = {}): void {
    const to = { ...target, mPerPx: clamp(target.mPerPx, this.minMPerPx, this.maxMPerPx) };
    const path = zoomPath([this.cx, this.cy, this.mPerPx * this.width], [to.cx, to.cy, to.mPerPx * this.width]);
    const duration = clamp(path.durationMs * 0.9, opts.minDurationMs ?? 450, opts.maxDurationMs ?? 1800);
    this.flight = { start: now, duration, at: path.at, width: this.width };
  }

  cancelFlight(): void {
    this.flight = null;
  }

  /** Advance an in-flight animation; returns true if the camera moved. */
  step(now: number): boolean {
    const f = this.flight;
    if (!f) return false;
    const t = clamp((now - f.start) / f.duration, 0, 1);
    const [x, y, w] = f.at(easeInOutCubic(t));
    this.cx = x;
    this.cy = y;
    this.mPerPx = clamp(w / f.width, this.minMPerPx, this.maxMPerPx);
    if (t >= 1) this.flight = null;
    this.constrain();
    this.version++;
    return true;
  }

  /**
   * Deepest scale (≥ `target`, ≤ the current one) for a zoom about world point
   * (wx, wy) at screen (sx, sy) that still keeps network content near the centre.
   */
  private limitZoomIn(wx: number, wy: number, sx: number, sy: number, target: number): number {
    const probe = this.probe;
    if (!probe) return target;
    const side = CONTENT_REACH * Math.min(this.width, this.height);
    let m = target;
    for (let i = 0; i < 6; i++) {
      const cx = wx - (sx - this.width / 2) * m;
      const cy = wy + (sy - this.height / 2) * m;
      const near = probe(cx, cy, side * m);
      if (!near || near.d <= side * m * 1.001) break;
      m = near.d / side;
    }
    return Math.min(Math.max(m, target), this.mPerPx);
  }

  private constrain(): void {
    const b = this.bounds;
    if (b) {
      // Allow the centre to roam a little past the network edge, never off into the void.
      const padX = Math.max(2000, (this.width * this.mPerPx) / 3);
      const padY = Math.max(2000, (this.height * this.mPerPx) / 3);
      this.cx = clamp(this.cx, b[0] - padX, b[2] + padX);
      this.cy = clamp(this.cy, b[1] - padY, b[3] + padY);
    }
    // Never an empty screen: pull the centre back towards the nearest line.
    const probe = this.probe;
    if (!probe || this.width <= 1) return;
    const reach = CONTENT_REACH * Math.min(this.width, this.height) * this.mPerPx;
    const near = probe(this.cx, this.cy, reach);
    if (!near || near.d <= reach) return;
    const k = (near.d - reach) / near.d;
    this.cx += (near.x - this.cx) * k;
    this.cy += (near.y - this.cy) * k;
  }
}
