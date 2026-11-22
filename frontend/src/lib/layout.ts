/**
 * Screen regions covered by HUD chrome. Camera fits and fly-tos centre content
 * inside the uncovered area. The App measures the real HUD layout (see
 * `App.tsx`) and publishes it with {@link setMeasuredInsets}; until then the
 * static per-breakpoint estimates in {@link mapInsets} are used.
 */
import type { CameraState } from "./camera";
import type { Bbox } from "./geometry";

export interface Insets {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export const BREAKPOINT_DESKTOP = 1100;
export const BREAKPOINT_MOBILE = 700;

/** Static estimates per breakpoint (used before the HUD has been measured). */
export function mapInsets(width: number, _height: number): Insets {
  if (width >= BREAKPOINT_DESKTOP) return { top: 136, right: 400, bottom: 100, left: 24 };
  if (width >= BREAKPOINT_MOBILE) return { top: 136, right: 24, bottom: 200, left: 24 };
  return { top: 160, right: 10, bottom: 230, left: 10 };
}

let measured: { insets: Insets; width: number; height: number } | null = null;

/** Publish the measured HUD insets for a viewport of `width` × `height` CSS px (null clears). */
export function setMeasuredInsets(insets: Insets | null, width = 0, height = 0): void {
  measured = insets ? { insets, width, height } : null;
}

/**
 * Insets for the current viewport: the measured layout when it was measured for
 * this viewport size, otherwise the static estimate.
 */
export function currentInsets(width: number, height: number): Insets {
  if (measured && Math.abs(measured.width - width) < 2 && Math.abs(measured.height - height) < 2) {
    return measured.insets;
  }
  return mapInsets(width, height);
}

/**
 * Sanitise measured insets so the uncovered area never collapses: each axis
 * keeps at least `minFree` CSS px (or 35 % of the viewport) uncovered.
 */
export function clampInsets(insets: Insets, width: number, height: number, minFree = 160): Insets {
  const fix = (a: number, b: number, size: number): [number, number] => {
    const lo = Math.max(0, a);
    const hi = Math.max(0, b);
    const free = Math.min(size, Math.max(minFree, size * 0.35));
    const excess = lo + hi - (size - free);
    if (excess <= 0) return [lo, hi];
    // Shrink both sides proportionally.
    const k = (size - free) / Math.max(1, lo + hi);
    return [lo * k, hi * k];
  };
  const [left, right] = fix(insets.left, insets.right, width);
  const [top, bottom] = fix(insets.top, insets.bottom, height);
  return { top, right, bottom, left };
}

/** Screen rectangle (CSS px, viewport coordinates). */
export interface ScreenRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

let occluders: { rects: ScreenRect[]; version: number } = { rects: [], version: 0 };

/** Publish the HUD rectangles that cover the map (labels avoid them). */
export function setOccluders(rects: ScreenRect[]): void {
  const same =
    rects.length === occluders.rects.length &&
    rects.every((r, i) => {
      const o = occluders.rects[i];
      return Math.abs(r.x0 - o.x0) < 1 && Math.abs(r.y0 - o.y0) < 1 && Math.abs(r.x1 - o.x1) < 1 && Math.abs(r.y1 - o.y1) < 1;
    });
  if (!same) occluders = { rects, version: occluders.version + 1 };
}

/** Current HUD occluders; `version` changes whenever they move. */
export function currentOccluders(): { rects: readonly ScreenRect[]; version: number } {
  return occluders;
}

/** Camera centre that puts world point (x, y) at the centre of the uncovered area. */
export function centerWithInsets(x: number, y: number, mPerPx: number, insets: Insets): CameraState {
  const ox = (insets.left - insets.right) / 2;
  const oy = (insets.top - insets.bottom) / 2;
  return { cx: x - ox * mPerPx, cy: y + oy * mPerPx, mPerPx };
}

/** Camera view that fits `bbox` inside the uncovered area. */
export function fitWithInsets(bbox: Bbox, width: number, height: number, insets: Insets, padPx = 12): CameraState {
  const w = Math.max(40, width - insets.left - insets.right - 2 * padPx);
  const h = Math.max(40, height - insets.top - insets.bottom - 2 * padPx);
  const mPerPx = Math.max((bbox[2] - bbox[0]) / w, (bbox[3] - bbox[1]) / h, 1e-3);
  return centerWithInsets((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, mPerPx, insets);
}

/**
 * Home view of the whole network. The floating scoreboard covers only the top
 * of the side column, so the network is centred across the full width (between
 * the top bar and the bottom controls) whenever no station would land under a
 * HUD panel; otherwise it slides toward the side-inset fit until it clears.
 * Without measured panels it falls back to the side-inset fit.
 */
export function homeView(
  bbox: Bbox,
  points: readonly { x: number; y: number }[],
  width: number,
  height: number,
  insets: Insets,
  rects: readonly ScreenRect[],
  padPx = 12,
): CameraState {
  const inset = fitWithInsets(bbox, width, height, insets, padPx);
  if (rects.length === 0 || insets.right <= insets.left) return inset;
  const full = fitWithInsets(bbox, width, height, { ...insets, right: insets.left }, padPx);
  const margin = 12;
  const clear = (v: CameraState): boolean => {
    for (const p of points) {
      const sx = (p.x - v.cx) / v.mPerPx + width / 2;
      const sy = (v.cy - p.y) / v.mPerPx + height / 2;
      for (const r of rects) {
        if (sx > r.x0 - margin && sx < r.x1 + margin && sy > r.y0 - margin && sy < r.y1 + margin) return false;
      }
    }
    return true;
  };
  for (const t of [0, 0.25, 0.5, 0.75]) {
    const v: CameraState = {
      cx: full.cx + (inset.cx - full.cx) * t,
      cy: full.cy + (inset.cy - full.cy) * t,
      mPerPx: full.mPerPx + (inset.mPerPx - full.mPerPx) * t,
    };
    if (clear(v)) return v;
  }
  return inset;
}
