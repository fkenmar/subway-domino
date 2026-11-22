/**
 * Deep links to a map view: `#map=<m/px>/<lat>/<lon>` (the OpenStreetMap
 * `zoom/lat/lon` convention, with the scale in metres per CSS pixel). The page
 * opens centred on that point instead of the citywide fit, which makes views
 * shareable and lets visual checks frame a district reproducibly.
 */
import type { CameraState } from "./camera";
import { type Projection, projectLatLon } from "./network";

export interface MapView {
  mPerPx: number;
  lat: number;
  lon: number;
}

const HASH_RE = /(?:^#|&)map=([^&]+)/;

/** Parse `#map=m/lat/lon` (other hash parameters are ignored). Returns null when absent or invalid. */
export function parseViewHash(hash: string): MapView | null {
  const m = HASH_RE.exec(hash.startsWith("#") ? hash : `#${hash}`);
  if (!m) return null;
  const parts = decodeURIComponent(m[1]).split("/").map(Number);
  if (parts.length !== 3 || !parts.every(Number.isFinite)) return null;
  const [mPerPx, lat, lon] = parts;
  if (mPerPx <= 0 || lat < -90 || lat > 90 || lon < -180 || lon > 180) return null;
  return { mPerPx, lat, lon };
}

/** Format a view as a `#map=` hash (5 decimals ≈ 1 m). */
export function formatViewHash(view: MapView): string {
  const scale = Number(view.mPerPx.toPrecision(3));
  return `#map=${scale}/${view.lat.toFixed(5)}/${view.lon.toFixed(5)}`;
}

/** Camera centre (world metres) and scale for a parsed view. */
export function viewToCamera(view: MapView, projection: Projection): CameraState {
  const p = projectLatLon(projection, view.lat, view.lon);
  return { cx: p.x, cy: p.y, mPerPx: view.mPerPx };
}
