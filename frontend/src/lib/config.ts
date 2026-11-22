/**
 * Deployment configuration shared by every module.
 *
 * The app is served under `basePath: "/subway"` (see next.config.ts). Next's
 * <Link> and next/image add the base path automatically, but raw `fetch()` and
 * `<img>` URLs do not — every `public/` URL must go through {@link asset}.
 */

/** Must match `basePath` in next.config.ts. */
export const BASE_PATH = "/subway";

/** Public path (relative to the base path) of the geometry file built by WP1 (§2.7). */
export const GEO_PATH = "/data/network_geo.json";

/** Backend used by `npm run dev` when NEXT_PUBLIC_WS_URL is not set. */
export const DEFAULT_DEV_WS_URL = "ws://localhost:8000/ws";

const ABSOLUTE_URL = /^(?:[a-z][a-z\d+.-]*:|\/\/)/i;

/**
 * Prefix a `public/` asset path with the base path.
 *
 * - `asset("/data/network_geo.json")` → `"/subway/data/network_geo.json"`
 * - relative paths are treated as rooted: `asset("icon.svg")` → `"/subway/icon.svg"`
 * - idempotent: an already-prefixed path is returned unchanged
 * - absolute URLs (`https:`, `data:`, `blob:`, protocol-relative) are returned unchanged
 */
export function asset(path: string): string {
  if (ABSOLUTE_URL.test(path)) return path;
  const rooted = path.startsWith("/") ? path : `/${path}`;
  if (rooted === BASE_PATH || rooted.startsWith(`${BASE_PATH}/`)) return rooted;
  return `${BASE_PATH}${rooted}`;
}

/**
 * Resolve the WebSocket endpoint.
 *
 * Production builds *require* `NEXT_PUBLIC_WS_URL` (browsers connect to Railway
 * directly over `wss://`, never through a Vercel rewrite). Development falls back
 * to {@link DEFAULT_DEV_WS_URL}. Returns `null` when no usable URL exists; the UI
 * then shows a configuration banner instead of silently connecting elsewhere.
 */
export function resolveWsUrl(envValue: string | undefined, nodeEnv: string | undefined): string | null {
  const value = envValue?.trim();
  if (value) {
    return /^wss?:\/\//i.test(value) ? value : null;
  }
  return nodeEnv === "production" ? null : DEFAULT_DEV_WS_URL;
}

/** The configured backend WebSocket URL (inlined at build time by Next). */
export const WS_URL: string | null = resolveWsUrl(process.env.NEXT_PUBLIC_WS_URL, process.env.NODE_ENV);

/**
 * The backend serves a copy of the geometry file at `GET /network_geo.json`
 * (§5.1, fallback). Derive its HTTP URL from the WebSocket URL.
 */
export function backendGeoUrl(wsUrl: string | null): string | null {
  if (!wsUrl) return null;
  try {
    const url = new URL(wsUrl);
    url.protocol = url.protocol === "wss:" ? "https:" : "http:";
    url.pathname = url.pathname.replace(/\/ws\/?$/, "") + "/network_geo.json";
    url.search = "";
    url.hash = "";
    return url.toString();
  } catch {
    return null;
  }
}
