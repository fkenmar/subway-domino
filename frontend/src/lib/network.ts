/**
 * Loads `network_geo.json` (§2.7 — CONTRACT) and builds the lookup tables the
 * renderer, hit-tester, interpolator and search need. Indices are identical to
 * `network.json`, so frames (edge / route / station indices) index straight in.
 */
import { asset, backendGeoUrl, GEO_PATH, WS_URL } from "./config";
import { type Bbox, makePolyline, type Polyline } from "./geometry";
import { type DisruptionKind, type DisruptionTarget, isProfile, type Profile } from "./protocol";

// ─── Raw schema ──────────────────────────────────────────────────────────────

export interface GeoRoute {
  idx: number;
  id: string;
  name: string;
  color: string;
  text_color: string;
  division: string;
}

export interface GeoComplex {
  idx: number;
  name: string;
  x: number;
  y: number;
  stations: number[];
  weekday_entries: number;
}

export interface GeoStation {
  idx: number;
  gtfs: string;
  name: string;
  complex: number;
  x: number;
  y: number;
  routes: number[];
  hub: boolean;
  /** 0 = most important (by entries). */
  label_rank: number;
}

export interface GeoSegment {
  idx: number;
  a: number;
  b: number;
  name: string;
  river: boolean;
  has_express: boolean;
  routes: number[];
  /** Polyline a → b. */
  pts: [number, number][];
}

export type EdgeKindName = "run" | "crossover" | "turnback" | "yard_lead";

export interface GeoEdge {
  idx: number;
  kind: EdgeKindName;
  segment: number;
  seg_dir: number;
  track: string;
  /** Station for crossover/turnback (and the lead station of a yard lead, when present). */
  station: number;
  /** Yard for yard_lead. */
  yard: number;
}

export interface GeoInterlocking {
  idx: number;
  station: number;
  x: number;
  y: number;
  kinds: string[];
}

export interface GeoYard {
  idx: number;
  name: string;
  x: number;
  y: number;
  division: string;
}

export interface GeoPresetDisruption {
  kind: DisruptionKind;
  target: DisruptionTarget;
  duration_s: number | null;
}

export interface GeoPreset {
  id: string;
  title: string;
  subtitle: string;
  profile: Profile;
  disruptions: GeoPresetDisruption[];
  focus?: { x: number; y: number; zoom_m_per_px: number };
}

export interface GeoFile {
  schema_version: number;
  graph_version: string;
  projection?: Projection;
  bbox: [number, number, number, number];
  routes: GeoRoute[];
  complexes: GeoComplex[];
  stations: GeoStation[];
  segments: GeoSegment[];
  edges: GeoEdge[];
  interlockings: GeoInterlocking[];
  yards: GeoYard[];
  presets: GeoPreset[];
  basemap: { land: [number, number][][] } | null;
}

export const GEO_SCHEMA_VERSION = 1;

// ─── Derived model ───────────────────────────────────────────────────────────

export const EDGE_RUN = 0;
export const EDGE_CROSSOVER = 1;
export const EDGE_TURNBACK = 2;
export const EDGE_YARD_LEAD = 3;
export const EDGE_UNKNOWN = 255;

export const TRACK_L = 0;
export const TRACK_X = 1;

const EDGE_KIND_CODE: Record<string, number> = {
  run: EDGE_RUN,
  crossover: EDGE_CROSSOVER,
  turnback: EDGE_TURNBACK,
  yard_lead: EDGE_YARD_LEAD,
};

export interface RouteInfo extends GeoRoute {
  /** Official colour, normalised to `#RRGGBB`. */
  color: string;
  textColor: string;
  /** Colour used for strokes on the dark map (contrast-adjusted official colour). */
  mapColor: string;
  /** Text inside a bullet: `6X` → `6`, `GS` → `S`. */
  bullet: string;
  /** Express variants (6X, 7X, FX) are drawn as diamonds. */
  diamond: boolean;
}

export interface SegmentInfo extends GeoSegment {
  line: Polyline;
  /** Distinct map colours of the routes on this segment, in a global canonical order. */
  colors: string[];
  /** +1 if the bundle should be offset along the polyline as given, −1 if reversed (bundle continuity). */
  bundleDir: 1 | -1;
}

export interface EdgeTable {
  count: number;
  kind: Uint8Array;
  segment: Int32Array;
  segDir: Int8Array;
  track: Uint8Array;
  station: Int32Array;
  yard: Int32Array;
  /** Station the edge leaves (−1 = a yard). */
  fromStation: Int32Array;
  /** Station the edge enters (−1 = a yard). */
  toStation: Int32Array;
  /** Geometric length in metres (0 for crossovers/turnbacks). */
  lengthM: Float64Array;
}

export interface Projection {
  lat0: number;
  lon0: number;
  kx: number;
  ky: number;
}

/** §2.6 local equirectangular projection (used when the geo file omits it). */
export const DEFAULT_PROJECTION: Projection = { lat0: 40.7128, lon0: -73.95, kx: 84384.0, ky: 111132.954 };

/** Project WGS84 lat/lon into the network's metric frame (§2.6). */
export function projectLatLon(p: Projection, lat: number, lon: number): { x: number; y: number } {
  return { x: (lon - p.lon0) * p.kx, y: (lat - p.lat0) * p.ky };
}

export interface Network {
  graphVersion: string;
  projection: Projection;
  bbox: Bbox;
  routes: RouteInfo[];
  routeById: Map<string, number>;
  complexes: GeoComplex[];
  stations: GeoStation[];
  segments: SegmentInfo[];
  edges: EdgeTable;
  interlockings: GeoInterlocking[];
  yards: GeoYard[];
  presets: GeoPreset[];
  land: Polyline[];
  /** Segments incident to each station. */
  stationSegments: number[][];
  /** Interlocking index per station, −1 if none. */
  stationInterlocking: Int32Array;
  /** 1 if the station has an express (X) track through it. */
  stationHasExpress: Uint8Array;
  /** 1 if any X run edge lies on the segment. */
  segmentHasXEdge: Uint8Array;
  /** Representative (lowest label_rank) station per complex. */
  complexRep: Int32Array;
  /** Route-colour set per station (map colours, canonical order) for markers. */
  stationColors: string[][];
}

export class GeoValidationError extends Error {
  override name = "GeoValidationError";
}

// ─── Colour helpers ──────────────────────────────────────────────────────────

/** Map background the line colours must stand out against (kept in sync with style.ts). */
export const MAP_BACKGROUND = "#0a0f17";

export function normalizeHex(color: string, fallback = "#808183"): string {
  const c = color.trim().replace(/^#/, "");
  if (/^[0-9a-f]{6}$/i.test(c)) return `#${c.toUpperCase()}`;
  if (/^[0-9a-f]{3}$/i.test(c)) return `#${c.replace(/./g, (ch) => ch + ch).toUpperCase()}`;
  return fallback;
}

function hexToRgb(hex: string): [number, number, number] {
  const h = normalizeHex(hex).slice(1);
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

function rgbToHex(r: number, g: number, b: number): string {
  const to = (v: number) => Math.round(Math.min(255, Math.max(0, v))).toString(16).padStart(2, "0");
  return `#${to(r)}${to(g)}${to(b)}`.toUpperCase();
}

export function relativeLuminance(hex: string): number {
  const lin = (v: number) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  const [r, g, b] = hexToRgb(hex);
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

export function contrastRatio(a: string, b: string): number {
  const la = relativeLuminance(a);
  const lb = relativeLuminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

/** Lighten `color` towards white until it reaches `minRatio` contrast against `bg`. */
export function ensureContrast(color: string, bg: string, minRatio: number): string {
  const [r, g, b] = hexToRgb(color);
  let out = normalizeHex(color);
  for (let k = 0; k <= 20 && contrastRatio(out, bg) < minRatio; k++) {
    const t = k / 20;
    out = rgbToHex(r + (255 - r) * t, g + (255 - g) * t, b + (255 - b) * t);
  }
  return out;
}

/** Bullet text for a route id: express variants and shuttles follow MTA signage. */
export function routeBullet(id: string): { text: string; diamond: boolean } {
  if (id === "GS" || id === "FS" || id === "H") return { text: "S", diamond: false };
  if (id.length === 2 && id.endsWith("X")) return { text: id[0] ?? id, diamond: true };
  return { text: id, diamond: false };
}

// ─── Validation ──────────────────────────────────────────────────────────────

function req<T>(cond: T, msg: string): asserts cond {
  if (!cond) throw new GeoValidationError(msg);
}

/** Structural validation of the raw JSON (fail fast with a readable message). */
export function validateGeo(raw: unknown): GeoFile {
  req(typeof raw === "object" && raw !== null, "geo file is not an object");
  const g = raw as Partial<GeoFile> & Record<string, unknown>;
  req(g.schema_version === GEO_SCHEMA_VERSION, `unsupported geo schema_version ${String(g.schema_version)}`);
  for (const key of ["routes", "complexes", "stations", "segments", "edges", "interlockings", "yards", "presets"] as const) {
    req(Array.isArray(g[key]), `geo file is missing array "${key}"`);
  }
  req(Array.isArray(g.bbox) && g.bbox.length === 4, "geo file has no bbox");
  const geo = g as GeoFile;
  geo.stations.forEach((s, i) => req(s.idx === i, `stations[${i}].idx = ${s.idx}`));
  geo.segments.forEach((s, i) => {
    req(s.idx === i, `segments[${i}].idx = ${s.idx}`);
    req(Array.isArray(s.pts) && s.pts.length >= 1, `segments[${i}] has no points`);
  });
  geo.edges.forEach((e, i) => req(e.idx === i, `edges[${i}].idx = ${e.idx}`));
  geo.routes.forEach((r, i) => req(r.idx === i, `routes[${i}].idx = ${r.idx}`));
  if (geo.basemap === undefined) geo.basemap = null;
  geo.presets = geo.presets.filter((p) => typeof p.id === "string" && isProfile(p.profile) && Array.isArray(p.disruptions));
  return geo;
}

// ─── Build ───────────────────────────────────────────────────────────────────

/** Minimum contrast of map line colours against the map background. */
const MIN_LINE_CONTRAST = 4.2;

function chordOrientation(line: Polyline): 1 | -1 {
  const n = line.xs.length;
  if (n < 2) return 1;
  const dx = line.xs[n - 1] - line.xs[0];
  const dy = line.ys[n - 1] - line.ys[0];
  // Canonical: mostly-vertical lines point north, mostly-horizontal lines point east.
  if (Math.abs(dy) >= Math.abs(dx)) return dy >= 0 ? 1 : -1;
  return dx >= 0 ? 1 : -1;
}

/**
 * Orient segment bundles so adjacent segments sharing colours continue in the
 * same sense through their shared station (prevents colour bundles from
 * "twisting" at stations). BFS over the segment adjacency, seeded canonically.
 */
function orientBundles(segments: Omit<SegmentInfo, "bundleDir">[], stationSegments: number[][]): Int8Array {
  const dir = new Int8Array(segments.length);
  const order = segments.map((_, i) => i).sort((a, b) => segments[b].line.length - segments[a].line.length);
  const queue: number[] = [];
  for (const seed of order) {
    if (dir[seed] !== 0) continue;
    dir[seed] = chordOrientation(segments[seed].line);
    queue.push(seed);
    while (queue.length) {
      const cur = queue.shift() as number;
      const seg = segments[cur];
      const start = dir[cur] === 1 ? seg.a : seg.b;
      const end = dir[cur] === 1 ? seg.b : seg.a;
      for (const [station, curEndsHere] of [
        [end, true],
        [start, false],
      ] as const) {
        for (const nb of stationSegments[station] ?? []) {
          if (dir[nb] !== 0) continue;
          const other = segments[nb];
          if (!other.colors.some((c) => seg.colors.includes(c))) continue;
          // Continue through the station: if cur ends here, the neighbour starts here.
          const nbStartsAtA = other.a === station;
          dir[nb] = curEndsHere === nbStartsAtA ? 1 : -1;
          queue.push(nb);
        }
      }
    }
  }
  return dir;
}

/** Nearest station to a yard that is served by a route of the yard's division (−1 if none). */
export function nearestLeadStation(geo: GeoFile, yardIdx: number, routes: readonly { division: string }[]): number {
  const yard = geo.yards[yardIdx];
  if (!yard) return -1;
  let best = -1;
  let bestD = Number.POSITIVE_INFINITY;
  for (const st of geo.stations) {
    const sameDivision = !yard.division || (st.routes ?? []).some((r) => routes[r]?.division === yard.division);
    if (!sameDivision) continue;
    const d = Math.hypot(st.x - yard.x, st.y - yard.y);
    if (d < bestD) {
      bestD = d;
      best = st.idx;
    }
  }
  return best;
}

export function buildNetwork(geo: GeoFile): Network {
  const routes: RouteInfo[] = geo.routes.map((r) => {
    const color = normalizeHex(r.color);
    const b = routeBullet(r.id);
    return {
      ...r,
      color,
      textColor: normalizeHex(r.text_color, "#FFFFFF"),
      mapColor: ensureContrast(color, MAP_BACKGROUND, MIN_LINE_CONTRAST),
      bullet: b.text,
      diamond: b.diamond,
    };
  });
  const routeById = new Map(routes.map((r) => [r.id, r.idx]));

  // Global canonical colour order: by the lowest route index using the colour.
  const colorRank = new Map<string, number>();
  routes.forEach((r) => {
    if (!colorRank.has(r.mapColor)) colorRank.set(r.mapColor, colorRank.size);
  });
  const colorsOf = (routeIdx: readonly number[]): string[] => {
    const set = new Set<string>();
    for (const ri of routeIdx) {
      const r = routes[ri];
      if (r) set.add(r.mapColor);
    }
    return [...set].sort((a, b) => (colorRank.get(a) ?? 0) - (colorRank.get(b) ?? 0));
  };

  const nStations = geo.stations.length;
  const stationSegments: number[][] = Array.from({ length: nStations }, () => []);
  const baseSegments = geo.segments.map((s) => {
    const line = makePolyline(s.pts);
    if (s.a >= 0 && s.a < nStations) stationSegments[s.a].push(s.idx);
    if (s.b >= 0 && s.b < nStations && s.b !== s.a) stationSegments[s.b].push(s.idx);
    return { ...s, line, colors: colorsOf(s.routes ?? []) };
  });
  const bundleDirs = orientBundles(baseSegments, stationSegments);
  const segments: SegmentInfo[] = baseSegments.map((s, i) => ({ ...s, bundleDir: bundleDirs[i] === -1 ? -1 : 1 }));

  // Edge table.
  const nE = geo.edges.length;
  const edges: EdgeTable = {
    count: nE,
    kind: new Uint8Array(nE),
    segment: new Int32Array(nE),
    segDir: new Int8Array(nE),
    track: new Uint8Array(nE),
    station: new Int32Array(nE),
    yard: new Int32Array(nE),
    fromStation: new Int32Array(nE),
    toStation: new Int32Array(nE),
    lengthM: new Float64Array(nE),
  };
  const segmentHasXEdge = new Uint8Array(segments.length);
  const leadCache = new Map<number, number>();
  for (let i = 0; i < nE; i++) {
    const e = geo.edges[i];
    let kind = EDGE_KIND_CODE[e.kind] ?? EDGE_UNKNOWN;
    const seg = e.segment;
    const segDir = e.seg_dir > 0 ? 1 : e.seg_dir < 0 ? -1 : 0;
    if (kind === EDGE_RUN && (seg < 0 || seg >= segments.length || segDir === 0)) kind = EDGE_UNKNOWN;
    edges.kind[i] = kind;
    edges.segment[i] = seg;
    edges.segDir[i] = segDir;
    edges.track[i] = e.track === "X" ? TRACK_X : TRACK_L;
    edges.station[i] = typeof e.station === "number" && e.station < nStations ? e.station : -1;
    edges.yard[i] = typeof e.yard === "number" && e.yard < geo.yards.length ? e.yard : -1;
    if (kind === EDGE_RUN) {
      const s = segments[seg];
      edges.fromStation[i] = segDir === 1 ? s.a : s.b;
      edges.toStation[i] = segDir === 1 ? s.b : s.a;
      edges.lengthM[i] = s.line.length;
      if (edges.track[i] === TRACK_X) segmentHasXEdge[seg] = 1;
    } else if (kind === EDGE_CROSSOVER || kind === EDGE_TURNBACK) {
      edges.fromStation[i] = edges.station[i];
      edges.toStation[i] = edges.station[i];
      edges.lengthM[i] = 0;
    } else if (kind === EDGE_YARD_LEAD) {
      const yardIdx = edges.yard[i];
      // §2.7 only guarantees `station` on crossover/turnback edges; when a yard lead
      // omits it, use the nearest station served by the yard's division.
      if (edges.station[i] < 0 && yardIdx >= 0) {
        let lead = leadCache.get(yardIdx);
        if (lead === undefined) {
          lead = nearestLeadStation(geo, yardIdx, routes);
          leadCache.set(yardIdx, lead);
        }
        edges.station[i] = lead;
      }
      const st = edges.station[i];
      const yard = geo.yards[yardIdx];
      // seg_dir −1 = put-in (station → yard); otherwise pull-out (yard → station).
      edges.fromStation[i] = segDir === -1 ? st : -1;
      edges.toStation[i] = segDir === -1 ? -1 : st;
      edges.lengthM[i] = yard && st >= 0 ? Math.hypot(geo.stations[st].x - yard.x, geo.stations[st].y - yard.y) : 0;
    } else {
      edges.fromStation[i] = -1;
      edges.toStation[i] = -1;
    }
  }

  const stationHasExpress = new Uint8Array(nStations);
  for (let s = 0; s < nStations; s++) {
    for (const seg of stationSegments[s]) {
      if (segmentHasXEdge[seg] || segments[seg].has_express) {
        stationHasExpress[s] = 1;
        break;
      }
    }
  }

  const stationInterlocking = new Int32Array(nStations).fill(-1);
  for (const il of geo.interlockings) {
    if (il.station >= 0 && il.station < nStations) stationInterlocking[il.station] = il.idx;
  }

  const complexRep = new Int32Array(geo.complexes.length).fill(-1);
  for (const c of geo.complexes) {
    let best = -1;
    for (const s of c.stations) {
      const st = geo.stations[s];
      if (st && (best < 0 || st.label_rank < geo.stations[best].label_rank)) best = s;
    }
    complexRep[c.idx] = best;
  }

  const stationColors = geo.stations.map((s) => colorsOf(s.routes ?? []));
  const land = (geo.basemap?.land ?? []).filter((ring) => ring.length >= 3).map((ring) => makePolyline(ring));

  const p = geo.projection;
  const projection =
    p && [p.lat0, p.lon0, p.kx, p.ky].every((v) => typeof v === "number" && Number.isFinite(v)) ? p : DEFAULT_PROJECTION;

  return {
    graphVersion: geo.graph_version,
    projection,
    bbox: geo.bbox,
    routes,
    routeById,
    complexes: geo.complexes,
    stations: geo.stations,
    segments,
    edges,
    interlockings: geo.interlockings,
    yards: geo.yards,
    presets: geo.presets,
    land,
    stationSegments,
    stationInterlocking,
    stationHasExpress,
    segmentHasXEdge,
    complexRep,
    stationColors,
  };
}

// ─── Loading ─────────────────────────────────────────────────────────────────

export interface LoadOptions {
  fetchImpl?: typeof fetch;
  signal?: AbortSignal;
  /** Candidate URLs tried in order (defaults: the static asset, then the backend's copy). */
  urls?: string[];
}

export function defaultGeoUrls(): string[] {
  const urls = [asset(GEO_PATH)];
  const fallback = backendGeoUrl(WS_URL);
  if (fallback) urls.push(fallback);
  return urls;
}

/**
 * Fetch and build the network. Tries the static asset first, then the backend's
 * `GET /network_geo.json` copy (§5.1), so a missing static file degrades gracefully.
 */
export async function loadNetwork(options: LoadOptions = {}): Promise<Network> {
  const fetchImpl = options.fetchImpl ?? fetch;
  const urls = options.urls ?? defaultGeoUrls();
  let lastError: unknown = new Error("no geo URL configured");
  for (const url of urls) {
    try {
      const res = await fetchImpl(url, { signal: options.signal });
      if (!res.ok) throw new Error(`GET ${url} → HTTP ${res.status}`);
      const raw: unknown = await res.json();
      return buildNetwork(validateGeo(raw));
    } catch (err) {
      if (options.signal?.aborted) throw err;
      lastError = err;
    }
  }
  throw lastError;
}
