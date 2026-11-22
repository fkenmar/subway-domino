/**
 * Map styling: a light palette after the printed NYC subway map, and zoom tiers. Every size is
 * in CSS (screen) pixels so the map stays crisp at every zoom (§6).
 */
import { MAP_BACKGROUND } from "@/lib/network";

export const PALETTE = {
  background: MAP_BACKGROUND,
  /** Printed-map basemap: cream land, pale-blue water. */
  water: MAP_BACKGROUND,
  land: "#f3eee4",
  landEdge: "#cfc5b2",
  /** Thin light gaps between parallel lines, with a faint warm hairline outside the bundle. */
  casing: "#fcfaf5",
  casingEdge: "rgba(92, 78, 52, 0.2)",
  riverCasing: "rgba(52, 104, 140, 0.4)",
  /** Local stops: black dots. */
  stationFill: "#151515",
  stationStroke: "#ffffff",
  /** Express stops and interchanges: white circles with black rims. */
  hubFill: "#ffffff",
  hubStroke: "#151515",
  /** Transfer connectors: white with black outlines. */
  transfer: "#ffffff",
  label: "#111111",
  labelMuted: "#3b3b3b",
  labelHalo: "rgba(250, 247, 240, 0.94)",
  borough: "#6f6a62",
  interlocking: "#3e5578",
  yard: "#857d6e",
  yardLead: "rgba(110, 100, 84, 0.5)",
  yardLabel: "#5f594f",
  calm: "#ffffff",
  amber: "#e89a00",
  red: "#d7261e",
  closed: "#111111",
  closedX: "#ffffff",
  block: "#d7261e",
  signal: "#e89a00",
  selection: "#111111",
  /** The AI dispatcher's accent on the light map (5.4:1 on the land). */
  highlight: "#0b5cc4",
  /** Held: a white ring with a black outline (independent of the route colour). */
  heldRing: "#ffffff",
  heldEdge: "#111111",
  /** "AI just acted" flash: black inside a white ring. */
  aiRing: "#111111",
  stuckHalo: "rgba(215, 38, 30, 0.5)",
  gap: "#14853b",
  /** Train pill outline and drop shadow on the light map. */
  trainEdge: "rgba(17, 17, 17, 0.88)",
  trainShadow: "rgba(40, 30, 12, 1)",
  glyph: "#111111",
  chipBg: "rgba(17, 17, 17, 0.96)",
  chipText: "#ffffff",
} as const;

export const FONT_STACK = '"Helvetica Neue", Helvetica, Arial, "Nimbus Sans", sans-serif';

/**
 * Borough names drawn under the lines on wide tiers so first-time visitors can
 * orient themselves. Anchors sit where no line runs at the city and borough
 * tiers, so the letters are never chopped up by route strokes and stations.
 */
export const BOROUGH_LABELS: readonly { name: string; lat: number; lon: number; angleDeg?: number }[] = [
  // Manhattan is solid with lines: the name runs up the Hudson along the west shore,
  // parallel to the island (its street grid runs ~29° east of north).
  { name: "MANHATTAN", lat: 40.787, lon: -73.9915, angleDeg: -61 },
  // Marine Park / Mill Basin: between the Brighton line and Canarsie.
  { name: "BROOKLYN", lat: 40.612, lon: -73.905 },
  { name: "QUEENS", lat: 40.7265, lon: -73.7935 },
  // Castle Hill / Throgs Neck, south of the Pelham line.
  { name: "THE BRONX", lat: 40.818, lon: -73.835 },
];

export interface ZoomTier {
  id: number;
  name: string;
  /** Tier applies while mPerPx >= minMPerPx (tiers are ordered widest first). */
  minMPerPx: number;
  /** Width of one route-colour stroke. */
  lineWidth: number;
  /** Gap between adjacent strokes in a bundle. */
  lineGap: number;
  casingExtra: number;
  stationRadius: number;
  hubRadius: number;
  stationStroke: number;
  /** Non-hub station markers are drawn from this tier on (hubs always). */
  showLocalStations: boolean;
  /** Stations whose complex label_rank is below this are labelled. */
  labelMaxRank: number;
  /**
   * Also label the other stations of a complex whose names differ from its main
   * station (e.g. 42 St-Port Authority beside Times Sq-42 St). Dense tiers only.
   */
  labelAllStations: boolean;
  /**
   * Fill pass: after the rank-limited pass, any further station may be labelled
   * where its label keeps at least this clearance (px) from every placed label,
   * so sparse outer-borough lines get names while dense Manhattan stays legible.
   * 0 disables the pass.
   */
  labelFillGap: number;
  /** Fill-pass candidates are limited to label_rank below this. */
  labelFillMaxRank: number;
  /** Fill pass skips bare numbered streets ("72 St", "86 St"): repeated citywide, they don't orient. */
  labelFillNamedOnly: boolean;
  labelSize: number;
  hubLabelSize: number;
  /** Station markers block label placement (dense tiers only). */
  labelsAvoidMarkers: boolean;
  /** Borough names for orientation (wide tiers). */
  showBoroughs: boolean;
  showInterlockings: boolean;
  showYards: boolean;
  /** Train glyph: dots on wide tiers, capsules (length × width) from `capsules` on. */
  capsules: boolean;
  trainLength: number;
  trainWidth: number;
  /** Lateral distance per lane unit (see positioning.laneFor). */
  laneWidth: number;
  /** Route lines are mixed this far (1 = full colour) from the land colour. Full colour at every tier (printed-map ribbons). */
  lineAlpha: number;
  /** Route lines are desaturated this far (0 = official hue) before mixing: wide tiers mute the track so trains separate from it. */
  lineDesat: number;
  showTrainLabels: boolean;
  showStatusIcons: boolean;
}

/** Widest → closest. Thresholds in metres per CSS pixel. */
export const ZOOM_TIERS: readonly ZoomTier[] = [
  {
    id: 0,
    name: "city",
    minMPerPx: 40,
    lineWidth: 3.6,
    lineGap: 0.9,
    casingExtra: 1.1,
    stationRadius: 1.2,
    hubRadius: 2.6,
    stationStroke: 0.7,
    showLocalStations: false,
    labelMaxRank: 10,
    labelAllStations: false,
    labelFillGap: 40,
    labelFillMaxRank: 60,
    labelFillNamedOnly: true,
    labelSize: 11,
    hubLabelSize: 11.5,
    labelsAvoidMarkers: false,
    showBoroughs: true,
    showInterlockings: false,
    showYards: false,
    capsules: true,
    trainLength: 8,
    trainWidth: 3.6,
    // Citywide beads stay inside the ribbon in both directions.
    laneWidth: 0.9,
    lineAlpha: 1,
    lineDesat: 0,
    showTrainLabels: false,
    showStatusIcons: false,
  },
  {
    id: 1,
    name: "borough",
    minMPerPx: 18,
    lineWidth: 4,
    lineGap: 1,
    casingExtra: 1.3,
    stationRadius: 1.5,
    hubRadius: 3.1,
    stationStroke: 0.8,
    showLocalStations: true,
    labelMaxRank: 34,
    labelAllStations: false,
    labelFillGap: 28,
    labelFillMaxRank: Number.POSITIVE_INFINITY,
    labelFillNamedOnly: false,
    labelSize: 11,
    hubLabelSize: 11.5,
    labelsAvoidMarkers: false,
    showBoroughs: true,
    showInterlockings: false,
    showYards: true,
    capsules: true,
    trainLength: 10.5,
    trainWidth: 4.4,
    laneWidth: 1.9,
    lineAlpha: 1,
    lineDesat: 0,
    showTrainLabels: false,
    showStatusIcons: false,
  },
  {
    id: 2,
    name: "district",
    minMPerPx: 8,
    lineWidth: 4.6,
    lineGap: 1.2,
    casingExtra: 1.5,
    stationRadius: 2.3,
    hubRadius: 4.2,
    stationStroke: 1.0,
    showLocalStations: true,
    labelMaxRank: 100,
    labelAllStations: false,
    labelFillGap: 14,
    labelFillMaxRank: Number.POSITIVE_INFINITY,
    labelFillNamedOnly: false,
    labelSize: 11.5,
    hubLabelSize: 12,
    labelsAvoidMarkers: true,
    showBoroughs: false,
    showInterlockings: true,
    showYards: true,
    capsules: true,
    trainLength: 14,
    trainWidth: 5.6,
    laneWidth: 3.6,
    lineAlpha: 1,
    lineDesat: 0,
    showTrainLabels: false,
    showStatusIcons: true,
  },
  {
    id: 3,
    name: "neighborhood",
    minMPerPx: 3.2,
    lineWidth: 5.8,
    lineGap: 1.5,
    casingExtra: 1.8,
    stationRadius: 3.2,
    hubRadius: 5.4,
    stationStroke: 1.3,
    showLocalStations: true,
    labelMaxRank: 280,
    labelAllStations: true,
    labelFillGap: 6,
    labelFillMaxRank: Number.POSITIVE_INFINITY,
    labelFillNamedOnly: false,
    labelSize: 12,
    hubLabelSize: 13,
    labelsAvoidMarkers: true,
    showBoroughs: false,
    showInterlockings: true,
    showYards: true,
    capsules: true,
    trainLength: 19,
    trainWidth: 7.6,
    laneWidth: 5.8,
    lineAlpha: 1,
    lineDesat: 0,
    showTrainLabels: true,
    showStatusIcons: true,
  },
  {
    id: 4,
    name: "street",
    minMPerPx: 0,
    lineWidth: 7.2,
    lineGap: 1.7,
    casingExtra: 2,
    stationRadius: 4.2,
    hubRadius: 6.8,
    stationStroke: 1.6,
    showLocalStations: true,
    labelMaxRank: Number.POSITIVE_INFINITY,
    labelAllStations: true,
    labelFillGap: 0,
    labelFillMaxRank: Number.POSITIVE_INFINITY,
    labelFillNamedOnly: false,
    labelSize: 12.5,
    hubLabelSize: 14,
    labelsAvoidMarkers: true,
    showBoroughs: false,
    showInterlockings: true,
    showYards: true,
    capsules: true,
    trainLength: 25,
    trainWidth: 9.6,
    laneWidth: 7.6,
    lineAlpha: 1,
    lineDesat: 0,
    showTrainLabels: true,
    showStatusIcons: true,
  },
];

export function tierFor(mPerPx: number): ZoomTier {
  for (const t of ZOOM_TIERS) if (mPerPx >= t.minMPerPx) return t;
  return ZOOM_TIERS[ZOOM_TIERS.length - 1];
}

/** Pill size anchors (m/px → length × width, CSS px); sizes are interpolated in log scale between them. */
const TRAIN_SIZE_ANCHORS: readonly [number, number, number][] = [
  [60, 8, 3.6],
  [25, 10.5, 4.4],
  [11, 14, 6],
  // From the neighbourhood tier (route letters on) pills are at least 16 × 8 CSS px,
  // so the letter stays readable on 1× displays.
  [8, 16.5, 8],
  [4.5, 20, 9],
  [1.6, 25, 10.5],
];

/**
 * Train pill size for a zoom: at least 8 px long at the citywide fit, growing
 * smoothly with zoom (no jumps at tier boundaries).
 */
export function trainSize(mPerPx: number): { length: number; width: number } {
  const a = TRAIN_SIZE_ANCHORS;
  if (mPerPx >= a[0][0]) return { length: a[0][1], width: a[0][2] };
  for (let i = 1; i < a.length; i++) {
    if (mPerPx >= a[i][0]) {
      const t = Math.log(a[i - 1][0] / mPerPx) / Math.log(a[i - 1][0] / a[i][0]);
      return { length: a[i - 1][1] + (a[i][1] - a[i - 1][1]) * t, width: a[i - 1][2] + (a[i][2] - a[i - 1][2]) * t };
    }
  }
  const last = a[a.length - 1];
  return { length: last[1], width: last[2] };
}

/** Pull a `#rrggbb` colour towards its own luminance grey (`t` = 1 → fully grey). */
export function desaturate(hex: string, t: number): string {
  if (t <= 0) return hex;
  const p = parseInt(hex.replace("#", ""), 16);
  const r = (p >> 16) & 255;
  const g = (p >> 8) & 255;
  const b = p & 255;
  const l = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  const grey = `#${((l << 16) | (l << 8) | l).toString(16).padStart(6, "0")}`;
  return mixHex(hex, grey, t);
}

/** Mix two `#rrggbb` colours: `t` = 0 → a, 1 → b. */
export function mixHex(a: string, b: string, t: number): string {
  const pa = parseInt(a.replace("#", ""), 16);
  const pb = parseInt(b.replace("#", ""), 16);
  const ch = (shift: number) => {
    const x = (pa >> shift) & 255;
    const y = (pb >> shift) & 255;
    return Math.round(x + (y - x) * t);
  };
  return `#${((ch(16) << 16) | (ch(8) << 8) | ch(0)).toString(16).padStart(6, "0")}`;
}

/** Total width of a segment bundle with `n` colours. */
export function bundleWidth(tier: ZoomTier, n: number): number {
  return n * tier.lineWidth + Math.max(0, n - 1) * tier.lineGap;
}

/** Offset (from the centre line) of colour `i` of `n` in a bundle. */
export function bundleOffset(tier: ZoomTier, i: number, n: number): number {
  const pitch = tier.lineWidth + tier.lineGap;
  return (i - (n - 1) / 2) * pitch;
}

/** Crowding below this (% of platform capacity) reads as calm: the base marker shows. */
export const CROWD_CALM = 40;
/** At or above this the marker is red and pulses (§6, OVERCROWD_FRAC = 0.80). */
export const CROWD_ALERT = 80;

/**
 * Crowding ramp (§6): calm → amber → red from `station_crowd` (100 = at capacity).
 * Returns null while calm (the base marker shows through).
 */
export function crowdColor(crowd: number): string | null {
  if (crowd < CROWD_CALM) return null;
  if (crowd >= CROWD_ALERT) return PALETTE.red;
  const t = (crowd - CROWD_CALM) / (CROWD_ALERT - CROWD_CALM); // 0 → amber, 1 → orange-red
  const g = Math.round(196 - (196 - 96) * t);
  const b = Math.round(32 + (48 - 32) * t);
  return `rgb(255, ${g}, ${b})`;
}

/** Hex → rgba() with alpha. */
export function withAlpha(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
