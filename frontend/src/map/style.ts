/**
 * Map styling: a dark transit-control-room palette and zoom tiers. Every size is
 * in CSS (screen) pixels so the map stays crisp at every zoom (§6).
 */
import { MAP_BACKGROUND } from "@/lib/network";

export const PALETTE = {
  background: MAP_BACKGROUND,
  water: MAP_BACKGROUND,
  land: "#121a26",
  landEdge: "#1c2738",
  grid: "rgba(120, 150, 190, 0.035)",
  casing: "rgba(6, 9, 14, 0.92)",
  riverCasing: "rgba(120, 180, 230, 0.55)",
  stationFill: "#f4f6fa",
  stationStroke: "#0a0f17",
  hubFill: "#ffffff",
  hubStroke: "#0a0f17",
  /** Transfer passages between the stations of a complex. */
  transfer: "rgba(226, 232, 242, 0.78)",
  label: "#dfe6f1",
  labelMuted: "#9aa8bc",
  labelHalo: "rgba(8, 12, 19, 0.92)",
  borough: "rgba(150, 172, 204, 0.26)",
  interlocking: "#8fb3ff",
  yard: "#9aa8bc",
  yardLead: "rgba(154, 168, 188, 0.45)",
  yardLabel: "#7d8aa0",
  calm: "#f4f6fa",
  amber: "#ffb020",
  red: "#ff453a",
  closed: "#05070b",
  closedX: "#ff453a",
  block: "#ff453a",
  signal: "#ffb020",
  selection: "#ffffff",
  highlight: "#5ac8fa",
  heldRing: "#ffb020",
  stuckHalo: "rgba(255, 69, 58, 0.55)",
  gap: "#34c759",
  chipBg: "rgba(14, 20, 31, 0.94)",
  chipText: "#f4f6fa",
} as const;

export const FONT_STACK = '"Helvetica Neue", Helvetica, Arial, system-ui, sans-serif';

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
  showTrainLabels: boolean;
  showStatusIcons: boolean;
}

/** Widest → closest. Thresholds in metres per CSS pixel. */
export const ZOOM_TIERS: readonly ZoomTier[] = [
  {
    id: 0,
    name: "city",
    minMPerPx: 40,
    lineWidth: 2.2,
    lineGap: 0.3,
    casingExtra: 1,
    stationRadius: 1.3,
    hubRadius: 2.2,
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
    capsules: false,
    trainLength: 4.6,
    trainWidth: 4.6,
    laneWidth: 1.4,
    showTrainLabels: false,
    showStatusIcons: false,
  },
  {
    id: 1,
    name: "borough",
    minMPerPx: 18,
    lineWidth: 2.6,
    lineGap: 0.4,
    casingExtra: 1.2,
    stationRadius: 1.6,
    hubRadius: 2.8,
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
    capsules: false,
    trainLength: 5.4,
    trainWidth: 5.4,
    laneWidth: 1.8,
    showTrainLabels: false,
    showStatusIcons: false,
  },
  {
    id: 2,
    name: "district",
    minMPerPx: 8,
    lineWidth: 3.2,
    lineGap: 0.6,
    casingExtra: 1.5,
    stationRadius: 2.6,
    hubRadius: 3.9,
    stationStroke: 1,
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
    trainLength: 11,
    trainWidth: 6,
    laneWidth: 3.6,
    showTrainLabels: false,
    showStatusIcons: true,
  },
  {
    id: 3,
    name: "neighborhood",
    minMPerPx: 3.2,
    lineWidth: 4.2,
    lineGap: 0.9,
    casingExtra: 2,
    stationRadius: 3.6,
    hubRadius: 5.2,
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
    trainLength: 17,
    trainWidth: 9,
    laneWidth: 5.8,
    showTrainLabels: true,
    showStatusIcons: true,
  },
  {
    id: 4,
    name: "street",
    minMPerPx: 0,
    lineWidth: 5.4,
    lineGap: 1,
    casingExtra: 2.4,
    stationRadius: 4.6,
    hubRadius: 6.6,
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
    trainLength: 23,
    trainWidth: 11,
    laneWidth: 7.6,
    showTrainLabels: true,
    showStatusIcons: true,
  },
];

export function tierFor(mPerPx: number): ZoomTier {
  for (const t of ZOOM_TIERS) if (mPerPx >= t.minMPerPx) return t;
  return ZOOM_TIERS[ZOOM_TIERS.length - 1];
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
