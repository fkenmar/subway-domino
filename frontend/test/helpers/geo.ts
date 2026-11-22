/**
 * Geo fixtures for unit tests.
 *
 * - {@link miniGeo} / {@link miniNetwork}: the hand-written §2.7 fixture
 *   (test/fixtures/mini_geo.json — 47 Manhattan/Brooklyn/Queens stations).
 * - {@link syntheticGeo}: a tiny network with exact, hand-checkable geometry
 *   for positioning and topology assertions.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { buildNetwork, type GeoFile, type Network, validateGeo } from "@/lib/network";

const MINI_GEO_PATH = fileURLToPath(new URL("../fixtures/mini_geo.json", import.meta.url));
const MINI_GEO_TEXT = readFileSync(MINI_GEO_PATH, "utf8");

/** A fresh parse of the mini fixture (safe to mutate). */
export function miniGeoRaw(): Record<string, unknown> {
  return JSON.parse(MINI_GEO_TEXT) as Record<string, unknown>;
}

export function miniGeo(): GeoFile {
  return validateGeo(miniGeoRaw());
}

export function miniNetwork(): Network {
  return buildNetwork(miniGeo());
}

/**
 * Synthetic network (metres, x east / y north):
 *
 * ```
 *                      S2 Charlie (300,400) ─── yard Y0 (600,400)
 *                       │ seg 1 (300 m, local only)
 *   S0 Alpha (0,0) ──── ┘ S1 Bravo (300,100)
 *        seg 0: (0,0) → (300,0) → (300,100), 400 m, has_express
 * ```
 *
 * Edges: 0 run seg0 +1 L · 1 run seg0 −1 L · 2 run seg0 +1 X · 3 run seg1 +1 L ·
 * 4 crossover @S0 · 5 turnback @S2 · 6 yard lead Y0→S2 (pull-out) ·
 * 7 yard lead S2→Y0 (put-in, seg_dir −1) · 8 yard lead without a lead station ·
 * 9 run edge with an invalid segment.
 */
export function syntheticGeo(): GeoFile {
  return validateGeo({
    schema_version: 1,
    graph_version: "synthetic000",
    projection: { lat0: 40.7128, lon0: -73.95, kx: 84384.0, ky: 111132.954 },
    bbox: [0, 0, 600, 400],
    routes: [
      { idx: 0, id: "4", name: "Lexington Avenue Express", color: "00933C", text_color: "FFFFFF", division: "A" },
      { idx: 1, id: "6X", name: "Pelham Bay Park Express", color: "#00933C", text_color: "#FFFFFF", division: "A" },
    ],
    complexes: [
      { idx: 0, name: "Alpha", x: 0, y: 0, stations: [0], weekday_entries: 1000 },
      { idx: 1, name: "Bravo", x: 300, y: 100, stations: [1], weekday_entries: 500 },
      { idx: 2, name: "Charlie", x: 300, y: 400, stations: [2], weekday_entries: 200 },
    ],
    stations: [
      { idx: 0, gtfs: "A01", name: "Alpha", complex: 0, x: 0, y: 0, routes: [0, 1], hub: true, label_rank: 0 },
      { idx: 1, gtfs: "A02", name: "Bravo", complex: 1, x: 300, y: 100, routes: [0, 1], hub: false, label_rank: 1 },
      { idx: 2, gtfs: "A03", name: "Charlie", complex: 2, x: 300, y: 400, routes: [0], hub: false, label_rank: 2 },
    ],
    segments: [
      {
        idx: 0,
        a: 0,
        b: 1,
        name: "Alpha – Bravo",
        river: false,
        has_express: true,
        routes: [0, 1],
        pts: [
          [0, 0],
          [300, 0],
          [300, 100],
        ],
      },
      {
        idx: 1,
        a: 1,
        b: 2,
        name: "Bravo – Charlie",
        river: true,
        has_express: false,
        routes: [0],
        pts: [
          [300, 100],
          [300, 400],
        ],
      },
    ],
    edges: [
      { idx: 0, kind: "run", segment: 0, seg_dir: 1, track: "L", station: -1, yard: -1 },
      { idx: 1, kind: "run", segment: 0, seg_dir: -1, track: "L", station: -1, yard: -1 },
      { idx: 2, kind: "run", segment: 0, seg_dir: 1, track: "X", station: -1, yard: -1 },
      { idx: 3, kind: "run", segment: 1, seg_dir: 1, track: "L", station: -1, yard: -1 },
      { idx: 4, kind: "crossover", segment: -1, seg_dir: 0, track: "L", station: 0, yard: -1 },
      { idx: 5, kind: "turnback", segment: -1, seg_dir: 0, track: "L", station: 2, yard: -1 },
      { idx: 6, kind: "yard_lead", segment: -1, seg_dir: 1, track: "L", station: 2, yard: 0 },
      { idx: 7, kind: "yard_lead", segment: -1, seg_dir: -1, track: "L", station: 2, yard: 0 },
      { idx: 8, kind: "yard_lead", segment: -1, seg_dir: 0, track: "L", station: -1, yard: 0 },
      { idx: 9, kind: "run", segment: 7, seg_dir: 1, track: "L", station: -1, yard: -1 },
    ],
    interlockings: [{ idx: 0, station: 0, x: 0, y: 0, kinds: ["L2X", "X2L"] }],
    yards: [{ idx: 0, name: "Test Yard", x: 600, y: 400, division: "A" }],
    presets: [
      {
        id: "close-bravo",
        title: "Close Bravo",
        subtitle: "4 6",
        profile: "am_peak",
        disruptions: [{ kind: "full_station_closure", target: { type: "station", idx: 1 }, duration_s: 1800 }],
        focus: { x: 300, y: 100, zoom_m_per_px: 4 },
      },
      { id: "bad-profile", title: "x", subtitle: "", profile: "rush_hour", disruptions: [] },
    ],
    basemap: null,
  });
}

export function syntheticNetwork(): Network {
  return buildNetwork(syntheticGeo());
}
