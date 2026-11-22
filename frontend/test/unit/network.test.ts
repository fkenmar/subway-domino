/** network_geo.json (§2.7) validation, lookup tables and loading. */
import { describe, expect, it, vi } from "vitest";
import {
  buildNetwork,
  contrastRatio,
  EDGE_CROSSOVER,
  EDGE_RUN,
  EDGE_TURNBACK,
  EDGE_UNKNOWN,
  EDGE_YARD_LEAD,
  ensureContrast,
  GeoValidationError,
  loadNetwork,
  MAP_BACKGROUND,
  SIGNAGE_BACKGROUND,
  normalizeHex,
  routeBullet,
  TRACK_L,
  TRACK_X,
  validateGeo,
} from "@/lib/network";
import { miniGeo, miniGeoRaw, miniNetwork, syntheticGeo, syntheticNetwork } from "../helpers/geo";

describe("validateGeo", () => {
  it("accepts the §2.7 fixture", () => {
    const geo = miniGeo();
    expect(geo.schema_version).toBe(1);
    expect(geo.stations).toHaveLength(47);
  });

  it.each([
    ["a non-object", () => 42, /not an object/],
    ["another schema version", () => ({ ...miniGeoRaw(), schema_version: 2 }), /schema_version/],
    ["a missing array", () => ({ ...miniGeoRaw(), edges: undefined }), /"edges"/],
    ["a missing bbox", () => ({ ...miniGeoRaw(), bbox: [0, 0] }), /bbox/],
    [
      "out-of-order station indices",
      () => {
        const raw = miniGeoRaw() as { stations: { idx: number }[] };
        raw.stations[3].idx = 99;
        return raw;
      },
      /stations\[3\]/,
    ],
    [
      "a segment without points",
      () => {
        const raw = miniGeoRaw() as { segments: { pts: unknown[] }[] };
        raw.segments[0].pts = [];
        return raw;
      },
      /segments\[0\]/,
    ],
  ])("rejects %s", (_name, make, message) => {
    expect(() => validateGeo(make())).toThrow(GeoValidationError);
    expect(() => validateGeo(make())).toThrow(message);
  });

  it("treats a missing basemap as none and drops presets it cannot run", () => {
    const raw = miniGeoRaw();
    delete raw.basemap;
    expect(validateGeo(raw).basemap).toBeNull();
    expect(syntheticGeo().presets.map((p) => p.id)).toEqual(["close-bravo"]);
  });
});

describe("buildNetwork: edge table", () => {
  const net = syntheticNetwork();
  const E = net.edges;

  it("classifies edge kinds and tracks", () => {
    expect([...E.kind]).toEqual([
      EDGE_RUN,
      EDGE_RUN,
      EDGE_RUN,
      EDGE_RUN,
      EDGE_CROSSOVER,
      EDGE_TURNBACK,
      EDGE_YARD_LEAD,
      EDGE_YARD_LEAD,
      EDGE_YARD_LEAD,
      EDGE_UNKNOWN, // run edge pointing at a segment that does not exist
    ]);
    expect(E.track[2]).toBe(TRACK_X);
    expect(E.track[0]).toBe(TRACK_L);
  });

  it("derives from/to stations from seg_dir (physical direction from station order)", () => {
    expect([E.fromStation[0], E.toStation[0]]).toEqual([0, 1]);
    expect([E.fromStation[1], E.toStation[1]]).toEqual([1, 0]);
    expect([E.fromStation[3], E.toStation[3]]).toEqual([1, 2]);
    expect([E.fromStation[4], E.toStation[4]]).toEqual([0, 0]);
    // Yard leads: −1 stands for the yard end.
    expect([E.fromStation[6], E.toStation[6]]).toEqual([-1, 2]);
    expect([E.fromStation[7], E.toStation[7]]).toEqual([2, -1]);
  });

  it("measures lengths from geometry", () => {
    expect(E.lengthM[0]).toBeCloseTo(400);
    expect(E.lengthM[3]).toBeCloseTo(300);
    expect(E.lengthM[4]).toBe(0);
    expect(E.lengthM[6]).toBeCloseTo(300);
  });

  it("flags express segments and stations", () => {
    expect([...net.segmentHasXEdge]).toEqual([1, 0]);
    expect([...net.stationHasExpress]).toEqual([1, 1, 0]);
    expect(net.stationInterlocking[0]).toBe(0);
    expect(net.stationInterlocking[2]).toBe(-1);
  });

  it("picks the best-ranked station of each complex", () => {
    const mini = miniNetwork();
    for (const c of mini.complexes) {
      const rep = mini.complexRep[c.idx];
      expect(c.stations).toContain(rep);
      const best = Math.min(...c.stations.map((s) => mini.stations[s].label_rank));
      expect(mini.stations[rep].label_rank).toBe(best);
    }
  });
});

describe("buildNetwork: route styling", () => {
  it("normalises colours and keeps the official route colours on the light map", () => {
    const mini = miniNetwork();
    for (const r of mini.routes) {
      expect(r.color).toMatch(/^#[0-9A-F]{6}$/);
      expect(r.mapColor).toBe(r.color);
    }
    expect(contrastRatio("#000000", MAP_BACKGROUND)).toBeGreaterThan(10);
    expect(syntheticNetwork().routes[0].color).toBe("#00933C");
  });

  it("uses MTA bullet conventions", () => {
    expect(routeBullet("6X")).toEqual({ text: "6", diamond: true });
    expect(routeBullet("FX")).toEqual({ text: "F", diamond: true });
    for (const shuttle of ["GS", "FS", "H"]) expect(routeBullet(shuttle)).toEqual({ text: "S", diamond: false });
    expect(routeBullet("A")).toEqual({ text: "A", diamond: false });
  });

  it("normalises hex colours with a fallback", () => {
    expect(normalizeHex("abc")).toBe("#AABBCC");
    expect(normalizeHex("#0062cf")).toBe("#0062CF");
    expect(normalizeHex("blue", "#808183")).toBe("#808183");
  });

  it("lightens dark colours only as far as needed", () => {
    const navy = "#0039A6"; // the old A/C/E blue is too dark as a text accent on black signage
    expect(contrastRatio(navy, SIGNAGE_BACKGROUND)).toBeLessThan(4.2);
    expect(contrastRatio(ensureContrast(navy, SIGNAGE_BACKGROUND, 4.2), SIGNAGE_BACKGROUND)).toBeGreaterThanOrEqual(4.2);
    expect(ensureContrast("#FCCC0A", SIGNAGE_BACKGROUND, 4.2)).toBe("#FCCC0A");
  });

  it("orders each segment's colour bundle canonically", () => {
    const mini = miniNetwork();
    const rank = new Map<string, number>();
    mini.routes.forEach((r) => rank.has(r.mapColor) || rank.set(r.mapColor, rank.size));
    for (const seg of mini.segments) {
      const ranks = seg.colors.map((c) => rank.get(c) ?? -1);
      expect(ranks).toEqual([...ranks].sort((a, b) => a - b));
      expect(new Set(seg.colors).size).toBe(seg.colors.length);
    }
  });
});

describe("buildNetwork: bundle orientation", () => {
  it("keeps colour bundles from twisting where segments meet head-to-head", () => {
    // A straight east-west chain S0–S1–S2–S3 whose middle polyline is stored reversed (a=2, b=1).
    const geo = syntheticGeo();
    const st = (idx: number, x: number) => ({
      idx,
      gtfs: `B0${idx}`,
      name: `S${idx}`,
      complex: 0,
      x,
      y: 0,
      routes: [0],
      hub: false,
      label_rank: idx,
    });
    geo.stations = [st(0, 0), st(1, 500), st(2, 1000), st(3, 1500)];
    geo.complexes = [{ idx: 0, name: "S", x: 0, y: 0, stations: [0, 1, 2, 3], weekday_entries: 1 }];
    const seg = (idx: number, a: number, b: number) => ({
      idx,
      a,
      b,
      name: `${a}-${b}`,
      river: false,
      has_express: false,
      routes: [0],
      pts: [
        [geo.stations[a].x, 0],
        [geo.stations[b].x, 0],
      ] as [number, number][],
    });
    geo.segments = [seg(0, 0, 1), seg(1, 2, 1), seg(2, 2, 3)];
    geo.edges = [];
    geo.interlockings = [];
    geo.yards = [];
    const net = buildNetwork(geo);
    // Effective direction of each bundle (start → end station) must chain through shared stations.
    const ends = net.segments.map((s) => (s.bundleDir === 1 ? [s.a, s.b] : [s.b, s.a]));
    expect(ends[0][1]).toBe(ends[1][0]);
    expect(ends[1][1]).toBe(ends[2][0]);
    expect(net.segments.map((s) => s.bundleDir)).toEqual([1, -1, 1]);
  });
});

describe("loadNetwork", () => {
  const okResponse = (body: unknown) => ({ ok: true, status: 200, json: async () => body }) as Response;
  const notFound = () => ({ ok: false, status: 404, json: async () => ({}) }) as Response;

  it("loads the first URL that serves a valid file", async () => {
    const fetchImpl = vi.fn(async (url: string | URL | Request) =>
      String(url).includes("static") ? notFound() : okResponse(miniGeoRaw()),
    );
    const net = await loadNetwork({ fetchImpl: fetchImpl as typeof fetch, urls: ["/static.json", "http://backend/geo"] });
    expect(net.stations).toHaveLength(47);
    expect(fetchImpl).toHaveBeenCalledTimes(2);
    expect(String(fetchImpl.mock.calls[1][0])).toBe("http://backend/geo");
  });

  it("falls through invalid files and reports the last error", async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce(okResponse({ schema_version: 99 }))
      .mockResolvedValueOnce(notFound());
    await expect(loadNetwork({ fetchImpl, urls: ["/a", "/b"] })).rejects.toThrow(/HTTP 404/);
  });

  it("stops immediately when aborted", async () => {
    const controller = new AbortController();
    const fetchImpl = vi.fn(async () => {
      controller.abort();
      throw new DOMException("aborted", "AbortError");
    });
    await expect(loadNetwork({ fetchImpl, urls: ["/a", "/b"], signal: controller.signal })).rejects.toThrow(/aborted/);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });
});
