/** `#map=m/lat/lon` deep links to a map view. */
import { describe, expect, it } from "vitest";
import { DEFAULT_PROJECTION } from "@/lib/network";
import { formatViewHash, parseViewHash, viewToCamera } from "@/lib/viewHash";

describe("parseViewHash", () => {
  it("parses m/px, lat and lon", () => {
    expect(parseViewHash("#map=3/40.758/-73.9855")).toEqual({ mPerPx: 3, lat: 40.758, lon: -73.9855 });
  });

  it("accepts other hash parameters around it and a missing #", () => {
    expect(parseViewHash("#foo=1&map=2.5/40.7/-73.9&bar")).toEqual({ mPerPx: 2.5, lat: 40.7, lon: -73.9 });
    expect(parseViewHash("map=2.5/40.7/-73.9")).toEqual({ mPerPx: 2.5, lat: 40.7, lon: -73.9 });
  });

  it("rejects absent, malformed and out-of-range views", () => {
    for (const h of ["", "#", "#map=", "#map=3/40.7", "#map=a/b/c", "#map=0/40.7/-73.9", "#map=-2/40.7/-73.9", "#map=3/95/-73.9", "#map=3/40/-190", "#mapx=3/40.7/-73.9"]) {
      expect(parseViewHash(h), h).toBeNull();
    }
  });

  it("round-trips through formatViewHash", () => {
    const v = { mPerPx: 4.25, lat: 40.69213, lon: -73.99042 };
    expect(formatViewHash(v)).toBe("#map=4.25/40.69213/-73.99042");
    expect(parseViewHash(formatViewHash(v))).toEqual(v);
  });
});

describe("viewToCamera", () => {
  it("projects into the §2.6 metric frame", () => {
    const c = viewToCamera({ mPerPx: 3, lat: DEFAULT_PROJECTION.lat0, lon: DEFAULT_PROJECTION.lon0 }, DEFAULT_PROJECTION);
    expect(c).toEqual({ cx: 0, cy: 0, mPerPx: 3 });
    const d = viewToCamera({ mPerPx: 3, lat: DEFAULT_PROJECTION.lat0 + 0.01, lon: DEFAULT_PROJECTION.lon0 + 0.01 }, DEFAULT_PROJECTION);
    expect(d.cx).toBeCloseTo(843.84, 2);
    expect(d.cy).toBeCloseTo(1111.33, 2);
  });
});
