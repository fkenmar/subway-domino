/** Station and route search (§6 UX flow 5). */
import { describe, expect, it } from "vitest";
import { normalizeName, SearchIndex } from "@/lib/search";
import { miniNetwork } from "../helpers/geo";

const net = miniNetwork();
const index = new SearchIndex(net);

describe("normalizeName", () => {
  it.each([
    ["42nd Street", "42 st"],
    ["Times Sq-42 St", "times sq 42 st"],
    ["Jackson Hts–Roosevelt Av", "jackson hts roosevelt av"],
    ["Jackson Heights Roosevelt Avenue", "jackson hts roosevelt av"],
    ["Bedford-Nostrand Avs", "bedford nostrand avs"],
    ["Café & Plaza", "cafe plz"],
    ["  1st   Ave ", "1 av"],
  ])("%j → %j", (input, expected) => {
    expect(normalizeName(input)).toBe(expected);
  });
});

describe("SearchIndex", () => {
  it("finds stations by prefix, ranked by relevance and ridership", () => {
    const [first] = index.query("times");
    expect(first).toMatchObject({ kind: "station", name: "Times Sq-42 St" });
  });

  it("centres on the highlighted station of a multi-station complex", () => {
    const [first] = index.query("times");
    if (first.kind !== "station") throw new Error("expected a station");
    const st = net.stations[first.stationIdx];
    expect(net.complexes[first.complexIdx].stations.length).toBeGreaterThan(1);
    expect(st.complex).toBe(first.complexIdx);
    expect([first.x, first.y]).toEqual([st.x, st.y]);
  });

  it("matches spelled-out street names against MTA abbreviations", () => {
    const names = index.query("42nd street").map((r) => r.name);
    expect(names).toContain("Times Sq-42 St");
    expect(names).toContain("Grand Central-42 St");
  });

  it("returns one result per station complex", () => {
    const results = index.query("union sq");
    const keys = results.map((r) => r.key);
    expect(new Set(keys).size).toBe(keys.length);
    expect(results.filter((r) => r.kind === "station" && r.name.includes("Union Sq"))).toHaveLength(1);
  });

  it("finds routes by bullet and by line name, with a fly-to extent", () => {
    const six = index.query("6")[0];
    expect(six).toMatchObject({ kind: "route", name: net.routes[net.routeById.get("6") ?? -1].name });
    const l = index.query("L train")[0];
    expect(l.kind).toBe("route");
    if (l.kind === "route") {
      expect(net.routes[l.routeIdx].id).toBe("L");
      expect(l.bbox[2]).toBeGreaterThan(l.bbox[0]);
    }
  });

  it("returns nothing for empty or unmatched queries and honours the limit", () => {
    expect(index.query("")).toEqual([]);
    expect(index.query("   -- ")).toEqual([]);
    expect(index.query("zzzzqx")).toEqual([]);
    expect(index.query("st", 3).length).toBeLessThanOrEqual(3);
  });
});
