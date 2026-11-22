/**
 * Station and route search with MTA-style name normalisation
 * ("42nd Street" ≈ "42 St", "Ave" ≈ "Av", accents and punctuation ignored).
 */
import { type Bbox } from "./geometry";
import type { Network } from "./network";

const ABBREVIATIONS: Record<string, string> = {
  street: "st",
  streets: "sts",
  avenue: "av",
  ave: "av",
  avenues: "avs",
  square: "sq",
  road: "rd",
  boulevard: "blvd",
  parkway: "pkwy",
  place: "pl",
  center: "ctr",
  centre: "ctr",
  heights: "hts",
  junction: "jct",
  plaza: "plz",
  saint: "st",
  fort: "ft",
  mount: "mt",
  terrace: "ter",
  highway: "hwy",
  and: "",
};

/** Normalise a station name or query for matching. */
export function normalizeName(s: string): string {
  return s
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((w) => w.replace(/^(\d+)(st|nd|rd|th)$/, "$1"))
    .map((w) => ABBREVIATIONS[w] ?? w)
    .filter(Boolean)
    .join(" ");
}

export type SearchResult =
  | {
      kind: "station";
      key: string;
      complexIdx: number;
      stationIdx: number;
      name: string;
      routes: number[];
      x: number;
      y: number;
      score: number;
    }
  | {
      kind: "route";
      key: string;
      routeIdx: number;
      name: string;
      bbox: Bbox;
      score: number;
    };

interface StationEntry {
  complexIdx: number;
  stationIdx: number;
  name: string;
  words: string[];
  norm: string;
  routes: number[];
  x: number;
  y: number;
  popularity: number;
}

interface RouteEntry {
  routeIdx: number;
  id: string;
  name: string;
  words: string[];
  bbox: Bbox;
}

function matchTokens(tokens: string[], words: string[]): number {
  let score = 0;
  for (const t of tokens) {
    let best = 0;
    for (const w of words) {
      if (w === t) best = Math.max(best, 3);
      else if (w.startsWith(t)) best = Math.max(best, 2);
      else if (t.length >= 3 && w.includes(t)) best = Math.max(best, 1);
    }
    if (best === 0) return 0;
    score += best;
  }
  return score;
}

export class SearchIndex {
  private readonly stations: StationEntry[];
  private readonly routes: RouteEntry[];

  constructor(net: Network) {
    // One entry per complex (a complex is what riders think of as "a station").
    this.stations = net.complexes
      .map((c): StationEntry | null => {
        const rep = net.complexRep[c.idx];
        const st = rep >= 0 ? net.stations[rep] : undefined;
        if (!st) return null;
        const names = new Set<string>([c.name, ...c.stations.map((s) => net.stations[s]?.name ?? "")]);
        const norm = normalizeName(st.name);
        const words = [...new Set([...names].flatMap((n) => normalizeName(n).split(" ")).filter(Boolean))];
        const routes = [...new Set(c.stations.flatMap((s) => net.stations[s]?.routes ?? []))].sort((a, b) => a - b);
        return {
          complexIdx: c.idx,
          stationIdx: rep,
          name: st.name,
          words,
          norm,
          routes,
          // Fly to the station that gets highlighted, not the complex centroid (which can
          // fall between platforms in large complexes like Times Sq).
          x: st.x,
          y: st.y,
          popularity: Math.log10(1 + Math.max(0, c.weekday_entries)),
        };
      })
      .filter((e): e is StationEntry => e !== null);

    const extents = new Map<number, [number, number, number, number]>();
    for (const seg of net.segments) {
      for (const r of seg.routes) {
        const b = extents.get(r) ?? [Infinity, Infinity, -Infinity, -Infinity];
        for (let i = 0; i < seg.line.xs.length; i++) {
          const x = seg.line.xs[i];
          const y = seg.line.ys[i];
          if (x < b[0]) b[0] = x;
          if (y < b[1]) b[1] = y;
          if (x > b[2]) b[2] = x;
          if (y > b[3]) b[3] = y;
        }
        extents.set(r, b);
      }
    }
    this.routes = net.routes
      .filter((r) => extents.has(r.idx))
      .map((r) => ({
        routeIdx: r.idx,
        id: r.id.toLowerCase(),
        name: r.name,
        words: normalizeName(r.name).split(" "),
        bbox: extents.get(r.idx) as Bbox,
      }));
  }

  query(raw: string, limit = 8): SearchResult[] {
    const q = normalizeName(raw);
    if (!q) return [];
    const tokens = q.split(" ");
    const out: SearchResult[] = [];

    // Routes: "6", "a train", "L line", or a line name ("Lexington").
    const routeQuery = q.replace(/\b(train|line|trains|lines)\b/g, "").trim();
    for (const r of this.routes) {
      let score = 0;
      if (routeQuery === r.id) score = 100;
      else if (tokens.length === 1 && q.length <= 2) score = 0;
      else score = matchTokens(tokens, r.words) > 0 ? 4 + matchTokens(tokens, r.words) : 0;
      if (score > 0) {
        out.push({ kind: "route", key: `r${r.routeIdx}`, routeIdx: r.routeIdx, name: r.name, bbox: r.bbox, score });
      }
    }

    for (const s of this.stations) {
      const m = matchTokens(tokens, s.words);
      if (m === 0) continue;
      let score = m + s.popularity * 0.6;
      if (s.norm.startsWith(q)) score += 4;
      else if (s.norm.includes(q)) score += 1.5;
      out.push({
        kind: "station",
        key: `c${s.complexIdx}`,
        complexIdx: s.complexIdx,
        stationIdx: s.stationIdx,
        name: s.name,
        routes: s.routes,
        x: s.x,
        y: s.y,
        score,
      });
    }

    out.sort((a, b) => b.score - a.score || a.key.localeCompare(b.key));
    return out.slice(0, limit);
  }
}
