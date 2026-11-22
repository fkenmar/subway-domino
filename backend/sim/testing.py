"""A tiny two-division network with the exact ``network.json`` schema (ARCHITECTURE §3.1).

``make_toy_graph()`` is used by unit tests and by parallel development before the real
artifact exists. Layout (y grows north)::

    A division (routes 1 local, 2 express)            B division (route L)

    A01 Northgate  (terminal)                         B01 Canarsie Jct (terminal)  CBTC
     |  2 tracks                                        |  1000 m                    CBTC
    A02 Express Center   L+X, crossovers, hub          B02 Linden                   CBTC
     |  4-track trunk (express track)                   |  2400 m                    CBTC
    A03 Midtown Local    X passes (no platform)        B03 Transfer Plaza  <-> yard (lead), hub
     |  4-track trunk                                   |  1000 m  (river tunnel)
    A04 Union Junction   L+X, crossovers, POCKET, hub  B04 Riverside
     |                                                  |  2400 m  fixed block
    A05 Split Junction   DIVERGE (west/east branches)  B05 Seaview (terminal)
     |\\
    A06 A07               route 1 via A06, route 2 via A07
     |/
    A08 Southgate (terminal)

Stations A04 and B03 form one cross-division complex (transfers, no shared track).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from sim import constants as C
from sim.graph import TRACK_L, TRACK_X, Graph, canonical_graph_version, graph_from_dict

KX = 84384.0
KY = 111132.954
LAT0 = 40.7128
LON0 = -73.95


@dataclass
class _Builder:
    routes: list[dict[str, Any]] = field(default_factory=list)
    complexes: list[dict[str, Any]] = field(default_factory=list)
    stations: list[dict[str, Any]] = field(default_factory=list)
    platforms: list[dict[str, Any]] = field(default_factory=list)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)
    patterns: list[dict[str, Any]] = field(default_factory=list)
    node_key: dict[tuple[str, str, str], int] = field(default_factory=dict)
    st_key: dict[str, int] = field(default_factory=dict)

    def route(self, rid: str, name: str, color: str, division: str) -> int:
        self.routes.append({"idx": len(self.routes), "id": rid, "name": name, "color": color,
                            "text_color": "#FFFFFF", "division": division, "shuttle": False})
        return len(self.routes) - 1

    def station(self, gtfs: str, name: str, division: str, x: float, y: float, routes: list[int],
                complex_idx: int, hub: bool = False, terminal: bool = False, structure: str = "Subway") -> int:
        idx = len(self.stations)
        self.stations.append({"idx": idx, "gtfs": gtfs, "name": name, "complex": complex_idx, "division": division,
                              "x": x, "y": y, "structure": structure, "routes": routes, "hub": hub,
                              "terminal": terminal})
        self.st_key[gtfs] = idx
        self.complexes[complex_idx]["stations"].append(idx)
        return idx

    def complex(self, cid: str, name: str, borough: str, x: float, y: float, entries: float) -> int:
        self.complexes.append({"idx": len(self.complexes), "complex_id": cid, "name": name, "borough": borough,
                               "x": x, "y": y, "stations": [], "weekday_entries": entries})
        return len(self.complexes) - 1

    def node(self, gtfs: str, direction: str, track: str) -> int:
        s = self.st_key[gtfs]
        idx = len(self.nodes)
        self.nodes.append({"idx": idx, "station": s, "platform": -1, "track": track,
                           "division": self.stations[s]["division"]})
        self.node_key[(gtfs, direction, track)] = idx
        return idx

    def platform(self, gtfs: str, direction: str, tracks: tuple[str, ...], capacity: float) -> int:
        s = self.st_key[gtfs]
        idx = len(self.platforms)
        nodes = [self.node_key[(gtfs, direction, t)] for t in tracks]
        self.platforms.append({"idx": idx, "station": s, "nodes": nodes, "capacity": capacity})
        for n in nodes:
            self.nodes[n]["platform"] = idx
        return idx

    def n(self, gtfs: str, direction: str, track: str = "L") -> int:
        return self.node_key[(gtfs, direction, track)]

    def edge(self, kind: str, a: int, b: int, *, track: str = "L", length: float = 0.0, vmax: float = 15.0,
             cbtc: bool = False, pocket: bool = False, run_s: float | None = None) -> int:
        idx = len(self.edges)
        if kind == "run":
            rs = length / vmax + C.ACCEL_DECEL_LOSS_S if run_s is None else run_s
        elif kind == "crossover":
            rs, length, vmax = C.CROSSOVER_S, 0.0, 0.0
        elif kind == "turnback":
            rs, length, vmax = (C.TURNBACK_S if pocket else C.LAYOVER_S), 0.0, 0.0
        else:  # yard_lead
            rs, length, vmax = (60.0 if run_s is None else run_s), 0.0, 0.0
        self.edges.append({"idx": idx, "kind": kind, "from": a, "to": b, "segment": -1, "seg_dir": 0,
                           "track": track, "length_m": float(length), "run_s": round(float(rs), 1),
                           "v_max": round(float(vmax), 2), "cbtc": cbtc, "bidir_group": -1, "pocket": pocket})
        return idx

    def find(self, a: int, b: int, kind: str = "run") -> int:
        for e in self.edges:
            if e["from"] == a and e["to"] == b and e["kind"] == kind:
                return int(e["idx"])
        raise KeyError((a, b, kind))

    def segment(self, ga: str, gb: str, edges: list[int], river: bool = False) -> int:
        idx = len(self.segments)
        a, b = self.st_key[ga], self.st_key[gb]
        has_x = any(self.edges[e]["track"] == "X" for e in edges)
        self.segments.append({"idx": idx, "a": a, "b": b,
                              "name": f"{self.stations[a]['name']} – {self.stations[b]['name']}",
                              "river": river, "has_express": has_x, "edges": sorted(edges)})
        for e in edges:
            ed = self.edges[e]
            ed["segment"] = idx
            ed["seg_dir"] = 1 if self.nodes[ed["from"]]["station"] == a else -1
        return idx

    def pattern(self, route: int, node_seq: list[tuple[int, str, int]], trips: int) -> int:
        """node_seq: [(node, kind_of_edge_into_node, stop_flag)], first entry is the origin."""
        edges, stops = [], []
        for (prev, _, _), (cur, kind, stop) in zip(node_seq[:-1], node_seq[1:]):
            edges.append(self.find(prev, cur, kind))
            stops.append(stop)
        idx = len(self.patterns)
        o = self.nodes[node_seq[0][0]]["station"]
        d = self.nodes[node_seq[-1][0]]["station"]
        self.patterns.append({"idx": idx, "route": route, "edges": edges, "stops": stops, "origin": o,
                              "dest": d, "trips_weekday": trips, "next_patterns": []})
        return idx


def _xy_to_ll(x: float, y: float) -> tuple[float, float]:
    return LAT0 + y / KY, LON0 + x / KX


def toy_network_dict() -> dict[str, Any]:
    """The toy network as a ``network.json``-schema dict (fresh copy on every call)."""
    b = _Builder()
    r1 = b.route("1", "Toy Local", "#EE352E", "A")
    r2 = b.route("2", "Toy Express", "#EE352E", "A")
    rl = b.route("L", "Toy Canarsie", "#A7A9AC", "B")

    # ---------------- complexes & stations
    a_st = [
        # gtfs, name, x, y, hub, terminal, routes
        ("A01", "Northgate", 0.0, 6200.0, False, True, [r1, r2]),
        ("A02", "Express Center", 0.0, 5000.0, True, False, [r1, r2]),
        ("A03", "Midtown Local", 0.0, 4000.0, False, False, [r1]),
        ("A04", "Union Junction", 0.0, 3000.0, True, False, [r1, r2]),
        ("A05", "Split Junction", 0.0, 2000.0, False, False, [r1, r2]),
        ("A06", "West Branch", -800.0, 900.0, False, False, [r1]),
        ("A07", "East Branch", 800.0, 900.0, False, False, [r2]),
        ("A08", "Southgate", 0.0, -200.0, False, True, [r1, r2]),
    ]
    b_st = [
        ("B01", "Canarsie Junction", 3000.0, 5200.0, False, True, [rl]),
        ("B02", "Linden", 3000.0, 4200.0, False, False, [rl]),
        ("B03", "Transfer Plaza", 3000.0, 1800.0, True, False, [rl]),
        ("B04", "Riverside", 3000.0, 800.0, False, False, [rl]),
        ("B05", "Seaview", 3000.0, -1600.0, False, True, [rl]),
    ]
    entries = {"A01": 9000, "A02": 42000, "A03": 12000, "A04": 30000, "A05": 8000, "A06": 6000,
               "A07": 6000, "A08": 9000, "B01": 7000, "B02": 9000, "B03": 22000, "B04": 8000, "B05": 6000}
    union_cpx = None
    for gtfs, name, x, y, hub, term, routes in a_st + b_st:
        if gtfs in ("A04", "B03"):
            if union_cpx is None:
                union_cpx = b.complex("900", "Union Junction-Transfer Plaza", "M", 1500.0, 2400.0,
                                      float(entries["A04"] + entries["B03"]))
            cpx = union_cpx
        else:
            cpx = b.complex(f"9{len(b.complexes):02d}", name, "M" if gtfs[0] == "A" else "Bk", x, y,
                            float(entries[gtfs]))
        b.station(gtfs, name, gtfs[0], x, y, routes, cpx, hub=hub, terminal=term)

    # ---------------- nodes & platforms (A: directions S/N; B: E/W)
    for gtfs in ["A01", "A05", "A06", "A07", "A08"]:
        for d in ("S", "N"):
            b.node(gtfs, d, "L")
    for gtfs in ["A02", "A03", "A04"]:
        for d in ("S", "N"):
            b.node(gtfs, d, "L")
            b.node(gtfs, d, "X")
    for gtfs in ["B01", "B02", "B03", "B04", "B05"]:
        for d in ("E", "W"):
            b.node(gtfs, d, "L")
    cap = {"A02": 2400.0, "A04": 2200.0, "B03": 1800.0}
    for gtfs in ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08"]:
        for d in ("S", "N"):
            tracks = ("L", "X") if gtfs in ("A02", "A04") else ("L",)
            b.platform(gtfs, d, tracks, cap.get(gtfs, 1200.0))
    for gtfs in ["B01", "B02", "B03", "B04", "B05"]:
        for d in ("E", "W"):
            b.platform(gtfs, d, ("L",), cap.get(gtfs, 1400.0))

    # ---------------- run edges
    def pair(ga: str, gb: str, length: float, track: str = "L", vmax: float = 15.0, cbtc: bool = False,
             dirs: tuple[str, str] = ("S", "N")) -> list[int]:
        fwd, rev = dirs
        e1 = b.edge("run", b.n(ga, fwd, track), b.n(gb, fwd, track), track=track, length=length, vmax=vmax, cbtc=cbtc)
        e2 = b.edge("run", b.n(gb, rev, track), b.n(ga, rev, track), track=track, length=length, vmax=vmax, cbtc=cbtc)
        return [e1, e2]

    seg_edges: dict[tuple[str, str], list[int]] = {}
    seg_edges[("A01", "A02")] = pair("A01", "A02", 1200.0)
    seg_edges[("A02", "A03")] = pair("A02", "A03", 1000.0) + pair("A02", "A03", 1000.0, "X", 20.0)
    seg_edges[("A03", "A04")] = pair("A03", "A04", 1000.0) + pair("A03", "A04", 1000.0, "X", 20.0)
    seg_edges[("A04", "A05")] = pair("A04", "A05", 1000.0)
    seg_edges[("A05", "A06")] = pair("A05", "A06", 1300.0)
    seg_edges[("A06", "A08")] = pair("A06", "A08", 1300.0)
    seg_edges[("A05", "A07")] = pair("A05", "A07", 1300.0)
    seg_edges[("A07", "A08")] = pair("A07", "A08", 1300.0)
    bdir = ("E", "W")
    seg_edges[("B01", "B02")] = pair("B01", "B02", 1000.0, cbtc=True, dirs=bdir)
    seg_edges[("B02", "B03")] = pair("B02", "B03", 2400.0, vmax=18.0, cbtc=True, dirs=bdir)
    seg_edges[("B03", "B04")] = pair("B03", "B04", 1000.0, dirs=bdir)
    seg_edges[("B04", "B05")] = pair("B04", "B05", 2400.0, vmax=18.0, dirs=bdir)

    # ---------------- crossovers (standard placement at express stations), turnbacks
    il_edges: dict[str, list[int]] = {}
    for gtfs in ("A02", "A04"):
        for d in ("S", "N"):
            il_edges.setdefault(gtfs, []).append(b.edge("crossover", b.n(gtfs, d, "L"), b.n(gtfs, d, "X"), track="L"))
            il_edges.setdefault(gtfs, []).append(b.edge("crossover", b.n(gtfs, d, "X"), b.n(gtfs, d, "L"), track="X"))
    il_edges["A04"].append(b.edge("turnback", b.n("A04", "S"), b.n("A04", "N"), pocket=True))
    il_edges["A04"].append(b.edge("turnback", b.n("A04", "N"), b.n("A04", "S"), pocket=True))
    il_edges["A01"] = [b.edge("turnback", b.n("A01", "N"), b.n("A01", "S"))]
    il_edges["A08"] = [b.edge("turnback", b.n("A08", "S"), b.n("A08", "N"))]
    il_edges["B01"] = [b.edge("turnback", b.n("B01", "W", "L"), b.n("B01", "E", "L"))]
    il_edges["B05"] = [b.edge("turnback", b.n("B05", "E", "L"), b.n("B05", "W", "L"))]

    for (ga, gb), es in seg_edges.items():
        b.segment(ga, gb, es, river=(ga, gb) == ("B03", "B04"))

    # ---------------- yard (node appended after station nodes)
    yard_node = len(b.nodes)
    b.nodes.append({"idx": yard_node, "station": -1, "platform": -1, "track": "Y", "division": "B"})
    lead_edges = [
        b.edge("yard_lead", yard_node, b.n("B03", "E"), track="Y", run_s=420.0),
        b.edge("yard_lead", yard_node, b.n("B03", "W"), track="Y", run_s=420.0),
        b.edge("yard_lead", b.n("B03", "E"), yard_node, track="Y", run_s=420.0),
        b.edge("yard_lead", b.n("B03", "W"), yard_node, track="Y", run_s=420.0),
    ]
    yx, yy = 3600.0, 1800.0
    yards = [{"idx": 0, "name": "Toy Yard", "division": "B", "x": yx, "y": yy, "node": yard_node, "fleet_cap": 4,
              "deadhead_s": 420, "lead_edges": lead_edges, "routes": [rl]}]

    # ---------------- patterns
    n = b.n
    p1s = b.pattern(r1, [(n("A01", "S"), "", 0), (n("A02", "S"), "run", 1), (n("A03", "S"), "run", 1),
                         (n("A04", "S"), "run", 1), (n("A05", "S"), "run", 1), (n("A06", "S"), "run", 1),
                         (n("A08", "S"), "run", 1)], 120)
    p1n = b.pattern(r1, [(n("A08", "N"), "", 0), (n("A06", "N"), "run", 1), (n("A05", "N"), "run", 1),
                         (n("A04", "N"), "run", 1), (n("A03", "N"), "run", 1), (n("A02", "N"), "run", 1),
                         (n("A01", "N"), "run", 1)], 120)
    p2s = b.pattern(r2, [(n("A01", "S"), "", 0), (n("A02", "S"), "run", 0), (n("A02", "S", "X"), "crossover", 1),
                         (n("A03", "S", "X"), "run", 0), (n("A04", "S", "X"), "run", 1),
                         (n("A04", "S"), "crossover", 0), (n("A05", "S"), "run", 1), (n("A07", "S"), "run", 1),
                         (n("A08", "S"), "run", 1)], 120)
    p2n = b.pattern(r2, [(n("A08", "N"), "", 0), (n("A07", "N"), "run", 1), (n("A05", "N"), "run", 1),
                         (n("A04", "N"), "run", 1), (n("A04", "N", "X"), "crossover", 0),
                         (n("A03", "N", "X"), "run", 0), (n("A02", "N", "X"), "run", 1),
                         (n("A02", "N"), "crossover", 0), (n("A01", "N"), "run", 1)], 120)
    ple = b.pattern(rl, [(n("B01", "E"), "", 0), (n("B02", "E"), "run", 1), (n("B03", "E"), "run", 1),
                         (n("B04", "E"), "run", 1), (n("B05", "E"), "run", 1)], 160)
    plw = b.pattern(rl, [(n("B05", "W"), "", 0), (n("B04", "W"), "run", 1), (n("B03", "W"), "run", 1),
                         (n("B02", "W"), "run", 1), (n("B01", "W"), "run", 1)], 160)
    b.patterns[p1s]["next_patterns"] = [p1n]
    b.patterns[p1n]["next_patterns"] = [p1s]
    b.patterns[p2s]["next_patterns"] = [p2n]
    b.patterns[p2n]["next_patterns"] = [p2s]
    b.patterns[ple]["next_patterns"] = [plw]
    b.patterns[plw]["next_patterns"] = [ple]

    # ---------------- interlockings (one per station with crossover/turnback/diverge)
    interlockings = []
    diverge_st = {"A05": [], "A08": []}
    for gtfs in ["A01", "A02", "A04", "A05", "A08", "B01", "B05"]:
        es = il_edges.get(gtfs, [])
        kinds: list[str] = []
        for e in es:
            ed = b.edges[e]
            if ed["kind"] == "crossover":
                kinds.append("L2X" if b.nodes[ed["from"]]["track"] == "L" else "X2L")
            elif ed["kind"] == "turnback":
                kinds.append("turnback")
        if gtfs in diverge_st:
            kinds.append("junction")
        interlockings.append({"idx": len(interlockings), "station": b.st_key[gtfs], "edges": sorted(es),
                              "kinds": sorted(set(kinds))})

    # ---------------- service (headway_s, phase_s) per profile
    # Routes 1 and 2 share the L track at both ends of the trunk; route 2's express run is
    # ~33 s faster in between, so interleaving them at half a headway would bunch them on
    # one of the shared sections. Offsetting the phases by half of that keeps both shared
    # sections near-even (like a real timetable written around the merge).
    merge_s = 16.5

    def svc(h1: float, h2: float, hl: float) -> list[dict[str, float]]:
        out = []
        for p, h, ph in ((p1s, h1, 0.0), (p1n, h1, 0.5 * h1 - merge_s), (p2s, h2, 0.5 * h2 + merge_s),
                         (p2n, h2, 0.0), (ple, hl, 0.0), (plw, hl, 0.5 * hl)):
            out.append({"pattern": p, "headway_s": float(h), "phase_s": float(ph % h)})
        return out

    service = {"am_peak": svc(300, 300, 240), "midday": svc(480, 480, 360), "pm_rush": svc(300, 300, 240),
               "late_night": svc(1200, 1200, 1200)}

    # ---------------- demand
    base_rate = {"A01": 0.20, "A02": 0.55, "A03": 0.22, "A04": 0.45, "A05": 0.15, "A06": 0.10, "A07": 0.10,
                 "A08": 0.18, "B01": 0.14, "B02": 0.18, "B03": 0.40, "B04": 0.15, "B05": 0.12}
    scale = {"am_peak": 1.0, "midday": 0.55, "pm_rush": 1.05, "late_night": 0.08}
    demand: dict[str, dict[str, list[float]]] = {}
    for prof, sc in scale.items():
        rates = []
        for p in b.platforms:
            gtfs = b.stations[p["station"]]["gtfs"]
            rates.append(round(base_rate[gtfs] * sc * 0.5, 4))
        attr = [base_rate[s["gtfs"]] for s in b.stations]
        tot = sum(attr)
        demand[prof] = {"platform_rate": rates, "attraction": [round(a / tot, 6) for a in attr]}

    presets = [
        {"id": "toy-union-closure", "title": "Close Union Junction", "subtitle": "1 2 · Toy trunk",
         "profile": "am_peak",
         "disruptions": [{"kind": "full_station_closure", "target": {"type": "station", "idx": b.st_key["A04"]},
                          "duration_s": 1800}],
         "focus": {"x": 0.0, "y": 3000.0, "zoom_m_per_px": 6.0}},
        {"id": "toy-river-block", "title": "Block the river tunnel", "subtitle": "L · Toy Canarsie",
         "profile": "am_peak",
         "disruptions": [{"kind": "local_block", "target": {"type": "segment", "idx": 10}, "duration_s": 1800}],
         "focus": {"x": 3000.0, "y": 1300.0, "zoom_m_per_px": 6.0}},
    ]
    assert b.segments[10]["a"] == b.st_key["B03"], "segment index of the river tunnel moved"

    data: dict[str, Any] = {
        "schema_version": 1,
        "graph_version": "",
        "projection": {"lat0": LAT0, "lon0": LON0, "kx": KX, "ky": KY},
        "routes": b.routes,
        "complexes": b.complexes,
        "stations": b.stations,
        "platforms": b.platforms,
        "nodes": b.nodes,
        "edges": b.edges,
        "segments": b.segments,
        "interlockings": interlockings,
        "yards": yards,
        "patterns": b.patterns,
        "service": service,
        "demand": demand,
        "peak_dir": [],
        "presets": presets,
    }
    data["graph_version"] = canonical_graph_version(data)
    return copy.deepcopy(data)


def make_toy_graph() -> Graph:
    """Build the toy :class:`Graph` (same schema and loader path as ``network.json``)."""
    return graph_from_dict(toy_network_dict(), source="<toy>")


def make_toy_variant(*, cbtc: bool | None = None, demand_scale: float = 1.0) -> Graph:
    """A toy graph with every B-division run edge CBTC (``cbtc=True``) or fixed-block (``False``).

    Used for A/B experiments (signaling) on otherwise identical networks.
    """
    data = toy_network_dict()
    if cbtc is not None:
        for e in data["edges"]:
            if e["kind"] == "run" and data["nodes"][e["from"]]["division"] == "B":
                e["cbtc"] = bool(cbtc)
    if demand_scale != 1.0:
        for prof in data["demand"].values():
            prof["platform_rate"] = [r * demand_scale for r in prof["platform_rate"]]
    data["graph_version"] = canonical_graph_version(data)
    return graph_from_dict(data, source=f"<toy cbtc={cbtc}>")


#: station index by GTFS id in the toy graph
TOY_STATION: dict[str, int] = {
    "A01": 0, "A02": 1, "A03": 2, "A04": 3, "A05": 4, "A06": 5, "A07": 6, "A08": 7,
    "B01": 8, "B02": 9, "B03": 10, "B04": 11, "B05": 12,
}
#: pattern index by name: route + direction (S/N southbound/northbound, E/W eastbound/westbound)
TOY_PATTERN: dict[str, int] = {"1S": 0, "1N": 1, "2S": 2, "2N": 3, "LE": 4, "LW": 5}


def toy_segment(g: Graph, a_gtfs: str, b_gtfs: str) -> int:
    """Index of the toy segment between two stations (either order)."""
    a, b = TOY_STATION[a_gtfs], TOY_STATION[b_gtfs]
    for i in range(g.n_segments):
        if {int(g.segment_a[i]), int(g.segment_b[i])} == {a, b}:
            return i
    raise KeyError((a_gtfs, b_gtfs))


def pattern_node(g: Graph, pattern: str | int, gtfs: str, track: str | None = None) -> int:
    """The node where ``pattern`` visits station ``gtfs`` (optionally on track "L"/"X")."""
    p = TOY_PATTERN[pattern] if isinstance(pattern, str) else int(pattern)
    s = TOY_STATION[gtfs]
    want = {"L": TRACK_L, "X": TRACK_X, None: None}[track]
    edges = g.pattern_edges(p)
    nodes = [int(g.edge_from[edges[0]])] + [int(g.edge_to[e]) for e in edges]
    for n in nodes:
        if int(g.node_station[n]) == s and (want is None or int(g.node_track[n]) == want):
            return n
    raise KeyError((pattern, gtfs, track))


def run_until(sim: Any, cond: Callable[[], bool], max_ticks: int) -> bool:
    """Step ``sim`` one tick at a time until ``cond()`` holds (checked before each step)."""
    for _ in range(int(max_ticks)):
        if cond():
            return True
        sim.step(1)
    return bool(cond())


def first_decision(
    sim: Any,
    action: int | None = None,
    *,
    station: int | None = None,
    route: str | None = None,
    max_ticks: int = 240,
) -> tuple[int, int]:
    """Step until a pending decision matches; returns ``(train_id, batch_row)``.

    Matches decisions where ``action`` is valid (any decision if ``None``), optionally at
    ``station`` and for route id ``route``. Raises ``AssertionError`` if none appears.
    """
    for _ in range(int(max_ticks)):
        b = sim.pending_decisions()
        for j, t in enumerate(b.train_ids.tolist()):
            if action is not None and not b.masks[j, action]:
                continue
            if station is not None and b.context[j]["station_idx"] != station:
                continue
            if route is not None and sim.g.route_id[b.context[j]["route_idx"]] != route:
                continue
            return int(t), j
        sim.step(1)
    raise AssertionError(f"no matching decision within {max_ticks} ticks (action={action}, station={station})")


def decisions_for(sim: Any, train_id: int) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """The pending mask row and decision context of ``train_id`` (``(None, None)`` if none)."""
    batch = sim.pending_decisions()
    hit = np.flatnonzero(batch.train_ids == int(train_id))
    if hit.size == 0:
        return None, None
    return batch.masks[int(hit[0])], batch.context[int(hit[0])]
