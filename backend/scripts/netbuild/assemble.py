"""Assemble ``network.json`` (§2.6) and ``network_geo.json`` (§2.7) from a :class:`NetworkModel`.

Both documents share every index. Floats are rounded (coordinates 0.1 m in ``network.json`` and
1 m integers in the geo file, rates 1e-4 riders/s, attraction shares 1e-7) and ``graph_version``
is the first 12 hex digits of the sha1 of the canonical ``network.json`` with ``graph_version`` "".

``yard_lead`` edges carry ``seg_dir`` +1 for a pull-out (yard -> lead station) and -1 for a put-in,
and their geo ``station`` is the lead station (§2.6, §2.7).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from scripts.netbuild.constants import (
    A_DIVISION_ROUTES,
    CAPACITY_DECIMALS,
    COORD_DECIMALS,
    ENTRIES_DECIMALS,
    KX,
    KY,
    LAT0,
    LENGTH_DECIMALS,
    LON0,
    PROFILES,
    RATE_DECIMALS,
    SCHEMA_VERSION,
    SHARE_DECIMALS,
    SPEED_DECIMALS,
    TIME_DECIMALS,
)
from scripts.netbuild.geo import project
from scripts.netbuild.geometry import bbox
from scripts.netbuild.jsonio import content_hash, rnd
from scripts.netbuild.pipeline import NetworkModel, shuttle

PROJECTION = {"lat0": LAT0, "lon0": LON0, "kx": KX, "ky": KY}


def _color(hex6: str) -> str:
    return "#" + hex6.strip().lstrip("#").upper()


def _xy(xy: tuple[float, float], decimals: int = COORD_DECIMALS) -> tuple[float, float]:
    return rnd(xy[0], decimals), rnd(xy[1], decimals)


def _iround(v: float) -> int:
    """Round half up to an integer meter (the same result in Python and JavaScript ``Math.round``)."""
    return int(math.floor(float(v) + 0.5))


def _ixy(xy: tuple[float, float]) -> tuple[int, int]:
    return _iround(xy[0]), _iround(xy[1])


def _dedup(pts: list[list[int]]) -> list[list[int]]:
    return [pts[0]] + [q for prev, q in zip(pts, pts[1:]) if q != prev]


def build_network_doc(m: NetworkModel) -> dict[str, Any]:
    topo, sidx = m.topo, m.station_idx
    routes = [
        {"idx": i, "id": r.route_id, "name": r.long_name, "color": _color(r.color), "text_color": _color(r.text_color),
         "division": "A" if r.route_id in A_DIVISION_ROUTES else "B",
         "shuttle": shuttle(r.route_id)}
        for i, r in enumerate(m.routes)
    ]
    weekday = m.demand.weekday_entries
    complexes = []
    for ci, c in enumerate(m.places.complexes):
        x, y = _xy(project(c.lat, c.lon))
        complexes.append({"idx": ci, "complex_id": c.complex_id, "name": c.name, "borough": c.borough, "x": x, "y": y,
                          "stations": [sidx[s] for s in c.stations],
                          "weekday_entries": rnd(float(weekday[ci]), ENTRIES_DECIMALS)})
    stations = []
    for i, s in enumerate(m.station_ids):
        meta = m.places.stations[s]
        x, y = _xy(m.station_xy[s])
        stations.append({
            "idx": i, "gtfs": s, "name": meta.name, "complex": m.places.complex_of[s], "division": meta.division,
            "x": x, "y": y, "structure": meta.structure,
            "routes": sorted(m.route_idx[r] for r in topo.station_routes.get(s, ())),
            "hub": bool(m.hub[i]), "terminal": s in topo.terminal_stations,
        })
    platforms = [{"idx": pl.idx, "station": sidx[pl.station], "nodes": list(pl.nodes),
                  "capacity": rnd(float(m.demand.capacity[pl.idx]), CAPACITY_DECIMALS)} for pl in topo.platforms]
    nodes = [{"idx": n.idx, "station": sidx[n.station] if n.station else -1, "platform": n.platform,
              "track": n.track, "division": n.division} for n in topo.nodes]
    seg_of_edge = {e: seg.idx for seg in topo.segments for e in seg.edges}
    edges = []
    for e in topo.edges:
        if e.kind == "run":
            seg = topo.segments[seg_of_edge[e.idx]]
            seg_dir = 1 if (e.hop[0], e.hop[1]) == (seg.a, seg.b) else -1
            edges.append({"idx": e.idx, "kind": "run", "from": e.src, "to": e.dst, "segment": seg.idx,
                          "seg_dir": seg_dir, "track": e.track,
                          "length_m": rnd(m.timing.length_m[e.idx], LENGTH_DECIMALS),
                          "run_s": rnd(m.timing.run_s[e.idx], TIME_DECIMALS),
                          "v_max": rnd(m.timing.v_max[e.idx], SPEED_DECIMALS),
                          "cbtc": e.cbtc, "bidir_group": e.bidir_group, "pocket": False})
        else:
            # yard_lead: +1 pull-out (yard -> station), -1 put-in; crossover/turnback: 0 (§2.6).
            seg_dir = 0 if e.kind != "yard_lead" else (1 if topo.nodes[e.src].yard >= 0 else -1)
            edges.append({"idx": e.idx, "kind": e.kind, "from": e.src, "to": e.dst, "segment": -1, "seg_dir": seg_dir,
                          "track": e.track, "length_m": 0.0, "run_s": rnd(e.run_s, TIME_DECIMALS), "v_max": 0.0,
                          "cbtc": False, "bidir_group": -1, "pocket": bool(e.pocket)})
    segments = []
    for seg in topo.segments:
        name = f"{m.places.stations[seg.a].name} – {m.places.stations[seg.b].name}"
        segments.append({"idx": seg.idx, "a": sidx[seg.a], "b": sidx[seg.b], "name": name,
                         "river": m.river[seg.idx],
                         "has_express": any(topo.edges[e].track == "X" for e in seg.edges),
                         "edges": list(seg.edges)})
    interlockings = [{"idx": il.idx, "station": sidx[il.station], "edges": list(il.edges), "kinds": list(il.kinds)}
                     for il in topo.interlockings]
    yards = []
    for y in topo.yards:
        x, yy = _xy(y.xy)
        yards.append({"idx": len(yards), "name": y.name, "division": y.division, "x": x, "y": yy, "node": y.node,
                      "fleet_cap": y.fleet_cap, "deadhead_s": y.deadhead_s, "lead_edges": list(y.lead_edges),
                      "routes": sorted(m.route_idx[r] for r in y.routes)})
    patterns = []
    for p in m.kept:
        path = topo.paths[p.pid]
        patterns.append({"idx": p.pid, "route": m.route_idx[p.route_id], "edges": list(path.edges),
                         "stops": [int(s) for s in path.stops], "origin": sidx[p.path[0]], "dest": sidx[p.path[-1]],
                         "trips_weekday": p.n_trips, "next_patterns": list(topo.next_patterns[p.pid])})
    demand = {}
    for prof in PROFILES:
        demand[prof] = {
            "platform_rate": [rnd(float(v), RATE_DECIMALS) for v in m.demand.platform_rate[prof]],
            "attraction": [rnd(float(v), SHARE_DECIMALS) for v in m.demand.attraction[prof]],
        }
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "graph_version": "",
        "projection": dict(PROJECTION),
        "routes": routes,
        "complexes": complexes,
        "stations": stations,
        "platforms": platforms,
        "nodes": nodes,
        "edges": edges,
        "segments": segments,
        "interlockings": interlockings,
        "yards": yards,
        "patterns": patterns,
        "service": {prof: m.service[prof] for prof in PROFILES},
        "demand": demand,
        "peak_dir": m.peak_dir,
        "presets": m.presets,
    }
    doc["graph_version"] = content_hash(doc)
    return doc


def build_geo_doc(m: NetworkModel, network: dict[str, Any]) -> dict[str, Any]:
    topo, sidx = m.topo, m.station_idx
    seg_routes: dict[int, set[int]] = {seg.idx: set() for seg in topo.segments}
    seg_of_edge = {e: seg.idx for seg in topo.segments for e in seg.edges}
    for p in m.kept:
        for e in topo.paths[p.pid].edges:
            if e in seg_of_edge:
                seg_routes[seg_of_edge[e]].add(m.route_idx[p.route_id])
    # Geo coordinates derive from network.json's 0.1 m values, so the two files agree exactly
    # (a station is at round(network x, y) in both).
    stations = []
    for ns in network["stations"]:
        x, y = _ixy((ns["x"], ns["y"]))
        stations.append({"idx": ns["idx"], "gtfs": ns["gtfs"], "name": ns["name"], "complex": ns["complex"], "x": x,
                         "y": y, "routes": ns["routes"], "hub": ns["hub"], "label_rank": int(m.label_rank[ns["idx"]])})
    segments = []
    for seg, nseg in zip(topo.segments, network["segments"]):
        pts = [list(_ixy((float(x), float(y)))) for x, y in m.geometry.pts[seg.idx]]
        a, b = stations[nseg["a"]], stations[nseg["b"]]
        pts[0], pts[-1] = [a["x"], a["y"]], [b["x"], b["y"]]  # polylines join exactly at station points
        segments.append({"idx": seg.idx, "a": nseg["a"], "b": nseg["b"], "name": nseg["name"], "river": nseg["river"],
                         "has_express": nseg["has_express"], "routes": sorted(seg_routes[seg.idx]),
                         "pts": _dedup(pts)})
    edges = []
    for e, ne in zip(topo.edges, network["edges"]):
        station = sidx[e.station] if e.kind in ("crossover", "turnback", "yard_lead") and e.station else -1
        edges.append({"idx": ne["idx"], "kind": ne["kind"], "segment": ne["segment"], "seg_dir": ne["seg_dir"],
                      "track": ne["track"], "station": station, "yard": e.yard if e.kind == "yard_lead" else -1})
    complexes = [{"idx": c["idx"], "name": c["name"], "x": _iround(c["x"]), "y": _iround(c["y"]),
                  "stations": c["stations"], "weekday_entries": c["weekday_entries"]} for c in network["complexes"]]
    interlockings = [{"idx": il["idx"], "station": il["station"],
                      "x": stations[il["station"]]["x"], "y": stations[il["station"]]["y"], "kinds": il["kinds"]}
                     for il in network["interlockings"]]
    yards = [{"idx": y["idx"], "name": y["name"], "x": _iround(y["x"]), "y": _iround(y["y"]),
              "division": y["division"]} for y in network["yards"]]
    routes = [{k: r[k] for k in ("idx", "id", "name", "color", "text_color", "division")} for r in network["routes"]]
    arrays = [np.asarray(s["pts"], dtype=np.float64) for s in segments]
    arrays.append(np.asarray([[s["x"], s["y"]] for s in stations], dtype=np.float64))
    if yards:
        arrays.append(np.asarray([[y["x"], y["y"]] for y in yards], dtype=np.float64))
    box = [int(np.floor(v)) if i < 2 else int(np.ceil(v)) for i, v in enumerate(bbox(arrays))]
    basemap = None
    if m.basemap:
        land = []
        for ring in m.basemap:
            pts = _dedup([list(_ixy((float(x), float(y)))) for x, y in ring])
            if len(pts) >= 3:
                land.append(pts)
        basemap = {"land": land}
    return {
        "schema_version": SCHEMA_VERSION,
        "graph_version": network["graph_version"],
        "projection": dict(PROJECTION),
        "bbox": box,
        "routes": routes,
        "complexes": complexes,
        "stations": stations,
        "segments": segments,
        "edges": edges,
        "interlockings": interlockings,
        "yards": yards,
        "presets": network["presets"],
        "basemap": basemap,
    }
