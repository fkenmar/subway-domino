"""Invariants of the built artifacts (ARCHITECTURE §2.6, §2.7), shared by ``--check`` and the tests.

Besides the contract invariants, every preset disruption must affect in-service patterns and leave
a structural dispatch maneuver upstream (:mod:`scripts.netbuild.maneuvers`).

``check_network`` and ``check_geo`` return ``(errors, warnings)``: errors are contract violations
(the build fails), warnings are soft targets (artifact size budgets).
"""

from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from scripts.netbuild.constants import EXPECTED_COMPLEXES, EXPECTED_ROUTES, EXPECTED_STATIONS, PROFILES
from scripts.netbuild.infra import KIND_TARGETS
from scripts.netbuild.jsonio import content_hash
from scripts.netbuild.maneuvers import check_preset_maneuvers

NODE_RANGE = (1100, 1700)
RUN_EDGE_RANGE = (1100, 1800)
SEGMENT_RANGE = (400, 560)
MIN_XOVER_INTERLOCKINGS = 40
MIN_POCKET_STATIONS = 30
EXPECTED_YARDS = 10
AM_PEAK_ENTRIES_H = (250_000.0, 500_000.0)
LATE_NIGHT_MAX_ENTRIES_H = 60_000.0
PRESET_COUNT = (6, 8)
NETWORK_MAX_BYTES = 3_000_000
GEO_MAX_BYTES = 400_000
GEO_MAX_GZIP_BYTES = 120_000
V_MAX_RANGE = (8.0, 24.6)


def _indexed(items: Sequence[Mapping[str, Any]], name: str, errors: list[str]) -> None:
    bad = [i for i, it in enumerate(items) if it.get("idx") != i]
    if bad:
        errors.append(f"{name}: idx != position at {bad[:5]}")


def system_entries_h(doc: Mapping[str, Any], profile: str) -> float:
    return float(sum(doc["demand"][profile]["platform_rate"]) * 3600.0)


def check_network(doc: Mapping[str, Any], *, routes_without_am_service: Iterable[str] = (),
                  size_bytes: int | None = None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    e_ = errors.append
    if doc.get("schema_version") != 1:
        e_("schema_version must be 1")
    if doc.get("graph_version") != content_hash(dict(doc)):
        e_("graph_version does not match the content hash")
    routes, complexes, stations = doc["routes"], doc["complexes"], doc["stations"]
    platforms, nodes, edges = doc["platforms"], doc["nodes"], doc["edges"]
    segments, interlockings, yards, patterns = doc["segments"], doc["interlockings"], doc["yards"], doc["patterns"]
    for name in ("routes", "complexes", "stations", "platforms", "nodes", "edges", "segments", "interlockings",
                 "yards", "patterns"):
        _indexed(doc[name], name, errors)
    n_st, n_pl, n_n, n_e = len(stations), len(platforms), len(nodes), len(edges)

    # ---------------------------------------------------------------- counts (§2.6)
    if len(routes) != EXPECTED_ROUTES:
        e_(f"{len(routes)} routes, expected {EXPECTED_ROUTES}")
    if n_st != EXPECTED_STATIONS:
        e_(f"{n_st} stations, expected {EXPECTED_STATIONS}")
    if len(complexes) != EXPECTED_COMPLEXES:
        e_(f"{len(complexes)} complexes, expected {EXPECTED_COMPLEXES}")
    run_edges = [e for e in edges if e["kind"] == "run"]
    for label, value, (lo, hi) in (("nodes", n_n, NODE_RANGE), ("run edges", len(run_edges), RUN_EDGE_RANGE),
                                   ("segments", len(segments), SEGMENT_RANGE)):
        if not lo <= value <= hi:
            e_(f"{value} {label}, expected {lo}..{hi}")
    if len(yards) != EXPECTED_YARDS:
        e_(f"{len(yards)} yards, expected exactly {EXPECTED_YARDS}")

    # ---------------------------------------------------------------- complexes, stations, platforms, nodes
    seen: Counter[int] = Counter(s for c in complexes for s in c["stations"])
    if set(seen) != set(range(n_st)) or any(v != 1 for v in seen.values()):
        e_("complexes do not partition the stations")
    for s in stations:
        if not 0 <= s["complex"] < len(complexes) or s["idx"] not in complexes[s["complex"]]["stations"]:
            e_(f"station {s['gtfs']}: complex {s['complex']} does not list it")
        if not s["routes"] or any(not 0 <= r < len(routes) for r in s["routes"]):
            e_(f"station {s['gtfs']}: no routes or a bad route index")
        if s["division"] not in ("A", "B"):
            e_(f"station {s['gtfs']}: bad division")
    node_station = [n["station"] for n in nodes]
    node_div = [n["division"] for n in nodes]
    for pl in platforms:
        for n in pl["nodes"]:
            if nodes[n]["platform"] != pl["idx"] or nodes[n]["station"] != pl["station"]:
                e_(f"platform {pl['idx']}: node {n} does not point back to it")
        if not 350.0 <= pl["capacity"] <= 3500.0:
            e_(f"platform {pl['idx']}: capacity {pl['capacity']} outside 350..3500")
    for n in nodes:
        if n["platform"] >= 0 and n["idx"] not in platforms[n["platform"]]["nodes"]:
            e_(f"node {n['idx']}: platform {n['platform']} does not list it")
        if n["station"] >= 0 and n["division"] != stations[n["station"]]["division"]:
            e_(f"node {n['idx']}: division differs from its station")
    stations_with_nodes = {s for s in node_station if s >= 0}
    if stations_with_nodes != set(range(n_st)):
        e_(f"stations without nodes: {sorted(set(range(n_st)) - stations_with_nodes)[:10]}")

    # ---------------------------------------------------------------- edges
    touched = set()
    for e in edges:
        a, b = e["from"], e["to"]
        if not (0 <= a < n_n and 0 <= b < n_n):
            e_(f"edge {e['idx']}: node index out of range")
            continue
        touched.update((a, b))
        if node_div[a] != node_div[b]:
            e_(f"edge {e['idx']}: crosses divisions ({node_div[a]} -> {node_div[b]})")
        kind = e["kind"]
        if kind == "run":
            if not (0 <= e["segment"] < len(segments)) or e["seg_dir"] not in (1, -1):
                e_(f"run edge {e['idx']}: bad segment/seg_dir")
            if e["length_m"] <= 0 or e["run_s"] <= 0 or not V_MAX_RANGE[0] <= e["v_max"] <= V_MAX_RANGE[1]:
                e_(f"run edge {e['idx']}: bad length/run_s/v_max")
            if e["track"] != nodes[a]["track"] or e["track"] != nodes[b]["track"] or e["track"] not in ("L", "X"):
                e_(f"run edge {e['idx']}: track {e['track']} differs from its nodes")
            if node_station[a] == node_station[b]:
                e_(f"run edge {e['idx']}: does not change station")
            if e["pocket"]:
                e_(f"run edge {e['idx']}: pocket must be false")
        else:
            if e["segment"] != -1 or e["length_m"] != 0.0 or e["run_s"] <= 0:
                e_(f"{kind} edge {e['idx']}: must have segment -1, length 0, run_s > 0")
            if kind == "yard_lead":
                want_dir = 1 if node_station[a] == -1 else -1  # +1 pull-out (yard -> station), -1 put-in
            else:
                want_dir = 0
            if e["seg_dir"] != want_dir:
                e_(f"{kind} edge {e['idx']}: seg_dir {e['seg_dir']}, expected {want_dir}")
            if kind in ("crossover", "turnback") and node_station[a] != node_station[b]:
                e_(f"{kind} edge {e['idx']}: endpoints at different stations")
            if kind == "crossover" and {nodes[a]["track"], nodes[b]["track"]} != {"L", "X"}:
                e_(f"crossover edge {e['idx']}: must connect L and X")
            if kind == "yard_lead" and (node_station[a] == -1) == (node_station[b] == -1):
                e_(f"yard_lead edge {e['idx']}: must connect a yard node and a station node")
            if kind not in ("crossover", "turnback", "yard_lead"):
                e_(f"edge {e['idx']}: unknown kind {kind!r}")
    if touched != set(range(n_n)):
        e_(f"nodes without any edge: {sorted(set(range(n_n)) - touched)[:10]}")

    # ---------------------------------------------------------------- segments, interlockings, yards
    for seg in segments:
        if seg["a"] == seg["b"] or not (0 <= seg["a"] < n_st and 0 <= seg["b"] < n_st):
            e_(f"segment {seg['idx']}: bad endpoints")
        for ei in seg["edges"]:
            e = edges[ei]
            if e["kind"] != "run" or e["segment"] != seg["idx"]:
                e_(f"segment {seg['idx']}: lists edge {ei} that is not its run edge")
                continue
            ends = (node_station[e["from"]], node_station[e["to"]])
            expect = (seg["a"], seg["b"]) if e["seg_dir"] == 1 else (seg["b"], seg["a"])
            if ends != expect:
                e_(f"segment {seg['idx']}: edge {ei} seg_dir does not match its stations")
        if seg["has_express"] != any(edges[ei]["track"] == "X" for ei in seg["edges"]):
            e_(f"segment {seg['idx']}: has_express inconsistent")
    listed = sorted(ei for seg in segments for ei in seg["edges"])
    if listed != sorted(e["idx"] for e in run_edges):
        e_("segments do not partition the run edges")
    kind_count: Counter[str] = Counter(k for il in interlockings for k in il["kinds"])
    for k in ("L2X", "X2L"):
        if kind_count[k] < MIN_XOVER_INTERLOCKINGS:
            e_(f"{k} at {kind_count[k]} interlockings, expected >= {MIN_XOVER_INTERLOCKINGS}")
    pocket_stations = {node_station[e["from"]] for e in edges if e["kind"] == "turnback" and e["pocket"]}
    if len(pocket_stations) < MIN_POCKET_STATIONS:
        e_(f"{len(pocket_stations)} stations with pocket turnbacks, expected >= {MIN_POCKET_STATIONS}")
    il_stations = [il["station"] for il in interlockings]
    if len(set(il_stations)) != len(il_stations):
        e_("more than one interlocking per station")
    xover_tb_stations = {node_station[e["from"]] for e in edges if e["kind"] in ("crossover", "turnback")}
    if not xover_tb_stations <= set(il_stations):
        e_("a station with crossover/turnback edges has no interlocking")
    for y in yards:
        node = nodes[y["node"]] if 0 <= y["node"] < n_n else None
        if node is None or node["station"] != -1 or node["division"] != y["division"]:
            e_(f"yard {y['name']}: bad yard node")
            continue
        if not y["lead_edges"]:
            e_(f"yard {y['name']}: no lead edges")
        for ei in y["lead_edges"]:
            e = edges[ei]
            if e["kind"] != "yard_lead" or y["node"] not in (e["from"], e["to"]):
                e_(f"yard {y['name']}: lead edge {ei} does not touch the yard node")
        if not 4 <= y["fleet_cap"] <= 8 or not 300 <= y["deadhead_s"] <= 720:
            e_(f"yard {y['name']}: fleet_cap/deadhead_s outside the contract ranges")

    # ---------------------------------------------------------------- patterns
    for p in patterns:
        pe, ps = p["edges"], p["stops"]
        if not pe or len(pe) != len(ps):
            e_(f"pattern {p['idx']}: edges/stops length mismatch")
            continue
        for x, y in zip(pe, pe[1:]):
            if edges[x]["to"] != edges[y]["from"]:
                e_(f"pattern {p['idx']}: edges {x} -> {y} are not consecutive")
                break
        for ei, s in zip(pe, ps):
            if s and nodes[edges[ei]["to"]]["platform"] < 0:
                e_(f"pattern {p['idx']}: stop at node {edges[ei]['to']} without a platform")
        if nodes[edges[pe[0]]["from"]]["platform"] < 0 or not ps[-1]:
            e_(f"pattern {p['idx']}: origin or terminal is not a platform stop")
        if node_station[edges[pe[0]]["from"]] != p["origin"] or node_station[edges[pe[-1]]["to"]] != p["dest"]:
            e_(f"pattern {p['idx']}: origin/dest do not match its edges")
        if any(edges[ei]["kind"] not in ("run", "crossover") for ei in pe):
            e_(f"pattern {p['idx']}: uses a turnback or yard edge")
        for q in p["next_patterns"]:
            if not 0 <= q < len(patterns) or patterns[q]["route"] != p["route"] or patterns[q]["origin"] != p["dest"]:
                e_(f"pattern {p['idx']}: bad next pattern {q}")

    # ---------------------------------------------------------------- service, demand, peak_dir
    service, demand = doc["service"], doc["demand"]
    if set(service) != set(PROFILES) or set(demand) != set(PROFILES):
        e_("service and demand must have exactly the four profiles")
    for prof in PROFILES:
        for row in service.get(prof, []):
            if not 0 <= row["pattern"] < len(patterns) or row["headway_s"] <= 0 or not 0 <= row["phase_s"] < row["headway_s"]:
                e_(f"service {prof}: bad row {row}")
        dp = demand.get(prof, {})
        if len(dp.get("platform_rate", [])) != n_pl or len(dp.get("attraction", [])) != n_st:
            e_(f"demand {prof}: vector lengths do not match platforms/stations")
            continue
        if min(dp["platform_rate"]) < 0 or min(dp["attraction"]) < 0 or abs(sum(dp["attraction"]) - 1.0) > 1e-4:
            e_(f"demand {prof}: negative values or attraction does not sum to 1")
    served = {patterns[r["pattern"]]["route"] for r in service.get("am_peak", [])}
    allowed = set(routes_without_am_service)
    for r in routes:
        if r["idx"] not in served and r["id"] not in allowed:
            e_(f"route {r['id']} has no am_peak service and is not recorded as having none")
    am, late = system_entries_h(doc, "am_peak"), system_entries_h(doc, "late_night")
    if not AM_PEAK_ENTRIES_H[0] <= am <= AM_PEAK_ENTRIES_H[1]:
        e_(f"am_peak system entries {am:.0f}/h outside {AM_PEAK_ENTRIES_H}")
    if not late < LATE_NIGHT_MAX_ENTRIES_H:
        e_(f"late_night system entries {late:.0f}/h >= {LATE_NIGHT_MAX_ENTRIES_H}")
    groups: dict[int, set[int]] = defaultdict(set)
    for e in edges:
        if e["bidir_group"] >= 0:
            groups[e["bidir_group"]].add(e["idx"])
            if e["kind"] != "run" or e["track"] != "X":
                e_(f"edge {e['idx']}: only X run edges may have a bidir_group")
    covered = set()
    for pd in doc["peak_dir"]:
        g = pd["bidir_group"]
        covered.add((g, pd["profile"]))
        if not set(pd["allowed_dir_edges"]) <= groups.get(g, set()) or not pd["allowed_dir_edges"]:
            e_(f"peak_dir group {g} {pd['profile']}: allowed edges are not X edges of the group")
    if covered != {(g, prof) for g in groups for prof in PROFILES}:
        e_("peak_dir must cover every bidir group in every profile")

    # ---------------------------------------------------------------- presets
    presets = doc["presets"]
    if not PRESET_COUNT[0] <= len(presets) <= PRESET_COUNT[1]:
        e_(f"{len(presets)} presets, expected {PRESET_COUNT[0]}..{PRESET_COUNT[1]}")
    preset_errors = check_presets(doc)
    errors.extend(preset_errors)
    if not preset_errors:  # the maneuver analysis needs resolvable targets
        errors.extend(check_preset_maneuvers(doc))
    if len({p["id"] for p in presets}) != len(presets):
        e_("preset ids are not unique")

    if size_bytes is not None and size_bytes > NETWORK_MAX_BYTES:
        warnings.append(f"network.json is {size_bytes} bytes (target <= {NETWORK_MAX_BYTES})")
    return errors, warnings


def check_presets(doc: Mapping[str, Any]) -> list[str]:
    errors = []
    limits = {"station": len(doc["stations"]), "complex": len(doc["complexes"]), "segment": len(doc["segments"]),
              "interlocking": len(doc["interlockings"])}
    for p in doc["presets"]:
        if p["profile"] not in PROFILES:
            errors.append(f"preset {p['id']}: unknown profile")
        if not p["disruptions"]:
            errors.append(f"preset {p['id']}: no disruptions")
        for d in p["disruptions"]:
            t = d["target"]
            if d["kind"] not in KIND_TARGETS or t["type"] not in KIND_TARGETS[d["kind"]]:
                errors.append(f"preset {p['id']}: {d['kind']} cannot target {t['type']}")
            elif not 0 <= t["idx"] < limits[t["type"]]:
                errors.append(f"preset {p['id']}: {t['type']} index {t['idx']} out of range")
            elif d["kind"] == "express_block" and not _has_express(doc, t["type"], t["idx"]):
                errors.append(f"preset {p['id']}: express_block target has no express track")
        focus = p.get("focus") or {}
        if not {"x", "y", "zoom_m_per_px"} <= set(focus) or focus["zoom_m_per_px"] <= 0:
            errors.append(f"preset {p['id']}: bad focus")
    return errors


def _has_express(doc: Mapping[str, Any], typ: str, idx: int) -> bool:
    if typ == "segment":
        return bool(doc["segments"][idx]["has_express"])
    station = idx if typ == "station" else doc["interlockings"][idx]["station"]
    return any(n["station"] == station and n["track"] == "X" for n in doc["nodes"])


def check_geo(geo: Mapping[str, Any], net: Mapping[str, Any], *, size_bytes: int | None = None,
              gzip_bytes: int | None = None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    e_ = errors.append
    if geo.get("schema_version") != 1 or geo.get("graph_version") != net.get("graph_version"):
        e_("geo schema_version/graph_version do not match network.json")
    for name in ("routes", "complexes", "stations", "segments", "edges", "interlockings", "yards", "presets"):
        if len(geo[name]) != len(net[name]):
            e_(f"geo {name}: {len(geo[name])} entries, network has {len(net[name])}")
            return errors, warnings
        if name != "presets":
            _indexed(geo[name], f"geo {name}", errors)
    mirrors = {
        "routes": ("id", "name", "color", "text_color", "division"),
        "complexes": ("name", "stations", "weekday_entries"),
        "stations": ("gtfs", "name", "complex", "routes", "hub"),
        "segments": ("a", "b", "name", "river", "has_express"),
        "edges": ("kind", "segment", "seg_dir", "track"),
        "interlockings": ("station", "kinds"),
        "yards": ("name", "division"),
    }
    for name, keys in mirrors.items():
        for g, n in zip(geo[name], net[name]):
            diff = [k for k in keys if g[k] != n[k]]
            if diff:
                e_(f"geo {name}[{g['idx']}]: {diff} differ from network.json")
                break
    if geo["presets"] != net["presets"]:
        e_("geo presets differ from network.json")
    stations = geo["stations"]
    ranks = sorted(s["label_rank"] for s in stations)
    if ranks != list(range(len(stations))):
        e_("label_rank is not a permutation of 0..n-1")
    for s, n in zip(stations, net["stations"]):
        if not (isinstance(s["x"], int) and isinstance(s["y"], int)) or abs(s["x"] - n["x"]) > 0.51 or abs(s["y"] - n["y"]) > 0.51:
            e_(f"geo station {s['idx']}: coordinates are not the rounded network coordinates")
            break
    for seg in geo["segments"]:
        pts = seg["pts"]
        if len(pts) < 2 or not all(isinstance(v, int) for pt in pts for v in pt):
            e_(f"geo segment {seg['idx']}: needs >= 2 integer points")
            continue
        a, b = stations[seg["a"]], stations[seg["b"]]
        if pts[0] != [a["x"], a["y"]] or pts[-1] != [b["x"], b["y"]]:
            e_(f"geo segment {seg['idx']}: polyline does not run from station a to station b")
        if any(not 0 <= r < len(geo["routes"]) for r in seg["routes"]) or not seg["routes"]:
            e_(f"geo segment {seg['idx']}: bad routes")
    node_station = [n["station"] for n in net["nodes"]]
    for g, n in zip(geo["edges"], net["edges"]):
        if n["kind"] in ("crossover", "turnback"):
            if g["station"] != node_station[n["from"]] or g["yard"] != -1:
                e_(f"geo edge {g['idx']}: station must be the {n['kind']} station")
        elif n["kind"] == "yard_lead":
            lead = node_station[n["to"]] if node_station[n["from"]] == -1 else node_station[n["from"]]
            if not 0 <= g["yard"] < len(geo["yards"]) or g["station"] != lead:
                e_(f"geo edge {g['idx']}: yard_lead needs its yard and lead station")
        elif g["station"] != -1 or g["yard"] != -1:
            e_(f"geo edge {g['idx']}: run edges have station -1 and yard -1")
    for il in geo["interlockings"]:
        s = stations[il["station"]]
        if (il["x"], il["y"]) != (s["x"], s["y"]):
            e_(f"geo interlocking {il['idx']}: not at its station")
    xmin, ymin, xmax, ymax = geo["bbox"]
    if any(not (xmin <= s["x"] <= xmax and ymin <= s["y"] <= ymax) for s in stations):
        e_("bbox does not contain every station")
    basemap = geo.get("basemap")
    if basemap is not None:
        land = basemap.get("land")
        if not isinstance(land, list) or any(len(r) < 3 for r in land):
            e_("basemap.land must be a list of rings with >= 3 points")
    if size_bytes is not None and size_bytes > GEO_MAX_BYTES:
        warnings.append(f"network_geo.json is {size_bytes} bytes (target <= {GEO_MAX_BYTES})")
    if gzip_bytes is not None and gzip_bytes > GEO_MAX_GZIP_BYTES:
        warnings.append(f"network_geo.json gzips to {gzip_bytes} bytes (target <= {GEO_MAX_GZIP_BYTES})")
    return errors, warnings


def gzip_size(data: bytes) -> int:
    return len(gzip.compress(data, compresslevel=9, mtime=0))


def load(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
