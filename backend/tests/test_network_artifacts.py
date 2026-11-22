"""WP1 artifact tests (ARCHITECTURE §2.6, §2.7, §8).

* every §2.6 invariant of the committed ``data/network.json``, one test per invariant;
* ``network_geo.json`` mirrors ``network.json`` index for index (§2.7) and the frontend copy is
  byte-identical;
* every preset resolves and leaves the dispatcher a physical maneuver upstream of each disruption;
* ``build --offline`` is deterministic: rebuilding from the ``data/raw/`` caches reproduces the
  committed artifacts byte for byte, in-process and in subprocesses with different hash seeds
  (skipped when the gitignored caches are absent);
* the CLI's ``--check`` and no-cache paths, plus unit tests of the de-quantization and snapping.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.netbuild import constants as K
from scripts.netbuild.artifacts import build_artifacts, load_routes_without_am_service
from scripts.netbuild.geometry import snap_ends
from scripts.netbuild.http import HttpClient
from scripts.netbuild.jsonio import content_hash
from scripts.netbuild.maneuvers import footprint, upstream_maneuvers
from scripts.netbuild.sources import has_raw_gtfs
from scripts.netbuild.timing import dequantize
from scripts.netbuild.validate import check_geo, check_network, gzip_size

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / "data"
NETWORK_PATH = DATA / K.NETWORK_FILE
GEO_PATH = DATA / K.GEO_FILE
MANIFEST_PATH = DATA / K.MANIFEST_FILE
FRONTEND_GEO_PATH = BACKEND.parent / "frontend" / "public" / "data" / K.GEO_FILE

requires_raw = pytest.mark.skipif(not has_raw_gtfs(DATA / "raw"),
                                  reason="data/raw/ download caches are absent (gitignored)")

PROFILES = ("am_peak", "midday", "pm_rush", "late_night")


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def net_bytes() -> bytes:
    return NETWORK_PATH.read_bytes()


@pytest.fixture(scope="module")
def geo_bytes() -> bytes:
    return GEO_PATH.read_bytes()


@pytest.fixture(scope="module")
def net(net_bytes: bytes) -> dict[str, Any]:
    return json.loads(net_bytes)


@pytest.fixture(scope="module")
def geo(geo_bytes: bytes) -> dict[str, Any]:
    return json.loads(geo_bytes)


@pytest.fixture(scope="module")
def node_station(net: dict[str, Any]) -> list[int]:
    return [n["station"] for n in net["nodes"]]


def _run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full_env = dict(os.environ)
    full_env.update(env or {})
    return subprocess.run([sys.executable, "-m", "scripts.build_nyc_network", *args], cwd=BACKEND, env=full_env,
                          capture_output=True, text=True, timeout=600, check=False)


# --------------------------------------------------------------------------- document level


def test_schema_projection_and_graph_version(net: dict[str, Any]) -> None:
    assert net["schema_version"] == 1
    assert net["projection"] == {"lat0": 40.7128, "lon0": -73.95, "kx": 84384.0, "ky": 111132.954}
    assert len(net["graph_version"]) == 12 and int(net["graph_version"], 16) >= 0
    assert net["graph_version"] == content_hash(net)


def test_graph_version_matches_the_simulator_formula(net: dict[str, Any]) -> None:
    from sim.graph import canonical_graph_version

    assert canonical_graph_version(net) == net["graph_version"]


def test_simulator_loads_the_network_without_repairs(net: dict[str, Any]) -> None:
    """The integration seam: sim.graph accepts the artifact as is (no defensive fixes, no dropped patterns)."""
    from sim.graph import graph_from_dict

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graph = graph_from_dict(net, source=str(NETWORK_PATH))
    assert not graph.warnings, graph.warnings
    assert not [w for w in caught if "[sim.graph]" in str(w.message)]
    assert graph.n_patterns == len(net["patterns"])


def test_indices_equal_array_positions(net: dict[str, Any]) -> None:
    for name in ("routes", "complexes", "stations", "platforms", "nodes", "edges", "segments", "interlockings",
                 "yards", "patterns"):
        assert [item["idx"] for item in net[name]] == list(range(len(net[name]))), name


def test_scope_counts(net: dict[str, Any]) -> None:
    assert len(net["routes"]) == 28
    assert len(net["stations"]) == 475
    assert len(net["complexes"]) == 424
    assert {r["id"] for r in net["routes"]} == set(K.IN_SCOPE_ROUTES)
    assert "SI" not in {r["id"] for r in net["routes"]}


def test_graph_size_ranges(net: dict[str, Any]) -> None:
    run_edges = [e for e in net["edges"] if e["kind"] == "run"]
    assert 1100 <= len(net["nodes"]) <= 1700
    assert 1100 <= len(run_edges) <= 1800
    assert 400 <= len(net["segments"]) <= 560


def test_artifact_sizes(net_bytes: bytes, geo_bytes: bytes) -> None:
    assert len(net_bytes) <= 3_000_000
    assert gzip_size(geo_bytes) <= 120_000
    # §2.7 targets 400 KB raw; the measured artifact is 431 KB (contract change request in
    # docs/progress/WP1.md: the 7 required keys x 1,915 edges alone take 168 KB).
    assert len(geo_bytes) <= 450_000


# --------------------------------------------------------------------------- §2.6 invariants


def test_pattern_edges_are_consecutive(net: dict[str, Any]) -> None:
    edges = net["edges"]
    for p in net["patterns"]:
        assert len(p["edges"]) == len(p["stops"]) > 0, p["idx"]
        for a, b in zip(p["edges"], p["edges"][1:]):
            assert edges[a]["to"] == edges[b]["from"], (p["idx"], a, b)


def test_every_stop_is_on_a_platform_node(net: dict[str, Any]) -> None:
    edges, nodes = net["edges"], net["nodes"]
    for p in net["patterns"]:
        assert nodes[edges[p["edges"][0]]["from"]]["platform"] >= 0, f"pattern {p['idx']} origin"
        assert p["stops"][-1] == 1, f"pattern {p['idx']} must stop at its destination"
        for e, stop in zip(p["edges"], p["stops"]):
            if stop:
                assert nodes[edges[e]["to"]]["platform"] >= 0, (p["idx"], e)


def test_no_edge_crosses_divisions(net: dict[str, Any]) -> None:
    nodes = net["nodes"]
    crossing = [e["idx"] for e in net["edges"] if nodes[e["from"]]["division"] != nodes[e["to"]]["division"]]
    assert crossing == []
    for n in nodes:
        if n["station"] >= 0:
            assert n["division"] == net["stations"][n["station"]]["division"]
    for y in net["yards"]:
        assert nodes[y["node"]]["division"] == y["division"]
        for r in y["routes"]:
            assert net["routes"][r]["division"] == y["division"]


def test_every_station_has_a_node_and_every_node_an_edge(net: dict[str, Any], node_station: list[int]) -> None:
    assert {s for s in node_station if s >= 0} == set(range(len(net["stations"])))
    touched = {e["from"] for e in net["edges"]} | {e["to"] for e in net["edges"]}
    assert touched == set(range(len(net["nodes"])))


def test_every_route_has_am_peak_service(net: dict[str, Any]) -> None:
    exempt = set(load_routes_without_am_service(DATA))
    patterns = net["patterns"]
    served = {patterns[row["pattern"]]["route"] for row in net["service"]["am_peak"]}
    missing = [r["id"] for r in net["routes"] if r["idx"] not in served and r["id"] not in exempt]
    assert missing == []


def test_crossover_interlockings_pockets_and_yards(net: dict[str, Any], node_station: list[int]) -> None:
    kinds = Counter(k for il in net["interlockings"] for k in il["kinds"])
    assert kinds["L2X"] >= 40
    assert kinds["X2L"] >= 40
    pocket_stations = {node_station[e["from"]] for e in net["edges"] if e["kind"] == "turnback" and e["pocket"]}
    assert len(pocket_stations) >= 30
    assert len(net["yards"]) == 10
    assert {y["name"].split(" Yard")[0].split(" (")[0] for y in net["yards"]} >= {
        "207 St", "Concourse", "Coney Island", "Jamaica", "Pitkin", "Livonia", "Corona", "Canarsie", "239 St"}


def test_system_entries_per_profile(net: dict[str, Any]) -> None:
    per_hour = {p: sum(net["demand"][p]["platform_rate"]) * 3600.0 for p in PROFILES}
    assert 250_000 <= per_hour["am_peak"] <= 500_000
    assert per_hour["late_night"] < 60_000
    assert per_hour["late_night"] < per_hour["midday"] < per_hour["am_peak"]


def test_contract_validator_reports_no_errors(net: dict[str, Any], geo: dict[str, Any], net_bytes: bytes,
                                              geo_bytes: bytes) -> None:
    errors, _ = check_network(net, routes_without_am_service=load_routes_without_am_service(DATA),
                              size_bytes=len(net_bytes))
    assert errors == []
    errors, _ = check_geo(geo, net, size_bytes=len(geo_bytes), gzip_bytes=gzip_size(geo_bytes))
    assert errors == []


# --------------------------------------------------------------------------- field-level rules


def test_edge_fields(net: dict[str, Any], node_station: list[int]) -> None:
    nodes, segments = net["nodes"], net["segments"]
    for e in net["edges"]:
        a, b = e["from"], e["to"]
        if e["kind"] == "run":
            seg = segments[e["segment"]]
            ends = (seg["a"], seg["b"]) if e["seg_dir"] == 1 else (seg["b"], seg["a"])
            assert (node_station[a], node_station[b]) == ends, e["idx"]
            assert e["track"] == nodes[a]["track"] == nodes[b]["track"] in ("L", "X")
            assert e["length_m"] > 0 and e["run_s"] > 0 and 8.0 <= e["v_max"] <= 24.6
            assert e["pocket"] is False
            assert e["bidir_group"] == -1 or e["track"] == "X"
            continue
        assert e["segment"] == -1 and e["length_m"] == 0.0 and e["run_s"] > 0 and e["bidir_group"] == -1
        if e["kind"] == "crossover":
            assert node_station[a] == node_station[b] and {nodes[a]["track"], nodes[b]["track"]} == {"L", "X"}
            assert e["seg_dir"] == 0 and e["run_s"] == K.CROSSOVER_S and not e["pocket"]
        elif e["kind"] == "turnback":
            assert node_station[a] == node_station[b] and e["seg_dir"] == 0
            assert e["run_s"] == (K.TURNBACK_S if e["pocket"] else K.LAYOVER_S)
        else:
            assert e["kind"] == "yard_lead"
            assert (node_station[a] == -1) != (node_station[b] == -1)
            assert e["seg_dir"] == (1 if node_station[a] == -1 else -1)  # +1 pull-out, -1 put-in


def test_segments_partition_run_edges(net: dict[str, Any]) -> None:
    edges = net["edges"]
    listed = sorted(e for seg in net["segments"] for e in seg["edges"])
    assert listed == [e["idx"] for e in edges if e["kind"] == "run"]
    for seg in net["segments"]:
        assert seg["a"] != seg["b"]
        assert seg["has_express"] == any(edges[e]["track"] == "X" for e in seg["edges"])
        assert all(edges[e]["segment"] == seg["idx"] for e in seg["edges"])


def test_every_segment_has_running_edges_both_ways(net: dict[str, Any]) -> None:
    """Every adjacent station pair is traversable in both directions on the local track."""
    edges = net["edges"]
    one_way = [seg["name"] for seg in net["segments"]
               if {edges[e]["seg_dir"] for e in seg["edges"] if edges[e]["track"] == "L"} != {1, -1}]
    assert one_way == []


def test_platforms_and_capacity(net: dict[str, Any]) -> None:
    nodes = net["nodes"]
    for pl in net["platforms"]:
        assert 350.0 <= pl["capacity"] <= 3500.0
        assert pl["nodes"]
        for n in pl["nodes"]:
            assert nodes[n]["platform"] == pl["idx"] and nodes[n]["station"] == pl["station"]
    for n in nodes:
        if n["platform"] >= 0:
            assert n["idx"] in net["platforms"][n["platform"]]["nodes"]


def test_complexes_partition_stations(net: dict[str, Any]) -> None:
    owner = Counter(s for c in net["complexes"] for s in c["stations"])
    assert set(owner) == set(range(len(net["stations"]))) and set(owner.values()) == {1}
    for s in net["stations"]:
        assert s["idx"] in net["complexes"][s["complex"]]["stations"]
        assert s["division"] == ("A" if s["gtfs"].isdigit() else "B")
        assert s["routes"]
    boroughs = Counter(c["borough"] for c in net["complexes"])
    assert set(boroughs) == {"M", "Bk", "Q", "Bx"}


def test_service_rows(net: dict[str, Any]) -> None:
    n_patterns = len(net["patterns"])
    for prof in PROFILES:
        rows = net["service"][prof]
        assert rows, prof
        assert len({r["pattern"] for r in rows}) == len(rows)
        for r in rows:
            assert 0 <= r["pattern"] < n_patterns
            assert r["headway_s"] > 0 and 0 <= r["phase_s"] < r["headway_s"]


def test_demand_vectors(net: dict[str, Any]) -> None:
    for prof in PROFILES:
        d = net["demand"][prof]
        assert len(d["platform_rate"]) == len(net["platforms"])
        assert len(d["attraction"]) == len(net["stations"])
        assert min(d["platform_rate"]) >= 0 and min(d["attraction"]) >= 0
        assert math.isclose(sum(d["attraction"]), 1.0, abs_tol=1e-4)


def test_patterns_and_next_patterns(net: dict[str, Any], node_station: list[int]) -> None:
    patterns, edges = net["patterns"], net["edges"]
    for p in patterns:
        assert node_station[edges[p["edges"][0]]["from"]] == p["origin"]
        assert node_station[edges[p["edges"][-1]]["to"]] == p["dest"]
        assert {edges[e]["kind"] for e in p["edges"]} <= {"run", "crossover"}
        assert p["trips_weekday"] >= 1
        for q in p["next_patterns"]:
            assert patterns[q]["route"] == p["route"] and patterns[q]["origin"] == p["dest"]


def test_peak_dir_covers_every_middle_track(net: dict[str, Any]) -> None:
    groups: dict[int, set[int]] = defaultdict(set)
    for e in net["edges"]:
        if e["bidir_group"] >= 0:
            groups[e["bidir_group"]].add(e["idx"])
    assert len(groups) == 7  # 5 three-track peak groups + 2 middle tracks (infrastructure.json)
    covered = {(pd["bidir_group"], pd["profile"]) for pd in net["peak_dir"]}
    assert covered == {(g, p) for g in groups for p in PROFILES}
    for pd in net["peak_dir"]:
        allowed = set(pd["allowed_dir_edges"])
        assert allowed and allowed < groups[pd["bidir_group"]]  # one direction only


def test_four_track_trunks_have_express_tracks_both_ways(net: dict[str, Any]) -> None:
    """PRD §4.1: Lexington, 7 Av, 8 Av, 6 Av, Broadway, Queens Blvd, Fulton and Eastern Pkwy are 4-track."""
    gidx = {s["gtfs"]: s["idx"] for s in net["stations"]}
    x_hops = {(net["nodes"][e["from"]]["station"], net["nodes"][e["to"]]["station"])
              for e in net["edges"] if e["kind"] == "run" and e["track"] == "X"}
    trunks = {
        "Lexington": ["635", "634", "633", "632", "631"], "7 Av": ["127", "128", "129", "130", "131", "132"],
        "8 Av": ["A27", "A28", "A30", "A31"], "6 Av": ["D16", "D17", "D18", "D19", "D20"],
        "Broadway": ["R16", "R17", "R18", "R19", "R20"], "Queens Blvd": ["G14", "G13", "G12", "G11", "G10"],
        "Fulton": ["A44", "A45", "A46", "A47", "A48"], "Eastern Pkwy": ["237", "238", "239"],
    }
    for name, chain in trunks.items():
        for a, b in zip(chain, chain[1:]):
            assert (gidx[a], gidx[b]) in x_hops and (gidx[b], gidx[a]) in x_hops, f"{name} {a}-{b}"


def test_river_crossings_are_marked(net: dict[str, Any]) -> None:
    gtfs = [s["gtfs"] for s in net["stations"]]
    river = {frozenset((gtfs[s["a"]], gtfs[s["b"]])) for s in net["segments"] if s["river"]}
    assert len(river) == 24
    for pair in (("A40", "A38"), ("L06", "L08"), ("420", "423"), ("D22", "R30"), ("Q01", "R30"), ("M18", "M16")):
        assert frozenset(pair) in river


# --------------------------------------------------------------------------- presets


def test_presets_resolve(net: dict[str, Any]) -> None:
    presets = net["presets"]
    assert 6 <= len(presets) <= 8
    assert len({p["id"] for p in presets}) == len(presets)
    limits = {"station": len(net["stations"]), "complex": len(net["complexes"]),
              "segment": len(net["segments"]), "interlocking": len(net["interlockings"])}
    allowed = {"local_block": {"segment", "station", "interlocking"},
               "express_block": {"segment", "station", "interlocking"},
               "full_station_closure": {"station", "complex"}, "signal_delay": {"segment", "interlocking"}}
    for p in presets:
        assert p["profile"] in PROFILES
        assert p["disruptions"]
        assert {"x", "y", "zoom_m_per_px"} <= set(p["focus"]) and p["focus"]["zoom_m_per_px"] > 0
        for d in p["disruptions"]:
            t = d["target"]
            assert t["type"] in allowed[d["kind"]]
            assert 0 <= t["idx"] < limits[t["type"]]
            assert d["duration_s"] is None or d["duration_s"] > 0
            if d["kind"] == "express_block":
                assert t["type"] != "segment" or net["segments"][t["idx"]]["has_express"]
    ids = {p["id"] for p in presets}
    assert {"lex-union-sq", "qbl-express-roosevelt", "times-sq-signals", "canarsie-tube", "manhattan-bridge",
            "8av-59st-local", "joralemon-tunnel"} <= ids


@pytest.mark.parametrize("preset_id", ["lex-union-sq", "qbl-express-roosevelt", "times-sq-signals", "canarsie-tube",
                                       "manhattan-bridge", "8av-59st-local", "joralemon-tunnel"])
def test_preset_leaves_a_maneuver_upstream(net: dict[str, Any], preset_id: str) -> None:
    """Every preset disruption hits in-service patterns, each of which can at least be held at an
    upstream platform, and the dispatcher has a structural maneuver (short-turn, express/local switch,
    reroute or gap train) upstream."""
    preset = next(p for p in net["presets"] if p["id"] == preset_id)
    for d in preset["disruptions"]:
        found = upstream_maneuvers(net, d, preset["profile"])
        assert found, f"{preset_id}: {d} affects no pattern in service"
        assert any(pm.structural for pm in found), f"{preset_id}: no structural maneuver upstream of {d}"
        assert all(pm.maneuvers for pm in found), f"{preset_id}: a pattern has no maneuver at all"


def test_canonical_preset_maneuvers(net: dict[str, Any]) -> None:
    """The showcase moves each preset is designed around."""
    def kinds(preset_id: str, j: int = 0) -> dict[str, set[str]]:
        preset = next(p for p in net["presets"] if p["id"] == preset_id)
        out: dict[str, set[str]] = defaultdict(set)
        for pm in upstream_maneuvers(net, preset["disruptions"][j], preset["profile"]):
            for k, v in pm.maneuvers.items():
                if v:
                    out[k].add(pm.route)
        return out

    assert {"E", "F"} <= kinds("qbl-express-roosevelt")["local"]          # express -> local around the block
    assert "L" in kinds("canarsie-tube")["short_turn"]                     # relay at Bedford Av
    assert "C" in kinds("8av-59st-local")["express"]                       # C runs express past 59 St
    assert {"N", "Q"} <= kinds("manhattan-bridge", 1)["reroute"]           # via the Montague St tunnel
    assert "5" in kinds("joralemon-tunnel")["reroute"]                     # via the Clark St tunnel
    assert {"4", "5"} <= kinds("lex-union-sq")["short_turn"]               # Brooklyn Bridge pocket


def test_footprints_follow_disruption_semantics(net: dict[str, Any]) -> None:
    edges, nodes = net["edges"], net["nodes"]
    seg = next(s for s in net["segments"] if s["has_express"])
    local = footprint(net, {"kind": "local_block", "target": {"type": "segment", "idx": seg["idx"]}})
    express = footprint(net, {"kind": "express_block", "target": {"type": "segment", "idx": seg["idx"]}})
    assert local.edges and {edges[e]["track"] for e in local.edges} == {"L"}
    assert express.edges and {edges[e]["track"] for e in express.edges} == {"X"}
    assert local.edges | express.edges == set(seg["edges"])
    closure = footprint(net, {"kind": "full_station_closure", "target": {"type": "station", "idx": seg["a"]}})
    assert closure.closed_nodes == {n["idx"] for n in nodes if n["station"] == seg["a"]} and not closure.edges


# --------------------------------------------------------------------------- network_geo.json (§2.7)


def test_geo_header_and_counts(geo: dict[str, Any], net: dict[str, Any]) -> None:
    assert geo["schema_version"] == 1 and geo["graph_version"] == net["graph_version"]
    assert geo["projection"] == net["projection"]
    for name in ("routes", "complexes", "stations", "segments", "edges", "interlockings", "yards"):
        assert [g["idx"] for g in geo[name]] == list(range(len(net[name]))), name
    assert geo["presets"] == net["presets"]


def test_geo_mirrors_network_fields(geo: dict[str, Any], net: dict[str, Any]) -> None:
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
            assert {k: g[k] for k in keys} == {k: n[k] for k in keys}, (name, g["idx"])
    def iround(v: float) -> int:  # half up, as the build (and JavaScript Math.round) does
        return math.floor(v + 0.5)

    for name in ("stations", "complexes", "yards"):
        for g, n in zip(geo[name], net[name]):
            assert (g["x"], g["y"]) == (iround(n["x"]), iround(n["y"])), (name, g["idx"])


def test_geo_coordinates_are_integer_meters(geo: dict[str, Any]) -> None:
    def ints(values: Any) -> bool:
        return all(isinstance(v, int) and not isinstance(v, bool) for v in values)

    assert ints(geo["bbox"])
    for s in geo["stations"]:
        assert ints((s["x"], s["y"]))
    for seg in geo["segments"]:
        assert ints(v for pt in seg["pts"] for v in pt)
    if geo["basemap"] is not None:
        for ring in geo["basemap"]["land"]:
            assert len(ring) >= 3 and ints(v for pt in ring for v in pt)


def test_geo_segment_polylines(geo: dict[str, Any], net: dict[str, Any]) -> None:
    stations = geo["stations"]
    for seg, nseg in zip(geo["segments"], net["segments"]):
        pts = seg["pts"]
        assert len(pts) >= 2
        assert pts[0] == [stations[seg["a"]]["x"], stations[seg["a"]]["y"]]
        assert pts[-1] == [stations[seg["b"]]["x"], stations[seg["b"]]["y"]]
        length = sum(math.dist(p, q) for p, q in zip(pts, pts[1:]))
        # polyline length ~ the run edges' length (rounding to 1 m moves it by at most a few meters)
        run_lengths = {net["edges"][e]["length_m"] for e in nseg["edges"]}
        assert all(abs(length - L) <= 2.0 + 0.002 * L for L in run_lengths), seg["idx"]
        assert seg["routes"] and all(0 <= r < len(geo["routes"]) for r in seg["routes"])
        # no hooks: interior turns are gentle (shapes are cut from real alignments)
        v = np.diff(np.asarray(pts, dtype=float), axis=0)
        v = v[np.hypot(v[:, 0], v[:, 1]) >= 5.0]
        if len(v) >= 2:
            cos = (v[:-1] * v[1:]).sum(axis=1) / (np.hypot(*v[:-1].T) * np.hypot(*v[1:].T))
            assert cos.min() > math.cos(math.radians(60)), (seg["idx"], seg["name"])


def test_geo_edge_station_and_yard_fields(geo: dict[str, Any], net: dict[str, Any], node_station: list[int]) -> None:
    for g, n in zip(geo["edges"], net["edges"]):
        if n["kind"] == "run":
            assert g["station"] == -1 and g["yard"] == -1
        elif n["kind"] in ("crossover", "turnback"):
            assert g["station"] == node_station[n["from"]] and g["yard"] == -1
        else:
            yard_node, lead = (n["from"], n["to"]) if node_station[n["from"]] == -1 else (n["to"], n["from"])
            assert g["station"] == node_station[lead]
            assert geo["yards"][g["yard"]]["name"] == net["yards"][g["yard"]]["name"]
            assert net["yards"][g["yard"]]["node"] == yard_node
            assert g["seg_dir"] == (1 if n["from"] == yard_node else -1)


def test_geo_stations_labels_interlockings_bbox(geo: dict[str, Any]) -> None:
    stations = geo["stations"]
    assert sorted(s["label_rank"] for s in stations) == list(range(len(stations)))
    for il in geo["interlockings"]:
        st = stations[il["station"]]
        assert (il["x"], il["y"]) == (st["x"], st["y"])
    xmin, ymin, xmax, ymax = geo["bbox"]
    for s in stations:
        assert xmin <= s["x"] <= xmax and ymin <= s["y"] <= ymax
    for seg in geo["segments"]:
        for x, y in seg["pts"]:
            assert xmin <= x <= xmax and ymin <= y <= ymax
    # label_rank follows complex entries: the busiest complex's stations come first.
    complexes = geo["complexes"]
    by_rank = sorted(stations, key=lambda s: s["label_rank"])
    entries = [complexes[s["complex"]]["weekday_entries"] for s in by_rank]
    assert entries == sorted(entries, reverse=True)
    assert by_rank[0]["hub"] and by_rank[0]["complex"] == max(complexes, key=lambda c: c["weekday_entries"])["idx"]


def test_frontend_copy_is_byte_identical(geo_bytes: bytes) -> None:
    assert FRONTEND_GEO_PATH.is_file()
    assert FRONTEND_GEO_PATH.read_bytes() == geo_bytes


def test_geo_is_valid_gzip_friendly_json(geo_bytes: bytes) -> None:
    assert gzip.decompress(gzip.compress(geo_bytes)) == geo_bytes
    json.loads(geo_bytes.decode("utf-8"))


# --------------------------------------------------------------------------- manifest


def test_manifest_records_provenance(net: dict[str, Any], net_bytes: bytes, geo_bytes: bytes) -> None:
    import hashlib

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["graph_version"] == net["graph_version"]
    assert set(manifest["sources"]) == {"gtfs", "complexes", "stations", "ridership", "basemap"}
    assert isinstance(manifest["fallbacks_used"], list)
    assert manifest["artifacts"]["network.json"]["sha1"] == hashlib.sha1(net_bytes).hexdigest()
    assert manifest["artifacts"]["network_geo.json"]["sha1"] == hashlib.sha1(geo_bytes).hexdigest()
    assert manifest["counts"]["stations"] == 475 and manifest["validation"]["errors"] == []
    assert len(manifest["preset_maneuvers"]) == len(net["presets"])


# --------------------------------------------------------------------------- determinism and the CLI


@requires_raw
def test_offline_rebuild_in_process_is_byte_identical(net_bytes: bytes, geo_bytes: bytes) -> None:
    out = build_artifacts(DATA, HttpClient(offline=True))
    assert out.errors == []
    assert out.network_bytes == net_bytes
    assert out.geo_bytes == geo_bytes


@requires_raw
@pytest.mark.parametrize("hash_seed", ["0", "4242"])
def test_offline_check_is_byte_identical_across_processes(hash_seed: str) -> None:
    """``build --offline --check`` rebuilds from the caches and compares with the committed bytes."""
    result = _run_cli("--offline", "--check", "-q", env={"PYTHONHASHSEED": hash_seed})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "check passed" in result.stdout


@requires_raw
def test_check_detects_stale_artifacts(tmp_path: Path, net_bytes: bytes, geo_bytes: bytes) -> None:
    (tmp_path / K.NETWORK_FILE).write_bytes(net_bytes)
    (tmp_path / K.GEO_FILE).write_bytes(geo_bytes.replace(b'"schema_version":1', b'"schema_version": 1', 1))
    result = _run_cli("--offline", "--check", "-q", "--out-dir", str(tmp_path), "--no-frontend-copy")
    assert result.returncode == 1
    assert "differs from the rebuild" in result.stderr
    assert (tmp_path / K.NETWORK_FILE).read_bytes() == net_bytes  # --check never writes


def _committed_copy(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    for name in (K.INFRASTRUCTURE_FILE, K.NETWORK_FILE, K.GEO_FILE, K.MANIFEST_FILE):
        shutil.copy2(DATA / name, data / name)
    return data


def test_offline_without_caches_validates_committed_artifacts(tmp_path: Path) -> None:
    data = _committed_copy(tmp_path)
    before = {p.name: p.read_bytes() for p in data.iterdir()}
    result = _run_cli("--offline", "--data-dir", str(data), "--no-frontend-copy")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "valid" in result.stdout
    assert {p.name: p.read_bytes() for p in data.iterdir()} == before  # nothing rebuilt or written


def test_offline_without_caches_rejects_corrupt_artifacts(tmp_path: Path) -> None:
    data = _committed_copy(tmp_path)
    doc = json.loads((data / K.NETWORK_FILE).read_text(encoding="utf-8"))
    doc["edges"][0]["to"], doc["edges"][1]["to"] = doc["edges"][1]["to"], doc["edges"][0]["to"]
    (data / K.NETWORK_FILE).write_text(json.dumps(doc), encoding="utf-8")
    result = _run_cli("--offline", "--data-dir", str(data), "--no-frontend-copy")
    assert result.returncode == 1
    assert "error:" in result.stderr


def test_offline_without_any_input_exits_unavailable(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy2(DATA / K.INFRASTRUCTURE_FILE, data / K.INFRASTRUCTURE_FILE)
    result = _run_cli("--offline", "--data-dir", str(data), "--no-frontend-copy")
    assert result.returncode == 2


# --------------------------------------------------------------------------- unit tests


def test_constants_mirror_the_simulator() -> None:
    from sim import constants as C

    assert K.CROSSOVER_S == C.CROSSOVER_S
    assert K.TURNBACK_S == C.TURNBACK_S
    assert K.LAYOVER_S == C.LAYOVER_S
    assert K.ACCEL_DECEL_LOSS_S == C.ACCEL_DECEL_LOSS_S
    assert K.V_MAX_CAP == C.V_MAX_CAP
    assert tuple(K.PROFILES) == tuple(C.PROFILES)
    assert {p: tuple(w) for p, w in K.PROFILE_WINDOWS_H.items()} == {p: tuple(w) for p, w in C.PROFILE_WINDOWS_H.items()}


def test_dequantize_removes_the_30s_rounding_and_keeps_trip_times() -> None:
    # Equal hops scheduled 120/60/120/60/120 s (30 s rounding) -> near-equal times, same total.
    hop_t = np.array([[120.0, 60.0, 120.0, 60.0, 120.0], [90.0, 90.0, 90.0, 90.0, 90.0]])
    out = dequantize(hop_t, np.full(5, 100.0), half_window=1)
    np.testing.assert_allclose(out.sum(axis=1), hop_t.sum(axis=1))
    assert np.ptp(out[0, 1:4]) < 0.4 * np.ptp(hop_t[0])  # a 3-hop window cuts a 2-periodic alternation ~3x
    np.testing.assert_allclose(out[1], hop_t[1])  # already smooth: unchanged
    # A longer hop keeps a proportionally larger share.
    out2 = dequantize(np.array([[90.0, 90.0, 90.0]]), np.array([50.0, 100.0, 50.0]), half_window=1)
    assert out2[0, 1] > out2[0, 0] and math.isclose(out2.sum(), 270.0)


def test_snap_ends_leaves_no_hook() -> None:
    # A shape vertex 3 m before the station and a station point 4.5 m off the shape:
    # replacing only the end point would leave a 4.5 m perpendicular leg.
    piece = np.array([[0.0, 0.0], [400.0, 0.0], [997.0, 0.0], [1000.0, 0.0]])
    out = snap_ends(piece, np.array([0.0, 0.0]), np.array([1000.0, 4.5]))
    assert out[-1].tolist() == [1000.0, 4.5] and out[0].tolist() == [0.0, 0.0]
    legs = np.diff(out, axis=0)
    headings = np.degrees(np.arctan2(legs[:, 1], legs[:, 0]))
    assert np.abs(np.diff(headings)).max() < 5.0
    np.testing.assert_allclose(out[1], piece[1])  # far from both ends: untouched
