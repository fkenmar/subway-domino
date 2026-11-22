"""Read-only network graph for the simulator (ARCHITECTURE §2.6, §3).

``load_graph()`` parses ``data/network.json`` (or any dict with the same schema, e.g. the
toy graph from :mod:`sim.testing`) into a :class:`Graph` of read-only NumPy arrays, and
precomputes everything the engine needs that depends only on the static network:

* CSR adjacency, per-station/platform/node lookups and crossover/turnback tables;
* per-pattern path templates (edges, stop flags, station visits, nominal timetable);
* **alternate paths** used by the action masks: express and local detours for
  RUN_EXPRESS / RUN_LOCAL, same-division reroutes at diverge junctions, short-turn
  continuations through pocket turnbacks, and yard lead positions.

The Graph is shared read-only by every simulator instance in a process (40 server
sessions, or one training worker). Profile-dependent views (service tables, scheduled
headways, demand) are built lazily and cached on the graph.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sim import constants as C

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
BACKEND_DIR: Path = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """``backend/data`` unless overridden by ``SUBWAY_DATA_DIR``."""
    return Path(os.environ.get("SUBWAY_DATA_DIR", str(BACKEND_DIR / "data")))


def model_path() -> Path:
    """``backend/models/nyc_subway_rcc_ppo.onnx`` unless overridden by ``SUBWAY_MODEL_PATH``."""
    default = BACKEND_DIR / "models" / "nyc_subway_rcc_ppo.onnx"
    return Path(os.environ.get("SUBWAY_MODEL_PATH", str(default)))


DATA_DIR: Path = data_dir()
MODEL_PATH: Path = model_path()

# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------
KIND_RUN, KIND_CROSSOVER, KIND_TURNBACK, KIND_YARD_LEAD = 0, 1, 2, 3
EDGE_KINDS: tuple[str, ...] = ("run", "crossover", "turnback", "yard_lead")
TRACK_L, TRACK_X, TRACK_OTHER = 0, 1, 2
XOVER_NONE, XOVER_L2X, XOVER_X2L = 0, 1, 2
DIV_A, DIV_B = 0, 1


def _track_code(value: Any) -> int:
    s = str(value or "").upper()
    if s == "L":
        return TRACK_L
    if s == "X":
        return TRACK_X
    return TRACK_OTHER


def _div_code(value: Any) -> int:
    return DIV_A if str(value).upper() == "A" else DIV_B


def _ro(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


def _csr(lists: Sequence[Sequence[int]], dtype: Any = np.int32) -> tuple[np.ndarray, np.ndarray]:
    ptr = np.zeros(len(lists) + 1, dtype=np.int64)
    for i, lst in enumerate(lists):
        ptr[i + 1] = ptr[i] + len(lst)
    idx = np.fromiter((v for lst in lists for v in lst), dtype=dtype, count=int(ptr[-1]))
    return _ro(ptr), _ro(idx)


def route_family(route_id: str) -> str:
    """Equipment pool family: express variants share trains with their route (6X -> 6)."""
    if len(route_id) == 2 and route_id.endswith("X") and route_id[0] != "G":
        return route_id[0]
    return route_id


# --------------------------------------------------------------------------------------
# Small value types
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Detour:
    """An alternate path from a pattern position (RUN_EXPRESS / RUN_LOCAL / REROUTE)."""

    kind: str  # "express" | "local" | "reroute"
    edges: tuple[int, ...]  # detour edges, the first one leaves the decision node
    stops: tuple[bool, ...]  # stop flag at each detour edge's to-node
    rejoin_pos: int  # pattern position resumed after the detour; -1 = ends at a terminal
    orig_edges: tuple[int, ...]  # pattern edges replaced by the detour
    rejoin_station: int
    via_station: int  # first station served by the detour that is off the original path
    bypassed_stations: tuple[int, ...]  # original stops the detour does not serve
    detour_time_s: float
    orig_time_s: float


@dataclass(frozen=True, eq=False)
class ShortTurnOption:
    turnback_edge: int
    pattern: int  # reverse pattern the train continues as
    position: int  # its position (edge index) leaving the turnback's to-node


@dataclass(frozen=True, eq=False)
class YardLead:
    lead_id: int
    yard: int
    pattern: int
    position: int  # pattern edge index leaving the lead node
    node: int
    station: int


@dataclass(frozen=True, eq=False)
class ServiceView:
    """Profile (and route subset) dependent service/demand tables, cached on the graph."""

    profile: str
    routes: tuple[int, ...] | None
    svc_pattern: np.ndarray  # int32[k]
    svc_headway: np.ndarray  # float64[k]
    svc_phase: np.ndarray  # float64[k]
    pattern_headway: np.ndarray  # float64[n_patterns] (inf if not in service)
    edge_sched_h: np.ndarray  # float64[n_edges] combined scheduled headway over the edge
    platform_sched_h: np.ndarray  # float64[n_platforms] combined headway of stopping service
    platform_rate: np.ndarray  # float64[n_platforms] riders/s (before demand_scale)
    attraction: np.ndarray  # float64[n_stations]
    tmpl_suffix: np.ndarray  # float64[n_patterns, S] attraction suffix sums per stop index
    pool_keys: tuple[tuple[int, int], ...]  # (family, station)
    pool_out_rate: np.ndarray  # departures / s
    pool_in_rate: np.ndarray  # arrivals / s
    pool_target: np.ndarray  # int32 seed / steady-state size
    pool_putin_period: np.ndarray  # float64, inf when no put-ins are needed
    pool_surplus: np.ndarray  # bool: structural surplus => lay up extra trains
    init_onboard: dict[int, np.ndarray]  # pattern -> float32[n_stops, N_DEST] steady load
    init_waiting: np.ndarray  # float32[n_platforms, N_AGE_BINS]


# --------------------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False, repr=False)
class Graph:
    """Read-only network arrays. Indices match ``network.json`` (and ``network_geo.json``)."""

    # meta
    schema_version: int
    graph_version: str
    projection: dict[str, float]
    source: str
    warnings: tuple[str, ...]

    # routes
    n_routes: int
    route_id: tuple[str, ...]
    route_name: tuple[str, ...]
    route_color: tuple[str, ...]
    route_text_color: tuple[str, ...]
    route_division: np.ndarray  # int8 (0=A, 1=B)
    route_shuttle: np.ndarray  # bool
    route_family: np.ndarray  # int16 pool family index
    route_index: dict[str, int]

    # complexes
    n_complexes: int
    complex_id: tuple[str, ...]
    complex_name: tuple[str, ...]
    complex_borough: tuple[str, ...]
    complex_x: np.ndarray
    complex_y: np.ndarray
    complex_entries: np.ndarray
    complex_station_ptr: np.ndarray
    complex_station_idx: np.ndarray

    # stations
    n_stations: int
    station_gtfs: tuple[str, ...]
    station_name: tuple[str, ...]
    station_complex: np.ndarray  # int32
    station_division: np.ndarray  # int8
    station_x: np.ndarray
    station_y: np.ndarray
    station_structure: tuple[str, ...]
    station_hub: np.ndarray  # bool
    station_terminal: np.ndarray  # bool
    station_route_ptr: np.ndarray
    station_route_idx: np.ndarray
    station_node_ptr: np.ndarray
    station_node_idx: np.ndarray
    station_platform_ptr: np.ndarray
    station_platform_idx: np.ndarray
    station_index: dict[str, int]

    # platforms
    n_platforms: int
    platform_station: np.ndarray  # int32
    platform_capacity: np.ndarray  # float64
    platform_node_ptr: np.ndarray
    platform_node_idx: np.ndarray

    # nodes
    n_nodes: int
    node_station: np.ndarray  # int32, -1 for yard nodes
    node_platform: np.ndarray  # int32, -1 if none
    node_track: np.ndarray  # int8
    node_division: np.ndarray  # int8
    node_yard: np.ndarray  # int32, -1 unless a yard node

    # edges
    n_edges: int
    edge_kind: np.ndarray  # int8
    edge_from: np.ndarray  # int32
    edge_to: np.ndarray  # int32
    edge_segment: np.ndarray  # int32
    edge_seg_dir: np.ndarray  # int8
    edge_track: np.ndarray  # int8
    edge_length: np.ndarray  # float64 (m)
    edge_run_s: np.ndarray  # float64
    edge_vmax: np.ndarray  # float64 (m/s)
    edge_cbtc: np.ndarray  # bool
    edge_bidir: np.ndarray  # int32
    edge_pocket: np.ndarray  # bool
    edge_timed: np.ndarray  # bool: zero-length edges traversed by time (crossover/turnback/yard)
    edge_xover: np.ndarray  # int8 XOVER_*
    edge_from_station: np.ndarray  # int32
    edge_to_station: np.ndarray  # int32
    edge_cruise_s: np.ndarray  # float64 nominal traversal time (length / v_max, or run_s)
    edge_pocket_group: np.ndarray  # int32 station of a pocket turnback, else -1
    node_out_ptr: np.ndarray
    node_out_idx: np.ndarray
    node_in_ptr: np.ndarray
    node_in_idx: np.ndarray
    node_l2x: np.ndarray  # int32 L2X crossover leaving the node, -1
    node_x2l: np.ndarray  # int32 X2L crossover leaving the node, -1
    node_has_pocket: np.ndarray  # bool: a pocket turnback leaves the node
    node_diverge: np.ndarray  # bool: >= 2 run out-edges to different stations
    node_dir_platform: np.ndarray  # int32: own platform, else same-direction platform of the station

    # segments
    n_segments: int
    segment_a: np.ndarray
    segment_b: np.ndarray
    segment_name: tuple[str, ...]
    segment_river: np.ndarray
    segment_has_express: np.ndarray
    segment_edge_ptr: np.ndarray
    segment_edge_idx: np.ndarray

    # interlockings
    n_interlockings: int
    interlocking_station: np.ndarray
    interlocking_edge_ptr: np.ndarray
    interlocking_edge_idx: np.ndarray
    interlocking_kinds: tuple[tuple[str, ...], ...]

    # yards
    n_yards: int
    yard_name: tuple[str, ...]
    yard_division: np.ndarray
    yard_x: np.ndarray
    yard_y: np.ndarray
    yard_node: np.ndarray
    yard_fleet_cap: np.ndarray  # int32
    yard_deadhead_s: np.ndarray  # float64
    yard_lead_edge_ptr: np.ndarray
    yard_lead_edge_idx: np.ndarray
    yard_route_ptr: np.ndarray
    yard_route_idx: np.ndarray

    # patterns
    n_patterns: int
    pat_route: np.ndarray  # int32
    pat_edge_ptr: np.ndarray
    pat_edge_idx: np.ndarray
    pat_stop_flat: np.ndarray  # bool aligned with pat_edge_idx
    pat_len: np.ndarray  # int32 number of edges
    pat_origin_node: np.ndarray
    pat_dest_node: np.ndarray
    pat_origin_station: np.ndarray
    pat_dest_station: np.ndarray
    pat_trips: np.ndarray
    pat_next_ptr: np.ndarray
    pat_next_idx: np.ndarray
    pat_n_stops: np.ndarray  # int32 stops after the origin
    route_pat_ptr: np.ndarray
    route_pat_idx: np.ndarray

    # path templates (padded to path_capacity / stop_capacity)
    path_capacity: int
    stop_capacity: int
    tmpl_edge: np.ndarray  # int32[n_patterns, P] (-1 pad)
    tmpl_stop: np.ndarray  # bool[n_patterns, P]
    tmpl_visit: np.ndarray  # int16[n_patterns, P] visit index of the edge's to-node
    tmpl_sched: np.ndarray  # float32[n_patterns, P] nominal arrival offset at the to-node
    tmpl_s_pos: np.ndarray  # int16[n_patterns, S] path position of the j-th stop (-1 pad)
    tmpl_s_station: np.ndarray  # int32[n_patterns, S]

    # demand / service raw tables
    service_raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    demand_raw: dict[str, tuple[np.ndarray, np.ndarray]]
    peak_ok: dict[str, np.ndarray]  # profile -> bool[n_edges] usable w.r.t. peak direction
    presets: tuple[dict[str, Any], ...]
    preset_index: dict[str, int]

    # station transfer structure
    station_transfer: np.ndarray  # bool: alighters may transfer here
    station_xfer_ptr: np.ndarray  # CSR -> destination platforms (other stations of the complex)
    station_xfer_idx: np.ndarray
    platform_spill_ptr: np.ndarray  # CSR -> spill destination platforms
    platform_spill_idx: np.ndarray
    station_interlocking: np.ndarray  # int32

    # alternates (keyed by flat pattern position pat_edge_ptr[p] + i)
    express_detours: dict[int, tuple[Detour, ...]]
    local_detours: dict[int, tuple[Detour, ...]]
    reroute_detours: dict[int, tuple[Detour, ...]]
    pp_has_express: np.ndarray  # bool[flat positions]
    pp_has_local: np.ndarray
    pp_has_reroute: np.ndarray
    short_turn_options: dict[int, ShortTurnOption]  # flat position -> option
    pp_has_short_turn: np.ndarray
    yard_leads: tuple[YardLead, ...]
    pat_lead_ptr: np.ndarray  # CSR pattern -> lead ids
    pat_lead_idx: np.ndarray

    _cache: dict[Any, Any] = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- convenience
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Graph(version={self.graph_version!r}, routes={self.n_routes}, stations={self.n_stations}, "
            f"platforms={self.n_platforms}, nodes={self.n_nodes}, edges={self.n_edges}, "
            f"patterns={self.n_patterns}, yards={self.n_yards})"
        )

    def pattern_edges(self, p: int) -> np.ndarray:
        return self.pat_edge_idx[self.pat_edge_ptr[p] : self.pat_edge_ptr[p + 1]]

    def pattern_stops(self, p: int) -> np.ndarray:
        return self.pat_stop_flat[self.pat_edge_ptr[p] : self.pat_edge_ptr[p + 1]]

    def station_nodes(self, s: int) -> np.ndarray:
        return self.station_node_idx[self.station_node_ptr[s] : self.station_node_ptr[s + 1]]

    def station_platforms(self, s: int) -> np.ndarray:
        return self.station_platform_idx[self.station_platform_ptr[s] : self.station_platform_ptr[s + 1]]

    def station_routes(self, s: int) -> np.ndarray:
        return self.station_route_idx[self.station_route_ptr[s] : self.station_route_ptr[s + 1]]

    def complex_stations(self, c: int) -> np.ndarray:
        return self.complex_station_idx[self.complex_station_ptr[c] : self.complex_station_ptr[c + 1]]

    def segment_edges(self, seg: int) -> np.ndarray:
        return self.segment_edge_idx[self.segment_edge_ptr[seg] : self.segment_edge_ptr[seg + 1]]

    def interlocking_edges(self, il: int) -> np.ndarray:
        return self.interlocking_edge_idx[self.interlocking_edge_ptr[il] : self.interlocking_edge_ptr[il + 1]]

    def node_out(self, n: int) -> np.ndarray:
        return self.node_out_idx[self.node_out_ptr[n] : self.node_out_ptr[n + 1]]

    def node_in(self, n: int) -> np.ndarray:
        return self.node_in_idx[self.node_in_ptr[n] : self.node_in_ptr[n + 1]]

    def yard_routes(self, y: int) -> np.ndarray:
        return self.yard_route_idx[self.yard_route_ptr[y] : self.yard_route_ptr[y + 1]]

    def route_patterns(self, r: int) -> np.ndarray:
        return self.route_pat_idx[self.route_pat_ptr[r] : self.route_pat_ptr[r + 1]]

    def pattern_leads(self, p: int) -> np.ndarray:
        return self.pat_lead_idx[self.pat_lead_ptr[p] : self.pat_lead_ptr[p + 1]]

    def flat_pos(self, p: int, i: int) -> int:
        return int(self.pat_edge_ptr[p]) + int(i)

    def train_capacity(self, division: int) -> float:
        return C.TRAIN_CAPACITY["A" if division == DIV_A else "B"]

    def train_length(self, division: int) -> float:
        return C.TRAIN_LENGTH_M["A" if division == DIV_A else "B"]

    def route_ids_to_idx(self, routes: Iterable[str] | None) -> tuple[int, ...] | None:
        if routes is None:
            return None
        out = []
        for r in routes:
            if r not in self.route_index:
                raise ValueError(f"unknown route id {r!r}")
            out.append(self.route_index[r])
        return tuple(sorted(set(out)))

    # ---------------------------------------------------------------- profile views
    def service_view(self, profile: str, routes: tuple[int, ...] | None = None) -> ServiceView:
        """Service/demand tables for ``profile`` restricted to ``routes`` (cached)."""
        if profile not in C.PROFILES:
            raise ValueError(f"unknown profile {profile!r}")
        key = ("service", profile, routes)
        view = self._cache.get(key)
        if view is None:
            view = _build_service_view(self, profile, routes)
            self._cache[key] = view
        return view


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
_GRAPH_CACHE: dict[tuple[str, int, int], Graph] = {}


def load_graph(path: str | os.PathLike[str] | None = None, *, use_cache: bool = True) -> Graph:
    """Load ``network.json`` (default ``DATA_DIR/network.json``) into a :class:`Graph`.

    Graphs are cached per resolved path and file mtime, so every simulator in a process
    shares one read-only instance.
    """
    p = Path(path) if path is not None else data_dir() / "network.json"
    p = p.resolve()
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    if use_cache and key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    g = graph_from_dict(data, source=str(p))
    if use_cache:
        _GRAPH_CACHE[key] = g
    return g


def canonical_graph_version(data: Mapping[str, Any]) -> str:
    """First 12 hex of sha1 of the canonical JSON with ``graph_version`` set to "" (§2.6)."""
    d = dict(data)
    d["graph_version"] = ""
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def graph_from_dict(data: Mapping[str, Any], *, source: str = "<dict>") -> Graph:
    """Build a :class:`Graph` from a parsed ``network.json`` dict (the exact §2.6 schema)."""
    return _GraphBuilder(data, source).build()


class _GraphBuilder:
    def __init__(self, data: Mapping[str, Any], source: str) -> None:
        self.d = data
        self.source = source
        self.warn: list[str] = []

    # ---------------------------------------------------------------- helpers
    def _w(self, msg: str) -> None:
        self.warn.append(msg)

    # ---------------------------------------------------------------- build
    def build(self) -> Graph:  # noqa: C901 - one linear pass over the schema
        d = self.d
        f: dict[str, Any] = {}
        f["schema_version"] = int(d.get("schema_version", 1))
        f["graph_version"] = str(d.get("graph_version") or "")
        f["projection"] = dict(d.get("projection") or {})
        f["source"] = self.source

        # ---------------- routes
        routes = list(d["routes"])
        self._check_idx(routes, "routes")
        n_routes = len(routes)
        f["n_routes"] = n_routes
        f["route_id"] = tuple(str(r["id"]) for r in routes)
        f["route_name"] = tuple(str(r.get("name", r["id"])) for r in routes)
        f["route_color"] = tuple(str(r.get("color", "#808183")) for r in routes)
        f["route_text_color"] = tuple(str(r.get("text_color", "#FFFFFF")) for r in routes)
        route_div = np.array([_div_code(r.get("division", "B")) for r in routes], dtype=np.int8)
        f["route_division"] = _ro(route_div)
        f["route_shuttle"] = _ro(np.array([bool(r.get("shuttle", False)) for r in routes], dtype=bool))
        fams: dict[str, int] = {}
        fam_idx = []
        for rid in f["route_id"]:
            fam = route_family(rid)
            fam_idx.append(fams.setdefault(fam, len(fams)))
        f["route_family"] = _ro(np.array(fam_idx, dtype=np.int16))
        f["route_index"] = {rid: i for i, rid in enumerate(f["route_id"])}

        # ---------------- complexes
        cpx = list(d["complexes"])
        self._check_idx(cpx, "complexes")
        f["n_complexes"] = len(cpx)
        f["complex_id"] = tuple(str(c.get("complex_id", i)) for i, c in enumerate(cpx))
        f["complex_name"] = tuple(str(c.get("name", "")) for c in cpx)
        f["complex_borough"] = tuple(str(c.get("borough", "")) for c in cpx)
        f["complex_x"] = _ro(np.array([float(c.get("x", 0.0)) for c in cpx], dtype=np.float64))
        f["complex_y"] = _ro(np.array([float(c.get("y", 0.0)) for c in cpx], dtype=np.float64))
        f["complex_entries"] = _ro(np.array([float(c.get("weekday_entries", 0.0)) for c in cpx]))

        # ---------------- stations
        sts = list(d["stations"])
        self._check_idx(sts, "stations")
        n_st = len(sts)
        f["n_stations"] = n_st
        f["station_gtfs"] = tuple(str(s.get("gtfs", "")) for s in sts)
        f["station_name"] = tuple(str(s.get("name", "")) for s in sts)
        st_cpx = np.array([int(s.get("complex", -1)) for s in sts], dtype=np.int32)
        f["station_complex"] = _ro(st_cpx)
        f["station_division"] = _ro(np.array([_div_code(s.get("division", "B")) for s in sts], dtype=np.int8))
        f["station_x"] = _ro(np.array([float(s.get("x", 0.0)) for s in sts], dtype=np.float64))
        f["station_y"] = _ro(np.array([float(s.get("y", 0.0)) for s in sts], dtype=np.float64))
        f["station_structure"] = tuple(str(s.get("structure", "")) for s in sts)
        f["station_hub"] = _ro(np.array([bool(s.get("hub", False)) for s in sts], dtype=bool))
        f["station_terminal"] = _ro(np.array([bool(s.get("terminal", False)) for s in sts], dtype=bool))
        f["station_route_ptr"], f["station_route_idx"] = _csr([list(s.get("routes", [])) for s in sts])
        f["station_index"] = {g: i for i, g in enumerate(f["station_gtfs"])}
        # complex -> stations (derive from stations if the complex lists are inconsistent)
        cpx_lists: list[list[int]] = [[] for _ in range(len(cpx))]
        for i, c in enumerate(st_cpx):
            if 0 <= c < len(cpx):
                cpx_lists[c].append(i)
        for ci, c in enumerate(cpx):
            listed = sorted(int(x) for x in c.get("stations", []))
            if listed and listed != sorted(cpx_lists[ci]):
                self._w(f"complex {ci} station list differs from station.complex; using station.complex")
        f["complex_station_ptr"], f["complex_station_idx"] = _csr(cpx_lists)

        # ---------------- nodes
        nodes = list(d["nodes"])
        self._check_idx(nodes, "nodes")
        n_nodes = len(nodes)
        f["n_nodes"] = n_nodes
        node_station = np.array([int(n.get("station", -1)) for n in nodes], dtype=np.int32)
        node_platform = np.array([int(n.get("platform", -1)) for n in nodes], dtype=np.int32)
        node_track = np.array([_track_code(n.get("track")) for n in nodes], dtype=np.int8)
        node_div = np.array([_div_code(n.get("division", "B")) for n in nodes], dtype=np.int8)

        # ---------------- platforms
        plats = list(d["platforms"])
        self._check_idx(plats, "platforms")
        n_pl = len(plats)
        f["n_platforms"] = n_pl
        f["platform_station"] = _ro(np.array([int(p["station"]) for p in plats], dtype=np.int32))
        cap = np.array([float(p.get("capacity", 1000.0)) for p in plats], dtype=np.float64)
        f["platform_capacity"] = _ro(np.maximum(cap, 1.0))
        f["platform_node_ptr"], f["platform_node_idx"] = _csr([list(p.get("nodes", [])) for p in plats])
        st_pl: list[list[int]] = [[] for _ in range(n_st)]
        for i, p in enumerate(plats):
            st_pl[int(p["station"])].append(i)
        f["station_platform_ptr"], f["station_platform_idx"] = _csr(st_pl)
        st_nodes: list[list[int]] = [[] for _ in range(n_st)]
        for i, s in enumerate(node_station):
            if s >= 0:
                st_nodes[s].append(i)
        f["station_node_ptr"], f["station_node_idx"] = _csr(st_nodes)

        # ---------------- edges
        edges = list(d["edges"])
        self._check_idx(edges, "edges")
        n_e = len(edges)
        f["n_edges"] = n_e
        kind_map = {k: i for i, k in enumerate(EDGE_KINDS)}
        e_kind = np.array([kind_map.get(str(e.get("kind", "run")), KIND_RUN) for e in edges], dtype=np.int8)
        e_from = np.array([int(e["from"]) for e in edges], dtype=np.int32)
        e_to = np.array([int(e["to"]) for e in edges], dtype=np.int32)
        e_len = np.array([float(e.get("length_m", 0.0)) for e in edges], dtype=np.float64)
        e_run = np.array([float(e.get("run_s", 0.0)) for e in edges], dtype=np.float64)
        e_vmax = np.array([float(e.get("v_max", 0.0)) for e in edges], dtype=np.float64)
        f["edge_segment"] = _ro(np.array([int(e.get("segment", -1)) for e in edges], dtype=np.int32))
        f["edge_seg_dir"] = _ro(np.array([int(e.get("seg_dir", 0)) for e in edges], dtype=np.int8))
        f["edge_track"] = _ro(np.array([_track_code(e.get("track")) for e in edges], dtype=np.int8))
        f["edge_cbtc"] = _ro(np.array([bool(e.get("cbtc", False)) for e in edges], dtype=bool))
        f["edge_bidir"] = _ro(np.array([int(e.get("bidir_group", -1)) for e in edges], dtype=np.int32))
        e_pocket = np.array([bool(e.get("pocket", False)) for e in edges], dtype=bool)

        # Defensive fixes: run edges need a positive length and speed.
        run = e_kind == KIND_RUN
        bad_len = run & ~(e_len > 1.0)
        if bad_len.any():
            self._w(f"{int(bad_len.sum())} run edges with length <= 1 m; set to 300 m")
            e_len[bad_len] = 300.0
        bad_v = run & ~(e_vmax > 0.5)
        if bad_v.any():
            self._w(f"{int(bad_v.sum())} run edges without v_max; derived from run_s")
            e_vmax[bad_v] = np.clip(e_len[bad_v] / np.maximum(e_run[bad_v] - C.ACCEL_DECEL_LOSS_S, 20.0), 8.0, C.V_MAX_CAP)
        e_vmax[run] = np.clip(e_vmax[run], 3.0, C.V_MAX_CAP)
        timed = ~run
        # Non-run edges are traversed by time at the station point (length 0).
        e_len[timed] = 0.0
        xo = e_kind == KIND_CROSSOVER
        e_run[xo & ~(e_run > 0)] = C.CROSSOVER_S
        tb = e_kind == KIND_TURNBACK
        e_run[tb & ~(e_run > 0)] = C.TURNBACK_S
        yl = e_kind == KIND_YARD_LEAD
        e_run[yl & ~(e_run > 0)] = 60.0
        bad_run = run & ~(e_run > 0)
        e_run[bad_run] = e_len[bad_run] / e_vmax[bad_run] + C.ACCEL_DECEL_LOSS_S

        f["edge_kind"] = _ro(e_kind)
        f["edge_from"] = _ro(e_from)
        f["edge_to"] = _ro(e_to)
        f["edge_length"] = _ro(e_len)
        f["edge_run_s"] = _ro(e_run)
        f["edge_vmax"] = _ro(e_vmax)
        f["edge_pocket"] = _ro(e_pocket & tb)
        f["edge_timed"] = _ro(timed.copy())
        f["edge_from_station"] = _ro(node_station[e_from].astype(np.int32))
        f["edge_to_station"] = _ro(node_station[e_to].astype(np.int32))
        cruise = np.where(run, e_len / np.maximum(e_vmax, 0.1), e_run)
        f["edge_cruise_s"] = _ro(cruise.astype(np.float64))
        xov = np.zeros(n_e, dtype=np.int8)
        xov[xo & (node_track[e_from] == TRACK_L) & (node_track[e_to] == TRACK_X)] = XOVER_L2X
        xov[xo & (node_track[e_from] == TRACK_X) & (node_track[e_to] == TRACK_L)] = XOVER_X2L
        f["edge_xover"] = _ro(xov)

        cross_div = node_div[e_from] != node_div[e_to]
        if cross_div.any():
            self._w(f"{int(cross_div.sum())} cross-division edges found; they are disabled")

        out_l: list[list[int]] = [[] for _ in range(n_nodes)]
        in_l: list[list[int]] = [[] for _ in range(n_nodes)]
        for e in range(n_e):
            if cross_div[e]:
                continue
            out_l[e_from[e]].append(e)
            in_l[e_to[e]].append(e)
        f["node_out_ptr"], f["node_out_idx"] = _csr(out_l)
        f["node_in_ptr"], f["node_in_idx"] = _csr(in_l)

        l2x = np.full(n_nodes, -1, dtype=np.int32)
        x2l = np.full(n_nodes, -1, dtype=np.int32)
        has_pocket = np.zeros(n_nodes, dtype=bool)
        diverge = np.zeros(n_nodes, dtype=bool)
        pocket_group = np.full(n_e, -1, dtype=np.int32)
        for n in range(n_nodes):
            run_to_st = set()
            for e in out_l[n]:
                if xov[e] == XOVER_L2X and l2x[n] < 0:
                    l2x[n] = e
                elif xov[e] == XOVER_X2L and x2l[n] < 0:
                    x2l[n] = e
                elif tb[e] and e_pocket[e]:
                    has_pocket[n] = True
                    pocket_group[e] = node_station[n]
                elif run[e]:
                    run_to_st.add(int(node_station[e_to[e]]))
            diverge[n] = len(run_to_st) >= 2
        f["node_l2x"] = _ro(l2x)
        f["node_x2l"] = _ro(x2l)
        f["node_has_pocket"] = _ro(has_pocket)
        f["node_diverge"] = _ro(diverge)
        f["edge_pocket_group"] = _ro(pocket_group)

        node_yard = np.full(n_nodes, -1, dtype=np.int32)

        # ---------------- segments
        segs = list(d.get("segments", []))
        self._check_idx(segs, "segments")
        f["n_segments"] = len(segs)
        f["segment_a"] = _ro(np.array([int(s["a"]) for s in segs], dtype=np.int32))
        f["segment_b"] = _ro(np.array([int(s["b"]) for s in segs], dtype=np.int32))
        f["segment_name"] = tuple(str(s.get("name", "")) for s in segs)
        f["segment_river"] = _ro(np.array([bool(s.get("river", False)) for s in segs], dtype=bool))
        f["segment_has_express"] = _ro(np.array([bool(s.get("has_express", False)) for s in segs], dtype=bool))
        seg_edges: list[list[int]] = [list(map(int, s.get("edges", []))) for s in segs]
        # Make sure every run edge that names a segment is listed by it.
        e_seg = f["edge_segment"]
        listed = {e for lst in seg_edges for e in lst}
        for e in range(n_e):
            if run[e] and 0 <= e_seg[e] < len(segs) and e not in listed:
                seg_edges[e_seg[e]].append(e)
        f["segment_edge_ptr"], f["segment_edge_idx"] = _csr([sorted(set(x)) for x in seg_edges])

        # ---------------- interlockings
        ils = list(d.get("interlockings", []))
        self._check_idx(ils, "interlockings")
        f["n_interlockings"] = len(ils)
        f["interlocking_station"] = _ro(np.array([int(i["station"]) for i in ils], dtype=np.int32))
        f["interlocking_edge_ptr"], f["interlocking_edge_idx"] = _csr([list(i.get("edges", [])) for i in ils])
        f["interlocking_kinds"] = tuple(tuple(str(k) for k in i.get("kinds", [])) for i in ils)
        st_il = np.full(n_st, -1, dtype=np.int32)
        for i, il in enumerate(ils):
            s = int(il["station"])
            if 0 <= s < n_st and st_il[s] < 0:
                st_il[s] = i
        f["station_interlocking"] = _ro(st_il)

        # ---------------- yards
        yards = list(d.get("yards", []))
        self._check_idx(yards, "yards")
        f["n_yards"] = len(yards)
        f["yard_name"] = tuple(str(y.get("name", f"Yard {i}")) for i, y in enumerate(yards))
        f["yard_division"] = _ro(np.array([_div_code(y.get("division", "B")) for y in yards], dtype=np.int8))
        f["yard_x"] = _ro(np.array([float(y.get("x", 0.0)) for y in yards]))
        f["yard_y"] = _ro(np.array([float(y.get("y", 0.0)) for y in yards]))
        yard_node = np.array([int(y.get("node", -1)) for y in yards], dtype=np.int32)
        f["yard_node"] = _ro(yard_node)
        f["yard_fleet_cap"] = _ro(np.array([int(y.get("fleet_cap", 4)) for y in yards], dtype=np.int32))
        f["yard_deadhead_s"] = _ro(np.array([float(y.get("deadhead_s", 480.0)) for y in yards]))
        f["yard_lead_edge_ptr"], f["yard_lead_edge_idx"] = _csr([list(y.get("lead_edges", [])) for y in yards])
        f["yard_route_ptr"], f["yard_route_idx"] = _csr([sorted(set(y.get("routes", []))) for y in yards])
        for yi, yn in enumerate(yard_node):
            if 0 <= yn < n_nodes:
                node_yard[yn] = yi
        f["node_station"] = _ro(node_station)
        f["node_platform"] = _ro(node_platform)
        f["node_track"] = _ro(node_track)
        f["node_division"] = _ro(node_div)
        f["node_yard"] = _ro(node_yard)

        # ---------------- patterns
        self._build_patterns(f, list(d.get("patterns", [])))

        # ---------------- service, demand, peak direction, presets
        self._build_service_demand(f)
        f["presets"] = tuple(dict(p) for p in d.get("presets", []))
        f["preset_index"] = {str(p.get("id")): i for i, p in enumerate(f["presets"])}

        # ---------------- direction siblings, transfers, spill
        self._build_platform_structure(f)

        g_partial = _Partial(f)
        self._build_alternates(f, g_partial)
        f["warnings"] = tuple(self.warn)
        f["_cache"] = {}
        for msg in self.warn:
            warnings.warn(f"[sim.graph] {msg}", stacklevel=3)
        return Graph(**f)

    def _check_idx(self, items: Sequence[Mapping[str, Any]], name: str) -> None:
        for i, it in enumerate(items):
            if "idx" in it and int(it["idx"]) != i:
                raise ValueError(f"{name}[{i}].idx == {it['idx']} (indices must equal array positions)")

    # ---------------------------------------------------------------- patterns
    def _build_patterns(self, f: dict[str, Any], pats: list[Mapping[str, Any]]) -> None:
        e_from, e_to = f["edge_from"], f["edge_to"]
        node_platform, node_station = f["node_platform"], f["node_station"]
        n_e = f["n_edges"]
        keep: list[tuple[int, list[int], list[bool], Mapping[str, Any]]] = []
        orig_index: dict[int, int] = {}
        for i, p in enumerate(pats):
            edges = [int(e) for e in p.get("edges", [])]
            stops = [bool(s) for s in p.get("stops", [1] * len(edges))]
            if not edges or len(stops) != len(edges) or any(not (0 <= e < n_e) for e in edges):
                self._w(f"pattern {i} malformed; dropped")
                continue
            if any(e_to[a] != e_from[b] for a, b in zip(edges[:-1], edges[1:])):
                self._w(f"pattern {i} edges not consecutive; dropped")
                continue
            # every stop must be on a platform node; the final node is always a stop
            stops[-1] = True
            for k, e in enumerate(edges):
                if stops[k] and node_platform[e_to[e]] < 0:
                    stops[k] = False
            if node_platform[e_to[edges[-1]]] < 0:
                self._w(f"pattern {i} ends at a node without platform")
            orig_index[i] = len(keep)
            keep.append((int(p["route"]), edges, stops, p))
        if len(keep) != len(pats):
            # Indices must stay aligned with network.json for service tables; we keep the
            # dropped slots as 1-edge dummies that are never scheduled.
            self._w("some patterns were dropped; their service entries are ignored")
        self._pattern_remap = orig_index
        n_p = len(keep)
        f["n_patterns"] = n_p
        f["pat_route"] = _ro(np.array([k[0] for k in keep], dtype=np.int32))
        f["pat_edge_ptr"], f["pat_edge_idx"] = _csr([k[1] for k in keep])
        f["pat_stop_flat"] = _ro(np.fromiter((s for k in keep for s in k[2]), dtype=bool))
        f["pat_len"] = _ro(np.array([len(k[1]) for k in keep], dtype=np.int32))
        on = np.array([e_from[k[1][0]] for k in keep], dtype=np.int32)
        dn = np.array([e_to[k[1][-1]] for k in keep], dtype=np.int32)
        f["pat_origin_node"] = _ro(on)
        f["pat_dest_node"] = _ro(dn)
        f["pat_origin_station"] = _ro(node_station[on].astype(np.int32))
        f["pat_dest_station"] = _ro(node_station[dn].astype(np.int32))
        f["pat_trips"] = _ro(np.array([float(k[3].get("trips_weekday", 0.0)) for k in keep]))
        f["pat_n_stops"] = _ro(np.array([int(sum(k[2])) for k in keep], dtype=np.int32))
        nxt = []
        for k in keep:
            lst = [orig_index[j] for j in k[3].get("next_patterns", []) if j in orig_index]
            nxt.append(lst)
        f["pat_next_ptr"], f["pat_next_idx"] = _csr(nxt)
        rp: list[list[int]] = [[] for _ in range(f["n_routes"])]
        for i, k in enumerate(keep):
            if 0 <= k[0] < f["n_routes"]:
                rp[k[0]].append(i)
        f["route_pat_ptr"], f["route_pat_idx"] = _csr(rp)

        max_len = max((len(k[1]) for k in keep), default=1)
        P = max(C.MAX_PATH, int(math.ceil((max_len + 40) / 8.0) * 8))
        max_stops = max((sum(k[2]) for k in keep), default=1)
        S = int(math.ceil((max_stops + 40) / 8.0) * 8)
        f["path_capacity"] = P
        f["stop_capacity"] = S
        tmpl_edge = np.full((n_p, P), -1, dtype=np.int32)
        tmpl_stop = np.zeros((n_p, P), dtype=bool)
        tmpl_visit = np.zeros((n_p, P), dtype=np.int16)
        tmpl_sched = np.zeros((n_p, P), dtype=np.float32)
        tmpl_s_pos = np.full((n_p, S), -1, dtype=np.int16)
        tmpl_s_station = np.full((n_p, S), -1, dtype=np.int32)
        for i, (_, edges, stops, _) in enumerate(keep):
            L = len(edges)
            tmpl_edge[i, :L] = edges
            tmpl_stop[i, :L] = stops
            tmpl_visit[i, :L] = path_visits(f, edges)
            tmpl_sched[i, :L] = nominal_schedule(f, edges, stops)
            spos = [k for k in range(L) if stops[k]]
            tmpl_s_pos[i, : len(spos)] = spos
            tmpl_s_station[i, : len(spos)] = [int(node_station[e_to[edges[k]]]) for k in spos]
        f["tmpl_edge"] = _ro(tmpl_edge)
        f["tmpl_stop"] = _ro(tmpl_stop)
        f["tmpl_visit"] = _ro(tmpl_visit)
        f["tmpl_sched"] = _ro(tmpl_sched)
        f["tmpl_s_pos"] = _ro(tmpl_s_pos)
        f["tmpl_s_station"] = _ro(tmpl_s_station)

    # ---------------------------------------------------------------- service & demand
    def _build_service_demand(self, f: dict[str, Any]) -> None:
        d = self.d
        n_pl, n_st, n_e = f["n_platforms"], f["n_stations"], f["n_edges"]
        svc_raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        service = d.get("service", {}) or {}
        for prof in C.PROFILES:
            pat, hw, ph = [], [], []
            for ent in service.get(prof, []) or []:
                pi = int(ent["pattern"])
                if pi not in self._pattern_remap:
                    continue
                h = float(ent.get("headway_s", 0.0))
                if not (h > 0):
                    continue
                pat.append(self._pattern_remap[pi])
                hw.append(h)
                ph.append(float(ent.get("phase_s", 0.0)) % h)
            svc_raw[prof] = (
                _ro(np.array(pat, dtype=np.int32)),
                _ro(np.array(hw, dtype=np.float64)),
                _ro(np.array(ph, dtype=np.float64)),
            )
        f["service_raw"] = svc_raw
        dem_raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        demand = d.get("demand", {}) or {}
        for prof in C.PROFILES:
            dp = demand.get(prof, {}) or {}
            rate = np.zeros(n_pl, dtype=np.float64)
            r_in = list(dp.get("platform_rate", []) or [])
            if r_in:
                if len(r_in) != n_pl:
                    self._w(f"demand[{prof}].platform_rate has {len(r_in)} values for {n_pl} platforms")
                m = min(len(r_in), n_pl)
                rate[:m] = np.maximum(np.asarray(r_in[:m], dtype=np.float64), 0.0)
            attr = np.zeros(n_st, dtype=np.float64)
            a_in = list(dp.get("attraction", []) or [])
            if a_in:
                m = min(len(a_in), n_st)
                attr[:m] = np.maximum(np.asarray(a_in[:m], dtype=np.float64), 0.0)
            if attr.sum() <= 0:
                attr[:] = 1.0
            attr = attr / attr.sum()
            dem_raw[prof] = (_ro(rate), _ro(attr))
        f["demand_raw"] = dem_raw

        peak_ok: dict[str, np.ndarray] = {}
        bidir = f["edge_bidir"]
        groups: dict[str, dict[int, set[int]]] = {p: {} for p in C.PROFILES}
        for ent in d.get("peak_dir", []) or []:
            prof = str(ent.get("profile", ""))
            if prof not in groups:
                continue
            groups[prof].setdefault(int(ent["bidir_group"]), set()).update(int(e) for e in ent.get("allowed_dir_edges", []))
        for prof in C.PROFILES:
            ok = np.ones(n_e, dtype=bool)
            for grp, allowed in groups[prof].items():
                members = np.nonzero(bidir == grp)[0]
                for e in members:
                    if int(e) not in allowed:
                        ok[e] = False
            peak_ok[prof] = _ro(ok)
        f["peak_ok"] = peak_ok

    # ---------------------------------------------------------------- platforms
    def _build_platform_structure(self, f: dict[str, Any]) -> None:
        n_nodes, n_st, n_pl = f["n_nodes"], f["n_stations"], f["n_platforms"]
        node_station, node_platform = f["node_station"], f["node_platform"]
        e_from, e_to, e_kind = f["edge_from"], f["edge_to"], f["edge_kind"]
        # same-direction siblings: nodes of a station linked by a crossover, or sharing a
        # predecessor/successor station along run edges.
        pred: list[set[int]] = [set() for _ in range(n_nodes)]
        succ: list[set[int]] = [set() for _ in range(n_nodes)]
        xlink: list[set[int]] = [set() for _ in range(n_nodes)]
        for e in range(f["n_edges"]):
            a, b = int(e_from[e]), int(e_to[e])
            if e_kind[e] == KIND_RUN:
                succ[a].add(int(node_station[b]))
                pred[b].add(int(node_station[a]))
            elif e_kind[e] == KIND_CROSSOVER:
                xlink[a].add(b)
                xlink[b].add(a)
        dir_pl = node_platform.copy()
        for n in range(n_nodes):
            if dir_pl[n] >= 0 or node_station[n] < 0:
                continue
            s = int(node_station[n])
            for m in f["station_node_idx"][f["station_node_ptr"][s] : f["station_node_ptr"][s + 1]]:
                m = int(m)
                if m == n or node_platform[m] < 0:
                    continue
                if m in xlink[n] or (pred[n] & pred[m]) or (succ[n] & succ[m]):
                    dir_pl[n] = node_platform[m]
                    break
        f["node_dir_platform"] = _ro(dir_pl.astype(np.int32))

        st_cpx = f["station_complex"]
        cpx_ptr, cpx_idx = f["complex_station_ptr"], f["complex_station_idx"]
        st_pl_ptr, st_pl_idx = f["station_platform_ptr"], f["station_platform_idx"]
        st_rt_ptr = f["station_route_ptr"]
        transfer = np.zeros(n_st, dtype=bool)
        xfer: list[list[int]] = [[] for _ in range(n_st)]
        for s in range(n_st):
            c = int(st_cpx[s])
            others: list[int] = []
            n_cpx_st = 0
            if c >= 0:
                members = cpx_idx[cpx_ptr[c] : cpx_ptr[c + 1]]
                n_cpx_st = len(members)
                for s2 in members:
                    if int(s2) != s:
                        others.extend(int(p) for p in st_pl_idx[st_pl_ptr[s2] : st_pl_ptr[s2 + 1]])
            n_routes = int(st_rt_ptr[s + 1] - st_rt_ptr[s])
            eligible = n_cpx_st >= C.TRANSFER_MIN_STATIONS or n_routes >= C.TRANSFER_MIN_ROUTES
            if eligible and others:
                transfer[s] = True
                xfer[s] = sorted(others)
        f["station_transfer"] = _ro(transfer)
        f["station_xfer_ptr"], f["station_xfer_idx"] = _csr(xfer)
        plat_st = f["platform_station"]
        spill = [xfer[int(plat_st[p])] for p in range(n_pl)]
        f["platform_spill_ptr"], f["platform_spill_idx"] = _csr(spill)

    # ---------------------------------------------------------------- alternates
    def _build_alternates(self, f: dict[str, Any], g: "_Partial") -> None:
        n_flat = int(f["pat_edge_ptr"][-1])
        express: dict[int, tuple[Detour, ...]] = {}
        local: dict[int, tuple[Detour, ...]] = {}
        reroute: dict[int, tuple[Detour, ...]] = {}
        shortturn: dict[int, ShortTurnOption] = {}
        has_x = np.zeros(n_flat, dtype=bool)
        has_l = np.zeros(n_flat, dtype=bool)
        has_r = np.zeros(n_flat, dtype=bool)
        has_st = np.zeros(n_flat, dtype=bool)
        node_track = f["node_track"]
        e_from = f["edge_from"]
        for p in range(f["n_patterns"]):
            pe = g.pattern_edges(p)
            ctx = _PatternCtx(g, p)
            base = int(f["pat_edge_ptr"][p])
            for i in range(len(pe)):
                n = int(e_from[pe[i]])
                if node_track[n] == TRACK_L and f["node_l2x"][n] >= 0:
                    alts = _track_detours(g, ctx, i, "express")
                    if alts:
                        express[base + i] = alts
                        has_x[base + i] = True
                if node_track[n] == TRACK_X and f["node_x2l"][n] >= 0:
                    alts = _track_detours(g, ctx, i, "local")
                    if alts:
                        local[base + i] = alts
                        has_l[base + i] = True
                if f["node_diverge"][n]:
                    alts = _reroute_detours(g, ctx, i)
                    if alts:
                        reroute[base + i] = alts
                        has_r[base + i] = True
                if f["node_has_pocket"][n]:
                    opt = _short_turn_option(g, ctx, i)
                    if opt is not None:
                        shortturn[base + i] = opt
                        has_st[base + i] = True
        f["express_detours"] = express
        f["local_detours"] = local
        f["reroute_detours"] = reroute
        f["short_turn_options"] = shortturn
        f["pp_has_express"] = _ro(has_x)
        f["pp_has_local"] = _ro(has_l)
        f["pp_has_reroute"] = _ro(has_r)
        f["pp_has_short_turn"] = _ro(has_st)
        leads, pat_leads = _yard_leads(g)
        f["yard_leads"] = leads
        f["pat_lead_ptr"], f["pat_lead_idx"] = _csr(pat_leads)


class _Partial:
    """Minimal attribute view over the builder's field dict (used before Graph exists)."""

    def __init__(self, f: dict[str, Any]) -> None:
        self.__dict__.update(f)

    def pattern_edges(self, p: int) -> np.ndarray:
        return self.pat_edge_idx[self.pat_edge_ptr[p] : self.pat_edge_ptr[p + 1]]

    def pattern_stops(self, p: int) -> np.ndarray:
        return self.pat_stop_flat[self.pat_edge_ptr[p] : self.pat_edge_ptr[p + 1]]

    def node_out(self, n: int) -> np.ndarray:
        return self.node_out_idx[self.node_out_ptr[n] : self.node_out_ptr[n + 1]]

    def route_patterns(self, r: int) -> np.ndarray:
        return self.route_pat_idx[self.route_pat_ptr[r] : self.route_pat_ptr[r + 1]]


# --------------------------------------------------------------------------------------
# Path helpers shared with the engine
# --------------------------------------------------------------------------------------
def path_visits(g: Any, edges: Sequence[int]) -> np.ndarray:
    """Visit index (station count from the path start) of each edge's to-node."""
    st = g["edge_to_station"] if isinstance(g, dict) else g.edge_to_station
    fr = g["edge_from_station"] if isinstance(g, dict) else g.edge_from_station
    out = np.zeros(len(edges), dtype=np.int16)
    v = 0
    for k, e in enumerate(edges):
        if int(st[e]) != int(fr[e]):
            v += 1
        out[k] = v
    return out


def shift_buckets(b: np.ndarray, q_tail: float) -> np.ndarray:
    """Advance onboard destination buckets by one stop (bucket 0 has just alighted).

    Bucket k <- k+1; the 12+ tail bucket splits geometrically: a share (1 - q) of its riders
    are destined to the stop that just became the 12th, the rest stay in the tail.
    """
    out = np.empty_like(b)
    out[..., :-2] = b[..., 1:-1]
    tail = b[..., -1]
    out[..., -2] = (1.0 - q_tail) * tail
    out[..., -1] = q_tail * tail
    return out


def run_time_s(lengths: Sequence[float], vmaxes: Sequence[float]) -> float:
    """Minimum running time from stand to stand over consecutive run edges.

    Uses the trapezoidal profile with ``ACCEL``/``BRAKE``: cruise time at each edge's
    speed plus the acceleration and braking losses at the ends (exact for one edge).
    """
    a, b = C.ACCEL, C.BRAKE
    if len(lengths) == 0:
        return 0.0
    total_len = float(sum(lengths))
    if len(lengths) == 1:
        L, v = float(lengths[0]), float(vmaxes[0])
        d_ab = v * v / (2 * a) + v * v / (2 * b)
        if L >= d_ab:
            return L / v + v / (2 * a) + v / (2 * b)
        vp = math.sqrt(2 * L * a * b / (a + b))
        return vp / a + vp / b
    t = sum(float(L) / float(v) for L, v in zip(lengths, vmaxes))
    v0, v1 = float(vmaxes[0]), float(vmaxes[-1])
    t += v0 / (2 * a) + v1 / (2 * b)
    # short multi-edge intervals cannot reach the cruise speeds
    vp = math.sqrt(2 * total_len * a * b / (a + b))
    t_min = vp / a + vp / b
    return max(t, t_min)


def nominal_schedule(g: Any, edges: Sequence[int], stops: Sequence[bool]) -> np.ndarray:
    """Nominal arrival offset (s) at each edge's to-node, from departure at the path start.

    Trains stop at scheduled stops and before every timed (zero-length) edge; each
    intermediate stop adds ``DWELL_SCHED_S``. Running times carry ``SCHEDULE_PADDING``.
    """
    get = (lambda k: g[k]) if isinstance(g, dict) else (lambda k: getattr(g, k))
    timed = get("edge_timed")
    length = get("edge_length")
    vmax = get("edge_vmax")
    run_s = get("edge_run_s")
    out = np.zeros(len(edges), dtype=np.float64)
    t = 0.0
    seg_len: list[float] = []
    seg_v: list[float] = []
    seg_idx: list[int] = []
    seg_t0 = 0.0

    def flush() -> float:
        nonlocal seg_len, seg_v, seg_idx
        if not seg_idx:
            return seg_t0
        total = run_time_s(seg_len, seg_v) * C.SCHEDULE_PADDING
        cum = np.cumsum(seg_len)
        span = float(cum[-1]) if cum[-1] > 0 else 1.0
        for j, k in enumerate(seg_idx):
            out[k] = seg_t0 + total * float(cum[j]) / span
        seg_len, seg_v, seg_idx = [], [], []
        return seg_t0 + total

    for k, e in enumerate(edges):
        if timed[e]:
            t = flush()
            t += float(run_s[e])
            out[k] = t
            seg_t0 = t
            if stops[k] and k < len(edges) - 1:
                t += C.DWELL_SCHED_S
                seg_t0 = t
            continue
        seg_len.append(float(length[e]))
        seg_v.append(float(vmax[e]))
        seg_idx.append(k)
        if stops[k] or k == len(edges) - 1 or timed[edges[k + 1]]:
            t = flush()
            if stops[k] and k < len(edges) - 1:
                t += C.DWELL_SCHED_S
            seg_t0 = t
    return out.astype(np.float32)


# --------------------------------------------------------------------------------------
# Alternate-path precompute
# --------------------------------------------------------------------------------------
class _PatternCtx:
    """Per-pattern lookups used while building detours."""

    def __init__(self, g: _Partial, p: int) -> None:
        self.p = p
        self.route = int(g.pat_route[p])
        self.edges = [int(e) for e in g.pattern_edges(p)]
        self.stops = [bool(s) for s in g.pattern_stops(p)]
        self.L = len(self.edges)
        ef, et = g.edge_from, g.edge_to
        self.node_pos: dict[int, int] = {}
        for k, e in enumerate(self.edges):
            self.node_pos.setdefault(int(ef[e]), k)
        self.node_pos.setdefault(int(et[self.edges[-1]]), self.L)
        st = g.node_station
        self.pos_station = [int(st[ef[e]]) for e in self.edges] + [int(st[et[self.edges[-1]]])]
        # stop stations by position of arrival (to-node of edge k)
        self.stop_station_at = {k: int(st[et[e]]) for k, e in enumerate(self.edges) if self.stops[k]}
        self.cum_run = np.concatenate([[0.0], np.cumsum([float(g.edge_run_s[e]) for e in self.edges])])

    def orig_time(self, i: int, k: int) -> float:
        return float(self.cum_run[k] - self.cum_run[i])

    def stops_between(self, i: int, k: int) -> list[int]:
        """Stations where the pattern stops on edges i .. k-1 (arrivals)."""
        return [self.stop_station_at[j] for j in range(i, k) if j in self.stop_station_at]


def _track_detours(g: _Partial, ctx: _PatternCtx, i: int, kind: str) -> tuple[Detour, ...]:
    """Express (L->X) or local (X->L) detours from pattern position ``i``; up to 3 rejoins."""
    ef, et = g.edge_from, g.edge_to
    node = int(ef[ctx.edges[i]])
    first_x = int(g.node_l2x[node] if kind == "express" else g.node_x2l[node])
    want_track = TRACK_X if kind == "express" else TRACK_L
    back_xover = XOVER_X2L if kind == "express" else XOVER_L2X
    detours: list[Detour] = []
    if first_x < 0 or ctx.edges[i] == first_x:
        return ()  # the pattern itself switches tracks here: nothing to choose
    cur = int(et[first_x])
    seq: list[int] = [first_x]
    seen_stations = {int(g.node_station[node])}
    remaining_stations = set(ctx.pos_station[i + 1 :])
    for _step in range(C.REROUTE_MAX_STATIONS):
        # follow the run edge on the same track class, staying on the pattern's corridor
        nxt = -1
        for e in g.node_out(cur):
            e = int(e)
            if g.edge_kind[e] != KIND_RUN or g.node_track[et[e]] != want_track:
                continue
            if int(g.node_station[et[e]]) in remaining_stations:
                nxt = e
                break
        if nxt < 0:
            break
        seq.append(nxt)
        cur = int(et[nxt])
        s = int(g.node_station[cur])
        if s in seen_stations:
            break
        seen_stations.add(s)
        k_direct = ctx.node_pos.get(cur, -1)
        rejoin_edges: list[int] | None = None
        k_rejoin = -1
        if k_direct > i:
            rejoin_edges = []
            k_rejoin = k_direct
        else:
            for e in g.node_out(cur):
                e = int(e)
                if g.edge_xover[e] == back_xover:
                    k2 = ctx.node_pos.get(int(et[e]), -1)
                    if k2 > i:
                        rejoin_edges = [e]
                        k_rejoin = k2
                        break
        if rejoin_edges is None:
            continue
        det = _make_detour(g, ctx, i, kind, seq + rejoin_edges, k_rejoin)
        if det is not None and det.edges != det.orig_edges:
            detours.append(det)
        if not rejoin_edges or len(detours) >= 3:
            break  # direct rejoin: beyond it the pattern itself runs on this track
    return tuple(detours)


def _make_detour(
    g: _Partial,
    ctx: _PatternCtx,
    i: int,
    kind: str,
    edges: list[int],
    k_rejoin: int,
    *,
    terminal: bool = False,
) -> Detour | None:
    """Build a :class:`Detour` replacing pattern edges ``i .. k_rejoin-1`` by ``edges``.

    ``terminal`` reroutes end at a terminal served by the route instead of rejoining:
    every remaining original stop is then bypassed.
    """
    et = g.edge_to
    if not edges:
        return None
    if terminal:
        k_rejoin = ctx.L
    elif k_rejoin > ctx.L or k_rejoin <= i:
        return None
    orig_edges = tuple(ctx.edges[i:k_rejoin])
    orig_stops = ctx.stops_between(i, k_rejoin)  # arrivals on edges i..k_rejoin-1
    orig_stop_set = set(orig_stops)
    if terminal:
        rejoin_station = int(g.node_station[et[edges[-1]]])
    else:
        rejoin_station = ctx.pos_station[k_rejoin]
    behind = set(ctx.pos_station[: i + 1])
    stops: list[bool] = []
    served: set[int] = set()
    last = len(edges) - 1
    for j, e in enumerate(edges):
        m = int(et[e])
        s = int(g.node_station[m])
        has_pl = bool(g.node_platform[m] >= 0)
        if g.edge_timed[e]:
            # crossover back onto the pattern: stop after it only if the rejoin stop was not served
            st = has_pl and s == rejoin_station and s in orig_stop_set and s not in served
        elif kind == "express":
            st = has_pl and s in orig_stop_set
        elif kind == "local":
            st = has_pl
        else:  # reroute: serve every platform on the alternate trunk
            st = has_pl and s not in behind
            if s == rejoin_station and not terminal:
                st = has_pl and s in orig_stop_set
        if j == last and (terminal or k_rejoin == ctx.L):
            st = has_pl  # the path ends here: a terminal stop
        stops.append(bool(st))
        if st:
            served.add(s)
    on_orig = set(ctx.pos_station[i : k_rejoin + 1])
    via = -1
    for e in edges:
        s = int(g.node_station[et[e]])
        if s not in on_orig:
            via = s
            break
    if terminal:
        bypassed = tuple(s for s in orig_stops if s not in served)
    else:
        bypassed = tuple(s for s in orig_stops if s not in served and s != rejoin_station)
    det_time = float(sum(float(g.edge_run_s[e]) for e in edges))
    return Detour(
        kind=kind,
        edges=tuple(int(e) for e in edges),
        stops=tuple(stops),
        rejoin_pos=-1 if (terminal or k_rejoin >= ctx.L) else int(k_rejoin),
        orig_edges=orig_edges,
        rejoin_station=int(rejoin_station),
        via_station=int(via),
        bypassed_stations=bypassed,
        detour_time_s=det_time,
        orig_time_s=ctx.orig_time(i, k_rejoin),
    )


def _reroute_detours(g: _Partial, ctx: _PatternCtx, i: int) -> tuple[Detour, ...]:
    """Same-division alternate via a different run out-edge of a diverge node (§4.2 #6).

    Shortest path by run time that rejoins the pattern within ``REROUTE_MAX_STATIONS``
    stations at <= ``REROUTE_MAX_TIME_RATIO`` x the original time, or that reaches a
    terminal served by the route.
    """
    ef, et = g.edge_from, g.edge_to
    node = int(ef[ctx.edges[i]])
    own_next = ctx.edges[i]
    targets: dict[int, int] = {}
    stations_seen = 0
    last_st = ctx.pos_station[i]
    k_far = i + 1
    for k in range(i + 1, ctx.L + 1):
        s = ctx.pos_station[k]
        if s != last_st:
            stations_seen += 1
            last_st = s
        if stations_seen > C.REROUTE_MAX_STATIONS:
            break
        if k >= i + 2:
            n_k = int(ef[ctx.edges[k]]) if k < ctx.L else int(et[ctx.edges[-1]])
            targets.setdefault(n_k, k)
        k_far = k
    route_terms = {int(g.pat_dest_node[q]) for q in g.route_patterns(ctx.route)}
    t_max = C.REROUTE_MAX_TIME_RATIO * max(ctx.orig_time(i, k_far), 60.0)
    best: Detour | None = None
    for e1 in g.node_out(node):
        e1 = int(e1)
        if e1 == own_next or g.edge_kind[e1] != KIND_RUN:
            continue
        if int(g.node_station[et[e1]]) == int(g.node_station[et[own_next]]):
            continue  # a parallel track to the same station is RUN_EXPRESS / RUN_LOCAL
        res = _dijkstra_to_targets(g, int(et[e1]), float(g.edge_run_s[e1]), targets, route_terms, t_max, ctx, i)
        if res is None:
            continue
        path_edges, k_rejoin, term = res
        det = _make_detour(g, ctx, i, "reroute", [e1] + path_edges, k_rejoin, terminal=term)
        if det is None:
            continue
        if best is None or det.detour_time_s < best.detour_time_s:
            best = det
    return (best,) if best is not None else ()


def _dijkstra_to_targets(
    g: _Partial,
    start: int,
    t0: float,
    targets: Mapping[int, int],
    route_terms: set[int],
    t_max: float,
    ctx: _PatternCtx,
    i: int,
) -> tuple[list[int], int, bool] | None:
    et = g.edge_to
    own_nodes_behind = {int(g.edge_from[e]) for e in ctx.edges[: i + 1]}
    dist = {start: t0}
    prev: dict[int, int] = {}
    heap = [(t0, start)]
    while heap:
        t, n = heapq.heappop(heap)
        if t > dist.get(n, math.inf) or t > t_max:
            continue
        if n in targets:
            k = targets[n]
            if t <= C.REROUTE_MAX_TIME_RATIO * max(ctx.orig_time(i, k), 1.0):
                return _unwind(prev, g.edge_from, start, n), k, False
        elif n in route_terms and n not in own_nodes_behind and g.node_platform[n] >= 0:
            return _unwind(prev, g.edge_from, start, n), -1, True
        for e in g.node_out(n):
            e = int(e)
            k = g.edge_kind[e]
            if k not in (KIND_RUN, KIND_CROSSOVER):
                continue
            m = int(et[e])
            if m in own_nodes_behind:
                continue
            nt = t + float(g.edge_run_s[e])
            if nt < dist.get(m, math.inf):
                dist[m] = nt
                prev[m] = e
                heapq.heappush(heap, (nt, m))
    return None


def _unwind(prev: Mapping[int, int], edge_from: np.ndarray, start: int, end: int) -> list[int]:
    out: list[int] = []
    n = end
    while n != start:
        e = prev[n]
        out.append(e)
        n = int(edge_from[e])
    out.reverse()
    return out


def _short_turn_option(g: _Partial, ctx: _PatternCtx, i: int) -> ShortTurnOption | None:
    ef, et = g.edge_from, g.edge_to
    node = int(ef[ctx.edges[i]])
    best: tuple[float, int, int, int] | None = None
    for e in g.node_out(node):
        e = int(e)
        if g.edge_kind[e] != KIND_TURNBACK or not g.edge_pocket[e]:
            continue
        n2 = int(et[e])
        for q in g.route_patterns(ctx.route):
            q = int(q)
            qe = g.pattern_edges(q)
            for j, qq in enumerate(qe):
                if int(ef[qq]) == n2:
                    if j >= len(qe):
                        continue
                    cand = (-float(g.pat_trips[q]), q, j, e)
                    if best is None or cand < best:
                        best = cand
                    break
    if best is None:
        return None
    return ShortTurnOption(turnback_edge=best[3], pattern=best[1], position=best[2])


def _yard_leads(g: _Partial) -> tuple[tuple[YardLead, ...], list[list[int]]]:
    leads: list[YardLead] = []
    pat_leads: list[list[int]] = [[] for _ in range(g.n_patterns)]
    ef, et = g.edge_from, g.edge_to
    for y in range(g.n_yards):
        yn = int(g.yard_node[y])
        lead_nodes: set[int] = set()
        for e in g.yard_lead_edge_idx[g.yard_lead_edge_ptr[y] : g.yard_lead_edge_ptr[y + 1]]:
            e = int(e)
            if int(ef[e]) == yn:
                lead_nodes.add(int(et[e]))
        if yn >= 0:
            for e in g.node_out(yn):
                if g.edge_kind[int(e)] == KIND_YARD_LEAD:
                    lead_nodes.add(int(et[int(e)]))
        routes = {int(r) for r in g.yard_route_idx[g.yard_route_ptr[y] : g.yard_route_ptr[y + 1]]}
        for p in range(g.n_patterns):
            r = int(g.pat_route[p])
            if r not in routes or int(g.route_division[r]) != int(g.yard_division[y]):
                continue
            pe = g.pattern_edges(p)
            for j, e in enumerate(pe):
                n = int(ef[e])
                if n in lead_nodes and g.node_platform[n] >= 0:
                    lid = len(leads)
                    leads.append(YardLead(lid, y, p, j, n, int(g.node_station[n])))
                    pat_leads[p].append(lid)
                    break
    return tuple(leads), pat_leads


# --------------------------------------------------------------------------------------
# Service views
# --------------------------------------------------------------------------------------
def _build_service_view(g: Graph, profile: str, routes: tuple[int, ...] | None) -> ServiceView:
    pat, hw, ph = g.service_raw[profile]
    if routes is not None:
        allowed = np.zeros(g.n_routes, dtype=bool)
        allowed[list(routes)] = True
        keep = allowed[g.pat_route[pat]] if len(pat) else np.zeros(0, dtype=bool)
        pat, hw, ph = pat[keep], hw[keep], ph[keep]
    pattern_hw = np.full(g.n_patterns, np.inf)
    inv_edge = np.zeros(g.n_edges)
    inv_plat = np.zeros(g.n_platforms)  # boarding service: departures that take riders
    for p, h in zip(pat.tolist(), hw.tolist()):
        pattern_hw[p] = min(pattern_hw[p], h)
        pe = g.pattern_edges(p)
        inv_edge[pe] += 1.0 / h
        ps = g.pattern_stops(p).copy()
        ps[-1] = False  # nobody boards at the destination: the train terminates there
        stop_nodes = g.edge_to[pe[ps]]
        pls = g.node_platform[stop_nodes]
        pls = pls[pls >= 0]
        np.add.at(inv_plat, pls, 1.0 / h)
        on = int(g.pat_origin_node[p])
        if g.node_platform[on] >= 0:
            inv_plat[g.node_platform[on]] += 1.0 / h
    with np.errstate(divide="ignore"):
        edge_h = np.where(inv_edge > 0, 1.0 / np.maximum(inv_edge, 1e-12), np.inf)
        plat_h = np.where(inv_plat > 0, 1.0 / np.maximum(inv_plat, 1e-12), np.inf)

    rate, attr = g.demand_raw[profile]
    rate = _route_demand_to_boarding(g, rate, inv_plat)
    attr = attr.copy()

    # attraction suffix sums for stop arrays (decay exp(-j/8) along the stop index)
    S = g.stop_capacity
    suffix = np.zeros((g.n_patterns, S), dtype=np.float64)
    j_idx = np.arange(S)
    w = np.exp(-j_idx / C.DEST_DECAY_STOPS)
    for p in range(g.n_patterns):
        ns = int(g.pat_n_stops[p])
        if ns == 0:
            continue
        sts = g.tmpl_s_station[p, :ns]
        vals = attr[sts] * w[:ns]
        suffix[p, :ns] = np.cumsum(vals[::-1])[::-1]

    # pools: (family, station) -> in/out rates
    pool_index: dict[tuple[int, int], int] = {}
    out_r: list[float] = []
    in_r: list[float] = []
    for p, h in zip(pat.tolist(), hw.tolist()):
        fam = int(g.route_family[g.pat_route[p]])
        for key, is_out in (((fam, int(g.pat_origin_station[p])), True), ((fam, int(g.pat_dest_station[p])), False)):
            if key not in pool_index:
                pool_index[key] = len(pool_index)
                out_r.append(0.0)
                in_r.append(0.0)
            if is_out:
                out_r[pool_index[key]] += 1.0 / h
            else:
                in_r[pool_index[key]] += 1.0 / h
    keys = tuple(sorted(pool_index, key=lambda k: pool_index[k]))
    out_a = np.array(out_r, dtype=np.float64)
    in_a = np.array(in_r, dtype=np.float64)
    target = np.where(out_a > 0, np.ceil(C.LAYOVER_S * out_a) + 1, 0).astype(np.int32)
    deficit = out_a - in_a
    with np.errstate(divide="ignore"):
        putin = np.where(deficit > 1e-9, 1.0 / np.maximum(deficit, 1e-12), np.inf)
    surplus = in_a > out_a + 1e-9

    init_onboard, init_wait = _steady_state(g, pat, hw, rate, attr, plat_h)
    return ServiceView(
        profile=profile,
        routes=routes,
        svc_pattern=_ro(pat.astype(np.int32)),
        svc_headway=_ro(hw.astype(np.float64)),
        svc_phase=_ro(ph.astype(np.float64)),
        pattern_headway=_ro(pattern_hw),
        edge_sched_h=_ro(edge_h),
        platform_sched_h=_ro(plat_h),
        platform_rate=_ro(rate),
        attraction=_ro(attr),
        tmpl_suffix=_ro(suffix),
        pool_keys=keys,
        pool_out_rate=_ro(out_a),
        pool_in_rate=_ro(in_a),
        pool_target=_ro(target),
        pool_putin_period=_ro(putin),
        pool_surplus=_ro(surplus),
        init_onboard=init_onboard,
        init_waiting=_ro(init_wait),
    )


def _route_demand_to_boarding(g: Graph, rate: np.ndarray, inv_board: np.ndarray) -> np.ndarray:
    """Send each platform's entering riders to a platform of the station that has service.

    Riders entering a station walk to a platform where trains depart. A platform without
    boarding service in this profile (a terminal's arrival side, an express platform with
    no express service, a route not operated in a Stage-1 scenario) passes its demand to
    the station's platforms with boarding service, in proportion to their departures. A
    station without any boarding service generates no riders.
    """
    out = np.zeros_like(rate, dtype=np.float64)
    served = inv_board > 0
    out[served] = rate[served]
    for s in range(g.n_stations):
        pls = g.station_platforms(s)
        if len(pls) == 0:
            continue
        orphan = float(rate[pls][~served[pls]].sum())
        if orphan <= 0.0:
            continue
        w = inv_board[pls]
        if w.sum() > 0:
            out[pls] += orphan * w / w.sum()
    return out


def _steady_state(
    g: Graph,
    pat: np.ndarray,
    hw: np.ndarray,
    rate: np.ndarray,
    attr: np.ndarray,
    plat_h: np.ndarray,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Approximate steady-state onboard buckets per pattern stop and platform queues.

    Used only to initialise a reset close to equilibrium (the warm-up does the rest).
    """
    nd = C.N_DEST
    k_idx = np.arange(nd)
    decay = np.exp(-k_idx / C.DEST_DECAY_STOPS)
    q_tail = math.exp(-1.0 / C.DEST_DECAY_STOPS)
    onboard: dict[int, np.ndarray] = {}
    for p in sorted(set(pat.tolist())):
        ns = int(g.pat_n_stops[p])
        cap = g.train_capacity(int(g.route_division[g.pat_route[p]]))
        sts = [int(g.node_station[g.pat_origin_node[p]])] + [int(s) for s in g.tmpl_s_station[p, :ns]]
        nodes = [int(g.pat_origin_node[p])] + [int(g.edge_to[g.tmpl_edge[p, k]]) for k in g.tmpl_s_pos[p, :ns]]
        loads = np.zeros((ns + 1, nd), dtype=np.float32)
        b = np.zeros(nd, dtype=np.float64)
        for j in range(ns + 1):
            if j > 0:
                b = shift_buckets(b, q_tail)
            pl = int(g.node_platform[nodes[j]])
            if pl >= 0 and j < ns and np.isfinite(plat_h[pl]):
                board = float(rate[pl] * min(plat_h[pl], 1800.0))
                board = max(0.0, min(board, cap - float(b.sum())))
                nxt = sts[j + 1 : j + 1 + nd]
                if nxt and board > 0:
                    w = np.zeros(nd)
                    w[: len(nxt)] = attr[nxt] * decay[: len(nxt)]
                    if len(sts) - (j + 1) > nd:
                        tail = attr[sts[j + 1 + nd - 1 :]] * np.exp(-np.arange(nd - 1, len(sts) - j - 1) / C.DEST_DECAY_STOPS)
                        w[nd - 1] = float(tail.sum())
                    if w.sum() > 0:
                        b = b + board * w / w.sum()
            loads[j] = b
        onboard[p] = _ro(loads)
    wait = np.zeros((g.n_platforms, C.N_AGE_BINS), dtype=np.float32)
    for pl in range(g.n_platforms):
        h = plat_h[pl]
        if not np.isfinite(h) or rate[pl] <= 0:
            continue
        nb = max(1, int(round(min(h, 1800.0) / (2 * C.AGE_BIN_S))))
        wait[pl, :nb] = rate[pl] * C.AGE_BIN_S
    return onboard, wait
