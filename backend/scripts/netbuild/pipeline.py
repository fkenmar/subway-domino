"""End-to-end network build: sources + curated overlay -> in-memory model (ARCHITECTURE §2.2-2.5).

``build_model`` runs every inference step in a fixed order and returns a :class:`NetworkModel`;
``scripts.netbuild.assemble`` turns it into the ``network.json`` / ``network_geo.json`` documents.
Nothing here touches the network or the file system.

Index conventions (all deterministic):
routes by GTFS ``route_sort_order``; stations and complexes by natural id order; patterns by
(route, most trips first, stop sequence); nodes, edges, platforms, segments and interlockings in
the order of :func:`scripts.netbuild.topology.build_topology`; yards in ``infrastructure.json`` order.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from scripts.netbuild import patterns as P
from scripts.netbuild.constants import CROSSOVER_S, EXPECTED_ROUTES, IN_SCOPE_ROUTES, SHUTTLE_ROUTES
from scripts.netbuild.corridors import cbtc_hop_set, resolve_corridors
from scripts.netbuild.demand import Demand, build_demand, hub_stations, label_ranks
from scripts.netbuild.geo import project
from scripts.netbuild.geometry import SegmentGeometry, basemap_land, polyline_point_at_half, segment_geometry
from scripts.netbuild.gtfs import GtfsFeed, GtfsRoute, parse_gtfs
from scripts.netbuild.infra import Infrastructure, InfrastructureError
from scripts.netbuild.places import Places, build_places, division_of, natural_key
from scripts.netbuild.presets import PresetIndex, resolve_presets
from scripts.netbuild.service import peak_directions, routes_without_service, service_table, stop_counts
from scripts.netbuild.sources import Sources
from scripts.netbuild.timing import EdgeTiming, edge_timing
from scripts.netbuild.topology import StationRef, Topology, TurnbackRequest, YardRequest, build_topology
from scripts.netbuild.tracks import TrackModel, X, infer_tracks


class BuildError(RuntimeError):
    """The network cannot be built consistently from the inputs."""


@dataclass
class NetworkModel:
    feed: GtfsFeed
    infra: Infrastructure
    routes: list[GtfsRoute]
    route_idx: dict[str, int]
    station_ids: list[str]
    station_idx: dict[str, int]
    station_xy: dict[str, tuple[float, float]]
    kept: list[P.Pattern]
    decomposition: P.Decomposition
    tracks: TrackModel
    topo: Topology
    geometry: SegmentGeometry
    timing: EdgeTiming
    places: Places
    stop_counts: np.ndarray
    service: dict[str, list[dict[str, Any]]]
    peak_dir: list[dict[str, Any]]
    demand: Demand
    hub: np.ndarray
    label_rank: np.ndarray
    river: list[bool]
    presets: list[dict[str, Any]]
    basemap: list[np.ndarray] | None
    turnback_evidence: dict[str, list[str]]
    stats: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)


def _turnback_evidence(all_patterns: list[P.Pattern]) -> dict[str, list[str]]:
    """§2.3.6(b): non-terminal stations where some weekday pattern starts or ends -> those routes.

    A station is non-terminal when some weekday pattern passes through it (stops there in the
    middle of its run). All weekday patterns count, however few trips they have.
    """
    ends: dict[str, set[str]] = defaultdict(set)
    passes: set[str] = set()
    for p in all_patterns:
        ends[p.stops[0]].add(p.route_id)
        ends[p.stops[-1]].add(p.route_id)
        passes.update(p.stops[1:-1])
    return {s: sorted(rs) for s, rs in sorted(ends.items()) if s in passes}


def build_model(sources: Sources, infra: Infrastructure, log: Callable[[str], None] = lambda _m: None) -> NetworkModel:
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[name] = round(now - clock, 3)
        clock = now
        log(f"  {name}: {timings[name]:.2f}s")

    notes: list[str] = []
    feed = parse_gtfs(sources.gtfs_zip)
    lap("parse_gtfs")

    # ---------------------------------------------------------------- patterns and scope (§2.2, §2.3.1-2)
    all_patterns = P.group_patterns(feed.trips)
    P.orient_patterns(all_patterns)
    kept = P.select_patterns(all_patterns)
    route_ids = sorted({p.route_id for p in kept}, key=lambda r: (feed.routes[r].sort_order, r))
    if set(route_ids) != set(IN_SCOPE_ROUTES) or len(route_ids) != EXPECTED_ROUTES:
        missing = sorted(set(IN_SCOPE_ROUTES) - set(route_ids))
        raise BuildError(f"expected the {EXPECTED_ROUTES} in-scope routes, missing {missing}")
    route_idx = {r: i for i, r in enumerate(route_ids)}
    for p in kept:
        if any(division_of(s) != division_of(p.stops[0]) for s in p.stops):
            raise BuildError(f"pattern of {p.route_id} {p.stops[0]}->{p.stops[-1]} mixes divisions")
    kept.sort(key=lambda p: (route_idx[p.route_id], -p.n_trips, p.stops))
    for i, p in enumerate(kept):
        p.pid = i
        p.profiles = P.service_profiles(p)
    station_ids = sorted({s for p in kept for s in p.stops}, key=natural_key)
    station_idx = {s: i for i, s in enumerate(station_ids)}
    station_xy = {s: project(feed.stations[s].lat, feed.stations[s].lon) for s in station_ids}
    dec = P.decompose_hops(kept, station_xy)
    P.expand_paths(kept, dec)
    for p in kept:
        if not set(p.path) <= set(station_idx):
            raise BuildError(f"pattern {p.pid} passes stations that are no pattern's stop")
    if dec.rejected:
        notes.extend(f"hop {h} kept elementary: {why}" for h, why in sorted(dec.rejected.items()))
    lap("patterns")

    # ---------------------------------------------------------------- tracks (§2.3.4, §2.3.8)
    track_corridors, cbtc_corridors = resolve_corridors(infra, kept, dec.elementary)
    hop_len = {h: math.dist(station_xy[h[0]], station_xy[h[1]]) for h in dec.elementary}
    tm = infer_tracks(kept, feed.trips, dec, track_corridors, hop_len)
    cbtc_hops = cbtc_hop_set(cbtc_corridors, dec.elementary)
    lap("tracks")

    # ---------------------------------------------------------------- topology (§2.3.3, 5-9)
    station_refs = {s: StationRef(s, division_of(s), station_xy[s]) for s in station_ids}
    evidence = _turnback_evidence(all_patterns)
    requests = [TurnbackRequest(s, tuple(rs), "evidence") for s, rs in evidence.items()]
    for t in infra.turnbacks:
        if t.gtfs not in station_idx:
            raise InfrastructureError(f"curated turnback at unknown station {t.gtfs!r}")
        requests.append(TurnbackRequest(t.gtfs, t.routes, "curated"))
    route_division = {p.route_id: division_of(p.stops[0]) for p in kept}
    yard_requests = []
    for y in infra.yards:
        wrong = [r for r in y.routes if route_division.get(r) != y.division]
        wrong += [s for s in y.lead_gtfs if division_of(s) != y.division]
        if wrong:
            raise InfrastructureError(f"yard {y.name!r} ({y.division} division): {wrong} are not in its division")
        yard_requests.append(YardRequest(y.name, y.division, project(y.lat, y.lon), y.lead_gtfs, y.routes,
                                         y.fleet_cap, y.deadhead_s))
    topo = build_topology(kept, tm.classes, set(tm.xtrack), tm.bidir_group, station_refs, station_ids,
                          requests, yard_requests, cbtc_hops, CROSSOVER_S)
    notes.extend(topo.warnings)
    lap("topology")

    # ---------------------------------------------------------------- geometry and timing (§2.3.10-11)
    geometry = segment_geometry(topo.segments, kept, feed.shapes, station_xy)
    notes.extend(geometry.warnings)
    timing = edge_timing(topo, kept, geometry.length_m)
    notes.extend(timing.warnings)
    basemap = basemap_land(sources.basemap)
    lap("geometry_timing")

    # ---------------------------------------------------------------- places, service, demand (§2.3.12-14)
    places = build_places(feed, station_ids, sources.complexes, sources.stations, sources.basemap)
    notes.extend(places.warnings)
    counts = stop_counts(kept, topo)
    service = service_table(kept)
    latlon = {s: (feed.stations[s].lat, feed.stations[s].lon) for s in station_ids}
    peak_dir, peak_notes = peak_directions(topo, kept, track_corridors, latlon)
    notes.extend(peak_notes)
    station_complex = np.asarray([places.complex_of[s] for s in station_ids], dtype=np.int64)
    platform_station = np.asarray([station_idx[pl.station] for pl in topo.platforms], dtype=np.int64)
    complex_routes = [sorted({r for s in c.stations for r in topo.station_routes.get(s, ())}) for c in places.complexes]
    demand = build_demand(places.complexes, station_complex, platform_station, counts,
                          [len(r) for r in complex_routes], sources.ridership.rows if sources.ridership else None)
    notes.extend(demand.warnings)
    hub = hub_stations(demand.weekday_entries, station_complex, [len(r) for r in complex_routes], station_ids,
                       infra.hubs_extra)
    ranks = label_ranks(demand.station_entries, demand.weekday_entries[station_complex])
    lap("demand_service")

    # ---------------------------------------------------------------- river crossings, presets (§2.4)
    seg_index: dict[tuple[str, str], int] = {}
    for seg in topo.segments:
        seg_index[(seg.a, seg.b)] = seg.idx
        seg_index[(seg.b, seg.a)] = seg.idx
    river = [False] * len(topo.segments)
    for rc in infra.river_crossings:
        si = seg_index.get((rc.a, rc.b))
        if si is None:
            raise InfrastructureError(f"river crossing {rc.name!r} {rc.a}-{rc.b} is not a segment")
        river[si] = True
    station_has_x = [False] * len(station_ids)
    for n in topo.nodes:
        if n.track == X and n.station:
            station_has_x[station_idx[n.station]] = True
    segment_has_x = [any(topo.edges[e].track == X for e in seg.edges) for seg in topo.segments]
    il_at = {station_idx[il.station]: il.idx for il in topo.interlockings}
    index = PresetIndex(
        station=station_idx,
        complex_by_id={c.complex_id: i for i, c in enumerate(places.complexes)},
        station_complex=[int(c) for c in station_complex],
        segment={(station_idx[a], station_idx[b]): si for (a, b), si in seg_index.items()},
        interlocking_at=il_at,
        station_has_x=station_has_x,
        segment_has_x=segment_has_x,
        station_xy=[station_xy[s] for s in station_ids],
        complex_xy=[project(c.lat, c.lon) for c in places.complexes],
        segment_mid=[polyline_point_at_half(geometry.pts[seg.idx]) for seg in topo.segments],
    )
    presets = resolve_presets(infra.presets, index)
    lap("presets")

    am_missing = routes_without_service(kept, "am_peak")
    stats = {
        "patterns_all": len(all_patterns),
        "patterns_kept": len(kept),
        "stop_hops": len({h for p in kept for h in zip(p.stops, p.stops[1:])}),
        "bypass_hops": len(dec.chains),
        "elementary_hops": len(dec.elementary),
        "x_hops": len(tm.xtrack),
        "x_hops_by_reason": {r: sum(1 for h in tm.xtrack if r in tm.reasons.get(h, ())) for r in
                             sorted({r for rs in tm.reasons.values() for r in rs})},
        "turnback_evidence_stations": len(evidence),
        "routes_without_am_peak_service": am_missing,
        "segment_geometry": {k: sum(1 for v in geometry.source.values() if v.startswith(k)) for k in ("shape", "straight")},
        "edge_timing_sources": {k: sum(1 for v in timing.source.values() if v == k) for k in
                                ("schedule", "parallel", "reverse", "fallback")},
        "complex_source": places.complex_source,
        "station_meta_source": places.meta_source,
        "demand_source": demand.source,
        "system_entries_h": {prof: round(demand.system_entries_h(prof), 1) for prof in demand.platform_rate},
        "unserved_entries_h": {k: round(v, 1) for k, v in demand.unserved_entries_h.items()},
        "weekday_entries": round(float(demand.weekday_entries.sum()), 1),
        "hubs": int(hub.sum()),
        **topo.stats,
    }
    return NetworkModel(
        feed=feed, infra=infra, routes=[feed.routes[r] for r in route_ids], route_idx=route_idx,
        station_ids=station_ids, station_idx=station_idx, station_xy=station_xy, kept=kept, decomposition=dec,
        tracks=tm, topo=topo, geometry=geometry, timing=timing, places=places, stop_counts=counts,
        service=service, peak_dir=peak_dir, demand=demand, hub=hub, label_rank=ranks, river=river,
        presets=presets, basemap=basemap, turnback_evidence=evidence, stats=stats, notes=notes,
        timings_s=timings,
    )


def shuttle(route_id: str) -> bool:
    return route_id in SHUTTLE_ROUTES
