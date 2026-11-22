"""Weekday service tables (ARCHITECTURE §2.3.14) and 3-track peak directions (§2.3.4, §3.3).

* ``stop_counts``: scheduled train stops per platform and clock hour (times mod 24 h), the basis
  for splitting a complex's entries over its platforms (§2.3.13).
* ``service_table``: per profile and pattern, ``headway_s`` = window length / departures in the
  window and ``phase_s`` = first departure offset mod ``headway_s`` (patterns without a departure
  in the window are omitted).
* ``peak_directions``: for every bidir group (a 3-track line's middle track) and profile, the
  direction the middle track may be used in. It is the direction of the majority of scheduled
  X traversals by trips that start in the window; with no such traversal it is the curated peak
  (``am_toward_manhattan`` / ``am_away_from_manhattan``) for the profile's AM/PM sense.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.netbuild.constants import (
    DAY_S,
    MANHATTAN_REF_LATLON,
    PEAK_PROFILE_SENSE,
    PROFILE_WINDOWS_H,
    PROFILES,
    TIME_DECIMALS,
)
from scripts.netbuild.geo import project
from scripts.netbuild.jsonio import rnd
from scripts.netbuild.patterns import Pattern
from scripts.netbuild.topology import Topology
from scripts.netbuild.tracks import ResolvedCorridor, X


def pattern_stop_nodes(topo: Topology, pid: int) -> list[int]:
    """Node of every scheduled stop of a pattern, origin first (one per ``Pattern.stops`` entry)."""
    path = topo.paths[pid]
    out = [topo.edges[path.edges[0]].src]
    out.extend(topo.edges[e].dst for e, s in zip(path.edges, path.stops) if s)
    return out


def stop_counts(kept: Sequence[Pattern], topo: Topology) -> np.ndarray:
    """``int64[n_platforms, 24]``: scheduled weekday train stops per platform and clock hour."""
    counts = np.zeros((len(topo.platforms), 24), dtype=np.int64)
    for p in kept:
        plats = [topo.nodes[n].platform for n in pattern_stop_nodes(topo, p.pid)]
        if len(plats) != len(p.stops) or min(plats) < 0:
            raise ValueError(f"pattern {p.pid} ({p.route_id}): stop nodes do not match its stops")
        arr = np.asarray([t.arr for t in p.trips], dtype=np.int64)  # (n_trips, n_stops)
        hours = (arr % DAY_S) // 3600
        for k, pl in enumerate(plats):
            counts[pl] += np.bincount(hours[:, k], minlength=24)
    return counts


def window_hours(profile: str) -> list[int]:
    h0, h1 = PROFILE_WINDOWS_H[profile]
    return list(range(h0, h1))


def service_table(kept: Sequence[Pattern]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for prof in PROFILES:
        h0, h1 = PROFILE_WINDOWS_H[prof]
        lo, hi = h0 * 3600, h1 * 3600
        rows = []
        for p in sorted(kept, key=lambda q: q.pid):
            deps = sorted(t.start_s for t in p.trips if lo <= t.start_s < hi)
            if not deps:
                continue
            headway = rnd((hi - lo) / len(deps), TIME_DECIMALS)
            phase = rnd(math.fmod(deps[0] - lo, headway), TIME_DECIMALS)
            rows.append({"pattern": p.pid, "headway_s": headway, "phase_s": phase if phase < headway else 0.0})
        out[prof] = rows
    return out


def routes_without_service(kept: Sequence[Pattern], profile: str) -> list[str]:
    h0, h1 = PROFILE_WINDOWS_H[profile]
    served = {p.route_id for p in kept if any(h0 * 3600 <= t.start_s < h1 * 3600 for t in p.trips)}
    return sorted({p.route_id for p in kept} - served)


def _toward_manhattan(corridor: ResolvedCorridor, station_latlon: Mapping[str, tuple[float, float]]) -> dict[tuple[str, str], bool]:
    """For each directed hop of the corridor: does it run toward the corridor's Manhattan end?"""
    ref = project(*MANHATTAN_REF_LATLON)
    chain = corridor.stations

    def dist(s: str) -> float:
        x, y = project(*station_latlon[s])
        return math.hypot(x - ref[0], y - ref[1])

    first_is_manhattan = dist(chain[0]) < dist(chain[-1])
    out = {}
    for a, b in zip(chain, chain[1:]):
        out[(b, a)] = first_is_manhattan  # b -> a walks toward chain[0]
        out[(a, b)] = not first_is_manhattan
    return out


def peak_directions(
    topo: Topology,
    kept: Sequence[Pattern],
    corridors: Sequence[ResolvedCorridor],
    station_latlon: Mapping[str, tuple[float, float]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """``peak_dir`` entries (sorted by group, then profile order) and consistency notes."""
    notes: list[str] = []
    group_edges: dict[int, list[int]] = defaultdict(list)
    for e in topo.edges:
        if e.kind == "run" and e.track == X and e.bidir_group >= 0:
            group_edges[e.bidir_group].append(e.idx)
    by_group = {c.group: c for c in corridors}
    sense_of: dict[int, dict[tuple[str, str], bool]] = {g: _toward_manhattan(c, station_latlon)
                                                        for g, c in by_group.items()}
    votes: dict[tuple[int, str], dict[bool, int]] = defaultdict(lambda: {True: 0, False: 0})
    for p in kept:
        path = topo.paths[p.pid]
        used: dict[int, set[bool]] = defaultdict(set)
        for ei in path.edges:
            e = topo.edges[ei]
            if e.kind == "run" and e.track == X and e.bidir_group >= 0:
                used[e.bidir_group].add(sense_of[e.bidir_group][e.hop])
        if not used:
            continue
        for prof in PROFILES:
            h0, h1 = PROFILE_WINDOWS_H[prof]
            n = sum(1 for t in p.trips if h0 * 3600 <= t.start_s < h1 * 3600)
            if n:
                for g, senses in used.items():
                    for toward in senses:
                        votes[(g, prof)][toward] += n
    out = []
    for g in sorted(group_edges):
        c = by_group.get(g)
        if c is None:
            raise ValueError(f"bidir group {g} has X edges but no curated corridor")
        am_toward = c.peak == "am_toward_manhattan"
        for prof in PROFILES:
            curated = am_toward if PEAK_PROFILE_SENSE[prof] == "am" else not am_toward
            v = votes.get((g, prof))
            if v and v[True] != v[False]:
                toward = v[True] > v[False]
                if prof in ("am_peak", "pm_rush") and toward != curated:
                    notes.append(f"{c.name} {prof}: scheduled X service runs "
                                 f"{'toward' if toward else 'away from'} Manhattan, against the curated peak")
            else:
                toward = curated
            allowed = sorted(ei for ei in group_edges[g] if sense_of[g][topo.edges[ei].hop] == toward)
            out.append({"bidir_group": g, "profile": prof, "allowed_dir_edges": allowed})
    return out, notes
