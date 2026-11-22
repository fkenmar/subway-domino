"""Disruption targets, effects and labels (ARCHITECTURE §3.5).

A disruption is a ``(kind, target_type, target_idx)`` triple. :func:`resolve` validates it
against the graph and returns the static :class:`Effect` (which edges are blocked or
signal-delayed, which nodes/stations are closed). The engine combines the effects of all
active disruptions into per-edge/per-node arrays whenever the set changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np

from sim import constants as C
from sim.graph import KIND_CROSSOVER, KIND_RUN, KIND_TURNBACK, TRACK_L, TRACK_X, Graph

KIND_CODE: dict[str, int] = {k: i + 1 for i, k in enumerate(C.DISRUPTION_KINDS)}  # 0 = none
TARGET_TYPES: tuple[str, ...] = ("station", "complex", "segment", "interlocking")


@dataclass(frozen=True, eq=False)
class Effect:
    kind: str
    target_type: str
    target_idx: int
    blocked_edges: np.ndarray  # run edges whose entry signal is held at red
    closed_nodes: np.ndarray
    closed_stations: np.ndarray
    delayed_edges: np.ndarray  # signal_delay: slow, single occupancy, key-by stop
    disabled_edges: np.ndarray  # crossovers/turnbacks made unusable
    station_idx: int  # representative station (callouts, distance features)
    label: str


def _arr(xs: Iterable[int]) -> np.ndarray:
    return np.array(sorted(set(int(x) for x in xs)), dtype=np.int64)


def _station_track_edges(g: Graph, s: int, track: int) -> list[int]:
    """Run edges of ``track`` entering or leaving the station's ``track`` nodes."""
    out: list[int] = []
    for n in g.station_nodes(s):
        n = int(n)
        if int(g.node_track[n]) != track:
            continue
        for e in list(g.node_out(n)) + list(g.node_in(n)):
            e = int(e)
            if int(g.edge_kind[e]) == KIND_RUN and int(g.edge_track[e]) == track:
                out.append(e)
    return out


def _station_all_edges(g: Graph, s: int) -> list[int]:
    out: list[int] = []
    for n in g.station_nodes(s):
        for e in list(g.node_out(int(n))) + list(g.node_in(int(n))):
            out.append(int(e))
    return out


def _seg_label(g: Graph, seg: int) -> str:
    a = g.station_name[int(g.segment_a[seg])]
    b = g.station_name[int(g.segment_b[seg])]
    return f"{a} ↔ {b}"


def resolve(g: Graph, kind: str, target_type: str, target_idx: int) -> Effect:
    """Validate ``(kind, target)`` and compute its effect; ``ValueError`` if invalid."""
    return _resolve_cached(g, str(kind), str(target_type), int(target_idx))


@lru_cache(maxsize=4096)
def _resolve_cached(g: Graph, kind: str, target_type: str, idx: int) -> Effect:
    if kind not in C.DISRUPTION_KINDS:
        raise ValueError(f"unknown disruption kind {kind!r}")
    if target_type not in C.DISRUPTION_TARGETS[kind]:
        raise ValueError(f"{kind} cannot target a {target_type}")
    limits = {"station": g.n_stations, "complex": g.n_complexes, "segment": g.n_segments, "interlocking": g.n_interlockings}
    if not (0 <= idx < limits[target_type]):
        raise ValueError(f"{target_type} index {idx} out of range")
    empty = np.zeros(0, dtype=np.int64)
    blocked, closed_n, closed_s, delayed, disabled = empty, empty, empty, empty, empty

    if target_type == "station":
        station = idx
    elif target_type == "interlocking":
        station = int(g.interlocking_station[idx])
    elif target_type == "segment":
        station = int(g.segment_a[idx])
    else:
        sts = g.complex_stations(idx)
        station = int(sts[0]) if len(sts) else -1

    if kind in ("local_block", "express_block"):
        track = TRACK_L if kind == "local_block" else TRACK_X
        word = "Local" if kind == "local_block" else "Express"
        if target_type == "segment":
            edges = [int(e) for e in g.segment_edges(idx) if int(g.edge_kind[e]) == KIND_RUN and int(g.edge_track[e]) == track]
            label = f"{word} track blocked: {_seg_label(g, idx)}"
        else:
            edges = _station_track_edges(g, station, track)
            label = f"{word} track blocked at {g.station_name[station]}"
        if not edges:
            raise ValueError(f"{target_type} {idx} has no {word.lower()} track")
        blocked = _arr(edges)
    elif kind == "full_station_closure":
        if target_type == "station":
            stations = [idx]
            label = f"Station closed: {g.station_name[idx]}"
        else:
            stations = [int(s) for s in g.complex_stations(idx)]
            if not stations:
                raise ValueError(f"complex {idx} has no stations")
            label = f"Station closed: {g.complex_name[idx]}"
        nodes = [int(n) for s in stations for n in g.station_nodes(s)]
        if not nodes:
            raise ValueError(f"{target_type} {idx} has no track nodes")
        closed_n = _arr(nodes)
        closed_s = _arr(stations)
    else:  # signal_delay
        if target_type == "segment":
            edges = [int(e) for e in g.segment_edges(idx) if int(g.edge_kind[e]) == KIND_RUN]
            label = f"Signal problems: {_seg_label(g, idx)}"
            dis: list[int] = []
        else:
            all_e = _station_all_edges(g, station)
            edges = [e for e in all_e if int(g.edge_kind[e]) == KIND_RUN]
            dis = [e for e in all_e if int(g.edge_kind[e]) in (KIND_CROSSOVER, KIND_TURNBACK)]
            label = f"Signal problems at {g.station_name[station]}"
        if not edges:
            raise ValueError(f"{target_type} {idx} has no running track")
        delayed = _arr(edges)
        disabled = _arr(dis)
    return Effect(
        kind=kind,
        target_type=target_type,
        target_idx=idx,
        blocked_edges=blocked,
        closed_nodes=closed_n,
        closed_stations=closed_s,
        delayed_edges=delayed,
        disabled_edges=disabled,
        station_idx=int(station),
        label=label,
    )


def is_valid(g: Graph, kind: str, target_type: str, target_idx: int) -> bool:
    try:
        resolve(g, kind, target_type, target_idx)
    except ValueError:
        return False
    return True


def valid_targets(g: Graph, kind: str) -> list[tuple[str, int]]:
    """All valid ``(target_type, idx)`` for ``kind`` (cached on the graph)."""
    key = ("valid_targets", kind)
    hit = g._cache.get(key)
    if hit is not None:
        return hit
    limits = {"station": g.n_stations, "complex": g.n_complexes, "segment": g.n_segments, "interlocking": g.n_interlockings}
    out: list[tuple[str, int]] = []
    for tt in C.DISRUPTION_TARGETS[kind]:
        for i in range(limits[tt]):
            if is_valid(g, kind, tt, i):
                out.append((tt, i))
    g._cache[key] = out
    return out


def target_stations(g: Graph, target_type: str, idx: int) -> tuple[int, ...]:
    """Stations a target touches (used to restrict Stage-1 disruptions to a trunk)."""
    if target_type == "station":
        return (idx,)
    if target_type == "interlocking":
        return (int(g.interlocking_station[idx]),)
    if target_type == "segment":
        return (int(g.segment_a[idx]), int(g.segment_b[idx]))
    return tuple(int(s) for s in g.complex_stations(idx))
