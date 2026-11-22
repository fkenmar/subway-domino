"""Run times and speed limits of running edges (ARCHITECTURE §2.3.11).

For every weekday trip and stop-to-stop hop, the scheduled time ``T = arr[k+1] - dep[k]`` is made
attributable to the hop's running edges:

* ``T`` is first **de-quantized** (:func:`dequantize`): GTFS stop times are rounded to 30 s, so the
  literal per-hop times alternate (the 1 train runs 72 St -> 79 St in 120 s and 79 St -> 86 St in
  60 s over hops of equal length), which would give consecutive edges the 8 m/s floor and the
  24.6 m/s cap. Each hop instead gets the scheduled time of the 3 hops centered on it, shared in
  proportion to a kinematic weight; trip end-to-end times are preserved exactly;
* ``A = max(T - NOMINAL_DWELL_S - n_crossovers * CROSSOVER_S, MIN_STOP_HOP_S)`` removes the nominal
  dwell (GTFS dwell is almost always 0, so ``T`` includes it) and the crossover moves the simulator
  times separately, with a 30 s floor;
* a single-edge hop attributes ``A`` to its edge;
* an express hop spanning several edges splits its cruise time ``A - ACCEL_DECEL_LOSS_S`` by edge
  length and gives each edge one ``ACCEL_DECEL_LOSS_S``. The §2.6 formula
  ``v_max = clamp(length / max(run_s - ACCEL_DECEL_LOSS_S, 20), 8, 24.6)`` then yields the hop's
  true cruise speed on every edge (the simulator accelerates and brakes once per stop-to-stop run).

``run_s`` is the median over all samples of an edge. Edges without a scheduled traversal (a middle
track without service, the reverse of a one-way express hop) copy the parallel edge of the same
hop, then the reverse hop, then fall back to ``FALLBACK_SPEED_MS``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from scripts.netbuild.constants import (
    ACCEL_DECEL_LOSS_S,
    CROSSOVER_S,
    DEQUANTIZE_HALF_WINDOW,
    DEQUANTIZE_REF_SPEED_MS,
    FALLBACK_SPEED_MS,
    MIN_RUNNING_S_FOR_VMAX,
    MIN_STOP_HOP_S,
    NOMINAL_DWELL_S,
    V_MAX_CAP,
    V_MAX_FLOOR_MS,
)
from scripts.netbuild.patterns import Pattern
from scripts.netbuild.topology import Topology


@dataclass
class EdgeTiming:
    length_m: dict[int, float]  # run edge idx -> meters
    run_s: dict[int, float]
    v_max: dict[int, float]
    samples: dict[int, int]  # number of scheduled traversals behind run_s
    source: dict[int, str]  # "schedule" | "parallel" | "reverse" | "fallback"
    warnings: list[str] = field(default_factory=list)


def v_max_for(length_m: float, run_s: float) -> float:
    return float(np.clip(length_m / max(run_s - ACCEL_DECEL_LOSS_S, MIN_RUNNING_S_FOR_VMAX), V_MAX_FLOOR_MS, V_MAX_CAP))


def stop_hop_groups(topo: Topology, pid: int) -> list[tuple[list[int], int]]:
    """The pattern's edges grouped per scheduled stop-to-stop hop: ``(run edges, n_crossovers)``."""
    edges = topo.edges
    path = topo.paths[pid]
    groups: list[tuple[list[int], int]] = []
    group: list[int] = []
    for ei, stop in zip(path.edges, path.stops):
        group.append(ei)
        if stop:
            groups.append(([g for g in group if edges[g].kind == "run"],
                           sum(1 for g in group if edges[g].kind == "crossover")))
            group = []
    return groups


def dequantize(hop_t: np.ndarray, weight: np.ndarray, half_window: int = DEQUANTIZE_HALF_WINDOW) -> np.ndarray:
    """Undo the 30 s rounding of GTFS stop times along each trip (rows of ``hop_t``).

    Every stop time is rounded to 30 s, so one hop's scheduled time is off by up to 30 s (a 60 s hop
    next to a 120 s hop of the same length), while the time over several consecutive hops carries
    the same small error. Each hop gets the scheduled time of the ``2 * half_window + 1`` hops
    centered on it, shared in proportion to ``weight`` (a kinematic estimate of the hop), and every
    trip is rescaled to keep its scheduled end-to-end time.
    """
    n = hop_t.shape[1]
    if n == 0:
        return hop_t.astype(np.float64, copy=True)
    t = hop_t.astype(np.float64)
    ct = np.concatenate([np.zeros((t.shape[0], 1)), np.cumsum(t, axis=1)], axis=1)
    cw = np.concatenate([[0.0], np.cumsum(weight)])
    k = np.arange(n)
    lo = np.clip(k - half_window, 0, n)
    hi = np.clip(k + half_window + 1, 0, n)
    window_t = ct[:, hi] - ct[:, lo]
    window_w = cw[hi] - cw[lo]
    out = weight[None, :] * window_t / window_w[None, :]
    got = out.sum(axis=1)
    want = t.sum(axis=1)
    scale = np.where(got > 0, want / np.where(got > 0, got, 1.0), 1.0)
    return out * scale[:, None]


def edge_timing(topo: Topology, kept: Sequence[Pattern], segment_length: Mapping[int, float]) -> EdgeTiming:
    edges = topo.edges
    seg_of_edge: dict[int, int] = {}
    for seg in topo.segments:
        for e in seg.edges:
            seg_of_edge[e] = seg.idx
    length = {e.idx: float(segment_length[seg_of_edge[e.idx]]) for e in edges if e.kind == "run"}

    samples: dict[int, list[np.ndarray]] = defaultdict(list)
    for p in kept:
        groups = stop_hop_groups(topo, p.pid)
        arr = np.asarray([t.arr for t in p.trips], dtype=np.float64)
        dep = np.asarray([t.dep for t in p.trips], dtype=np.float64)
        hop_t = arr[:, 1:] - dep[:, :-1]  # (n_trips, n_stops - 1), quantized to 30 s
        if len(groups) != hop_t.shape[1]:
            raise ValueError(f"pattern {p.pid} ({p.route_id}): {len(groups)} stop hops on the path, "
                             f"{hop_t.shape[1]} in GTFS")
        hop_len = np.asarray([sum(length[g] for g in runs) for runs, _ in groups], dtype=np.float64)
        n_xover = np.asarray([n_x for _, n_x in groups], dtype=np.float64)
        weight = (NOMINAL_DWELL_S + n_xover * CROSSOVER_S + ACCEL_DECEL_LOSS_S
                  + hop_len / DEQUANTIZE_REF_SPEED_MS)
        smooth = dequantize(hop_t, weight)
        for k, (runs, n_x) in enumerate(groups):
            avail = np.maximum(smooth[:, k] - NOMINAL_DWELL_S - n_x * CROSSOVER_S, MIN_STOP_HOP_S)
            if len(runs) == 1:
                samples[runs[0]].append(avail)
            elif runs:
                total = sum(length[g] for g in runs)
                cruise = np.maximum(avail - ACCEL_DECEL_LOSS_S, 1.0)
                for g in runs:
                    samples[g].append(cruise * (length[g] / total) + ACCEL_DECEL_LOSS_S)

    run_s: dict[int, float] = {}
    count: dict[int, int] = {}
    source: dict[int, str] = {}
    for ei, chunks in samples.items():
        allv = np.concatenate(chunks)
        run_s[ei] = float(np.median(allv))
        count[ei] = int(allv.size)
        source[ei] = "schedule"

    by_hop: dict[tuple[str, str], list[int]] = defaultdict(list)
    for e in edges:
        if e.kind == "run":
            by_hop[e.hop].append(e.idx)
    warnings: list[str] = []
    for e in edges:
        if e.kind != "run" or e.idx in run_s:
            continue
        speed = None
        for kind, pool in (("parallel", by_hop[e.hop]), ("reverse", by_hop.get((e.hop[1], e.hop[0]), []))):
            timed = sorted((g for g in pool if g in run_s and source[g] == "schedule"),
                           key=lambda g: (edges[g].track != e.track, g))
            if timed:
                speed = (kind, v_max_for(length[timed[0]], run_s[timed[0]]))
                break
        if speed is None:
            speed = ("fallback", FALLBACK_SPEED_MS)
            warnings.append(f"edge {e.idx} {e.hop} ({e.track}): no timing source; {FALLBACK_SPEED_MS} m/s assumed")
        run_s[e.idx] = length[e.idx] / speed[1] + ACCEL_DECEL_LOSS_S
        count[e.idx] = 0
        source[e.idx] = speed[0]
    vmax = {ei: v_max_for(length[ei], run_s[ei]) for ei in run_s}
    return EdgeTiming(length, run_s, vmax, count, source, warnings)
