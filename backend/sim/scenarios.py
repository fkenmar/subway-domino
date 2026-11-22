"""Scenarios: what a simulation episode runs (ARCHITECTURE §3.1, §4.6, §4.7).

``Scenario`` and ``ScheduledDisruption`` are the CONTRACT dataclasses. The samplers build
them deterministically from a NumPy generator or a seed:

* :func:`sample_training_scenario` — Stage 1 (one trunk, 1-2 disruptions on it, headway
  jitter) and Stage 2 (citywide, 1-4 disruptions of any kind, domain randomization);
* :func:`eval_scenario` — the fixed meltdown seeds used by ``rl/evaluate.py``
  (``EVAL_SEED_BASE + i``; training never draws seeds that large);
* :func:`preset_scenario` — the curated presets from ``network.json``.

Disruption targets are drawn among valid targets that actually carry service in the
scenario's profile, weighted by how many trains per hour they affect, so episodes are
meltdowns rather than no-ops on an unused crossover.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim import constants as C
from sim import disruptions as D
from sim.graph import Graph


@dataclass(frozen=True)
class ScheduledDisruption:
    kind: str
    target_type: str
    target_idx: int
    start_s: float  # relative to the episode start (after warm-up)
    duration_s: float | None


@dataclass(frozen=True)
class Scenario:
    seed: int = 0
    profile: str = "am_peak"
    demand_scale: float = 1.0
    routes: tuple[str, ...] | None = None  # subset of route ids to operate (Stage 1); None = all
    disruptions: tuple[ScheduledDisruption, ...] = ()
    episode_s: float = 5400.0
    dwell_scale: float = 1.0  # domain randomization
    speed_scale: float = 1.0
    # Backwards-compatible superset of the §3.1 contract: initial-position and dispatch
    # jitter as a fraction of the scheduled headway (Stage-1 "random headway jitter").
    headway_jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.profile not in C.PROFILES:
            raise ValueError(f"unknown profile {self.profile!r}")
        if not (self.demand_scale >= 0.0):
            raise ValueError("demand_scale must be >= 0")
        if not (self.dwell_scale > 0.0 and self.speed_scale > 0.0):
            raise ValueError("dwell_scale and speed_scale must be > 0")
        if not (0.0 <= self.headway_jitter <= 1.0):
            raise ValueError("headway_jitter must be in [0, 1]")


# --------------------------------------------------------------------------------------
# Target catalogue
# --------------------------------------------------------------------------------------
def _route_set(g: Graph, routes: tuple[str, ...] | None) -> set[int]:
    if routes is None:
        return set(range(g.n_routes))
    return {g.route_index[r] for r in routes if r in g.route_index}


def _target_catalogue(g: Graph, profile: str, routes: tuple[str, ...] | None) -> list[tuple[str, str, int, float]]:
    """``(kind, target_type, idx, weight)`` for targets that affect operated service."""
    key = ("target_catalogue", profile, routes)
    hit = g._cache.get(key)
    if hit is not None:
        return hit
    view = g.service_view(profile, g.route_ids_to_idx(routes))
    trains_per_s = np.zeros(g.n_edges)
    finite = np.isfinite(view.edge_sched_h)
    trains_per_s[finite] = 1.0 / view.edge_sched_h[finite]
    st_rate = np.zeros(g.n_stations)
    for n in range(g.n_nodes):
        s = int(g.node_station[n])
        if s >= 0:
            outs = g.node_out(n)
            if len(outs):
                st_rate[s] += float(trains_per_s[outs].sum())
    out: list[tuple[str, str, int, float]] = []
    for kind in C.DISRUPTION_KINDS:
        for tt, idx in D.valid_targets(g, kind):
            eff = D.resolve(g, kind, tt, idx)
            if kind == "full_station_closure":
                w = float(st_rate[eff.closed_stations].sum())
            else:
                edges = eff.blocked_edges if len(eff.blocked_edges) else eff.delayed_edges
                w = float(trains_per_s[edges].sum())
            if w > 0.0:
                out.append((kind, tt, idx, w))
    g._cache[key] = out
    return out


def _trunk_stations(g: Graph, route_idx: set[int]) -> set[int]:
    out: set[int] = set()
    for s in range(g.n_stations):
        if route_idx & {int(r) for r in g.station_routes(s)}:
            out.add(s)
    return out


def _draw_disruptions(
    rng: np.random.Generator,
    g: Graph,
    profile: str,
    routes: tuple[str, ...] | None,
    n: int,
    episode_s: float,
    start_range: tuple[float, float],
    dur_range: tuple[float, float],
    p_until_end: float,
) -> tuple[ScheduledDisruption, ...]:
    cat = _target_catalogue(g, profile, routes)
    if routes is not None:
        sts = _trunk_stations(g, _route_set(g, routes))
        cat = [c for c in cat if set(D.target_stations(g, c[1], c[2])) <= sts]
    if not cat:
        return ()
    kinds = sorted({c[0] for c in cat})
    chosen: list[ScheduledDisruption] = []
    used: set[tuple[str, int]] = set()
    for _ in range(n):
        kind = kinds[int(rng.integers(len(kinds)))]
        pool = [c for c in cat if c[0] == kind and (c[1], c[2]) not in used]
        if not pool:
            continue
        w = np.sqrt(np.array([c[3] for c in pool]))
        j = int(rng.choice(len(pool), p=w / w.sum()))
        _, tt, idx, _ = pool[j]
        used.add((tt, idx))
        start = float(rng.uniform(*start_range))
        if rng.random() < p_until_end:
            dur: float | None = None
        else:
            dur = float(rng.uniform(*dur_range))
        start = min(start, max(episode_s - 600.0, 0.0))
        chosen.append(ScheduledDisruption(kind, tt, int(idx), round(start, 1), None if dur is None else round(dur, 1)))
    return tuple(sorted(chosen, key=lambda d: (d.start_s, d.kind, d.target_type, d.target_idx)))


def _trunk_groups_available(g: Graph) -> list[tuple[str, tuple[str, ...]]]:
    out = []
    for name, rids in C.TRUNK_GROUPS.items():
        have = tuple(r for r in rids if r in g.route_index)
        if have:
            out.append((name, have))
    if not out:  # e.g. the toy graph: every division is a "trunk"
        by_div: dict[int, list[str]] = {}
        for r in range(g.n_routes):
            by_div.setdefault(int(g.route_division[r]), []).append(g.route_id[r])
        out = [(f"division-{d}", tuple(v)) for d, v in sorted(by_div.items())]
    return out


# --------------------------------------------------------------------------------------
# Samplers
# --------------------------------------------------------------------------------------
def sample_training_scenario(rng: np.random.Generator, stage: int, graph: Graph, *, calm_frac: float = 0.0) -> Scenario:
    """Draw a training scenario for curriculum ``stage`` (1 or 2), see ARCHITECTURE §4.6.

    ``calm_frac`` (Stage 2 only): the probability of a *calm anchor* episode with no
    disruptions; headway jitter and the domain randomization are drawn as usual. With the
    default 0 no extra random number is drawn, so the draws are exactly the pre-WP5b ones.
    """
    if not 0.0 <= float(calm_frac) <= 1.0:
        raise ValueError(f"calm_frac must be in [0, 1], got {calm_frac!r}")
    seed = int(rng.integers(0, C.TRAINING_SEED_MAX))
    if stage == 1:
        groups = _trunk_groups_available(graph)
        _, routes = groups[int(rng.integers(len(groups)))]
        profile = C.PROFILES[int(rng.integers(len(C.PROFILES)))]
        episode_s = C.STAGE1_EPISODE_S
        n_dis = int(rng.integers(1, 3))
        dis = _draw_disruptions(rng, graph, profile, routes, n_dis, episode_s,
                                C.DISRUPTION_START_RANGE_S, C.DISRUPTION_DURATION_RANGE_S, 0.15)
        jitter = float(rng.uniform(*C.STAGE1_JITTER_RANGE))
        return Scenario(seed=seed, profile=profile, routes=routes, disruptions=dis, episode_s=episode_s,
                        headway_jitter=round(jitter, 3))
    if stage == 2:
        profile = C.PROFILES[int(rng.integers(len(C.PROFILES)))]
        episode_s = C.STAGE2_EPISODE_S
        if calm_frac > 0.0 and rng.random() < calm_frac:
            dis: tuple[ScheduledDisruption, ...] = ()  # calm anchor: nothing to fix, jitter still on
        else:
            n_dis = int(rng.integers(1, 5))
            dis = _draw_disruptions(rng, graph, profile, None, n_dis, episode_s,
                                    C.DISRUPTION_START_RANGE_S, C.DISRUPTION_DURATION_RANGE_S, 0.15)
        return Scenario(
            seed=seed,
            profile=profile,
            demand_scale=round(float(rng.uniform(*C.DR_DEMAND_SCALE)), 4),
            disruptions=dis,
            episode_s=episode_s,
            dwell_scale=round(float(rng.uniform(*C.DR_DWELL_SCALE)), 4),
            speed_scale=round(float(rng.uniform(*C.DR_SPEED_SCALE)), 4),
            headway_jitter=round(float(rng.uniform(*C.STAGE2_JITTER_RANGE)), 3),
        )
    raise ValueError(f"unknown curriculum stage {stage!r}")


def eval_scenario(seed: int, graph: Graph) -> Scenario:
    """Deterministic citywide meltdown for evaluation seed ``seed`` (``EVAL_SEED_BASE + i``)."""
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    profile = C.PROFILES[int(seed) % len(C.PROFILES)]
    n_dis = int(rng.integers(1, 4))
    dis = _draw_disruptions(rng, graph, profile, None, n_dis, C.STAGE2_EPISODE_S,
                            C.EVAL_DISRUPTION_START_RANGE_S, C.EVAL_DISRUPTION_DURATION_RANGE_S, 0.1)
    return Scenario(
        seed=int(seed),
        profile=profile,
        demand_scale=round(float(rng.uniform(*C.EVAL_DEMAND_SCALE)), 4),
        disruptions=dis,
        episode_s=C.STAGE2_EPISODE_S,
        headway_jitter=round(float(rng.uniform(*C.EVAL_JITTER_RANGE)), 3),
    )


def preset_scenario(preset_id: str, graph: Graph) -> Scenario:
    """The curated preset ``preset_id`` from ``network.json`` (``KeyError`` if unknown)."""
    if preset_id not in graph.preset_index:
        raise KeyError(f"unknown preset {preset_id!r}")
    p: dict[str, Any] = graph.presets[graph.preset_index[preset_id]]
    dis = []
    for d in p.get("disruptions", []):
        tgt = d.get("target", {})
        dur = d.get("duration_s")
        dis.append(ScheduledDisruption(str(d["kind"]), str(tgt["type"]), int(tgt["idx"]), 0.0,
                                       None if dur is None else float(dur)))
    seed = zlib.crc32(preset_id.encode("utf-8")) & 0x7FFFFFFF
    return Scenario(seed=int(seed), profile=str(p.get("profile", "am_peak")), disruptions=tuple(dis))
