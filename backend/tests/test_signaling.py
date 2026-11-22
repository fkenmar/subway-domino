"""Fixed-block vs CBTC signaling (ARCHITECTURE §3.3, §8; PRD Epic 2).

The same toy network is built twice, with every B-division run edge fixed-block or CBTC.
The B04–B05 segment is blocked and eastbound L trains queue behind it:

* fixed-block queues: front-to-front spacing >= train length + 0.5 * BLOCK_M, and larger
  than the CBTC spacing by >= 50 m;
* CBTC queues close up to the moving-block margin behind the leader's rear;
* after the block clears, the CBTC queue discharges faster.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim import NYCSubwaySim, Scenario
from sim import constants as C
from sim.graph import DIV_B, Graph
from sim.testing import TOY_PATTERN, make_toy_variant, toy_segment

QUEUE_MIN = 25  # minutes of blockage before measuring


def _eastbound_positions(sim: NYCSubwaySim) -> dict[int, float]:
    """Front position (m from the origin) of active trains on the eastbound L pattern's edges."""
    g = sim.g
    edges = g.pattern_edges(TOY_PATTERN["LE"])
    start = dict(zip(edges.tolist(), (np.cumsum(g.edge_length[edges]) - g.edge_length[edges]).tolist()))
    out: dict[int, float] = {}
    for t in np.flatnonzero(sim.tr_active).tolist():
        e = int(sim.p_edge[t, sim.tr_ptr[t]])
        if e in start and not g.edge_timed[e]:
            out[t] = start[e] + float(sim.tr_off[t])
    return out


def _queue(cbtc: bool) -> tuple[NYCSubwaySim, int, list[int], np.ndarray]:
    g = make_toy_variant(cbtc=cbtc)
    sim = NYCSubwaySim(g, Scenario(seed=0, profile="am_peak"))
    seg = toy_segment(g, "B04", "B05")
    did = sim.add_disruption("local_block", "segment", seg)
    sim.step(QUEUE_MIN * 6)
    pos = _eastbound_positions(sim)
    queued = sorted((t for t in pos if sim.tr_v[t] < C.STOPPED_V), key=lambda t: -pos[t])
    fronts = np.array([pos[t] for t in queued])
    return sim, did, queued, fronts


@pytest.fixture(scope="module")
def queues() -> dict[bool, tuple[NYCSubwaySim, int, list[int], np.ndarray]]:
    return {cbtc: _queue(cbtc) for cbtc in (False, True)}


def test_queues_form_behind_the_block(queues: dict) -> None:
    for cbtc, (sim, _, queued, _) in queues.items():
        assert len(queued) >= 4, f"cbtc={cbtc}: expected a queue, got {len(queued)} stopped trains"
        assert sim.metrics()["tunnel_stuck_trains"] >= 2


def test_fixed_block_spacing_exceeds_length_plus_half_block(queues: dict) -> None:
    _, _, _, fronts = queues[False]
    spacing = -np.diff(fronts)
    assert spacing.min() >= C.TRAIN_LENGTH_M["B"] + 0.5 * C.BLOCK_M - 1e-6, spacing


def test_cbtc_queue_is_tighter_by_at_least_50m(queues: dict) -> None:
    _, _, _, f_fixed = queues[False]
    _, _, _, f_cbtc = queues[True]
    s_fixed = -np.diff(f_fixed)
    s_cbtc = -np.diff(f_cbtc)
    # CBTC: the follower stops CBTC_MARGIN_M behind the leader's rear (or at its platform)
    assert s_cbtc.min() >= C.TRAIN_LENGTH_M["B"] + C.CBTC_MARGIN_M - 1e-6, s_cbtc
    assert s_fixed.min() >= s_cbtc.max() + 50.0, (s_fixed, s_cbtc)
    # a CBTC queue holds more trains in the same length of track
    span = 2000.0
    assert np.sum(f_cbtc >= f_cbtc[0] - span) > np.sum(f_fixed >= f_fixed[0] - span)


def _discharge_times(sim: NYCSubwaySim, did: int, queued: list[int], k: int) -> list[float]:
    """Seconds after clearing until each of the first ``k`` queued trains enters B04→B05."""
    g = sim.g
    exit_edge = int(g.pattern_edges(TOY_PATTERN["LE"])[-1])
    assert sim.clear_disruption(did)
    t0 = sim.clock_s
    passed: dict[int, float] = {}
    for _ in range(90):
        sim.step(1)
        for t in queued[:k]:
            if t not in passed and int(sim.p_edge[t, sim.tr_ptr[t]]) == exit_edge and sim.tr_active[t]:
                passed[t] = sim.clock_s - t0
        if len(passed) == k:
            break
    assert len(passed) == k, f"only {len(passed)} of {k} queued trains discharged"
    return [passed[t] for t in queued[:k]]


def test_cbtc_queue_discharges_faster(queues: dict) -> None:
    k = 3
    fixed = _discharge_times(*queues[False][:3], k=k)
    cbtc = _discharge_times(*queues[True][:3], k=k)
    assert cbtc[-1] < fixed[-1], (cbtc, fixed)
    # the head-of-queue train leaves at the same time in both (it waits at the red signal)
    assert abs(cbtc[0] - fixed[0]) <= C.DT_S


def test_no_train_ever_passes_a_red_signal() -> None:
    """While the block is active, no train is ever on a blocked edge."""
    g = make_toy_variant(cbtc=False)
    sim = NYCSubwaySim(g, Scenario(seed=2))
    seg = toy_segment(g, "B03", "B04")
    sim.add_disruption("local_block", "segment", seg)
    blocked = set(int(e) for e in g.segment_edges(seg))
    for _ in range(240):
        sim.step(1)
        act = np.flatnonzero(sim.tr_active)
        on = {int(sim.p_edge[t, sim.tr_ptr[t]]) for t in act if not sim.tr_at_start[t]}
        assert not (on & blocked)
    assert int(np.count_nonzero(sim.tr_div[np.flatnonzero(sim.tr_active)] == DIV_B)) > 0


def test_toy_has_both_signaling_kinds(toy_graph: Graph) -> None:
    run_b = (toy_graph.edge_kind == 0) & (toy_graph.node_division[toy_graph.edge_from] == DIV_B)
    assert toy_graph.edge_cbtc[run_b].any() and (~toy_graph.edge_cbtc[run_b]).any()


# --------------------------------------------------------------------------------------
# Real network, Normal Schedule, no disruptions (slow): interlocking liveness and calibration
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def calm_am_peak_2h(real_graph: Graph):
    from sim.calibrate import run_profile

    return run_profile(real_graph, "am_peak", seed=0, duration_s=7200.0)


@pytest.mark.slow
def test_calm_citywide_service_never_jams(calm_am_peak_2h) -> None:
    """Regression for the crossover-merge deadlock: 2 sim-hours of citywide am_peak, tunnel-stuck
    trains <= 3 on average and no train standing still for more than 10 min."""
    from sim.calibrate import MAX_STATIONARY_S, STUCK_MEAN_MAX

    r = calm_am_peak_2h
    assert r.stuck_mean <= STUCK_MEAN_MAX, r
    assert r.max_stationary_s <= MAX_STATIONARY_S, r


@pytest.mark.slow
@pytest.mark.parametrize("profile", C.PROFILES)
def test_calm_service_calibration_targets(real_graph: Graph, profile: str, calm_am_peak_2h) -> None:
    """sim.calibrate targets: trains within +-15 % of the §0 weekday counts, median dwell 25-45 s,
    am_peak wait 1.5-5 min, < 2 % platforms overcrowded, tunnel-stuck <= 3 on average."""
    from sim.calibrate import format_profile, run_profile

    r = calm_am_peak_2h if profile == "am_peak" else run_profile(real_graph, profile, seed=0, duration_s=3600.0)
    failed = [k for k, ok in r.checks().items() if not ok]
    assert not failed, f"{failed}\n{format_profile(r)}"
