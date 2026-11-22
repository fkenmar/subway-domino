"""DISPATCH_YARD_GAP_TRAIN (ARCHITECTURE §4.2 #7, §8; PRD §4.1 yard latencies).

The toy yard (B division, route L, fleet 4, deadhead 420 s) leads onto Transfer Plaza (B03)
in both directions. Blocking the river tunnel B03–B04 starves westbound service at B03.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim import NYCSubwaySim, Scenario
from sim import constants as C
from sim.graph import Graph
from sim.testing import TOY_PATTERN, TOY_STATION, toy_segment


def _starved(g: Graph, seed: int = 4) -> NYCSubwaySim:
    sim = NYCSubwaySim(g, Scenario(seed=seed))
    sim.add_disruption("local_block", "segment", toy_segment(g, "B03", "B04"))
    return sim


def _yard_valid_rows(sim: NYCSubwaySim) -> list[tuple[int, int]]:
    b = sim.pending_decisions()
    return [(int(t), j) for j, t in enumerate(b.train_ids.tolist()) if b.masks[j, C.A_YARD]]


def _dispatch_next(sim: NYCSubwaySim, max_ticks: int = 120) -> tuple[float, object]:
    for _ in range(max_ticks):
        rows = _yard_valid_rows(sim)
        if rows:
            t, _ = rows[0]
            (ev,) = sim.apply_actions(np.array([t]), np.array([C.A_YARD]))
            assert ev.applied
            return sim.clock_s, ev
        sim.step(1)
    raise AssertionError("DISPATCH_YARD_GAP_TRAIN never became valid")


def test_gap_train_enters_after_exactly_deadhead(toy_graph: Graph) -> None:
    g = toy_graph
    sim = _starved(g)
    y = 0
    stock0 = int(sim.yard_stock[y])
    cost0 = sim.cost_totals()
    t_disp, ev = _dispatch_next(sim)
    assert ev.detail["yard_idx"] == y
    assert ev.detail["lead_station_idx"] == TOY_STATION["B03"]
    assert ev.detail["deadhead_s"] == pytest.approx(float(g.yard_deadhead_s[y]))
    assert int(sim.yard_stock[y]) == stock0 - 1
    after = sim.cost_totals()
    assert after["yard"] - cost0["yard"] == pytest.approx(C.W_YARD)
    assert sim.action_counts()["DISPATCH_YARD_GAP_TRAIN"] == 1

    before = set(np.flatnonzero(sim.tr_active & sim.tr_gap).tolist())
    appeared: dict[int, tuple[float, dict, int]] = {}
    for _ in range(int(g.yard_deadhead_s[y] / C.DT_S) + 3):
        sim.step(1)
        for t in np.flatnonzero(sim.tr_active & sim.tr_gap).tolist():
            if t not in before and t not in appeared:
                f = sim.frame()
                status = int(f["status"][np.flatnonzero(f["train_id"] == t)[0]])
                appeared[t] = (sim.clock_s, sim.train_info(t), status)
    assert len(appeared) == 1
    (t_gap, (t_seen, info, status)), = appeared.items()
    assert abs((t_seen - t_disp) - float(g.yard_deadhead_s[y])) <= C.DT_S
    # it enters at the starved lead node and runs the westbound pattern from there
    lead_nodes = {ld.node for ld in g.yard_leads if ld.yard == y}
    assert info["node"] in lead_nodes
    assert info["station"] == TOY_STATION["B03"]
    assert info["gap_train"] and status & C.ST_GAP_TRAIN
    assert int(sim.tr_pattern[t_gap]) == TOY_PATTERN["LW"]


def test_cooldown_and_stock_are_enforced(toy_graph: Graph) -> None:
    g = toy_graph
    sim = _starved(g)
    y = 0
    cap = int(g.yard_fleet_cap[y])
    times = []
    for k in range(cap):
        t_disp, _ = _dispatch_next(sim, max_ticks=200)
        times.append(t_disp)
        assert int(sim.yard_stock[y]) == cap - k - 1
        # during the cooldown the action is masked for every train
        for _ in range(int(C.YARD_COOLDOWN_S / C.DT_S) - 1):
            sim.step(1)
            assert _yard_valid_rows(sim) == []
    assert np.all(np.diff(times) >= C.YARD_COOLDOWN_S - 1e-6)
    # stock exhausted: masked for the rest of the episode although service is still starved
    assert int(sim.yard_stock[y]) == 0
    for _ in range(120):
        sim.step(1)
        assert _yard_valid_rows(sim) == []


def test_masked_without_starvation(toy_graph: Graph) -> None:
    sim = NYCSubwaySim(toy_graph, Scenario(seed=4))
    ready_l = 0
    for _ in range(240):
        sim.step(1)
        assert _yard_valid_rows(sim) == []
        for t in np.flatnonzero(sim.tr_active):
            if toy_graph.route_id[int(sim.tr_route[t])] == "L" and sim.train_info(t)["mode"] == "ready":
                ready_l += 1
                assert not sim.masks_for([t])[0][C.A_YARD]
    assert ready_l > 0


def test_only_compatible_routes_can_dispatch(toy_graph: Graph) -> None:
    """The B-division yard (routes: L) is never offered to A-division trains."""
    sim = _starved(toy_graph)
    sim.add_disruption("full_station_closure", "station", TOY_STATION["A05"])
    l_route = toy_graph.route_index["L"]
    offered = 0
    for _ in range(240):
        for t, _ in _yard_valid_rows(sim):
            assert int(sim.tr_route[t]) == l_route
            offered += 1
        sim.step(1)
    assert offered > 0


# --------------------------------------------------------------------------------------
# Starvation ETA (masks.lead_etas): vectorized over every lead, equal to the scalar definition
# --------------------------------------------------------------------------------------
def _scalar_pair_eta(sim: NYCSubwaySim, q: int, j: int) -> float:
    """Reference: next train of pattern q to reach template edge j's from-node (s)."""
    g = sim.g
    tmpl = g.tmpl_sched[q].astype(np.float64)
    te = g.tmpl_edge[q, : int(g.pat_len[q])].astype(np.intp)
    flags = sim.e_blocked[te] | sim.n_closed[sim._S.eto[te]]
    cum = np.concatenate([[0], np.cumsum(flags)])
    t_lead = float(tmpl[j - 1]) if j >= 1 else 0.0
    best = np.inf
    for t in np.flatnonzero(sim.tr_active & (sim.tr_pattern == q) & ~sim.tr_deviated).tolist():
        tp = int(sim.p_tpos[t, sim.tr_ptr[t]])
        if tp < 0 or not (tp <= j if sim.tr_at_start[t] else tp <= j - 1):
            continue
        if cum[j] - cum[min(tp, j)] > 0:
            continue
        best = min(best, max(t_lead - (float(tmpl[tp - 1]) if tp >= 1 else 0.0), 0.0))
    if cum[j] - cum[0] == 0:
        for k in sim._P.pat_svc.get(q, []):
            best = min(best, max(float(sim.svc_next[k]) - sim.clock_s, 0.0) + t_lead)
    return best


def _check_lead_etas(sim: NYCSubwaySim) -> int:
    from sim.masks import lead_etas

    P = sim._P
    vec = lead_etas(sim)
    for a, lid in enumerate(P.lead_svc.tolist()):
        pairs = range(int(P.lead_pair_ptr[a]), int(P.lead_pair_ptr[a + 1]))
        ref = min(_scalar_pair_eta(sim, int(P.lead_pair_q[b]), int(P.lead_pair_j[b])) for b in pairs)
        assert vec[lid] == pytest.approx(ref, abs=1e-6), (lid, vec[lid], ref)
    return int(P.lead_svc.size)


def test_lead_etas_match_the_scalar_definition_on_the_toy(toy_graph: Graph) -> None:
    sim = _starved(toy_graph)
    n = 0
    for _ in range(12):
        sim.step(10)
        n += _check_lead_etas(sim)
    assert n > 0


@pytest.mark.parametrize("preset", [None, "lex-union-sq"])
def test_lead_etas_match_the_scalar_definition_on_the_real_network(real_graph: Graph, preset: str | None) -> None:
    from sim.scenarios import preset_scenario

    sc = preset_scenario(preset, real_graph) if preset else Scenario(seed=2, profile="am_peak")
    sim = NYCSubwaySim(real_graph, sc)
    for _ in range(6):
        sim.step(15)
        assert _check_lead_etas(sim) > 10


def test_served_trunk_is_not_starved_by_a_rare_variant_pattern(real_graph: Graph) -> None:
    """A lead's "next train" is any in-service train on the lead's track, not only its own
    (possibly hourly) variant pattern; calm am_peak service therefore rarely offers a yard train."""
    from sim.masks import lead_etas

    sim = NYCSubwaySim(real_graph, Scenario(seed=0, profile="am_peak"))
    P = sim._P
    own_only = []
    for a, lid in enumerate(P.lead_svc.tolist()):
        lead = real_graph.yard_leads[lid]
        if int(P.lead_pair_ptr[a + 1] - P.lead_pair_ptr[a]) > 1 and P.pat_h[lead.pattern] > 3 * P.lead_thr[lid]:
            own_only.append((lid, _scalar_pair_eta(sim, int(lead.pattern), int(lead.position))))
    assert own_only, "expected rare variant patterns sharing a served trunk on the real network"
    etas = lead_etas(sim)
    assert any(eta > P.lead_thr[lid] for lid, eta in own_only)  # the old, per-pattern reading
    assert all(etas[lid] <= P.lead_thr[lid] for lid, _ in own_only)
    valid = 0
    for _ in range(180):
        sim.step(1)
        valid += int(sim._pend_masks[:, C.A_YARD].sum())
    assert valid < 200, valid  # was ~2,500 per 30 sim-min before
