"""A/B division isolation (ARCHITECTURE §2.3 #7, §8; PRD §4.1 "Division Separation").

Static: no edge, detour, reroute, short-turn continuation or yard lead joins the divisions.
Dynamic: three simulated hours with random disruptions and random *valid* actions never
put a train on an edge of the other division.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim import NYCSubwaySim
from sim import constants as C
from sim.disruptions import valid_targets
from sim.graph import Graph
from sim.scenarios import sample_training_scenario

THREE_HOURS_TICKS = int(3 * 3600 / C.DT_S)


def _edge_div(g: Graph) -> np.ndarray:
    return g.node_division[g.edge_from]


def test_no_edge_crosses_divisions(any_graph: Graph) -> None:
    g = any_graph
    usable = np.zeros(g.n_edges, dtype=bool)
    usable[g.node_out_idx] = True  # cross-division edges are never wired into adjacency
    assert np.all(g.node_division[g.edge_from[usable]] == g.node_division[g.edge_to[usable]])


def test_patterns_and_alternates_stay_in_their_division(any_graph: Graph) -> None:
    g = any_graph
    ediv = _edge_div(g)
    for p in range(g.n_patterns):
        d = int(g.route_division[g.pat_route[p]])
        assert np.all(ediv[g.pattern_edges(p)] == d), f"pattern {p}"
    for table in (g.express_detours, g.local_detours, g.reroute_detours):
        for flat, dets in table.items():
            p = int(np.searchsorted(g.pat_edge_ptr, flat, side="right") - 1)
            d = int(g.route_division[g.pat_route[p]])
            for det in dets:
                assert np.all(ediv[list(det.edges)] == d), (p, det)
    for flat, opt in g.short_turn_options.items():
        p = int(np.searchsorted(g.pat_edge_ptr, flat, side="right") - 1)
        d = int(g.route_division[g.pat_route[p]])
        assert int(ediv[opt.turnback_edge]) == d
        assert int(g.route_division[g.pat_route[opt.pattern]]) == d
    for lead in g.yard_leads:
        d = int(g.yard_division[lead.yard])
        assert int(g.route_division[g.pat_route[lead.pattern]]) == d
        assert int(g.node_division[lead.node]) == d
    for y in range(g.n_yards):
        d = int(g.yard_division[y])
        assert np.all(g.route_division[g.yard_routes(y)] == d)


def _check_trains(sim: NYCSubwaySim) -> None:
    g = sim.g
    act = np.flatnonzero(sim.tr_active)
    if act.size == 0:
        return
    ediv = np.append(_edge_div(g), -1)
    cur = sim.p_edge[act, sim.tr_ptr[act]].astype(np.intp)
    assert np.all(ediv[cur] == sim.tr_div[act]), "a train is on an edge of the other division"
    # the whole remaining path too
    rows = sim.p_edge[act].astype(np.intp)
    valid = np.arange(rows.shape[1])[None, :] < sim.tr_plen[act][:, None]
    assert np.all(np.where(valid, ediv[rows] == sim.tr_div[act][:, None], True))
    assert np.all(g.route_division[sim.tr_route[act]] == sim.tr_div[act])


def test_random_disruptions_and_actions_never_cross_divisions(any_graph: Graph) -> None:
    g = any_graph
    rng = np.random.Generator(np.random.PCG64(2026))
    sim = NYCSubwaySim(g, sample_training_scenario(rng, 2, g))
    catalogue = [(k, tt, i) for k in C.DISRUPTION_KINDS for tt, i in valid_targets(g, k)]
    applied = np.zeros(C.N_ACTIONS, dtype=np.int64)
    for tick in range(THREE_HOURS_TICKS):
        if tick % 60 == 0:  # every 10 sim-minutes: add a random disruption, sometimes clear one
            k, tt, i = catalogue[int(rng.integers(len(catalogue)))]
            sim.add_disruption(k, tt, i, float(rng.uniform(600, 2400)))
            if sim.disruptions() and rng.random() < 0.3:
                sim.clear_disruption(sim.disruptions()[0]["id"])
        b = sim.pending_decisions()
        if len(b):
            acts = np.array([rng.choice(np.flatnonzero(m)) for m in b.masks], dtype=np.int64)
            for ev in sim.apply_actions(b.train_ids, acts):
                if ev.applied:
                    applied[ev.action] += 1
        sim.step(1)
        _check_trains(sim)
    assert applied[1:].sum() > 0, "random valid actions must actually be exercised"


@pytest.mark.parametrize("seed", [0, 1])
def test_reroutes_and_gap_trains_keep_their_division(toy_graph: Graph, seed: int) -> None:
    """Exercise the structural actions specifically (always pick the first structural one)."""
    from sim import Scenario
    from sim.testing import TOY_STATION, toy_segment

    g = toy_graph
    sim = NYCSubwaySim(g, Scenario(seed=seed))
    sim.add_disruption("local_block", "segment", toy_segment(g, "A06", "A08"))
    sim.add_disruption("local_block", "segment", toy_segment(g, "B03", "B04"))
    sim.add_disruption("local_block", "station", TOY_STATION["A03"])
    structural = [C.A_REROUTE, C.A_YARD, C.A_EXPRESS, C.A_LOCAL, C.A_SHORT_TURN]
    done = set()
    for _ in range(600):
        b = sim.pending_decisions()
        acts = np.zeros(len(b), dtype=np.int64)
        for j in range(len(b)):
            for a in structural:
                if b.masks[j, a]:
                    acts[j] = a
                    break
        for ev in sim.apply_actions(b.train_ids, acts):
            if ev.applied:
                done.add(ev.action)
        sim.step(1)
        _check_trains(sim)
    assert {C.A_REROUTE, C.A_YARD, C.A_EXPRESS} <= done
