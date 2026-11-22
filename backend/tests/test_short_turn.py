"""SHORT_TURN_RELAY (ARCHITECTURE §4.2 #5, §8; PRD §4.1 "3–5 simulated minutes").

The toy's only pocket turnback is at A04 (Union Junction). Closing A05 blocks every
southbound path between A04 and the terminal, so southbound local trains at A04 may relay.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sim import NYCSubwaySim, Scenario
from sim import constants as C
from sim.graph import Graph
from sim.testing import TOY_PATTERN, TOY_STATION, first_decision, make_toy_graph, pattern_node, run_until


@pytest.fixture()
def closed_a05(toy_graph: Graph) -> NYCSubwaySim:
    sim = NYCSubwaySim(toy_graph, Scenario(seed=4))
    sim.add_disruption("full_station_closure", "station", TOY_STATION["A05"])
    return sim


@pytest.mark.parametrize("seed", [4, 9, 23])
def test_relay_latency_discharge_and_cost(seed: int) -> None:
    g = make_toy_graph()
    sim = NYCSubwaySim(g, Scenario(seed=seed))
    sim.add_disruption("full_station_closure", "station", TOY_STATION["A05"])
    t, _ = first_decision(sim, C.A_SHORT_TURN, station=TOY_STATION["A04"])
    a04_south_pl = int(g.node_platform[pattern_node(g, "1S", "A04")])
    riders = float(sim.tr_onboard[t].sum())
    assert riders > 1.0
    waiting_before = float(sim.waiting[a04_south_pl].sum())
    cost_before = sim.cost_totals()
    t_decision = sim.clock_s

    (ev,) = sim.apply_actions(np.array([t]), np.array([C.A_SHORT_TURN]))
    assert ev.applied and ev.action == C.A_SHORT_TURN and ev.train_id == t
    assert ev.station_idx == TOY_STATION["A04"]
    assert ev.detail["riders_forced_off"] == pytest.approx(riders)
    assert ev.detail["blocked_ahead"] is not None
    latency = ev.detail["turnback_latency_s"]
    assert C.TURNBACK_MIN_S <= latency <= C.TURNBACK_MAX_S
    # every onboard rider is discharged to the platform (they wait for the next train)
    assert sim.tr_onboard[t].sum() == 0.0
    assert float(sim.waiting[a04_south_pl].sum()) == pytest.approx(waiting_before + riders)
    after = sim.cost_totals()
    assert after["short_turn"] - cost_before["short_turn"] == pytest.approx(C.W_ST * riders)
    assert after["total"] - cost_before["total"] == pytest.approx(C.W_ST * riders)
    assert sim.action_counts()["SHORT_TURN_RELAY"] == 1

    # the train relays through the pocket (status bit, never tunnel-stuck) ...
    reverse_node = pattern_node(g, "1N", "A04")

    def back_in_service() -> bool:
        info = sim.train_info(t)
        return info["mode"] == "dwell" and info["node"] == reverse_node

    seen_short_turning = False
    for _ in range(40):
        if back_in_service():
            break
        sim.step(1)
        f = sim.frame()
        row = np.flatnonzero(f["train_id"] == t)
        status = int(f["status"][row[0]])
        seen_short_turning |= bool(status & C.ST_SHORT_TURNING)
        assert not status & C.ST_TUNNEL_STUCK
    assert back_in_service()
    assert seen_short_turning
    elapsed = sim.clock_s - t_decision
    # ... and re-enters service in the reverse direction 180-300 s after the decision
    assert C.TURNBACK_MIN_S <= elapsed <= C.TURNBACK_MAX_S
    assert elapsed == pytest.approx(math.ceil(latency / C.DT_S - 1e-9) * C.DT_S)
    assert int(sim.tr_pattern[t]) == TOY_PATTERN["1N"]
    # it now runs northbound: its next station is A03
    k = int(sim.tr_ptr[t]) + (0 if sim.tr_at_start[t] else 1)
    assert int(g.edge_to_station[int(sim.p_edge[t, k])]) == TOY_STATION["A03"]


def test_masked_without_need(toy_graph: Graph) -> None:
    """Calm service: a southbound train ready at A04 (pocket, not a terminal) has no need."""
    sim = NYCSubwaySim(toy_graph, Scenario(seed=4))
    node = pattern_node(toy_graph, "1S", "A04")

    def ready_at_a04() -> bool:
        return any(sim.train_info(t)["mode"] == "ready" and sim.train_info(t)["node"] == node
                   for t in np.flatnonzero(sim.tr_active))

    assert run_until(sim, ready_at_a04, 120)
    t = next(t for t in np.flatnonzero(sim.tr_active)
             if sim.train_info(t)["mode"] == "ready" and sim.train_info(t)["node"] == node)
    assert toy_graph.node_has_pocket[node]
    assert not sim.masks_for([t])[0][C.A_SHORT_TURN]


def test_masked_without_a_pocket(closed_a05: NYCSubwaySim) -> None:
    """A03 has no pocket track: never valid there even with the path blocked."""
    sim = closed_a05
    for _ in range(150):
        b = sim.pending_decisions()
        for j, ctx in enumerate(b.context):
            if ctx["station_idx"] != TOY_STATION["A04"]:
                assert not b.masks[j, C.A_SHORT_TURN], ctx
        sim.step(1)


def test_masked_at_terminals(toy_graph: Graph) -> None:
    """A train at its pattern's origin or destination terminal never relays."""
    sim = NYCSubwaySim(toy_graph, Scenario(seed=3))
    sim.add_disruption("full_station_closure", "station", TOY_STATION["A05"])
    terminals = {TOY_STATION[s] for s in ("A01", "A08", "B01", "B05")}
    checked = 0
    for _ in range(150):
        ready = [t for t in np.flatnonzero(sim.tr_active) if sim.train_info(t)["mode"] == "ready"]
        for t in ready:
            if sim.train_info(t)["station"] in terminals:
                assert not sim.masks_for([t])[0][C.A_SHORT_TURN]
                checked += 1
        sim.step(1)
    assert checked > 0


def test_pocket_is_exclusive(closed_a05: NYCSubwaySim) -> None:
    """While one train occupies the pocket, no other train may relay there."""
    sim = closed_a05
    t, _ = first_decision(sim, C.A_SHORT_TURN, station=TOY_STATION["A04"])
    (ev,) = sim.apply_actions(np.array([t]), np.array([C.A_SHORT_TURN]))
    assert ev.applied
    latency = ev.detail["turnback_latency_s"]
    for _ in range(int(latency // C.DT_S) - 1):
        sim.step(1)
        b = sim.pending_decisions()
        for j, ctx in enumerate(b.context):
            if ctx["station_idx"] == TOY_STATION["A04"]:
                assert not b.masks[j, C.A_SHORT_TURN]


def test_invalid_request_is_replaced_by_continue(closed_a05: NYCSubwaySim) -> None:
    """Re-validation: an action that is not (or no longer) valid is not applied."""
    sim = closed_a05
    t, j = first_decision(sim, C.A_SHORT_TURN, station=TOY_STATION["A04"])
    b = sim.pending_decisions()
    invalid = [a for a in range(1, C.N_ACTIONS) if not b.masks[j, a]]
    assert invalid
    before = sim.state_hash()
    cost = sim.cost_totals()["total"]
    (ev,) = sim.apply_actions(np.array([t]), np.array([invalid[0]]))
    assert not ev.applied and ev.action == invalid[0]
    assert sim.state_hash() == before
    assert sim.cost_totals()["total"] == cost
    assert sum(sim.action_counts().values()) == 0
