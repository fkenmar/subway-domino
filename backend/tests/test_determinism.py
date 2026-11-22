"""Determinism (ARCHITECTURE §3.6): same scenario + same actions at the same ticks ⇒ same state."""

from __future__ import annotations

import numpy as np

from sim import NYCSubwaySim, Scenario, ScheduledDisruption
from sim import constants as C
from sim.graph import Graph
from sim.testing import TOY_STATION, toy_segment


def _scenario(g: Graph, seed: int = 7) -> Scenario:
    return Scenario(
        seed=seed,
        profile="am_peak",
        demand_scale=1.1,
        headway_jitter=0.2,
        dwell_scale=1.05,
        disruptions=(
            ScheduledDisruption("local_block", "segment", toy_segment(g, "B03", "B04"), 60.0, 900.0),
            ScheduledDisruption("full_station_closure", "station", TOY_STATION["A05"], 300.0, 1200.0),
        ),
    )


def _rollout(g: Graph, scenario: Scenario, ticks: int, action_seed: int) -> tuple[NYCSubwaySim, list[str]]:
    """Run with seeded random *valid* actions; returns the sim and the per-tick hashes."""
    sim = NYCSubwaySim(g, scenario)
    rng = np.random.Generator(np.random.PCG64(action_seed))
    hashes = [sim.state_hash()]
    for _ in range(ticks):
        b = sim.pending_decisions()
        if len(b):
            acts = np.array([rng.choice(np.flatnonzero(m)) for m in b.masks], dtype=np.int64)
            sim.apply_actions(b.train_ids, acts)
        sim.step(1)
        hashes.append(sim.state_hash())
    return sim, hashes


def test_same_scenario_and_actions_give_identical_state(toy_graph: Graph) -> None:
    sc = _scenario(toy_graph)
    a, ha = _rollout(toy_graph, sc, 240, action_seed=3)
    b, hb = _rollout(toy_graph, sc, 240, action_seed=3)
    assert ha == hb
    assert a.cost_totals() == b.cost_totals()
    assert a.action_counts() == b.action_counts()
    assert sum(a.action_counts().values()) > 0, "the rollout should exercise some actions"


def test_different_actions_or_seeds_diverge(toy_graph: Graph) -> None:
    sc = _scenario(toy_graph)
    _, h1 = _rollout(toy_graph, sc, 240, action_seed=3)
    _, h2 = _rollout(toy_graph, sc, 240, action_seed=4)
    assert h1[-1] != h2[-1]
    sim_a = NYCSubwaySim(toy_graph, _scenario(toy_graph, seed=1))
    sim_b = NYCSubwaySim(toy_graph, _scenario(toy_graph, seed=2))
    assert sim_a.state_hash() != sim_b.state_hash()


def test_reset_reproduces_the_initial_state(toy_graph: Graph) -> None:
    sc = _scenario(toy_graph)
    sim = NYCSubwaySim(toy_graph, sc)
    h0 = sim.state_hash()
    sim.step(50)
    sim.reset(sc)
    assert sim.state_hash() == h0
    assert sim.tick == 0
    assert sim.cost_totals()["total"] == 0.0


def test_copy_is_independent_and_evolves_identically(toy_graph: Graph) -> None:
    sim = NYCSubwaySim(toy_graph, _scenario(toy_graph))
    sim.step(30)
    twin = sim.copy()
    assert twin.state_hash() == sim.state_hash()
    assert twin.g is sim.g  # the read-only graph is shared
    sim.step(40)
    assert twin.tick == 30
    twin.step(40)
    assert twin.state_hash() == sim.state_hash()
    assert twin.cost_totals() == sim.cost_totals()
    # mutating one never touches the other
    twin.add_disruption("express_block", "segment", toy_segment(toy_graph, "A02", "A03"))
    assert len(twin.disruptions()) == len(sim.disruptions()) + 1


def test_disruption_ids_increase_monotonically(toy_graph: Graph) -> None:
    sim = NYCSubwaySim(toy_graph, Scenario(seed=0))
    ids = [sim.add_disruption("signal_delay", "segment", s) for s in range(3)]
    assert ids == sorted(ids) and len(set(ids)) == 3
    assert sim.clear_disruption(ids[1])
    assert not sim.clear_disruption(ids[1])
    nxt = sim.add_disruption("local_block", "segment", 0)
    assert nxt > ids[-1]


def test_no_global_rng_is_used(toy_graph: Graph) -> None:
    """The simulator never touches NumPy's legacy global RNG state."""
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    sim = NYCSubwaySim(toy_graph, _scenario(toy_graph))
    for _ in range(60):
        b = sim.pending_decisions()
        sim.apply_actions(b.train_ids, np.full(len(b), C.A_CONTINUE))
        sim.step(1)
    assert np.array_equal(np.random.get_state()[1], before)


# --------------------------------------------------------------------------------------
# Real network: determinism and passenger conservation under random valid actions
# --------------------------------------------------------------------------------------
def _rider_stock(sim: NYCSubwaySim) -> float:
    """Riders in the system plus those who left it (served or displaced by closures)."""
    return float(sim.waiting.sum() + sim.tr_onboard[sim.tr_active].sum() + sim.riders_served + sim.riders_displaced)


def test_real_network_is_deterministic_and_conserves_riders(real_graph: Graph) -> None:
    from sim.scenarios import preset_scenario

    sc = preset_scenario("manhattan-bridge", real_graph)
    a, ha = _rollout(real_graph, sc, 150, action_seed=5)
    b, hb = _rollout(real_graph, sc, 150, action_seed=5)
    assert ha == hb
    assert a.cost_totals() == b.cost_totals()
    used = {k for k, v in a.action_counts().items() if v}
    assert len(used) >= 5, used  # the random policy exercises most action kinds
    # conservation: every arrival is waiting, onboard, served or displaced (never created or lost)
    sim = NYCSubwaySim(real_graph, sc)
    rng = np.random.Generator(np.random.PCG64(5))
    s0, arrivals = _rider_stock(sim), 0.0
    for _ in range(150):
        batch = sim.pending_decisions()
        if len(batch):
            sim.apply_actions(batch.train_ids, np.array([rng.choice(np.flatnonzero(m)) for m in batch.masks]))
        sim.step(1)
        arrivals += float(sim.rate_eff.sum()) * C.DT_S
        assert _rider_stock(sim) - s0 == np.float64(arrivals) or abs(_rider_stock(sim) - s0 - arrivals) < 1e-9 * arrivals
    assert float(sim.tr_onboard[~sim.tr_active].sum()) == 0.0
