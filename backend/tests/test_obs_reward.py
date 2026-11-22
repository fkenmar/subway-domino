"""Observations, costs/reward and the wire contract (ARCHITECTURE §3.8, §3.9, §4.3, §4.4)."""

from __future__ import annotations

import numpy as np
import pytest

from sim import FEATURE_NAMES, NYCSubwaySim, Scenario, ScheduledDisruption
from sim import constants as C
from sim import cost as K
from sim.graph import Graph
from sim.testing import TOY_STATION, toy_segment


def _meltdown(g: Graph, seed: int = 11) -> Scenario:
    if g.n_stations < 50:  # toy
        dis = (
            ScheduledDisruption("local_block", "segment", toy_segment(g, "B03", "B04"), 0.0, 1500.0),
            ScheduledDisruption("full_station_closure", "station", TOY_STATION["A05"], 120.0, 1200.0),
            ScheduledDisruption("signal_delay", "segment", toy_segment(g, "A02", "A03"), 0.0, None),
        )
        return Scenario(seed=seed, disruptions=dis, headway_jitter=0.15)
    from sim.scenarios import eval_scenario

    return eval_scenario(C.EVAL_SEED_BASE + seed, g)


def _collect_batches(g: Graph, ticks: int) -> list:
    sim = NYCSubwaySim(g, _meltdown(g))
    rng = np.random.Generator(np.random.PCG64(0))
    out = []
    for _ in range(ticks):
        b = sim.pending_decisions()
        if len(b):
            out.append(b)
            acts = np.array([rng.choice(np.flatnonzero(m)) for m in b.masks], dtype=np.int64)
            sim.apply_actions(b.train_ids, acts)
        sim.step(1)
    return out


# --------------------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------------------
def test_feature_names_cover_the_observation() -> None:
    assert C.OBS_DIM == 216 == len(FEATURE_NAMES)
    assert len(set(FEATURE_NAMES)) == C.OBS_DIM
    assert FEATURE_NAMES[-C.N_ACTIONS:] == tuple(f"mask_{a}" for a in C.ACTION_NAMES)


def test_obs_shape_dtype_range_and_mask_bits(any_graph: Graph) -> None:
    batches = _collect_batches(any_graph, 360)
    assert batches, "a meltdown must request decisions"
    n_total = 0
    for b in batches:
        n = len(b)
        n_total += n
        assert b.obs.shape == (n, C.OBS_DIM) and b.obs.dtype == np.float32
        assert b.masks.shape == (n, C.N_ACTIONS) and b.masks.dtype == bool
        assert b.train_ids.dtype == np.int32 and b.train_ids.shape == (n,)
        assert np.isfinite(b.obs).all()
        assert b.obs.min() >= -1.0 and b.obs.max() <= 1.0
        assert b.masks[:, 0].all(), "CONTINUE is always valid"
        assert b.masks[:, 1:].any(axis=1).all(), "every decision has a valid non-CONTINUE action"
        np.testing.assert_array_equal(b.obs[:, -C.N_ACTIONS:], b.masks.astype(np.float32))
        assert len(b.context) == n
        for ctx in b.context:
            assert set(ctx) == {"station_idx", "route_idx", "node_idx"}
            assert 0 <= ctx["station_idx"] < any_graph.n_stations
    assert n_total >= 5
    # the features actually vary (no dead observation)
    obs = np.concatenate([b.obs for b in batches])
    assert (obs.std(axis=0) > 0).sum() >= 40


def test_pending_decisions_is_one_cached_snapshot(toy_graph: Graph) -> None:
    sim = NYCSubwaySim(toy_graph, _meltdown(toy_graph))
    for _ in range(200):
        b1 = sim.pending_decisions()
        if len(b1):
            break
        sim.step(1)
    assert len(b1)
    b2 = sim.pending_decisions()
    assert b2 is b1
    sim.step(1)
    b3 = sim.pending_decisions()
    assert b3 is not b1


def test_unanswered_decisions_resolve_as_continue(toy_graph: Graph) -> None:
    """Not answering a batch is exactly the same as answering CONTINUE (Normal Schedule)."""
    a = NYCSubwaySim(toy_graph, _meltdown(toy_graph))
    b = NYCSubwaySim(toy_graph, _meltdown(toy_graph))
    for _ in range(240):
        batch = a.pending_decisions()
        ev = a.apply_actions(batch.train_ids, np.zeros(len(batch), dtype=np.int64))
        assert ev == []  # events are only returned for non-CONTINUE actions
        a.step(1)
        b.step(1)
    assert a.state_hash() == b.state_hash()


# --------------------------------------------------------------------------------------
# Counterfactual twin
# --------------------------------------------------------------------------------------
def test_twin_difference_is_exactly_zero_under_all_continue(any_graph: Graph) -> None:
    ai = NYCSubwaySim(any_graph, _meltdown(any_graph))
    twin = ai.copy()
    extra = None
    for t in range(300):
        if t == 100:  # a user disruption mirrored to the twin (server semantics)
            seg = int(np.flatnonzero(any_graph.segment_has_express)[0]) if any_graph.segment_has_express.any() else 0
            extra = ai.add_disruption("signal_delay", "segment", seg, 600.0)
            twin.add_disruption("signal_delay", "segment", seg, 600.0)
        before_ai, before_tw = ai.cost_totals()["total"], twin.cost_totals()["total"]
        b = ai.pending_decisions()
        ai.apply_actions(b.train_ids, np.zeros(len(b), dtype=np.int64))
        ai.step(1)
        twin.step(1)
        d_ai = ai.cost_totals()["total"] - before_ai
        d_tw = twin.cost_totals()["total"] - before_tw
        assert d_ai - d_tw == 0.0
    assert extra is not None
    assert ai.state_hash() == twin.state_hash()
    assert ai.cost_totals() == twin.cost_totals()
    assert ai.cost_totals()["total"] > 0


# --------------------------------------------------------------------------------------
# Cost terms (pure functions) are monotone in their drivers
# --------------------------------------------------------------------------------------
def test_wait_cost_grows_with_riders_and_age() -> None:
    w = np.zeros((3, C.N_AGE_BINS))
    w[0, 2] = 10.0
    base = K.wait_cost(w, 1.0)
    more = w.copy()
    more[0, 2] = 20.0
    older = w.copy()
    older[0, 2], older[0, 10] = 0.0, 10.0
    assert 0 < base < K.wait_cost(more, 1.0)
    assert base < K.wait_cost(older, 1.0)
    # the age weight is superlinear (equity: long waits dominate)
    ratio = K.AGE_WEIGHTS[20] / K.AGE_WEIGHTS[5]
    assert ratio > (20.5 / 5.5)
    assert K.AGE_WEIGHTS[-1] == pytest.approx(30.0**1.5)


def test_crowd_cost_is_zero_below_threshold_and_steep_above() -> None:
    cap = np.array([1000.0])
    vals = [K.crowd_cost(np.array([x]), cap, 1.0) for x in (0.0, 700.0, 800.0, 850.0, 900.0, 1000.0)]
    assert vals[0] == vals[1] == vals[2] == 0.0
    assert 0 < vals[3] < vals[4] < vals[5]
    assert vals[5] == pytest.approx(C.W_CROWD)  # (1.0 - 0.8) / 0.2 = 1


def test_tunnel_cost_dominates_hold_cost() -> None:
    onboard = np.array([0.0, 500.0, 1000.0])
    one = np.array([True, False, False])
    assert K.train_cost(one, onboard, C.W_TUNNEL_TRAIN, C.W_TUNNEL_RIDER, 1.0) == C.W_TUNNEL_TRAIN
    for sel in (np.array([False, True, False]), np.array([False, False, True])):
        tunnel = K.train_cost(sel, onboard, C.W_TUNNEL_TRAIN, C.W_TUNNEL_RIDER, 1.0)
        hold = K.train_cost(sel, onboard, C.W_HOLD_TRAIN, C.W_HOLD_RIDER, 1.0)
        assert tunnel > 2 * hold  # even the first tunnel minute costs more than an open-door hold
        # PRD "tunnel >> hold" over a realistic 10-minute stoppage (trapped riders weigh a^1.5)
        tunnel_10 = sum(
            K.train_cost(sel, onboard * K.trapped_age_weight(np.full(3, float(a))), C.W_TUNNEL_TRAIN, C.W_TUNNEL_RIDER, 1.0)
            for a in range(1, 11)
        )
        assert tunnel_10 > 15 * (10 * hold)
    more = K.train_cost(np.array([True, True, False]), onboard, C.W_TUNNEL_TRAIN, C.W_TUNNEL_RIDER, 1.0)
    assert more > K.train_cost(one, onboard, C.W_TUNNEL_TRAIN, C.W_TUNNEL_RIDER, 1.0)


def test_tick_costs_order_and_scaling() -> None:
    w = np.zeros((2, C.N_AGE_BINS))
    w[0, 0] = 100.0
    w[1, 5] = 950.0
    cap = np.array([1000.0, 1000.0])
    onboard = np.array([100.0, 200.0])
    terms = K.tick_costs(w, cap, onboard, np.array([True, False]), np.array([False, True]), dt_s=60.0)
    assert terms.shape == (4,)
    assert terms[K.I_WAIT] == pytest.approx(100.0 * 0.5**1.5 + 950.0 * 5.5**1.5)
    assert terms[K.I_TUNNEL] == pytest.approx(C.W_TUNNEL_TRAIN + C.W_TUNNEL_RIDER * 100.0)
    assert terms[K.I_HOLD] == pytest.approx(C.W_HOLD_TRAIN + C.W_HOLD_RIDER * 200.0)
    assert terms[K.I_CROWD] == pytest.approx(C.W_CROWD * ((0.95 - 0.8) / 0.2) ** 2)
    half = K.tick_costs(w, cap, onboard, np.array([True, False]), np.array([False, True]), dt_s=30.0)
    np.testing.assert_allclose(half * 2.0, terms)


def test_trapped_riders_weigh_like_waiting_riders_of_the_same_age() -> None:
    """Rider weight a^1.5 after a minutes stopped (first minute 1, capped at the last wait bin)."""
    w = K.trapped_age_weight(np.array([0.0, 0.5, 1.0, 4.0, 9.0, 30.0, 300.0]))
    np.testing.assert_allclose(w, [1.0, 1.0, 1.0, 8.0, 27.0, 30.0 ** 1.5, 30.0 ** 1.5])
    # a rider trapped for 30 min is never cheaper than a rider waiting 30 min on a platform
    minutes = np.arange(30) + 0.5
    trapped = float((K.trapped_age_weight(minutes) * C.W_TUNNEL_RIDER).sum())
    waiting = float((minutes ** C.WAIT_AGE_EXPONENT * C.W_WAIT).sum())
    assert trapped >= 0.95 * waiting
    # the per-train term: 200 per train-minute plus the age-weighted riders
    sel = np.array([True, True])
    onboard = np.array([100.0, 100.0])
    terms = K.tick_costs(np.zeros((1, C.N_AGE_BINS)), np.ones(1), onboard, sel, ~sel, 60.0, np.array([1.0, 4.0]))
    assert terms[K.I_TUNNEL] == pytest.approx(2 * C.W_TUNNEL_TRAIN + C.W_TUNNEL_RIDER * 100.0 * (1.0 + 8.0))


def test_trapped_rider_cost_grows_with_the_time_stopped(toy_graph: Graph) -> None:
    """Under a long block the tunnel cost per stuck train-minute rises as the queue ages."""
    sim = NYCSubwaySim(toy_graph, Scenario(seed=5))
    sim.add_disruption("local_block", "segment", toy_segment(toy_graph, "B03", "B04"))
    per_train_min = []
    for _ in range(4):
        before = sim.cost_totals()["tunnel"]
        stuck = 0.0
        for _ in range(30):
            sim.step(1)
            stuck += float(np.count_nonzero(sim.tr_tunnel)) * C.DT_S / 60.0
        if stuck:
            per_train_min.append((sim.cost_totals()["tunnel"] - before) / stuck)
    assert len(per_train_min) >= 3
    assert per_train_min[-1] > per_train_min[0]


def test_engine_charges_tunnel_cost_under_a_block(toy_graph: Graph) -> None:
    calm = NYCSubwaySim(toy_graph, Scenario(seed=5))
    melt = NYCSubwaySim(toy_graph, Scenario(seed=5))
    melt.add_disruption("local_block", "segment", toy_segment(toy_graph, "B03", "B04"))
    calm.step(180)
    melt.step(180)
    c, m = calm.cost_totals(), melt.cost_totals()
    assert set(c) == set(C.COST_KEYS)
    assert c["tunnel"] == 0.0
    assert m["tunnel"] > 10 * C.W_TUNNEL_TRAIN
    assert m["wait"] > c["wait"]
    for totals in (c, m):
        parts = sum(v for k, v in totals.items() if k != "total")
        assert totals["total"] == pytest.approx(parts)
    assert melt.metrics()["tunnel_stuck_trains"] > 0


# --------------------------------------------------------------------------------------
# Metrics and frame
# --------------------------------------------------------------------------------------
def test_metrics_keys_and_types(toy_graph: Graph) -> None:
    sim = NYCSubwaySim(toy_graph, Scenario(seed=1))
    sim.step(30)
    m = sim.metrics()
    assert tuple(m) == C.METRIC_KEYS
    assert isinstance(m["overcrowded_platforms"], int) and isinstance(m["tunnel_stuck_trains"], int)
    assert isinstance(m["trains_active"], int)
    assert m["clock_s"] == pytest.approx(C.PROFILE_START_S["am_peak"] + 30 * C.DT_S)
    assert 0.0 < m["avg_wait_min"] < 30.0
    assert m["riders_served"] > 0 and m["riders_waiting"] > 0 and m["riders_onboard"] > 0


def test_frame_wire_arrays(toy_graph: Graph) -> None:
    sim = NYCSubwaySim(toy_graph, Scenario(seed=1))
    sim.add_disruption("full_station_closure", "station", TOY_STATION["A06"])
    sim.step(20)
    f = sim.frame()
    n = sim.metrics()["trains_active"]
    expect = {"train_id": np.uint16, "edge": np.uint16, "frac": np.uint16, "route": np.uint8, "status": np.uint8,
              "load": np.uint8, "station_crowd": np.uint8, "station_wait": np.uint8}
    assert set(f) == set(expect)
    for k, dt in expect.items():
        assert f[k].dtype == dt, k
    for k in ("train_id", "edge", "frac", "route", "status", "load"):
        assert f[k].shape == (n,)
    assert f["station_crowd"].shape == f["station_wait"].shape == (toy_graph.n_stations,)
    assert np.all(np.diff(f["train_id"].astype(np.int64)) > 0), "ascending slot order"
    assert f["station_crowd"][TOY_STATION["A06"]] == 255
    assert (f["station_crowd"][np.arange(toy_graph.n_stations) != TOY_STATION["A06"]] < 255).all()
    timed = toy_graph.edge_timed[f["edge"].astype(np.int64)]
    assert (f["frac"][timed] == 0).all(), "zero-length edges report frac 0"
    assert (f["status"] & C.ST_AT_PLATFORM).any()
