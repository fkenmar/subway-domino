"""Action masks (ARCHITECTURE §4.1–§4.2, §8; PRD §4.2): each predicate true and false.

Situations are constructed on the toy network (see ``sim/testing.py`` for its layout):
A-division routes 1 (local) and 2 (express) share a 4-track trunk A02–A04 with crossovers
at A02/A04, a pocket at A04 and a diverge at A05; the B-division L line has a yard at B03.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sim import NYCSubwaySim, Scenario, ScheduledDisruption
from sim import constants as C
from sim.graph import Graph, load_graph
from sim.masks import compute_masks
from sim.testing import TOY_PATTERN, TOY_STATION, first_decision, pattern_node, run_until, toy_network_dict, toy_segment


def _sim(g: Graph, *disruptions: tuple[str, str, int], seed: int = 4, profile: str = "am_peak") -> NYCSubwaySim:
    sim = NYCSubwaySim(g, Scenario(seed=seed, profile=profile))
    for d in disruptions:
        sim.add_disruption(*d)
    return sim


def _row(sim: NYCSubwaySim, t: int) -> int:
    return int(np.flatnonzero(sim.pending_decisions().train_ids == t)[0])


def _info(sim: NYCSubwaySim, t: int) -> dict:
    j = _row(sim, t)
    return {k: (v[j] if isinstance(v, np.ndarray) else v[j]) for k, v in sim._pend_info.items()}


# --------------------------------------------------------------------------------------
# Need clauses: calm service is (nearly) decision-free
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("profile", C.PROFILES)
def test_calm_service_is_decision_free(toy_graph: Graph, profile: str) -> None:
    sim = _sim(toy_graph, profile=profile)
    decisions = 0
    for _ in range(360):
        decisions += len(sim.pending_decisions())
        sim.step(1)
    departures = sim.diagnostics()["dwell_samples"]
    assert departures > 30
    assert decisions <= 0.02 * departures


# --------------------------------------------------------------------------------------
# 1 HOLD_IN_STATION
# --------------------------------------------------------------------------------------
def test_hold_valid_when_blocked_ahead_and_capped(toy_graph: Graph) -> None:
    sim = _sim(toy_graph, ("local_block", "segment", toy_segment(toy_graph, "B03", "B04")))
    t, _ = first_decision(sim, C.A_HOLD, route="L")
    info = _info(sim, t)
    assert info["blocked"] and info["lead_gap"] < C.HOLD_GAP_WIDE_MULT * info["lead_h"]
    holds = 0
    while True:
        b = sim.pending_decisions()
        hit = np.flatnonzero(b.train_ids == t)
        if hit.size == 0:
            sim.step(1)
            continue
        if not b.masks[hit[0], C.A_HOLD]:
            break
        (ev,) = sim.apply_actions(np.array([t]), np.array([C.A_HOLD]))
        assert ev.applied
        holds += 1
        f = sim.frame()
        assert f["status"][np.flatnonzero(f["train_id"] == t)[0]] & C.ST_HELD
        assert sim.train_info(t)["mode"] == "dwell"  # doors open, boarding continues
        assert holds <= C.MAX_HOLD_S / C.HOLD_STEP_S
        sim.step(1)
    # MAX_HOLD_S reached: masked from then on
    assert holds == int(C.MAX_HOLD_S / C.HOLD_STEP_S)
    assert sim.cost_totals()["hold"] > 0


def test_hold_masked_in_calm_service_and_when_headway_wide(toy_graph: Graph) -> None:
    sim = _sim(toy_graph)
    checked = 0
    for _ in range(200):
        sim.step(1)
        for t in np.flatnonzero(sim.tr_active):
            if sim.train_info(t)["mode"] == "ready":
                assert not sim.masks_for([t])[0][C.A_HOLD]
                checked += 1
    assert checked > 20


def test_hold_masked_when_headway_already_wide_even_if_blocked(toy_graph: Graph) -> None:
    """Blocked ahead, but the nearest obstacle is > 1.5 headways away: holding only widens the gap."""
    g = toy_graph
    sim = _sim(g, ("local_block", "segment", toy_segment(g, "B04", "B05")))
    b01 = pattern_node(g, "LE", "B01")

    def ready_at_origin() -> list[int]:
        return [t for t in np.flatnonzero(sim.tr_active)
                if sim.train_info(t)["mode"] == "ready" and sim.train_info(t)["node"] == b01]

    assert run_until(sim, lambda: bool(ready_at_origin()), 120)
    (t,) = ready_at_origin()
    # remove every eastbound train ahead of it: the block is now the nearest obstacle
    le = set(int(e) for e in g.pattern_edges(TOY_PATTERN["LE"]))
    for other in np.flatnonzero(sim.tr_active).tolist():
        if other != t and int(sim.p_edge[other, sim.tr_ptr[other]]) in le:
            sim._free_train(other)
    sim._occupancy()
    masks, info = compute_masks(sim, np.array([t]))
    assert info["blocked"][0]
    assert info["lead_gap"][0] >= C.HOLD_GAP_WIDE_MULT * info["lead_h"][0]
    assert not masks[0, C.A_HOLD]


def test_hold_valid_for_bunched_trains_without_disruption(toy_graph: Graph) -> None:
    """Bunching alone (leader < 0.75 headway, nothing blocked) is a reason to hold."""
    sim = _sim(toy_graph)
    seg = toy_segment(toy_graph, "B01", "B02")
    did = sim.add_disruption("local_block", "segment", seg)
    sim.step(60)  # trains bunch up behind the short block
    sim.clear_disruption(did)
    found = False
    for _ in range(120):
        b = sim.pending_decisions()
        info = sim._pend_info
        for j in range(len(b)):
            if b.masks[j, C.A_HOLD] and not info["blocked"][j]:
                assert info["lead_gap"][j] < C.HOLD_GAP_BUNCHED_MULT * info["lead_h"][j]
                found = True
        if found:
            break
        sim.step(1)
    assert found


# --------------------------------------------------------------------------------------
# 2 RUN_EXPRESS / 3 RUN_LOCAL
# --------------------------------------------------------------------------------------
def test_run_express_valid_around_a_local_blockage(toy_graph: Graph) -> None:
    g = toy_graph
    sim = _sim(g, ("local_block", "station", TOY_STATION["A03"]))
    t, _ = first_decision(sim, C.A_EXPRESS, station=TOY_STATION["A02"], route="1")
    assert sim.train_info(t)["node"] == pattern_node(g, "1S", "A02")  # on the L track
    riders = float(sim.tr_onboard[t].sum())
    (ev,) = sim.apply_actions(np.array([t]), np.array([C.A_EXPRESS]))
    assert ev.applied and ev.detail["bypassed_station_idx"] == [TOY_STATION["A03"]]
    assert ev.detail["rejoin_station_idx"] == TOY_STATION["A04"]
    assert 0 <= ev.detail["riders_displaced"] <= riders
    assert sim.cost_totals()["skip"] == pytest.approx(C.W_SKIP * ev.detail["riders_displaced"])
    # it dwells next at the A04 express platform, never at A03
    x_a04 = pattern_node(g, "2S", "A04", "X")
    assert run_until(sim, lambda: sim.train_info(t)["mode"] == "dwell", 30)
    assert sim.train_info(t)["node"] == x_a04
    f = sim.frame()
    assert f["status"][np.flatnonzero(f["train_id"] == t)[0]] & C.ST_TRACK_SWITCH


def test_run_express_masked_without_need_or_open_express(toy_graph: Graph) -> None:
    g = toy_graph
    a02 = TOY_STATION["A02"]
    # no need: calm service never offers it
    calm = _sim(g)
    for _ in range(200):
        assert not calm.pending_decisions().masks[:, C.A_EXPRESS].any()
        calm.step(1)
    # need, but the express track is blocked too: masked
    both = _sim(g, ("local_block", "station", TOY_STATION["A03"]), ("express_block", "segment", toy_segment(g, "A03", "A04")))
    for _ in range(200):
        b = both.pending_decisions()
        for j, ctx in enumerate(b.context):
            if ctx["station_idx"] == a02:
                assert not b.masks[j, C.A_EXPRESS]
        both.step(1)


def test_run_local_valid_around_an_express_blockage(toy_graph: Graph) -> None:
    g = toy_graph
    sim = _sim(g, ("express_block", "segment", toy_segment(g, "A02", "A03")))
    t, _ = first_decision(sim, C.A_LOCAL, station=TOY_STATION["A02"], route="2")
    assert int(g.node_track[sim.train_info(t)["node"]]) == 1  # standing on the express track
    (ev,) = sim.apply_actions(np.array([t]), np.array([C.A_LOCAL]))
    assert ev.applied and ev.detail["bypassed_station_idx"] == []
    assert ev.detail["riders_displaced"] == 0.0  # a local run serves every original stop
    stops = []
    for _ in range(60):
        sim.step(1)
        info = sim.train_info(t)
        if info["mode"] == "dwell" and (not stops or stops[-1] != info["station"]):
            stops.append(info["station"])
        if len(stops) >= 2:
            break
    assert stops[:2] == [TOY_STATION["A03"], TOY_STATION["A04"]]


def test_run_local_masked_when_local_track_blocked_too(toy_graph: Graph) -> None:
    g = toy_graph
    sim = _sim(g, ("express_block", "segment", toy_segment(g, "A02", "A03")),
               ("local_block", "segment", toy_segment(g, "A02", "A03")))
    for _ in range(200):
        assert not sim.pending_decisions().masks[:, C.A_LOCAL].any()
        sim.step(1)


# --------------------------------------------------------------------------------------
# 4 SKIP_STOP
# --------------------------------------------------------------------------------------
def test_skip_valid_for_a_late_train_and_never_twice(toy_graph: Graph) -> None:
    g = toy_graph
    sim = _sim(g, ("signal_delay", "segment", toy_segment(g, "A01", "A02")))
    t, _ = first_decision(sim, C.A_SKIP, station=TOY_STATION["A02"], route="1")
    info = _info(sim, t)
    assert info["late"] > C.SKIP_LATENESS_S or info["foll_gap"] < C.SKIP_FOLLOWER_MULT * info["foll_h"]
    (ev,) = sim.apply_actions(np.array([t]), np.array([C.A_SKIP]))
    assert ev.applied and ev.detail["skipped_station_idx"] == TOY_STATION["A03"]
    assert sim.cost_totals()["skip"] == pytest.approx(C.W_SKIP * ev.detail["riders_overshot"])
    assert ev.detail["riders_overshot"] > 0
    f = sim.frame()
    assert f["status"][np.flatnonzero(f["train_id"] == t)[0]] & C.ST_SKIPPING_NEXT
    # the train passes A03 without stopping and next dwells at A04 ...
    assert run_until(sim, lambda: sim.train_info(t)["mode"] == "dwell", 40)
    assert sim.train_info(t)["station"] == TOY_STATION["A04"]
    # ... where SKIP is masked (it skipped its previous stop)
    assert run_until(sim, lambda: sim.train_info(t)["mode"] == "ready", 20)
    assert not sim.masks_for([t])[0][C.A_SKIP]


def test_skip_masked_before_hubs_terminals_and_critical_waits(toy_graph: Graph) -> None:
    g = toy_graph
    hubs_or_terminals = {s for s in range(g.n_stations) if g.station_hub[s] or g.station_terminal[s]}
    sim = _sim(g, ("signal_delay", "segment", toy_segment(g, "A01", "A02")),
               ("signal_delay", "segment", toy_segment(g, "A02", "A03")))
    late_seen = 0
    for _ in range(300):
        b = sim.pending_decisions()
        for j, t in enumerate(b.train_ids.tolist()):
            sptr = int(sim.tr_sptr[t])
            nxt_station = int(sim.s_station[t, sptr]) if sptr < int(sim.tr_nstops[t]) else -1
            if nxt_station in hubs_or_terminals:
                assert not b.masks[j, C.A_SKIP]
            if sim._pend_info["late"][j] > C.SKIP_LATENESS_S:
                late_seen += 1
        sim.step(1)
    assert late_seen > 0
    # a critical wait at the next stop masks SKIP, unless that stop is closed
    sim2 = _sim(g, ("signal_delay", "segment", toy_segment(g, "A01", "A02")))
    t, _ = first_decision(sim2, C.A_SKIP, station=TOY_STATION["A02"], route="1")
    a03 = int(g.node_platform[pattern_node(g, "1S", "A03")])
    sim2.waiting[a03, 25] += 50.0  # riders waiting 25 min: critical
    sim2._pl_stats = None
    assert not compute_masks(sim2, np.array([t]))[0][0, C.A_SKIP]
    sim2.add_disruption("full_station_closure", "station", TOY_STATION["A03"])
    assert compute_masks(sim2, np.array([t]))[0][0, C.A_SKIP]


# --------------------------------------------------------------------------------------
# 6 REROUTE_VIA_COMPATIBLE_TRUNK
# --------------------------------------------------------------------------------------
def test_reroute_valid_at_a_diverge_with_the_branch_blocked(toy_graph: Graph) -> None:
    g = toy_graph
    sim = _sim(g, ("local_block", "segment", toy_segment(g, "A06", "A08")))
    t, _ = first_decision(sim, C.A_REROUTE, station=TOY_STATION["A05"], route="1")
    cost0 = sim.cost_totals()["reroute"]
    (ev,) = sim.apply_actions(np.array([t]), np.array([C.A_REROUTE]))
    assert ev.applied and ev.detail["via_station_idx"] == TOY_STATION["A07"]
    assert sim.cost_totals()["reroute"] - cost0 == pytest.approx(C.W_REROUTE + C.W_REROUTE_RIDER * ev.detail["riders_displaced"])
    # the rerouted train stays in its division and serves the alternate branch
    assert run_until(sim, lambda: sim.train_info(t)["mode"] == "dwell", 40)
    assert sim.train_info(t)["station"] == TOY_STATION["A07"]
    f = sim.frame()
    assert f["status"][np.flatnonzero(f["train_id"] == t)[0]] & C.ST_REROUTED


def test_reroute_masked_without_need_or_with_alternate_blocked(toy_graph: Graph) -> None:
    g = toy_graph
    calm = _sim(g)
    both = _sim(g, ("local_block", "segment", toy_segment(g, "A06", "A08")),
                ("local_block", "segment", toy_segment(g, "A05", "A07")))
    for _ in range(240):
        assert not calm.pending_decisions().masks[:, C.A_REROUTE].any()
        assert not both.pending_decisions().masks[:, C.A_REROUTE].any()
        calm.step(1)
        both.step(1)


# --------------------------------------------------------------------------------------
# Structural invariants and consistency
# --------------------------------------------------------------------------------------
def _random_meltdown_rollout(g: Graph, ticks: int, seed: int):
    sc = Scenario(seed=seed, headway_jitter=0.2, disruptions=(
        ScheduledDisruption("local_block", "segment", toy_segment(g, "B03", "B04"), 0.0, 1800.0),
        ScheduledDisruption("full_station_closure", "station", TOY_STATION["A05"], 200.0, 900.0),
        ScheduledDisruption("express_block", "segment", toy_segment(g, "A03", "A04"), 600.0, 900.0),
    ))
    sim = NYCSubwaySim(g, sc)
    rng = np.random.Generator(np.random.PCG64(seed))
    for _ in range(ticks):
        b = sim.pending_decisions()
        yield sim, b
        if len(b):
            acts = np.array([rng.choice(np.flatnonzero(m)) for m in b.masks], dtype=np.int64)
            sim.apply_actions(b.train_ids, acts)
        sim.step(1)


def test_mask_rows_follow_the_contract_predicates(toy_graph: Graph) -> None:
    n = 0
    for sim, b in _random_meltdown_rollout(toy_graph, 400, seed=1):
        if not len(b):
            continue
        info = sim._pend_info
        assert (info["lead_ref"] <= info["lead_h"] + 1e-9).all() and (info["foll_ref"] <= info["foll_h"] + 1e-9).all()
        hold = ((info["lead_gap"] < C.HOLD_GAP_WIDE_MULT * info["lead_h"])
                & ((info["lead_gap"] < C.HOLD_GAP_BUNCHED_MULT * info["lead_ref"]) | info["blocked"])
                & (sim.tr_hold_total[b.train_ids] < C.MAX_HOLD_S - 1e-6))
        assert not (b.masks[:, C.A_HOLD] & ~hold).any()
        skip_need = (info["late"] > C.SKIP_LATENESS_S) | (info["foll_gap"] < C.SKIP_FOLLOWER_MULT * info["foll_ref"])
        assert not (b.masks[:, C.A_SKIP] & ~skip_need).any()
        st = b.masks[:, C.A_SHORT_TURN]
        imbalance = ((info["opp_gap"] > C.SHORT_TURN_OPPOSITE_GAP_MULT * info["opp_h"])
                     & (info["foll_gap"] < C.SHORT_TURN_FOLLOWER_MULT * info["foll_ref"]))
        for j in np.flatnonzero(st & ~imbalance).tolist():  # else: a blockage within the horizon
            t = int(b.train_ids[j])
            rem = sim.p_edge[t, int(sim.tr_ptr[t]) + (0 if sim.tr_at_start[t] else 1): int(sim.tr_plen[t])]
            assert (sim.e_blocked[rem] | sim.n_closed[sim._S.eto[rem]]).any()
        assert b.masks[:, 0].all() and b.masks[:, 1:].any(axis=1).all()
        # re-computing the masks from the same snapshot gives the same rows
        np.testing.assert_array_equal(compute_masks(sim, b.train_ids.astype(np.int64))[0], b.masks)
        n += len(b)
    assert n > 20


def test_actions_are_revalidated_against_the_current_state(toy_graph: Graph) -> None:
    """Two trains offered the same yard: after the first dispatch the second is refused."""
    g = toy_graph
    sim = _sim(g, ("local_block", "segment", toy_segment(g, "B03", "B04")))
    for _ in range(300):
        b = sim.pending_decisions()
        rows = np.flatnonzero(b.masks[:, C.A_YARD])
        if rows.size >= 2:
            ids = b.train_ids[rows[:2]]
            evs = sim.apply_actions(ids, np.array([C.A_YARD, C.A_YARD]))
            assert [e.applied for e in evs] == [True, False]
            assert sim.action_counts()["DISPATCH_YARD_GAP_TRAIN"] == 1
            return
        if rows.size == 1:
            sim.apply_actions(b.train_ids[rows], np.array([C.A_HOLD if b.masks[rows[0], C.A_HOLD] else 0]))
        sim.step(1)
    pytest.skip("no tick offered the yard to two trains at once")


def test_masks_identical_between_env_and_pending_decisions(tmp_path: Path) -> None:
    """The Gym env (WP5) serializes exactly the sim's DecisionBatch rows."""
    env_mod = pytest.importorskip("rl.env")
    path = tmp_path / "network.json"
    path.write_text(json.dumps(toy_network_dict()))
    g = load_graph(path)
    sc = Scenario(seed=4, disruptions=(ScheduledDisruption("local_block", "segment", toy_segment(g, "B03", "B04"), 0.0, None),))
    try:
        env = env_mod.SubwayDominoEnv(stage=2, seed=0, graph_path=str(path))
        obs, _ = env.reset(seed=0, options={"scenario": sc})
    except Exception as exc:  # pragma: no cover - depends on WP5's implementation state
        pytest.skip(f"rl.env not usable yet: {exc!r}")
    sim = NYCSubwaySim(g, sc)
    first_decision(sim)
    b = sim.pending_decisions()
    np.testing.assert_array_equal(env.action_masks(), b.masks[0])
    np.testing.assert_allclose(obs, b.obs[0])
    for j in range(1, len(b)):
        obs, *_ = env.step(0)
        sim.apply_actions(b.train_ids[j - 1 : j], np.zeros(1, dtype=np.int64))
        np.testing.assert_array_equal(env.action_masks(), b.masks[j])
        np.testing.assert_allclose(obs, b.obs[j])


# --------------------------------------------------------------------------------------
# Timetable-relative gap references (sim.masks module docstring, WP2 CR 4)
# --------------------------------------------------------------------------------------
def test_planned_ref_judges_gaps_against_the_timetable(toy_graph: Graph) -> None:
    from sim.masks import planned_ref

    sim = _sim(toy_graph)
    a, b = np.flatnonzero(sim.tr_active)[:2]
    gap, h = np.array([60.0]), np.array([120.0])
    ahead, behind = np.array([a]), np.array([b])
    sim.tr_late[[a, b]] = 0.0  # both on time: a 60 s gap is the plan (an uncoordinated merge)
    assert planned_ref(sim, gap, h, ahead, behind)[0] == pytest.approx(60.0)
    assert not 60.0 < C.HOLD_GAP_BUNCHED_MULT * 60.0
    sim.tr_late[a] = 40.0  # the leader runs 40 s late: planned 100 s, the gap has shrunk
    assert planned_ref(sim, gap, h, ahead, behind)[0] == pytest.approx(100.0)
    assert 60.0 < C.HOLD_GAP_BUNCHED_MULT * 100.0
    sim.tr_late[a] = 200.0  # planned gap beyond the track headway: the track headway rules
    assert planned_ref(sim, gap, h, ahead, behind)[0] == pytest.approx(120.0)
    sim.tr_late[[a, b]] = [0.0, 90.0]  # the follower is the late one: planned gap < 0 -> floor
    assert planned_ref(sim, gap, h, ahead, behind)[0] == pytest.approx(C.GAP_PLAN_MIN_S)
    # unknown leader (a red signal standing in for it): the track headway
    assert planned_ref(sim, gap, h, np.array([-1]), behind)[0] == pytest.approx(120.0)


def test_calm_real_network_is_nearly_decision_free(real_graph: Graph) -> None:
    """Calm am_peak: decisions at well under 10 % of departures (was ~58 % with the track
    headway as the only reference: merges of uncoordinated patterns look bunched by timetable)."""
    sim = NYCSubwaySim(real_graph, Scenario(seed=0, profile="am_peak"))
    decisions = departures = 0
    for _ in range(180):
        before = sim.nd_last_dep.copy()
        sim.step(1)
        decisions += int(sim._pend_ids.size)
        departures += int(np.count_nonzero(sim.nd_last_dep != before))
    assert departures > 3000
    assert decisions < 0.10 * departures, (decisions, departures)


def test_short_turn_needs_the_blockage_within_the_horizon(real_graph: Graph) -> None:
    """lex-union-sq: relays are offered in front of 14 St (Bowling Green, ~8 min before it), never
    at pockets far up the line (the Bronx, ~25 min) whose trains still serve every station
    before the closure."""
    from sim.scenarios import preset_scenario

    sim = NYCSubwaySim(real_graph, preset_scenario("lex-union-sq", real_graph))
    S = sim._S
    seen = set()
    for _ in range(180):
        sim.step(1)
        if not sim._pend_ids.size:
            continue
        info = sim._pend_info
        for j in np.flatnonzero(sim._pend_masks[:, C.A_SHORT_TURN]).tolist():
            imbalance = (info["opp_gap"][j] > C.SHORT_TURN_OPPOSITE_GAP_MULT * info["opp_h"][j]
                         and info["foll_gap"][j] < C.SHORT_TURN_FOLLOWER_MULT * info["foll_ref"][j])
            if not imbalance:
                t = int(sim._pend_ids[j])
                rem = sim.p_edge[t, int(sim.tr_ptr[t]): int(sim.tr_plen[t])]
                assert sim.n_closed[S.eto[rem]].any()
                seen.add(real_graph.station_name[int(S.nst[info["node"][j]])])
    assert seen == {"Bowling Green"}, seen
