"""Action leverage on the real network (marker ``slow``).

The RL problem is only learnable if the action space can beat Normal Schedule. A deliberately
simple scripted controller, defined **only in this file** (it is a test instrument, never a
dispatch heuristic of the product), must beat all-CONTINUE on the total PRD cost and on
tunnel-stuck train-minutes for every preset in ``network.json`` and for three random Stage-2
training scenarios. The same runs report the decision rate and the distribution of valid
actions; each of the 8 actions must be valid somewhere.

The random seeds are fixed (11, 12, 13), not tuned. The controller acts on incidents only (see
:func:`make_scripted_policy`); the naive variant that also switches tracks and holds on mere
congestion loses on 8av-59st-local (every local sent onto the express saturates it) and on
times-sq-signals (holding behind a capacity restriction moves the queue into the tunnel), which
is precisely the judgement left to the learned policy. Track switches are shown to pay off where
they bypass the blockage.

The calibration preconditions of the presets (a meltdown under Normal Schedule: tunnel-stuck
trains and waits rising within 15 sim-min, see ``sim.calibrate``) are asserted here too.

Run with ``-s`` to see the report.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from sim import constants as C
from sim.engine import DecisionBatch, NYCSubwaySim
from sim.graph import Graph
from sim.obs import FEATURE_NAMES
from sim.scenarios import Scenario, preset_scenario, sample_training_scenario

pytestmark = pytest.mark.slow

RANDOM_STAGE2_SEEDS = (11, 12, 13)
EPISODE_S = 3600.0  # presets disrupt for 30 min; the second half measures the recovery

# ---------------------------------------------------------------------------------------
# The scripted controller (test instrument). It reads only what the policy network reads:
# the observation and the mask of each decision.
# ---------------------------------------------------------------------------------------
_FI = {name: i for i, name in enumerate(FEATURE_NAMES)}
_AHEAD = ["current"] + [f"ahead{k}" for k in range(1, C.K_AHEAD + 1)]


def _slot_cols(feature: str) -> list[int]:
    return [_FI[f"{slot}.{feature}"] for slot in _AHEAD]


_HARD_COLS = _slot_cols("next_edge_blocked") + _slot_cols("node_closed")
_STOPPED_COLS = _slot_cols("stopped_trains_next_edge_over_3")


def disruption_ahead(obs: np.ndarray) -> np.ndarray:
    """A blocked edge or a closed station in the current or next ``K_AHEAD`` stations' slots."""
    return (obs[:, _HARD_COLS] > 0).any(axis=1)


def queue_ahead(obs: np.ndarray) -> np.ndarray:
    """At least two trains stopped in the tunnel within the next ``K_AHEAD`` stations."""
    return obs[:, _STOPPED_COLS].sum(axis=1) * 3.0 >= 2.0 - 1e-6


def make_scripted_policy(*, track_switches: bool) -> Callable[[DecisionBatch], np.ndarray]:
    """Incident response by rote, in priority order (reads only the observation and the mask).

    1. SHORT_TURN_RELAY when valid and either the line ahead already queues (>= 2 trains
       stopped) or no blockage is ahead (the mask then fired for a service imbalance): do not
       send more trains into a dead end than the stations in front of it need.
    2. (``track_switches``) REROUTE / RUN_EXPRESS / RUN_LOCAL when valid and a hard disruption
       (blocked track or closed station) is ahead.
    3. HOLD_IN_STATION when valid and a hard disruption is ahead: wait at the platform with the
       doors open instead of joining the queue in the tunnel.
    4. DISPATCH_YARD_GAP_TRAIN whenever valid (its mask requires a starved segment).
    5. Otherwise CONTINUE_SCHEDULED.

    Routine congestion without an incident is left alone: holding behind a capacity restriction
    (signal delay) only moves the queue into the tunnel behind the platform.
    """

    def policy(batch: DecisionBatch) -> np.ndarray:
        m = batch.masks
        hard = disruption_ahead(batch.obs)
        always = np.ones(len(batch), dtype=bool)
        switch = hard if track_switches else np.zeros(len(batch), dtype=bool)
        order = (
            (C.A_SHORT_TURN, queue_ahead(batch.obs) | ~hard),
            (C.A_REROUTE, switch),
            (C.A_EXPRESS, switch),
            (C.A_LOCAL, switch),
            (C.A_HOLD, hard),
            (C.A_YARD, always),
        )
        act = np.zeros(len(batch), dtype=np.int64)
        for a, cond in reversed(order):  # highest priority written last
            act = np.where(m[:, a] & cond, a, act)
        return act

    return policy


scripted_policy = make_scripted_policy(track_switches=False)
switching_policy = make_scripted_policy(track_switches=True)


def all_continue(batch: DecisionBatch) -> np.ndarray:
    return np.zeros(len(batch), dtype=np.int64)


# ---------------------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------------------
@dataclass
class Outcome:
    cost: float
    stuck_train_min: float
    decisions: int
    valid: np.ndarray
    applied: dict[str, int]
    cost_terms: dict[str, float]


def run_episode(graph: Graph, sc: Scenario, policy: Callable[[DecisionBatch], np.ndarray]) -> Outcome:
    sim = NYCSubwaySim(graph, sc)
    n_ticks = int(round(sc.episode_s / C.DT_S))
    stuck_ticks = 0
    decisions = 0
    valid = np.zeros(C.N_ACTIONS, dtype=np.int64)
    for _ in range(n_ticks):
        batch = sim.pending_decisions()
        if len(batch):
            decisions += len(batch)
            valid += batch.masks.sum(axis=0)
            sim.apply_actions(batch.train_ids, policy(batch))
        sim.step(1)
        stuck_ticks += sim.metrics()["tunnel_stuck_trains"]
    costs = sim.cost_totals()
    return Outcome(
        cost=costs["total"],
        stuck_train_min=stuck_ticks * C.DT_S / 60.0,
        decisions=decisions,
        valid=valid,
        applied=sim.action_counts(),
        cost_terms=costs,
    )


def leverage_scenarios(graph: Graph) -> list[tuple[str, Scenario]]:
    out = [(p["id"], dataclasses.replace(preset_scenario(p["id"], graph), episode_s=EPISODE_S)) for p in graph.presets]
    for seed in RANDOM_STAGE2_SEEDS:
        sc = sample_training_scenario(np.random.default_rng(seed), 2, graph)
        out.append((f"stage2-seed{seed}", dataclasses.replace(sc, episode_s=EPISODE_S)))
    return out


@pytest.fixture(scope="module")
def leverage_results(real_graph: Graph) -> dict[str, tuple[Scenario, Outcome, Outcome]]:
    res = {}
    for name, sc in leverage_scenarios(real_graph):
        res[name] = (sc, run_episode(real_graph, sc, all_continue), run_episode(real_graph, sc, scripted_policy))
    _report(res)
    return res


def _report(res: dict[str, tuple[Scenario, Outcome, Outcome]]) -> None:
    hours = EPISODE_S / 3600.0
    print("\nscenario                 profile     cost NORMAL -> SCRIPTED (delta)        stuck train-min        decisions/h")
    for name, (sc, base, ctl) in res.items():
        d = 100.0 * (ctl.cost - base.cost) / max(base.cost, 1e-9)
        print(f"{name:24s} {sc.profile:10s} {base.cost:12.0f} -> {ctl.cost:12.0f} ({d:+6.1f} %)  "
              f"{base.stuck_train_min:8.1f} -> {ctl.stuck_train_min:8.1f}  {base.decisions / hours:8.0f}")
        applied = {k.split("_")[0] + "_" + k.split("_")[1]: v for k, v in ctl.applied.items() if v}
        print(f"{'':24s} applied {applied}")
    tot = sum((b.valid for _, b, _ in res.values()), np.zeros(C.N_ACTIONS, dtype=np.int64))
    print("valid-action counts under Normal Schedule (all scenarios):",
          {n: int(c) for n, c in zip(C.ACTION_NAMES, tot)})


# ---------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------
#: Where Normal Schedule leaves (almost) no train stuck in a tunnel there is nothing to beat on
#: that metric (a late-night scenario: one train held at a red signal for 20 s); the controller
#: must then merely not be worse.
STUCK_NOISE_FLOOR_TRAIN_MIN = 1.0


def _beats(base: Outcome, ctl: Outcome) -> bool:
    if not ctl.cost < base.cost:
        return False
    if base.stuck_train_min < STUCK_NOISE_FLOOR_TRAIN_MIN:
        return ctl.stuck_train_min <= base.stuck_train_min
    return ctl.stuck_train_min < base.stuck_train_min


def test_scripted_controller_beats_normal_schedule(leverage_results: dict[str, tuple[Scenario, Outcome, Outcome]]) -> None:
    losers = []
    for name, (_, base, ctl) in leverage_results.items():
        if not _beats(base, ctl):
            losers.append(f"{name}: cost {base.cost:.0f}->{ctl.cost:.0f}, stuck {base.stuck_train_min:.1f}->{ctl.stuck_train_min:.1f}")
    assert not losers, "scripted controller did not beat Normal Schedule:\n" + "\n".join(losers)


@pytest.mark.parametrize("preset_id", ["qbl-express-roosevelt", "manhattan-bridge"])
def test_track_switches_have_leverage_where_they_bypass_the_blockage(
        real_graph: Graph, preset_id: str, leverage_results: dict[str, tuple[Scenario, Outcome, Outcome]]) -> None:
    """Express blocked at Jackson Hts (RUN_LOCAL around it) and the Manhattan Bridge closed
    (REROUTE via the tunnel): adding track switches to the scripted controller pays off.

    Switching is not a free win: on 8av-59st-local, sending every local onto the express
    saturates it (that judgement is what the policy has to learn).
    """
    sc, _, no_switch = leverage_results[preset_id]
    with_switch = run_episode(real_graph, sc, switching_policy)
    used = sum(with_switch.applied[C.ACTION_NAMES[a]] for a in (C.A_EXPRESS, C.A_LOCAL, C.A_REROUTE))
    assert used > 0
    assert with_switch.cost < no_switch.cost, (with_switch.cost, no_switch.cost)
    assert with_switch.stuck_train_min < no_switch.stuck_train_min


def test_every_action_is_valid_somewhere(leverage_results: dict[str, tuple[Scenario, Outcome, Outcome]]) -> None:
    tot = np.zeros(C.N_ACTIONS, dtype=np.int64)
    for _, base, ctl in leverage_results.values():
        tot += base.valid + ctl.valid
    never = [C.ACTION_NAMES[a] for a in range(C.N_ACTIONS) if tot[a] == 0]
    assert not never, f"actions never valid in any preset/random scenario: {never}"


def test_meltdowns_generate_more_decisions_than_calm(real_graph: Graph,
                                                    leverage_results: dict[str, tuple[Scenario, Outcome, Outcome]]) -> None:
    calm = run_episode(real_graph, Scenario(seed=0, profile="am_peak", episode_s=EPISODE_S), all_continue)
    for pid in ("lex-union-sq", "manhattan-bridge", "qbl-express-roosevelt"):
        _, base, _ = leverage_results[pid]
        assert base.decisions > 1.3 * calm.decisions, (pid, base.decisions, calm.decisions)


def test_preset_meltdown_calibration(real_graph: Graph) -> None:
    """lex-union-sq under Normal Schedule: >= 6 tunnel-stuck trains and a clearly rising wait in 15 min."""
    from sim.calibrate import PRESET_STUCK_MIN, REFERENCE_MELTDOWN, run_preset

    r = run_preset(real_graph, REFERENCE_MELTDOWN)
    assert r.stuck_max >= PRESET_STUCK_MIN, r.stuck_series
    assert r.checks()["wait_rising"], (r.wait_series, r.calm_wait_series)


@pytest.mark.parametrize("preset_id", ["qbl-express-roosevelt", "times-sq-signals", "canarsie-tube",
                                       "manhattan-bridge", "8av-59st-local", "joralemon-tunnel"])
def test_every_preset_hurts_under_normal_schedule(real_graph: Graph, preset_id: str) -> None:
    """Track blockages jam the tunnel (>= 6 stuck trains in 15 min); the signal delay stops more
    trains than calm service; every preset raises the citywide wait above its calm twin."""
    from sim.calibrate import format_preset, run_preset

    r = run_preset(real_graph, preset_id)
    failed = [k for k, ok in r.checks().items() if not ok]
    assert not failed, f"{failed}\n{format_preset(r)}"
