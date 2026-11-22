"""Gymnasium environment: one decision of one train per step (ARCHITECTURE §4.1, §4.4, §4.5).

``SubwayDominoEnv`` runs two simulators built from the same :class:`sim.Scenario`:

* the **AI sim**, whose decisions the agent answers one ego at a time, and
* the **Normal twin**, advanced in lockstep with every decision resolved as
  ``CONTINUE_SCHEDULED``.

Decision serialization (§4.1). All decisions of one tick come from one snapshot
(:class:`sim.DecisionBatch`). The env hands them to the agent in batch order and applies each
action immediately through ``apply_actions`` (re-validated by the sim) without recomputing
observations, exactly like the server applies a batch. When the batch is exhausted both sims
advance tick by tick until the AI sim has a non-empty batch or the episode ends.

Reward (§4.4)::

    r = clip(-REWARD_SCALE * (dC_ai - dC_twin), +-REWARD_CLIP)

where ``dC`` is the change of each sim's ``cost_totals()["total"]`` inside this step. The sim
charges the immediate action cost (relay riders, skipped riders, ...) into the same total when
the action is applied, so ``dC_ai`` is "ticks advanced + immediate cost of this action". The
twin is action-independent, so under all-CONTINUE play the two sims stay bit-identical and the
reward is exactly 0.

Twin construction (``twin_mode``) - all three are equivalent, the tests prove it with
``state_hash()`` and bit-identical reward sequences:

* ``"reset"``: an independent ``NYCSubwaySim`` reset from the same scenario (the literal
  contract); its ``state_hash()`` must equal the AI sim's right after reset.
* ``"copy"``: ``ai.copy()`` right after reset.
* ``"lazy"`` (default, fastest): the twin is forked with ``ai.copy()`` just before the first
  non-CONTINUE action. Until then the AI sim *is* the Normal trajectory (deterministic sim,
  identical CONTINUE-only inputs), so no second simulator has to be ticked during calm
  stretches before the agent first intervenes.

Per-train credit data (``credit_info=True``, WP5b, ARCHITECTURE §4.4): both sims track
``cost_vectors()`` and ``info`` adds ``t_s``, ``route``, ``imm_cost``, ``d_flow_global`` and
``d_route`` (route-scoped flow deltas, :func:`rl.credit.scope_weights`). The reward above is
unchanged: ``reward == clip(-REWARD_SCALE * (imm_cost + d_flow_global), +-REWARD_CLIP)``.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl.credit import NEIGHBOR_W, scope_weights
from sim import NYCSubwaySim, Scenario, load_graph, sample_training_scenario
from sim import constants as C
from sim.engine import DecisionBatch, empty_batch
from sim.graph import Graph

log = logging.getLogger(__name__)

#: ``scenario_fn(rng, stage, graph) -> Scenario`` (default :func:`sim.sample_training_scenario`)
ScenarioFn = Callable[[np.random.Generator, int, Graph], Scenario]
TwinMode = Literal["lazy", "copy", "reset"]

#: an episode without any decision is resampled at most this many times (§4.5)
MAX_SCENARIO_TRIES: int = 5
STAGES: tuple[int, ...] = (1, 2)

_I_TOTAL: int = C.COST_KEYS.index("total")
_ZERO_OBS = np.zeros(C.OBS_DIM, dtype=np.float32)
_ZERO_OBS.flags.writeable = False
_CONTINUE_ONLY = np.zeros(C.N_ACTIONS, dtype=bool)
_CONTINUE_ONLY[C.A_CONTINUE] = True
_CONTINUE_ONLY.flags.writeable = False


def _cost_vector(sim: NYCSubwaySim) -> np.ndarray:
    """``cost_totals()`` as a float64 vector in ``COST_KEYS`` order (a copy)."""
    return np.array(sim.cost, dtype=np.float64, copy=True)


def _cost_dict(vec: np.ndarray) -> dict[str, float]:
    return dict(zip(C.COST_KEYS, vec.tolist()))


class EpisodeStats:
    """Per-episode accumulators reported in ``info["episode_summary"]`` at the episode end."""

    __slots__ = ("decisions", "chosen", "applied", "valid", "reward_sum", "clipped", "ticks")

    def __init__(self) -> None:
        self.decisions = 0
        self.chosen = np.zeros(C.N_ACTIONS, dtype=np.int64)
        self.applied = np.zeros(C.N_ACTIONS, dtype=np.int64)
        self.valid = np.zeros(C.N_ACTIONS, dtype=np.int64)
        self.reward_sum = 0.0
        self.clipped = 0
        self.ticks = 0


class SubwayDominoEnv(gym.Env[np.ndarray, np.int64]):
    """MaskablePPO-ready env over the citywide sim and its Normal twin (§4.5)."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        stage: int = 2,
        seed: int | None = None,
        graph_path: str | None = None,
        scenario_fn: ScenarioFn | None = None,
        *,
        twin_mode: TwinMode = "lazy",
        credit_info: bool = True,
        scope_neighbor_weight: float = NEIGHBOR_W,
        calm_frac: float = 0.0,
    ) -> None:
        super().__init__()
        if twin_mode not in ("lazy", "copy", "reset"):
            raise ValueError(f"unknown twin_mode {twin_mode!r}")
        if not 0.0 <= float(calm_frac) <= 1.0:
            raise ValueError(f"calm_frac must be in [0, 1], got {calm_frac!r}")
        self.stage = self._check_stage(stage)
        self.graph_path = graph_path
        self.graph: Graph = load_graph(graph_path)  # cached per process, shared read-only
        if scenario_fn is None:
            scenario_fn = (functools.partial(sample_training_scenario, calm_frac=float(calm_frac)) if calm_frac > 0
                           else sample_training_scenario)
        self.scenario_fn: ScenarioFn = scenario_fn
        self.twin_mode: TwinMode = twin_mode
        #: per-step credit data in ``info`` (WP5b, ``rl.credit``): the sims track cost vectors
        self.credit_info = bool(credit_info)
        self.scope_neighbor_weight = float(scope_neighbor_weight)
        if self.credit_info:
            self._w_plat, self._w_rt = scope_weights(self.graph, self.scope_neighbor_weight)
        self.observation_space = spaces.Box(-1.0, 1.0, (C.OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(C.N_ACTIONS)
        self._init_seed = seed
        self._seeded = False
        self._ai: NYCSubwaySim | None = None
        self._twin: NYCSubwaySim | None = None
        self._scenario: Scenario | None = None
        self._batch: DecisionBatch = empty_batch()
        self._row = 0
        self._needs_reset = True
        self._dummy = False
        self._ep = EpisodeStats()
        self._ep_cost0 = np.zeros(len(C.COST_KEYS))
        self._ep_twin_cost0 = np.zeros(len(C.COST_KEYS))
        self._resamples = 0

    # ------------------------------------------------------------------ public helpers
    @staticmethod
    def _check_stage(stage: int) -> int:
        if int(stage) not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
        return int(stage)

    def set_stage(self, stage: int) -> None:
        """Switch the curriculum stage; it takes effect at the next ``reset()``."""
        self.stage = self._check_stage(stage)

    @property
    def sim(self) -> NYCSubwaySim | None:
        """The AI simulator (``None`` before the first reset)."""
        return self._ai

    @property
    def twin(self) -> NYCSubwaySim | None:
        """The Normal twin (``None`` while a lazy twin has not been forked yet)."""
        return self._twin

    @property
    def scenario(self) -> Scenario | None:
        return self._scenario

    def action_masks(self) -> np.ndarray:
        """``bool[N_ACTIONS]``: the current ego's mask (CONTINUE-only once the episode ended)."""
        if self._needs_reset or self._dummy or self._row >= len(self._batch):
            return _CONTINUE_ONLY.copy()
        return self._batch.masks[self._row].copy()

    # ------------------------------------------------------------------ gymnasium API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is None and not self._seeded and self._init_seed is not None:
            seed = int(self._init_seed)
        super().reset(seed=seed)
        self._seeded = True
        fixed = (options or {}).get("scenario")
        if fixed is not None and not isinstance(fixed, Scenario):
            raise TypeError('options["scenario"] must be a sim.Scenario')
        self._resamples = 0
        for attempt in range(MAX_SCENARIO_TRIES):
            sc = fixed if fixed is not None else self.scenario_fn(self.np_random, self.stage, self.graph)
            self._start_episode(sc)
            if self._advance(check_current=True):
                return self._current_obs(), self._reset_info()
            if fixed is not None:
                break  # a fixed scenario cannot be resampled
            self._resamples = attempt + 1
            log.info("scenario seed=%s produced no decision; resampling", sc.seed)
        # No decision at all: one dummy CONTINUE-only step that truncates (§4.5).
        self._dummy = True
        self._needs_reset = False
        return _ZERO_OBS.copy(), self._reset_info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset or self._ai is None:
            raise RuntimeError("step() called before reset() or after the episode ended")
        a = int(action)
        if not 0 <= a < C.N_ACTIONS:
            raise ValueError(f"action {action!r} outside Discrete({C.N_ACTIONS})")
        if self._dummy:
            self._needs_reset = True
            zero = np.zeros(len(C.COST_KEYS))
            info = self._step_info(a, True, zero, zero, 0, -1)
            if self.credit_info:
                info.update(self._credit_info(0.0, -1, 0.0))
            info["episode_summary"] = self._episode_summary()
            return _ZERO_OBS.copy(), 0.0, False, True, info

        ai = self._ai
        batch, row = self._batch, self._row
        tid = int(batch.train_ids[row])
        ep = self._ep
        ep.decisions += 1
        ep.chosen[a] += 1
        ep.valid += batch.masks[row]

        if a != C.A_CONTINUE and self._twin is None:
            self._fork_twin()  # lazy twin: the AI sim is still exactly the Normal trajectory
        ai_before = _cost_vector(ai)
        twin = self._twin
        twin_before = _cost_vector(twin) if twin is not None else ai_before
        credit = self.credit_info
        if credit:
            t_s, route = ai.episode_time_s, int(ai.tr_route[tid])

        events = ai.apply_actions(np.array([tid], dtype=np.int64), np.array([a], dtype=np.int64))
        applied = a == C.A_CONTINUE or (bool(events) and bool(events[0].applied))
        if applied:
            ep.applied[a] += 1
        if credit:
            after_apply = float(ai.cost[_I_TOTAL])  # the sim charges immediate costs at apply time

        self._row += 1
        ticks = 0
        truncated = False
        snap = None
        if self._row >= len(batch):
            if credit and twin is not None:
                snap = (ai.cost_plat.copy(), ai.cost_route_train.copy(), twin.cost_plat.copy(),
                        twin.cost_route_train.copy())
            t0 = ep.ticks
            truncated = not self._advance(check_current=False)
            ticks = ep.ticks - t0

        d_ai = _cost_vector(ai) - ai_before
        # Before the lazy fork the twin and the AI sim are the same trajectory (only CONTINUE
        # was ever applied), so the twin's increments equal the AI's.
        d_twin = (_cost_vector(twin) - twin_before) if twin is not None else d_ai
        advantage = float(d_twin[_I_TOTAL] - d_ai[_I_TOTAL])
        raw = C.REWARD_SCALE * advantage
        reward = float(np.clip(raw, -C.REWARD_CLIP, C.REWARD_CLIP))
        ep.reward_sum += reward
        ep.clipped += int(reward != raw)

        info = self._step_info(a, applied, d_ai, d_twin, ticks, tid)
        if credit:
            imm = after_apply - float(ai_before[_I_TOTAL])
            if snap is None or ticks == 0:  # no ticks, or before the lazy fork: no flow difference
                info.update(self._credit_info(t_s, route, imm))
            else:
                d_flow = (float(ai.cost[_I_TOTAL]) - after_apply) - float(d_twin[_I_TOTAL])
                pa, ra, pt, rt = snap
                d_plat = (ai.cost_plat - pa) - (twin.cost_plat - pt)  # type: ignore[union-attr]
                d_rt = (ai.cost_route_train - ra) - (twin.cost_route_train - rt)  # type: ignore[union-attr]
                d_route = (self._w_plat @ d_plat + self._w_rt @ d_rt).astype(np.float32)
                info.update(self._credit_info(t_s, route, imm, d_flow, d_route))
        if truncated:
            self._needs_reset = True
            info["episode_summary"] = self._episode_summary()
            return _ZERO_OBS.copy(), reward, False, True, info
        return self._current_obs(), reward, False, False, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        self._ai = None
        self._twin = None
        self._batch = empty_batch()
        self._needs_reset = True
        super().close()

    # ------------------------------------------------------------------ internals
    def _start_episode(self, sc: Scenario) -> None:
        if self._ai is None:
            self._ai = NYCSubwaySim(self.graph, sc)
            self._ai.track_cost_vectors = self.credit_info
        else:
            self._ai.reset(sc)
        self._scenario = sc
        self._twin = None
        if self.twin_mode == "copy":
            self._twin = self._ai.copy()
        elif self.twin_mode == "reset":
            self._twin = NYCSubwaySim(self.graph, sc)
            self._twin.track_cost_vectors = self.credit_info
            ha, ht = self._ai.state_hash(), self._twin.state_hash()
            if ha != ht:  # pragma: no cover - would mean the sim is not deterministic
                raise RuntimeError(f"AI sim and Normal twin differ after reset ({ha} != {ht})")
        self._batch = empty_batch()
        self._row = 0
        self._dummy = False
        self._needs_reset = False
        self._ep = EpisodeStats()
        self._ep_cost0 = _cost_vector(self._ai)
        self._ep_twin_cost0 = self._ep_cost0.copy()

    def _fork_twin(self) -> None:
        assert self._ai is not None
        self._twin = self._ai.copy()

    def _advance(self, *, check_current: bool) -> bool:
        """Tick both sims until the AI sim has a non-empty batch (True) or the episode ends."""
        ai = self._ai
        assert ai is not None and self._scenario is not None
        end_s = float(self._scenario.episode_s)
        if check_current and ai.episode_time_s < end_s:
            b = ai.pending_decisions()
            if len(b):
                self._batch, self._row = b, 0
                return True
        twin = self._twin
        ep = self._ep
        while ai.episode_time_s < end_s:
            ai.step(1)
            if twin is not None:
                twin.step(1)
            ep.ticks += 1
            if ai.episode_time_s >= end_s:
                break
            b = ai.pending_decisions()
            if len(b):
                self._batch, self._row = b, 0
                return True
        self._batch, self._row = empty_batch(), 0
        return False

    def _current_obs(self) -> np.ndarray:
        return np.array(self._batch.obs[self._row], dtype=np.float32, copy=True)

    def _reset_info(self) -> dict[str, Any]:
        sc = self._scenario
        return {
            "stage": self.stage,
            "scenario_seed": None if sc is None else int(sc.seed),
            "profile": None if sc is None else sc.profile,
            "routes": None if sc is None or sc.routes is None else list(sc.routes),
            "n_disruptions": 0 if sc is None else len(sc.disruptions),
            "resamples": self._resamples,
            "dummy": self._dummy,
        }

    @staticmethod
    def _step_info(a: int, applied: bool, d_ai: np.ndarray, d_twin: np.ndarray, ticks: int, tid: int) -> dict[str, Any]:
        return {
            "action": C.ACTION_NAMES[a],
            "applied": applied,
            "train_id": tid,
            "ticks": ticks,
            "cost_ai": _cost_dict(d_ai),
            "cost_twin": _cost_dict(d_twin),
            "twin_advantage": float(d_twin[_I_TOTAL] - d_ai[_I_TOTAL]),
        }

    def _credit_info(self, t_s: float, route: int, imm: float, d_flow: float = 0.0,
                     d_route: np.ndarray | None = None) -> dict[str, Any]:
        """Per-step credit data (``rl.credit``): unscaled costs, AI minus twin, flow terms only."""
        return {
            "t_s": float(t_s),  # episode seconds at this decision
            "route": int(route),  # the ego's route index (-1: no ego)
            "imm_cost": float(imm),  # immediate cost charged by this action (0 if not applied)
            "d_flow_global": float(d_flow),  # (dC_ai - dC_twin) of the ticks of this step, no immediate costs
            "d_route": np.zeros(self.graph.n_routes, dtype=np.float32) if d_route is None else d_route,
        }

    def _episode_summary(self) -> dict[str, Any]:
        """Whole-episode numbers for logging (twin advantage in raw cost units)."""
        ai, twin, sc, ep = self._ai, self._twin, self._scenario, self._ep
        assert ai is not None and sc is not None
        c_ai = _cost_vector(ai) - self._ep_cost0
        c_twin = (_cost_vector(twin) - self._ep_twin_cost0) if twin is not None else c_ai
        tot_ai, tot_twin = float(c_ai[_I_TOTAL]), float(c_twin[_I_TOTAL])
        return {
            "stage": self.stage,
            "profile": sc.profile,
            "scenario_seed": int(sc.seed),
            "decisions": ep.decisions,
            "ticks": ep.ticks,
            "reward_sum": ep.reward_sum,
            "clipped_steps": ep.clipped,
            "cost_ai": tot_ai,
            "cost_twin": tot_twin,
            "twin_advantage": tot_twin - tot_ai,
            "twin_advantage_rel": (tot_twin - tot_ai) / tot_twin if tot_twin > 0 else 0.0,
            "chosen": ep.chosen.tolist(),
            "applied": ep.applied.tolist(),
            "valid": ep.valid.tolist(),
            "twin_forked": twin is not None,
            "dummy": self._dummy,
        }
