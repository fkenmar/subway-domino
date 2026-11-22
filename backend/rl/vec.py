"""Vectorized environments for training and benchmarking (ARCHITECTURE §4.6, §4.10).

* :class:`EnvFactory` is a picklable, top-level env constructor, so ``SubprocVecEnv`` works with
  the ``spawn`` start method (macOS default; also used on Linux/Colab for uniformity).
* :class:`ObsMaskVecEnv` answers MaskablePPO's per-step ``env_method("action_masks")`` from the
  observations already received: the last ``N_ACTIONS`` features of every observation are the
  current ego's mask as 0/1 (§4.3), so no extra round-trip to each worker is needed.
* :func:`resolve_n_envs` / :func:`resolve_n_steps` / :func:`resolve_torch_threads` implement the
  §4.10 auto-scaling rules.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvStepReturn

from rl.credit import NEIGHBOR_W
from rl.env import SubwayDominoEnv, TwinMode
from sim import constants as C

#: rollout size per PPO update targeted by ``--n-steps auto`` (§4.10)
ROLLOUT_TARGET: int = 15360
#: BLAS/OpenMP thread caps for env workers (each worker is one single-threaded NumPy sim)
_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def cpu_count() -> int:
    """Usable CPUs (the affinity mask where available, else ``os.cpu_count()``)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def resolve_n_envs(value: str | int, cpus: int | None = None) -> int:
    """``auto`` -> ``clamp(cpu_count - 2, 2, 64)`` (§4.10); integers pass through (>= 1)."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        n = cpu_count() if cpus is None else int(cpus)
        return int(min(64, max(2, n - 2)))
    n_envs = int(value)
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    return n_envs


def resolve_n_steps(value: str | int, n_envs: int) -> int:
    """``auto`` -> ``max(128, round(15360 / n_envs / 64) * 64)`` (§4.10)."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        return int(max(128, round(ROLLOUT_TARGET / n_envs / 64) * 64))
    n_steps = int(value)
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    return n_steps


def resolve_torch_threads(value: str | int, cpus: int | None = None) -> int:
    """``auto`` -> ``clamp(cpu_count // 8, 1, 8)`` learner threads (§4.10)."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        n = cpu_count() if cpus is None else int(cpus)
        return int(min(8, max(1, n // 8)))
    return max(1, int(value))


def resolve_batch_size(rollout: int, target: int = 2048) -> int:
    """The mini-batch size closest to ``target`` that divides ``rollout`` evenly.

    ``n_envs * n_steps`` with ``--n-steps auto`` is a multiple of 64 but not always of 2048;
    an exact divisor avoids PPO's truncated trailing mini-batch.
    """
    if rollout <= target:
        return rollout
    best = None
    for k in range(1, rollout + 1):
        if rollout % k:
            continue
        size = rollout // k
        if size < 64 and best is not None:
            break
        if best is None or abs(size - target) < abs(best - target):
            best = size
    return int(best if best is not None else target)


def limit_worker_threads() -> None:
    """Make spawned workers single-threaded in BLAS/OpenMP (inherited through ``os.environ``)."""
    for k in _THREAD_ENV_VARS:
        os.environ.setdefault(k, "1")


@dataclass(frozen=True)
class EnvFactory:
    """Picklable constructor of one monitored :class:`SubwayDominoEnv` (one per worker)."""

    stage: int = 2
    seed: int | None = None
    graph_path: str | None = None
    twin_mode: TwinMode = "lazy"
    monitor: bool = True
    credit_info: bool = True  # per-step credit data in info (rl.credit); rl.train --credit serial turns it off
    scope_neighbor_weight: float = NEIGHBOR_W
    calm_frac: float = 0.0  # Stage 2 calm anchor episodes (no disruptions)

    def __call__(self) -> gym.Env:
        if mp.parent_process() is not None:
            # Workers ignore Ctrl+C: the trainer catches it, checkpoints, then closes them.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        env: gym.Env = SubwayDominoEnv(
            stage=self.stage, seed=self.seed, graph_path=self.graph_path, twin_mode=self.twin_mode,
            credit_info=self.credit_info, scope_neighbor_weight=self.scope_neighbor_weight, calm_frac=self.calm_frac,
        )
        if self.monitor:
            env = Monitor(env, info_keywords=())
        return env


class ObsMaskVecEnv(VecEnvWrapper):
    """Serves ``env_method("action_masks")`` from the mask bits of the latest observations.

    ``SubwayDominoEnv`` places the current ego's mask in ``obs[-N_ACTIONS:]`` (§4.3). After an
    auto-reset the returned observation is the new episode's first one, so the bits always
    describe the ego that acts next. CONTINUE is always valid (§4.2), which also covers the
    all-zero observation of a dummy (decision-free) episode.
    """

    def __init__(self, venv: VecEnv) -> None:
        super().__init__(venv)
        self._masks = np.zeros((self.num_envs, C.N_ACTIONS), dtype=bool)
        self._masks[:, C.A_CONTINUE] = True

    def _update(self, obs: np.ndarray) -> None:
        m = np.asarray(obs)[:, -C.N_ACTIONS :] > 0.5
        m[:, C.A_CONTINUE] = True
        self._masks = m

    def reset(self) -> Any:
        obs = self.venv.reset()
        self._update(obs)
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        self._update(obs)
        return obs, rewards, dones, infos

    def action_masks(self) -> np.ndarray:
        return self._masks.copy()

    def env_method(
        self, method_name: str, *method_args: Any, indices: VecEnvIndices = None, **method_kwargs: Any
    ) -> list[Any]:
        if method_name == "action_masks" and not method_args and not method_kwargs:
            return [self._masks[i].copy() for i in self._get_indices(indices)]
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)


def make_vec_env(
    n_envs: int,
    *,
    stage: int = 2,
    seed: int = 0,
    graph_path: str | None = None,
    twin_mode: TwinMode = "lazy",
    start_method: str = "spawn",
    use_subprocess: bool = True,
    obs_masks: bool = True,
) -> VecEnv:
    """``n_envs`` monitored envs (worker ``i`` seeded ``seed + i``), optionally in subprocesses."""
    factories: Sequence[EnvFactory] = [
        EnvFactory(stage=stage, seed=seed + i, graph_path=graph_path, twin_mode=twin_mode) for i in range(n_envs)
    ]
    if use_subprocess and n_envs > 1:
        limit_worker_threads()
        venv: VecEnv = SubprocVecEnv(list(factories), start_method=start_method)
    else:
        venv = DummyVecEnv(list(factories))
    venv.seed(seed)
    return ObsMaskVecEnv(venv) if obs_masks else venv
