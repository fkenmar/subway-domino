"""MaskablePPO with chunked, worker-side-inference rollouts (see :mod:`rl.rollout`).

``ChunkedMaskablePPO`` is ``sb3_contrib.MaskablePPO``: the same policy class, loss, update and
saved format. Only ``collect_rollouts`` differs. The rollout comes from
:class:`~rl.rollout.RolloutWorkers` as one trajectory segment per worker, and is stored in a
``MaskableRolloutBuffer`` of ``n_envs * n_steps`` rows (one column) with GAE computed per
segment:

* a segment that ends mid-episode bootstraps with ``V(next obs)``;
* the episode horizon ``episode_s`` is a **terminal** of the finite-horizon objective, so it
  does not bootstrap. The observation carries ``time_in_episode_frac``, and evaluation scores
  exactly that horizon. The env still reports it as ``truncated`` (§4.5).

Rewards may be normalized by a running estimate of the discounted-return standard deviation
(the ``VecNormalize(norm_reward=True)`` rule that §4.6 allows). This keeps value targets O(1),
so the shared global gradient-norm clip does not starve the policy gradient. Raw rewards are
still what gets logged.

Checkpoints saved by this class load with plain ``MaskablePPO.load`` (``sim.policy.Sb3ZipPolicy``,
``rl.export_onnx``).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from rl.rollout import RolloutWorkers, Segment, mlp_params_from_policy
from sim import constants as C


class ReturnNormalizer:
    """Running std of the discounted return per worker (SB3 ``VecNormalize`` reward rule)."""

    def __init__(self, gamma: float, clip: float = 10.0, epsilon: float = 1e-8) -> None:
        self.gamma = float(gamma)
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
        self._ret: dict[int, float] = {}

    def _update(self, x: np.ndarray) -> None:
        b_mean, b_var, b_n = float(x.mean()), float(x.var()), int(x.size)
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta * delta * self.count * b_n / tot
        self.var = m2 / tot
        self.count = tot

    def normalize(self, worker: int, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Update with this worker's next rewards (time order) and return them normalized."""
        ret = self._ret.get(worker, 0.0)
        rets = np.empty(rewards.shape[0], dtype=np.float64)
        for t in range(rewards.shape[0]):
            ret = ret * self.gamma + float(rewards[t])
            rets[t] = ret
            if dones[t]:
                ret = 0.0
        self._ret[worker] = ret
        self._update(rets)
        out = rewards / np.sqrt(self.var + self.epsilon)
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count, "gamma": self.gamma, "clip": self.clip}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.mean, self.var, self.count = float(d["mean"]), float(d["var"]), float(d["count"])


def segment_gae(seg_rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: float,
                gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """GAE(lambda) advantages and returns of one time-ordered segment (dones = terminal steps)."""
    n = seg_rewards.shape[0]
    adv = np.zeros(n, dtype=np.float32)
    last = 0.0
    for t in range(n - 1, -1, -1):
        nonterminal = 0.0 if dones[t] else 1.0
        next_v = last_value if t == n - 1 else float(values[t + 1])
        delta = float(seg_rewards[t]) + gamma * next_v * nonterminal - float(values[t])
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values.astype(np.float32)


class ChunkedMaskablePPO(MaskablePPO):
    """MaskablePPO whose rollouts come from :class:`RolloutWorkers` (one barrier per rollout)."""

    def __init__(self, *args: Any, rollout_chunk: int = 64, normalize_reward: bool = True, **kwargs: Any) -> None:
        self.rollout_chunk = int(rollout_chunk)
        self.normalize_reward = bool(normalize_reward)
        self.reward_normalizer: ReturnNormalizer | None = None
        self.should_stop: Any = None  # optional callable polled between chunks (graceful SIGINT)
        self.last_rollout: dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ setup / save
    def _setup_model(self) -> None:
        super()._setup_model()
        self.rollout_buffer = MaskableRolloutBuffer(
            self.n_steps * self.n_envs, self.observation_space, self.action_space, device=self.device,
            gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=1,
        )
        if self.normalize_reward and self.reward_normalizer is None:
            self.reward_normalizer = ReturnNormalizer(self.gamma)

    def _excluded_save_params(self) -> list[str]:
        # keep the zip loadable by plain MaskablePPO.load (no rl.* objects pickled inside)
        return [*super()._excluded_save_params(), "reward_normalizer", "should_stop", "last_rollout"]

    # ------------------------------------------------------------------ rollouts
    def collect_rollouts(  # type: ignore[override]
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: MaskableRolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        if not isinstance(env, RolloutWorkers):
            raise TypeError("ChunkedMaskablePPO needs a RolloutWorkers env")
        if not use_masking:
            raise ValueError("action masking is mandatory for Subway Domino")
        self.policy.set_training_mode(False)
        callback.on_rollout_start()
        total = n_rollout_steps * env.num_envs
        t0 = time.perf_counter()
        segments = env.collect(mlp_params_from_policy(self.policy), total, self.rollout_chunk, self.should_stop)
        collect_s = time.perf_counter() - t0
        if segments is None:
            return False
        self._fill_buffer(rollout_buffer, segments)
        n = sum(len(s) for s in segments)
        self.num_timesteps += n
        self._update_info_buffer([{"episode": {k: e[k] for k in ("r", "l", "t")}} for s in segments for e in s.episodes])
        self._log_rollout(segments, collect_s)
        callback.update_locals(locals())
        if not callback.on_step():
            return False
        callback.on_rollout_end()
        return True

    def _fill_buffer(self, buf: MaskableRolloutBuffer, segments: list[Segment]) -> None:
        buf.reset()
        rows = buf.buffer_size
        if sum(len(s) for s in segments) != rows:
            raise RuntimeError("rollout size does not match the buffer")
        pos = 0
        for s in segments:
            n = len(s)
            sl = slice(pos, pos + n)
            r = s.rewards
            if self.reward_normalizer is not None:
                r = self.reward_normalizer.normalize(s.worker, s.rewards, s.dones)
            adv, ret = segment_gae(r, s.values, s.dones, s.last_value, self.gamma, self.gae_lambda)
            buf.observations[sl, 0] = s.obs
            buf.actions[sl, 0, 0] = s.actions
            buf.rewards[sl, 0] = r
            buf.values[sl, 0] = s.values
            buf.log_probs[sl, 0] = s.log_probs
            buf.advantages[sl, 0] = adv
            buf.returns[sl, 0] = ret
            buf.action_masks[sl, 0] = s.masks
            starts = np.zeros(n, dtype=np.float32)
            starts[1:] = s.dones[:-1]
            buf.episode_starts[sl, 0] = starts
            pos += n
        buf.pos = rows
        buf.full = True

    def _log_rollout(self, segments: list[Segment], collect_s: float) -> None:
        n = sum(len(s) for s in segments)
        acts = np.concatenate([s.actions for s in segments])
        masks = np.concatenate([s.masks for s in segments])
        raw = np.concatenate([s.rewards for s in segments])
        chosen = np.bincount(acts, minlength=C.N_ACTIONS)
        applied = np.sum([s.applied for s in segments], axis=0)
        valid = masks.sum(axis=0)
        busy = sum(s.busy_s for s in segments)
        ticks = sum(s.ticks for s in segments)
        rec = self.logger.record
        rec("rollout/reward_raw_mean", float(raw.mean()))
        rec("rollout/reward_raw_std", float(raw.std()))
        rec("rollout/reward_clip_frac", float(np.mean(np.abs(raw) >= C.REWARD_CLIP - 1e-6)))
        rec("rollout/decisions_per_sim_hour", n / max(ticks * C.DT_S / 3600.0, 1e-9))
        for k, name in enumerate(C.ACTION_NAMES):
            rec(f"actions/rate_{name}", float(chosen[k] / n))
            rec(f"masks/valid_{name}", float(valid[k] / n))
        non_cont = int(chosen[1:].sum())
        rec("actions/noncontinue_rate", non_cont / n)
        rec("masks/applied_rate", float(applied[1:].sum() / non_cont) if non_cont else 1.0)
        rec("masks/mean_valid_actions", float(valid.sum() / n))
        eps = [e["summary"] for s in segments for e in s.episodes if e.get("summary")]
        if eps:
            rec("twin/advantage_mean", float(np.mean([e["twin_advantage"] for e in eps])))
            rec("twin/advantage_rel_mean", float(np.mean([e["twin_advantage_rel"] for e in eps])))
            rec("twin/win_rate", float(np.mean([e["twin_advantage"] > 0 for e in eps])))
            rec("twin/episodes", len(eps))
        rec("perf/env_steps_per_s", n / collect_s)
        rec("perf/worker_busy_frac", busy / (collect_s * len(segments)) if segments else 0.0)
        rec("perf/collect_s", collect_s)
        self.last_rollout = {"steps": n, "collect_s": collect_s, "episodes": eps, "chosen": chosen.tolist()}
