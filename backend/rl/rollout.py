"""Chunked rollout workers with worker-side inference (throughput: ARCHITECTURE §4.6, §4.10).

Why not SB3's per-step ``SubprocVecEnv`` loop? One ``env.step()`` here means "advance the AI
sim and its Normal twin to the next decision", so step durations are heavy-tailed: mean about
2 ms, p99 about 16 ms, and about 65 ms for a reset (scenario build and warm-up). A
synchronous vectorized step waits for the slowest of N workers. Measured on 14 workers, that
left 16 % parallel efficiency (about 1,000 env-steps/s), and it gets worse as N grows, which is
the Colab case with 64 workers.

Here every worker owns one :class:`~rl.env.SubwayDominoEnv` and a NumPy copy of the current
actor-critic (:class:`NumpyActorCritic`, the exact MlpPolicy forward pass). Per PPO rollout
the learner broadcasts the weights once. It then hands out chunks of ``chunk`` steps
dynamically until the rollout holds exactly ``total`` transitions. Each worker samples
actions, values and log-probabilities locally, so the only barrier is the end of the rollout,
and faster cores (performance cores, or less loaded ones) simply collect more chunks. Each
worker's chunks are contiguous in time, so every worker contributes one trajectory segment
per rollout, bootstrapped with the value of its next observation.

Workers are started with ``spawn`` (spawn-safe, inherited single-thread BLAS settings) and
ignore SIGINT, so a Ctrl+C reaches only the trainer, which checkpoints and then closes them.
"""

from __future__ import annotations

import multiprocessing as mp
import signal
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from typing import Any, Callable, Sequence

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices

from rl.vec import EnvFactory, limit_worker_threads
from sim import constants as C

Layer = tuple[np.ndarray, np.ndarray]  # (W[in, out], b[out]), float32


@dataclass(frozen=True)
class MlpParams:
    """Weights of an SB3 ``MlpPolicy`` actor-critic with separate pi / vf MLPs and Tanh."""

    pi: tuple[Layer, ...]
    vf: tuple[Layer, ...]
    action: Layer
    value: Layer


def mlp_params_from_policy(policy: Any) -> MlpParams:
    """Extract :class:`MlpParams` from a MaskableActorCriticPolicy (validates the architecture)."""
    import torch.nn as nn  # rl-only dependency
    from stable_baselines3.common.torch_layers import FlattenExtractor

    if not isinstance(policy.features_extractor, FlattenExtractor):
        raise TypeError("worker-side inference supports FlattenExtractor policies only")

    def linears(seq: nn.Module) -> tuple[Layer, ...]:
        out: list[Layer] = []
        for m in seq.modules():
            if m is seq or isinstance(m, nn.Sequential):
                continue
            if isinstance(m, nn.Linear):
                out.append(_layer(m))
            elif not isinstance(m, nn.Tanh):
                raise TypeError(f"unsupported layer {type(m).__name__} (expected Linear/Tanh)")
        return tuple(out)

    return MlpParams(
        pi=linears(policy.mlp_extractor.policy_net),
        vf=linears(policy.mlp_extractor.value_net),
        action=_layer(policy.action_net),
        value=_layer(policy.value_net),
    )


def _layer(m: Any) -> Layer:
    w = m.weight.detach().cpu().numpy().astype(np.float32).T.copy()
    b = m.bias.detach().cpu().numpy().astype(np.float32).copy()
    return w, b


class NumpyActorCritic:
    """NumPy forward pass of the MlpPolicy: ``obs -> (logits, value)``."""

    def __init__(self, params: MlpParams) -> None:
        self.p = params

    @staticmethod
    def _mlp(x: np.ndarray, layers: Sequence[Layer]) -> np.ndarray:
        for w, b in layers:
            x = np.tanh(x @ w + b)
        return x

    def logits(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        w, b = self.p.action
        return self._mlp(x, self.p.pi) @ w + b

    def value(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        w, b = self.p.value
        return (self._mlp(x, self.p.vf) @ w + b)[..., 0]

    def act_deterministic(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Masked argmax (the ONNX/server decision rule)."""
        lg = np.where(np.asarray(masks, dtype=bool), self.logits(obs).astype(np.float64), -np.inf)
        return np.argmax(lg, axis=-1).astype(np.int64)


def sample_masked(logits: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[int, float]:
    """Sample from the masked categorical; returns ``(action, log_prob)`` (MaskableCategorical)."""
    z = np.where(mask, logits.astype(np.float64), -np.inf)
    z -= z.max()
    p = np.exp(z)
    s = p.sum()
    a = int(np.searchsorted(np.cumsum(p), rng.random() * s, side="right"))
    a = min(a, C.N_ACTIONS - 1)
    if not mask[a]:  # numerical edge case at the CDF boundary
        a = int(np.flatnonzero(mask)[-1])
    return a, float(z[a] - np.log(s))


# --------------------------------------------------------------------------------------
# Worker process
# --------------------------------------------------------------------------------------
@dataclass
class Chunk:
    """``n`` consecutive transitions of one worker (rewards are raw env rewards)."""

    obs: np.ndarray  # float32[n, OBS_DIM]
    actions: np.ndarray  # int64[n]
    log_probs: np.ndarray  # float32[n]
    values: np.ndarray  # float32[n]
    masks: np.ndarray  # bool[n, N_ACTIONS]
    rewards: np.ndarray  # float32[n]
    dones: np.ndarray  # bool[n]: the step ended its episode (time horizon reached)
    last_value: float  # V(next observation) for bootstrapping the segment
    applied: np.ndarray  # int64[N_ACTIONS]: chosen actions the sim accepted on re-validation
    ticks: int
    busy_s: float
    episodes: list[dict[str, Any]] = field(default_factory=list)


def _collect(env: Any, net: NumpyActorCritic, n: int, rng: np.random.Generator, state: dict[str, Any]) -> Chunk:
    obs_b = np.empty((n, C.OBS_DIM), dtype=np.float32)
    act_b = np.empty(n, dtype=np.int64)
    logp_b = np.empty(n, dtype=np.float32)
    val_b = np.empty(n, dtype=np.float32)
    mask_b = np.empty((n, C.N_ACTIONS), dtype=bool)
    rew_b = np.empty(n, dtype=np.float32)
    done_b = np.zeros(n, dtype=bool)
    applied = np.zeros(C.N_ACTIONS, dtype=np.int64)
    episodes: list[dict[str, Any]] = []
    ticks = 0
    t0 = time.perf_counter()
    obs = state["obs"]
    for t in range(n):
        mask = env.action_masks()
        a, logp = sample_masked(net.logits(obs), mask, rng)
        obs_b[t] = obs
        mask_b[t] = mask
        act_b[t] = a
        logp_b[t] = logp
        val_b[t] = net.value(obs)
        obs, r, terminated, truncated, info = env.step(a)
        rew_b[t] = r
        ticks += int(info["ticks"])
        applied[a] += bool(info["applied"])
        state["ep_r"] += float(r)
        state["ep_l"] += 1
        if terminated or truncated:
            done_b[t] = True
            episodes.append({"r": state["ep_r"], "l": state["ep_l"], "t": time.time() - state["ep_t0"],
                             "summary": info.get("episode_summary", {})})
            obs, _ = env.reset()
            state.update(ep_r=0.0, ep_l=0, ep_t0=time.time())
    state["obs"] = obs
    return Chunk(obs_b, act_b, logp_b, val_b, mask_b, rew_b, done_b, float(net.value(obs)), applied, ticks,
                 time.perf_counter() - t0, episodes)


def _worker_main(conn: Connection, factory: EnvFactory, seed: int, worker_id: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the trainer handles Ctrl+C
    env = None
    try:
        env = factory()
        rng = np.random.default_rng([int(seed), int(worker_id), 0x5D])
        obs, _ = env.reset(seed=int(seed) + int(worker_id))
        state: dict[str, Any] = {"obs": obs, "ep_r": 0.0, "ep_l": 0, "ep_t0": time.time()}
        net: NumpyActorCritic | None = None
        while True:
            cmd, data = conn.recv()
            if cmd == "collect":
                if net is None:
                    raise RuntimeError("collect before weights")
                conn.send(("chunk", _collect(env, net, int(data), rng, state)))
            elif cmd == "weights":
                net = NumpyActorCritic(data)
            elif cmd == "evaluate":
                from rl.evaluate import run_episode  # lazy: only validation needs it

                eval_params, scenario = data
                act = None if eval_params is None else NumpyActorCritic(eval_params).act_deterministic
                conn.send(("result", run_episode(env.graph, scenario, act)))
            elif cmd == "obs":
                conn.send(("result", state["obs"]))
            elif cmd == "env_method":
                name, args, kwargs = data
                conn.send(("result", getattr(env, name)(*args, **kwargs)))
            elif cmd == "get_attr":
                conn.send(("result", getattr(env, data)))
            elif cmd == "set_attr":
                setattr(env, data[0], data[1])
                conn.send(("result", None))
            elif cmd == "has_attr":
                conn.send(("result", hasattr(env, data)))
            elif cmd == "close":
                break
            else:
                raise ValueError(f"unknown command {cmd!r}")
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    except Exception:  # report to the learner, which raises it
        try:
            conn.send(("error", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        conn.close()


# --------------------------------------------------------------------------------------
# Learner-side manager
# --------------------------------------------------------------------------------------
@dataclass
class Segment:
    """One worker's contribution to a rollout: its chunks concatenated in time order."""

    worker: int
    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    masks: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    last_value: float
    applied: np.ndarray
    ticks: int
    busy_s: float
    episodes: list[dict[str, Any]]

    def __len__(self) -> int:
        return int(self.actions.shape[0])


def _concat(worker: int, chunks: list[Chunk]) -> Segment:
    cat = np.concatenate
    return Segment(
        worker=worker,
        obs=cat([c.obs for c in chunks]),
        actions=cat([c.actions for c in chunks]),
        log_probs=cat([c.log_probs for c in chunks]),
        values=cat([c.values for c in chunks]),
        masks=cat([c.masks for c in chunks]),
        rewards=cat([c.rewards for c in chunks]),
        dones=cat([c.dones for c in chunks]),
        last_value=chunks[-1].last_value,
        applied=np.sum([c.applied for c in chunks], axis=0),
        ticks=sum(c.ticks for c in chunks),
        busy_s=sum(c.busy_s for c in chunks),
        episodes=[e for c in chunks for e in c.episodes],
    )


class RolloutWorkers(VecEnv):
    """``n_workers`` env processes collecting chunked on-policy rollouts (a VecEnv for SB3).

    Only the parts of the VecEnv API SB3 needs around ``collect_rollouts`` are implemented;
    per-step ``step()`` is intentionally unsupported (use :meth:`collect`).
    """

    def __init__(self, n_workers: int, factory: EnvFactory, *, seed: int = 0, start_method: str = "spawn") -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        if factory.monitor:
            raise ValueError("RolloutWorkers does its own episode accounting: use EnvFactory(monitor=False)")
        limit_worker_threads()
        ctx = mp.get_context(start_method)
        self._conns: list[Connection] = []
        self._procs: list[Any] = []
        self._closed = False
        for i in range(n_workers):
            parent, child = ctx.Pipe(duplex=True)
            proc = ctx.Process(target=_worker_main, args=(child, factory, seed, i), daemon=True,
                               name=f"subway-domino-worker-{i}")
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)
        super().__init__(n_workers, spaces.Box(-1.0, 1.0, (C.OBS_DIM,), dtype=np.float32), spaces.Discrete(C.N_ACTIONS))

    # ---------------------------------------------------------------- collection
    def collect(
        self,
        params: MlpParams,
        total: int,
        chunk: int = 64,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[Segment] | None:
        """Exactly ``total`` transitions with the policy ``params``; ``None`` if stopped early."""
        if total < 1 or chunk < 1:
            raise ValueError("total and chunk must be >= 1")
        for c in self._conns:
            c.send(("weights", params))
        assigned = 0
        outstanding: dict[int, int] = {}
        chunks: list[list[Chunk]] = [[] for _ in self._conns]
        stopping = False

        def assign(i: int) -> None:
            nonlocal assigned
            k = min(chunk, total - assigned)
            if k > 0:
                self._conns[i].send(("collect", k))
                assigned += k
                outstanding[i] = k

        for i in range(self.num_envs):
            assign(i)
        index = {id(c): i for i, c in enumerate(self._conns)}
        while outstanding:
            for conn in wait([self._conns[i] for i in outstanding]):
                i = index[id(conn)]
                chunks[i].append(self._recv(i, "chunk"))
                del outstanding[i]
                if should_stop is not None and should_stop():
                    stopping = True
                if not stopping:
                    assign(i)
        if stopping:
            return None
        return [_concat(i, cs) for i, cs in enumerate(chunks) if cs]

    def run_episodes(self, params: MlpParams | None, scenarios: Sequence[Any]) -> list[dict[str, Any]]:
        """Run full episodes in parallel on the (idle) workers: deterministic masked argmax of
        ``params``, or Normal Schedule for ``None``. Results are in ``scenarios`` order."""
        results: list[Any] = [None] * len(scenarios)
        queue = list(enumerate(scenarios))
        busy: dict[int, int] = {}
        index = {id(c): i for i, c in enumerate(self._conns)}
        while queue or busy:
            for i in range(self.num_envs):
                if queue and i not in busy:
                    j, sc = queue.pop(0)
                    self._conns[i].send(("evaluate", (params, sc)))
                    busy[i] = j
            for conn in wait([self._conns[i] for i in busy]):
                i = index[id(conn)]
                results[busy.pop(i)] = self._recv(i, "result")
        return results

    def _recv(self, i: int, expect: str) -> Any:
        try:
            tag, data = self._conns[i].recv()
        except EOFError as exc:
            raise RuntimeError(f"rollout worker {i} died (exit code {self._procs[i].exitcode})") from exc
        if tag == "error":
            raise RuntimeError(f"rollout worker {i} failed:\n{data}")
        if tag != expect:
            raise RuntimeError(f"rollout worker {i}: expected {expect!r}, got {tag!r}")
        return data

    def _call(self, cmd: str, data: Any, indices: VecEnvIndices = None) -> list[Any]:
        idx = list(self._get_indices(indices))
        for i in idx:
            self._conns[i].send((cmd, data))
        return [self._recv(i, "result") for i in idx]

    # ---------------------------------------------------------------- VecEnv API
    def reset(self) -> np.ndarray:
        """The workers' current observations (they keep running episodes across rollouts)."""
        return np.stack(self._call("obs", None)).astype(np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        raise NotImplementedError("RolloutWorkers collects whole chunks: use collect()")

    def step_wait(self) -> Any:
        raise NotImplementedError("RolloutWorkers collects whole chunks: use collect()")

    def env_method(self, method_name: str, *method_args: Any, indices: VecEnvIndices = None, **method_kwargs: Any) -> list[Any]:
        return self._call("env_method", (method_name, method_args, method_kwargs), indices)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        return self._call("get_attr", attr_name, indices)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        self._call("set_attr", (attr_name, value), indices)

    def has_attr(self, attr_name: str) -> bool:
        return all(self._call("has_attr", attr_name))

    def env_is_wrapped(self, wrapper_class: type, indices: VecEnvIndices = None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]

    def get_images(self) -> Sequence[np.ndarray | None]:
        return [None for _ in range(self.num_envs)]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for c in self._conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        for c in self._conns:
            c.close()
