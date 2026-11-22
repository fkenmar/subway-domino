"""WP5b per-train credit assignment through the env and PPO (marker ``train``).

Spec tests 1 (cost vectors sum to ``cost_totals`` for the AI sim and the twin), 2 (all-CONTINUE
gives exactly zero credit in all twin modes), 5 (``--credit serial`` is the segment GAE), the
PPO buffer glue, a toy-network smoke run and the 60k-step real-network smoke run (test 6).
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

pytestmark = pytest.mark.train

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from rl import vec as V  # noqa: E402
from rl.credit import PerTrainCredit, TransitionReturnNormalizer, assemble, per_rollout_batch_size  # noqa: E402
from rl.env import SubwayDominoEnv  # noqa: E402
from rl.ppo import ChunkedMaskablePPO, ReturnNormalizer, segment_gae  # noqa: E402
from rl.rollout import NumpyActorCritic, RolloutWorkers, Segment, mlp_params_from_policy  # noqa: E402
from rl.train import POLICY_KWARGS  # noqa: E402
from sim import constants as C  # noqa: E402
from sim import eval_scenario  # noqa: E402
from sim.graph import data_dir  # noqa: E402
from sim.testing import toy_network_dict  # noqa: E402

needs_real = pytest.mark.skipif(not (data_dir() / "network.json").is_file(), reason="data/network.json not built")


@pytest.fixture(scope="module")
def toy_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    p = tmp_path_factory.mktemp("toy") / "network.json"
    p.write_text(json.dumps(toy_network_dict()))
    return str(p)


def _short(sc: Any, episode_s: float) -> Any:
    return dataclasses.replace(sc, episode_s=float(episode_s))


def _run_stream(env: SubwayDominoEnv, sc: Any, choose: Callable[[int, np.ndarray], int]) -> SimpleNamespace:
    """One episode as a Segment-like stream with the per-step credit fields."""
    obs, _ = env.reset(seed=0, options={"scenario": sc})
    rows: dict[str, list[Any]] = {k: [] for k in ("obs", "actions", "t_s", "step_ticks", "train_ids", "routes",
                                                  "imm", "d_glob", "d_route", "dones")}
    for i in range(100_000):
        a = choose(i, env.action_masks())
        rows["obs"].append(obs)
        rows["actions"].append(a)
        obs, _, _, trunc, info = env.step(a)
        rows["t_s"].append(info["t_s"])
        rows["step_ticks"].append(info["ticks"])
        rows["train_ids"].append(info["train_id"])
        rows["routes"].append(info["route"])
        rows["imm"].append(info["imm_cost"])
        rows["d_glob"].append(info["d_flow_global"])
        rows["d_route"].append(info["d_route"])
        rows["dones"].append(trunc)
        if trunc:
            break
    n = len(rows["actions"])
    out = {k: np.asarray(v) for k, v in rows.items()}
    return SimpleNamespace(worker=0, log_probs=np.zeros(n, np.float32), values=np.zeros(n, np.float32),
                           masks=np.ones((n, C.N_ACTIONS), bool), ep=np.zeros(n, np.int64), **out)


def _check_vectors(sim: Any) -> None:
    ct, cv = sim.cost_totals(), sim.cost_vectors()
    flow_pl, flow_rt = ct["wait"] + ct["crowd"], ct["tunnel"] + ct["hold"]
    assert math.isclose(float(cv["platform"].sum()), flow_pl, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(float(cv["route_train"].sum()), flow_rt, rel_tol=1e-9, abs_tol=1e-9)
    cv["platform"][:] = -1.0  # copies: the sim is untouched
    assert sim.cost_plat.min() >= 0.0


# --------------------------------------------------------------------------------------
# Test 1 and test 2: env-level credit data
# --------------------------------------------------------------------------------------
@needs_real
def test_cost_vectors_sum_to_cost_totals_for_ai_and_twin() -> None:
    env = SubwayDominoEnv(stage=2, seed=0)
    sc = _short(eval_scenario(C.EVAL_SEED_BASE + 3, env.graph), 2400.0)
    rng = np.random.default_rng(0)
    seg = _run_stream(env, sc, lambda i, m: 0 if i < 10 else int(rng.choice(np.flatnonzero(m))))
    assert env.twin is not None and seg.dones[-1]
    assert seg.t_s[-1] + seg.step_ticks[-1] * C.DT_S >= 2400.0 - 1e-6  # the last step reaches the horizon
    for sim in (env.sim, env.twin):
        _check_vectors(sim)
        ct = sim.cost_totals()
        assert ct["wait"] > 0 and ct["tunnel"] + ct["hold"] > 0
    assert np.any(seg.d_route != 0) and np.any(seg.d_glob != 0) and np.any(seg.imm != 0)
    # the reward is still exactly the citywide twin delta, and decomposes into imm + d_flow_global
    env2 = SubwayDominoEnv(stage=2, seed=0)
    env2.reset(seed=0, options={"scenario": sc})
    rng = np.random.default_rng(0)
    for i in range(2000):
        m = env2.action_masks()
        _, r, _, trunc, info = env2.step(0 if i < 10 else int(rng.choice(np.flatnonzero(m))))
        raw = -C.REWARD_SCALE * (info["imm_cost"] + info["d_flow_global"])
        assert r == pytest.approx(float(np.clip(raw, -C.REWARD_CLIP, C.REWARD_CLIP)), rel=1e-9, abs=1e-12)
        assert info["applied"] or info["imm_cost"] == 0.0
        if trunc:
            break


@needs_real
@pytest.mark.parametrize("twin_mode", ["lazy", "copy", "reset"])
def test_all_continue_credit_is_exactly_zero(twin_mode: str) -> None:
    env = SubwayDominoEnv(stage=2, seed=0, twin_mode=twin_mode)  # type: ignore[arg-type]
    sc = _short(eval_scenario(C.EVAL_SEED_BASE + 3, env.graph), 1500.0)  # disruptions start by 900 s
    seg = _run_stream(env, sc, lambda i, m: 0)
    assert len(seg.actions) > 100 and seg.dones[-1]
    assert np.all(seg.d_route == 0) and np.all(seg.d_glob == 0) and np.all(seg.imm == 0)
    tr = PerTrainCredit().process(0, seg)
    assert tr.closed.sum() > 100
    assert np.all(tr.r_scope[tr.closed] == 0) and np.all(tr.r_global[tr.closed] == 0)
    batch = assemble(PerTrainCredit(), [seg], lambda o: np.zeros(len(o), np.float32), 0.95,
                     normalizer=TransitionReturnNormalizer())
    assert len(batch) == tr.closed.sum() and np.all(batch.rewards == 0)
    if twin_mode != "lazy":
        assert env.twin is not None and env.twin.state_hash() == env.sim.state_hash()
        assert np.array_equal(env.twin.cost_plat, env.sim.cost_plat)


@needs_real
def test_calm_anchor_episodes() -> None:
    from sim import load_graph, sample_training_scenario

    g = load_graph()
    a = [sample_training_scenario(np.random.default_rng(s), 2, g) for s in range(40)]
    b = [sample_training_scenario(np.random.default_rng(s), 2, g, calm_frac=0.0) for s in range(40)]
    assert a == b and all(len(sc.disruptions) > 0 for sc in a)  # calm_frac 0: the pre-WP5b draws
    calm = [sample_training_scenario(np.random.default_rng(s), 2, g, calm_frac=1.0) for s in range(40)]
    assert all(len(sc.disruptions) == 0 and sc.headway_jitter >= 0 for sc in calm)
    assert len({sc.headway_jitter for sc in calm}) > 30  # jitter stays on (randomized)
    some = [sample_training_scenario(np.random.default_rng(s), 2, g, calm_frac=0.12) for s in range(400)]
    frac = np.mean([len(sc.disruptions) == 0 for sc in some])
    assert 0.06 < frac < 0.2
    s1 = [sample_training_scenario(np.random.default_rng(s), 1, g, calm_frac=1.0) for s in range(10)]
    assert all(len(sc.disruptions) > 0 for sc in s1)  # Stage 2 only


# --------------------------------------------------------------------------------------
# Test 5 and the PPO glue (no workers)
# --------------------------------------------------------------------------------------
class _Spaces(gym.Env[np.ndarray, np.int64]):
    observation_space = spaces.Box(-1.0, 1.0, (C.OBS_DIM,), dtype=np.float32)
    action_space = spaces.Discrete(C.N_ACTIONS)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        return np.zeros(C.OBS_DIM, np.float32), {}

    def step(self, action: Any) -> Any:
        return np.zeros(C.OBS_DIM, np.float32), 0.0, False, False, {}


def _bare_model(credit: str, n_rows: int, **kw: Any) -> ChunkedMaskablePPO:
    model = ChunkedMaskablePPO("MlpPolicy", DummyVecEnv([_Spaces]), n_steps=n_rows, batch_size=64, n_epochs=2,
                               gamma=0.995, policy_kwargs=POLICY_KWARGS, seed=0, device="cpu", verbose=0,
                               credit=credit, **kw)
    model.set_logger(configure(None, []))
    return model


def _synthetic_segments(rng: np.random.Generator, n_workers: int, n: int, n_routes: int = 3) -> list[Segment]:
    segs = []
    for w in range(n_workers):
        dones = np.zeros(n, bool)
        dones[rng.choice(n, 2, replace=False)] = True
        ep = np.concatenate([[0], np.cumsum(dones[:-1])]).astype(np.int64)
        ticks = rng.integers(0, 3, n).astype(np.int32)
        t_s = np.zeros(n)
        for t in range(1, n):
            t_s[t] = 0.0 if dones[t - 1] else t_s[t - 1] + ticks[t - 1] * C.DT_S
        segs.append(Segment(
            worker=w, obs=rng.uniform(-1, 1, (n, C.OBS_DIM)).astype(np.float32),
            actions=rng.integers(0, C.N_ACTIONS, n), log_probs=rng.uniform(-2, 0, n).astype(np.float32),
            values=rng.normal(size=n).astype(np.float32), masks=np.ones((n, C.N_ACTIONS), bool),
            rewards=rng.normal(size=n).astype(np.float32), dones=dones, last_value=float(rng.normal()),
            applied=np.zeros(C.N_ACTIONS, np.int64), ticks=int(ticks.sum()), busy_s=0.0, episodes=[],
            t_s=t_s, step_ticks=ticks, train_ids=rng.integers(0, 12, n), routes=rng.integers(0, n_routes, n),
            imm=np.where(rng.random(n) < 0.2, rng.exponential(100.0, n), 0.0),
            d_glob=np.where(ticks > 0, rng.normal(0, 500, n), 0.0),
            d_route=np.where(ticks[:, None] > 0, rng.normal(0, 200, (n, n_routes)), 0.0).astype(np.float32), ep=ep,
        ))
    return segs


def test_serial_credit_is_the_segment_gae() -> None:
    """Test 5: ``--credit serial`` fills the buffer with exactly the WP5 segment GAE (regression)."""
    rng = np.random.default_rng(7)
    segs = _synthetic_segments(rng, 3, 60)
    model = _bare_model("serial", 180)
    assert isinstance(model.reward_normalizer, ReturnNormalizer) and model.train_credit is None
    assert model.target_kl is None
    ref = ReturnNormalizer(model.gamma)
    for _ in range(2):  # two rollouts: the per-worker running return carries over
        model._fill_buffer(model.rollout_buffer, segs)
        pos = 0
        for s in segs:
            r = ref.normalize(s.worker, s.rewards, s.dones)
            adv, ret = segment_gae(r, s.values, s.dones, s.last_value, model.gamma, model.gae_lambda)
            sl = slice(pos, pos + len(s))
            assert np.array_equal(model.rollout_buffer.rewards[sl, 0], r)
            assert np.array_equal(model.rollout_buffer.advantages[sl, 0], adv)
            assert np.array_equal(model.rollout_buffer.returns[sl, 0], ret)
            assert np.array_equal(model.rollout_buffer.values[sl, 0], s.values)
            pos += len(s)
        assert ref.state_dict() == model.reward_normalizer.state_dict() and "kind" not in ref.state_dict()


def test_per_train_buffer_matches_assemble_and_trains() -> None:
    rng = np.random.default_rng(8)
    rollouts = [_synthetic_segments(rng, 2, 300) for _ in range(3)]
    model = _bare_model("per-train", 600, target_kl=0.02)
    assert isinstance(model.reward_normalizer, TransitionReturnNormalizer) and model.train_credit is not None
    ref_credit, ref_norm = PerTrainCredit(model.gamma_min), TransitionReturnNormalizer()
    for segs in rollouts:
        params = mlp_params_from_policy(model.policy)
        stats = model._fill_buffer_per_train(segs, params)
        ref = assemble(ref_credit, segs, NumpyActorCritic(params).value, model.gae_lambda, normalizer=ref_norm)
        buf = model.rollout_buffer
        n = len(ref)
        assert buf.buffer_size == n == stats["closed"] and buf.full and buf.pos == n
        assert model.batch_size == per_rollout_batch_size(n)
        for name, col in (("advantages", ref.advantages), ("returns", ref.returns), ("values", ref.values),
                          ("rewards", ref.rewards), ("log_probs", ref.log_probs)):
            assert np.array_equal(getattr(buf, name)[:, 0], col), name
        assert np.array_equal(buf.observations[:, 0], ref.obs) and np.array_equal(buf.action_masks[:, 0], ref.masks)
        model.train()  # SB3's update runs on the variable-size buffer
    assert stats["carried_closed"] > 0 and model.reward_normalizer.state_dict() == ref_norm.state_dict()


def test_per_train_save_excludes_carry_and_loads(tmp_path: Path) -> None:
    model = _bare_model("per-train", 128)
    path = tmp_path / "m.zip"
    model.save(str(path))
    plain = MaskablePPO.load(str(path), device="cpu")
    assert getattr(plain, "credit") == "per-train" and not hasattr(plain, "train_credit")
    again = ChunkedMaskablePPO.load(str(path), env=DummyVecEnv([_Spaces]), device="cpu")
    assert again.credit == "per-train" and again.train_credit is not None and again.train_credit.n_open() == 0
    assert isinstance(again.reward_normalizer, TransitionReturnNormalizer)


# --------------------------------------------------------------------------------------
# Smoke runs
# --------------------------------------------------------------------------------------
def test_per_train_chunked_smoke_run(toy_path: str, tmp_path: Path) -> None:
    """Toy network: RolloutWorkers + per-train credit learn 3 rollouts; the zip loads in plain MaskablePPO."""
    workers = RolloutWorkers(2, V.EnvFactory(stage=2, graph_path=toy_path, monitor=False, credit_info=True,
                                             calm_frac=0.12), seed=0)
    try:
        model = ChunkedMaskablePPO("MlpPolicy", workers, n_steps=512, batch_size=256, n_epochs=2, gamma=0.995,
                                   policy_kwargs=POLICY_KWARGS, seed=0, device="cpu", verbose=0, rollout_chunk=32,
                                   credit="per-train", target_kl=0.02)
        model.set_logger(configure(None, []))
        model.learn(3_072)
        assert model.num_timesteps == 3_072  # env steps, not closed transitions
        assert isinstance(model.reward_normalizer, TransitionReturnNormalizer) and model.reward_normalizer.count > 1
        st = model.last_rollout["credit"]
        assert 0 < st["closed"] <= 1024 + st["carried_closed"] and st["mean_interval_s"] > 0
        path = tmp_path / "m.zip"
        model.save(str(path))
    finally:
        workers.close()
    plain = MaskablePPO.load(str(path), device="cpu")
    obs = np.zeros((1, C.OBS_DIM), dtype=np.float32)
    a, _ = plain.predict(obs, action_masks=np.eye(C.N_ACTIONS, dtype=bool)[:1], deterministic=True)
    assert int(a[0]) == 0


@needs_real
@pytest.mark.slow
def test_per_train_60k_smoke_real_network(tmp_path: Path) -> None:
    """Test 6: 60k steps per-train on 2 workers; the checkpoint loads and exports with parity."""
    from rl.export_onnx import export
    from rl.train import main

    run = tmp_path / "pertrain"
    rc = main(["--run-dir", str(run), "--n-envs", "2", "--total-steps", "60000", "--stage", "2",
               "--credit", "per-train", "--checkpoint-every", "30000", "--validate-every", "1000000000",
               "--validation-seeds", "2", "--print-every", "15000", "--torch-threads", "2"])
    assert rc == 0
    state = json.loads((run / "train_state.json").read_text())
    assert state["steps_total"] >= 60_000 and state["finished"]
    assert state["credit"]["mode"] == "per-train" and state["reward_normalizer"]["kind"] == "per_train"
    model = MaskablePPO.load(str(run / "latest.zip"), device="cpu")
    assert model.num_timesteps == state["steps_total"]
    res = export(run / "latest.zip", tmp_path / "out" / "nyc_subway_rcc_ppo.onnx", n_parity=300)
    assert res["parity"]["passed"] and res["parity"]["argmax_agreement"] == 1.0
    card = json.loads((tmp_path / "out" / "nyc_subway_rcc_ppo.json").read_text())
    assert card["hyperparameters"]["credit"] == "per-train"
