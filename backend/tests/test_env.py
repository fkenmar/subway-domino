"""Gym env, twin reward and training-path tests (ARCHITECTURE §4.1, §4.4-§4.6, §8; marker ``train``).

The toy network (``sim.testing``) keeps most tests fast; the real network is used where the
contract is about real service (twin reward, twin construction). Requires
``requirements-train.txt``.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("sb3_contrib")
pytestmark = pytest.mark.train

import gymnasium  # noqa: E402
import torch  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from rl import vec as V  # noqa: E402
from rl.env import MAX_SCENARIO_TRIES, SubwayDominoEnv  # noqa: E402
from rl.ppo import ChunkedMaskablePPO, segment_gae  # noqa: E402
from rl.rollout import NumpyActorCritic, RolloutWorkers, mlp_params_from_policy, sample_masked  # noqa: E402
from rl.train import POLICY_KWARGS, build_policy  # noqa: E402
from sim import Scenario, eval_scenario  # noqa: E402
from sim import constants as C  # noqa: E402
from sim.testing import toy_network_dict  # noqa: E402

from sim.graph import data_dir  # noqa: E402

needs_real = pytest.mark.skipif(not (data_dir() / "network.json").is_file(), reason="data/network.json not built")


@pytest.fixture(scope="module")
def toy_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    p = tmp_path_factory.mktemp("toy") / "network.json"
    p.write_text(json.dumps(toy_network_dict()))
    return str(p)


def _random_valid(rng: np.random.Generator, mask: np.ndarray) -> int:
    return int(rng.choice(np.flatnonzero(mask)))


def _short(sc: Scenario, episode_s: float) -> Scenario:
    return dataclasses.replace(sc, episode_s=float(episode_s))


# --------------------------------------------------------------------------------------
# Gymnasium / SB3 API conformance
# --------------------------------------------------------------------------------------
def test_gymnasium_env_checker(toy_path: str) -> None:
    from gymnasium.utils.env_checker import check_env
    from stable_baselines3.common.env_checker import check_env as sb3_check_env

    env = SubwayDominoEnv(stage=2, seed=0, graph_path=toy_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(env, skip_render_check=True)
        sb3_check_env(SubwayDominoEnv(stage=1, seed=1, graph_path=toy_path), warn=False)
    assert env.observation_space == gymnasium.spaces.Box(-1.0, 1.0, (C.OBS_DIM,), np.float32)
    assert env.action_space == gymnasium.spaces.Discrete(C.N_ACTIONS)


@needs_real
def test_gymnasium_env_checker_real_network() -> None:
    from gymnasium.utils.env_checker import check_env

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(SubwayDominoEnv(stage=1, seed=3), skip_render_check=True)


def test_reset_is_seed_deterministic(toy_path: str) -> None:
    a, b = SubwayDominoEnv(stage=2, graph_path=toy_path), SubwayDominoEnv(stage=2, graph_path=toy_path)
    oa, ia = a.reset(seed=7)
    ob, ib = b.reset(seed=7)
    np.testing.assert_array_equal(oa, ob)
    assert ia == ib and a.scenario == b.scenario
    assert a.sim is not None and b.sim is not None and a.sim.state_hash() == b.sim.state_hash()


# --------------------------------------------------------------------------------------
# Masks and batch serialization (§4.1)
# --------------------------------------------------------------------------------------
def test_action_masks_consistent_with_obs_and_batch(toy_path: str) -> None:
    env = SubwayDominoEnv(stage=2, seed=0, graph_path=toy_path)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    episodes = steps = 0
    while episodes < 2:
        m = env.action_masks()
        assert m.dtype == np.bool_ and m.shape == (C.N_ACTIONS,) and m[C.A_CONTINUE]
        np.testing.assert_array_equal(m, obs[-C.N_ACTIONS:] > 0.5)  # obs mask bits are the mask
        b = env.sim.pending_decisions()
        np.testing.assert_array_equal(m, b.masks[env._row])  # the current ego's row of the snapshot
        np.testing.assert_array_equal(obs, b.obs[env._row])  # obs never recomputed mid-batch
        obs, r, term, trunc, info = env.step(_random_valid(rng, m))
        steps += 1
        assert term is False and np.isfinite(r) and abs(r) <= C.REWARD_CLIP
        assert obs.dtype == np.float32 and obs.shape == (C.OBS_DIM,) and np.all(np.abs(obs) <= 1.0)
        assert set(info["cost_ai"]) == set(C.COST_KEYS) == set(info["cost_twin"])
        assert info["action"] in C.ACTION_NAMES
        if trunc:
            assert not obs.any()
            np.testing.assert_array_equal(env.action_masks(), np.eye(C.N_ACTIONS, dtype=bool)[0])
            assert env.sim.episode_time_s >= env.scenario.episode_s
            with pytest.raises(RuntimeError):
                env.step(0)
            assert info["episode_summary"]["decisions"] > 0
            episodes += 1
            obs, _ = env.reset()
    assert steps > 50


def test_invalid_action_is_rejected_by_revalidation(toy_path: str) -> None:
    env = SubwayDominoEnv(stage=2, seed=2, graph_path=toy_path)
    env.reset(seed=2)
    for _ in range(500):
        m = env.action_masks()
        if not m.all():
            _, _, _, _, info = env.step(int(np.flatnonzero(~m)[0]))
            assert info["applied"] is False  # the sim re-validates and ignores it
            return
        env.step(0)
    pytest.skip("no masked action found")


# --------------------------------------------------------------------------------------
# Twin-difference reward (§4.4)
# --------------------------------------------------------------------------------------
def _run_continue(env: SubwayDominoEnv, sc: Scenario) -> tuple[list[float], dict[str, Any]]:
    env.reset(seed=0, options={"scenario": sc})
    rewards: list[float] = []
    while True:
        _, r, _, trunc, info = env.step(C.A_CONTINUE)
        rewards.append(r)
        assert info["cost_ai"] == info["cost_twin"]
        if trunc:
            return rewards, info["episode_summary"]


@needs_real
@pytest.mark.parametrize("twin_mode", ["lazy", "copy", "reset"])
def test_twin_reward_exactly_zero_under_all_continue(twin_mode: str) -> None:
    env = SubwayDominoEnv(stage=2, seed=0, twin_mode=twin_mode)  # type: ignore[arg-type]
    sc = _short(eval_scenario(C.EVAL_SEED_BASE + 3, env.graph), 1500.0)  # disruptions start by 900 s
    rewards, summary = _run_continue(env, sc)
    assert len(rewards) > 100 and all(r == 0.0 for r in rewards)
    assert summary["twin_advantage"] == 0.0 and summary["cost_ai"] == summary["cost_twin"] > 0
    if twin_mode != "lazy":
        assert env.twin is not None and env.twin.state_hash() == env.sim.state_hash()
    else:
        assert env.twin is None  # never forked: the AI sim *is* the Normal trajectory


@needs_real
def test_twin_starts_identical_to_ai_sim() -> None:
    """The twin is reset from the same Scenario: identical start state (state_hash)."""
    for mode in ("copy", "reset"):
        env = SubwayDominoEnv(stage=1, seed=5, twin_mode=mode)  # type: ignore[arg-type]
        sc = env.scenario_fn(np.random.default_rng(5), 1, env.graph)
        env._start_episode(sc)
        assert env.twin is not None and env.sim.state_hash() == env.twin.state_hash()


@needs_real
def test_twin_modes_are_bit_identical() -> None:
    """lazy (fork before the first non-CONTINUE action) == copy == independent reset."""
    out: dict[str, tuple[list[float], str]] = {}
    for mode in ("lazy", "copy", "reset"):
        env = SubwayDominoEnv(stage=1, seed=11, twin_mode=mode)  # type: ignore[arg-type]
        sc = _short(env.scenario_fn(np.random.default_rng(11), 1, env.graph), 1800.0)
        env.reset(seed=11, options={"scenario": sc})
        rng = np.random.default_rng(0)
        rewards: list[float] = []
        for i in range(10_000):
            a = 0 if i < 5 else _random_valid(rng, env.action_masks())  # lazy forks mid-episode
            _, r, _, trunc, info = env.step(a)
            rewards.append(r)
            if trunc:
                break
        assert env.twin is not None
        out[mode] = (rewards, env.twin.state_hash())
    assert out["lazy"] == out["copy"] == out["reset"]
    assert any(r != 0.0 for r in out["lazy"][0])


def test_reward_matches_cost_difference(toy_path: str) -> None:
    env = SubwayDominoEnv(stage=2, seed=4, graph_path=toy_path)
    rng = np.random.default_rng(1)
    env.reset(seed=4)
    for _ in range(300):
        _, r, _, trunc, info = env.step(_random_valid(rng, env.action_masks()))
        raw = -C.REWARD_SCALE * (info["cost_ai"]["total"] - info["cost_twin"]["total"])
        assert r == pytest.approx(float(np.clip(raw, -C.REWARD_CLIP, C.REWARD_CLIP)), abs=1e-12)
        if trunc:
            env.reset()


def test_decision_free_episode_resamples_then_returns_dummy(toy_path: str) -> None:
    calls: list[int] = []

    def calm(rng: np.random.Generator, stage: int, graph: Any) -> Scenario:
        calls.append(stage)
        return Scenario(seed=len(calls), profile="late_night", episode_s=C.DT_S)  # one tick, no decision

    env = SubwayDominoEnv(stage=2, seed=0, graph_path=toy_path, scenario_fn=calm)
    obs, info = env.reset(seed=0)
    if not info["dummy"]:
        pytest.skip("the calm toy scenario produced a decision")
    assert len(calls) == MAX_SCENARIO_TRIES and info["resamples"] == MAX_SCENARIO_TRIES
    assert not obs.any() and env.action_masks().tolist() == [True] + [False] * (C.N_ACTIONS - 1)
    obs, r, term, trunc, info = env.step(0)
    assert (r, term, trunc) == (0.0, False, True) and info["episode_summary"]["dummy"]


def test_set_stage_applies_at_next_reset(toy_path: str) -> None:
    env = SubwayDominoEnv(stage=1, seed=0, graph_path=toy_path)
    env.reset(seed=0)
    assert env.scenario is not None and env.scenario.episode_s == C.STAGE1_EPISODE_S
    env.set_stage(2)
    env.reset()
    assert env.scenario is not None and env.scenario.episode_s == C.STAGE2_EPISODE_S
    with pytest.raises(ValueError):
        env.set_stage(3)


# --------------------------------------------------------------------------------------
# Vectorization, auto-scaling and the worker-side policy
# --------------------------------------------------------------------------------------
def test_subproc_spawn_and_obs_masks(toy_path: str) -> None:
    raw = SubprocVecEnv([V.EnvFactory(stage=2, seed=i, graph_path=toy_path) for i in range(2)], start_method="spawn")
    venv = V.ObsMaskVecEnv(raw)
    try:
        venv.seed(0)
        venv.reset()
        rng = np.random.default_rng(0)
        for _ in range(60):
            m = np.stack(venv.env_method("action_masks"))
            np.testing.assert_array_equal(m, np.stack(raw.env_method("action_masks")))
            venv.step(np.array([_random_valid(rng, row) for row in m]))
    finally:
        venv.close()


def test_auto_scaling_rules() -> None:
    assert [V.resolve_n_envs("auto", c) for c in (1, 4, 15, 96, 200)] == [2, 2, 13, 64, 64]
    assert V.resolve_n_steps("auto", 14) == 1088 and V.resolve_n_steps("auto", 64) == 256
    assert V.resolve_n_steps("auto", 2) == 7680 and V.resolve_n_steps(512, 14) == 512
    assert [V.resolve_torch_threads("auto", c) for c in (2, 15, 32, 96)] == [1, 1, 4, 8]
    assert V.resolve_batch_size(14 * 1024) == 2048
    rollout = 13 * V.resolve_n_steps("auto", 13)
    bs = V.resolve_batch_size(rollout)
    assert rollout % bs == 0 and 1024 <= bs <= 4096


def test_numpy_policy_matches_torch() -> None:
    policy = build_policy(seed=0)
    net = NumpyActorCritic(mlp_params_from_policy(policy))
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1, 1, (64, C.OBS_DIM)).astype(np.float32)
    masks = rng.random((64, C.N_ACTIONS)) < 0.5
    masks[:, 0] = True
    with torch.no_grad():
        t_obs = torch.as_tensor(obs)
        feats = policy.extract_features(t_obs, policy.pi_features_extractor)
        logits = policy.action_net(policy.mlp_extractor.forward_actor(feats)).numpy()
        values = policy.predict_values(t_obs).numpy()[:, 0]
        dist = policy.get_distribution(t_obs, action_masks=masks)
    np.testing.assert_allclose(net.logits(obs), logits, atol=1e-5)
    np.testing.assert_allclose(net.value(obs), values, atol=1e-5)
    for i in range(8):
        a, lp = sample_masked(net.logits(obs[i]), masks[i], rng)
        assert masks[i, a]
        with torch.no_grad():
            ref = dist.log_prob(torch.as_tensor(np.array([a] * 64)))[i].item()
        assert lp == pytest.approx(ref, abs=1e-4)
    assert POLICY_KWARGS["net_arch"] == {"pi": [256, 256], "vf": [256, 256]}


def test_segment_gae_matches_sb3_buffer() -> None:
    from gymnasium import spaces
    from stable_baselines3.common.buffers import RolloutBuffer

    rng = np.random.default_rng(0)
    n, gamma, lam = 40, 0.995, 0.95
    rewards, values = rng.normal(size=n).astype(np.float32), rng.normal(size=n).astype(np.float32)
    dones = np.zeros(n, dtype=bool)
    dones[[9, 25]] = True
    last_v = 0.7
    buf = RolloutBuffer(n, spaces.Box(-1, 1, (1,)), spaces.Discrete(2), gamma=gamma, gae_lambda=lam, n_envs=1)
    starts = np.concatenate([[1.0], dones[:-1].astype(np.float32)])
    for t in range(n):
        buf.add(np.zeros((1, 1)), np.zeros((1, 1)), rewards[t:t + 1], starts[t:t + 1],
                torch.tensor([values[t]]), torch.zeros(1))
    buf.compute_returns_and_advantage(torch.tensor([last_v]), np.array([False]))
    adv, ret = segment_gae(rewards, values, dones, last_v, gamma, lam)
    np.testing.assert_allclose(adv, buf.advantages[:, 0], atol=1e-5)
    np.testing.assert_allclose(ret, buf.returns[:, 0], atol=1e-5)


# --------------------------------------------------------------------------------------
# MaskablePPO smoke runs (2,000 steps)
# --------------------------------------------------------------------------------------
def test_maskableppo_smoke_run(toy_path: str) -> None:
    """Plain sb3-contrib MaskablePPO on the env (DummyVecEnv): learns 2,000 steps, valid actions."""
    venv = DummyVecEnv([V.EnvFactory(stage=2, seed=i, graph_path=toy_path) for i in range(2)])
    model = MaskablePPO("MlpPolicy", venv, n_steps=512, batch_size=256, n_epochs=2, gamma=0.995,
                        policy_kwargs=POLICY_KWARGS, seed=0, device="cpu", verbose=0)
    model.learn(2_000)
    assert model.num_timesteps >= 2_000
    obs = venv.reset()
    masks = np.stack(venv.env_method("action_masks"))
    actions, _ = model.predict(obs, action_masks=masks, deterministic=True)
    assert all(masks[i, a] for i, a in enumerate(actions))
    venv.close()


def test_chunked_maskableppo_smoke_run(toy_path: str, tmp_path: Path) -> None:
    """The rl.train path: RolloutWorkers (spawn) + ChunkedMaskablePPO; zip loads in plain MaskablePPO."""
    workers = RolloutWorkers(2, V.EnvFactory(stage=2, graph_path=toy_path, monitor=False), seed=0)
    try:
        model = ChunkedMaskablePPO("MlpPolicy", workers, n_steps=512, batch_size=256, n_epochs=2, gamma=0.995,
                                   policy_kwargs=POLICY_KWARGS, seed=0, device="cpu", verbose=0, rollout_chunk=32)
        model.learn(2_000)
        assert model.num_timesteps >= 2_000 and model.num_timesteps % 1024 == 0
        assert model.reward_normalizer is not None and model.reward_normalizer.count > 1_000
        results = workers.run_episodes(mlp_params_from_policy(model.policy),
                                       [Scenario(seed=1, episode_s=600.0)] * 3)
        assert results[0] == results[1] == results[2] and results[0]["decisions"] >= 0
        path = tmp_path / "m.zip"
        model.save(str(path))
    finally:
        workers.close()
    plain = MaskablePPO.load(str(path), device="cpu")
    obs = np.zeros((1, C.OBS_DIM), dtype=np.float32)
    a, _ = plain.predict(obs, action_masks=np.eye(C.N_ACTIONS, dtype=bool)[:1], deterministic=True)
    assert int(a[0]) == 0
