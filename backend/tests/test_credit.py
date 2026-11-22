"""Per-train semi-MDP credit assignment (WP5b, ``rl.credit``): synthetic streams, no sim, no torch.

Spec tests 3 (hand-computed 2-train stream with a terminal and a carry) and 4 (splitting a
worker stream into rollouts), plus the carry limit, the normalizer and the batch-size rule.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from rl.credit import (
    PerTrainCredit,
    TransitionReturnNormalizer,
    assemble,
    discounted_returns,
    per_rollout_batch_size,
    per_train_gae,
)
from sim import constants as C

G = 0.96
S = C.REWARD_SCALE


def _seg(worker: int = 0, **cols: Any) -> SimpleNamespace:
    """A ``rl.rollout.Segment`` look-alike with the per-step credit fields."""
    n = len(cols["train_ids"])
    obs = np.asarray(cols.pop("obs"), dtype=np.float32)
    d = dict(
        worker=worker, obs=obs, actions=np.zeros(n, np.int64), log_probs=np.full(n, -0.5, np.float32),
        masks=np.ones((n, C.N_ACTIONS), bool), values=np.asarray(cols.pop("values"), np.float32),
        dones=np.asarray(cols.pop("dones"), bool), t_s=np.asarray(cols.pop("t_s"), np.float64),
        step_ticks=np.asarray(cols.pop("ticks"), np.int32), train_ids=np.asarray(cols.pop("train_ids"), np.int64),
        routes=np.asarray(cols.pop("routes"), np.int16), imm=np.asarray(cols.pop("imm"), np.float64),
        d_glob=np.asarray(cols.pop("d_glob"), np.float64), d_route=np.asarray(cols.pop("d_route"), np.float32),
        ep=np.asarray(cols.pop("ep"), np.int64),
    )
    assert not cols, cols
    return SimpleNamespace(**d)


def _slice(seg: SimpleNamespace, a: int, b: int) -> SimpleNamespace:
    out = {k: (v[a:b] if isinstance(v, np.ndarray) else v) for k, v in vars(seg).items()}
    return SimpleNamespace(**out)


def _v1(obs: np.ndarray) -> np.ndarray:
    return obs[:, 0].astype(np.float32)


def _v2(obs: np.ndarray) -> np.ndarray:
    return (2.0 * obs[:, 0]).astype(np.float32)


# --------------------------------------------------------------------------------------
# Test 3: hand-computed reference
# --------------------------------------------------------------------------------------
def _hand_stream() -> SimpleNamespace:
    # m:      0     1     2     3     4     5 (new episode)
    # train   A     B     A     B     A     B          A = slot 5 (route 0), B = slot 9 (route 1)
    # t_s     0     0     30    90    90    0
    # ticks   0     3     6     0     12    2
    obs = np.array([[0.1, 0], [0.2, 0], [0.3, 0], [0.4, 0], [0.5, 0], [0.6, 0]])
    return _seg(
        obs=obs, values=np.zeros(6), dones=[0, 0, 0, 0, 1, 0], t_s=[0, 0, 30, 90, 90, 0], ticks=[0, 3, 6, 0, 12, 2],
        train_ids=[5, 9, 5, 9, 5, 9], routes=[0, 1, 0, 1, 0, 1], imm=[2.0, 0, 0, 1.0, 0, 0.5],
        d_glob=[0, 25, 10, 0, 7, 4], d_route=[[0, 0], [10, 20], [4, 8], [0, 0], [6, 2], [1, 3]], ep=[0, 0, 0, 0, 0, 1],
    )


def test_hand_computed_two_train_stream_with_terminal_and_carry() -> None:
    lam = 0.9
    full = _hand_stream()
    r1, r2 = _slice(full, 0, 3), _slice(full, 3, 6)
    r1.values = _v1(r1.obs)  # rollout 1 critic
    r2.values = _v2(r2.obs)  # rollout 2 critic (an update happened in between)
    credit = PerTrainCredit(gamma_min=G)

    b1 = assemble(credit, [r1], _v1, lam, normalizer=None, reward_clip=1e9)
    assert b1.pos.tolist() == [0] and b1.stats["carried"] == 2  # T1 (B) and T2 (A) are open
    R0 = -S * (2.0 + 10 * G ** 0.25)  # m1's ticks are centred at t = 15 s
    d0 = R0 + G ** 0.5 * 0.3 - 0.1  # next own decision of A at t = 30 (m2, still open): truncated
    np.testing.assert_allclose(b1.rewards, [R0], rtol=1e-6)
    np.testing.assert_allclose(b1.deltas, [d0], rtol=1e-6)
    np.testing.assert_allclose(b1.advantages, [d0], rtol=1e-6)
    np.testing.assert_allclose(b1.returns, b1.advantages + np.float32(0.1), rtol=1e-6)

    b2 = assemble(credit, [r2], _v2, lam, normalizer=None, reward_clip=1e9)
    by = {int(p): i for i, p in enumerate(b2.pos)}
    assert sorted(by) == [1, 2, 3, 4] and b2.stats["carried"] == 1  # T5 (B, episode 1) stays open
    assert b2.stats["carried_closed"] == 2 and b2.stats["terminal"] == 2
    R1 = -S * (20 * G ** 0.25 + 8 * G)  # m1 at t_mid 15, m2 at t_mid 60
    R2 = -S * (4 * G ** 0.5)  # m2 at t_mid 60 seen from t = 30
    R3 = -S * (1.0 + 2 * G)  # imm of B at m3 + m4 at t_mid 150 seen from t = 90; terminal
    R4 = -S * (6 * G)
    A3 = R3 - 0.8
    A4 = R4 - 1.0
    d1 = R1 + G ** 1.5 * 0.8 - 0.4  # V(o_1) recomputed with the rollout-2 critic: 2 * 0.2
    d2 = R2 + G * 1.0 - 0.6
    A1 = d1 + G ** 1.5 * lam * A3
    A2 = d2 + G * lam * A4
    for p, R, d, A, V in ((1, R1, d1, A1, 0.4), (2, R2, d2, A2, 0.6), (3, R3, A3, A3, 0.8), (4, R4, A4, A4, 1.0)):
        i = by[p]
        np.testing.assert_allclose(b2.rewards[i], R, rtol=1e-6)
        np.testing.assert_allclose(b2.deltas[i], d, rtol=1e-6)
        np.testing.assert_allclose(b2.advantages[i], A, rtol=1e-5)
        np.testing.assert_allclose(b2.values[i], V, rtol=1e-6)
        np.testing.assert_allclose(b2.returns[i], A + V, rtol=1e-5, atol=1e-6)
    # global scope: same intervals, citywide deltas
    g_credit = PerTrainCredit(gamma_min=G)
    assemble(g_credit, [_slice(full, 0, 3)], _v1, lam, scope="global", reward_clip=1e9)
    g2 = assemble(g_credit, [_slice(full, 3, 6)], _v1, lam, scope="global", reward_clip=1e9)
    i = {int(p): k for k, p in enumerate(g2.pos)}[1]
    np.testing.assert_allclose(g2.rewards[i], -S * (25 * G ** 0.25 + 10 * G), rtol=1e-6)


def test_all_zero_deltas_give_exactly_zero_rewards() -> None:
    seg = _hand_stream()
    seg.imm[:] = 0.0
    seg.d_glob[:] = 0.0
    seg.d_route[:] = 0.0
    b = assemble(PerTrainCredit(), [seg], _v1, 0.95, normalizer=TransitionReturnNormalizer())
    assert len(b) == 5 and np.all(b.rewards == 0.0)


# --------------------------------------------------------------------------------------
# Test 4: rollout splits
# --------------------------------------------------------------------------------------
def _random_stream(rng: np.random.Generator, n_trains: int = 9, n_routes: int = 4, n_eps: int = 3) -> SimpleNamespace:
    cols: dict[str, list[Any]] = {k: [] for k in ("t_s", "ticks", "train_ids", "routes", "imm", "d_glob", "d_route",
                                                  "dones", "ep")}
    route_of = rng.integers(0, n_routes, size=n_trains)
    for ep in range(n_eps):
        t = 0.0
        steps = int(rng.integers(30, 70))
        k = 0
        while k < steps:
            batch = rng.choice(n_trains, size=int(rng.integers(1, 4)), replace=False)
            for b, tid in enumerate(batch):
                last = b == len(batch) - 1
                ticks = int(rng.integers(1, 7)) if last else 0
                cols["t_s"].append(t)
                cols["ticks"].append(ticks)
                cols["train_ids"].append(int(tid))
                cols["routes"].append(int(route_of[tid]))
                cols["imm"].append(float(rng.exponential(50.0)) if rng.random() < 0.3 else 0.0)
                cols["d_glob"].append(float(rng.normal(0, 300)) if ticks else 0.0)
                cols["d_route"].append(rng.normal(0, 100, n_routes) if ticks else np.zeros(n_routes))
                cols["dones"].append(False)
                cols["ep"].append(ep)
                t += ticks * C.DT_S
                k += 1
        cols["dones"][-1] = True
    n = len(cols["t_s"])
    obs = rng.normal(size=(n, 3))
    return _seg(obs=obs, values=_v1(obs.astype(np.float32)), **cols)


def _reference(seg: SimpleNamespace, lam: float, cut_of: dict[int, int] | None) -> dict[int, tuple[float, float]]:
    """Unsplit transitions; GAE truncated where the next transition closes in a later rollout."""
    tr = PerTrainCredit(gamma_min=G).process(0, seg)
    r = tr.r_scope
    _, _, delta = per_train_gae(r, tr, 0.0)
    adv: dict[int, float] = {}
    for j in range(len(tr) - 1, -1, -1):
        if not tr.closed[j]:
            continue
        a = float(delta[j])
        q = int(tr.nxt[j])
        if not tr.terminal[j] and tr.closed[q] and (cut_of is None or cut_of[int(tr.pos[q])] == cut_of[int(tr.pos[j])]):
            a += float(tr.gamma_dt[j]) * lam * adv[int(tr.pos[q])]
        adv[int(tr.pos[j])] = a
    return {p: (float(delta[i]), adv[p]) for i, p in enumerate(tr.pos.tolist()) if tr.closed[i]}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n_parts", [1, 2, 3])
def test_split_rollouts_identical_deltas_and_truncated_gae(seed: int, n_parts: int) -> None:
    rng = np.random.default_rng(seed)
    full = _random_stream(rng)
    n = len(full.t_s)
    cuts = [0, *sorted(rng.choice(np.arange(1, n), size=n_parts - 1, replace=False).tolist()), n]
    for lam in (0.0, 0.95):
        credit = PerTrainCredit(gamma_min=G)
        got: dict[int, tuple[float, float, int]] = {}
        for q, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            batch = assemble(credit, [_slice(full, a, b)], _v1, lam, normalizer=None, reward_clip=1e12)
            for p, d, A in zip(batch.pos.tolist(), batch.deltas.tolist(), batch.advantages.tolist()):
                assert p not in got
                got[p] = (d, A, q)
        cut_of = {p: v[2] for p, v in got.items()}
        ref = _reference(full, lam, cut_of)
        assert set(got) == set(ref)
        for p, (d, A, _) in got.items():
            assert d == ref[p][0]  # bit-identical TD errors whatever the split
            assert A == np.float32(ref[p][1]) if lam == 0.0 else math.isclose(A, ref[p][1], rel_tol=1e-6, abs_tol=1e-6)
        if lam == 0.0:  # lambda = 0: the advantage is delta, identical across splits
            unsplit = _reference(full, 0.0, None)
            assert all(np.float32(unsplit[p][1]) == A for p, (_, A, _) in got.items())


def test_carried_too_long_is_closed_as_terminal() -> None:
    # train 1 decides once; train 2 keeps deciding (ticking) for 5 rollouts, no episode end
    n = 10
    seg = _seg(obs=np.zeros((n, 2)), values=np.zeros(n), dones=np.zeros(n), t_s=np.arange(n) * 60.0,
               ticks=np.full(n, 6), train_ids=[1] + [2] * (n - 1), routes=np.zeros(n), imm=np.zeros(n),
               d_glob=np.ones(n), d_route=np.ones((n, 1)), ep=np.zeros(n))
    credit = PerTrainCredit(gamma_min=G, max_carry=3)
    first_closed_at = None
    for q in range(5):
        tr = credit.process(0, _slice(seg, 2 * q, 2 * q + 2))
        j = np.flatnonzero(tr.pos == 0)
        if j.size and tr.closed[j[0]]:
            first_closed_at = q
            assert tr.forced[j[0]] and tr.terminal[j[0]]
            # its flow sum covers every step seen while open: t_mid = 30 + 60 m for m = 0..7
            expect = -S * sum(G ** ((30.0 + 60.0 * m) / 60.0) for m in range(8))
            assert math.isclose(tr.r_scope[j[0]], expect, rel_tol=1e-12)
    assert first_closed_at == 3  # open after rollouts 0, 1, 2 (carried 3 times), closed at the 4th end


def test_normalizer_welford_and_state() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(2.0, 3.0, 5000)
    norm = TransitionReturnNormalizer()
    for part in np.array_split(x, 7):
        norm.update(part)
    assert math.isclose(norm.mean, x.mean(), rel_tol=1e-3) and math.isclose(norm.var, x.var(), rel_tol=1e-3)
    out = norm.normalize(np.array([1e9, -1e9, 3.0]))
    assert out[0] == 10.0 and out[1] == -10.0 and math.isclose(out[2], 3.0 / norm.std)
    other = TransitionReturnNormalizer()
    other.load_state_dict(norm.state_dict())
    assert other.state_dict() == norm.state_dict()
    with pytest.raises(ValueError):
        other.load_state_dict({"mean": 0.0, "var": 1.0, "count": 1.0, "gamma": 0.995, "clip": 10.0})


def test_discounted_returns_follow_own_chain() -> None:
    seg = _hand_stream()
    tr = PerTrainCredit(gamma_min=G).process(0, seg)
    g = discounted_returns(tr.r_scope, tr)
    # T0 -> T2 -> T4 (terminal): G0 = R0 + Gamma0 * (R2 + Gamma2 * R4)
    expect = tr.r_scope[0] + G ** 0.5 * (tr.r_scope[2] + G * tr.r_scope[4])
    assert math.isclose(g[0], expect, rel_tol=1e-12)


def test_per_rollout_batch_size() -> None:
    assert per_rollout_batch_size(15_360) == 1920  # round(7.5) = 8 mini-batches (round half to even)
    assert per_rollout_batch_size(2047) == 2047
    assert per_rollout_batch_size(1000) == 1000  # round(0.49) = 0 -> one batch
    assert per_rollout_batch_size(0) == 1
    for n in range(1, 20_000, 997):
        b = per_rollout_batch_size(n)
        k = math.ceil(n / b)
        assert n - (k - 1) * b >= b - k  # the last mini-batch is not a tiny remainder
