"""Per-train semi-MDP credit assignment for training (WP5b; ARCHITECTURE §4.4, §4.6).

The serial scheme (``--credit serial``) treats each worker's interleaved decision stream as one
trajectory: delta bootstraps from whichever train decides next, gamma and lambda apply per
*global* decision, and the reward is the citywide twin delta. Here every train is its own
semi-MDP instead. Train i decides at step k (episode time t_k) and next decides at step k'
(time t_k'), or the episode ends::

    R_k     = -REWARD_SCALE * (imm_k + sum_{m=k}^{k'-1} gamma_min^((t_mid(m) - t_k)/60) * d_scope_i(m))
    t_mid(m) = t_m + ticks_m * DT_S / 2
    Gamma_k = gamma_min^((t_k' - t_k)/60)            (gamma per sim-minute, not per decision)
    delta_k = R_k + Gamma_k * V(o_k') - V(o_k)       (the episode end is terminal: no bootstrap)
    A_k     = delta_k + Gamma_k * lambda * A_k'      (lambda per *own* decision); returns A + V

``imm_k`` is the immediate cost of train i's own action. ``d_scope_i(m)`` is step m's flow-cost
difference AI - twin restricted to the scope of train i's route (:func:`scope_weights`), or the
citywide difference with ``--reward-scope global``.

Rollout boundaries. A transition whose train has not decided again yet, in an episode that is
still running, is *open*. The env state persists across rollouts, so an open transition is
carried into its worker's next rollout and closed there. The lambda-recursion is truncated at
it, the way truncated GAE treats a rollout end: a closed transition whose next transition is
still open uses ``A_k = delta_k``. Flow sums accumulate step by step in time order, and the
per-worker slot table survives the boundary unchanged, so splitting a stream into rollouts
changes nothing but that truncation (tests/test_credit.py). A transition still open after more
than ``max_carry`` rollouts is closed as terminal.

NumPy only: :mod:`rl.env` imports this module inside the rollout workers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from sim import constants as C
from sim.engine import route_platform_incidence
from sim.graph import Graph

#: ``--gamma-min``: discount per sim-minute (half-life ln 0.5 / ln 0.96 = 17 min)
GAMMA_MIN: float = 0.96
#: ``--scope-neighbor-weight``: weight of platforms / trains of track-sharing routes
NEIGHBOR_W: float = 0.35
#: an open transition is closed as terminal once it has been carried over more rollouts than this
MAX_CARRY: int = 3
#: normalized per-train rewards are clipped to +-NORM_CLIP
NORM_CLIP: float = 10.0
SCOPES: tuple[str, ...] = ("route", "global")

ValueFn = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------------------
# Scope weights (C)
# --------------------------------------------------------------------------------------
def track_sharing(g: Graph) -> np.ndarray:
    """``bool[n_routes, n_routes]``: routes whose patterns share at least one run edge (no diagonal)."""
    run = ~np.asarray(g.edge_timed, dtype=bool)
    use = np.zeros((g.n_routes, g.n_edges), dtype=np.int64)
    for p in range(g.n_patterns):
        e = np.asarray(g.pattern_edges(p), dtype=np.intp)
        use[int(g.pat_route[p]), e[run[e]]] = 1
    share = (use @ use.T) > 0
    np.fill_diagonal(share, False)
    return share


def scope_weights(g: Graph, neighbor_w: float = NEIGHBOR_W) -> tuple[np.ndarray, np.ndarray]:
    """``(W_plat float64[n_routes, n_platforms], W_rt float64[n_routes, n_routes])``.

    For an ego on route r: a platform weighs 1 if r serves it, ``neighbor_w`` if a route that
    shares track with r serves it, else 0; the trains of route r weigh 1, those of track-sharing
    routes ``neighbor_w``, all others 0. ``d_route[r] = W_plat[r] . dplat + W_rt[r] . drt``.
    """
    nw = float(neighbor_w)
    if not 0.0 <= nw <= 1.0:
        raise ValueError(f"neighbor weight must be in [0, 1], got {neighbor_w!r}")
    inc = np.asarray(route_platform_incidence(g)) > 0
    share = track_sharing(g)
    near = (share.astype(np.int64) @ inc.astype(np.int64)) > 0
    w_plat = np.where(inc, 1.0, np.where(near, nw, 0.0))
    w_rt = np.where(share, nw, 0.0)
    np.fill_diagonal(w_rt, 1.0)
    return w_plat, w_rt


# --------------------------------------------------------------------------------------
# Transitions (D)
# --------------------------------------------------------------------------------------
@dataclass
class Transitions:
    """One worker's transitions in one rollout.

    Rows ``0..c-1`` are the carried-in transitions (slot order), then one row per step of this
    rollout in time order, so ``nxt[j] > j`` always. Only ``closed`` rows go into the PPO
    buffer. Open rows are carried to the next rollout; closed rows that point at them through
    ``nxt`` bootstrap from their value but truncate the lambda-recursion there.
    """

    obs: np.ndarray  # float32[T, OBS_DIM]
    actions: np.ndarray  # int64[T]
    log_probs: np.ndarray  # float32[T] behavior policy
    values: np.ndarray  # float32[T] V(o_k); carried-in rows must be recomputed (current critic)
    masks: np.ndarray  # bool[T, N_ACTIONS]
    r_scope: np.ndarray  # float64[T] R_k with the route-scoped flow deltas
    r_global: np.ndarray  # float64[T] R_k with the citywide flow deltas
    gamma_dt: np.ndarray  # float64[T] Gamma_k (1 where terminal or open)
    interval_s: np.ndarray  # float64[T] t_k' - t_k (nan where terminal or open)
    nxt: np.ndarray  # int64[T] row of the train's next own transition (-1: terminal or open)
    terminal: np.ndarray  # bool[T] closed without a bootstrap (episode end, or carried too long)
    closed: np.ndarray  # bool[T]
    carried_in: np.ndarray  # bool[T] opened in an earlier rollout
    forced: np.ndarray  # bool[T] closed as terminal after MAX_CARRY rollouts
    pos: np.ndarray  # int64[T] position of the decision in the worker's stream

    def __len__(self) -> int:
        return int(self.actions.shape[0])


_PAYLOAD = ("obs", "actions", "log_probs", "values", "masks", "t_k", "route", "imm", "pos")


@dataclass
class _WorkerStream:
    """Learner-side state of one worker between rollouts: the slot table of the current episode.

    Slot s belongs to train ``tid[s]`` and holds the accumulator of its open transition. Rows
    ``0..n-1`` of ``carried`` are those open transitions in slot order, so a carried-in
    transition s is slot s again in the next rollout and every array keeps its layout.
    """

    cap: int = C.MAX_TRAINS + 2
    n: int = 0
    ep: int | None = None
    pos: int = 0
    slot_of: dict[int, int] = field(default_factory=dict)
    carried: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self._alloc(self.cap)

    def _alloc(self, cap: int) -> None:
        old = {k: getattr(self, k) for k in ("tid", "tk", "rt", "sc", "gl", "age", "tr")} if self.n else {}
        self.cap = cap
        self.tid = np.zeros(cap, dtype=np.int64)  # train slot id of the open transition
        self.tk = np.zeros(cap, dtype=np.float64)  # its decision time t_k
        self.rt = np.zeros(cap, dtype=np.intp)  # its route (column of d_route; -1 mapped to 0)
        self.sc = np.zeros(cap, dtype=np.float64)  # discounted scoped flow sum so far
        self.gl = np.zeros(cap, dtype=np.float64)  # discounted global flow sum so far
        self.age = np.zeros(cap, dtype=np.int64)  # rollout ends survived while open
        self.tr = np.zeros(cap, dtype=np.int64)  # its row in the current Transitions
        for k, v in old.items():
            getattr(self, k)[: self.n] = v[: self.n]


class PerTrainCredit:
    """Builds per-train semi-MDP transitions from each worker's time-ordered step stream.

    ``process(worker, seg)`` consumes one rollout segment of a worker (``rl.rollout.Segment``
    with the per-step credit fields) and returns its :class:`Transitions`; open transitions
    stay inside this object until a later call closes them. ``reset()`` drops them (resume).
    """

    def __init__(self, gamma_min: float = GAMMA_MIN, max_carry: int = MAX_CARRY) -> None:
        if not 0.0 < float(gamma_min) <= 1.0:
            raise ValueError(f"gamma_min must be in (0, 1], got {gamma_min!r}")
        if int(max_carry) < 0:
            raise ValueError("max_carry must be >= 0")
        self.gamma_min = float(gamma_min)
        self.max_carry = int(max_carry)
        self._k = math.log(self.gamma_min) / 60.0  # ln(gamma) per sim-second
        self._workers: dict[int, _WorkerStream] = {}

    def reset(self) -> None:
        self._workers.clear()

    def n_open(self) -> int:
        """Open transitions carried into the next rollout (all workers)."""
        return sum(w.n for w in self._workers.values())

    def process(self, worker: int, seg: Any) -> Transitions:
        ws = self._workers.get(worker)
        if ws is None:
            ws = self._workers[worker] = _WorkerStream()
        car = ws.carried
        c = 0 if car is None else int(car["actions"].shape[0])
        if c != ws.n:  # pragma: no cover - internal invariant
            raise RuntimeError("carried transitions out of sync with the slot table")
        n = int(seg.actions.shape[0])
        T = c + n
        st = np.asarray(seg.t_s, dtype=np.float64)
        ticks = np.asarray(seg.step_ticks, dtype=np.int64)
        tids = np.asarray(seg.train_ids, dtype=np.int64)
        routes = np.asarray(seg.routes, dtype=np.int64)
        dgl = np.asarray(seg.d_glob, dtype=np.float64)
        drt = np.asarray(seg.d_route)
        dones = np.asarray(seg.dones, dtype=bool)
        eps = np.asarray(seg.ep, dtype=np.int64)
        fresh = {
            "obs": np.asarray(seg.obs, dtype=np.float32), "actions": np.asarray(seg.actions, dtype=np.int64),
            "log_probs": np.asarray(seg.log_probs, dtype=np.float32), "values": np.asarray(seg.values, dtype=np.float32),
            "masks": np.asarray(seg.masks, dtype=bool), "t_k": st, "route": routes,
            "imm": np.asarray(seg.imm, dtype=np.float64), "pos": ws.pos + np.arange(n, dtype=np.int64),
        }
        rows = {k: (np.concatenate([car[k], v]) if c else v.copy()) for k, v in fresh.items()}  # type: ignore[index]
        t_k = rows["t_k"]

        s_sc = np.zeros(T, dtype=np.float64)
        s_gl = np.zeros(T, dtype=np.float64)
        gamma_dt = np.ones(T, dtype=np.float64)
        interval = np.full(T, np.nan, dtype=np.float64)
        nxt = np.full(T, -1, dtype=np.int64)
        terminal = np.zeros(T, dtype=bool)
        closed = np.zeros(T, dtype=bool)
        forced = np.zeros(T, dtype=bool)
        carried_in = np.zeros(T, dtype=bool)
        carried_in[:c] = True
        ws.tr[:c] = np.arange(c, dtype=np.int64)

        def close_all() -> None:  # every open transition of the episode ends (terminal)
            k = ws.n
            jt = ws.tr[:k]
            s_sc[jt] = ws.sc[:k]
            s_gl[jt] = ws.gl[:k]
            terminal[jt] = True
            closed[jt] = True
            ws.n = 0
            ws.slot_of.clear()

        half = 0.5 * C.DT_S
        kk = self._k
        for m in range(n):
            e = int(eps[m])
            if ws.ep != e:
                if ws.n:  # defensive: an episode boundary without a done flag
                    close_all()
                ws.ep = e
            j = c + m
            tid = int(tids[m])
            tm = float(st[m])
            s = ws.slot_of.get(tid)
            if s is None:
                if ws.n == ws.cap:
                    ws._alloc(2 * ws.cap)
                s = ws.n
                ws.n += 1
                ws.slot_of[tid] = s
                ws.tid[s] = tid
            else:  # the train's previous transition closes at its next own decision
                jp = int(ws.tr[s])
                s_sc[jp] = ws.sc[s]
                s_gl[jp] = ws.gl[s]
                dt = tm - float(t_k[jp])
                interval[jp] = dt
                gamma_dt[jp] = math.exp(kk * dt)
                nxt[jp] = j
                closed[jp] = True
            ws.tr[s] = j
            ws.tk[s] = tm
            ws.rt[s] = max(int(routes[m]), 0)
            ws.sc[s] = 0.0
            ws.gl[s] = 0.0
            ws.age[s] = 0
            if ticks[m] > 0:
                k = ws.n
                w = np.exp(kk * ((tm + float(ticks[m]) * half) - ws.tk[:k]))
                ws.sc[:k] += w * drt[m, ws.rt[:k]]
                ws.gl[:k] += w * float(dgl[m])
            if dones[m]:
                close_all()

        # rollout end: what is still open is carried (or closed as terminal once carried too long)
        k = ws.n
        if k:
            ws.age[:k] += 1
            over = ws.age[:k] > self.max_carry
            if over.any():
                jo = ws.tr[:k][over]
                s_sc[jo] = ws.sc[:k][over]
                s_gl[jo] = ws.gl[:k][over]
                terminal[jo] = forced[jo] = closed[jo] = True
                keep = np.flatnonzero(~over)
                for name in ("tid", "tk", "rt", "sc", "gl", "age", "tr"):
                    arr = getattr(ws, name)
                    arr[: keep.size] = arr[keep]
                ws.n = k = int(keep.size)
                ws.slot_of = {int(t): i for i, t in enumerate(ws.tid[:k].tolist())}
        if k:
            jt = ws.tr[:k].copy()
            ws.carried = {name: rows[name][jt] for name in _PAYLOAD}
        else:
            ws.carried = None
        ws.pos += n

        scale = -C.REWARD_SCALE
        return Transitions(
            obs=rows["obs"], actions=rows["actions"], log_probs=rows["log_probs"], values=rows["values"],
            masks=rows["masks"], r_scope=scale * (rows["imm"] + s_sc), r_global=scale * (rows["imm"] + s_gl),
            gamma_dt=gamma_dt, interval_s=interval, nxt=nxt, terminal=terminal, closed=closed,
            carried_in=carried_in, forced=forced, pos=rows["pos"],
        )


# --------------------------------------------------------------------------------------
# GAE over own-decision chains
# --------------------------------------------------------------------------------------
def _chain(tr: Transitions) -> tuple[np.ndarray, np.ndarray]:
    """``(boot, cont)``: closed rows that bootstrap from V(next), and those whose next row is closed too."""
    boot = tr.closed & ~tr.terminal
    cont = boot.copy()
    cont[boot] = tr.closed[tr.nxt[boot]]
    return boot, cont


def per_train_gae(r: np.ndarray, tr: Transitions, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(adv float32[T], ret float32[T], delta float64[T])`` for the closed rows (0 elsewhere).

    ``A_k = delta_k + Gamma_k * lam * A_k'`` while the next own transition k' is closed in the
    same rollout; at an open k' the recursion is truncated (``A_k = delta_k``).
    """
    v = tr.values.astype(np.float64)
    boot, cont = _chain(tr)
    v_next = np.zeros(len(tr), dtype=np.float64)
    v_next[boot] = v[tr.nxt[boot]]
    delta = np.where(tr.closed, np.asarray(r, dtype=np.float64) + tr.gamma_dt * v_next - v, 0.0)
    coef = (tr.gamma_dt * float(lam)).tolist()
    nl, cl = tr.nxt.tolist(), cont.tolist()
    adv = delta.tolist()
    for j in range(len(adv) - 1, -1, -1):
        if cl[j]:
            adv[j] = adv[j] + coef[j] * adv[nl[j]]
    a32 = np.asarray(adv, dtype=np.float32)
    return a32, a32 + tr.values.astype(np.float32), delta


def discounted_returns(r: np.ndarray, tr: Transitions) -> np.ndarray:
    """``G_k = R_k + Gamma_k * G_k'`` along own-decision chains (truncated like the GAE), float64[T]."""
    _, cont = _chain(tr)
    g = np.where(tr.closed, np.asarray(r, dtype=np.float64), 0.0).tolist()
    gl, nl, cl = tr.gamma_dt.tolist(), tr.nxt.tolist(), cont.tolist()
    for j in range(len(g) - 1, -1, -1):
        if cl[j]:
            g[j] = g[j] + gl[j] * g[nl[j]]
    return np.asarray(g, dtype=np.float64)


class TransitionReturnNormalizer:
    """Running std of the per-train discounted returns ``G_k`` (Welford/Chan merge per rollout).

    Rewards are divided by that std and clipped to ``+-clip``. Same initial state as
    :class:`rl.ppo.ReturnNormalizer`; ``state_dict()["kind"]`` tells the two apart on resume.
    """

    kind = "per_train"

    def __init__(self, clip: float = NORM_CLIP, epsilon: float = 1e-8) -> None:
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, g: np.ndarray) -> None:
        x = np.asarray(g, dtype=np.float64)
        if x.size == 0:
            return
        b_mean, b_var, b_n = float(x.mean()), float(x.var()), int(x.size)
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta * delta * self.count * b_n / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> float:
        return math.sqrt(self.var + self.epsilon)

    def normalize(self, r: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(r, dtype=np.float64) / self.std, -self.clip, self.clip)

    def state_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "mean": self.mean, "var": self.var, "count": self.count, "clip": self.clip}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        if d.get("kind") != self.kind:
            raise ValueError(f"not a {self.kind} normalizer state: {d.get('kind')!r}")
        self.mean, self.var, self.count = float(d["mean"]), float(d["var"]), float(d["count"])


# --------------------------------------------------------------------------------------
# One rollout -> PPO rows
# --------------------------------------------------------------------------------------
@dataclass
class CreditBatch:
    """The closed transitions of one rollout as PPO rows (worker order, then time order)."""

    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    masks: np.ndarray
    rewards: np.ndarray  # float32: the R used by the GAE (normalized and clipped)
    advantages: np.ndarray
    returns: np.ndarray
    deltas: np.ndarray  # float64 TD errors (tests, diagnostics)
    pos: np.ndarray  # int64 stream position of each decision (tests, diagnostics)
    stats: dict[str, float]

    def __len__(self) -> int:
        return int(self.actions.shape[0])


def assemble(
    credit: PerTrainCredit,
    segments: Sequence[Any],
    value_fn: ValueFn,
    lam: float,
    *,
    scope: str = "route",
    normalizer: TransitionReturnNormalizer | None = None,
    reward_clip: float = C.REWARD_CLIP,
) -> CreditBatch:
    """Transitions -> recompute carried values -> normalize R -> per-train GAE -> PPO rows."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    trs = [credit.process(int(s.worker), s) for s in segments]
    for tr in trs:  # V(o_k) of carried-in transitions with the current critic (log_prob stays)
        if tr.carried_in.any():
            tr.values[tr.carried_in] = np.asarray(value_fn(tr.obs[tr.carried_in]), dtype=np.float32)
    raws = [tr.r_global if scope == "global" else tr.r_scope for tr in trs]
    if normalizer is not None:
        g = [discounted_returns(r, tr)[tr.closed] for r, tr in zip(raws, trs)]
        normalizer.update(np.concatenate(g) if g else np.zeros(0))
        rews = [normalizer.normalize(r) for r in raws]
    else:
        rews = [np.clip(r, -reward_clip, reward_clip) for r in raws]

    if not trs:
        raise ValueError("no segments")
    cols: dict[str, list[np.ndarray]] = {k: [] for k in ("obs", "actions", "log_probs", "values", "masks",
                                                         "rewards", "advantages", "returns", "deltas", "pos")}
    for tr, r in zip(trs, rews):
        adv, ret, delta = per_train_gae(r, tr, lam)
        idx = np.flatnonzero(tr.closed)
        for name, arr in (("obs", tr.obs), ("actions", tr.actions), ("log_probs", tr.log_probs),
                          ("values", tr.values), ("masks", tr.masks), ("rewards", r.astype(np.float32)),
                          ("advantages", adv), ("returns", ret), ("deltas", delta), ("pos", tr.pos)):
            cols[name].append(arr[idx])
    out = {k: np.concatenate(v) for k, v in cols.items()}

    closed = np.concatenate([tr.closed for tr in trs])
    raw_c = np.concatenate(raws)[closed]
    sc_c = np.concatenate([tr.r_scope for tr in trs])[closed]
    gl_c = np.concatenate([tr.r_global for tr in trs])[closed]
    boot = np.concatenate([tr.closed & ~tr.terminal for tr in trs])
    iv = np.concatenate([tr.interval_s for tr in trs])[boot]
    n = int(closed.sum())
    clip = normalizer.clip if normalizer is not None else float(reward_clip)
    stats = {
        "closed": float(n),
        "carried": float(credit.n_open()),
        "carried_closed": float(sum(int((tr.carried_in & tr.closed).sum()) for tr in trs)),
        "forced_terminal": float(sum(int(tr.forced.sum()) for tr in trs)),
        "terminal": float(sum(int(tr.terminal.sum()) for tr in trs)),
        "mean_interval_s": float(iv.mean()) if iv.size else 0.0,
        "R_mean": float(raw_c.mean()) if n else 0.0,
        "R_std": float(raw_c.std()) if n else 0.0,
        "R_norm_std": float(out["rewards"].std()) if n else 0.0,
        "R_clip_frac": float(np.mean(np.abs(out["rewards"]) >= clip - 1e-6)) if n else 0.0,
        "scope_global_corr": _corr(sc_c, gl_c),
    }
    if normalizer is not None:
        stats["norm_std"] = normalizer.std
    return CreditBatch(stats=stats, **out)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def per_rollout_batch_size(n: int, target: int = 2048) -> int:
    """Mini-batch size ``ceil(n / max(1, round(n / target)))``: no tiny remainder batch."""
    return max(1, math.ceil(int(n) / max(1, round(int(n) / int(target)))))
