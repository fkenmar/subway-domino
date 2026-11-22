"""Vectorized train kinematics (ARCHITECTURE §3.3).

Each tick, every moving train gets a *movement authority*: the distance to the nearest
obstacle along its path (leader's rear under fixed-block or CBTC rules, a red signal at a
blocked edge, a closed station, its next stop, a timed edge entry, the look-ahead horizon).
The train then follows the exact accelerate / cruise / brake profile that stops at the
authority end, integrated analytically over the tick. Using the closed form instead of an
Euler step removes the 10 s tick quantisation from running times: a train arriving at a
platform mid-tick starts its dwell at the exact arrival instant.
"""

from __future__ import annotations

import numpy as np

from sim import constants as C

_EPS = 1e-6


def motion_profile(
    v0: np.ndarray,
    vm: np.ndarray,
    d: np.ndarray,
    T: np.ndarray,
    a: float = C.ACCEL,
    b: float = C.BRAKE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Motion over ``T`` seconds towards a mandatory stop at distance ``d``.

    The train accelerates at ``a`` (or decelerates at ``b`` if above ``vm``), cruises at
    ``vm`` and brakes at ``b`` to stand exactly at ``d``. A train already above its braking
    curve (``v0^2 > 2 b d``, e.g. when an obstacle appeared) brakes harder to stop at ``d``.

    Returns ``(x, v, t_arr)``: distance travelled, final speed, and the arrival time at
    ``d`` (``inf`` when the train does not reach ``d`` within ``T``).
    """
    v0 = np.maximum(v0, 0.0)
    d = np.maximum(d, 0.0)
    vm = np.maximum(vm, 0.1)

    over = v0 * v0 > 2.0 * b * d + 1e-9

    # --- normal case -------------------------------------------------------------------
    accel = v0 <= vm
    vi = np.sqrt((2.0 * a * b * d + b * v0 * v0) / (a + b))
    vp = np.where(accel, np.minimum(vm, vi), vm)
    alpha = np.where(accel, a, -b)
    t1 = np.abs(vp - v0) / np.abs(alpha)
    x1 = (vp * vp - v0 * v0) / (2.0 * alpha)
    x3 = vp * vp / (2.0 * b)
    t3 = vp / b
    x2 = np.maximum(d - x1 - x3, 0.0)
    t2 = np.where(vp > _EPS, x2 / np.maximum(vp, _EPS), 0.0)
    t_arr_n = t1 + t2 + t3

    in1 = T <= t1
    in2 = (~in1) & (T <= t1 + t2)
    in3 = (~in1) & (~in2) & (T < t_arr_n)
    tau = np.maximum(T - t1 - t2, 0.0)
    x_n = np.where(
        in1,
        v0 * T + 0.5 * alpha * T * T,
        np.where(in2, x1 + vp * (T - t1), np.where(in3, x1 + x2 + vp * tau - 0.5 * b * tau * tau, d)),
    )
    v_n = np.where(in1, v0 + alpha * T, np.where(in2, vp, np.where(in3, vp - b * tau, 0.0)))
    arrived_n = ~(in1 | in2 | in3)

    # --- over-speed: brake at whatever rate stops the train at d --------------------------
    d_safe = np.maximum(d, _EPS)
    be = v0 * v0 / (2.0 * d_safe)
    ts = np.where(d > _EPS, 2.0 * d_safe / np.maximum(v0, _EPS), 0.0)
    arrived_o = T >= ts
    x_o = np.where(arrived_o, d, v0 * T - 0.5 * be * T * T)
    v_o = np.where(arrived_o, 0.0, v0 - be * T)

    x = np.where(over, x_o, x_n)
    v = np.where(over, v_o, v_n)
    t_arr = np.where(over, np.where(arrived_o, ts, np.inf), np.where(arrived_n, t_arr_n, np.inf))
    x = np.clip(x, 0.0, d)
    v = np.clip(v, 0.0, None)
    return x, v, t_arr


def time_to_reach(v0: float, vm: float, d: float, x_target: float, a: float = C.ACCEL, b: float = C.BRAKE) -> float:
    """Time at which the :func:`motion_profile` trajectory reaches ``x_target`` (scalar)."""
    v0 = max(v0, 0.0)
    d = max(d, 0.0)
    vm = max(vm, 0.1)
    x_target = min(max(x_target, 0.0), d)
    if v0 * v0 > 2.0 * b * d + 1e-9:
        if d <= _EPS:
            return 0.0
        be = v0 * v0 / (2.0 * d)
        disc = max(v0 * v0 - 2.0 * be * x_target, 0.0)
        return (v0 - disc**0.5) / be
    accel = v0 <= vm
    vi = ((2.0 * a * b * d + b * v0 * v0) / (a + b)) ** 0.5
    vp = min(vm, vi) if accel else vm
    alpha = a if accel else -b
    t1 = abs(vp - v0) / abs(alpha)
    x1 = (vp * vp - v0 * v0) / (2.0 * alpha)
    x3 = vp * vp / (2.0 * b)
    x2 = max(d - x1 - x3, 0.0)
    if x_target <= x1 + 1e-9:
        # solve v0 t + 0.5 alpha t^2 = x
        disc = max(v0 * v0 + 2.0 * alpha * x_target, 0.0)
        return (disc**0.5 - v0) / alpha if abs(alpha) > 0 else x_target / max(v0, _EPS)
    if x_target <= x1 + x2 + 1e-9:
        return t1 + (x_target - x1) / max(vp, _EPS)
    t2 = x2 / max(vp, _EPS)
    rem = x_target - x1 - x2
    disc = max(vp * vp - 2.0 * b * rem, 0.0)
    return t1 + t2 + (vp - disc**0.5) / b


def fixed_block_authority(obs_dist: np.ndarray, starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Authority end for an obstacle (leader's rear) at path distance ``obs_dist``.

    Fixed-block rule: the follower must stop ``FIXED_BLOCK_STOP_MARGIN_M`` before the
    signal at the start of the block that contains ``obs_dist - FIXED_BLOCK_OVERLAP_M``
    (block boundaries every ``BLOCK_M`` from each edge's start). ``starts``/``lens`` are
    the look-ahead edges' start distances and lengths, shape ``[n, K]`` (``inf`` start for
    invalid slots).
    """
    p = obs_dist - C.FIXED_BLOCK_OVERLAP_M
    # edge holding point p: last k with start_k <= p among edges with positive length
    has_len = lens > 0.0
    cand = (starts <= p[:, None] + 1e-9) & has_len
    k = np.where(cand.any(axis=1), cand.shape[1] - 1 - np.argmax(cand[:, ::-1], axis=1), -1)
    rows = np.arange(len(p))
    kk = np.maximum(k, 0)
    s_k = starts[rows, kk]
    o = p - s_k
    block = np.floor(np.maximum(o, 0.0) / C.BLOCK_M) * C.BLOCK_M
    auth = s_k + block - C.FIXED_BLOCK_STOP_MARGIN_M
    auth = np.where(k >= 0, auth, -1.0)
    return np.where(np.isfinite(obs_dist), auth, np.inf)
