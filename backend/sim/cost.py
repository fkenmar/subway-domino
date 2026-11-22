"""PRD §4.3 cost terms (ARCHITECTURE §4.4): pure functions shared by every simulator.

Per-tick terms accumulate over ``dt_min = DT_S / 60`` minutes::

    C_wait   = W_WAIT  * sum_p sum_b waiting[p, b] * age_b^1.5          * dt   (age_b = b + 0.5 min; last bin 30)
    C_tunnel = sum_trains tunnel_stopped * (W_TUNNEL_TRAIN + W_TUNNEL_RIDER * onboard * a^1.5) * dt
               (a = minutes the train has been stopped, clipped to [1, 30]; exponent TUNNEL_RIDER_AGE_EXPONENT)
    C_hold   = sum_trains held_by_action * (W_HOLD_TRAIN   + W_HOLD_RIDER   * onboard) * dt
    C_crowd  = W_CROWD * sum_p (max(0, waiting_p / cap_p - 0.8) / 0.2)^2 * dt

The trapped-rider age weight (docs/progress/WP2.md, contract change request 5) applies the
platform wait kernel to riders stuck in a tunnel: with a flat 1 per rider-minute a rider trapped
for 30 min cost 30 while a rider waiting 30 min on a platform cost ~1,970, so discharging
riders (short-turn) or holding them at a platform looked ~65x worse than leaving them trapped.
The first minute weighs 1, as before.

Immediate (action-time) terms are charged by :mod:`sim.actions`: short-turn
``W_ST * riders_forced_off``, skip ``W_SKIP * riders_overshot``, reroute
``W_REROUTE + W_REROUTE_RIDER * riders_displaced`` and yard ``W_YARD``.
"""

from __future__ import annotations

import numpy as np

from sim import constants as C

#: wait-age weights per histogram bin: age_b^1.5 with bin centres in minutes
AGE_WEIGHTS: np.ndarray = np.asarray(C.age_bin_centers_min(), dtype=np.float64) ** C.WAIT_AGE_EXPONENT
AGE_WEIGHTS.flags.writeable = False

I_WAIT, I_TUNNEL, I_HOLD, I_CROWD = 0, 1, 2, 3
I_TOTAL = C.COST_KEYS.index("total")


def wait_cost(waiting: np.ndarray, dt_min: float) -> float:
    """Quadratic-ish wait-age cost of the platform queues ``waiting[p, b]``."""
    return C.W_WAIT * float((waiting @ AGE_WEIGHTS).sum()) * dt_min


def crowd_cost(total_waiting: np.ndarray, capacity: np.ndarray, dt_min: float) -> float:
    """Steep penalty for platforms above ``CROWD_THRESHOLD`` of their capacity."""
    ex = np.maximum(0.0, total_waiting / capacity - C.CROWD_THRESHOLD) / C.CROWD_SPAN
    return C.W_CROWD * float((ex * ex).sum()) * dt_min


def train_cost(selected: np.ndarray, onboard: np.ndarray, w_train: float, w_rider: float, dt_min: float) -> float:
    """``sum(selected * (w_train + w_rider * onboard)) * dt`` (tunnel and hold terms)."""
    return float(((w_train + w_rider * onboard) * selected).sum()) * dt_min


def trapped_age_weight(stopped_min: np.ndarray) -> np.ndarray:
    """Per-rider tunnel weight ``a^TUNNEL_RIDER_AGE_EXPONENT`` after ``stopped_min`` minutes (a in [1, 30]).

    The same age kernel as the platform wait penalty, so a rider trapped in a tunnel is never
    cheaper than a rider waiting on a platform for as long; the first minute weighs 1.
    """
    if C.TUNNEL_RIDER_AGE_EXPONENT == 0.0:
        return np.ones_like(stopped_min, dtype=np.float64)
    a = np.clip(stopped_min, 1.0, C.WAIT_AGE_CAP_MIN)
    return a ** C.TUNNEL_RIDER_AGE_EXPONENT


def tick_costs(
    waiting: np.ndarray,
    capacity: np.ndarray,
    onboard: np.ndarray,
    tunnel_stopped: np.ndarray,
    held: np.ndarray,
    dt_s: float = C.DT_S,
    tunnel_stopped_min: np.ndarray | None = None,
) -> np.ndarray:
    """The four per-tick terms ``[wait, tunnel, hold, crowd]`` for one tick of ``dt_s``.

    ``tunnel_stopped_min`` (per train, minutes stopped so far) weights the trapped riders by
    :func:`trapped_age_weight`; omitted, every trapped rider weighs 1 (first minute).
    """
    dtm = dt_s / 60.0
    ob_t = onboard if tunnel_stopped_min is None else onboard * trapped_age_weight(tunnel_stopped_min)
    return np.array(
        [
            wait_cost(waiting, dtm),
            train_cost(tunnel_stopped, ob_t, C.W_TUNNEL_TRAIN, C.W_TUNNEL_RIDER, dtm),
            train_cost(held, onboard, C.W_HOLD_TRAIN, C.W_HOLD_RIDER, dtm),
            crowd_cost(waiting.sum(axis=1), capacity, dtm),
        ],
        dtype=np.float64,
    )
