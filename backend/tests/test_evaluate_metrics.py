"""§4.8 excess-wait recovery metric (decided with the user 2026-11-22)."""
import math

import pytest

from rl import evaluate as ev
from sim import constants as C


def _row(normal_w: float, ai_w: float, calm_w: float | None, seed: int) -> dict:
    cost = {k: 0.0 for k in C.COST_KEYS}

    def mk(w: float) -> dict:
        return {"avg_wait_min": w, "overcrowded_platform_min": 1.0, "tunnel_stuck_train_min": 10.0,
                "riders_served": 100.0, "cost": {**cost, "total": 100.0 * w}, "actions": {}, "decisions": 5}

    r = {"seed": seed, "profile": "am_peak", "n_disruptions": 1, "demand_scale": 1.0, "normal": mk(normal_w), "ai": mk(ai_w)}
    if calm_w is not None:
        r["calm"] = {"avg_wait_min": calm_w}
    return r


def test_recovery_is_ratio_of_sums() -> None:
    rows = [_row(5.0, 4.5, 4.0, 1), _row(4.0, 3.9, 3.9, 2), _row(6.0, 5.0, 4.0, 3)]
    e = ev.excess_wait_recovery(rows, resamples=200)
    assert e["recovery"] == pytest.approx((0.5 + 0.1 + 1.0) / (1.0 + 0.1 + 2.0))
    assert e["excess_mean"] == pytest.approx(3.1 / 3)
    lo, hi = e["recovery_ci95"]
    assert lo <= e["recovery"] <= hi


def test_bar_uses_recovery_not_citywide_cut() -> None:
    # AI removes 20 % of the excess: fails the recovery bar even though wait drops 4 % citywide
    rows = [_row(5.0, 4.8, 4.0, s) for s in range(4)]
    bar = ev.summarize(rows)["acceptance_bar"]
    assert bar["excess_wait_recovery"] == pytest.approx(0.2)
    assert bar["wait_ok"] is (0.2 >= ev.SHIP_EXCESS_WAIT_RECOVERY)
    assert "wait_reduction" in bar and "stuck_reduction" in bar  # keys the Colab notebook prints


def test_rows_without_calm_give_nan_and_fail() -> None:
    rows = [_row(5.0, 4.0, None, 1)]
    s = ev.summarize(rows)
    assert math.isnan(s["excess_wait"]["recovery"])
    assert s["acceptance_bar"]["wait_ok"] is False
    assert "n/a" in ev._excess_line(s)
