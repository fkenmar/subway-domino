"""Calibration report on the real network: ``python -m sim.calibrate``.

Normal Schedule only (every decision resolves as CONTINUE_SCHEDULED), so the numbers describe
the physics, timetable and demand model, not a controller.

* **Calm service** per profile, no disruptions: concurrent trains against the ARCHITECTURE §0
  weekday counts (±15 %), median dwell, rider-weighted wait, overcrowded platforms, tunnel-stuck
  trains and the longest time any train stood still (a deadlock detector).
* **Presets** (the curated meltdowns in ``network.json``): tunnel-stuck trains and the average
  wait over the first ``PRESET_WINDOW_S`` against a calm twin of the same scenario without its
  disruptions.

The targets are module constants; ``tests/test_signaling.py`` (calm) and
``tests/test_action_leverage.py`` (presets) assert them as slow tests.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from sim import constants as C
from sim.engine import NYCSubwaySim
from sim.graph import Graph, load_graph
from sim.scenarios import Scenario, preset_scenario

# --------------------------------------------------------------------------- targets
#: ARCHITECTURE §0 concurrent weekday trains (08h, 12h, 17h, 02h) as (low, high).
TRAINS_REF: dict[str, tuple[float, float]] = {
    "am_peak": (491.0, 535.0),
    "midday": (400.0, 400.0),
    "pm_rush": (504.0, 504.0),
    "late_night": (83.0, 116.0),
}
TRAINS_TOL = 0.15
DWELL_MEDIAN_RANGE_S = (25.0, 45.0)
AM_WAIT_RANGE_MIN = (1.5, 5.0)
OVERCROWD_MAX_FRAC = 0.02
STUCK_MEAN_MAX = 3.0
MAX_STATIONARY_S = 600.0
#: Presets under Normal Schedule, first PRESET_WINDOW_S: a preset that blocks track (local/express
#: block, station closure) jams the tunnel (>= PRESET_STUCK_MIN stuck trains); a capacity
#: restriction (signal delay) at least stops more trains than calm service. Every preset raises
#: the citywide wait above its calm twin; the reference meltdown raises it clearly.
PRESET_WINDOW_S = 900.0
PRESET_STUCK_MIN = 6
BLOCKING_KINDS = frozenset({"local_block", "express_block", "full_station_closure"})
REFERENCE_MELTDOWN = "lex-union-sq"
#: "Clearly rising" (reference meltdown): the citywide wait grows over the window by at least this
#: much (min) and ends at least this much above the calm twin. Other presets are local enough that
#: the citywide average moves by ~0.05-0.15 min in 15 min; for them the excess must be positive.
PRESET_WAIT_RISE_MIN = 0.10

SAMPLE_S = 60.0


def trains_band(profile: str) -> tuple[float, float]:
    lo, hi = TRAINS_REF[profile]
    return lo * (1.0 - TRAINS_TOL), hi * (1.0 + TRAINS_TOL)


# --------------------------------------------------------------------------- sampling
@dataclass
class _Series:
    t_min: list[float] = field(default_factory=list)
    trains: list[int] = field(default_factory=list)
    wait: list[float] = field(default_factory=list)
    over: list[int] = field(default_factory=list)
    stuck: list[int] = field(default_factory=list)
    decisions: int = 0
    decision_ticks: int = 0
    valid_counts: np.ndarray = field(default_factory=lambda: np.zeros(C.N_ACTIONS, dtype=np.int64))
    max_stationary_s: float = 0.0


def _position_key(sim: NYCSubwaySim) -> np.ndarray:
    """(ptr, offset) per slot; equal keys at consecutive samples mean the train did not move."""
    return np.stack([sim.tr_ptr.astype(np.float64), sim.tr_off.astype(np.float64), sim.tr_pattern.astype(np.float64)], 1)


def run_series(sim: NYCSubwaySim, duration_s: float, sample_s: float = SAMPLE_S) -> _Series:
    """Advance ``sim`` under Normal Schedule, sampling metrics every ``sample_s``.

    Decisions are counted from the engine's per-tick snapshot (the same rows
    :meth:`NYCSubwaySim.pending_decisions` would return) without building observations.
    """
    out = _Series()
    per = max(1, int(round(sample_s / C.DT_S)))
    n_ticks = int(round(duration_s / C.DT_S))
    last_key = _position_key(sim)
    still_since = np.full(sim.tr_active.shape[0], sim.clock_s)
    for k in range(n_ticks):
        sim.step(1)
        n = int(sim._pend_ids.size)
        if n:
            out.decisions += n
            out.decision_ticks += 1
            out.valid_counts += sim._pend_masks.sum(axis=0)
        if (k + 1) % per == 0:
            m = sim.metrics()
            out.t_min.append(sim.episode_time_s / 60.0)
            out.trains.append(int(m["trains_active"]))
            out.wait.append(float(m["avg_wait_min"]))
            out.over.append(int(m["overcrowded_platforms"]))
            out.stuck.append(int(m["tunnel_stuck_trains"]))
            key = _position_key(sim)
            moved = ~np.all(key == last_key, axis=1) | ~sim.tr_active
            still_since[moved] = sim.clock_s
            last_key = key
            # timed manoeuvres (turnbacks, key-by stops) stand still legitimately for minutes;
            # dwell/hold is capped by DWELL_MAX_S + MAX_HOLD_S. Anything beyond 10 min is a jam.
            act = sim.tr_active
            if act.any():
                out.max_stationary_s = max(out.max_stationary_s, float((sim.clock_s - still_since[act]).max()))
    return out


# --------------------------------------------------------------------------- calm profiles
@dataclass(frozen=True)
class ProfileReport:
    profile: str
    seed: int
    duration_s: float
    trains_mean: float
    trains_min: int
    trains_max: int
    dwell_median_s: float
    dwell_p90_s: float
    avg_wait_min: float
    avg_wait_max_min: float
    overcrowded_frac_mean: float
    overcrowded_frac_max: float
    stuck_mean: float
    stuck_max: int
    max_stationary_s: float
    missed_departures: int
    decisions_per_hour: float
    valid_action_counts: tuple[int, ...]
    wall_s: float

    def checks(self) -> dict[str, bool]:
        lo, hi = trains_band(self.profile)
        res = {
            "trains_active": lo <= self.trains_mean <= hi,
            "dwell_median": DWELL_MEDIAN_RANGE_S[0] <= self.dwell_median_s <= DWELL_MEDIAN_RANGE_S[1],
            "overcrowded": self.overcrowded_frac_mean < OVERCROWD_MAX_FRAC,
            "tunnel_stuck": self.stuck_mean <= STUCK_MEAN_MAX,
            "no_jam": self.max_stationary_s <= MAX_STATIONARY_S,
        }
        if self.profile == "am_peak":
            res["avg_wait"] = AM_WAIT_RANGE_MIN[0] <= self.avg_wait_min <= AM_WAIT_RANGE_MIN[1]
        return res


def run_profile(graph: Graph, profile: str, *, seed: int = 0, duration_s: float = 3600.0) -> ProfileReport:
    t0 = time.perf_counter()
    sim = NYCSubwaySim(graph, Scenario(seed=seed, profile=profile, episode_s=duration_s))
    s = run_series(sim, duration_s)
    d = sim.diagnostics()
    n_pl = graph.n_platforms
    return ProfileReport(
        profile=profile,
        seed=seed,
        duration_s=duration_s,
        trains_mean=float(np.mean(s.trains)),
        trains_min=int(np.min(s.trains)),
        trains_max=int(np.max(s.trains)),
        dwell_median_s=float(d["dwell_median_s"]),
        dwell_p90_s=float(d["dwell_p90_s"]),
        avg_wait_min=float(np.mean(s.wait)),
        avg_wait_max_min=float(np.max(s.wait)),
        overcrowded_frac_mean=float(np.mean(s.over)) / n_pl,
        overcrowded_frac_max=float(np.max(s.over)) / n_pl,
        stuck_mean=float(np.mean(s.stuck)),
        stuck_max=int(np.max(s.stuck)),
        max_stationary_s=float(s.max_stationary_s),
        missed_departures=int(d["missed_departures"]),
        decisions_per_hour=s.decisions * 3600.0 / duration_s,
        valid_action_counts=tuple(int(x) for x in s.valid_counts),
        wall_s=time.perf_counter() - t0,
    )


# --------------------------------------------------------------------------- presets
@dataclass(frozen=True)
class PresetReport:
    preset_id: str
    profile: str
    window_s: float
    stuck_series: tuple[int, ...]
    wait_series: tuple[float, ...]
    calm_wait_series: tuple[float, ...]
    calm_stuck_series: tuple[int, ...]
    decisions_per_hour: float
    calm_decisions_per_hour: float
    valid_action_counts: tuple[int, ...]
    blocking: bool

    @property
    def stuck_max(self) -> int:
        return max(self.stuck_series)

    @property
    def wait_rise_min(self) -> float:
        """Growth of the average wait over the window (end minus the first sample)."""
        return self.wait_series[-1] - self.wait_series[0]

    @property
    def wait_excess_min(self) -> float:
        """Average wait above the calm twin at the end of the window."""
        return self.wait_series[-1] - self.calm_wait_series[-1]

    def checks(self) -> dict[str, bool]:
        res = {
            "tunnel_stuck": (self.stuck_max >= PRESET_STUCK_MIN if self.blocking
                             else self.stuck_max > max(self.calm_stuck_series)),
            "wait_excess": self.wait_excess_min > 0.0,
        }
        if self.preset_id == REFERENCE_MELTDOWN:
            res["wait_rising"] = self.wait_rise_min >= PRESET_WAIT_RISE_MIN and self.wait_excess_min >= PRESET_WAIT_RISE_MIN
        return res


def calm_twin(sc: Scenario) -> Scenario:
    return dataclasses.replace(sc, disruptions=())


def run_preset(graph: Graph, preset_id: str, *, window_s: float = PRESET_WINDOW_S) -> PresetReport:
    sc = dataclasses.replace(preset_scenario(preset_id, graph), episode_s=window_s)
    hot = run_series(NYCSubwaySim(graph, sc), window_s)
    calm = run_series(NYCSubwaySim(graph, calm_twin(sc)), window_s)
    return PresetReport(
        preset_id=preset_id,
        profile=sc.profile,
        window_s=window_s,
        stuck_series=tuple(hot.stuck),
        wait_series=tuple(hot.wait),
        calm_wait_series=tuple(calm.wait),
        calm_stuck_series=tuple(calm.stuck),
        decisions_per_hour=hot.decisions * 3600.0 / window_s,
        calm_decisions_per_hour=calm.decisions * 3600.0 / window_s,
        valid_action_counts=tuple(int(x) for x in hot.valid_counts),
        blocking=any(d.kind in BLOCKING_KINDS for d in sc.disruptions),
    )


# --------------------------------------------------------------------------- report
def _mark(ok: bool) -> str:
    return "ok " if ok else "OUT"


def format_profile(r: ProfileReport) -> str:
    ck = r.checks()
    lo, hi = trains_band(r.profile)
    lines = [
        f"[{r.profile}] seed={r.seed} {r.duration_s / 60:.0f} sim-min ({r.wall_s:.1f} s wall)",
        f"  {_mark(ck['trains_active'])} trains_active   mean {r.trains_mean:6.1f} (min {r.trains_min}, max {r.trains_max}); "
        f"target {lo:.0f}-{hi:.0f}",
        f"  {_mark(ck['dwell_median'])} dwell median    {r.dwell_median_s:6.1f} s (p90 {r.dwell_p90_s:.1f} s); "
        f"target {DWELL_MEDIAN_RANGE_S[0]:.0f}-{DWELL_MEDIAN_RANGE_S[1]:.0f} s",
    ]
    wait = f"  {'   ' if 'avg_wait' not in ck else _mark(ck['avg_wait'])} avg_wait_min    {r.avg_wait_min:6.2f} (max {r.avg_wait_max_min:.2f})"
    if "avg_wait" in ck:
        wait += f"; target {AM_WAIT_RANGE_MIN[0]}-{AM_WAIT_RANGE_MIN[1]}"
    lines += [
        wait,
        f"  {_mark(ck['overcrowded'])} overcrowded     {100 * r.overcrowded_frac_mean:6.2f} % of platforms "
        f"(max {100 * r.overcrowded_frac_max:.2f} %); target < {100 * OVERCROWD_MAX_FRAC:.0f} %",
        f"  {_mark(ck['tunnel_stuck'])} tunnel_stuck    mean {r.stuck_mean:6.2f} (max {r.stuck_max}); target <= {STUCK_MEAN_MAX:.0f}",
        f"  {_mark(ck['no_jam'])} max stationary  {r.max_stationary_s:6.0f} s; target <= {MAX_STATIONARY_S:.0f} s",
        f"      missed departures {r.missed_departures}; decisions/sim-h {r.decisions_per_hour:.1f}; "
        f"valid {_valid_str(r.valid_action_counts)}",
    ]
    return "\n".join(lines)


_SHORT_NAMES = ("CONTINUE", "HOLD", "EXPRESS", "LOCAL", "SKIP", "SHORT_TURN", "REROUTE", "YARD")


def _valid_str(counts: Sequence[int]) -> str:
    """Decision rows on which each non-CONTINUE action was valid."""
    return " ".join(f"{name}={c}" for name, c in zip(_SHORT_NAMES[1:], counts[1:]))


def format_preset(r: PresetReport) -> str:
    ck = r.checks()
    stuck_target = f">= {PRESET_STUCK_MIN}" if r.blocking else f"> calm max {max(r.calm_stuck_series)}"
    wait_ok = ck["wait_excess"] and ck.get("wait_rising", True)
    wait_target = (f"rise and excess >= {PRESET_WAIT_RISE_MIN}" if "wait_rising" in ck else "excess > 0")
    return "\n".join([
        f"[preset {r.preset_id}] {r.profile}, first {r.window_s / 60:.0f} sim-min",
        f"  {_mark(ck['tunnel_stuck'])} tunnel_stuck max {r.stuck_max} (series {list(r.stuck_series)}); "
        f"calm max {max(r.calm_stuck_series)}; target {stuck_target}",
        f"  {_mark(wait_ok)} avg_wait {r.wait_series[0]:.2f} -> {r.wait_series[-1]:.2f} min "
        f"(calm {r.calm_wait_series[-1]:.2f}); rise {r.wait_rise_min:+.2f}, excess {r.wait_excess_min:+.2f}; "
        f"target {wait_target}",
        f"      decisions/sim-h {r.decisions_per_hour:.1f} (calm {r.calm_decisions_per_hour:.1f}); "
        f"valid {_valid_str(r.valid_action_counts)}",
    ])


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.calibrate", description=__doc__.splitlines()[0])
    ap.add_argument("--profiles", nargs="*", default=list(C.PROFILES))
    ap.add_argument("--presets", nargs="*", default=None, help="preset ids (default: all in network.json)")
    ap.add_argument("--duration", type=float, default=3600.0, help="calm-service sim seconds per profile")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    g = load_graph()
    failures = 0
    for prof in args.profiles:
        r = run_profile(g, prof, seed=args.seed, duration_s=args.duration)
        print(format_profile(r), flush=True)
        failures += sum(not v for v in r.checks().values())
    presets = args.presets if args.presets is not None else [p["id"] for p in g.presets]
    for pid in presets:
        r2 = run_preset(g, pid)
        print(format_preset(r2), flush=True)
        failures += sum(not v for v in r2.checks().values())
    print(f"\n{failures} target(s) out of range")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
