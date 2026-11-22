"""Simulator micro-benchmarks for the ARCHITECTURE §3.7 targets: ``python -m sim.bench``.

Measures, single-threaded on the real network (``data/network.json``):

* the citywide ``am_peak`` tick under Normal Schedule (median over many ticks),
* :meth:`NYCSubwaySim.pending_decisions` for a 32-decision batch (observation building; masks
  are evaluated inside the tick, so the mask cost of the same 32 trains is reported too),
* :meth:`NYCSubwaySim.reset` with warm-up and :meth:`NYCSubwaySim.copy`.

``tests/test_perf.py`` (marker ``slow``) asserts the targets through :func:`run_all`.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from sim import constants as C
from sim.engine import M_DWELL, M_READY, NYCSubwaySim
from sim.graph import Graph, load_graph
from sim.masks import compute_masks
from sim.scenarios import Scenario

#: ARCHITECTURE §3.7 targets (milliseconds).
TARGET_TICK_MS = 0.6
TARGET_PENDING32_MS = 0.3
TARGET_RESET_MS = 250.0
TARGET_COPY_MS = 2.0
BATCH_N = 32


@dataclass(frozen=True)
class BenchResult:
    tick_median_ms: float
    tick_p90_ms: float
    sim_s_per_cpu_s: float
    pending32_ms: float
    masks32_ms: float
    reset_ms: float
    copy_ms: float
    trains_active: int
    n_ticks: int

    def checks(self) -> dict[str, bool]:
        return {
            "tick": self.tick_median_ms <= TARGET_TICK_MS,
            "pending32": self.pending32_ms <= TARGET_PENDING32_MS,
            "reset": self.reset_ms <= TARGET_RESET_MS,
            "copy": self.copy_ms <= TARGET_COPY_MS,
        }


def _median_ms(fn: Callable[[], object], repeat: int) -> float:
    """Median wall ms of ``fn()`` over ``repeat`` calls, best of three rounds (see :func:`bench_tick`)."""
    rounds = []
    for _ in range(3):
        samples = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        rounds.append(statistics.median(samples))
    return 1e3 * min(rounds)


def bench_tick(sim: NYCSubwaySim, n_ticks: int, batches: int = 8) -> tuple[float, float]:
    """(median, p90) wall ms of ``sim.step(1)`` under Normal Schedule.

    The ticks are timed in ``batches`` consecutive batches and the batch with the lowest
    median is reported (best-of-N, as ``timeit`` does): on a laptop SoC a single-threaded
    process may be parked on an efficiency core while other work runs, which says nothing
    about the code.
    """
    per = max(1, n_ticks // batches)
    best: tuple[float, float] | None = None
    for _ in range(batches):
        dts = np.empty(per)
        for k in range(per):
            t0 = time.perf_counter()
            sim.step(1)
            dts[k] = time.perf_counter() - t0
        res = (1e3 * float(np.median(dts)), 1e3 * float(np.percentile(dts, 90)))
        if best is None or res[0] < best[0]:
            best = res
    assert best is not None
    return best


def _stopped_trains(sim: NYCSubwaySim, n: int) -> np.ndarray:
    """``n`` trains standing at a stop (ready first), for a synthetic decision batch."""
    ready = np.flatnonzero(sim.tr_active & (sim.tr_mode == M_READY))
    dwell = np.flatnonzero(sim.tr_active & (sim.tr_mode == M_DWELL))
    ids = np.concatenate([ready, dwell])[:n]
    if ids.size < n:
        raise RuntimeError(f"only {ids.size} trains at stops; cannot build a {n}-decision batch")
    return ids.astype(np.int64)


def bench_decisions(sim: NYCSubwaySim, n: int = BATCH_N, repeat: int = 100) -> tuple[float, float]:
    """(pending_decisions ms, compute_masks ms) for an ``n``-decision batch.

    The batch is installed exactly as the tick's decision collection does, so
    :meth:`NYCSubwaySim.pending_decisions` does its full work (observations and context) on
    every call; its per-tick cache is invalidated between repetitions.
    """
    from sim.masks import subset_info

    ids = _stopped_trains(sim, n)
    masks, info = compute_masks(sim, ids)
    masks_ms = _median_ms(lambda: compute_masks(sim, ids), repeat)
    sim._pend_ids = ids
    sim._pend_masks = masks
    sim._pend_info = subset_info(info, np.arange(ids.size))
    sim._pend_pos = {int(t): i for i, t in enumerate(ids.tolist())}

    def once() -> None:
        sim._batch = None
        sim.pending_decisions()

    once()
    return _median_ms(once, repeat), masks_ms


def run_all(graph: Graph | None = None, *, n_ticks: int = 720, repeat: int = 7) -> BenchResult:
    g = graph if graph is not None else load_graph()
    sc = Scenario(seed=0, profile="am_peak")
    NYCSubwaySim(g, sc)  # warm caches shared through the graph (profile views, static tables)
    reset_sim = NYCSubwaySim(g, sc, warmup=False)
    reset_ms = _median_ms(lambda: reset_sim.reset(sc), max(1, repeat // 3))
    sim = NYCSubwaySim(g, sc)
    copy_ms = _median_ms(sim.copy, 30)
    gc_was = gc.isenabled()
    gc.disable()
    try:
        bench_tick(sim, 30)  # warm-up
        tick_med, tick_p90 = bench_tick(sim, n_ticks)
        pending_ms, masks_ms = bench_decisions(sim)
    finally:
        if gc_was:
            gc.enable()
    return BenchResult(
        tick_median_ms=tick_med,
        tick_p90_ms=tick_p90,
        sim_s_per_cpu_s=C.DT_S / (tick_med / 1e3),
        pending32_ms=pending_ms,
        masks32_ms=masks_ms,
        reset_ms=reset_ms,
        copy_ms=copy_ms,
        trains_active=int(np.count_nonzero(sim.tr_active)),
        n_ticks=n_ticks,
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.bench", description=__doc__.splitlines()[0])
    ap.add_argument("--ticks", type=int, default=720, help="ticks timed for the tick median")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)
    r = run_all(n_ticks=args.ticks)
    if args.json:
        print(json.dumps({**asdict(r), "checks": r.checks()}, indent=2))
    else:
        ck = r.checks()
        mark = {True: "ok ", False: "OUT"}
        print(f"citywide am_peak, {r.trains_active} trains, {r.n_ticks} ticks, pid {os.getpid()}")
        print(f"  {mark[ck['tick']]} tick           median {r.tick_median_ms:.3f} ms (p90 {r.tick_p90_ms:.3f}); "
              f"{r.sim_s_per_cpu_s:,.0f} sim-s/CPU-s; target <= {TARGET_TICK_MS} ms")
        print(f"  {mark[ck['pending32']]} pending x32    {r.pending32_ms:.3f} ms (masks x32 {r.masks32_ms:.3f} ms, inside the tick); "
              f"target <= {TARGET_PENDING32_MS} ms")
        print(f"  {mark[ck['reset']]} reset+warmup   {r.reset_ms:.1f} ms; target <= {TARGET_RESET_MS} ms")
        print(f"  {mark[ck['copy']]} copy()         {r.copy_ms:.3f} ms; target <= {TARGET_COPY_MS} ms")
    return 0 if all(r.checks().values()) else 1


if __name__ == "__main__":
    sys.exit(main())
