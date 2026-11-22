"""Environment throughput benchmark (ARCHITECTURE §4.6: >= 2,000 env-steps/s on 14 workers).

Reports aggregate and per-env env-steps/s, simulated sim-hours/s and decisions per sim-hour for:

* ``single``: one in-process env (no IPC) - the per-core upper bound;
* ``subproc``: SB3 ``SubprocVecEnv`` stepped synchronously with random *valid* actions (or
  all-CONTINUE) - every vectorized step waits for the slowest worker;
* ``chunked``: :class:`rl.rollout.RolloutWorkers`, the collector ``rl.train`` uses - workers run
  an (untrained) policy locally for chunks of steps, one barrier per rollout.

``aggregate / (n_envs * single)`` is the parallel efficiency.

    .venv/bin/python -m rl.bench_env --n-envs 14 --seconds 60 --stage 1
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from rl.env import SubwayDominoEnv
from rl.vec import cpu_count, make_vec_env, resolve_n_envs
from sim import constants as C


def sample_valid(rng: np.random.Generator, masks: np.ndarray, policy: str) -> np.ndarray:
    """One action per row: uniform over valid actions (``random``) or CONTINUE (``continue``)."""
    if policy == "continue":
        return np.zeros(masks.shape[0], dtype=np.int64)
    u = rng.random(masks.shape) * masks
    return np.argmax(u, axis=1).astype(np.int64)


def bench_single(stage: int, seconds: float, policy: str, seed: int, graph_path: str | None) -> dict[str, Any]:
    """In-process single env (no IPC): the per-worker upper bound."""
    env = SubwayDominoEnv(stage=stage, seed=seed, graph_path=graph_path)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    steps = ticks = episodes = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = sample_valid(rng, env.action_masks()[None, :], policy)[0]
        _, _, _, trunc, info = env.step(a)
        steps += 1
        ticks += int(info["ticks"])
        if trunc:
            episodes += 1
            env.reset()
    dt = time.perf_counter() - t0
    env.close()
    return {"steps": steps, "seconds": dt, "steps_per_s": steps / dt, "ticks": ticks, "episodes": episodes,
            "decisions_per_sim_hour": steps / max(ticks * C.DT_S / 3600.0, 1e-9)}


def bench_vec(n_envs: int, stage: int, seconds: float, warmup_s: float, policy: str, seed: int,
              graph_path: str | None, start_method: str) -> dict[str, Any]:
    t_start = time.perf_counter()
    venv = make_vec_env(n_envs, stage=stage, seed=seed, graph_path=graph_path, start_method=start_method)
    venv.reset()
    startup = time.perf_counter() - t_start
    rng = np.random.default_rng(seed)
    steps = ticks = episodes = 0
    t_measure = time.perf_counter() + warmup_s
    t0 = None
    while True:
        now = time.perf_counter()
        if t0 is None and now >= t_measure:
            t0, steps, ticks, episodes = now, 0, 0, 0
        if t0 is not None and now - t0 >= seconds:
            break
        masks = np.stack(venv.env_method("action_masks"))
        _, _, dones, infos = venv.step(sample_valid(rng, masks, policy))
        steps += n_envs
        ticks += sum(int(i["ticks"]) for i in infos)
        episodes += int(np.count_nonzero(dones))
    dt = time.perf_counter() - t0
    venv.close()
    agg = steps / dt
    return {"n_envs": n_envs, "steps": steps, "seconds": dt, "startup_s": startup, "aggregate_steps_per_s": agg,
            "per_env_steps_per_s": agg / n_envs, "ticks": ticks, "episodes": episodes,
            "sim_hours_per_s": ticks * C.DT_S / 3600.0 / dt,
            "decisions_per_sim_hour": steps / max(ticks * C.DT_S / 3600.0, 1e-9)}


def bench_chunked(n_envs: int, stage: int, seconds: float, warmup_s: float, seed: int, graph_path: str | None,
                  start_method: str, chunk: int = 64) -> dict[str, Any]:
    """Chunked worker-side-inference collection with a freshly initialized MlpPolicy."""
    import torch

    from rl.rollout import RolloutWorkers, mlp_params_from_policy
    from rl.train import build_policy
    from rl.vec import EnvFactory

    torch.manual_seed(seed)
    params = mlp_params_from_policy(build_policy())
    t_start = time.perf_counter()
    workers = RolloutWorkers(n_envs, EnvFactory(stage=stage, graph_path=graph_path, monitor=False), seed=seed,
                             start_method=start_method)
    workers.reset()
    startup = time.perf_counter() - t_start
    per_rollout = n_envs * chunk * 4
    t_measure = time.perf_counter() + warmup_s
    t0 = None
    steps = ticks = episodes = 0
    busy = 0.0
    try:
        while True:
            now = time.perf_counter()
            if t0 is None and now >= t_measure:
                t0, steps, ticks, episodes, busy = now, 0, 0, 0, 0.0
            if t0 is not None and now - t0 >= seconds:
                break
            segs = workers.collect(params, per_rollout, chunk)
            assert segs is not None
            steps += sum(len(s) for s in segs)
            ticks += sum(s.ticks for s in segs)
            episodes += sum(len(s.episodes) for s in segs)
            busy += sum(s.busy_s for s in segs)
        dt = time.perf_counter() - t0
    finally:
        workers.close()
    agg = steps / dt
    return {"n_envs": n_envs, "chunk": chunk, "steps": steps, "seconds": dt, "startup_s": startup,
            "aggregate_steps_per_s": agg, "per_env_steps_per_s": agg / n_envs, "ticks": ticks, "episodes": episodes,
            "worker_busy_frac": busy / (dt * n_envs), "sim_hours_per_s": ticks * C.DT_S / 3600.0 / dt,
            "decisions_per_sim_hour": steps / max(ticks * C.DT_S / 3600.0, 1e-9)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-envs", default="auto", help="workers (int or 'auto' = clamp(cpus-2, 2, 64))")
    ap.add_argument("--seconds", type=float, default=60.0, help="measured wall-clock seconds")
    ap.add_argument("--warmup-seconds", type=float, default=5.0, help="unmeasured seconds after startup")
    ap.add_argument("--single-seconds", type=float, default=10.0, help="in-process baseline (0 = skip)")
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    ap.add_argument("--policy", choices=("random", "continue"), default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graph", default=None, help="network.json path (default: data/network.json)")
    ap.add_argument("--start-method", default="spawn", choices=("spawn", "forkserver", "fork"))
    ap.add_argument("--mode", choices=("chunked", "subproc", "both"), default="both")
    ap.add_argument("--chunk", type=int, default=64, help="steps per chunk (chunked mode)")
    ap.add_argument("--json", type=Path, default=None, help="also write the report here")
    args = ap.parse_args(argv)

    n_envs = resolve_n_envs(args.n_envs)
    report: dict[str, Any] = {
        "stage": args.stage, "policy": args.policy, "cpu_count": cpu_count(),
        "machine": platform.machine(), "python": sys.version.split()[0],
    }
    if args.single_seconds > 0:
        report["single"] = bench_single(args.stage, args.single_seconds, args.policy, args.seed, args.graph)
        s = report["single"]
        print(f"single env (in-process): {s['steps_per_s']:.0f} steps/s, "
              f"{s['decisions_per_sim_hour']:.0f} decisions/sim-h, {s['episodes']} episodes")
    best = 0.0
    if args.mode in ("subproc", "both"):
        report["subproc"] = bench_vec(n_envs, args.stage, args.seconds, args.warmup_seconds, args.policy, args.seed,
                                      args.graph, args.start_method)
        _print_vec("SubprocVecEnv", report["subproc"], report.get("single"))
        best = max(best, report["subproc"]["aggregate_steps_per_s"])
    if args.mode in ("chunked", "both"):
        report["chunked"] = bench_chunked(n_envs, args.stage, args.seconds, args.warmup_seconds, args.seed,
                                          args.graph, args.start_method, args.chunk)
        _print_vec("RolloutWorkers (chunked, used by rl.train)", report["chunked"], report.get("single"))
        best = max(best, report["chunked"]["aggregate_steps_per_s"])
    report["target_met"] = bool(best >= 2000.0)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
    return 0


def _print_vec(name: str, v: dict[str, Any], single: dict[str, Any] | None) -> None:
    line = (f"{name} x{v['n_envs']}: aggregate {v['aggregate_steps_per_s']:.0f} env-steps/s "
            f"({v['per_env_steps_per_s']:.0f}/env), {v['sim_hours_per_s']:.2f} sim-h/s, "
            f"{v['decisions_per_sim_hour']:.0f} decisions/sim-h, {v['episodes']} episodes, startup {v['startup_s']:.1f} s")
    if single is not None:
        eff = v["aggregate_steps_per_s"] / (v["n_envs"] * single["steps_per_s"])
        v["parallel_efficiency"] = eff
        line += f", parallel efficiency {eff:.0%}"
    print(line, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
