"""MaskablePPO training for Subway Domino (ARCHITECTURE §4.6, Colab §4.10).

    .venv/bin/python -m rl.train --run-dir runs/local --n-envs auto --resume auto

* Algorithm: MaskablePPO (:class:`rl.ppo.ChunkedMaskablePPO`, same update and saved format),
  ``MlpPolicy`` with ``net_arch=dict(pi=[256, 256], vf=[256, 256])`` and Tanh, the §4.6
  hyperparameters, linear learning-rate (3e-4 -> 5e-5) and entropy (0.01 -> 0.001) schedules
  over the whole run.
* Parallelism (§4.10): ``--n-envs auto`` = clamp(cpus - 2, 2, 64) env workers;
  ``--n-steps auto`` keeps the rollout near 15,360 steps;
  learner threads = clamp(cpus // 8, 1, 8).
* Curriculum: ``--stage auto`` trains Stage 1 (trunk school) for ``--stage1-steps``, then
  Stage 2 (citywide, domain randomization).
* Durability: atomic checkpoints every 250k steps (the last 5 kept), ``latest.zip``,
  ``best_model.zip`` (validated every 500k steps on 8 fixed seeds) and ``train_state.json``.
  ``--resume auto`` continues from ``latest.zip``, so re-running the same command after a
  Colab disconnect loses at most one checkpoint interval. Ctrl+C or SIGTERM checkpoints
  before exiting.
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
import torch
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import LinearSchedule

from rl.callbacks import (
    LATEST,
    STATE_FILE,
    CheckpointCallback,
    CurriculumCallback,
    EntropyScheduleCallback,
    GracefulStop,
    ProgressCallback,
    TrainState,
    ValidationCallback,
    stage_for,
)
from rl.ppo import ChunkedMaskablePPO
from rl.rollout import RolloutWorkers
from rl.vec import EnvFactory, cpu_count, resolve_batch_size, resolve_n_envs, resolve_n_steps, resolve_torch_threads
from sim import constants as C
from sim import load_graph

#: §4.6 network
POLICY_KWARGS: dict[str, Any] = {"net_arch": {"pi": [256, 256], "vf": [256, 256]}, "activation_fn": nn.Tanh}
#: §4.6 hyperparameters (n_steps / n_envs / batch_size are resolved at run time)
HYPERPARAMS: dict[str, Any] = {
    "n_epochs": 5,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "max_grad_norm": 0.5,
    "learning_rate_start": 3e-4,
    "learning_rate_end": 5e-5,
    "ent_coef_start": 0.01,
    "ent_coef_end": 0.001,
    "batch_size_target": 2048,
}
STAGE1_STEPS: int = 3_000_000
STAGE2_STEPS: int = 10_000_000


def build_policy(seed: int | None = None) -> MaskableActorCriticPolicy:
    """A freshly initialized §4.6 policy (benchmarks and tests)."""
    from gymnasium import spaces

    if seed is not None:
        torch.manual_seed(seed)
    return MaskableActorCriticPolicy(
        spaces.Box(-1.0, 1.0, (C.OBS_DIM,), dtype=np.float32), spaces.Discrete(C.N_ACTIONS),
        lr_schedule=lambda _: HYPERPARAMS["learning_rate_start"], **POLICY_KWARGS,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="outputs (checkpoints, logs, state)")
    ap.add_argument("--n-envs", default="auto", help="env workers: int or 'auto' (clamp(cpus-2, 2, 64))")
    ap.add_argument("--n-steps", default="auto", help="steps per worker per rollout: int or 'auto' (~15,360 total)")
    ap.add_argument("--batch-size", default="auto", help="mini-batch: int or 'auto' (divisor of the rollout closest to 2048)")
    ap.add_argument("--total-steps", type=int, default=STAGE1_STEPS + STAGE2_STEPS, help="total env steps of the run")
    ap.add_argument("--stage", choices=("auto", "1", "2"), default="auto", help="curriculum stage")
    ap.add_argument("--stage1-steps", type=int, default=STAGE1_STEPS, help="Stage 1 length with --stage auto")
    ap.add_argument("--resume", choices=("auto", "never"), default="auto",
                    help="auto: continue from <run-dir>/latest.zip if present")
    ap.add_argument("--force", action="store_true", help="with --resume never: overwrite an existing run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=250_000)
    ap.add_argument("--keep-checkpoints", type=int, default=5)
    ap.add_argument("--validate-every", type=int, default=500_000)
    ap.add_argument("--validation-seeds", type=int, default=8)
    ap.add_argument("--print-every", type=int, default=100_000, help="progress line every N steps")
    ap.add_argument("--torch-threads", default="auto", help="learner threads: int or 'auto' (clamp(cpus//8, 1, 8))")
    ap.add_argument("--chunk", type=int, default=64, help="steps per worker chunk")
    ap.add_argument("--no-reward-norm", action="store_true", help="disable return-based reward normalization")
    ap.add_argument("--graph", default=None, help="network.json (default: data/network.json)")
    ap.add_argument("--start-method", default="spawn", choices=("spawn", "forkserver", "fork"))
    ap.add_argument("--device", default="cpu", help="torch device for the learner (cpu is fastest for this MLP)")
    return ap.parse_args(argv)


def _new_model(args: argparse.Namespace, workers: RolloutWorkers, n_steps: int, batch: int) -> ChunkedMaskablePPO:
    hp = HYPERPARAMS
    return ChunkedMaskablePPO(
        "MlpPolicy",
        workers,
        learning_rate=LinearSchedule(hp["learning_rate_start"], hp["learning_rate_end"], 1.0),
        n_steps=n_steps,
        batch_size=batch,
        n_epochs=hp["n_epochs"],
        gamma=hp["gamma"],
        gae_lambda=hp["gae_lambda"],
        clip_range=hp["clip_range"],
        ent_coef=hp["ent_coef_start"],
        max_grad_norm=hp["max_grad_norm"],
        policy_kwargs=POLICY_KWARGS,
        seed=args.seed,
        device=args.device,
        verbose=0,
        rollout_chunk=args.chunk,
        normalize_reward=not args.no_reward_norm,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path, latest = run_dir / STATE_FILE, run_dir / LATEST

    cpus = cpu_count()
    n_envs = resolve_n_envs(args.n_envs)
    n_steps = resolve_n_steps(args.n_steps, n_envs)
    batch = resolve_batch_size(n_envs * n_steps, HYPERPARAMS["batch_size_target"]) if str(args.batch_size) == "auto" \
        else int(args.batch_size)
    threads = resolve_torch_threads(args.torch_threads)
    torch.set_num_threads(threads)

    resume = args.resume == "auto" and latest.is_file() and state_path.is_file()
    if args.resume == "never" and latest.exists() and not args.force:
        print(f"{latest} exists: use --resume auto to continue, or --force to start over", file=sys.stderr)
        return 2
    graph = load_graph(args.graph)
    if resume:
        state = TrainState.load(state_path)
        state.resumes += 1
    else:
        state = TrainState(stage_mode=args.stage, stage1_steps=args.stage1_steps)
    state.total_steps_target = int(args.total_steps)
    state.stage_mode, state.stage1_steps = args.stage, int(args.stage1_steps)
    state.n_envs, state.n_steps, state.batch_size = n_envs, n_steps, batch
    state.graph_version, state.constants_hash = graph.graph_version, C.constants_hash()
    if state.steps_total >= args.total_steps:
        print(f"[rl.train] {run_dir} already reached {state.steps_total:,} >= {args.total_steps:,} steps; nothing to do")
        return 0
    stage = stage_for(args.stage, state.steps_total, args.stage1_steps)
    if stage != state.stage:
        state.stage, state.stage_start_step = stage, state.steps_total

    print(f"[rl.train] run {state.run_id} in {run_dir} | {'resume' if resume else 'new'} at {state.steps_total:,} "
          f"steps | stage {stage} ({args.stage}) | cpus {cpus}: {n_envs} workers x {n_steps} steps "
          f"(rollout {n_envs * n_steps:,}, batch {batch}), torch threads {threads} | Python "
          f"{platform.python_version()} torch {torch.__version__} | graph {graph.graph_version}", flush=True)

    stop = GracefulStop()
    workers = RolloutWorkers(n_envs, EnvFactory(stage=stage, graph_path=args.graph, monitor=False),
                             seed=args.seed + 7919 * state.resumes + state.steps_total, start_method=args.start_method)
    try:
        if resume:
            model = ChunkedMaskablePPO.load(str(latest), env=workers, device=args.device, n_steps=n_steps,
                                            batch_size=batch, rollout_chunk=args.chunk, verbose=0)
            if model.reward_normalizer is not None and state.reward_normalizer:
                model.reward_normalizer.load_state_dict(state.reward_normalizer)
        else:
            model = _new_model(args, workers, n_steps, batch)
        model.set_logger(configure(str(run_dir / "tb"), ["tensorboard"]))
        model.should_stop = stop.is_set
        callbacks = CallbackList([
            CurriculumCallback(run_dir, state),
            EntropyScheduleCallback(HYPERPARAMS["ent_coef_start"], HYPERPARAMS["ent_coef_end"]),
            ValidationCallback(run_dir, state, graph, every=args.validate_every, n_seeds=args.validation_seeds,
                               should_skip_final=stop.is_set),
            CheckpointCallback(run_dir, state, every=args.checkpoint_every, keep=args.keep_checkpoints),
            ProgressCallback(run_dir, state, every_steps=args.print_every),
        ])
        stop.install()
        t0 = time.perf_counter()
        remaining = int(args.total_steps) - int(model.num_timesteps)
        model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=not resume,
                    tb_log_name="MaskablePPO", log_interval=1, progress_bar=False)
        dt = time.perf_counter() - t0
    finally:
        stop.restore()
        workers.close()
    done = int(model.num_timesteps)
    print(f"[rl.train] {'stopped' if stop.requested else 'finished'} at {done:,} steps after {dt / 60:.1f} min; "
          f"best validation score {state.best_score if state.best_score is None else f'{100 * state.best_score:+.2f} %'}"
          f" (step {state.best_step}); state: {json.dumps({k: getattr(state, k) for k in ('stage', 'steps_total', 'steps_stage', 'best_score', 'run_id')})}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
