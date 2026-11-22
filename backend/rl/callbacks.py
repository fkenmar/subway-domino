"""Training callbacks and run-directory state (ARCHITECTURE §4.6 outputs, §4.10 durability).

Run directory layout (local ``runs/<name>`` or Colab ``MyDrive/subway-domino/runs/<name>``)::

    checkpoints/ckpt_<steps>.zip   every --checkpoint-every steps, atomic, the last 5 kept
    latest.zip                     the newest checkpoint (what --resume auto loads)
    best_model.zip                 best validation score so far
    train_state.json               {stage, steps_total, steps_stage, best_score, run_id, ...}
    validation_normal.json         cached Normal-Schedule results of the validation seeds
    progress.jsonl                 one compact JSON line per printed progress report
    tb/                            TensorBoard event files

Every file is written to a temporary name and renamed into place, so an interrupted Colab
runtime never leaves a truncated checkpoint behind. A checkpoint is taken at the start of a
rollout, so its weights have been trained on all of its ``num_timesteps``.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any, Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl.rollout import RolloutWorkers, mlp_params_from_policy
from sim import constants as C
from sim import eval_scenario
from sim.graph import Graph

STATE_FILE = "train_state.json"
LATEST = "latest.zip"
BEST = "best_model.zip"
CKPT_DIR = "checkpoints"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_save_model(model: Any, path: Path) -> None:
    """``model.save`` to a temporary ``.zip.tmp`` file, then rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    model.save(str(tmp))
    os.replace(tmp, path)


def atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


@dataclass
class TrainState:
    """``train_state.json`` (§4.6 contract keys first, then bookkeeping)."""

    stage: int = 1
    steps_total: int = 0
    steps_stage: int = 0
    best_score: float | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    stage_start_step: int = 0
    best_step: int | None = None
    total_steps_target: int = 0
    stage1_steps: int = 0
    stage_mode: str = "auto"
    n_envs: int = 0
    n_steps: int = 0
    batch_size: int = 0
    graph_version: str = ""
    constants_hash: str = ""
    resumes: int = 0
    created_utc: str = field(default_factory=utc_now)
    updated_utc: str = field(default_factory=utc_now)
    finished: bool = False
    reward_normalizer: dict[str, Any] | None = None  # ReturnNormalizer, or kind "per_train" (rl.credit)
    last_validation: dict[str, Any] | None = None
    credit: dict[str, Any] | None = None  # resolved WP5b credit settings (rl.train.credit_settings)

    @classmethod
    def load(cls, path: Path) -> "TrainState":
        d = json.loads(path.read_text())
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: Path) -> None:
        self.updated_utc = utc_now()
        self.steps_stage = self.steps_total - self.stage_start_step
        atomic_write_text(path, json.dumps(asdict(self), indent=2))


def stage_for(stage_mode: str, steps_total: int, stage1_steps: int) -> int:
    """Curriculum stage for ``--stage auto|1|2`` at ``steps_total``."""
    if stage_mode in ("1", "2"):
        return int(stage_mode)
    return 1 if steps_total < stage1_steps else 2


class GracefulStop:
    """First SIGINT/SIGTERM requests a checkpointed stop; a second SIGINT aborts immediately."""

    def __init__(self) -> None:
        self.requested = False
        self._old: dict[int, Any] = {}

    def install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._old[sig] = signal.signal(sig, self._handle)

    def restore(self) -> None:
        for sig, h in self._old.items():
            signal.signal(sig, h)
        self._old.clear()

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        if self.requested and signum == signal.SIGINT:
            raise KeyboardInterrupt
        self.requested = True
        print(f"\n[rl.train] {signal.Signals(signum).name}: finishing the current chunk, then checkpointing "
              "(press Ctrl+C again to abort without saving)", file=sys.stderr, flush=True)

    def is_set(self) -> bool:
        return self.requested


class RunCallback(BaseCallback):
    """Base class with access to the run directory and its :class:`TrainState`."""

    def __init__(self, run_dir: Path, state: TrainState) -> None:
        super().__init__(verbose=0)
        self.run_dir = Path(run_dir)
        self.state = state

    def _on_step(self) -> bool:
        return True

    def save_state(self) -> None:
        self.state.steps_total = int(self.model.num_timesteps)
        norm = getattr(self.model, "reward_normalizer", None)
        self.state.reward_normalizer = None if norm is None else norm.state_dict()
        self.state.save(self.run_dir / STATE_FILE)


class CurriculumCallback(RunCallback):
    """Stage 1 -> 2 once ``steps_total >= stage1_steps`` (``--stage auto``)."""

    def _on_training_start(self) -> None:
        self.model.logger.record("curriculum/stage", self.state.stage)

    def _on_rollout_end(self) -> None:
        stage = stage_for(self.state.stage_mode, int(self.model.num_timesteps), self.state.stage1_steps)
        if stage != self.state.stage:
            self.training_env.env_method("set_stage", stage)
            self.state.stage = stage
            self.state.stage_start_step = int(self.model.num_timesteps)
            print(f"[rl.train] curriculum: stage {stage} from step {self.model.num_timesteps:,} "
                  "(takes effect at each worker's next episode)", flush=True)
            self.save_state()
        self.model.logger.record("curriculum/stage", self.state.stage)


class EntropyScheduleCallback(BaseCallback):
    """Linear ``ent_coef`` from ``start`` to ``end`` over the whole run (SB3 PPO takes a float)."""

    def __init__(self, start: float, end: float) -> None:
        super().__init__(verbose=0)
        self.start, self.end = float(start), float(end)

    def _value(self) -> float:
        total = max(int(getattr(self.model, "_total_timesteps", 0)), 1)
        progress_remaining = 1.0 - min(1.0, self.model.num_timesteps / total)
        return self.end + (self.start - self.end) * progress_remaining

    def _on_training_start(self) -> None:
        self.model.ent_coef = self._value()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.model.ent_coef = self._value()
        self.model.logger.record("train/ent_coef", self.model.ent_coef)


class CheckpointCallback(RunCallback):
    """Atomic ``checkpoints/ckpt_<steps>.zip`` + ``latest.zip`` + ``train_state.json``."""

    def __init__(self, run_dir: Path, state: TrainState, every: int = 250_000, keep: int = 5) -> None:
        super().__init__(run_dir, state)
        self.every, self.keep = int(every), int(keep)
        self._next = 0

    def _on_training_start(self) -> None:
        self._next = (int(self.model.num_timesteps) // self.every + 1) * self.every

    def _on_rollout_start(self) -> None:
        if self.model.num_timesteps >= self._next:
            self.save()
            self._next = (int(self.model.num_timesteps) // self.every + 1) * self.every

    def _on_training_end(self) -> None:
        self.state.finished = bool(self.model.num_timesteps >= self.state.total_steps_target)
        self.save()

    def save(self) -> Path:
        steps = int(self.model.num_timesteps)
        ckpt = self.run_dir / CKPT_DIR / f"ckpt_{steps}.zip"
        if ckpt.is_file() and (self.run_dir / LATEST).is_file() and steps == getattr(self, "_saved_step", None):
            self.save_state()  # nothing new since the last checkpoint (e.g. Ctrl+C right after it)
            return ckpt
        self._saved_step = steps
        atomic_save_model(self.model, ckpt)
        atomic_copy(ckpt, self.run_dir / LATEST)
        self._prune()
        self.save_state()
        print(f"[rl.train] checkpoint {ckpt.name} (latest.zip updated)", flush=True)
        return ckpt

    def _prune(self) -> None:
        def step_of(p: Path) -> int:
            try:
                return int(p.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                return -1

        ckpts = sorted((p for p in (self.run_dir / CKPT_DIR).glob("ckpt_*.zip")), key=step_of)
        for p in ckpts[: max(0, len(ckpts) - self.keep)]:
            p.unlink(missing_ok=True)


class ValidationCallback(RunCallback):
    """Every ``every`` steps: deterministic policy on the 8 fixed validation seeds (§4.6).

    The score is the mean twin-relative cost reduction ``(C_normal - C_ai) / C_normal`` over
    ``eval_scenario(VALIDATION_SEED_BASE + i)``. The episodes run in parallel on the rollout
    workers, which are idle between rollouts. Normal results are cached in
    ``validation_normal.json``, keyed by graph version and constants hash. A new best is
    saved atomically to ``best_model.zip``.
    """

    def __init__(self, run_dir: Path, state: TrainState, graph: Graph, every: int = 500_000, n_seeds: int = 8,
                 should_skip_final: Callable[[], bool] | None = None) -> None:
        super().__init__(run_dir, state)
        self.graph, self.every, self.n_seeds = graph, int(every), int(n_seeds)
        self.should_skip_final = should_skip_final
        self._next = 0
        self._last_step = -1

    def _on_training_start(self) -> None:
        self._next = (int(self.model.num_timesteps) // self.every + 1) * self.every

    def _on_rollout_start(self) -> None:
        if self.model.num_timesteps >= self._next:
            self.validate()
            self._next = (int(self.model.num_timesteps) // self.every + 1) * self.every

    def _on_training_end(self) -> None:
        skip = self.should_skip_final() if self.should_skip_final is not None else False
        if not skip and self.model.num_timesteps != self._last_step:
            self.validate()

    def _scenarios(self) -> list[Any]:
        return [eval_scenario(C.VALIDATION_SEED_BASE + i, self.graph) for i in range(self.n_seeds)]

    def _normal_results(self, workers: RolloutWorkers, scenarios: list[Any]) -> list[dict[str, Any]]:
        path = self.run_dir / "validation_normal.json"
        key = f"{self.graph.graph_version}:{C.constants_hash()}"
        cache: dict[str, Any] = {}
        if path.is_file():
            try:
                cache = json.loads(path.read_text())
            except json.JSONDecodeError:
                cache = {}
        rows = cache.get(key, {})
        missing = [sc for sc in scenarios if str(sc.seed) not in rows]
        if missing:
            for sc, res in zip(missing, workers.run_episodes(None, missing)):
                rows[str(sc.seed)] = res
            atomic_write_text(path, json.dumps({key: rows}))
        return [rows[str(sc.seed)] for sc in scenarios]

    def validate(self) -> dict[str, Any]:
        workers = self.training_env
        if not isinstance(workers, RolloutWorkers):
            raise TypeError("validation runs on RolloutWorkers")
        t0 = time.perf_counter()
        scenarios = self._scenarios()
        normal = self._normal_results(workers, scenarios)
        ai = workers.run_episodes(mlp_params_from_policy(self.model.policy), scenarios)
        rel = np.array([(n["cost"]["total"] - a["cost"]["total"]) / max(n["cost"]["total"], 1e-9)
                        for n, a in zip(normal, ai)])

        def red(key: str) -> float:
            num = sum(n[key] for n in normal)
            return 0.0 if num <= 0 else 1.0 - sum(a[key] for a in ai) / num

        result = {
            "step": int(self.model.num_timesteps),
            "score": float(rel.mean()),
            "win_rate": float(np.mean(rel > 0)),
            "per_seed": [round(float(x), 5) for x in rel],
            "stuck_reduction": red("tunnel_stuck_train_min"),
            "wait_reduction": red("avg_wait_min"),
            "overcrowd_reduction": red("overcrowded_platform_min"),
            "riders_served_gain": -red("riders_served"),
            "actions_per_episode": float(np.mean([sum(a["actions"].values()) for a in ai])),
            "seconds": time.perf_counter() - t0,
        }
        rec = self.model.logger.record
        for k in ("score", "win_rate", "stuck_reduction", "wait_reduction", "overcrowd_reduction",
                  "riders_served_gain", "actions_per_episode"):
            rec(f"validation/{k}", result[k])
        improved = self.state.best_score is None or result["score"] > self.state.best_score
        if improved:
            atomic_save_model(self.model, self.run_dir / BEST)
            self.state.best_score = result["score"]
            self.state.best_step = result["step"]
        result["best"] = improved
        self.state.last_validation = result
        self._last_step = result["step"]
        self.save_state()
        print(f"[rl.train] validation @ {result['step']:,}: score {100 * result['score']:+.2f} % "
              f"(win {100 * result['win_rate']:.0f} %, stuck {100 * result['stuck_reduction']:+.1f} %, "
              f"wait {100 * result['wait_reduction']:+.1f} %){' NEW BEST' if improved else ''} "
              f"[{result['seconds']:.0f} s]", flush=True)
        return result


class ProgressCallback(RunCallback):
    """A compact progress line every ``every_steps`` (stdout + ``progress.jsonl``)."""

    def __init__(self, run_dir: Path, state: TrainState, every_steps: int = 100_000) -> None:
        super().__init__(run_dir, state)
        self.every_steps = int(every_steps)
        self._next = 0
        self._t0 = time.perf_counter()
        self._s0 = 0

    def _on_training_start(self) -> None:
        self._t0, self._s0 = time.perf_counter(), int(self.model.num_timesteps)
        self._next = (self._s0 // self.every_steps + 1) * self.every_steps

    def _on_rollout_end(self) -> None:
        steps = int(self.model.num_timesteps)
        if steps < self._next:
            return
        self._next = (steps // self.every_steps + 1) * self.every_steps
        v = self.model.logger.name_to_value
        last = getattr(self.model, "last_rollout", {}) or {}
        eps = self.model.ep_info_buffer
        def num(x: Any) -> float | None:
            return None if x is None else float(x)

        line = {
            "step": steps, "stage": self.state.stage, "utc": utc_now(),
            "fps": (steps - self._s0) / max(time.perf_counter() - self._t0, 1e-9),
            "collect_fps": last.get("steps", 0) / max(last.get("collect_s", 1e-9), 1e-9),
            "ep_rew_mean": float(np.mean([e["r"] for e in eps])) if eps else None,
            "twin_adv_rel": num(v.get("twin/advantage_rel_mean")),
            "noncontinue_rate": num(v.get("actions/noncontinue_rate")),
            "entropy": num(-v["train/entropy_loss"]) if "train/entropy_loss" in v else None,
            "explained_variance": num(v.get("train/explained_variance")),
            "approx_kl": num(v.get("train/approx_kl")),
        }
        with (self.run_dir / "progress.jsonl").open("a") as fh:
            fh.write(json.dumps(line) + "\n")

        def f(x: Any, spec: str) -> str:
            return "n/a" if x is None else format(x, spec)

        total = max(self.state.total_steps_target, 1)
        print(f"[rl.train] {steps:>11,} / {total:,} ({100 * steps / total:5.1f} %) stage {line['stage']} | "
              f"{line['fps']:,.0f} steps/s | twin adv {f(line['twin_adv_rel'] and 100 * line['twin_adv_rel'], '+.2f')} % | "
              f"act {f(line['noncontinue_rate'] and 100 * line['noncontinue_rate'], '.1f')} % | "
              f"H {f(line['entropy'], '.3f')} | EV {f(line['explained_variance'], '.3f')}", flush=True)
