"""Export the MaskablePPO actor to ONNX (ARCHITECTURE §4.9) and write the model card.

    .venv/bin/python -m rl.export_onnx --checkpoint runs/local/best_model.zip \
        --out models/nyc_subway_rcc_ppo.onnx

* Actor only: ``obs -> features -> mlp_extractor.forward_actor -> action_net`` = ``logits``.
  Masking and argmax happen in ``sim.policy.OnnxPolicy``.
* One self-contained file at opset 18: the dynamo exporter with ``external_data=False``, or the
  legacy exporter (``dynamo=False``) as a fallback.
* I/O: ``"obs"`` float32 ``[batch, OBS_DIM]`` (dynamic batch) -> ``"logits"`` float32 ``[batch, 8]``.
* Metadata props: ``obs_dim``, ``n_actions``, ``action_names`` (JSON), ``graph_version``,
  ``constants_hash``, ``trained_steps``, ``created_utc`` (plus ``mask_revision`` and ``producer``).
* Parity gate: on 1,000 observations recorded from real episodes, the max abs logit difference
  vs SB3 must be <= 1e-4 and the masked argmax must equal ``MaskablePPO.predict(deterministic=True)``
  for every row. Otherwise nothing is written and the exit code is 1.
* Model card ``<out>.json``: hyperparameters, steps, curriculum and run state, reward and mask
  coefficients, parity results, and the eval summary when an ``eval_report.json`` is available.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO

from rl.env import SubwayDominoEnv
from rl.evaluate import evaluation_summary
from rl.rollout import NumpyActorCritic, mlp_params_from_policy, sample_masked
from sim import constants as C
from sim import load_graph
from sim.policy import masked_argmax

OPSET = 18
PARITY_ATOL = 1e-4
N_PARITY_OBS = 1_000


class ActorLogits(nn.Module):
    """The deterministic actor head of an SB3 ``MaskableActorCriticPolicy``: obs -> logits."""

    def __init__(self, policy: Any) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        p = self.policy
        features = p.extract_features(obs, p.pi_features_extractor)
        return p.action_net(p.mlp_extractor.forward_actor(features))


def record_observations(model: MaskablePPO, n: int = N_PARITY_OBS, seed: int = 0,
                        graph_path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``n`` (obs, mask) pairs from real episodes (both stages), acting with the policy itself."""
    net = NumpyActorCritic(mlp_params_from_policy(model.policy))
    rng = np.random.default_rng(seed)
    obs_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    for k, stage in enumerate((2, 1, 2, 1, 2, 1, 2, 1)):
        env = SubwayDominoEnv(stage=stage, seed=seed + k, graph_path=graph_path)
        obs, _ = env.reset(seed=seed + k)
        per_env = (n - len(obs_rows) + 1) // 2 if k < 7 else n - len(obs_rows)
        for _ in range(per_env):
            mask = env.action_masks()
            obs_rows.append(obs)
            mask_rows.append(mask)
            obs, _, _, trunc, _ = env.step(sample_masked(net.logits(obs), mask, rng)[0])
            if trunc:
                obs, _ = env.reset()
        env.close()
        if len(obs_rows) >= n:
            break
    return np.stack(obs_rows[:n]).astype(np.float32), np.stack(mask_rows[:n])


def export_actor(model: MaskablePPO, path: Path) -> str:
    """Write the single-file ONNX graph; returns the exporter used."""
    actor = ActorLogits(model.policy).eval()
    dummy = torch.zeros(2, C.OBS_DIM, dtype=torch.float32)
    logging.getLogger("torch.onnx._internal.exporter._registration").setLevel(logging.ERROR)  # torchvision noise
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(actor, (dummy,), str(path), input_names=["obs"], output_names=["logits"],
                              opset_version=OPSET, dynamo=True, external_data=False, verbose=False,
                              dynamic_shapes={"obs": {0: torch.export.Dim("batch", min=1, max=65536)}})
            return "dynamo"
        except Exception as exc:  # pragma: no cover - depends on the torch build
            print(f"dynamo export failed ({type(exc).__name__}: {exc}); using the legacy exporter", file=sys.stderr)
            torch.onnx.export(actor, (dummy,), str(path), input_names=["obs"], output_names=["logits"],
                              opset_version=OPSET, dynamo=False,
                              dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}})
            return "legacy"


def add_metadata(path: Path, props: dict[str, str]) -> None:
    import onnx

    m = onnx.load(str(path))
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in m.graph.initializer):
        raise RuntimeError("the exported model uses external data; a single self-contained file is required")
    del m.metadata_props[:]
    for k, v in props.items():
        e = m.metadata_props.add()
        e.key, e.value = k, str(v)
    onnx.checker.check_model(m)
    onnx.save_model(m, str(path), save_as_external_data=False)


def parity_check(model: MaskablePPO, onnx_path: Path, obs: np.ndarray, masks: np.ndarray) -> dict[str, Any]:
    """ONNX vs SB3: max |logit diff| and masked-argmax agreement with ``predict(deterministic=True)``."""
    from sim.policy import OnnxPolicy

    pol = OnnxPolicy(onnx_path)
    lg_onnx = pol.logits(obs)
    with torch.no_grad():
        lg_sb3 = ActorLogits(model.policy).eval()(torch.as_tensor(obs)).numpy()
    sb3_actions, _ = model.predict(obs, action_masks=masks, deterministic=True)
    onnx_actions = masked_argmax(lg_onnx, masks)
    max_diff = float(np.max(np.abs(lg_onnx - lg_sb3)))
    agree = float(np.mean(onnx_actions == np.asarray(sb3_actions).reshape(-1)))
    return {"n_obs": int(obs.shape[0]), "max_abs_logit_diff": max_diff, "argmax_agreement": agree,
            "atol": PARITY_ATOL, "batch_dynamic_ok": bool(np.allclose(pol.logits(obs[:1]), lg_onnx[:1], atol=1e-6)),
            "passed": bool(max_diff <= PARITY_ATOL and agree == 1.0),
            "metadata_warnings": list(pol.warnings)}


def _reward_coefficients() -> dict[str, float]:
    names = ["W_WAIT", "WAIT_AGE_EXPONENT", "W_TUNNEL_TRAIN", "W_TUNNEL_RIDER", "TUNNEL_RIDER_AGE_EXPONENT",
             "W_HOLD_TRAIN", "W_HOLD_RIDER", "W_CROWD", "CROWD_THRESHOLD", "CROWD_SPAN", "W_ST", "W_SKIP",
             "W_REROUTE", "W_REROUTE_RIDER", "W_YARD", "REWARD_SCALE", "REWARD_CLIP"]
    return {k: float(getattr(C, k)) for k in names}


def _hyperparameters(model: MaskablePPO) -> dict[str, Any]:
    from rl.train import HYPERPARAMS, POLICY_KWARGS

    return {
        "algorithm": "MaskablePPO (sb3-contrib), chunked worker-side-inference rollouts (rl.ppo)",
        "policy": "MlpPolicy", "net_arch": POLICY_KWARGS["net_arch"], "activation_fn": "Tanh",
        "n_steps_per_worker": int(model.n_steps), "n_workers": int(model.n_envs),
        "batch_size": int(model.batch_size), "n_epochs": int(model.n_epochs), "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda), "clip_range": HYPERPARAMS["clip_range"],
        "max_grad_norm": float(model.max_grad_norm),
        "learning_rate": f"linear {HYPERPARAMS['learning_rate_start']} -> {HYPERPARAMS['learning_rate_end']}",
        "ent_coef": f"linear {HYPERPARAMS['ent_coef_start']} -> {HYPERPARAMS['ent_coef_end']}",
        "reward_normalization": bool(getattr(model, "normalize_reward", False)),
        "episode_horizon": "terminal (no bootstrap; time_in_episode_frac is observed)",
    }


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def build_model_card(model: MaskablePPO, checkpoint: Path, onnx_path: Path, meta: dict[str, str],
                     parity: dict[str, Any], exporter: str, eval_report: dict[str, Any] | None,
                     train_state: dict[str, Any] | None) -> dict[str, Any]:
    card: dict[str, Any] = {
        "name": onnx_path.stem,
        "description": "Subway Domino RCC dispatcher: masked actor logits over 8 dispatch actions per decision.",
        "created_utc": meta["created_utc"],
        "checkpoint": str(checkpoint),
        "trained_steps": int(meta["trained_steps"]),
        "graph_version": meta["graph_version"],
        "constants_hash": meta["constants_hash"],
        "mask_revision": C.MASK_REVISION,
        "io": {"input": {"name": "obs", "dtype": "float32", "shape": ["batch", C.OBS_DIM]},
               "output": {"name": "logits", "dtype": "float32", "shape": ["batch", C.N_ACTIONS]},
               "opset": OPSET, "exporter": exporter, "decision_rule": "argmax over valid actions (masks)"},
        "action_names": list(C.ACTION_NAMES),
        "hyperparameters": _hyperparameters(model),
        "curriculum": {"stage1": "one trunk group, 1-2 disruptions, headway jitter, 3600 s episodes",
                       "stage2": "citywide, 1-4 disruptions, domain randomization, 5400 s episodes"},
        "reward": {"formula": "clip(-REWARD_SCALE * (dC_ai - dC_twin), +-REWARD_CLIP), twin = Normal Schedule",
                   "coefficients": _reward_coefficients()},
        "parity": parity,
    }
    if train_state is not None:
        card["training_run"] = {k: train_state.get(k) for k in (
            "run_id", "stage", "steps_total", "steps_stage", "best_score", "best_step", "stage1_steps",
            "total_steps_target", "n_envs", "n_steps", "batch_size", "resumes", "created_utc", "updated_utc",
            "last_validation")}
    if eval_report is not None:
        card["evaluation"] = evaluation_summary(eval_report, eval_report.get("_path"))
    return card


def export(checkpoint: Path, out: Path, *, n_parity: int = N_PARITY_OBS, seed: int = 0, graph_path: str | None = None,
           eval_report: Path | None = None) -> dict[str, Any]:
    model = MaskablePPO.load(str(checkpoint), device="cpu")
    graph = load_graph(graph_path)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {
        "obs_dim": str(C.OBS_DIM), "n_actions": str(C.N_ACTIONS), "action_names": json.dumps(list(C.ACTION_NAMES)),
        "graph_version": graph.graph_version, "constants_hash": C.constants_hash(),
        "trained_steps": str(int(model.num_timesteps)), "created_utc": created,
        "mask_revision": str(C.MASK_REVISION), "producer": "subway-domino rl.export_onnx",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        tmp_onnx = Path(tmp) / out.name
        exporter = export_actor(model, tmp_onnx)
        add_metadata(tmp_onnx, meta)
        obs, masks = record_observations(model, n_parity, seed, graph_path)
        parity = parity_check(model, tmp_onnx, obs, masks)
        if not parity["passed"]:
            raise RuntimeError(f"ONNX parity check failed: {parity}")
        er_path = eval_report or (out.parent / "eval_report.json")
        er = _read_json(er_path)
        if er is not None:
            er["_path"] = str(er_path)
        ts = _read_json(checkpoint.parent / "train_state.json")
        card = build_model_card(model, checkpoint, out, meta, parity, exporter, er, ts)
        shutil.move(str(tmp_onnx), str(out))
    card_path = out.with_suffix(".json")
    tmp_card = card_path.with_name(card_path.name + ".tmp")
    tmp_card.write_text(json.dumps(card, indent=2))
    os.replace(tmp_card, card_path)
    return {"onnx": str(out), "model_card": str(card_path), "parity": parity, "exporter": exporter,
            "size_bytes": out.stat().st_size, "trained_steps": int(model.num_timesteps)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="MaskablePPO .zip (e.g. best_model.zip)")
    ap.add_argument("--out", type=Path, default=Path("models/nyc_subway_rcc_ppo.onnx"))
    ap.add_argument("--parity-obs", type=int, default=N_PARITY_OBS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graph", default=None, help="network.json (default: data/network.json)")
    ap.add_argument("--eval-report", type=Path, default=None,
                    help="eval_report.json for the model card (default: next to --out, if present)")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    try:
        res = export(args.checkpoint, args.out, n_parity=args.parity_obs, seed=args.seed, graph_path=args.graph,
                     eval_report=args.eval_report)
    except RuntimeError as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 1
    p = res["parity"]
    print(f"wrote {res['onnx']} ({res['size_bytes'] / 1024:.0f} KiB, {res['exporter']} exporter, "
          f"{res['trained_steps']:,} steps) and {res['model_card']}")
    print(f"parity on {p['n_obs']} recorded obs: max |dlogit| {p['max_abs_logit_diff']:.2e} (<= {PARITY_ATOL}), "
          f"masked-argmax agreement {100 * p['argmax_agreement']:.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
