"""Policy runners shared by the server and ``rl/evaluate.py`` (ARCHITECTURE §3.1, §4.9).

* :class:`OnnxPolicy` runs the exported actor with ONNX Runtime (CPU, one thread by
  default), sets masked logits to ``-inf`` and takes the argmax: deterministic.
* :class:`Sb3ZipPolicy` runs a MaskablePPO ``.zip`` checkpoint (local development only;
  needs ``sb3_contrib`` from ``requirements-train.txt``).

Both heavy dependencies are imported lazily inside the constructors so that importing
``sim`` stays within the import firewall (no torch/gymnasium/sb3 in production).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from sim import constants as C

log = logging.getLogger(__name__)


class Policy(Protocol):
    def act(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray: ...


def masked_argmax(logits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Argmax over valid actions (``masks`` True); rows without any valid action give 0."""
    lg = np.asarray(logits, dtype=np.float64)
    m = np.asarray(masks, dtype=bool)
    if lg.shape != m.shape:
        raise ValueError(f"logits {lg.shape} and masks {m.shape} differ")
    lg = np.where(m, lg, -np.inf)
    out = np.argmax(lg, axis=1).astype(np.int64)
    out[~m.any(axis=1)] = C.A_CONTINUE
    return out


def _check_batch(obs: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    o = np.ascontiguousarray(obs, dtype=np.float32)
    m = np.asarray(masks, dtype=bool)
    if o.ndim == 1:
        o = o[None, :]
    if m.ndim == 1:
        m = m[None, :]
    if o.shape[1] != C.OBS_DIM:
        raise ValueError(f"obs must be [n, {C.OBS_DIM}], got {o.shape}")
    if m.shape != (o.shape[0], C.N_ACTIONS):
        raise ValueError(f"masks must be [n, {C.N_ACTIONS}], got {m.shape}")
    return o, m


class OnnxPolicy:
    """Deterministic masked-argmax policy over an exported ONNX actor."""

    def __init__(self, path: str | Path, intra_op_threads: int = 1) -> None:
        import onnxruntime as ort  # lazy: only the policy runner needs it

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(str(self.path))
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(intra_op_threads)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.path), sess_options=so, providers=["CPUExecutionProvider"])
        inputs = {i.name: i for i in self.session.get_inputs()}
        outputs = {o.name: o for o in self.session.get_outputs()}
        if "obs" not in inputs or "logits" not in outputs:
            raise ValueError(f"{self.path.name}: expected input 'obs' and output 'logits', got {list(inputs)} -> {list(outputs)}")
        shp = inputs["obs"].shape
        if len(shp) != 2 or (isinstance(shp[1], int) and shp[1] != C.OBS_DIM):
            raise ValueError(f"{self.path.name}: input 'obs' has shape {shp}, expected [batch, {C.OBS_DIM}]")
        self.metadata: dict[str, str] = dict(self.session.get_modelmeta().custom_metadata_map or {})
        self.warnings: list[str] = []
        self._validate_metadata()

    def _validate_metadata(self) -> None:
        md = self.metadata
        if "obs_dim" in md and int(md["obs_dim"]) != C.OBS_DIM:
            raise ValueError(f"model obs_dim {md['obs_dim']} != {C.OBS_DIM}")
        if "n_actions" in md and int(md["n_actions"]) != C.N_ACTIONS:
            raise ValueError(f"model n_actions {md['n_actions']} != {C.N_ACTIONS}")
        if "action_names" in md:
            try:
                names = json.loads(md["action_names"])
            except json.JSONDecodeError:
                names = None
            if names is not None and list(names) != list(C.ACTION_NAMES):
                raise ValueError("model action_names differ from sim.constants.ACTION_NAMES")
        if "constants_hash" in md and md["constants_hash"] != C.constants_hash():
            self.warnings.append("constants_hash mismatch: the model was trained with different obs/mask/cost constants")
        for w in self.warnings:
            log.warning("%s: %s", self.path.name, w)

    def check_graph_version(self, graph_version: str) -> bool:
        """True when the model's ``graph_version`` matches (logs a warning otherwise)."""
        mv = self.metadata.get("graph_version")
        if mv is not None and graph_version and mv != graph_version:
            msg = f"graph_version mismatch: model {mv}, network {graph_version}"
            if msg not in self.warnings:
                self.warnings.append(msg)
                log.warning("%s: %s", self.path.name, msg)
            return False
        return True

    def logits(self, obs: np.ndarray) -> np.ndarray:
        o = np.ascontiguousarray(obs, dtype=np.float32)
        if o.ndim == 1:
            o = o[None, :]
        return np.asarray(self.session.run(["logits"], {"obs": o})[0])

    def act(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        o, m = _check_batch(obs, masks)
        if o.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)
        return masked_argmax(self.logits(o), m)


class Sb3ZipPolicy:
    """MaskablePPO checkpoint runner (local development; requires ``sb3_contrib``)."""

    def __init__(self, path: str | Path) -> None:
        try:
            from sb3_contrib import MaskablePPO  # lazy; training dependency only
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError("loading a .zip policy requires sb3_contrib (requirements-train.txt)") from exc
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(str(self.path))
        self.model: Any = MaskablePPO.load(str(self.path), device="cpu")
        self.metadata: dict[str, str] = {}
        self.warnings: list[str] = []

    def check_graph_version(self, graph_version: str) -> bool:
        return True

    def act(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        o, m = _check_batch(obs, masks)
        if o.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)
        actions, _ = self.model.predict(o, action_masks=m, deterministic=True)
        out = np.asarray(actions, dtype=np.int64).reshape(-1)
        # never trust a runner to respect masks: fall back to CONTINUE on an invalid choice
        bad = ~m[np.arange(out.size), out]
        out[bad] = C.A_CONTINUE
        return out


def load_policy(path: str | Path) -> OnnxPolicy | Sb3ZipPolicy:
    """Load ``.onnx`` (production) or ``.zip`` (local, needs sb3_contrib) lazily."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() == ".zip":
        return Sb3ZipPolicy(p)
    return OnnxPolicy(p)
