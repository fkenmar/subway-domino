"""ONNX export parity and shipped-model metadata (ARCHITECTURE §4.9, §8).

* ``model`` marker: the shipped ``models/nyc_subway_rcc_ppo.onnx`` has the I/O and metadata
  contract, matches the current network and constants, and has a model card. Parity against
  its training checkpoint runs when that checkpoint is available (``SUBWAY_DOMINO_CHECKPOINT``
  or the model card's ``checkpoint``). Skips cleanly while no model is shipped.
* ``train`` marker: an export round trip of a freshly trained toy policy (needs torch / sb3).
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from sim import constants as C
from sim.graph import data_dir, load_graph
from sim.policy import OnnxPolicy, masked_argmax
from sim.testing import toy_network_dict

MODEL = Path(os.environ.get("SUBWAY_DOMINO_MODEL", Path(__file__).resolve().parents[1] / "models" / "nyc_subway_rcc_ppo.onnx"))
REQUIRED_META = ("obs_dim", "n_actions", "action_names", "graph_version", "constants_hash", "trained_steps", "created_utc")


def _check_onnx_contract(path: Path) -> OnnxPolicy:
    try:
        import onnx  # graph-proto checks need the `onnx` package (requirements-train.txt only)
    except ModuleNotFoundError:
        onnx = None
    if onnx is not None:
        m = onnx.load(str(path))
        assert [o.version for o in m.opset_import if o.domain in ("", "ai.onnx")] == [18]
        assert not any(t.data_location == onnx.TensorProto.EXTERNAL for t in m.graph.initializer)
        assert [i.name for i in m.graph.input] == ["obs"] and [o.name for o in m.graph.output] == ["logits"]
        dims = m.graph.input[0].type.tensor_type.shape.dim
        assert dims[0].dim_param and dims[1].dim_value == C.OBS_DIM  # dynamic batch
    pol = OnnxPolicy(path)
    md = pol.metadata
    assert all(k in md for k in REQUIRED_META), md
    assert int(md["obs_dim"]) == C.OBS_DIM and int(md["n_actions"]) == C.N_ACTIONS
    assert json.loads(md["action_names"]) == list(C.ACTION_NAMES)
    rng = np.random.default_rng(0)
    for n in (1, 37):
        obs = rng.uniform(-1, 1, (n, C.OBS_DIM)).astype(np.float32)
        lg = pol.logits(obs)
        assert lg.shape == (n, C.N_ACTIONS) and lg.dtype == np.float32 and np.isfinite(lg).all()
    return pol


@pytest.mark.model
def test_shipped_model_contract() -> None:
    if not MODEL.is_file():
        pytest.skip(f"{MODEL} not shipped yet (WP5 training run)")
    pol = _check_onnx_contract(MODEL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = load_graph(data_dir() / "network.json")
    assert pol.metadata["graph_version"] == graph.graph_version
    assert pol.metadata["constants_hash"] == C.constants_hash(), "model trained against other obs/mask/cost constants"
    assert int(pol.metadata["trained_steps"]) > 0
    card = json.loads(MODEL.with_suffix(".json").read_text())
    assert card["parity"]["passed"] and card["parity"]["max_abs_logit_diff"] <= 1e-4
    assert card["constants_hash"] == pol.metadata["constants_hash"]


@pytest.mark.model
def test_shipped_model_parity_vs_checkpoint() -> None:
    if not MODEL.is_file():
        pytest.skip(f"{MODEL} not shipped yet")
    pytest.importorskip("sb3_contrib")
    card = json.loads(MODEL.with_suffix(".json").read_text())
    ckpt = Path(os.environ.get("SUBWAY_DOMINO_CHECKPOINT", card.get("checkpoint", "")))
    if not ckpt.is_absolute():
        ckpt = Path(__file__).resolve().parents[1] / ckpt  # relative to backend/
    if not ckpt.is_file():
        pytest.skip(f"training checkpoint {ckpt} not available")
    from sb3_contrib import MaskablePPO

    from rl.export_onnx import parity_check, record_observations

    model = MaskablePPO.load(str(ckpt), device="cpu")
    obs, masks = record_observations(model, 1_000, seed=1)
    res = parity_check(model, MODEL, obs, masks)
    assert res["max_abs_logit_diff"] <= 1e-4 and res["argmax_agreement"] == 1.0, res


@pytest.mark.train
def test_export_roundtrip_parity(tmp_path: Path) -> None:
    """Train briefly on the toy network, export, and check §4.9 parity + metadata + model card."""
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from rl.export_onnx import export
    from rl.train import POLICY_KWARGS
    from rl.vec import EnvFactory

    graph_path = tmp_path / "network.json"
    graph_path.write_text(json.dumps(toy_network_dict()))
    venv = DummyVecEnv([EnvFactory(stage=2, seed=0, graph_path=str(graph_path))])
    model = MaskablePPO("MlpPolicy", venv, n_steps=256, batch_size=128, n_epochs=1, policy_kwargs=POLICY_KWARGS,
                        seed=0, device="cpu", verbose=0)
    model.learn(512)
    ckpt = tmp_path / "best_model.zip"
    model.save(str(ckpt))
    venv.close()
    out = tmp_path / "models" / "nyc_subway_rcc_ppo.onnx"
    res = export(ckpt, out, n_parity=300, graph_path=str(graph_path))
    assert res["parity"]["passed"] and res["parity"]["max_abs_logit_diff"] <= 1e-4
    assert res["parity"]["argmax_agreement"] == 1.0
    pol = _check_onnx_contract(out)
    assert pol.metadata["graph_version"] == load_graph(graph_path).graph_version
    assert pol.metadata["constants_hash"] == C.constants_hash()
    assert int(pol.metadata["trained_steps"]) == model.num_timesteps
    card = json.loads(out.with_suffix(".json").read_text())
    assert card["reward"]["coefficients"]["REWARD_SCALE"] == C.REWARD_SCALE
    assert card["hyperparameters"]["net_arch"] == {"pi": [256, 256], "vf": [256, 256]}
    rng = np.random.default_rng(1)
    obs = rng.uniform(-1, 1, (64, C.OBS_DIM)).astype(np.float32)
    masks = rng.random((64, C.N_ACTIONS)) < 0.4
    masks[:, 0] = True
    sb3_actions, _ = model.predict(obs, action_masks=masks, deterministic=True)
    np.testing.assert_array_equal(masked_argmax(pol.logits(obs), masks), sb3_actions)
