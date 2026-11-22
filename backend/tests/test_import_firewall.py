"""Import firewall (ARCHITECTURE §1, §8; PRD Epic 6).

``sim/*`` and ``app/*`` must never import torch, gymnasium, stable-baselines3,
sb3-contrib or tensorboard, and ``onnxruntime`` is imported lazily by ``sim.policy`` only.
Every check runs in a fresh interpreter so modules imported by other tests cannot mask a
violation.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
FORBIDDEN = ("torch", "gymnasium", "gym", "stable_baselines3", "sb3_contrib", "tensorboard")


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BACKEND)
    return subprocess.run([sys.executable, "-c", code], cwd=BACKEND, env=env, capture_output=True, text=True, timeout=120)


def _modules_after(import_stmt: str) -> set[str]:
    code = f"import json, sys\n{import_stmt}\nprint(json.dumps(sorted(sys.modules)))"
    res = _run(code)
    assert res.returncode == 0, res.stderr
    return set(json.loads(res.stdout.strip().splitlines()[-1]))


def _forbidden_in(mods: set[str]) -> list[str]:
    return sorted(m for m in mods if m.split(".")[0] in FORBIDDEN)


def test_importing_sim_loads_no_training_stack() -> None:
    mods = _modules_after(
        "import sim, sim.engine, sim.graph, sim.masks, sim.obs, sim.actions, sim.cost, sim.policy, "
        "sim.scenarios, sim.testing, sim.disruptions, sim.kinematics, sim.paths, sim.bench, sim.calibrate"
    )
    assert _forbidden_in(mods) == []


def test_onnxruntime_is_imported_lazily() -> None:
    mods = _modules_after("import sim, sim.policy")
    assert "onnxruntime" not in mods


def test_sim_runs_with_training_stack_blocked() -> None:
    """A full reset/step/decide cycle works when the forbidden packages cannot be imported."""
    code = f"""
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in {FORBIDDEN!r}:
            raise ImportError('blocked by test: ' + name)
        return None
sys.meta_path.insert(0, Block())
import numpy as np
from sim import NYCSubwaySim, Scenario
from sim.testing import make_toy_graph
g = make_toy_graph()
sim = NYCSubwaySim(g, Scenario(seed=1))
sim.add_disruption('local_block', 'segment', 10)
for _ in range(60):
    b = sim.pending_decisions()
    sim.apply_actions(b.train_ids, np.zeros(len(b), dtype=np.int64))
    sim.step(1)
sim.frame(); sim.metrics(); sim.cost_totals()
print('ok')
"""
    res = _run(code)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().endswith("ok")


@pytest.mark.skipif(not (BACKEND / "app" / "__init__.py").is_file(), reason="app package (WP4) not present")
def test_importing_app_loads_no_training_stack() -> None:
    main = BACKEND / "app" / "main.py"
    stmt = "import app" + (", app.main" if main.is_file() else "")
    assert _forbidden_in(_modules_after(stmt)) == []


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module.split(".")[0])
    return out


@pytest.mark.parametrize("package", ["sim", "app"])
def test_source_never_mentions_forbidden_imports(package: str) -> None:
    """Static check: even lazily imported code paths in sim/app avoid the training stack."""
    root = BACKEND / package
    if not root.is_dir():
        pytest.skip(f"{package}/ not present")
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        imported = _imports_of(path)
        if package == "sim" and path.name == "policy.py":
            # Sb3ZipPolicy lazily imports sb3_contrib for local .zip checkpoints (ARCHITECTURE §3.1)
            imported -= {"sb3_contrib"}
        bad = sorted(imported & set(FORBIDDEN))
        if bad:
            offenders.append(f"{path.relative_to(BACKEND)}: {bad}")
    assert offenders == []
