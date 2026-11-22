"""Import firewall for the server package (ARCHITECTURE §1, §7, §8; PRD Epic 6).

``app/*`` must never import torch, gymnasium, stable-baselines3, sb3-contrib or tensorboard,
and must run on ``requirements-prod.txt`` alone (exactly fastapi, uvicorn[standard], numpy,
onnxruntime, pydantic). WP2's ``test_import_firewall.py`` covers ``sim``; this file covers
``app`` with fresh interpreters, so modules imported by other tests cannot mask a violation.
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
APP = BACKEND / "app"
FORBIDDEN = ("torch", "gymnasium", "gym", "stable_baselines3", "sb3_contrib", "tensorboard")
APP_MODULES = sorted(f"app.{p.stem}" for p in APP.glob("*.py") if p.stem != "__init__")
# Third-party top-level packages that requirements-prod.txt installs (uvicorn[standard] brings
# websockets, used by app.smoke inside the image) plus the backend's own packages.
PROD_IMPORTS = {"fastapi", "starlette", "pydantic", "numpy", "uvicorn", "websockets", "onnxruntime", "sim", "app"}
PROD_REQUIREMENTS = [
    "fastapi==0.165.0",
    "uvicorn[standard]==0.54.0",
    "numpy==2.4.6",
    "onnxruntime==1.32.0",
    "pydantic==2.14.3",
]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BACKEND)
    return subprocess.run([sys.executable, "-c", code], cwd=BACKEND, env=env, capture_output=True, text=True,
                          timeout=180)


def _modules_after(import_stmt: str) -> set[str]:
    res = _run(f"import json, sys\n{import_stmt}\nprint(json.dumps(sorted(sys.modules)))")
    assert res.returncode == 0, res.stderr
    return set(json.loads(res.stdout.strip().splitlines()[-1]))


def _forbidden_in(mods: set[str]) -> list[str]:
    return sorted(m for m in mods if m.split(".")[0] in FORBIDDEN)


def test_app_package_has_the_expected_modules() -> None:
    for name in ("main", "config", "manager", "session", "protocol", "callouts", "smoke"):
        assert f"app.{name}" in APP_MODULES, name


def test_importing_every_app_module_loads_no_training_stack() -> None:
    mods = _modules_after("import app, " + ", ".join(APP_MODULES))
    assert _forbidden_in(mods) == []


def test_creating_the_app_loads_no_training_stack_and_onnxruntime_stays_lazy() -> None:
    mods = _modules_after("from app.main import create_app\napp = create_app()")
    assert _forbidden_in(mods) == []
    assert "onnxruntime" not in mods  # only loaded when a model is actually loaded


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module.split(".")[0])
    return out


@pytest.mark.parametrize("path", sorted(APP.glob("*.py")), ids=lambda p: p.name)
def test_app_sources_import_only_stdlib_and_production_packages(path: Path) -> None:
    """Static check (covers lazy imports too): the image built from requirements-prod.txt has them all."""
    imported = _imports_of(path)
    assert not imported & set(FORBIDDEN), sorted(imported & set(FORBIDDEN))
    third_party = {m for m in imported if m not in sys.stdlib_module_names}
    assert third_party <= PROD_IMPORTS, sorted(third_party - PROD_IMPORTS)


def test_a_full_session_runs_with_the_training_stack_blocked() -> None:
    """hello → welcome → frames → preset → AI refused, with torch & co. unimportable."""
    code = f"""
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in {FORBIDDEN!r}:
            raise ImportError('blocked by test: ' + name)
        return None
sys.meta_path.insert(0, Block())
from fastapi.testclient import TestClient
from app.config import Settings
from app.main import create_app
from app.protocol import decode_frame
from sim.testing import make_toy_graph
settings = Settings(allow_no_origin=True, model_path='/nonexistent/model.onnx', data_dir='/nonexistent', log_level='ERROR')
app = create_app(graph=make_toy_graph(), settings=settings, prewarm=False, offload=False)
with TestClient(app) as tc:
    assert tc.get('/health').json() == {{'status': 'ok', 'model_loaded': False}}
    with tc.websocket_connect('/ws') as ws:
        ws.send_text(json.dumps({{'type': 'hello', 'protocol': 1, 'resume_token': None, 'restore': None}}))
        assert json.loads(ws.receive_text())['type'] == 'welcome'
        ws.send_text(json.dumps({{'type': 'preset', 'id': 'toy-union-closure'}}))
        ws.send_text(json.dumps({{'type': 'set_mode', 'mode': 'ai'}}))
        frames, types = 0, set()
        while frames < 3 or 'error' not in types:
            msg = ws.receive()
            if msg.get('bytes') is not None:
                decode_frame(msg['bytes']); frames += 1
            else:
                types.add(json.loads(msg['text'])['type'])
assert not [m for m in sys.modules if m.split('.')[0] in {FORBIDDEN!r}]
print('ok')
"""
    res = _run(code)
    assert res.returncode == 0, res.stderr[-3000:]
    assert res.stdout.strip().endswith("ok")


def test_requirements_prod_is_exactly_the_five_runtime_packages() -> None:
    lines = [ln.strip() for ln in (BACKEND / "requirements-prod.txt").read_text(encoding="utf-8").splitlines()]
    reqs = [ln for ln in lines if ln and not ln.startswith("#")]
    assert reqs == PROD_REQUIREMENTS


def test_requirements_dev_extends_prod_with_the_test_tools() -> None:
    lines = [ln.strip() for ln in (BACKEND / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()]
    reqs = [ln for ln in lines if ln and not ln.startswith("#")]
    assert reqs[0] == "-r requirements-prod.txt"
    assert reqs[1:] == ["pytest==9.2.0", "pytest-asyncio==1.5.0", "httpx==0.28.1", "websockets==17.4"]
