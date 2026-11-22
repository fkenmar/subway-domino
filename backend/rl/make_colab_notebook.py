"""Generate ``colab/train_subway_domino_colab.ipynb`` (ARCHITECTURE §4.10) with nbformat.

    .venv/bin/python -m rl.make_colab_notebook          # write the notebook
    .venv/bin/python -m rl.make_colab_notebook --check  # exit 1 if the committed notebook is stale

The notebook is plain Python apart from three cells tagged ``colab-only``: pip install,
TensorBoard and the download. Every setting can be overridden through ``SUBWAY_DOMINO_*``
environment variables, so the same cells can be executed outside Colab with Drive mocked
(see ``docs/progress/WP5.md``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "colab" / "train_subway_domino_colab.ipynb"
DEFAULT_REPO_URL = "https://github.com/fkenmar/subway-domino.git"

INTRO = """# Subway Domino: train the RCC dispatcher on Google Colab

This notebook trains the MaskablePPO dispatcher of [Subway Domino](https://kenmar.dev/subway)
(`backend/rl`, ARCHITECTURE §4.6) on a Colab runtime. It then evaluates the policy against
Normal Schedule on 50 fixed seeds (§4.7) and exports the ONNX model the server ships (§4.9).

**What it does**
1. Detects the runtime (vCPUs, RAM, GPU/TPU, Python) and sizes the training to it.
2. Mounts Google Drive. Everything durable lives in `MyDrive/subway-domino/`:
   `runs/<name>/` holds checkpoints, TensorBoard logs and `train_state.json`;
   `models/` receives the exported model.
3. Gets the code (a `git clone` of the public repo, or an uploaded zip) and installs the pinned
   training packages on top of Colab's own `torch`.
4. Trains with `--resume auto`. **If Colab disconnects or times out, reconnect and use
   *Runtime → Run all*.** Training continues from `latest.zip` (checkpoints every 250k steps).
5. Exports `nyc_subway_rcc_ppo.onnx` and its model card, evaluates them, copies everything to
   `MyDrive/subway-domino/models/` and offers a download.
"""

RUNTIME_MD = """## 1. Pick the right runtime

Training is **CPU-bound**. Every env step advances a NumPy simulation of the whole subway (plus
its Normal-Schedule twin), and the policy is a small MLP, so a GPU barely helps.

* Choose the runtime with the **most vCPUs per compute unit** (*Runtime → Change runtime type*)
  and check the `vCPUs` line below.
  TPU runtimes usually come with very large host CPUs; High-RAM CPU runtimes come next.
* **Don't spend GPU credits on this.**
* Throughput scales with the vCPU count: one env worker per vCPU, minus two kept for the
  learner and the system.
* **Disconnect the runtime when training finishes** (last cell), so it stops consuming compute units.
"""

RUNTIME_CODE = r'''# Runtime detection (ARCHITECTURE §4.10): CPUs, RAM, GPU/TPU, Python
import os, platform, shutil, subprocess, sys

cpus = os.cpu_count() or 1

def _ram_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (AttributeError, ValueError, OSError):
        return float("nan")

gpu = ""
if shutil.which("nvidia-smi"):
    gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
tpu = any(os.environ.get(k) for k in ("COLAB_TPU_ADDR", "TPU_NAME", "TPU_ACCELERATOR_TYPE")) or os.path.exists("/dev/accel0")
print(f"Python {platform.python_version()} | vCPUs {cpus} | RAM {_ram_gb():.1f} GB | GPU: {gpu or 'none'} | TPU: {'yes' if tpu else 'no'}")

n_envs = min(64, max(2, cpus - 2))
n_steps = max(128, round(15360 / n_envs / 64) * 64)
print(f"-> rl.train --n-envs auto: {n_envs} env workers x {n_steps} steps per rollout; "
      f"learner torch threads {min(8, max(1, cpus // 8))}")
if not (3, 11) <= sys.version_info[:2] <= (3, 13):
    print("WARNING: backend/ supports Python 3.11-3.13")
if gpu:
    print("NOTE: this GPU runtime spends GPU credits, but training is CPU-bound: prefer the runtime with the most vCPUs.")
if cpus < 8:
    print(f"NOTE: only {cpus} vCPUs, so training will be slow. A TPU runtime (large host CPU) or High-RAM CPU runtime is better.")
'''

SETTINGS_MD = """## 2. Settings

`RUN_NAME` names the run directory on Drive. Keep it unchanged to resume a run, or change it
to start a new one. `TOTAL_STEPS` defaults to the full curriculum: Stage 1 for 3M steps, then
Stage 2 for 10M. See the expected wall-clock in `docs/progress/WP5.md`.
"""

SETTINGS_CODE = r'''#@title Settings
RUN_NAME = "run1"  # @param {type:"string"}
REPO_URL = "https://github.com/fkenmar/subway-domino.git"  # @param {type:"string"}
BRANCH = "main"  # @param {type:"string"}
CODE_SOURCE = "git"  # @param ["git", "drive_zip"]
TOTAL_STEPS = 13000000  # @param {type:"integer"}
STAGE = "auto"  # @param ["auto", "1", "2"]
EVAL_SEEDS = 50  # @param {type:"integer"}

import os

# Optional SUBWAY_DOMINO_* overrides (used to test this notebook outside Colab; unset on Colab).
_env = os.environ.get
RUN_NAME = _env("SUBWAY_DOMINO_RUN_NAME", RUN_NAME)
REPO_URL = _env("SUBWAY_DOMINO_REPO_URL", REPO_URL)
BRANCH = _env("SUBWAY_DOMINO_BRANCH", BRANCH)
CODE_SOURCE = _env("SUBWAY_DOMINO_CODE_SOURCE", CODE_SOURCE)
TOTAL_STEPS = int(_env("SUBWAY_DOMINO_TOTAL_STEPS", TOTAL_STEPS))
STAGE = _env("SUBWAY_DOMINO_STAGE", STAGE)
EVAL_SEEDS = int(_env("SUBWAY_DOMINO_EVAL_SEEDS", EVAL_SEEDS))
TRAIN_EXTRA_ARGS = _env("SUBWAY_DOMINO_TRAIN_ARGS", "")  # extra rl.train flags, e.g. "--checkpoint-every 100000"

DRIVE_MOUNT = _env("SUBWAY_DOMINO_DRIVE", "/content/drive")
BASE = os.path.join(DRIVE_MOUNT, "MyDrive", "subway-domino")
CODE_DIR = _env("SUBWAY_DOMINO_CODE_DIR", "/content/subway-domino")
BACKEND = os.path.join(CODE_DIR, "backend")
RUN_DIR = os.path.join(BASE, "runs", RUN_NAME)
EXPORT_DIR = os.path.join(RUN_DIR, "export")
MODELS_DIR = os.path.join(BASE, "models")
print(f"run dir:  {RUN_DIR}\ncode dir: {CODE_DIR} (source: {CODE_SOURCE})\nmodels:   {MODELS_DIR}")
'''

DRIVE_MD = """## 3. Mount Google Drive

Checkpoints go straight to Drive, so a disconnect loses at most one checkpoint interval.
"""

DRIVE_CODE = r'''import signal, subprocess, sys

def sh(argv, cwd=None, check=True):
    """Run a command and stream its output. Stopping the cell forwards Ctrl+C (SIGINT), so
    rl.train writes a checkpoint before exiting."""
    print("$ " + " ".join(str(a) for a in argv), flush=True)
    proc = subprocess.Popen([str(a) for a in argv], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        raise
    if check and rc != 0:
        raise RuntimeError(f"command failed with exit code {rc}")
    return rc

from google.colab import drive

drive.mount(DRIVE_MOUNT)
for d in (os.path.join(BASE, "runs"), MODELS_DIR):
    os.makedirs(d, exist_ok=True)
print("Drive ready:", BASE)
'''

CODE_MD = f"""## 4. Get the code

* **`git`** (default): clones the public repository `{DEFAULT_REPO_URL}`; no token needed.
  Re-running this cell runs `git pull` to fetch the latest commit.
* **`drive_zip`**: on your machine, run `backend/.venv/bin/python -m rl.make_repo_zip` (from
  `backend/`). Upload `dist/subway-domino.zip` to `MyDrive/subway-domino/subway-domino.zip`,
  then set `CODE_SOURCE = "drive_zip"`.
"""

CODE_CODE = r'''import shutil, zipfile

if CODE_SOURCE == "git":
    if os.path.isdir(os.path.join(CODE_DIR, ".git")):
        sh(["git", "-C", CODE_DIR, "pull", "--ff-only", "origin", BRANCH])
    else:
        sh(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, CODE_DIR])
    sh(["git", "-C", CODE_DIR, "log", "-1", "--format=commit %h %cd %s", "--date=iso"])
elif CODE_SOURCE == "drive_zip":
    zpath = os.path.join(BASE, "subway-domino.zip")
    if not os.path.isfile(zpath):
        raise FileNotFoundError(f"upload dist/subway-domino.zip (rl.make_repo_zip) to {zpath}")
    tmp = CODE_DIR + ".unzip"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(tmp)
    shutil.rmtree(CODE_DIR, ignore_errors=True)
    shutil.move(os.path.join(tmp, "subway-domino"), CODE_DIR)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"unzipped {zpath} -> {CODE_DIR}")
else:
    raise ValueError(f"unknown CODE_SOURCE {CODE_SOURCE!r}")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
'''

INSTALL_MD = """## 5. Install the training packages

The notebook keeps Colab's preinstalled `torch` and pins it through a constraint, so nothing
reinstalls it. It installs the pinned `stable-baselines3`, `sb3-contrib`, `gymnasium`,
`tensorboard`, `onnx`, `onnxruntime` and `onnxscript` from `backend/requirements-colab.txt`.
"""

INSTALL_CODE = r'''import torch

constraint = "/tmp/subway-domino-torch-constraint.txt"
with open(constraint, "w") as fh:
    fh.write(f"torch=={torch.__version__}\n")
sh([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(BACKEND, "requirements-colab.txt"), "-c", constraint])
'''

VERIFY_MD = """## 6. Verify the environment and artifacts

This step checks package versions and the committed network artifacts (`backend/data/network.json`).
It finishes with a 10-second environment-throughput benchmark on this runtime.
"""

VERIFY_CODE = r'''import importlib, json

versions = {m: importlib.import_module(m).__version__
            for m in ("numpy", "torch", "gymnasium", "stable_baselines3", "sb3_contrib", "onnx", "onnxruntime")}
print(versions)
assert int(versions["numpy"].split(".")[0]) >= 2, "backend/ needs numpy >= 2"
net = os.path.join(BACKEND, "data", "network.json")
assert os.path.isfile(net), f"missing {net}: the repository checkout is incomplete"
with open(net) as fh:
    meta = json.load(fh)
print(f"network.json: graph_version {meta.get('graph_version')}, {len(meta.get('stations', []))} stations, "
      f"{len(meta.get('routes', []))} routes")
sh([sys.executable, "-c", "from sim import constants as C; print('constants_hash', C.constants_hash())"], cwd=BACKEND)
sh([sys.executable, "-m", "rl.bench_env", "--n-envs", "auto", "--seconds", "10", "--warmup-seconds", "3",
    "--single-seconds", "3", "--stage", "1", "--mode", "chunked"], cwd=BACKEND)
'''

TRAIN_MD = """## 7. Train (re-run this cell to resume)

The command is `python -m rl.train --run-dir <Drive run dir> --n-envs auto --n-steps auto --resume auto`.

* Every 250k steps it writes an atomic checkpoint to `runs/<name>/checkpoints/` and updates `latest.zip`.
* Every 500k steps it validates on 8 fixed seeds and keeps `best_model.zip`.
* Stopping the cell (■) writes a checkpoint first.
* After a disconnect or timeout, reconnect, pick the same runtime type and choose
  *Runtime → Run all*: it continues from the last checkpoint.
"""

TRAIN_CODE = r'''import shlex

train_cmd = [sys.executable, "-m", "rl.train", "--run-dir", RUN_DIR, "--n-envs", "auto", "--n-steps", "auto",
             "--total-steps", str(TOTAL_STEPS), "--stage", STAGE, "--resume", "auto", *shlex.split(TRAIN_EXTRA_ARGS)]
sh(train_cmd, cwd=BACKEND)
'''

PROGRESS_MD = """## 8. Progress

This cell reads `train_state.json` and `progress.jsonl` from Drive. It works at any time,
also from a second notebook while training runs.
"""

PROGRESS_CODE = r'''import glob, json

state_path = os.path.join(RUN_DIR, "train_state.json")
if not os.path.isfile(state_path):
    print("no train_state.json yet")
else:
    with open(state_path) as fh:
        st = json.load(fh)
    total = max(st.get("total_steps_target") or TOTAL_STEPS, 1)
    print(f"run {st['run_id']}: {st['steps_total']:,} / {total:,} steps ({100 * st['steps_total'] / total:.1f} %), "
          f"stage {st['stage']} ({st['steps_stage']:,} steps in stage), resumes {st.get('resumes', 0)}, "
          f"updated {st.get('updated_utc')}")
    best = st.get("best_score")
    print(f"best validation score: {'n/a' if best is None else f'{100 * best:+.2f} %'} at step {st.get('best_step')}")
    if st.get("last_validation"):
        v = st["last_validation"]
        print(f"last validation @ {v['step']:,}: score {100 * v['score']:+.2f} %, win {100 * v['win_rate']:.0f} %, "
              f"stuck {100 * v['stuck_reduction']:+.1f} %, wait {100 * v['wait_reduction']:+.1f} %")
    ckpts = sorted(glob.glob(os.path.join(RUN_DIR, "checkpoints", "ckpt_*.zip")), key=lambda p: int(p.rsplit("_", 1)[1][:-4]))
    print("checkpoints:", [os.path.basename(p) for p in ckpts])
    prog = os.path.join(RUN_DIR, "progress.jsonl")
    if os.path.isfile(prog):
        with open(prog) as fh:
            lines = fh.readlines()[-5:]
        for line in lines:
            p = json.loads(line)
            print(f"  step {p['step']:>11,}  stage {p['stage']}  {p['fps']:,.0f} steps/s  "
                  f"twin adv {p['twin_adv_rel'] if p['twin_adv_rel'] is None else round(100 * p['twin_adv_rel'], 2)} %  "
                  f"entropy {p['entropy'] if p['entropy'] is None else round(p['entropy'], 3)}  "
                  f"EV {p['explained_variance'] if p['explained_variance'] is None else round(p['explained_variance'], 3)}")
'''

TB_CODE = r'''# TensorBoard: reward, twin advantage, per-action rates, mask validity, validation metrics
%load_ext tensorboard
%tensorboard --logdir "$RUN_DIR/tb"
'''

FINISH_MD = """## 9. Export, evaluate and save the model

* Exports `best_model.zip` (or `latest.zip` if no validation has run) to ONNX. The export
  aborts unless logits match SB3 within 1e-4 and the masked argmax matches on 1,000 recorded
  observations.
* Evaluates the ONNX model against Normal Schedule on `EVAL_SEEDS` seeds, using every CPU.
* Copies the model, model card and evaluation reports to `MyDrive/subway-domino/models/`.

To ship the model, copy `nyc_subway_rcc_ppo.onnx` and `nyc_subway_rcc_ppo.json` into
`backend/models/` in the repository.
"""

FINISH_CODE = r'''import shutil

ckpt = os.path.join(RUN_DIR, "best_model.zip")
if not os.path.isfile(ckpt):
    ckpt = os.path.join(RUN_DIR, "latest.zip")
onnx_path = os.path.join(EXPORT_DIR, "nyc_subway_rcc_ppo.onnx")
sh([sys.executable, "-m", "rl.export_onnx", "--checkpoint", ckpt, "--out", onnx_path], cwd=BACKEND)
sh([sys.executable, "-m", "rl.evaluate", "--policy", onnx_path, "--seeds", str(EVAL_SEEDS), "--workers", "auto",
    "--out", EXPORT_DIR], cwd=BACKEND)
copied = []
for name in ("nyc_subway_rcc_ppo.onnx", "nyc_subway_rcc_ppo.json", "eval_report.json", "eval_report.md", "eval_per_seed.csv"):
    src = os.path.join(EXPORT_DIR, name)
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(MODELS_DIR, name))
        copied.append(name)
print("copied to", MODELS_DIR, ":", copied)
'''

DOWNLOAD_CODE = r'''from google.colab import files

for name in ("nyc_subway_rcc_ppo.onnx", "nyc_subway_rcc_ppo.json", "eval_report.md"):
    files.download(os.path.join(MODELS_DIR, name))
'''

DISCONNECT_MD = """## 10. Disconnect

When training and export are done, disconnect so the runtime stops using compute units:
*Runtime → Disconnect and delete runtime*, or tick the box below and run the cell.
Everything you need is on Drive.
"""

DISCONNECT_CODE = r'''DISCONNECT_NOW = False  # @param {type:"boolean"}
if DISCONNECT_NOW:
    from google.colab import runtime

    runtime.unassign()
'''


def build_notebook() -> nbformat.NotebookNode:
    colab = {"tags": ["colab-only"]}
    cells = [
        new_markdown_cell(INTRO),
        new_markdown_cell(RUNTIME_MD),
        new_code_cell(RUNTIME_CODE, metadata={"id": "runtime"}),
        new_markdown_cell(SETTINGS_MD),
        new_code_cell(SETTINGS_CODE, metadata={"id": "settings", "cellView": "form"}),
        new_markdown_cell(DRIVE_MD),
        new_code_cell(DRIVE_CODE, metadata={"id": "drive", "tags": ["drive"]}),
        new_markdown_cell(CODE_MD),
        new_code_cell(CODE_CODE, metadata={"id": "code"}),
        new_markdown_cell(INSTALL_MD),
        new_code_cell(INSTALL_CODE, metadata={"id": "install", **colab}),
        new_markdown_cell(VERIFY_MD),
        new_code_cell(VERIFY_CODE, metadata={"id": "verify"}),
        new_markdown_cell(TRAIN_MD),
        new_code_cell(TRAIN_CODE, metadata={"id": "train"}),
        new_markdown_cell(PROGRESS_MD),
        new_code_cell(PROGRESS_CODE, metadata={"id": "progress"}),
        new_code_cell(TB_CODE, metadata={"id": "tensorboard", **colab}),
        new_markdown_cell(FINISH_MD),
        new_code_cell(FINISH_CODE, metadata={"id": "finish"}),
        new_code_cell(DOWNLOAD_CODE, metadata={"id": "download", **colab}),
        new_markdown_cell(DISCONNECT_MD),
        new_code_cell(DISCONNECT_CODE, metadata={"id": "disconnect", "cellView": "form", **colab}),
    ]
    for i, cell in enumerate(cells):  # deterministic nbformat 4.5 cell ids (stable --check output)
        cell["id"] = cell.metadata.get("id") or f"md-{i:02d}"
    nb = new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True, "name": "train_subway_domino_colab.ipynb"},
    })
    nbformat.validate(nb)
    return nb


def render() -> str:
    return nbformat.writes(build_notebook(), version=4)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=NOTEBOOK)
    ap.add_argument("--check", action="store_true", help="exit 1 if --out differs from the generated notebook")
    args = ap.parse_args(argv)
    text = render()
    if args.check:
        ok = args.out.is_file() and args.out.read_text() == text
        print(f"{args.out}: {'up to date' if ok else 'STALE (re-run rl.make_colab_notebook)'}")
        return 0 if ok else 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out} ({len(build_notebook().cells)} cells, nbformat-validated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
