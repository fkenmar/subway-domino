# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Subway Domino is a Human vs. AI NYC subway operations simulator, served at `https://kenmar.dev/subway`. `PRD.md` is the product spec of record.

**`docs/ARCHITECTURE.md` is the binding engineering contract.** It defines:
- data schemas and the `NYCSubwaySim` API;
- observations, actions, masks and reward;
- ONNX I/O;
- the WebSocket protocol;
- deployment, tests and work packages.

Its CONTRACT sections are interfaces other components depend on. Change them only by updating that doc, every consumer, and the golden fixtures in the same change. Work-package progress notes live in `docs/progress/WP*.md`: read them before continuing interrupted work.

## Layout (see ARCHITECTURE §1)

- **`backend/` is the Python root and the working directory for all Python commands.** Packages are `sim` (shared physics, obs, masks, cost, ONNX policy runner; pure NumPy), `rl` (gymnasium/sb3 training; the only place torch, gymnasium or sb3 may be imported), `app` (FastAPI and WebSocket server) and `scripts` (data build; stdlib plus numpy only).
- **`backend/data/`** holds committed artifacts: `network.json` and `network_geo.json` are built by `scripts/build_nyc_network.py`, and `infrastructure.json` is the hand-curated overlay. `data/raw/` is a gitignored cache.
- **`backend/models/nyc_subway_rcc_ppo.onnx`** is the shipped policy.
- **`frontend/`** is a Next.js 16 App Router app (TypeScript) with `basePath: "/subway"`. Every `public/` URL goes through `asset()`.

## Commands

Python commands run from `backend/` with the venv at `backend/.venv` (Python 3.11):

| Task | Command |
|---|---|
| Build data (network required) | `.venv/bin/python -m scripts.build_nyc_network` |
| Build offline / verify | `.venv/bin/python -m scripts.build_nyc_network --offline --check` |
| Fast tests | `.venv/bin/python -m pytest -q -m "not slow and not train"` |
| Single test | `.venv/bin/python -m pytest tests/test_short_turn.py -q -k latency` |
| Dev server | `.venv/bin/uvicorn app.main:app --port 8000 --reload` |
| Train (local) | `.venv/bin/python -m rl.train --run-dir runs/local --n-envs 14 --resume auto` |
| Evaluate | `.venv/bin/python -m rl.evaluate --policy models/nyc_subway_rcc_ppo.onnx --seeds 50 --out models/` |
| Export ONNX | `.venv/bin/python -m rl.export_onnx --checkpoint runs/local/best_model.zip --out models/nyc_subway_rcc_ppo.onnx` |

Frontend commands run from `frontend/`:
- Dev server: `npm run dev` (open `http://localhost:3000/subway`).
- Checks: `npm run typecheck && npm run lint && npm test && npm run build`.

## Hard rules

- **Honesty.** Normal Schedule means every decision is `CONTINUE_SCHEDULED`. AI mode means the ONNX policy with action masks only. Never add hand-coded dispatch heuristics, and never fall back silently when the model is missing (AI mode is unavailable instead).
- **Import firewall.** `sim/*` and `app/*` must not import torch, gymnasium, stable_baselines3, sb3_contrib or tensorboard. `requirements-prod.txt` contains exactly fastapi, uvicorn[standard], numpy, onnxruntime and pydantic.
- **One shared code path.** The observation builder, masks and cost live in `sim` and are shared by training and serving. One `DT_S = 10` s tick is used everywhere. All constants live in `sim/constants.py`.
- **Physical direction comes from station order.** Never derive it from GTFS `direction_id` or the N/S suffix: they are inconsistent on the shared J/M track.
- **Scope.** Manhattan, Brooklyn, Queens and the Bronx only (Staten Island Railway excluded): 28 routes, 475 stations, 424 complexes.
- **Production.** The server runs as a single uvicorn worker, with at most 40 WebSocket sessions, each with its own `NYCSubwaySim`; they share only the read-only graph and one `InferenceSession`. Browsers connect to Railway directly over `wss://`, never through a Vercel rewrite.
