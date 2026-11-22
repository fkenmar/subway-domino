# WP5 — RL pipeline progress

## Orchestrator notes (read first; added before WP5 started)
- **The user has Colab Pro credits (student subscription). Colab is the PRIMARY venue for the long training run.** The local M5 Pro (15 cores) is for smoke tests and quick iteration.
- Implement ARCHITECTURE **§4.10** exactly:
  - runtime detection;
  - runtime advice: the job is CPU-bound, so pick the most vCPUs per compute unit, typically TPU host CPUs, and don't burn GPU credits;
  - `--n-envs auto` / `--n-steps auto` scaling with the core count;
  - use Colab's preinstalled torch, plus `backend/requirements-colab.txt` for everything else;
  - Python 3.11–3.13 compatibility;
  - Drive run dir with `--resume auto`;
  - git-clone or Drive-zip code access;
  - evaluate, export and copy artifacts to `MyDrive/subway-domino/models/`.
- **Code access:** the user chose GitHub. The repo is **public** at `https://github.com/fkenmar/subway-domino.git`. That is the notebook's default clone URL (form field, no token needed); re-running the cell pulls the latest commit.
- Also provide, as the fallback path, a tiny helper that builds the repo zip for Drive upload: `backend/rl/make_repo_zip.py` → `dist/subway-domino.zip`, excluding `.venv`, `node_modules`, `.next`, `data/raw`, `runs` and `__pycache__`.
- Validate the generated notebook locally as far as possible:
  - nbformat validation;
  - execute the non-Colab cells (install skipped, Drive mount mocked) via a small runner script in the scratchpad, including a very short training run with `--n-envs auto`.

## Status: COMPLETE (the long training run is deliberately not launched; command below)

## Verification (2026-11-21)
- `pytest -m train`: 20 passed (25 s). `pytest -m "not slow and not train"`: 218 passed, 2 skipped (no
  shipped model). After the REWARD_CLIP change, `pytest -m "not train"` (incl. slow sim suite):
  239 passed, 2 skipped, 1 xfailed (WP2 tick target), exit 0.
- `rl.make_colab_notebook --check`: up to date. Notebook executed locally (see Done).

## Done
- `rl/env.py` `SubwayDominoEnv` (§4.5): AI sim + Normal twin from one Scenario, §4.1 batch
  serialization (obs never recomputed), §4.4 twin-difference reward, `action_masks()`, truncation
  (zero obs + CONTINUE-only mask), info (`cost_ai`/`cost_twin` deltas, action name, applied, ticks,
  `episode_summary` at the end), resample decision-free episodes (5 tries, then dummy step),
  `set_stage()`. `twin_mode` = `lazy` (default: fork `ai.copy()` just before the first non-CONTINUE
  action) | `copy` | `reset` (independent reset, state_hash asserted equal). The three are proven
  bit-identical (rewards + final twin state_hash) in tests.
- `rl/vec.py`: picklable `EnvFactory` (spawn-safe, workers ignore SIGINT), `ObsMaskVecEnv` (masks
  from obs bits, no extra IPC), §4.10 auto rules (`resolve_n_envs/n_steps/torch_threads`,
  `resolve_batch_size` = divisor of the rollout closest to 2048).
- `rl/rollout.py` + `rl/ppo.py`: **chunked worker-side-inference collector** (see Perf) and
  `ChunkedMaskablePPO` (MaskablePPO subclass; only `collect_rollouts` differs; zips load with plain
  `MaskablePPO.load` / `sim.policy.Sb3ZipPolicy`). GAE per worker segment (verified == SB3 buffer),
  horizon = terminal, `ReturnNormalizer` (= VecNormalize(norm_reward) rule, state in train_state.json).
- `rl/callbacks.py` + `rl/train.py` (§4.6/§4.10): net arch/hyperparameters/LR+entropy linear
  schedules, curriculum Stage 1 -> 2 by step count, atomic ckpts every 250k (keep 5) + latest.zip,
  best_model.zip via validation (8 seeds `eval_scenario(2_000_000+i)`, score = mean twin-relative cost
  reduction, run in parallel on the idle workers, Normal results cached), train_state.json
  (`{stage, steps_total, steps_stage, best_score, run_id, ...}`), TB scalars (rollout reward/clip,
  twin advantage/win rate, per-action rates, mask validity + applied rate, validation/*, perf/*,
  curriculum/stage, train/ent_coef), progress.jsonl, `--resume auto|never`, `--run-dir`,
  `--n-envs/--n-steps/--batch-size/--torch-threads auto`, `--total-steps`, `--stage auto|1|2`,
  graceful SIGINT/SIGTERM checkpoint (exit 0). Verified: resume (resumes=1), curriculum switch,
  SIGINT, pruning, no orphan workers.
- `rl/evaluate.py` (§4.7): Normal vs AI per eval seed, `.onnx`/`.zip` via `load_policy`, `--workers`
  (spawn pool), JSON/MD/CSV, 95 % bootstrap CIs (10k), % change, seed win rate, `--require-win`,
  §4.8 acceptance-bar block, merges its summary into `<policy>.json` (model card).
- `rl/export_onnx.py` (§4.9): actor-only wrapper, dynamo exporter (legacy fallback), opset 18, single
  file, `obs`->`logits` dynamic batch, metadata (obs_dim, n_actions, action_names, graph_version,
  constants_hash, trained_steps, created_utc, mask_revision, producer), parity gate on 1,000 recorded
  obs (max |dlogit| <= 1e-4 AND identical masked argmax vs `predict(deterministic=True)`), model card.
- `rl/bench_env.py`: single in-process / SubprocVecEnv / chunked RolloutWorkers; per-env + aggregate
  steps/s, sim-h/s, decisions per sim-hour, parallel efficiency, `--json`.
- `rl/make_repo_zip.py` (dist/subway-domino.zip, 23 MB, 229 files), `rl/make_colab_notebook.py`
  -> `colab/train_subway_domino_colab.ipynb` (23 cells, deterministic cell ids, `--check`).
- `backend/requirements-train.txt` (§7), `backend/requirements-colab.txt` (no torch; the notebook
  pins Colab's torch with a pip constraint so nothing can reinstall it).
- Tests: `tests/test_env.py` (train, 19 tests, ~21 s), `tests/test_onnx_parity.py` (model: shipped
  model contract + parity vs its checkpoint, skip while unshipped, `SUBWAY_DOMINO_MODEL` override;
  train: export round trip).
- Notebook validated locally (scratchpad `wp5/run_notebook.py`): nbformat validate; all non-Colab
  cells executed with Drive mocked; drive_zip code path; 20k-step `--n-envs auto` training; progress;
  export; evaluate; copy to models/; then the git path (clone + re-run = pull) from a file:// repo.

## Perf (M5 Pro, 5 P + 10 E cores)
- Single env in-process, Stage 1: 460 steps/s (~250-280 decisions per sim-hour; ~1.4 ticks/step).
- SB3 SubprocVecEnv x14, Stage 1: **1,006 env-steps/s (16 % parallel efficiency)**. Cause: step
  durations are heavy-tailed (mean 2.3 ms, p99 16 ms, reset+warm-up 64 ms) and every vectorized step
  waits for the slowest worker; simulated k-envs-per-worker batching only reaches 42 % at k=16.
- Fix: `rl.rollout.RolloutWorkers` (each worker runs the current policy in NumPy for 64-step chunks,
  dynamic assignment, one barrier per rollout). Stage 1: 5 workers 1,535/s, 8: 1,771, 11: 1,925,
  13: 2,215, 14: 1,995 (E-cores add ~70/s each). Stage 2 x14: 5,044/s. Twin cost is inherent (the
  engine's `_decide` flag also drives boarding top-ups/disruptions, so the twin needs full ticks).

## Calibration change (sim/constants.py, allowed for WP5)
- `REWARD_CLIP` 10 -> 100 (REWARD_SCALE unchanged). Smoke policy, unclipped |r|: Stage 1 13 % > 10,
  4.9 % > 50, 1.6 % > 100, max 416; the best am_peak meltdown (+30.9 % twin advantage) had 27 % of its
  steps clipped -> the clip erased the signal where actions pay off most. Stage 2: 1.2 % > 10, max 27.
  Training normalizes returns, so only the tail cap moves. Documented in the model card coefficients.
  Full test suite re-run after the change: see Verification.

## Contract change requests
1. §4.6 "n_envs 14 (SubprocVecEnv, spawn)": training collects with `rl.rollout.RolloutWorkers`
   (spawned env workers with worker-side NumPy inference, one barrier per rollout) instead of the
   per-step SubprocVecEnv loop (16 % efficiency, measured). Same MaskablePPO update and saved format;
   the env stays SubprocVecEnv(spawn)-safe (tested) and bench_env reports both. Proposed text: "n_envs
   spawned env workers (`rl.rollout`, chunked on-policy collection)".
2. §4.6 truncation: GAE treats `episode_s` as terminal (no bootstrap from the zero terminal obs; the
   finite-horizon objective is what evaluation scores, `time_in_episode_frac` is observed). The env
   still returns `truncated=True` as contracted.
3. §4.6 `VecNormalize(norm_reward=True)` is implemented inside `ChunkedMaskablePPO`
   (`ReturnNormalizer`, identical rule; state persisted in train_state.json for resume).
4. §4.6 table `n_steps 1024` vs §4.10 auto: `--n-steps auto` is the default everywhere (14 workers ->
   1088, rollout 15,232, batch auto 2176); `--n-steps 1024 --batch-size 2048` reproduces the table.
5. §4.5 `SubwayDominoEnv.__init__` adds keyword-only `twin_mode` (backwards-compatible superset).
6. §4.6 validation seeds: `eval_scenario(VALIDATION_SEED_BASE + i)` (the citywide meltdown generator,
   so model selection mirrors §4.7 evaluation), also during Stage 1.
7. §4.4 `REWARD_CLIP` recalibrated 10 -> 100 (see above; RL-owner recalibration, constants_hash changes).

## Smoke results (final config, REWARD_CLIP 100; runs/smoke, gitignored)
- Training: `rl.train --run-dir runs/smoke --n-envs 14 --total-steps 150000 --stage 1`: 152,320 steps in
  1.3 min (~2,000 steps/s end to end incl. PPO updates + 3 validations). Rollout 14 x 1088, batch 2176.
- Learning health (first -> last rollout): explained variance -0.32 -> 0.78; entropy 0.734 -> 0.708
  (no collapse); approx_kl 0.005 -> 0.0024, clip_fraction 0.0125 -> 0.0002 (the LR schedule anneals over
  150k here); value loss small (normalized returns); reward clip fraction 0.12-0.16 % (was 6-8 % at
  clip 10); twin advantage of stochastic Stage-1 training episodes +1.5..+3.3 % (noisy, no trend yet);
  masks applied rate 99.1 %. Action mix (training rollouts): CONTINUE 48 -> 45 %, HOLD 37 -> 33 %,
  SKIP 12 -> 18 %, YARD 0.6 -> 1.1 %, SHORT_TURN 0.7 -> 1.0 %, LOCAL 0.9 -> 1.1 %, EXPRESS 0.4 -> 0.6 %,
  REROUTE 0.3 -> 0.4 %. Validation (8 citywide seeds, deterministic): +13.7 % @61k, +14.7 % @107k
  (best), +12.4 % @152k; stuck -36..-40 %, wait -4 %.
- Export (best @106,624): 492 KiB, dynamo, parity max |dlogit| 3.6e-7, argmax agreement 100 %.
- Eval, 5 seeds (`rl.evaluate --seeds 5`): total cost -28.8 % (21.5 M -> 15.3 M), seed win rate 80 %,
  tunnel-stuck train-min -38.7 %, avg wait -2.4 %, riders served +0.4 %, overcrowded platform-min
  0.33 -> 4.7 (worse, small absolute). `--require-win` PASS; §4.8 bar FAIL (wait), expected at 1 % of the run.
- Stage 2 end to end (`--n-envs auto` = 13, runs/smoke_s2): 5,000-5,800 steps/s; 135k steps -> validation
  +17.9 % (win 100 %). Memory: ~200 MB per worker, ~3.7 GB for 13 workers + learner.
- bench_env 60 s, Stage 1, 14 workers: SubprocVecEnv 993 env-steps/s (71/env, 15 % of 14 x single);
  chunked RolloutWorkers 2,057 env-steps/s (147/env); single in-process 466/s; 252 decisions/sim-h.
  Stage 2 (30 s): 4,977 env-steps/s (356/env), 1,094 decisions/sim-h. Target >= 2,000: MET (chunked).
- Wait headroom on eval seeds (calm vs Normal, same scenario): disruptions add 38 % / 54 % of Normal
  wait on seeds 0/1 (smoke AI already -10.5 % / -10.7 % there), but only 13 / 1.4 / 9.3 % on seeds
  2/3/4, where the smoke AI over-intervenes (+3.2 %, +13.8 % wait) while still winning on cost
  (tunnel-stuck dominates the cost). If the long run misses the §4.8 wait bar, the first levers are
  W_HOLD_TRAIN/W_HOLD_RIDER (holds lengthen platform waits) or more Stage-2 steps.

## Recommended training run (NOT launched)
- Local (M5 Pro, from backend/):
  `.venv/bin/python -m rl.train --run-dir runs/local --n-envs auto --n-steps auto --total-steps 13000000 --stage auto --stage1-steps 3000000 --resume auto`
  = 13 workers x 1152 steps; Stage 1 3M steps at ~2,000/s (~25 min) + Stage 2 10M at ~5,500/s
  (~30 min) + 26 validations/checkpoints: **expected wall-clock ~1 h**. Re-running the same command
  resumes after any interruption (Ctrl+C checkpoints first).
  Then: `.venv/bin/python -m rl.export_onnx --checkpoint runs/local/best_model.zip --out models/nyc_subway_rcc_ppo.onnx`
  and `.venv/bin/python -m rl.evaluate --policy models/nyc_subway_rcc_ppo.onnx --seeds 50 --out models/ --require-win`
  (~1 min with 14 workers). If the §4.8 bar is not met, start a fresh longer run (schedules span the
  whole run): `--run-dir runs/local30m --total-steps 30000000` (~2 h).
- Colab (primary venue): run the notebook top to bottom on the runtime with the most vCPUs; the
  verify cell's 10-s bench shows that runtime's steps/s (expect roughly 150-350 steps/s per vCPU-worker,
  e.g. ~10-20k steps/s on a 64-worker TPU host -> the 13M default in well under an hour).

## Known issues / notes
- The lazy twin keeps the AI sim as the Normal trajectory until the first intervention; `twin_mode`
  `reset` is available for audits (asserts state_hash equality at every reset).
- Validation runs on the rollout workers between rollouts (8 citywide episodes ~2-3 s on 13+ workers).
- `rl.evaluate` on a `.zip` uses torch per worker (local only); production evaluation uses `.onnx`.
- The 150k smoke model passes `--require-win` but not the §4.8 bar (expected at 1 % of the run).
