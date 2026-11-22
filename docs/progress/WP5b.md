# WP5b: Per-train semi-MDP credit assignment

Status: **done** (2026-11-22, attempt 2): code, tests and docs complete; awaiting the orchestrator's 13M-step A/B.

- Branch: `rl/per-train-credit`
- Worktree: `.worktrees/rl2`
- Run Python from `<worktree>/backend` with `backend/.venv/bin/python`. That venv has no editable install, so the current directory's `sim`/`rl` packages are the ones imported.
- **Never edit anything under `backend` (the main tree).** A baseline training run imports from it.
- Do not commit and do not push. The orchestrator merges.

## Why

**Baseline run** `b8ab1440189e` (main tree `runs/local`, 13M steps): serial GAE, γ = 0.995 per decision, citywide twin-delta reward.
- Validation peaked at +12.95 % at 1.0M steps, then fell to 4–7 % through Stage 1.
- The training twin advantage stayed at only +1–4 %.
- Policy entropy fell from 0.70 to 0.33 and the non-CONTINUE rate from 52 % to 36 %: the policy drifts toward inaction.
- The smoke model over-intervened on mild seeds (+3 % and +14 % wait).

**Diagnosis.** It agrees with the Gemini Deep Research review the user commissioned, and our code confirms it (`rl/ppo.py: segment_gae`).
1. Each worker has one serialized trajectory that mixes trains.
   - δ_t bootstraps from a *different* train's V(o_{t+1}).
   - λ = 0.95 applies per *global* decision. That makes the GAE window about 18 decisions: roughly 1 sim-minute in Stage 2 (~1,090 decisions per sim-hour) and ~5 min in Stage 1.
   - Holds, skips and short-turns pay off over 5–15 min, and that payoff reaches the acting train only through other trains' value estimates.
2. γ per decision ties the physical horizon to the decision rate: about 48 min in Stage 1 but about 11 min in Stage 2.
3. A citywide reward makes every advantage absorb the effects of every other train's actions across the city.

## What to build

The serial path must stay bit-identical under `--credit serial`, so the A/B is clean.

### A. Per-route cost decomposition (sim; training-only; no behavior change)

- The sim keeps cumulative cost vectors since reset:
  - per-platform flow cost (wait + crowd): float64 `[n_platforms]`;
  - per-route train flow cost (tunnel + hold): float64 `[n_routes]`.
- Update them in the same code that produces `cost_totals()`, so there is one code path. The scalar totals must equal the sums of the vectors within 1e-9 relative; add a test.
- Expose `cost_vectors() -> {"platform": ..., "route_train": ...}` as copies. `copy()` must carry them.
- Immediate action costs stay separate: they belong to the acting train only.
- Keep `state_hash()`, frames and the protocol golden fixtures unchanged.
- Tick overhead must stay ≤ 3 % (`python -m sim.bench`).

### B. Env per-step data (`rl/env.py`)

`info` gains these keys:
- `t_s`: sim seconds at this decision.
- `route`: the ego's route index.
- `imm_cost`: unscaled immediate cost of this action; 0 if it was not applied.
- `d_flow_global`: unscaled (ΔC_ai − ΔC_twin) over the ticks advanced in this step, excluding immediate costs.
- `d_route`: float32 `[n_routes]`, the scoped flow deltas over the same ticks (see C).

The returned scalar `reward` stays exactly as it is today. The env checkers, plain MaskablePPO and the existing tests depend on it. Lazy-twin mode: before the fork, all deltas are 0.

### C. Scope weights (new `rl/credit.py`; precomputed from the graph at env or learner init)

For the ego's route r:
- **Platform weight:** 1.0 if r serves the platform; `NEIGHBOR_W` if a route that shares at least one run edge with r serves it; otherwise 0.
- **Route-train weight:** 1.0 for r; `NEIGHBOR_W` for track-sharing routes; otherwise 0.
- Scoped delta: `d_route[r] = W_plat[r]·(Δplat_ai − Δplat_twin) + W_rt[r]·(Δrt_ai − Δrt_twin)`.
- `--scope-neighbor-weight` sets `NEIGHBOR_W`; default 0.35.
- `--reward-scope global` uses `d_flow_global` instead, for the ablation.

### D. Per-train semi-MDP GAE (`rl/credit.py` + `rl/ppo.py` + `rl/rollout.py`)

Each worker's steps are time-ordered. A train's key is `(worker, episode ordinal, train_id)`. Train i decides at step k; its next own decision is at step k′ (or the episode ends).

**Per-transition quantities**
- Reward:
  `R_k = −REWARD_SCALE · ( imm_k + Σ_{m=k}^{k′−1} γ_min^{(t_mid(m) − t_k)/60} · d_scope_i(m) )`, where `t_mid(m) = t_m + ticks_m·DT_S/2`.
- Discount: `Γ_k = γ_min^{(t_{k′} − t_k)/60}`. The episode end is terminal: no bootstrap, finite horizon as today.
- TD error: `δ_k = R_k + Γ_k·V(o_{k′}) − V(o_k)`.
- Advantage: `A_k = δ_k + Γ_k·λ·A_{k′}`, with λ applied per *own* decision. Returns are `A + V`.

**Rollout boundaries and open transitions**
- A transition with no next own decision yet, and whose episode is not over, is *open*.
- At rollout end, carry open transitions to that worker's next rollout (the env state persists across rollouts) and close them there.
- The λ-recursion is truncated at an open transition, exactly as standard truncated GAE treats the rollout end: a closed transition whose next transition is open uses `A_k = δ_k`.
- Before running GAE, recompute V for carried transitions with the current critic. They keep their behavior `log_prob`.
- A transition still open after more than 3 rollouts is closed as terminal. On `--resume` the carry set is dropped.

**Buffer**
- Rebuild a `MaskableRolloutBuffer` every rollout holding exactly the closed transitions; the size varies.
- `num_timesteps` keeps counting env steps collected.
- Per-rollout minibatch size: `ceil(N / max(1, round(N / 2048)))`, so no tiny remainder batches.

**Reward normalization**
- In per-train mode, replace `ReturnNormalizer` with a running std (Welford) of the per-train discounted returns `G_k = R_k + Γ_k·G_{k′}` from the same backward pass.
- Divide R by that std; clip the normalized R to ±10.
- Save its state in `train_state.json`. `--no-reward-norm` keeps working.

**Flags and defaults**
- `--credit {per-train,serial}`. Make `per-train` the default only once the tests pass; `serial` is today's exact behavior.
- `--gamma-min 0.96`: per sim-minute, a 17-min half-life.
- λ = 0.95.
- `--target-kl`: per-train default 0.02; serial default None, so it reproduces the baseline.

**Logging**
- `rollout/closed`, `rollout/carried`, `rollout/mean_interval_s`, `rollout/R_mean`, `rollout/R_std`.
- `credit/scope_global_corr`: correlation of the scoped R with the global R over the same intervals.
- The existing action and mask rates.

### E. Calm anchor episodes

- The Stage 2 sampler gets a `calm_frac` parameter: with that probability an episode has 0 disruptions, with headway jitter on.
- CLI `--calm-frac`: per-train default 0.12; serial default 0.0.
- This targets the over-intervention on mild seeds. `eval_scenario` stays unchanged.

## Tests (add to `-m train` unless they are fast)

1. `cost_vectors` sums equal `cost_totals` for the AI sim and the twin, over many ticks of a disrupted scenario.
2. Under an all-CONTINUE policy, every `d_route`, `d_flow_global` and R is exactly 0, in all 3 twin modes.
3. A hand-built 2-train interleaved stream with known times, a terminal and a carry gives per-train GAE equal to a hand-computed reference.
4. Splitting one worker stream into 2 or 3 rollouts gives identical δ and identical λ = 0 advantages, and equals the truncated-GAE reference for λ > 0.
5. `--credit serial` reproduces `segment_gae` exactly (regression).
6. Smoke: 60k steps in per-train mode with 2 workers finishes; the checkpoint loads with `MaskablePPO.load`; `rl.export_onnx` passes parity.
7. The fast suite and `-m train` stay green, and the golden protocol fixtures are unchanged.

## Out of scope

Asymmetric/centralized critic, PopArt, the 3-stage curriculum, PLR replay, heuristic baselines, and any change to obs, actions, masks or the ONNX contract. The actor stays 216 → 8.

## CPU etiquette

- While `pgrep -f "rl.train --run-dir runs/local"` prints a PID (the baseline uses 13 of 15 cores), keep runs to ≤ 2 workers and ≤ 60k steps.
- Once it has exited, up to 8 workers and 300k steps are fine.
- Never start long training. The orchestrator runs the 13M-step A/B.

## Deliverables

- Code and tests in the worktree.
- Update ARCHITECTURE §4.4 and §4.6 (in the worktree) to describe the per-train scheme as the training default, with serial kept as an ablation.
- Keep this file current. Append progress below as you go, so an interrupted agent can resume.

## Code map (line numbers as of commit 642d67a; read only these spots)

**Simulator**
- `sim/cost.py:64-88`: `tick_costs()` returns the scalar terms `[wait, tunnel, hold, crowd]`; its helpers are just above.
  - Add a separate `tick_cost_vectors()` that returns per-platform wait+crowd and per-train tunnel+hold.
  - Leave the scalar path untouched so serial stays bit-identical, and test that the vector sums match it within 1e-9.
- `sim/engine.py:1700-1712`: per-tick accumulation (`terms = tick_costs(...)`, `self.cost += ...`).
  - Accumulate the vectors here only when an opt-in flag is set, for example `self.track_cost_vectors` (set by the env), so the server pays nothing.
  - Map trains to routes with `self.tr_route` (L437; −1 means inactive). Platform service is the route×platform 0/1 incidence matrix `self.route_pl` (L140-148).
- `sim/engine.py` methods: `copy()` L827 (it must copy the new vectors), `pending_decisions()` L854, `cost_totals()` L950, `state_hash()` L978 (leave its semantics unchanged).
- Track-sharing routes: use `g.pattern_edges(p)` (sim/graph.py L355) and `g.pat_route[p]`. Two routes share track if any edge appears in patterns of both.

**Env** (`rl/env.py`)
- `_cost_vector` L70.
- `step()` L189-245; the lazy fork is at L210.
- `_start_episode` L257, `_fork_twin` L279, `_advance` L283.
- The info dict is about L325-332 and already carries `train_id`, `ticks`, `cost_ai` and `cost_twin`.

**Rollouts** (`rl/rollout.py`)
- `Chunk` L132.
- `_collect` L149-185: the per-step loop, which reads `info["ticks"]` and `info["applied"]` at L172-173. Add per-step `t_s`, `train_id`, `route`, `imm`, `d_scope` and an episode ordinal.
- `_worker_main` L187, `Segment` L243, `_concat` L264, `RolloutWorkers.collect` L311.

**PPO** (`rl/ppo.py`)
- `ReturnNormalizer` L38-77, `segment_gae` L80-93.
- `ChunkedMaskablePPO._setup_model` L107, `collect_rollouts` L121.
- `_fill_buffer` L152-178 is where the per-train path branches. Keep the serial branch verbatim.
- `_log_rollout` L180.

**Train and callbacks**
- `rl/train.py`: `HYPERPARAMS` about L60-75, `parse_args` L88, `_new_model` L115, `main` L138 (normalizer resume at L185).
- `rl/callbacks.py`: `TrainState` L76-110 (`reward_normalizer` field L98), `RunCallback.save_state` L158-161.

**Scenarios**
- `sim/scenarios.py:169` `sample_training_scenario`. Stage 2 is L183-198 and `_draw_disruptions` is L113.

**Attempt 1** (2026-11-22) hit the 5-hour usage limit after about 170k tokens of exploration, with no code written.
- Write code early, in this order: D (the GAE core plus its unit tests on synthetic streams; this needs no sim), then B and C, then A, then E.
- Log each milestone below as you go.

## Progress log

### 2026-11-22 attempt 2: start
- Read the spec, rl/ppo.py, rl/rollout.py, rl/env.py, rl/train.py, callbacks (TrainState/save_state/Progress), sim/cost.py, engine cost/copy/state_hash/reset, scenarios Stage 2.
- Plan (design decisions, so a restart need not re-derive them):
  - `rl/credit.py`: `scope_weights(g, nw)`, `PerTrainCredit` (per-worker open set keyed by train slot within the
    current episode; forward pass over steps opens/closes transitions and accumulates discounted scoped/global
    flow sums sequentially per step, so splitting a stream into rollouts is bit-identical), `per_train_gae`
    (backward pass in stream-position order; λ truncated where the next own decision is still open),
    `TransitionReturnNormalizer` (Chan/Welford merge of G_k, state has `kind: "per_train"`).
  - sim: new `tick_cost_vectors()` in cost.py; engine accumulates `cost_plat`/`cost_route_train` in
    `_accumulate_costs` only if `self.track_cost_vectors`; these 3 attrs are excluded from `state_hash()`.
  - env: `credit_info=True` ctor flag (trainer passes False in serial mode so serial pays nothing);
    imm = Δtotal over `apply_actions`; d_flow_global = (ΔC_ai − imm) − ΔC_twin; d_route = W_plat@Δplat + W_rt@Δrt.

### 2026-11-22 milestone D-core done
- `backend/rl/credit.py` written: `scope_weights`/`track_sharing` (C), `PerTrainCredit.process` (forward pass),
  `per_train_gae`, `discounted_returns`, `TransitionReturnNormalizer`, `assemble` (segments -> CreditBatch rows +
  stats), `per_rollout_batch_size`. `sim/engine.py` gained `route_platform_incidence(g)` (read-only view of `_static(g).route_pl`).
- `backend/tests/test_credit.py` (fast, no torch/sim): hand-computed 2-train stream w/ terminal + carry (test 3),
  split invariance 1/2/3 parts (test 4, exact δ), forced terminal after 3 carries, normalizer, batch size. 15 pass.
- Gotchas: carried rows come first in *slot* order (not time order); nxt[j] > j still holds. Γ uses math.exp at
  closure (scalar, layout-independent). Python round(7.5) = 8 -> 15,360 rows give batch 1920.
- Next: A (sim cost vectors) + B (env info) together, then rollout/ppo wiring, train flags, E.

### 2026-11-22 milestones A, B, C, E + wiring written
- A: `sim/cost.py tick_cost_vectors`; engine `track_cost_vectors` flag (init False), `cost_plat`/`cost_route_train`
  (alloc in `_alloc`, zeroed after warm-up), accumulation in `_accumulate_costs` (bincount by `tr_route`),
  `cost_vectors()`; `_ACCOUNTING_ATTRS` excluded from `state_hash()`. Sanity: sums match 1e-16, hash unchanged.
- B: env ctor `credit_info=True, scope_neighbor_weight, calm_frac`; info keys t_s/route/imm_cost/d_flow_global/d_route;
  reward == clip(-S*(imm+d_flow_global)) exactly (checked 3000 random steps).
- rollout: `CREDIT_FIELDS` per-step arrays in Chunk.credit -> Segment fields; worker `ep_ord`.
- ppo: `credit=` ctor kw (class default serial), `_fill_buffer_per_train` (assemble -> new MaskableRolloutBuffer,
  batch_size per rollout), `train()` skips buffers < 2 rows in per-train, `_log_credit`; serial `_fill_buffer` verbatim.
- train: `--credit/--gamma-min/--reward-scope/--scope-neighbor-weight/--target-kl/--calm-frac`, `CREDIT_DEFAULTS`,
  `credit_settings()` stored in TrainState.credit; resume loads normalizer only if `kind` matches. export card has credit.
- E: `sample_training_scenario(..., calm_frac=0.0)`; no extra RNG draw at 0.
- Next: fast suite, -m train, new env/PPO tests (tests 1, 2, 5), bench, serial bit-identity check, smoke 60k, docs.

### 2026-11-22 tests green (fast + new train tests)
- Fast suite: 231 passed. `tests/test_credit_train.py` (-m train): tests 1, 2 (3 twin modes), calm anchors,
  test 5 serial == segment_gae, per-train buffer == assemble + SB3 train() on variable buffer, save/load,
  toy per-train smoke; 9 passed. Test 6 (`test_per_train_60k_smoke_real_network`, slow+train) not yet run.
- Bench: server tick unchanged (0.828/0.843 vs main 0.832/0.841 ms); tracked tick (env) +2.25 %.
- Gotcha: ticks advanced inside reset() before the first decision belong to no step (as in WP5).
- Next: serial bit-identity A/B vs main tree (chunk == n_steps for determinism), test 6, full -m train, docs.

### 2026-11-22 serial A/B, test 6, docs
- Serial bit-identity: main tree (642d67a, untouched) vs worktree `--credit serial`, 2 workers, `--n-steps 512
  --chunk 512` (one chunk per worker = deterministic), 3,072 steps: policy + optimizer weights bit-identical,
  normalizer state identical; only validation wall-clock `seconds` differs.
- Test 6 (`pytest tests/test_credit_train.py -m "train and slow"`): 61,440 steps, 2 workers, stage 2, 32 s; loads with
  MaskablePPO.load, export parity passed. Metrics: closed ~15.1k/15,360, carried 250-440 (all closed next rollout,
  forced 0), mean own interval ~340 s, scope/global corr 0.51-0.61, EV -0.02 -> 0.19, approx_kl <= 0.01.
- Docs: ARCHITECTURE §3.1 (cost_vectors), §4.4 (training credit), §4.5 (env kwargs/info keys), §4.6 (per-train
  default vs serial table, rollout boundaries, calm anchors). Added `perf/credit_s` log.
- Next: full -m train + fast suite rerun, protocol fixtures check, final self-review, structured result.

### 2026-11-22 done
- Full `-m train`: 30 passed (incl. test 6, 52 s). Fast suite: 231 passed, 6 skipped. Main tree untouched (git clean).
- Learner-side `perf/credit_s` ~0.025 s per 15,360-step rollout (collect 4-6 s).
- GAE self-review vs spec: R = -REWARD_SCALE*(imm + sum w*d) with d = AI - twin (minus sign, tested by hand
  stream); discount exp(ln(gamma_min)/60 * dt) = gamma_min^(dt/60) per sim-minute for both w(t_mid) and Gamma;
  t_mid = t_m + ticks_m*DT_S/2 with t_m = episode_time_s before the action (start of the step's ticks); episode end
  and >3 carries close as terminal (v_next = 0); lambda-recursion only when nxt is closed in the same rollout;
  carried-in V recomputed with the rollout's broadcast critic (NumpyActorCritic(params)) before G and GAE.
- A/B command (baseline b8ab1440189e used 13 workers, n_steps auto = 1152):
  `cd <worktree>/backend && backend/.venv/bin/python -m rl.train --run-dir runs/pertrain
  --n-envs 13 --total-steps 13000000 --stage auto --credit per-train --resume auto`

### 2026-11-22: orchestrator, A/B results and WP5c
**50 held-out seeds** (§4.7). Baseline is serial credit, best checkpoint at 4.0M steps; per-train is the best checkpoint at 4.5M steps.

| Metric | Baseline | Per-train |
|---|---|---|
| Seed win rate | 68 % | 84 % |
| Total cost | −21.3 % | −21.4 % |
| Avg wait | −4.6 % | −5.3 % |
| Tunnel-stuck | −32.9 % | −18.0 % |
| Overcrowded | −18.6 % | −15.1 % |

Neither passes §4.8.

**Per-train validation** was stable at +17–19 % through 13M steps. The baseline collapsed to +8 %.

**Presets at 60 min, avg wait.** Per-train is better on lex (−3.4 %), Times Sq (−0.1 % vs +7.5 %), Canarsie (−12.9 %) and 8 Av (+3.3 % vs +8.1 %). Per-train is the current champion: `.worktrees/rl2/backend/runs/pertrain/export_best`.

**Wait ceiling** (`scratchpad/evaltools/wait_ceiling.py`):
- Disruptions add 1.03 min of avg wait (20.9 % of Normal's 4.92). Calm is 3.90.
- 20 of 50 seeds add less than 10 %, so −10 % is impossible for any dispatcher on those seeds.
- The models recover 22 % (baseline) and 25 % (per-train) of the disruption's excess wait.

**Diagnosis.** About 95 % of non-CONTINUE actions are holds, and the entropy collapsed to 0.06.

**WP5c (run `runs/pertrain_hold`):**
- `W_HOLD_RIDER` 0.05 → 0.5 (holds were about 100× cheaper per rider-minute than platform waits).
- `ent_coef_end` 0.001 → 0.003.
- `test_tunnel_cost_dominates_hold_cost` now checks tunnel ≫ hold over a 10-min stoppage (~28×) and tunnel > 2× hold in minute 1.

### 2026-11-22: orchestrator, WP5c result, rule-based benchmark, next run
**Evaluator.** `rl/evaluate.py` now runs a calm counterfactual per seed and reports `excess_wait_recovery`. The user approved the new bar design.

| On 50 seeds | Champion (per-train) | Hold-cost best | Rules, no switches | Rules, with switches | Baseline (serial) |
|---|---|---|---|---|---|
| Excess wait removed | **25 %** (CI 9–40) | 19 % | 20 % (CI 12–28) | 15 % | 22 % |
| Tunnel-stuck | −18 % | −22 % | −33 % | −33 % | −33 % |
| Seed win rate | 82 % | 80 % | 90 % | 84 % | 68 % |

- The rule-based policy is WP2's leverage policy, run through the evaluator: `scratchpad/evaltools/scripted_eval.py`. It is evaluation-only and never shipped.
- **WP5c (`W_HOLD_RIDER` 0.5) is reverted.** Holding went down, but wait got no better.
- **Proposed bar for the user:** excess-wait removal ≥ 20 % (≥ the rule-based dispatcher). The champion meets it. Its only remaining failure is tunnel-stuck (needs ≥ 30 %).
- **Next run** (`runs/pertrain_global`, 13 workers): the champion config with `--reward-scope global`. The route scope may be what costs tunnel performance.

### 2026-11-22: orchestrator, citywide-reward result, 20 % bar confirmed, 25M run
**Bar.** The user confirmed the excess-wait bar at **20 %** (the rule-based benchmark).

**Per-train with `--reward-scope global`** (`runs/pertrain_global`, best checkpoint at 11.0M steps), on 50 seeds:
- total cost −23.1 %; seed win rate 86 %;
- excess wait removed 24 % (CI 9–39), which passes;
- tunnel-stuck −27.4 %, which fails (the bar is 30 %);
- crowding −9.5 %.

It passes 3 of the 4 bar checks and is the **new champion**: `runs/pertrain_global/export_best`.

**Presets at 60 min, wait:** lex −2.0, QBL −18.6, Times Sq 0.0, Canarsie −6.5, Manhattan Bridge −2.3, 8 Av +9.3, Joralemon −5.8 (all %).

**Next run:** the same config for 25M steps (`runs/pertrain_global_25m`), to push tunnel-stuck past 30 %.

### 2026-11-22: orchestrator, 25M result and no-calm ablation
**25M-step citywide run** (`runs/pertrain_global_25m`, best checkpoint at 22.5M steps), on 50 seeds:
- cost −19.4 %, win 86 %, excess wait 21 %, stuck −26.4 %, crowding −2.1 %.
- It is **not better** than the 13M champion; training longer doesn't help.
- Presets: QBL −22.4 %, Canarsie −14.9 % (both better); 8 Av +9.6 %.

**Ablation** `runs/pertrain_global_nocalm`: the champion config with `--calm-frac 0`. The serial baseline reached −33 % stuck without calm anchors, and the calm episodes may be teaching restraint that costs tunnel performance.

### 2026-11-22: orchestrator, no-calm ablation result and model summary
**No-calm ablation** (`runs/pertrain_global_nocalm`, best checkpoint at 8.0M steps), on 50 seeds:
- cost −19.5 %, win 80 %, excess wait 19 % (fails), stuck −20.0 % (fails), crowding −12.3 %.
- **Worse** than the champion. The calm anchors were not the cause of the tunnel shortfall.

**Champion stays** `runs/pertrain_global/export_best` (per-train credit, global scope, calm 0.12, 13M steps):
- cost −23.1 %, win 86 %, excess wait 24 % (passes), stuck −27.4 % (fails; the bar is 30 %).
- Run-to-run variance across single-seed runs is large (stuck −18 to −33 % across configs), so the 2.6-pt gap is near the noise floor.
- Options: ship with an honest model card (stuck −27 % vs the 30 % target), or later try multi-seed training / selection by 50-seed eval. The user decides.

### 2026-11-22: rule-based benchmark, full numbers (cited in the README)
- `scratchpad/evaltools/scripted_eval.py noswitch` on the 50 eval seeds:
  - win 90 %, total cost −19.2 %, avg wait −4.1 %, stuck −33.2 %, crowding −6.4 %, excess wait removed 20 % (CI 12–28 %).
  - The README fact-check reproduced these numbers.
- With track switches: win 84 %, cost −16.3 %, wait −3.1 %, stuck −33.1 %, crowding +4.0 %, excess 15 %.
