# Product Requirements Document: Subway Domino

- **Version:** 1.1
- **Status:** Ready for Engineering
- **Live Production URL:** `https://kenmar.dev/subway`
- **Core Stack:** Python 3.11+ (`Gymnasium`, `sb3-contrib` `MaskablePPO`, `FastAPI`, `ONNX Runtime`), Google Colab (RL Training), Next.js (App Router, TypeScript)
- **Production Hosting:** Vercel Hobby (Frontend via `kenmar.dev` subpath rewrite) + Railway Hobby (Always-on FastAPI/WebSocket ONNX Backend)
- **Data Foundation:** Official MTA / NY Open Data — GTFS Static (`fgm6-ccue`), Station Complexes (`5f5g-n3cz`), Hourly Ridership: Beginning 2025 (`5wq4-mkjj`)

---

## 1. Executive Summary & Problem Statement

Public transit datasets and reinforcement learning (RL) models are typically presented as static charts, notebooks, or opaque benchmark tables that fail to resonate with non-technical audiences—while playful transit games usually rely on hand-coded heuristics rather than genuine machine learning.

**Subway Domino** bridges that gap as an interactive, citywide **Human vs. AI New York City Subway operations simulator**, accessible directly from `kenmar.dev` at `kenmar.dev/subway`. Users explore the entire NYC subway network across Manhattan, Brooklyn, Queens, and the Bronx, trigger realistic rail disruptions at any station or track segment, and watch a trained **Reinforcement Learning Rail Control Center (AI Dispatcher)** contain the meltdown in real time using authentic MTA dispatching maneuvers.

### Product Promise
- **For non-technical viewers:** Click a project on `kenmar.dev`, land straight in a live running NYC subway map with zero cold-start delay, break a familiar station, flip on the AI, and visually see trains reroute and crowds clear within 10 seconds.
- **For technical & ML reviewers:** Authentic railway constraints (A/B division isolation, fixed-block vs. CBTC signaling, real turnaround latencies), zero fake heuristic fallbacks in AI mode, real NYC Open Data grounding, reproducible Google Colab `MaskablePPO` training, and production CPU inference via ONNX Runtime.

---

## 2. Target Audience

| Audience | Primary Goal | What Convinces Them |
| :--- | :--- | :--- |
| **Non-Technical Viewers** (Recruiters, hiring managers, executives, NYC commuters) | Launch from `kenmar.dev` and play immediately without instructions, wait screens, or ML jargon. | Instant load, recognizable NYC lines and stations, immediate cause-and-effect meltdowns, and clear before/after proof that the AI saves riders. |
| **Technical & ML Reviewers** (ML engineers, software engineers, data scientists) | Evaluate systems engineering, RL rigor, data pipelines, and production deployment craft. | Strict physical action masking, anti-reward-hacking rules, real MTA data ingestion, reproducible Colab benchmarks, multi-user WebSocket isolation, and lean ONNX production architecture. |

---

## 3. Core Product Loop

1. **Enter From Portfolio:** The user clicks Subway Domino on `kenmar.dev` and lands directly at `https://kenmar.dev/subway` on a live, smoothly running simulation of all active NYC subway lines and ~420+ station complexes.
2. **Break the City:** The user pans, zooms, searches, or selects a curated preset—or clicks any station, track segment, interlocking, or river crossing—to trigger a realistic disruption (`local_block`, `express_block`, `full_station_closure`, or `signal_delay`).
3. **Observe Baseline Collapse:** Under **Normal Schedule**, trains obey fixed timetables and signals without dynamic rescheduling—stacking into tunnel queues behind blockages while crowding dwell delays trigger cascading train bunching and transfer-hub overflow.
4. **Activate the AI Dispatcher:** The user flips a single toggle to **AI Dispatcher**. The trained `MaskablePPO` ONNX model takes real-time control, executing authentic MTA Rail Control Center (RCC) maneuvers as if operating real trains carrying real human passengers.
5. **Compare Real Rider Outcomes:** Live passenger-first metrics (Average Wait Time, Overcrowded Platforms, Tunnel-Stuck Trains, Riders Served) and plain-English dispatch callouts show *how* and *why* the AI contains the meltdown.

---

## 4. Technical & Operational Specifications

### 4.1 Real-Life Subway Physics & Safety Rules
The shared simulation engine (`backend/sim/NYCSubwaySim`) enforces real railway mechanics so the RL agent cannot exploit arcade loopholes:
- **Division Separation:** Strict **A Division (IRT: 1–7, 42 St S)** vs. **B Division (IND/BMT: A–Z, Franklin/Rockaway S)** loading-gauge isolation. Trains never cross onto incompatible division tracks.
- **Multi-Track Topology:** Models separate Local and Express tracks on 4-track trunks (Lexington, 7th, 8th, 6th, Broadway, Queens Blvd, Fulton, Eastern Pkwy), 3-track peak-direction express lines (7, Jerome, Concourse, Pelham, West End), and 2-track local-only lines (L, G).
- **Signaling:** Legacy lines enforce **Fixed-Block Signaling** (block occupancy, red/yellow/green aspects, safe braking curves, cascade queueing); CBTC-equipped corridors (L, 7, Queens Blvd, 8th Ave CBTC) operate with tighter headways and faster recovery.
- **Physical Turnaround & Yard Latencies:** Short-turning requires discharging all passengers at a physical crossover/pocket track plus 3–5 simulated minutes for crew cab-walk and brake test before entering the reverse track. Yard gap trains draw from real MTA yards (207 St, Concourse, Coney Island, Jamaica, Pitkin, Livonia, Corona, Canarsie, 239 St, Pelham) with finite fleet caps and deadhead entry lead times.
- **Passenger & Crowding Realism:** Tracks platform wait-time age distributions (not just headcount), nonlinear door-open dwell delays causing organic train bunching, inter-line transfer spillover at disrupted hubs, and onboard passenger destination buckets so skips/short-turns produce real displacement/overshoot costs.

### 4.2 Authentic MTA Rail Control Center (RCC) Action Space
Implemented via an event-driven, parameter-shared **`MaskablePPO`** policy evaluating a fixed-size ego-centric corridor window ($K=6$ stations ahead, $K=3$ behind) with `action_masks()` defined directly on the `gymnasium.Env` class for `SubprocVecEnv` compatibility:

| Action ID | RCC Command | Physical Masking Constraint |
| :--- | :--- | :--- |
| `0` | `CONTINUE_SCHEDULED` | Always valid. |
| `1` | `HOLD_IN_STATION` | Masked if max hold exceeded or lead-train headway is already wide. Preferred over trapping riders in tunnels. |
| `2` | `RUN_EXPRESS` | Masked unless at a physical local-to-express interlocking on a 3/4-track trunk with an open express path ahead. |
| `3` | `RUN_LOCAL` | Masked unless at a physical express-to-local interlocking with an open local path ahead. |
| `4` | `SKIP_STOP` | Masked at major hubs/terminals, on consecutive skips, or if next stop has critical wait time unless physically closed. |
| `5` | `SHORT_TURN_RELAY` | Masked unless station has a physical turnaround switch/pocket track, is not a terminal, and genuine downstream blockage/imbalance exists. |
| `6` | `REROUTE_VIA_COMPATIBLE_TRUNK` | Masked unless at a real **same-division** junction with a valid alternate corridor path. |
| `7` | `DISPATCH_YARD_GAP_TRAIN` | Masked if nearest compatible yard reserve fleet is exhausted, cooldown is active, or no downstream segment is starved. |

### 4.3 Anti-Reward-Hacking Objective
- **Quadratic Wait-Age & Equity Penalty:** Penalizes `sum(waiting_riders * wait_age^1.5)` so starving quiet outer-borough local stations is heavily punished.
- **Tunnel-Trapping Penalty:** Penalizes train-minutes stopped between stations in tunnels much more heavily than controlled open-door station holds.
- **Platform Overcrowding Penalty:** Steep penalty above 80% safe platform capacity.
- **Passenger Displacement Honesty:** `SHORT_TURN_RELAY` incurs cost proportional to actual onboard riders forced off; `SKIP_STOP` incurs cost proportional to onboard riders whose destination was bypassed; trunk reroutes and yard put-ins carry realistic operational friction costs.

---

## 5. User Stories & Acceptance Criteria

### Epic 1: Citywide NYC Open Data & Network Graph
*As a user, I want to explore the real NYC subway system across all four connected boroughs so that disruptions feel authentic to New York City rather than a toy map.*

**Acceptance Criteria:**
- [ ] Automated ingestion script (`backend/scripts/build_nyc_network.py`) builds the complete NYC rail graph from MTA GTFS Static, Station Complexes (`5f5g-n3cz`), and Hourly Ridership (`5wq4-mkjj`) without hard-coding station subsets.
- [ ] Queries hourly ridership via server-side SoQL aggregation (`$group`) to extract `AM Peak`, `Midday`, `PM Rush`, and `Late Night` profiles per complex without downloading raw 40M+ row tables.
- [ ] Covers Manhattan, Brooklyn, Queens, and the Bronx across all active lines (`1–7`, `A–Z`, `S` shuttles) and ~420+ complexes, including inferred express/local tracks, interlockings, and rail yards.
- [ ] Saves compact cached artifacts in `backend/data/` (with deterministic fallback generation) so local dev, Colab training, and Railway production boot completely self-contained.

### Epic 2: Real-Life Subway Physics & Disruption Sandbox
*As a user, I want to break any station or track and see realistic railway chain reactions—not arcade behavior.*

**Acceptance Criteria:**
- [ ] Users can pan, zoom, search any station or line across NYC, and click to trigger or clear `local_block`, `express_block`, `full_station_closure`, or `signal_delay` disruptions.
- [ ] Users can control playback speed (`Pause`, `1x`, `2x`, `5x`), switch time-of-day demand profiles, trigger curated meltdown presets, and reset cleanly in one click.
- [ ] Zero A/B division crossing violations, realistic fixed-block vs. CBTC spacing, enforced 3–5 min short-turn relay latencies, and realistic yard deadhead delays verified by automated `pytest` suites.

### Epic 3: Real-Life AI Rail Control Center (RL Dispatcher)
*As a user, I want to toggle on a real trained AI Dispatcher that makes the best possible operational decisions as if carrying real human passengers.*

**Acceptance Criteria:**
- [ ] **Single Source of Truth:** FastAPI runs the exact same `NYCSubwaySim` physics engine used in training and executes the trained Colab model (`nyc_subway_rcc_ppo.onnx` via ONNX Runtime, with optional `.zip` support locally)—never hard-coded `if/else` heuristics in AI mode.
- [ ] All 8 RCC actions are strictly guarded by `action_masks()` in both training and live inference.
- [ ] Live UI surfaces plain-English callouts whenever the AI issues non-trivial dispatch commands so non-technical visitors understand the strategy at a glance.

### Epic 4: Reproducible Google Colab Training Pipeline
*As the developer, I want to train, resume, benchmark, and export the RL model cleanly on Google Colab without rewriting code.*

**Acceptance Criteria:**
- [ ] Core simulation physics step is NumPy-vectorized to sustain high throughput across parallel `SubprocVecEnv` workers on Colab.
- [ ] Ready-to-run `colab/train_subway_domino_colab.ipynb` mounts Google Drive (`/content/drive/MyDrive/subway-domino/`) for persistent checkpoints, TensorBoard logs, and automatic resume-from-checkpoint recovery across Colab timeouts.
- [ ] Implements 2-stage curriculum + domain randomization: Stage 1 (single-trunk bunching and track faults) -> Stage 2 (citywide multi-disruption meltdowns across all 4 demand profiles).
- [ ] `backend/rl/evaluate.py` benchmarks **Normal Schedule vs. Trained AI** across 50 unseen meltdown seeds and `backend/rl/export_onnx.py` exports a portable `nyc_subway_rcc_ppo.onnx` ready to drop into `backend/models/`.

### Epic 5: Interactive UX & Portfolio Integration (`kenmar.dev/subway`)
*As a visitor clicking Subway Domino on `kenmar.dev`, I want an intuitive, flagship-grade experience that feels seamlessly part of Kenmar's portfolio.*

**Acceptance Criteria:**
- [ ] Visual aesthetic, layout, map projection/renderer, and HUD design are left to engineering/design discretion, provided the map remains crisp and legible at both full-NYC overview zoom and dense Midtown hub zoom.
- [ ] First-time users can trigger a meltdown and toggle `Normal` vs. `AI Dispatcher` within 10 seconds without external instructions.
- [ ] App runs seamlessly under subpath `basePath: "/subway"` both locally (`http://localhost:3000/subway`) and behind the `kenmar.dev` Vercel rewrite (`https://kenmar.dev/subway`), with zero broken static asset paths.
- [ ] Includes a subtle `<a href="https://kenmar.dev">` back-link in the header/corner so visitors can return to the main portfolio in one click.
- [ ] Includes an unobtrusive methodology/info modal documenting the MTA Open Data datasets, RL reward/masking rules, and live session action breakdown.

### Epic 6: Production Deployment, Multi-User Isolation & Cost Efficiency
*As the site owner deploying on Vercel Hobby + Railway Hobby, I want zero cold starts, multi-user safety, and predictable ~$5/month backend costs.*

**Acceptance Criteria:**
- [ ] Production backend uses `backend/requirements-prod.txt` containing ONLY lightweight runtime packages (`fastapi`, `uvicorn[standard]`, `numpy`, `onnxruntime`, `pydantic`)—strictly excluding `torch`, `stable-baselines3`, `sb3-contrib`, `gymnasium`, and `tensorboard` (kept in `requirements-train.txt`).
- [ ] `backend/Dockerfile` (`python:3.11-slim`) bundles code, cached `backend/data/`, and `backend/models/nyc_subway_rcc_ppo.onnx`, binds dynamically to `${PORT:-8000}` with `--workers 1`, and exposes `GET /health` returning `{"status": "ok", "model_loaded": true}`.
- [ ] Every WebSocket connection instantiates an isolated `NYCSubwaySim` instance while sharing only the read-only NYC graph arrays and the single loaded `onnxruntime.InferenceSession` in memory.
- [ ] Server enforces a concurrent session cap (`40` active connections with a clean at-capacity message) and a 5-minute server-side idle timeout; frontend disconnects/pauses WebSockets if the browser tab is hidden for >60 seconds (`visibilitychange`) and reconnects seamlessly on return.
- [ ] WebSockets connect directly from the browser to Railway (`wss://...`)—never proxied through Vercel rewrites—and `CORSMiddleware` allows `https://kenmar.dev` and `http://localhost:3000` via `ALLOWED_ORIGINS`.

---

## 6. Deployment & Portfolio Routing Architecture

### 6.1 Subpath Integration (`kenmar.dev/subway`)
Subway Domino lives in its own independent GitHub repository and Vercel project, surfaced seamlessly under `https://kenmar.dev/subway`:

1. **Subway Domino Repo (`frontend/next.config.ts`):**
   ```ts
   import type { NextConfig } from "next";

   const nextConfig: NextConfig = {
     basePath: "/subway",
   };

   export default nextConfig;