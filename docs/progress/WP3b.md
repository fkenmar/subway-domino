# WP3b: Immersive, fluid UI redesign

Status: **done** (2026-11-22: all five phases plus final verification; see the progress log). Work in the main tree, `frontend/` only.

`next dev` is already running on :3000 against the real backend on :8000, so the user sees changes live through hot reload. Keep the app working between edits. Do not touch `backend/`. Another agent is working in `.worktrees/rl2`.

## The user's feedback (2026-11-22)

- **"The UI isn't very fluid and tasteful."**
- Chosen direction: **Immersive map**:
  - the map runs edge to edge and is the hero;
  - trains are drawn as glowing pills you can follow;
  - one compact floating scoreboard;
  - presets live in a single menu;
  - panels slide in only when needed;
  - transitions are smooth.
- What feels least fluid: **trains jump / choppy**, **static UI** (abrupt changes, no transitions) and **layout feels busy**.
- The user's mock:
  ```
  ┌─────────────────────────────────────────────┐
  │ Subway Domino           8:22:05 AM · 1 min/s │
  │      ╲   map fills the whole screen  ┌─────┐ │
  │   ━━━━(▬)━━━━━(▬)━━━  trains = pills │ AI  │ │
  │      ╱                               │-12% │ │
  │                                      │wait │ │
  │                                      └─────┘ │
  │ [Break the city ▾]  ▶ 1·2·5 min/s [Normal|AI]│
  └─────────────────────────────────────────────┘
  ```
- Already done: the clock shows seconds, and the speed buttons read "1 min/s · 2 min/s · 5 min/s". 1x is 60 sim-s/s. Keep both.

## Measurements (orchestrator, before the redesign)

**Rendering** is a steady 60 fps (p95 16.8 ms) in Chromium at both citywide and zoomed views. Frames arrive every 250 ms (p95 260 ms). Neither the server nor the canvas is the problem.

**Trains teleport.** The interpolator in `src/lib/interp.ts` glides only when the new edge *directly* continues the old one; otherwise it snaps.
- At 5 min/s (75 sim-s per frame), **7 %** of train updates snap: about 31 trains jump each frame.
- At 1 min/s, 1.4 % snap (about 6 per frame).
- The measurement script is `<scratchpad>/evaltools/frame_speed.py`. Run it from `backend/` with `.venv/bin/python <script> 5x 8`; the backend must be running. It prints the snap percentage.

**Before screenshots** (1440×900 @2x): `.../scratchpad/ui/ui_citywide.png`, `.../scratchpad/ui/ui_zoomed.png`.
- Trains are tiny dots that look like station dots.
- The network fills about a third of the screen.
- A preset chip row, a tutorial popover, a big metrics panel, a dispatch log, a wide bottom bar, a map key and zoom controls all compete for attention.
- Sparklines are jagged.
- Deep zoom shows an empty dark screen.

`.../scratchpad/evaltools/ui_probe.mjs` takes screenshots and measures FPS. Run it from `frontend/` with `node <script> <outdir>`.

## Goals and acceptance criteria

### 1. Motion: no teleporting

- When a train moves across 2 or more edges between frames, interpolate along the connecting path at constant speed.
  - Search the graph of continuing edges (to-station of one equals from-station of the next), depth ≤ 6.
  - Prefer edges the train's route uses (the geo segments list their routes). Cache paths per (prev, next) pair.
- Snap only on real discontinuities: a new trip in a reused slot, a reversal at a terminal or turnback, or no path within the depth limit.
- Acceptance:
  - the snap rate at 5 min/s is ≤ 0.5 % of updates;
  - unit tests cover the path interpolation;
  - the render stays at 60 fps.
- Optional, if cheap: ease velocity changes between frames (for example, interpolate along the path using a smoothed cumulative distance) without adding more than one extra frame of latency.

### 2. Trains as glowing pills; quieter base map

- Draw each train as a route-colored capsule, oriented along the track, with a soft glow. It must be readable at citywide zoom: at least about 7 px long on a 1440×900 viewport, scaling up with zoom.
- Show load subtly, for example by fill intensity.
- Status cues:
  - stuck in a tunnel: a red pulse;
  - held by an action: an amber ring;
  - gap train or AI-affected: a short highlight when an action applies.
- De-emphasize stations: small, low-contrast ticks. Hubs and interchanges get white rings. Lines can be slightly thicker with soft casing.
- The map is the hero. On first load, fit the network to the viewport minus the overlay insets.
- Clamp the max zoom so the screen never shows an empty map.

### 3. Immersive layout: calm, fewer boxes

- The map runs edge to edge (full-bleed canvas).
- **Top bar (slim, translucent):**
  - left: logo and "Subway Domino";
  - right: the clock (`8:22:05 AM · AM Peak · 1 min/s`), a search icon that expands into search (keep `/` as the shortcut), and the "How it works" info button.
  - Keep the "← kenmar.dev" back link, small.
- **Bottom-center control pill:** play/pause, `1 · 2 · 5 min/s`, a **Normal | AI** segmented toggle with a sliding indicator, the profile menu, and reset. It should be slimmer than today's bar.
- **Bottom-left "Break the city ▾" menu:**
  - all 7 presets, with route bullets and one-line descriptions;
  - a note that any station, track or interlocking on the map can be clicked;
  - the active disruptions list, with clear-one and clear-all.
- **Compact floating scoreboard (top-right):**
  - **Normal mode:** average wait, tunnel-stuck trains, overcrowded platforms and riders served, small, each with a smoothed sparkline.
  - **AI mode:** the headline becomes AI vs Normal. Show the live delta for each metric against the twin, for example "Wait −12 % vs Normal", green when better and red when worse, plus "since AI took over m:ss".
  - Values must come from the existing live and twin metrics. Never invent or extrapolate numbers (honesty rule).
- **Dispatch log:**
  - closed by default, as a small "Dispatch log · N" chip that opens a slide-in drawer;
  - AI callouts appear as brief toasts (near the bottom-right, above the controls) and fade out;
  - nothing stays on screen permanently except the scoreboard.
- **Onboarding:** at most one small, dismissible hint, anchored to "Break the city ▾". It must never cover the map center.
- Map key and zoom controls: compact, bottom-right, secondary styling.
- Hide Next's dev indicator if it's configurable; it only shows in dev.

### 4. Transitions: a UI that feels alive, not static

- Tween numbers (about 250–400 ms, ease-out) and smooth the sparklines (a moving average or curve, consistent scale).
- Panels and drawers enter and exit with fade and slide (160–240 ms, ease-out). The mode toggle has a sliding indicator; menus pop with a slight scale.
- Respect `prefers-reduced-motion`.
- Mode switch: a subtle, brief map-wide cue (for example, a vignette pulse or a scoreboard header color change) so the user feels AI take over.

### 5. Constraints

- Keep the WebSocket protocol, the frame decoder, the store's contract and the golden-decoder tests. All `public/` URLs go through `asset()`, and `basePath` stays `/subway`.
- Keep the PRD features:
  - Pause / 1 / 2 / 5 speeds, profiles, presets, one-click reset;
  - click any station, track or interlocking to disrupt it;
  - the info modal, search, callouts and accessibility labels.
- Responsive: desktop 1440 and laptop 1280 must be excellent. Mobile 390 must be usable: the scoreboard collapses into a pill and the controls stack.
- Green before finishing: `npm run typecheck && npm run lint && npm test && npm run build`. Update the e2e `screenshots.spec.ts` if selectors change, and re-run `npm run screenshots`. It uses the mock server on :8765 and port 3100, not the user's :3000.

## Process

1. **Work in phases**, logging each one below as you go: **motion**, then **trains and base map**, then **layout**, then **transitions**, then **polish**.
2. After each visual phase, take screenshots: run `ui_probe.mjs` against `:3000`, or Playwright against `:3100`. Look at them critically and fix anything that doesn't read as calm, premium and legible before moving on.
3. Final screenshots: desktop citywide in Normal mode, desktop with AI on during the `lex-union-sq` preset, laptop, and mobile.
4. Save them to `.../scratchpad/ui/after_*.png` and list them in the log.

## Code map

- `src/lib/interp.ts`: the interpolator (`interpolateTrain` and `FrameBuffer`); `src/lib/network.ts` builds the geo network.
- `src/map/renderDynamic.ts`: trains, chips, highlights. `src/map/renderBase.ts`: lines, stations, labels. `src/map/style.ts`: zoom tiers and colors. `src/map/engine.ts`: the frame loop and camera. `src/lib/camera.ts`: camera fitting and clamps.
- `src/components/*`: today's panels. The ones that change are `App`, `Header`, `PresetBar`, `MetricsHUD`, `ActivityFeed`, `CalloutLayer`, `Onboarding`, `MapControls`, `ModeToggle`, `SpeedControl`, `ProfileSelect`, `ResetButton`, `DisruptionList` and `Sparkline`. `src/lib/store.ts` holds the state; it is a contract, so extend it but don't break it.

> **Update 2026-11-22 (orchestrator):** the user asked to close the live app. The servers on :3000 and :8000 are stopped. Do **not** restart them.
> - For visual checks, use the self-contained Playwright setup (`npm run screenshots`: mock server on :8765 plus `next start` on :3100).
> - Or temporarily start the mock server and `next dev -p 3100` (or a real backend on :8010) for screenshots and the motion check.
> - Stop every server you start before finishing.

## Progress log

### 2026-11-22: phase 1 (motion), started
- Baseline from `frame_speed.py 5x 8`: 7.01 % snaps.
- Recorded 5x frame sequences from the live server (Normal; AI during `lex-union-sq`) with `scratchpad/evaltools/record_frames.py`. A Python prototype of the path rule (`path_proto.py`) leaves:
  - ~0.03 % true snaps (same slot and route, no path);
  - ~1.5 % **route changes**. These are new trips in reused slots: median 13 km apart. Short turns and reroutes keep the route, per ARCHITECTURE §4 actions.
- Plan:
  - `PathFinder` in `interp.ts`: layered min-cost search, depth ≤ 6, preferring the train's route edges, with extra cost for crossovers, turnbacks and reversals, cached per (prev, next, route);
  - a plausibility cap: path length ≤ 1.25 · V_MAX_CAP · Δclock + 150 m;
  - real discontinuities cross-fade (the old train fades out, the new one fades in) instead of sliding;
  - speed eased with a monotone Hermite, adding no latency.

### 2026-11-22: phase 1 (motion), done
- `src/lib/interp.ts`:
  - `PathFinder` (CSR out-edges by station; layered min-cost search, depth `MAX_PATH_HOPS = 6`; route bits per edge, with extra cost for crossovers (150 m), turnbacks and reversals (1500 m) and off-route edges (3000 m); cache keyed by prev, next and route);
  - `walkPath`, and `easeDistance`, a monotone Hermite whose start speed is the previous interval's mean, adding no latency;
  - the `FrameBuffer` resolves motion lazily on the first sample per frame and keeps `stats`;
  - discontinuities, appearing and vanishing trains, and resets cross-fade: `InterpolatedTrains.fade`, plus a `ghosts` output of `sample()`.
- `src/lib/network.ts`: `EdgeTable.routeBits` / `routeWords`.
- Tests:
  - `test/unit/interp.test.ts` covers the path finder, multi-edge glide, cross-fades, easing and reset fades;
  - `test/unit/interpRecorded.test.ts` replays `test/fixtures/frames_5x_{normal,ai_lex}.bin.gz` (live 5x recordings for graph e184af352bc6) through the real client code.
- Results at 5x:
  - snaps fall to **0.028 %** (Normal) and **0.049 %** (AI + lex-union-sq), and no continuing train steps more than 165 m per 1/10 frame;
  - new trips in reused slots are 1.5 %, and these cross-fade;
  - extended `frame_speed.py 5x 8` on the live server: old rule 7.01 % → client rule **0.007 %**; at 1x, 1.39 % → 0.000 %.
- `docs/ARCHITECTURE.md` §6 interpolation sentence updated to the new rule (the WP3 contract-change request). The frontend is its only consumer.
- Next: the engine must pass `ghosts` to `sample()` and draw `fade`. That is done in phase 2 together with the pill renderer.

### 2026-11-22: phase 2 (trains and base map), done
- **Trains** (`renderDynamic.ts` `drawTrains`) are route-coloured capsules oriented along the track:
  - sized by `style.trainSize(mPerPx)`: 8 × 3.6 px at the citywide fit, growing smoothly to 25 × 9.6;
  - the heading is smoothed per train id, so pills turn through curves;
  - load shows as brightness: the route colour is mixed 6–50 % towards white in 4 load buckets;
  - glow comes from two translucent fattened copies of each pill, batched per colour as fills.
  - Two other glow approaches were measured and rejected: per-train glow sprites ran at 38 fps, and a low-resolution bloom blit at 9 fps in headless Chromium.
  - At close zoom: a dark casing and a white "headlight" on the front cap. Route letters show from the neighborhood tier.
- **Status cues:**
  - tunnel-stuck trains get a pulsing red glow and a red outline;
  - held trains get an amber ring;
  - when a train gains an action status bit (hold, short turn, track switch, reroute, gap train, skip), it flashes a cyan ring for 1.5 s (green for gap trains);
  - rerouted trains get a dashed ring at close zoom;
  - ghosts and new trips are drawn with their fade.
- **Base map** (`renderBase.ts`, `style.ts`):
  - no grid;
  - land is `#111823`;
  - lines are mixed towards the land (`tier.lineAlpha`: 0.5 at city, 0.75 at street), so trains lead;
  - soft two-pass casing;
  - ordinary stations are small, low-contrast ticks;
  - hubs and multi-station complexes (`isInterchange`) are white rings on a dark core, with matching transfer links.
- **Never empty:**
  - `HitTester.nearestContent` and `Camera.setContentProbe` keep some network within 40 % of the smaller viewport side from the centre;
  - zoom-in about an empty spot stops (`limitZoomIn`), and pans are pulled back;
  - tests are in `camera.test.ts`.
- Probe (machine under load from the RL agent, load average 13 on 15 cores): **60 fps** citywide and in Midtown (p95 16.8 ms).
- Screenshots: `scratchpad/ui/p2b/{city,midtown}.png`.
- Next: phase 3 (layout).

### 2026-11-22: phase 3 (layout), first pass
- **Environment:**
  - `next dev` on :3000 exited around 02:59, and I restarted it (`npm run dev`, log in `scratchpad/frontend.log`);
  - the backend on :8000 was shut down gracefully at 02:59, and `backend/models/` is gone (another actor is changing it), so I did **not** restart it;
  - UI screenshots now use the protocol mock: `node scripts/mock-server.mjs --port 8766`, plus `NEXT_PUBLIC_WS_URL=ws://127.0.0.1:8766/ws next build && next start -p 3200`;
  - the tour script is `scratchpad/evaltools/tour3.mjs` (`BASE=http://127.0.0.1:3200`).
- **New layout** (`App.tsx` and `App.module.css`): a full-bleed map, plus:
  - a slim top bar (`Header`): logo, title, a small "← kenmar.dev", the clock with profile and speed, collapsible search (`/` still works) and the info icon;
  - the scoreboard (`MetricsHUD`) at the top right. Normal mode shows 4 metrics with tweened values and smoothed sparklines. AI mode shows "AI vs Normal", with honest twin deltas coloured green or red and "since AI took over 8m 40s". On phones it is a pill that expands;
  - the dispatch log (`ActivityFeed`): a chip under the scoreboard that opens a drawer, closed by default. While it is open, toasts step aside;
  - "Break the city ▾" (`PresetBar`) at the bottom left: all 7 presets with route bullets, a click-the-map note, and the active disruptions (`DisruptionList`) with clear and clear-all;
  - the control pill at the bottom centre: play/pause, a 1·2·5 min/s segment with a sliding thumb, the Normal|AI toggle with a sliding indicator, the profile menu and an icon reset;
  - map tools at the bottom right: key, −, + and fit;
  - toasts (`CalloutLayer`) at the bottom right: two lines each, fading in and out, lasting 6.5 s;
  - a single onboarding hint anchored to "Break the city".
- **New pieces:**
  - `lib/motion.ts`: `useDisclosure` (exit animations), `useDismiss`, and `useTweenedText` (rAF-driven DOM text, so no per-frame React render);
  - `ModeCue` (vignette pulse);
  - `store.trend` and `appendTrend`: the last 40 metrics points, so sparklines have data at 5x (the 10-sim-minute `history` holds only about 2 points at 5x);
  - `format.formatDuration`.
- **e2e spec updated:**
  - presets are opened through the menu;
  - search opens from its icon;
  - the AI header text is now "since AI took over";
  - the inset mirror: "side" is the right column at ≥700 px and the top on phones.

### 2026-11-22: phase 3 fixes and phase 4 (transitions), done
- **Layout fixes after the screenshot review:**
  - clock spacing;
  - scoreboard 296 px wide, so "Riders served" fits in AI mode;
  - "since AI took over 8m 40s" (`formatDuration`), because m:ss read as a time of day;
  - glass at 0.84 / 0.94 alpha, because backdrop blur doesn't render in headless Chromium;
  - toasts step aside while the log is open, and are clamped to two lines;
  - the log chip is a label occluder;
  - the phone card is opaque when expanded.
- **Transitions:**
  - numbers tween for 320 ms with an ease-out (`useTweenedText`);
  - sparklines are smoothed: a 5-point moving average with a monotone cubic, on a zero-based scale for wait and counts;
  - menus, drawer, toasts, map key, profile menu and search dropdown enter and exit with fade plus slide or scale (160–260 ms);
  - Normal|AI and speed segments have sliding thumbs;
  - the scoreboard border, kicker and dot shift to cyan in AI mode;
  - `ModeCue` adds a cyan edge vignette that breathes in for 1.3 s on AI takeover and a neutral one on hand-back. I verified its opacity peaks at 1.0 around 300–450 ms (`scratchpad/ui/p3/cue_450.png`).
- **Reduced motion:**
  - the global CSS rule collapses animations;
  - `useDisclosure` and `useTweenedText` skip animation;
  - on the canvas, the stuck pulse, action flash, crowding pulse and marching signal dashes hold still.
- 227 unit tests pass.

### 2026-11-22: phase 5 (polish), in progress (resumed run)
- **Environment:** stopped leftover servers from the interrupted run (`next dev` on :3000, `next start` on :3200, mock on :8766). Temp servers for this phase: backend on :8010 (`SUBWAY_MODEL_PATH=runs/local/export_best/…onnx`, `ALLOWED_ORIGINS=http://localhost:3200`) and `next dev -p 3200` with `NEXT_PUBLIC_WS_URL=ws://localhost:8010/ws`. Tour: `BASE=http://localhost:3200 node scratchpad/evaltools/tour3.mjs <out>`; contact sheets with `scratchpad/evaltools/sheet.py`.
- **Fixed:**
  - the at-capacity card and the idle "Paused" veil rendered ~140 px wide (word-per-line), because `.banner` was a shrink-to-fit box with a `transform`, which also trapped the `position: fixed` veil. The banner now spans the gutters without a transform, and `.cardWrap` is fixed to the viewport;
  - citywide home view: the network was centred left of the scoreboard column, leaving the right third empty. `layout.homeView` centres it across the full width and slides back toward the inset fit only as far as needed for no station to sit under a HUD panel (first load, Fit button, `0` key). Tests: `test/unit/homeView.test.ts`;
  - top bar legibility: an eased, denser scrim (112 px), and the clock in the UI font with tabular figures instead of monospace;
  - glass panels at 0.9 alpha (was 0.84): lines no longer read through the scoreboard rows when backdrop blur is unavailable.
- **Motion re-verified on the temp backend:** `frame_speed.py 5x 8` (now honours `WS_URL` / `WS_ORIGIN`): old rule 7.01 % → client rule **0.007 %** snaps (new trips in reused slots 1.51 %, cross-faded). Recorded-frame vitest (real client code): 0.028 % Normal, 0.049 % AI + lex-union-sq.

### 2026-11-22: phase 5 (polish) and final verification, done
- **More polish after the screenshot review:**
  - narrow tablets (700–959 px): the control pill takes its own centred row under "Break the city" and the map tools, and the trigger never wraps (it wrapped to two lines at 820 px);
  - dispatch-log and disruption times use the UI font with tabular figures, like the clock;
  - the onboarding hint stands aside while the at-capacity card or the idle veil is up (it floated above the veil);
  - glass at 0.92 alpha.
- **Checks (all green):** `npm run typecheck`, `npm run lint`, `npm test` (19 files, 230 tests), `npm run build`, and `npm run screenshots` (5 passed; the mobile banner test is skipped by design).
- **Render:** 60 fps on the real backend at 1440×900 @2x: citywide p95 16.8 ms, AI + lex-union-sq zoomed p95 16.7 ms.
- **Motion:** at 5x, snaps are 0.007 % on the live server (old rule: 7.01 %) and 0.028 % / 0.049 % in the recorded-frame vitest (Normal / AI + lex-union-sq).
- **Final screenshots** (real backend, 1440×900 and 1280×760 @2x, 390×844 @3x, 820×1180 @2x) in `scratchpad/ui/`:
  - `after_desktop_first_visit.png`, `after_desktop_city.png` (Normal), `after_desktop_menu.png`, `after_desktop_ai_lex.png` (AI on during `lex-union-sq`), `after_desktop_log.png`;
  - `after_laptop_city.png`, `after_laptop_ai_lex.png`;
  - `after_mobile_city.png`, `after_mobile_ai_lex.png`, `after_mobile_scoreboard.png`;
  - `after_tablet_820.png`.
  - Script: `scratchpad/evaltools/final_shots.mjs`.
- All temp servers are stopped (:8010, :3200, and the e2e :8765 / :3100).
- **Open:**
  - backdrop blur does not render in headless Chromium, so glass was judged without it. It should be checked once in a real browser;
  - on narrow tablets the home view falls back to the inset fit (the network is width-limited there).

### 2026-11-22: critique fixes, done
- **Environment:** temp servers only (backend on :8010 with `runs/local/export_best/…onnx`, `next dev -p 3200`); both stopped. Review scripts: `scratchpad/evaltools/cf_review.mjs <out> [url] [dpr]` (preset, then AI at 5 min/s, sampled every second for 20 s; key; search; back to Normal; zoom) and `cf_extra.mjs` (1× DPR, citywide crop, phone).
- **Scoreboard (high):**
  - new `lib/aiScore.ts`: since-takeover totals from live and twin metrics only (trapezoid integrals of riders waiting, stuck trains and overcrowded platforms; the riders-served gap, since the twin forks at takeover). It applies a dead band (3 % or 1 unit) with hysteresis, so a total at the edge doesn't flip. It restarts on a new takeover or reset. After a mid-takeover resume it says "last Xm" instead of "since takeover";
  - the AI card leads with a verdict block: "AI ahead on N of 4", "Rider-min waited AI 1.63M vs 1.62M ±0%", and four tone chips (Wait / Stuck / Crowding / Served). The tooltip shows all four totals;
  - the rows are captioned "Right now · vs Normal twin". Row deltas are grey inside a dead band (< 3 % of the twin, or ±1 for counts); the tooltip shows the AI and twin absolute values;
  - the phone pill shows the verdict ("vs Normal · ahead on 1 of 4");
  - sparklines draw the twin as a solid grey line (60 %) and shade disruption windows (red band; a dotted line marks the real end). Windows are kept in `store.disruptionWindows` after the disruption ends;
  - measured over 20 s at 5 min/s: the verdict held "AI ahead on 1 of 4" in every sample (before: deltas flipped sign every few seconds).
- **Disruption lifecycle (high):**
  - `disruption_expired` now raises a notice toast ("14 St-Union Sq reopened · 8:30 AM"; it is also in the log as "Ended");
  - a preset that lasts < 15 s of wall time at 2 or 5 min/s gets a one-time hint ("This closure lasts ~6 s at 5 min/s. Try 1 min/s…") with a "Slow to 1 min/s" button. Speed never changes on its own, and the hint is dismissed when the crisis ends.
- **AI callouts (high):**
  - new `lib/calloutPacer.ts`: at most one callout on screen, swapped at most every 4.5 s of wall time. It prefers an action inside the viewport, drops ones older than 6 s and counts actions in the last 10 s. Every action still goes to the dispatch log;
  - the card is a single line (`AI [bullet] To local at 125 St · Trains are stopped in the tunnel ahead near 86 St`, ellipsised, with the full sentence in the tooltip), plus an "N in 10 s" rate tag and a small locate pin;
  - toasts dock centred above the control pill, so they no longer sit in the bottom-right corner.
- **Mode switch (medium):** switching to Normal clears the pacer and fades the callout out. Callouts that arrive while Normal is selected are logged, never toasted (verified: no toasts 600 ms after the switch).
- **Search (medium):** the field and dropdown are opaque (`--panel-solid`). The clock hides while search is open (`:has`). The dropdown is 340 px wide, right-aligned to the bar gutter, and covers the scoreboard header with nothing peeking out.
- **Map key (medium):**
  - open state is `store.keyOpen`, and toasts step aside while it is open;
  - two columns at ≥ 960 px, so it stays clear of the scoreboard;
  - new entries: "AI just acted on this train" (cyan ring), "Skipping its next stop" (»), "Switching track (express ⇄ local)" (⇄);
  - status glyphs sit just above the pill's own extent instead of floating beside the track;
  - the action-flash ring is smaller (+9 px close, +6 px wide).
- **Labels (medium):**
  - 5 px opaque dark halo;
  - placement prefers spots no route line or transfer link crosses (soft obstacle; fill-pass labels must be clear);
  - same-name stations within 140 px get one label (the best-ranked);
  - minor labels fade out while a pill passes over them.
- **Citywide (medium):**
  - route lines are muted on wide tiers: city tier at 40 % and 40 % desaturated, borough tier at 48 % and 25 % (new `tier.lineDesat`, `style.desaturate`);
  - pills get a dark casing at every zoom, a brighter fill (+0.16 towards white) and a light core stroke on wide tiers, so they read as objects rather than dashes;
  - glow radius and strength scale with zoom (×0.45 radius and ×0.55 alpha citywide).
- **Pills at 1× DPR (medium):** size anchors give at least 16.5 × 8 CSS px from 8 m/px (where route letters turn on), and letters are at least 8 px. Verified readable in a 1× DPR crop (`final_zoom_dpr1_pills_crop.png`). Hover tooltips on pills already existed.
- **Glass:** `--glass` alpha raised to 0.975, so lines no longer read through the scoreboard.
- **Tests:** `test/unit/aiFeedback.test.ts` adds 11 tests covering the score integrals, dead band, verdict, twin-delta dead band, pacer (gap, in-view preference, stale, clear), headline and reason, disruption windows and desaturate.
- **Checks (all green):** typecheck, lint, `npm test` (20 files, 241 tests), build, `npm run screenshots` (5 passed, 1 skipped by design). No console errors in any review run.
- **Final screenshots:** `scratchpad/ui/final_*.png`.
- **Open:**
  - ARCHITECTURE §6 UX flow 4 still describes the deltas only as "green/red". The neutral dead band and the cumulative verdict are additions to that UX contract, so the doc should be updated to match;
  - the critique text was cut off after the 1× DPR pill item, so any later items were not addressed;
  - the "14 St" label at 6 Av still sits between transfer links. It is legible on its halo, but it is placed over lines because no clear spot exists.
