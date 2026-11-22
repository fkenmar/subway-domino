# WP3c: MTA-themed reskin (subway design language)

Status: **not started**. Work in the main tree, `frontend/` only.

> **OVERRIDE (orchestrator, user request 2026-11-22):** the user CLOSED the live app so their RL training runs well. Ports :3000 and :8000 are stopped. Do **not** restart anything on them, and ignore any instruction to browse :3000. A 13-worker training run is using the CPU until about 30 min from now.
> - Keep CPU use light: batch your visual checks, and avoid repeated `npm run build` / `npm run screenshots` until the end.
> - For screenshots, start temporary servers only while you need them, then stop them:
>   - backend: `cd backend && SUBWAY_MODEL_PATH=.worktrees/rl2/backend/runs/pertrain/export_best/nyc_subway_rcc_ppo.onnx ALLOWED_ORIGINS=http://localhost:3200 .venv/bin/uvicorn app.main:app --port 8010`
>   - frontend: `cd frontend && NEXT_PUBLIC_WS_URL=ws://localhost:8010/ws npx next dev -p 3200`
>   - then browse http://localhost:3200/subway (`ui_probe.mjs <outdir> http://localhost:3200/subway`).

## What the user asked for (2026-11-22)

- **Feedback:** "Instead of a basic dashboard design it would have been creative to make it MTA themed."
- **The user picked all four elements:**
  1. **Station signage UI.** The top bar, menus and controls look like NYC subway platform signs: black bands, white Helvetica, the thin white rule along the top edge, and route bullets in every label and button.
  2. **Official map look.** A light basemap like the printed subway map:
     - cream/beige land, light-blue water (parks in green are optional, see below);
     - express stops as white circles with black rims, local stops as black dots;
     - transfers as white rounded connectors with black outlines;
     - colored line ribbons.
  3. **Countdown-clock scoreboard.** The AI-vs-Normal scoreboard becomes a station arrival board: a black panel, rows led by bullets, amber LED / dot-matrix digits.
  4. **Service-change notices.** AI callout toasts and dispatch-log entries look like service-change posters:
     - a bold "SERVICE CHANGE" header bar;
     - the route bullet, large;
     - a plain-language instruction ("Trains run local · 125 St → Grand Central");
     - small "AI Dispatcher" attribution.
- **Keep WP3b's work:**
  - the full-screen map layout and component structure (`docs/progress/WP3b.md`);
  - the fluid motion (path-following interpolation, transitions, reduced motion);
  - the honest AI scoreboard semantics (since-takeover verdict, grey dead band, twin line on sparklines);
  - the toast pacing;
  - the clock "8:22:05 AM · 1 min/s";
  - all tests.

## Hard constraints

- **No MTA logo, and no claim or look of being official.** Evoke the design language only: Helvetica, bullets, signage, map conventions.
  - Add a small, quiet "Not affiliated with the MTA" line in the info modal and in an unobtrusive corner credit.
  - Route colors and letters are factual service designations; keep them.
- **Honesty:** every scoreboard number comes from the live and twin metrics. Never fake or embellish.
- **Contracts:** keep the protocol, the store and the decoder golden tests. `asset()` for all `public/` URLs; `basePath` stays `/subway`.
- **Fonts:**
  - UI: `"Helvetica Neue", Helvetica, Arial, "Nimbus Sans", sans-serif`. Use bold and medium weights for signage. No commercial font files.
  - LED digits: a free OFL dot-matrix font via `next/font/google`, for example **Doto** or DotGothic16. Use it only for the countdown-board numbers, in amber (about `#FFB100`) on black, with a subtle glow. Keep tabular alignment, and fall back to monospace.
- **Map palette** (light, inspired by the printed map; tune for contrast):
  - land about `#F3EEE4`, water about `#BFDDEB`, borough labels in grey uppercase Helvetica;
  - lines are route-colored ribbons, slightly thicker, with thin light gaps between parallel lines;
  - labels are black Helvetica with a light halo;
  - trains are route-colored pills with a white letter and a thin dark outline, plus a soft shadow so they pop on the light map;
  - status cues re-tuned for the light background: stuck = red pulse ring, held = amber ring, AI-touched = small marker;
  - disruptions in strong signage colors: red "no service" segments, a closed-station badge in the poster style.
- **Parks are optional.** Add them only if you can build a separate simplified asset of the largest parks (Central, Prospect, Flushing Meadows, Van Cortlandt, Pelham Bay, Forest, and similar), ≤ 25 KB, from NYC Open Data with a cached download in a tiny script under `backend/scripts/`. Do not touch `network_geo.json` or its contract. Skip parks if that is not cheap and clean.
- **Responsive:** desktop 1440 and laptop 1280 must be excellent; mobile 390 must be usable.
- **Accessibility:** WCAG AA contrast for text on both the light map and the black signage. Keep focus states and aria labels.

## The user's live app is running. Do not stop or restart it.

- `next dev` runs on :3000 against the backend on :8000, which serves the champion model. The user may be browsing it.
- Your edits hot-reload into it, which is fine. Just keep the app working between edits.
- For visual checks you may browse it read-only: `scratchpad/evaltools/ui_probe.mjs <outdir> http://localhost:3000/subway`.
- E2E: `npm run screenshots` (mock server on :8765 plus `next start` on :3100) is self-contained.
- Stop any extra server you start.

## Process

1. Work in phases, logging each one below as you go:
   1. design tokens and fonts;
   2. map palette and symbols;
   3. signage UI (top bar, controls, menus, drawer);
   4. countdown-clock scoreboard;
   5. service-change notices (toasts and log);
   6. polish, mobile, info modal, not-affiliated note.
2. After each visual phase, screenshot and look critically: does it read as unmistakably NYC subway, tasteful, legible and premium?
3. Green before finishing: `npm run typecheck && npm run lint && npm test && npm run build`, and `npm run screenshots`.
4. Save the final screenshots to `<scratchpad>/ui/mta_*.png`. Cover desktop citywide, desktop zoomed with AI on during a preset, the scoreboard close-up, a service-change toast, the menu, laptop and mobile.

## Progress log

### 2026-11-22: started
- Plan: tokens and Doto font first, then the map palette (`style.ts`, `renderBase.ts`, `renderDynamic.ts`), then signage CSS per component, then rewrites of the scoreboard (`MetricsHUD`) and of the callouts/log (`CalloutLayer`, `ActivityFeed`).
- Visual checks run against the user's live :3000 app (read-only) with `BASE=http://localhost:3000` and a copy of `scratchpad/evaltools/final_shots.mjs` (`mta_shots.mjs`).
- The user's :3000 / :8000 servers were already gone at the first visual check (not stopped by this run). Not restarted. Temp servers for visual checks: backend on :8010 (`SUBWAY_MODEL_PATH=runs/local/export_best/…onnx`, `ALLOWED_ORIGINS=http://localhost:3200`) and `NEXT_PUBLIC_WS_URL=ws://localhost:8010/ws next dev -p 3200`. Quick tour: `BASE=http://localhost:3200 node scratchpad/evaltools/mta_quick.mjs <out> <tag>`.

### 2026-11-22: phases 1–2 (tokens, fonts, map palette), code done
- `globals.css`: black signage tokens, `--sign-rule` (thin white rule 3 px below the top edge, as an inset shadow on `.glass`), `--led*`, `--poster*`, `--ai-fill`, small radii; `.led` class. `layout.tsx`: Doto (OFL) via `next/font/google` as `--font-led`.
- Top bar is a full-width black sign band (`App.module.css`); the dark scrims are gone. All `999px` pill radii are now `--radius-sm`.
- Map: `style.ts` palette (cream land, blue water, light casing gaps, black dots / white circles, dark-outlined pills with a drop shadow); thicker ribbons with wider gaps. `network.ts`: `mapColor` is the official colour; new `SIGNAGE_BACKGROUND`; `network.test.ts` updated. `renderBase.ts`: `expressStops()` (white circle where every route through a station with an express track stops), transfers from the borough tier. `renderDynamic.ts`: shadow instead of glow, deeper fill = fuller. Map key swatches redrawn on a land tile.

### 2026-11-22: phases 4–5 (countdown board, service-change notices), code done
- `MetricsHUD`: black board, white-rimmed pictogram bullets per row (`MetricBullet`; pictograms, never letters, so they can't pass for real route bullets), Doto amber values and LED-coloured deltas, faint dot-matrix field, verdict block with a tone bar. Same numbers and semantics as WP3b. Scoreboard column 316 px.
- `CalloutLayer`: poster card (black "SERVICE CHANGE" bar with "AI Dispatcher" byline, 34 px route bullet, plain-language line from new `serviceChangeText()` in `calloutPacer.ts`, reason below). Notices are white slips with a black "NOTICE" / red "NOT APPLIED" tab.
- `ActivityFeed`: each entry is a white slip with a "SERVICE CHANGE" / red "SERVICE ALERT" / green "SERVICE RESTORED" header line, big bullet, instruction, the server's full text, and "AI Dispatcher · <action>".

### 2026-11-22: phases 3 and 6 (signage UI, polish, not-affiliated note), done
- Signage: dock, map tools and "Break the city" are square-cornered black signs with the top rule; the Normal | AI toggle uses a white tile (Normal) or a solid `--ai-fill` tile (AI); the preset menu has a sign header and ruled rows; the station-name-sized title. Popover, banners, info modal and map skeleton lost the navy tints (black signage, white primary buttons, flat red "Close this station").
- Onboarding hint is a white notice slip with a black top band.
- Not affiliated: a quiet corner credit above the map tools ("Fan-made simulator · Not affiliated with the MTA", hidden under 960 px) and a line under the info-modal title (the Data tab footnote stays).
- Board columns tightened so 6-digit "Riders served" fits; sparklines sized to the column.
- New `test/unit/mtaTheme.test.ts` (service-change copy, express-stop rule, light-palette contrast). typecheck, lint, 245 tests, build and `npm run screenshots` (5 passed, 1 skipped by design) were green before this last polish; re-run below.

### 2026-11-22: final verification, done
- The AI callout headline wraps to two lines (line clamp) instead of ellipsising after one.
- **Checks (all green):** `npm run typecheck`, `npm run lint`, `npm test` (21 files, 245 tests), `npm run build`, `npm run screenshots` (5 passed, 1 skipped by design; `test/screenshots/*` regenerated in the new look).
- **Final screenshots** (temp backend + `next dev -p 3200`; 1440×900 and 1280×760 @2x, 390×844 @3x) in `scratchpad/ui/`: `mta_desktop_first_visit`, `mta_desktop_city`, `mta_desktop_menu`, `mta_desktop_ai_lex`, `mta_desktop_ai_zoomed`, `mta_scoreboard`, `mta_service_change_toast`, `mta_desktop_log`, `mta_desktop_info`, `mta_laptop_city`, `mta_laptop_ai_lex`, `mta_mobile_city`, `mta_mobile_menu`, `mta_mobile_ai_lex`, `mta_mobile_scoreboard`. Script: `scratchpad/evaltools/mta_shots.mjs <out> [prefix]`. No console errors.
- Temp servers (:8010, :3200) and the e2e servers are stopped. The user's :3000 / :8000 were not running during this work and were not restarted.
- **Open:**
  - parks were skipped (optional; needs a new NYC Open Data download script);
  - the express-stop rule is a heuristic from segment routes (e.g. 79 St on the D reads as express), because the geo file has no per-station stopping pattern;
  - `models/nyc_subway_rcc_ppo.onnx` was absent from `backend/models/` during this run, so checks used `runs/local/export_best/…onnx`;
  - the `--sign-rule` inset shadow curves slightly at rounded corners; this is barely visible at 6–8 px radii.

### 2026-11-22: critique fixes (`WP3c-critique.md`, 21 issues), done
- **Resumed work:** an earlier interrupted run had already rewritten `calloutPacer.ts` (`posterCopy`, `parseGapTrain`, `parseSkip`, per-action headers) and `CalloutLayer.*` without logging. Its unit tests were stale (`serviceChangeText` skip copy); fixed here.
- **#1 (honesty, high):** gap-train headlines name the entry station parsed from the server text ("Extra train enters service at Forest Hills-71 Av"); the gap goes on the reason line ("Fills a gap near 42 St-Port Authority · from Jamaica Yard · in ~7 min"); if the text can't be read, no place is named. Toasts and the dispatch log both use `posterCopy`. Tests in `test/unit/mtaTheme.test.ts` assert headline == entry station (four texts, incl. "+N more" suffix and a 7-express phrase) and that the gap station never appears in it.
- **#2 LED:** Doto → DotGothic16 (OFL, `next/font/google`): a real full stop, a plain 0, "±" and "−" in-font. Units ("min") in white Helvetica; dead-band deltas #c8c8c8 (≥ 10:1); mobile pill splits value and unit.
- **#3 / #10 / #14 / #15 / #16:** "No stop at Wall St" headlines; headers per action (HOLDING / SKIP STOP on a light grey tab, SERVICE CHANGE / EXTRA SERVICE on black); "+N more" chip with an explicit title; notices and the poster share one width; log slips framed (1 px light frame) on a #1c1c1c drawer; the log hides the reason line when the server text repeats it verbatim.
- **#4 express/local:** `network_geo.json` can't tell stopping patterns (station `routes` include late-night/peak-direction service, e.g. Steinway lists E F M R; 33 St lists 7 and 7X), so the express heuristic is gone. Every station is a black dot; white circles mark transfer complexes only (`transferStations`, multi-station complexes). The map key says "Station" / "Transfer between stations". Not done: real express circles, which would need per-route stop patterns (in `network.json` `patterns[].stops`) exported as a new frontend asset.
- **#5 citywide:** ribbons at full official colour on every tier (`lineAlpha 1`, `lineDesat 0`), a bit thicker on the city/borough tiers; at the city tier trains are small uncased white beads inset on the ribbon (lane width 0.9 so they stay inside), and pills come back from the borough tier on.
- **#6 status cues:** held = white ring with a black outline; stuck = white gap + red ring + the red pulse; "AI just acted" = black-in-white double ring (gap trains keep green inside). Checked on QBL blue/orange/yellow and Lex green/red.
- **#7:** sparkline disruption = thin red baseline with start/end ticks; footer key "— disruption".
- **#8:** credit is a cream chip; the scoreboard column (and log drawer) ends above it (≥ 960 px); dock fully opaque; on phones the open log takes the column to the bottom and the floating controls/toasts hide until it closes.
- **#9:** `.glass` no longer carries the rule; `.glass.sign` (2 px corners) only on the scoreboard and the Break-the-city menu (plus the top band and poster heads as before).
- **#11** "Riders served" ≥ 1,000,000 shows "1.39M". **#12** labels never fade under trains (drawn on top with the halo). **#13** info modal: black sign header with the rule, white tab underline and numerals, "No tricks" as a white poster slip. **#17** expanded phone board: the pill becomes the header (card header hidden). **#18** not-affiliated line at the foot of the board under 960 px. **#19** "color", connector swatch, closed swatch as drawn (black disc ×). **#20** connectors are one fat white pill (circle diameter) with a single black outline. **#21** `presetSubtitle()` shows "Broadway–7 Av Line", "Lexington Av Line", … (data file untouched; tested).
- **Skipped on purpose:** the LED punctuation keeps DotGothic16's monospace cell, so "2. 9" / "22, 235" show a cell-wide gap (true to a matrix board, all glyphs unambiguous). Status-cue "pause tick" not added (the white/black ring already reads on every line).
- **Checks (all green):** `npm run typecheck`, `npm run lint`, `npm test` (21 files, 251 tests), `npm run build`, `npm run screenshots` (5 passed, 1 skipped by design; `test/screenshots/*` regenerated).
- **Screenshots:** temp backend :8010 (`runs/pertrain_global/export_best`) + `next dev -p 3200`, both stopped; `scratchpad/ui/mta_final_*.png` (desktop/laptop/mobile tour via `evaltools/mta_shots.mjs`, plus `mta_final_check_{queens,courtsq,mapkey,m_log,m_board}` via `evaltools/fix3c.mjs`). 0 console errors. :3000/:8000 untouched.
