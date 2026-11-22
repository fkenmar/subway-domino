# WP3 — Frontend progress

## Done
- [x] Scaffold (package.json pinned, tsconfig, next.config.ts, eslint flat config, vitest) — deps installed
- [x] lib: config/asset, protocol decoder, network loader, geometry, positioning, interp, camera, layout, search, format, copy, disruptions, store, session, wsClient
- [x] map: MapCanvas, engine, renderBase (cached per zoom tier), renderDynamic, hitTest, style, mapView
- [x] components: Header, ModeToggle, SpeedControl, ProfileSelect, ResetButton, PresetBar, MetricsHUD(+Sparkline), ActivityFeed, DisruptionList, SearchBox, RouteBullet
- [x] test/fixtures/mini_geo.json (hand-written §2.7 fixture, 47 stations)
- [x] components: App (page shell + measured map insets), CalloutLayer, DisruptionPopover, InfoModal, ConnectionBanner, Onboarding
- [x] scripts/mock-server.mjs (protocol v1 mock; also serves GET /network_geo.json like the real backend)
- [x] Visual pass 1 (desktop/mobile, citywide/Midtown, preset+AI) against the mock + a throwaway full-network geo approximated from raw GTFS (scratchpad only)
- [x] vitest suites (170 tests): protocol decoder + JSON parsing, golden fixtures (skipped until WP4 writes them), interp, positioning rule, asset()/basePath/WS URL, network build/validate/load, wsClient lifecycle (fake clock/sockets/visibility), camera, disruptions targeting, search, store, format, layout/popover
- [x] typecheck / lint / test green (vitest config renamed to vitest.config.mts — removes the ESM/CJS warning)
- [x] mock-server integration suite (in-process, ephemeral port): welcome/frames decode + every train locatable, AI twin metrics, preset events, invalid_target, ping/resume, at_capacity 1013, idle 1000, --no-ai
- [x] `npm run build` green; playwright.config.ts + test/e2e/screenshots.spec.ts (desktop 1440, laptop 1280, mobile 390; connection banners via routeWebSocket) — all passing
- [x] Visual pass 2 fixes: borough labels moved to line-free spots (Manhattan up the Hudson), slim transfer links, dense-tier labels for differently named complex stations, labels kept on screen, map controls as label occluders (+ MutationObserver for late panels), compact older toast, ±0.0 deltas, touch-aware onboarding copy, search centres on the highlighted station, touch taps prefer the station marker over the adjacent interlocking, idle card accessible name

- [x] Final round (real WP1 geo): mock/dev/playwright already prefer public/data/network_geo.json; golden decoder test passes against WP4's frame_v1.bin/json + messages_v1.json
- [x] `npm run start` serves /subway (200), /subway/data/network_geo.json (200), / → 307 /subway
- [x] Real-data visual pass (tour script in scratchpad): label fill pass (tier.labelFillGap/labelFillMaxRank/labelFillNamedOnly) names sparse outer-borough lines; yard names go through label collision (no more labels under HUD panels); off-screen stations never get labels straddling the viewport edge; station labels yield to disruption chips (chips laid out before labels each frame)
- [x] test/unit/realGeo.test.ts: scope counts, every edge positionable, yard leads, presets valid + focus near target, station + segment hit precision on real geometry (206 tests)

- [x] Callout toasts opaque (backdrop-filter saturate let map labels ghost through on mobile); enter animation reaches full opacity at 30%
- [x] e2e: geo must load from /subway/data/network_geo.json only (no backend fallback), console errors no longer filtered
- [x] Final full check green: typecheck, lint, 206 vitest, build, Playwright 5 passed / 1 skipped (mobile banners covered at desktop); test/screenshots refreshed (26 PNGs, real geo)

## Next
- WP3 complete. Optional polish: per-label halo knock-out of train capsules; label placement could also avoid line strokes.

## Known issues
- `next dev` (16.4) writes frontend/AGENTS.md (Next's agent-rules block); next.config.ts is contract-exact so `agentRules: false` can't be set. Harmless; Next recommends committing it.
- Verification runs use ports 8765 (mock) / 3100 (next) to stay clear of WP4 on :8000.

## Contract change requests
- §2.7 edges: for `yard_lead` edges please emit `station` = the lead station (the non-yard endpoint) and `seg_dir` = +1 for yard→station (pull-out), −1 for station→yard (put-in). The positioning rule needs the lead station and the direction; today the schema only promises `station` for crossover/turnback. Frontend fallback (implemented): nearest station served by the yard's division; seg_dir 0 is treated as pull-out.
- §6 interpolation: "different edge → snap" makes every station departure jump. The client also interpolates (constant speed) when the new edge directly continues the old one (old edge's to-station == new edge's from-station, derived from the geo file); unrelated edges still snap. Suggest blessing this in §6.
