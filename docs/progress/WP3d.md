# WP3d: Disruptions are blockages that last until the user clears them

Status: **done** (2026-11-22, not committed)

## The user's request (2026-11-22)

> "Instead of signal disruption or things being blocked for a few minutes, it's not significant enough to see anything. Just make the only option be blocked indefinitely until the user make it not blocked."

## Decisions (orchestrator)

1. **Every disruption is indefinite.** There is no duration picker, and the client always sends `duration_s: null`. The protocol and engine already support this: `AddDisruption.duration_s` is optional, and `None` means no end (`engine.py`, `end_s` is `None`).
2. **Blockage kinds only.** The UI offers `full_station_closure`, `local_block` and `express_block`, whichever is valid for the target. `signal_delay` is removed from the UI. Keep it in the sim, protocol and training: RL training and eval still use it, so do not touch them.
3. **Presets block indefinitely too, without touching the data artifacts.**
   - Do **not** edit `network.json` or `network_geo.json` (or the netbuild preset code). That would change `graph_version` and make the server reject the shipped ONNX model.
   - Instead, in the server (`backend/app/session.py`, `_on_preset`, which calls `preset_scenario(msg.id, graph)`), rewrite the preset scenario's disruptions with `duration_s=None` (`dataclasses.replace`).
   - In the same place, convert a `signal_delay` disruption to `local_block` on the **same target**. Both kinds can target segments and interlockings (`sim/constants.py`, `DISRUPTION_TARGETS`).
   - Leave `sim.scenarios.preset_scenario` itself unchanged, so WP2's calibration and leverage tests keep their 1800 s presets.
4. **Times Sq preset.** Its UI copy must say it is a blockage, for example "Tracks blocked at Times Sq–42 St". Presets come from `network_geo.json`, so add a small frontend copy override keyed by preset id (`times-sq-signals`), and use the same override in the menu and anywhere else the preset title shows. Keep the id.
5. **Unblocking is easy and obvious:**
   - clicking a blocked target shows a clear "Restore service" (clear) button;
   - the "Break the city" menu lists active blocks, each with a clear button, plus "Clear all";
   - map chips and labels show "CLOSED" / "BLOCKED" with no countdown, optionally "since 8:12 AM";
   - the "Reopened" / "SERVICE RESTORED" notices fire when the user clears a block;
   - remove any expiry / "crisis ended" logic or copy that assumed timed disruptions (e.g., the "Slow to 1 min/s" hint for short crises) if it no longer applies.
6. **Honesty and AI** are unchanged. The AI was trained with blocks lasting 10–40 min or until the episode ended, so indefinite blocks are in-distribution.

## Constraints

- Work in the main tree. `frontend/` and `backend/app/` (session preset handling) only, plus tests and docs.
- Do not change `backend/sim/*`, `backend/rl/*`, `backend/data/*` or `backend/models/*`.
- Update or add tests:
  - frontend unit tests for the valid kinds (no `signal_delay`), no duration UI, the preset copy override and chips without countdowns;
  - a backend `test_server` test that a preset's disruptions arrive with no end and that `signal_delay` presets become `local_block`.
- Protocol golden fixtures stay byte-identical. The protocol is unchanged.
- Docs:
  - ARCHITECTURE §6 UX flow 5 (the disruption popover) and any §5 text that implies timed UI disruptions: describe "blockages until cleared";
  - README.md: if it mentions signal delays or timed disruptions, adjust that wording only.
- Green before finishing:
  - `cd backend && .venv/bin/python -m pytest -q -m "not slow and not train"`;
  - `cd frontend && npm run typecheck && npm run lint && npm test && npm run build`;
  - `npm run screenshots`.
- The user's app is closed. For any visual check, start temporary servers and stop them afterwards:
  - backend: `cd backend && ALLOWED_ORIGINS=http://localhost:3200 .venv/bin/uvicorn app.main:app --port 8010` (it loads the shipped `models/` ONNX by default);
  - frontend: `cd frontend && NEXT_PUBLIC_WS_URL=ws://localhost:8010/ws npx next dev -p 3200`.
- Do not commit. The orchestrator commits and pushes.

## Progress log

- 2026-11-22: backend done. `app/session.py` gains `as_blockages(scenario)` (dataclasses.replace: `duration_s=None`, `signal_delay` to `local_block` on the same target), used only by `_on_preset`. Tests in `tests/test_server.py`: the preset test now expects `end_t: null`; new `test_as_blockages_*` (pure) and `test_signal_delay_preset_arrives_as_an_indefinite_local_block` (toy graph plus a signal_delay preset, live server). Next: frontend.
- 2026-11-22: frontend done (typecheck, lint, unit tests green). `lib/disruptions.ts`: no `DURATION_CHOICES`, `validKinds` without `signal_delay`, new `chipText` / `sinceText` / `blockedWord` / `restoredText`. Popover: no duration fieldset, "Restore service" buttons with "Blocked since 8:12 AM", form collapsed when the target is already blocked, "It stays blocked until you restore service." `DisruptionList`: "Blocked now", per-row "Restore", "Clear all" always. Chips: CLOSED/BLOCKED (LOCAL/EXPRESS BLOCKED from tier 1, "since 8:12 AM" from tier 2). `session.ts`: `addDisruption(kind, target)` sends `duration_s: null`; speed hint ("Slow to 1 min/s") and Toast `cta` removed; user clears (tracked in `restoring`) and expiries post one "Service restored" notice per batch (green tab, `kicker`/`tone` on Toast); reset-caused clears stay silent. `copy.ts`: `PRESET_TITLE_OVERRIDES` / `presetTitle` (times-sq-signals: "Tracks blocked at Times Sq–42 St"), used by PresetBar. Map key: signal swatch removed. Mock server mirrors `_on_preset`. Tests: new `test/unit/blockages.test.ts` (SSR of popover and list, session controller), `disruptions.test.ts`, `mockServer.test.ts`, `realGeo.test.ts`, `format.test.ts` (formatCountdown removed), e2e spec (no Duration group, Restore/Clear all, restored toast). Next: build, screenshots, docs, visual check.
- 2026-11-22: finished. Phones hid the "Service restored" notice behind the AI callout (only the last card shows); a restored notice now wins on phones while it lasts (`CalloutLayer.module.css`). Docs: ARCHITECTURE §2 preset list note, §5.3 "Blockages until cleared" note (wire format unchanged, fixtures byte-identical), §6 disruptions rendering and UX flow 5; README wording. Green: backend fast suite (241 passed, 1 skipped), frontend typecheck, lint, 261 unit tests, build, `npm run screenshots` (5 passed, 1 skipped). Visual check against the real backend (port 8010) and `next dev` (port 3200): Times Sq preset titled "Tracks blocked at Times Sq–42 St" arrives as `local_block` on interlocking 7 with `end_t: null`; chip "LOCAL BLOCKED since 5:00 PM"; menu "Blocked now" with Restore and Clear all; Restore posts "SERVICE RESTORED: Times Sq-42 St interlocking: local track clear again". Temporary servers stopped.
