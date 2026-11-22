# WP4 — Server + deployment (progress)

## Done
- [x] Read CLAUDE.md, ARCHITECTURE (§5, §7, §8), PRD; studied sim API on the toy graph and WP3's
      protocol.ts / wsClient.ts / goldenFixtures.test.ts / mock-server.mjs (the consumers).
- [x] app/: config, protocol (strict models, builders, frame codec), callouts, ratelimit, transport
      (single writer, skip-on-backpressure), session, manager (governor, registry, resume), main, smoke
- [x] manual e2e on toy (uvicorn in-process) and on the real network (smoke OK: 475 st, ~468 trains,
      3.9 Hz, 5.2 KB/frame)
- [x] golden fixtures (tests/fixtures/protocol/frame_v1.bin|json, messages_v1.json) + tests:
      test_server.py, test_protocol_fixtures.py, test_import_firewall_app.py — 80 pass (2026-11-20, run 2)
- [x] requirements-prod/dev, Dockerfile, .dockerignore, railway.json written

- [x] (2026-11-21 run 3) review vs current sim API: ActionEvent detail keys, blocked_ahead shape,
      frame/metrics/disruptions API all match; WP4 suite green
- [x] docker build: layer sum 356 MB uncompressed (< 400 MB target; Docker Desktop's containerd
      store lists 461 MB = unpacked + compressed blobs), 105 MB gzip; run -e PORT=8080: /health
      exact body, container HEALTHCHECK healthy, app.smoke OK (475 st, 455 trains, 3.97 Hz, 5.1 KB/frame)
- [x] load test (scratch loadtest.py, container --cpus=1, real network, 40 sessions):
      Normal 1x: 0.49 core (12 m-core/session), 4.00 Hz, 20 KiB/s/session, 0.996x speed, scale 1.0
      Normal 5x: governor holds 0.69-0.71 core, effective_speed 0.32 (~95 sim-s/s), 4.00 Hz, 0 skips
      AI (dummy model) 1x: needs ~0.93 core (twin doubles ticks; inference 0.013 ms) -> scale ~0.75
- [x] governor: run_blocking (reset/construct) CPU now metered and excluded from the load
      (a 40-join burst floored scale to 0.05-0.07 for ~6 s); unit test added; /stats reset_cpu_s

- [x] rebuilt image + rerun: 40-join burst keeps scale 1.0 (welcomes take 11.5 s total instead of
      5.8 s: resets get the headroom, existing sessions keep full speed); Normal 1x 0.47 core,
      5x governor 0.70 core / effective 0.34; container RSS ~240 MB (40 sessions, Normal)
- [x] AI smoke with a dummy model on the real network: twin on, 8 callouts, all 7 actions applied
- [x] repo fast suite: 220 passed, 2 skipped (ONNX parity: no shipped model yet)

## Next
- [ ] once WP5 ships models/nyc_subway_rcc_ppo.onnx: rebuild image, `app.smoke --require-ai`,
      repeat the AI load test with the real policy

## Decisions
- Real uvicorn server in a background thread for integration tests (true close codes, Origin header,
  permessage-deflate), websockets 17 asyncio client.
- uvicorn 0.54 defaults to the websockets sans-I/O impl: `send()` awaits transport writability
  (64 KiB high-water mark), so "a send still pending" = previous frame not yet written.

## Known issues
- 40 sessions all in AI mode exceed CPU_BUDGET at 1x (~0.93 core on an M5 Pro core; Railway vCPU
  likely slower): cost is sim.step x2 (live + twin, ~1.3-1.6 ms each under 40-sim cache pressure),
  WP2 scope (tick 0.83 ms bench vs 0.6 target). Governor degrades all sessions equally (~0.75x).
- Full fast suite: only failure is WP2's test_import_firewall (sim.bench not written yet) — not WP4.

## Contract change requests
- (none yet)
