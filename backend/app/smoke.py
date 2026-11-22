"""End-to-end WebSocket smoke test: ``python -m app.smoke --url ws://localhost:8000/ws --seconds 10``.

Against a running server it checks ``/health`` (exact body), fetches ``/network_geo.json``
(for the station/edge counts and preset ids), connects, says ``hello``, waits for
``welcome``, triggers a preset, switches to AI when the server says it is available,
decodes every binary frame strictly and validates every JSON message, then prints a
summary. Exit status: 0 = healthy, 1 = protocol/behavior error, 2 = cannot connect.

The server must accept the socket's Origin: pass ``--origin http://localhost:3000`` (an
allowed origin) or run the server with ``ALLOW_NO_ORIGIN=1``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from app import protocol as P

EXIT_OK, EXIT_PROTOCOL, EXIT_CONNECT = 0, 1, 2
SERVER_TYPES = {"welcome", "metrics", "events", "disruptions", "error", "pong", "idle_timeout", "at_capacity"}


@dataclass
class Report:
    frames: int = 0
    frame_bytes: int = 0
    json_messages: int = 0
    json_bytes: int = 0
    callouts: int = 0
    errors: list[str] = field(default_factory=list)  # protocol violations (exit 1)
    notes: list[str] = field(default_factory=list)
    server_errors: list[dict[str, Any]] = field(default_factory=list)
    last_metrics: dict[str, Any] | None = None
    last_frame: P.DecodedFrame | None = None
    welcome: dict[str, Any] | None = None
    ai_confirmed: bool = False
    preset_started: bool = False

    def fail(self, msg: str) -> None:
        self.errors.append(msg)


def http_base(ws_url: str) -> str:
    parts = urlsplit(ws_url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    path = parts.path[: -len("/ws")] if parts.path.endswith("/ws") else parts.path
    return urlunsplit((scheme, parts.netloc, path.rstrip("/"), "", ""))


def http_json(url: str, timeout: float = 10.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "subway-domino-smoke/1.0", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - operator-supplied URL
        return json.loads(resp.read().decode("utf-8"))


def check_frame(frame: P.DecodedFrame, prev: P.DecodedFrame | None, geo: dict[str, Any] | None, rep: Report) -> None:
    if prev is not None:
        if frame.seq <= prev.seq and not (prev.seq > 0xFFFF0000 and frame.seq < 0x10000):
            rep.fail(f"frame seq not increasing: {prev.seq} -> {frame.seq}")
        if frame.n_stations != prev.n_stations:
            rep.fail(f"station count changed: {prev.n_stations} -> {frame.n_stations}")
    ids = frame.train_id
    if ids.size > 1 and not bool((ids[1:] > ids[:-1]).all()):
        rep.fail("train_id not strictly ascending")
    if geo is not None:
        n_st, n_edges, n_routes = len(geo["stations"]), len(geo["edges"]), len(geo["routes"])
        if frame.n_stations != n_st:
            rep.fail(f"frame has {frame.n_stations} stations, geo has {n_st}")
        if frame.edge.size and int(frame.edge.max()) >= n_edges:
            rep.fail(f"edge index {int(frame.edge.max())} >= {n_edges}")
        if frame.route.size and int(frame.route.max()) >= n_routes:
            rep.fail(f"route index {int(frame.route.max())} >= {n_routes}")


def check_json(msg: Any, rep: Report) -> None:
    if not isinstance(msg, dict) or msg.get("type") not in SERVER_TYPES:
        rep.fail(f"unknown server message: {str(msg)[:120]}")
        return
    t = msg["type"]
    if t == "metrics":
        for key in ("t", "mode", "effective_speed", "live", "twin", "ai_since_t", "actions"):
            if key not in msg:
                rep.fail(f"metrics without {key!r}")
        live = msg.get("live") or {}
        for key in ("avg_wait_min", "overcrowded_platforms", "tunnel_stuck_trains", "riders_served",
                    "riders_waiting", "trains_active"):
            if not isinstance(live.get(key), (int, float)):
                rep.fail(f"metrics.live.{key} missing")
        if set((msg.get("actions") or {}).keys()) != set(P.DISPATCH_ACTIONS):
            rep.fail("metrics.actions must list the 7 dispatch actions")
        rep.last_metrics = msg
        if msg.get("mode") == "ai" and msg.get("twin") is not None:
            rep.ai_confirmed = True
    elif t == "events":
        for item in msg.get("items", []):
            if item.get("kind") == "callout":
                rep.callouts += 1
                if not isinstance(item.get("text"), str) or not item["text"]:
                    rep.fail("callout without text")
            elif item.get("kind") == "disruption_started":
                rep.preset_started = True
    elif t == "error":
        rep.server_errors.append(msg)


async def run(args: argparse.Namespace) -> int:
    try:
        from websockets.asyncio.client import connect  # uvicorn[standard] ships websockets
    except ImportError:  # pragma: no cover
        print("websockets is not installed", file=sys.stderr)
        return EXIT_CONNECT

    rep = Report()
    base = http_base(args.url)
    geo: dict[str, Any] | None = None
    try:
        health = http_json(f"{base}/health")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"FAIL cannot reach {base}/health: {exc}", file=sys.stderr)
        return EXIT_CONNECT
    print(f"health: {json.dumps(health)}")
    if set(health) != {"status", "model_loaded"} or health.get("status") != "ok":
        rep.fail(f"/health body is not exactly {{status, model_loaded}}: {health}")
    try:
        geo = http_json(f"{base}/network_geo.json")
        print(f"geo: graph_version={geo.get('graph_version')} stations={len(geo['stations'])} "
              f"edges={len(geo['edges'])} presets={[p['id'] for p in geo.get('presets', [])]}")
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        rep.notes.append(f"no /network_geo.json ({exc}); frame indices not cross-checked")

    preset = args.preset or (geo["presets"][0]["id"] if geo and geo.get("presets") else None)
    t_start = time.monotonic()
    try:
        ws_cm = connect(args.url, origin=args.origin, open_timeout=15, max_size=4 * 1024 * 1024)
        ws = await ws_cm.__aenter__()
    except Exception as exc:  # noqa: BLE001 - report any connect failure
        print(f"FAIL cannot open {args.url}: {exc!r}", file=sys.stderr)
        return EXIT_CONNECT
    try:
        await ws.send(json.dumps({"type": "hello", "protocol": P.PROTOCOL_VERSION, "resume_token": None,
                                  "restore": {"mode": "normal", "speed": args.speed, "profile": args.profile,
                                              "disruptions": []}}))
        first = await asyncio.wait_for(ws.recv(), 20)
        if isinstance(first, bytes):
            rep.fail("first message is a frame, expected welcome")
        else:
            msg = json.loads(first)
            if msg.get("type") == "at_capacity":
                print(f"server at capacity: {msg.get('message')}", file=sys.stderr)
                return EXIT_CONNECT
            if msg.get("type") != "welcome":
                rep.fail(f"first message is {msg.get('type')!r}, expected welcome")
            else:
                rep.welcome = msg
                print(f"welcome: session={msg['session_id'][:8]} graph_version={msg['graph_version']} "
                      f"model_loaded={msg['model_loaded']} ai_available={msg['ai_available']} settings={msg['settings']}")
                if geo is not None and geo.get("graph_version") != msg["graph_version"]:
                    rep.notes.append("graph_version differs between /network_geo.json and welcome")
        ai_available = bool(rep.welcome and rep.welcome.get("ai_available"))
        deadline = time.monotonic() + args.seconds
        preset_at = time.monotonic() + min(1.0, args.seconds / 4)
        ai_at = preset_at + 0.5
        sent_preset = sent_ai = False
        prev: P.DecodedFrame | None = None
        while (now := time.monotonic()) < deadline:
            if not sent_preset and preset and now >= preset_at:
                await ws.send(json.dumps({"type": "preset", "id": preset}))
                sent_preset = True
            if sent_preset and not sent_ai and ai_available and not args.no_ai and now >= ai_at:
                await ws.send(json.dumps({"type": "set_mode", "mode": "ai"}))
                sent_ai = True
            try:
                data = await asyncio.wait_for(ws.recv(), max(0.05, min(0.5, deadline - now)))
            except asyncio.TimeoutError:
                continue
            if isinstance(data, bytes):
                rep.frames += 1
                rep.frame_bytes += len(data)
                try:
                    frame = P.decode_frame(data)
                except P.FrameError as exc:
                    rep.fail(f"undecodable frame: {exc}")
                    continue
                check_frame(frame, prev, geo, rep)
                prev = rep.last_frame = frame
            else:
                rep.json_messages += 1
                rep.json_bytes += len(data)
                try:
                    check_json(json.loads(data), rep)
                except json.JSONDecodeError:
                    rep.fail("server sent invalid JSON")
        elapsed = time.monotonic() - t_start
    except Exception as exc:  # noqa: BLE001 - a dropped socket is a failure of the run
        rep.fail(f"connection error: {exc!r}")
        elapsed = time.monotonic() - t_start
    finally:
        await ws_cm.__aexit__(None, None, None)

    if rep.frames == 0:
        rep.fail("no binary frames received")
    if preset and not rep.preset_started:
        rep.fail(f"preset {preset!r} produced no disruption_started event")
    if rep.welcome and rep.welcome.get("ai_available") and not args.no_ai and not rep.ai_confirmed:
        rep.fail("AI mode was available but no metrics with mode=ai and a twin arrived")
    if args.require_ai and not (rep.welcome and rep.welcome.get("ai_available")):
        rep.fail("--require-ai: the server reports ai_available=false")
    unexpected = [e for e in rep.server_errors if not (e.get("code") == "ai_unavailable" and args.no_ai)]
    for e in unexpected:
        rep.fail(f"server error {e.get('code')}: {e.get('message')}")

    secs = max(elapsed, 1e-9)
    print(f"frames: {rep.frames} in {elapsed:.1f} s = {rep.frames / secs:.2f} Hz; "
          f"{(rep.frame_bytes + rep.json_bytes) / secs / 1024:.1f} KiB/s payload "
          f"(frames {rep.frame_bytes / max(rep.frames, 1):.0f} B avg); json messages: {rep.json_messages}; "
          f"callouts: {rep.callouts}")
    if rep.last_frame is not None:
        f = rep.last_frame
        print(f"last frame: seq={f.seq} clock_s={f.clock_s:.0f} trains={f.n_trains} stations={f.n_stations} "
              f"mode={f.mode} speed={f.speed} twin={f.twin_active}")
    if rep.last_metrics is not None:
        m = rep.last_metrics
        print(f"last metrics: mode={m['mode']} effective_speed={m['effective_speed']} live={m['live']}")
        if m.get("twin"):
            print(f"              twin={m['twin']} actions={m['actions']}")
    for note in rep.notes:
        print(f"note: {note}")
    for err in rep.errors:
        print(f"FAIL {err}", file=sys.stderr)
    print("OK" if not rep.errors else f"FAILED ({len(rep.errors)} problem(s))")
    return EXIT_OK if not rep.errors else EXIT_PROTOCOL


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.smoke", description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="ws://localhost:8000/ws", help="WebSocket URL (ws:// or wss://)")
    ap.add_argument("--seconds", type=float, default=10.0, help="how long to stream after welcome")
    ap.add_argument("--origin", default=None, help="Origin header to send (default: none)")
    ap.add_argument("--preset", default=None, help="preset id (default: the first in /network_geo.json)")
    ap.add_argument("--speed", choices=P.SPEEDS, default="2x")
    ap.add_argument("--profile", choices=P.PROFILES, default="am_peak")
    ap.add_argument("--no-ai", action="store_true", help="do not switch to AI mode")
    ap.add_argument("--require-ai", action="store_true", help="fail unless the server has a model loaded")
    args = ap.parse_args(argv)
    if args.seconds <= 0:
        ap.error("--seconds must be > 0")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
