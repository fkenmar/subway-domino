"""Server tests (ARCHITECTURE §5, §8 ``test_server.py``; PRD Epic 6).

Integration tests run the real stack — ``create_app`` → uvicorn (websockets sans-I/O
protocol, permessage-deflate) on an ephemeral port in a background thread — against the
toy graph, with the websockets asyncio client. So close codes, the ``Origin`` header,
idle/grace timers and frame decoding are tested on the wire, not through a test transport.

AI-mode tests generate a tiny random-weight ONNX actor with the §4.9 I/O contract
(``obs`` float32 [batch, 216] → ``logits`` float32 [batch, 8]) and let the lifespan load it
through ``sim.policy.OnnxPolicy``; they are skipped when the ``onnx`` package is missing.
Unit tests at the end cover the accumulator, the CPU governor, callouts, settings, the
parser and the writer's skip-on-backpressure rule.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import uvicorn
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from app import protocol as P
from app.callouts import CalloutGate, CalloutWriter, format_duration, format_riders
from app.config import ConfigError, Settings, normalize_origin
from app.main import create_app
from app.manager import CpuGovernor, SessionManager
from app.session import Session, SessionContext, as_blockages, ticks_due
from app.transport import Outbox
from sim import NYCSubwaySim, Scenario, constants as C, preset_scenario
from sim.engine import ActionEvent
from sim.graph import Graph, canonical_graph_version, graph_from_dict
from sim.scenarios import ScheduledDisruption
from sim.testing import TOY_STATION, toy_network_dict, toy_segment

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

GEO_STUB = {"schema_version": 1, "graph_version": "stub", "stations": [], "presets": []}


# ======================================================================================
# Live server harness
# ======================================================================================
class LiveServer:
    """``uvicorn.Server`` for an app on 127.0.0.1:<ephemeral> in a daemon thread."""

    def __init__(self, app: Any) -> None:
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning",
                                                    lifespan="on"))
        self.thread = threading.Thread(target=self.server.run, name="live-uvicorn", daemon=True)
        self.port = 0

    def __enter__(self) -> "LiveServer":
        self.thread.start()
        deadline = time.monotonic() + 20.0
        while not self.server.started:
            if not self.thread.is_alive():
                raise RuntimeError("uvicorn failed to start")
            if time.monotonic() > deadline:
                raise TimeoutError("uvicorn did not start")
            time.sleep(0.01)
        self.port = self.server.servers[0].sockets[0].getsockname()[1]
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(15.0)

    @property
    def http(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def ws(self) -> str:
        return f"ws://127.0.0.1:{self.port}/ws"

    def get(self, path: str, **kw: Any) -> httpx.Response:
        return httpx.get(self.http + path, timeout=10.0, **kw)

    def stats(self) -> dict[str, Any]:
        return self.get("/stats").json()


def make_settings(tmp: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {"allow_no_origin": True, "data_dir": tmp, "model_path": tmp / "no-model.onnx",
                            "log_level": "WARNING"}
    base.update(overrides)
    return Settings(**base)


@contextmanager
def live(graph: Graph, tmp: Path, **overrides: Any) -> Iterator[LiveServer]:
    app = create_app(graph=graph, settings=make_settings(tmp, **overrides), prewarm=False)
    with LiveServer(app) as srv:
        yield srv


def wait_for(cond: Any, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return bool(cond())


# ======================================================================================
# Client helper
# ======================================================================================
class Client:
    """Thin protocol client: JSON helpers, decoded frames, close-code capture."""

    def __init__(self, ws: ClientConnection) -> None:
        self.ws = ws
        self.frames: list[P.DecodedFrame] = []
        self.messages: list[dict[str, Any]] = []

    async def send(self, **msg: Any) -> None:
        await self.ws.send(json.dumps(msg))

    async def hello(self, *, resume_token: str | None = None, restore: dict[str, Any] | None = None) -> dict[str, Any]:
        await self.send(type="hello", protocol=1, resume_token=resume_token, restore=restore)
        return await self.expect("welcome")

    async def next(self, timeout: float = 5.0) -> P.DecodedFrame | dict[str, Any]:
        data = await asyncio.wait_for(self.ws.recv(), timeout)
        if isinstance(data, bytes):
            frame = P.decode_frame(data)
            self.frames.append(frame)
            return frame
        msg = json.loads(data)
        self.messages.append(msg)
        return msg

    async def expect(self, type_: str, timeout: float = 5.0, where: Any = None) -> dict[str, Any]:
        """Next JSON message of ``type_`` (matching ``where``), skipping frames and others."""
        async with asyncio.timeout(timeout):
            while True:
                item = await self.next(timeout)
                if isinstance(item, dict) and item.get("type") == type_ and (where is None or where(item)):
                    return item

    async def expect_frame(self, timeout: float = 5.0, where: Any = None) -> P.DecodedFrame:
        async with asyncio.timeout(timeout):
            while True:
                item = await self.next(timeout)
                if isinstance(item, P.DecodedFrame) and (where is None or where(item)):
                    return item

    async def collect(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while (left := deadline - time.monotonic()) > 0:
            try:
                await self.next(left)
            except (asyncio.TimeoutError, TimeoutError):
                return

    async def expect_close(self, timeout: float = 6.0) -> int | None:
        """Read until the server closes; returns the close code it sent."""
        try:
            async with asyncio.timeout(timeout):
                while True:
                    await self.next(timeout)
        except ConnectionClosed as exc:
            return exc.rcvd.code if exc.rcvd is not None else None
        return None  # pragma: no cover

    def json_of(self, type_: str) -> list[dict[str, Any]]:
        return [m for m in self.messages if m.get("type") == type_]


@asynccontextmanager
async def client(srv: LiveServer, *, origin: str | None = None) -> AsyncIterator[Client]:
    async with connect(srv.ws, origin=origin, open_timeout=10) as ws:  # type: ignore[arg-type]
        yield Client(ws)


# ======================================================================================
# Fixtures
# ======================================================================================
@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("data")
    (d / "network_geo.json").write_text(json.dumps(GEO_STUB), encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def server(toy_graph: Graph, data_dir: Path) -> Iterator[LiveServer]:
    """Default server: no model, ALLOW_NO_ORIGIN=1, default limits."""
    with live(toy_graph, data_dir) as srv:
        yield srv


def write_dummy_onnx(path: Path, graph_version: str, seed: int = 7) -> Path:
    """Random-weight linear actor with the §4.9 I/O contract and metadata."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, (C.OBS_DIM, C.N_ACTIONS)).astype(np.float32)
    b = rng.normal(0.0, 1.0, (C.N_ACTIONS,)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["obs", "W", "B"], ["logits"])],
        "dummy_actor",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", C.OBS_DIM])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", C.N_ACTIONS])],
        initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="test_server")
    model.ir_version = 9
    helper.set_model_props(model, {
        "obs_dim": str(C.OBS_DIM), "n_actions": str(C.N_ACTIONS), "action_names": json.dumps(list(C.ACTION_NAMES)),
        "graph_version": graph_version, "constants_hash": C.constants_hash(), "trained_steps": "0",
        "created_utc": "2026-11-20T00:00:00Z",
    })
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


@pytest.fixture(scope="module")
def dummy_model(tmp_path_factory: pytest.TempPathFactory, toy_graph: Graph) -> Path:
    return write_dummy_onnx(tmp_path_factory.mktemp("model") / "dummy.onnx", toy_graph.graph_version)


@pytest.fixture(scope="module")
def ai_server(toy_graph: Graph, data_dir: Path, dummy_model: Path) -> Iterator[LiveServer]:
    """A server whose lifespan loads the dummy ONNX model from SUBWAY_MODEL_PATH."""
    with live(toy_graph, data_dir, model_path=dummy_model) as srv:
        yield srv


SEG_RIVER = 10  # toy: Transfer Plaza – Riverside (B division, local only)


# ======================================================================================
# HTTP
# ======================================================================================
def test_health_exact_body_without_model(server: LiveServer) -> None:
    r = server.get("/health")
    assert r.status_code == 200
    assert r.content == b'{"status":"ok","model_loaded":false}'


def test_health_exact_body_with_model(ai_server: LiveServer) -> None:
    r = ai_server.get("/health")
    assert r.status_code == 200
    assert r.content == b'{"status":"ok","model_loaded":true}'


def test_stats_reports_sessions_governor_uptime_and_graph(server: LiveServer, toy_graph: Graph) -> None:
    s = server.stats()
    for key in ("sessions", "detached", "max_sessions", "governor_load", "governor_scale", "uptime_s",
                "process_cpu_s", "graph_version", "model_loaded", "totals"):
        assert key in s, key
    assert s["graph_version"] == toy_graph.graph_version
    assert s["max_sessions"] == 40 and s["model_loaded"] is False


def test_network_geo_is_served_from_the_data_dir(server: LiveServer) -> None:
    r = server.get("/network_geo.json")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json() == GEO_STUB


def test_network_geo_missing_is_404(toy_graph: Graph, tmp_path: Path) -> None:
    with live(toy_graph, tmp_path) as srv:
        assert srv.get("/network_geo.json").status_code == 404


def test_cors_allows_only_configured_origins(server: LiveServer) -> None:
    ok = server.get("/health", headers={"Origin": "https://kenmar.dev"})
    assert ok.headers.get("access-control-allow-origin") == "https://kenmar.dev"
    bad = server.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


# ======================================================================================
# Hello, welcome, frames
# ======================================================================================
@pytest.mark.asyncio
async def test_hello_welcome_then_frames_decode(server: LiveServer, toy_graph: Graph) -> None:
    async with client(server) as c:
        w = await c.hello()
        assert w["protocol"] == 1 and w["resumed"] is False
        assert w["graph_version"] == toy_graph.graph_version
        assert w["model_loaded"] is False and w["ai_available"] is False
        assert w["settings"] == {"mode": "normal", "speed": "1x", "profile": "am_peak"}
        assert w["constants"] == {"dt_s": 10, "frame_hz": 4, "speeds": {"pause": 0, "1x": 60, "2x": 120, "5x": 300},
                                  "idle_timeout_s": 300}
        assert len(w["resume_token"]) >= 32 and w["session_id"]
        assert (await c.expect("disruptions"))["items"] == []
        m = await c.expect("metrics")
        assert m["mode"] == "normal" and m["twin"] is None and m["ai_since_t"] is None
        assert set(m["actions"]) == set(C.ACTION_NAMES[1:]) and sum(m["actions"].values()) == 0
        t0 = time.monotonic()
        frames = [await c.expect_frame() for _ in range(6)]
        elapsed = time.monotonic() - t0
    assert elapsed < 3.0  # ~4 Hz
    for f in frames:
        assert f.n_stations == toy_graph.n_stations and f.mode == "normal" and f.speed == "1x"
        assert not f.twin_active and f.n_trains > 0
        assert np.all(np.diff(f.train_id.astype(int)) > 0)
        assert int(f.edge.max()) < toy_graph.n_edges and int(f.route.max()) < toy_graph.n_routes
    seqs = [f.seq for f in frames]
    clocks = [f.clock_s for f in frames]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert clocks == sorted(clocks) and clocks[-1] > clocks[0]  # 1x: 60 sim-s per real s


@pytest.mark.asyncio
async def test_hello_restore_recreates_settings_and_disruptions(server: LiveServer) -> None:
    restore = {"mode": "ai", "speed": "2x", "profile": "midday",
               "disruptions": [{"kind": "local_block", "target": {"type": "segment", "idx": SEG_RIVER},
                                "duration_s": None},
                               {"kind": "express_block", "target": {"type": "segment", "idx": 0},  # invalid: skipped
                                "duration_s": 600}]}
    async with client(server) as c:
        w = await c.hello(restore=restore)
        assert w["settings"] == {"mode": "normal", "speed": "2x", "profile": "midday"}  # no model: AI not restored
        d = await c.expect("disruptions")
        assert [(i["kind"], i["target"]) for i in d["items"]] == [("local_block", {"type": "segment", "idx": SEG_RIVER})]
        f = await c.expect_frame()
        assert f.speed == "2x" and C.PROFILE_START_S["midday"] <= f.clock_s < C.PROFILE_START_S["pm_rush"]


@pytest.mark.asyncio
async def test_set_speed_is_reflected_in_frames_and_pause_stops_the_clock(server: LiveServer) -> None:
    async with client(server) as c:
        await c.hello()
        await c.send(type="set_speed", speed="pause")
        f1 = await c.expect_frame(where=lambda f: f.speed == "pause")
        f2 = await c.expect_frame()
        f3 = await c.expect_frame()
        assert f1.clock_s == f2.clock_s == f3.clock_s and f3.seq > f1.seq
        await c.send(type="set_speed", speed="5x")
        g1 = await c.expect_frame(where=lambda f: f.speed == "5x")
        g2 = await c.expect_frame()
        assert g2.clock_s - g1.clock_s >= 5 * C.DT_S  # 7.5 ticks per frame at 5x


# ======================================================================================
# AI mode
# ======================================================================================
@pytest.mark.asyncio
async def test_set_mode_ai_without_model_is_ai_unavailable(server: LiveServer) -> None:
    async with client(server) as c:
        w = await c.hello()
        assert w["ai_available"] is False
        await c.send(type="set_mode", mode="ai")
        err = await c.expect("error")
        assert err["code"] == "ai_unavailable"
        m = await c.expect("metrics")
        assert m["mode"] == "normal" and m["twin"] is None
        f = await c.expect_frame()
        assert f.mode == "normal" and not f.twin_active


@pytest.mark.asyncio
async def test_set_mode_ai_runs_the_policy_with_a_normal_twin(ai_server: LiveServer) -> None:
    async with client(ai_server) as c:
        w = await c.hello()
        assert w["model_loaded"] is True and w["ai_available"] is True
        await c.send(type="set_mode", mode="ai")
        m = await c.expect("metrics", where=lambda m: m["mode"] == "ai")
        assert m["twin"] is not None and isinstance(m["ai_since_t"], float)
        assert set(m["twin"]) == set(m["live"])
        await c.send(type="set_speed", speed="5x")
        await c.send(type="preset", id="toy-union-closure")
        started = await c.expect("events", where=lambda e: any(i["kind"] == "disruption_started" for i in e["items"]))
        assert started["items"][0]["disruption"]["target"] == {"type": "station", "idx": TOY_STATION["A04"]}
        after = await c.expect("metrics", where=lambda m: m["mode"] == "ai")
        assert after["ai_since_t"] == float(C.PROFILE_START_S["am_peak"])  # twin restarted with the preset
        await c.collect(4.0)
        f = await c.expect_frame()
        assert f.mode == "ai" and f.twin_active
        metrics = c.json_of("metrics")
        assert all(m["twin"] is not None for m in metrics if m["mode"] == "ai")
        assert sum(metrics[-1]["actions"].values()) > 0, "the policy applied no dispatch action"
        callouts = [i for e in c.json_of("events") for i in e["items"] if i["kind"] == "callout"]
        assert callouts, "no callout for applied AI actions"
        for co in callouts:
            assert co["action"] in C.ACTION_NAMES[1:] and co["route"] in ("1", "2", "L") and co["text"]
            assert co["more"] >= 0 and (co["more"] == 0) == ("more dispatch action" not in co["text"])
        await c.send(type="set_mode", mode="normal")
        m = await c.expect("metrics", where=lambda m: m["mode"] == "normal")
        assert m["twin"] is None and m["ai_since_t"] is None
        f = await c.expect_frame(where=lambda f: f.mode == "normal")
        assert not f.twin_active


@pytest.mark.asyncio
async def test_callouts_are_rate_limited_to_one_per_1_5_s(ai_server: LiveServer) -> None:
    async with client(ai_server) as c:
        await c.hello(restore={"mode": "ai", "speed": "5x", "profile": "am_peak", "disruptions": [
            {"kind": "full_station_closure", "target": {"type": "station", "idx": TOY_STATION["A04"]},
             "duration_s": None}]})
        arrivals: list[float] = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                item = await c.next(0.5)
            except (asyncio.TimeoutError, TimeoutError):
                continue
            if isinstance(item, dict) and item["type"] == "events" and any(i["kind"] == "callout" for i in item["items"]):
                arrivals.append(time.monotonic())
    assert len(arrivals) >= 2
    gaps = np.diff(arrivals)
    assert float(gaps.min()) >= 1.3  # 1.5 s real time, minus network jitter on the client clock


# ======================================================================================
# Capacity, idle, origin
# ======================================================================================
@pytest.mark.asyncio
async def test_capacity_cap_sends_at_capacity_and_closes_1013(toy_graph: Graph, data_dir: Path) -> None:
    with live(toy_graph, data_dir, max_sessions=2) as srv:
        async with client(srv) as a, client(srv) as b:
            await a.hello()
            await b.hello()
            async with client(srv) as c:
                msg = await c.next()
                assert msg == {"type": "at_capacity",
                               "message": "Subway Domino is at capacity right now — please try again in a minute."}
                assert await c.expect_close() == 1013
            assert srv.stats()["sessions"] == 2
            await a.ws.close()
            assert wait_for(lambda: srv.stats()["detached"] == 1)
            # detached sessions do not count toward MAX_SESSIONS
            async with client(srv) as d:
                assert (await d.hello())["resumed"] is False
        assert srv.stats()["totals"]["rejected_capacity"] == 1


@pytest.mark.asyncio
async def test_idle_timeout_closes_1000_and_ping_does_not_count(toy_graph: Graph, data_dir: Path) -> None:
    with live(toy_graph, data_dir, idle_timeout_s=2.0) as srv:
        async with client(srv) as c:
            w = await c.hello()
            assert w["constants"]["idle_timeout_s"] == 2
            t0 = time.monotonic()

            async def pinger() -> None:
                while True:
                    await asyncio.sleep(0.4)
                    await c.send(type="ping", t=time.monotonic())

            task = asyncio.create_task(pinger())
            try:
                await c.expect("idle_timeout", timeout=6.0)
                idle_after = time.monotonic() - t0
                assert await c.expect_close() == 1000
            finally:
                task.cancel()
            assert c.json_of("pong"), "pings were answered"
        assert 1.7 <= idle_after <= 3.5
        assert wait_for(lambda: srv.stats()["detached"] == 1)  # resumable after the idle close


@pytest.mark.asyncio
async def test_activity_resets_the_idle_timer(toy_graph: Graph, data_dir: Path) -> None:
    with live(toy_graph, data_dir, idle_timeout_s=2.0) as srv:
        async with client(srv) as c:
            await c.hello()
            t0 = time.monotonic()
            await c.collect(1.3)
            await c.send(type="activity")
            await c.expect("idle_timeout", timeout=6.0)
            assert time.monotonic() - t0 >= 3.0
            assert await c.expect_close() == 1000


@pytest.mark.asyncio
async def test_origin_check(toy_graph: Graph, data_dir: Path) -> None:
    with live(toy_graph, data_dir, allow_no_origin=False) as srv:
        for origin in (None, "https://evil.example", "null", "https://kenmar.dev.evil.example"):
            async with client(srv, origin=origin) as c:
                assert await c.expect_close() == 1008, origin
        for origin in ("https://kenmar.dev", "http://localhost:3000"):
            async with client(srv, origin=origin) as c:
                assert (await c.hello())["protocol"] == 1
        assert srv.stats()["totals"]["rejected_origin"] == 4


# ======================================================================================
# Resume
# ======================================================================================
@pytest.mark.asyncio
async def test_resume_token_reattaches_to_the_exact_state(server: LiveServer) -> None:
    async with client(server) as a:
        w1 = await a.hello()
        await a.send(type="set_speed", speed="2x")
        await a.send(type="add_disruption", kind="local_block", target={"type": "segment", "idx": SEG_RIVER},
                     duration_s=None)
        started = await a.expect("events")
        did = started["items"][0]["disruption"]["id"]
        before = await a.expect_frame(where=lambda f: f.speed == "2x")
    await asyncio.sleep(0.3)
    async with client(server) as b:
        w2 = await b.hello(resume_token=w1["resume_token"],
                           restore={"mode": "normal", "speed": "5x", "profile": "pm_rush", "disruptions": []})
        assert w2["resumed"] is True and w2["session_id"] == w1["session_id"]
        assert w2["resume_token"] != w1["resume_token"]  # rotated on every reattach
        assert w2["settings"] == {"mode": "normal", "speed": "2x", "profile": "am_peak"}  # restore ignored
        d = await b.expect("disruptions")
        assert [i["id"] for i in d["items"]] == [did]
        after = await b.expect_frame()
        assert after.seq > before.seq and after.clock_s >= before.clock_s  # continued, not reset
        assert after.clock_s - before.clock_s < 300  # paused while detached
        # the token of an *attached* session is not resumable (no stealing): restore is used
        async with client(server) as c:
            w3 = await c.hello(resume_token=w2["resume_token"],
                               restore={"mode": "normal", "speed": "5x", "profile": "midday", "disruptions": []})
            assert w3["resumed"] is False and w3["session_id"] != w1["session_id"]
            assert w3["settings"] == {"mode": "normal", "speed": "5x", "profile": "midday"}
    # the rotated-away token is dead too
    async with client(server) as e:
        w4 = await e.hello(resume_token=w1["resume_token"])
        assert w4["resumed"] is False


@pytest.mark.asyncio
async def test_detached_session_expires_after_the_grace_period(toy_graph: Graph, data_dir: Path) -> None:
    with live(toy_graph, data_dir, detach_grace_s=0.6) as srv:
        async with client(srv) as a:
            w1 = await a.hello()
        assert wait_for(lambda: srv.stats()["detached"] == 1, timeout=2.0) or srv.stats()["detached"] == 0
        assert wait_for(lambda: srv.stats()["detached"] == 0, timeout=3.0)
        async with client(srv) as b:
            w2 = await b.hello(resume_token=w1["resume_token"],
                               restore={"mode": "normal", "speed": "2x", "profile": "late_night", "disruptions": []})
            assert w2["resumed"] is False and w2["settings"]["profile"] == "late_night"
        assert srv.stats()["totals"]["sessions_expired"] >= 1


@pytest.mark.asyncio
async def test_resuming_needs_a_free_slot(toy_graph: Graph, data_dir: Path) -> None:
    with live(toy_graph, data_dir, max_sessions=1, max_detached_sessions=4) as srv:
        async with client(srv) as a:
            w1 = await a.hello()
        assert wait_for(lambda: srv.stats()["detached"] == 1)
        async with client(srv) as b:
            await b.hello()  # the detached session does not hold the only slot
            async with client(srv) as c:
                assert (await c.next())["type"] == "at_capacity"
                assert await c.expect_close() == 1013
        async with client(srv) as d:
            assert (await d.hello(resume_token=w1["resume_token"]))["resumed"] is True


@pytest.mark.asyncio
async def test_the_detached_pool_is_bounded_oldest_first(toy_graph: Graph, data_dir: Path) -> None:
    with live(toy_graph, data_dir, max_detached_sessions=1) as srv:
        async with client(srv) as a:
            wa = await a.hello()
        assert wait_for(lambda: srv.stats()["detached"] == 1)
        async with client(srv) as b:
            wb = await b.hello()
        assert wait_for(lambda: srv.stats()["totals"].get("sessions_evicted", 0) == 1)
        assert srv.stats()["detached"] == 1
        async with client(srv) as d:
            assert (await d.hello(resume_token=wb["resume_token"]))["resumed"] is True
            async with client(srv) as c:
                assert (await c.hello(resume_token=wa["resume_token"]))["resumed"] is False  # a was evicted


# ======================================================================================
# Presets, disruptions, profile, reset
# ======================================================================================
@pytest.mark.asyncio
async def test_preset_resets_to_its_profile_and_starts_its_disruptions(server: LiveServer, toy_graph: Graph) -> None:
    async with client(server) as c:
        await c.hello(restore={"mode": "normal", "speed": "1x", "profile": "late_night", "disruptions": []})
        await c.send(type="preset", id="toy-river-block")
        ev = await c.expect("events")
        assert [(i["kind"], i["disruption"]["kind"], i["disruption"]["target"]) for i in ev["items"]] == [
            ("disruption_started", "local_block", {"type": "segment", "idx": SEG_RIVER})]
        assert ev["items"][0]["disruption"]["end_t"] is None  # network.json says 1800 s; the UI blocks until cleared
        d = await c.expect("disruptions")
        assert len(d["items"]) == 1 and d["items"][0]["end_t"] is None
        m = await c.expect("metrics")
        assert m["t"] == C.PROFILE_START_S["am_peak"]
        f = await c.expect_frame(where=lambda f: f.clock_s >= C.PROFILE_START_S["am_peak"])
        assert f.clock_s < C.PROFILE_START_S["am_peak"] + 120
        await c.send(type="preset", id="no-such-preset")
        err = await c.expect("error")
        assert err["code"] == "bad_request" and "no-such-preset" in err["message"]


def test_as_blockages_makes_presets_indefinite_and_turns_signal_delay_into_a_local_block() -> None:
    timed = Scenario(seed=3, profile="pm_rush", disruptions=(
        ScheduledDisruption("signal_delay", "interlocking", 7, 0.0, 1800.0),
        ScheduledDisruption("express_block", "segment", 4, 0.0, 600.0),
        ScheduledDisruption("full_station_closure", "station", 2, 0.0, None),
    ))
    out = as_blockages(timed)
    assert [(d.kind, d.target_type, d.target_idx, d.start_s, d.duration_s) for d in out.disruptions] == [
        ("local_block", "interlocking", 7, 0.0, None),  # same target, now a blockage
        ("express_block", "segment", 4, 0.0, None),
        ("full_station_closure", "station", 2, 0.0, None),
    ]
    assert (out.seed, out.profile) == (3, "pm_rush")
    assert timed.disruptions[0].kind == "signal_delay"  # the source scenario is untouched


@pytest.mark.asyncio
async def test_signal_delay_preset_arrives_as_an_indefinite_local_block(tmp_path: Path) -> None:
    """A preset whose data says ``signal_delay`` for 1800 s is served as a local block with no end."""
    data = toy_network_dict()
    data["presets"].append({"id": "toy-signals", "title": "Signal problems", "subtitle": "1 2 · Toy trunk",
                            "profile": "pm_rush", "focus": {"x": 0.0, "y": 0.0, "zoom_m_per_px": 6.0},
                            "disruptions": [{"kind": "signal_delay", "target": {"type": "segment", "idx": SEG_RIVER},
                                             "duration_s": 1800}]})
    data["graph_version"] = canonical_graph_version(data)
    graph = graph_from_dict(data, source="<toy signals>")
    assert preset_scenario("toy-signals", graph).disruptions[0].kind == "signal_delay"  # sim side unchanged
    (tmp_path / "network_geo.json").write_text(json.dumps(GEO_STUB), encoding="utf-8")
    with live(graph, tmp_path) as srv:
        async with client(srv) as c:
            await c.hello()
            await c.expect("disruptions")
            await c.send(type="preset", id="toy-signals")
            ev = await c.expect("events", where=lambda e: any(i["kind"] == "disruption_started" for i in e["items"]))
            started = [i["disruption"] for i in ev["items"] if i["kind"] == "disruption_started"]
            assert [(d["kind"], d["target"], d["end_t"]) for d in started] == [
                ("local_block", {"type": "segment", "idx": SEG_RIVER}, None)]
            d = await c.expect("disruptions", where=lambda d: len(d["items"]) == 1)
            assert d["items"][0]["kind"] == "local_block" and d["items"][0]["end_t"] is None
            m = await c.expect("metrics")
            assert m["t"] == C.PROFILE_START_S["pm_rush"]


@pytest.mark.asyncio
async def test_disruptions_add_clear_clear_all_and_expiry(server: LiveServer) -> None:
    async with client(server) as c:
        await c.hello()
        await c.expect("disruptions")
        await c.send(type="add_disruption", kind="local_block", target={"type": "segment", "idx": SEG_RIVER},
                     duration_s=None)
        ev = await c.expect("events")
        info = ev["items"][0]["disruption"]
        assert ev["items"][0]["kind"] == "disruption_started" and info["end_t"] is None
        assert info["label"] == "Local track blocked: Transfer Plaza ↔ Riverside"
        assert [i["id"] for i in (await c.expect("disruptions"))["items"]] == [info["id"]]

        await c.send(type="add_disruption", kind="express_block", target={"type": "segment", "idx": 0},
                     duration_s=600)
        assert (await c.expect("error"))["code"] == "invalid_target"
        await c.send(type="add_disruption", kind="local_block", target={"type": "segment", "idx": 999},
                     duration_s=600)
        assert (await c.expect("error"))["code"] == "invalid_target"
        await c.send(type="add_disruption", kind="flood", target={"type": "segment", "idx": 1}, duration_s=600)
        assert (await c.expect("error"))["code"] == "bad_request"

        await c.send(type="clear_disruption", id=info["id"])
        ev = await c.expect("events")
        assert ev["items"][0]["kind"] == "disruption_cleared" and ev["items"][0]["disruption"]["id"] == info["id"]
        assert (await c.expect("disruptions"))["items"] == []
        n_err = len(c.json_of("error"))
        await c.send(type="clear_disruption", id=info["id"])  # idempotent: no error, current list
        assert (await c.expect("disruptions"))["items"] == []
        assert len(c.json_of("error")) == n_err

        for target in ({"type": "segment", "idx": SEG_RIVER}, {"type": "station", "idx": TOY_STATION["A02"]}):
            await c.send(type="add_disruption", kind="local_block", target=target, duration_s=1200)
            await c.expect("events")
        await c.send(type="clear_all")
        ev = await c.expect("events")
        assert [i["kind"] for i in ev["items"]] == ["disruption_cleared", "disruption_cleared"]
        assert (await c.expect("disruptions"))["items"] == []

        await c.send(type="set_speed", speed="5x")
        await c.send(type="add_disruption", kind="signal_delay", target={"type": "segment", "idx": SEG_RIVER},
                     duration_s=30)
        did = (await c.expect("events"))["items"][0]["disruption"]["id"]
        ev = await c.expect("events", timeout=6.0,
                            where=lambda e: any(i["kind"] == "disruption_expired" for i in e["items"]))
        assert ev["items"][0]["disruption"]["id"] == did
        assert (await c.expect("disruptions"))["items"] == []


@pytest.mark.asyncio
async def test_disruption_ids_increase_across_resets(server: LiveServer) -> None:
    async with client(server) as c:
        await c.hello()
        ids = []
        for _ in range(2):
            await c.send(type="add_disruption", kind="local_block", target={"type": "segment", "idx": SEG_RIVER},
                         duration_s=None)
            ids.append((await c.expect("events"))["items"][0]["disruption"]["id"])
            assert len((await c.expect("disruptions"))["items"]) == 1
            await c.send(type="reset")
            assert (await c.expect("disruptions"))["items"] == []
        assert ids[1] > ids[0]


@pytest.mark.asyncio
async def test_resets_report_the_disruptions_they_end(server: LiveServer) -> None:
    """reset / set_profile / preset end every active disruption: each gets a ``disruption_cleared``."""
    async with client(server) as c:
        await c.hello()
        await c.expect("disruptions")
        for reset in ({"type": "reset"}, {"type": "set_profile", "profile": "midday"},
                      {"type": "preset", "id": "toy-river-block"}):
            await c.send(type="add_disruption", kind="local_block",
                         target={"type": "station", "idx": TOY_STATION["A02"]}, duration_s=None)
            did = (await c.expect("events"))["items"][0]["disruption"]["id"]
            await c.expect("disruptions", where=lambda d: len(d["items"]) >= 1)
            await c.send(**reset)
            ev = await c.expect("events", timeout=6.0)
            kinds = [(i["kind"], i["disruption"]["id"]) for i in ev["items"]]
            assert kinds[0] == ("disruption_cleared", did), reset
            if reset["type"] == "preset":  # the preset's own disruption starts after the old ones end
                assert [k for k, _ in kinds] == ["disruption_cleared", "disruption_started"]
                assert kinds[1][1] > did
            else:
                assert len(kinds) == 1
            d = await c.expect("disruptions")
            assert did not in [i["id"] for i in d["items"]]  # (3 resets fit the reset burst of 3)


@pytest.mark.asyncio
async def test_set_profile_resets_and_clears_disruptions(server: LiveServer) -> None:
    async with client(server) as c:
        await c.hello()
        await c.send(type="add_disruption", kind="local_block", target={"type": "segment", "idx": SEG_RIVER},
                     duration_s=None)
        await c.expect("events")
        await c.send(type="set_profile", profile="late_night")
        d = await c.expect("disruptions", where=lambda d: d["items"] == [])
        assert d["items"] == []
        m = await c.expect("metrics", where=lambda m: m["t"] < C.PROFILE_START_S["am_peak"])
        assert m["t"] == C.PROFILE_START_S["late_night"]
        f = await c.expect_frame(where=lambda f: f.clock_s < C.PROFILE_START_S["am_peak"])
        assert f.clock_s >= C.PROFILE_START_S["late_night"]
        await c.send(type="set_speed", speed="5x")
        await c.collect(1.0)
        await c.send(type="reset")
        m = await c.expect("metrics", where=lambda m: m["t"] == C.PROFILE_START_S["late_night"])
        assert m["live"]["riders_served"] == 0


# ======================================================================================
# Protocol errors and limits
# ======================================================================================
@pytest.mark.asyncio
async def test_malformed_messages_get_bad_request_and_the_socket_stays_open(server: LiveServer) -> None:
    async with client(server) as c:
        await c.hello()
        for raw in ("{not json", '{"type":"warp"}', '{"type":"set_speed","speed":"3x"}',
                    '{"type":"set_mode","mode":"ai","extra":1}', '{"type":"ping","t":"soon"}', "[]"):
            await c.ws.send(raw)
            err = await c.expect("error")
            assert err["code"] == "bad_request", raw
        await c.ws.send(b"\x00\x01")
        assert (await c.expect("error"))["code"] == "bad_request"
        await c.send(type="hello", protocol=1, resume_token=None, restore=None)
        err = await c.expect("error")
        assert err["code"] == "bad_request" and "hello" in err["message"].lower()
        await c.send(type="ping", t=42.5)
        assert (await c.expect("pong")) == {"type": "pong", "t": 42.5}


@pytest.mark.asyncio
async def test_first_message_must_be_a_protocol_1_hello(server: LiveServer) -> None:
    async with client(server) as c:
        await c.send(type="set_mode", mode="ai")
        assert (await c.expect("error"))["code"] == "bad_request"
        assert await c.expect_close() == 1008
    async with client(server) as c:
        await c.send(type="hello", protocol=2, resume_token=None, restore=None)
        err = await c.expect("error")
        assert "protocol" in err["message"]
        assert await c.expect_close() == 1008


@pytest.mark.asyncio
async def test_hello_timeout(toy_graph: Graph, data_dir: Path) -> None:
    with live(toy_graph, data_dir, hello_timeout_s=0.5) as srv:
        async with client(srv) as c:
            t0 = time.monotonic()
            assert (await c.expect("error", timeout=3.0))["code"] == "bad_request"
            assert await c.expect_close() == 1008
            assert time.monotonic() - t0 < 2.5
        assert srv.stats()["connecting"] == 0  # the reservation was released


@pytest.mark.asyncio
async def test_message_flood_is_rate_limited(server: LiveServer) -> None:
    async with client(server) as c:
        await c.hello()
        for i in range(150):
            await c.send(type="ping", t=float(i))
        err = await c.expect("error", timeout=5.0)
        assert err["code"] == "rate_limited"
        await c.collect(0.5)
        assert 0 < len(c.json_of("pong")) < 150


@pytest.mark.asyncio
async def test_resets_are_rate_limited(server: LiveServer) -> None:
    async with client(server) as c:
        await c.hello()
        for _ in range(6):
            await c.send(type="reset")
        err = await c.expect("error", timeout=5.0)
        assert err["code"] == "rate_limited"


@pytest.mark.asyncio
async def test_active_disruptions_are_capped(toy_graph: Graph, data_dir: Path) -> None:
    with live(toy_graph, data_dir, max_disruptions=2) as srv:
        async with client(srv) as c:
            await c.hello()
            for target in ({"type": "segment", "idx": SEG_RIVER}, {"type": "segment", "idx": 8}):
                await c.send(type="add_disruption", kind="local_block", target=target, duration_s=None)
                await c.expect("events")
            await c.send(type="add_disruption", kind="local_block", target={"type": "segment", "idx": 9},
                         duration_s=None)
            err = await c.expect("error")
            assert err["code"] == "bad_request" and "2" in err["message"]


# ======================================================================================
# Session semantics without a socket (twin mirroring, policy failure)
# ======================================================================================
class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[Any] = []
        self.closed: tuple[int, str | None] | None = None

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(P.decode_frame(data))

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed = (code, reason)


class LastValidPolicy:
    """Test double: the highest-index valid action (deterministic, exercises apply_actions)."""

    def act(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.array([int(np.flatnonzero(row)[-1]) for row in masks], dtype=np.int64)


class BrokenPolicy:
    def act(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        raise RuntimeError("inference exploded")


async def _session(graph: Graph, tmp: Path, policy: Any) -> tuple[Session, Outbox, FakeSocket]:
    ctx = SessionContext(graph=graph, policy=policy, settings=make_settings(tmp))
    s = await Session.create(ctx, session_id="unit", restore=None)
    sock = FakeSocket()
    out = Outbox(sock)
    out.start()
    s.attach(out, resume_token="tok", resumed=False, model_loaded=policy is not None)
    return s, out, sock


async def _drain(s: Session, out: Outbox) -> None:
    while s._inbox:  # noqa: SLF001 - unit test drives the handlers deterministically
        await s._handle(s._inbox.popleft())  # noqa: SLF001
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_twin_mirrors_disruptions_and_runs_in_lockstep(toy_graph: Graph, tmp_path: Path) -> None:
    s, out, sock = await _session(toy_graph, tmp_path, LastValidPolicy())
    s.submit(P.SetMode(type="set_mode", mode="ai"))
    s.submit(P.AddDisruption(type="add_disruption", kind="full_station_closure",
                             target=P.Target(type="station", idx=TOY_STATION["A04"]), duration_s=None))
    await _drain(s, out)
    assert s.twin is not None and s.mode == "ai"
    assert [d["target"] for d in s.twin.disruptions()] == [d["target"] for d in s.sim.disruptions()]
    s.speed = "5x"
    for _ in range(40):
        s._cycle(out, send_metrics=False)  # noqa: SLF001
        await asyncio.sleep(0)
    assert s.twin.clock_s == s.sim.clock_s
    assert sum(s.sim.action_counts().values()) > 0 and sum(s.twin.action_counts().values()) == 0
    assert s.sim.state_hash() != s.twin.state_hash()  # the AI acted; the twin only CONTINUEd
    did = int(s.sim.disruptions()[0]["id"])
    s.submit(P.ClearDisruption(type="clear_disruption", id=did))
    await _drain(s, out)
    assert s.sim.disruptions() == [] and s.twin.disruptions() == []
    s.submit(P.SetMode(type="set_mode", mode="normal"))
    await _drain(s, out)
    assert s.twin is None and s.ai_since_t is None
    await s.close()


@pytest.mark.asyncio
async def test_twin_matches_a_never_controlled_sim_exactly(toy_graph: Graph, tmp_path: Path) -> None:
    """The twin is Normal Schedule: it equals a sim that was simply stepped (§4.4 twin)."""
    s, out, _ = await _session(toy_graph, tmp_path, LastValidPolicy())
    reference = s.sim.copy()
    s.submit(P.SetMode(type="set_mode", mode="ai"))
    await _drain(s, out)
    s.speed = "2x"
    ticks = 0
    for _ in range(12):
        before = s.stats.ticks
        s._cycle(out, send_metrics=False)  # noqa: SLF001
        ticks += s.stats.ticks - before
    reference.step(ticks)
    assert s.twin is not None and s.twin.state_hash() == reference.state_hash()
    await s.close()


@pytest.mark.asyncio
async def test_reattach_during_a_reset_leaves_the_sim_to_the_worker_thread(toy_graph: Graph, tmp_path: Path) -> None:
    """A socket that resumes while a reset runs off-loop must not read the half-reset sim."""
    gate = threading.Event()
    gate.set()

    async def gated_thread(fn: Any, *args: Any) -> Any:
        def run() -> Any:
            assert gate.wait(5.0)
            return fn(*args)
        return await asyncio.to_thread(run)

    ctx = SessionContext(graph=toy_graph, policy=None, settings=make_settings(tmp_path), run_blocking=gated_thread)
    s = await Session.create(ctx, session_id="unit", restore=None)
    out1 = Outbox(FakeSocket())
    out1.start()
    s.attach(out1, resume_token="t1", resumed=False, model_loaded=False)
    s.submit(P.AddDisruption(type="add_disruption", kind="local_block",
                             target=P.Target(type="segment", idx=SEG_RIVER), duration_s=None))
    await _drain(s, out1)
    did = int(s.sim.disruptions()[0]["id"])

    reads_during_reset: list[bool] = []
    real_metrics = s.sim.metrics

    def spy_metrics() -> dict[str, Any]:
        reads_during_reset.append(s._resetting)  # noqa: SLF001
        return real_metrics()

    s.sim.metrics = spy_metrics  # type: ignore[method-assign]
    gate.clear()  # the reset now waits in the worker thread
    reset = asyncio.create_task(s._handle(P.Reset(type="reset")))  # noqa: SLF001
    while not s._resetting:  # noqa: SLF001
        await asyncio.sleep(0.005)
    s.detach()  # the client drops and resumes while the warm-up runs
    sock2 = FakeSocket()
    out2 = Outbox(sock2)
    out2.start()
    s.attach(out2, resume_token="t2", resumed=True, model_loaded=False)
    assert reads_during_reset == []  # attach did not touch the sim
    gate.set()
    await asyncio.wait_for(reset, 5.0)
    await asyncio.sleep(0.01)
    assert reads_during_reset == [False]  # metrics were read once, after the reset
    types = [m["type"] for m in sock2.sent]
    assert types == ["welcome", "disruptions", "events", "disruptions", "metrics"]
    assert sock2.sent[2]["items"] == [{"kind": "disruption_cleared", "disruption": sock2.sent[1]["items"][0]}]
    assert sock2.sent[1]["items"][0]["id"] == did and sock2.sent[3]["items"] == []
    await s.close()


@pytest.mark.asyncio
async def test_policy_failure_disables_ai_explicitly(toy_graph: Graph, tmp_path: Path) -> None:
    s, out, sock = await _session(toy_graph, tmp_path, BrokenPolicy())
    s.submit(P.SetMode(type="set_mode", mode="ai"))
    s.submit(P.AddDisruption(type="add_disruption", kind="full_station_closure",
                             target=P.Target(type="station", idx=TOY_STATION["A04"]), duration_s=None))
    await _drain(s, out)
    s.speed = "5x"
    for _ in range(40):
        s._cycle(out, send_metrics=False)  # noqa: SLF001
        await asyncio.sleep(0)
        if s.mode == "normal":
            break
    await asyncio.sleep(0.01)
    assert s.mode == "normal" and s.twin is None and not s.ai_available
    errors = [m for m in sock.sent if isinstance(m, dict) and m.get("type") == "error"]
    assert errors and errors[-1]["code"] == "ai_unavailable"
    s.submit(P.SetMode(type="set_mode", mode="ai"))
    await _drain(s, out)
    await asyncio.sleep(0.01)
    assert s.mode == "normal"
    await s.close()


# ======================================================================================
# Unit: accumulator, governor
# ======================================================================================
def test_ticks_due_follows_the_section_5_2_accumulator() -> None:
    acc, total = 0.0, 0
    for _ in range(40):  # 10 s at 1x = 60 ticks
        n, acc = ticks_due(acc, C.SPEEDS["1x"], 1.0)
        assert n in (1, 2)
        total += n
    assert total == 60
    acc, total = 0.0, 0
    for _ in range(4):  # 1 s at 5x = 30 ticks (7.5 per frame)
        n, acc = ticks_due(acc, C.SPEEDS["5x"], 1.0)
        total += n
    assert total == 30
    assert ticks_due(0.3, C.SPEEDS["pause"], 1.0) == (0, 0.3)
    n, acc = ticks_due(0.5, 1200.0, 1.0)  # 30 owed: capped at 8, backlog dropped
    assert n == 8 and 0.0 <= acc < 1.0
    acc, total = 0.0, 0
    for _ in range(40):  # governor at 0.5: half speed for everyone
        n, acc = ticks_due(acc, C.SPEEDS["1x"], 0.5)
        total += n
    assert total == 30


def test_governor_scales_down_on_overload_and_converges_to_the_budget() -> None:
    gov = CpuGovernor(0.70, alpha=0.5)
    fixed, demand = 0.08, 2.4  # load = fixed + demand * scale (one core saturates at 1.0)
    for _ in range(40):
        gov.observe(min(1.0, fixed + demand * gov.scale))
    assert 0.6 <= fixed + demand * gov.scale <= 0.75
    assert gov.scale == pytest.approx((0.70 - fixed) / demand, rel=0.12)
    for _ in range(40):  # load gone: recovers to full speed, gradually
        prev = gov.scale
        gov.observe(fixed)
        assert gov.scale <= prev * gov.max_raise + 1e-12
    assert gov.scale == 1.0


def test_governor_holds_inside_the_deadband_and_ignores_light_load() -> None:
    gov = CpuGovernor(0.70)
    for x in (0.1, 0.3, 0.5, 0.69):
        gov.observe(x)
    assert gov.scale == 1.0
    gov = CpuGovernor(0.70, alpha=1.0)
    gov.observe(1.4)
    assert gov.scale == pytest.approx(0.5)
    gov.observe(0.71)  # within ±5 %: unchanged
    assert gov.scale == pytest.approx(0.5)


@pytest.mark.parametrize("offload", [True, False])
def test_governor_load_excludes_one_off_reset_work(toy_graph: Graph, offload: bool) -> None:
    """Resets/construction run via run_blocking; tick scaling cannot reduce them, so they
    must not throttle every session (a burst of arrivals used to floor the scale)."""
    mgr = SessionManager(graph=toy_graph, policy=None, settings=Settings(), offload=offload)

    def burn() -> str:
        t0 = time.thread_time()
        while time.thread_time() - t0 < 0.2:
            pass
        return "done"

    async def go() -> str:
        try:
            return await mgr.run_blocking(burn)
        finally:
            await mgr.shutdown()

    governed0, process0 = mgr._governed_cpu_s(), time.process_time()
    assert asyncio.run(go()) == "done"
    assert time.process_time() - process0 >= 0.2
    assert mgr._governed_cpu_s() - governed0 < 0.05
    assert mgr.stats()["reset_cpu_s"] >= 0.2


def test_governor_samples_process_cpu_per_wall_second() -> None:
    clock = {"cpu": 0.0, "wall": 0.0}
    gov = CpuGovernor(0.7, alpha=1.0, cpu_clock=lambda: clock["cpu"], wall_clock=lambda: clock["wall"])
    clock["cpu"], clock["wall"] = 0.45, 0.5
    gov.sample()
    assert gov.load == pytest.approx(0.9)
    assert gov.scale == pytest.approx(0.7 / 0.9)


# ======================================================================================
# Unit: callouts
# ======================================================================================
def _event(action: int, *, t: float = 30000.0, train: int = 5, station: int = 1, route: int = 0, applied: bool = True,
           **detail: Any) -> ActionEvent:
    return ActionEvent(t=t, train_id=train, route_idx=route, action=action, station_idx=station, applied=applied,
                       detail=detail)


def test_callout_templates_are_factual_plain_english(toy_graph: Graph) -> None:
    w = CalloutWriter(toy_graph)
    blocked = {"kind": "local_block", "station_idx": TOY_STATION["A03"]}
    texts = {
        "hold_blocked": w.text(_event(C.A_HOLD, blocked_ahead=blocked, riders_onboard=840.4)),
        "hold_bunched": w.text(_event(C.A_HOLD, blocked_ahead=None, riders_onboard=12, leader_gap_s=95.0,
                                      sched_headway_s=300.0)),
        "express": w.text(_event(C.A_EXPRESS, blocked_ahead=blocked, rejoin_station_idx=TOY_STATION["A04"],
                                 bypassed_station_idx=[TOY_STATION["A03"]], riders_displaced=1.0)),
        "local": w.text(_event(C.A_LOCAL, route=1, blocked_ahead={"kind": "express_block", "station_idx": 1},
                               rejoin_station_idx=TOY_STATION["A04"])),
        "skip": w.text(_event(C.A_SKIP, skipped_station_idx=TOY_STATION["A03"], riders_overshot=9.2)),
        "skip_closed": w.text(_event(C.A_SKIP, skipped_station_idx=TOY_STATION["A04"],
                                     blocked_ahead={"kind": "full_station_closure", "station_idx": TOY_STATION["A04"]})),
        "short_turn": w.text(_event(C.A_SHORT_TURN, station=TOY_STATION["A04"], riders_forced_off=1.0,
                                    turnback_latency_s=227.0, blocked_ahead={"kind": "congestion", "station_idx": 4})),
        "reroute": w.text(_event(C.A_REROUTE, station=TOY_STATION["A05"], via_station_idx=TOY_STATION["A07"],
                                 rejoin_station_idx=TOY_STATION["A08"], bypassed_station_idx=[5],
                                 riders_displaced=31.0)),
        "yard": w.text(_event(C.A_YARD, route=2, station=10, yard_idx=0, lead_station_idx=10, deadhead_s=420.0)),
    }
    assert texts["hold_blocked"] == ("Holding a 1 train at Express Center with doors open — the local track ahead is "
                                     "blocked at Midtown Local, so its 840 riders wait on a platform instead of in a "
                                     "stalled tunnel.")
    assert "only 1.6 min ahead" in texts["hold_bunched"] and "every 5 min" in texts["hold_bunched"]
    assert "rejoins its route at Union Junction, passing 1 local stop" in texts["express"]
    assert "1 rider bound for those stops rides on" in texts["express"]
    assert texts["local"].startswith("Moving a 2 train to the local track at Express Center — the express track just "
                                     "ahead is blocked.")
    assert texts["skip"] == ("A 1 train at Express Center will pass Midtown Local without stopping; 9 riders bound "
                             "there ride on to the next stop.")
    assert texts["skip_closed"].endswith("through Union Junction without stopping — the station is closed.")
    assert "1 rider steps off" in texts["short_turn"] and "about 3.8 min" in texts["short_turn"]
    assert texts["reroute"].startswith("Rerouting a 1 train at Split Junction via East Branch") and \
        "It runs to Southgate." in texts["reroute"]
    assert texts["yard"] == "Sending a gap train out of Toy Yard — it enters service as an L train at Transfer Plaza in about 7 min."
    for name, text in texts.items():
        lowered = f" {text.lower()} "
        for phrase in (" i ", " we ", " i'm ", "decided", " wants ", " thinks ", "because i", " my "):
            assert phrase not in lowered, (name, phrase, text)


def test_route_phrases_and_formatting(toy_graph: Graph) -> None:
    w = CalloutWriter(toy_graph)
    assert w.route_phrase(0) == "a 1 train" and w.route_phrase(2) == "an L train"
    assert format_riders(0.4) == "0 riders" and format_riders(1.2) == "1 rider" and format_riders(843) == "840 riders"
    assert format_riders(1249.0) == "1,250 riders"
    assert format_duration(45) == "45 s" and format_duration(210) == "3.5 min" and format_duration(240) == "4 min"
    assert format_duration(725) == "12 min"


def test_callout_gate_rate_limits_and_aggregates(toy_graph: Graph) -> None:
    now = {"t": 100.0}
    gate = CalloutGate(CalloutWriter(toy_graph), clock=lambda: now["t"])
    assert gate.poll() is None
    gate.offer(_event(C.A_HOLD, train=1, riders_onboard=10))
    gate.offer(_event(C.A_SHORT_TURN, train=2, riders_forced_off=5, turnback_latency_s=200))
    gate.offer(_event(C.A_SKIP, train=3, skipped_station_idx=2, riders_overshot=1))
    gate.offer(_event(C.A_YARD, train=4, applied=False, yard_idx=0))  # not applied: never a callout
    first = gate.poll()
    assert first is not None and first.action == "SHORT_TURN_RELAY" and first.more == 2
    assert first.text.endswith("(+2 more dispatch actions)") and first.id == 1
    gate.offer(_event(C.A_SKIP, train=6, skipped_station_idx=2))
    now["t"] += 1.0
    assert gate.poll() is None  # inside the 1.5 s window
    now["t"] += 0.6
    second = gate.poll()
    assert second is not None and second.more == 0 and "more dispatch" not in second.text and second.id == 2
    assert gate.poll() is None


def test_hold_extensions_are_not_new_callouts(toy_graph: Graph) -> None:
    now = {"t": 0.0}
    gate = CalloutGate(CalloutWriter(toy_graph), clock=lambda: now["t"])
    for k in range(5):  # one train holding 5 × 30 s at the same station
        gate.offer(_event(C.A_HOLD, train=9, station=1, t=30000.0 + 30.0 * k, riders_onboard=50))
    c = gate.poll()
    assert c is not None and c.more == 0
    now["t"] += 2.0
    gate.offer(_event(C.A_HOLD, train=9, station=2, t=30400.0, riders_onboard=50))  # a new station: new hold
    assert gate.poll() is not None


# ======================================================================================
# Unit: settings and parser
# ======================================================================================
def test_settings_from_env() -> None:
    s = Settings.from_env({})
    assert s.allowed_origins == ("https://kenmar.dev", "http://localhost:3000")
    assert (s.max_sessions, s.idle_timeout_s, s.detach_grace_s, s.cpu_budget) == (40, 300.0, 90.0, 0.70)
    assert s.allow_no_origin is False and s.log_level == "INFO"
    s = Settings.from_env({"ALLOWED_ORIGINS": " https://kenmar.dev/ ,https://preview.vercel.app", "MAX_SESSIONS": "2",
                           "IDLE_TIMEOUT_S": "2", "DETACH_GRACE_S": "5", "ALLOW_NO_ORIGIN": "1", "CPU_BUDGET": "0.5",
                           "LOG_LEVEL": "debug", "SUBWAY_DATA_DIR": "/tmp/d", "SUBWAY_MODEL_PATH": "/tmp/m.onnx"})
    assert s.allowed_origins == ("https://kenmar.dev", "https://preview.vercel.app")
    assert (s.max_sessions, s.idle_timeout_s, s.detach_grace_s, s.cpu_budget) == (2, 2.0, 5.0, 0.5)
    assert s.allow_no_origin and s.log_level == "DEBUG"
    assert str(s.data_dir) == "/tmp/d" and str(s.model_path) == "/tmp/m.onnx"
    assert s.network_path.name == "network.json" and s.geo_path.name == "network_geo.json"
    for env in ({"MAX_SESSIONS": "many"}, {"CPU_BUDGET": "1.5"}, {"ALLOW_NO_ORIGIN": "maybe"}, {"IDLE_TIMEOUT_S": "0"},
                {"LOG_LEVEL": "chatty"}, {"ALLOWED_ORIGINS": " , "}):
        with pytest.raises(ConfigError):
            Settings.from_env(env)


def test_origin_matching() -> None:
    s = Settings()
    assert s.origin_allowed("https://kenmar.dev") and s.origin_allowed("http://localhost:3000")
    assert s.origin_allowed("HTTPS://KENMAR.DEV:443")
    for bad in ("https://kenmar.dev.evil.example", "http://kenmar.dev", "null", "https://localhost:3000", None, ""):
        assert not s.origin_allowed(bad), bad
    assert Settings(allow_no_origin=True).origin_allowed(None)
    assert Settings(allowed_origins=("*",)).origin_allowed("https://anything.example")
    assert normalize_origin("http://Example.com:80/") == "http://example.com"


def test_parser_is_strict() -> None:
    ok = P.parse_client_message('{"type":"add_disruption","kind":"signal_delay","target":{"type":"interlocking","idx":7},'
                                '"duration_s":1800}')
    assert isinstance(ok, P.AddDisruption) and ok.duration_s == 1800.0
    for raw in ('{"type":"add_disruption","kind":"signal_delay","target":{"type":"line","idx":7},"duration_s":1}',
                '{"type":"add_disruption","kind":"signal_delay","target":{"type":"segment","idx":1.5},"duration_s":1}',
                '{"type":"add_disruption","kind":"signal_delay","target":{"type":"segment","idx":1},"duration_s":0}',
                '{"type":"add_disruption","kind":"signal_delay","target":{"type":"segment","idx":true},"duration_s":1}',
                '{"type":"clear_disruption","id":"3"}', '{"type":"ping","t":NaN}', '{"type":"reset","now":true}',
                '{"type":"preset","id":""}', '{"type":"hello","protocol":1,"resume_token":7}'):
        with pytest.raises(P.ProtocolError) as exc:
            P.parse_client_message(raw)
        assert exc.value.code == "bad_request"
    with pytest.raises(P.ProtocolError):
        P.parse_client_message("x" * 100, max_bytes=64)


# ======================================================================================
# Unit: writer backpressure and ordering
# ======================================================================================
class SlowSocket:
    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.sent: list[tuple[str, Any]] = []

    async def send_text(self, data: str) -> None:
        self.sent.append(("text", json.loads(data)))

    async def send_bytes(self, data: bytes) -> None:
        await self.gate.wait()  # the client is not reading
        self.sent.append(("bytes", data))

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.sent.append(("close", code))


@pytest.mark.asyncio
async def test_outbox_skips_frames_while_one_is_pending_and_keeps_json_order() -> None:
    sock = SlowSocket()
    out = Outbox(sock)
    out.start()
    assert out.offer_frame(b"f1") is True
    await asyncio.sleep(0)
    assert out.frame_pending
    assert out.offer_frame(b"f2") is False and out.offer_frame(b"f3") is False
    assert out.stats.frames_skipped == 2
    out.send_json({"type": "pong", "t": 1.0})
    sock.gate.set()
    await asyncio.sleep(0.01)
    assert not out.frame_pending
    assert out.offer_frame(b"f4") is True
    await out.close(1000)
    assert sock.sent == [("bytes", b"f1"), ("text", {"type": "pong", "t": 1.0}), ("bytes", b"f4"), ("close", 1000)]
    assert out.stats.frames_sent == 2 and out.closed
    assert out.send_json({"type": "pong", "t": 2.0}) is False
