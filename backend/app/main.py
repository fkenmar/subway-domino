"""FastAPI application: HTTP routes and the ``/ws`` WebSocket (ARCHITECTURE §5.1).

``create_app(graph=None, policy=None, settings=None)`` builds an app; anything not injected
is loaded once by the lifespan from ``settings`` (``network.json`` and the ONNX model). The
module-level ``app`` is what ``uvicorn app.main:app`` serves (one worker).

Routes:

* ``GET /health`` → exactly ``{"status":"ok","model_loaded":<bool>}``
* ``GET /stats`` → sessions, governor load/scale, uptime, graph_version, totals
* ``GET /network_geo.json`` → the same file as the frontend's copy (fallback)
* ``WS /ws`` → protocol v1 (:mod:`app.protocol`). The ``Origin`` header is checked by hand
  against ``ALLOWED_ORIGINS`` (mismatch → close 1008; missing only with ``ALLOW_NO_ORIGIN``).
  At capacity the socket is accepted, told ``at_capacity`` and closed with 1013.

A model that fails to load is logged and leaves ``model_loaded = false``: Normal mode keeps
working and AI mode is unavailable. There is never a heuristic fallback.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse

from sim.graph import Graph, load_graph

from app import __version__
from app import protocol as P
from app.config import Settings
from app.manager import SessionManager
from app.ratelimit import TokenBucket
from app.session import PolicyLike, Session
from app.transport import CLOSE_POLICY, CLOSE_TRY_AGAIN_LATER, Outbox

log = logging.getLogger("app.main")

# Per-connection message rate limit (every client message, ping included).
MESSAGE_RATE_PER_S = 20.0
MESSAGE_BURST = 60
MAX_DROPPED_MESSAGES = 600  # a client this far over the limit is closed (1008)


def configure_logging(level: str) -> None:
    """Give the ``app`` loggers a handler (uvicorn only configures its own loggers)."""
    logger = logging.getLogger("app")
    logger.setLevel(level)
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False


def load_model(settings: Settings, graph: Graph) -> PolicyLike | None:
    """Load the ONNX policy once (1 intra-op thread, sequential); ``None`` if unavailable."""
    path = settings.model_path
    if not path.is_file():
        log.warning("no model at %s: AI mode unavailable (Normal Schedule only)", path)
        return None
    try:
        from sim.policy import OnnxPolicy  # imports onnxruntime lazily

        policy = OnnxPolicy(path, intra_op_threads=1)
        policy.check_graph_version(graph.graph_version)
    except Exception:
        log.exception("failed to load model %s: AI mode unavailable (Normal Schedule only)", path)
        return None
    log.info("loaded model %s (metadata: %s)", path.name, {k: v for k, v in policy.metadata.items() if k != "action_names"})
    return policy


def create_app(
    graph: Graph | None = None,
    policy: PolicyLike | None = None,
    settings: Settings | None = None,
    *,
    prewarm: bool = True,
    offload: bool = True,
) -> FastAPI:
    """Build the application. Injected ``graph``/``policy`` are used as is (tests)."""
    cfg = settings if settings is not None else Settings.from_env()
    configure_logging(cfg.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        g = graph if graph is not None else await asyncio.to_thread(load_graph, cfg.network_path)
        pol = policy if policy is not None else await asyncio.to_thread(load_model, cfg, g)
        manager = SessionManager(graph=g, policy=pol, settings=cfg, offload=offload)
        app.state.manager = manager
        if prewarm:
            dt = await manager.prewarm()
            log.info("graph %s ready (%d stations, %d edges); profiles prewarmed in %.2f s; model_loaded=%s",
                     g.graph_version, g.n_stations, g.n_edges, dt, manager.model_loaded)
        await manager.start()
        try:
            yield
        finally:
            await manager.shutdown()

    app = FastAPI(
        title="Subway Domino",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = cfg
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["*"],
        max_age=600,
    )

    @app.get("/health")
    async def health() -> JSONResponse:
        manager: SessionManager = app.state.manager
        return JSONResponse({"status": "ok", "model_loaded": manager.model_loaded})

    @app.get("/stats")
    async def stats() -> JSONResponse:
        manager: SessionManager = app.state.manager
        return JSONResponse(manager.stats(), headers={"Cache-Control": "no-store"})

    @app.get("/network_geo.json", response_model=None)
    async def network_geo() -> FileResponse | JSONResponse:
        path = cfg.geo_path
        if not path.is_file():
            return JSONResponse({"detail": "network_geo.json not found"}, status_code=404)
        return FileResponse(path, media_type="application/json", headers={"Cache-Control": "public, max-age=300"})

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await serve_socket(websocket, app.state.manager)

    return app


async def serve_socket(websocket: WebSocket, manager: SessionManager) -> None:
    """One WebSocket connection: Origin check, capacity, hello, then the receive loop."""
    settings = manager.settings
    origin = websocket.headers.get("origin")
    if not settings.origin_allowed(origin):
        manager.counters["rejected_origin"] += 1
        log.info("rejected WebSocket from origin %r", origin)
        await websocket.accept()
        await websocket.close(code=CLOSE_POLICY, reason="origin not allowed")
        return
    if not manager.try_reserve():
        await websocket.accept()
        await websocket.send_text(P.dumps(P.at_capacity_message()))
        await websocket.close(code=CLOSE_TRY_AGAIN_LATER, reason="at capacity")
        return

    reserved = True
    session: Session | None = None
    out = Outbox(websocket)
    try:
        await websocket.accept()
        out.start()
        manager.register_outbox(out)
        hello = await _receive_hello(websocket, out, settings.hello_timeout_s, settings.max_message_bytes)
        if hello is None:
            return
        reserved = False  # open() consumes the reservation, whatever happens
        session = await manager.open(out, hello)
        await _receive_loop(websocket, out, session, manager)
    except Exception:  # pragma: no cover - defensive: never let one socket take the server down
        log.exception("WebSocket handler failed")
        out.close_soon(1011, "internal error")
    finally:
        if reserved:
            manager.release()
        if session is not None:
            manager.connection_closed(session, out)
        else:
            manager.retire_outbox(out)  # the connection never became a session
        if out.open:
            out.abort()
        else:
            await out.wait_closed(2.0)


async def _receive_hello(websocket: WebSocket, out: Outbox, timeout_s: float, max_bytes: int) -> P.Hello | None:
    """Wait for the mandatory first ``hello`` (protocol 1); on anything else close 1008."""
    try:
        async with asyncio.timeout(timeout_s):
            msg = await websocket.receive()
    except TimeoutError:
        out.send_json(P.error_message("bad_request", "Send hello (protocol 1) first."))
        await out.close(CLOSE_POLICY, "hello timeout")
        return None
    if msg["type"] == "websocket.disconnect":
        out.abort()
        return None
    text = msg.get("text")
    try:
        if text is None:
            raise P.ProtocolError("bad_request", "messages must be JSON text")
        parsed = P.parse_client_message(text, max_bytes=max_bytes)
        if not isinstance(parsed, P.Hello):
            raise P.ProtocolError("bad_request", "Send hello (protocol 1) first.")
        if parsed.protocol != P.PROTOCOL_VERSION:
            raise P.ProtocolError(
                "bad_request", f"unsupported protocol {parsed.protocol}; this server speaks {P.PROTOCOL_VERSION}"
            )
    except P.ProtocolError as exc:
        out.send_json(P.error_message(exc.code, exc.message))
        await out.close(CLOSE_POLICY, "hello required")
        return None
    return parsed


async def _receive_loop(websocket: WebSocket, out: Outbox, session: Session, manager: SessionManager) -> None:
    bucket = TokenBucket(MESSAGE_RATE_PER_S, MESSAGE_BURST, clock=manager.clock)
    last_rate_error = -1e9
    dropped = 0
    max_bytes = manager.settings.max_message_bytes
    while True:
        msg = await websocket.receive()
        if msg["type"] == "websocket.disconnect":
            return
        if not bucket.take():
            dropped += 1
            now = manager.clock()
            if now - last_rate_error >= 1.0:
                last_rate_error = now
                out.send_json(P.error_message("rate_limited", "Too many messages — slow down."))
            if dropped >= MAX_DROPPED_MESSAGES:
                log.warning("session=%s flooding: closing", session.session_id[:8])
                out.close_soon(CLOSE_POLICY, "too many messages")
                return
            continue
        text = msg.get("text")
        if text is None:
            out.send_json(P.error_message("bad_request", "messages must be JSON text"))
            continue
        try:
            parsed: Any = P.parse_client_message(text, max_bytes=max_bytes)
        except P.ProtocolError as exc:
            manager.counters["protocol_errors"] += 1
            out.send_json(P.error_message(exc.code, exc.message))
            continue
        if isinstance(parsed, P.Ping):
            out.send_json(P.pong_message(parsed.t))  # answered here: RTT excludes frame cycles
            continue
        session.touch()
        if isinstance(parsed, P.Hello):
            out.send_json(P.error_message("bad_request", "Already said hello."))
            continue
        session.submit(parsed)


app = create_app()
