"""Subway Domino FastAPI / WebSocket server (ARCHITECTURE §5, WP4).

Modules:

* :mod:`app.config` — environment-driven :class:`~app.config.Settings` and §5.2 constants;
* :mod:`app.protocol` — WebSocket protocol v1: strict client-message models, server JSON
  builders and the binary train-frame codec (§5.3);
* :mod:`app.callouts` — plain-English dispatch callouts with rate limiting (§5.4);
* :mod:`app.transport` — the per-connection writer (ordered JSON, skip-on-backpressure frames);
* :mod:`app.session` — one isolated ``NYCSubwaySim`` per visitor and its asyncio loop (§5.2);
* :mod:`app.manager` — session registry, capacity cap, resume tokens, CPU governor;
* :mod:`app.main` — the FastAPI application factory and HTTP / WebSocket routes;
* :mod:`app.smoke` — ``python -m app.smoke`` end-to-end WebSocket check.

Import firewall (tested): nothing here imports torch, gymnasium, stable-baselines3,
sb3-contrib or tensorboard; ``onnxruntime`` is only loaded lazily through :mod:`sim.policy`.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
