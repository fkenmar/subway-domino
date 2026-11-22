"""Server configuration (ARCHITECTURE §5.1, §5.2, §7).

All runtime knobs come from environment variables and are parsed once into an immutable
:class:`Settings`. Tests build ``Settings(...)`` directly (or ``Settings.from_env(mapping)``)
and inject it into :func:`app.main.create_app`.

==========================  =====================================  ==========================
Variable                    Meaning                                Default
==========================  =====================================  ==========================
``ALLOWED_ORIGINS``         comma-separated CORS + WebSocket       ``https://kenmar.dev,``
                            Origin allow-list (``*`` = any)        ``http://localhost:3000``
``MAX_SESSIONS``            concurrent attached sessions           40
``IDLE_TIMEOUT_S``          close after this long w/o activity     300
``DETACH_GRACE_S``          keep a dropped session resumable       90
``ALLOW_NO_ORIGIN``         accept sockets without ``Origin``      0 (tests and smoke: 1)
``SUBWAY_DATA_DIR``         ``network.json`` + geo file            ``backend/data``
``SUBWAY_MODEL_PATH``       the shipped ONNX policy                ``backend/models/...onnx``
``CPU_BUDGET``              governor target, CPU-s per wall-s      0.70
``LOG_LEVEL``               ``app.*`` log level                    ``INFO``
==========================  =====================================  ==========================

Beyond the contract (safety limits, all optional): ``MAX_DETACHED_SESSIONS`` (default
``MAX_SESSIONS``; bounds the memory held by dropped-but-resumable sessions),
``MAX_DISRUPTIONS`` (32 active per session), ``HELLO_TIMEOUT_S`` (10) and
``MAX_MESSAGE_BYTES`` (16 KiB per client message).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from sim.graph import BACKEND_DIR

# --------------------------------------------------------------------------------------
# §5.2 session-loop constants (CONTRACT)
# --------------------------------------------------------------------------------------
FRAME_HZ: int = 4
METRICS_HZ: int = 1
MAX_TICKS_PER_FRAME: int = 8
CALLOUT_MIN_INTERVAL_S: float = 1.5

DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = ("https://kenmar.dev", "http://localhost:3000")
DEFAULT_MAX_SESSIONS: int = 40
DEFAULT_IDLE_TIMEOUT_S: float = 300.0
DEFAULT_DETACH_GRACE_S: float = 90.0
DEFAULT_CPU_BUDGET: float = 0.70
DEFAULT_DATA_DIR: Path = BACKEND_DIR / "data"
DEFAULT_MODEL_PATH: Path = BACKEND_DIR / "models" / "nyc_subway_rcc_ppo.onnx"

_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


class ConfigError(ValueError):
    """An environment variable has an invalid value (the message names it)."""


def normalize_origin(origin: str) -> str:
    """Canonical ``scheme://host[:port]`` (lower-case, default port and trailing slash dropped).

    Browsers already send origins in this form; normalizing the configured allow-list makes
    ``https://kenmar.dev/`` or ``HTTPS://Kenmar.dev:443`` in an env var behave as intended.
    """
    raw = origin.strip()
    if raw == "*":
        return raw
    parts = urlsplit(raw)
    if not parts.scheme or not parts.hostname:
        return raw.rstrip("/").lower()
    scheme = parts.scheme.lower()
    host = parts.hostname.lower()
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        return raw.rstrip("/").lower()
    if port is not None and port != _DEFAULT_PORTS.get(scheme):
        return f"{scheme}://{host}:{port}"
    return f"{scheme}://{host}"


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable server settings; see the module docstring for the environment mapping."""

    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS
    max_sessions: int = DEFAULT_MAX_SESSIONS
    idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S
    detach_grace_s: float = DEFAULT_DETACH_GRACE_S
    allow_no_origin: bool = False
    data_dir: Path = DEFAULT_DATA_DIR
    model_path: Path = DEFAULT_MODEL_PATH
    cpu_budget: float = DEFAULT_CPU_BUDGET
    log_level: str = "INFO"
    # safety limits beyond the contract
    max_detached_sessions: int | None = None  # None: same as max_sessions
    max_disruptions: int = 32
    hello_timeout_s: float = 10.0
    max_message_bytes: int = 16 * 1024
    _origin_set: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.allowed_origins:
            raise ConfigError("ALLOWED_ORIGINS must list at least one origin")
        if self.max_sessions < 1:
            raise ConfigError("MAX_SESSIONS must be >= 1")
        if self.max_detached_sessions is not None and self.max_detached_sessions < 0:
            raise ConfigError("MAX_DETACHED_SESSIONS must be >= 0")
        for name, value in (("IDLE_TIMEOUT_S", self.idle_timeout_s), ("DETACH_GRACE_S", self.detach_grace_s),
                            ("HELLO_TIMEOUT_S", self.hello_timeout_s)):
            if not (math.isfinite(value) and value > 0):
                raise ConfigError(f"{name} must be a positive number of seconds, got {value!r}")
        if not (math.isfinite(self.cpu_budget) and 0.0 < self.cpu_budget <= 1.0):
            raise ConfigError(f"CPU_BUDGET must be in (0, 1] (fraction of one core), got {self.cpu_budget!r}")
        if self.max_disruptions < 1:
            raise ConfigError("MAX_DISRUPTIONS must be >= 1")
        if self.max_message_bytes < 256:
            raise ConfigError("MAX_MESSAGE_BYTES must be >= 256")
        level = self.log_level.upper()
        if level not in logging.getLevelNamesMapping():
            raise ConfigError(f"LOG_LEVEL must be a logging level name, got {self.log_level!r}")
        object.__setattr__(self, "log_level", level)
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "model_path", Path(self.model_path))
        origins = tuple(normalize_origin(o) for o in self.allowed_origins if o.strip())
        if not origins:
            raise ConfigError("ALLOWED_ORIGINS must list at least one origin")
        object.__setattr__(self, "allowed_origins", origins)
        object.__setattr__(self, "_origin_set", frozenset(origins))

    # ------------------------------------------------------------------ derived values
    @property
    def network_path(self) -> Path:
        return self.data_dir / "network.json"

    @property
    def geo_path(self) -> Path:
        return self.data_dir / "network_geo.json"

    @property
    def detached_cap(self) -> int:
        return self.max_sessions if self.max_detached_sessions is None else self.max_detached_sessions

    @property
    def cors_origins(self) -> list[str]:
        return ["*"] if "*" in self._origin_set else list(self.allowed_origins)

    def origin_allowed(self, origin: str | None) -> bool:
        """WebSocket ``Origin`` check (§5.1): exact allow-list match; missing only if allowed."""
        if origin is None or origin == "":
            return self.allow_no_origin
        if "*" in self._origin_set:
            return True
        return normalize_origin(origin) in self._origin_set

    # ------------------------------------------------------------------ environment
    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Parse settings from ``env`` (default: ``os.environ``); raises :class:`ConfigError`."""
        if env is None:
            import os

            env = os.environ
        origins_raw = env.get("ALLOWED_ORIGINS")
        origins = (
            tuple(o.strip() for o in origins_raw.split(",") if o.strip())
            if origins_raw is not None
            else DEFAULT_ALLOWED_ORIGINS
        )
        max_detached_raw = env.get("MAX_DETACHED_SESSIONS")
        return cls(
            allowed_origins=origins,
            max_sessions=_int(env, "MAX_SESSIONS", DEFAULT_MAX_SESSIONS),
            idle_timeout_s=_float(env, "IDLE_TIMEOUT_S", DEFAULT_IDLE_TIMEOUT_S),
            detach_grace_s=_float(env, "DETACH_GRACE_S", DEFAULT_DETACH_GRACE_S),
            allow_no_origin=_bool(env, "ALLOW_NO_ORIGIN", False),
            data_dir=Path(env.get("SUBWAY_DATA_DIR") or DEFAULT_DATA_DIR),
            model_path=Path(env.get("SUBWAY_MODEL_PATH") or DEFAULT_MODEL_PATH),
            cpu_budget=_float(env, "CPU_BUDGET", DEFAULT_CPU_BUDGET),
            log_level=(env.get("LOG_LEVEL") or "INFO").strip(),
            max_detached_sessions=None if not max_detached_raw else _int(env, "MAX_DETACHED_SESSIONS", 0),
            max_disruptions=_int(env, "MAX_DISRUPTIONS", 32),
            hello_timeout_s=_float(env, "HELLO_TIMEOUT_S", 10.0),
            max_message_bytes=_int(env, "MAX_MESSAGE_BYTES", 16 * 1024),
        )


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from None


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None
    if not math.isfinite(value):
        raise ConfigError(f"{name} must be finite, got {raw!r}")
    return value


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ConfigError(f"{name} must be a boolean (1/0, true/false), got {raw!r}")
