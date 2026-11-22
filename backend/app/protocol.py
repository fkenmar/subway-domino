"""WebSocket protocol v1 (ARCHITECTURE §5.3 — CONTRACT).

Three parts, each the single implementation used by the server, the smoke test and the
golden-fixture tests (``tests/fixtures/protocol``; WP3's decoder tests the same files):

* **Client → server** JSON messages: strict pydantic v2 models (unknown fields, wrong types,
  non-finite numbers and unknown ``type`` values are rejected) behind one discriminated
  union, parsed by :func:`parse_client_message`.
* **Server → client** JSON messages: small typed builders returning plain dicts, serialized
  compactly by :func:`dumps` (``allow_nan=False``: a NaN never reaches the wire).
* **Binary train frames** (``kind = 1``, little-endian): :func:`encode_frame` writes
  ``24 + 9n + 2m`` bytes laid out exactly as the §5.3 table; :func:`decode_frame` is the
  strict inverse (exact length, valid enums).
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Final, Literal, TypeAlias, TypedDict, get_args

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, TypeAdapter, ValidationError

from sim import constants as C

PROTOCOL_VERSION: Final = 1

# --------------------------------------------------------------------------------------
# Enumerations (wire names; the binary frame encodes mode/speed as indices of these)
# --------------------------------------------------------------------------------------
Mode: TypeAlias = Literal["normal", "ai"]
Speed: TypeAlias = Literal["pause", "1x", "2x", "5x"]
Profile: TypeAlias = Literal["am_peak", "midday", "pm_rush", "late_night"]
DisruptionKind: TypeAlias = Literal["local_block", "express_block", "full_station_closure", "signal_delay"]
TargetType: TypeAlias = Literal["station", "complex", "segment", "interlocking"]
ErrorCode: TypeAlias = Literal["bad_request", "invalid_target", "ai_unavailable", "rate_limited"]
DisruptionEventKind: TypeAlias = Literal["disruption_started", "disruption_cleared", "disruption_expired"]

MODES: Final[tuple[Mode, ...]] = get_args(Mode)
SPEEDS: Final[tuple[Speed, ...]] = get_args(Speed)
PROFILES: Final[tuple[Profile, ...]] = get_args(Profile)
DISRUPTION_KINDS: Final[tuple[DisruptionKind, ...]] = get_args(DisruptionKind)
TARGET_TYPES: Final[tuple[TargetType, ...]] = get_args(TargetType)
MODE_CODE: Final[dict[str, int]] = {m: i for i, m in enumerate(MODES)}
SPEED_CODE: Final[dict[str, int]] = {s: i for i, s in enumerate(SPEEDS)}
DISPATCH_ACTIONS: Final[tuple[str, ...]] = tuple(C.ACTION_NAMES[1:])

# The protocol enumerations must agree with the simulator's single source of truth.
assert PROFILES == C.PROFILES, (PROFILES, C.PROFILES)
assert DISRUPTION_KINDS == C.DISRUPTION_KINDS, (DISRUPTION_KINDS, C.DISRUPTION_KINDS)
assert set(SPEEDS) == set(C.SPEEDS), (SPEEDS, C.SPEEDS)

AT_CAPACITY_MESSAGE: Final = "Subway Domino is at capacity right now — please try again in a minute."

# Limits on client input (the sim validates indices against the graph; these only bound
# the parser's work and keep values in their wire ranges).
MAX_INDEX: Final = 65535
MAX_DURATION_S: Final = 24 * 3600.0
MAX_RESTORE_DISRUPTIONS: Final = 64
MAX_TOKEN_CHARS: Final = 128
MAX_PRESET_ID_CHARS: Final = 64


# --------------------------------------------------------------------------------------
# Client -> server messages
# --------------------------------------------------------------------------------------
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


Index = Annotated[int, Field(ge=0, le=MAX_INDEX)]
DurationS = Annotated[FiniteFloat, Field(gt=0.0, le=MAX_DURATION_S)]


class Target(_Strict):
    type: TargetType
    idx: Index


class RestoreDisruption(_Strict):
    kind: DisruptionKind
    target: Target
    duration_s: DurationS | None = None


class Restore(_Strict):
    mode: Mode = "normal"
    speed: Speed = "1x"
    profile: Profile = "am_peak"
    disruptions: Annotated[tuple[RestoreDisruption, ...], Field(max_length=MAX_RESTORE_DISRUPTIONS)] = ()


class Hello(_Strict):
    type: Literal["hello"]
    protocol: int
    resume_token: Annotated[str, Field(max_length=MAX_TOKEN_CHARS)] | None = None
    restore: Restore | None = None


class SetMode(_Strict):
    type: Literal["set_mode"]
    mode: Mode


class SetSpeed(_Strict):
    type: Literal["set_speed"]
    speed: Speed


class SetProfile(_Strict):
    type: Literal["set_profile"]
    profile: Profile


class AddDisruption(_Strict):
    type: Literal["add_disruption"]
    kind: DisruptionKind
    target: Target
    duration_s: DurationS | None = None


class ClearDisruption(_Strict):
    type: Literal["clear_disruption"]
    id: Annotated[int, Field(ge=0)]


class ClearAll(_Strict):
    type: Literal["clear_all"]


class Preset(_Strict):
    type: Literal["preset"]
    id: Annotated[str, Field(min_length=1, max_length=MAX_PRESET_ID_CHARS)]


class Reset(_Strict):
    type: Literal["reset"]


class Activity(_Strict):
    type: Literal["activity"]


class Ping(_Strict):
    type: Literal["ping"]
    t: FiniteFloat


ClientMessage: TypeAlias = Annotated[
    Hello | SetMode | SetSpeed | SetProfile | AddDisruption | ClearDisruption | ClearAll | Preset | Reset
    | Activity | Ping,
    Field(discriminator="type"),
]
CLIENT_MESSAGE_TYPES: Final[tuple[str, ...]] = (
    "hello", "set_mode", "set_speed", "set_profile", "add_disruption", "clear_disruption", "clear_all",
    "preset", "reset", "activity", "ping",
)
_CLIENT_ADAPTER: Final = TypeAdapter(ClientMessage)


class ProtocolError(ValueError):
    """A client message violates the protocol; ``code`` is the §5.3 error code to send."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code: ErrorCode = code
        self.message = message


def parse_client_message(data: str | bytes, *, max_bytes: int | None = None) -> ClientMessage:
    """Validate one client JSON message; raises :class:`ProtocolError` (``bad_request``)."""
    if max_bytes is not None and len(data) > max_bytes:
        raise ProtocolError("bad_request", f"message too large ({len(data)} bytes, limit {max_bytes})")
    try:
        return _CLIENT_ADAPTER.validate_json(data)
    except ValidationError as exc:
        raise ProtocolError("bad_request", _describe(exc)) from None


def _describe(exc: ValidationError) -> str:
    err = exc.errors(include_url=False)[0]
    etype = err.get("type", "")
    if etype == "json_invalid":
        return "invalid JSON"
    if etype in ("union_tag_not_found", "dict_type", "model_type"):
        return "expected a JSON object with a string 'type'"
    if etype == "union_tag_invalid":
        tag = (err.get("ctx") or {}).get("tag", "?")
        return f"unknown message type {tag!r}"
    loc = ".".join(str(p) for p in err.get("loc", ()))
    return f"invalid {loc}: {err.get('msg', 'invalid value')}" if loc else str(err.get("msg", "invalid message"))


# --------------------------------------------------------------------------------------
# Server -> client JSON messages
# --------------------------------------------------------------------------------------
class SettingsDict(TypedDict):
    mode: Mode
    speed: Speed
    profile: Profile


class TargetDict(TypedDict):
    type: str
    idx: int


class DisruptionInfo(TypedDict):
    id: int
    kind: str
    target: TargetDict
    label: str
    start_t: float
    end_t: float | None


class MetricValues(TypedDict):
    avg_wait_min: float
    overcrowded_platforms: int
    tunnel_stuck_trains: int
    riders_served: int
    riders_waiting: int
    riders_onboard: int
    trains_active: int


class CalloutItem(TypedDict):
    kind: Literal["callout"]
    id: int
    t: float
    train_id: int
    route: str
    action: str
    station_idx: int
    text: str
    more: int  # dispatch actions aggregated into this callout (superset of §5.3; text says "+N more")


class DisruptionEvent(TypedDict):
    kind: DisruptionEventKind
    disruption: DisruptionInfo


EventItem: TypeAlias = CalloutItem | DisruptionEvent


def dumps(message: Mapping[str, Any]) -> str:
    """Compact JSON for the wire (UTF-8 text, no NaN/Infinity)."""
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _num(x: float) -> int | float:
    """Integral floats as ints (``10`` not ``10.0``) for the human-facing constants."""
    xf = float(x)
    return int(xf) if xf.is_integer() else xf


def _finite_or_none(x: float | None) -> float | None:
    if x is None:
        return None
    xf = float(x)
    return xf if math.isfinite(xf) else None


def welcome_message(
    *,
    session_id: str,
    resume_token: str,
    resumed: bool,
    graph_version: str,
    model_loaded: bool,
    ai_available: bool,
    settings: SettingsDict,
    idle_timeout_s: float,
    frame_hz: int,
) -> dict[str, Any]:
    return {
        "type": "welcome",
        "protocol": PROTOCOL_VERSION,
        "session_id": session_id,
        "resume_token": resume_token,
        "resumed": bool(resumed),
        "graph_version": graph_version,
        "model_loaded": bool(model_loaded),
        "ai_available": bool(ai_available),
        "settings": {"mode": settings["mode"], "speed": settings["speed"], "profile": settings["profile"]},
        "constants": {
            "dt_s": _num(C.DT_S),
            "frame_hz": int(frame_hz),
            "speeds": {name: _num(C.SPEEDS[name]) for name in SPEEDS},
            "idle_timeout_s": _num(idle_timeout_s),
        },
    }


def metric_values(m: Mapping[str, Any]) -> MetricValues:
    """The §3.8 metrics of one sim, rounded for the wire (riders as whole numbers)."""
    return {
        "avg_wait_min": round(float(m["avg_wait_min"]), 2),
        "overcrowded_platforms": int(m["overcrowded_platforms"]),
        "tunnel_stuck_trains": int(m["tunnel_stuck_trains"]),
        "riders_served": int(round(float(m["riders_served"]))),
        "riders_waiting": int(round(float(m["riders_waiting"]))),
        "riders_onboard": int(round(float(m["riders_onboard"]))),
        "trains_active": int(m["trains_active"]),
    }


def metrics_message(
    *,
    t: float,
    mode: Mode,
    effective_speed: float,
    live: MetricValues,
    twin: MetricValues | None,
    ai_since_t: float | None,
    actions: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "type": "metrics",
        "t": float(t),
        "mode": mode,
        "effective_speed": round(float(effective_speed), 3),
        "live": live,
        "twin": twin,
        "ai_since_t": _finite_or_none(ai_since_t),
        "actions": {name: int(actions.get(name, 0)) for name in DISPATCH_ACTIONS},
    }


def disruption_info(d: Mapping[str, Any]) -> DisruptionInfo:
    """Wire form of one ``NYCSubwaySim.disruptions()`` entry (``start_s/end_s`` → ``start_t/end_t``)."""
    tgt = d["target"]
    end = d.get("end_s")
    return {
        "id": int(d["id"]),
        "kind": str(d["kind"]),
        "target": {"type": str(tgt["type"]), "idx": int(tgt["idx"])},
        "label": str(d.get("label", "")),
        "start_t": float(d["start_s"]),
        "end_t": None if end is None else float(end),
    }


def callout_item(
    *, id: int, t: float, train_id: int, route: str, action: str, station_idx: int, text: str, more: int = 0
) -> CalloutItem:
    return {
        "kind": "callout",
        "id": int(id),
        "t": float(t),
        "train_id": int(train_id),
        "route": route,
        "action": action,
        "station_idx": int(station_idx),
        "text": text,
        "more": int(more),
    }


def disruption_event(kind: DisruptionEventKind, info: DisruptionInfo) -> DisruptionEvent:
    return {"kind": kind, "disruption": info}


def events_message(items: Sequence[EventItem]) -> dict[str, Any]:
    return {"type": "events", "items": list(items)}


def disruptions_message(items: Sequence[DisruptionInfo]) -> dict[str, Any]:
    return {"type": "disruptions", "items": list(items)}


def error_message(code: ErrorCode, message: str) -> dict[str, Any]:
    return {"type": "error", "code": code, "message": message}


def pong_message(t: float) -> dict[str, Any]:
    return {"type": "pong", "t": float(t)}


def idle_timeout_message() -> dict[str, Any]:
    return {"type": "idle_timeout"}


def at_capacity_message(message: str = AT_CAPACITY_MESSAGE) -> dict[str, Any]:
    return {"type": "at_capacity", "message": message}


# --------------------------------------------------------------------------------------
# Binary train frame (kind = 1)
# --------------------------------------------------------------------------------------
FRAME_MAGIC: Final = 0x5344
FRAME_VERSION: Final = 1
FRAME_KIND_TRAINS: Final = 1
FRAME_FLAG_TWIN_ACTIVE: Final = 1
#   magic u16 | version u8 | kind u8 | seq u32 | clock_s f64 | n u16 | m u16 | mode u8 | speed u8 | flags u16
FRAME_HEADER: Final = struct.Struct("<HBBIdHHBBH")
FRAME_HEADER_SIZE: Final = FRAME_HEADER.size
assert FRAME_HEADER_SIZE == 24

TRAIN_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("train_id", "<u2"),
    ("edge", "<u2"),
    ("frac", "<u2"),
    ("route", "u1"),
    ("status", "u1"),
    ("load", "u1"),
)
STATION_FIELDS: Final[tuple[tuple[str, str], ...]] = (("station_crowd", "u1"), ("station_wait", "u1"))
FRAME_ARRAYS: Final[tuple[str, ...]] = tuple(name for name, _ in TRAIN_FIELDS + STATION_FIELDS)


class FrameError(ValueError):
    """A binary frame is malformed (bad header, enum value or length)."""


def frame_size(n_trains: int, n_stations: int) -> int:
    """Exact byte length of a v1 frame: ``24 + 9n + 2m``."""
    return FRAME_HEADER_SIZE + 9 * int(n_trains) + 2 * int(n_stations)


def _wire_array(values: Any, dtype: str, name: str) -> np.ndarray:
    arr = np.asarray(values)
    want = np.dtype(dtype)
    if arr.ndim != 1:
        raise ValueError(f"frame array {name!r} must be 1-D, got shape {arr.shape}")
    if arr.dtype == want or (arr.dtype.kind == "u" and arr.dtype.itemsize == want.itemsize):
        return arr  # already in range; the byte order is fixed by the frombuffer view
    if arr.size and arr.dtype.kind in "iub":
        lo, hi = int(arr.min()), int(arr.max())
        if lo < 0 or hi > np.iinfo(want).max:
            raise ValueError(f"frame array {name!r} has values outside {want.name}: [{lo}, {hi}]")
        return arr
    if arr.size:
        raise ValueError(f"frame array {name!r} must hold integers, got dtype {arr.dtype}")
    return arr


def encode_frame(
    arrays: Mapping[str, Any],
    *,
    seq: int,
    clock_s: float,
    mode: Mode,
    speed: Speed,
    twin_active: bool,
) -> bytes:
    """Encode ``NYCSubwaySim.frame()`` arrays (§3.9) into one v1 binary frame (§5.3)."""
    n = int(np.asarray(arrays["train_id"]).shape[0])
    m = int(np.asarray(arrays["station_crowd"]).shape[0])
    if n > MAX_INDEX or m > MAX_INDEX:
        raise ValueError(f"frame too large: n={n}, m={m} (u16 counts)")
    if not math.isfinite(float(clock_s)):
        raise ValueError("clock_s must be finite")
    buf = bytearray(frame_size(n, m))
    FRAME_HEADER.pack_into(
        buf, 0, FRAME_MAGIC, FRAME_VERSION, FRAME_KIND_TRAINS, int(seq) & 0xFFFFFFFF, float(clock_s),
        n, m, MODE_CODE[mode], SPEED_CODE[speed], FRAME_FLAG_TWIN_ACTIVE if twin_active else 0,
    )
    offset = FRAME_HEADER_SIZE
    for fields, count in ((TRAIN_FIELDS, n), (STATION_FIELDS, m)):
        for name, dtype in fields:
            arr = _wire_array(arrays[name], dtype, name)
            if arr.shape[0] != count:
                raise ValueError(f"frame array {name!r} has length {arr.shape[0]}, expected {count}")
            view = np.frombuffer(buf, dtype=dtype, count=count, offset=offset)
            view[:] = arr  # writes through to ``buf`` with the wire dtype and byte order
            offset += view.nbytes
    assert offset == len(buf)
    return bytes(buf)


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    seq: int
    clock_s: float
    mode: Mode
    speed: Speed
    flags: int
    n_trains: int
    n_stations: int
    train_id: np.ndarray
    edge: np.ndarray
    frac: np.ndarray
    route: np.ndarray
    status: np.ndarray
    load: np.ndarray
    station_crowd: np.ndarray
    station_wait: np.ndarray

    @property
    def twin_active(self) -> bool:
        return bool(self.flags & FRAME_FLAG_TWIN_ACTIVE)

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in FRAME_ARRAYS}

    def to_json(self) -> dict[str, Any]:
        """Decoded values as plain JSON (the ``frame_v1.json`` golden-fixture layout)."""
        out: dict[str, Any] = {
            "seq": self.seq,
            "clock_s": self.clock_s,
            "mode": self.mode,
            "speed": self.speed,
            "flags": self.flags,
            "twin_active": self.twin_active,
            "n_trains": self.n_trains,
            "n_stations": self.n_stations,
        }
        for name in FRAME_ARRAYS:
            out[name] = [int(v) for v in getattr(self, name).tolist()]
        return out


def decode_frame(data: bytes | bytearray | memoryview) -> DecodedFrame:
    """Strict inverse of :func:`encode_frame`; raises :class:`FrameError`."""
    buf = bytes(data)
    if len(buf) < FRAME_HEADER_SIZE:
        raise FrameError(f"frame too short: {len(buf)} bytes")
    magic, version, kind, seq, clock_s, n, m, mode_code, speed_code, flags = FRAME_HEADER.unpack_from(buf, 0)
    if magic != FRAME_MAGIC:
        raise FrameError(f"bad magic 0x{magic:04x}")
    if version != FRAME_VERSION:
        raise FrameError(f"unsupported frame version {version}")
    if kind != FRAME_KIND_TRAINS:
        raise FrameError(f"unsupported frame kind {kind}")
    if mode_code >= len(MODES):
        raise FrameError(f"bad mode code {mode_code}")
    if speed_code >= len(SPEEDS):
        raise FrameError(f"bad speed code {speed_code}")
    if len(buf) != frame_size(n, m):
        raise FrameError(f"frame length {len(buf)} != {frame_size(n, m)} for n={n}, m={m}")
    values: dict[str, np.ndarray] = {}
    offset = FRAME_HEADER_SIZE
    for fields, count in ((TRAIN_FIELDS, n), (STATION_FIELDS, m)):
        for name, dtype in fields:
            arr = np.frombuffer(buf, dtype=dtype, count=count, offset=offset)
            values[name] = arr.astype(np.dtype(dtype).newbyteorder("="))
            offset += arr.nbytes
    return DecodedFrame(
        seq=int(seq),
        clock_s=float(clock_s),
        mode=MODES[mode_code],
        speed=SPEEDS[speed_code],
        flags=int(flags),
        n_trains=int(n),
        n_stations=int(m),
        **values,
    )
