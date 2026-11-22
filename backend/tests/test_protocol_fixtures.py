"""Golden wire fixtures for protocol v1 (ARCHITECTURE §5.3, §8; PRD Epic 5/6).

``tests/fixtures/protocol/`` holds:

* ``frame_v1.bin`` — one binary train frame encoded by :func:`app.protocol.encode_frame`
  from the deterministic state in :func:`golden_frame_state`;
* ``frame_v1.json`` — its decoded values;
* ``messages_v1.json`` — one example of every client → server and server → client JSON
  message, built by the :mod:`app.protocol` builders and accepted by its parser.

WP3's vitest decodes the same files, so both ends of the wire test against one source.
The state is hand-built (not taken from a simulation) so the fixture pins the *encoding*
and does not drift with simulator physics; it exercises the value ranges that matter
(u32 sequence above 2^31, u16 values across the byte boundary and at 0xFFFF, every status
bit, a closed station = 255).

Regenerate after a deliberate protocol change (and update ARCHITECTURE §5.3 and WP3)::

    .venv/bin/python tests/test_protocol_fixtures.py --write
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:  # allow ``python tests/test_protocol_fixtures.py --write``
    sys.path.insert(0, str(BACKEND))

from app import protocol as P  # noqa: E402
from sim import constants as C  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "protocol"
FRAME_BIN = FIXTURES / "frame_v1.bin"
FRAME_JSON = FIXTURES / "frame_v1.json"
MESSAGES_JSON = FIXTURES / "messages_v1.json"


# --------------------------------------------------------------------------------------
# Deterministic inputs
# --------------------------------------------------------------------------------------
def golden_frame_state() -> dict[str, Any]:
    """Header fields and §3.9 wire arrays of the golden frame (6 trains, 13 stations)."""
    arrays = {
        "train_id": np.array([0, 3, 17, 255, 256, 699], dtype=np.uint16),
        "edge": np.array([0, 1, 0x1234, 1913, 300, 65535], dtype=np.uint16),
        "frac": np.array([0, 65535, 32768, 1, 0x00FF, 0xFF00], dtype=np.uint16),
        "route": np.array([0, 1, 27, 9, 14, 255], dtype=np.uint8),
        "status": np.array([0, C.ST_AT_PLATFORM, C.ST_TUNNEL_STUCK | C.ST_SHORT_TURNING,
                            C.ST_HELD | C.ST_AT_PLATFORM, C.ST_TRACK_SWITCH | C.ST_REROUTED | C.ST_GAP_TRAIN, 255],
                           dtype=np.uint8),
        "load": np.array([0, 50, 100, 120, 254, 255], dtype=np.uint8),
        "station_crowd": np.array([0, 10, 79, 80, 100, 150, 254, 255, 0, 33, 66, 99, 1], dtype=np.uint8),
        "station_wait": np.array([0, 4, 8, 20, 40, 60, 120, 255, 1, 2, 3, 250, 0], dtype=np.uint8),
    }
    return {"seq": 4_000_000_123, "clock_s": 30123.25, "mode": "ai", "speed": "5x", "twin_active": True,
            "arrays": arrays}


def build_frame() -> bytes:
    s = golden_frame_state()
    return P.encode_frame(s["arrays"], seq=s["seq"], clock_s=s["clock_s"], mode=s["mode"], speed=s["speed"],
                          twin_active=s["twin_active"])


def build_frame_json(data: bytes) -> dict[str, Any]:
    decoded = P.decode_frame(data).to_json()
    return {
        "description": "Decoded values of frame_v1.bin (ARCHITECTURE §5.3 binary frame, kind = 1, little-endian).",
        "size_bytes": len(data),
        **decoded,
    }


_DISRUPTION_A = {"id": 3, "kind": "local_block", "target": {"type": "segment", "idx": 12},
                 "label": "Local track blocked: 59 St ↔ 51 St", "start_s": 29000.0, "end_s": 30800.0}
_DISRUPTION_B = {"id": 4, "kind": "full_station_closure", "target": {"type": "station", "idx": 148},
                 "label": "Station closed: 14 St-Union Sq", "start_s": 29050.0, "end_s": None}
_DISRUPTION_C = {"id": 2, "kind": "signal_delay", "target": {"type": "interlocking", "idx": 7},
                 "label": "Signal problems at Times Sq-42 St", "start_s": 28800.0, "end_s": 29400.0}
_DISRUPTION_D = {"id": 1, "kind": "express_block", "target": {"type": "segment", "idx": 342},
                 "label": "Express track blocked: Jackson Hts-Roosevelt Av ↔ 65 St", "start_s": 28800.0,
                 "end_s": 29900.0}
_LIVE = {"avg_wait_min": 4.2, "overcrowded_platforms": 3, "tunnel_stuck_trains": 2, "riders_served": 123456.4,
         "riders_waiting": 23000.2, "riders_onboard": 81234.6, "trains_active": 498}
_TWIN = {"avg_wait_min": 6.85, "overcrowded_platforms": 9, "tunnel_stuck_trains": 14, "riders_served": 119876.0,
         "riders_waiting": 31877.0, "riders_onboard": 79001.0, "trains_active": 497}
_ACTIONS = {"HOLD_IN_STATION": 12, "RUN_EXPRESS": 3, "RUN_LOCAL": 1, "SKIP_STOP": 4, "SHORT_TURN_RELAY": 2,
            "REROUTE_VIA_COMPATIBLE_TRUNK": 1, "DISPATCH_YARD_GAP_TRAIN": 1}


def build_messages() -> dict[str, Any]:
    """One example of every protocol v1 JSON message, keyed by a descriptive name."""
    client: dict[str, Any] = {
        "hello_fresh": {"type": "hello", "protocol": 1, "resume_token": None, "restore": None},
        "hello_resume": {
            "type": "hello", "protocol": 1, "resume_token": "q2Zb7ZK1y3n8m0Yw2b9tq0p4Rr7Zc1Qe5Xv8Lw3Ja6M",
            "restore": {"mode": "ai", "speed": "2x", "profile": "am_peak", "disruptions": [
                {"kind": "local_block", "target": {"type": "segment", "idx": 12}, "duration_s": None},
                {"kind": "full_station_closure", "target": {"type": "station", "idx": 148}, "duration_s": 1200},
            ]},
        },
        "set_mode": {"type": "set_mode", "mode": "ai"},
        "set_speed": {"type": "set_speed", "speed": "5x"},
        "set_profile": {"type": "set_profile", "profile": "pm_rush"},
        "add_disruption": {"type": "add_disruption", "kind": "express_block",
                           "target": {"type": "segment", "idx": 342}, "duration_s": 1800},
        "add_disruption_until_cleared": {"type": "add_disruption", "kind": "signal_delay",
                                         "target": {"type": "interlocking", "idx": 7}, "duration_s": None},
        "clear_disruption": {"type": "clear_disruption", "id": 3},
        "clear_all": {"type": "clear_all"},
        "preset": {"type": "preset", "id": "lex-union-sq"},
        "reset": {"type": "reset"},
        "activity": {"type": "activity"},
        "ping": {"type": "ping", "t": 1234.5},
    }
    welcome = P.welcome_message(
        session_id="5f0c3a9e8b7d4c2a9e1f6b3d7a8c9e0f", resume_token="Jr0kQe2b-5n8V1Yw2b9tq0p4Rr7Zc1Qe5Xv8Lw3Ja6M",
        resumed=False, graph_version="e184af352bc6", model_loaded=True, ai_available=True,
        settings={"mode": "normal", "speed": "1x", "profile": "am_peak"}, idle_timeout_s=300.0, frame_hz=4,
    )
    metrics_ai = P.metrics_message(
        t=30120.0, mode="ai", effective_speed=1.0, live=P.metric_values(_LIVE), twin=P.metric_values(_TWIN),
        ai_since_t=29400.0, actions=_ACTIONS,
    )
    metrics_normal = P.metrics_message(
        t=29010.0, mode="normal", effective_speed=0.85, live=P.metric_values(_TWIN), twin=None, ai_since_t=None,
        actions={name: 0 for name in P.DISPATCH_ACTIONS},
    )
    events = P.events_message([
        P.callout_item(
            id=17, t=30110.0, train_id=12, route="4", action="HOLD_IN_STATION", station_idx=123,
            text=("Holding a 4 train at 59 St with doors open — the local track ahead is blocked at "
                  "Grand Central-42 St, so its 840 riders wait on a platform instead of in a stalled tunnel."),
        ),
        P.callout_item(
            id=18, t=30125.0, train_id=231, route="F", action="SHORT_TURN_RELAY", station_idx=301,
            text=("Short-turning an F train at Jackson Hts-Roosevelt Av — the express track ahead is blocked at "
                  "65 St. 410 riders step off to wait for the next train, and it reverses after about 4 min for "
                  "the crew to change ends and test the brakes. (+3 more dispatch actions)"),
            more=3,
        ),
        P.disruption_event("disruption_started", P.disruption_info(_DISRUPTION_A)),
        P.disruption_event("disruption_cleared", P.disruption_info(_DISRUPTION_C)),
        P.disruption_event("disruption_expired", P.disruption_info(_DISRUPTION_D)),
    ])
    server: dict[str, Any] = {
        "welcome": welcome,
        "metrics_ai": metrics_ai,
        "metrics_normal": metrics_normal,
        "events": events,
        "disruptions": P.disruptions_message([P.disruption_info(_DISRUPTION_A), P.disruption_info(_DISRUPTION_B)]),
        "disruptions_empty": P.disruptions_message([]),
        "error_bad_request": P.error_message("bad_request", "unknown message type 'warp'"),
        "error_invalid_target": P.error_message("invalid_target", "segment 3 has no express track."),
        "error_ai_unavailable": P.error_message("ai_unavailable",
                                                "The AI Dispatcher is unavailable right now (no model loaded)."),
        "error_rate_limited": P.error_message("rate_limited", "Too many messages — slow down."),
        "pong": P.pong_message(1234.5),
        "idle_timeout": P.idle_timeout_message(),
        "at_capacity": P.at_capacity_message(),
    }
    return {
        "description": ("Subway Domino WebSocket protocol v1 (ARCHITECTURE §5.3): one example of every JSON "
                        "message, generated by app.protocol. Binary frames: frame_v1.bin / frame_v1.json."),
        "protocol": P.PROTOCOL_VERSION,
        "client": client,
        "server": server,
    }


def _json_text(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def write_fixtures() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    data = build_frame()
    FRAME_BIN.write_bytes(data)
    FRAME_JSON.write_text(_json_text(build_frame_json(data)), encoding="utf-8")
    MESSAGES_JSON.write_text(_json_text(build_messages()), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------
def _require(path: Path) -> None:
    if not path.is_file():
        pytest.fail(f"{path.relative_to(BACKEND)} missing: run .venv/bin/python tests/test_protocol_fixtures.py --write")


def test_frame_bin_is_reproduced_byte_for_byte() -> None:
    _require(FRAME_BIN)
    assert build_frame() == FRAME_BIN.read_bytes()


def test_frame_json_matches_the_decoded_bin() -> None:
    _require(FRAME_BIN)
    _require(FRAME_JSON)
    data = FRAME_BIN.read_bytes()
    assert json.loads(FRAME_JSON.read_text(encoding="utf-8")) == build_frame_json(data)
    assert FRAME_JSON.read_text(encoding="utf-8") == _json_text(build_frame_json(data))


def test_frame_layout_follows_the_section_5_3_table() -> None:
    """Read the golden bytes at the documented offsets, independently of the encoder."""
    _require(FRAME_BIN)
    data = FRAME_BIN.read_bytes()
    s = golden_frame_state()
    a = s["arrays"]
    n, m = len(a["train_id"]), len(a["station_crowd"])
    assert len(data) == 24 + 9 * n + 2 * m
    assert struct.unpack_from("<H", data, 0)[0] == 0x5344
    assert data[2] == 1 and data[3] == 1  # version, kind
    assert struct.unpack_from("<I", data, 4)[0] == s["seq"]
    assert struct.unpack_from("<d", data, 8)[0] == s["clock_s"]
    assert struct.unpack_from("<HH", data, 16) == (n, m)
    assert data[20] == 1 and data[21] == 3  # mode ai, speed 5x
    assert struct.unpack_from("<H", data, 22)[0] == 1  # twin active
    o = 24
    assert list(struct.unpack_from(f"<{n}H", data, o)) == a["train_id"].tolist()
    assert list(struct.unpack_from(f"<{n}H", data, o + 2 * n)) == a["edge"].tolist()
    assert list(struct.unpack_from(f"<{n}H", data, o + 4 * n)) == a["frac"].tolist()
    assert list(data[o + 6 * n: o + 7 * n]) == a["route"].tolist()
    assert list(data[o + 7 * n: o + 8 * n]) == a["status"].tolist()
    assert list(data[o + 8 * n: o + 9 * n]) == a["load"].tolist()
    assert list(data[o + 9 * n: o + 9 * n + m]) == a["station_crowd"].tolist()
    assert list(data[o + 9 * n + m:]) == a["station_wait"].tolist()
    # little-endian on the wire: 0x1234 is stored as 34 12
    k = a["edge"].tolist().index(0x1234)
    assert data[o + 2 * n + 2 * k: o + 2 * n + 2 * k + 2] == b"\x34\x12"


def test_messages_json_is_reproduced_byte_for_byte() -> None:
    _require(MESSAGES_JSON)
    assert MESSAGES_JSON.read_text(encoding="utf-8") == _json_text(build_messages())


def test_messages_cover_every_message_type() -> None:
    msgs = build_messages()
    client_types = {m["type"] for m in msgs["client"].values()}
    server_types = {m["type"] for m in msgs["server"].values()}
    assert client_types == set(P.CLIENT_MESSAGE_TYPES)
    assert server_types == {"welcome", "metrics", "events", "disruptions", "error", "pong", "idle_timeout",
                            "at_capacity"}
    errors = {m["code"] for m in msgs["server"].values() if m["type"] == "error"}
    assert errors == {"bad_request", "invalid_target", "ai_unavailable", "rate_limited"}
    event_kinds = {i["kind"] for i in msgs["server"]["events"]["items"]}
    assert event_kinds == {"callout", "disruption_started", "disruption_cleared", "disruption_expired"}


@pytest.mark.parametrize("name", list(build_messages()["client"]))
def test_every_client_example_is_accepted_by_the_server_parser(name: str) -> None:
    example = build_messages()["client"][name]
    parsed = P.parse_client_message(json.dumps(example))
    assert parsed.type == example["type"]
    # the strict model round-trips the example exactly
    assert parsed.model_dump(mode="json", exclude_unset=True) == example


def test_server_examples_match_the_contract_shapes() -> None:
    s = build_messages()["server"]
    w = s["welcome"]
    assert set(w) == {"type", "protocol", "session_id", "resume_token", "resumed", "graph_version", "model_loaded",
                      "ai_available", "settings", "constants"}
    assert w["constants"] == {"dt_s": 10, "frame_hz": 4, "speeds": {"pause": 0, "1x": 60, "2x": 120, "5x": 300},
                              "idle_timeout_s": 300}
    for key in ("metrics_ai", "metrics_normal"):
        m = s[key]
        assert set(m) == {"type", "t", "mode", "effective_speed", "live", "twin", "ai_since_t", "actions"}
        assert set(m["actions"]) == set(C.ACTION_NAMES[1:])
        assert set(m["live"]) >= {"avg_wait_min", "overcrowded_platforms", "tunnel_stuck_trains", "riders_served",
                                  "riders_waiting", "trains_active"}
    assert s["metrics_normal"]["twin"] is None and s["metrics_normal"]["ai_since_t"] is None
    for item in s["disruptions"]["items"]:
        assert set(item) == {"id", "kind", "target", "label", "start_t", "end_t"}
    for item in s["events"]["items"]:
        if item["kind"] == "callout":
            assert set(item) == {"kind", "id", "t", "train_id", "route", "action", "station_idx", "text", "more"}
            assert item["action"] in C.ACTION_NAMES[1:]
    assert s["at_capacity"]["message"] == "Subway Domino is at capacity right now — please try again in a minute."
    # every example is valid compact wire JSON (no NaN / Infinity)
    for msg in s.values():
        assert json.loads(P.dumps(msg)) == msg


if __name__ == "__main__":
    if "--write" not in sys.argv[1:]:
        print(__doc__)
        sys.exit(2)
    write_fixtures()
    print(f"wrote {FRAME_BIN.name}, {FRAME_JSON.name}, {MESSAGES_JSON.name} in {FIXTURES}")
