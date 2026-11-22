"""Build, validate, serialize and compare the WP1 artifacts (shared by the CLI and the tests).

``build_artifacts`` runs the whole pipeline in memory: sources (network/cache/fallback ladder)
-> :func:`scripts.netbuild.pipeline.build_model` -> ``network.json`` / ``network_geo.json``
documents -> §2.6/§2.7 invariants. Nothing is written except the ``data/raw/`` download cache
(by :mod:`scripts.netbuild.sources`); writing the artifacts is the caller's decision, so
``--check`` can verify without touching anything.

The two artifacts depend only on the input bytes: same inputs give byte-identical files. Only
``build_manifest.json`` carries timestamps, timings and host details.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from scripts.netbuild import constants as K
from scripts.netbuild.assemble import build_geo_doc, build_network_doc
from scripts.netbuild.http import HttpClient
from scripts.netbuild.infra import load_infrastructure
from scripts.netbuild.jsonio import dumps_artifact, sha1_hex
from scripts.netbuild.maneuvers import preset_maneuver_summary
from scripts.netbuild.pipeline import NetworkModel, build_model
from scripts.netbuild.service import routes_without_service
from scripts.netbuild.sources import Sources, fetch_sources
from scripts.netbuild.validate import check_geo, check_network, gzip_size

Log = Callable[[str], None]


@dataclass
class BuildOutput:
    """Everything one build produced, in memory."""

    model: NetworkModel
    sources: Sources
    network: dict[str, Any]
    geo: dict[str, Any]
    network_bytes: bytes
    geo_bytes: bytes
    errors: list[str]
    warnings: list[str]
    routes_without_am_service: list[str]
    timings_s: dict[str, float] = field(default_factory=dict)

    @property
    def graph_version(self) -> str:
        return str(self.network["graph_version"])


def serialize(doc: dict[str, Any]) -> bytes:
    """The committed byte form of an artifact (deterministic, diff-friendly compact JSON)."""
    return dumps_artifact(doc).encode("utf-8")


def validate_documents(network: dict[str, Any], geo: dict[str, Any], network_bytes: bytes, geo_bytes: bytes,
                       routes_without_am_service: list[str]) -> tuple[list[str], list[str]]:
    """All §2.6/§2.7 invariants of a pair of artifacts: ``(errors, warnings)``.

    A document too malformed to inspect (missing keys, wrong types) is reported as an error
    instead of raising.
    """
    try:
        errors, warnings = check_network(network, routes_without_am_service=routes_without_am_service,
                                         size_bytes=len(network_bytes))
        g_errors, g_warnings = check_geo(geo, network, size_bytes=len(geo_bytes),
                                         gzip_bytes=gzip_size(geo_bytes))
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        return [f"malformed artifact ({type(exc).__name__}: {exc})"], []
    return errors + g_errors, warnings + g_warnings


def build_artifacts(data_dir: Path, client: HttpClient, log: Log = lambda _m: None) -> BuildOutput:
    """Fetch the inputs (with fallbacks), build the model and the two documents, and validate them.

    Raises :class:`scripts.netbuild.sources.SourceError` when no GTFS feed is available and
    :class:`scripts.netbuild.infra.InfrastructureError` / ``BuildError`` / ``TopologyError`` when the
    curated overlay or the inferred network is inconsistent.
    """
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    infra = load_infrastructure(data_dir / K.INFRASTRUCTURE_FILE)
    log("fetching sources" + (" (offline: caches only)" if client.offline else ""))
    sources = fetch_sources(data_dir / "raw", client)
    timings["sources"] = round(time.perf_counter() - t0, 3)
    for name, rec in sources.records.items():
        log(f"  {name}: {rec.origin} ({rec.bytes} bytes)")
    for msg in sources.fallbacks:
        log(f"  fallback: {msg}")

    t1 = time.perf_counter()
    log("building the network model")
    model = build_model(sources, infra, log=log)
    timings["model"] = round(time.perf_counter() - t1, 3)

    t2 = time.perf_counter()
    network = build_network_doc(model)
    geo = build_geo_doc(model, network)
    network_bytes, geo_bytes = serialize(network), serialize(geo)
    timings["assemble"] = round(time.perf_counter() - t2, 3)

    t3 = time.perf_counter()
    no_am = routes_without_service(model.kept, "am_peak")
    errors, warnings = validate_documents(network, geo, network_bytes, geo_bytes, no_am)
    timings["validate"] = round(time.perf_counter() - t3, 3)
    timings["total"] = round(time.perf_counter() - t0, 3)
    return BuildOutput(model, sources, network, geo, network_bytes, geo_bytes, errors, warnings, no_am, timings)


def artifact_counts(network: dict[str, Any]) -> dict[str, Any]:
    """Headline counts of a ``network.json`` document (reported in the manifest and the CLI)."""
    kinds: dict[str, int] = {}
    for e in network["edges"]:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    il_kinds: dict[str, int] = {}
    for il in network["interlockings"]:
        for k in il["kinds"]:
            il_kinds[k] = il_kinds.get(k, 0) + 1
    nodes = network["nodes"]
    pocket_stations = {nodes[e["from"]]["station"] for e in network["edges"] if e["kind"] == "turnback" and e["pocket"]}
    return {
        "routes": len(network["routes"]),
        "stations": len(network["stations"]),
        "complexes": len(network["complexes"]),
        "platforms": len(network["platforms"]),
        "nodes": len(nodes),
        "edges": len(network["edges"]),
        "edges_by_kind": dict(sorted(kinds.items())),
        "segments": len(network["segments"]),
        "segments_with_express": sum(1 for s in network["segments"] if s["has_express"]),
        "river_segments": sum(1 for s in network["segments"] if s["river"]),
        "interlockings": len(network["interlockings"]),
        "interlockings_by_kind": dict(sorted(il_kinds.items())),
        "pocket_turnback_stations": len(pocket_stations),
        "yards": len(network["yards"]),
        "patterns": len(network["patterns"]),
        "presets": len(network["presets"]),
        "hubs": sum(1 for s in network["stations"] if s["hub"]),
        "service_rows": {p: len(rows) for p, rows in network["service"].items()},
        "system_entries_h": {p: round(sum(d["platform_rate"]) * 3600.0, 1) for p, d in network["demand"].items()},
    }


def manifest_doc(out: BuildOutput, *, offline: bool, built_utc: dt.datetime | None = None) -> dict[str, Any]:
    """``build_manifest.json``: provenance, fallbacks, counts, validation, sizes and timings.

    This is the only artifact with timestamps and host details; it is excluded from the
    byte-identical rebuild guarantee.
    """
    when = (built_utc or dt.datetime.now(dt.timezone.utc)).replace(microsecond=0)
    m = out.model
    ridership = out.sources.ridership
    return {
        "schema_version": K.MANIFEST_SCHEMA_VERSION,
        "built_utc": when.isoformat().replace("+00:00", "Z"),
        "graph_version": out.graph_version,
        "mode": {"offline": offline},
        "toolchain": {"python": platform.python_version(), "numpy": np.__version__},
        "sources": {name: rec.to_json() for name, rec in out.sources.records.items()},
        "fallbacks_used": list(out.sources.fallbacks),
        "gtfs_feed_info": dict(m.feed.feed_info),
        "ridership_window": None if ridership is None else {"since_ts": ridership.since_ts,
                                                            "max_ts": ridership.max_ts,
                                                            "rows": len(ridership.rows)},
        "routes_without_am_peak_service": list(out.routes_without_am_service),
        "counts": artifact_counts(out.network),
        "preset_maneuvers": preset_maneuver_summary(out.network),
        "stats": m.stats,
        "notes": list(m.notes),
        "validation": {"errors": list(out.errors), "warnings": list(out.warnings)},
        "artifacts": {
            K.NETWORK_FILE: {"bytes": len(out.network_bytes), "sha1": sha1_hex(out.network_bytes)},
            K.GEO_FILE: {"bytes": len(out.geo_bytes), "sha1": sha1_hex(out.geo_bytes),
                         "gzip_bytes": gzip_size(out.geo_bytes)},
        },
        "timings_s": {**out.timings_s, **{f"model.{k}": v for k, v in m.timings_s.items()}},
    }


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    """The manifest is a human-read provenance record: pretty-printed, keys in insertion order."""
    return (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def load_routes_without_am_service(data_dir: Path) -> list[str]:
    """The routes the last build recorded as having no weekday AM service (from the manifest)."""
    path = data_dir / K.MANIFEST_FILE
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    value = doc.get("routes_without_am_peak_service", [])
    return [str(r) for r in value] if isinstance(value, list) else []


def validate_committed(data_dir: Path) -> tuple[list[str], list[str]]:
    """Validate the committed ``network.json`` / ``network_geo.json`` in ``data_dir`` as they are."""
    net_path, geo_path = data_dir / K.NETWORK_FILE, data_dir / K.GEO_FILE
    missing = [p.name for p in (net_path, geo_path) if not p.is_file()]
    if missing:
        return [f"missing artifact(s): {', '.join(missing)}"], []
    network_bytes, geo_bytes = net_path.read_bytes(), geo_path.read_bytes()
    try:
        network = json.loads(network_bytes.decode("utf-8"))
        geo = json.loads(geo_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"artifact is not valid JSON: {exc}"], []
    return validate_documents(network, geo, network_bytes, geo_bytes, load_routes_without_am_service(data_dir))
