"""Fetch the build inputs with caching and deterministic fallbacks (ARCHITECTURE §2.1).

Every source follows the same ladder: network (unless ``--offline``) -> cached copy in
``data/raw/`` -> documented fallback. Network results are validated before they are
accepted and cached. Each step down the ladder is recorded in ``fallbacks`` so that
``build_manifest.json`` says exactly what the artifacts were built from.

The artifacts depend only on the bytes returned here, so a rebuild from the same caches
is byte-identical (the manifest is the only file that carries timestamps).
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from scripts.netbuild import constants as K
from scripts.netbuild.gtfs import validate_gtfs_zip
from scripts.netbuild.http import FetchError, HttpClient, soql_url
from scripts.netbuild.jsonio import sha1_hex, write_atomic


class SourceError(RuntimeError):
    """A required input (the GTFS feed) is unavailable from every rung of the ladder."""


@dataclass(frozen=True)
class SourceRecord:
    """Provenance of one input, reported in the build manifest."""

    name: str
    origin: str  # "network" | "cache" | "fallback"
    url: str
    sha1: str
    bytes: int
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"origin": self.origin, "url": self.url, "sha1": self.sha1, "bytes": self.bytes, "note": self.note}


@dataclass(frozen=True)
class Ridership:
    """Weekday entries aggregated per complex x hour (one SoQL ``$group`` query)."""

    max_ts: str
    since_ts: str
    rows: tuple[dict[str, str], ...]


@dataclass
class Sources:
    gtfs_zip: bytes
    complexes: list[dict[str, Any]] | None
    stations: list[dict[str, Any]] | None
    ridership: Ridership | None
    basemap: dict[str, Any] | None
    records: dict[str, SourceRecord] = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- validators


def _valid_complexes(doc: Any) -> str:
    if not isinstance(doc, list) or len(doc) < 400:
        return "expected a JSON list of >= 400 complex rows"
    if not all(isinstance(r, dict) and "complex_id" in r and "gtfs_stop_ids" in r for r in doc):
        return "rows lack complex_id / gtfs_stop_ids"
    return ""


def _valid_stations(doc: Any) -> str:
    if not isinstance(doc, list) or len(doc) < 400:
        return "expected a JSON list of >= 400 station rows"
    if not all(isinstance(r, dict) and "gtfs_stop_id" in r for r in doc):
        return "rows lack gtfs_stop_id"
    return ""


def _valid_ridership(doc: Any) -> str:
    if not isinstance(doc, dict) or not isinstance(doc.get("rows"), list) or len(doc["rows"]) < 1000:
        return "expected {max_ts, since_ts, rows: [>= 1000 aggregate rows]}"
    need = {"station_complex_id", "hh", "r", "days"}
    if not all(isinstance(r, dict) and need <= set(r) for r in doc["rows"]):
        return f"aggregate rows lack {sorted(need)}"
    return ""


def _valid_basemap(doc: Any) -> str:
    if not isinstance(doc, dict) or doc.get("type") != "FeatureCollection" or not doc.get("features"):
        return "expected a GeoJSON FeatureCollection"
    return ""


# --------------------------------------------------------------------------- helpers


def _load_json_cache(path: Path, validate: Callable[[Any], str]) -> tuple[Any, bytes] | None:
    if not path.is_file():
        return None
    data = path.read_bytes()
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return (doc, data) if not validate(doc) else None


def _canonical_cache_bytes(doc: Any) -> bytes:
    """Cached JSON is re-serialized canonically so identical content gives identical bytes."""
    return (json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


class _Ladder:
    def __init__(self, raw_dir: Path, client: HttpClient) -> None:
        self.raw_dir = raw_dir
        self.client = client
        self.records: dict[str, SourceRecord] = {}
        self.fallbacks: list[str] = []

    def note_fallback(self, msg: str) -> None:
        self.fallbacks.append(msg)

    def json_source(
        self,
        name: str,
        filename: str,
        validate: Callable[[Any], str],
        fetch: Callable[[], Any],
        url: str,
        *,
        required_note: str,
    ) -> Any | None:
        path = self.raw_dir / filename
        error = "offline mode"
        if not self.client.offline:
            try:
                doc = fetch()
                problem = validate(doc)
                if problem:
                    raise FetchError(f"invalid response: {problem}")
                data = _canonical_cache_bytes(doc)
                write_atomic(path, data)
                self.records[name] = SourceRecord(name, "network", url, sha1_hex(data), len(data))
                return doc
            except FetchError as exc:
                error = str(exc)
        cached = _load_json_cache(path, validate)
        if cached is not None:
            doc, data = cached
            if not self.client.offline:
                self.note_fallback(f"{name}: fetch failed ({error}); used cached data/raw/{filename}")
            self.records[name] = SourceRecord(name, "cache", url, sha1_hex(data), len(data), note=error)
            return doc
        self.note_fallback(f"{name}: unavailable ({error}; no valid cache); {required_note}")
        self.records[name] = SourceRecord(name, "fallback", url, "", 0, note=required_note)
        return None


# --------------------------------------------------------------------------- GTFS


def _gtfs_candidate_urls(client: HttpClient) -> list[str]:
    meta = client.get_json(K.GTFS_METADATA_URL)
    text = json.dumps(meta)
    urls = {u.rstrip("\\") for u in re.findall(r"https?://[^\s\"'<>\\]+", text) if K.GTFS_URL_MARKER in u}
    return sorted(urls)


def _fetch_gtfs(ladder: _Ladder) -> bytes:
    client = ladder.client
    path = ladder.raw_dir / K.RAW_GTFS_ZIP
    problems: list[str] = []
    if not client.offline:
        candidates: list[tuple[str, str]] = []
        try:
            candidates.extend(("metadata", u) for u in _gtfs_candidate_urls(client))
        except FetchError as exc:
            problems.append(f"metadata: {exc}")
        candidates.append(("s3", K.GTFS_FALLBACK_URL))
        for kind, url in candidates:
            try:
                data = client.get(url)
            except FetchError as exc:
                problems.append(f"{url}: {exc}")
                continue
            bad = validate_gtfs_zip(data)
            if bad:
                problems.append(f"{url}: {'; '.join(bad)}")
                continue
            if kind == "s3":
                ladder.note_fallback("gtfs: no valid zip at the fgm6-ccue metadata URL(s) ("
                                     + " | ".join(problems) + f"); used {url}")
            write_atomic(path, data)
            ladder.records["gtfs"] = SourceRecord(
                "gtfs", "network", url, sha1_hex(data), len(data), note=" | ".join(problems)
            )
            return data
    if path.is_file():
        data = path.read_bytes()
        bad = validate_gtfs_zip(data)
        if not bad:
            if not client.offline:
                ladder.note_fallback("gtfs: every URL failed (" + " | ".join(problems) + "); used cached zip")
            ladder.records["gtfs"] = SourceRecord(
                "gtfs", "cache", K.GTFS_FALLBACK_URL, sha1_hex(data), len(data),
                note="offline mode" if client.offline else "network failed",
            )
            return data
        problems.append(f"cache: {'; '.join(bad)}")
    raise SourceError("GTFS feed unavailable: " + (" | ".join(problems) or "offline mode and no cached zip"))


# --------------------------------------------------------------------------- ridership


def _iso(ts: dt.datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def _fetch_ridership(client: HttpClient) -> dict[str, Any]:
    probe = client.get_json(soql_url(K.RIDERSHIP_URL, {"$select": "max(transit_timestamp) AS max_ts"}))
    if not probe or "max_ts" not in probe[0]:
        raise FetchError("max(transit_timestamp) query returned no value")
    max_ts = dt.datetime.fromisoformat(str(probe[0]["max_ts"]).rstrip("Z").split(".")[0])
    since = max_ts - dt.timedelta(days=K.RIDERSHIP_LOOKBACK_DAYS)
    where = f"transit_timestamp >= '{_iso(since)}' AND date_extract_dow(transit_timestamp) BETWEEN 1 AND 5"
    params = {
        "$select": "station_complex_id, date_extract_hh(transit_timestamp) AS hh, sum(ridership) AS r, "
        "count(distinct date_trunc_ymd(transit_timestamp)) AS days",
        "$where": where,
        "$group": "station_complex_id, hh",
        "$limit": "50000",
    }
    try:
        rows = client.get_json(soql_url(K.RIDERSHIP_URL, params), timeout_s=K.RIDERSHIP_TIMEOUT_S)
        distinct_days_query = False
    except FetchError as exc:
        if "HTTP 400" not in str(exc):
            raise
        # count(distinct ...) unsupported: aggregate without it and count weekdays separately.
        params["$select"] = "station_complex_id, date_extract_hh(transit_timestamp) AS hh, sum(ridership) AS r"
        rows = client.get_json(soql_url(K.RIDERSHIP_URL, params), timeout_s=K.RIDERSHIP_TIMEOUT_S)
        days = client.get_json(soql_url(K.RIDERSHIP_URL, {
            "$select": "date_trunc_ymd(transit_timestamp) AS d", "$where": where, "$group": "d", "$limit": "1000",
        }))
        n_days = str(len(days))
        rows = [dict(r, days=n_days) for r in rows]
        distinct_days_query = True
    if not isinstance(rows, list):
        raise FetchError("ridership aggregate is not a list")
    rows = sorted(
        ({k: str(v) for k, v in r.items()} for r in rows if isinstance(r, dict)),
        key=lambda r: (r.get("station_complex_id", ""), int(float(r.get("hh", 0)))),
    )
    return {
        "max_ts": _iso(max_ts),
        "since_ts": _iso(since),
        "lookback_days": K.RIDERSHIP_LOOKBACK_DAYS,
        "separate_distinct_day_query": distinct_days_query,
        "rows": rows,
    }


# --------------------------------------------------------------------------- entry point


def fetch_sources(raw_dir: Path, client: HttpClient) -> Sources:
    """Collect every input. Raises :class:`SourceError` only when no GTFS feed is available."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    ladder = _Ladder(raw_dir, client)
    gtfs = _fetch_gtfs(ladder)
    complexes = ladder.json_source(
        "complexes", K.RAW_COMPLEXES, _valid_complexes, lambda: client.get_json(K.COMPLEXES_URL), K.COMPLEXES_URL,
        required_note="complexes derived from GTFS parent stations unioned via transfers.txt",
    )
    stations = ladder.json_source(
        "stations", K.RAW_STATIONS, _valid_stations, lambda: client.get_json(K.STATIONS_URL), K.STATIONS_URL,
        required_note="station structure taken from the complex row",
    )
    ridership_doc = ladder.json_source(
        "ridership", K.RAW_RIDERSHIP, _valid_ridership, lambda: _fetch_ridership(client), K.RIDERSHIP_URL,
        required_note="ridership synthesized from routes x CBD x borough x the fixed hourly curve",
    )
    basemap = ladder.json_source(
        "basemap", K.RAW_BASEMAP, _valid_basemap, lambda: client.get_json(K.BASEMAP_URL, accept=None), K.BASEMAP_URL,
        required_note="no basemap emitted",
    )
    ridership = None
    if ridership_doc is not None:
        ridership = Ridership(
            max_ts=str(ridership_doc["max_ts"]),
            since_ts=str(ridership_doc["since_ts"]),
            rows=tuple(ridership_doc["rows"]),
        )
    return Sources(
        gtfs_zip=gtfs,
        complexes=complexes,
        stations=stations,
        ridership=ridership,
        basemap=basemap,
        records=dict(sorted(ladder.records.items())),
        fallbacks=ladder.fallbacks,
    )


def has_raw_gtfs(raw_dir: Path) -> bool:
    path = raw_dir / K.RAW_GTFS_ZIP
    return path.is_file() and not validate_gtfs_zip(path.read_bytes())
