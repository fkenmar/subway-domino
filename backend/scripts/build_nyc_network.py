"""Build the NYC subway network artifacts (ARCHITECTURE §2, WP1).

Run from ``backend/``::

    .venv/bin/python -m scripts.build_nyc_network                    # fetch, build, validate, write
    .venv/bin/python -m scripts.build_nyc_network --offline          # build from data/raw/ caches only
    .venv/bin/python -m scripts.build_nyc_network --offline --check  # verify; writes nothing

A build fetches every input through its fallback ladder (network -> ``data/raw/`` cache ->
documented fallback, §2.1), infers the track graph, validates every §2.6/§2.7 invariant and only
then writes ``data/network.json``, ``data/network_geo.json`` (plus a byte-identical copy in
``frontend/public/data/``) and ``data/build_manifest.json``. An invalid network is never written.

``--offline`` never touches the network. Without a cached GTFS feed it validates the committed
artifacts instead and exits 0 when they are valid.

``--check`` rebuilds in memory and verifies that the committed artifacts (and the frontend copy)
are valid and byte-identical to the rebuild: the determinism and freshness check. Nothing is written.

Exit codes: 0 success, 1 invariant violation / inconsistent overlay / stale artifacts (``--check``),
2 inputs unavailable (no GTFS feed from any source and no committed artifacts).

On macOS, Python's CA bundle may lack the root of ``data.cityofnewyork.us`` (basemap); the build
then uses the cached basemap. Point ``SSL_CERT_FILE`` at a bundle that includes the system roots
to fetch it fresh.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from scripts.netbuild import constants as K
from scripts.netbuild.artifacts import (
    BuildOutput,
    artifact_counts,
    build_artifacts,
    manifest_bytes,
    manifest_doc,
    validate_committed,
)
from scripts.netbuild.gtfs import GtfsError
from scripts.netbuild.http import HttpClient
from scripts.netbuild.infra import InfrastructureError
from scripts.netbuild.jsonio import write_atomic
from scripts.netbuild.pipeline import BuildError
from scripts.netbuild.places import PlacesError
from scripts.netbuild.sources import SourceError, has_raw_gtfs
from scripts.netbuild.topology import TopologyError

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BACKEND_DIR / "data"
DEFAULT_FRONTEND_DATA_DIR = BACKEND_DIR.parent / "frontend" / "public" / "data"

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_UNAVAILABLE = 2

log = logging.getLogger("build_nyc_network")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.build_nyc_network",
        description="Build network.json, network_geo.json and build_manifest.json from MTA / NY Open Data.",
    )
    parser.add_argument("--offline", action="store_true",
                        help="no network access: build from data/raw/ caches; without caches, validate the "
                             "committed artifacts and exit")
    parser.add_argument("--check", action="store_true",
                        help="verify only: rebuild in memory and require the committed artifacts to be valid and "
                             "byte-identical to the rebuild; writes nothing")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="directory with infrastructure.json and the raw/ cache (default: backend/data)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where the artifacts are written / compared (default: --data-dir)")
    parser.add_argument("--frontend-dir", type=Path, default=DEFAULT_FRONTEND_DATA_DIR,
                        help="directory of the frontend's copy of network_geo.json (default: frontend/public/data)")
    parser.add_argument("--no-frontend-copy", action="store_true",
                        help="neither write nor check the frontend copy of network_geo.json")
    parser.add_argument("-q", "--quiet", action="store_true", help="only print warnings, errors and the summary")
    return parser.parse_args(argv)


def _configure_logging(quiet: bool) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(logging.WARNING if quiet else logging.INFO)
    log.propagate = False


def _report(errors: Sequence[str], warnings: Sequence[str]) -> None:
    for w in warnings:
        log.warning("warning: %s", w)
    for e in errors:
        log.error("error: %s", e)


def _summary(out: BuildOutput) -> str:
    c = artifact_counts(out.network)
    run = c["edges_by_kind"].get("run", 0)
    il = ", ".join(f"{k} {v}" for k, v in c["interlockings_by_kind"].items())
    return (
        f"graph_version {out.graph_version}: {c['routes']} routes, {c['stations']} stations, "
        f"{c['complexes']} complexes, {c['platforms']} platforms, {c['nodes']} nodes, {run} run edges "
        f"({c['edges']} edges), {c['segments']} segments ({c['segments_with_express']} with express), "
        f"{c['interlockings']} interlockings ({il}), {c['pocket_turnback_stations']} pocket-turnback stations, "
        f"{c['yards']} yards, {c['presets']} presets; network.json {len(out.network_bytes):,} B, "
        f"network_geo.json {len(out.geo_bytes):,} B"
    )


def _differs(path: Path, data: bytes) -> str:
    if not path.is_file():
        return f"{path} is missing"
    if path.read_bytes() != data:
        return f"{path} differs from the rebuild"
    return ""


def _check_frontend_copy(frontend_dir: Path | None, out_dir: Path) -> list[str]:
    if frontend_dir is None:
        return []
    geo = out_dir / K.GEO_FILE
    if not geo.is_file():
        return []
    problem = _differs(frontend_dir / K.GEO_FILE, geo.read_bytes())
    return [f"frontend copy: {problem}"] if problem else []


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.quiet)
    data_dir = args.data_dir.resolve()
    out_dir = (args.out_dir or args.data_dir).resolve()
    frontend_dir = None if args.no_frontend_copy else args.frontend_dir.resolve()

    if args.offline and not has_raw_gtfs(data_dir / "raw"):
        log.info("offline and no cached GTFS feed in %s: validating the committed artifacts", data_dir / "raw")
        if not (out_dir / K.NETWORK_FILE).is_file():
            log.error("error: no cached inputs and no committed artifacts in %s", out_dir)
            return EXIT_UNAVAILABLE
        errors, warnings = validate_committed(out_dir)
        errors += _check_frontend_copy(frontend_dir, out_dir)
        _report(errors, warnings)
        if errors:
            return EXIT_INVALID
        print(f"committed artifacts in {out_dir} are valid (not rebuilt: no cached inputs)")
        return EXIT_OK

    client = HttpClient(offline=args.offline)
    try:
        out = build_artifacts(data_dir, client, log=log.info)
    except SourceError as exc:
        log.error("error: %s", exc)
        return EXIT_UNAVAILABLE
    except (InfrastructureError, BuildError, TopologyError, PlacesError, GtfsError) as exc:
        log.error("error: %s: %s", type(exc).__name__, exc)
        return EXIT_INVALID

    for note in out.model.notes:
        log.info("note: %s", note)
    _report(out.errors, out.warnings)
    if out.errors:
        log.error("%d invariant violation(s); nothing written", len(out.errors))
        return EXIT_INVALID

    targets = [(out_dir / K.NETWORK_FILE, out.network_bytes), (out_dir / K.GEO_FILE, out.geo_bytes)]
    if frontend_dir is not None:
        targets.append((frontend_dir / K.GEO_FILE, out.geo_bytes))

    if args.check:
        stale = [msg for msg in (_differs(path, data) for path, data in targets) if msg]
        for msg in stale:
            log.error("error: %s", msg)
        print(_summary(out))
        if stale:
            log.error("artifacts are not up to date with the inputs (run the build without --check)")
            return EXIT_INVALID
        print("check passed: committed artifacts are valid and byte-identical to the rebuild")
        return EXIT_OK

    for path, data in targets:
        write_atomic(path, data)
        log.info("wrote %s (%d bytes)", path, len(data))
    manifest = manifest_doc(out, offline=args.offline)
    write_atomic(out_dir / K.MANIFEST_FILE, manifest_bytes(manifest))
    log.info("wrote %s", out_dir / K.MANIFEST_FILE)
    print(_summary(out))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
