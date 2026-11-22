"""Deterministic JSON output: rounding, diff-friendly layout, canonical hashing.

Artifacts are written as compact JSON with one array element per line at the top two
levels, so ``git diff`` stays readable while the files stay small. ``graph_version`` is
computed over the *canonical* form (sorted keys, no whitespace), independent of layout.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def rnd(value: float, decimals: int) -> float:
    """Round half-even like ``round`` but never emit ``-0.0`` and reject non-finite values."""
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"non-finite value {value!r} cannot be serialized")
    r = round(v, decimals)
    return 0.0 if r == 0 else r


def _compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _layout(obj: Any, depth: int, max_depth: int) -> str:
    if depth >= max_depth:
        return _compact(obj)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        parts = [f"{_compact(str(k))}:{_layout(v, depth + 1, max_depth)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(parts) + "}"
    if isinstance(obj, list) and obj and all(isinstance(v, (dict, list)) for v in obj):
        return "[\n" + ",\n".join(_compact(v) for v in obj) + "]"
    return _compact(obj)


def dumps_artifact(doc: dict[str, Any]) -> str:
    """Line-oriented compact JSON (valid JSON, deterministic for a given ``doc``)."""
    return _layout(doc, 0, 2) + "\n"


def canonical_bytes(doc: Any) -> bytes:
    return json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def content_hash(doc: dict[str, Any], *, field: str = "graph_version", length: int = 12) -> str:
    """sha1 of the canonical JSON of ``doc`` with ``field`` set to ``""`` (ARCHITECTURE §2.6)."""
    probe = dict(doc)
    probe[field] = ""
    return hashlib.sha1(canonical_bytes(probe)).hexdigest()[:length]


def sha1_hex(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def write_atomic(path: Path, data: bytes, mode: int = 0o644) -> None:
    """Write ``data`` to ``path`` via a temp file + rename (never leaves a partial file).

    ``mkstemp`` creates the temp file as 0600; the result gets ``mode`` (world-readable by
    default: the artifacts are served by the frontend and copied into the server image).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
