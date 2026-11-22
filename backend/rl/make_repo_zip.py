"""Build ``dist/subway-domino.zip``: the code-upload fallback for Colab (ARCHITECTURE §4.10).

Upload the zip to ``MyDrive/subway-domino/subway-domino.zip`` and pick ``CODE_SOURCE =
"drive_zip"`` in the notebook. Everything sits under one top-level ``subway-domino/`` folder.
Excluded: ``.venv``, ``node_modules``, ``.next``, ``data/raw``, ``runs`` and ``__pycache__``
(plus ``.git``, ``dist``, caches and temporary files).

    .venv/bin/python -m rl.make_repo_zip            # from backend/
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_DIRS = {".venv", "node_modules", ".next", "runs", "__pycache__", ".git", "dist", ".pytest_cache",
                 ".mypy_cache", ".ruff_cache", "test-results", "playwright-report"}
EXCLUDED_PATHS = {("backend", "data", "raw")}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp"}
EXCLUDED_NAMES = {".DS_Store", ".env", ".env.local"}


def iter_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        dirnames[:] = sorted(d for d in dirnames
                             if d not in EXCLUDED_DIRS and (*rel.parts, d) not in EXCLUDED_PATHS)
        for f in sorted(filenames):
            if f in EXCLUDED_NAMES or Path(f).suffix in EXCLUDED_SUFFIXES:
                continue
            out.append(Path(dirpath) / f)
    return out


def build_zip(root: Path = REPO_ROOT, out: Path | None = None, prefix: str = "subway-domino") -> Path:
    out = out if out is not None else root / "dist" / "subway-domino.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in iter_files(root):
            if p.resolve() == out.resolve():
                continue
            zf.write(p, Path(prefix) / p.relative_to(root))
    os.replace(tmp, out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/dist/subway-domino.zip")
    args = ap.parse_args(argv)
    out = build_zip(args.root, args.out)
    with zipfile.ZipFile(out) as zf:
        n = len(zf.namelist())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {n} files); upload it to "
          "MyDrive/subway-domino/subway-domino.zip and set CODE_SOURCE = 'drive_zip' in the notebook")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
