#!/usr/bin/env python3
"""Regenerate/check the artifact file list, excluding Git internals and caches."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache"}


def manifest_text(root: Path = ROOT) -> str:
    paths = sorted(
        "./" + path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not (set(path.relative_to(root).parts) & EXCLUDED_DIRS)
        and path.suffix not in {".pyc", ".pyo"}
    )
    return "\n".join(paths) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check without rewriting the manifest.")
    args = parser.parse_args()
    path = ROOT / "MANIFEST.txt"
    expected = manifest_text()
    if args.check:
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            raise SystemExit("MANIFEST.txt is stale; run tools/update_manifest.py")
        print("MANIFEST.txt matches the current artifact files")
    else:
        path.write_text(expected, encoding="utf-8")
        print(f"Updated {path}: {len(expected.splitlines())} files")


if __name__ == "__main__":
    main()
