#!/usr/bin/env python3
"""Build the distributable ZIP with a single legacy-compatible ``code/`` root."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", "validation_output"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if EXCLUDED_PARTS.intersection(relative.parts) or path.suffix in EXCLUDED_SUFFIXES:
                continue
            if path.resolve() == args.output.resolve():
                continue
            archive.write(path, (Path("code") / relative).as_posix())
    print(f"Built {args.output}")


if __name__ == "__main__":
    main()
