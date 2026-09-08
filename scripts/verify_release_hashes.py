#!/usr/bin/env python3
"""Verify every file listed in ``SHA256SUMS.txt``."""

from __future__ import annotations

import hashlib
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    lines = (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        if actual != expected:
            failures.append(relative)
    if failures:
        raise SystemExit("Hash verification failed: " + ", ".join(failures))
    print(f"Verified {len(lines)} release files.")


if __name__ == "__main__":
    main()
