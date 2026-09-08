#!/usr/bin/env python3
"""Write deterministic per-file hashes and a compact release inventory."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


EXCLUDED = {"SHA256SUMS.txt", "RELEASE_MANIFEST.json"}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", "validation_output"}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in EXCLUDED or EXCLUDED_PARTS.intersection(path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
    top_level = Counter(record["path"].split("/", 1)[0] for record in records)
    manifest = {
        "release": "2026-09-08",
        "archive_root": "code/",
        "file_count": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "files_by_top_level_entry": dict(sorted(top_level.items())),
        "files": records,
    }
    (root / "RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [f"{record['sha256']}  {record['path']}" for record in records]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Indexed {len(records)} files ({manifest['total_bytes']} bytes).")


if __name__ == "__main__":
    main()
