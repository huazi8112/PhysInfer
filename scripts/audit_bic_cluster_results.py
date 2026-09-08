#!/usr/bin/env python3
"""Validate and optionally merge the eight cluster-level BIC result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = ["gene_index", "gene_name", "predicted_state", "confidence"]


def load_cluster_tables(directory: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    files = sorted(directory.glob("cluster_*_identification_results.csv"))
    if len(files) != 8:
        raise ValueError(f"Expected 8 cluster files, found {len(files)} in {directory}")
    frames: list[pd.DataFrame] = []
    rows_by_file: dict[str, int] = {}
    for path in files:
        frame = pd.read_csv(path)
        if list(frame.columns) != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected columns in {path.name}: {list(frame.columns)}")
        frame = frame.copy()
        frame.insert(0, "cluster", int(path.stem.split("_")[1]))
        frame["source_file"] = path.name
        frames.append(frame)
        rows_by_file[path.name] = len(frame)
    combined = pd.concat(frames, ignore_index=True)
    resolved = combined.dropna(subset=["predicted_state"]).copy()
    resolved["predicted_state"] = resolved["predicted_state"].astype(int)
    counts = resolved.predicted_state.value_counts().sort_index().to_dict()
    summary = {
        "cluster_files": len(files),
        "rows_by_file": rows_by_file,
        "total_rows": len(combined),
        "unresolved_rows": int(combined.predicted_state.isna().sum()),
        "resolved_rows": len(resolved),
        "state_counts": {str(key): int(value) for key, value in counts.items()},
        "unique_gene_names": int(combined.gene_name.nunique()),
        "duplicate_gene_names": int(combined.gene_name.duplicated().sum()),
        "confidence_min": float(combined.confidence.min()),
        "confidence_max": float(combined.confidence.max()),
    }
    return combined, summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "frozen_results/secondary_bic/gene_level_clusters",
    )
    parser.add_argument("--output", type=Path, help="Optional merged CSV output path")
    parser.add_argument("--summary", type=Path, help="Optional JSON summary output path")
    args = parser.parse_args()
    combined, summary = load_cluster_tables(args.input)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.output, index=False)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
