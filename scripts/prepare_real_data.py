#!/usr/bin/env python3
"""Reconstruct the frozen eligible manifest from the processed GSE176044 matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_gene_by_cell(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return numeric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path, default=Path("real_data_preflight"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    matrix = read_gene_by_cell(args.matrix)
    rows = []
    missing_entries = int((matrix == -1).sum().sum())
    for position, (gene, series) in enumerate(matrix.iterrows()):
        values = series.to_numpy(dtype=float)
        finite = values[np.isfinite(values) & (values != -1)]
        eligible = finite.size >= 200 and finite.max(initial=0) <= 300
        reason = "eligible" if eligible else ("fewer_than_200_finite" if finite.size < 200 else "max_count_above_300")
        rows.append(
            {
                "manifest_position": position,
                "gene": gene,
                "n_finite": finite.size,
                "max_finite_count": float(finite.max()) if finite.size else np.nan,
                "eligible": eligible,
                "reason": reason,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output / "all_gene_eligibility.csv", index=False)
    manifest.loc[manifest.eligible].to_csv(args.output / "eligible_manifest_2135.csv", index=False)
    summary = {
        "matrix_shape": list(matrix.shape),
        "missing_minus_one_entries": missing_entries,
        "n_eligible": int(manifest.eligible.sum()),
        "eligibility": "n_finite >= 200 and max_finite_count <= 300",
        "completed_cohort_rule": "first 473 genes completing all 30 repetitions in frozen manifest order",
    }
    (args.output / "preflight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
