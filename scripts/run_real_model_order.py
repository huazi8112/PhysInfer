#!/usr/bin/env python3
"""Checkpointed 30-repeat adjacent-order analysis of the frozen real-data manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from physinfer.inference import adjacent_heldout_gain
from physinfer.order_selection import joint_bootstrap_decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path, help="gene-by-cell processed count CSV")
    parser.add_argument("manifest", type=Path, help="eligible manifest from prepare_real_data.py")
    parser.add_argument("null_gains", type=Path, help="condition-matched K2-null gains CSV")
    parser.add_argument("--output", type=Path, default=Path("real_model_order"))
    parser.add_argument("--n-genes", type=int, default=473)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(args.matrix, index_col=0).apply(pd.to_numeric, errors="coerce")
    manifest = pd.read_csv(args.manifest).head(args.n_genes)
    null = pd.read_csv(args.null_gains).gain_per_test_cell.to_numpy()
    checkpoint = args.output / "repeated_gains.csv"
    records = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    seen = {(row["gene"], int(row["repetition"])) for row in records}
    for gene_index, gene in enumerate(manifest.gene.astype(str)):
        values = matrix.loc[gene].to_numpy(dtype=float)
        values = np.rint(values[np.isfinite(values) & (values != -1)]).astype(int)
        for repetition in range(args.repeats):
            if (gene, repetition) in seen:
                continue
            seed = args.seed + gene_index * 1000 + repetition
            result = adjacent_heldout_gain(values, 2, eta=args.eta, seed=seed)
            records.append({"gene": gene, "manifest_position": gene_index, "repetition": repetition, **result})
            pd.DataFrame(records).to_csv(checkpoint, index=False)
    gains = pd.DataFrame(records)
    calls = []
    for gene_index, (gene, group) in enumerate(gains.groupby("gene", sort=False)):
        decision = joint_bootstrap_decision(
            group.gain_per_test_cell.to_numpy(),
            null,
            lower_order=2,
            n_draws=3000,
            seed=args.seed + 10_000 + gene_index,
        )
        calls.append({"gene": gene, "manifest_position": int(group.manifest_position.iloc[0]), **decision.__dict__})
    pd.DataFrame(calls).sort_values("manifest_position").to_csv(args.output / "initial_model_order_calls.csv", index=False)


if __name__ == "__main__":
    main()

