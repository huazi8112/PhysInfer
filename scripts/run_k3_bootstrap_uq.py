#!/usr/bin/env python3
"""Cell-bootstrap BS/BF uncertainty for a separately sampled fixed-K3 cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from physinfer.inference import fit_order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path, help="gene-by-cell CSV")
    parser.add_argument("genes", type=Path, help="one gene name per line")
    parser.add_argument("--output", type=Path, default=Path("k3_bootstrap_uq"))
    parser.add_argument("--bootstraps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(args.matrix, index_col=0).apply(pd.to_numeric, errors="coerce")
    genes = [line.strip() for line in args.genes.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = np.random.default_rng(args.seed)
    records = []
    checkpoint = args.output / "bootstrap_estimates.csv"
    if checkpoint.exists():
        records = pd.read_csv(checkpoint).to_dict("records")
    seen = {(row["gene"], int(row["bootstrap"])) for row in records}
    for gene_index, gene in enumerate(genes):
        x = matrix.loc[gene].to_numpy(dtype=float)
        x = np.rint(x[np.isfinite(x) & (x != -1)]).astype(int)
        for bootstrap in range(args.bootstraps):
            if (gene, bootstrap) in seen:
                continue
            sample = rng.choice(x, size=x.size, replace=True)
            fit = fit_order(sample, 3, eta=args.eta, seed=args.seed + gene_index * 1000 + bootstrap)
            model = fit.model
            records.append(
                {
                    "gene": gene,
                    "bootstrap": bootstrap,
                    "BS": model.burst_size,
                    "BF": model.burst_frequency,
                    "p_on": model.p_on,
                    "nll": fit.nll,
                    "max_mrna": fit.max_mrna,
                    "boundary_mass": fit.boundary_mass,
                }
            )
            pd.DataFrame(records).to_csv(checkpoint, index=False)
    estimates = pd.DataFrame(records)
    summary = estimates.groupby("gene").agg(
        n_completed=("bootstrap", "count"),
        BS_median=("BS", "median"),
        BS_q025=("BS", lambda x: x.quantile(0.025)),
        BS_q975=("BS", lambda x: x.quantile(0.975)),
        BF_median=("BF", "median"),
        BF_q025=("BF", lambda x: x.quantile(0.025)),
        BF_q975=("BF", lambda x: x.quantile(0.975)),
    )
    summary.to_csv(args.output / "bootstrap_summary.csv")


if __name__ == "__main__":
    main()

