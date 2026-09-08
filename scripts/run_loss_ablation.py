#!/usr/bin/env python3
"""Paired full-loss versus distribution-only direct-backprop experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from physinfer.training import train_branch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("counts", type=Path, help="NPY array [units,cells]")
    parser.add_argument("--truth", type=Path, help="CSV with state and eta columns; rows align with counts")
    parser.add_argument("--order", type=int, choices=(2, 3), required=True)
    parser.add_argument("--output", type=Path, default=Path("loss_ablation"))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    counts = np.load(args.counts)
    capture_efficiencies = np.ones(counts.shape[0], dtype=float)
    if args.truth is not None:
        truth = pd.read_csv(args.truth)
        if len(truth) != len(counts):
            raise ValueError("Truth table and count array must have the same number of rows.")
        keep = truth.state.to_numpy(dtype=int) == args.order
        counts = counts[keep]
        capture_efficiencies = truth.loc[keep, "eta"].to_numpy(dtype=float)
    rows = []
    for retained_auxiliaries in (True, False):
        mode = "full_loss" if retained_auxiliaries else "distribution_only"
        for seed in args.seeds:
            result = train_branch(
                counts,
                args.order,
                retained_auxiliaries=retained_auxiliaries,
                seed=seed,
                epochs=args.epochs,
                capture_efficiencies=capture_efficiencies,
            )
            torch.save(result["model"].state_dict(), args.output / f"K{args.order}_{mode}_seed{seed}.pt")
            pd.DataFrame({"epoch": np.arange(args.epochs), "loss": result["history"]}).to_csv(
                args.output / f"K{args.order}_{mode}_seed{seed}_history.csv", index=False
            )
            rows.append({"order": args.order, "mode": mode, "seed": seed, "final_training_loss": result["history"][-1]})
    pd.DataFrame(rows).to_csv(args.output / "paired_training_summary.csv", index=False)


if __name__ == "__main__":
    main()
