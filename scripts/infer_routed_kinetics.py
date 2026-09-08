#!/usr/bin/env python3
"""Train routed K2/K3 branches and retain the lowest-NLL neural restart per gene."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from physinfer.features import extract_features15
from physinfer.training import differentiable_stationary_pmf, train_branch


def infer_one_model(model, features, center, scale, order, counts, max_mrna):
    standardized = torch.tensor((features - center) / scale, dtype=torch.float64)
    with torch.no_grad():
        mapped = model.physical_map(model(standardized))
        pmf = differentiable_stationary_pmf(order, mapped, max_mrna)
        nll = -torch.log(pmf[torch.tensor(counts, dtype=torch.long)].clamp_min(1e-12)).mean().item()
    if order == 2:
        return {
            "k_on": mapped["k_on"].item(),
            "k_off": mapped["k_off"].item(),
            "k_syn": mapped["k_syn"].item(),
            "p_on": mapped["p_on"].item(),
            "BS": (mapped["k_syn"] / mapped["k_off"]).item(),
            "BF": (mapped["p_on"] * mapped["k_off"]).item(),
            "stationary_nll_per_cell": nll,
        }
    return {
        "k01": mapped["k01"].item(),
        "k10": mapped["k10"].item(),
        "k12": mapped["k12"].item(),
        "k21": mapped["k21"].item(),
        "k_syn": mapped["k_syn"].item(),
        "p_on": mapped["p_on"].item(),
        "BS": mapped["burst_size"].item(),
        "BF": mapped["burst_frequency"].item(),
        "stationary_nll_per_cell": nll,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("roles", type=Path)
    parser.add_argument("--output", type=Path, default=Path("routed_kinetics"))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(args.matrix, index_col=0).apply(pd.to_numeric, errors="coerce")
    roles = pd.read_csv(args.roles)
    all_predictions = []
    for order, role in ((2, "stable_K2"), (3, "formal_K3")):
        genes = roles.loc[roles.final_role == role, "gene"].astype(str).tolist()
        counts = []
        for gene in genes:
            values = matrix.loc[gene].to_numpy(dtype=float)
            values = np.rint(values[np.isfinite(values) & (values != -1)]).astype(int)
            counts.append(values)
        if not counts:
            continue
        minimum = min(map(len, counts))
        count_matrix = np.vstack([row[:minimum] for row in counts])
        feature_matrix = np.vstack([extract_features15(row) for row in count_matrix])
        for seed in args.seeds:
            trained = train_branch(
                count_matrix,
                order,
                retained_auxiliaries=True,
                seed=seed,
                epochs=args.epochs,
            )
            torch.save(trained["model"].state_dict(), args.output / f"K{order}_seed{seed}.pt")
            for index, gene in enumerate(genes):
                prediction = infer_one_model(
                    trained["model"],
                    feature_matrix[index],
                    trained["feature_center"],
                    trained["feature_scale"],
                    order,
                    count_matrix[index],
                    trained["max_mrna"],
                )
                all_predictions.append({"gene": gene, "final_role": role, "order": order, "seed": seed, **prediction})
    frame = pd.DataFrame(all_predictions)
    frame.to_csv(args.output / "all_neural_restarts.csv", index=False)
    best = frame.loc[frame.groupby("gene").stationary_nll_per_cell.idxmin()].sort_values("gene")
    best.to_csv(args.output / "final_lowest_nll_neural_estimates.csv", index=False)


if __name__ == "__main__":
    main()

