#!/usr/bin/env python3
"""Split-matched confirmation of every initial stable-K3 candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from physinfer.order_selection import joint_bootstrap_decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_calls", type=Path)
    parser.add_argument("confirmation_gains", type=Path, help="70/30 gains with gene and gain_per_test_cell")
    parser.add_argument("split_matched_null", type=Path, help="70/30 matched K2-null gains")
    parser.add_argument("--output", type=Path, default=Path("final_model_order_roles.csv"))
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    initial = pd.read_csv(args.initial_calls)
    confirmation = pd.read_csv(args.confirmation_gains)
    null = pd.read_csv(args.split_matched_null).gain_per_test_cell.to_numpy()
    candidates = initial.loc[initial.call == "K3", "gene"].astype(str).tolist()
    confirmation_rows = []
    retained = set()
    for index, gene in enumerate(candidates):
        gains = confirmation.loc[confirmation.gene.astype(str) == gene, "gain_per_test_cell"].to_numpy()
        decision = joint_bootstrap_decision(
            gains, null, lower_order=2, n_draws=3000, seed=args.seed + index
        )
        confirmation_rows.append({"gene": gene, **decision.__dict__})
        if decision.call == "K3":
            retained.add(gene)
    roles = initial[["gene", "manifest_position"]].copy()
    initial_call = dict(zip(initial.gene.astype(str), initial.call))

    def final_role(gene: str) -> str:
        call = initial_call[str(gene)]
        if call == "K2":
            return "stable_K2"
        if call == "K3" and str(gene) in retained:
            return "formal_K3"
        return "ambiguous"

    roles["final_role"] = [final_role(gene) for gene in roles.gene]
    roles.to_csv(args.output, index=False)
    pd.DataFrame(confirmation_rows).to_csv(args.output.with_name("k3_split_matched_confirmation.csv"), index=False)


if __name__ == "__main__":
    main()

