#!/usr/bin/env python3
"""Generate secondary BIC-plus-veto preferences without changing primary calls."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from physinfer.bic_annotation import annotate_secondary_bic_preference


REQUIRED = {
    "gene",
    "n_cells",
    "total_nll_2state",
    "total_nll_3state",
    "k01",
    "k10",
    "k12",
    "k21",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="CSV containing paired model fits")
    parser.add_argument("--output", type=Path, default=Path("secondary_bic_annotations.csv"))
    parser.add_argument("--parameters-2state", type=int, default=3)
    parser.add_argument("--parameters-3state", type=int, default=5)
    parser.add_argument("--tau", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=2.5)
    parser.add_argument(
        "--scope",
        choices=("all", "ambiguous-only"),
        default="all",
        help="Restrict annotation to primary ambiguous rows when requested.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    if "primary_call" not in frame:
        frame["primary_call"] = "unavailable"
    if args.scope == "ambiguous-only":
        frame = frame[frame.primary_call.str.lower() == "ambiguous"].copy()

    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        result = annotate_secondary_bic_preference(
            total_nll_2state=row.total_nll_2state,
            total_nll_3state=row.total_nll_3state,
            n_parameters_2state=args.parameters_2state,
            n_parameters_3state=args.parameters_3state,
            n_observations=int(row.n_cells),
            k3_rates=(row.k01, row.k10, row.k12, row.k21),
            primary_call=str(row.primary_call),
            tau=args.tau,
            gamma=args.gamma,
        )
        rows.append({"gene": row.gene, **asdict(result)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} secondary annotations to {args.output}")


if __name__ == "__main__":
    main()
