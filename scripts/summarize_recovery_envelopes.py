#!/usr/bin/env python3
"""Derive condition- and tolerance-specific BS/BF practical-recovery envelopes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predictions",
        type=Path,
        help="CSV containing estimator, condition, p_on, true_bs, true_bf, estimated_bs, estimated_bf",
    )
    parser.add_argument("--output", type=Path, default=Path("recovery_envelopes.csv"))
    parser.add_argument("--fold-tolerances", type=float, nargs="+", default=[1.2, 1.3, 1.4, 1.5, 2.0])
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    frame["bs_abs_log10_error"] = np.abs(np.log10(frame.estimated_bs / frame.true_bs))
    frame["bf_abs_log10_error"] = np.abs(np.log10(frame.estimated_bf / frame.true_bf))
    grid = (
        frame.groupby(["estimator", "condition", "p_on"], as_index=False)
        .agg(
            n_units=("true_bs", "size"),
            median_bs_abs_log10_error=("bs_abs_log10_error", "median"),
            median_bf_abs_log10_error=("bf_abs_log10_error", "median"),
        )
    )
    rows = []
    for tolerance in args.fold_tolerances:
        cutoff = math.log10(tolerance)
        for (estimator, condition), subset in grid.groupby(["estimator", "condition"], sort=False):
            passed = subset[
                (subset.median_bs_abs_log10_error <= cutoff)
                & (subset.median_bf_abs_log10_error <= cutoff)
            ]
            rows.append(
                {
                    "estimator": estimator,
                    "fold_tolerance": tolerance,
                    "condition": condition,
                    "largest_passing_p_on": float(passed.p_on.max()) if len(passed) else np.nan,
                    "criterion": "simultaneous median absolute log10 error for BS and productive BF",
                }
            )
    pd.DataFrame(rows).to_csv(args.output, index=False)
    grid.to_csv(args.output.with_name(args.output.stem + "_grid.csv"), index=False)


if __name__ == "__main__":
    main()

