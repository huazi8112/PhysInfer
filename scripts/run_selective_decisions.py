#!/usr/bin/env python3
"""Apply the frozen joint-bootstrap selective policy to a repeated-gain table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from physinfer.order_selection import joint_bootstrap_decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gains", type=Path, help="CSV with unit_id, regime, gain_per_test_cell")
    parser.add_argument("null_gains", type=Path, help="CSV with regime, gain_per_test_cell")
    parser.add_argument("--output", type=Path, default=Path("selective_calls.csv"))
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    gains = pd.read_csv(args.gains)
    null = pd.read_csv(args.null_gains)
    rows = []
    for index, (unit_id, group) in enumerate(gains.groupby("unit_id", sort=False)):
        regime = group.regime.iloc[0]
        matched = null.loc[null.regime == regime, "gain_per_test_cell"].to_numpy()
        decision = joint_bootstrap_decision(
            group.gain_per_test_cell.to_numpy(),
            matched,
            lower_order=2,
            n_draws=args.draws,
            seed=args.seed + index,
        )
        rows.append({"unit_id": unit_id, "regime": regime, **decision.__dict__})
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

