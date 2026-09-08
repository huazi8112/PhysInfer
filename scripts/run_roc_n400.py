#!/usr/bin/env python3
"""Generate and score the independent 400-parameter-unit ROC panel.

Final protocol defaults: 240 K2 and 160 K3 units; 100 independent parameter
units per observation regime; 30 repeated simulations/held-out fits per unit;
20 label-phase repetitions form the unit-level detection-power score and 10 are
reserved as an evaluation phase.  A separate K2-null panel freezes the matched
90th-percentile threshold.  Use --smoke for a short installation exercise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from physinfer.inference import adjacent_heldout_gain
from physinfer.metrics import stratified_unit_auc_ci
from physinfer.models import SequentialSingleOn, sample_observed_counts

REGIMES = ((200, 0.3), (200, 1.0), (500, 0.3), (500, 1.0))


def sample_model(order: int, rng: np.random.Generator) -> SequentialSingleOn:
    if order == 2:
        return SequentialSingleOn(
            (float(10 ** rng.uniform(-1.2, 0.8)),),
            (float(10 ** rng.uniform(-1.2, 0.8)),),
            float(10 ** rng.uniform(0.3, 1.5)),
        )
    return SequentialSingleOn(
        tuple(10 ** rng.uniform(-1.2, 0.8, 2)),
        tuple(10 ** rng.uniform(-1.2, 0.8, 2)),
        float(10 ** rng.uniform(0.3, 1.5)),
    )


def build_manifest(seed: int, smoke: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for regime_index, (cells, eta) in enumerate(REGIMES):
        n_k2, n_k3 = ((2, 2) if smoke else (60, 40))
        for order, count in ((2, n_k2), (3, n_k3)):
            for within in range(count):
                model = sample_model(order, rng)
                rows.append(
                    {
                        "unit_id": f"r{regime_index}_K{order}_{within:03d}",
                        "label": order - 2,
                        "true_order": order,
                        "regime": f"n{cells}_eta{eta:g}",
                        "n_cells": cells,
                        "eta": eta,
                        "forward_rates": ";".join(map(str, model.forward_rates)),
                        "reverse_rates": ";".join(map(str, model.reverse_rates)),
                        "ksyn": model.ksyn,
                    }
                )
    return pd.DataFrame(rows)


def decode_model(row) -> SequentialSingleOn:
    return SequentialSingleOn(
        tuple(map(float, str(row.forward_rates).split(";"))),
        tuple(map(float, str(row.reverse_rates).split(";"))),
        float(row.ksyn),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("run_roc_n400_output"))
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "parameter_unit_manifest.csv"
    manifest = pd.read_csv(manifest_path) if args.resume and manifest_path.exists() else build_manifest(args.seed, args.smoke)
    manifest.to_csv(manifest_path, index=False)
    repeats = 2 if args.smoke else 30
    null_units = 2 if args.smoke else 24
    rng = np.random.default_rng(args.seed + 91)

    null_by_regime = {}
    for cells, eta in REGIMES:
        regime = f"n{cells}_eta{eta:g}"
        null_gains = []
        for unit in range(null_units):
            model = sample_model(2, rng)
            for repetition in range(repeats):
                seed = args.seed + 1_000_000 + cells * 1000 + unit * 100 + repetition
                counts = sample_observed_counts(model, cells, capture_efficiency=eta, seed=seed)
                result = adjacent_heldout_gain(
                    counts, 2, eta=eta, seed=seed + 17, n_random_starts=(1 if args.smoke else 6)
                )
                null_gains.append(result["gain_per_test_cell"])
        null_by_regime[regime] = np.asarray(null_gains)

    gain_path = args.output / "repeated_heldout_gains.csv"
    completed = pd.read_csv(gain_path) if args.resume and gain_path.exists() else pd.DataFrame()
    seen = set(zip(completed.get("unit_id", []), completed.get("repetition", [])))
    records = completed.to_dict("records")
    for row in manifest.itertuples(index=False):
        model = decode_model(row)
        for repetition in range(repeats):
            if (row.unit_id, repetition) in seen:
                continue
            seed = args.seed + len(records) * 101 + repetition
            counts = sample_observed_counts(model, row.n_cells, capture_efficiency=row.eta, seed=seed)
            result = adjacent_heldout_gain(
                counts, 2, eta=row.eta, seed=seed + 17, n_random_starts=(1 if args.smoke else 6)
            )
            records.append({"unit_id": row.unit_id, "repetition": repetition, **result})
            pd.DataFrame(records).to_csv(gain_path, index=False)

    gains = pd.DataFrame(records).merge(manifest, on="unit_id", how="left")
    units = []
    label_repeats = min(20, repeats)
    for unit_id, group in gains.groupby("unit_id", sort=False):
        first = group.iloc[0]
        threshold = float(np.quantile(null_by_regime[first.regime], 0.90))
        phase = group.sort_values("repetition").head(label_repeats)
        units.append(
            {
                "unit_id": unit_id,
                "label": int(first.label),
                "true_order": int(first.true_order),
                "regime": first.regime,
                "threshold": threshold,
                "detection_power": float(np.mean(phase.gain_per_test_cell > threshold)),
                "median_gain": float(np.median(phase.gain_per_test_cell)),
            }
        )
    unit_frame = pd.DataFrame(units)
    unit_frame.to_csv(args.output / "independent_unit_scores.csv", index=False)
    if unit_frame.label.nunique() == 2:
        summary = stratified_unit_auc_ci(
            unit_frame,
            label_column="label",
            score_column="detection_power",
            n_bootstrap=(200 if args.smoke else 20_000),
            seed=args.seed + 33,
        )
        serializable = {key: value for key, value in summary.items() if key not in {"fpr", "tpr", "threshold"}}
        (args.output / "auc_summary.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        pd.DataFrame({"fpr": summary["fpr"], "tpr": summary["tpr"], "threshold": summary["threshold"]}).to_csv(
            args.output / "roc_curve.csv", index=False
        )
        print(serializable)


if __name__ == "__main__":
    main()
