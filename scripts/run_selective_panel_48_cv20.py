#!/usr/bin/env python3
"""Run a fresh 48-unit selective validation panel at one fixed condition.

Default scientific condition
----------------------------
* 48 independent parameter units: 24 effective K2 and 24 effective K3.
* 500 cells per simulated observation.
* 30% capture efficiency (eta=0.3; 70% technical transcript loss).
* 20% extrinsic noise (CV_ext=0.20).

CV_ext is operationally defined here as cell-to-cell lognormal heterogeneity in
the transcription-synthesis rate: k_syn,c = z_c * k_syn, where E[z_c] = 1 and
CV[z_c] = CV_ext.  The continuous lognormal mixture is evaluated by
Gauss-Hermite quadrature before binomial capture thinning.  This CV is not the
empirical coefficient of variation of the observed transcript counts.

The K2-null calibration panel is generated under the same n, eta, and CV_ext
condition.  Every parameter unit is evaluated by repeated independent count
simulations and independent 70/30 held-out fits.  The final call uses the same
joint-bootstrap 0.80/0.20 abstaining rule as the revised identifier.

Use --smoke for a small verification run.  Smoke output is diagnostic only and
must not be reported as the 48-unit manuscript result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.stats import beta

from physinfer.inference import adjacent_heldout_gain
from physinfer.models import (
    SequentialSingleOn,
    adaptive_stationary_pmf,
    thin_pmf,
)
from physinfer.order_selection import joint_bootstrap_decision


def sample_model(order: int, rng: np.random.Generator) -> SequentialSingleOn:
    """Draw one independent base parameter unit from the validation range."""
    if order == 2:
        return SequentialSingleOn(
            (float(10 ** rng.uniform(-1.2, 0.8)),),
            (float(10 ** rng.uniform(-1.2, 0.8)),),
            float(10 ** rng.uniform(0.3, 1.5)),
        )

    # Exclude nearly empty effective states so that the truth label represents
    # a genuine K3 generating unit rather than an intentionally nested K2 limit.
    for _ in range(10_000):
        model = SequentialSingleOn(
            tuple(10 ** rng.uniform(-1.2, 0.8, 2)),
            tuple(10 ** rng.uniform(-1.2, 0.8, 2)),
            float(10 ** rng.uniform(0.3, 1.5)),
        )
        occupancy = model.stationary_promoter
        if occupancy.min() >= 0.05 and 0.03 <= model.p_on <= 0.75:
            return model
    raise RuntimeError("Unable to draw an admissible K3 parameter unit.")


def model_record(unit_id: str, model: SequentialSingleOn, true_order: int) -> dict:
    return {
        "unit_id": unit_id,
        "true_order": true_order,
        "true_label": f"K{true_order}",
        "forward_rates": ";".join(map(str, model.forward_rates)),
        "reverse_rates": ";".join(map(str, model.reverse_rates)),
        "ksyn": model.ksyn,
        "p_on": model.p_on,
        "burst_size": model.burst_size,
        "burst_frequency": model.burst_frequency,
    }


def decode_model(row) -> SequentialSingleOn:
    return SequentialSingleOn(
        tuple(map(float, str(row.forward_rates).split(";"))),
        tuple(map(float, str(row.reverse_rates).split(";"))),
        float(row.ksyn),
    )


def build_manifest(seed: int, units_per_order: int, prefix: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for order in (2, 3):
        for index in range(units_per_order):
            model = sample_model(order, rng)
            rows.append(model_record(f"{prefix}_K{order}_{index:03d}", model, order))
    return pd.DataFrame(rows)


def lognormal_quadrature(cv: float, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return mean-one lognormal factors and normalized quadrature weights."""
    if cv < 0:
        raise ValueError("extrinsic CV must be non-negative")
    if cv == 0:
        return np.asarray([1.0]), np.asarray([1.0])
    if n_nodes < 3:
        raise ValueError("At least three quadrature nodes are required when CV > 0.")
    sigma2 = math.log1p(cv * cv)
    sigma = math.sqrt(sigma2)
    mu = -0.5 * sigma2
    nodes, weights = hermgauss(n_nodes)
    factors = np.exp(mu + math.sqrt(2.0) * sigma * nodes)
    weights = weights / math.sqrt(math.pi)
    weights = weights / weights.sum()
    # Remove the tiny numerical quadrature error in E[z] while preserving CV.
    factors = factors / np.sum(weights * factors)
    return factors, weights


def observed_mixture_pmf(
    model: SequentialSingleOn,
    *,
    eta: float,
    extrinsic_cv: float,
    quadrature_nodes: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Observed PMF under lognormal cell-to-cell ksyn heterogeneity."""
    factors, weights = lognormal_quadrature(extrinsic_cv, quadrature_nodes)
    components: list[tuple[float, np.ndarray]] = []
    max_support = 0
    for factor, weight in zip(factors, weights):
        perturbed = SequentialSingleOn(
            model.forward_rates,
            model.reverse_rates,
            model.ksyn * float(factor),
            model.kdeg,
        )
        latent, _ = adaptive_stationary_pmf(perturbed)
        observed = thin_pmf(latent, eta)
        components.append((float(weight), observed))
        max_support = max(max_support, observed.size)

    mixture = np.zeros(max_support, dtype=float)
    for weight, component in components:
        mixture[: component.size] += weight * component
    mixture = np.maximum(mixture, 0.0)
    mixture /= mixture.sum()
    weighted_mean = float(np.sum(weights * factors))
    weighted_variance = float(np.sum(weights * (factors - weighted_mean) ** 2))
    realized_cv = math.sqrt(weighted_variance) / weighted_mean
    return mixture, {
        "quadrature_factor_mean": weighted_mean,
        "quadrature_factor_cv": realized_cv,
    }


def sample_from_pmf(pmf: np.ndarray, n_cells: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(pmf.size), size=n_cells, replace=True, p=pmf)


def exact_binomial_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials == 0:
        return math.nan, math.nan
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def run_repeated_gains(
    manifest: pd.DataFrame,
    *,
    output_path: Path,
    n_cells: int,
    eta: float,
    extrinsic_cv: float,
    quadrature_nodes: int,
    repeats: int,
    seed: int,
    n_random_starts: int,
    n_base_starts: int | None,
    maxiter: int,
    resume: bool,
) -> pd.DataFrame:
    completed = pd.read_csv(output_path) if resume and output_path.exists() else pd.DataFrame()
    seen = set(zip(completed.get("unit_id", []), completed.get("repetition", [])))
    records = completed.to_dict("records")

    for unit_index, row in enumerate(manifest.itertuples(index=False)):
        model = decode_model(row)
        mixture, diagnostics = observed_mixture_pmf(
            model,
            eta=eta,
            extrinsic_cv=extrinsic_cv,
            quadrature_nodes=quadrature_nodes,
        )
        for repetition in range(repeats):
            if (row.unit_id, repetition) in seen:
                continue
            simulation_seed = seed + unit_index * 100_003 + repetition * 101
            counts = sample_from_pmf(mixture, n_cells, simulation_seed)
            fit_seed = simulation_seed + 37
            result = adjacent_heldout_gain(
                counts,
                2,
                eta=eta,
                seed=fit_seed,
                n_random_starts=n_random_starts,
                n_base_starts=n_base_starts,
                maxiter=maxiter,
            )
            records.append(
                {
                    "unit_id": row.unit_id,
                    "repetition": repetition,
                    "simulation_seed": simulation_seed,
                    "observed_mean": float(np.mean(counts)),
                    "observed_zero_fraction": float(np.mean(counts == 0)),
                    **diagnostics,
                    **result,
                }
            )
            pd.DataFrame(records).to_csv(output_path, index=False)
            print(f"completed {row.unit_id}, repeat {repetition + 1}/{repeats}", flush=True)
    return pd.DataFrame(records)


def save_summary_plot(calls: pd.DataFrame, summary: dict, path: Path) -> None:
    call_counts = calls["call"].value_counts().reindex(["K2", "K3", "ambiguous"], fill_value=0)
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].bar(call_counts.index, call_counts.values, color=["#4C78A8", "#E45756", "#B8B8B8"])
    axes[0].set_ylabel("Independent parameter units")
    axes[0].set_title("Selective calls")
    for index, value in enumerate(call_counts.values):
        axes[0].text(index, value + 0.2, str(int(value)), ha="center")

    agreement = summary["selective_agreement"]
    metrics = [summary["stable_call_coverage"], 0.0 if agreement is None else agreement]
    axes[1].bar(["Coverage", "Agreement"], metrics, color=["#72B7B2", "#54A24B"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Reliability–coverage")
    for index, value in enumerate(metrics):
        axes[1].text(index, value + 0.025, f"{100 * value:.1f}%", ha="center")
    figure.suptitle("500 cells, 30% capture efficiency, CV_ext = 20%")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("selective_panel_48_cv20_output"))
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--smoke", action="store_true", help="Run 4 units with reduced repeats and fitting budget.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-cells", type=int, default=500)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--extrinsic-cv", type=float, default=0.20)
    parser.add_argument("--units-per-order", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--null-units", type=int)
    parser.add_argument("--bootstrap-draws", type=int)
    parser.add_argument("--quadrature-nodes", type=int)
    parser.add_argument("--random-starts", type=int)
    parser.add_argument("--base-starts", type=int)
    parser.add_argument("--maxiter", type=int)
    args = parser.parse_args()

    units_per_order = args.units_per_order if args.units_per_order is not None else (2 if args.smoke else 24)
    repeats = args.repeats if args.repeats is not None else (2 if args.smoke else 30)
    null_units = args.null_units if args.null_units is not None else (5 if args.smoke else 24)
    bootstrap_draws = args.bootstrap_draws if args.bootstrap_draws is not None else (100 if args.smoke else 3000)
    quadrature_nodes = args.quadrature_nodes if args.quadrature_nodes is not None else (3 if args.smoke else 7)
    random_starts = args.random_starts if args.random_starts is not None else (0 if args.smoke else 6)
    base_starts = args.base_starts if args.base_starts is not None else (1 if args.smoke else None)
    maxiter = args.maxiter if args.maxiter is not None else (25 if args.smoke else 350)

    if args.n_cells < 50:
        raise ValueError("n_cells must be at least 50")
    if not 0 < args.eta <= 1:
        raise ValueError("eta must lie in (0,1]")
    if units_per_order < 1 or repeats < 2 or null_units * repeats < 10:
        raise ValueError("Need >=1 unit/order, >=2 repeats, and >=10 matched-null gains.")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "parameter_unit_manifest.csv"
    null_manifest_path = args.output / "matched_k2_null_manifest.csv"
    manifest = (
        pd.read_csv(manifest_path)
        if args.resume and manifest_path.exists()
        else build_manifest(args.seed, units_per_order, "validation")
    )
    null_manifest = (
        pd.read_csv(null_manifest_path)
        if args.resume and null_manifest_path.exists()
        else build_manifest(args.seed + 10_000_019, null_units, "null").query("true_order == 2").reset_index(drop=True)
    )
    manifest.to_csv(manifest_path, index=False)
    null_manifest.to_csv(null_manifest_path, index=False)

    protocol = {
        "purpose": "fresh fixed-condition selective validation; not the historical frozen 48-unit result",
        "n_parameter_units": int(len(manifest)),
        "n_K2_units": int((manifest.true_order == 2).sum()),
        "n_K3_units": int((manifest.true_order == 3).sum()),
        "n_cells": args.n_cells,
        "capture_efficiency": args.eta,
        "technical_transcript_loss": 1.0 - args.eta,
        "extrinsic_cv": args.extrinsic_cv,
        "extrinsic_noise_definition": "mean-one lognormal cell-to-cell multiplier on ksyn",
        "repeats_per_unit": repeats,
        "matched_K2_null_units": int(len(null_manifest)),
        "joint_bootstrap_draws": bootstrap_draws,
        "quadrature_nodes": quadrature_nodes,
        "train_fraction": 0.7,
        "q_high": 0.80,
        "q_low": 0.20,
        "probability_cutoff": 0.80,
        "smoke": args.smoke,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    null_gains = run_repeated_gains(
        null_manifest,
        output_path=args.output / "matched_k2_null_gains.csv",
        n_cells=args.n_cells,
        eta=args.eta,
        extrinsic_cv=args.extrinsic_cv,
        quadrature_nodes=quadrature_nodes,
        repeats=repeats,
        seed=args.seed + 20_000_033,
        n_random_starts=random_starts,
        n_base_starts=base_starts,
        maxiter=maxiter,
        resume=args.resume,
    )
    validation_gains = run_repeated_gains(
        manifest,
        output_path=args.output / "validation_repeated_heldout_gains.csv",
        n_cells=args.n_cells,
        eta=args.eta,
        extrinsic_cv=args.extrinsic_cv,
        quadrature_nodes=quadrature_nodes,
        repeats=repeats,
        seed=args.seed + 30_000_041,
        n_random_starts=random_starts,
        n_base_starts=base_starts,
        maxiter=maxiter,
        resume=args.resume,
    )

    matched_null = null_gains["gain_per_test_cell"].to_numpy(dtype=float)
    rows = []
    truth = manifest.set_index("unit_id")["true_label"].to_dict()
    for index, (unit_id, group) in enumerate(validation_gains.groupby("unit_id", sort=False)):
        decision = joint_bootstrap_decision(
            group["gain_per_test_cell"].to_numpy(dtype=float),
            matched_null,
            lower_order=2,
            n_draws=bootstrap_draws,
            seed=args.seed + 40_000_063 + index,
        )
        rows.append(
            {
                "unit_id": unit_id,
                "true_label": truth[unit_id],
                "p_lower": decision.p_lower,
                "p_higher": decision.p_higher,
                "call": decision.call,
                "stable_call": decision.call != "ambiguous",
                "correct_if_called": decision.call == truth[unit_id] if decision.call != "ambiguous" else pd.NA,
                "median_gain": float(group["gain_per_test_cell"].median()),
                "null_threshold_median": decision.null_threshold_median,
            }
        )
    calls = pd.DataFrame(rows)
    calls.to_csv(args.output / "selective_calls.csv", index=False)

    called = calls[calls.stable_call]
    correct = int(called.correct_if_called.astype(bool).sum()) if len(called) else 0
    ci_low, ci_high = exact_binomial_interval(correct, len(called))
    summary = {
        **protocol,
        "n_called": int(len(called)),
        "n_ambiguous": int((calls.call == "ambiguous").sum()),
        "n_stable_K2": int((calls.call == "K2").sum()),
        "n_stable_K3": int((calls.call == "K3").sum()),
        "stable_call_coverage": float(len(called) / len(calls)),
        "n_correct_among_called": correct,
        "selective_agreement": float(correct / len(called)) if len(called) else None,
        "exact_95_ci_lower": ci_low if math.isfinite(ci_low) else None,
        "exact_95_ci_upper": ci_high if math.isfinite(ci_high) else None,
        "reporting_guardrail": (
            "Smoke results are diagnostic only." if args.smoke else
            "Report these values as the fresh CV_ext=0.20 fixed-condition panel, not as the historical frozen panel."
        ),
    }
    (args.output / "selective_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_summary_plot(calls, summary, args.output / "selective_panel_summary.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
