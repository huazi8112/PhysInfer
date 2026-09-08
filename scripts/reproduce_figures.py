#!/usr/bin/env python3
"""Rebuild the supplied numerical summary figures from frozen result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_selective(root: Path, output: Path) -> None:
    frame = pd.read_csv(root / "validation" / "selective_policy_48_units.csv")
    labels = [f"{high:.2f}/{low:.2f}" for high, low in zip(frame.policy_high, frame.policy_low)]
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(x - 0.18, frame.coverage, 0.36, label="Coverage", color="#3973B7")
    ax.bar(x + 0.18, frame.selective_accuracy, 0.36, label="Selective accuracy", color="#E28E2C")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("High/low evidence policy")
    ax.set_ylabel("Fraction")
    ax.legend(frameon=False)
    save(fig, output, "selective_reliability_coverage")


def plot_loss(root: Path, output: Path) -> None:
    frame = pd.read_csv(root / "loss_ablation" / "Table1_loss_ablation.csv")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    for ax, row in zip(axes.ravel(), frame.itertuples(index=False)):
        values = [row.full_loss_mean, row.distribution_only_mean]
        errors = [row.full_loss_sd, row.distribution_only_sd]
        bars = ax.bar(["Full", "Distribution\nonly"], values, yerr=errors, color=["#2F69BF", "#A9B6C6"], capsize=3)
        ax.set_title(row.metric, fontsize=10)
        lower = min(0.0, min(value - error for value, error in zip(values, errors)))
        upper = max(value + error for value, error in zip(values, errors))
        padding = max((upper - lower) * 0.15, 1e-5)
        ax.set_ylim(lower - padding, upper + padding)
        ax.bar_label(bars, labels=[f"{value:.4g}" for value in values], padding=3, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Paired loss ablation (mean ± SD)", fontsize=13)
    fig.tight_layout()
    save(fig, output, "loss_ablation_summary")


def plot_model_order(root: Path, output: Path) -> None:
    roles = pd.read_csv(root / "real_data" / "FINAL_MODEL_ORDER_ROLES_473.csv")
    column = "final_analysis_role" if "final_analysis_role" in roles else "final_role"
    order = ["stable_K2", "formal_K3", "ambiguous"]
    counts = roles[column].value_counts().reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    bars = ax.bar(["Stable K2", "Formal K3", "Ambiguous"], counts, color=["#4477AA", "#CC6677", "#BBBBBB"])
    ax.bar_label(bars)
    ax.set_ylabel("Completed genes")
    ax.set_title("Completed cohort (N=473)")
    save(fig, output, "real_data_model_order_flow")


def plot_kinetics(root: Path, output: Path) -> None:
    frame = pd.read_csv(root / "real_data" / "FINAL_LOWEST_NLL_NEURAL_ESTIMATES.csv")
    role_column = "analysis_role" if "analysis_role" in frame else "final_role"
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    for role, color, label in (("stable_K2", "#4477AA", "Stable K2"), ("formal_K3", "#CC6677", "Formal K3")):
        subset = frame[frame[role_column] == role]
        ax.scatter(subset.BS, subset.BF, s=20, alpha=0.72, color=color, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Burst size")
    ax.set_ylabel("Productive burst frequency")
    ax.legend(frameon=False)
    save(fig, output, "real_data_bs_bf_landscape")


def plot_recovery(root: Path, output: Path) -> None:
    wide = pd.read_csv(root / "recovery" / "V77_STRICT_RECOVERY_ENVELOPES_WIDE.csv")
    primary = wide[np.isclose(wide.fold_tolerance, 1.5)]
    x = np.arange(len(primary))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - 0.18, primary.nn, 0.36, label="Direct PhysInfer", color="#3973B7")
    ax.bar(x + 0.18, primary.mle, 0.36, label="Likelihood reference", color="#E28E2C")
    ax.set_xticks(x, primary.condition, rotation=20)
    ax.set_ylabel("Largest passing ON-state occupancy")
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=False)
    save(fig, output, "practical_recovery_1p5fold")


def plot_secondary_bic(root: Path, output: Path) -> None:
    frame = pd.read_csv(root / "secondary_bic" / "full_cohort_preference_summary.csv")
    counts = frame.n_genes.to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    wedges, _, autotexts = ax.pie(
        counts,
        colors=["#AFC3DE", "#D62728"],
        startangle=88,
        autopct=lambda pct: f"{pct:.1f}%",
        pctdistance=0.78,
        wedgeprops={"width": 0.36, "edgecolor": "white"},
        textprops={"fontsize": 12},
    )
    for text in autotexts:
        text.set_color("white")
        text.set_fontweight("bold")
    labels = [f"{name}\nN={count}" for name, count in zip(frame.preference, counts)]
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    ax.text(0, 0, "Secondary BIC-plus-veto\nQC-retained genes\n(N = 2,137)", ha="center", va="center")
    ax.set_title("Full-Cohort Model-Order Preferences from Secondary BIC Analysis")
    save(fig, output, "secondary_bic_full_cohort_preferences")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("frozen_results"))
    parser.add_argument("--output", type=Path, default=Path("reproduced_figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_selective(args.results, args.output)
    plot_loss(args.results, args.output)
    plot_model_order(args.results, args.output)
    plot_kinetics(args.results, args.output)
    plot_recovery(args.results, args.output)
    plot_secondary_bic(args.results, args.output)


if __name__ == "__main__":
    main()
