#!/usr/bin/env python3
"""Audit frozen tables against the locked manuscript counts and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from audit_bic_cluster_results import load_cluster_tables


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    roles = pd.read_csv(root / "frozen_results/real_data/FINAL_MODEL_ORDER_ROLES_473.csv")
    role_counts = roles.final_analysis_role.value_counts().to_dict()
    initial_counts = roles.initial_state_call.value_counts().to_dict()
    selective = pd.read_csv(root / "frozen_results/validation/selective_policy_48_units.csv")
    policy = selective[(selective.policy_high == 0.8) & (selective.policy_low == 0.2)].iloc[0]
    roc = json.loads((root / "frozen_results/validation/roc_n400_summary.json").read_text(encoding="utf-8"))
    uq = pd.read_csv(root / "frozen_results/focused_K3_UQ/v651_final_K3_BS_BF_UQ_table.csv")
    bic = pd.read_csv(root / "frozen_results/secondary_bic/full_cohort_preference_summary.csv")
    bic_counts = dict(zip(bic.preference, bic.n_genes))
    bic_gene_rows, bic_gene_summary = load_cluster_tables(
        root / "frozen_results/secondary_bic/gene_level_clusters"
    )
    checks = {
        "completed_real_genes_473": len(roles) == 473,
        "initial_stable_K2_223": initial_counts.get("K2", 0) == 223,
        "initial_stable_K3_22": initial_counts.get("K3", 0) == 22,
        "initial_ambiguous_228": initial_counts.get("ambiguous", 0) == 228,
        "final_stable_K2_223": role_counts.get("stable_K2", 0) == 223,
        "final_formal_K3_20": role_counts.get("formal_K3", 0) == 20,
        "final_ambiguous_230": role_counts.get("ambiguous", 0) == 230,
        "selective_total_48": int(policy.n_total) == 48,
        "selective_called_27": int(policy.n_called) == 27,
        "selective_coverage_0p5625": abs(float(policy.coverage) - 0.5625) < 1e-12,
        "selective_accuracy_1": abs(float(policy.selective_accuracy) - 1.0) < 1e-12,
        "roc_units_400": roc["n_independent_units"] == 400,
        "roc_auc_0p978": abs(roc["auc"] - 0.978) < 1e-12,
        "focused_uq_genes_7": uq.gene.nunique() == 7,
        "focused_uq_700_completed": int(uq.n_successful_bootstraps.sum()) == 700,
        "secondary_bic_total_2137": int(bic.n_genes.sum()) == 2137,
        "secondary_bic_2state_1765": int(bic_counts.get("BIC-based 2-state preference", 0)) == 1765,
        "secondary_bic_3state_372": int(bic_counts.get("BIC-based 3-state preference", 0)) == 372,
        "secondary_bic_fraction_17p4pct": abs(float(bic.loc[bic.preference == "BIC-based 3-state preference", "fraction"].iloc[0]) - 372 / 2137) < 1e-9,
        "secondary_bic_cluster_files_8": bic_gene_summary["cluster_files"] == 8,
        "secondary_bic_source_rows_2161": len(bic_gene_rows) == 2161,
        "secondary_bic_unresolved_rows_24": bic_gene_summary["unresolved_rows"] == 24,
        "secondary_bic_resolved_rows_2137": bic_gene_summary["resolved_rows"] == 2137,
        "secondary_bic_gene_level_2state_1765": bic_gene_summary["state_counts"].get("2", 0) == 1765,
        "secondary_bic_gene_level_3state_372": bic_gene_summary["state_counts"].get("3", 0) == 372,
        "secondary_bic_gene_names_unique": bic_gene_summary["duplicate_gene_names"] == 0,
    }
    report = {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "initial_counts": initial_counts,
        "final_role_counts": role_counts,
        "secondary_bic_counts": bic_counts,
        "secondary_bic_gene_level_source": bic_gene_summary,
    }
    (root / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["all_checks_passed"]:
        raise SystemExit(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
