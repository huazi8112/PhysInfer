"""Unit-level validation metrics and stratified bootstrap intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def stratified_unit_auc_ci(
    frame: pd.DataFrame,
    *,
    label_column: str,
    score_column: str,
    strata_columns: tuple[str, ...] = ("label", "regime"),
    n_bootstrap: int = 20_000,
    seed: int = 1,
) -> dict[str, float]:
    """Bootstrap complete independent parameter units jointly by label and regime."""
    work = frame.reset_index(drop=True)
    labels = work[label_column].astype(int).to_numpy()
    scores = work[score_column].astype(float).to_numpy()
    auc = float(roc_auc_score(labels, scores))
    groups = [group.index.to_numpy() for _, group in work.groupby(list(strata_columns), sort=False)]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap)
    for draw in range(n_bootstrap):
        sampled = np.concatenate([rng.choice(index, size=index.size, replace=True) for index in groups])
        draws[draw] = roc_auc_score(labels[sampled], scores[sampled])
    fpr, tpr, threshold = roc_curve(labels, scores)
    return {
        "n_independent_units": int(work.shape[0]),
        "auc": auc,
        "ci025": float(np.quantile(draws, 0.025)),
        "ci975": float(np.quantile(draws, 0.975)),
        "fpr": fpr,
        "tpr": tpr,
        "threshold": threshold,
    }

