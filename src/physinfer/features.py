"""Exchangeable gene-level distributional features used by PhysInfer.

Cells are bootstrap-resampled as exchangeable observations.  No sliding window,
cell-column order, pseudotime, or temporal interpretation is used.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

FEATURE_NAMES = (
    "mean",
    "variance",
    "fano",
    "zero_fraction",
    "cv",
    "skewness",
    "excess_kurtosis",
    "q00_over_mean",
    "q25_over_mean",
    "q50_over_mean",
    "q75_over_mean",
    "q90_over_mean",
    "q95_over_mean",
    "q99_over_mean",
    "q100_over_mean",
)


def clean_counts(counts: np.ndarray, *, missing_value: float = -1.0) -> np.ndarray:
    """Return finite non-negative integer counts, treating ``missing_value`` as missing."""
    x = np.asarray(counts, dtype=float).reshape(-1)
    x = x[np.isfinite(x) & (x != missing_value)]
    if x.size < 10:
        raise ValueError("At least 10 finite observed counts are required.")
    if np.any(x < 0) or not np.allclose(x, np.rint(x), atol=1e-8):
        raise ValueError("Observed counts must be non-negative integers; -1 may encode missingness.")
    return np.rint(x).astype(np.int64)


def extract_features15(counts: np.ndarray) -> np.ndarray:
    """Compute the frozen 5 + 2 + 8 feature vector from one empirical distribution."""
    x = clean_counts(counts)
    mean = float(np.mean(x))
    variance = float(np.var(x, ddof=0))
    scale = mean + 1e-10
    fano = variance / scale
    zero = float(np.mean(x == 0))
    cv = float(np.sqrt(max(variance, 0.0)) / scale)
    sk = float(skew(x, bias=False)) if variance > 0 else 0.0
    ku = float(kurtosis(x, fisher=True, bias=False)) if variance > 0 else 0.0
    if not np.isfinite(sk):
        sk = 0.0
    if not np.isfinite(ku):
        ku = 0.0
    quantiles = np.percentile(x, [0, 25, 50, 75, 90, 95, 99, 100]) / scale
    out = np.asarray([mean, variance, fano, zero, cv, sk, ku, *quantiles], dtype=float)
    out[~np.isfinite(out)] = 0.0
    if out.shape != (15,):
        raise RuntimeError("The PhysInfer feature vector must contain 15 entries.")
    return out


def bootstrap_feature_views(
    counts: np.ndarray,
    n_views: int,
    *,
    seed: int,
) -> np.ndarray:
    """Create exchangeable cell-bootstrap views and map each view to 15 features."""
    x = clean_counts(counts)
    rng = np.random.default_rng(seed)
    views = np.empty((n_views, 15), dtype=float)
    for index in range(n_views):
        views[index] = extract_features15(rng.choice(x, size=x.size, replace=True))
    return views

