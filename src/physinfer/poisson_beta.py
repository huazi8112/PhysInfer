"""Exact stationary Poisson-Beta likelihood for the effective K2 branch."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp1f1, logsumexp, roots_jacobi


def _logpmf_hyp1f1(counts: np.ndarray, kon: float, koff: float, ksyn_observed: float) -> np.ndarray:
    n = np.asarray(counts, dtype=float)
    prefactor = (
        gammaln(kon + n)
        + gammaln(kon + koff)
        - gammaln(kon)
        - gammaln(kon + koff + n)
        + n * math.log(ksyn_observed)
        - gammaln(n + 1)
    )
    h = hyp1f1(kon + n, kon + koff + n, -ksyn_observed)
    if np.any(~np.isfinite(h)) or np.any(h <= 0):
        raise FloatingPointError
    return prefactor + np.log(h)


def _logpmf_quadrature(
    counts: np.ndarray, kon: float, koff: float, ksyn_observed: float, n_nodes: int = 80
) -> np.ndarray:
    nodes, weights = roots_jacobi(n_nodes, koff - 1, kon - 1)
    x = (nodes + 1) / 2
    log_w = np.log(weights) - (kon + koff - 1) * math.log(2) - betaln(kon, koff)
    log_lambda = np.log(np.clip(ksyn_observed * x, 1e-300, None))
    out = []
    for count in np.asarray(counts, dtype=int):
        terms = log_w - ksyn_observed * x + count * log_lambda - gammaln(count + 1)
        out.append(logsumexp(terms))
    return np.asarray(out)


def logpmf(counts: np.ndarray, kon: float, koff: float, ksyn: float, eta: float = 1.0) -> np.ndarray:
    """Evaluate observed-count log probabilities; binomial thinning maps ksyn -> eta*ksyn."""
    observed_ksyn = eta * ksyn
    try:
        return _logpmf_hyp1f1(counts, kon, koff, observed_ksyn)
    except Exception:
        return _logpmf_quadrature(counts, kon, koff, observed_ksyn)


def nll(counts: np.ndarray, rates: np.ndarray, *, eta: float = 1.0) -> float:
    x = np.asarray(counts, dtype=int)
    unique, frequency = np.unique(x, return_counts=True)
    values = logpmf(unique, *np.asarray(rates, dtype=float), eta=eta)
    if np.any(~np.isfinite(values)) or np.any(values > 1e-7):
        return 1e100
    return -float(np.sum(frequency * values))


@dataclass(frozen=True)
class K2Fit:
    kon: float
    koff: float
    ksyn: float
    nll: float
    success: bool

    @property
    def burst_size(self) -> float:
        return self.ksyn / self.koff

    @property
    def burst_frequency(self) -> float:
        return self.kon * self.koff / (self.kon + self.koff)


def fit_multistart(
    counts: np.ndarray,
    *,
    eta: float = 1.0,
    seed: int,
    n_random_starts: int = 8,
    n_deterministic_starts: int = 5,
    maxiter: int = 700,
) -> K2Fit:
    """Bounded multistart L-BFGS-B fit on training cells only."""
    x = np.asarray(counts, dtype=float)
    x = np.rint(x[np.isfinite(x)]).astype(int)
    if x.size < 20 or np.any(x < 0):
        raise ValueError("At least 20 non-negative training counts are required.")
    mean = max(float(np.mean(x)), 1e-3)
    base = [
        (0.1, 1.0, max(mean / eta * 11, 1.0)),
        (0.5, 0.5, max(mean / eta * 2, 1.0)),
        (1.0, 1.0, max(mean / eta * 2, 1.0)),
        (5.0, 1.0, max(mean / eta * 1.2, 1.0)),
        (1.0, 10.0, max(mean / eta * 11, 1.0)),
    ]
    if not 1 <= n_deterministic_starts <= len(base):
        raise ValueError(f"n_deterministic_starts must lie in [1,{len(base)}].")
    base = base[:n_deterministic_starts]
    rng = np.random.default_rng(seed)
    for _ in range(n_random_starts):
        kon, koff = 10 ** rng.uniform(-2, 1.7, 2)
        pon = kon / (kon + koff)
        ksyn = np.clip(mean / eta / max(pon, 1e-4) * 10 ** rng.uniform(-0.5, 0.5), 0.02, 1500)
        base.append((kon, koff, float(ksyn)))
    bounds = [(math.log(1e-3), math.log(300)), (math.log(1e-3), math.log(300)), (math.log(0.02), math.log(1500))]
    fits = []
    for start in base:
        result = minimize(
            lambda z: nll(x, np.exp(z), eta=eta),
            np.log(start),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6},
        )
        if np.isfinite(result.fun) and result.fun < 1e90:
            fits.append(result)
    if not fits:
        raise RuntimeError("All K2 multistart fits failed.")
    best = min(fits, key=lambda result: result.fun)
    rates = np.exp(best.x)
    return K2Fit(*map(float, rates), float(best.fun), bool(best.success))
