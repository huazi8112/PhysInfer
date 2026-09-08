"""Exact/convergence-checked likelihood fitting for adjacent effective orders."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize

from .models import SequentialSingleOn, adaptive_stationary_pmf, thin_pmf
from .poisson_beta import fit_multistart as fit_k2_multistart
from .poisson_beta import logpmf as k2_logpmf


@dataclass(frozen=True)
class OrderFit:
    order: int
    rates: np.ndarray
    nll: float
    success: bool
    max_mrna: int | None
    boundary_mass: float | None

    @property
    def model(self) -> SequentialSingleOn:
        return SequentialSingleOn.from_flat_rates(self.order, self.rates)


def observed_logpmf(model: SequentialSingleOn, counts: np.ndarray, eta: float) -> np.ndarray:
    x = np.asarray(counts, dtype=int)
    if model.order == 2:
        return k2_logpmf(x, model.forward_rates[0], model.reverse_rates[0], model.ksyn, eta=eta)
    latent, _ = adaptive_stationary_pmf(model)
    observed = thin_pmf(latent, eta)
    if x.max(initial=0) >= observed.size:
        return np.full(x.shape, -math.inf)
    return np.log(np.maximum(observed[x], 1e-300))


def heldout_nll(fit: OrderFit, counts: np.ndarray, eta: float) -> float:
    values = observed_logpmf(fit.model, np.asarray(counts, dtype=int), eta)
    return -float(np.sum(values)) if np.all(np.isfinite(values)) else 1e100


def fit_order(
    counts: np.ndarray,
    order: int,
    *,
    eta: float,
    seed: int,
    n_random_starts: int = 6,
    n_base_starts: int | None = None,
    maxiter: int = 350,
) -> OrderFit:
    """Fit one prespecified effective order on training counts."""
    x = np.rint(np.asarray(counts, dtype=float)).astype(int)
    if order == 2:
        fit = fit_k2_multistart(
            x,
            eta=eta,
            seed=seed,
            n_random_starts=n_random_starts,
            n_deterministic_starts=(5 if n_base_starts is None else n_base_starts),
            maxiter=maxiter,
        )
        rates = np.asarray([fit.kon, fit.koff, fit.ksyn])
        return OrderFit(2, rates, fit.nll, fit.success, None, None)

    n_rates = 2 * (order - 1) + 1
    mean = max(float(np.mean(x)) / eta, 1e-3)
    rng = np.random.default_rng(seed)
    starts = []
    k3_base_starts = 4 if n_base_starts is None else n_base_starts
    if k3_base_starts < 1:
        raise ValueError("n_base_starts must be positive.")
    for index in range(n_random_starts + k3_base_starts):
        edges = 10 ** rng.uniform(-1.3, 1.0, n_rates - 1)
        occupancy_guess = 1 / order
        ksyn = np.clip(mean / occupancy_guess * 10 ** rng.uniform(-0.4, 0.4), 0.05, 1500)
        starts.append(np.log(np.r_[edges, ksyn]))
    bounds = [(math.log(1e-3), math.log(300))] * (n_rates - 1) + [(math.log(0.02), math.log(1500))]
    diagnostics = {}

    def objective(log_rates: np.ndarray) -> float:
        nonlocal diagnostics
        try:
            model = SequentialSingleOn.from_flat_rates(order, np.exp(log_rates))
            latent, diagnostics = adaptive_stationary_pmf(model)
            observed = thin_pmf(latent, eta)
            if x.max(initial=0) >= observed.size:
                return 1e100
            return -float(np.log(np.maximum(observed[x], 1e-300)).sum())
        except Exception:
            return 1e100

    fits = []
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-5},
        )
        if np.isfinite(result.fun) and result.fun < 1e90:
            fits.append(result)
    if not fits:
        raise RuntimeError(f"All K={order} multistart fits failed.")
    best = min(fits, key=lambda result: result.fun)
    model = SequentialSingleOn.from_flat_rates(order, np.exp(best.x))
    _, final_diagnostics = adaptive_stationary_pmf(model)
    return OrderFit(
        order,
        np.exp(best.x),
        float(best.fun),
        bool(best.success),
        int(final_diagnostics["max_mrna"]),
        float(final_diagnostics["boundary_mass"]),
    )


def adjacent_heldout_gain(
    counts: np.ndarray,
    lower_order: int,
    *,
    eta: float,
    seed: int,
    train_fraction: float = 0.7,
    n_random_starts: int = 6,
    n_base_starts: int | None = None,
    maxiter: int = 350,
) -> dict[str, float]:
    """Fit adjacent orders on training cells and return held-out per-cell gain."""
    x = np.asarray(counts, dtype=int)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(x.size)
    split = max(20, min(x.size - 10, int(train_fraction * x.size)))
    train, test = x[indices[:split]], x[indices[split:]]
    lower = fit_order(
        train,
        lower_order,
        eta=eta,
        seed=seed,
        n_random_starts=n_random_starts,
        n_base_starts=n_base_starts,
        maxiter=maxiter,
    )
    higher = fit_order(
        train,
        lower_order + 1,
        eta=eta,
        seed=seed + 100003,
        n_random_starts=n_random_starts,
        n_base_starts=n_base_starts,
        maxiter=maxiter,
    )
    lower_test = heldout_nll(lower, test, eta)
    higher_test = heldout_nll(higher, test, eta)
    return {
        "lower_order": float(lower_order),
        "higher_order": float(lower_order + 1),
        "n_train": float(train.size),
        "n_test": float(test.size),
        "lower_test_nll": lower_test,
        "higher_test_nll": higher_test,
        "gain_per_test_cell": (lower_test - higher_test) / test.size,
    }
