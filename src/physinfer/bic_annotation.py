"""Secondary BIC-plus-veto annotations that never overwrite primary routing."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SecondaryBICAnnotation:
    """Result of the manuscript secondary model-order preference analysis."""

    primary_call: str
    bic_2state: float
    bic_3state: float
    delta_bic: float
    rate_separation: float | None
    preference: str
    reason: str


def bic_from_nll(nll: float, n_parameters: int, n_observations: int) -> float:
    """Compute BIC from a *total* negative log-likelihood."""
    if n_parameters < 0:
        raise ValueError("n_parameters must be non-negative")
    if n_observations <= 0:
        raise ValueError("n_observations must be positive")
    return 2.0 * float(nll) + int(n_parameters) * log(int(n_observations))


def rate_separation_degree(rates: Iterable[float], eps: float = 1e-6) -> float:
    """Return max(kappa_3)/(min(kappa_3)+eps) for positive K3 transition rates."""
    values = np.asarray(tuple(rates), dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("rates must contain four finite positive values: k01,k10,k12,k21")
    return float(values.max() / (values.min() + eps))


def annotate_secondary_bic_preference(
    *,
    total_nll_2state: float,
    total_nll_3state: float,
    n_parameters_2state: int,
    n_parameters_3state: int,
    n_observations: int,
    k3_rates: Iterable[float],
    primary_call: str = "ambiguous",
    tau: float = 5.0,
    gamma: float = 2.5,
) -> SecondaryBICAnnotation:
    """Apply the frozen secondary BIC rule while preserving ``primary_call``.

    The manuscript convention is delta_BIC = BIC_2state - BIC_3state. Values below
    ``tau`` imply a 2-state preference. At or above ``tau``, the K3 candidate must
    also pass the transition-rate separation veto to receive a 3-state preference.
    """
    bic2 = bic_from_nll(total_nll_2state, n_parameters_2state, n_observations)
    bic3 = bic_from_nll(total_nll_3state, n_parameters_3state, n_observations)
    delta = bic2 - bic3
    if delta <= 0:
        return SecondaryBICAnnotation(
            primary_call, bic2, bic3, delta, None, "2-state preference", "delta_BIC<=0"
        )
    if delta < tau:
        return SecondaryBICAnnotation(
            primary_call, bic2, bic3, delta, None, "2-state preference", "0<delta_BIC<tau"
        )
    separation = rate_separation_degree(k3_rates)
    if separation < gamma:
        return SecondaryBICAnnotation(
            primary_call,
            bic2,
            bic3,
            delta,
            separation,
            "2-state preference",
            "delta_BIC>=tau but rate_separation<gamma",
        )
    return SecondaryBICAnnotation(
        primary_call,
        bic2,
        bic3,
        delta,
        separation,
        "3-state preference",
        "delta_BIC>=tau and rate_separation>=gamma",
    )
