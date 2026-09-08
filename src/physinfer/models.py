"""Sequential single-ON promoter models and converged stationary CME/FSP evaluation.

Public notation follows the manuscript: rows of Q are source states and columns
are destination states.  The row stationary distribution satisfies pi @ Q = 0;
the column-vector CME therefore uses Q.T.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import gammaln


@dataclass(frozen=True)
class SequentialSingleOn:
    """Effective K-state chain G0 <-> ... <-> G(K-1), with only G(K-1) productive."""

    forward_rates: tuple[float, ...]
    reverse_rates: tuple[float, ...]
    ksyn: float
    kdeg: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "forward_rates", tuple(float(v) for v in self.forward_rates))
        object.__setattr__(self, "reverse_rates", tuple(float(v) for v in self.reverse_rates))
        if len(self.forward_rates) != len(self.reverse_rates) or not self.forward_rates:
            raise ValueError("K>=2 and equal numbers of forward/reverse rates are required.")
        values = np.asarray([*self.forward_rates, *self.reverse_rates, self.ksyn, self.kdeg])
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("All kinetic rates must be positive and finite.")

    @property
    def order(self) -> int:
        return len(self.forward_rates) + 1

    @property
    def stationary_promoter(self) -> np.ndarray:
        ratios = np.asarray(self.forward_rates) / np.asarray(self.reverse_rates)
        weights = np.ones(self.order, dtype=float)
        weights[1:] = np.cumprod(ratios)
        return weights / weights.sum()

    @property
    def p_on(self) -> float:
        return float(self.stationary_promoter[-1])

    @property
    def burst_size(self) -> float:
        return float(self.ksyn / self.reverse_rates[-1])

    @property
    def burst_frequency(self) -> float:
        pi = self.stationary_promoter
        return float(pi[-2] * self.forward_rates[-1])

    @property
    def mean(self) -> float:
        return float(self.p_on * self.ksyn / self.kdeg)

    @classmethod
    def from_flat_rates(
        cls, order: int, rates: np.ndarray, *, kdeg: float = 1.0
    ) -> "SequentialSingleOn":
        values = np.asarray(rates, dtype=float)
        expected = 2 * (order - 1) + 1
        if values.shape != (expected,):
            raise ValueError(f"K={order} requires {expected} values.")
        edge_count = order - 1
        return cls(
            tuple(values[:edge_count]),
            tuple(values[edge_count : 2 * edge_count]),
            float(values[-1]),
            kdeg,
        )


def promoter_generator(model: SequentialSingleOn) -> np.ndarray:
    """Return the row-source promoter generator Q used in the manuscript."""
    q = np.zeros((model.order, model.order), dtype=float)
    for edge, (forward, reverse) in enumerate(zip(model.forward_rates, model.reverse_rates)):
        q[edge, edge + 1] = forward
        q[edge + 1, edge] = reverse
    q[np.diag_indices(model.order)] = -q.sum(axis=1)
    if not np.allclose(q.sum(axis=1), 0.0, atol=1e-12):
        raise RuntimeError("Rows of the promoter generator do not sum to zero.")
    return q


def k3_stationary_closed_form(k01: float, k10: float, k12: float, k21: float) -> np.ndarray:
    """Closed-form occupancies for (OFF1, OFF2, ON)."""
    denominator = k10 * k21 + k01 * k21 + k01 * k12
    return np.asarray(
        [k10 * k21, k01 * k21, k01 * k12], dtype=float
    ) / denominator


def joint_generator(model: SequentialSingleOn, max_mrna: int) -> csc_matrix:
    """Build the column-conservative joint promoter/count CME generator."""
    if max_mrna < 1:
        raise ValueError("max_mrna must be positive.")
    k = model.order
    dim = k * (max_mrna + 1)
    generator = lil_matrix((dim, dim), dtype=float)
    q_column = promoter_generator(model).T

    def idx(mrna: int, state: int) -> int:
        return k * mrna + state

    for mrna in range(max_mrna + 1):
        for source in range(k):
            for destination in range(k):
                rate = q_column[destination, source]
                if rate:
                    generator[idx(mrna, destination), idx(mrna, source)] += rate
        if mrna > 0:
            for state in range(k):
                rate = model.kdeg * mrna
                generator[idx(mrna - 1, state), idx(mrna, state)] += rate
                generator[idx(mrna, state), idx(mrna, state)] -= rate
        if mrna < max_mrna:
            generator[idx(mrna + 1, k - 1), idx(mrna, k - 1)] += model.ksyn
            generator[idx(mrna, k - 1), idx(mrna, k - 1)] -= model.ksyn
    out = generator.tocsc()
    if not np.allclose(np.asarray(out.sum(axis=0)).ravel(), 0.0, atol=1e-9):
        raise RuntimeError("The joint CME generator is not probability conserving.")
    return out


def stationary_pmf(
    model: SequentialSingleOn,
    max_mrna: int,
) -> tuple[np.ndarray, float, float]:
    """Solve the truncated stationary CME and return PMF, boundary mass, residual L1."""
    generator = joint_generator(model, max_mrna)
    dim = generator.shape[0]
    system = generator.tolil(copy=True)
    system[-1, :] = np.ones(dim)
    rhs = np.zeros(dim)
    rhs[-1] = 1.0
    stationary = np.asarray(spsolve(system.tocsc(), rhs), dtype=float)
    if np.any(~np.isfinite(stationary)) or stationary.min() < -1e-8:
        raise FloatingPointError("Invalid stationary CME solution.")
    stationary = np.maximum(stationary, 0.0)
    stationary /= stationary.sum()
    pmf = stationary.reshape(max_mrna + 1, model.order).sum(axis=1)
    boundary = float(pmf[-1])
    residual = float(np.linalg.norm(generator @ stationary, ord=1))
    return pmf, boundary, residual


def adaptive_stationary_pmf(
    model: SequentialSingleOn,
    *,
    min_max_mrna: int = 64,
    max_max_mrna: int = 4000,
    boundary_tolerance: float = 1e-9,
    residual_tolerance: float = 1e-7,
    l1_tolerance: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """Enlarge transcript support until boundary, residual, and PMF change are stable."""
    guess = int(math.ceil(model.mean + 12 * math.sqrt(max(model.mean, 1.0)) + 30))
    support = min(max(min_max_mrna, guess), max_max_mrna)
    previous = None
    while True:
        pmf, boundary, residual = stationary_pmf(model, support)
        l1_change = math.inf
        if previous is not None:
            common = min(previous.size, pmf.size)
            l1_change = float(np.abs(pmf[:common] - previous[:common]).sum() + pmf[common:].sum())
        if boundary <= boundary_tolerance and residual <= residual_tolerance and l1_change <= l1_tolerance:
            return pmf, {
                "max_mrna": float(support),
                "boundary_mass": boundary,
                "stationary_residual_l1": residual,
                "pmf_l1_change": l1_change,
            }
        if support >= max_max_mrna:
            raise RuntimeError(
                "Adaptive truncation did not converge: "
                f"boundary={boundary:.3e}, residual={residual:.3e}, L1={l1_change:.3e}."
            )
        previous = pmf
        support = min(max_max_mrna, max(support + 64, 2 * support))


def thin_pmf(pmf: np.ndarray, capture_efficiency: float) -> np.ndarray:
    """Apply binomial capture thinning to a latent stationary count PMF."""
    if not 0 < capture_efficiency <= 1:
        raise ValueError("capture_efficiency must lie in (0,1].")
    latent = np.asarray(pmf, dtype=float)
    latent = latent / latent.sum()
    if capture_efficiency == 1.0:
        return latent.copy()
    eta = capture_efficiency
    observed = np.zeros_like(latent)
    for latent_count, mass in enumerate(latent):
        if mass <= 0:
            continue
        y = np.arange(latent_count + 1)
        log_prob = (
            gammaln(latent_count + 1)
            - gammaln(y + 1)
            - gammaln(latent_count - y + 1)
            + y * math.log(eta)
            + (latent_count - y) * math.log1p(-eta)
        )
        observed[: latent_count + 1] += mass * np.exp(log_prob)
    observed = np.maximum(observed, 0.0)
    return observed / observed.sum()


def _mean_one_lognormal_quadrature(
    extrinsic_cv: float, quadrature_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize a mean-one lognormal factor with the requested CV."""
    if not np.isfinite(extrinsic_cv) or extrinsic_cv < 0:
        raise ValueError("extrinsic_cv must be finite and non-negative.")
    if extrinsic_cv == 0:
        return np.asarray([1.0]), np.asarray([1.0])
    if quadrature_nodes < 3:
        raise ValueError("quadrature_nodes must be at least 3 when extrinsic_cv > 0.")
    sigma2 = math.log1p(extrinsic_cv**2)
    sigma = math.sqrt(sigma2)
    mu = -0.5 * sigma2
    nodes, weights = hermgauss(quadrature_nodes)
    factors = np.exp(mu + math.sqrt(2.0) * sigma * nodes)
    weights = weights / math.sqrt(math.pi)
    weights = weights / weights.sum()
    factors = factors / np.sum(weights * factors)
    return factors, weights


@lru_cache(maxsize=512)
def _extrinsic_ksyn_observed_pmf_cached(
    model: SequentialSingleOn,
    capture_efficiency: float,
    extrinsic_cv: float,
    quadrature_nodes: int,
) -> tuple[float, ...]:
    """Observed PMF under cell-to-cell lognormal ksyn heterogeneity."""
    factors, weights = _mean_one_lognormal_quadrature(extrinsic_cv, quadrature_nodes)
    components: list[tuple[float, np.ndarray]] = []
    support = 0
    for factor, weight in zip(factors, weights):
        perturbed = SequentialSingleOn(
            model.forward_rates,
            model.reverse_rates,
            model.ksyn * float(factor),
            model.kdeg,
        )
        latent, _ = adaptive_stationary_pmf(perturbed)
        observed = thin_pmf(latent, capture_efficiency)
        components.append((float(weight), observed))
        support = max(support, observed.size)
    mixture = np.zeros(support, dtype=float)
    for weight, component in components:
        mixture[: component.size] += weight * component
    mixture = np.maximum(mixture, 0.0)
    mixture /= mixture.sum()
    return tuple(map(float, mixture))


def sample_observed_counts(
    model: SequentialSingleOn,
    n_cells: int,
    *,
    capture_efficiency: float,
    seed: int,
    extrinsic_cv: float = 0.0,
    quadrature_nodes: int = 7,
) -> np.ndarray:
    """Draw independent stationary counts, optionally with extrinsic ksyn noise.

    ``extrinsic_cv`` is the CV of a mean-one lognormal, cell-specific multiplier
    on ``k_syn``.  It is distinct from the empirical CV of the sampled counts.
    The default value 0.0 follows the original noise-free parameter protocol.
    """
    if n_cells < 1:
        raise ValueError("n_cells must be positive.")
    if extrinsic_cv == 0:
        latent, _ = adaptive_stationary_pmf(model)
        observed = thin_pmf(latent, capture_efficiency)
    else:
        observed = np.asarray(
            _extrinsic_ksyn_observed_pmf_cached(
                model,
                float(capture_efficiency),
                float(extrinsic_cv),
                int(quadrature_nodes),
            ),
            dtype=float,
        )
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(observed.size), size=n_cells, replace=True, p=observed)
