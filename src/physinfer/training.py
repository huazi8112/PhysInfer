"""Compact direct-backprop implementation for paired PhysInfer loss experiments.

The routine is intentionally transparent and intended as reproducibility/base
code.  It maps the 15 exchangeable-distribution features to physical rates and
backpropagates through a dense stationary CME solve.
"""

from __future__ import annotations

import numpy as np
import torch

from .features import extract_features15
from .losses import full_branch_loss, stationary_nll
from .networks import K2Branch, K3Branch


def _row_generator(order: int, forward, reverse):
    q = torch.zeros((order, order), dtype=forward.dtype, device=forward.device)
    for edge in range(order - 1):
        q[edge, edge + 1] = forward[edge]
        q[edge + 1, edge] = reverse[edge]
    q = q - torch.diag(q.sum(dim=1))
    return q


def differentiable_stationary_pmf(order: int, mapped: dict, max_mrna: int) -> torch.Tensor:
    """Solve the normalized stationary column-CME system with a reflecting truncation."""
    if order == 2:
        forward = torch.stack([mapped["k_on"]])
        reverse = torch.stack([mapped["k_off"]])
    elif order == 3:
        forward = torch.stack([mapped["k01"], mapped["k12"]])
        reverse = torch.stack([mapped["k10"], mapped["k21"]])
    else:
        raise ValueError("Neural branches are instantiated for K2 and K3 in the reported analyses.")
    q_column = _row_generator(order, forward, reverse).T
    dim = order * (max_mrna + 1)
    generator = torch.zeros((dim, dim), dtype=forward.dtype, device=forward.device)

    def index(mrna: int, state: int) -> int:
        return order * mrna + state

    for mrna in range(max_mrna + 1):
        block = slice(order * mrna, order * (mrna + 1))
        generator[block, block] = generator[block, block] + q_column
        if mrna > 0:
            rate = torch.as_tensor(float(mrna), dtype=forward.dtype, device=forward.device)
            for state in range(order):
                generator[index(mrna - 1, state), index(mrna, state)] += rate
                generator[index(mrna, state), index(mrna, state)] -= rate
        if mrna < max_mrna:
            ksyn = mapped["k_syn"]
            generator[index(mrna + 1, order - 1), index(mrna, order - 1)] += ksyn
            generator[index(mrna, order - 1), index(mrna, order - 1)] -= ksyn
    system = generator.clone()
    system[-1, :] = 1.0
    rhs = torch.zeros(dim, dtype=forward.dtype, device=forward.device)
    rhs[-1] = 1.0
    stationary = torch.linalg.solve(system, rhs).clamp_min(0)
    stationary = stationary / stationary.sum()
    return stationary.reshape(max_mrna + 1, order).sum(dim=1)


def differentiable_thin_pmf(pmf: torch.Tensor, capture_efficiency: float) -> torch.Tensor:
    """Differentiable binomial thinning of a latent count PMF."""
    eta = float(capture_efficiency)
    if not 0 < eta <= 1:
        raise ValueError("capture efficiency must lie in (0,1].")
    if eta == 1.0:
        return pmf
    maximum = pmf.numel() - 1
    observed = torch.zeros_like(pmf)
    log_eta = torch.log(torch.as_tensor(eta, dtype=pmf.dtype, device=pmf.device))
    log_one_minus = torch.log1p(torch.as_tensor(-eta, dtype=pmf.dtype, device=pmf.device))
    for latent_count in range(maximum + 1):
        y = torch.arange(latent_count + 1, dtype=pmf.dtype, device=pmf.device)
        n = torch.as_tensor(float(latent_count), dtype=pmf.dtype, device=pmf.device)
        log_binomial = (
            torch.lgamma(n + 1)
            - torch.lgamma(y + 1)
            - torch.lgamma(n - y + 1)
            + y * log_eta
            + (n - y) * log_one_minus
        )
        observed[: latent_count + 1] = observed[: latent_count + 1] + pmf[latent_count] * torch.exp(log_binomial)
    return observed / observed.sum()


def _standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (features - center) / scale, center, scale


def train_branch(
    counts_matrix: np.ndarray,
    order: int,
    *,
    retained_auxiliaries: bool,
    seed: int,
    epochs: int = 250,
    learning_rate: float = 2e-3,
    max_mrna: int | None = None,
    capture_efficiencies: np.ndarray | None = None,
) -> dict:
    """Train one paired branch and return the model, scaler, and epoch history."""
    torch.manual_seed(seed)
    counts = np.asarray(counts_matrix, dtype=int)
    features = np.vstack([extract_features15(row) for row in counts])
    standardized, center, scale = _standardize(features)
    feature_tensor = torch.tensor(standardized, dtype=torch.float64)
    count_tensors = [torch.tensor(row, dtype=torch.long) for row in counts]
    if capture_efficiencies is None:
        capture_efficiencies = np.ones(counts.shape[0], dtype=float)
    capture_efficiencies = np.asarray(capture_efficiencies, dtype=float)
    if capture_efficiencies.shape != (counts.shape[0],):
        raise ValueError("capture_efficiencies must contain one value per parameter unit.")
    model = (K2Branch() if order == 2 else K3Branch()).to(dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if max_mrna is None:
        max_mrna = max(80, int(np.max(counts)) + 40)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        raw = model(feature_tensor)
        losses = []
        for unit in range(counts.shape[0]):
            mapped = model.physical_map(raw[unit])
            latent_pmf = differentiable_stationary_pmf(order, mapped, max_mrna)
            pmf = differentiable_thin_pmf(latent_pmf, capture_efficiencies[unit])
            if retained_auxiliaries:
                loss = full_branch_loss(
                    order,
                    pmf,
                    count_tensors[unit],
                    burst_size=(mapped["k_syn"] / mapped["k_off"] if order == 2 else mapped["burst_size"]),
                )["total"]
            else:
                loss = stationary_nll(pmf, count_tensors[unit])
            losses.append(loss)
        objective = torch.stack(losses).mean()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        history.append(float(objective.detach()))
    return {
        "model": model,
        "feature_center": center,
        "feature_scale": scale,
        "history": np.asarray(history),
        "max_mrna": max_mrna,
    }
