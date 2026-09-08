"""Branch-specific stationary objective and separately weighted auxiliaries."""

from __future__ import annotations

import torch


def empirical_pmf(counts: torch.Tensor, max_mrna: int) -> torch.Tensor:
    values = counts.long().clamp(0, max_mrna)
    return torch.bincount(values, minlength=max_mrna + 1).to(torch.float64) / values.numel()


def stationary_nll(
    predicted_pmf: torch.Tensor, counts: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Per-cell stationary negative log-likelihood used by both branches."""
    probabilities = predicted_pmf[counts.long()].clamp_min(eps)
    return -torch.log(probabilities).mean()


def zero_count_penalty(
    predicted_pmf: torch.Tensor, empirical: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Squared log-probability discrepancy at zero counts."""
    return (
        torch.log(predicted_pmf[0] + eps) - torch.log(empirical[0] + eps)
    ).square()


def k2_shape_penalty(predicted_pmf: torch.Tensor, empirical: torch.Tensor) -> torch.Tensor:
    support = torch.arange(predicted_pmf.numel(), dtype=predicted_pmf.dtype, device=predicted_pmf.device)
    predicted_m3 = torch.sum(predicted_pmf * support.pow(3))
    empirical_m3 = torch.sum(empirical * support.pow(3))
    return (torch.log1p(predicted_m3) - torch.log1p(empirical_m3)).square()


def k2_burst_proxy_penalty(
    burst_size: torch.Tensor, counts: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    x = counts.to(torch.float64)
    f_obs = x.var(unbiased=False) / (x.mean() + eps)
    f_star = torch.maximum(f_obs, torch.as_tensor(1.05, dtype=x.dtype, device=x.device))
    proxy = torch.maximum(f_star - 1, torch.as_tensor(0.05, dtype=x.dtype, device=x.device))
    return (torch.log1p(burst_size) - torch.log1p(proxy)).square()


def k3_tail_penalty(predicted_pmf: torch.Tensor, empirical: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    threshold = int(torch.floor(counts.to(torch.float64).mean()).item()) + 1
    predicted_cdf = torch.cumsum(predicted_pmf, dim=0)
    empirical_cdf = torch.cumsum(empirical, dim=0)
    if threshold >= predicted_pmf.numel():
        return torch.zeros((), dtype=predicted_pmf.dtype, device=predicted_pmf.device)
    return torch.mean((predicted_cdf[threshold:] - empirical_cdf[threshold:]).square())


def full_branch_loss(
    order: int,
    predicted_pmf: torch.Tensor,
    counts: torch.Tensor,
    *,
    burst_size: torch.Tensor,
    auxiliary_coefficient: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Return total and named components; 0.01 is applied to each applicable auxiliary."""
    empirical = empirical_pmf(counts, predicted_pmf.numel() - 1).to(predicted_pmf)
    primary = stationary_nll(predicted_pmf, counts)
    zero = zero_count_penalty(predicted_pmf, empirical)
    total = primary + auxiliary_coefficient * zero
    components = {"stationary_nll": primary, "zero": zero}
    if order == 2:
        shape = k2_shape_penalty(predicted_pmf, empirical)
        burst = k2_burst_proxy_penalty(burst_size, counts)
        total = total + auxiliary_coefficient * shape + auxiliary_coefficient * burst
        components.update({"shape": shape, "burst_proxy": burst})
    elif order == 3:
        tail = k3_tail_penalty(predicted_pmf, empirical, counts)
        total = total + auxiliary_coefficient * tail
        components["tail"] = tail
    components["total"] = total
    return components
