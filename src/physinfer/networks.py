"""Model-order-specific PhysInfer neural branches."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Install the 'neural' dependencies to use PhysInfer neural branches.") from exc


class _Branch(nn.Module):
    def __init__(self, hidden: tuple[int, int, int], output_dim: int):
        super().__init__()
        h1, h2, h3 = hidden
        self.layers = nn.Sequential(
            nn.Linear(15, h1),
            nn.LayerNorm(h1),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(h3, output_dim),
        )

    def forward(self, features):
        return self.layers(features)


class K2Branch(_Branch):
    """Outputs unconstrained coordinates for p_on, k_off, and k_syn."""

    def __init__(self):
        super().__init__((128, 128, 64), 3)

    @staticmethod
    def physical_map(raw):
        p_on = torch.sigmoid(raw[..., 0]).clamp(1e-4, 1 - 1e-4)
        k_off = torch.nn.functional.softplus(raw[..., 1]) + 1e-4
        k_syn = torch.nn.functional.softplus(raw[..., 2]) + 1e-4
        k_on = k_off * p_on / (1 - p_on)
        return {"p_on": p_on, "k_on": k_on, "k_off": k_off, "k_syn": k_syn}


class K3Branch(_Branch):
    """Outputs k01, k10, k12, k21, and BS; k_syn is reconstructed as BS*k21."""

    def __init__(self):
        super().__init__((192, 192, 64), 5)

    @staticmethod
    def physical_map(raw):
        positive = torch.nn.functional.softplus(raw) + 1e-4
        k01, k10, k12, k21, burst_size = positive.unbind(-1)
        k_syn = burst_size * k21
        denominator = k10 * k21 + k01 * k21 + k01 * k12
        p_on = k01 * k12 / denominator
        burst_frequency = p_on * k21
        return {
            "k01": k01,
            "k10": k10,
            "k12": k12,
            "k21": k21,
            "k_syn": k_syn,
            "p_on": p_on,
            "burst_size": burst_size,
            "burst_frequency": burst_frequency,
        }

