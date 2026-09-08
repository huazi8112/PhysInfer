#!/usr/bin/env python3
"""Fast convention, normalization, and selective-routing checks."""

from __future__ import annotations

import numpy as np

from physinfer.models import (
    SequentialSingleOn,
    adaptive_stationary_pmf,
    k3_stationary_closed_form,
    promoter_generator,
)
from physinfer.order_selection import joint_bootstrap_decision


def main() -> None:
    k3 = SequentialSingleOn((0.4, 1.2), (0.7, 0.9), 8.0)
    q = promoter_generator(k3)
    assert np.allclose(q.sum(axis=1), 0.0)
    assert np.allclose(k3.stationary_promoter, k3_stationary_closed_form(0.4, 0.7, 1.2, 0.9))
    assert np.isclose(k3.burst_size, 8.0 / 0.9)
    assert np.isclose(k3.burst_frequency, k3.p_on * 0.9)
    pmf, diagnostic = adaptive_stationary_pmf(k3)
    assert np.isclose(pmf.sum(), 1.0)
    assert diagnostic["boundary_mass"] <= 1e-9
    decision = joint_bootstrap_decision(
        np.linspace(0.02, 0.04, 30),
        np.linspace(-0.01, 0.01, 200),
        lower_order=2,
        n_draws=500,
        seed=7,
    )
    assert decision.call == "K3"
    print("PhysInfer validation passed.")
    print(f"K3 states=(OFF1, OFF2, ON); BS={k3.burst_size:.6g}; BF={k3.burst_frequency:.6g}")
    print(diagnostic)


if __name__ == "__main__":
    main()

