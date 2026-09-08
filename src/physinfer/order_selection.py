"""Calibrated held-out adjacent-order evidence with selective abstention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SelectiveDecision:
    lower_order: int
    higher_order: int
    p_lower: float
    p_higher: float
    call: str
    null_threshold_median: float
    n_joint_bootstrap: int


def joint_bootstrap_decision(
    unit_gains: np.ndarray,
    matched_null_gains: np.ndarray,
    *,
    lower_order: int,
    n_draws: int = 3000,
    null_quantile: float = 0.90,
    q_high: float = 0.80,
    q_low: float = 0.20,
    probability_cutoff: float = 0.80,
    seed: int = 1,
) -> SelectiveDecision:
    """Propagate null-threshold and repeated-held-out uncertainty jointly."""
    gains = np.asarray(unit_gains, dtype=float)
    null = np.asarray(matched_null_gains, dtype=float)
    gains = gains[np.isfinite(gains)]
    null = null[np.isfinite(null)]
    if gains.size < 2 or null.size < 10:
        raise ValueError("Insufficient repeated gains or matched-null gains.")
    rng = np.random.default_rng(seed)
    q_values = np.empty(n_draws)
    thresholds = np.empty(n_draws)
    for draw in range(n_draws):
        threshold = float(np.quantile(rng.choice(null, size=null.size, replace=True), null_quantile))
        resampled = rng.choice(gains, size=gains.size, replace=True)
        thresholds[draw] = threshold
        q_values[draw] = np.mean(resampled > threshold)
    p_higher = float(np.mean(q_values >= q_high))
    p_lower = float(np.mean(q_values <= q_low))
    if p_higher >= probability_cutoff:
        call = f"K{lower_order + 1}"
    elif p_lower >= probability_cutoff:
        call = f"K{lower_order}"
    else:
        call = "ambiguous"
    return SelectiveDecision(
        lower_order,
        lower_order + 1,
        p_lower,
        p_higher,
        call,
        float(np.median(thresholds)),
        n_draws,
    )


def sequential_adjacent_search(
    evaluate_boundary: Callable[[int], SelectiveDecision],
    *,
    starting_order: int = 2,
) -> tuple[int | None, list[SelectiveDecision]]:
    """Advance while an added effective state has stable support; otherwise stop/abstain.

    No universal methodological Kmax is imposed here.  A caller may enforce a
    resource policy externally, but an unresolved boundary must remain ambiguous.
    """
    order = starting_order
    history = []
    while True:
        decision = evaluate_boundary(order)
        history.append(decision)
        if decision.call == f"K{order + 1}":
            order += 1
            continue
        if decision.call == f"K{order}":
            return order, history
        return None, history

