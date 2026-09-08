import numpy as np
import torch

from physinfer.bic_annotation import annotate_secondary_bic_preference
from physinfer.features import extract_features15
from physinfer.losses import zero_count_penalty
from physinfer.models import (
    SequentialSingleOn,
    adaptive_stationary_pmf,
    k3_stationary_closed_form,
    promoter_generator,
    sample_observed_counts,
)
from physinfer.order_selection import joint_bootstrap_decision


def test_k3_row_generator_and_closed_form_agree():
    model = SequentialSingleOn((0.4, 1.2), (0.7, 0.9), 8.0)
    q = promoter_generator(model)
    assert np.allclose(q.sum(axis=1), 0.0)
    expected = k3_stationary_closed_form(0.4, 0.7, 1.2, 0.9)
    assert np.allclose(model.stationary_promoter, expected)
    assert np.allclose(expected @ q, 0.0, atol=1e-12)
    assert np.isclose(model.burst_size, 8.0 / 0.9)
    assert np.isclose(model.burst_frequency, expected[1] * 1.2)
    assert np.isclose(model.burst_frequency, expected[2] * 0.9)


def test_adaptive_stationary_pmf_is_normalized():
    model = SequentialSingleOn((0.3,), (0.8,), 7.0)
    pmf, diagnostic = adaptive_stationary_pmf(model)
    assert np.isclose(pmf.sum(), 1.0)
    assert np.all(pmf >= 0)
    assert diagnostic["boundary_mass"] <= 1e-9


def test_features_are_exchangeable():
    counts = np.asarray([0, 1, 1, 2, 3, 0, 4, 1, 0, 2, 5, 1])
    assert np.allclose(extract_features15(counts), extract_features15(counts[::-1]))


def test_joint_bootstrap_can_abstain():
    gains = np.linspace(-0.004, 0.004, 30)
    null = np.linspace(-0.01, 0.01, 300)
    decision = joint_bootstrap_decision(gains, null, lower_order=2, n_draws=500, seed=11)
    assert decision.call in {"K2", "K3", "ambiguous"}
    assert 0 <= decision.p_lower <= 1
    assert 0 <= decision.p_higher <= 1


def test_optional_extrinsic_noise_preserves_valid_counts_and_default_path():
    model = SequentialSingleOn((0.4,), (0.8,), 7.0)
    baseline_a = sample_observed_counts(model, 80, capture_efficiency=0.3, seed=17)
    baseline_b = sample_observed_counts(
        model, 80, capture_efficiency=0.3, seed=17, extrinsic_cv=0.0
    )
    noisy = sample_observed_counts(
        model,
        80,
        capture_efficiency=0.3,
        seed=17,
        extrinsic_cv=0.20,
        quadrature_nodes=3,
    )
    assert np.array_equal(baseline_a, baseline_b)
    assert noisy.shape == baseline_a.shape
    assert np.issubdtype(noisy.dtype, np.integer)
    assert np.all(noisy >= 0)


def test_zero_count_anchor_uses_log_probability_scale():
    predicted = torch.tensor([0.20, 0.80], dtype=torch.float64)
    empirical = torch.tensor([0.25, 0.75], dtype=torch.float64)
    expected = (torch.log(predicted[0] + 1e-5) - torch.log(empirical[0] + 1e-5)).square()
    assert torch.allclose(zero_count_penalty(predicted, empirical), expected)


def test_secondary_bic_annotation_does_not_change_primary_call():
    result = annotate_secondary_bic_preference(
        total_nll_2state=125.0,
        total_nll_3state=100.0,
        n_parameters_2state=3,
        n_parameters_3state=5,
        n_observations=500,
        k3_rates=(0.1, 1.0, 0.8, 0.5),
        primary_call="ambiguous",
    )
    assert result.preference == "3-state preference"
    assert result.primary_call == "ambiguous"
