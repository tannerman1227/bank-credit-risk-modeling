import numpy as np

from credit_risk.metrics import (
    expected_calibration_error,
    gini_coefficient,
    ks_statistic,
    population_stability_index,
    roc_auc,
)


def test_perfect_separation_has_auc_gini_and_ks_of_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert roc_auc(y, score) == 1
    assert gini_coefficient(y, score) == 1
    assert ks_statistic(y, score) == 1


def test_gini_is_twice_auc_minus_one():
    y = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    score = np.array([0.2, 0.8, 0.4, 0.55, 0.33, 0.9, 0.1, 0.6])
    assert gini_coefficient(y, score) == 2 * roc_auc(y, score) - 1


def test_ece_is_zero_when_bin_predictions_match_outcomes():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probability = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    assert expected_calibration_error(y, probability, n_bins=2) == 0


def test_identical_populations_have_near_zero_psi():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=2000)
    psi = population_stability_index(reference, reference)
    assert psi < 1e-6
