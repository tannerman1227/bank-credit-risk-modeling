import numpy as np
import pandas as pd

from credit_risk.expected_loss import (
    assign_risk_grades,
    cutoff_curve,
    exposure_at_default,
    expected_loss,
)


def test_exposure_handles_credit_balances_and_overlimit():
    ead = exposure_at_default(
        limit_bal=[200, 200, 200],
        bill_amt=[100, -50, 250],
        ccf=0.10,
    )
    np.testing.assert_allclose(ead, [110, 20, 250])


def test_expected_loss_is_the_product():
    el = expected_loss(pd=0.10, lgd=0.50, ead=1000)
    assert el == 50


def test_grades_are_equal_counts_with_one_as_safest():
    score = np.arange(100, dtype=float)  # higher = riskier
    grades = assign_risk_grades(score, n_grades=5)
    assert set(grades) == {1, 2, 3, 4, 5}
    assert grades[0] == 1
    assert grades[-1] == 5
    counts = pd.Series(grades).value_counts()
    assert counts.max() - counts.min() <= 1


def test_stricter_cutoff_does_not_raise_the_bad_rate_when_rank_is_perfect():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score = np.array([0.05, 0.10, 0.15, 0.20, 0.60, 0.70, 0.80, 0.90])
    el = score * 100
    ead = np.full(len(y), 100.0)
    curve = cutoff_curve(y, score, el, ead, grid=(0.50, 0.75, 1.00))
    bad_rates = curve["observed_bad_rate"].to_list()
    assert bad_rates[0] <= bad_rates[1] <= bad_rates[2]
    assert curve.iloc[0]["approval_rate"] == 0.5
    assert curve.iloc[0]["observed_bad_rate"] == 0
