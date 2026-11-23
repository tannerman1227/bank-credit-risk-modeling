import pandas as pd

from credit_risk.expected_loss import exposure_at_default, expected_loss
from credit_risk.features import engineer_features
from credit_risk.sql_analysis import load_queries, run_queries
from tests.synthetic import make_book


def test_named_queries_run_and_snapshot_matches_the_book():
    book = engineer_features(make_book(80, seed=6))
    ead = exposure_at_default(book["limit_bal"], book["bill_amt1"], ccf=0.1)
    pd_hat = book["pay_1"].clip(lower=0) / 10
    scored = pd.DataFrame(
        {
            "account_id": book["account_id"],
            "risk_grade": pd.qcut(book["pay_1"].rank(method="first"), 5, labels=False) + 1,
            "pd": pd_hat,
            "default_flag": book["default_flag"],
            "ead": ead,
            "expected_loss": expected_loss(pd_hat, 0.75, ead),
            "limit_bal": book["limit_bal"],
            "bill_amt1": book["bill_amt1"],
        }
    )
    queries = load_queries()
    assert "portfolio_snapshot" in queries
    assert "expected_loss_by_risk_grade" in queries
    results = run_queries(book, scored, queries)
    snapshot = results["portfolio_snapshot"].iloc[0]
    assert int(snapshot["accounts"]) == len(book)
    assert abs(snapshot["default_rate"] - book["default_flag"].mean()) < 1e-4
    grades = results["expected_loss_by_risk_grade"]
    assert set(grades["risk_grade"]) <= {1, 2, 3, 4, 5}
    concentration = results["expected_loss_concentration"].iloc[0]
    assert 0 < concentration["share_of_el_in_highest_grade"] <= 1
