import pandas as pd

from credit_risk.config import GBM_FEATURES, PROTECTED_COLUMNS, SCORECARD_FEATURES
from credit_risk.data import clean
from credit_risk.features import engineer_features
from tests.synthetic import make_book


def _raw_row():
    row = {
        "ID": 1,
        "LIMIT_BAL": 20000,
        "SEX": 2,
        "EDUCATION": 0,
        "MARRIAGE": 0,
        "AGE": 24,
        "PAY_0": 2,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": -1,
        "PAY_6": -2,
        "default payment next month": 1,
    }
    for i in range(1, 7):
        row[f"BILL_AMT{i}"] = 1000
        row[f"PAY_AMT{i}"] = 200
    return row


def test_clean_recodes_undocumented_levels_and_renames_pay_0():
    frame = pd.DataFrame([_raw_row(), {**_raw_row(), "ID": 2, "EDUCATION": 5, "MARRIAGE": 1, "default payment next month": 0}])
    cleaned = clean(frame)
    assert list(cleaned["pay_1"]) == [2, 2]
    assert "pay_0" not in cleaned.columns
    assert set(cleaned["education"]) == {4}
    assert cleaned.loc[cleaned["account_id"] == 1, "marriage"].item() == 3
    assert cleaned.attrs["cleaning_report"]["education_undocumented"] == 2
    assert cleaned.attrs["cleaning_report"]["marriage_undocumented"] == 1


def test_utilization_and_delinquency_features():
    book = engineer_features(make_book(40, seed=1))
    row = book.iloc[0]
    assert row["util_1"] == row["bill_amt1"] / row["limit_bal"]
    assert row["months_late"] == int((row[[f"pay_{i}" for i in range(1, 7)]] >= 1).sum())
    assert 0 <= row["pay_ratio_1"] <= 5 or pd.isna(row["pay_ratio_1"])


def test_models_do_not_receive_protected_attributes_or_the_target():
    banned = set(PROTECTED_COLUMNS) | {"default_flag", "account_id"}
    assert banned.isdisjoint(SCORECARD_FEATURES)
    assert banned.isdisjoint(GBM_FEATURES)
