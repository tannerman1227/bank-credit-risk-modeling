"""Small books with the cleaned schema, for tests that should not download data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import EDUCATION_LABELS, MARRIAGE_LABELS, SEX_LABELS


def make_book(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pay = rng.choice([-2, -1, 0, 1, 2, 3], size=(n, 6), p=[0.08, 0.12, 0.50, 0.15, 0.10, 0.05])
    limit = rng.choice([20000, 50000, 80000, 140000, 240000], size=n).astype(float)
    util = np.clip(rng.normal(0.4, 0.35, size=(n, 6)), -0.1, 1.6)
    bill = util * limit[:, None]
    pay_amt = np.clip(bill, 0, None) * rng.uniform(0, 1.1, size=(n, 6))
    education = rng.choice([1, 2, 3, 4], size=n, p=[0.25, 0.45, 0.22, 0.08])
    sex = rng.choice([1, 2], size=n)
    marriage = rng.choice([1, 2, 3], size=n, p=[0.45, 0.48, 0.07])
    age = rng.integers(21, 70, size=n)
    logit = (
        -2.3
        + 0.95 * np.clip(pay[:, 0], 0, 4)
        + 0.25 * (pay[:, 1] >= 1)
        + 1.15 * np.clip(util[:, 0], 0, 1.5)
    )
    probability = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, probability)
    frame = {
        "account_id": np.arange(1, n + 1),
        "limit_bal": limit,
        "sex": sex,
        "education": education,
        "marriage": marriage,
        "age": age,
        "default_flag": default,
        "sex_label": pd.Series(sex).map(SEX_LABELS).to_numpy(),
        "education_label": pd.Series(education).map(EDUCATION_LABELS).to_numpy(),
        "marriage_label": pd.Series(marriage).map(MARRIAGE_LABELS).to_numpy(),
    }
    for i in range(6):
        frame[f"pay_{i + 1}"] = pay[:, i].astype(int)
        frame[f"bill_amt{i + 1}"] = bill[:, i]
        frame[f"pay_amt{i + 1}"] = pay_amt[:, i]
    return pd.DataFrame(frame)
