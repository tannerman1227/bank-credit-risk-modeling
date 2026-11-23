"""Row-wise feature engineering. No statistics are fit here, so there is no leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add utilization, payment ratios, and delinquency summaries.

    ``pay_ratio`` is missing when the statement balance is not positive
    (no bill to pay). Ratios are capped at 5 so a tiny bill does not create
    an extreme payment multiple. Utilization is left uncapped apart from
    replacing a non-positive limit, which cleaning already removes.
    """
    out = df.copy()
    limit = out["limit_bal"].astype(float)
    pay_cols = [f"pay_{i}" for i in range(1, 7)]
    util_cols = []
    ratio_cols = []
    for i in range(1, 7):
        util_col = f"util_{i}"
        ratio_col = f"pay_ratio_{i}"
        bill = out[f"bill_amt{i}"].astype(float)
        paid = out[f"pay_amt{i}"].astype(float)
        out[util_col] = bill / limit
        ratio = np.divide(paid, bill, out=np.full(len(out), np.nan), where=bill > 0)
        out[ratio_col] = np.clip(ratio, 0, 5)
        util_cols.append(util_col)
        ratio_cols.append(ratio_col)

    out["max_delay"] = out[pay_cols].max(axis=1).astype(int)
    out["months_late"] = (out[pay_cols] >= 1).sum(axis=1).astype(int)
    out["months_severe"] = (out[pay_cols] >= 2).sum(axis=1).astype(int)
    out["avg_util"] = out[util_cols].mean(axis=1)
    out["util_recent_minus_old"] = out["util_1"] - out["util_6"]
    out["avg_pay_ratio"] = out[ratio_cols].mean(axis=1)
    return out


def repayment_band(pay_status: pd.Series) -> pd.Series:
    """Collapse the raw status codes into the bands used on the scorecard."""
    bands = pd.Series(index=pay_status.index, dtype=object)
    bands.loc[pay_status <= 0] = "Current or prepaid (<=0)"
    bands.loc[pay_status == 1] = "1 month late"
    bands.loc[pay_status == 2] = "2 months late"
    bands.loc[pay_status >= 3] = "3+ months late"
    return bands


def utilization_band(util: pd.Series) -> pd.Series:
    bands = pd.Series(index=util.index, dtype=object)
    bands.loc[util < 0] = "Credit balance"
    bands.loc[(util >= 0) & (util < 0.3)] = "0–30%"
    bands.loc[(util >= 0.3) & (util < 0.7)] = "30–70%"
    bands.loc[(util >= 0.7) & (util < 1.0)] = "70–100%"
    bands.loc[util >= 1.0] = "Overlimit"
    return bands
