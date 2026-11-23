"""Monitoring cuts and adverse-action reasons.

Sex and marital status are not model inputs. Approval rates are still
compared across those groups because other characteristics can proxy for
them. This is a screening table, not a determination of fair-lending
compliance. Age and education are scorecard inputs; a US lender would
review both under Regulation B before using them in a decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import FEATURE_LABELS


REASON_TEMPLATES = {
    "pay_1": "Recent repayment status is {bin}",
    "months_late": "Late-payment history is {bin}",
    "util_1": "Current utilization is {bin}",
    "pay_ratio_1": "Recent payment pattern is {bin}",
    "limit_bal": "Credit limit is {bin}",
    "age": "Age band is {bin}",
    "education": "Education category is {bin}",
}


def acceptance_mask(score, approval_rate: float) -> np.ndarray:
    """True for the safest ``approval_rate`` share. Higher score is riskier."""
    score = np.asarray(score, dtype=float)
    n = len(score)
    k = max(1, min(n, int(round(float(approval_rate) * n))))
    order = np.argsort(score, kind="mergesort")
    accepted = np.zeros(n, dtype=bool)
    accepted[order[:k]] = True
    return accepted


def group_outcomes(y_true, rank_score, group, approval_rate: float, calibrated_pd=None) -> pd.DataFrame:
    """Approval rate and bad rate by group at one global cutoff.

    ``rank_score`` decides who is accepted (higher = riskier). ``calibrated_pd``,
    when supplied, is what gets averaged in ``mean_pd``.
    """
    y = np.asarray(y_true).astype(int)
    groups = pd.Series(group).astype(str).reset_index(drop=True)
    accepted = acceptance_mask(rank_score, approval_rate)
    probability = np.asarray(rank_score if calibrated_pd is None else calibrated_pd, dtype=float)
    frame = pd.DataFrame({"y": y, "group": groups, "accepted": accepted, "score": probability})
    rows = []
    for name, part in frame.groupby("group"):
        rows.append(
            {
                "group": name,
                "accounts": int(len(part)),
                "approval_rate": float(part["accepted"].mean()),
                "bad_rate_if_approved": float(part.loc[part["accepted"], "y"].mean()) if part["accepted"].any() else np.nan,
                "observed_default_rate": float(part["y"].mean()),
                "mean_pd": float(part["score"].mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    top = table["approval_rate"].max()
    table["approval_vs_highest_group"] = table["approval_rate"] / top if top else np.nan
    return table


def four_fifths_ratio(table: pd.DataFrame) -> float:
    """Lowest group approval rate divided by the highest. A screening ratio, not a legal test."""
    rates = table["approval_rate"].to_numpy(dtype=float)
    if len(rates) == 0 or np.nanmax(rates) == 0:
        return float("nan")
    return float(np.nanmin(rates) / np.nanmax(rates))


def scorecard_reasons(model, account, top_n: int = 4) -> list[dict]:
    """Characteristics on which this account scores worse than the training average.

    The shortfall is average points minus this account's points. A positive
    shortfall costs the applicant points relative to a typical account.
    ``account`` is a one-row DataFrame (preferred) or a Series.
    """
    frame = account.iloc[[0]].copy() if isinstance(account, pd.DataFrame) else account.to_frame().T
    bins = model.woe_.bin_labels(frame)
    woe = model.transform_woe(frame)
    reasons = []
    for feature in model.kept_features_:
        account_points = -model.factor_ * model.coefs_[feature] * float(woe[feature].iloc[0])
        shortfall = model.avg_points_[feature] - account_points
        if shortfall <= 0:
            continue
        bin_label = str(bins[feature].iloc[0])
        template = REASON_TEMPLATES.get(feature, "{feature} is {bin}")
        reasons.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "bin": bin_label,
                "reason": template.format(bin=bin_label, feature=FEATURE_LABELS.get(feature, feature)),
                "points": account_points,
                "average_points": model.avg_points_[feature],
                "shortfall": shortfall,
            }
        )
    reasons.sort(key=lambda item: item["shortfall"], reverse=True)
    return reasons[:top_n]


def shap_reasons(feature_names, shap_values, feature_row: pd.Series, top_n: int = 4) -> list[dict]:
    """Largest positive SHAP contributions (those pushing probability of default up)."""
    values = np.asarray(shap_values, dtype=float).ravel()
    order = np.argsort(-values)
    reasons = []
    for idx in order:
        contribution = float(values[idx])
        if contribution <= 0:
            break
        feature = feature_names[idx]
        raw_value = feature_row[feature]
        reasons.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "value": _plain_value(raw_value),
                "shap": contribution,
                "reason": (
                    f"{FEATURE_LABELS.get(feature, feature)} "
                    f"({_plain_value(raw_value)}) increased predicted default risk"
                ),
            }
        )
        if len(reasons) >= top_n:
            break
    return reasons


def _plain_value(value) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer() and abs(value) >= 10:
            return f"{value:,.0f}"
        return f"{value:.2f}"
    return str(value)
