"""Exposure, expected loss, risk grades, and approval-cutoff arithmetic.

Expected loss is PD x LGD x EAD. PD comes from the calibrated model.
LGD and the undrawn credit-conversion factor are assumptions: this dataset
has no recoveries and no balance-at-default history. They are applied the
same way to every account, so they scale the currency loss but do not
change the rank order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import ASSUMED_CCF, ASSUMED_LGD, N_GRADES


def exposure_at_default(limit_bal, bill_amt, ccf: float = ASSUMED_CCF) -> np.ndarray:
    """Drawn balance, floored at zero, plus ``ccf`` times the undrawn limit.

    A negative statement balance is an overpayment, so drawn exposure is zero
    and the undrawn portion is the full limit. A balance above the limit has
    no undrawn piece; the overlimit balance is the exposure.
    """
    limit_bal = np.asarray(limit_bal, dtype=float)
    bill_amt = np.asarray(bill_amt, dtype=float)
    drawn = np.maximum(bill_amt, 0.0)
    undrawn = np.maximum(limit_bal - drawn, 0.0)
    return drawn + float(ccf) * undrawn


def expected_loss(pd, lgd, ead) -> np.ndarray:
    return np.asarray(pd, dtype=float) * np.asarray(lgd, dtype=float) * np.asarray(ead, dtype=float)


def assign_risk_grades(score, n_grades: int = N_GRADES) -> np.ndarray:
    """Grade 1 is the lowest score (safest). Equal counts, ties broken by order."""
    series = pd.Series(np.asarray(score, dtype=float))
    grades = pd.qcut(series.rank(method="first"), n_grades, labels=False) + 1
    return grades.to_numpy().astype(int)


def grade_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a scored book. Requires risk_grade, pd, default_flag, ead, expected_loss."""
    grouped = (
        frame.groupby("risk_grade", as_index=False)
        .agg(
            accounts=("default_flag", "size"),
            defaults=("default_flag", "sum"),
            observed_default_rate=("default_flag", "mean"),
            mean_pd=("pd", "mean"),
            total_ead=("ead", "sum"),
            total_expected_loss=("expected_loss", "sum"),
        )
        .sort_values("risk_grade")
    )
    grouped["el_rate"] = grouped["total_expected_loss"] / grouped["total_ead"]
    grouped["share_of_accounts"] = grouped["accounts"] / grouped["accounts"].sum()
    grouped["share_of_expected_loss"] = grouped["total_expected_loss"] / grouped["total_expected_loss"].sum()
    return grouped.reset_index(drop=True)


def portfolio_totals(frame: pd.DataFrame) -> dict:
    ead = float(frame["ead"].sum())
    el = float(frame["expected_loss"].sum())
    return {
        "accounts": int(len(frame)),
        "defaults": int(frame["default_flag"].sum()),
        "observed_default_rate": float(frame["default_flag"].mean()),
        "mean_pd": float(frame["pd"].mean()),
        "total_ead": ead,
        "total_expected_loss": el,
        "el_rate": el / ead if ead else float("nan"),
    }


def segment_totals(frame: pd.DataFrame, mask: pd.Series, name: str) -> dict:
    totals = portfolio_totals(frame.loc[mask])
    totals["segment"] = name
    return totals


def cutoff_curve(y_true, rank_score, el, ead, calibrated_pd=None, grid=None) -> pd.DataFrame:
    """Accept the safest accounts first. Each row is a target approval rate.

    ``rank_score`` orders the book (higher = riskier). When ``calibrated_pd``
    is provided, the threshold and the mean PD are stated on that scale, which
    is the probability used for expected loss. Isotonic calibration can tie
    many accounts, so the rank score keeps a stable order inside a tied PD.
    """
    y = np.asarray(y_true).astype(int)
    score = np.asarray(rank_score).astype(float)
    probability = score if calibrated_pd is None else np.asarray(calibrated_pd).astype(float)
    el = np.asarray(el).astype(float)
    ead = np.asarray(ead).astype(float)
    if grid is None:
        grid = tuple(np.round(np.linspace(0.20, 1.00, 17), 2))
    n = len(y)
    order = np.argsort(score, kind="mergesort")  # safest first
    total_bads = int(y.sum())
    total_goods = n - total_bads
    rows = []
    for rate in grid:
        k = max(1, min(n, int(round(float(rate) * n))))
        chosen = order[:k]
        accepted = np.zeros(n, dtype=bool)
        accepted[chosen] = True
        declined = ~accepted
        y_acc = y[accepted]
        bads_declined = int(y[declined].sum()) if declined.any() else 0
        goods_declined = int((1 - y[declined]).sum()) if declined.any() else 0
        accepted_ead = float(ead[accepted].sum())
        accepted_el = float(el[accepted].sum())
        rows.append(
            {
                "approval_rate": k / n,
                "accounts_accepted": int(k),
                "pd_threshold": float(probability[chosen].max()),
                "observed_bad_rate": float(y_acc.mean()),
                "mean_pd": float(probability[accepted].mean()),
                "defaults_excluded_share": bads_declined / total_bads if total_bads else float("nan"),
                "non_defaults_excluded_share": goods_declined / total_goods if total_goods else float("nan"),
                "accepted_ead": accepted_ead,
                "accepted_expected_loss": accepted_el,
                "accepted_el_rate": accepted_el / accepted_ead if accepted_ead else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def policy_row(curve: pd.DataFrame, approval_rate: float) -> pd.Series:
    idx = (curve["approval_rate"] - approval_rate).abs().idxmin()
    return curve.loc[idx]
