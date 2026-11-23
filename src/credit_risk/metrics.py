"""Discrimination, calibration, and stability metrics used in credit scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def roc_auc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


def gini_coefficient(y_true, y_score) -> float:
    return 2.0 * roc_auc(y_true, y_score) - 1.0


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov: the largest gap between the good and bad CDFs.

    Returned on the 0–1 scale. Multiply by 100 for the "KS points" convention.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def ranking_metrics(y_true, y_score) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    return {
        "auc": roc_auc(y_true, y_score),
        "gini": gini_coefficient(y_true, y_score),
        "ks": ks_statistic(y_true, y_score),
    }


def brier(y_true, y_prob) -> float:
    return float(brier_score_loss(np.asarray(y_true).astype(int), np.asarray(y_prob).astype(float)))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Equal-frequency ECE: average absolute gap between outcome and predicted PD."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    frame = pd.DataFrame({"y": y_true, "p": y_prob})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), n_bins, labels=False)
    n = len(frame)
    ece = 0.0
    for _, group in frame.groupby("bin"):
        ece += (len(group) / n) * abs(group["y"].mean() - group["p"].mean())
    return float(ece)


def reliability_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(y_prob).astype(float)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), n_bins, labels=False) + 1
    table = (
        frame.groupby("bin", as_index=False)
        .agg(n=("y", "size"), observed_default_rate=("y", "mean"), mean_predicted_pd=("p", "mean"))
        .sort_values("bin")
    )
    table["gap"] = table["observed_default_rate"] - table["mean_predicted_pd"]
    return table.reset_index(drop=True)


def decile_table(y_true, score, calibrated_pd=None, n_bins: int = 10) -> pd.DataFrame:
    """Decile 1 is the highest score (riskiest). Cumulative capture starts there."""
    frame = pd.DataFrame(
        {
            "y": np.asarray(y_true).astype(int),
            "score": np.asarray(score).astype(float),
        }
    )
    if calibrated_pd is None:
        frame["calibrated_pd"] = frame["score"]
    else:
        frame["calibrated_pd"] = np.asarray(calibrated_pd).astype(float)
    # rank(method="first") breaks ties so every decile has the same count.
    frame["decile"] = (
        n_bins
        - pd.qcut(frame["score"].rank(method="first"), n_bins, labels=False)
    ).astype(int)
    grouped = (
        frame.groupby("decile", as_index=False)
        .agg(
            accounts=("y", "size"),
            defaults=("y", "sum"),
            observed_default_rate=("y", "mean"),
            mean_score=("score", "mean"),
            mean_calibrated_pd=("calibrated_pd", "mean"),
        )
        .sort_values("decile")
    )
    goods = grouped["accounts"] - grouped["defaults"]
    grouped["share_of_defaults"] = grouped["defaults"] / grouped["defaults"].sum()
    grouped["share_of_non_defaults"] = goods / goods.sum()
    grouped["cumulative_default_capture"] = grouped["share_of_defaults"].cumsum()
    grouped["cumulative_non_default_capture"] = grouped["share_of_non_defaults"].cumsum()
    grouped["ks_gap"] = (
        grouped["cumulative_default_capture"] - grouped["cumulative_non_default_capture"]
    )
    return grouped.reset_index(drop=True)


def cumulative_gains(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    """Population depth and share of defaults captured, walking from riskiest to safest."""
    y = np.asarray(y_true).astype(int)
    score = np.asarray(y_score).astype(float)
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    total_bad = y_sorted.sum()
    if total_bad == 0:
        depth = np.arange(1, len(y) + 1) / len(y)
        return depth, np.zeros(len(y))
    captured = np.cumsum(y_sorted) / total_bad
    depth = np.arange(1, len(y) + 1) / len(y)
    return depth, captured


def population_stability_index(reference, current, n_bins: int = 10) -> float:
    """PSI of ``current`` against quantile bins of ``reference``.

    Under 0.10 is usually read as stable, 0.10–0.25 as a shift worth watching.
    A random split should land well under 0.10.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(reference, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return float("nan")
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    n_used = len(ref_counts)
    ref_pct = (ref_counts + 1e-6) / (ref_counts.sum() + 1e-6 * n_used)
    cur_pct = (cur_counts + 1e-6) / (cur_counts.sum() + 1e-6 * n_used)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
