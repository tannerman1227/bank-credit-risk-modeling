"""Weight-of-evidence binning for the logistic scorecard.

WOE is ln(share of non-defaults / share of defaults), with 0.5 smoothing.
A higher WOE bin is safer. Information value summarizes how far the goods
and bads diverge across bins. Bins are fit on the training rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Logical scorecard bins. Quantile edges for the credit limit are fit on train.
SCORECARD_SPECS = {
    "pay_1": {
        "kind": "numeric",
        "edges": [-np.inf, 0, 1, 2, np.inf],
        "labels": [
            "Current or prepaid (<=0)",
            "1 month late",
            "2 months late",
            "3+ months late",
        ],
    },
    "months_late": {
        "kind": "numeric",
        "edges": [-np.inf, 0, 1, 2, np.inf],
        "labels": ["0 months late", "1 month late", "2 months late", "3+ months late"],
    },
    "util_1": {"kind": "utilization"},
    "pay_ratio_1": {"kind": "pay_ratio"},
    "limit_bal": {"kind": "quantile", "q": 5, "currency": True},
    "age": {
        "kind": "numeric",
        "edges": [-np.inf, 25, 35, 45, 55, np.inf],
        "labels": ["21–25", "26–35", "36–45", "46–55", "56+"],
    },
    "education": {
        "kind": "categorical",
        "categories": {
            1: "Graduate school",
            2: "University",
            3: "High school",
            4: "Other",
        },
        "other": "Other",
    },
}

PAY_RATIO_ORDER = [
    "No positive statement balance",
    "No payment",
    "Paid under 20% of bill",
    "Paid 20–100% of bill",
    "Paid more than the bill",
]
UTILIZATION_ORDER = [
    "Credit balance",
    "0–30%",
    "30–70%",
    "70–100%",
    "Overlimit",
    "Missing",
]


def assign_bins(series: pd.Series, spec: dict, edges=None, labels=None) -> pd.Series:
    kind = spec["kind"]
    values = series.astype(float) if kind != "categorical" else series
    if kind == "numeric":
        use_edges = spec["edges"] if edges is None else edges
        use_labels = spec["labels"] if labels is None else labels
        binned = pd.cut(values, bins=use_edges, labels=use_labels, include_lowest=True, right=True)
        out = pd.Series(binned.astype(object), index=series.index)
        out[values.isna() | out.isna()] = "Missing"
        return out
    if kind == "quantile":
        if edges is None or labels is None:
            raise ValueError("Quantile bins need fitted edges and labels")
        binned = pd.cut(values, bins=edges, labels=labels, include_lowest=True, right=True)
        out = pd.Series(binned.astype(object), index=series.index)
        out[values.isna() | out.isna()] = "Missing"
        return out
    if kind == "categorical":
        mapped = series.map(spec["categories"])
        return mapped.fillna(spec.get("other", "Other")).astype(object)
    if kind == "pay_ratio":
        out = pd.Series("No positive statement balance", index=series.index, dtype=object)
        known = series.notna()
        out.loc[known & (series <= 0)] = "No payment"
        out.loc[known & (series > 0) & (series <= 0.2)] = "Paid under 20% of bill"
        out.loc[known & (series > 0.2) & (series <= 1.0)] = "Paid 20–100% of bill"
        out.loc[known & (series > 1.0)] = "Paid more than the bill"
        return out
    if kind == "utilization":
        out = pd.Series("Missing", index=series.index, dtype=object)
        known = series.notna()
        out.loc[known & (series < 0)] = "Credit balance"
        out.loc[known & (series >= 0) & (series < 0.3)] = "0–30%"
        out.loc[known & (series >= 0.3) & (series < 0.7)] = "30–70%"
        out.loc[known & (series >= 0.7) & (series < 1.0)] = "70–100%"
        out.loc[known & (series >= 1.0)] = "Overlimit"
        return out
    raise ValueError(f"Unknown bin kind {kind}")


def _fit_quantile_edges(series: pd.Series, q: int) -> tuple[np.ndarray, list[str]]:
    clean = series.dropna().astype(float)
    _, raw_edges = pd.qcut(clean, q=q, retbins=True, duplicates="drop")
    raw_edges = np.asarray(raw_edges, dtype=float)
    display = []
    for left, right in zip(raw_edges[:-1], raw_edges[1:]):
        display.append(f"NT${left:,.0f}–{right:,.0f}")
    edges = raw_edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges, display


def _woe_table(bins: pd.Series, y: np.ndarray, order: list[str] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({"bin": bins.astype(object), "y": y})
    counts = frame.groupby("bin", as_index=False).agg(n=("y", "size"), n_bad=("y", "sum"))
    counts["n_good"] = counts["n"] - counts["n_bad"]
    n_bins = len(counts)
    n_good = counts["n_good"].to_numpy(dtype=float)
    n_bad = counts["n_bad"].to_numpy(dtype=float)
    dist_good = (n_good + 0.5) / (n_good.sum() + 0.5 * n_bins)
    dist_bad = (n_bad + 0.5) / (n_bad.sum() + 0.5 * n_bins)
    counts["pct_good"] = dist_good
    counts["pct_bad"] = dist_bad
    counts["woe"] = np.log(dist_good / dist_bad)
    counts["iv_contrib"] = (dist_good - dist_bad) * counts["woe"]
    if order:
        rank = {label: i for i, label in enumerate(order)}
        counts["_order"] = counts["bin"].map(lambda label: rank.get(label, 10_000))
        counts = counts.sort_values(["_order", "bin"]).drop(columns="_order")
    else:
        counts = counts.sort_values("bin")
    return counts.reset_index(drop=True)


class WoeTransformer(BaseEstimator, TransformerMixin):
    """Map scorecard characteristics to weight of evidence.

    Unknown or missing bins become WOE 0, which adds nothing to the log-odds.
    """

    def __init__(self, features: list[str] | None = None):
        self.features = features

    def fit(self, X, y):
        frame = _as_frame(X)
        target = np.asarray(y).astype(int)
        features = list(self.features) if self.features is not None else [
            c for c in SCORECARD_SPECS if c in frame.columns
        ]
        self.features_ = features
        self.tables_ = {}
        self.mappings_ = {}
        self.fitted_ = {}
        self.iv_ = {}
        for name in features:
            spec = SCORECARD_SPECS[name]
            fitted = {"spec": spec}
            labels = None
            edges = None
            if spec["kind"] == "quantile":
                edges, labels = _fit_quantile_edges(frame[name], spec["q"])
                fitted["edges"] = edges
                fitted["labels"] = labels
            bins = assign_bins(frame[name], spec, edges=edges, labels=labels)
            order = _bin_order(spec, labels)
            table = _woe_table(bins, target, order=order)
            self.tables_[name] = table
            self.mappings_[name] = dict(zip(table["bin"], table["woe"]))
            self.iv_[name] = float(table["iv_contrib"].sum())
            self.fitted_[name] = fitted
        return self

    def transform(self, X):
        frame = _as_frame(X)
        data = {}
        for name in self.features_:
            spec = self.fitted_[name]["spec"]
            edges = self.fitted_[name].get("edges")
            labels = self.fitted_[name].get("labels")
            bins = assign_bins(frame[name], spec, edges=edges, labels=labels)
            data[name] = bins.map(self.mappings_[name]).fillna(0.0).to_numpy(dtype=float)
        return pd.DataFrame(data, index=frame.index)

    def bin_labels(self, X) -> pd.DataFrame:
        frame = _as_frame(X)
        data = {}
        for name in self.features_:
            spec = self.fitted_[name]["spec"]
            data[name] = assign_bins(
                frame[name],
                spec,
                edges=self.fitted_[name].get("edges"),
                labels=self.fitted_[name].get("labels"),
            ).to_numpy()
        return pd.DataFrame(data, index=frame.index)

    def information_value(self) -> pd.DataFrame:
        rows = []
        for name in self.features_:
            iv = self.iv_[name]
            rows.append(
                {
                    "feature": name,
                    "information_value": iv,
                    "n_bins": int(len(self.tables_[name])),
                    "strength": _iv_strength(iv),
                }
            )
        return pd.DataFrame(rows).sort_values("information_value", ascending=False).reset_index(drop=True)


def _bin_order(spec: dict, quantile_labels: list[str] | None) -> list[str]:
    if spec["kind"] == "numeric":
        return list(spec["labels"]) + ["Missing"]
    if spec["kind"] == "pay_ratio":
        return list(PAY_RATIO_ORDER)
    if spec["kind"] == "utilization":
        return list(UTILIZATION_ORDER)
    if spec["kind"] == "categorical":
        return list(dict.fromkeys(list(spec["categories"].values()) + [spec.get("other", "Other")]))
    if spec["kind"] == "quantile":
        return list(quantile_labels or []) + ["Missing"]
    return []


def _iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "unused"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "very strong"


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    raise TypeError("WoeTransformer expects a pandas DataFrame with named columns")
