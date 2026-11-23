"""Charts for the credit committee pack. One function, one figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from credit_risk.config import FEATURE_LABELS
from credit_risk.features import repayment_band, utilization_band
from credit_risk.metrics import cumulative_gains

NAVY = "#0B2545"
TEAL = "#1F7A8C"
CORAL = "#C44536"
GOLD = "#A6792D"
GREEN = "#2F6D4F"
SLATE = "#5C6B7A"
GRID = "#E6E8EC"
INK = "#1B2430"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#FBFBFD",
        "axes.edgecolor": "#D0D5DD",
        "axes.labelcolor": INK,
        "axes.titlecolor": NAVY,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "text.color": INK,
        "xtick.color": "#3D4A5C",
        "ytick.color": "#3D4A5C",
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
    }
)


def save_fig(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _footnote(fig, text: str) -> None:
    fig.text(0.0, -0.02, text, fontsize=8, color=SLATE, ha="left")


def _bar_colors(values) -> list[str]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return []
    peak = np.nanmax(values)
    return [CORAL if np.isclose(v, peak) else TEAL for v in values]


def plot_default_rate_by_repayment(train: pd.DataFrame, path: Path) -> Path:
    frame = train.copy()
    frame["band"] = repayment_band(frame["pay_1"])
    order = ["Current or prepaid (<=0)", "1 month late", "2 months late", "3+ months late"]
    summary = (
        frame.groupby("band", as_index=False)
        .agg(n=("default_flag", "size"), rate=("default_flag", "mean"))
        .set_index("band")
        .reindex(order)
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    labels = [f"{name}\n(n={int(n):,})" for name, n in zip(summary.index, summary["n"])]
    bars = ax.bar(labels, summary["rate"], color=_bar_colors(summary["rate"]), width=0.72)
    ax.set_ylabel("Default rate")
    ax.set_ylim(0, min(1.0, float(summary["rate"].max()) * 1.25 + 0.05))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Default rate by most recent repayment status")
    for bar, rate in zip(bars, summary["rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=NAVY,
        )
    ax.grid(axis="x", visible=False)
    _footnote(fig, "Training sample. Status is the most recent statement (PAY_0 in the UCI file).")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_default_rate_by_utilization(train: pd.DataFrame, path: Path) -> Path:
    frame = train.copy()
    frame["band"] = utilization_band(frame["util_1"])
    order = ["Credit balance", "0–30%", "30–70%", "70–100%", "Overlimit"]
    summary = (
        frame.groupby("band", as_index=False)
        .agg(n=("default_flag", "size"), rate=("default_flag", "mean"))
        .set_index("band")
        .reindex(order)
        .dropna(subset=["n"])
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    labels = [f"{name}\n(n={int(n):,})" for name, n in zip(summary.index, summary["n"])]
    bars = ax.bar(labels, summary["rate"], color=_bar_colors(summary["rate"]), width=0.72)
    ax.set_ylabel("Default rate")
    ax.set_ylim(0, min(1.0, float(summary["rate"].max()) * 1.28 + 0.05))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Default rate by current utilization")
    for bar, rate in zip(bars, summary["rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=NAVY,
        )
    ax.grid(axis="x", visible=False)
    _footnote(fig, "Training sample. Utilization is the current statement balance divided by the credit limit.")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_limit_by_default(train: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    groups = [
        train.loc[train["default_flag"] == 0, "limit_bal"] / 1000,
        train.loc[train["default_flag"] == 1, "limit_bal"] / 1000,
    ]
    box = ax.boxplot(
        groups,
        tick_labels=["No default next month", "Default next month"],
        patch_artist=True,
        medianprops={"color": NAVY, "linewidth": 1.6},
        whiskerprops={"color": SLATE},
        capprops={"color": SLATE},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": SLATE, "markeredgecolor": "none", "alpha": 0.35},
        widths=0.55,
    )
    for patch, color in zip(box["boxes"], [TEAL, CORAL]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(NAVY)
    ax.set_ylabel("Credit limit (NT$ thousands)")
    ax.set_title("Credit limits are lower on accounts that default")
    ax.grid(axis="x", visible=False)
    _footnote(fig, "Training sample. Limits are New Taiwan dollars, shown in thousands.")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_correlation(train: pd.DataFrame, path: Path) -> Path:
    columns = [
        "pay_1",
        "months_late",
        "max_delay",
        "util_1",
        "avg_util",
        "pay_ratio_1",
        "limit_bal",
        "age",
        "bill_amt1",
        "pay_amt1",
    ]
    labels = [FEATURE_LABELS.get(c, c) for c in columns]
    corr = train[columns].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        square=True,
        cbar_kws={"shrink": 0.75},
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_title("Correlation among the characteristics a scorecard would consider")
    ax.tick_params(axis="x", labelrotation=40)
    ax.tick_params(axis="y", labelrotation=0)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    _footnote(fig, "Training sample. Pearson correlation. Delinquency fields move together; the scorecard regularizes that.")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    colors = {"Repayment status only": SLATE, "Scorecard": TEAL, "Gradient boosting": CORAL}
    for name, (y, score) in curves.items():
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        ax.plot(fpr, tpr, color=colors.get(name, NAVY), lw=2.0, label=f"{name}  AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#C5CAD3", lw=1, ls="--", label="No discrimination")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Holdout ROC: model versus a repayment-status rule")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _footnote(fig, "Untouched 20% holdout. The repayment-status score is the raw PAY status, higher meaning later.")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_gains(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    colors = {"Repayment status only": SLATE, "Scorecard": TEAL, "Gradient boosting": CORAL}
    for name, (y, score) in curves.items():
        depth, captured = cumulative_gains(y, score)
        # Plot a few hundred points so the file stays light.
        step = max(1, len(depth) // 400)
        ax.plot(depth[::step], captured[::step], color=colors.get(name, NAVY), lw=2.0, label=name)
    ax.plot([0, 1], [0, 1], color="#C5CAD3", lw=1, ls="--", label="Random")
    ax.set_xlabel("Share of holdout reviewed, riskiest first")
    ax.set_ylabel("Share of defaults captured")
    ax.set_title("Gains chart on the holdout")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _footnote(fig, "Walking from the highest predicted risk to the lowest. A steeper curve concentrates defaults earlier.")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_deciles(deciles: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    x = np.arange(len(deciles))
    width = 0.38
    ax.bar(x - width / 2, deciles["observed_default_rate"], width=width, color=CORAL, label="Observed default rate")
    ax.bar(x + width / 2, deciles["mean_calibrated_pd"], width=width, color=TEAL, label="Mean calibrated PD")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in deciles["decile"]])
    ax.set_xlabel("Risk decile (1 = highest predicted risk)")
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Holdout deciles: observed default rate versus calibrated PD")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    _footnote(fig, "Deciles are equal counts on the champion's raw score. PD is the isotonic-calibrated probability.")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_calibration(raw_table: pd.DataFrame, calibrated_table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.plot([0, 1], [0, 1], color="#C5CAD3", lw=1, ls="--", label="Perfect calibration")
    ax.plot(
        raw_table["mean_predicted_pd"],
        raw_table["observed_default_rate"],
        color=SLATE,
        marker="o",
        lw=1.6,
        label="Before calibration",
    )
    ax.plot(
        calibrated_table["mean_predicted_pd"],
        calibrated_table["observed_default_rate"],
        color=TEAL,
        marker="o",
        lw=2.0,
        label="After isotonic calibration",
    )
    cap = max(
        0.05,
        float(raw_table["mean_predicted_pd"].max()),
        float(raw_table["observed_default_rate"].max()),
        float(calibrated_table["mean_predicted_pd"].max()),
        float(calibrated_table["observed_default_rate"].max()),
    )
    cap = min(1.0, cap * 1.08)
    ax.set_xlim(0, cap)
    ax.set_ylim(0, cap)
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Holdout calibration of the champion model")
    ax.legend(frameon=False, loc="upper left")
    _footnote(fig, "Equal-frequency bins on the untouched holdout. The calibrator was fit on training out-of-fold scores only.")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_score_distribution(score, y, path: Path) -> Path:
    score = np.asarray(score, dtype=float)
    y = np.asarray(y).astype(int)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bins = 35
    ax.hist(score[y == 0], bins=bins, density=True, color=TEAL, alpha=0.75, label="No default")
    ax.hist(score[y == 1], bins=bins, density=True, color=CORAL, alpha=0.65, label="Default")
    ax.set_xlabel("Scorecard points (higher is safer)")
    ax.set_ylabel("Density")
    ax.set_title("Holdout scorecard points by next-month outcome")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    _footnote(fig, "Points double the odds of non-default every 20 points. Scaling does not change the rank order.")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_importance(table: pd.DataFrame, path: Path, method: str) -> Path:
    top = table.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.barh(top["label"], top["mean_abs_shap"], color=TEAL)
    ax.set_xlabel("Mean absolute SHAP value" if method == "shap" else "Permutation importance (AUC drop)")
    ax.set_title("What moves the gradient boosting score")
    ax.grid(axis="y", visible=False)
    _footnote(fig, "Holdout sample. A larger bar means the feature moves the predicted default risk more, in either direction.")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_cutoff(curve: pd.DataFrame, policy_rate: float, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    ax = axes[0]
    ax.plot(curve["approval_rate"], curve["observed_bad_rate"], color=CORAL, lw=2.2)
    policy = curve.iloc[(curve["approval_rate"] - policy_rate).abs().argmin()]
    ax.scatter([policy["approval_rate"]], [policy["observed_bad_rate"]], color=NAVY, zorder=3)
    ax.annotate(
        f"{policy['approval_rate']:.0%} accepted\nbad rate {policy['observed_bad_rate']:.1%}",
        (policy["approval_rate"], policy["observed_bad_rate"]),
        textcoords="offset points",
        xytext=(-78, 12),
        fontsize=8.5,
        color=NAVY,
    )
    book_bad = curve.iloc[(curve["approval_rate"] - 1).abs().argmin()]["observed_bad_rate"]
    ax.axhline(book_bad, color=SLATE, lw=1, ls="--", label=f"Full-book bad rate {book_bad:.1%}")
    ax.set_xlabel("Share of holdout accepted, safest first")
    ax.set_ylabel("Default rate among accepted accounts")
    ax.set_title("Acceptance versus bad rate")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    ax.plot(curve["approval_rate"], curve["defaults_excluded_share"], color=CORAL, lw=2.2, label="Defaults removed")
    ax.plot(
        curve["approval_rate"],
        curve["non_defaults_excluded_share"],
        color=TEAL,
        lw=2.2,
        label="Non-defaults removed",
    )
    ax.axvline(policy["approval_rate"], color=NAVY, lw=1, ls=":")
    ax.set_xlabel("Share of holdout accepted")
    ax.set_ylabel("Share removed from the book")
    ax.set_title("What the cutoff takes out")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False, loc="upper right")
    _footnote(fig, "Holdout book. Accounts are ordered by the champion score, safest first. The marker is the 70% acceptance policy.")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_expected_loss_by_grade(grades: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    millions = grades["total_expected_loss"] / 1_000_000
    colors = [TEAL if g < grades["risk_grade"].max() else CORAL for g in grades["risk_grade"]]
    bars = ax.bar([str(int(g)) for g in grades["risk_grade"]], millions, color=colors, width=0.72)
    ax.set_xlabel("Risk grade (1 = lowest predicted risk)")
    ax.set_ylabel("Expected loss (NT$ millions)")
    ax.set_title("Holdout expected loss sits in the worst grades")
    for bar, row in zip(bars, grades.itertuples(index=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f" {row.el_rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=NAVY,
        )
    ax.set_ylim(0, float(millions.max()) * 1.2)
    ax.grid(axis="x", visible=False)
    _footnote(fig, "Labels are EL / EAD. LGD and the undrawn conversion factor are assumptions, so levels scale with those choices.")
    fig.tight_layout()
    return save_fig(fig, path)


def plot_approval_by_group(table: pd.DataFrame, path: Path, title: str, footnote: str | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ordered = table.sort_values("approval_rate")
    colors = [CORAL if r < 0.8 else TEAL for r in ordered["approval_vs_highest_group"]]
    bars = ax.barh(ordered["group"], ordered["approval_rate"], color=colors)
    ax.set_xlabel("Approval rate at the 70% book cutoff")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    for bar, rate in zip(bars, ordered["approval_rate"]):
        ax.text(rate + 0.015, bar.get_y() + bar.get_height() / 2, f"{rate:.1%}", va="center", fontsize=9, color=NAVY)
    ax.grid(axis="y", visible=False)
    _footnote(
        fig,
        footnote
        or "Same score cutoff for every group. Coral marks a group under 80% of the highest approval rate.",
    )
    fig.tight_layout()
    return save_fig(fig, path)
