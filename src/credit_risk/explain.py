"""Gradient-boosting explanations via SHAP, with a permutation fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from credit_risk.config import FEATURE_LABELS, SEED


def shap_values_matrix(model, frame: pd.DataFrame, max_rows: int = 1500, seed: int = SEED):
    """Return (sample frame, shap matrix aligned to model.feature_names_)."""
    import shap

    sample = _sample(frame, max_rows, seed)
    matrix = model._matrix(sample)
    explainer = shap.TreeExplainer(model.model_)
    values = explainer.shap_values(matrix)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        # Some explainers return (n, features, classes). Use the positive class.
        values = values[:, :, -1]
    return sample, values


def importance_table(model, frame: pd.DataFrame, max_rows: int = 1500, seed: int = SEED) -> tuple[pd.DataFrame, str, pd.DataFrame | None, np.ndarray | None]:
    """Mean absolute SHAP, falling back to permutation importance.

    Also returns the sampled frame and SHAP matrix when SHAP succeeds, so
    adverse-action examples can cite the same explanations.
    """
    try:
        sample, values = shap_values_matrix(model, frame, max_rows=max_rows, seed=seed)
        table = pd.DataFrame(
            {
                "feature": model.feature_names_,
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }
        )
        table["label"] = table["feature"].map(lambda name: FEATURE_LABELS.get(name, name))
        table = table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return table, "shap", sample, values
    except Exception as exc:  # pragma: no cover - exercised only if SHAP breaks
        table = _permutation_table(model, frame, max_rows=max_rows, seed=seed)
        table.attrs["fallback_error"] = f"{exc.__class__.__name__}: {exc}"
        return table, "permutation_importance", None, None


def _permutation_table(model, frame, max_rows, seed) -> pd.DataFrame:
    sample = _sample(frame, max_rows, seed)
    y = sample["default_flag"].to_numpy() if "default_flag" in sample.columns else None
    if y is None:
        raise ValueError("Permutation importance fallback needs default_flag on the frame")
    result = permutation_importance(
        model,
        sample[model.feature_names_],
        y,
        n_repeats=5,
        random_state=seed,
        scoring="roc_auc",
    )
    table = pd.DataFrame(
        {
            "feature": model.feature_names_,
            "mean_abs_shap": result.importances_mean,
        }
    )
    table["label"] = table["feature"].map(lambda name: FEATURE_LABELS.get(name, name))
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def _sample(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    return frame.sample(n=max_rows, random_state=seed)
