"""Load and clean the UCI Default of Credit Card Clients file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_risk.config import (
    EDUCATION_LABELS,
    MARRIAGE_LABELS,
    RAW_CSV,
    SAMPLE_PATH,
    SEX_LABELS,
)
from credit_risk.download import ensure_raw_csv


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the full CSV, downloading the UCI workbook if it is not local yet."""
    csv_path = Path(path) if path is not None else ensure_raw_csv()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path)


def load_sample() -> pd.DataFrame:
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Sample not found at {SAMPLE_PATH}. Run python scripts/download_data.py."
        )
    return pd.read_csv(SAMPLE_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize names, recode undocumented categories, and drop broken rows.

    The source file labels the most recent repayment status ``PAY_0`` and then
    skips to ``PAY_2``. ``PAY_0`` is renamed ``pay_1`` so the six statements
    line up with ``bill_amt1`` (most recent) through ``bill_amt6``.

    Education codes 0, 5, and 6 and marriage code 0 are undocumented in the
    UCI dictionary. They are folded into the published "other" level.
    """
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    rename = {}
    if "pay_0" in out.columns:
        rename["pay_0"] = "pay_1"
    if "default_payment_next_month" in out.columns:
        rename["default_payment_next_month"] = "default_flag"
    if "id" in out.columns:
        rename["id"] = "account_id"
    out = out.rename(columns=rename)

    required = [
        "account_id",
        "limit_bal",
        "sex",
        "education",
        "marriage",
        "age",
        "default_flag",
        *[f"pay_{i}" for i in range(1, 7)],
        *[f"bill_amt{i}" for i in range(1, 7)],
        *[f"pay_amt{i}" for i in range(1, 7)],
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns after rename: {missing}")

    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    cleaning_report = _cleaning_report(out)

    out = out.dropna(subset=required).copy()
    out = out[out["limit_bal"] > 0].copy()
    out["account_id"] = out["account_id"].astype(int)
    out = out.drop_duplicates(subset=["account_id"], keep="first")

    out["default_flag"] = out["default_flag"].astype(int)
    if not set(out["default_flag"].unique()).issubset({0, 1}):
        raise ValueError("default_flag must be 0/1 after cleaning")

    out["sex"] = out["sex"].astype(int)
    out["age"] = out["age"].astype(int)
    out["limit_bal"] = out["limit_bal"].astype(float)
    out["education"] = out["education"].astype(int)
    out["marriage"] = out["marriage"].astype(int)
    for i in range(1, 7):
        out[f"pay_{i}"] = out[f"pay_{i}"].astype(int)

    out.loc[~out["education"].isin([1, 2, 3, 4]), "education"] = 4
    out.loc[~out["marriage"].isin([1, 2, 3]), "marriage"] = 3
    out["education_label"] = out["education"].map(EDUCATION_LABELS)
    out["marriage_label"] = out["marriage"].map(MARRIAGE_LABELS)
    out["sex_label"] = out["sex"].map(SEX_LABELS).fillna("Unknown")

    out.attrs["cleaning_report"] = {
        **cleaning_report,
        "rows_out": int(len(out)),
        "default_rate": float(out["default_flag"].mean()),
        "defaults": int(out["default_flag"].sum()),
    }
    return out.reset_index(drop=True)


def _cleaning_report(numeric: pd.DataFrame) -> dict:
    edu = numeric["education"]
    mar = numeric["marriage"]
    return {
        "rows_in": int(len(numeric)),
        "education_undocumented": int((~edu.isin([1, 2, 3, 4]) & edu.notna()).sum()),
        "marriage_undocumented": int((~mar.isin([1, 2, 3]) & mar.notna()).sum()),
        "rows_with_any_null_in_required": int(numeric.isna().any(axis=1).sum()),
        "nonpositive_limit": int((numeric["limit_bal"].fillna(0) <= 0).sum()),
        "duplicate_ids": int(numeric["account_id"].duplicated().sum()),
    }


def load_clean(path: Path | None = None) -> pd.DataFrame:
    raw = load_raw(path) if path is not None else load_raw(RAW_CSV if RAW_CSV.exists() else None)
    return clean(raw)
