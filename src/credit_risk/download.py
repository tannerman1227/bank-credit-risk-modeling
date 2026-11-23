"""Download the UCI Default of Credit Card Clients workbook and write a CSV."""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.config import RAW_CSV, RAW_DIR, RAW_XLS, SAMPLE_PATH, SAMPLE_ROWS, SEED, UCI_XLS_URL


def download(force: bool = False) -> Path:
    """Download the public XLS if needed and return the path to the full CSV.

    The full extract is written under ``data/raw`` and is gitignored.
    A small stratified sample is written to ``data/sample`` so the schema
    is visible without the download.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if force or not RAW_XLS.exists() or RAW_XLS.stat().st_size < 1_000_000:
        print(f"Downloading UCI credit-card default file\n  {UCI_XLS_URL}")
        urllib.request.urlretrieve(UCI_XLS_URL, RAW_XLS)
        print(f"Saved {RAW_XLS} ({RAW_XLS.stat().st_size:,} bytes)")

    if force or not RAW_CSV.exists() or RAW_CSV.stat().st_size == 0:
        _xls_to_csv(RAW_XLS, RAW_CSV)
        print(f"Converted workbook to {RAW_CSV}")

    _write_sample(RAW_CSV, SAMPLE_PATH, SAMPLE_ROWS)
    return RAW_CSV


def _xls_to_csv(xls_path: Path, csv_path: Path) -> None:
    """Read the legacy .xls with xlrd 1.x. Row 0 is X1..Y; row 1 is the names."""
    import xlrd

    book = xlrd.open_workbook(str(xls_path))
    sheet = book.sheet_by_index(0)
    if sheet.nrows < 100:
        raise RuntimeError(f"Workbook {xls_path} has only {sheet.nrows} rows; parse failed.")
    headers = [str(sheet.cell_value(1, c)).strip() for c in range(sheet.ncols)]
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for r in range(2, sheet.nrows):
            writer.writerow([sheet.cell_value(r, c) for c in range(sheet.ncols)])


def _write_sample(csv_path: Path, sample_path: Path, n_rows: int) -> None:
    frame = pd.read_csv(csv_path)
    target = "default payment next month"
    if target not in frame.columns:
        raise RuntimeError(f"Expected column {target!r} in {csv_path}")
    sample, _ = train_test_split(
        frame,
        train_size=n_rows,
        stratify=frame[target],
        random_state=SEED,
    )
    sample = sample.sort_values("ID")
    sample.to_csv(sample_path, index=False)
    print(f"Wrote {len(sample)}-row stratified sample to {sample_path}")


def ensure_raw_csv() -> Path:
    if RAW_CSV.exists() and RAW_CSV.stat().st_size > 0:
        return RAW_CSV
    return download()
