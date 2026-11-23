"""Run the portfolio SQL in SQLite and return one dataframe per query."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from credit_risk.config import SQL_PATH

ACCOUNT_COLUMNS = [
    "account_id",
    "limit_bal",
    "sex",
    "sex_label",
    "education",
    "education_label",
    "marriage",
    "marriage_label",
    "age",
    "pay_1",
    "months_late",
    "bill_amt1",
    "util_1",
    "default_flag",
]

SCORED_COLUMNS = [
    "account_id",
    "risk_grade",
    "pd",
    "default_flag",
    "ead",
    "expected_loss",
    "limit_bal",
    "bill_amt1",
]


def load_queries(path: Path | None = None) -> dict[str, str]:
    text = (path or SQL_PATH).read_text()
    parts = re.split(r"(?m)^-- name:\s*", text)
    queries = {}
    for part in parts[1:]:
        name, _, body = part.partition("\n")
        sql = body.strip().rstrip(";").strip()
        if sql:
            queries[name.strip()] = sql
    if not queries:
        raise ValueError(f"No named queries found in {path or SQL_PATH}")
    return queries


def run_queries(
    accounts: pd.DataFrame,
    scored_book: pd.DataFrame,
    queries: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    queries = queries if queries is not None else load_queries()
    connection = sqlite3.connect(":memory:")
    try:
        accounts.loc[:, ACCOUNT_COLUMNS].to_sql("accounts", connection, index=False, if_exists="replace")
        scored_book.loc[:, SCORED_COLUMNS].to_sql("scored_book", connection, index=False, if_exists="replace")
        results = {}
        for name, sql in queries.items():
            results[name] = pd.read_sql_query(sql, connection)
        return results
    finally:
        connection.close()
