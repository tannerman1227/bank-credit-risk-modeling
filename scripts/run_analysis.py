#!/usr/bin/env python3
"""Fit the PD models and write the figures, tables, and metrics pack."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.pipeline import run_all


if __name__ == "__main__":
    run_all(save=True, verbose=True)
