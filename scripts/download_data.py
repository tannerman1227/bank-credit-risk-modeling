#!/usr/bin/env python3
"""Download the UCI Default of Credit Card Clients file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.download import download


if __name__ == "__main__":
    download()
