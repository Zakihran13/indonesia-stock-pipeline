from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

IDX_STOCK_LIST_URL = os.getenv(
    "IDX_STOCK_LIST_URL",
    "https://idx.co.id/id/data-pasar/data-saham/daftar-saham/",
)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/processed/portfolio.db")
