"""Transforms raw free cash flow periods into the fundamental_data table.

Raw documents live in Mongo (`raw_cashflow_data`, written by
``stock_market.extract.cashflow_data_ingestion``) with one row per
``(ticker, period_type, period_end)``. This flow maps each fiscal period onto
a ``stock_market.fundamental_data`` snapshot keyed by
``(stock_id, created_at = period_end)`` - the table that already carries
``free_cashflow`` / ``operating_cashflow`` - so the indicators materialization
can asof-merge real statement history instead of only the TTM ``.info`` value.

On conflict only the cashflow columns are patched (``update_columns`` in
``upsert_table``), so same-day ``.info`` snapshots keep their valuation
columns (market_cap, ratios, ...).

Several raw periods can share one period_end (a fiscal year end appears in
both the annual and the quarterly statement, and the TTM snapshot lands on
the most recent quarter), while the table holds one row per date - so each
period_end keeps the widest window available: annual > ttm > quarterly.
"""

from dotenv import load_dotenv
import sys
from pathlib import Path

import asyncio
from datetime import datetime
from loguru import logger
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)


from data.db.statements import (
    insert_cashflow_data,
    fetch_stock_ids,
    attach_stock_ids,
)
from data.db.client import init_async_db, get_async_mongodb
from data.db.statements_mongo import fetch_cashflow_raw

# Rank used to keep one period per period_end: the widest reporting window
# wins (annual FY > trailing-twelve-months > single quarter).
PERIOD_PREFERENCE = {"annual": 0, "ttm": 1, "quarterly": 2}


def drop_missing_period_end(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows that cannot form the (stock_id, created_at) primary key.

    `created_at` is rebuilt from `period_end` and is part of a NOT NULL primary
    key, so a row without one would abort the whole batch instead of being
    skipped.
    """
    if df.empty or "period_end" not in df.columns:
        return df

    missing = df["period_end"].isna()
    if not missing.any():
        return df

    logger.warning(f"Dropping {int(missing.sum())} cashflow row(s) without period_end")
    return df.loc[~missing].copy()


def dedupe_shared_period_ends(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps one row per (ticker, period_end), preferring annual > ttm > quarterly."""

    if df.empty:
        return df

    ranked = df.assign(
        _period_rank=df["period_type"].map(PERIOD_PREFERENCE).fillna(len(PERIOD_PREFERENCE))
    )
    ranked = ranked.sort_values("_period_rank", kind="stable")
    ranked = ranked.drop_duplicates(subset=["ticker", "period_end"], keep="first")
    return ranked.drop(columns=["_period_rank"])


async def exec_cashflow_data(
    ticker: list[str] | None = None, start_date: datetime | None = None
):
    """Moves raw free cash flow periods from Mongo into `stock_market.fundamental_data`.

    `start_date` defaults to today, matching the daily incremental transform
    (raw docs are selected by extraction time, `fetched_at`, because their
    `period_end` values are historical fiscal dates). Pass an earlier date to
    transform a wider raw backfill range.
    """

    logger.info("Starting the DB Engine...")
    engine = init_async_db()
    engine_mongo = get_async_mongodb("raw_stock_data_ingestion")
    cashflow_raw = engine_mongo["raw_cashflow_data"]

    query_date = start_date or datetime.now()
    logger.info(f"Fetching cashflow data extracted since {query_date.date()}!")

    try:
        async with engine.connect() as conn:
            all_tickers = await fetch_stock_ids(conn, ticker)
        if not all_tickers:
            raise RuntimeError(
                "No transformed metadata records found. Run metadata ingestion first."
            )

        # Consume and insert one batch at a time so memory stays bounded on
        # large historical backfills.
        found_any = False
        async for cashflow_df in fetch_cashflow_raw(cashflow_raw, query_date, ticker):
            found_any = True
            cashflow_df = cashflow_df.replace({np.nan: None})

            # created_at is a NOT NULL primary key component.
            cashflow_df = drop_missing_period_end(cashflow_df)

            # Rows without an actual free cash flow value add nothing here and
            # would blank the carried-forward value during the indicators asof
            # merge; they stay available in the raw Mongo collection.
            if "free_cashflow" in cashflow_df.columns:
                cashflow_df = cashflow_df[cashflow_df["free_cashflow"].notna()]
            if cashflow_df.empty:
                continue

            cashflow_df = dedupe_shared_period_ends(cashflow_df)

            cashflow_df = attach_stock_ids(cashflow_df, cashflow_df, all_tickers)
            if cashflow_df.empty:
                continue

            async with engine.begin() as conn:
                await insert_cashflow_data(conn, cashflow_df)

        if not found_any:
            logger.error(f"No data was found for: {ticker}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(exec_cashflow_data(start_date=datetime(2025,1,1)))
