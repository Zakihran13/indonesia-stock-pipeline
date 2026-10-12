from dotenv import load_dotenv
import sys
from pathlib import Path

import asyncio
import nest_asyncio
from datetime import datetime
from loguru import logger

nest_asyncio.apply()
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)


from data.db.statements import (
    insert_dynamic_data,
    fetch_stock_ids,
    attach_stock_ids,
    drop_missing_market_dates,
)
from data.db.client import init_async_db, get_async_mongodb
from data.db.statements_mongo import fetch_dynamic_raw


async def exec_dynamic_data(
    ticker: list[str] | None = None, start_date: datetime | None = None
):
    """Fetches dynamic snapshot data from the raw Mongo store and loads it into the analytics database.

    `start_date` defaults to today, matching the daily incremental transform. Pass an
    earlier date (e.g. to backfill historical gaps) to pull a wider range from Mongo.
    """

    logger.info("Starting the DB Engine...")
    engine = init_async_db()
    engine_mongo = get_async_mongodb("raw_stock_data_ingestion")
    stock_raw = engine_mongo["raw_stock_data"]

    query_date = start_date or datetime.now()
    logger.info(f"Fetching dynamic data since {query_date.date()}!")

    try:
        async with engine.connect() as conn:
            all_tickers = await fetch_stock_ids(conn, ticker)
        if not all_tickers:
            raise RuntimeError(
                "No transformed metadata records found. Run metadata ingestion first."
            )

        # Consume and insert one batch at a time so we never hold the full
        # backfill range (which embeds the full yfinance info payload) in memory.
        found_any = False
        async for metadata_df in fetch_dynamic_raw(stock_raw, query_date, ticker):
            found_any = True
            metadata_df = metadata_df.replace({np.nan: None})

            # market_date becomes created_at (a NOT NULL primary key) downstream,
            # so rows without one would abort the batch instead of inserting.
            metadata_df = drop_missing_market_dates(metadata_df)
            if metadata_df.empty:
                continue

            metadata_df = attach_stock_ids(metadata_df, metadata_df, all_tickers)
            if metadata_df.empty:
                continue

            # insert_dynamic_data() rebuilds created_at from market_date, so dedupe
            # on that same key -- the extraction timestamp never reaches the tables.
            metadata_df = metadata_df.drop_duplicates(
                subset=["stock_id", "market_date"], keep="last"
            )

            async with engine.begin() as conn:
                await insert_dynamic_data(conn, metadata_df)

        if not found_any:
            logger.error(f"No data was found for: {ticker}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(exec_dynamic_data(start_date=datetime(2025, 1, 1)))
