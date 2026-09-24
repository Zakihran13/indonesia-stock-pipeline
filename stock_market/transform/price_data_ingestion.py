import pandas as pd
import asyncio
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
from dotenv import load_dotenv
from loguru import logger

# Make direct script execution work by ensuring the project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

from data.db.client import init_async_db, get_async_mongodb
from data.db.statements import fetch_stock_ids, attach_stock_ids, insert_price_data
from data.db.statements_mongo import fetch_price_raw


async def exec_price_data(ticker: list[str] | None = None, start_date: datetime | None = None):
    """Fetches price data from the raw Mongo store and loads it into the analytics database.

    `start_date` defaults to today, matching the daily incremental transform. Pass an
    earlier date (e.g. to backfill historical gaps) to pull a wider range from Mongo.
    """

    logger.info("Starting the DB Engine...")
    engine = init_async_db()
    engine_mongo = get_async_mongodb("raw_stock_data_ingestion")
    stock_raw = engine_mongo["raw_price_data"]

    query_date = start_date or datetime.now()
    logger.info(f"Fetching price data since {query_date.date()}!")
    price_df = await fetch_price_raw(stock_raw, query_date, ticker)

    if price_df is None or price_df.empty:
        logger.error(f"No data was found for: {ticker}")
        return

    # store data
    try:
        async with engine.begin() as conn:
            all_tickers = await fetch_stock_ids(conn, ticker)
            if not all_tickers:
                raise RuntimeError(
                    "No transformed metadata records found. Run metadata ingestion first."
                )
            price_df = price_df.replace({np.nan: None})
            price_df = attach_stock_ids(price_df, price_df, all_tickers)

            price_df["date"] = pd.to_datetime(price_df["date"]).dt.normalize()
            price_df = price_df.drop_duplicates(
                subset=["stock_id", "date"], keep="last"
            )

            await insert_price_data(conn, price_df)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(exec_price_data(start_date=datetime(1990,1,1)))
