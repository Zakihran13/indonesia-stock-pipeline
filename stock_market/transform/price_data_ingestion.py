import pandas as pd
import asyncio
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
from dotenv import load_dotenv
from loguru import logger
from functools import partial
from sqlalchemy.ext.asyncio import AsyncEngine
from motor.motor_asyncio import AsyncIOMotorCollection
import aiometer

# Make direct script execution work by ensuring the project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

from data.db.client import init_async_db, get_async_mongodb
from data.db.statements import fetch_stock_ids, attach_stock_ids, insert_price_data
from data.db.statements_mongo import fetch_partial_price_raw
from utils.helper import split_batch


async def process_ticker_ingestion(
    engine: AsyncEngine,
    stock_raw: AsyncIOMotorCollection,
    nr_tickers: int,
    tickers: list[dict],
    start_date: datetime,
):
    tickers_list = [n["ticker"] for n in tickers]
    tickers_str = ", ".join(n for n in tickers_list)
    data_inserted = 0

    logger.info(f"processing {nr_tickers}: **{tickers_str}**")
    async for data in fetch_partial_price_raw(stock_raw, start_date, tickers_list):
        if not data.empty:
            data = data.replace({np.nan: None})
            data = attach_stock_ids(data, data, tickers)

            data["date"] = pd.to_datetime(data["date"]).dt.normalize()
            data = data.drop_duplicates(subset=["stock_id", "date"], keep="last")

            try:
                async with engine.begin() as app:
                    await insert_price_data(app, data)

                    data_inserted += len(data)
                    logger.info(f"inserted **{data_inserted}** for {tickers_str}")
            except Exception as e:
                logger.error(f"error while inserting data {nr_tickers}: {e}")

    logger.info(f"successfull consume {nr_tickers}: **{tickers_str}**")


async def exec_price_data(
    ticker: list[str] | None = None,
    start_date: datetime | None = None,
    batch_size: int = 100,
):
    """Fetches price data from the raw Mongo store and loads it into the analytics database.

    `start_date` defaults to today, matching the daily incremental transform. Pass an
    earlier date (e.g. to backfill historical gaps) to pull a wider range from Mongo.
    """

    logger.info("Starting the DB Engine...")
    engine = init_async_db()
    engine_mongo = get_async_mongodb("raw_stock_data_ingestion")
    stock_raw = engine_mongo["raw_price_data"]

    query_date = start_date or datetime.now()

    # fetch tickers data
    async with engine.connect() as conn:
        all_tickers = await fetch_stock_ids(conn, ticker)

        if not all_tickers:
            raise RuntimeError(
                "No transformed metadata records found. Run metadata ingestion first."
            )

    batch_tickers = split_batch(all_tickers, batch_size)
    logger.info(f"found {len(batch_tickers)} batches for every {batch_size} groups")

    tasks = [
        partial(
            process_ticker_ingestion, engine, stock_raw, nr_tickers, tickers, query_date
        )
        for nr_tickers, tickers in enumerate(batch_tickers)
    ]

    try:
        await aiometer.run_all(tasks, max_at_once=2)
    except Exception as e:
        logger.error(f"error while runing the job: {e}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(exec_price_data())
