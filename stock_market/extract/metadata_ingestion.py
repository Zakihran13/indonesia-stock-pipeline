from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
import sys
import yfinance as yf

from functools import partial
import asyncio
import aiometer
import nest_asyncio

nest_asyncio.apply()
import numpy as np
from datetime import datetime, UTC
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorCollection

# Make direct script execution work by ensuring the project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)


from data.db.statements_mongo import upsert_data
from utils.helper import split_batch
from data.db.client import get_async_mongodb
from stock_market.config import get_stock_list_path


async def process_ticker(ticker: list[str], collection: AsyncIOMotorCollection):
    """fetches the metadata for a given ticker using yfinance and returns a DataFrame."""

    flatten_ticker = " ".join(ticker)
    metadata_list = []
    logger.info(f"Fetching metadata for batch: {flatten_ticker}")

    try:
        tickers = yf.Tickers(flatten_ticker)
        for s, t in tickers.tickers.items():
            metadata_list.append(
                {"ticker": s, "created_at": datetime.now(UTC), **t.info}
            )

        if metadata_list:
            await upsert_data(collection, pd.DataFrame(metadata_list), ["ticker"])

    except Exception as e:
        print(f"Error fetching metadata for {flatten_ticker}: {e}")


async def exec_metadata():
    logger.info("Starting metadata ingestion process...")
    stock_list_path = get_stock_list_path()
    logger.info(f"Loading stock list from: {stock_list_path}")

    all_tickers = pd.read_json(stock_list_path)
    all_tickers["Kode Jakarta"] = all_tickers["Kode"] + ".JK"
    logger.info(f"Total tickers to process: {len(all_tickers)}")

    logger.info("Initializing async database connection...")
    db = get_async_mongodb("raw_stock_data_ingestion")
    collection = db["raw_stock_data"]

    # creating index for first running
    await collection.create_index(
        [("ticker", 1)],
        unique=1,
        name="ticker_unique",
    )

    try:
        batches = split_batch(all_tickers["Kode Jakarta"].tolist()[:500], 50)
        tasks = [partial(process_ticker, batch, collection) for batch in batches]

        await aiometer.run_all(tasks, max_at_once=5)
    except Exception as e:
        logger.error(f"Error while processing the batch flow: {e}")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(exec_metadata())