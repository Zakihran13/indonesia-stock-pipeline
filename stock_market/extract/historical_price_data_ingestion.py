import asyncio
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from loguru import logger
from functools import partial
import aiometer

from motor.motor_asyncio import AsyncIOMotorCollection

# Make direct script execution work by ensuring the project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

from data.db.client import get_async_mongodb
from utils.helper import snake_case_columns, split_batch
from data.db.statements_mongo import fetch_stock_ids, upsert_data

logger.remove()
logger.add(
    sys.stderr,
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    backtrace=True,
    diagnose=False,
)

YF_HISTORICAL_PERIOD = os.getenv("YF_HISTORICAL_PERIOD", "max")
YF_HISTORICAL_START = os.getenv("YF_HISTORICAL_START")
YF_HISTORICAL_END = os.getenv("YF_HISTORICAL_END")
YF_HISTORICAL_INTERVAL = os.getenv("YF_HISTORICAL_INTERVAL", "1d")
YF_HISTORICAL_TIMEOUT_SECONDS = int(os.getenv("YF_HISTORICAL_TIMEOUT_SECONDS", "30"))


def _build_history_kwargs() -> dict[str, Any]:
    """Builds yfinance history kwargs from configured date/period settings."""

    history_kwargs: dict[str, Any] = {"interval": YF_HISTORICAL_INTERVAL}

    if YF_HISTORICAL_START:
        history_kwargs["start"] = YF_HISTORICAL_START

    if YF_HISTORICAL_END:
        history_kwargs["end"] = YF_HISTORICAL_END

    if "start" not in history_kwargs and "end" not in history_kwargs:
        history_kwargs["period"] = YF_HISTORICAL_PERIOD

    history_kwargs["timeout"] = YF_HISTORICAL_TIMEOUT_SECONDS

    return history_kwargs


def _fetch_history_blocking(flatten_ticker: str, history_kwargs: dict[str, Any]) -> pd.DataFrame | None:
    """Executes the blocking yfinance history call outside the event loop."""

    tickers = yf.Tickers(flatten_ticker)
    return tickers.history(**history_kwargs)


async def process_historical_price_data(collection: AsyncIOMotorCollection, ticker: list[str]):
    """Fetches historical price data for a ticker batch using yfinance."""

    flatten_ticker = " ".join(ticker)
    price_data = pd.DataFrame([])
    logger.debug("Fetching historical price data for batch of {} tickers", len(ticker))

    try:
        history_kwargs = _build_history_kwargs()
        logger.debug("Calling yfinance history with params: {}", history_kwargs)
        price_data = _fetch_history_blocking(flatten_ticker, history_kwargs)

        if price_data is None or price_data.empty:
            logger.warning("No historical data returned for batch: {}", flatten_ticker)
            return

        price_data = (
            price_data.stack(level=1).rename_axis(["Date", "Ticker"]).reset_index()
        )
        price_data = snake_case_columns(price_data)
        price_data = price_data.replace({np.nan: None})

        logger.info(
            "Fetched {} historical rows for batch: {}",
            price_data.shape[0],
            flatten_ticker,
        )

        await upsert_data(collection, price_data, conflict_cols=["ticker", "date"])

    except asyncio.CancelledError:
        logger.warning("Fetch task cancelled for batch: {}", flatten_ticker)
        raise

    except Exception as e:
        logger.exception(
            "Error fetching historical data for batch {}: {}", flatten_ticker, e
        )



async def exec_historical_price_data(tickers: list[str] | None = None) -> None:
    """Fetches historical price data for all tickers and stores it in the database."""

    logger.info("Starting historical price ingestion")

    db = get_async_mongodb("raw_stock_data_ingestion")
    collection = db["raw_price_data"]
    metadata_col = db["raw_stock_data"]

    # creating index for first running
    await collection.create_index(
        [("ticker", 1), ("date", 1)],
        unique=1,
        name="ticker_date_unique",
    )


    try:
        if not tickers:
            tickers = await fetch_stock_ids(metadata_col)

        if not tickers:
            logger.warning("No tickers found in metadata table. Skipping ingestion.")
            return

        batches = split_batch(tickers, 5)
        logger.info(
            "Prepared {} ticker batches (batch size: 5) from {} tickers",
            len(batches),
            len(tickers),
        )

        try:
            tasks = [
                partial(process_historical_price_data, collection, batch)
                for batch in batches
            ]

            await aiometer.run_all(
                tasks,
                max_at_once=5
            )

        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.warning("Ctrl+C detected. Cancelling outstanding fetch tasks...")
            logger.info("Task cleanup complete.")
            return
    finally:
        logger.info("Process is Done")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(exec_historical_price_data())
