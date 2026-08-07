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

# Make direct script execution work by ensuring the project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

from data.db.client import init_async_db
from data.db.statements import attach_stock_ids, fetch_stock_ids, insert_price_data
from utils.helper import snake_case_columns, split_batch

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

YF_GLOBAL_MIN_INTERVAL_SECONDS = float(
    os.getenv("YF_GLOBAL_MIN_INTERVAL_SECONDS", "1.5")
)
YF_HISTORICAL_PERIOD = os.getenv("YF_HISTORICAL_PERIOD", "max")
YF_HISTORICAL_START = os.getenv("YF_HISTORICAL_START")
YF_HISTORICAL_END = os.getenv("YF_HISTORICAL_END")
YF_HISTORICAL_INTERVAL = os.getenv("YF_HISTORICAL_INTERVAL", "1d")
YF_HISTORICAL_TIMEOUT_SECONDS = int(os.getenv("YF_HISTORICAL_TIMEOUT_SECONDS", "30"))

_yf_rate_limit_lock = asyncio.Lock()
_yf_last_request_ts = 0.0


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


async def wait_for_global_yf_rate_limit() -> None:
    """Enforces a process-wide minimum interval between yfinance requests."""

    global _yf_last_request_ts

    async with _yf_rate_limit_lock:
        now = time.monotonic()
        elapsed = now - _yf_last_request_ts

        if elapsed < YF_GLOBAL_MIN_INTERVAL_SECONDS:
            await asyncio.sleep(YF_GLOBAL_MIN_INTERVAL_SECONDS - elapsed)

        _yf_last_request_ts = time.monotonic()


def _fetch_history_blocking(flatten_ticker: str, history_kwargs: dict[str, Any]):
    """Executes the blocking yfinance history call outside the event loop."""

    tickers = yf.Tickers(flatten_ticker)
    return tickers.history(**history_kwargs)


async def process_historical_price_data(ticker: list[str]) -> pd.DataFrame:
    """Fetches historical price data for a ticker batch using yfinance."""

    flatten_ticker = " ".join(ticker)
    price_data = pd.DataFrame([])
    logger.debug("Fetching historical price data for batch of {} tickers", len(ticker))

    try:
        await wait_for_global_yf_rate_limit()

        history_kwargs = _build_history_kwargs()
        logger.debug("Calling yfinance history with params: {}", history_kwargs)
        history_result = await asyncio.to_thread(
            _fetch_history_blocking, flatten_ticker, history_kwargs
        )
        price_data = history_result if history_result is not None else pd.DataFrame([])

        if price_data.empty:
            logger.warning("No historical data returned for batch: {}", flatten_ticker)
            return price_data

        price_data = (
            price_data.stack(level=1).rename_axis(["Date", "Ticker"]).reset_index()
        )
        logger.info(
            "Fetched {} historical rows for batch: {}",
            price_data.shape[0],
            flatten_ticker,
        )

    except asyncio.CancelledError:
        logger.warning("Fetch task cancelled for batch: {}", flatten_ticker)
        raise

    except Exception as e:
        logger.exception(
            "Error fetching historical data for batch {}: {}", flatten_ticker, e
        )

    return price_data if price_data is not None else pd.DataFrame([])


async def _cancel_pending_tasks(tasks: list[asyncio.Task]) -> None:
    """Cancels pending tasks and waits for cancellation to settle."""

    pending_tasks = [task for task in tasks if not task.done()]
    for task in pending_tasks:
        task.cancel()

    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)


async def exec_historical_price_data() -> None:
    """Fetches historical price data for all tickers and stores it in the database."""

    logger.info("Starting historical price ingestion")
    engine = init_async_db()
    try:
        async with engine.begin() as conn:
            all_tickers = await fetch_stock_ids(conn, None)

        if not all_tickers:
            logger.warning("No tickers found in metadata table. Skipping ingestion.")
            return

        batches = split_batch(list(all_tickers.keys())[:15], 5)
        logger.info(
            "Prepared {} ticker batches (batch size: 5) from {} tickers",
            len(batches),
            len(all_tickers),
        )

        tasks = [
            asyncio.create_task(
                process_historical_price_data(batch), name=f"historical-batch-{idx}"
            )
            for idx, batch in enumerate(batches, start=1)
        ]

        try:
            results = await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.warning("Ctrl+C detected. Cancelling outstanding fetch tasks...")
            await _cancel_pending_tasks(tasks)
            logger.info("Task cleanup complete.")
            return

        valid_results = [
            result for result in results if result is not None and not result.empty
        ]

        if not valid_results:
            logger.warning("No historical price data fetched from any batch.")
            return

        price_data_df = pd.concat(valid_results, ignore_index=True)
        if price_data_df.empty:
            logger.warning("Merged historical price DataFrame is empty.")
            return

        price_data_df = snake_case_columns(price_data_df)
        price_data_df = price_data_df.replace({np.nan: None})
        price_data_df = attach_stock_ids(price_data_df, price_data_df, all_tickers)
        logger.info("Prepared {} rows for database upsert", len(price_data_df))

        # store data
        async with engine.begin() as conn:
            await insert_price_data(conn, price_data_df)
        logger.success(
            "Historical price ingestion completed successfully with {} rows",
            len(price_data_df),
        )
    finally:
        await engine.dispose()
        logger.info("Database engine disposed")


if __name__ == "__main__":
    asyncio.run(exec_historical_price_data())
