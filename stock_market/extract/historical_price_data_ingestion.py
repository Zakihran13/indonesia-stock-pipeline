import asyncio
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# Make direct script execution work by ensuring the project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

from data.db.client import init_async_db
from data.db.statements import attach_stock_ids, fetch_stock_ids, insert_price_data
from utils.helper import snake_case_columns, split_batch

YF_GLOBAL_MIN_INTERVAL_SECONDS = float(
    os.getenv("YF_GLOBAL_MIN_INTERVAL_SECONDS", "1.5")
)
YF_HISTORICAL_PERIOD = os.getenv("YF_HISTORICAL_PERIOD", "max")
YF_HISTORICAL_START = os.getenv("YF_HISTORICAL_START")
YF_HISTORICAL_END = os.getenv("YF_HISTORICAL_END")
YF_HISTORICAL_INTERVAL = os.getenv("YF_HISTORICAL_INTERVAL", "1d")

_yf_rate_limit_lock = threading.Lock()
_yf_last_request_ts = 0.0


def wait_for_global_yf_rate_limit() -> None:
    """Enforces a process-wide minimum interval between yfinance requests."""

    global _yf_last_request_ts

    with _yf_rate_limit_lock:
        now = time.monotonic()
        elapsed = now - _yf_last_request_ts

        if elapsed < YF_GLOBAL_MIN_INTERVAL_SECONDS:
            time.sleep(YF_GLOBAL_MIN_INTERVAL_SECONDS - elapsed)

        _yf_last_request_ts = time.monotonic()


def process_historical_price_data(ticker: list[str]) -> pd.DataFrame | None:
    """Fetches historical price data for a ticker batch using yfinance."""

    flatten_ticker = " ".join(ticker)
    price_data = pd.DataFrame([])

    try:
        tickers = yf.Tickers(flatten_ticker)
        wait_for_global_yf_rate_limit()

        if YF_HISTORICAL_START and YF_HISTORICAL_END:
            price_data = tickers.history(
                start=YF_HISTORICAL_START,
                end=YF_HISTORICAL_END,
                interval=YF_HISTORICAL_INTERVAL,
            )
        elif YF_HISTORICAL_START:
            price_data = tickers.history(
                start=YF_HISTORICAL_START,
                interval=YF_HISTORICAL_INTERVAL,
            )
        elif YF_HISTORICAL_END:
            price_data = tickers.history(
                end=YF_HISTORICAL_END,
                interval=YF_HISTORICAL_INTERVAL,
            )
        else:
            price_data = tickers.history(
                period=YF_HISTORICAL_PERIOD,
                interval=YF_HISTORICAL_INTERVAL,
            )

        if price_data is None or price_data.empty:
            return price_data

        price_data = (
            price_data.stack(level=1).rename_axis(["Date", "Ticker"]).reset_index()
        )

    except Exception as e:
        print(f"Error fetching historical data for {flatten_ticker}: {e}")

    return price_data


async def exec_historical_price_data() -> None:
    """Fetches historical price data for all tickers and stores it in the database."""

    engine = init_async_db()
    async with engine.begin() as conn:
        all_tickers = await fetch_stock_ids(conn, None)

    batches = split_batch(list(all_tickers.keys()), 20)

    tasks = [
        asyncio.to_thread(process_historical_price_data, batch) for batch in batches
    ]
    results = await asyncio.gather(*tasks)

    if not results:
        print("No historical price data fetched.")
        await engine.dispose()
        return

    price_data_df = pd.concat(results, ignore_index=True)
    if price_data_df.empty:
        print("No historical price data fetched.")
        await engine.dispose()
        return

    price_data_df = snake_case_columns(price_data_df)
    price_data_df = price_data_df.replace({np.nan: None})
    price_data_df = attach_stock_ids(price_data_df, price_data_df, all_tickers)

    # store data
    try:
        async with engine.begin() as conn:
            await insert_price_data(conn, price_data_df)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(exec_historical_price_data())
