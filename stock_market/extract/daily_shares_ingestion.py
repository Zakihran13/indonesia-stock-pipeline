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
from data.db.statements import attach_stock_ids, fetch_stock_ids, insert_shares_data
from utils.helper import snake_case_columns, split_batch


YF_GLOBAL_MIN_INTERVAL_SECONDS = float(
	os.getenv("YF_GLOBAL_MIN_INTERVAL_SECONDS", "1.5")
)
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


def process_shares_data(tickers_batch: list[str]) -> pd.DataFrame:
	"""Fetches full shares history for a batch of tickers from yfinance."""

	flatten_ticker = " ".join(tickers_batch)
	shares_frames: list[pd.DataFrame] = []

	try:
		tickers = yf.Tickers(flatten_ticker)
		for symbol, ticker_obj in tickers.tickers.items():
			wait_for_global_yf_rate_limit()
			shares_series = ticker_obj.get_shares_full()

			if shares_series is None or shares_series.empty:
				continue

			shares_df = shares_series.rename("shares_outstanding").to_frame().reset_index()

			if "Date" in shares_df.columns:
				shares_df = shares_df.rename(columns={"Date": "retrieve_at"})
			elif "index" in shares_df.columns:
				shares_df = shares_df.rename(columns={"index": "retrieve_at"})

			shares_df["ticker"] = symbol
			shares_frames.append(shares_df)

	except Exception as e:
		print(f"Error fetching shares for {flatten_ticker}: {e}")

	if not shares_frames:
		return pd.DataFrame([])

	return pd.concat(shares_frames, ignore_index=True)


async def exec_shares_data() -> None:
	"""Fetches shares history for all tickers and upserts into fundamental_data."""

	engine = init_async_db()
	async with engine.begin() as conn:
		all_tickers = await fetch_stock_ids(conn, None)

	batches = split_batch(list(all_tickers.keys()), 20)

	tasks = [asyncio.to_thread(process_shares_data, batch) for batch in batches]
	results = await asyncio.gather(*tasks)

	if not results:
		print("No shares data fetched.")
		await engine.dispose()
		return

	shares_data_df = pd.concat(results, ignore_index=True)
	if shares_data_df.empty:
		print("No shares data fetched.")
		await engine.dispose()
		return

	shares_data_df = snake_case_columns(shares_data_df)
	shares_data_df["retrieve_at"] = pd.to_datetime(
		shares_data_df["retrieve_at"], errors="coerce", utc=True
	).dt.tz_convert(None)
	shares_data_df = shares_data_df.dropna(subset=["ticker", "retrieve_at"])
	shares_data_df = shares_data_df.replace({np.nan: None})
	shares_data_df = attach_stock_ids(shares_data_df, shares_data_df, all_tickers)

	# store data
	try:
		async with engine.begin() as conn:
			await insert_shares_data(conn, shares_data_df)
	finally:
		await engine.dispose()


if __name__ == "__main__":
	asyncio.run(exec_shares_data())
