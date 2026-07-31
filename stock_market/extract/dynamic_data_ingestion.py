from dotenv import load_dotenv
import pandas as pd
import sys
from pathlib import Path
import yfinance as yf

from functools import partial
import asyncio
import aiometer
import nest_asyncio

nest_asyncio.apply()
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)


from data.db.statements import insert_dynamic_data, fetch_stock_ids, attach_stock_ids
from utils.helper import split_batch
from data.db.client import init_async_db


def process_ticker(ticker: list[str]):
    """fetches the metadata for a given ticker using yfinance and returns a DataFrame."""

    flatten_ticker = " ".join(ticker)
    metadata_list = []

    try:
        tickers = yf.Tickers(flatten_ticker)
        for s, t in tickers.tickers.items():
            metadata_list.append({"ticker": s, **t.info})

    except Exception as e:
        print(f"Error fetching metadata for {flatten_ticker}: {e}")

    return pd.DataFrame(metadata_list)


async def exec_metadata():
    engine = init_async_db()
    async with engine.begin() as conn:
        all_tickers = await fetch_stock_ids(conn, None)

    batches = split_batch(list(all_tickers.keys()), 50)

    tasks = [asyncio.to_thread(process_ticker, batch) for batch in batches]
    results = await asyncio.gather(*tasks)

    metadata_df = pd.concat(results, ignore_index=True)
    metadata_df = metadata_df.replace({np.nan: None})
    metadata_df = attach_stock_ids(metadata_df, metadata_df, all_tickers)

    # store data
    try:
        async with engine.begin() as conn:
            await insert_dynamic_data(conn, metadata_df)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(exec_metadata())
