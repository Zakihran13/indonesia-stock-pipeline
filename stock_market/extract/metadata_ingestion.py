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
from loguru import logger

from stock_market.config import get_stock_list_path

# Make direct script execution work by ensuring the project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)


from data.db.statements import insert_metadata
from utils.helper import split_batch
from data.db.client import init_async_db



def process_ticker(ticker: list[str]):
    """fetches the metadata for a given ticker using yfinance and returns a DataFrame."""

    flatten_ticker = " ".join(ticker)
    metadata_list = []
    logger.info(f"Fetching metadata for batch: {flatten_ticker}")

    try:
        tickers = yf.Tickers(flatten_ticker)
        for s, t in tickers.tickers.items():
            metadata_list.append({
                "ticker": s,
                **t.info
            })

    except Exception as e:
        print(f"Error fetching metadata for {flatten_ticker}: {e}")

    return pd.DataFrame(metadata_list)




async def exec_metadata():
    logger.info("Starting metadata ingestion process...")
    stock_list_path = get_stock_list_path()
    logger.info(f"Loading stock list from: {stock_list_path}")
    all_tickers = pd.read_json(stock_list_path)
    all_tickers['Kode Jakarta'] = all_tickers['Kode'] + ".JK"
    logger.info(f"Total tickers to process: {len(all_tickers)}")

    batches = split_batch(all_tickers["Kode Jakarta"].tolist(), 50)

    tasks = [asyncio.to_thread(process_ticker, batch) for batch in batches]
    results = await asyncio.gather(*tasks)
    metadata_df = pd.concat(results, ignore_index=True)
    metadata_df = metadata_df.replace({np.nan: None})
    logger.info(f"Total metadata records fetched: {len(metadata_df)}")

    # store data
    engine = init_async_db()
    try:
        async with engine.begin() as conn:
            await insert_metadata(conn, metadata_df) 
            logger.info("Metadata ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Error during metadata ingestion: {e}")
    finally:
        await engine.dispose()




if __name__ == "__main__":
    asyncio.run(exec_metadata())