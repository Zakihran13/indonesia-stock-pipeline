from dotenv import load_dotenv
import sys
from pathlib import Path

import asyncio
import nest_asyncio
from datetime import datetime
from loguru import logger
import pandas as pd

nest_asyncio.apply()
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)


from data.db.statements import insert_dynamic_data, fetch_stock_ids, attach_stock_ids
from data.db.client import init_async_db, get_async_mongodb
from data.db.statements_mongo import fetch_dynamic_raw


async def exec_dynamic_data(ticker: list[str] | None = None):
    logger.info("Starting the DB Engine...")
    engine = init_async_db()
    engine_mongo = get_async_mongodb("raw_stock_data_ingestion")
    stock_raw = engine_mongo["raw_stock_data"]

    logger.info("Fetching dynamic data!")
    metadata_df = await fetch_dynamic_raw(stock_raw, datetime.now(), ticker)

    if metadata_df is None or metadata_df.empty:
        logger.error(f"No data was found for: {ticker}")
        return

    # store data
    try:
        async with engine.begin() as conn:
            all_tickers = await fetch_stock_ids(conn, ticker)
            metadata_df = metadata_df.replace({np.nan: None})
            metadata_df = attach_stock_ids(metadata_df, metadata_df, all_tickers)

            metadata_df["created_at"] = pd.to_datetime(
                metadata_df["created_at"]
            ).dt.normalize()
            metadata_df = metadata_df.drop_duplicates(
                subset=["stock_id", "created_at"], keep="last"
            )

            await insert_dynamic_data(conn, metadata_df)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(exec_dynamic_data())
