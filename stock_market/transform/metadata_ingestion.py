from dotenv import load_dotenv
import sys
from pathlib import Path

import asyncio
import nest_asyncio
from datetime import datetime, timedelta
from loguru import logger

nest_asyncio.apply()
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)


from data.db.statements import insert_metadata, fetch_stock_ids, attach_stock_ids
from data.db.client import init_async_db, get_async_mongodb
from data.db.statements_mongo import fetch_dynamic_raw


async def ingest_metadata(ticker: list[str] | None = None):
    logger.info("Starting the DB Engine...")
    engine = init_async_db()
    engine_mongo = get_async_mongodb("raw_stock_data_ingestion")
    stock_raw = engine_mongo["raw_stock_data"]

    logger.info("Fetching dynamic data!")

    # store data
    try:
        found_any = False
        async for metadata_df in fetch_dynamic_raw(
            stock_raw, datetime.now() - timedelta(days=60), ticker
        ):
            found_any = True
            metadata_df = metadata_df.replace({np.nan: None})
            metadata_df = metadata_df.drop_duplicates(subset=["ticker"], keep="last")
            metadata_df = metadata_df.dropna(subset=["symbol"])

            async with engine.begin() as conn:
                await insert_metadata(conn, metadata_df)

        if not found_any:
            logger.error(f"No data was found for: {ticker}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(ingest_metadata())
