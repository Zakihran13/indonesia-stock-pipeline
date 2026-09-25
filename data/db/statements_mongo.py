from motor.motor_asyncio import AsyncIOMotorCollection
from typing import List, Dict, Any
import pandas as pd
from pymongo import UpdateOne
from loguru import logger
from functools import partial
import asyncio
from datetime import datetime

from utils.helper import split_batch, log_bulk_write_results


async def fetch_stock_ids(coll: AsyncIOMotorCollection) -> List[str]:
    cursor = coll.find({}, {"ticker": 1})
    data = await cursor.to_list()

    return [n["ticker"] for n in data if "ticker"]


async def fetch_metadata_ticker_dates(
    coll: AsyncIOMotorCollection, ticker: List[str] | None = None
) -> pd.DataFrame:
    """Fetches ticker/market_date pairs only, for gap/staleness detection."""
    params: dict[str, Any] = {}
    if ticker:
        params["ticker"] = {"$in": ticker}

    cursor = coll.find(params, {"ticker": 1, "market_date": 1, "_id": 0})
    data = await cursor.to_list(length=None)

    return (
        pd.DataFrame(data) if data else pd.DataFrame(columns=["ticker", "market_date"])
    )


async def fetch_price_start_dates(
    coll: AsyncIOMotorCollection, ticker: List[str] | None = None
) -> pd.DataFrame:
    """Fetches each ticker's earliest recorded price date, for alignment checks."""
    pipeline: list[dict] = []
    if ticker:
        pipeline.append({"$match": {"ticker": {"$in": ticker}}})
    pipeline.append({"$group": {"_id": "$ticker", "start_date": {"$min": "$date"}}})

    cursor = coll.aggregate(pipeline)
    data = await cursor.to_list(length=None)

    if not data:
        return pd.DataFrame(columns=["ticker", "start_date"])

    return pd.DataFrame(
        [{"ticker": d["_id"], "start_date": d["start_date"]} for d in data]
    )


async def process_chunk(
    coll: AsyncIOMotorCollection,
    chunk: list[dict],
    conflict_cols: list[str],
    semaphore: asyncio.Semaphore,
):
    batch_operations = [
        UpdateOne(
            {col: record[col] for col in conflict_cols}, {"$set": record}, upsert=True
        )
        for record in chunk
    ]

    # Bound concurrency so we don't hold every chunk's BSON payload in memory at once.
    async with semaphore:
        return await coll.bulk_write(batch_operations, ordered=False)


async def upsert_data(
    coll: AsyncIOMotorCollection,
    data: pd.DataFrame,
    conflict_cols: list[str],
    batch_size: int = 2_000,
    max_concurrency: int = 4,
):
    if not conflict_cols:
        raise ValueError("conflict_cols cannot be empty.")

    if data.empty:
        logger.info("DataFrame is empty. Nothing to upsert.")
        return

    logger.info(f"inserting data for: **{' '.join(pd.unique(data["ticker"]))}**")
    records = data.to_dict("records")
    batch_data = split_batch(records, batch_size)
    semaphore = asyncio.Semaphore(max_concurrency)
    operation_tasks = [
        process_chunk(coll, chunk, conflict_cols, semaphore) for chunk in batch_data
    ]

    if operation_tasks:
        try:
            results = await asyncio.gather(*operation_tasks, return_exceptions=True)
            log_bulk_write_results(results)
        except Exception as e:
            logger.error(f"An error occurred during bulk write: {e}")


async def fetch_dynamic_raw(
    coll: AsyncIOMotorCollection,
    current_date: datetime,
    ticker: List[str] | None = None,
) -> pd.DataFrame | None:
    params: dict[str, Any] = {
        "created_at": {
            "$gte": current_date.replace(hour=0, minute=0, second=0, microsecond=0)
        }
    }

    if ticker:
        params["ticker"] = {"$in": ticker}

    cursor = coll.find(params)
    data = await cursor.to_list(length=None)

    if data:
        return pd.DataFrame(data)
    return


async def fetch_price_raw(
    coll: AsyncIOMotorCollection,
    current_date: datetime,
    ticker: List[str] | None = None,
) -> pd.DataFrame | None:
    params: dict[str, Any] = {
        "date": {
            "$gte": current_date.replace(hour=0, minute=0, second=0, microsecond=0)
        }
    }

    if ticker:
        params["ticker"] = {"$in": ticker}

    cursor = coll.find(params)
    data = await cursor.to_list(length=None)

    if data:
        return pd.DataFrame(data)
    return


async def fetch_partial_price_raw(
    coll: AsyncIOMotorCollection,
    current_date: datetime,
    tickers: list[str] | None = None,
    batch_size: int = 25_000,
):
    params: dict[str, Any] = {
        "date": {
            "$gte": current_date.replace(hour=0, minute=0, second=0, microsecond=0)
        }
    }

    if tickers:
        params["ticker"] = {"$in": tickers}

    cursor = coll.find(params)

    batch_data = []
    async for doc in cursor:
        batch_data.append(doc)

        if len(batch_data) >= batch_size:
            yield pd.DataFrame(batch_data)
            batch_data = []

    if batch_data:
        yield pd.DataFrame(batch_data)
