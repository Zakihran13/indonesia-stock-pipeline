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


async def process_chunk(
    coll: AsyncIOMotorCollection, chunk: list[dict], conflict_cols: list[str]
):
    batch_operations = [
        UpdateOne(
            {col: record[col] for col in conflict_cols}, {"$set": record}, upsert=True
        )
        for record in chunk
    ]

    return await coll.bulk_write(batch_operations, ordered=False)


async def upsert_data(
    coll: AsyncIOMotorCollection, data: pd.DataFrame, conflict_cols: list[str]
):
    if not conflict_cols:
        raise ValueError("conflict_cols cannot be empty.")

    if data.empty:
        logger.info("DataFrame is empty. Nothing to upsert.")
        return

    logger.info(f"inserting data for: **{' '.join(pd.unique(data["ticker"]))}**")
    records = data.to_dict("records")
    batch_data = split_batch(records, 10_000)
    operation_tasks = [
        process_chunk(coll, chunk, conflict_cols) for chunk in batch_data
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
