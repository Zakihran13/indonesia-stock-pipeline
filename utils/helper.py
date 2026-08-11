from typing import List, Any
import pandas as pd
import inflection
import re
import os
import asyncio
import time
from loguru import logger

YF_GLOBAL_MIN_INTERVAL_SECONDS = float(
    os.getenv("YF_GLOBAL_MIN_INTERVAL_SECONDS", "1.5")
)

_yf_rate_limit_lock = asyncio.Lock()
_yf_last_request_ts = 0.0


def split_batch(data: List[Any], batch_size: int) -> List[List[Any]]:
    """Splits a list into smaller lists of the specified size."""
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0.")

    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def _to_snake_case(col: Any) -> str:
    col_str = str(col).strip()
    col_str = col_str.replace("&", "_and_")
    col_str = col_str.replace("%", "_percent")
    col_str = col_str.replace("#", "_num_")
    col_str = inflection.underscore(col_str)
    col_str = re.sub(r"[^a-zA-Z0-9]+", "_", col_str)
    col_str = re.sub(r"_+", "_", col_str)
    return col_str.strip("_").lower()


def snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames all columns in a pandas DataFrame to snake_case
    using the inflection library and regex formatting.
    """
    df = df.copy()
    df.columns = [_to_snake_case(col) for col in df.columns]
    return df


async def wait_for_global_yf_rate_limit() -> None:
    """Enforces a process-wide minimum interval between yfinance requests."""

    global _yf_last_request_ts

    async with _yf_rate_limit_lock:
        now = time.monotonic()
        elapsed = now - _yf_last_request_ts

        if elapsed < YF_GLOBAL_MIN_INTERVAL_SECONDS:
            await asyncio.sleep(YF_GLOBAL_MIN_INTERVAL_SECONDS - elapsed)

        _yf_last_request_ts = time.monotonic()


async def cancel_pending_tasks(tasks: list[asyncio.Task]) -> None:
    """Cancels pending tasks and waits for cancellation to settle."""

    pending_tasks = [task for task in tasks if not task.done()]
    for task in pending_tasks:
        task.cancel()

    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)


def log_bulk_write_results(results: list) -> None:
    """
    Parses a list of BulkWriteResult objects and Exceptions from asyncio.gather,
    aggregates the counts, and logs the final result.
    """
    total_matched = 0
    total_modified = 0
    total_upserted = 0

    for res in results:
        if isinstance(res, Exception):
            # Log individual chunk failures
            logger.error(f"A specific chunk failed during bulk write: {res}")
        else:
            # Aggregate success metrics
            total_matched += res.matched_count
            total_modified += res.modified_count
            total_upserted += res.upserted_count

    logger.info(
        f"**_Upsert data successful!_** "
        f"Total Matched: {total_matched} | "
        f"Total Modified: {total_modified} | "
        f"Total Inserted: {total_upserted}"
    )
