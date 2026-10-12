from typing import Any, Callable, List, TypeVar
import re
import os
import random
import threading
import asyncio
import time
import math
import datetime
import pandas as pd
import inflection
from loguru import logger
from decimal import Decimal, InvalidOperation
from sqlalchemy import String, Text, Numeric, Integer, BigInteger, DateTime, Date

YF_GLOBAL_MIN_INTERVAL_SECONDS = float(
    os.getenv("YF_GLOBAL_MIN_INTERVAL_SECONDS", "1.5")
)
YF_BACKOFF_MAX_RETRIES = int(os.getenv("YF_BACKOFF_MAX_RETRIES", "5"))
YF_BACKOFF_BASE_SECONDS = float(os.getenv("YF_BACKOFF_BASE_SECONDS", "5"))
YF_BACKOFF_MAX_SECONDS = float(os.getenv("YF_BACKOFF_MAX_SECONDS", "120"))

# Substrings identifying Yahoo throttling / transient upstream failures.
# yfinance surfaces these inconsistently across versions (YFRateLimitError,
# HTTP errors, or plain text messages), so message matching is the fallback.
_TRANSIENT_YF_ERROR_NEEDLES = (
    "429",
    "too many requests",
    "rate limit",
    "403",
    "forbidden",
    "502",
    "503",
    "504",
    "service unavailable",
    "gateway",
    "timed out",
    "timeout",
    "connection reset",
    "temporarily",
    "max retries exceeded",
)

T = TypeVar("T")


class GlobalYfRateLimiter:
    """Process-wide pacing for outbound yfinance requests.

    Every caller - asyncio tasks and ``asyncio.to_thread`` workers alike -
    reserves the next request slot from a single shared timestamp under a
    thread lock, so concurrent batches are spaced by at least ``min_interval``
    seconds no matter which concurrency primitive runs them. Slots are
    reserved ahead of time, which keeps the spacing even and prevents
    concurrent workers from collapsing into a burst.

    The interval is adaptive: a rate-limit response multiplies it
    (``penalize``) and every successful request decays it back toward the base
    value (``reward``), so the whole flow slows down while Yahoo is
    throttling instead of hammering it harder.
    """

    def __init__(
        self,
        min_interval: float,
        penalty_factor: float = 2.0,
        ceiling: float = 60.0,
    ):
        self._base_interval = max(0.0, float(min_interval))
        self._interval = self._base_interval
        self._penalty_factor = max(1.0, float(penalty_factor))
        self._ceiling = max(self._base_interval, float(ceiling))
        self._lock = threading.Lock()
        self._next_slot = 0.0

    @property
    def interval(self) -> float:
        """Currently effective per-request interval in seconds."""
        with self._lock:
            return self._interval

    @property
    def base_interval(self) -> float:
        return self._base_interval

    def _reserve_slot(self) -> float:
        """Claims the next request slot and returns how long the caller waits."""
        with self._lock:
            now = time.monotonic()
            slot_start = max(now, self._next_slot)
            self._next_slot = slot_start + self._interval
            return slot_start - now

    def acquire_blocking(self) -> float:
        """Waits for this caller's slot. Safe to call from worker threads."""
        wait = self._reserve_slot()
        if wait > 0:
            time.sleep(wait)
        return wait

    async def acquire(self) -> float:
        """Async variant of :meth:`acquire_blocking`."""
        wait = self._reserve_slot()
        if wait > 0:
            await asyncio.sleep(wait)
        return wait

    def penalize(self) -> float:
        """Escalates pacing after a rate-limit response; returns new interval."""
        with self._lock:
            self._interval = min(
                self._ceiling,
                max(
                    self._interval * self._penalty_factor,
                    self._base_interval * self._penalty_factor,
                ),
            )
            return self._interval

    def reward(self) -> float:
        """Decays the interval back toward the base after a successful request."""
        with self._lock:
            if self._interval > self._base_interval:
                self._interval = max(
                    self._base_interval, self._interval * 0.9
                )
            return self._interval


# Shared by every flow in this process; import and call
# `await wait_for_global_yf_rate_limit()` (async) or
# `yf_rate_limiter.acquire_blocking()` (threads) before each yfinance request.
yf_rate_limiter = GlobalYfRateLimiter(YF_GLOBAL_MIN_INTERVAL_SECONDS)


def is_transient_yf_error(error: BaseException) -> bool:
    """True for Yahoo rate limiting and other transient transport failures."""

    try:
        from yfinance.exceptions import YFRateLimitError

        if isinstance(error, YFRateLimitError):
            return True
    except Exception:
        pass

    message = str(error).lower()
    return any(needle in message for needle in _TRANSIENT_YF_ERROR_NEEDLES)


def compute_backoff_delay(
    attempt: int,
    base: float = YF_BACKOFF_BASE_SECONDS,
    maximum: float = YF_BACKOFF_MAX_SECONDS,
    jitter: float = 0.2,
) -> float:
    """Exponential backoff with jitter: ``base * 2**attempt``, capped at ``maximum``."""

    delay = min(maximum, base * (2 ** max(0, attempt)))
    if jitter:
        delay *= random.uniform(1 - jitter, 1 + jitter)
    return max(0.0, min(maximum, delay))


def with_yf_backoff(
    fn: Callable[[], T],
    description: str = "yfinance request",
    max_retries: int | None = None,
) -> T:
    """Runs a blocking yfinance call, retrying transient failures with backoff.

    Each retry sleeps ``base * 2**attempt`` (jittered) and escalates the global
    rate limiter, so *all* workers slow down while Yahoo is throttling us.
    Non-transient errors are raised immediately; the last transient error is
    re-raised once retries are exhausted.
    """

    retries = YF_BACKOFF_MAX_RETRIES if max_retries is None else max_retries
    attempt = 0

    while True:
        try:
            return fn()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not is_transient_yf_error(error) or attempt >= retries:
                raise

            delay = compute_backoff_delay(attempt)
            new_interval = yf_rate_limiter.penalize()
            logger.warning(
                "{} hit a transient error ({}); backing off {:.1f}s "
                "(retry {}/{}), global yfinance interval now {:.1f}s",
                description,
                error,
                delay,
                attempt + 1,
                retries,
                new_interval,
            )
            time.sleep(delay)
            attempt += 1


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

    await yf_rate_limiter.acquire()


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


def dataframe_to_records(table, df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame into DB-safe records.

    Pandas uses NaN/NaT for missing values, but the database layer expects None
    so those values become SQL NULL.

    Some Yahoo payload fields come as numeric values for string DB columns
    (e.g. fax/phone can be 0). Coerce those to strings before binding.

    Some numeric fields can exceed DB precision (NUMERIC(p,s)); these are
    converted to None to avoid asyncpg numeric overflow errors.
    """
    sanitized = df.astype(object).where(pd.notna(df), None)
    records = sanitized.to_dict(orient="records")

    # Group columns by their SQLAlchemy type
    string_columns = {
        column.name
        for column in table.__table__.columns
        if isinstance(column.type, (String, Text))
    }

    numeric_columns = {
        column.name: column.type
        for column in table.__table__.columns
        if isinstance(column.type, Numeric)
    }

    integer_columns = {
        column.name: column.type
        for column in table.__table__.columns
        if isinstance(column.type, (Integer, BigInteger))
    }

    # FIX: Added `Date` to the types we need to catch
    datetime_columns = {
        column.name: column.type
        for column in table.__table__.columns
        if isinstance(column.type, (DateTime, Date))
    }

    def sanitize_numeric(value, numeric_type: Numeric):
        if value is None:
            return None

        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return None

        if not decimal_value.is_finite():
            return None

        precision = numeric_type.precision
        scale = numeric_type.scale

        if precision is not None and scale is not None:
            limit = Decimal(10) ** (precision - scale)
            if abs(decimal_value) >= limit:
                return None

        return float(decimal_value)

    def sanitize_integer(value, integer_type):
        if value is None:
            return None

        if isinstance(value, bool):
            return int(value)

        if isinstance(value, float) and not math.isfinite(value):
            return None

        try:
            int_value = int(value)
        except (TypeError, ValueError):
            return None

        if isinstance(integer_type, BigInteger):
            if int_value < -9223372036854775808 or int_value > 9223372036854775807:
                return None

        return int_value

    def sanitize_datetime(value, datetime_type):
        if value is None or pd.isna(value):
            return None

        # FIX: Handle numeric timestamps (like 1776988800.0) from Yahoo Finance
        if isinstance(value, (int, float)):
            try:
                # Timestamps around 1.7 billion are in seconds (matches 2026)
                value = pd.to_datetime(value, unit="s")
            except (ValueError, TypeError):
                return None

        # If it's a string, attempt to parse it
        elif isinstance(value, str):
            try:
                value = pd.to_datetime(value)
            except (ValueError, TypeError):
                return None

        # By this point, it should be a Pandas Timestamp if it was valid data
        if isinstance(value, pd.Timestamp):
            ts = value
            # SQLAlchemy Date columns don't have a timezone attribute, so use getattr to be safe
            has_tz = getattr(datetime_type, "timezone", False)

            if ts.tzinfo is not None and not has_tz:
                ts = ts.tz_convert("UTC").tz_localize(None)

            # FIX: If the target column is Date (not DateTime), return a date object
            if isinstance(datetime_type, Date):
                return ts.date()

            return ts.to_pydatetime()

        # Fallback if the dataframe directly contained a native datetime/date object
        if isinstance(value, datetime.datetime):
            if isinstance(datetime_type, Date):
                return value.date()
            return value

        if isinstance(value, datetime.date):
            return value

        return None

    # Apply sanitization to the records
    for record in records:
        for column_name in string_columns:
            value = record.get(column_name)
            if value is not None and not isinstance(value, str):
                record[column_name] = str(value)

        for column_name, numeric_type in numeric_columns.items():
            if column_name in record:
                record[column_name] = sanitize_numeric(
                    record.get(column_name), numeric_type
                )

        for column_name, integer_type in integer_columns.items():
            if column_name in record:
                record[column_name] = sanitize_integer(
                    record.get(column_name), integer_type
                )

        for column_name, datetime_type in datetime_columns.items():
            if column_name in record:
                record[column_name] = sanitize_datetime(
                    record.get(column_name), datetime_type
                )

    return records
