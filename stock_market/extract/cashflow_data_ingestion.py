"""Free cash flow ingestion (historical + daily) built on yfinance.

Yahoo exposes free cash flow in two places, both with limited coverage:

1. The cash flow statement (``ticker.cashflow`` / ``ticker.quarterly_cashflow``)
   only keeps a handful of periods (~4 annual / ~5 quarterly), and individual
   periods frequently ship with the ``Free Cash Flow`` line missing entirely.
2. ``ticker.info`` carries a trailing-twelve-months snapshot
   (``freeCashflow`` / ``operatingCashflow``) that is often null for Indonesian
   tickers.

So instead of reading a single field, every period is assembled through a
fallback chain:

========================  ==================================================
fcf_source                 how ``free_cashflow`` is obtained
========================  ==================================================
``cashflow_statement``     ``Free Cash Flow`` row of the statement
``derived_ocf_capex``      ``Operating Cash Flow`` +/- ``Capital Expenditure``
``info_ttm``               ``info["freeCashflow"]`` (TTM row only)
``derived_ttm_ocf_capex``  TTM ``info["operatingCashflow"]`` +/- latest CapEx
========================  ==================================================

Rows are upserted into the ``raw_cashflow_data`` Mongo collection (new table)
under the ``raw_stock_data_ingestion`` database, keyed by
``(ticker, period_type, period_end)``.

Modes:
    * ``historical`` - all annual + quarterly statement periods plus the TTM
      snapshot, optionally filtered to ``[start, end]``.
    * ``daily``      - only the newest quarterly period (falls back to the
      newest annual period when the quarterly statement is empty) plus a
      refreshed TTM snapshot, so the daily DAG stays cheap.

TTM records require a period date from Yahoo or a statement. Records without
one are skipped rather than being assigned the extraction date.

Rate limiting / backoff (Yahoo rejects abusive request patterns):
    1. Every yfinance request first reserves a slot on the process-wide
       ``utils.helper.yf_rate_limiter``, so all concurrent batches share one
       ``YF_GLOBAL_MIN_INTERVAL_SECONDS`` spacing instead of bursting.
    2. Transient errors (429/403/5xx/timeouts, ``YFRateLimitError``) are
       retried with exponential backoff + jitter via ``with_yf_backoff``;
       each retry also multiplies the shared interval (adaptive penalty) so
       every worker slows down, and each success decays it back to base.
    3. A batch that comes back completely empty (silent throttling) is
       re-fetched after a longer backoff.
"""

import asyncio
import math
import os
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable

import aiometer
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorCollection

# Make direct script execution work by ensuring the project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

from data.db.client import get_async_mongodb
from data.db.statements_mongo import fetch_stock_ids, upsert_data
from utils.helper import (
    compute_backoff_delay,
    split_batch,
    with_yf_backoff,
    yf_rate_limiter,
)

logger.remove()
logger.add(
    sys.stderr,
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    backtrace=True,
    diagnose=False,
)

CASHFLOW_DB_NAME = "raw_stock_data_ingestion"
CASHFLOW_COLLECTION = "raw_cashflow_data"
METADATA_COLLECTION = "raw_stock_data"

# Cash flow statements are fetched per ticker (2-3 requests each), so batches
# stay smaller than the price flow's 50-ticker batches.
YF_CASHFLOW_BATCH_SIZE = int(os.getenv("YF_CASHFLOW_BATCH_SIZE", "25"))
YF_CASHFLOW_MAX_AT_ONCE = int(os.getenv("YF_CASHFLOW_MAX_AT_ONCE", "5"))

# A batch that returns nothing for every ticker is usually a silent Yahoo
# throttle rather than a batch of data-less tickers, so it gets re-fetched.
YF_CASHFLOW_EMPTY_RETRIES = int(os.getenv("YF_CASHFLOW_EMPTY_RETRIES", "3"))
YF_CASHFLOW_EMPTY_BACKOFF_SECONDS = float(
    os.getenv("YF_CASHFLOW_EMPTY_BACKOFF_SECONDS", "10")
)

VALID_MODES = ("historical", "daily")

# Statement row labels as published by yfinance (matched case-insensitively),
# ordered from canonical to looser alternatives used to rebuild missing values.
FCF_ROW_ALIASES = ("free cash flow",)
OCF_ROW_ALIASES = (
    "operating cash flow",
    "cash flowsfromusedin operating activities direct",
    "total cash from operating activities",
)
CAPEX_ROW_ALIASES = (
    "capital expenditure",
    "purchase of ppe",
    "net ppe purchase and sale",
)


def _safe_float(value: Any) -> float | None:
    """Converts a yfinance cell into a finite float, or None when unusable."""

    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def _as_naive_date(value: Any) -> datetime | None:
    """Normalizes any timestamp-like value to a tz-naive UTC date."""

    if value is None:
        return None

    try:
        ts = pd.to_datetime(value, errors="coerce")
    except (TypeError, ValueError):
        return None

    if pd.isna(ts):
        return None

    if isinstance(ts, pd.DatetimeIndex):
        if len(ts) == 0:
            return None
        ts = ts[0]

    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)

    return ts.normalize().to_pydatetime()


def _period_from_epoch(seconds: Any) -> datetime | None:
    """Converts Yahoo's unix-second fields (e.g. mostRecentQuarter) to a date."""

    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        return None
    if seconds <= 0:
        return None

    return _as_naive_date(pd.to_datetime(seconds, unit="s"))


def derive_free_cash_flow(
    operating_cashflow: float | None, capital_expenditure: float | None
) -> float | None:
    """FCF = OCF - CapEx.

    Yahoo normally reports CapEx as a negative outflow, but some statements
    report it positive; ``abs()`` makes the derivation correct for both.
    """

    if operating_cashflow is None or capital_expenditure is None:
        return None

    return operating_cashflow - abs(capital_expenditure)


def _build_index_lookup(statement: pd.DataFrame) -> dict[str, int]:
    """Maps lowercased statement row labels to their positional index."""

    lookup: dict[str, int] = {}
    for position, label in enumerate(statement.index):
        lookup.setdefault(str(label).strip().lower(), position)
    return lookup


def _row_value(
    statement: pd.DataFrame,
    index_lookup: dict[str, int],
    aliases: Iterable[str],
    column: Any,
) -> float | None:
    """Returns the first non-null value among the alias rows for a period."""

    for alias in aliases:
        position = index_lookup.get(alias.strip().lower())
        if position is None:
            continue

        value = _safe_float(statement.iloc[position][column])
        if value is not None:
            return value

    return None


def _latest_statement_value(
    statement: pd.DataFrame | None, aliases: Iterable[str]
) -> float | None:
    """Newest non-null value across statement periods (newest column first)."""

    if statement is None or getattr(statement, "empty", True):
        return None

    index_lookup = _build_index_lookup(statement)
    for column in sorted(statement.columns, key=lambda c: str(c), reverse=True):
        value = _row_value(statement, index_lookup, aliases, column)
        if value is not None:
            return value

    return None


def _latest_statement_period(statement: pd.DataFrame | None) -> datetime | None:
    """Newest usable period_end of a statement, or None."""

    if statement is None or getattr(statement, "empty", True):
        return None

    periods = [
        period
        for period in (_as_naive_date(column) for column in statement.columns)
        if period is not None
    ]
    return max(periods) if periods else None


def _info_period_end(info: dict[str, Any]) -> datetime | None:
    """Period end for the TTM row: mostRecentQuarter, then lastFiscalYearEnd."""

    for key in ("mostRecentQuarter", "lastFiscalYearEnd"):
        period = _period_from_epoch(info.get(key))
        if period is not None:
            return period

    return None


def _record(
    ticker: str,
    period_type: str,
    period_end: datetime,
    free_cashflow: float | None,
    operating_cashflow: float | None,
    capital_expenditure: float | None,
    fcf_source: str | None,
    currency: str | None,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "period_type": period_type,
        "period_end": period_end,
        "free_cashflow": free_cashflow,
        "operating_cashflow": operating_cashflow,
        "capital_expenditure": capital_expenditure,
        "fcf_source": fcf_source,
        "currency": currency,
        "fetched_at": datetime.now(UTC),
    }


def build_statement_records(
    ticker: str,
    statement: pd.DataFrame | None,
    period_type: str,
    start: datetime | None = None,
    end: datetime | None = None,
    latest_only: bool = False,
    currency: str | None = None,
) -> list[dict[str, Any]]:
    """Flattens a yfinance cash flow statement into Mongo-ready records.

    Each period column becomes one record. ``free_cashflow`` is read from the
    ``Free Cash Flow`` row when present, otherwise rebuilt from
    ``Operating Cash Flow`` and ``Capital Expenditure`` so periods without the
    canonical row still contribute data.

    ``latest_only`` keeps only the newest period that yields usable values
    (used by the daily flow to stay cheap).
    """

    records: list[dict[str, Any]] = []
    if statement is None or getattr(statement, "empty", True):
        return records

    start_date = _as_naive_date(start)
    end_date = _as_naive_date(end)
    index_lookup = _build_index_lookup(statement)

    periods: list[tuple[datetime, Any]] = []
    for column in statement.columns:
        period_end = _as_naive_date(column)
        if period_end is None:
            continue
        if start_date and period_end < start_date:
            continue
        if end_date and period_end > end_date:
            continue
        periods.append((period_end, column))

    # Newest periods first so `latest_only` keeps the freshest snapshot.
    periods.sort(key=lambda item: item[0], reverse=True)

    for period_end, column in periods:
        operating_cashflow = _row_value(
            statement, index_lookup, OCF_ROW_ALIASES, column
        )
        capital_expenditure = _row_value(
            statement, index_lookup, CAPEX_ROW_ALIASES, column
        )
        free_cashflow = _row_value(statement, index_lookup, FCF_ROW_ALIASES, column)

        fcf_source = None
        if free_cashflow is not None:
            fcf_source = "cashflow_statement"
        else:
            derived = derive_free_cash_flow(operating_cashflow, capital_expenditure)
            if derived is not None:
                free_cashflow = derived
                fcf_source = "derived_ocf_capex"

        # Skip periods with no usable cash flow data at all.
        if (
            free_cashflow is None
            and operating_cashflow is None
            and capital_expenditure is None
        ):
            continue

        records.append(
            _record(
                ticker,
                period_type,
                period_end,
                free_cashflow,
                operating_cashflow,
                capital_expenditure,
                fcf_source,
                currency,
            )
        )

        if latest_only:
            break

    return records


def build_ttm_record(
    ticker: str,
    info: dict[str, Any] | None,
    statement: pd.DataFrame | None = None,
    currency: str | None = None,
) -> dict[str, Any] | None:
    """Builds the trailing-twelve-months snapshot.

    Falls back to ``info["operatingCashflow"]`` combined with the latest
    statement CapEx when ``info["freeCashflow"]`` is null (common for IDX
    tickers). Returns None when nothing usable is available.
    """

    info = info or {}
    free_cashflow = _safe_float(info.get("freeCashflow"))
    operating_cashflow = _safe_float(info.get("operatingCashflow"))

    fcf_source = "info_ttm" if free_cashflow is not None else None
    capital_expenditure: float | None = None

    if free_cashflow is None:
        capital_expenditure = _latest_statement_value(statement, CAPEX_ROW_ALIASES)
        derived = derive_free_cash_flow(operating_cashflow, capital_expenditure)
        if derived is not None:
            free_cashflow = derived
            fcf_source = "derived_ttm_ocf_capex"

    if free_cashflow is None and operating_cashflow is None:
        return None

    period_end = _info_period_end(info) or _latest_statement_period(statement)
    if period_end is None:
        logger.warning(
            "Skipping TTM cashflow record for {} because Yahoo provided no "
            "usable period end",
            ticker,
        )
        return None

    return _record(
        ticker,
        "ttm",
        period_end,
        free_cashflow,
        operating_cashflow,
        capital_expenditure,
        fcf_source,
        currency,
    )


def _yf_request(label: str, fn: Callable[[], Any]) -> Any:
    """One paced, retry-protected yfinance request.

    Waits for this call's slot on the process-wide rate limiter (so concurrent
    batches cannot burst), runs the fetch, and on success decays the adaptive
    interval back toward its base. Transient Yahoo errors are retried with
    exponential backoff inside ``with_yf_backoff``, which also escalates the
    shared interval so every other worker slows down while we are throttled.
    """

    def _paced_call() -> Any:
        yf_rate_limiter.acquire_blocking()
        result = fn()
        yf_rate_limiter.reward()
        return result

    return with_yf_backoff(_paced_call, description=label)


def _statement_has_data(statement: Any) -> bool:
    return statement is not None and not bool(getattr(statement, "empty", True))


def _collect_ticker_cashflow(
    ticker_symbol: str,
    ticker: Any,
    mode: str,
    start: datetime | None,
    end: datetime | None,
) -> list[dict[str, Any]]:
    """Fetches statements + info for one ticker and flattens them to records."""

    annual = None
    quarterly = None
    info: dict[str, Any] = {}

    try:
        quarterly = _yf_request(
            f"quarterly cashflow[{ticker_symbol}]",
            lambda: ticker.quarterly_cashflow,
        )
    except Exception as e:
        logger.warning(
            "Quarterly cashflow statement unavailable for {}: {}", ticker_symbol, e
        )

    try:
        info = _yf_request(f"info[{ticker_symbol}]", lambda: ticker.info) or {}
    except Exception as e:
        logger.warning("Info payload unavailable for {}: {}", ticker_symbol, e)

    # The annual statement is only worth a request when it adds something the
    # daily refresh cannot get elsewhere: full history (historical mode), a
    # quarterly statement with no data, or a CapEx figure used to rebuild a
    # null info freeCashflow. Keeps the daily flow at 2 requests/ticker when
    # Yahoo already reports TTM free cash flow.
    annual_needed = (
        mode == "historical"
        or not _statement_has_data(quarterly)
        or (
            info.get("freeCashflow") is None
            and info.get("operatingCashflow") is not None
        )
    )
    if annual_needed:
        try:
            annual = _yf_request(
                f"annual cashflow[{ticker_symbol}]", lambda: ticker.cashflow
            )
        except Exception as e:
            logger.warning(
                "Annual cashflow statement unavailable for {}: {}", ticker_symbol, e
            )

    currency = info.get("currency") or info.get("financialCurrency")
    records: list[dict[str, Any]] = []

    if mode == "historical":
        records += build_statement_records(
            ticker_symbol, annual, "annual", start, end, currency=currency
        )
        records += build_statement_records(
            ticker_symbol, quarterly, "quarterly", start, end, currency=currency
        )
    else:
        # Daily refresh: only the newest period, preferring the quarterly
        # statement and falling back to annual when it has little data.
        latest = build_statement_records(
            ticker_symbol,
            quarterly,
            "quarterly",
            start,
            end,
            latest_only=True,
            currency=currency,
        )
        if not latest:
            latest = build_statement_records(
                ticker_symbol,
                annual,
                "annual",
                start,
                end,
                latest_only=True,
                currency=currency,
            )
        records += latest

    ttm_record = build_ttm_record(
        ticker_symbol,
        info,
        statement=annual if _statement_has_data(annual) else quarterly,
        currency=currency,
    )
    if ttm_record is not None:
        records.append(ttm_record)

    return records


def _fetch_cashflow_records(
    tickers: list[str],
    mode: str,
    start: datetime | None,
    end: datetime | None,
) -> list[dict[str, Any]]:
    """Blocking yfinance calls for a ticker batch (run via asyncio.to_thread)."""

    yf_tickers = yf.Tickers(" ".join(tickers))
    records: list[dict[str, Any]] = []

    for ticker_symbol, ticker in yf_tickers.tickers.items():
        try:
            logger.debug("Collecting cashflow records for {}", ticker_symbol)
            records += _collect_ticker_cashflow(ticker_symbol, ticker, mode, start, end)
        except Exception:
            logger.exception("Error fetching cashflow data for {}", ticker_symbol)

    return records


async def process_cashflow_batch(
    collection: AsyncIOMotorCollection,
    ticker_batch: list[str],
    mode: str,
    start: datetime | None,
    end: datetime | None,
    batch_label: str = "",
) -> int:
    """Fetches cash flow data for a ticker batch and upserts it into Mongo.

    ``batch_label`` (e.g. ``"3/20"``) prefixes every log line so concurrent
    batches stay tellable apart. Returns how many rows were written, which the
    caller aggregates into the overall progress report.
    """

    flatten_ticker = " ".join(ticker_batch)
    tag = f"[batch {batch_label}] " if batch_label else ""
    logger.info(
        "{}Fetching cashflow data for {} tickers: {}",
        tag,
        len(ticker_batch),
        flatten_ticker,
    )

    try:
        records: list[dict[str, Any]] = []

        for attempt in range(YF_CASHFLOW_EMPTY_RETRIES):
            records = await asyncio.to_thread(
                _fetch_cashflow_records, ticker_batch, mode, start, end
            )
            if records:
                break

            # The whole batch came back empty: most likely a silent Yahoo
            # throttle rather than a batch of data-less tickers. Escalate the
            # shared interval and retry after an exponential backoff.
            if attempt >= YF_CASHFLOW_EMPTY_RETRIES - 1:
                break

            delay = compute_backoff_delay(
                attempt, base=YF_CASHFLOW_EMPTY_BACKOFF_SECONDS
            )
            new_interval = yf_rate_limiter.penalize()
            logger.warning(
                "{}No cashflow data (attempt {}/{}, likely rate limiting); "
                "backing off {:.1f}s, global yfinance interval now {:.1f}s",
                tag,
                attempt + 1,
                YF_CASHFLOW_EMPTY_RETRIES,
                delay,
                new_interval,
            )
            await asyncio.sleep(delay)

        if not records:
            logger.warning("{}No cashflow data returned for: {}", tag, flatten_ticker)
            return 0

        cashflow_data = pd.DataFrame(records).replace({np.nan: None})

        logger.info(
            "{}Fetched {} cashflow rows from {} tickers",
            tag,
            cashflow_data.shape[0],
            len(ticker_batch),
        )

        await upsert_data(
            collection,
            cashflow_data,
            conflict_cols=["ticker", "period_type", "period_end"],
        )

        return cashflow_data.shape[0]

    except asyncio.CancelledError:
        logger.warning("{}Fetch task cancelled", tag)
        raise

    except Exception as e:
        logger.exception("{}Error fetching cashflow data: {}", tag, e)
        return 0


async def _run_cashflow_batch(
    collection: AsyncIOMotorCollection,
    ticker_batch: list[str],
    mode: str,
    start: datetime | None,
    end: datetime | None,
    batch_no: int,
    total_batches: int,
    progress: dict[str, int],
    started_at: float,
) -> None:
    """Runs one batch and reports overall ingestion progress when it finishes.

    Batches run concurrently (``aiometer``), so completions arrive out of
    order; the report shows both the batch that just finished and the running
    totals (batches, rows, elapsed, ETA).
    """

    rows_written = await process_cashflow_batch(
        collection,
        ticker_batch,
        mode,
        start,
        end,
        batch_label=f"{batch_no}/{total_batches}",
    )

    progress["batches_done"] += 1
    progress["rows"] += rows_written

    done = progress["batches_done"]
    elapsed = time.monotonic() - started_at
    eta = (elapsed / done) * (total_batches - done)
    logger.info(
        "Progress: {}/{} batches done (batch {} just finished, {} rows), "
        "{} rows total, {:.0f}s elapsed, {:.0f}s eta",
        done,
        total_batches,
        batch_no,
        rows_written,
        progress["rows"],
        elapsed,
        eta,
    )


async def exec_cashflow_data(
    tickers: list[str] | None = None,
    mode: str = "historical",
    start: datetime | None = None,
    end: datetime | None = None,
) -> None:
    """Fetches free cash flow data for all tickers and stores it in Mongo.

    Writes to the ``raw_cashflow_data`` collection with a unique index on
    ``(ticker, period_type, period_end)`` so re-runs refresh values instead of
    duplicating them.
    """

    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    started_at = time.monotonic()
    logger.info(
        "Starting {} cashflow ingestion (start={}, end={}, tickers={})",
        mode,
        start,
        end,
        f"{len(tickers)} explicit" if tickers else "all from metadata",
    )

    db = get_async_mongodb(CASHFLOW_DB_NAME)
    collection = db[CASHFLOW_COLLECTION]
    metadata_col = db[METADATA_COLLECTION]

    # creating index for first running
    logger.info(
        "[1/5] Connected to Mongo db {}; ensuring unique index on "
        "(ticker, period_type, period_end)",
        CASHFLOW_DB_NAME,
    )
    await collection.create_index(
        [("ticker", 1), ("period_type", 1), ("period_end", 1)],
        unique=1,
        name="ticker_period_end_unique",
    )

    try:
        logger.info("[2/5] Resolving tickers from {}", METADATA_COLLECTION)
        if not tickers:
            tickers = await fetch_stock_ids(metadata_col)

        if not tickers:
            logger.warning("No tickers found in metadata table. Skipping ingestion.")
            return

        batches = split_batch(tickers, YF_CASHFLOW_BATCH_SIZE)
        total_batches = len(batches)
        logger.info(
            "[3/5] Prepared {} ticker batches (batch size {}) from {} tickers",
            total_batches,
            YF_CASHFLOW_BATCH_SIZE,
            len(tickers),
        )

        # Shared counters updated by every batch as it completes; used for the
        # per-batch progress lines and the final summary.
        progress: dict[str, int] = {"batches_done": 0, "rows": 0}

        try:
            logger.info(
                "[4/5] Fetching + upserting {} batches ({} concurrent, "
                "yfinance min interval {:.1f}s, mode={})",
                total_batches,
                YF_CASHFLOW_MAX_AT_ONCE,
                yf_rate_limiter.interval,
                mode,
            )

            semaphore = asyncio.Semaphore(YF_CASHFLOW_MAX_AT_ONCE)

            async def run_limited(batch: list[str], batch_no: int):
                async with semaphore:
                    await _run_cashflow_batch(
                        collection,
                        batch,
                        mode,
                        start,
                        end,
                        batch_no=batch_no,
                        total_batches=total_batches,
                        progress=progress,
                        started_at=started_at,
                    )

            await asyncio.gather(*(run_limited(batch, batch_no) for batch_no, batch in enumerate(batches, start=1)))

            logger.info(
                "[5/5] Done: {} batches, {} cashflow rows from {} tickers "
                "in {:.0f}s",
                total_batches,
                progress["rows"],
                len(tickers),
                time.monotonic() - started_at,
            )

        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.warning("Ctrl+C detected. Cancelling outstanding fetch tasks...")
            logger.info(
                "Stopped after {}/{} batches ({} rows) in {:.0f}s",
                progress["batches_done"],
                total_batches,
                progress["rows"],
                time.monotonic() - started_at,
            )
            return
    finally:
        logger.info(
            "Cashflow ingestion process exited (mode={}, elapsed {:.0f}s)",
            mode,
            time.monotonic() - started_at,
        )


async def exec_historical_cashflow_data(
    tickers: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> None:
    """Full backfill: every annual + quarterly period plus the TTM snapshot.

    ``start``/``end`` optionally filter the statement periods; both default to
    keeping everything yfinance exposes.
    """

    await exec_cashflow_data(tickers=tickers, mode="historical", start=start, end=end)


async def exec_daily_cashflow_data(
    tickers: list[str] | None = None,
) -> None:
    """Cheap daily refresh: newest quarterly period + refreshed TTM snapshot."""

    await exec_cashflow_data(tickers=tickers, mode="daily")


if __name__ == "__main__":
    asyncio.run(exec_historical_cashflow_data(start=datetime(2025,1,1)))
