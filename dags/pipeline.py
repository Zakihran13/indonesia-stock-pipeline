import os
from pathlib import Path
import sys
from dotenv import load_dotenv
from loguru import logger
import asyncio

# Make direct script execution work by ensuring the project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

from datetime import datetime, timedelta, timezone
from airflow.sdk import Asset, Param, dag, task

from stock_market.extract.raw_stock_data import exec_metadata
from stock_market.extract.raw_price_data_ingestion import (
    exec_historical_price_data,
)
from stock_market.transform.metadata_ingestion import ingest_metadata
from stock_market.transform.dynamic_data_ingestion import exec_dynamic_data
from stock_market.transform.price_data_ingestion import exec_price_data
from stock_market.transform.indicators import materialize_indicators

from data.db.run_db_transformed import run_db

default_args = {"owner": "suzaki", "retries": 2, "retry_delay": timedelta(minutes=5)}

# 2. Define the logical assets that connect the indonesia-stock-pipeline DAGs
raw_daily_asset = Asset("indonesia-stock-pipeline://raw_daily")
db_prep_asset = Asset("indonesia-stock-pipeline://db_prepared")
metadata_asset = Asset("indonesia-stock-pipeline://metadata_transformed")
indicators_asset = Asset("indonesia-stock-pipeline://indicators_transformed")

ingestion_period = os.getenv("YF_HISTORICAL_PERIOD") or "5y"

# Static, parse-time-safe default for the backfill start date. When START_DATE is
# unset, the fallback (365 days before run time) is resolved inside the tasks so
# the Dag version does not change on every parse.
default_start_date = os.getenv("START_DATE")


def _resolve_start_date(start_date: str | None) -> datetime:
    if start_date:
        return datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - timedelta(days=365)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


# =========================== raw data flow ==================================================
@dag(
    dag_id="daily_raw_ingestion",
    default_args=default_args,
    start_date=datetime(2026, 8, 14),
    schedule=None,
    catchup=False,
    tags=["ingestion", "yfinance"],
    is_paused_upon_creation=True,
)
def daily_raw_ingestion():
    @task
    def get_stock_raw():
        return asyncio.run(exec_metadata())

    @task
    def get_price_raw():
        return asyncio.run(exec_historical_price_data(period="1d"))

    # 3. Create a final task that runs after both async extraction tasks finish.
    # The 'outlets' parameter tells Airflow to update the asset when this succeeds.
    @task(outlets=[raw_daily_asset])
    def mark_raw_done():
        logger.info("Raw extraction complete. Triggering downstream DAGs.")

    stock_data = get_stock_raw()
    price_data = get_price_raw()

    # Set the bitshift dependencies so mark_raw_done waits for both
    [stock_data, price_data] >> mark_raw_done()


@dag(
    dag_id="historical_price_raw",
    default_args=default_args,
    start_date=datetime(2026, 8, 14),
    schedule=None,
    catchup=False,
    tags=["ingestion", "yfinance"],
    is_paused_upon_creation=True,
)
def historical_price_raw():
    @task
    def get_price_raw():
        return asyncio.run(exec_historical_price_data(period=ingestion_period))

    price_data = get_price_raw()


@dag(
    dag_id="historical_data_transform",
    default_args=default_args,
    start_date=datetime(2026, 8, 14),
    schedule=[metadata_asset],
    catchup=False,
    tags=["ingestion", "yfinance"],
    is_paused_upon_creation=True,
    params={
        "start_date": Param(
            default_start_date,
            type=["string", "null"],
            format="date",
            description="Backfill start date (YYYY-MM-DD). When empty, falls "
            "back to 365 days before the run date.",
        )
    },
)
def historical_data_transform():
    @task
    def historical_metadata_transform():
        return asyncio.run(ingest_metadata())

    @task
    def historical_stock_transform(start_date: str | None):
        return asyncio.run(
            exec_dynamic_data(start_date=_resolve_start_date(start_date))
        )

    @task
    def historical_price_transform(start_date: str | None):
        return asyncio.run(
            exec_price_data(start_date=_resolve_start_date(start_date))
        )

    @task(outlets=[indicators_asset])
    def historical_indicators_transform(start_date: str | None):
        return asyncio.run(
            materialize_indicators(start_date=_resolve_start_date(start_date))
        )

    start_date = "{{ params.start_date or '' }}"
    # metadata = historical_metadata_transform()
    stock_data = historical_stock_transform(start_date)
    price_data = historical_price_transform(start_date)
    # metadata >> [stock_data, price_data]
    [stock_data, price_data] >> historical_indicators_transform(start_date)


# =========================== daily data flow ==================================================
@dag(
    dag_id="prepare_db_transform",
    default_args=default_args,
    start_date=datetime(2026, 8, 14),
    schedule=[raw_daily_asset],  # 4. Listens for the raw_daily_asset
    catchup=False,
    tags=["ingestion", "yfinance"],
    is_paused_upon_creation=True,
)
def prepare_db_transform():
    # 5. When SQLAlchemy preparation finishes, update the next asset
    @task(outlets=[db_prep_asset])
    def db_transform():
        return asyncio.run(run_db())

    db_transform()


@dag(
    dag_id="metadata_ingestion",
    default_args=default_args,
    start_date=datetime(2026, 8, 14),
    schedule=[db_prep_asset],  # 6. Listens for db_prep_asset
    catchup=False,
    tags=["ingestion", "yfinance"],
    is_paused_upon_creation=True,
)
def metadata_ingestion():
    @task(outlets=[metadata_asset])
    def transform_metadata():
        return asyncio.run(ingest_metadata())

    transform_metadata()


@dag(
    dag_id="daily_data_transform",
    default_args=default_args,
    start_date=datetime(2026, 8, 14),
    schedule=[metadata_asset],  # 7. Listens for metadata_asset
    catchup=False,
    tags=["ingestion", "yfinance"],
    is_paused_upon_creation=True,
)
def daily_data_transform():
    @task
    def daily_stock_transform():
        return asyncio.run(exec_dynamic_data())

    @task
    def daily_price_transform():
        return asyncio.run(exec_price_data())

    @task(outlets=[indicators_asset])
    def daily_indicators_transform():
        return asyncio.run(materialize_indicators())

    stock_data = daily_stock_transform()
    price_data = daily_price_transform()
    [stock_data, price_data] >> daily_indicators_transform()


daily_pipeline = daily_raw_ingestion()
historical_raw_data = historical_price_raw()
historical_ingestion = historical_data_transform()

db_transform_preparation = prepare_db_transform()
metadata_transform_flow = metadata_ingestion()
daily_data_transform_flow = daily_data_transform()
