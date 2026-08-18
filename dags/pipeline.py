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

from datetime import datetime, timedelta
from airflow.sdk import dag, task, Asset

from stock_market.extract.raw_stock_data import exec_metadata
from stock_market.extract.raw_price_data_ingestion import (
    exec_historical_price_data,
)
from stock_market.transform.metadata_ingestion import ingest_metadata
from stock_market.transform.dynamic_data_ingestion import exec_dynamic_data
from stock_market.transform.price_data_ingestion import exec_price_data

from data.db.run_db_transformed import run_db

default_args = {"owner": "suzaki", "retries": 2, "retry_delay": timedelta(minutes=5)}

# 2. Define the logical assets that connect the indonesia-stock-pipeline DAGs
raw_daily_asset = Asset("indonesia-stock-pipeline://raw_daily")
db_prep_asset = Asset("indonesia-stock-pipeline://db_prepared")
metadata_asset = Asset("indonesia-stock-pipeline://metadata_transformed")


# =========================== raw data flow ==================================================
@dag(
    dag_id="daily_raw_ingestion",
    default_args=default_args,
    start_date=datetime(2026, 8, 14),
    schedule="@daily",
    catchup=False,
    tags=["ingestion", "yfinance"],
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
    schedule="@daily",
    catchup=False,
    tags=["ingestion", "yfinance"],
)
def historical_price_raw():
    @task
    def get_price_raw():
        return asyncio.run(exec_historical_price_data(period="5y"))

    price_data = get_price_raw()


# =========================== daily data flow ==================================================
@dag(
    dag_id="prepare_db_transform",
    default_args=default_args,
    start_date=datetime(2026, 8, 14),
    schedule=[raw_daily_asset],  # 4. Listens for the raw_daily_asset
    catchup=False,
    tags=["ingestion", "yfinance"],
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
)
def daily_data_transform():
    @task
    def daily_stock_transform():
        return asyncio.run(exec_dynamic_data())

    @task
    def daily_price_transform():
        return asyncio.run(exec_price_data())

    daily_stock_transform()
    daily_price_transform()


daily_pipeline = daily_raw_ingestion()
historical_raw_data = historical_price_raw()

db_transform_preparation = prepare_db_transform()
metadata_transform_flow = metadata_ingestion()
daily_data_transform_flow = daily_data_transform()
