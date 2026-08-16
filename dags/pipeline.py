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
from airflow.sdk import dag, task

from stock_market.extract.raw_stock_data import exec_metadata
from stock_market.extract.raw_price_data_ingestion import (
    exec_historical_price_data,
)
from stock_market.transform.metadata_ingestion import ingest_metadata
from stock_market.transform.dynamic_data_ingestion import exec_dynamic_data

from data.db.run_db_transformed import run_db

default_args = {"owner": "suzaki", "retries": 2, "retry_delay": timedelta(minutes=5)}


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

    stock_data = get_stock_raw()
    price_data = get_price_raw()


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
    schedule=None,
    catchup=False,
    tags=["ingestion", "yfinance"],
)
def prepare_db_transform():
    @task
    def db_transform():
        return asyncio.run(run_db())

    db = db_transform()

@dag(
    dag_id="metadata_ingestion",
    default_args=default_args,
    start_date=datetime(2026, 8, 14),
    schedule=None,
    catchup=False,
    tags=["ingestion", "yfinance"],
)
def metadata_ingestion():
    @task
    def transform_metadata():
        return asyncio.run(ingest_metadata())

    metadata_transformed = transform_metadata()


@dag(
    dag_id="daily_data_transform",
    default_args=default_args,
    start_date=datetime(2026, 8, 14),
    schedule="@daily",
    catchup=False,
    tags=["ingestion", "yfinance"],
)
def daily_data_transform():
    @task
    def daily_transform():
        return asyncio.run(exec_dynamic_data())

    daily_transform_data = daily_transform()



daily_pipeline = daily_raw_ingestion()
historical_raw_data = historical_price_raw()

db_transform_preparation = prepare_db_transform()
metadata_ingestion_flow = metadata_ingestion()
daily_dynamic_ingestion_flow = daily_data_transform()
