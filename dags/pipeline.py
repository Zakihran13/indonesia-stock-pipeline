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

default_args = {"owner": "suzaki", "retries": 2, "retry_delay": timedelta(minutes=5)}


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
        return asyncio.run(exec_historical_price_data(period="5d"))

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


daily_pipeline = daily_raw_ingestion()
historical_raw_data = historical_price_raw()
