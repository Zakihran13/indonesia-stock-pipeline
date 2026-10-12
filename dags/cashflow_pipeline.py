"""Airflow DAGs for the free cash flow ingestion flow.

Mirrors the structure of ``dags/pipeline.py``: raw yfinance extraction tasks
are wired together with assets so the cash flow data lands in the
``raw_cashflow_data`` Mongo table.

* ``historical_cashflow_raw`` - manual/backfill run over every statement
  period yfinance exposes (optionally filtered by start/end date).
* ``daily_cashflow_raw``      - runs after the daily raw ingestion finishes,
  refreshing only the newest quarterly period + the TTM snapshot.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys

from dotenv import load_dotenv
from loguru import logger

# Make direct script execution work by ensuring the project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

from airflow.sdk import Asset, Param, dag, task

from stock_market.extract.cashflow_data_ingestion import (
    exec_daily_cashflow_data,
    exec_historical_cashflow_data,
)

default_args = {"owner": "suzaki", "retries": 2, "retry_delay": timedelta(minutes=5)}

# Logical assets connecting this DAG file to the rest of the pipeline.
raw_daily_asset = Asset("indonesia-stock-pipeline://raw_daily")
cashflow_raw_asset = Asset("indonesia-stock-pipeline://raw_cashflow")


def _parse_date(value: str | None) -> datetime | None:
    """Renders a DAG param into a date, tolerating Airflow's empty strings."""

    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        logger.warning("Ignoring unparseable date param: {}", value)
        return None


# =========================== historical free cash flow flow ===========================
@dag(
    dag_id="historical_cashflow_raw",
    default_args=default_args,
    start_date=datetime(2026, 8, 14),
    schedule=None,
    catchup=False,
    tags=["ingestion", "yfinance", "cashflow"],
    is_paused_upon_creation=True,
    params={
        "start_date": Param(
            default=None,
            type=["string", "null"],
            format="date",
            description="Earliest statement period to keep (YYYY-MM-DD). "
            "When empty, keeps the full yfinance history.",
        ),
        "end_date": Param(
            default=None,
            type=["string", "null"],
            format="date",
            description="Latest statement period to keep (YYYY-MM-DD). "
            "When empty, keeps everything up to the newest period.",
        ),
    },
)
def historical_cashflow_raw():
    @task(outlets=[cashflow_raw_asset])
    def get_cashflow_raw(start_date: str | None, end_date: str | None):
        return asyncio.run(
            exec_historical_cashflow_data(
                start=_parse_date(start_date),
                end=_parse_date(end_date),
            )
        )

    start_date = "{{ params.start_date or '' }}"
    end_date = "{{ params.end_date or '' }}"
    get_cashflow_raw(start_date, end_date)


# =========================== daily free cash flow flow ===========================
@dag(
    dag_id="daily_cashflow_raw",
    default_args=default_args,
    start_date=datetime(2026, 8, 14),
    schedule=[raw_daily_asset],  # runs once the daily raw extraction finishes
    catchup=False,
    tags=["ingestion", "yfinance", "cashflow"],
    is_paused_upon_creation=True,
)
def daily_cashflow_raw():
    @task(outlets=[cashflow_raw_asset])
    def get_cashflow_daily():
        return asyncio.run(exec_daily_cashflow_data())

    get_cashflow_daily()


historical_cashflow_flow = historical_cashflow_raw()
daily_cashflow_flow = daily_cashflow_raw()
