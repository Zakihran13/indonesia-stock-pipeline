import asyncio
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from data.db import entities_transformed as et
from data.db.client import init_async_db
from data.db.statements import insert_indicators_data

RECOMMENDATION_SCORES = {
    "strong buy": 2,
    "buy": 1,
    "hold": 0,
    "neutral": 0,
    "sell": -1,
    "strong sell": -2,
}


async def _fetch_frame(conn, model) -> pd.DataFrame:
    result = await conn.execute(select(model))
    return pd.DataFrame(result.mappings().all())


def _asof_merge(
    prices: pd.DataFrame, snapshots: pd.DataFrame, snapshot_date: str
) -> pd.DataFrame:
    if snapshots.empty:
        return prices

    snapshots = snapshots.copy()
    snapshots[snapshot_date] = pd.to_datetime(snapshots[snapshot_date])
    frames = []
    for stock_id, price_group in prices.groupby("stock_id", sort=False):
        snapshot_group = snapshots[snapshots["stock_id"] == stock_id]
        price_group = price_group.sort_values("indicator_date")
        if snapshot_group.empty:
            frames.append(price_group)
            continue
        snapshot_group = snapshot_group.copy()
        snapshot_group["__snapshot_date"] = snapshot_group[snapshot_date]
        snapshot_columns = [
            "__snapshot_date",
            *[
                column
                for column in snapshot_group.columns
                if column not in price_group.columns
                and column not in {"stock_id", snapshot_date, "__snapshot_date"}
            ],
        ]
        snapshot_group = snapshot_group.reindex(columns=snapshot_columns)
        merged = pd.merge_asof(
            price_group,
            snapshot_group.sort_values("__snapshot_date"),
            left_on="indicator_date",
            right_on="__snapshot_date",
            direction="backward",
            suffixes=("", "_snapshot"),
        )
        frames.append(merged.drop(columns="__snapshot_date"))
    return pd.concat(frames, ignore_index=True)


def build_indicator_frame(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    fundamentals: pd.DataFrame,
    dynamic: pd.DataFrame,
    analytics: pd.DataFrame,
    start_date: datetime | None = None,
) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()

    indicators = prices.copy()
    indicators["indicator_date"] = pd.to_datetime(indicators["trade_date"])
    indicators = indicators.sort_values(["stock_id", "indicator_date"])
    indicators = _asof_merge(indicators, metadata, "updated_at")
    indicators = _asof_merge(indicators, fundamentals, "retrieve_at")
    indicators = _asof_merge(indicators, dynamic, "retrieve_at")
    indicators = _asof_merge(indicators, analytics, "retrieve_at")

    for column in (
        "sector",
        "market_cap",
        "free_cashflow",
        "current_price",
        "target_mean_price",
        "recommendation_status",
        "earnings_start_date",
        "return_on_equity",
        "payout_ratio",
        "total_debt",
        "average_analyst_rating",
    ):
        if column not in indicators:
            indicators[column] = np.nan

    grouped = indicators.groupby("stock_id", sort=False)
    indicators["daily_return"] = grouped["close"].pct_change()
    indicators["moving_average_30d"] = grouped["close"].transform(
        lambda values: values.rolling(30, min_periods=30).mean()
    )
    indicators["momentum_30d"] = grouped["close"].transform(
        lambda values: values.pct_change(30)
    )
    indicators["volatility_5d"] = grouped["daily_return"].transform(
        lambda values: values.rolling(5, min_periods=5).std() * np.sqrt(252)
    )
    indicators["volatility_20d"] = grouped["daily_return"].transform(
        lambda values: values.rolling(20, min_periods=20).std() * np.sqrt(252)
    )
    indicators["average_volume_20d"] = grouped["volume"].transform(
        lambda values: values.rolling(20, min_periods=1).mean()
    )

    market_cap = pd.to_numeric(indicators["market_cap"], errors="coerce")
    free_cashflow = pd.to_numeric(indicators["free_cashflow"], errors="coerce")
    indicators["free_cashflow_to_market_cap"] = free_cashflow.div(
        market_cap.replace(0, np.nan)
    )

    current_price = pd.to_numeric(indicators["current_price"], errors="coerce")
    target_price = pd.to_numeric(indicators["target_mean_price"], errors="coerce")
    indicators["target_price_deviation"] = target_price.div(
        current_price.replace(0, np.nan)
    ).sub(1)

    recommendation = (
        indicators["recommendation_status"]
        .astype("string")
        .str.lower()
        .map(RECOMMENDATION_SCORES)
    )
    indicators["recommendation_score"] = recommendation
    indicators["recommendation_score_change"] = grouped["recommendation_score"].diff()

    earnings_date = pd.to_datetime(
        indicators["earnings_start_date"], errors="coerce"
    ).dt.normalize()
    indicator_dates = indicators["indicator_date"].dt.normalize()
    indicators["days_until_earnings"] = (earnings_date - indicator_dates).dt.days
    indicators["earnings_within_7d"] = indicators["days_until_earnings"].between(
        0, 7, inclusive="both"
    )

    if start_date is not None:
        start_timestamp = pd.Timestamp(start_date).tz_localize(None).normalize()
        indicators = indicators[indicators["indicator_date"] >= start_timestamp]

    indicators["indicator_date"] = indicators["indicator_date"].dt.date
    output_columns = list(et.IndicatorsData.__table__.columns.keys())
    return indicators.reindex(columns=output_columns).replace({np.nan: None})


async def materialize_indicators(start_date: datetime | None = None) -> int:
    """Build and upsert indicators, optionally starting from a historical date."""
    engine = init_async_db()
    try:
        async with engine.begin() as conn:
            frames = await asyncio.gather(
                _fetch_frame(conn, et.PriceData),
                _fetch_frame(conn, et.StockMetadata),
                _fetch_frame(conn, et.FundamentalData),
                _fetch_frame(conn, et.DynamicData),
                _fetch_frame(conn, et.AnalyticData),
            )
            indicator_frame = build_indicator_frame(*frames, start_date=start_date)
            await insert_indicators_data(conn, indicator_frame)
            logger.info("Materialized {} indicator rows", len(indicator_frame))
            return len(indicator_frame)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(materialize_indicators())
