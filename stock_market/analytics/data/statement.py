import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection
from datetime import date, datetime
from typing import List

from data.db import entities_transformed as et
from data.db.client import init_async_db
from stock_market.config import SECTOR_MAPPING


async def get_price_data(
    app: AsyncConnection, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    stmt = select(
        et.PriceData.stock_id,
        et.PriceData.trade_date,
        et.PriceData.high,
        et.PriceData.low,
        et.PriceData.close,
    ).where(et.PriceData.trade_date.between(start_date, end_date))

    data = await app.execute(stmt)
    data = pd.DataFrame(data.mappings().all())

    metadata_stmt = select(et.StockMetadata.stock_id, et.StockMetadata.sector).where(
        et.StockMetadata.sector.is_not(None)
    )

    metadata = await app.execute(metadata_stmt)
    metadata = pd.DataFrame(metadata.mappings().all())
    metadata["sector"] = metadata["sector"].map(SECTOR_MAPPING)

    return pd.merge(data, metadata, how="inner", on="stock_id")


async def get_indicators_data(
    app: AsyncConnection,
    start_date: date | datetime,
    end_date: date | datetime,
    stock_ids: List[int] | None = None,
) -> pd.DataFrame:
    stmt = select(et.IndicatorsData).where(
        et.IndicatorsData.indicator_date.between(start_date, end_date)
    )

    if stock_ids:
        stmt = stmt.where(et.IndicatorsData.stock_id.in_(stock_ids))

    data = await app.execute(stmt)
    data = pd.DataFrame(data.mappings().all())
    data["sector"] = data["sector"].map(SECTOR_MAPPING)

    return data


async def fetch_training_data(
    start_date: date | datetime,
    end_date: date | datetime,
    stock_ids: List[int] | None = None,
) -> pd.DataFrame:
    """Load indicator rows for training, managing the database connection."""
    client = init_async_db()
    try:
        async with client.connect() as app:
            data = await get_indicators_data(
                app,
                start_date,
                end_date,
                stock_ids or None,
            )
    finally:
        await client.dispose()

    if data.empty:
        raise ValueError(
            f"No indicator data found between {start_date} and {end_date} "
            f"for stock_ids={stock_ids or 'all'}."
        )
    return data
