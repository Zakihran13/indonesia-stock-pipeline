import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    insert,
    literal,
    select,
    text,
)
from sqlalchemy.ext.asyncio import AsyncConnection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from data.db import entities_transformed as et
from data.db.client import init_async_db
from data.db.statements import fetch_frame, insert_indicators_data
from stock_market.config import (
    DEFAULT_STOCK_BATCH_SIZE,
    FEATURE_WARMUP_DAYS,
    RANK_FEATURES,
)
from utils import indicators_calc

_CROSS_SECTION_STAGE = Table(
    "indicator_cross_section_stage",
    MetaData(),
    Column("stock_id", Integer, primary_key=True),
    Column("indicator_date", Date, primary_key=True),
    Column("sector", String),
    *(Column(feature, Float) for feature in RANK_FEATURES),
    Column("is_target", Boolean, nullable=False),
)


async def _stage_cross_section(conn: AsyncConnection, indicators: pd.DataFrame) -> None:
    columns = ["stock_id", "indicator_date", "sector", *RANK_FEATURES]
    staged = indicators[columns].astype(object)
    staged = staged.where(pd.notna(staged), None)
    staged["is_target"] = True
    if not staged.empty:
        await conn.execute(insert(_CROSS_SECTION_STAGE), staged.to_dict("records"))


async def _stage_existing_peers(conn: AsyncConnection, target_ids: set[int]) -> None:
    columns = ["stock_id", "indicator_date", "sector", *RANK_FEATURES]
    target_dates = select(_CROSS_SECTION_STAGE.c.indicator_date).distinct()
    peers = select(
        *(getattr(et.IndicatorsData, column) for column in columns),
        literal(False),
    ).where(
        et.IndicatorsData.stock_id.not_in(sorted(target_ids)),
        et.IndicatorsData.indicator_date.in_(target_dates),
    )
    await conn.execute(
        insert(_CROSS_SECTION_STAGE).from_select([*columns, "is_target"], peers)
    )


async def _update_cross_section(conn: AsyncConnection) -> None:
    # Average percentile ranks match pandas rank(method="average", pct=True).
    rank_expressions = [f"""CASE WHEN {feature} IS NOT NULL THEN
            (RANK() OVER (PARTITION BY indicator_date ORDER BY {feature} NULLS LAST)
             + (COUNT(*) OVER (PARTITION BY indicator_date, {feature}) - 1) / 2.0)
            / NULLIF(COUNT({feature}) OVER (PARTITION BY indicator_date), 0)
        END AS {rank_column}""" for feature, rank_column in RANK_FEATURES.items()]
    sector_expressions = [f"""CASE WHEN sector IS NOT NULL THEN {feature} -
            AVG({feature}) OVER (PARTITION BY indicator_date, sector)
        END AS sector_rel_{feature}""" for feature in ("momentum_20d", "momentum_120d")]
    output_columns = [
        *RANK_FEATURES.values(),
        "sector_rel_momentum_20d",
        "sector_rel_momentum_120d",
    ]
    assignments = ", ".join(f"{column} = ranked.{column}" for column in output_columns)
    expressions = ", ".join([*rank_expressions, *sector_expressions])
    await conn.execute(text(f"""
            UPDATE stock_market.indicators_data AS target
            SET {assignments}
            FROM (
                SELECT stock_id, indicator_date, is_target, {expressions}
                FROM indicator_cross_section_stage
            ) AS ranked
            WHERE ranked.is_target
              AND target.stock_id = ranked.stock_id
              AND target.indicator_date = ranked.indicator_date
            """))


async def materialize_indicators(
    stock_ids: List[int] | None = None,
    start_date: datetime | None = None,
    batch_size: int = DEFAULT_STOCK_BATCH_SIZE,
) -> int | None:
    """Build indicators in stock batches and calculate cross-sectional features.

    With ``stock_ids``, only those stocks are calculated. Their ranks and sector
    means use existing indicator rows for other stocks on the same dates; missing
    or stale peer rows can make the resulting cross-section incomplete.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    if not start_date:
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    target_ids = set(stock_ids) if stock_ids is not None else None
    if target_ids is not None and not target_ids:
        logger.warning("stock_ids is empty; nothing to materialize")
        return 0

    engine = init_async_db()
    try:
        async with engine.begin() as conn:
            fetch_start_date = start_date - timedelta(days=FEATURE_WARMUP_DAYS)
            await conn.execute(text("""
                    CREATE TEMPORARY TABLE indicator_cross_section_stage (
                        stock_id INTEGER NOT NULL,
                        indicator_date DATE NOT NULL,
                        sector VARCHAR,
                        momentum_20d DOUBLE PRECISION,
                        momentum_60d DOUBLE PRECISION,
                        momentum_120d DOUBLE PRECISION,
                        momentum_12_1 DOUBLE PRECISION,
                        volatility_20d_ann DOUBLE PRECISION,
                        risk_adjusted_momentum DOUBLE PRECISION,
                        distance_high_52w DOUBLE PRECISION,
                        is_target BOOLEAN NOT NULL,
                        PRIMARY KEY (stock_id, indicator_date)
                    ) ON COMMIT DROP
                    """))

            last_stock_id = None
            row_count = 0
            found_prices = False
            while True:
                stmt = (
                    select(et.PriceData.stock_id)
                    .where(et.PriceData.trade_date >= fetch_start_date)
                    .distinct()
                    .order_by(et.PriceData.stock_id)
                    .limit(batch_size)
                )
                if target_ids is not None:
                    stmt = stmt.where(et.PriceData.stock_id.in_(sorted(target_ids)))

                if last_stock_id is not None:
                    stmt = stmt.where(et.PriceData.stock_id > last_stock_id)

                result = await conn.execute(stmt)
                batch_ids = list(result.scalars().all())

                if not batch_ids:
                    break

                last_stock_id = batch_ids[-1]
                found_prices = True

                price_frame = await fetch_frame(
                    conn,
                    et.PriceData,
                    et.PriceData.trade_date,
                    batch_ids,
                    fetch_start_date,
                )

                if price_frame.empty:
                    continue

                metadata_result = await conn.execute(
                    select(et.StockMetadata).where(
                        et.StockMetadata.stock_id.in_(batch_ids),
                        et.StockMetadata.sector.is_not(None),
                    )
                )
                metadata_frame = pd.DataFrame(metadata_result.mappings().all())

                indicator_frame = indicators_calc.build_indicator_frame(
                    price_frame,
                    metadata_frame,
                    start_date=start_date,
                    calculate_cross_section=False,
                )
                await _stage_cross_section(conn, indicator_frame)

                await insert_indicators_data(conn, indicator_frame)

                row_count += len(indicator_frame)
                logger.info(
                    "Processed {} stocks through ID {}; materialized {} indicator rows",
                    len(batch_ids),
                    last_stock_id,
                    row_count,
                )
                del price_frame, metadata_frame, indicator_frame

            if not found_prices:
                logger.error(f"NO DATA IS FOUND FOR: **{start_date}**")
                return

            if target_ids is not None:
                logger.warning(
                    "Targeted cross-sectional calculations use existing peer "
                    "indicators; missing or stale rows may produce incomplete results"
                )
                await _stage_existing_peers(conn, target_ids)
            await _update_cross_section(conn)
            logger.info("Materialized {} indicator rows", row_count)
            return row_count
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(
        materialize_indicators(
            start_date=datetime(2016, 1, 1),
        )
    )
