from utils.helper import snake_case_columns, dataframe_to_records
import data.db.entities_transformed as et
from decimal import Decimal, InvalidOperation
import math

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import DateTime
from sqlalchemy.sql.sqltypes import BigInteger, Integer, Numeric, String, Text
import pandas as pd

MAX_BIND_PARAMS_PER_STATEMENT = 30000


def utc_now_naive() -> pd.Timestamp:
    """Returns current UTC time as a naive timestamp for TIMESTAMP columns."""

    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def metadata_separation(df: pd.DataFrame):
    metadata_df = (
        df.reindex(columns=et.StockMetadata.__table__.columns.keys())
        .copy()
        .drop(columns=["stock_id"], errors="ignore")
    )
    analytics_df = df.reindex(columns=et.AnalyticData.__table__.columns.keys()).copy()
    fundamental_df = df.reindex(
        columns=et.FundamentalData.__table__.columns.keys()
    ).copy()
    dynamic_df = df.reindex(columns=et.DynamicData.__table__.columns.keys()).copy()

    return metadata_df, analytics_df, fundamental_df, dynamic_df


def _iter_record_batches(records: list[dict], columns_per_row: int):
    """Yields record chunks that stay below the DB bind-parameter threshold."""
    if not records:
        return

    batch_size = max(1, MAX_BIND_PARAMS_PER_STATEMENT // max(1, columns_per_row))

    for i in range(0, len(records), batch_size):
        yield records[i : i + batch_size]


async def upsert_table(
    conn: AsyncConnection,
    table,
    df: pd.DataFrame,
    on_conflict_columns: list[str] | None = None,
):
    """Upserts data into the specified table."""
    if df.empty:
        return

    conflict_columns = on_conflict_columns or list(
        table.__table__.primary_key.columns.keys()
    )
    primary_key_columns = set(table.__table__.primary_key.columns.keys())
    records = dataframe_to_records(table, df)

    if not records:
        return

    # Chunk inserts so we do not exceed asyncpg/Postgres bind parameter limits.
    for record_batch in _iter_record_batches(records, len(table.__table__.columns)):
        stmt = insert(table).values(record_batch)

        stmt = stmt.on_conflict_do_update(
            index_elements=conflict_columns,
            set_={
                c.name: getattr(stmt.excluded, c.name)
                for c in stmt.excluded
                if c.name not in conflict_columns and c.name not in primary_key_columns
            },
        )

        await conn.execute(stmt)


async def fetch_stock_ids(
    conn: AsyncConnection, tickers: list[str] | None = None
) -> dict[str, int]:
    if not tickers:
        stmt = select(et.StockMetadata.ticker, et.StockMetadata.stock_id)
    else:
        stmt = select(et.StockMetadata.ticker, et.StockMetadata.stock_id).where(
            et.StockMetadata.ticker.in_(tickers)
        )

    result = await conn.execute(stmt)
    return {ticker: stock_id for ticker, stock_id in result.all()}


def attach_stock_ids(
    child_df: pd.DataFrame, metadata_df: pd.DataFrame, stock_ids: dict[str, int]
) -> pd.DataFrame:
    child_df = child_df.copy()
    child_df["stock_id"] = metadata_df["ticker"].map(stock_ids)
    child_df = child_df.dropna(subset=["stock_id"])

    if not child_df.empty:
        child_df["stock_id"] = child_df["stock_id"].astype(int)

    return child_df


async def insert_metadata(conn, raw_df: pd.DataFrame):
    """Inserts metadata into the database."""

    # rename mapping
    renamed_df = snake_case_columns(raw_df)

    # separate data: metadata, analyctics, fundamental, dynamic
    metadata_df, analytics_df, fundamental_df, dynamic_df = metadata_separation(
        renamed_df
    )
    now_utc = utc_now_naive().date()
    metadata_df[["created_at", "updated_at"]] = now_utc
    analytics_df[["retrieve_at"]] = now_utc
    fundamental_df[["retrieve_at"]] = now_utc
    dynamic_df[["retrieve_at"]] = now_utc

    # insertion
    await upsert_table(
        conn, et.StockMetadata, metadata_df, on_conflict_columns=["ticker"]
    )

    stock_ids = await fetch_stock_ids(
        conn, metadata_df["ticker"].dropna().astype(str).tolist()
    )

    analytics_df = attach_stock_ids(analytics_df, metadata_df, stock_ids)
    fundamental_df = attach_stock_ids(fundamental_df, metadata_df, stock_ids)
    dynamic_df = attach_stock_ids(dynamic_df, metadata_df, stock_ids)

    await upsert_table(
        conn,
        et.AnalyticData,
        analytics_df,
        on_conflict_columns=["stock_id", "retrieve_at"],
    )
    await upsert_table(
        conn,
        et.FundamentalData,
        fundamental_df,
        on_conflict_columns=["stock_id", "retrieve_at"],
    )
    await upsert_table(
        conn,
        et.DynamicData,
        dynamic_df,
        on_conflict_columns=["stock_id", "retrieve_at"],
    )


async def insert_dynamic_data(conn, raw_df: pd.DataFrame):
    """Inserts dynamic data into the database."""

    # rename mapping
    renamed_df = snake_case_columns(raw_df)

    # separate data: metadata, analyctics, fundamental, dynamic
    _, analytics_df, fundamental_df, dynamic_df = metadata_separation(renamed_df)
    now_utc = utc_now_naive()
    analytics_df[["retrieve_at"]] = now_utc
    fundamental_df[["retrieve_at"]] = now_utc
    dynamic_df[["retrieve_at"]] = now_utc

    await upsert_table(
        conn,
        et.AnalyticData,
        analytics_df,
        on_conflict_columns=["stock_id", "retrieve_at"],
    )
    await upsert_table(
        conn,
        et.FundamentalData,
        fundamental_df,
        on_conflict_columns=["stock_id", "retrieve_at"],
    )
    await upsert_table(
        conn,
        et.DynamicData,
        dynamic_df,
        on_conflict_columns=["stock_id", "retrieve_at"],
    )


async def insert_price_data(conn, raw_df: pd.DataFrame):
    """Inserts price data into the database."""

    # attach stock_ids
    raw_df = raw_df.rename(columns={"date": "trade_date"})
    price_df = raw_df.reindex(columns=et.PriceData.__table__.columns.keys()).copy()

    await upsert_table(
        conn, et.PriceData, price_df, on_conflict_columns=["stock_id", "trade_date"]
    )


async def insert_shares_data(conn, raw_df: pd.DataFrame):
    """Inserts shares history into the fundamental_data table."""

    shares_df = (
        raw_df.drop_duplicates(subset=["stock_id", "retrieve_at"], keep="last")
        .dropna(subset=["stock_id", "retrieve_at"])
        .reindex(columns=et.FundamentalData.__table__.columns.keys())
        .copy()
    )

    await upsert_table(
        conn,
        et.FundamentalData,
        shares_df,
        on_conflict_columns=["stock_id", "retrieve_at"],
    )


async def get_metadata(conn, tickers: list[str] | None = None) -> pd.DataFrame:
    """Fetches metadata from the database."""
    if not tickers:
        stmt = select(et.StockMetadata)
    else:
        stmt = select(et.StockMetadata).where(et.StockMetadata.ticker.in_(tickers))

    result = await conn.execute(stmt)
    rows = result.fetchall()
    df = pd.DataFrame(rows, columns=result.keys())
    return df


async def get_fundamental_data(conn, tickers: list[str] | None = None) -> pd.DataFrame:
    """Fetches fundamental data from the database."""
    if not tickers:
        stmt = select(et.FundamentalData)
    else:
        stmt = (
            select(et.FundamentalData)
            .join(
                et.FundamentalData,
                et.FundamentalData.stock_id == et.StockMetadata.stock_id,
            )
            .where(et.FundamentalData.ticker.in_(tickers))
        )

    result = await conn.execute(stmt)
    rows = result.fetchall()
    df = pd.DataFrame(rows, columns=result.keys())
    return df
