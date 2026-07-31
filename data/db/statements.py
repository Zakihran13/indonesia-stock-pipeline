from utils.helper import snake_case_columns
import data.db.entities as e
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
        df.reindex(columns=e.StockMetadata.__table__.columns.keys())
        .copy()
        .drop(columns=["stock_id"], errors="ignore")
    )
    analytics_df = df.reindex(columns=e.AnalyticData.__table__.columns.keys()).copy()
    fundamental_df = df.reindex(
        columns=e.FundamentalData.__table__.columns.keys()
    ).copy()
    dynamic_df = df.reindex(columns=e.DynamicData.__table__.columns.keys()).copy()

    return metadata_df, analytics_df, fundamental_df, dynamic_df


def dataframe_to_records(table, df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame into DB-safe records.

    Pandas uses NaN/NaT for missing values, but the database layer expects None
    so those values become SQL NULL.

    Some Yahoo payload fields come as numeric values for string DB columns
    (e.g. fax/phone can be 0). Coerce those to strings before binding.

    Some numeric fields can exceed DB precision (NUMERIC(p,s)); these are
    converted to None to avoid asyncpg numeric overflow errors.
    """
    sanitized = df.astype(object).where(pd.notna(df), None)
    records = sanitized.to_dict(orient="records")

    column_by_name = {column.name: column for column in table.__table__.columns}

    string_columns = {
        column.name
        for column in table.__table__.columns
        if isinstance(column.type, (String, Text))
    }

    numeric_columns = {
        column.name: column.type
        for column in table.__table__.columns
        if isinstance(column.type, Numeric)
    }

    integer_columns = {
        column.name: column.type
        for column in table.__table__.columns
        if isinstance(column.type, (Integer, BigInteger))
    }

    datetime_columns = {
        column.name: column.type
        for column in table.__table__.columns
        if isinstance(column.type, DateTime)
    }

    def sanitize_numeric(value, numeric_type: Numeric):
        if value is None:
            return None

        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return None

        if not decimal_value.is_finite():
            return None

        precision = numeric_type.precision
        scale = numeric_type.scale

        if precision is not None and scale is not None:
            limit = Decimal(10) ** (precision - scale)
            if abs(decimal_value) >= limit:
                return None

        return float(decimal_value)

    def sanitize_integer(value, integer_type):
        if value is None:
            return None

        if isinstance(value, bool):
            return int(value)

        if isinstance(value, float) and not math.isfinite(value):
            return None

        try:
            int_value = int(value)
        except (TypeError, ValueError):
            return None

        if isinstance(integer_type, BigInteger):
            if int_value < -9223372036854775808 or int_value > 9223372036854775807:
                return None

        return int_value

    def sanitize_datetime(value, datetime_type: DateTime):
        if value is None:
            return None

        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                return None

            ts = value
            if ts.tzinfo is not None and not datetime_type.timezone:
                ts = ts.tz_convert("UTC").tz_localize(None)

            return ts.to_pydatetime()

        if pd.isna(value):
            return None

        return value

    for record in records:
        for column_name in string_columns:
            value = record.get(column_name)
            if value is not None and not isinstance(value, str):
                record[column_name] = str(value)

        for column_name, numeric_type in numeric_columns.items():
            if column_name in record:
                record[column_name] = sanitize_numeric(
                    record.get(column_name), numeric_type
                )

        for column_name, integer_type in integer_columns.items():
            if column_name in record:
                record[column_name] = sanitize_integer(
                    record.get(column_name), integer_type
                )

        for column_name, datetime_type in datetime_columns.items():
            if column_name in record:
                record[column_name] = sanitize_datetime(
                    record.get(column_name), datetime_type
                )

    return records


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
        stmt = select(e.StockMetadata.ticker, e.StockMetadata.stock_id)
    else:
        stmt = select(e.StockMetadata.ticker, e.StockMetadata.stock_id).where(
            e.StockMetadata.ticker.in_(tickers)
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
    now_utc = utc_now_naive()
    metadata_df[["created_at", "updated_at"]] = now_utc
    analytics_df[["retrieve_at"]] = now_utc
    fundamental_df[["retrieve_at"]] = now_utc
    dynamic_df[["retrieve_at"]] = now_utc

    # insertion
    await upsert_table(
        conn, e.StockMetadata, metadata_df, on_conflict_columns=["ticker"]
    )

    stock_ids = await fetch_stock_ids(
        conn, metadata_df["ticker"].dropna().astype(str).tolist()
    )

    analytics_df = attach_stock_ids(analytics_df, metadata_df, stock_ids)
    fundamental_df = attach_stock_ids(fundamental_df, metadata_df, stock_ids)
    dynamic_df = attach_stock_ids(dynamic_df, metadata_df, stock_ids)

    await upsert_table(conn, e.AnalyticData, analytics_df)
    await upsert_table(conn, e.FundamentalData, fundamental_df)
    await upsert_table(conn, e.DynamicData, dynamic_df)


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

    await upsert_table(conn, e.AnalyticData, analytics_df)
    await upsert_table(conn, e.FundamentalData, fundamental_df)
    await upsert_table(conn, e.DynamicData, dynamic_df)


async def insert_price_data(conn, raw_df: pd.DataFrame):
    """Inserts price data into the database."""

    # attach stock_ids
    price_df = (
        raw_df.drop_duplicates(subset=["stock_id", "date"], keep="last")
        .reindex(columns=e.PriceData.__table__.columns.keys())
        .copy()
    )

    await upsert_table(
        conn, e.PriceData, price_df, on_conflict_columns=["stock_id", "date"]
    )


async def insert_shares_data(conn, raw_df: pd.DataFrame):
    """Inserts shares history into the fundamental_data table."""

    shares_df = (
        raw_df.drop_duplicates(subset=["stock_id", "retrieve_at"], keep="last")
        .dropna(subset=["stock_id", "retrieve_at"])
        .reindex(columns=e.FundamentalData.__table__.columns.keys())
        .copy()
    )

    await upsert_table(
        conn,
        e.FundamentalData,
        shares_df,
        on_conflict_columns=["stock_id", "retrieve_at"],
    )
