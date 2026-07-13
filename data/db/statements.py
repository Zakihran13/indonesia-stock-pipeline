from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Hashable, Mapping

import pandas as pd
from data.db import entities as e

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection


_SPECIAL_KEY_MAP = {
    "Symbol": "symbol",
    "symbol": "symbol",
    "52WeekChange": "fifty_two_week_change",
    "SandP52WeekChange": "s_and_p_52_week_change",
}


def _camel_to_snake(name: str) -> str:
    step1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step1).lower()


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    if pd.isna(value):
        return None
    return value


def _normalize_metadata_input(data: Mapping[Hashable, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    valid_columns = set(e.TickerMetadata.__mapper__.column_attrs.keys())

    for key, value in data.items():
        key_name = str(key)
        mapped_key = _SPECIAL_KEY_MAP.get(key_name)
        if mapped_key is None:
            mapped_key = _camel_to_snake(key_name)

        if mapped_key in valid_columns:
            normalized[mapped_key] = _normalize_value(value)

    now = datetime.utcnow()
    normalized.setdefault("created_at", now)
    normalized["updated_at"] = now

    return normalized


async def insert_metadata(conn: AsyncConnection, data: Mapping[Hashable, Any]) -> None:
    payload = _normalize_metadata_input(data)
    if not payload.get("symbol"):
        return

    stmt = insert(e.TickerMetadata).values(**payload)

    update_map = {
        column.name: getattr(stmt.excluded, column.name)
        for column in e.TickerMetadata.__table__.columns
        if column.name not in {"symbol", "created_at"}
    }
    update_map["updated_at"] = datetime.utcnow()

    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=[e.TickerMetadata.__table__.c.symbol],
        set_=update_map,
    )
    await conn.execute(upsert_stmt)

