import asyncio
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from data.db import entities_transformed as et
from data.db.client import init_async_db


def default_artifact_path(name: str) -> Path:
    path = Path("artifacts") / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


async def load_indicators() -> pd.DataFrame:
    engine = init_async_db()
    try:
        async with engine.connect() as conn:
            result = await conn.execute(select(et.IndicatorsData))
            return pd.DataFrame(result.mappings().all())
    finally:
        await engine.dispose()


def load_indicators_sync() -> pd.DataFrame:
    return asyncio.run(load_indicators())


def numeric_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    values = frame.reindex(columns=features).copy()
    return values.apply(pd.to_numeric, errors="coerce")