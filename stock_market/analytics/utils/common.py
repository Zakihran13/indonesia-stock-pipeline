import pandas as pd
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine
from decimal import Decimal
from loguru import logger
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import xgboost as xgb


from data.db import entities_transformed as et
from data.db.statements import fetch_frame
from stock_market.analytics.data.preprocess import LABEL_NAMES
from stock_market.config import SECTOR_MAPPING, HORIZON, ROOT_DIR
from utils.helper import DEFAULT_TRAIN_CONFIG, load_train_config

DEFAULT_START_DATE = date(2020, 1, 1)


def resolve_project_path(value: str | Path) -> Path:
    """Resolve ``value`` against the project root when it is relative."""
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def parse_date(value: Any, default: date) -> date:
    """Accept YAML dates, ISO strings, or datetime values."""
    if value is None or value == "":
        return default
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError as exc:
        raise ValueError(f"Invalid date value: {value!r}") from exc


def load_training_config(
    path: str | Path = DEFAULT_TRAIN_CONFIG,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load ``train.yaml`` and validate its ``training`` section."""
    path = resolve_project_path(path)
    config = load_train_config(path)
    if not config:
        raise FileNotFoundError(f"Training config is missing or empty: {path}")

    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError(
            f"Expected a `training` mapping in {path}; run the setup again."
        )
    return config, training


def label_distribution(
    frame: pd.DataFrame, label_column: str = "label"
) -> dict[int, float]:
    """Return the class distribution of a labelled frame."""
    return {
        int(label): float(value)
        for label, value in frame[label_column]
        .value_counts(normalize=True)
        .sort_index()
        .items()
    }


def build_split_metadata(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    split_config: dict[str, Any],
    date_column: str = "indicator_date",
) -> dict[str, Any]:
    """Describe the chronological split written to ``inference.yaml``."""
    return {
        "train_fraction": float(split_config.get("train_fraction", 0.7)),
        "validation_fraction": float(split_config.get("validation_fraction", 0.15)),
        "embargo_days": int(split_config.get("embargo_days", HORIZON * 2)),
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        "train_start": train_frame[date_column].min().isoformat(),
        "train_end": train_frame[date_column].max().isoformat(),
        "validation_start": validation_frame[date_column].min().isoformat(),
        "validation_end": validation_frame[date_column].max().isoformat(),
        "test_start": test_frame[date_column].min().isoformat(),
        "test_end": test_frame[date_column].max().isoformat(),
    }


def _best_iteration(model: xgb.XGBClassifier) -> int | None:
    try:
        iteration = model.best_iteration
    except (AttributeError, TypeError):
        return None
    return int(iteration) if iteration is not None else None


def _relative_model_path(model_path: Path) -> str:
    try:
        return model_path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return model_path.resolve().as_posix()


def build_inference_config(
    *,
    model: xgb.XGBClassifier,
    model_path: Path,
    features: list[str],
    feature_metadata: dict[str, Any],
    start_date: date,
    end_date: date,
    stock_ids: list[int],
    row_count: int,
    label_distribution: dict[int, float],
    split_metadata: dict[str, Any],
    search_metadata: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Build the metadata consumed by inference/serving code."""
    trained_at = datetime.now().isoformat(timespec="seconds")
    return {
        "schema_version": 1,
        "generated_at": trained_at,
        "label": {
            "horizon": HORIZON,
            "classes": LABEL_NAMES,
        },
        "model": {
            "path": _relative_model_path(model_path),
            "format": model_path.suffix.lstrip(".") or "ubj",
            "trained_at": trained_at,
            "xgboost_version": xgb.__version__,
            "device": device,
            "n_estimators": int(model.n_estimators),
            "best_iteration": _best_iteration(model),
            "n_features": len(features),
        },
        "features": features,
        "feature_engineering": feature_metadata,
        "data": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "stock_ids": stock_ids,
            "rows": int(row_count),
            "label_distribution": label_distribution,
        },
        "split": split_metadata,
        "search": search_metadata,
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
    }


def purged_time_split(df, train_frac=0.6, val_frac=0.2, date_col: str = "trade_date"):
    """Chronological split by date. Rows whose label window reaches into the next
    split are dropped, so no train/val label overlaps val/test dates."""
    d = df.dropna(subset=["label"]).copy()
    udates = np.sort(d[date_col].unique())
    val_start = udates[int(len(udates) * train_frac)]
    test_start = udates[int(len(udates) * (train_frac + val_frac))]

    train = d[(d[date_col] < val_start) & (d["label_end"] < val_start)]
    val = d[
        (d[date_col] >= val_start)
        & (d[date_col] < test_start)
        & (d["label_end"] < test_start)
    ]
    test = d[d[date_col] >= test_start]
    return train, val, test
