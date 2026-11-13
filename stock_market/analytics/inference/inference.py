"""Batch inference for the latest exported triple-barrier model."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sqlalchemy import func, select

from data.db import entities_transformed as et
from data.db.client import init_async_db
from data.db.statements import upsert_table, utc_now_naive
from stock_market.analytics.utils.common import parse_date, resolve_project_path
from stock_market.config import SECTOR_MAPPING
from utils.helper import DEFAULT_INFERENCE_CONFIG, load_inference_config


def load_model_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the metadata written by the training flow."""
    path = resolve_project_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Inference config not found: {path}")
    config = load_inference_config(path)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError(f"Invalid or unsupported inference config: {path}")

    model = config.get("model")
    features = config.get("features")
    label = config.get("label")
    metrics = config.get("metrics")
    if (
        not isinstance(model, dict)
        or not isinstance(model.get("path"), str)
        or not model["path"]
    ):
        raise ValueError("Inference config must contain model.path.")
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(feature, str) or not feature for feature in features)
        or len(set(features)) != len(features)
    ):
        raise ValueError("Inference config must contain unique feature names.")
    if model.get("n_features") != len(features):
        raise ValueError("model.n_features does not match the feature list.")
    if not isinstance(label, dict) or not isinstance(label.get("classes"), dict):
        raise ValueError("Inference config must contain label.classes.")
    classes = label["classes"]
    if (
        set(classes) != {0, 1, 2}
        or any(not isinstance(name, str) or not name for name in classes.values())
        or len(set(classes.values())) != 3
    ):
        raise ValueError("label.classes must map 0, 1 and 2 to unique names.")
    if not isinstance(label.get("horizon"), int) or label["horizon"] < 1:
        raise ValueError("label.horizon must be a positive integer.")
    if not isinstance(metrics, dict) or any(
        not isinstance(metrics.get(split), dict) for split in ("validation", "test")
    ):
        raise ValueError("Inference config must contain validation and test metrics.")
    return config


def load_model(config: dict[str, Any]) -> xgb.XGBClassifier:
    """Load the configured artifact on CPU and verify its feature contract."""
    model_path = resolve_project_path(config["model"]["path"])
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    model.set_params(device="cpu")
    booster = model.get_booster()
    features = config["features"]
    if booster.num_features() != len(features) or booster.feature_names != features:
        raise ValueError(
            "Model artifact feature names/order do not match inference config."
        )
    configured_iteration = config["model"].get("best_iteration")
    try:
        artifact_iteration = model.best_iteration
    except AttributeError:
        artifact_iteration = None
    if configured_iteration != artifact_iteration:
        raise ValueError(
            "Model artifact best_iteration does not match inference config."
        )
    return model


async def fetch_inference_data(
    features: list[str],
    as_of: date | None = None,
    stock_ids: list[int] | None = None,
) -> tuple[date, pd.DataFrame]:
    """Fetch one global indicator-date snapshot, optionally filtered by stock."""
    unknown = [
        feature for feature in features if not hasattr(et.IndicatorsData, feature)
    ]
    if unknown:
        raise ValueError(f"Features missing from indicator schema: {unknown}")
    client = init_async_db()
    try:
        async with client.connect() as connection:
            snapshot = as_of
            if snapshot is None:
                result = await connection.execute(
                    select(func.max(et.IndicatorsData.indicator_date)).where(
                        et.IndicatorsData.indicator_date <= date.today()
                    )
                )
                snapshot = result.scalar_one()
            if snapshot is None:
                raise ValueError(
                    "No indicator snapshot is available on or before today."
                )

            columns = [
                "stock_id",
                "ticker",
                "indicator_date",
                "open",
                "high",
                "low",
                "close",
                "returns",
                "momentum_5d",
                "momentum_20d",
                "volatility_20d_ann",
                *features,
            ]
            columns = list(dict.fromkeys(columns))
            statement = (
                select(
                    *(getattr(et.IndicatorsData, column) for column in columns),
                    et.IndicatorsData.sector.label("market_sector"),
                    et.PriceData.volume,
                )
                .outerjoin(
                    et.PriceData,
                    (et.PriceData.stock_id == et.IndicatorsData.stock_id)
                    & (et.PriceData.trade_date == et.IndicatorsData.indicator_date),
                )
                .where(et.IndicatorsData.indicator_date == snapshot)
            )
            if stock_ids:
                statement = statement.where(et.IndicatorsData.stock_id.in_(stock_ids))
            result = await connection.execute(
                statement.order_by(et.IndicatorsData.stock_id)
            )
            frame = pd.DataFrame(
                result.mappings().all(),
                columns=[*columns, "market_sector", "volume"],
                dtype=object,
            )
    finally:
        await client.dispose()

    if frame.empty:
        raise ValueError(
            f"No indicator rows found for {snapshot} and stock_ids={stock_ids or 'all'}."
        )
    if stock_ids:
        missing = sorted(set(stock_ids) - set(frame["stock_id"]))
        if missing:
            raise ValueError(
                f"Stocks missing from indicator snapshot {snapshot}: {missing}"
            )
    if "sector" in features:
        frame["sector"] = frame["sector"].map(SECTOR_MAPPING)
    return snapshot, frame


async def fetch_inference_dates(start_date: date, end_date: date) -> list[date]:
    """Find available global snapshots in an inclusive date range."""
    client = init_async_db()
    try:
        async with client.connect() as connection:
            result = await connection.execute(
                select(et.IndicatorsData.indicator_date)
                .where(et.IndicatorsData.indicator_date.between(start_date, end_date))
                .distinct()
                .order_by(et.IndicatorsData.indicator_date)
            )
            dates = list(result.scalars().all())
    finally:
        await client.dispose()
    if not dates:
        raise ValueError(
            f"No indicator snapshots found between {start_date} and {end_date}."
        )
    return dates


def predict_snapshot(
    model: xgb.XGBClassifier,
    config: dict[str, Any],
    frame: pd.DataFrame,
    *,
    allow_empty: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Predict valid rows, explicitly reporting rows with unusable features."""
    features = config["features"]
    missing = [
        column
        for column in ["stock_id", "indicator_date", *features]
        if column not in frame
    ]
    if missing:
        raise ValueError(f"Inference data is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Inference data is empty.")
    if frame["stock_id"].isna().any() or frame["stock_id"].duplicated().any():
        raise ValueError("Inference data must have one non-null row per stock.")

    values = frame[features].copy()
    for feature in features:
        values[feature] = pd.to_numeric(values[feature], errors="coerce")
    invalid = values.isna() | ~np.isfinite(values)
    rejected = [
        {
            "stock_id": int(frame.iloc[index]["stock_id"]),
            "invalid_features": invalid.columns[invalid.iloc[index]].tolist(),
        }
        for index in range(len(frame))
        if invalid.iloc[index].any()
    ]
    valid = ~invalid.any(axis=1)
    if not valid.any():
        if allow_empty:
            return [], rejected
        raise ValueError(
            f"No rows have valid model features; rejected rows: {rejected}"
        )

    valid_frame = frame.loc[valid]
    probabilities = model.predict_proba(values.loc[valid, features])
    if (
        probabilities.shape != (len(valid_frame), 3)
        or not np.isfinite(probabilities).all()
    ):
        raise ValueError("Model returned invalid class probabilities.")
    classes = config["label"]["classes"]
    predictions = []
    for (_, row), probs in zip(valid_frame.iterrows(), probabilities):
        predicted_class = int(np.argmax(probs))
        ticker = row.get("ticker")
        predictions.append(
            {
                "stock_id": int(row["stock_id"]),
                "ticker": None if pd.isna(ticker) else str(ticker),
                "indicator_date": pd.Timestamp(row["indicator_date"])
                .date()
                .isoformat(),
                "predicted_class": predicted_class,
                "predicted_label": classes[predicted_class],
                "probabilities": {
                    classes[index]: float(probs[index]) for index in range(3)
                },
            }
        )
    return predictions, rejected


async def persist_monitoring(
    frame: pd.DataFrame,
    config: dict[str, Any],
    predictions: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> None:
    """Atomically upsert the processed stocks without changing other rows."""
    outcomes = {row["stock_id"]: row for row in predictions}
    failures = {row["stock_id"]: row for row in rejected}
    scored_at = utc_now_naive()
    classes = config["label"]["classes"]
    records = []
    for _, row in frame.iterrows():
        stock_id = int(row["stock_id"])
        prediction = outcomes.get(stock_id)
        failure = failures.get(stock_id)
        if (prediction is None) == (failure is None):
            raise ValueError(
                f"Expected exactly one inference outcome for stock {stock_id}."
            )
        probabilities = prediction["probabilities"] if prediction else {}
        records.append(
            {
                "stock_id": stock_id,
                "indicator_date": pd.Timestamp(row["indicator_date"]).date(),
                "ticker": row.get("ticker"),
                "sector": row.get("market_sector"),
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume"),
                "daily_return": row.get("returns"),
                "momentum_5d": row.get("momentum_5d"),
                "momentum_20d": row.get("momentum_20d"),
                "volatility_20d_ann": row.get("volatility_20d_ann"),
                "status": "predicted" if prediction else "rejected",
                "predicted_class": (
                    prediction["predicted_class"] if prediction else None
                ),
                "predicted_label": (
                    prediction["predicted_label"] if prediction else None
                ),
                "probability_lower": probabilities.get(classes[0]),
                "probability_neither": probabilities.get(classes[1]),
                "probability_upper": probabilities.get(classes[2]),
                "invalid_features": failure["invalid_features"] if failure else [],
                "horizon": config["label"]["horizon"],
                "model_path": config["model"]["path"],
                "model_trained_at": config["model"].get("trained_at"),
                "scored_at": scored_at,
            }
        )
    engine = init_async_db()
    try:
        async with engine.begin() as connection:
            await upsert_table(
                connection, et.InferenceMonitoring, pd.DataFrame(records, dtype=object)
            )
    finally:
        await engine.dispose()


async def run_inference(
    config_path: str | Path = DEFAULT_INFERENCE_CONFIG,
    as_of: str | date | datetime | None = None,
    stock_ids: list[int] | None = None,
    *,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
) -> dict[str, Any]:
    """Score one snapshot or an inclusive range of snapshots, one day at a time."""
    is_range = start_date is not None or end_date is not None
    if is_range:
        if as_of is not None:
            raise ValueError("as_of cannot be combined with start_date or end_date.")
        if start_date is None or end_date is None or start_date == "" or end_date == "":
            raise ValueError(
                "Both start_date and end_date are required for a date range."
            )
        range_start = parse_date(start_date, date.today())
        range_end = parse_date(end_date, date.today())
        if range_start > range_end:
            raise ValueError("start_date must be on or before end_date.")

    config = load_model_config(config_path)
    model = load_model(config)
    if is_range:
        dates = await fetch_inference_dates(range_start, range_end)
        summaries = []
        total_predictions = 0
        for day in dates:
            _, frame = await fetch_inference_data(config["features"], day, stock_ids)
            predictions, rejected = predict_snapshot(
                model, config, frame, allow_empty=True
            )
            await persist_monitoring(frame, config, predictions, rejected)
            summaries.append(
                {
                    "indicator_date": day.isoformat(),
                    "predicted_rows": len(predictions),
                    "rejected_rows": len(rejected),
                }
            )
            total_predictions += len(predictions)
            logger.info(
                "Inference date {} committed | predicted={} rejected={}",
                day,
                len(predictions),
                len(rejected),
            )
        if not total_predictions:
            raise ValueError(
                "No rows have valid model features in the date range; "
                "rejected rows were persisted."
            )
        return {
            "start_date": range_start.isoformat(),
            "end_date": range_end.isoformat(),
            "model": config["model"],
            "label": config["label"],
            "training_metrics": config["metrics"],
            "batches": summaries,
            "predicted_rows": total_predictions,
            "rejected_rows": sum(batch["rejected_rows"] for batch in summaries),
        }

    snapshot_date = parse_date(as_of, date.today()) if as_of is not None else None
    snapshot, frame = await fetch_inference_data(
        config["features"], snapshot_date, stock_ids
    )
    predictions, rejected = predict_snapshot(model, config, frame, allow_empty=True)
    await persist_monitoring(frame, config, predictions, rejected)
    if not predictions:
        raise ValueError(
            f"No rows have valid model features; rejected rows were persisted: {rejected}"
        )
    return {
        "indicator_date": snapshot.isoformat(),
        "model": config["model"],
        "label": config["label"],
        "training_metrics": config["metrics"],
        "predictions": predictions,
        "rejected_rows": rejected,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batch stock-model inference.")
    parser.add_argument("--config", type=Path, default=DEFAULT_INFERENCE_CONFIG)
    parser.add_argument(
        "--as-of",
        help="Exact indicator date (YYYY-MM-DD); defaults to latest snapshot.",
    )
    parser.add_argument(
        "--start-date", help="Inclusive start date for daily batches (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end-date", help="Inclusive end date for daily batches (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--stock-ids",
        type=int,
        nargs="+",
        help="Restrict the snapshot to these stock IDs.",
    )
    parser.add_argument(
        "--output", type=Path, help="Write JSON to this file instead of stdout."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = asyncio.run(
        run_inference(
            args.config,
            args.as_of,
            args.stock_ids,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    )
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
