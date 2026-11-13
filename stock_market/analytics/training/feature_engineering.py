import numpy as np
import pandas as pd
from loguru import logger
import time
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone
import shap
from datetime import date, datetime, timedelta
from typing import Any, List
from tqdm.auto import tqdm

from stock_market.config import NUMERIC_FEATURE_COLUMNS, HORIZON, CURRENT_DATE
from stock_market.analytics.models.model import xgb_model
from stock_market.analytics.data.statement import get_indicators_data
from stock_market.analytics.data.preprocess import process_data_training
from data.db.client import init_async_db
from utils.helper import (
    DEFAULT_TRAIN_CONFIG,
    load_train_config,
    save_selected_features,
)


def _estimated_rfe_iterations(n_features: int, min_features: int, step: float) -> int:
    """Estimate SHAP-RFE iterations so the progress bar has a total."""
    iterations = 1
    current = n_features
    while current > min_features:
        current = max(min_features, current - max(1, int(current * step)))
        iterations += 1
    return iterations


def shap_rfe(
    model,
    X,
    y,
    step=0.1,
    min_features=10,
    cv=3,
    progress_position: int = 0,
):
    feats = list(X.columns)
    rows = []

    skf = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42,
    )

    iteration = 1
    total_start = time.perf_counter()

    logger.info(
        "Starting SHAP-RFE | Initial Features={} | Min Features={} | Step={} | CV={}",
        len(feats),
        min_features,
        step,
        cv,
    )

    progress = tqdm(
        total=_estimated_rfe_iterations(len(feats), min_features, step),
        desc="SHAP-RFE",
        unit="iter",
        position=progress_position,
        leave=False,
        nrows=20,
        disable=None,
    )

    try:
        while True:
            iter_start = time.perf_counter()
            progress.set_description(f"SHAP-RFE | {len(feats)} features")

            logger.info(
                "[Iteration {}] Evaluating {} features...",
                iteration,
                len(feats),
            )

            score = cross_val_score(
                clone(model),
                X[feats],
                y,
                cv=skf,
                scoring="f1_macro",
                n_jobs=-1,
            ).mean()

            rows.append((len(feats), score, list(feats)))
            progress.set_postfix(features=len(feats), score=f"{score:.4f}")

            logger.info(
                "[Iteration {}] CV F1 Macro = {:.6f}",
                iteration,
                score,
            )

            if len(feats) <= min_features:
                progress.update(1)
                logger.success(
                    "Reached minimum feature count ({}). Stopping.",
                    min_features,
                )
                break

            logger.info(
                "[Iteration {}] Training model for SHAP importance...",
                iteration,
            )

            m = clone(model).fit(X[feats], y)

            logger.info(
                "[Iteration {}] Computing SHAP values...",
                iteration,
            )

            sv = shap.TreeExplainer(m).shap_values(X[feats])

            if isinstance(sv, list):
                sv = np.stack(sv, axis=-1)

            imp = np.abs(sv).mean(axis=(0, 2))

            drop = max(1, int(len(feats) * step))
            keep_n = max(min_features, len(feats) - drop)

            ranked = np.argsort(imp)[::-1]
            removed = [feats[i] for i in ranked[keep_n:]]

            logger.info(
                "[Iteration {}] Removing {} feature(s): {}",
                iteration,
                len(removed),
                ", ".join(removed[:10]) + ("..." if len(removed) > 10 else ""),
            )

            feats = [feats[i] for i in ranked[:keep_n]]

            logger.info(
                "[Iteration {}] Remaining features: {} | Elapsed: {:.2f}s",
                iteration,
                len(feats),
                time.perf_counter() - iter_start,
            )

            progress.update(1)
            iteration += 1
    finally:
        progress.close()

    logger.success(
        "SHAP-RFE completed in {:.2f}s after {} iteration(s). Final feature count: {}",
        time.perf_counter() - total_start,
        iteration,
        len(feats),
    )

    return rows


def sample_feature_selection_data(
    data: pd.DataFrame,
    sample_fraction: float,
    cv: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample labeled rows proportionally by year and label for SHAP-RFE."""
    if not np.isfinite(sample_fraction) or not 0 < sample_fraction <= 1:
        raise ValueError("Feature-selection sample_fraction must be in (0, 1].")
    if data.empty:
        raise ValueError("No rows remain after feature-selection data preparation.")
    if cv < 2:
        raise ValueError("Feature-selection cv must be at least 2.")

    sample_size = max(1, int(len(data) * sample_fraction))
    if sample_size == len(data):
        sample = data.copy()
    else:
        years = pd.to_datetime(data["indicator_date"]).dt.year
        groups = data.groupby([years, data["label"]], sort=True).indices
        sizes = np.array([len(indices) for indices in groups.values()])
        exact_sizes = sizes * (sample_size / len(data))
        counts = np.floor(exact_sizes).astype(int)
        remaining = sample_size - int(counts.sum())
        order = np.argsort(-(exact_sizes - counts), kind="stable")
        counts[order[:remaining]] += 1

        rng = np.random.default_rng(random_state)
        positions = [
            rng.choice(indices, size=count, replace=False)
            for indices, count in zip(groups.values(), counts)
            if count
        ]
        sample = data.iloc[np.sort(np.concatenate(positions))].copy()

    label_counts = sample["label"].value_counts()
    if (
        len(label_counts) < 2
        or len(label_counts) != data["label"].nunique()
        or (label_counts < cv).any()
    ):
        raise ValueError(
            "The feature-selection sample must include every label class with "
            f"at least {cv} rows each for cross-validation. Increase "
            "training.feature_selection.sample_fraction or use more data."
        )
    logger.info(
        "SHAP-RFE sample: {} of {} prepared rows ({:.1f}%), stratified by "
        "year and label | class counts={}",
        len(sample),
        len(data),
        100 * len(sample) / len(data),
        label_counts.sort_index().to_dict(),
    )
    return sample


async def select_features(
    stock_ids: List[int] | None = None,
    start_date: date | datetime = CURRENT_DATE - timedelta(400),
    end_date: date | datetime = CURRENT_DATE,
    n_estimators: int = 500,
    min_features: int = 10,
    step: float = 0.1,
    cv: int = 3,
    device: str = "cuda",
    save_config: bool = True,
    config_path: str | Path = DEFAULT_TRAIN_CONFIG,
    sample_fraction: float = 0.25,
    random_state: int = 42,
    progress_position: int = 0,
):
    client = init_async_db()
    model = xgb_model(n_estimators, device=device)

    try:
        async with client.connect() as app:
            data = await get_indicators_data(app, start_date, end_date, stock_ids)
    finally:
        await client.dispose()

    if data.empty:
        logger.warning(
            "No indicator data found between {} and {}; skipping feature selection.",
            start_date,
            end_date,
        )
        return

    data = process_data_training(data, start_date, end_date, HORIZON)
    prepared_rows = len(data)
    data = sample_feature_selection_data(data, sample_fraction, cv, random_state)
    X = data[NUMERIC_FEATURE_COLUMNS]
    Y = data["label"]

    rows = shap_rfe(
        model,
        X,
        Y,
        step=step,
        min_features=min_features,
        cv=cv,
        progress_position=progress_position,
    )
    best = max(rows, key=lambda r: r[1])
    logger.info(
        "Best feature set: {} feature(s) | CV F1 Macro = {:.6f}",
        len(best[2]),
        best[1],
    )

    if save_config:
        save_selected_features(
            features=best[2],
            score=best[1],
            params={
                "stock_ids": [str(stock_id) for stock_id in (stock_ids or [])],
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "n_estimators": n_estimators,
                "min_features": min_features,
                "step": step,
                "cv": cv,
                "device": device,
                "horizon": HORIZON,
                "sample_fraction": sample_fraction,
                "sample_random_state": random_state,
                "prepared_rows": prepared_rows,
                "sampled_rows": len(data),
            },
            path=config_path,
        )

    return best[2]


def _configured_features(
    config: dict[str, Any], training: dict[str, Any]
) -> tuple[list[str], str]:
    """Read the training feature list when feature engineering is disabled."""
    explicit = training.get("features") or []
    if explicit:
        return [str(feature) for feature in explicit], "training.features"

    selected = (config.get("feature_selection") or {}).get("features") or []
    if selected:
        return [str(feature) for feature in selected], "feature_selection.features"

    logger.warning(
        "No features configured in train.yaml; falling back to all {} numeric "
        "feature columns.",
        len(NUMERIC_FEATURE_COLUMNS),
    )
    return list(NUMERIC_FEATURE_COLUMNS), "NUMERIC_FEATURE_COLUMNS"


def _feature_engineering_params(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Read the SHAP-RFE parameters stored by a previous selection run."""
    params = (config.get("feature_selection") or {}).get("params") or {}
    return {
        "n_estimators": int(params.get("n_estimators", 500)),
        "min_features": int(params.get("min_features", 10)),
        "step": float(params.get("step", 0.1)),
        "cv": int(params.get("cv", 3)),
    }


async def resolve_features(
    config: dict[str, Any],
    training: dict[str, Any],
    train_config_path: Path,
    use_feature_engineering: bool,
    start_date: date,
    end_date: date,
    stock_ids: list[int],
    device: str,
    progress_position: int = 0,
) -> tuple[list[str], dict[str, Any]]:
    """Return the features used for training and their selection metadata."""
    if not use_feature_engineering:
        features, source = _configured_features(config, training)
        logger.info("Using {} configured feature(s) from {}", len(features), source)
        return features, {
            "enabled": False,
            "source": source,
            "n_features": len(features),
            "cv_f1_macro": None,
            "params": {},
        }

    logger.info("Running SHAP-RFE feature engineering...")
    select_params = _feature_engineering_params(config)
    sampling_config = training.get("feature_selection") or {}
    features = await select_features(
        stock_ids=stock_ids,
        start_date=start_date,
        end_date=end_date,
        config_path=train_config_path,
        device=device,
        sample_fraction=float(sampling_config.get("sample_fraction", 0.25)),
        random_state=int(sampling_config.get("random_state", 42)),
        progress_position=progress_position,
        **select_params,
    )

    if not features:
        raise RuntimeError(
            "Feature engineering returned no features; check the indicator data "
            "and the date range in train.yaml."
        )

    updated = load_train_config(train_config_path)
    selection = updated.get("feature_selection") or {}
    metadata = {
        "enabled": True,
        "source": "shap_rfe",
        "n_features": len(features),
        "cv_f1_macro": (selection.get("best") or {}).get("cv_f1_macro"),
        "params": selection.get("params") or select_params,
    }
    logger.success("SHAP-RFE selected {} feature(s).", len(features))
    return [str(feature) for feature in features], metadata


if __name__ == "__main__":
    import asyncio

    asyncio.run(
        select_features(
            stock_ids=[1, 2, 3, 4, 5],
            start_date=datetime(2017, 1, 1),
            end_date=datetime(2018, 1, 1),
            n_estimators=800,
        )
    )
