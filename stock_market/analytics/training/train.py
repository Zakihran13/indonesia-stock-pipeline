"""End-to-end training flow for the triple-barrier outperformer model.

The flow intentionally mirrors the research notebook but keeps each step small
and reusable:

1. Load ``train.yaml``.
2. Pick the feature set: SHAP-RFE via ``feature_engineering.py`` or the
   existing feature list in ``train.yaml``.
3. Load indicator data and build triple-barrier labels.
4. Split chronologically into train / validation / test with an embargo.
5. Run a randomized hyperparameter search (optional).
6. Fit the final model with early stopping.
7. Evaluate on the validation and holdout sets.
8. Export the model to ``stock_market/analytics/models`` and write the latest
   result metadata to ``inference.yaml``.

Run it with::

    poetry run python -m stock_market.analytics.training.train \
        --no-feature-engineering

Use ``--feature-engineering`` to run SHAP-RFE and use its returned feature
list for training. Without the flag, the flow reads features from
``training.features`` (or ``feature_selection.features``) in ``train.yaml``.
SHAP-RFE uses ``training.feature_selection.sample_fraction`` of the prepared
rows (default 0.25); final training still uses the full dataset. Feature
selection currently samples from the entire date range, including holdouts.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm.auto import tqdm

from stock_market.analytics.data.preprocess import (
    prepare_training_frame,
    temporal_split,
)
from stock_market.analytics.data.statement import fetch_training_data
from stock_market.analytics.models.model import (
    evaluate_model,
    fit_final_model,
    resolve_device,
    save_model_artifact,
    search_hyperparameters,
)
from stock_market.analytics.training.feature_engineering import resolve_features
from stock_market.analytics.utils.common import (
    DEFAULT_START_DATE,
    build_inference_config,
    build_split_metadata,
    label_distribution,
    load_training_config,
    parse_date,
    resolve_project_path,
)
from stock_market.config import CURRENT_DATE, HORIZON
from utils.helper import (
    DEFAULT_INFERENCE_CONFIG,
    DEFAULT_TRAIN_CONFIG,
    save_inference_config,
)


async def run_training(
    train_config_path: str | Path = DEFAULT_TRAIN_CONFIG,
    feature_engineering: bool | None = None,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    stock_ids: list[int] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run the full learning flow and return the generated inference config."""
    train_config_path = resolve_project_path(train_config_path)

    with tqdm(
        total=8,
        desc="Training pipeline",
        unit="step",
        position=0,
        leave=True,
        nrows=20,
        disable=None,
    ) as progress:
        progress.set_description("Step 1/8 | loading config")
        logger.info("Step 1/8: Loading training config from {}", train_config_path)
        config, training = load_training_config(train_config_path)

        data_config = training.get("data") or {}
        resolved_start = parse_date(
            start_date if start_date is not None else data_config.get("start_date"),
            DEFAULT_START_DATE,
        )
        resolved_end = parse_date(
            end_date if end_date is not None else data_config.get("end_date"),
            CURRENT_DATE,
        )
        if resolved_start >= resolved_end:
            raise ValueError(
                f"start_date ({resolved_start}) must be before end_date "
                f"({resolved_end})."
            )

        configured_stock_ids = (
            stock_ids if stock_ids is not None else data_config.get("stock_ids") or []
        )
        resolved_stock_ids = [int(stock_id) for stock_id in configured_stock_ids]

        use_feature_engineering = (
            bool(training.get("feature_engineering", False))
            if feature_engineering is None
            else bool(feature_engineering)
        )

        model_config = training.get("model") or {}
        resolved_device = resolve_device(
            str(device or model_config.get("device", "auto"))
        )
        progress.update(1)

        progress.set_description("Step 2/8 | resolving features")
        features, feature_metadata = await resolve_features(
            config=config,
            training=training,
            train_config_path=train_config_path,
            use_feature_engineering=use_feature_engineering,
            start_date=resolved_start,
            end_date=resolved_end,
            stock_ids=resolved_stock_ids,
            device=resolved_device,
            progress_position=1,
        )
        progress.update(1)

        progress.set_description("Step 3/8 | loading indicator data")
        logger.info(
            "Step 3/8: Loading indicator data | {} -> {} | stocks={}",
            resolved_start,
            resolved_end,
            resolved_stock_ids or "all",
        )
        raw_data = await fetch_training_data(
            resolved_start,
            resolved_end,
            resolved_stock_ids,
        )
        frame = prepare_training_frame(
            raw_data,
            resolved_start,
            resolved_end,
            features,
        )
        progress.update(1)

        progress.set_description("Step 4/8 | splitting train/validation/test")
        split_config = training.get("split") or {}
        train_frame, validation_frame, test_frame = temporal_split(
            frame,
            train_fraction=float(split_config.get("train_fraction", 0.7)),
            validation_fraction=float(split_config.get("validation_fraction", 0.15)),
            embargo_days=int(split_config.get("embargo_days", HORIZON * 2)),
        )
        progress.update(1)

        X_train = train_frame[features]
        y_train = train_frame["label"]
        X_validation = validation_frame[features]
        y_validation = validation_frame["label"]
        X_test = test_frame[features]
        y_test = test_frame["label"]

        progress.set_description("Step 5/8 | hyperparameter search")
        best_params, _, search_metadata = search_hyperparameters(
            X_train,
            y_train,
            training,
            resolved_device,
        )
        progress.update(1)

        progress.set_description("Step 6/8 | fitting final model")
        model = fit_final_model(
            X_train,
            y_train,
            X_validation,
            y_validation,
            best_params,
            training,
            resolved_device,
        )
        progress.update(1)

        progress.set_description("Step 7/8 | evaluating model")
        validation_metrics = evaluate_model(
            model, X_validation, y_validation, "Validation"
        )
        test_metrics = evaluate_model(model, X_test, y_test, "Test")
        progress.update(1)

        progress.set_description("Step 8/8 | exporting artifacts")
        logger.info("Step 8/8: Exporting model and inference metadata...")
        model_path = save_model_artifact(model, training)
        output_config = training.get("output") or {}
        inference_path = resolve_project_path(
            output_config.get("inference_config", DEFAULT_INFERENCE_CONFIG)
        )

        inference_config = build_inference_config(
            model=model,
            model_path=model_path,
            features=features,
            feature_metadata=feature_metadata,
            start_date=resolved_start,
            end_date=resolved_end,
            stock_ids=resolved_stock_ids,
            row_count=len(frame),
            label_distribution=label_distribution(frame),
            split_metadata=build_split_metadata(
                train_frame, validation_frame, test_frame, split_config
            ),
            search_metadata=search_metadata,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            device=resolved_device,
        )
        save_inference_config(inference_config, inference_path)
        progress.update(1)
        progress.set_description("Training complete")

    logger.success(
        "Training complete | model={} | inference_config={}",
        model_path,
        inference_path,
    )
    return inference_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the triple-barrier outperformer model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG,
        help="Path to train.yaml.",
    )
    parser.add_argument(
        "--feature-engineering",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Run SHAP-RFE and train on its output. Overrides "
            "training.feature_engineering in train.yaml."
        ),
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Override training.data.start_date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Override training.data.end_date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--stock-ids",
        type=int,
        nargs="+",
        default=None,
        help="Override training.data.stock_ids.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Override training.model.device.",
    )
    return parser


def _configure_logging() -> None:
    """Send loguru output through tqdm so active progress bars redraw cleanly."""
    logger.remove()
    logger.add(
        lambda message: tqdm.write(str(message).rstrip("\n")),
        level="INFO",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
    )


def main() -> None:
    args = build_parser().parse_args()
    _configure_logging()
    asyncio.run(
        run_training(
            train_config_path=args.config,
            feature_engineering=args.feature_engineering,
            start_date=args.start_date,
            end_date=args.end_date,
            stock_ids=args.stock_ids,
            device=args.device,
        )
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(
        run_training(
            feature_engineering=True,
            start_date=datetime(2017, 1, 1),
            end_date=datetime(2020, 1, 1),
            device="cuda",
        )
    )
