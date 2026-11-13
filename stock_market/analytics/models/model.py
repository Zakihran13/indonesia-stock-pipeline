"""XGBoost model creation, training, evaluation, and export."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from stock_market.analytics.data.preprocess import LABEL_NAMES, LABELS
from stock_market.config import ROOT_DIR
from utils.helper import to_builtin


def xgb_model(n_estimator: int = 500, device: str = "cuda"):
    return XGBClassifier(
        n_estimators=n_estimator,
        learning_rate=0.1,
        max_depth=5,
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        device=device,
        random_state=42,
        enable_categorical=True,
    )


def resolve_device(preferred: str = "auto") -> str:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if preferred and preferred != "auto":
        return preferred

    try:
        probe_x = np.array([[0.0], [1.0], [2.0], [3.0]])
        probe_y = np.array([0, 1, 0, 1])
        xgb.XGBClassifier(
            n_estimators=1,
            max_depth=1,
            tree_method="hist",
            device="cuda",
        ).fit(probe_x, probe_y, verbose=False)
        logger.info("CUDA is available; training on GPU.")
        return "cuda"
    except Exception as exc:  # pragma: no cover - depends on host hardware
        logger.warning("CUDA is unavailable ({}); training on CPU.", exc)
        return "cpu"


def _fixed_model_params(training: dict[str, Any], device: str) -> dict[str, Any]:
    """Merge fixed XGBoost parameters from train.yaml."""
    model_config = training.get("model") or {}
    params = dict(model_config.get("fixed_params") or {})
    params.setdefault("objective", "multi:softprob")
    params.setdefault("num_class", 3)
    params.setdefault("eval_metric", "mlogloss")
    params.setdefault("tree_method", "hist")
    params.setdefault("random_state", 42)
    params.setdefault("enable_categorical", True)
    params["device"] = device

    # These belong to fit()/the training loop, not the search estimator.
    params.pop("n_estimators", None)
    params.pop("early_stopping_rounds", None)
    return params


def _build_estimator(
    params: dict[str, Any],
    n_estimators: int,
    device: str,
) -> xgb.XGBClassifier:
    kwargs = {**params, "n_estimators": int(n_estimators), "device": device}
    return xgb.XGBClassifier(**kwargs)


def search_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    training: dict[str, Any],
    device: str,
) -> tuple[dict[str, Any], float | None, dict[str, Any]]:
    """Run RandomizedSearchCV using chronological folds."""
    search_config = training.get("search") or {}
    if not search_config.get("enabled", True):
        logger.info("Hyperparameter search disabled.")
        return (
            {},
            None,
            {
                "enabled": False,
                "scoring": str(search_config.get("scoring", "f1_macro")),
                "cv": int(search_config.get("cv", 3)),
                "n_iter": int(search_config.get("n_iter", 50)),
                "best_score": None,
                "best_params": {},
            },
        )

    distributions = search_config.get("param_distributions") or {}
    if not distributions:
        raise ValueError(
            "training.search.param_distributions is empty; add a search space "
            "or set training.search.enabled to false."
        )

    cv_splits = int(search_config.get("cv", 3))
    if len(X_train) <= cv_splits + 1:
        raise ValueError(
            f"Not enough training rows ({len(X_train)}) for {cv_splits}-fold CV."
        )

    params = _fixed_model_params(training, device)
    estimator = _build_estimator(
        params,
        n_estimators=int(search_config.get("n_estimators", 500)),
        device=device,
    )
    n_iter = int(search_config.get("n_iter", 50))
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=distributions,
        n_iter=n_iter,
        scoring=str(search_config.get("scoring", "f1_macro")),
        cv=TimeSeriesSplit(n_splits=cv_splits),
        random_state=int(search_config.get("random_state", 42)),
        n_jobs=int(search_config.get("n_jobs", 1)),
        verbose=int(search_config.get("verbose", 1)),
        refit=False,
    )

    logger.info(
        "Searching {} candidates with {} time-series folds...",
        n_iter,
        cv_splits,
    )
    search.fit(X_train, y_train)
    best_params = dict(search.best_params_)
    best_score = float(search.best_score_)
    logger.success(
        "Best CV {} = {:.6f} | params={}",
        search_config.get("scoring", "f1_macro"),
        best_score,
        best_params,
    )
    return (
        best_params,
        best_score,
        {
            "enabled": True,
            "scoring": str(search_config.get("scoring", "f1_macro")),
            "cv": cv_splits,
            "n_iter": n_iter,
            "best_score": best_score,
            "best_params": best_params,
        },
    )


def fit_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    best_params: dict[str, Any],
    training: dict[str, Any],
    device: str,
) -> xgb.XGBClassifier:
    """Fit on train and use validation for early stopping."""
    model_config = training.get("model") or {}
    params = _fixed_model_params(training, device)
    params.update(best_params)

    early_stopping_rounds = model_config.get("early_stopping_rounds")
    if early_stopping_rounds:
        params["early_stopping_rounds"] = int(early_stopping_rounds)

    model = _build_estimator(
        params,
        n_estimators=int(model_config.get("n_estimators", 3000)),
        device=device,
    )

    logger.info(
        "Fitting final model | n_estimators={} early_stopping_rounds={}",
        model.n_estimators,
        early_stopping_rounds,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=bool(model_config.get("verbose", False)),
    )
    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y_true: pd.Series,
    split_name: str,
) -> dict[str, Any]:
    """Compute class and aggregate metrics for one split."""
    y_pred = model.predict(X)
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=[LABEL_NAMES[label] for label in LABELS],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "rows": int(len(X)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "label_distribution": {
            int(label): float(value)
            for label, value in y_true.value_counts(normalize=True).sort_index().items()
        },
    }

    logger.info(
        "{} metrics | rows={} accuracy={:.4f} balanced_accuracy={:.4f} f1_macro={:.4f}",
        split_name,
        metrics["rows"],
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["f1_macro"],
    )
    return to_builtin(metrics)


def save_model_artifact(
    model: xgb.XGBClassifier,
    training: dict[str, Any],
) -> Path:
    """Export the trained booster into the configured models directory."""
    output_config = training.get("output") or {}
    model_dir = Path(output_config.get("model_dir", "stock_market/analytics/models"))
    if not model_dir.is_absolute():
        model_dir = ROOT_DIR / model_dir

    model_name = str(output_config.get("model_name", "outperformer_xgb_latest.ubj"))
    model_path = model_dir / model_name
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    logger.success("Saved model artifact to {}", model_path)
    return model_path
