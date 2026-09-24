from pathlib import Path

import numpy as np
import pandas as pd

from stock_market.analytics.common import (
    default_artifact_path,
    load_indicators_sync,
    numeric_features,
)

FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "daily_return",
    "volatility_5d",
    "volatility_20d",
    "earnings_within_7d",
]


def make_training_set(indicators: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    frame = indicators.sort_values(["stock_id", "indicator_date"]).copy()

    def future_volatility(values: pd.Series) -> pd.Series:
        return values.iloc[::-1].rolling(5, min_periods=5).std().iloc[::-1].shift(-1)

    target = frame.groupby("stock_id")["daily_return"].transform(future_volatility)
    valid = target.notna()
    return numeric_features(frame.loc[valid], FEATURES), target.loc[valid]


def train(indicators: pd.DataFrame | None = None, artifact_path: Path | None = None):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    import joblib

    indicators = load_indicators_sync() if indicators is None else indicators
    features, target = make_training_set(indicators)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )
    model.fit(features, target)
    artifact_path = artifact_path or default_artifact_path("volatility_regime")
    joblib.dump({"model": model, "features": FEATURES}, artifact_path)
    return model


def predict_latest(indicators: pd.DataFrame, artifact_path: Path | None = None):
    import joblib

    artifact_path = artifact_path or default_artifact_path("volatility_regime")
    artifact = joblib.load(artifact_path)
    latest = indicators.sort_values("indicator_date").groupby("stock_id").tail(1)
    result = latest[["stock_id", "indicator_date"]].copy()
    result["forecast_volatility_5d"] = artifact["model"].predict(
        numeric_features(latest, artifact["features"])
    )
    percentile = result["forecast_volatility_5d"].rank(method="first", pct=True)
    result["risk_regime"] = pd.cut(
        percentile,
        bins=[0, 1 / 3, 2 / 3, 1],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )
    return result