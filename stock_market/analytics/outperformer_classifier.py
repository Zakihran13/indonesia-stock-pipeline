from pathlib import Path

import pandas as pd

from stock_market.analytics.common import (
    default_artifact_path,
    load_indicators_sync,
    numeric_features,
)

FEATURES = [
    "momentum_30d",
    "moving_average_30d",
    "free_cashflow_to_market_cap",
    "target_price_deviation",
    "recommendation_score_change",
    "volatility_20d",
]


def make_training_set(indicators: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    frame = indicators.sort_values(["stock_id", "indicator_date"]).copy()
    future_close = frame.groupby("stock_id")["close"].shift(-30)
    forward_return = future_close.div(frame["close"]).sub(1)
    labels = (forward_return >= 0.05).astype("int8")
    valid = forward_return.notna() & frame[FEATURES].notna().all(axis=1)
    return numeric_features(frame.loc[valid], FEATURES), labels.loc[valid]


def train(indicators: pd.DataFrame | None = None, artifact_path: Path | None = None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    import joblib

    indicators = load_indicators_sync() if indicators is None else indicators
    features, labels = make_training_set(indicators)
    if labels.nunique() < 2:
        raise ValueError("Outperformer training requires both target classes.")
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(features, labels)
    artifact_path = artifact_path or default_artifact_path("outperformer_classifier")
    joblib.dump({"model": model, "features": FEATURES}, artifact_path)
    return model


def predict_latest(indicators: pd.DataFrame, artifact_path: Path | None = None):
    import joblib

    artifact_path = artifact_path or default_artifact_path("outperformer_classifier")
    artifact = joblib.load(artifact_path)
    latest = indicators.sort_values("indicator_date").groupby("stock_id").tail(1)
    result = latest[["stock_id", "indicator_date"]].copy()
    result["outperformer_probability"] = artifact["model"].predict_proba(
        numeric_features(latest, artifact["features"])
    )[:, 1]
    result["outperformer_prediction"] = (
        result["outperformer_probability"] >= 0.5
    ).astype("int8")
    return result



if __name__ == "__main__":
    train()