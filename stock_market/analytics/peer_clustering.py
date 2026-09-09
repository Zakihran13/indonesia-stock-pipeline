from pathlib import Path

import pandas as pd

from stock_market.analytics.common import (
    default_artifact_path,
    load_indicators_sync,
    numeric_features,
)

NUMERIC_FEATURES = [
    "return_on_equity",
    "payout_ratio",
    "total_debt",
    "free_cashflow_to_market_cap",
    "momentum_30d",
    "volatility_20d",
]


def latest_peer_frame(indicators: pd.DataFrame) -> pd.DataFrame:
    return indicators.sort_values("indicator_date").groupby("stock_id").tail(1)


def train(
    indicators: pd.DataFrame | None = None,
    artifact_path: Path | None = None,
    clusters: int = 5,
):
    from sklearn.cluster import KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import joblib

    indicators = load_indicators_sync() if indicators is None else indicators
    frame = latest_peer_frame(indicators)
    if len(frame) < clusters:
        raise ValueError("The number of clusters cannot exceed the number of stocks.")
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            (
                "sector",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["sector"],
            ),
        ]
    )
    transformed = preprocessor.fit_transform(frame)
    cluster_model = KMeans(n_clusters=clusters, n_init=10, random_state=42)
    anomaly_model = IsolationForest(contamination="auto", random_state=42)
    cluster_model.fit(transformed)
    anomaly_model.fit(transformed)
    artifact_path = artifact_path or default_artifact_path("peer_clustering")
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "cluster_model": cluster_model,
            "anomaly_model": anomaly_model,
            "numeric_features": NUMERIC_FEATURES,
        },
        artifact_path,
    )
    return cluster_model


def predict_latest(indicators: pd.DataFrame, artifact_path: Path | None = None):
    import joblib

    artifact_path = artifact_path or default_artifact_path("peer_clustering")
    artifact = joblib.load(artifact_path)
    latest = latest_peer_frame(indicators)
    transformed = artifact["preprocessor"].transform(latest)
    result = latest[["stock_id", "indicator_date"]].copy()
    result["peer_cluster"] = artifact["cluster_model"].predict(transformed)
    result["anomaly_score"] = artifact["anomaly_model"].decision_function(transformed)
    result["is_anomaly"] = artifact["anomaly_model"].predict(transformed) == -1
    return result