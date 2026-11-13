import pandas as pd
import numpy as np
from datetime import date, datetime
from loguru import logger

from stock_market.config import HORIZON, NUMERIC_FEATURE_COLUMNS

LABEL_NAMES = {
    0: "lower_barrier",
    1: "neither_barrier",
    2: "upper_barrier",
}
LABELS = [0, 1, 2]


def make_triple_barrier_labels(
    data: pd.DataFrame,
    horizon: int = HORIZON,
    upper_return: float = 0.05,
    lower_return: float = 0.05,
) -> pd.Series:
    """Create 0/1/2 lower/neither/upper labels for rows sorted by stock and date.
    A bar touching both barriers is labeled lower; incomplete windows remain NA.
    """
    required = {"stock_id", "close", "high", "low"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"Missing columns for triple-barrier labels: {sorted(missing)}"
        )
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if (
        not np.isfinite(upper_return)
        or not np.isfinite(lower_return)
        or upper_return <= 0
        or not 0 < lower_return < 1
    ):
        raise ValueError(
            "upper_return must be positive and lower_return must be between 0 and 1"
        )

    close = pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype="float64")
    high = pd.to_numeric(data["high"], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(data["low"], errors="coerce").to_numpy(dtype="float64")
    labels = np.full(len(data), np.nan, dtype="float64")

    for positions in data.groupby("stock_id", sort=False).indices.values():
        positions = np.asarray(positions)
        event_count = len(positions) - horizon
        if event_count <= 0:
            continue

        event_positions = positions[:event_count]
        entry_prices = close[event_positions]
        active = np.isfinite(entry_prices) & (entry_prices > 0)
        upper_barriers = entry_prices * (1 + upper_return)
        lower_barriers = entry_prices * (1 - lower_return)

        for offset in range(1, horizon + 1):
            future_positions = positions[offset : offset + event_count]
            valid_bar = np.isfinite(high[future_positions]) & np.isfinite(
                low[future_positions]
            )
            active[~valid_bar] = False

            upper_hit = active & (high[future_positions] >= upper_barriers)
            lower_hit = active & (low[future_positions] <= lower_barriers)
            touched = upper_hit | lower_hit
            labels[event_positions[touched]] = np.where(lower_hit[touched], 0, 2)
            active[touched] = False

        labels[event_positions[active]] = 1

    return pd.Series(pd.array(labels, dtype="Int8"), index=data.index, name="label")


def process_data_training(
    data: pd.DataFrame,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    horizon: int = HORIZON,
    upper_return: float = 0.05,
    lower_return: float = 0.05,
    non_overlapping: bool = True,
) -> pd.DataFrame:
    data = (
        data.sort_values(["stock_id", "indicator_date"]).reset_index(drop=True).copy()
    )
    data["label"] = make_triple_barrier_labels(
        data,
        horizon=horizon,
        upper_return=upper_return,
        lower_return=lower_return,
    )

    if start_date is not None:
        data = data.loc[
            pd.to_datetime(data["indicator_date"]) >= pd.Timestamp(start_date)
        ]
    if end_date is not None:
        data = data.loc[
            pd.to_datetime(data["indicator_date"]) <= pd.Timestamp(end_date)
        ]

    if non_overlapping:
        data = data.loc[data.groupby("stock_id", sort=False).cumcount() % horizon == 0]

    for col in NUMERIC_FEATURE_COLUMNS:
        data[col] = data[col].replace(["NaN", "null", "NA", "?"], np.nan)
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return (
        data.dropna().assign(label=lambda frame: frame["label"].astype("int8")).copy()
    )


def prepare_training_frame(
    data: pd.DataFrame,
    start_date: date | datetime,
    end_date: date | datetime,
    features: list[str],
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Build labels, validate feature columns, and drop unusable rows."""
    frame = process_data_training(
        data,
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
    )
    if frame.empty:
        raise ValueError(
            "No training rows remain after label creation and cleaning. "
            "Use a longer date range or a smaller feature set."
        )

    if not features:
        raise ValueError("The training feature list is empty.")

    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError(f"Features missing from the dataset: {missing}")

    for feature in features:
        if feature in NUMERIC_FEATURE_COLUMNS:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    frame = frame.dropna(subset=[*features, "label"]).copy()
    frame["label"] = frame["label"].astype(int)
    if frame.empty:
        raise ValueError("No training rows remain after dropping NaN feature values.")
    return frame


def temporal_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    embargo_days: int = HORIZON * 2,
    date_column: str = "indicator_date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split with a gap so label windows cannot cross splits."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be < 1.")

    data = frame.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    dates = np.sort(data[date_column].unique())
    if len(dates) < 20:
        raise ValueError(
            "Not enough distinct dates for a chronological train/validation/test split."
        )

    train_end = pd.Timestamp(dates[int(len(dates) * train_fraction)])
    validation_end = pd.Timestamp(
        dates[int(len(dates) * (train_fraction + validation_fraction))]
    )
    gap = pd.Timedelta(days=max(0, int(embargo_days)))

    train = data[data[date_column] < train_end - gap].copy()
    validation = data[
        (data[date_column] >= train_end) & (data[date_column] < validation_end - gap)
    ].copy()
    test = data[data[date_column] >= validation_end].copy()

    for name, part in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        if part.empty:
            raise ValueError(
                f"The {name} split is empty. Adjust the split fractions or "
                "date range."
            )

    if train["label"].nunique() < 2:
        raise ValueError("The training split contains fewer than two label classes.")

    logger.info(
        "Split rows | train={} validation={} test={}",
        len(train),
        len(validation),
        len(test),
    )
    return (
        train.sort_values(date_column),
        validation.sort_values(date_column),
        test.sort_values(date_column),
    )
