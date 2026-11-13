from datetime import datetime

import numpy as np
import pandas as pd

from data.db import entities_transformed as et
from stock_market.config import RANK_FEATURES


def _rolling_r_squared(values: np.ndarray) -> float:
    if not np.isfinite(values).all():
        return np.nan

    correlation = np.corrcoef(np.arange(len(values)), values)[0, 1]
    return float(correlation**2) if np.isfinite(correlation) else np.nan


def _rolling_max_drawdown(values: np.ndarray) -> float:
    if not np.isfinite(values).all() or np.any(values <= 0):
        return np.nan

    drawdowns = values / np.maximum.accumulate(values) - 1
    return float(drawdowns.min())


def _rolling_ulcer_index(values: np.ndarray) -> float:
    if not np.isfinite(values).all() or np.any(values <= 0):
        return np.nan

    drawdowns = values / np.maximum.accumulate(values) - 1
    return float(np.sqrt(np.mean(drawdowns**2)))


def build_indicator_frame(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    start_date: datetime | None = None,
    *,
    calculate_cross_section: bool = True,
) -> pd.DataFrame:
    output_columns = list(et.IndicatorsData.__table__.columns.keys())
    if prices.empty:
        return pd.DataFrame(columns=output_columns)

    indicators = prices.copy()
    indicators["indicator_date"] = pd.to_datetime(indicators["trade_date"])
    indicators = indicators.sort_values(["stock_id", "indicator_date"])
    if not metadata.empty:
        metadata_columns = [
            "stock_id",
            *[
                column
                for column in metadata.columns
                if column not in indicators.columns
                and column not in {"stock_id", "created_at"}
            ],
        ]
        indicators = indicators.merge(
            metadata[metadata_columns],
            on="stock_id",
            how="left",
            validate="many_to_one",
        )

    for column in ("ticker", "sector", "open", "high", "low", "close"):
        if column not in indicators:
            indicators[column] = np.nan

    for column in ("open", "high", "low", "close"):
        indicators[column] = pd.to_numeric(indicators[column], errors="coerce")

    grouped = indicators.groupby("stock_id", sort=False)
    close = grouped["close"]
    indicators["returns"] = close.pct_change(fill_method=None)
    moving_avg_30d = close.transform(
        lambda values: values.rolling(30, min_periods=30).mean()
    )
    indicators["distance_moving_avg"] = (
        indicators["close"].div(moving_avg_30d).sub(1).astype("float32")
    )
    indicators["high_52w"] = grouped["high"].transform(
        lambda values: values.shift(1).rolling(252, min_periods=252).max()
    )
    indicators["low_52w"] = grouped["low"].transform(
        lambda values: values.shift(1).rolling(252, min_periods=252).min()
    )

    for period in (5, 10, 20, 30, 60, 120, 250):
        indicators[f"momentum_{period}d"] = close.pct_change(
            period, fill_method=None
        ).astype("float32")

    indicators["momentum_6_1"] = close.shift(21).div(close.shift(126)).sub(1)
    indicators["momentum_12_1"] = close.shift(21).div(close.shift(252)).sub(1)

    for period in range(5, 31, 5):
        daily_volatility = grouped["returns"].transform(
            lambda values, period=period: values.rolling(
                period, min_periods=period
            ).std()
        )
        indicators[f"volatility_{period}d_ann"] = daily_volatility * np.sqrt(252)

    moving_avg_30d_grouped = moving_avg_30d.groupby(indicators["stock_id"], sort=False)
    for period in (3, 5, 10, 15):
        prior_average = moving_avg_30d_grouped.shift(period)
        indicators[f"moving_avg_slope_{period}d"] = moving_avg_30d.div(
            prior_average
        ).sub(1)

    indicators["distance_high_52w"] = (
        indicators["close"].div(indicators["high_52w"]).sub(1).astype("float32")
    )
    indicators["distance_low_52w"] = (
        indicators["close"].div(indicators["low_52w"]).sub(1).astype("float32")
    )
    indicators["risk_adjusted_momentum"] = indicators["momentum_30d"].div(
        indicators["volatility_30d_ann"].replace(0, np.nan)
    )

    if calculate_cross_section:
        for period in (20, 120):
            feature = f"momentum_{period}d"
            indicators[f"sector_rel_{feature}"] = indicators[feature].sub(
                indicators.groupby(["indicator_date", "sector"], dropna=True)[
                    feature
                ].transform("mean")
            )

    sma = {
        period: close.transform(
            lambda values, period=period: values.rolling(
                period, min_periods=period
            ).mean()
        )
        for period in (20, 50, 200)
    }
    indicators["ma_alignment_score"] = (
        (sma[20] > sma[50]).astype("int8")
        + (sma[50] > sma[200]).astype("int8")
        + (sma[20] > sma[200]).astype("int8")
    ).where(sma[20].notna() & sma[50].notna() & sma[200].notna())
    std_20d = close.transform(lambda values: values.rolling(20, min_periods=20).std())
    indicators["bollinger_pct_b_20"] = (
        indicators["close"].sub(sma[20]).div(std_20d.mul(2).replace(0, np.nan))
    )
    indicators["trend_r2_60d"] = close.transform(
        lambda values: values.rolling(60, min_periods=60).apply(
            _rolling_r_squared, raw=True
        )
    )
    absolute_changes = close.diff().abs()
    change_sum_20d = absolute_changes.groupby(
        indicators["stock_id"], sort=False
    ).transform(lambda values: values.rolling(20, min_periods=20).sum())
    indicators["efficiency_ratio_20d"] = (
        indicators["close"]
        .sub(close.shift(20))
        .abs()
        .div(change_sum_20d.replace(0, np.nan))
    )

    previous_close = close.shift(1)
    open_price = indicators["open"]
    high = indicators["high"]
    low = indicators["low"]
    candle_range = high.sub(low)
    candle_range_nonzero = candle_range.replace(0, np.nan)
    previous_close_nonzero = previous_close.replace(0, np.nan)

    indicators["gap"] = open_price.sub(previous_close).div(previous_close_nonzero)
    indicators["daily_range_normalized"] = high.sub(low).div(previous_close_nonzero)
    indicators["candle_body_ratio"] = (
        indicators["close"].sub(open_price).abs().div(candle_range_nonzero)
    )
    indicators["close_position_in_range"] = (
        indicators["close"].sub(low).div(candle_range_nonzero)
    )
    indicators["upper_shadow_ratio"] = high.sub(
        pd.concat([open_price, indicators["close"]], axis=1).max(axis=1)
    ).div(candle_range_nonzero)
    indicators["lower_shadow_ratio"] = (
        pd.concat([open_price, indicators["close"]], axis=1)
        .min(axis=1)
        .sub(low)
        .div(candle_range_nonzero)
    )
    indicators["doji_flag"] = (indicators["candle_body_ratio"] < 0.1).where(
        candle_range_nonzero.notna()
    )
    indicators["vol_ratio_5_20"] = indicators["volatility_5d_ann"].div(
        indicators["volatility_20d_ann"].replace(0, np.nan)
    )

    high_14d = grouped["high"].transform(
        lambda values: values.rolling(14, min_periods=14).max()
    )
    low_14d = grouped["low"].transform(
        lambda values: values.rolling(14, min_periods=14).min()
    )
    indicators["stochastic_pct_k_14"] = (
        indicators["close"].sub(low_14d).div(high_14d.sub(low_14d).replace(0, np.nan))
    )
    indicators["williams_r_14"] = high_14d.sub(indicators["close"]).div(
        high_14d.sub(low_14d).replace(0, np.nan)
    )

    daily_change = close.diff()
    average_gain = (
        daily_change.clip(lower=0)
        .groupby(indicators["stock_id"], sort=False)
        .transform(lambda values: values.rolling(14, min_periods=14).mean())
    )
    average_loss = (
        -daily_change.clip(upper=0)
        .groupby(indicators["stock_id"], sort=False)
        .transform(lambda values: values.rolling(14, min_periods=14).mean())
    )
    relative_strength = average_gain.div(average_loss)
    indicators["rsi_14"] = 100 - 100 / (1 + relative_strength)

    true_range = pd.concat(
        [
            high.sub(low),
            high.sub(previous_close).abs(),
            low.sub(previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = true_range.groupby(indicators["stock_id"], sort=False).transform(
        lambda values: values.rolling(14, min_periods=14).mean()
    )
    indicators["atr_14_normalized"] = atr_14.div(indicators["close"].replace(0, np.nan))

    indicators["rolling_max_drawdown_60d"] = close.transform(
        lambda values: values.rolling(60, min_periods=60).apply(
            _rolling_max_drawdown, raw=True
        )
    )
    indicators["current_drawdown_from_rolling_max"] = (
        indicators["close"]
        .div(
            grouped["close"].transform(
                lambda values: values.rolling(60, min_periods=1).max()
            )
        )
        .sub(1)
    )
    indicators["realized_skewness_60d"] = grouped["returns"].transform(
        lambda values: values.rolling(60, min_periods=60).skew()
    )
    indicators["realized_kurtosis_60d"] = (
        grouped["returns"].transform(
            lambda values: values.rolling(60, min_periods=60).kurt()
        )
        + 3
    )
    downside_deviation = (
        indicators["returns"]
        .clip(upper=0)
        .pow(2)
        .groupby(indicators["stock_id"], sort=False)
        .transform(lambda values: values.rolling(60, min_periods=60).mean().pow(0.5))
    )
    mean_return_60d = grouped["returns"].transform(
        lambda values: values.rolling(60, min_periods=60).mean()
    )
    indicators["sortino_60d"] = mean_return_60d.div(
        downside_deviation.replace(0, np.nan)
    )
    drawdown_120d = close.transform(
        lambda values: values.rolling(120, min_periods=120).apply(
            _rolling_max_drawdown, raw=True
        )
    )
    annualized_return_120d = (1 + indicators["momentum_120d"]).pow(252 / 120) - 1
    indicators["calmar_120d"] = annualized_return_120d.div(
        drawdown_120d.abs().replace(0, np.nan)
    )
    indicators["ulcer_index_60d"] = close.transform(
        lambda values: values.rolling(60, min_periods=60).apply(
            _rolling_ulcer_index, raw=True
        )
    )
    up_days = close.diff().gt(0)
    indicators["consecutive_up_days_streak"] = up_days.groupby(
        indicators["stock_id"], sort=False
    ).transform(
        lambda values: values.astype("int8").groupby((~values).cumsum()).cumsum()
    )
    prior_high_20d = grouped["high"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=20).max()
    )
    indicators["breakout_flag_20d"] = (indicators["close"] > prior_high_20d).where(
        prior_high_20d.notna()
    )

    if calculate_cross_section:
        for feature, rank_column in RANK_FEATURES.items():
            indicators[rank_column] = indicators.groupby("indicator_date")[
                feature
            ].rank(method="average", pct=True, ascending=True)
    else:
        for column in (
            *RANK_FEATURES.values(),
            "sector_rel_momentum_20d",
            "sector_rel_momentum_120d",
        ):
            indicators[column] = np.nan

    if start_date is not None:
        start_timestamp = pd.Timestamp(start_date).tz_localize(None).normalize()
        indicators = indicators[indicators["indicator_date"] >= start_timestamp].copy()

    indicators["indicator_date"] = indicators["indicator_date"].dt.date
    return indicators.loc[:, output_columns].replace({np.nan: None})
