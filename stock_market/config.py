from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

CURRENT_DATE = datetime.now().date()
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

IDX_STOCK_LIST_URL = os.getenv(
    "IDX_STOCK_LIST_URL",
    "https://idx.co.id/id/data-pasar/data-saham/daftar-saham/",
)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/processed/portfolio.db")


HORIZON = 10

# Rolling 252-day features need history before the requested start date.
FEATURE_WARMUP_DAYS = 400
DEFAULT_STOCK_BATCH_SIZE = 50


SECTOR_MAPPING = {
    "Basic Materials": 0,
    "Communication Services": 1,
    "Consumer Cyclical": 2,
    "Consumer Defensive": 3,
    "Energy": 4,
    "Financial Services": 5,
    "Healthcare": 6,
    "Industrials": 7,
    "Real Estate": 8,
    "Technology": 9,
    "Utilities": 10,
}


# Keep these aligned with features_list in
# stock_market/analytics/outperformer_explore_cuda.ipynb.
NUMERIC_FEATURE_COLUMNS = [
    "returns",
    "distance_moving_avg",
    "momentum_5d",
    "momentum_10d",
    "momentum_20d",
    "momentum_30d",
    "momentum_60d",
    "momentum_120d",
    "momentum_250d",
    "momentum_6_1",
    "momentum_12_1",
    "volatility_5d_ann",
    "volatility_10d_ann",
    "volatility_15d_ann",
    "volatility_20d_ann",
    "volatility_25d_ann",
    "volatility_30d_ann",
    "moving_avg_slope_3d",
    "moving_avg_slope_5d",
    "moving_avg_slope_10d",
    "moving_avg_slope_15d",
    "distance_high_52w",
    "distance_low_52w",
    "risk_adjusted_momentum",
    "rank_momentum_20d",
    "rank_momentum_60d",
    "rank_momentum_120d",
    "rank_momentum_12_1",
    "sector_rel_momentum_20d",
    "sector_rel_momentum_120d",
    "rank_volatility_20d_ann",
    "rank_risk_adjusted_momentum",
    "rank_distance_high_52w",
    "trend_r2_60d",
    "efficiency_ratio_20d",
    "ma_alignment_score",
    "bollinger_pct_b_20",
    "gap",
    "vol_ratio_5_20",
    "atr_14_normalized",
    "daily_range_normalized",
    "rsi_14",
    "stochastic_pct_k_14",
    "williams_r_14",
    "rolling_max_drawdown_60d",
    "current_drawdown_from_rolling_max",
    "realized_skewness_60d",
    "realized_kurtosis_60d",
    "sortino_60d",
    "calmar_120d",
    "ulcer_index_60d",
    "candle_body_ratio",
    "close_position_in_range",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "consecutive_up_days_streak",
]
FEATURE_COLUMNS = ["sector", *NUMERIC_FEATURE_COLUMNS]


RANK_FEATURES = {
    "momentum_20d": "rank_momentum_20d",
    "momentum_60d": "rank_momentum_60d",
    "momentum_120d": "rank_momentum_120d",
    "momentum_12_1": "rank_momentum_12_1",
    "volatility_20d_ann": "rank_volatility_20d_ann",
    "risk_adjusted_momentum": "rank_risk_adjusted_momentum",
    "distance_high_52w": "rank_distance_high_52w",
}
