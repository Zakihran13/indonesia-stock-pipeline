from sqlalchemy import (
    Column,
    Integer,
    String,
    Numeric,
    DateTime,
    Date,
    BigInteger,
    Boolean,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()


class StockMetadata(Base):
    __tablename__ = "metadata"
    __table_args__ = {"schema": "stock_market"}

    # Primary Key
    stock_id = Column(Integer, primary_key=True, autoincrement=True)

    # Columns
    ticker = Column(String)
    symbol = Column(String)
    company_name = Column(String)
    sector = Column(String)
    industry = Column(String)
    exchange = Column(String)
    market = Column(String)
    full_time_employees = Column(Integer)
    ceo_name = Column(String)
    full_address = Column(String)
    gmt_offset_hours = Column(Numeric)
    exchange_timezone_name = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


class FundamentalData(Base):
    __tablename__ = "fundamental_data"
    __table_args__ = {"schema": "stock_market"}

    # Composite Primary Key for snapshot data
    stock_id = Column(
        Integer, ForeignKey("stock_market.metadata.stock_id"), primary_key=True
    )
    created_at = Column(DateTime, primary_key=True)
    retrieve_at = Column(DateTime)

    # Columns
    market_cap = Column(BigInteger)
    enterprise_value = Column(BigInteger)
    shares_outstanding = Column(BigInteger)
    trailing_pe = Column(Numeric)
    forward_pe = Column(Numeric)
    dividend_yield = Column(Numeric)
    payout_ratio = Column(Numeric)
    last_fiscal_year_end_date = Column(DateTime)
    most_recent_quarter_date = Column(DateTime)
    ex_dividend_date = Column(DateTime)
    last_dividend_date = Column(DateTime)
    total_cash = Column(BigInteger)
    total_debt = Column(BigInteger)
    free_cashflow = Column(BigInteger)
    operating_cashflow = Column(BigInteger)
    return_on_equity = Column(Numeric)


class DynamicData(Base):
    __tablename__ = "dynamic_data"
    __table_args__ = {"schema": "stock_market"}

    # Composite Primary Key for snapshot data
    stock_id = Column(
        Integer, ForeignKey("stock_market.metadata.stock_id"), primary_key=True
    )
    created_at = Column(DateTime, primary_key=True)
    retrieve_at = Column(DateTime)

    # Columns
    current_price = Column(Numeric)
    previous_close = Column(Numeric)
    open = Column(Numeric)
    day_low = Column(Numeric)
    day_high = Column(Numeric)
    volume = Column(BigInteger)
    average_volume_10days = Column(BigInteger)
    regular_market_time_utc = Column(DateTime)
    first_trade_date_utc = Column(DateTime)
    fifty_two_week_low = Column(Numeric)
    fifty_two_week_high = Column(Numeric)
    fifty_two_week_change_percent = Column(Numeric)
    fifty_day_average = Column(Numeric)
    two_hundred_day_average = Column(Numeric)
    market_state = Column(String)


class AnalyticData(Base):
    __tablename__ = "analytic_data"
    __table_args__ = {"schema": "stock_market"}

    # Composite Primary Key for snapshot data
    stock_id = Column(
        Integer, ForeignKey("stock_market.metadata.stock_id"), primary_key=True
    )
    created_at = Column(DateTime, primary_key=True)
    retrieve_at = Column(DateTime)

    # Columns
    target_low_price = Column(Numeric)
    target_mean_price = Column(Numeric)
    target_high_price = Column(Numeric)
    recommendation_status = Column(String)
    average_analyst_rating = Column(String)
    number_of_analyst_opinions = Column(Integer)
    earnings_start_date = Column(DateTime)
    earnings_end_date = Column(DateTime)
    is_earnings_date_estimate = Column(Boolean)


class PriceData(Base):
    __tablename__ = "price_data"
    __table_args__ = {"schema": "stock_market"}

    # Composite Primary Key for time-series data
    stock_id = Column(
        Integer, ForeignKey("stock_market.metadata.stock_id"), primary_key=True
    )
    trade_date = Column(Date, primary_key=True)

    # Columns
    ticker = Column(String)
    open = Column(Numeric)
    high = Column(Numeric)
    low = Column(Numeric)
    close = Column(Numeric)
    volume = Column(BigInteger)
    dividends = Column(Numeric)
    stock_splits = Column(Numeric)


class IndicatorsData(Base):
    __tablename__ = "indicators_data"
    __table_args__ = {"schema": "stock_market"}

    stock_id = Column(
        Integer,
        ForeignKey("stock_market.metadata.stock_id"),
        primary_key=True,
    )
    indicator_date = Column(Date, primary_key=True)

    ticker = Column(String)
    sector = Column(String)
    open = Column(Numeric)
    high = Column(Numeric)
    low = Column(Numeric)
    close = Column(Numeric)
    returns = Column(Numeric)
    distance_moving_avg = Column(Numeric)
    momentum_5d = Column(Numeric)
    momentum_10d = Column(Numeric)
    momentum_20d = Column(Numeric)
    momentum_30d = Column(Numeric)
    momentum_60d = Column(Numeric)
    momentum_120d = Column(Numeric)
    momentum_250d = Column(Numeric)
    momentum_6_1 = Column(Numeric)
    momentum_12_1 = Column(Numeric)
    volatility_5d_ann = Column(Numeric)
    volatility_10d_ann = Column(Numeric)
    volatility_15d_ann = Column(Numeric)
    volatility_20d_ann = Column(Numeric)
    volatility_25d_ann = Column(Numeric)
    volatility_30d_ann = Column(Numeric)
    moving_avg_slope_3d = Column(Numeric)
    moving_avg_slope_5d = Column(Numeric)
    moving_avg_slope_10d = Column(Numeric)
    moving_avg_slope_15d = Column(Numeric)
    distance_high_52w = Column(Numeric)
    distance_low_52w = Column(Numeric)
    risk_adjusted_momentum = Column(Numeric)
    rank_momentum_20d = Column(Numeric)
    rank_momentum_60d = Column(Numeric)
    rank_momentum_120d = Column(Numeric)
    rank_momentum_12_1 = Column(Numeric)
    sector_rel_momentum_20d = Column(Numeric)
    sector_rel_momentum_120d = Column(Numeric)
    rank_volatility_20d_ann = Column(Numeric)
    rank_risk_adjusted_momentum = Column(Numeric)
    rank_distance_high_52w = Column(Numeric)
    trend_r2_60d = Column(Numeric)
    efficiency_ratio_20d = Column(Numeric)
    ma_alignment_score = Column(Integer)
    bollinger_pct_b_20 = Column(Numeric)
    gap = Column(Numeric)
    vol_ratio_5_20 = Column(Numeric)
    atr_14_normalized = Column(Numeric)
    daily_range_normalized = Column(Numeric)
    rsi_14 = Column(Numeric)
    stochastic_pct_k_14 = Column(Numeric)
    williams_r_14 = Column(Numeric)
    rolling_max_drawdown_60d = Column(Numeric)
    current_drawdown_from_rolling_max = Column(Numeric)
    realized_skewness_60d = Column(Numeric)
    realized_kurtosis_60d = Column(Numeric)
    sortino_60d = Column(Numeric)
    calmar_120d = Column(Numeric)
    ulcer_index_60d = Column(Numeric)
    candle_body_ratio = Column(Numeric)
    close_position_in_range = Column(Numeric)
    upper_shadow_ratio = Column(Numeric)
    lower_shadow_ratio = Column(Numeric)
    doji_flag = Column(Boolean)
    consecutive_up_days_streak = Column(Integer)
    breakout_flag_20d = Column(Boolean)


class InferenceMonitoring(Base):
    __tablename__ = "inference_monitoring"
    __table_args__ = {"schema": "stock_market"}

    stock_id = Column(
        Integer, ForeignKey("stock_market.metadata.stock_id"), primary_key=True
    )
    indicator_date = Column(Date, primary_key=True)
    ticker = Column(String)
    sector = Column(String)
    open = Column(Numeric)
    high = Column(Numeric)
    low = Column(Numeric)
    close = Column(Numeric)
    volume = Column(BigInteger)
    daily_return = Column(Numeric)
    momentum_5d = Column(Numeric)
    momentum_20d = Column(Numeric)
    volatility_20d_ann = Column(Numeric)
    status = Column(String, nullable=False)
    predicted_class = Column(Integer)
    predicted_label = Column(String)
    probability_lower = Column(Numeric)
    probability_neither = Column(Numeric)
    probability_upper = Column(Numeric)
    invalid_features = Column(JSONB, nullable=False)
    horizon = Column(Integer, nullable=False)
    model_path = Column(String, nullable=False)
    model_trained_at = Column(String)
    scored_at = Column(DateTime, nullable=False)
