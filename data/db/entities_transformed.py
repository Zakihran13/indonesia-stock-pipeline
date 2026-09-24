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
    volume = Column(BigInteger)
    moving_average_30d = Column(Numeric)
    momentum_30d = Column(Numeric)
    daily_return = Column(Numeric)
    volatility_5d = Column(Numeric)
    volatility_20d = Column(Numeric)
    average_volume_20d = Column(Numeric)
    free_cashflow_to_market_cap = Column(Numeric)
    target_price_deviation = Column(Numeric)
    recommendation_score = Column(Numeric)
    recommendation_score_change = Column(Numeric)
    days_until_earnings = Column(Numeric)
    earnings_within_7d = Column(Boolean)
    return_on_equity = Column(Numeric)
    payout_ratio = Column(Numeric)
    total_debt = Column(BigInteger)
    average_analyst_rating = Column(String)
