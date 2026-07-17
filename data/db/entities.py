from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy import (
    Integer,
    String,
    Text,
    JSON,
    BigInteger,
    Numeric,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class StockMetadata(Base):
    __tablename__ = "metadata"
    __table_args__ = {"schema": "stock_market"}

    stock_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    symbol: Mapped[str] = mapped_column(String(50), index=True)

    long_name: Mapped[Optional[str]] = mapped_column(String(255))
    short_name: Mapped[Optional[str]] = mapped_column(String(255))
    exchange: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    full_exchange_name: Mapped[Optional[str]] = mapped_column(String(255))
    market: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    sector: Mapped[Optional[str]] = mapped_column(String(150), index=True)
    sector_key: Mapped[Optional[str]] = mapped_column(String(150))
    sector_disp: Mapped[Optional[str]] = mapped_column(String(150))
    industry: Mapped[Optional[str]] = mapped_column(String(150), index=True)
    industry_key: Mapped[Optional[str]] = mapped_column(String(150))
    industry_disp: Mapped[Optional[str]] = mapped_column(String(150))

    long_business_summary: Mapped[Optional[str]] = mapped_column(Text)
    full_time_employees: Mapped[Optional[int]] = mapped_column(Integer)

    # Using JSON type; type hinting can be mapped to list or dict
    company_officers: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    executive_team: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)

    address1: Mapped[Optional[str]] = mapped_column(String(255))
    address2: Mapped[Optional[str]] = mapped_column(String(255))
    city: Mapped[Optional[str]] = mapped_column(String(100))
    zip: Mapped[Optional[str]] = mapped_column(String(50))
    country: Mapped[Optional[str]] = mapped_column(String(100))
    phone: Mapped[Optional[str]] = mapped_column(String(50))
    fax: Mapped[Optional[str]] = mapped_column(String(50))
    website: Mapped[Optional[str]] = mapped_column(String(255))
    currency: Mapped[Optional[str]] = mapped_column(String(10))
    financial_currency: Mapped[Optional[str]] = mapped_column(String(10))
    language: Mapped[Optional[str]] = mapped_column(String(20))
    region: Mapped[Optional[str]] = mapped_column(String(50))

    quote_type: Mapped[Optional[str]] = mapped_column(String(50))
    type_disp: Mapped[Optional[str]] = mapped_column(String(50))
    message_board_id: Mapped[Optional[str]] = mapped_column(String(100))
    exchange_timezone_name: Mapped[Optional[str]] = mapped_column(String(100))
    exchange_timezone_short_name: Mapped[Optional[str]] = mapped_column(String(20))
    gmt_off_set_milliseconds: Mapped[Optional[int]] = mapped_column(BigInteger)

    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, server_default=func.current_timestamp()
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )

    # Optional: SQLAlchemy relationships to access child records easily
    dynamic_data = relationship(
        "DynamicData", back_populates="stock", cascade="all, delete-orphan"
    )
    fundamental_data = relationship(
        "FundamentalData", back_populates="stock", cascade="all, delete-orphan"
    )
    analytic_data = relationship(
        "AnalyticData", back_populates="stock", cascade="all, delete-orphan"
    )


class DynamicData(Base):
    __tablename__ = "dynamic_data"
    __table_args__ = (
        Index("idx_dynamic_retrieve_at", "retrieve_at", postgresql_using="brin"),
        {"schema": "stock_market"},
    )

    stock_id: Mapped[int] = mapped_column(
        ForeignKey("stock_market.metadata.stock_id", ondelete="CASCADE"),
        primary_key=True,
    )
    retrieve_at: Mapped[datetime] = mapped_column(DateTime, primary_key=True)

    current_price: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    previous_close: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    open: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    day_low: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    day_high: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    regular_market_previous_close: Mapped[Optional[float]] = mapped_column(
        Numeric(15, 4)
    )
    regular_market_open: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    regular_market_day_low: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    regular_market_day_high: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    regular_market_price: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    regular_market_change: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    regular_market_change_percent: Mapped[Optional[float]] = mapped_column(
        Numeric(8, 4)
    )

    volume: Mapped[Optional[int]] = mapped_column(BigInteger)
    regular_market_volume: Mapped[Optional[int]] = mapped_column(BigInteger)
    average_volume: Mapped[Optional[int]] = mapped_column(BigInteger)
    average_volume_10days: Mapped[Optional[int]] = mapped_column(BigInteger)
    average_daily_volume_10day: Mapped[Optional[int]] = mapped_column(BigInteger)
    average_daily_volume_3month: Mapped[Optional[int]] = mapped_column(BigInteger)

    bid: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    ask: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    bid_size: Mapped[Optional[int]] = mapped_column(Integer)
    ask_size: Mapped[Optional[int]] = mapped_column(Integer)

    market_state: Mapped[Optional[str]] = mapped_column(String(50))
    regular_market_time: Mapped[Optional[int]] = mapped_column(BigInteger)

    fifty_two_week_low: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    fifty_two_week_high: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    all_time_high: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    all_time_low: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    fifty_day_average: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    two_hundred_day_average: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))

    fifty_two_week_change: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    s_and_p_52_week_change: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    fifty_two_week_low_change: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    fifty_two_week_low_change_percent: Mapped[Optional[float]] = mapped_column(
        Numeric(8, 4)
    )
    fifty_two_week_range: Mapped[Optional[str]] = mapped_column(String(50))
    fifty_two_week_high_change: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    fifty_two_week_high_change_percent: Mapped[Optional[float]] = mapped_column(
        Numeric(8, 4)
    )
    fifty_two_week_change_percent: Mapped[Optional[float]] = mapped_column(
        Numeric(8, 4)
    )

    fifty_day_average_change: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    fifty_day_average_change_percent: Mapped[Optional[float]] = mapped_column(
        Numeric(8, 4)
    )
    two_hundred_day_average_change: Mapped[Optional[float]] = mapped_column(
        Numeric(15, 4)
    )
    two_hundred_day_average_change_percent: Mapped[Optional[float]] = mapped_column(
        Numeric(8, 4)
    )

    tradeable: Mapped[Optional[bool]] = mapped_column(Boolean)
    triggerable: Mapped[Optional[bool]] = mapped_column(Boolean)
    crypto_tradeable: Mapped[Optional[bool]] = mapped_column(Boolean)
    has_pre_post_market_data: Mapped[Optional[bool]] = mapped_column(Boolean)

    first_trade_date_milliseconds: Mapped[Optional[int]] = mapped_column(BigInteger)
    regular_market_day_range: Mapped[Optional[str]] = mapped_column(String(50))

    stock = relationship("StockMetadata", back_populates="dynamic_data")


class FundamentalData(Base):
    __tablename__ = "fundamental_data"
    __table_args__ = (
        Index("idx_fundamental_retrieve_at", "retrieve_at", postgresql_using="brin"),
        {"schema": "stock_market"},
    )

    stock_id: Mapped[int] = mapped_column(
        ForeignKey("stock_market.metadata.stock_id", ondelete="CASCADE"),
        primary_key=True,
    )
    retrieve_at: Mapped[datetime] = mapped_column(DateTime, primary_key=True)

    market_cap: Mapped[Optional[int]] = mapped_column(BigInteger)
    non_diluted_market_cap: Mapped[Optional[int]] = mapped_column(BigInteger)
    enterprise_value: Mapped[Optional[int]] = mapped_column(BigInteger)
    float_shares: Mapped[Optional[int]] = mapped_column(BigInteger)
    shares_outstanding: Mapped[Optional[int]] = mapped_column(BigInteger)
    implied_shares_outstanding: Mapped[Optional[int]] = mapped_column(BigInteger)

    profit_margins: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    book_value: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    price_to_book: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    trailing_pe: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    forward_pe: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    price_to_sales_trailing_12_months: Mapped[Optional[float]] = mapped_column(
        Numeric(10, 4)
    )
    peg_ratio: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    trailing_peg_ratio: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))

    net_income_to_common: Mapped[Optional[int]] = mapped_column(BigInteger)
    total_revenue: Mapped[Optional[int]] = mapped_column(BigInteger)
    revenue_per_share: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    ebitda: Mapped[Optional[int]] = mapped_column(BigInteger)
    enterprise_to_revenue: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    enterprise_to_ebitda: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))

    earnings_growth: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    revenue_growth: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    gross_profits: Mapped[Optional[int]] = mapped_column(BigInteger)
    gross_margins: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    ebitda_margins: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    operating_margins: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    return_on_assets: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    return_on_equity: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))

    total_cash: Mapped[Optional[int]] = mapped_column(BigInteger)
    total_cash_per_share: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    total_debt: Mapped[Optional[int]] = mapped_column(BigInteger)
    quick_ratio: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    current_ratio: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    free_cashflow: Mapped[Optional[int]] = mapped_column(BigInteger)
    operating_cashflow: Mapped[Optional[int]] = mapped_column(BigInteger)

    trailing_eps: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    forward_eps: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    eps_trailing_twelve_months: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    eps_forward: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    eps_current_year: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    price_eps_current_year: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))

    held_percent_insiders: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    held_percent_institutions: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))

    last_fiscal_year_end: Mapped[Optional[int]] = mapped_column(BigInteger)
    next_fiscal_year_end: Mapped[Optional[int]] = mapped_column(BigInteger)
    most_recent_quarter: Mapped[Optional[int]] = mapped_column(BigInteger)

    dividend_rate: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    dividend_yield: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    ex_dividend_date: Mapped[Optional[int]] = mapped_column(BigInteger)
    payout_ratio: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    five_year_avg_dividend_yield: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    trailing_annual_dividend_rate: Mapped[Optional[float]] = mapped_column(
        Numeric(15, 4)
    )
    trailing_annual_dividend_yield: Mapped[Optional[float]] = mapped_column(
        Numeric(8, 4)
    )
    last_dividend_value: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    last_dividend_date: Mapped[Optional[int]] = mapped_column(BigInteger)

    corporate_actions: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)

    stock = relationship("StockMetadata", back_populates="fundamental_data")


class AnalyticData(Base):
    __tablename__ = "analytic_data"
    __table_args__ = (
        Index("idx_analytic_retrieve_at", "retrieve_at", postgresql_using="brin"),
        {"schema": "stock_market"},
    )

    stock_id: Mapped[int] = mapped_column(
        ForeignKey("stock_market.metadata.stock_id", ondelete="CASCADE"),
        primary_key=True,
    )
    retrieve_at: Mapped[datetime] = mapped_column(DateTime, primary_key=True)

    target_high_price: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    target_low_price: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    target_mean_price: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))
    target_median_price: Mapped[Optional[float]] = mapped_column(Numeric(15, 4))

    recommendation_mean: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    recommendation_key: Mapped[Optional[str]] = mapped_column(String(50))
    number_of_analyst_opinions: Mapped[Optional[int]] = mapped_column(Integer)
    average_analyst_rating: Mapped[Optional[str]] = mapped_column(String(100))

    earnings_timestamp_start: Mapped[Optional[int]] = mapped_column(BigInteger)
    earnings_timestamp_end: Mapped[Optional[int]] = mapped_column(BigInteger)
    is_earnings_date_estimate: Mapped[Optional[bool]] = mapped_column(Boolean)

    stock = relationship("StockMetadata", back_populates="analytic_data")
