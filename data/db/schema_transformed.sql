-- Create the schema
CREATE SCHEMA IF NOT EXISTS stock_market;

-- 1. Create the parent Metadata table
CREATE TABLE IF NOT EXISTS stock_market.metadata (
    stock_id SERIAL PRIMARY KEY,
    ticker VARCHAR UNIQUE NOT NULL,
    symbol VARCHAR NOT NULL,
    company_name VARCHAR,
    sector VARCHAR,
    industry VARCHAR,
    exchange VARCHAR,
    market VARCHAR,
    full_time_employees INT,
    ceo_name VARCHAR,
    full_address VARCHAR,
    gmt_offset_hours NUMERIC,
    exchange_timezone_name VARCHAR,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- 2. Create Fundamental Data table
CREATE TABLE IF NOT EXISTS stock_market.fundamental_data (
    stock_id INT,
    created_at TIMESTAMP,
    retrieve_at TIMESTAMP,
    market_cap BIGINT,
    enterprise_value BIGINT,
    shares_outstanding BIGINT,
    trailing_pe NUMERIC,
    forward_pe NUMERIC,
    dividend_yield NUMERIC,
    payout_ratio NUMERIC,
    last_fiscal_year_end_date TIMESTAMP,
    most_recent_quarter_date TIMESTAMP,
    ex_dividend_date TIMESTAMP,
    last_dividend_date TIMESTAMP,
    total_cash BIGINT,
    total_debt BIGINT,
    free_cashflow BIGINT,
    operating_cashflow BIGINT,
    return_on_equity NUMERIC,
    PRIMARY KEY (stock_id, created_at),
    FOREIGN KEY (stock_id) REFERENCES stock_market.metadata (stock_id) ON DELETE CASCADE
);

-- 3. Create Dynamic Data table
CREATE TABLE IF NOT EXISTS stock_market.dynamic_data (
    stock_id INT,
    created_at TIMESTAMP,
    retrieve_at TIMESTAMP,
    current_price NUMERIC,
    previous_close NUMERIC,
    open NUMERIC,
    day_low NUMERIC,
    day_high NUMERIC,
    volume BIGINT,
    average_volume_10days BIGINT,
    regular_market_time_utc TIMESTAMP,
    first_trade_date_utc TIMESTAMP,
    fifty_two_week_low NUMERIC,
    fifty_two_week_high NUMERIC,
    fifty_two_week_change_percent NUMERIC,
    fifty_day_average NUMERIC,
    two_hundred_day_average NUMERIC,
    market_state VARCHAR,
    PRIMARY KEY (stock_id, created_at),
    FOREIGN KEY (stock_id) REFERENCES stock_market.metadata (stock_id) ON DELETE CASCADE
);

-- 4. Create Analytic Data table
CREATE TABLE IF NOT EXISTS stock_market.analytic_data (
    stock_id INT,
    created_at TIMESTAMP,
    retrieve_at TIMESTAMP,
    target_low_price NUMERIC,
    target_mean_price NUMERIC,
    target_high_price NUMERIC,
    recommendation_status VARCHAR,
    average_analyst_rating VARCHAR,
    number_of_analyst_opinions INT,
    earnings_start_date TIMESTAMP,
    earnings_end_date TIMESTAMP,
    is_earnings_date_estimate BOOLEAN,
    PRIMARY KEY (stock_id, created_at),
    FOREIGN KEY (stock_id) REFERENCES stock_market.metadata (stock_id) ON DELETE CASCADE
);

-- 5. Create Price Data table
CREATE TABLE IF NOT EXISTS stock_market.price_data (
    stock_id INT,
    trade_date DATE,
    ticker VARCHAR,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    dividends NUMERIC,
    stock_splits NUMERIC,
    PRIMARY KEY (stock_id, trade_date),
    FOREIGN KEY (stock_id) REFERENCES stock_market.metadata (stock_id) ON DELETE CASCADE
);

-- 6. Point-in-time indicator features used by analytics and production scoring
CREATE TABLE IF NOT EXISTS stock_market.indicators_data (
    stock_id INT,
    indicator_date DATE,
    ticker VARCHAR,
    sector VARCHAR,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    returns NUMERIC,
    distance_moving_avg NUMERIC,
    momentum_5d NUMERIC,
    momentum_10d NUMERIC,
    momentum_20d NUMERIC,
    momentum_30d NUMERIC,
    momentum_60d NUMERIC,
    momentum_120d NUMERIC,
    momentum_250d NUMERIC,
    momentum_6_1 NUMERIC,
    momentum_12_1 NUMERIC,
    volatility_5d_ann NUMERIC,
    volatility_10d_ann NUMERIC,
    volatility_15d_ann NUMERIC,
    volatility_20d_ann NUMERIC,
    volatility_25d_ann NUMERIC,
    volatility_30d_ann NUMERIC,
    moving_avg_slope_3d NUMERIC,
    moving_avg_slope_5d NUMERIC,
    moving_avg_slope_10d NUMERIC,
    moving_avg_slope_15d NUMERIC,
    distance_high_52w NUMERIC,
    distance_low_52w NUMERIC,
    risk_adjusted_momentum NUMERIC,
    rank_momentum_20d NUMERIC,
    rank_momentum_60d NUMERIC,
    rank_momentum_120d NUMERIC,
    rank_momentum_12_1 NUMERIC,
    sector_rel_momentum_20d NUMERIC,
    sector_rel_momentum_120d NUMERIC,
    rank_volatility_20d_ann NUMERIC,
    rank_risk_adjusted_momentum NUMERIC,
    rank_distance_high_52w NUMERIC,
    trend_r2_60d NUMERIC,
    efficiency_ratio_20d NUMERIC,
    ma_alignment_score INT,
    bollinger_pct_b_20 NUMERIC,
    gap NUMERIC,
    vol_ratio_5_20 NUMERIC,
    atr_14_normalized NUMERIC,
    daily_range_normalized NUMERIC,
    rsi_14 NUMERIC,
    stochastic_pct_k_14 NUMERIC,
    williams_r_14 NUMERIC,
    rolling_max_drawdown_60d NUMERIC,
    current_drawdown_from_rolling_max NUMERIC,
    realized_skewness_60d NUMERIC,
    realized_kurtosis_60d NUMERIC,
    sortino_60d NUMERIC,
    calmar_120d NUMERIC,
    ulcer_index_60d NUMERIC,
    candle_body_ratio NUMERIC,
    close_position_in_range NUMERIC,
    upper_shadow_ratio NUMERIC,
    lower_shadow_ratio NUMERIC,
    doji_flag BOOLEAN,
    consecutive_up_days_streak INT,
    breakout_flag_20d BOOLEAN,
    PRIMARY KEY (stock_id, indicator_date),
    FOREIGN KEY (stock_id) REFERENCES stock_market.metadata (stock_id) ON DELETE CASCADE
);

-- 7. Latest inference result per stock and indicator date for daily monitoring
CREATE TABLE IF NOT EXISTS stock_market.inference_monitoring (
    stock_id INT NOT NULL,
    indicator_date DATE NOT NULL,
    ticker VARCHAR,
    sector VARCHAR,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    daily_return NUMERIC,
    momentum_5d NUMERIC,
    momentum_20d NUMERIC,
    volatility_20d_ann NUMERIC,
    status VARCHAR NOT NULL CHECK (status IN ('predicted', 'rejected')),
    predicted_class INT,
    predicted_label VARCHAR,
    probability_lower NUMERIC,
    probability_neither NUMERIC,
    probability_upper NUMERIC,
    invalid_features JSONB NOT NULL,
    horizon INT NOT NULL,
    model_path VARCHAR NOT NULL,
    model_trained_at VARCHAR,
    scored_at TIMESTAMP NOT NULL,
    PRIMARY KEY (stock_id, indicator_date),
    FOREIGN KEY (stock_id) REFERENCES stock_market.metadata (stock_id) ON DELETE CASCADE,
    CHECK (
        (status = 'predicted' AND predicted_class IS NOT NULL AND predicted_class IN (0, 1, 2)
         AND predicted_label IS NOT NULL AND probability_lower IS NOT NULL
         AND probability_neither IS NOT NULL AND probability_upper IS NOT NULL)
        OR
        (status = 'rejected' AND predicted_class IS NULL
         AND predicted_label IS NULL AND probability_lower IS NULL
         AND probability_neither IS NULL AND probability_upper IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS inference_monitoring_date_idx
    ON stock_market.inference_monitoring (indicator_date);