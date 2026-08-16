-- Create the schema
CREATE SCHEMA IF NOT EXISTS stock_market;

-- 1. Create the parent Metadata table
CREATE TABLE stock_market.metadata (
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
CREATE TABLE stock_market.fundamental_data (
    stock_id INT,
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
    PRIMARY KEY (stock_id, retrieve_at),
    FOREIGN KEY (stock_id) REFERENCES stock_market.metadata (stock_id) ON DELETE CASCADE
);

-- 3. Create Dynamic Data table
CREATE TABLE stock_market.dynamic_data (
    stock_id INT,
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
    PRIMARY KEY (stock_id, retrieve_at),
    FOREIGN KEY (stock_id) REFERENCES stock_market.metadata (stock_id) ON DELETE CASCADE
);

-- 4. Create Analytic Data table
CREATE TABLE stock_market.analytic_data (
    stock_id INT,
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
    PRIMARY KEY (stock_id, retrieve_at),
    FOREIGN KEY (stock_id) REFERENCES stock_market.metadata (stock_id) ON DELETE CASCADE
);

-- 5. Create Price Data table
CREATE TABLE stock_market.price_data (
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