CREATE SCHEMA IF NOT EXISTS stock_market;

CREATE TABLE IF NOT EXISTS stock_market.metadata (
    stock_id SERIAL PRIMARY KEY,       
    ticker VARCHAR(50) UNIQUE NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    long_name VARCHAR(255),
    short_name VARCHAR(255),
    exchange VARCHAR(100),
    full_exchange_name VARCHAR(255),
    market VARCHAR(100),
    sector VARCHAR(150),
    sector_key VARCHAR(150),
    sector_disp VARCHAR(150),
    industry VARCHAR(150),
    industry_key VARCHAR(150),
    industry_disp VARCHAR(150),
    long_business_summary TEXT,  
    full_time_employees INTEGER,
    company_officers JSON, 
    executive_team JSON, 
    address1 VARCHAR(255),
    address2 VARCHAR(255),
    city VARCHAR(100),
    zip VARCHAR(50),
    country VARCHAR(100),
    phone VARCHAR(50),
    fax VARCHAR(50),
    website VARCHAR(255),
    currency VARCHAR(10),
    financial_currency VARCHAR(10),
    language VARCHAR(20),
    region VARCHAR(50),
    quote_type VARCHAR(50),
    type_disp VARCHAR(50),
    message_board_id VARCHAR(100),
    exchange_timezone_name VARCHAR(100),
    exchange_timezone_short_name VARCHAR(20),
    gmt_off_set_milliseconds BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_metadata_ticker ON stock_market.metadata (ticker);
CREATE INDEX IF NOT EXISTS idx_metadata_symbol ON stock_market.metadata (symbol);
CREATE INDEX IF NOT EXISTS idx_metadata_exchange ON stock_market.metadata (exchange);
CREATE INDEX IF NOT EXISTS idx_metadata_sector ON stock_market.metadata (sector);
CREATE INDEX IF NOT EXISTS idx_metadata_industry ON stock_market.metadata (industry);
CREATE INDEX IF NOT EXISTS idx_metadata_market ON stock_market.metadata (market);


CREATE TABLE IF NOT EXISTS stock_market.dynamic_data (
    stock_id INTEGER NOT NULL,
    retrieve_at TIMESTAMP NOT NULL,
    current_price NUMERIC(15, 4),
    previous_close NUMERIC(15, 4),
    open NUMERIC(15, 4),
    day_low NUMERIC(15, 4),
    day_high NUMERIC(15, 4),
    regular_market_previous_close NUMERIC(15, 4),
    regular_market_open NUMERIC(15, 4),
    regular_market_day_low NUMERIC(15, 4),
    regular_market_day_high NUMERIC(15, 4),
    regular_market_price NUMERIC(15, 4),
    regular_market_change NUMERIC(15, 4),
    regular_market_change_percent NUMERIC(8, 4),
    volume BIGINT,
    regular_market_volume BIGINT,
    average_volume BIGINT,
    average_volume_10days BIGINT,
    average_daily_volume_10day BIGINT,
    average_daily_volume_3month BIGINT,
    bid NUMERIC(15, 4),
    ask NUMERIC(15, 4),
    bid_size INTEGER,
    ask_size INTEGER,
    market_state VARCHAR(50),
    regular_market_time BIGINT,          
    fifty_two_week_low NUMERIC(15, 4),
    fifty_two_week_high NUMERIC(15, 4),
    all_time_high NUMERIC(15, 4),
    all_time_low NUMERIC(15, 4),
    fifty_day_average NUMERIC(15, 4),
    two_hundred_day_average NUMERIC(15, 4),
    fifty_two_week_change NUMERIC(8, 4),  
    s_and_p_52_week_change NUMERIC(8, 4),
    fifty_two_week_low_change NUMERIC(15, 4),
    fifty_two_week_low_change_percent NUMERIC(8, 4),
    fifty_two_week_range VARCHAR(50),
    fifty_two_week_high_change NUMERIC(15, 4),
    fifty_two_week_high_change_percent NUMERIC(8, 4),
    fifty_two_week_change_percent NUMERIC(8, 4),
    fifty_day_average_change NUMERIC(15, 4),
    fifty_day_average_change_percent NUMERIC(8, 4),
    two_hundred_day_average_change NUMERIC(15, 4),
    two_hundred_day_average_change_percent NUMERIC(8, 4),
    tradeable BOOLEAN,
    triggerable BOOLEAN,
    crypto_tradeable BOOLEAN,
    has_pre_post_market_data BOOLEAN,
    first_trade_date_milliseconds BIGINT,
    regular_market_day_range VARCHAR(50),

    -- Constraints
    PRIMARY KEY (stock_id, retrieve_at),
    FOREIGN KEY (stock_id) REFERENCES stock_market.metadata(stock_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_dynamic_data_retrieve_at 
    ON stock_market.dynamic_data USING BRIN (retrieve_at);


CREATE TABLE IF NOT EXISTS stock_market.fundamental_data (
    stock_id INTEGER NOT NULL,
    retrieve_at TIMESTAMP NOT NULL,
    
    -- Valuation & Market Size
    market_cap BIGINT,
    non_diluted_market_cap BIGINT,
    enterprise_value BIGINT,
    float_shares BIGINT,
    shares_outstanding BIGINT,
    implied_shares_outstanding BIGINT,
    
    -- Price Ratios & Margins
    profit_margins NUMERIC(8, 4),          
    book_value NUMERIC(15, 4),
    price_to_book NUMERIC(10, 4),
    trailing_pe NUMERIC(10, 4),
    forward_pe NUMERIC(10, 4),
    price_to_sales_trailing_12_months NUMERIC(10, 4),
    peg_ratio NUMERIC(10, 4),
    trailing_peg_ratio NUMERIC(10, 4),
    
    -- Income & Revenue Metrics
    net_income_to_common BIGINT,
    total_revenue BIGINT,
    revenue_per_share NUMERIC(15, 4),
    ebitda BIGINT,
    enterprise_to_revenue NUMERIC(10, 4),
    enterprise_to_ebitda NUMERIC(10, 4),
    
    -- Growth & Efficiency
    earnings_growth NUMERIC(8, 4),
    revenue_growth NUMERIC(8, 4),
    gross_profits BIGINT,
    gross_margins NUMERIC(8, 4),
    ebitda_margins NUMERIC(8, 4),
    operating_margins NUMERIC(8, 4),
    return_on_assets NUMERIC(8, 4),
    return_on_equity NUMERIC(8, 4),
    
    -- Balance Sheet Health & Cash Flow
    total_cash BIGINT,
    total_cash_per_share NUMERIC(15, 4),
    total_debt BIGINT,
    quick_ratio NUMERIC(10, 4),
    current_ratio NUMERIC(10, 4),
    free_cashflow BIGINT,
    operating_cashflow BIGINT,
    
    -- EPS Metrics
    trailing_eps NUMERIC(10, 4),
    forward_eps NUMERIC(10, 4),
    eps_trailing_twelve_months NUMERIC(10, 4),
    eps_forward NUMERIC(10, 4),
    eps_current_year NUMERIC(10, 4),
    price_eps_current_year NUMERIC(10, 4),
    
    -- Ownership Structure
    held_percent_insiders NUMERIC(8, 4),
    held_percent_institutions NUMERIC(8, 4),
    
    -- Fiscal Timestamps (Unix Epoch formats from Yahoo Finance)
    last_fiscal_year_end BIGINT,
    next_fiscal_year_end BIGINT,
    most_recent_quarter BIGINT,
    
    -- Dividend Profiles
    dividend_rate NUMERIC(15, 4),
    dividend_yield NUMERIC(8, 4),
    ex_dividend_date BIGINT,
    payout_ratio NUMERIC(8, 4),
    five_year_avg_dividend_yield NUMERIC(8, 4),
    trailing_annual_dividend_rate NUMERIC(15, 4),
    trailing_annual_dividend_yield NUMERIC(8, 4),
    last_dividend_value NUMERIC(15, 4),
    last_dividend_date BIGINT,
    
    -- Miscellaneous / Complex Metadata
    corporate_actions JSON,                

    -- Constraints
    PRIMARY KEY (stock_id, retrieve_at),
    FOREIGN KEY (stock_id) REFERENCES stock_market.metadata(stock_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_fundamental_data_retrieve_at 
    ON stock_market.fundamental_data USING BRIN (retrieve_at);


CREATE TABLE IF NOT EXISTS stock_market.analytic_data (
    stock_id INTEGER NOT NULL,
    retrieve_at TIMESTAMP NOT NULL,
    
    -- Price Targets (Forecasts for the next 12 months)
    target_high_price NUMERIC(15, 4),
    target_low_price NUMERIC(15, 4),
    target_mean_price NUMERIC(15, 4),
    target_median_price NUMERIC(15, 4),
    
    -- Analyst Sentiment & Coverage Scale
    recommendation_mean NUMERIC(5, 2),
    recommendation_key VARCHAR(50),
    number_of_analyst_opinions INTEGER,
    average_analyst_rating VARCHAR(100),
    
    -- Earnings Calendar Forecasts
    earnings_timestamp_start BIGINT,
    earnings_timestamp_end BIGINT,
    is_earnings_date_estimate BOOLEAN,
    
    -- Constraints
    PRIMARY KEY (stock_id, retrieve_at),
    FOREIGN KEY (stock_id) REFERENCES stock_market.metadata(stock_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_analytic_data_retrieve_at 
    ON stock_market.analytic_data USING BRIN (retrieve_at);
