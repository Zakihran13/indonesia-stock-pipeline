CREATE SCHEMA IF NOT EXISTS stock_market_clean;

CREATE
OR REPLACE VIEW stock_market_clean.metadata AS
SELECT
    stock_id,
    ticker,
    symbol,
    COALESCE(short_name, long_name) AS company_name,
    sector_disp AS sector,
    industry_disp AS industry,
    exchange,
    market,
    full_time_employees,
    (
        SELECT
            officer ->> 'name'
        FROM
            json_array_elements(company_officers) AS officer
        WHERE
            officer ->> 'title' ILIKE '%CEO%'
        LIMIT
            1
    ) AS ceo_name,
    CONCAT_WS(', ', address1, city, country) AS full_address,
    (gmt_off_set_milliseconds / 3600000.0) AS gmt_offset_hours,
    exchange_timezone_name,
    created_at AT TIME ZONE 'UTC' AS created_at_utc
FROM
    stock_market.metadata;

CREATE
OR REPLACE VIEW stock_market_clean.fundamental_data AS
SELECT
    stock_id,
    retrieve_at AT TIME ZONE 'UTC' AS retrieve_at_utc,
    market_cap,
    enterprise_value,
    shares_outstanding,
    trailing_pe,
    forward_pe,
    COALESCE(dividend_yield, 0) AS dividend_yield,
    COALESCE(payout_ratio, 0) AS payout_ratio,
    TO_TIMESTAMP(last_fiscal_year_end) AS last_fiscal_year_end_date,
    TO_TIMESTAMP(most_recent_quarter) AS most_recent_quarter_date,
    TO_TIMESTAMP(ex_dividend_date) AS ex_dividend_date,
    TO_TIMESTAMP(last_dividend_date) AS last_dividend_date,
    total_cash,
    total_debt,
    free_cashflow,
    operating_cashflow,
    return_on_equity
FROM
    stock_market.fundamental_data;

CREATE
OR REPLACE VIEW stock_market_clean.dynamic_data AS
SELECT
    stock_id,
    retrieve_at AT TIME ZONE 'UTC' AS retrieve_at_utc,
    current_price,
    previous_close,
    open,
    day_low,
    day_high,
    volume,
    average_volume_10days,
    TO_TIMESTAMP(regular_market_time) AS regular_market_time_utc,
    TO_TIMESTAMP(first_trade_date_milliseconds / 1000) AS first_trade_date_utc,
    fifty_two_week_low,
    fifty_two_week_high,
    fifty_two_week_change_percent,
    fifty_day_average,
    two_hundred_day_average,
    market_state
FROM
    stock_market.dynamic_data;

CREATE
OR REPLACE VIEW stock_market_clean.analytic_data AS
SELECT
    stock_id,
    retrieve_at AT TIME ZONE 'UTC' AS retrieve_at_utc,
    target_low_price,
    target_mean_price,
    target_high_price,
    UPPER(recommendation_key) AS recommendation_status,
    average_analyst_rating,
    number_of_analyst_opinions,
    TO_TIMESTAMP(earnings_timestamp_start) AS earnings_start_date,
    TO_TIMESTAMP(earnings_timestamp_end) AS earnings_end_date,
    is_earnings_date_estimate
FROM
    stock_market.analytic_data;

CREATE
OR REPLACE VIEW stock_market_clean.price_data AS
SELECT
    stock_id,
    ticker,
    DATE(date) AS trade_date,
    open,
    high,
    low,
    close,
    volume,
    COALESCE(dividends, 0) AS dividends,
    COALESCE(stock_splits, 0) AS stock_splits
FROM
    stock_market.price_data;