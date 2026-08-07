# Indonesia Stock Pipeline

This repository is a portfolio project for Data Engineering, Data Science, and MLOps-focused roles. It demonstrates how to build a practical data product around public market data: ingesting financial data, transforming it into a structured schema, loading it into a relational database, and preparing the foundation for analytics and model-based workflows.

## Portfolio story

The goal of this project is to show that I can move from raw external data to a reliable, queryable data layer and then extend it into analytics and ML-ready workflows. The project is intentionally focused on one domain—Indonesian stock data—so it is easier to explain in interviews and easier to evolve into a strong hiring portfolio.

## What this project demonstrates

- Data ingestion from public sources using Python and pandas
- Structured data modeling and relational loading with SQLAlchemy
- Async processing patterns for external API work
- A repeatable project layout suitable for data engineering and MLOps discussions
- A foundation for future work in forecasting, analytics, and model deployment

## Current repository structure

- `stock_market/extract/`: ingestion scripts for metadata, prices, and related market data
- `data/db/`: database schema, SQL statements, and connection helpers
- `stock_market/analytics/`: notebooks and analysis work
- `dashboard/`: dashboard-related components and app entrypoints
- `tests/`: quality checks and future validation coverage

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Create a local environment file from `.env.example` and fill in the required values.
4. Initialize the database schema:
   `python data/db/run_db.py`
5. Run the metadata ingestion script:
   `python stock_market/extract/metadata_ingestion.py`
6. Run the price ingestion script:
   `python stock_market/extract/daily_price_data_ingestion.py`

## Environment variables

Copy `.env.example` to `.env` and set the values required for your local environment.

Typical variables include:

- `POSTGRESQL_DB_USER`
- `POSTGRESQL_DB_PASSWORD`
- `POSTGRESQL_DB_HOST`
- `POSTGRESQL_DB_PORT`
- `POSTGRESQL_DB_NAME`
- `STOCK_LIST_PATH`

## Hiring-focused roadmap

This project is being shaped into a portfolio-ready example for roles such as:

- Data Engineer
- Data Scientist
- Machine Learning Engineer
- AI Engineer
- MLOps / Machine Learning Operations Engineer

The near-term plan is to strengthen the project with:

- a clearer end-to-end workflow
- stronger tests and validation
- a simple ML baseline or forecasting use case
- better deployment and observability practices

See `PORTFOLIO_ROADMAP.md` for the step-by-step plan.
