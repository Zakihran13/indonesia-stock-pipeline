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

## Docker services

Start PostgreSQL, MongoDB, Grafana, and MLflow with:

```bash
docker compose up -d --build
```

The MLflow UI and tracking server are available at `http://localhost:5000` by
default (`MLFLOW_PORT` changes the host port). Run metadata is stored in
PostgreSQL and artifacts persist in the `mlflow_artifacts` Docker volume. Local
Python processes should use `MLFLOW_TRACKING_URI=http://localhost:5000`; other
Compose services should use `http://mlflow:5000`.

The MLflow backend URI is assembled from the PostgreSQL settings in `.env`.
Percent-encode reserved characters in `POSTGRESQL_DB_USER` or
`POSTGRESQL_DB_PASSWORD` because they are embedded in a database URL.

Typical variables include:

- `POSTGRESQL_DB_USER`
- `POSTGRESQL_DB_PASSWORD`
- `POSTGRESQL_DB_HOST`
- `POSTGRESQL_DB_PORT`
- `POSTGRESQL_DB_NAME`
- `STOCK_LIST_PATH`

## Indicator materialization

`materialize_indicators` processes price histories in batches of 50 stock IDs by
default. Pass `batch_size` to adjust the maximum number of stocks held in memory
at once, for example:

```python
await materialize_indicators(start_date=start_date, batch_size=25)
```

Each batch still includes the price history needed for rolling calculations.
The run stages cross-sectional inputs in a PostgreSQL temporary table and updates
ranks and sector-relative momentum after all batches. Without `stock_ids`, all
stocks with prices are calculated. When `stock_ids` is specified, only those
stocks are calculated and written; cross-sectional comparisons use previously
materialized indicator rows for other stocks on the same dates. Missing peers
are excluded, and stale peer rows produce stale comparisons; a single available
stock has a percentile rank of 1 and sector-relative momentum of 0 when defined.
Run a full materialization first when complete, current comparisons are needed.
Existing non-target ranks are not recomputed. All writes remain in one
transaction; large backfills may require substantial PostgreSQL temporary space
and will hold the transaction open until the run completes.
Metadata is joined by `stock_id` without a date filter: the current sector is
assumed to apply to all historical prices. Sector-relative indicators are not
point-in-time accurate if a stock's sector has changed.

## Batch model inference

Install the project's Poetry dependencies and configure the PostgreSQL
environment variables above. Materialize indicators before inference:

```bash
poetry run python -m stock_market.analytics.inference.inference \
  --output predictions.json
```

The flow loads the model path, ordered features, label names, and historical
validation/test metrics from `stock_market/analytics/configs/inference.yaml`.
It verifies the saved model's feature names and early-stopping iteration against
that configuration and runs on CPU, even if training used CUDA. Publish the
model and config from the same training run together; replacing either while
inference is loading can produce an inconsistent deployment.

By default, the flow selects the latest indicator date on or before today
across the whole database, then predicts available stocks on that date. It does
not backfill older rows for stocks missing from the snapshot. Use `--as-of
YYYY-MM-DD` for an exact date and `--stock-ids 1 2` to restrict the result;
explicitly requested stocks missing from the snapshot cause an error. Snapshot
freshness depends on indicator materialization.

For a historical minibatch backfill, specify both inclusive boundaries:

```bash
poetry run python -m stock_market.analytics.inference.inference \
  --start-date 2024-01-01 --end-date 2024-01-31 --output summary.json
```

The flow loads the model once and processes one available global indicator date
at a time in chronological order. Dates with no snapshot (such as weekends) are
skipped; a range with no snapshots fails. The JSON contains per-date and total
prediction/rejection counts, not full row details; query the monitoring table
for those. `--as-of` cannot be combined with a range. With `--stock-ids`, every
requested stock must exist on each processed date, as in single-date mode.
Each date commits separately, so a later failure leaves earlier dates committed;
the progress log identifies committed dates and rerunning is safe. Dates where
all rows are rejected are saved and processing continues; if the entire range
has no predictions, the command fails after saving the rejection details.
Backfills use the currently configured model, not the model available on each
historical date; reruns replace existing results for the same stock and date.

The JSON contains per-stock class probabilities and predictions, the label
horizon, model metadata, and the saved **training holdout** metrics. These
historical metrics are not live performance measures or confidence thresholds;
the probabilities are model outputs and are not guaranteed to be calibrated.
Rows with missing, nonnumeric, or non-finite selected features appear in
`rejected_rows` instead of receiving predictions. If all rows are rejected,
the command fails. Output defaults to stdout and errors go to stderr.

Inference also writes `stock_market.inference_monitoring` in PostgreSQL. Apply
the updated schema first with `poetry run python -m data.db.run_db_transformed`
(using the project's PostgreSQL environment settings). The table has one row
per stock and indicator date; a rerun replaces the result for processed stocks
only, including transitions between predicted and rejected. The write is
transactional, and a database failure fails the run. Even if every row is
rejected, rejection details are written before the command fails. Schedule the
inference command after indicator materialization to refresh the dashboard;
creating the table alone does not schedule daily runs.

For dashboards, the table includes status, class probabilities or invalid
features, model path/training timestamp, horizon, UTC scoring time, OHLC,
volume, sector, daily return, 5- and 20-day momentum, and annualized 20-day
volatility. OHLC and indicators come from the materialized indicator snapshot;
volume comes from the same-date price row and can be NULL. `daily_return` is
the fractional close-to-previous-close change (0.02 means 2%); momentum and
volatility are also fractions. Missing market values remain NULL. The close is
the snapshot close, not a live quote. Scoring time uses a UTC timestamp without
timezone. Model probabilities describe barrier classes, not predicted returns
or realized performance. Historical training metrics remain in the JSON/config.

Example monitoring query:

```sql
SELECT indicator_date, ticker, close, daily_return, volume, status,
       predicted_label, probability_upper, invalid_features, scored_at
FROM stock_market.inference_monitoring
WHERE indicator_date = (SELECT max(indicator_date) FROM stock_market.inference_monitoring)
ORDER BY probability_upper DESC NULLS LAST;
```

The same flow can be called from Python:

```python
from stock_market.analytics.inference.inference import run_inference

result = await run_inference(as_of="2024-01-02", stock_ids=[1, 2])
summary = await run_inference(start_date="2024-01-01", end_date="2024-01-31")
```

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

For the proposed model lifecycle, indicator governance, and drift monitoring
architecture, see the [outperformer analytics implementation plan](stock_market/analytics/OUTPERFORMER_IMPLEMENTATION_PLAN.md).
