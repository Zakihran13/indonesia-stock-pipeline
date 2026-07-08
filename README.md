# IDX Portfolio Pipeline

End-to-end ETL pipeline for Indonesian stock portfolio analytics.

## Project Structure

- `.github/workflows/daily_etl.yml`: CI/CD workflow and daily ETL schedule.
- `data/raw`: Raw market and trade CSV files.
- `data/processed`: Cleaned outputs and local SQLite database.
- `dags/pipeline_dag.py`: Optional Airflow DAG.
- `src`: Core extract-transform-load modules.
- `dashboard`: Streamlit dashboard app and reusable components.
- `tests`: Unit tests for transformation and API wrappers.

## Quickstart

1. Create and activate a Python environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Install Playwright browser runtime:
   `playwright install chromium`
4. Run extractor:
   `python -m src.extract.idx_api`
5. Run tests:
   `pytest`
6. Launch dashboard:
   `streamlit run dashboard/app.py`

## Environment Variables

Copy `.env.example` to `.env` and set:

- `IDX_STOCK_LIST_URL`
- `DATABASE_URL`
