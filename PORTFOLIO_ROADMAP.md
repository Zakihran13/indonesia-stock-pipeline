# Portfolio roadmap

This roadmap describes the full maturity path of the project, from a first prototype to something that looks production-ready in a hiring review. It is written so you can explain the evolution of the system clearly in interviews.

## Phase 0 - Problem framing and scope
- Define the business problem clearly: Indonesian stock data as a public market dataset for analytics and modeling.
- Choose the audience for the project: recruiter, data engineer, data scientist, ML engineer, or MLOps reviewer.
- Decide the minimum useful workflow: ingest -> transform -> store -> query -> visualize -> model.
- Document what is in scope and what is intentionally excluded.

## Phase 1 - Prototype version
- Prove that the data source can be collected reliably.
- Build the first ingestion scripts for stock metadata and price data.
- Store the data locally in a simple and inspectable format.
- Show one notebook or small script that validates the raw data.

## Phase 2 - Structured pipeline version
- Normalize the codebase into clear extract, transform, and load layers.
- Define a relational schema for metadata, prices, and derived tables.
- Make the ingestion flow idempotent so reruns do not corrupt data.
- Add configuration management so the project runs on another machine without code changes.

## Phase 3 - Reliable data engineering version
- Add data validation and transformation checks.
- Add tests for core logic, especially parsing, cleaning, and DB writes.
- Add logging so pipeline runs are easy to trace.
- Add a reproducible local setup using Docker or a documented environment.
- Add a CI workflow that runs tests automatically.

## Phase 4 - Analytics and product version
- Build a small dashboard or reporting layer.
- Add useful aggregated tables or views for analysis.
- Show business-facing metrics, not only raw tables.
- Create a clear README that explains how to run the pipeline and how the data flows.

## Phase 5 - ML prototype version
- Choose one target use case, such as price movement classification, volatility estimation, or return forecasting.
- Build a baseline model using the ingested historical data.
- Track features, labels, metrics, and evaluation results.
- Save the model artifact and document the training assumptions.
- Compare the baseline to a naive benchmark.

## Phase 6 - ML engineering version
- Separate training, batch inference, and evaluation into distinct steps.
- Add feature generation that can be reused consistently.
- Make model outputs reproducible and versioned.
- Add experiment tracking or at least structured run logs.
- Add unit tests for feature engineering and inference behavior.

## Phase 7 - MLOps / production-ready version
- Add orchestration or scheduling for recurring runs.
- Add monitoring for pipeline health, data freshness, and basic model quality.
- Add retry, backfill, and failure handling for ingestion jobs.
- Add deployment documentation and rollback notes.
- Add environment separation for local, staging, and production-like runs.

## How to explain it in interviews
- For data engineering roles, emphasize the move from raw data collection to reliable schema design and repeatable pipelines.
- For data science roles, emphasize the dataset creation process, feature work, and model evaluation.
- For machine learning engineer roles, emphasize training/inference separation, artifact handling, and reproducibility.
- For MLOps roles, emphasize orchestration, observability, automated checks, and operational readiness.

## Scope guardrails
- Keep the project centered on one business domain.
- Avoid adding features that do not improve hiring clarity.
- Prefer reliability, reproducibility, and documentation over extra complexity.
