# Outperformer analytics: implementation plan

Status: proposed design; not yet implemented. This plan uses the existing
ingestion flows, MLflow for model management, and Grafana for monitoring.

## 1. Goal and decisions

Produce a daily, reproducible score for each eligible Indonesian stock after
the trading session. The research target is **not benchmark-relative
outperformance**: it predicts which price barrier is touched first within the
next 10 observed trading bars for that stock. Treat the product as a
first-passage signal, not a guaranteed return or investment recommendation.
If benchmark-relative outperformance is required, define a separate target and
model version rather than silently changing the meaning of this one.

Recommended first release:

- Keep the notebook's three classes: `0` lower, `1` neither, `2` upper. Rank
  stocks by `P(upper)` and publish all three probabilities; a prediction is
  not a buy/sell recommendation.
- Run batch inference after validated price and indicator data are ready.
  Avoid an online serving system and a feature-store service until needed.
- Store indicators and monitoring data in PostgreSQL, model artifacts and
  lineage in MLflow, and dashboards and alerts in the existing Grafana.
- Start with manual promotion and drift-triggered investigation. Retraining
  creates a challenger; it must never automatically replace the champion.

The horizon, barrier levels, execution timing, stock universe, threshold or
top-K policy, baseline, acceptance margins, and owners must be approved
before production. Values below marked **initial proposal** are operating
defaults to calibrate, not proven thresholds.

## 2. Current state and research findings

| Area | Evidence today | Implication |
| --- | --- | --- |
| Research | [CUDA notebook](./outperformer_explore_cuda.ipynb) calculates 21 numeric indicators plus `sector`, selects features with SHAP-RFE, and trains a three-class XGBoost model. | Extract reusable, tested functions; the notebook is research evidence, not the production pipeline. |
| Target | Entry is the close at date `t`. Within the next `HORIZON = 10` stock bars, a high at or above `close(t) * 1.05` is upper and a low at or below `close(t) * 0.95` is lower. Neither is class 1; a same-bar double touch resolves to lower. Incomplete/invalid paths remain unknown unless a barrier was already hit. | Version these exact rules, including tie handling, valid-price policy, corporate-action treatment, and event-end timestamp. No data after `t` may enter features. |
| Existing classifier | [outperformer_classifier.py](./outperformer_classifier.py) is a different binary random forest: 30-bar forward return >= 5%, 21 numeric features, local joblib artifact and 0.5 cutoff. | Do not deploy it as the notebook model or reuse its probability semantics. Preserve it as a separate baseline/legacy path until an explicit migration. |
| Ingestion and indicators | [pipeline.py](../../dags/pipeline.py) emits `indicators_transformed` after daily stock and price transforms. Its daily raw DAG is manually scheduled; historical indicator materialization is commented out. [indicators.py](../transform/indicators.py) defaults to today and fetches 400 calendar days of warm-up. | Add a calendar-aware trigger and a validated historical backfill; do not assume backfill emits the indicator asset or 400 calendar days always contain 252 stock bars. |
| Storage and tools | [schema_transformed.sql](../../data/db/schema_transformed.sql) contains price and indicator tables. [docker-compose.yml](../../docker-compose.yml) already includes PostgreSQL, MongoDB and Grafana. MLflow is not configured in the project dependencies or compose file. | Add MLflow and observability tables as future work; Grafana can query PostgreSQL directly. |
| Research results | One saved feature-selection result contains 20 features and reports macro F1 around 0.518. Saved later evaluation cells report balanced accuracy around 0.44 and class-2 recall around 0.14; different saved cells report slightly different results. | These are historical notebook outputs, not verified or reproducible acceptance results. Rerun cleanly and record dataset, versions and splits before comparison. |

Important research-to-production gaps:

1. SHAP-RFE uses shuffled `StratifiedKFold`; hyperparameter search also uses
   ordinary cross-validation. Both can mix past and future observations.
   The notebook's `Timedelta(days=HORIZON)` gap is calendar days, not 10
   stock trading bars. Its `purged_time_split` expects `label_end`, which is
   not created in the displayed training data. Replace these with
   event-end-aware chronological folds throughout selection, tuning and
   evaluation.
2. The notebook reads a current metadata sector mapping, while
   `indicators_data` stores a sector value and the indicator builder merges
   metadata using its `created_at`. The current metadata table is not a
   historical sector snapshot. Verify actual availability and avoid
   pretending it is point-in-time history; exclude `sector` if trustworthy
   as-of values cannot be reconstructed, or create a historical source.
3. Notebook preprocessing drops rows with any missing values. Production must
   explicitly define eligibility, warm-up, missing/unknown sector, numeric
   invalid values, and any fitted imputation. Scoring must use the exact same
   transformations and ordered features as training.
4. The loader fetches future bars using a calendar-day buffer and later
   requests a different start date without refetching earlier data. Define
   data windows from available trading bars, verify actual coverage, and
   record effective dates rather than trusting requested dates.
5. Ingestion code includes logged-and-swallowed batch exceptions and a
   500-ticker limit. A successful Airflow asset event alone is therefore not
   proof of completeness. Add explicit upstream failure propagation and
   reconciliation gates before scoring.
6. Confirm yfinance adjustment settings and consistent OHLC adjustment across
   history, splits and dividends. Historical universe membership,
   delistings, suspensions, and data revisions can introduce survivorship or
   availability bias; document limitations and use point-in-time snapshots
   where available.

## 3. Proposed architecture

```text
daily / historical ingestion (MongoDB raw -> PostgreSQL price_data)
    -> validated, as-of indicator materialization -> indicators_data
    -> quality + expected-session/universe gate
    -> resolve MLflow @champion once -> batch score -> prediction revisions
    -> quality/drift aggregates -> PostgreSQL -> Grafana

price bars + frozen features + label contract -> mature outcomes
    -> realized performance aggregates -> PostgreSQL -> Grafana

versioned dataset snapshot + purged time folds -> selection/tuning/training
    -> MLflow experiment, artifacts and candidate model version
    -> holdout + backtest + shadow review -> manual @champion promotion
```

### Daily flow

1. Use the existing Airflow asset as a dependency, **plus** a readiness gate
   for the requested exchange session: expected universe, successfully
   ingested bars, duplicates, freshness, indicator coverage and warm-up.
   Reconcile upstream ticker-level failures; stop and alert on failed gates.
2. Resolve a specific registered MLflow model version from `@champion` once
   per run. Load the packaged preprocessing, feature contract and model as
   one unit; validate signature, types, category mapping and feature version.
   Fail closed if the alias, artifact, required inputs or contract is invalid.
3. Score only the agreed eligible universe for the session. Record excluded
   tickers and reasons, not just successful rows. Do not silently use the
   previous session's indicators as today's.
4. Write predictions and run status transactionally, then emit the scoring
   completion signal only after validation and commit. Run quality and drift
   calculations even if scoring fails when possible; alert on missing runs
   separately so a stale dashboard cannot look healthy.
5. Use exchange calendar/session dates and `Asia/Jakarta` for scheduling;
   store run timestamps in UTC. Holidays and suspended stocks are not
   necessarily ingestion failures. No same-close trading assumption: scores
   use final close/high/low and become available only after ingestion.

### Historical flow and replay

Backfill prices and indicators from explicit session ranges, including enough
pre-start stock bars for the longest lookback (252 prior bars for the
52-week extrema). Verify per-stock history, not just an assumed calendar
buffer. Materialize historical labels only when sufficient future bars exist
or a valid barrier hit is observable; record actual event end and data
availability. Replay predictions using the model version and frozen input
snapshot specified for the replay, clearly marking research/backfill runs
versus predictions actually produced at the time. Recompute affected
features, labels and reports when corrected source data arrive; retain prior
revisions for audit.

## 4. Data and output contracts

Implement migrations and constraints rather than modifying the existing
tables blindly. Suggested new logical entities:

| Entity | Minimum fields and rules |
| --- | --- |
| `analytics_runs` | `run_id`, run type, session date, status, start/end timestamps, input snapshot/watermark, expected/processed/excluded counts, model version, feature/label contract versions, error details. |
| `outperformer_predictions` | `run_id`, `stock_id`, decision session, `scored_at`, resolved model name/version and MLflow run ID, feature contract/snapshot ID, `prob_lower`, `prob_neither`, `prob_upper`, predicted class, rank and optional policy version. Probabilities are finite, in [0,1] and sum to 1 within tolerance. |
| `outperformer_outcomes` | prediction reference, label version, entry close, upper/lower barriers, event end, outcome class, source snapshot/revision, status and maturity/availability timestamps. Pending, invalid and mature are distinct; never encode missing labels as class 1. |
| `analytics_quality_metrics`, `analytics_drift_metrics`, `analytics_performance_metrics` | computation window/session, metric name/value, sample size, model and feature versions, baseline ID, run ID and applicable segment. Store alertable aggregate time series, not only logs. |
| `feature_snapshot` or immutable snapshot artifacts | Input values or a durable content-addressed dataset reference, schema and source watermarks sufficient to reproduce training and each published score. |

Make `(run_id, stock_id, decision_session)` unique for retry-safe writes. A
rerun of the *same* run must not duplicate results; a deliberate recompute,
new model, or source correction creates a new revision. Publish an explicit
current-result view using a controlled promotion rule, not arbitrary
last-write-wins. Keep original prediction-time inputs and results immutable
for honest performance and rollback. Restrict PostgreSQL/Grafana access to
the necessary read/write roles and never place credentials in the repository.

## 5. Indicator and feature governance

### Recommended design

Keep the full **candidate indicator catalog** separate from each model's
**selected ordered feature set**. Maintain the catalog and definitions in
version-controlled configuration/code; MLflow stores an immutable copy and
hash with every trained model. `indicators_data` is a materialized feature
source, not the source of truth for which features a model consumes.

Each feature contract should include:

- Name, description, formula and source columns, unit/dtype, as-of and
  availability rules, lookback in stock bars, valid range and null policy.
- Feature implementation/schema version and, for sector, the categorical
  mapping, unknown-category handling and mapping version.
- Ordered model inputs, fitted preprocessing, training selection method,
  training date range, baseline statistics and dataset fingerprint.

The saved SHAP-RFE result selected these 20 inputs, in this order:
`volatility_30d_ann`, `distance_high_52w`, `distance_low_52w`, `returns`,
`volatility_10d_ann`, `volatility_20d_ann`, `momentum_120d`,
`volatility_5d_ann`, `volatility_25d_ann`, `momentum_5d`,
`volatility_15d_ann`, `sector`, `distance_moving_avg`, `momentum_250d`,
`moving_avg_slope_15d`, `moving_avg_slope_3d`, `momentum_10d`,
`momentum_20d`, `moving_avg_slope_10d`, `moving_avg_slope_5d`. It excluded
`momentum_30d` and `risk_adjusted_momentum` from the 22 candidates.
This is a **research snapshot**, not an approved production list: the
selection used shuffled folds and sector's historical availability is
uncertain. Repeat selection with point-in-time data and purged time folds.

### Change policy

1. Never silently change a selected input, formula, order, dtype, category
   map or preprocessing for an existing model version. Formula/semantic
   changes require a new feature contract and a compatible trained model.
2. Maintain old feature logic and model artifacts long enough to score or
   reproduce the champion during a migration. Backfill changed features into
   versioned storage/snapshots; the current `(stock_id, indicator_date)` key
   cannot preserve parallel formula versions by itself.
3. For each training run, choose features only inside its training folds.
   Compare the full candidate set, the previous selection and a newly
   selected set with the same purged folds, including a stability analysis
   across periods and correlated feature groups. Record selection scores,
   importance and decisions in MLflow; SHAP importance is not causation.
4. Promote a changed feature set only as part of a candidate model that meets
   the same untouched holdout and shadow gates. Version the contract together
   with model and monitoring baseline; do not update a live selection merely
   because a feature drifts.

## 6. Training, evaluation and MLflow lifecycle

1. Create a reproducible snapshot with source revision, eligible historical
   universe, indicator contract, label contract, code commit, environment,
   seeds, decision dates and actual coverage. Separate training,
   validation/model-selection and final test periods chronologically across
   all stocks.
2. Implement expanding or rolling **date-based, purged** folds. Purge any
   training event whose actual label end reaches the next fold; account for
   overlapping event windows and use an embargo where applicable. Keep all
   stocks on the same side of each date boundary. Fit selection, imputation
   and encoding using training folds only; use nested temporal validation
   for feature and parameter selection. Keep the final test untouched.
3. Compare XGBoost against simple majority/prevalence, momentum/ranking and
   the existing binary classifier only where targets are comparable. Report
   macro F1, balanced accuracy, per-class precision/recall, class-2
   precision/recall, confusion matrix, multiclass log loss and calibration.
   For ranking, report precision@K/lift against prevalence and forward
   realized outcomes, with sample counts and uncertainty by time and sector.
   Define K, threshold and required improvements before the test.
4. Evaluate a realistic walk-forward strategy separately: signal availability
   after close, executable entry assumption such as next open, costs,
   liquidity, slippage, suspensions, turnover and market benchmark. The
   close-based research barrier label is not itself executable P&L.
5. Log to an MLflow experiment: model and preprocessing with signature,
   class order and input example; feature and label contracts; dataset
   snapshot reference/hash; splits and event-end purge; code/environment;
   training parameters, selection report, baselines, metrics, confusion
   matrix, calibration and backtest reports. Persist model artifacts in
   durable shared storage accessible to both training and scoring workers.
6. Register a named model such as `idx-outperformer-triple-barrier`; tag each
   version with contract IDs and review status. Use aliases such as
   `@candidate` and `@champion`; resolve the alias once and log the concrete
   version on each score. Use an MLflow tracking server with a persistent
   database-backed registry and persistent artifact store, protected with
   appropriate network and access controls. Pin and test compatible versions.
7. Approve only after reproducibility, leakage checks, minimum support,
   predefined business/statistical gates and shadow scoring pass. Save the
   prior champion version and baseline before alias promotion; verify the new
   version after deployment. Roll back by restoring the prior alias and its
   compatible feature contract. If it cannot be loaded safely, stop publishing
   new scores and alert rather than silently falling back.

Start with GPU training if available, but document and test an explicit CPU
execution option; inference need not require CUDA. Track hardware and
library versions since they can affect reproducibility.

## 7. Drift monitoring and response

Compare production to a **fixed, versioned reference** built from the
champion's training data after the exact production preprocessing. Log its
dates, population, bin edges/categories, sample counts and null rates in
MLflow and persist its ID with every monitoring result. Use aligned eligible
universes and matching decision-time semantics. Review a rolling comparison
as context, not as a replacement for the fixed reference.

| Signal | Proposed calculation and cadence | Initial response |
| --- | --- | --- |
| Data quality | Every session: freshness, eligible coverage, missing/non-finite rates, schema/categories, duplicate keys, impossible prices, warm-up and sector availability. Compare to prior and expected universe. | Hard contract failures block affected publication and alert. Warn/stop coverage levels must be agreed per source; low sample is `insufficient data`, never healthy by default. |
| Feature drift | Daily aggregates; compare a rolling 20-session distribution to fixed reference for each selected numeric feature using PSI with reference-derived, frozen bins and explicit underflow/overflow/missing bins. For sector, compare proportions including missing/unknown. | **Initial proposal:** PSI >= 0.10 warning, >= 0.25 investigation; require adequate effective samples, monitor persistence (e.g. 3 consecutive eligible sessions), and calibrate using historical market regimes. |
| Prediction drift | Track `P(upper)` distribution, class/rank mix, score coverage and confidence by sector and model version. | Investigate sustained shift alongside feature drift; do not infer performance degradation from score shifts alone. |
| Realized performance / concept drift | After outcomes mature, monitor rolling cohort macro F1, class-2 precision/recall, calibration and precision@K/lift with counts and confidence intervals. Compare the same cohort/universe and threshold to historical baseline and champion expectations. | Escalate persistent, statistically and operationally meaningful deterioration; set margins/minimum support after backtesting. |

Stock-day observations are dependent, particularly with daily overlapping
10-bar windows and correlated sectors. Report counts by session and stock,
use time-blocked uncertainty or appropriately aggregated cohorts, and avoid
claiming significance from raw row count alone. Segment checks should have
minimum support and multiple-comparison/persistence controls. An early
barrier hit can be known before 10 bars, but calculate cohort performance
only once every eligible prediction has had its full horizon or has an
explicit unresolved/censored status; report label coverage and censoring so
early hits do not bias metrics. Suspensions and missing bars require an
explicit maturity/timeout policy, not calendar days masquerading as bars.

### Runbook

1. **Triage:** check run status, lag, source coverage, corporate actions,
   schema/mapping changes, indicator warm-up and data corrections. If inputs
   are invalid or incomplete, block/mark affected scores, repair ingestion,
   backfill and recompute. Alert the pipeline owner.
2. **Characterize:** if data are sound, examine feature and score drift by
   sector/time/regime and review mature outcome performance. A market regime
   shift is not automatically a model failure; drift alone is not a reason to
   replace features or reset the reference.
3. **Challenge:** trigger a candidate retraining run using only available
   labels, a documented rolling/expanding history and purged time folds.
   Compare current features with a newly selected set and unchanged-model
   baseline. A scheduled review (initial proposal: monthly) complements
   alerts; it does not force deployment.
4. **Decide:** model owner reviews validation, untouched test or subsequent
   time holdout, shadow period, stability and costs; record the decision in
   MLflow. If degraded enough to breach agreed safety criteria, stop
   publication or roll back to a validated compatible version. Otherwise
   retain champion while investigating.
5. **Promote and monitor:** manually switch the alias only after gates pass;
   create a new model-specific reference from its training set and preserve
   the previous reference for comparison and rollback. Never reset a baseline
   merely to make an alert disappear.

## 8. Grafana dashboards and alerts

Use the existing Grafana service with its built-in PostgreSQL datasource and
a read-only database role. Provision datasource, dashboards and alert rules
as version-controlled configuration; keep credentials in environment/secret
management. PostgreSQL views can expose stable query interfaces for:

1. **Pipeline health:** latest expected versus completed session, ingestion
   and indicator coverage, missing/stale rows, excluded stock reasons,
   durations, failures, retries and backfill/revision counts.
2. **Analytics results:** latest ranked stocks with all three probabilities,
   model version, score timestamp and feature freshness; distribution and
   sector coverage. Clearly label outputs as research signals.
3. **Data and feature drift:** missingness, distributions/PSI, unknown
   categories, baseline ID, sample counts, low-support flags and model
   version; annotate promotions and source revisions.
4. **Model quality:** mature cohort counts and coverage, class distribution,
   confusion matrix, class-2 precision/recall, calibration and precision@K
   with appropriate uncertainty; distinguish pending from missing outcomes.

Set alerts for missed expected sessions, failed/stale scoring, failed quality
gates, persistent drift and sustained mature-performance degradation.
Configure no-data/error behavior explicitly and route to an accountable
owner; avoid alerts on exchange holidays and suppress duplicate alerts during
acknowledged backfills. Test alert delivery and recovery, not just panels.

## 9. Delivery phases and acceptance gates

| Phase | Deliverables | Exit criteria |
| --- | --- | --- |
| 0. Product and source contracts | Approve target semantics, eligible universe, bar/session rules, corporate actions, expected timing, policy/baselines and owners. Audit historical availability, sector, source failures and effective coverage. | Known limitations and acceptance thresholds recorded; enough valid history/future bars demonstrated. |
| 1. Reusable research pipeline | Extract label generation, indicator parity and training dataset construction; add versioned feature/label contracts and tests. Fix point-in-time, warm-up, missingness and correction handling. | Golden notebook-versus-production fixtures agree; boundary/tie/incomplete windows and historical/daily parity pass. |
| 2. Time-safe model and MLflow | Implement purged temporal selection/tuning, baselines, tracked runs and persistent registry/artifact storage. | Fresh run reproduces artifacts and metrics; no train/validation/test event overlap; candidate meets pre-agreed untouched-test gates. |
| 3. Batch scoring and outcomes | Add migrations, validated Airflow dependencies, calendar-aware daily scoring, run/revision records and label-maturity jobs; backfill/replay. | Retry is idempotent, failures do not emit success, concrete model/inputs are traceable, incomplete data block publication, rollback tested. |
| 4. Grafana and drift | Persist quality, drift and mature-performance metrics; provision dashboards, alerts and documented runbook. | Correct reference and cohorts, sample/no-data handling, notification delivery and incident recovery demonstrated on simulated drift and missing runs. |
| 5. Shadow to production | Shadow candidate across multiple market sessions; review prediction coverage, outcomes, stability and operational readiness. | Owner approves gates and manually sets champion alias; repeat scoring, monitoring and rollback checks after promotion. |

### Minimum test matrix

- Labels: upper/lower/none, same-bar double touch, missing/invalid OHLC,
  early hit followed by missing data, incomplete horizon, different stock
  calendars and corporate-action scenarios.
- Features: no look-ahead, exact warm-up bar requirements, rolling boundaries,
  unknown sector, non-finite/zero volatility, training/inference parity,
  formula-version coexistence and source corrections.
- Splits: chronological global boundaries, event-end purge, overlapping
  windows, selection inside folds and untouched test.
- Operations: late and partial ingestion, asset success with missing tickers,
  holidays, suspensions, MLflow unavailable, alias missing, schema mismatch,
  idempotent retries, revision policy, shadow promotion and rollback.
- Monitoring: fixed baseline bins including missing/out-of-range values, low
  support, delayed/censored labels, actual mature cohorts, alert no-data and
  recovery, version and baseline isolation.

## 10. References

- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
  for model versions, lineage and aliases.
- [Grafana PostgreSQL datasource](https://grafana.com/docs/grafana/latest/datasources/postgres/)
  for PostgreSQL querying, dashboards and alerting.
