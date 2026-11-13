# Target Contract: Stock Market Outperformer Model

> Formal specification of the target variable definition, feature schema, labeling rules, and data contracts for the Outperformer analysis pipeline.

## 01 · Overview & Scope

| Item | Contract |
| --- | --- |
| Document Version | `1.0.0` — proposed; production use requires owner approval of the open policy decisions below |
| Target Asset Universe | Eligible Indonesian/IDX listed stocks; the point-in-time eligibility and suspension policy must be versioned and approved before production |
| Primary Model | Three-class XGBoost (`multi:softprob`) with SHAP-RFE feature selection |
| Reference Notebook | [`stock_market/analytics/outperformer_explore_cuda.ipynb`](../stock_market/analytics/outperformer_explore_cuda.ipynb) |

This contract separates the research label, feature inputs, and production lifecycle so that training and daily scoring have identical, auditable semantics. It formalizes the notebook’s triple-barrier experiment and the implementation plan’s requirements for point-in-time data, versioned features, time-safe evaluation, and MLflow lineage. “Outperformer” is a product name here: the target does **not** measure benchmark-relative return or guarantee investment performance.

## 02 · Target Variable & Labeling Specification

### 2.1 Label semantics

For each stock \(s\) and decision session \(t\), the entry reference is the adjusted closing price \(C_{s,t}\). The target observes the next \(H=10\) available stock trading bars, excluding the decision bar. Calendar days, exchange-wide sessions on which a stock has no bar, and rows belonging to other stocks do not count toward the horizon.

| Class | Name | Exact condition |
| --- | --- | --- |
| `0` | Lower | The first observed barrier touch is the lower barrier, \(L_{s,t}=C_{s,t}(1-0.05)\). A future bar touches it when its low is \(\le L_{s,t}\). |
| `1` | Neither | All 10 future bars are valid and neither barrier is touched during the horizon. |
| `2` | Upper | The first observed barrier touch is the upper barrier, \(U_{s,t}=C_{s,t}(1+0.05)\). A future bar touches it when its high is \(\ge U_{s,t}\). |
| Unknown | Unresolved / censored | No barrier touch is established before an invalid or unavailable bar, or the full 10-bar horizon has not elapsed without a touch. Unknown is null/status, never class `1`. |

Apply the following deterministic rules in order:

1. Require a finite, strictly positive entry close. Construct both fixed barriers from this close; barriers do not trail or reset.
2. Examine future bars in chronological order, from offset 1 through 10. An observed barrier touch terminates the event immediately, even if later bars are missing.
3. If both thresholds are crossed within the same OHLC bar, assign class `0` (lower-first conservative tie-break). OHLC data do not establish intrabar ordering, so no finer ordering may be inferred.
4. If only one barrier is crossed, assign its class (`0` for lower, `2` for upper). A price gap through a barrier counts as a touch under the high/low test; the label is not an executable fill price.
5. Assign class `1` only after ten consecutive valid future bars with no touch. An incomplete horizon or invalid bar before a touch remains unresolved.
6. Store the event end as the date of the first touching bar, or the tenth future bar for a mature class-`1` event. Retain the maturity/availability timestamp and a reason/status for unresolved events.

These are stock-relative absolute-return barriers, not a comparison with an index. No market benchmark return enters the target equation. If benchmark-relative alpha is required, define a distinct target, label contract version, dataset, and model rather than changing these labels in place.

### 2.2 Validity, look-ahead, and label-window controls

- **Price validity:** require finite positive close, high, and low; require `low <= high` and `low <= close <= high`. Reject or mark invalid bars that violate these rules. Do not impute OHLC values or silently repair corporate-action discontinuities.
- **Adjustment consistency:** entry close and all future high/low values must use a consistent split/dividend adjustment basis. Record the source, adjustment policy, and data revision. A discontinuity not resolved by verified corporate-action-adjusted data must be treated as invalid/censored, not as a genuine barrier touch.
- **As-of boundary:** features for decision session \(t\) may use only source values available by the close of \(t\). No future OHLC, label, event-end field, later metadata state, or future data correction may leak into the feature row.
- **Event end for validation:** persist `label_end` (the realized event-end session) for every resolved label. Chronological training/validation folds must share global date boundaries across stocks and purge training events whose `label_end` reaches or crosses the next fold boundary; use an embargo appropriate to the overlapping horizon. The notebook’s calendar-day gap and its split helper’s `label_end` expectation are not sufficient production controls.
- **Overlapping observations:** the notebook optionally samples one observation per stock every 10 rows (`cumcount() % horizon == 0`) to form non-overlapping label windows. Treat this as a research sampling choice, not an implicit target rule. Any production sampling must be versioned, deterministic, anchored to a documented per-stock phase, and applied after the target is formed. Otherwise retain all eligible decision sessions and use event-end purging.
- **Tail and missing-path handling:** the notebook’s displayed implementation leaves rows without a full future window null, and its blanket `dropna()` removes them. The production labeler must additionally preserve an already-observed first barrier touch even if the remaining horizon is incomplete; absence of a touch before an invalid or missing future bar remains unresolved.

### 2.3 Rolling feature buffers and training-row eligibility

- Calculate features on per-stock bars sorted by normalized session date, using no cross-stock rolling windows.
- The 52-week high/low features require **252 prior observed stock bars** because the implementation shifts the rolling extrema by one bar. Fetching 400 calendar days is only a warm-up approximation; verify the actual bar count per stock. Do not equate 400 calendar days with 252 valid observations.
- The longest other lookbacks include 250-bar momentum; require all indicator-specific history needed for the model’s features. Do not backfill warm-up NaNs from future observations.
- Exclude warm-up rows and rows with invalid required inputs from model eligibility, recording exclusion reason/counts. Compute these flags before selecting a training date range so that the first eligible observation is determined from the complete available prehistory.
- For supervised training, additionally require a resolved target. Unresolved target rows may be retained in outcome/audit storage but must not be coerced into a class.
- Training and prediction must use the same feature builder and eligibility rules. The production readiness gate must verify history coverage using observed stock bars, not just requested start dates or pipeline success events.

### 2.4 `data_label` and imputation separation

`data_label` identifies the feature-data variant; it is not the outcome label and must never be used as a model feature.

| Exact `data_label` value | Meaning | Contract |
| --- | --- | --- |
| `non_imputed` | Source/calculated feature values without imputation | Canonical research comparison. Incomplete required features are excluded with recorded reasons; do not fill nulls or non-finite values. |
| `imputed` | Feature values after the registered imputation transform | Use only when the model’s versioned preprocessing explicitly enables imputation. Fit the imputer on each training fold only, persist its parameters with the model, and apply that exact transform at inference. |

- Every row and dataset snapshot must carry exactly one of these case-sensitive strings. A single training/evaluation run must not silently mix variants; filter by and log one `data_label` value, or compare the variants as separate experiments.
- `data_label` must agree with the preprocessing artifact. Reject scoring if the input variant does not match the model contract.
- Never impute target labels, entry/future OHLC, timestamps, stock identifiers, event ends, or eligibility flags. Do not impute rolling warm-up gaps. For categorical `sector`, use a versioned category/unknown policy; do not infer missing historical sector from current metadata.
- If an imputed variant is not approved for a model, use `non_imputed` and fail closed or exclude the affected observation rather than silently switching paths.

## 03 · Feature & Indicator Contract

### 3.1 Candidate indicator schema

The candidate catalog is sourced from [`indicators.py`](../stock_market/transform/indicators.py) and the notebook. It consists of 21 numeric indicators plus `sector`. Feature timestamps are the decision session; formulas and source/availability rules are part of the feature version, not merely column names.

| Core category | Candidate columns | Data type at model boundary | Null / invalid handling |
| --- | --- | --- | --- |
| Sector category | `sector` | Categorical/string with a versioned mapping; not an ordinal numeric value | Require a historically valid as-of mapping. Preserve an explicit missing/unknown category only if trained and declared in the contract; otherwise exclude the row. Exclude the feature if point-in-time sector history cannot be established. |
| One-bar return | `returns` | Finite numeric; normalize to the contract’s declared floating dtype | First return and gaps without a valid prior close are null; exclude in `non_imputed`, or use only the packaged fitted imputer in `imputed`. |
| Moving-average distance | `distance_moving_avg` | Finite numeric | Requires 30 valid closes; do not impute warm-up. |
| Momentum | `momentum_5d`, `momentum_10d`, `momentum_20d`, `momentum_30d`, `momentum_120d`, `momentum_250d` | Finite numeric | Requires the named number of prior bars and a valid reference close; exclude warm-up/invalid values or use the declared imputer. |
| Annualized realized volatility | `volatility_5d_ann`, `volatility_10d_ann`, `volatility_15d_ann`, `volatility_20d_ann`, `volatility_25d_ann`, `volatility_30d_ann` | Finite numeric | Rolling standard deviation of returns multiplied by \(\sqrt{252}\); zero/non-finite denominators are invalid where used downstream. Do not impute warm-up. |
| Moving-average slope | `moving_avg_slope_3d`, `moving_avg_slope_5d`, `moving_avg_slope_10d`, `moving_avg_slope_15d` | Finite numeric | Ratio of the 30-bar moving average to its value the named number of bars earlier, minus one; exclude missing/zero denominator. |
| 52-week price distance | `distance_high_52w`, `distance_low_52w` | Finite numeric | Compare current close with extrema from the 252 prior bars; current bar is excluded from extrema. Require full history and positive denominator. |
| Risk-adjusted momentum | `risk_adjusted_momentum` | Finite numeric | `momentum_30d / volatility_30d_ann`; zero or non-finite volatility produces null, not infinity. |

At model input, numeric features must be finite and cast to the exact dtype recorded in the feature contract. Column presence, names, order, type, category mapping, and preprocessing are strict invariants. Reject undeclared extra inputs if they alter the signed schema; never silently substitute a similarly named feature.

### 3.2 SHAP-RFE selection and versioning

- Keep the candidate indicator catalog separate from the selected ordered feature list. SHAP-RFE results are model-specific; a selection does not redefine the global catalog.
- Store the ordered candidate list, selected list, definitions/formula versions, source columns, lookback, units, dtypes, null policy, point-in-time availability, sector mapping version, preprocessing/data-label policy, target-contract ID, and selection metadata in the version-controlled `stock_market/analytics/config/outperformer_features.yaml` (or its approved successor).
- Compute a deterministic SHA-256 hash over a canonical serialization of the complete contract. Any change to order, formula, dtype, mapping, imputation, or semantics requires a new feature-contract version and hash.
- SHAP-RFE selection, imputation, and category handling must be fit inside each training fold using only fold-training rows. Use event-end-purged chronological folds for selection and tuning; do not reproduce the notebook’s shuffled `StratifiedKFold` as production validation.
- The notebook/plan’s saved 20-feature selection is a **research snapshot**, not an approved production subset. Its ordered list is:
  `volatility_30d_ann`, `distance_high_52w`, `distance_low_52w`, `returns`, `volatility_10d_ann`, `volatility_20d_ann`, `momentum_120d`, `volatility_5d_ann`, `volatility_25d_ann`, `momentum_5d`, `volatility_15d_ann`, `sector`, `distance_moving_avg`, `momentum_250d`, `moving_avg_slope_15d`, `moving_avg_slope_3d`, `momentum_10d`, `momentum_20d`, `moving_avg_slope_10d`, `moving_avg_slope_5d`.
- Re-run selection on point-in-time data and safe folds before production approval. Specifically validate `sector` availability; if historical values cannot be reconstructed, omit it from the candidate champion contract and record that decision.
- At training, log the exact ordered selected feature list and contract artifact with the run. At prediction, load that same immutable contract with the model and align by its declared order. Do not re-run RFE, consult a mutable latest config, or derive feature order from database/query order at inference.
- The model signature must record named ordered inputs and types. The model artifact must package preprocessing and the feature contract or an immutable reference to it. Contract hash/signature mismatch is a hard failure, not a warning.

## 04 · Pipeline Integration & Validation Rules

### 🚂 **Training Phase Contract**

- **Snapshot and reproducibility:** freeze the eligible historical universe, source watermark/revision, indicator and target contract versions, `data_label`, code commit, environment/dependency versions, random seeds, date range, and actual per-stock history coverage. Persist a durable snapshot or content-addressed reference sufficient to recreate the dataset.
- **Time-safe evaluation:** use global chronological date boundaries across stocks. Purge training events by actual `label_end`, including overlapping label windows, and use an explicit embargo. Fit SHAP-RFE, imputation, encoding, and all learned preprocessing on training folds only; reserve a final untouched temporal test.
- **Evaluation record:** report class support, confusion matrix, macro F1, balanced accuracy, per-class precision/recall, multiclass log loss and calibration. For ranking, report precision@K/lift against prevalence with K, cohort, sample counts, and uncertainty defined in advance. Evaluate executable strategy performance separately from the close-based label.
- **MLflow tracking:** log to a named experiment, such as `idx-outperformer`, with the model and preprocessing, ordered signature, class order `[0, 1, 2]`, input example, feature/target contracts and hashes, `data_label`, dataset fingerprint, code/environment, splits/purge details, selection report, parameters, baselines, metrics, and evaluation/backtest artifacts. Persist artifacts in shared durable storage accessible to training and scoring workers.
- **Registry:** register a named model such as `idx-outperformer-triple-barrier`; tag each model version with target-contract ID, feature-contract ID/hash, data-label variant, training MLflow run ID, and review status. Create a candidate version first. Promotion to `@champion` is manual and gated by reproducibility, leakage checks, minimum support, pre-agreed acceptance criteria, and shadow scoring; retraining must never auto-promote.
- **Rollback:** preserve the prior champion and its compatible feature contract. Record promotion/rollback decisions. If the registry, alias, artifact, or contract cannot be resolved and validated, stop publication and alert rather than silently falling back to a local or stale model.
- **Execution parity:** support and test CUDA where available and an explicit CPU path. Record runtime/library/hardware details relevant to reproducibility; inference does not require CUDA.

### ⚡ **Prediction Phase Contract**

- Run one after-close batch per validated exchange session, downstream of ready price and indicator data. Use the exchange session date as the decision key, schedule with `Asia/Jakarta`, and store run timestamps in UTC. Do not score intraday or claim same-close executability.
- Before scoring, gate on expected-universe/session coverage, ingestion failures, duplicate keys, freshness, indicator coverage, per-stock warm-up, input finiteness, and contract/schema compatibility. Holidays and suspensions require explicit policy; a successful upstream asset event alone does not establish completeness.
- Resolve `@champion` to one concrete model version exactly once per run. Load the packaged preprocessing, feature schema, category map, and classifier together. Record model name/version, MLflow run ID, feature/target contract versions and hashes, input snapshot/watermark, session, and `data_label` on each run/result.
- Score only eligible stocks. Emit the three probabilities in class order `[P(lower), P(neither), P(upper)]`, each finite in `[0,1]` and summing to one within a declared numeric tolerance; store predicted class and ranking derived from `P(upper)`. Scores are research signals, not buy/sell instructions.
- Write run status, predictions, and exclusions transactionally. A retry of the same run must be idempotent (recommended unique key: `(run_id, stock_id, decision_session)`). A deliberate recompute, model change, or corrected source creates a traceable revision; do not use uncontrolled last-write-wins. Emit success only after validation and commit.
- Record exclusions with reason codes, including missing/stale price, insufficient warm-up, invalid feature, missing required category, and contract mismatch. Do not silently substitute the prior session’s features or model.

### 🛡️ **Data Quality & Edge Cases**

- **Missingness:** distinguish expected warm-up/missing source values, invalid numeric values, and imputed values. Enforce the selected `data_label`; report null and non-finite rates per feature and block scoring when required values violate the model’s policy. Never turn unresolved labels into class `1`.
- **Timestamps:** normalize source trade dates to exchange-local session dates (`Asia/Jakarta`) before joining or rolling. Use the normalized session date for per-stock ordering and event horizons; store operational timestamps in UTC. Reject duplicate `(stock_id, session_date)` bars or resolve them through a deterministic, audited source-revision policy.
- **Point-in-time data:** map metadata as-of the decision session. Current sector metadata is not evidence of historical sector state. Preserve source/adjustment revisions and avoid survivorship-biased universe reconstruction where historical membership is available.
- **Schema and arithmetic checks:** validate required columns, exact feature order/types, finite values, positive prices, OHLC consistency, barrier math, probability bounds/sum, unique result keys, and observed warm-up bars. Validate training/inference parity on a shared golden fixture.
- **Outcome maturity:** keep pending, mature, invalid, and censored outcomes distinct. Do not calculate mature-cohort performance until outcomes have reached their event end or are explicitly unresolved under a documented timeout policy. Report outcome coverage and sample support.
- **Operational failures:** fail closed on missing champion/alias, artifact or signature mismatch, insufficient coverage, invalid inputs, or database write failure. Surface the error, mark the run failed, and alert; do not emit success-shaped partial results or silently fall back.
- **Contract tests:** cover upper/lower/neither, same-bar double touch, early hit with a later missing bar, invalid bar before any touch, incomplete horizon, different stock calendars, corporate-action discontinuity, warm-up boundary, unknown sector, zero volatility, no look-ahead, chronological event-end purge, exact feature alignment, imputation separation, idempotent retry, revision handling, and registry failure/rollback.

> ⚠️ **Production approval required:** confirm eligible-universe rules, point-in-time sector policy, corporate-action-adjusted price source, imputation approval, any non-overlap sampling policy, label maturity/timeout handling, acceptance thresholds, and model owners. Until approved and implemented, notebook metrics and saved feature selections are research evidence, not production guarantees.
