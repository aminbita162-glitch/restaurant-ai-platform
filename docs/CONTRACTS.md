# Restaurant AI Platform — Current Contracts

> **Status:** Frozen as of PHASE C0.
> This document describes the system **as it exists today**.
> Future intentions are collected in the [FUTURE](#future) section at the bottom.
> Do not treat any FUTURE item as implemented.

---

## 1. Pipeline Steps

The pipeline is executed by `orchestrator.run_pipeline()`.
Steps run in the order listed below. Each step may be skipped, included, or
excluded via caller options.

| Order | Step key               | Module                      | Description                                    |
|-------|------------------------|-----------------------------|------------------------------------------------|
| 1     | `real_data_ingestion`  | `real_data_ingestion`       | Optional live-data ingestion (skipped when module absent) |
| 2     | `1_data_ingestion`     | `data_ingestion`            | Reads tenant CSV sales file from disk          |
| 3     | `2_data_warehouse`     | `data_warehouse`            | Passes sales records downstream                |
| 4     | `3_feature_engineering`| `feature_engineering`       | Derives features from sales records            |
| 5     | `4_feature_store_sync` | `feature_store_sync`        | Syncs features (no-op if module absent)        |
| 6     | `model_registry`       | `model_registry`            | Registers a simple model artifact              |
| 7     | `5_ml_prediction`      | `ml_prediction`             | Produces 7-day heuristic forecast              |
| 8     | `6_optimization`       | `optimization`              | Produces staffing plan via ratio rule          |
| 9     | `gpt_insight`          | `gpt_insight`               | Calls OpenAI if key is set; skipped otherwise  |
| 10    | `7_api_serving`        | `api_serving`               | API housekeeping (init persistence)            |
| 11    | `8_dashboard_update`   | `dashboard_update`          | Assembles inventory, waste, and insight output |
| 12    | `9_model_training`     | `model_training`            | Lightweight model training (no-op if absent)   |

Pipeline runs **inside the HTTP request** (synchronous). There is no async job queue yet.

---

## 2. Public Error Codes

All error responses carry a stable `error_code` string. No raw tracebacks are
returned to callers. The codes below are the ones currently defined in the
codebase.

### 2.1 Data Ingestion (`data_ingestion.py`)

| Code                          | Meaning                                                      |
|-------------------------------|--------------------------------------------------------------|
| `DATA_INGESTION_FILE_NOT_FOUND` | Tenant CSV file does not exist on disk                     |
| `DATA_INGESTION_READ_FAILED`    | CSV file exists but could not be read (encoding, I/O error)|
| `DATA_INGESTION_EMPTY`          | CSV file exists and was read, but contains zero valid rows  |

### 2.2 Feature Engineering (`feature_engineering.py`)

| Code                                | Meaning                                                |
|-------------------------------------|--------------------------------------------------------|
| `FEATURE_ENGINEERING_MISSING_SALES` | Sales data absent from context                        |
| `FEATURE_ENGINEERING_INVALID_SALES` | Sales data present but not in expected format         |

### 2.3 Forecast / ML Prediction (`ml_prediction.py`)

| Code                          | Meaning                                                      |
|-------------------------------|--------------------------------------------------------------|
| `FORECAST_MISSING_SALES`      | Sales list is absent or empty; cannot forecast               |
| `FORECAST_INSUFFICIENT_DATA`  | Fewer than 7 valid daily rows; forecast refused              |
| `FORECAST_INVALID_SALES`      | Sales rows present but not numerically valid                 |

### 2.4 Labor / Optimization (`optimization.py`)

| Code                              | Meaning                                                    |
|-----------------------------------|------------------------------------------------------------|
| `LABOR_MISSING_SALES_OR_FORECAST` | Sales absent; forecast cannot run; staffing refused        |
| `LABOR_FORECAST_FAILED`           | Forecast step returned an error; staffing refused          |

### 2.5 Inventory / Waste (`dashboard_update.py`)

| Code                       | Meaning                                                         |
|----------------------------|-----------------------------------------------------------------|
| `INVENTORY_MISSING_SALES`  | Sales absent from context; inventory and waste refused          |
| `INVENTORY_INVALID_SALES`  | Sales rows present but missing numeric `daily_sales_total`      |

### 2.6 Persistence (`core/persistence.py`)

| Code                          | Meaning                                                      |
|-------------------------------|--------------------------------------------------------------|
| `PERSISTENCE_MISSING_TENANT_KEYS` | `restaurant_id` or `location_id` absent; read refused   |

### 2.7 API Layer (`api_serving.py`)

| Code                   | HTTP | Meaning                                                             |
|------------------------|------|---------------------------------------------------------------------|
| `AUTH_NOT_CONFIGURED`  | 503  | `RESTAURANT_AI_API_KEY` env var is empty; auth cannot work         |
| `AUTH_UNAUTHORIZED`    | 401  | `X-Api-Key` header missing or does not match configured key        |
| `MISSING_TENANT_KEYS`  | 400  | `restaurant_id` and/or `location_id` absent on a protected route   |
| `PIPELINE_ERROR`       | 500  | Unhandled exception during pipeline execution                      |

---

## 3. Tenant Keys

Every protected route and every pipeline run requires both keys:

| Key             | Where supplied              | Notes                                          |
|-----------------|-----------------------------|------------------------------------------------|
| `restaurant_id` | JSON body (POST) or query param (GET) | Must be a non-empty string. Never defaulted by the agent. |
| `location_id`   | JSON body (POST) or query param (GET) | Must be a non-empty string. Never defaulted by the agent. |

The agent must not invent `restaurant_001` / `location_001` when the caller
omits these keys.

---

## 4. Authentication

| Mechanism   | Header       | Env var                  | Behavior                                      |
|-------------|--------------|--------------------------|-----------------------------------------------|
| `X-Api-Key` | `X-Api-Key`  | `RESTAURANT_AI_API_KEY`  | Present on all non-health routes. 503 if env not set; 401 if header wrong. |

This is a single shared key checked in-process. It is **not** token-bound
tenant isolation.

---

## 5. Sales Input (step `1_data_ingestion`)

CSV file resolved per tenant. Resolution order:

1. `<restaurant_id>__<location_id>__sales.csv`
2. `<restaurant_id>__sales.csv`

If no file exists, `DATA_INGESTION_FILE_NOT_FOUND` is returned.

Required columns (per row):

| Column              | Type    | Notes                                   |
|---------------------|---------|-----------------------------------------|
| `date`              | string  | ISO date recommended; not validated yet |
| `daily_sales_total` | float   | Must be numeric                         |
| `restaurant_id`     | string  | Optional in CSV; falls back to context  |
| `location_id`       | string  | Optional in CSV; falls back to context  |

Demo mode: caller must explicitly pass `demo=true` in context. The agent
must never silently fall back to synthetic data.

---

## 6. Forecast Output (step `5_ml_prediction`)

| Field                  | Type            | Notes                                                   |
|------------------------|-----------------|---------------------------------------------------------|
| `ml_prediction_status` | `"ok"`          | Present on success                                      |
| `method`               | `"heuristic"`   | Always `heuristic`; not a trained model                 |
| `method_description`   | string          | Human-readable description of the algorithm             |
| `sales_rows_used`      | int             | Number of input rows consumed                           |
| `avg_daily_sales_used` | float           | Average of input daily totals                           |
| `horizon`              | int (7)         | Number of forecast days                                 |
| `forecast`             | list of objects | Each: `{"predicted_sales": float}`                      |
| `restaurant_id`        | string          |                                                         |
| `location_id`          | string          |                                                         |
| `timestamp`            | ISO string      |                                                         |

Algorithm: simple average of all supplied daily totals, compounded at a
fixed 2% daily growth factor. Requires a minimum of 7 valid daily rows.

---

## 7. Labor / Staffing Output (step `6_optimization`)

| Field                   | Type          | Notes                                              |
|-------------------------|---------------|----------------------------------------------------|
| `optimization_status`   | `"ok"`        | Present on success                                 |
| `method`                | `"rule"`      | Always `rule`; not a constraint optimizer          |
| `method_description`    | string        | `"1 staff per 600 sales units (minimum 1)"`        |
| `staffing_plan`         | list          | Per day: `day_index`, `predicted_sales`, `recommended_staff` |
| `restaurant_id`         | string        |                                                    |
| `location_id`           | string        |                                                    |
| `timestamp`             | ISO string    |                                                    |

Rule constant: 1 staff per 600 sales units, minimum 1.

---

## 8. Inventory Output (step `8_dashboard_update`)

| Field                          | Type       | Notes                                                     |
|--------------------------------|------------|-----------------------------------------------------------|
| `inventory_plan`               | list       | Per day: `day_index`, `daily_sales_total`, `ingredients_needed`, `estimated_meals` |
| `inventory_method`             | `"rule"`   | Always `rule`; not MRP                                    |
| `inventory_method_description` | string     | Fixed ratio constants described                           |

Rule constants:
- `ingredients_needed = daily_sales_total × 0.45`
- `estimated_meals = daily_sales_total × (1/25)`

---

## 9. Waste Output (step `8_dashboard_update`)

| Field                       | Type      | Notes                                                        |
|-----------------------------|-----------|--------------------------------------------------------------|
| `waste_plan`                | list      | Per day: `day_index`, `daily_sales_total`, `estimated_waste` |
| `waste_method`              | `"rule"`  | Always `rule`; not a trained model                           |
| `waste_method_description`  | string    | Fixed ratio constant described                               |

Rule constant: `estimated_waste = daily_sales_total × 0.08`

This is an **estimate from a fixed ratio**, not a measured event store.

---

## 10. AI / GPT Insight Output (step `gpt_insight`)

| Field               | Type       | Notes                                                              |
|---------------------|------------|--------------------------------------------------------------------|
| `gpt_insight_status`| string     | `"ok"`, `"skipped"`, or `"error"`                                 |
| `insight_json`      | dict/null  | Parsed JSON from OpenAI; keys: `summary`, `staffing`, `inventory`, `waste`, `notes`, `risk_level`, `actions` |
| `insight_json_raw`  | string     | Raw string from OpenAI before parsing                             |
| `openai_model`      | string     | Model name used (e.g. `gpt-3.5-turbo`)                           |

Skipped when `OPENAI_API_KEY` is not set. The `actions` field in `insight_json`
is expected to be a list of exactly 3 short strings. No schema enforcement
exists yet on the response (see FUTURE).

---

## 11. Persistence

- **Backend:** SQLite via `core/persistence.py`. Path: `/tmp/restaurant_ai_pipeline.db`
  (overridable with `PIPELINE_DB_PATH` env var).
- **Scope:** Reads and writes are tenant-scoped by `restaurant_id` + `location_id`.
- **Limitation:** SQLite on ephemeral disk (e.g. Render free tier) is not
  authoritative across restarts.
- **Postgres:** Not yet implemented. `DATABASE_URL` is not read.

---

## 12. HTTP Endpoints (current)

| Method | Path                          | Auth     | Description                                       |
|--------|-------------------------------|----------|---------------------------------------------------|
| GET    | `/health`                     | None     | Liveness check                                    |
| GET    | `/api/v1/health`              | None     | Liveness check (versioned)                        |
| GET    | `/pipeline/status`            | None     | Pipeline configuration summary                    |
| GET    | `/api/v1/pipeline/status`     | None     | Pipeline configuration summary (versioned)        |
| GET    | `/pipeline/last-run`          | X-Api-Key| Last stored run for tenant (requires tenant keys) |
| GET    | `/api/v1/pipeline/last-run`   | X-Api-Key| Last stored run for tenant (versioned)            |
| GET    | `/pipeline/run`               | None     | Returns instructions; **never executes pipeline** |
| GET    | `/api/v1/pipeline/run`        | None     | Returns instructions; **never executes pipeline** |
| POST   | `/pipeline/run`               | X-Api-Key| Executes pipeline synchronously (requires tenant keys) |
| POST   | `/api/v1/pipeline/run`        | X-Api-Key| Executes pipeline synchronously (versioned)       |

`GET /pipeline/run` must never execute the pipeline. It returns a
`how_to_execute` hint only.

---

## 13. CI

Smoke CI exists (`.github/workflows/ci.yml`). It is not an enterprise test
suite. The fail-closed tests live in `backend/tests/test_fail_closed.py`.

---

## FUTURE

> Items in this section are **not implemented**. They are recorded here only
> to distinguish planned work from current reality.

- **C1** — Enforce `source_type` REAL/DEMO on ingestion; reject invalid rows with a stable code.
- **C2** — Optional Postgres persistence when `DATABASE_URL` is set.
- **C3** — Async pipeline job: `POST /pipeline/run` returns 202 + job id.
- **C4** — Single forecast artifact: downstream steps read context; do not re-run `ml_prediction`.
- **C5** — Day-of-week heuristic forecast (`method=dow_heuristic`).
- **C6** — Backtest gate: MAPE recorded; `labor_recommendation_allowed` field.
- **C7** — Block shift planner: 2-hour blocks with `start`, `end`, `role`, `n`.
- **C8** — BOM inventory: suggestions from explicit recipe structure, not fixed ratio.
- **C9** — Waste events: numeric from inputs or error; `method=rule`; weekly target field.
- **C10** — Tighter tenant API gate; in-process rate limit per tenant.
- **C11** — AI output schema gate: invalid model JSON rejected or degraded with `method=rule`.
- **C12** — Structured logs: `run_id`, `step`, `duration_ms` on every log line.
- **C13** — Manager payload: compact summary with `approval=proposed`.
- **C14** — README honesty matrix: Implemented / Experimental / Planned.
- **C15** — Demo install pack: labeled `DEMO` in name and header.
