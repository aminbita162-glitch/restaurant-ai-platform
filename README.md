# Restaurant AI Platform

An AI-powered backend platform for restaurant operations intelligence.

This system is designed to ingest operational data, generate features, run machine learning predictions, produce GPT-based business insights, optimize staffing recommendations, and prepare a dashboard-ready payload for restaurant decision-making.

---

## Project Overview

Restaurant AI Platform is a modular AI operations pipeline for restaurants.

It provides a structured backend workflow that transforms raw restaurant data into operational intelligence through the following stages:

- data ingestion
- warehouse preparation
- feature engineering
- feature synchronization
- model registration
- sales forecasting
- GPT insight generation
- staffing optimization
- dashboard payload generation

The platform is built as a backend-first system with API endpoints for health checks, pipeline execution, pipeline status inspection, and last-run retrieval.

---

## Problem

Restaurants make daily operational decisions under uncertainty.

Typical challenges include:

- predicting next-day or next-week sales
- deciding staffing levels
- planning inventory requirements
- reducing waste
- translating forecast data into business actions

In many cases, restaurant data exists but is not transformed into usable intelligence. Teams often rely on spreadsheets, intuition, or fragmented dashboards, which creates slow and inconsistent decision-making.

---

## Solution

This platform solves that problem by providing a structured AI pipeline that converts restaurant data into operational outputs.

The system:

- ingests restaurant sales and related operational inputs
- prepares structured warehouse-like records
- derives operational features
- synchronizes features for downstream usage
- registers a prediction model
- forecasts short-term sales
- generates GPT-based structured business insights
- recommends staffing actions
- builds a dashboard-ready payload

The solution is intentionally designed so that AI is not used as uncontrolled free text generation. Instead, AI output is constrained into structured JSON fields that can be used reliably by downstream systems.

---

## Architecture

The platform follows a modular pipeline architecture.

### Core pipeline flow

1. `1_data_ingestion`
2. `2_data_warehouse`
3. `3_feature_engineering`
4. `4_feature_store_sync`
5. `model_registry`
6. `5_ml_prediction`
7. `gpt_insight`
8. `6_optimization`
9. `8_dashboard_update`

### Main components

- **Data Ingestion**  
  Loads restaurant operational input data from CSV or fallback sources.

- **Data Warehouse**  
  Converts input data into structured warehouse-style records.

- **Feature Engineering**  
  Generates derived operational features for downstream ML and optimization.

- **Feature Store Sync**  
  Syncs engineered features into a feature-ready representation.

- **Model Registry**  
  Registers the forecasting model used by prediction logic.

- **ML Prediction**  
  Produces a 7-day sales forecast.

- **GPT Insight**  
  Converts prediction output into structured business recommendations.

- **Optimization**  
  Produces staffing recommendations based on forecast demand.

- **Dashboard Update**  
  Combines forecast, staffing, inventory planning, and GPT insights into one payload.

### API layer

The backend exposes operational API endpoints for:

- health checks
- pipeline status inspection
- last-run retrieval
- manual pipeline execution

### Persistence layer

The system stores pipeline run metadata so the most recent execution can be inspected later.

---

## Example Workflow

A typical full pipeline execution works like this:

### Input
Restaurant sales records are loaded for a specific tenant:

- `restaurant_id = restaurant_001`
- `location_id = location_001`

### Processing
The platform executes:

- ingestion
- warehouse shaping
- feature engineering
- feature sync
- model registration
- ML prediction
- GPT insight generation
- optimization
- dashboard preparation

### Output
The final dashboard payload includes:

- `forecast`
- `staffing_plan`
- `inventory_plan`
- `insight_json`
- `actions`
- `risk_level`

### Example outcome
A successful pipeline run returns:

- `status = ok`
- `summary.error_count = 0`
- `summary.ok_count = 9`

This indicates that the full operational AI pipeline completed successfully.

---

## API Endpoints

### Health
- `GET /health`
- `GET /api/v1/health`

### Pipeline Status
- `GET /pipeline/status`
- `GET /api/v1/pipeline/status`

### Last Pipeline Run
- `GET /pipeline/last-run`
- `GET /api/v1/pipeline/last-run`

Tenant-aware example:

- `/api/v1/pipeline/last-run?restaurant_id=restaurant_001&location_id=location_001`

### Execute Pipeline
- `POST /pipeline/run`
- `POST /api/v1/pipeline/run`

> **Note:** `GET /pipeline/run` does NOT execute the pipeline. It returns instructions to use POST. The pipeline must never be triggered by a safe (GET) request.

---

## Current Capabilities

| Capability | Status | Notes |
|---|---|---|
| Data ingestion (CSV, tenant-scoped) | Implemented | Required row fields: `date`, `net_sales`, `restaurant_id`, `location_id`. Missing field → `DATA_INGESTION_MISSING_REQUIRED_FIELD`. No silent fill. DEMO only with `demo=true`. |
| Pipeline orchestrator (step selection, dry-run) | Implemented | Async job enqueue via `POST /pipeline/run` (202 + job_id). `GET /pipeline/run` does not execute. |
| Worker entrypoint (`backend/worker.py`) | Implemented | Standalone process that executes queued jobs. Run separately from the API server (`python worker.py`). |
| Postgres persistence (optional) | Implemented | When `DATABASE_URL` is set; SQLite local-only otherwise. Tenant-scoped. |
| Day-of-week sales forecast | Implemented | `method=dow_heuristic`; 7-day horizon from weekday buckets. Thin history (<7 rows) errors. |
| Backtest quality gate | Implemented | Hold-out MAPE; `labor_recommendation_allowed=false` when MAPE > 50% or insufficient history. |
| Block shift planner | Implemented | 2-hour blocks with `start`, `end`, `role`, `n`. `method=rule` — not a constraint solver. |
| BOM inventory | Implemented | Requires explicit recipe/BOM structure; errors if missing. `inventory_method=bom_rule`. No MRP. |
| Waste numeric contract | Implemented | Numeric from inputs; errors on missing/invalid. `waste_method=rule` + `weekly_waste_target`. |
| Tenant API gate | Implemented | `X-Api-Key` required. Tenant keys must not be defaulted. Per-tenant rate limit on `POST /pipeline/run`. |
| AI output schema gate | Implemented | Invalid model JSON → `degraded` with `method=rule`; valid → `method=openai`. Raw text never passes as actions. |
| Structured run logs | Implemented | Every log line includes `run_id`, `step`, `duration_ms`. |
| Manager decision payload | Implemented | Compact payload with `approval=proposed`, `actions[]`, forecast/staffing/inventory/waste summary. |
| GPT insight generation | Experimental | Quality depends on external model. Schema-validated; degraded fallback when invalid. |
| Forecast accuracy | Experimental | Heuristic (day-of-week average + 2% growth). Not a trained ML model. |
| Staffing optimization | Experimental | Fixed ratio (1 per 600 sales). Not a constraint solver. |
| Real-data connectors (POS/ERP) | Planned | CSV-only ingestion today. |
| Full authentication / IdP | Planned | `X-Api-Key` only; not token-bound tenant isolation. |
| Frontend dashboard | Planned | Backend payload only; no UI. |
| Production observability | Planned | Structured logs exist; no alerting/tracing/metrics dashboards. |
| Multi-tenant isolation at scale | Planned | In-process state; not million-tenant isolated cells. |

---

## Limitations

This project is a development baseline. It is not a production-hardened, enterprise-authenticated, or multi-tenant-isolated deployment.

- Forecasting is heuristic, not a trained ML model.
- Staffing is a fixed-ratio rule, not a constraint optimizer.
- Inventory requires an explicit recipe/BOM; no MRP or demand-driven optimization.
- Waste is a fixed-ratio estimate, not a measured event store.
- API auth is `X-Api-Key` only; no full IdP or token-bound tenant isolation.
- No customer-facing frontend.
- No live POS/ERP connectors.

---

## Roadmap

### Phase 1
- stable backend pipeline
- structured GPT insight generation
- tenant-aware execution
- last-run persistence
- dashboard-ready payloads

### Phase 2
- stronger real-data ingestion connectors
- improved forecasting model
- production monitoring and alerting
- stronger feature store design
- richer optimization logic
- better GPT business intelligence layer

### Phase 3
- customer-facing SaaS dashboard
- multi-tenant authentication
- role-based access
- reporting and analytics views
- production deployment hardening

---

## Why this project matters

This project is not just a demo model.

It is an operational AI system designed to show how restaurant data can move through a controlled pipeline and become business-ready intelligence.

The key design principle is simple:

**AI should be structured, constrained, testable, and connected to a complete user flow.**

---

## Baseline Status

Current system status: **Development baseline.**

The system has been verified against:

- health endpoint validation
- pipeline status validation
- tenant-aware run validation
- fail-closed smoke tests (4 tests)
- last-run persistence validation

Note: this is a development baseline. It is not a production-hardened, enterprise-authenticated, or multi-tenant-isolated deployment.

---

## Releases

No GitHub Release has been published for this repository yet.

---

## Engineering Plan

The items below are planned future work written in future tense. None of these capabilities exist in the current codebase.

- The forecasting engine will be replaced with a statistically validated model and benchmarked against held-out data.
- The staffing module will be extended from the current sales/600 heuristic to a constraint-based shift optimizer.
- Inventory planning will move from fixed-ratio estimates to a demand-driven calculation.
- Food waste tracking will be implemented as a measured output, not just a named field.
- A production authentication and tenant isolation layer will be added before any multi-tenant deployment.
- A structured error catalog with trace IDs and tenant context will replace raw exception propagation.
- A versioned artifact registry will be introduced so engineered shapes are reused rather than regenerated.
- Live POS, ERP, and inventory system connectors will replace the current CSV-based ingestion.
- A customer-facing dashboard application will be built on top of the existing payload API.
- Full observability — monitoring, alerting, distributed tracing, and metrics — will be added in a later phase.

---

## Author

Amin Azimi — AI Architect
Amin Azimi and System Development
Business Challenge
Azimi Innovation Lab