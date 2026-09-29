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

Browser execution example:

- `/api/v1/pipeline/run?execute=1&confirm=yes`

Tenant-aware browser execution example:

- `/api/v1/pipeline/run?execute=1&confirm=yes&restaurant_id=restaurant_001&location_id=location_001`

---

## Current Capabilities

The current version supports:

- end-to-end pipeline execution
- tenant-aware execution using `restaurant_id` and `location_id`
- 7-day sales forecasting
- GPT-generated structured business insights
- staffing recommendations
- inventory planning payload generation
- persistence of last successful pipeline runs
- dashboard-ready structured output
- browser-based execution for manual testing

---

## Limitations

This project is currently a Phase 1 backend pipeline baseline with important limitations.

### 1. Simplified model logic
The forecasting logic is still relatively simple and not yet a fully production-grade ML forecasting engine.

### 2. Limited real-world data integration
The current ingestion layer supports CSV-style inputs and controlled examples, but is not yet integrated with live POS, ERP, or inventory systems.

### 3. No full authentication / authorization layer
API access control, tenant security isolation, and production auth are not yet fully implemented.

### 4. Limited observability
The platform has execution logging and persistence, but not yet full production monitoring, alerting, tracing, or metrics dashboards.

### 5. No full frontend product yet
The backend prepares dashboard payloads, but a full customer-facing dashboard application is still pending.

### 6. GPT output depends on external model behavior
Although GPT output is structured, the quality of business insight still depends on prompt design and model consistency.

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

Current system status: **Phase 1 backend pipeline baseline.**

The system has been verified against:

- health endpoint validation
- pipeline status validation
- tenant-aware run validation
- full golden pipeline execution
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