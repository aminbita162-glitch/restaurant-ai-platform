# DEMO DATA — Restaurant AI Platform

> **Label:** DEMO
> This directory contains demonstration data only.
> It is not real restaurant data. Do not use it as a production source.

---

## What is in this directory

Sample CSV files and fixtures used when `demo=true` is passed to the pipeline.

The demo path is the **only** path to synthetic data. The system will never silently
fall back to demo data. A caller must explicitly set `demo=true` in the request body.

---

## How to use demo mode

Pass `demo=true` in the POST body:

```json
{
  "restaurant_id": "demo_rest",
  "location_id": "demo_loc",
  "demo": true
}
```

The pipeline returns an empty sales list and placeholder inventory/attendance values.
No forecast or BOM calculation is performed on demo data.

---

## What demo mode does NOT do

- Does not load or simulate real sales rows.
- Does not produce a valid forecast (no real `net_sales` data).
- Does not represent any actual restaurant's operational data.
- Is not suitable for staffing, inventory, or waste decisions.

---

## For a live restaurant

Supply a tenant CSV file at:

```
backend/data/<restaurant_id>__<location_id>__sales.csv
```

Required columns per row: `date`, `net_sales`, `restaurant_id`, `location_id`.

A persistent Postgres database (`DATABASE_URL`) and the worker process (`python worker.py`)
are required before running the pipeline against real data.

See the main `README.md` — **30-Day Install** section for setup steps.
