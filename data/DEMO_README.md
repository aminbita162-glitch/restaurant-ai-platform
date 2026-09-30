# DEMO Sales Data — Demo Install Pack

> **DEMO** — This file is for demonstration only. Do not use in production ingestion.

## Files

| File | Purpose |
|---|---|
| `demo_sales.csv` | DEMO sample sales data (15 daily rows) for testing the pipeline. |
| `DEMO_README.md` | This file. DEMO documentation for the demo install pack. |

## Usage

1. Copy `demo_sales.csv` to a tenant-specific path, for example:
   ```
   data/<restaurant_id>__<location_id>__sales.csv
   ```
2. Run the pipeline with `demo=true` to use DEMO sample data.
3. The production ingestion default (`upload_sales.csv`) is **not** overwritten by this DEMO pack.

## Format

The CSV has two columns:

| Column | Type | Description |
|---|---|---|
| `date` | ISO date string | The calendar date of the sales row |
| `daily_sales_total` | float | Total sales value for that date |

## DEMO Label

Both files in this pack are labeled **DEMO** in the file name or header. This pack is not a production data source.
