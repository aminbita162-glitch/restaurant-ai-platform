from __future__ import annotations

from datetime import datetime
from typing import Dict, Any
import csv
import os
import sqlite3


DEFAULT_DB_PATH = os.getenv("PIPELINE_DB_PATH", "/tmp/restaurant_ai_pipeline.db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp")


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _connect_db():
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            date TEXT PRIMARY KEY,
            daily_sales_total REAL NOT NULL
        )
    """)
    return conn


def _validate_row(date_str: str, sales_str: str):
    try:
        datetime.fromisoformat(date_str)
    except Exception:
        return False, "invalid_date"

    try:
        val = float(sales_str)
        if val < 0:
            return False, "negative_sales"
    except Exception:
        return False, "invalid_sales"

    return True, ""


def run() -> Dict[str, Any]:
    """
    Expects a file:
    /tmp/upload_sales.csv

    CSV format:
    date,daily_sales_total
    2025-01-01,1450.5
    """

    csv_path = os.path.join(UPLOAD_DIR, "upload_sales.csv")

    if not os.path.exists(csv_path):
        return {
            "data": {
                "ingestion_status": "error",
                "reason": "upload_sales.csv_not_found",
                "timestamp": _utc_ts(),
            },
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

    conn = _connect_db()
    cur = conn.cursor()

    inserted = 0
    updated = 0
    skipped = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        if "date" not in reader.fieldnames or "daily_sales_total" not in reader.fieldnames:
            return {
                "data": {
                    "ingestion_status": "error",
                    "reason": "invalid_csv_headers",
                    "timestamp": _utc_ts(),
                },
                "errors": [],
                "warnings": [],
                "metrics": {},
            }

        for row in reader:
            date_str = row["date"]
            sales_str = row["daily_sales_total"]

            valid, reason = _validate_row(date_str, sales_str)
            if not valid:
                skipped += 1
                continue

            value = float(sales_str)

            cur.execute("SELECT date FROM sales WHERE date = ?", (date_str,))
            exists = cur.fetchone()

            if exists:
                cur.execute(
                    "UPDATE sales SET daily_sales_total = ? WHERE date = ?",
                    (value, date_str),
                )
                updated += 1
            else:
                cur.execute(
                    "INSERT INTO sales (date, daily_sales_total) VALUES (?, ?)",
                    (date_str, value),
                )
                inserted += 1

    conn.commit()
    conn.close()

    return {
        "data": {
            "ingestion_status": "ok",
            "inserted": inserted,
            "updated": updated,
            "skipped": skipped,
            "timestamp": _utc_ts(),
        },
        "errors": [],
        "warnings": [],
        "metrics": {},
    }