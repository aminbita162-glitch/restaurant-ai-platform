import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

DB_PATH = Path(__file__).parent / "pipeline.db"

DEFAULT_RESTAURANT_ID = "restaurant_001"
DEFAULT_LOCATION_ID = "location_001"


def _get_connection():
    return sqlite3.connect(DB_PATH)


def _extract_restaurant_context(payload: Dict[str, Any]) -> tuple[str, str]:
    restaurant_id = str(payload.get("restaurant_id") or DEFAULT_RESTAURANT_ID)
    location_id = str(payload.get("location_id") or DEFAULT_LOCATION_ID)

    if restaurant_id != DEFAULT_RESTAURANT_ID or location_id != DEFAULT_LOCATION_ID:
        return restaurant_id, location_id

    result = payload.get("result")
    if isinstance(result, dict):
        results = result.get("results")
        if isinstance(results, list):
            for step in results:
                if not isinstance(step, dict):
                    continue
                data = step.get("data")
                if not isinstance(data, dict):
                    continue

                rid = data.get("restaurant_id")
                lid = data.get("location_id")

                if rid or lid:
                    return (
                        str(rid or DEFAULT_RESTAURANT_ID),
                        str(lid or DEFAULT_LOCATION_ID),
                    )

    return DEFAULT_RESTAURANT_ID, DEFAULT_LOCATION_ID


def init_db():
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            request_id TEXT,
            restaurant_id TEXT,
            location_id TEXT,
            status TEXT,
            duration_ms INTEGER,
            payload_json TEXT,
            result_json TEXT,
            created_at TEXT
        )
        """
    )

    conn.commit()

    cursor.execute("PRAGMA table_info(pipeline_runs)")
    columns = [row[1] for row in cursor.fetchall()]

    if "restaurant_id" not in columns:
        cursor.execute("ALTER TABLE pipeline_runs ADD COLUMN restaurant_id TEXT")
    if "location_id" not in columns:
        cursor.execute("ALTER TABLE pipeline_runs ADD COLUMN location_id TEXT")

    conn.commit()
    conn.close()


def save_run(payload: Dict[str, Any]):
    run_id = str(payload.get("run_id", ""))
    request_id = str(payload.get("request_id", ""))
    status = str(payload.get("status", "unknown"))
    duration_ms = int(payload.get("duration_ms", 0))
    result = payload.get("result", {})

    restaurant_id, location_id = _extract_restaurant_context(payload)

    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO pipeline_runs
        (run_id, request_id, restaurant_id, location_id, status, duration_ms, payload_json, result_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            request_id,
            restaurant_id,
            location_id,
            status,
            duration_ms,
            json.dumps(payload),
            json.dumps(result),
            datetime.utcnow().isoformat(),
        ),
    )

    conn.commit()
    conn.close()


def get_last_run(
    restaurant_id: Optional[str] = None,
    location_id: Optional[str] = None,
):
    conn = _get_connection()
    cursor = conn.cursor()

    if restaurant_id and location_id:
        cursor.execute(
            """
            SELECT run_id, request_id, restaurant_id, location_id, status, duration_ms, payload_json, result_json, created_at
            FROM pipeline_runs
            WHERE restaurant_id = ? AND location_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (restaurant_id, location_id),
        )
    else:
        cursor.execute(
            """
            SELECT run_id, request_id, restaurant_id, location_id, status, duration_ms, payload_json, result_json, created_at
            FROM pipeline_runs
            ORDER BY id DESC
            LIMIT 1
            """
        )

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "run_id": row[0],
        "request_id": row[1],
        "restaurant_id": row[2],
        "location_id": row[3],
        "status": row[4],
        "duration_ms": row[5],
        "payload": json.loads(row[6]),
        "result": json.loads(row[7]),
        "created_at": row[8],
    }