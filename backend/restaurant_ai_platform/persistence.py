import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "pipeline.db"


def _get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            request_id TEXT,
            status TEXT,
            duration_ms INTEGER,
            payload_json TEXT,
            result_json TEXT,
            created_at TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def save_run(
    run_id: str,
    request_id: str,
    status: str,
    duration_ms: int,
    payload: dict,
    result: dict,
):
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO pipeline_runs
        (run_id, request_id, status, duration_ms, payload_json, result_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            request_id,
            status,
            duration_ms,
            json.dumps(payload),
            json.dumps(result),
            datetime.utcnow().isoformat(),
        ),
    )

    conn.commit()
    conn.close()


def get_last_run():
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT run_id, request_id, status, duration_ms, payload_json, result_json, created_at
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
        "status": row[2],
        "duration_ms": row[3],
        "payload": json.loads(row[4]),
        "result": json.loads(row[5]),
        "created_at": row[6],
    }