import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional

# Stable public error code returned when tenant keys are required but absent.
PERSISTENCE_ERROR_MISSING_TENANT = "PERSISTENCE_MISSING_TENANT_KEYS"

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
# When DATABASE_URL is set, all operations use Postgres.
# When DATABASE_URL is absent, SQLite is used (local / development only).
# ---------------------------------------------------------------------------

_DATABASE_URL: Optional[str] = (os.getenv("DATABASE_URL") or "").strip() or None


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


# ---------------------------------------------------------------------------
# Postgres helpers
# ---------------------------------------------------------------------------

def _pg_conn():  # type: ignore[return]
    """Return a psycopg2 connection using DATABASE_URL."""
    import psycopg2  # type: ignore
    return psycopg2.connect(_DATABASE_URL)


def _pg_init_db() -> None:
    conn = _pg_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id          SERIAL PRIMARY KEY,
                stored_at   TEXT NOT NULL,
                run_id      TEXT,
                request_id  TEXT,
                restaurant_id TEXT,
                location_id   TEXT,
                status      TEXT,
                payload_json TEXT NOT NULL
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_pg_pipeline_runs_stored_at"
            " ON pipeline_runs(stored_at)"
        )
        conn.commit()
    finally:
        conn.close()


def _pg_save_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    _pg_init_db()

    stored_at = _utc_ts()
    run_id = str(payload.get("run_id") or "")
    request_id = str(payload.get("request_id") or "")
    status = str(payload.get("status") or "")
    restaurant_id = str(payload.get("restaurant_id") or "")
    location_id = str(payload.get("location_id") or "")

    to_store = {**payload, "stored_at": stored_at}
    payload_json = json.dumps(to_store, ensure_ascii=False)

    conn = _pg_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO pipeline_runs
            (stored_at, run_id, request_id, restaurant_id, location_id, status, payload_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (stored_at, run_id, request_id, restaurant_id, location_id, status, payload_json),
        )
        conn.commit()
    finally:
        conn.close()

    return to_store


def _pg_get_last_run(
    restaurant_id: Optional[str],
    location_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    _pg_init_db()

    conn = _pg_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT payload_json
            FROM pipeline_runs
            WHERE restaurant_id = %s AND location_id = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (restaurant_id, location_id),
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return None

    try:
        return json.loads(row[0])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SQLite helpers (unchanged from before C2)
# ---------------------------------------------------------------------------

def _db_path() -> str:
    # Render: /tmp is writable, ephemeral (ok for local / no-DATABASE_URL use)
    return os.getenv("PIPELINE_DB_PATH", "/tmp/restaurant_ai_pipeline.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def _sqlite_init_db() -> None:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stored_at TEXT NOT NULL,
            run_id TEXT,
            request_id TEXT,
            restaurant_id TEXT,
            location_id TEXT,
            status TEXT,
            payload_json TEXT NOT NULL
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_stored_at ON pipeline_runs(stored_at)"
    )
    # Add tenant columns to existing databases that predate this schema.
    cur.execute("PRAGMA table_info(pipeline_runs)")
    existing = {row["name"] for row in cur.fetchall()}
    if "restaurant_id" not in existing:
        cur.execute("ALTER TABLE pipeline_runs ADD COLUMN restaurant_id TEXT")
    if "location_id" not in existing:
        cur.execute("ALTER TABLE pipeline_runs ADD COLUMN location_id TEXT")
    conn.commit()
    conn.close()


def _sqlite_save_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    _sqlite_init_db()

    stored_at = _utc_ts()
    run_id = str(payload.get("run_id") or "")
    request_id = str(payload.get("request_id") or "")
    status = str(payload.get("status") or "")
    restaurant_id = str(payload.get("restaurant_id") or "")
    location_id = str(payload.get("location_id") or "")

    to_store = {**payload, "stored_at": stored_at}
    payload_json = json.dumps(to_store, ensure_ascii=False)

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO pipeline_runs
        (stored_at, run_id, request_id, restaurant_id, location_id, status, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (stored_at, run_id, request_id, restaurant_id, location_id, status, payload_json),
    )
    conn.commit()
    conn.close()

    return to_store


def _sqlite_get_last_run(
    restaurant_id: Optional[str],
    location_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    _sqlite_init_db()

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT payload_json
        FROM pipeline_runs
        WHERE restaurant_id = ? AND location_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (restaurant_id, location_id),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    try:
        return json.loads(row["payload_json"])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API — route to Postgres or SQLite based on DATABASE_URL
# ---------------------------------------------------------------------------

def init_db() -> None:
    if _DATABASE_URL:
        _pg_init_db()
    else:
        _sqlite_init_db()


def save_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    if _DATABASE_URL:
        return _pg_save_run(payload)
    return _sqlite_save_run(payload)


def get_last_run(
    restaurant_id: Optional[str] = None,
    location_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the latest run for the given tenant.

    Both restaurant_id and location_id are required. If either is absent,
    return an error dict rather than a global cross-tenant result.
    Backend is Postgres when DATABASE_URL is set, SQLite otherwise.
    """
    if not restaurant_id or not location_id:
        return {
            "error_code": PERSISTENCE_ERROR_MISSING_TENANT,
            "message": "restaurant_id and location_id are required to read last run",
        }

    if _DATABASE_URL:
        return _pg_get_last_run(restaurant_id, location_id)
    return _sqlite_get_last_run(restaurant_id, location_id)
