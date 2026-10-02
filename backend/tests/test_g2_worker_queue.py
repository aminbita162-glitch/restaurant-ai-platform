"""PHASE G2 tests — Postgres job queue for worker.

Tests:
1. Job enqueued by one process context is claimed by worker code in another
   process context (cross-process contract verified via a shared SQLite DB
   that is injected by patching _pg_conn — the schema and SQL are identical).
2. enqueue_job without DATABASE_URL returns QUEUE_NO_DATABASE error dict.
3. worker.run_once without DATABASE_URL returns 0 without executing any job.

SQLite shim note:
  psycopg2 and sqlite3 share the same SQL for this schema except for two
  Postgres-specific features:
    - SERIAL PRIMARY KEY  → INTEGER PRIMARY KEY AUTOINCREMENT
    - FOR UPDATE SKIP LOCKED → not needed in single-writer tests; replaced
      with a plain SELECT … LIMIT 1 for the shim.
  The shim re-implements _pg_conn, _pg_init_jobs_table, _pg_enqueue_job,
  _pg_claim_next_job, _pg_finish_job, and _pg_get_job against SQLite so the
  full orchestrator → worker path can be exercised without a live Postgres.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest.mock as mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# SQLite shim — same schema, same logic, sqlite3 syntax
# ---------------------------------------------------------------------------

class _SqliteJobQueue:
    """In-memory SQLite database that implements the pipeline_jobs contract."""

    def __init__(self) -> None:
        # Use a file-based SQLite DB so multiple function calls share state
        # (mimicking two processes sharing a Postgres DB).
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._path = self._tmp.name
        self._init()

    def _conn(self):
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_jobs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id        TEXT NOT NULL UNIQUE,
                status        TEXT NOT NULL DEFAULT 'queued',
                queued_at     TEXT NOT NULL,
                started_at    TEXT,
                ended_at      TEXT,
                restaurant_id TEXT,
                location_id   TEXT,
                options_json  TEXT NOT NULL,
                result_json   TEXT,
                error         TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def enqueue(self, job_id: str, options: dict, queued_at: str) -> None:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO pipeline_jobs
            (job_id, status, queued_at, restaurant_id, location_id, options_json)
            VALUES (?, 'queued', ?, ?, ?, ?)
            """,
            (
                job_id,
                queued_at,
                str(options.get("restaurant_id") or ""),
                str(options.get("location_id") or ""),
                json.dumps(options),
            ),
        )
        conn.commit()
        conn.close()

    def claim_next(self, started_at: str):
        """Claim one queued job atomically. Returns (job_id, options) or None."""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT job_id, options_json FROM pipeline_jobs"
            " WHERE status = 'queued' ORDER BY id ASC LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            conn.close()
            return None
        job_id = row["job_id"]
        options = json.loads(row["options_json"])
        cur.execute(
            "UPDATE pipeline_jobs SET status = 'running', started_at = ?"
            " WHERE job_id = ?",
            (started_at, job_id),
        )
        conn.commit()
        conn.close()
        return job_id, options

    def finish(self, job_id: str, status: str, result=None, error=None,
               ended_at: str = "") -> None:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE pipeline_jobs SET status = ?, ended_at = ?,"
            " result_json = ?, error = ? WHERE job_id = ?",
            (
                status,
                ended_at,
                json.dumps(result) if result is not None else None,
                error,
                job_id,
            ),
        )
        conn.commit()
        conn.close()

    def get(self, job_id: str):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT job_id, status, queued_at, started_at, ended_at,"
            " restaurant_id, location_id, result_json, error"
            " FROM pipeline_jobs WHERE job_id = ?",
            (job_id,),
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        result = None
        if row["result_json"]:
            try:
                result = json.loads(row["result_json"])
            except Exception:
                pass
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "queued_at": row["queued_at"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "restaurant_id": row["restaurant_id"],
            "location_id": row["location_id"],
            "result": result,
            "error": row["error"],
        }

    def cleanup(self):
        try:
            os.unlink(self._path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test 1 — Cross-process contract: enqueue in one context, claim in another
# ---------------------------------------------------------------------------

def test_job_enqueued_by_one_process_claimed_by_worker():
    """A job written by _pg_enqueue_job (API-process context) is claimed and
    executed by worker.run_once (worker-process context) using the same
    shared job queue (Postgres; here replaced by a shared SQLite shim)."""
    from restaurant_ai_platform import orchestrator
    import worker as worker_mod

    db = _SqliteJobQueue()
    original_db_url = orchestrator._DATABASE_URL

    try:
        # Patch orchestrator._DATABASE_URL so both paths take the Postgres branch.
        orchestrator._DATABASE_URL = "sqlite-shim://test"

        # Patch the Postgres helpers to use the SQLite shim.
        def shim_enqueue(job_id, options):
            db.enqueue(job_id, options, orchestrator._utc_ts())

        def shim_claim():
            return db.claim_next(orchestrator._utc_ts())

        def shim_finish(job_id, status, result=None, error=None):
            db.finish(job_id, status, result=result, error=error,
                      ended_at=orchestrator._utc_ts())

        def shim_get(job_id):
            return db.get(job_id)

        with mock.patch.object(orchestrator, "_pg_enqueue_job", side_effect=shim_enqueue), \
             mock.patch.object(orchestrator, "_pg_claim_next_job", side_effect=shim_claim), \
             mock.patch.object(orchestrator, "_pg_finish_job", side_effect=shim_finish), \
             mock.patch.object(orchestrator, "_pg_get_job", side_effect=shim_get):

            # --- API-process context: enqueue a job ---
            options = {
                "restaurant_id": "rest_g2",
                "location_id": "loc_g2",
                "dry_run": True,  # dry_run so no real pipeline steps execute
            }
            result = orchestrator.enqueue_job(options)

            assert isinstance(result, str), (
                f"enqueue_job must return a job_id string when DATABASE_URL set, got: {result}"
            )
            job_id = result

            # Verify the job is in the queue as 'queued'.
            job_record = db.get(job_id)
            assert job_record is not None, f"Job {job_id} must be in the DB after enqueue"
            assert job_record["status"] == "queued", (
                f"Job must be 'queued' before worker runs, got: {job_record['status']}"
            )

            # --- Worker-process context: claim and execute ---
            count = worker_mod.run_once()

            assert count == 1, (
                f"worker.run_once must have processed 1 job, got: {count}"
            )

            # Verify the job is now done.
            job_after = db.get(job_id)
            assert job_after is not None, "Job record must still exist after worker runs"
            assert job_after["status"] == "done", (
                f"Job must be 'done' after worker runs, got: {job_after['status']}"
            )

    finally:
        orchestrator._DATABASE_URL = original_db_url
        db.cleanup()


# ---------------------------------------------------------------------------
# Test 2 — enqueue_job without DATABASE_URL returns QUEUE_NO_DATABASE
# ---------------------------------------------------------------------------

def test_enqueue_without_database_url_returns_error():
    from restaurant_ai_platform import orchestrator

    original = orchestrator._DATABASE_URL
    try:
        orchestrator._DATABASE_URL = None
        result = orchestrator.enqueue_job({"restaurant_id": "r", "location_id": "l"})
        assert isinstance(result, dict), (
            f"Expected error dict when DATABASE_URL absent, got: {result!r}"
        )
        assert result.get("error_code") == orchestrator.QUEUE_ERROR_NO_DATABASE, (
            f"Expected QUEUE_NO_DATABASE, got: {result.get('error_code')!r}"
        )
    finally:
        orchestrator._DATABASE_URL = original


# ---------------------------------------------------------------------------
# Test 3 — worker.run_once without DATABASE_URL returns 0
# ---------------------------------------------------------------------------

def test_worker_run_once_without_database_url_returns_zero():
    from restaurant_ai_platform import orchestrator
    import worker as worker_mod

    original = orchestrator._DATABASE_URL
    try:
        orchestrator._DATABASE_URL = None
        count = worker_mod.run_once()
        assert count == 0, (
            f"worker.run_once must return 0 when DATABASE_URL absent, got: {count}"
        )
    finally:
        orchestrator._DATABASE_URL = original


if __name__ == "__main__":
    test_job_enqueued_by_one_process_claimed_by_worker()
    print("PASS test_job_enqueued_by_one_process_claimed_by_worker")

    test_enqueue_without_database_url_returns_error()
    print("PASS test_enqueue_without_database_url_returns_error")

    test_worker_run_once_without_database_url_returns_zero()
    print("PASS test_worker_run_once_without_database_url_returns_zero")

    print("\nAll PHASE G2 tests passed.")
