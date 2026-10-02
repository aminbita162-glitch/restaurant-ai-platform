"""Worker entrypoint — executes queued pipeline jobs.

Run this process separately from the API server:

    python worker.py

The worker drains the in-process job queue produced by
orchestrator.enqueue_job(). Each queued job calls run_pipeline()
in this process, not inside the API request handler.

Environment variables used (same as the API server):
  DATABASE_URL           Optional Postgres URL for run persistence.
  RESTAURANT_AI_API_KEY  Not needed by the worker directly.
  OPENAI_API_KEY         Required only if gpt_insight step is enabled.

Note: the current job store is in-process memory shared via import.
For multi-process deployments a shared queue (e.g. Redis) is required.
"""
from __future__ import annotations

import os
import sys
import time

# Ensure the backend package is importable when run from the repo root.
sys.path.insert(0, os.path.dirname(__file__))

from restaurant_ai_platform import orchestrator  # noqa: E402


_POLL_INTERVAL_S = float(os.environ.get("WORKER_POLL_INTERVAL_S", "1"))


def run_once() -> int:
    """Drain all jobs currently in the QUEUED state. Return count processed."""
    processed = 0
    with orchestrator._JOB_STORE_LOCK:
        job_ids = list(orchestrator._JOB_STORE.keys())

    for job_id in job_ids:
        record = orchestrator.get_job(job_id)
        if record is None:
            continue
        if record.get("status") != "queued":
            continue

        # Mark running before executing so a second worker won't double-run.
        started_at = orchestrator._utc_ts()
        orchestrator._job_store_set(job_id, {
            **record,
            "status": "running",
            "started_at": started_at,
        })

        options = {k: v for k, v in record.items()
                   if k not in {"job_id", "status", "queued_at", "started_at", "ended_at"}}

        try:
            result = orchestrator.run_pipeline(options)
            ended_at = orchestrator._utc_ts()
            orchestrator._job_store_set(job_id, {
                **orchestrator._JOB_STORE.get(job_id, {}),
                "status": "done",
                "started_at": started_at,
                "ended_at": ended_at,
                "result": result,
            })
        except Exception as exc:
            ended_at = orchestrator._utc_ts()
            orchestrator._job_store_set(job_id, {
                **orchestrator._JOB_STORE.get(job_id, {}),
                "status": "error",
                "started_at": started_at,
                "ended_at": ended_at,
                "error": f"{type(exc).__name__}: {exc}",
            })

        processed += 1

    return processed


def main() -> None:
    print(f"[worker] starting, poll_interval={_POLL_INTERVAL_S}s", flush=True)
    while True:
        count = run_once()
        if count:
            print(f"[worker] processed {count} job(s)", flush=True)
        time.sleep(_POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
