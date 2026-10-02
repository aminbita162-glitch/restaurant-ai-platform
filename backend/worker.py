"""Worker entrypoint — executes queued pipeline jobs from Postgres.

Run this process separately from the API server:

    python worker.py

When DATABASE_URL is set, the worker claims jobs from the Postgres
pipeline_jobs table (written by enqueue_job) and executes them.
This is the only supported mode for a live restaurant deployment.

When DATABASE_URL is unset, run_once() returns 0 and logs a warning.
The in-process memory queue is not a cross-process worker substitute.

Environment variables:
  DATABASE_URL           Required for live use. Postgres connection URL.
  WORKER_POLL_INTERVAL_S Poll interval in seconds (default: 1).
  OPENAI_API_KEY         Required only if gpt_insight step is enabled.
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
    """Claim and execute one batch of queued jobs. Return count processed.

    When DATABASE_URL is set, claims jobs atomically from Postgres.
    When DATABASE_URL is unset, logs a warning and returns 0.
    """
    if not orchestrator._DATABASE_URL:
        print(
            "[worker] DATABASE_URL is not set — no Postgres job queue available."
            " Set DATABASE_URL to enable cross-process job execution.",
            flush=True,
        )
        return 0

    processed = 0
    while True:
        claimed = orchestrator._pg_claim_next_job()
        if claimed is None:
            break

        job_id, options = claimed

        try:
            result = orchestrator.run_pipeline(options)
            orchestrator._pg_finish_job(job_id, "done", result=result)
        except Exception as exc:
            orchestrator._pg_finish_job(
                job_id, "error", error=f"{type(exc).__name__}: {exc}"
            )

        processed += 1

    return processed


def main() -> None:
    print(f"[worker] starting, poll_interval={_POLL_INTERVAL_S}s", flush=True)
    if not orchestrator._DATABASE_URL:
        print(
            "[worker] WARNING: DATABASE_URL is not set."
            " Worker will poll but cannot claim jobs.",
            flush=True,
        )
    while True:
        count = run_once()
        if count:
            print(f"[worker] processed {count} job(s)", flush=True)
        time.sleep(_POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
