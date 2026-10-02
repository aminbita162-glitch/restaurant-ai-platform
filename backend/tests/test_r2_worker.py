"""PHASE R2 tests — separate pipeline worker.

Tests:
1. POST /pipeline/run does not call run_pipeline inline (job is queued, not executed).
2. GET /pipeline/run has no side effect (no job created, no pipeline called).
"""
from __future__ import annotations

import os
import sys
import unittest.mock as mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app():
    """Create a minimal Flask test app with the api_serving blueprint."""
    from flask import Flask  # type: ignore
    from restaurant_ai_platform import api_serving

    app = Flask(__name__)
    app.config["TESTING"] = True
    # Provide a dummy API key so auth passes.
    os.environ["RESTAURANT_AI_API_KEY"] = "test-key-r2"
    api_serving.register(app)
    return app


# ---------------------------------------------------------------------------
# Test 1 — POST does not call run_pipeline inline
# ---------------------------------------------------------------------------

def test_post_pipeline_run_does_not_call_run_pipeline_inline():
    """POST /pipeline/run must return 202 before run_pipeline executes.

    Strategy: replace run_pipeline with a version that blocks on an Event
    until the request has already returned. If the request handler were
    calling run_pipeline inline (synchronously), the response would never
    arrive while the gate is closed. The 202 arriving before the gate is
    opened proves the pipeline did not run inside the request.
    """
    import threading

    app = _make_app()

    gate = threading.Event()           # blocks the background thread
    pipeline_started = threading.Event()  # signals the thread began

    def blocking_run_pipeline(options=None, **kwargs):
        pipeline_started.set()
        gate.wait(timeout=5)  # block until test releases it
        return {"status": "ok", "run_id": "test-run"}

    with mock.patch(
        "restaurant_ai_platform.orchestrator.run_pipeline",
        side_effect=blocking_run_pipeline,
    ):
        with app.test_client() as client:
            # The gate is closed — if POST called run_pipeline inline this
            # would hang until timeout; instead it must return 202 immediately.
            resp = client.post(
                "/api/v1/pipeline/run",
                json={"restaurant_id": "rest_r2", "location_id": "loc_r2"},
                headers={"X-Api-Key": "test-key-r2"},
            )

        # Response arrived while gate was still closed: pipeline did not run inline.
        assert resp.status_code == 202, (
            f"Expected 202, got {resp.status_code}: {resp.get_data(as_text=True)}"
        )

        data = resp.get_json()
        assert data.get("job_id"), f"Expected job_id in response, got: {data}"
        assert data.get("status") == "queued", f"Expected status=queued, got: {data}"

    # Release the background thread so it can finish cleanly.
    gate.set()


# ---------------------------------------------------------------------------
# Test 2 — GET /pipeline/run has no side effect
# ---------------------------------------------------------------------------

def test_get_pipeline_run_has_no_side_effect():
    """GET /pipeline/run must not create a job or call run_pipeline."""
    app = _make_app()

    from restaurant_ai_platform import orchestrator

    # Record the job store size before the request.
    with orchestrator._JOB_STORE_LOCK:
        job_count_before = len(orchestrator._JOB_STORE)

    with mock.patch(
        "restaurant_ai_platform.orchestrator.run_pipeline"
    ) as mock_run:
        with mock.patch(
            "restaurant_ai_platform.orchestrator.enqueue_job"
        ) as mock_enqueue:
            with app.test_client() as client:
                resp = client.get(
                    "/api/v1/pipeline/run",
                    headers={"X-Api-Key": "test-key-r2"},
                )

    # Must return 200 (info page) — not 202, not 4xx.
    assert resp.status_code == 200, (
        f"Expected 200 info response, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )

    # No pipeline run or job enqueue must have occurred.
    mock_run.assert_not_called(), "run_pipeline must not be called by GET"
    mock_enqueue.assert_not_called(), "enqueue_job must not be called by GET"

    # Job store must not have grown.
    with orchestrator._JOB_STORE_LOCK:
        job_count_after = len(orchestrator._JOB_STORE)

    assert job_count_after == job_count_before, (
        f"GET /pipeline/run must not create jobs; store grew from {job_count_before} to {job_count_after}"
    )


if __name__ == "__main__":
    test_post_pipeline_run_does_not_call_run_pipeline_inline()
    print("PASS test_post_pipeline_run_does_not_call_run_pipeline_inline")

    test_get_pipeline_run_has_no_side_effect()
    print("PASS test_get_pipeline_run_has_no_side_effect")

    print("\nAll PHASE R2 tests passed.")
