"""PHASE R8 tests — Manager approval page.

Tests:
1. GET /manager page exists and does not call GET /pipeline/run.
2. POST /manager/approve without token (no X-Api-Key) returns 401.
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
    from flask import Flask  # type: ignore
    from restaurant_ai_platform import api_serving

    app = Flask(__name__)
    app.config["TESTING"] = True
    os.environ["RESTAURANT_AI_API_KEY"] = "test-key-r8"
    api_serving.register(app)
    return app


# ---------------------------------------------------------------------------
# Test 1 — Page exists and does NOT call GET /pipeline/run
# ---------------------------------------------------------------------------

def test_manager_page_exists_and_does_not_call_get_pipeline_run():
    """GET /manager must return 200 with HTML content.
    The page source must not contain a fetch/XHR call to GET /pipeline/run."""
    app = _make_app()

    with app.test_client() as client:
        resp = client.get("/manager")

    assert resp.status_code == 200, (
        f"Expected 200 for GET /manager, got {resp.status_code}"
    )
    content_type = resp.content_type or ""
    assert "text/html" in content_type, (
        f"Expected text/html content type, got: {content_type}"
    )

    html = resp.get_data(as_text=True)
    assert html.strip(), "manager.html must not be empty"

    # The page must not contain a call to GET /pipeline/run.
    # Check for the forbidden pattern in JS: fetching /pipeline/run without POST method.
    # A safe call would be POST /pipeline/run — GET side effects are forbidden.
    import re
    # Look for fetch("/pipeline/run") or fetch("/api/v1/pipeline/run") without method:POST nearby.
    # Simple check: the string "/pipeline/run" must not appear as a GET call.
    # We check that any occurrence of the path is not used in a GET context.
    forbidden_patterns = [
        r'fetch\s*\(\s*["\'](?:[^"\']*/)pipeline/run["\']',
    ]
    for pat in forbidden_patterns:
        matches = re.findall(pat, html)
        for m in matches:
            # Allow if followed by method: "POST" context — but the simplest safe
            # check is that the string "/pipeline/run" does not appear at all in JS
            # without an explicit POST method qualifier.
            # The page must never use GET /pipeline/run.
            assert False, (
                f"Page must not contain a GET call to /pipeline/run; found: {m!r}"
            )

    # Confirm the page contains the approve mechanism (POST to /manager/approve).
    assert "/manager/approve" in html, (
        "manager.html must contain a reference to POST /manager/approve"
    )


# ---------------------------------------------------------------------------
# Test 2 — Approve without token returns 401
# ---------------------------------------------------------------------------

def test_approve_without_api_key_returns_401():
    """POST /manager/approve without X-Api-Key must return 401."""
    app = _make_app()

    with app.test_client() as client:
        # No X-Api-Key header supplied.
        resp = client.post(
            "/manager/approve",
            json={"run_id": "run-test-r8"},
            # deliberately no X-Api-Key header
        )

    assert resp.status_code == 401, (
        f"Expected 401 without X-Api-Key, got {resp.status_code}: "
        f"{resp.get_data(as_text=True)}"
    )
    data = resp.get_json()
    assert data and not data.get("ok"), "Response must not be ok=True without auth"


def test_approve_with_valid_key_returns_ok():
    """POST /manager/approve with valid X-Api-Key and run_id must return 200 approval=approved."""
    app = _make_app()

    with app.test_client() as client:
        resp = client.post(
            "/manager/approve",
            json={"run_id": "run-r8-test-001"},
            headers={"X-Api-Key": "test-key-r8"},
        )

    assert resp.status_code == 200, (
        f"Expected 200 for valid approve, got {resp.status_code}: "
        f"{resp.get_data(as_text=True)}"
    )
    data = resp.get_json()
    assert data.get("ok") is True, f"Expected ok=True, got: {data}"
    assert data.get("approval") == "approved", (
        f"Expected approval=approved, got: {data.get('approval')!r}"
    )
    assert data.get("run_id") == "run-r8-test-001", (
        f"Expected run_id echoed back, got: {data.get('run_id')!r}"
    )


if __name__ == "__main__":
    test_manager_page_exists_and_does_not_call_get_pipeline_run()
    print("PASS test_manager_page_exists_and_does_not_call_get_pipeline_run")

    test_approve_without_api_key_returns_401()
    print("PASS test_approve_without_api_key_returns_401")

    test_approve_with_valid_key_returns_ok()
    print("PASS test_approve_with_valid_key_returns_ok")

    print("\nAll PHASE R8 tests passed.")
