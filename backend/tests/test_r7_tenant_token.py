"""PHASE R7 tests — Tenant from token.

Tests:
1. Unknown X-Tenant-Token does not run the pipeline (returns 401).
2. Body tenant ids cannot override the token-resolved tenant.
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
    os.environ["RESTAURANT_AI_API_KEY"] = "test-key-r7"
    api_serving.register(app)
    return app


# ---------------------------------------------------------------------------
# Test 1 — Unknown token does not run the pipeline (returns 401)
# ---------------------------------------------------------------------------

def test_unknown_token_returns_401_and_does_not_enqueue():
    """An X-Tenant-Token that is not in TENANT_TOKENS must return 401
    and must not enqueue or execute the pipeline."""
    app = _make_app()

    # Ensure TENANT_TOKENS does not contain the token we send.
    original = os.environ.get("TENANT_TOKENS")
    os.environ["TENANT_TOKENS"] = "valid_tok:rest_a:loc_a"
    try:
        with mock.patch(
            "restaurant_ai_platform.orchestrator.enqueue_job"
        ) as mock_enqueue:
            with app.test_client() as client:
                resp = client.post(
                    "/api/v1/pipeline/run",
                    json={"restaurant_id": "rest_x", "location_id": "loc_x"},
                    headers={
                        "X-Api-Key": "test-key-r7",
                        "X-Tenant-Token": "unknown_token_xyz",
                    },
                )

        assert resp.status_code == 401, (
            f"Expected 401 for unknown token, got {resp.status_code}: "
            f"{resp.get_data(as_text=True)}"
        )
        mock_enqueue.assert_not_called(), (
            "enqueue_job must not be called when token is unknown"
        )
    finally:
        if original is None:
            os.environ.pop("TENANT_TOKENS", None)
        else:
            os.environ["TENANT_TOKENS"] = original


# ---------------------------------------------------------------------------
# Test 2 — Body tenant cannot override the token tenant
# ---------------------------------------------------------------------------

def test_body_tenant_cannot_override_token_tenant():
    """When X-Tenant-Token resolves a tenant, body restaurant_id/location_id
    must be ignored — the enqueued job must use the token-resolved tenant."""
    app = _make_app()

    original = os.environ.get("TENANT_TOKENS")
    # Token maps to rest_token / loc_token.
    os.environ["TENANT_TOKENS"] = "my_token:rest_token:loc_token"

    captured_options: dict = {}

    def capture_enqueue(options):
        captured_options.update(options)
        return "job-test-r7"

    try:
        with mock.patch(
            "restaurant_ai_platform.orchestrator.enqueue_job",
            side_effect=capture_enqueue,
        ):
            with app.test_client() as client:
                resp = client.post(
                    "/api/v1/pipeline/run",
                    # Body provides different tenant ids — must be ignored.
                    json={
                        "restaurant_id": "rest_body_override",
                        "location_id": "loc_body_override",
                    },
                    headers={
                        "X-Api-Key": "test-key-r7",
                        "X-Tenant-Token": "my_token",
                    },
                )

        assert resp.status_code == 202, (
            f"Expected 202 for valid token, got {resp.status_code}: "
            f"{resp.get_data(as_text=True)}"
        )

        # The enqueued options must carry the token-resolved tenant, not the body tenant.
        assert captured_options.get("restaurant_id") == "rest_token", (
            f"restaurant_id must come from token, got: {captured_options.get('restaurant_id')!r}"
        )
        assert captured_options.get("location_id") == "loc_token", (
            f"location_id must come from token, got: {captured_options.get('location_id')!r}"
        )

    finally:
        if original is None:
            os.environ.pop("TENANT_TOKENS", None)
        else:
            os.environ["TENANT_TOKENS"] = original


# ---------------------------------------------------------------------------
# Test 3 — _parse_tenant_tokens and _resolve_tenant_token unit tests
# ---------------------------------------------------------------------------

def test_parse_tenant_tokens_valid():
    from restaurant_ai_platform import api_serving

    original = os.environ.get("TENANT_TOKENS")
    os.environ["TENANT_TOKENS"] = "tokA:restA:locA,tokB:restB:locB"
    try:
        mapping = api_serving._parse_tenant_tokens()
        assert mapping.get("tokA") == ("restA", "locA"), f"Unexpected: {mapping}"
        assert mapping.get("tokB") == ("restB", "locB"), f"Unexpected: {mapping}"
    finally:
        if original is None:
            os.environ.pop("TENANT_TOKENS", None)
        else:
            os.environ["TENANT_TOKENS"] = original


def test_resolve_unknown_token_returns_none():
    from restaurant_ai_platform import api_serving

    original = os.environ.get("TENANT_TOKENS")
    os.environ["TENANT_TOKENS"] = "tokA:restA:locA"
    try:
        result = api_serving._resolve_tenant_token("not_a_real_token")
        assert result is None, f"Expected None for unknown token, got: {result}"
    finally:
        if original is None:
            os.environ.pop("TENANT_TOKENS", None)
        else:
            os.environ["TENANT_TOKENS"] = original


if __name__ == "__main__":
    test_unknown_token_returns_401_and_does_not_enqueue()
    print("PASS test_unknown_token_returns_401_and_does_not_enqueue")

    test_body_tenant_cannot_override_token_tenant()
    print("PASS test_body_tenant_cannot_override_token_tenant")

    test_parse_tenant_tokens_valid()
    print("PASS test_parse_tenant_tokens_valid")

    test_resolve_unknown_token_returns_none()
    print("PASS test_resolve_unknown_token_returns_none")

    print("\nAll PHASE R7 tests passed.")
