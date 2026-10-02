"""PHASE R1 tests — Live Postgres / tenant isolation.

Tests:
1. Tenant A read does not return tenant B.
2. Missing keys do not return a global last run.

Both tests run against SQLite (no DATABASE_URL needed) because the tenant
isolation logic is identical regardless of backend.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Test 1 — Tenant A read does not return tenant B data
# ---------------------------------------------------------------------------

def test_tenant_isolation_a_does_not_return_b():
    from restaurant_ai_platform.core import persistence

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    original = os.environ.get("PIPELINE_DB_PATH")
    try:
        os.environ["PIPELINE_DB_PATH"] = db_path
        persistence.init_db()

        # Save a run for tenant B.
        persistence.save_run({
            "restaurant_id": "rest_b",
            "location_id": "loc_b",
            "run_id": "run-b-001",
            "status": "ok",
        })

        # Read for tenant A — must return None, not tenant B's data.
        result = persistence.get_last_run(
            restaurant_id="rest_a",
            location_id="loc_a",
        )

        assert result is None, (
            f"Tenant A read must not return tenant B data; got: {result}"
        )

    finally:
        if original is None:
            os.environ.pop("PIPELINE_DB_PATH", None)
        else:
            os.environ["PIPELINE_DB_PATH"] = original
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 2 — Missing keys do not return a global last run
# ---------------------------------------------------------------------------

def test_missing_tenant_keys_do_not_return_global_run():
    from restaurant_ai_platform.core import persistence

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    original = os.environ.get("PIPELINE_DB_PATH")
    try:
        os.environ["PIPELINE_DB_PATH"] = db_path
        persistence.init_db()

        # Save a run so there IS data in the DB.
        persistence.save_run({
            "restaurant_id": "rest_x",
            "location_id": "loc_x",
            "run_id": "run-x-001",
            "status": "ok",
        })

        # Call without tenant keys — must return the error dict, not a run.
        result = persistence.get_last_run(restaurant_id=None, location_id=None)

        assert result is not None, "Expected an error dict, got None"
        assert result.get("error_code") == persistence.PERSISTENCE_ERROR_MISSING_TENANT, (
            f"Expected PERSISTENCE_MISSING_TENANT_KEYS, got: {result.get('error_code')}"
        )
        assert "run_id" not in result, (
            "Error response must not include run data"
        )

    finally:
        if original is None:
            os.environ.pop("PIPELINE_DB_PATH", None)
        else:
            os.environ["PIPELINE_DB_PATH"] = original
        os.unlink(db_path)


if __name__ == "__main__":
    test_tenant_isolation_a_does_not_return_b()
    print("PASS test_tenant_isolation_a_does_not_return_b")

    test_missing_tenant_keys_do_not_return_global_run()
    print("PASS test_missing_tenant_keys_do_not_return_global_run")

    print("\nAll PHASE R1 tests passed.")
