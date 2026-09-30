"""Smoke tests — fail-closed behaviour for PHASE 13.

Covers:
1. Missing sales CSV → data_ingestion returns DATA_INGESTION_FILE_NOT_FOUND.
2. Missing tenant keys → persistence.get_last_run returns PERSISTENCE_MISSING_TENANT_KEYS.
3. Forecast with fewer than 7 daily totals → ml_prediction returns FORECAST_INSUFFICIENT_DATA.

No product modules are edited here.
"""
import os
import sys
import tempfile

# Ensure the backend package is importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Test 1 — Missing sales CSV returns DATA_INGESTION_FILE_NOT_FOUND
# ---------------------------------------------------------------------------

def test_ingestion_missing_file_returns_error():
    from restaurant_ai_platform import data_ingestion

    # Use a tenant combination that will never have a CSV on disk.
    context = {
        "restaurant_id": "nonexistent_tenant",
        "location_id": "nonexistent_location",
        "demo": False,
    }
    result = data_ingestion.run(context)

    assert result.get("status") == "error", (
        f"Expected status=error, got: {result}"
    )
    assert result.get("error_code") == data_ingestion.DATA_INGESTION_ERROR_FILE_NOT_FOUND, (
        f"Expected DATA_INGESTION_FILE_NOT_FOUND, got: {result.get('error_code')}"
    )
    assert "sales" not in result or result.get("sales") is None, (
        "No sales data must be returned on file-not-found error"
    )


# ---------------------------------------------------------------------------
# Test 2 — Missing tenant keys → PERSISTENCE_MISSING_TENANT_KEYS
# ---------------------------------------------------------------------------

def test_last_run_missing_tenant_keys_returns_error():
    from restaurant_ai_platform.core import persistence

    # Use a fresh isolated DB so this test is side-effect-free.
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    original = os.environ.get("PIPELINE_DB_PATH")
    try:
        os.environ["PIPELINE_DB_PATH"] = db_path
        persistence.init_db()

        result = persistence.get_last_run(restaurant_id=None, location_id=None)

        assert result is not None, "Expected an error dict, got None"
        assert result.get("error_code") == persistence.PERSISTENCE_ERROR_MISSING_TENANT, (
            f"Expected PERSISTENCE_MISSING_TENANT_KEYS, got: {result.get('error_code')}"
        )
        # Must not contain any run payload data.
        assert "payload_json" not in result and "run_id" not in result, (
            "Error response must not include run data"
        )
    finally:
        if original is None:
            os.environ.pop("PIPELINE_DB_PATH", None)
        else:
            os.environ["PIPELINE_DB_PATH"] = original
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 3 — Forecast with fewer than 7 daily totals → FORECAST_INSUFFICIENT_DATA
# ---------------------------------------------------------------------------

def test_forecast_under_7_rows_returns_error():
    from restaurant_ai_platform import ml_prediction

    # 6 rows — one short of the required minimum.
    sales = [{"daily_sales_total": float(100 + i * 10)} for i in range(6)]
    result = ml_prediction.run(context={"sales": sales})

    assert result.get("status") == "error", (
        f"Expected status=error, got: {result}"
    )
    assert result.get("error_code") == ml_prediction.FORECAST_ERROR_INSUFFICIENT_DATA, (
        f"Expected FORECAST_INSUFFICIENT_DATA, got: {result.get('error_code')}"
    )
    assert "forecast" not in result, (
        "No forecast list must be returned when data is insufficient"
    )


# ---------------------------------------------------------------------------
# Test 4 — C12: structured log lines include run_id, step, duration_ms
# ---------------------------------------------------------------------------

def test_structured_logs_include_run_id_step_duration_ms():
    import io
    import contextlib

    from restaurant_ai_platform import orchestrator

    # Run a dry-run pipeline so steps are planned but no heavy work is done.
    # Capture stdout to inspect the structured log lines.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        orchestrator.run_pipeline(options={"dry_run": True})

    output = buf.getvalue()
    lines = [l.strip() for l in output.splitlines() if l.strip()]

    assert lines, "Expected at least one log line from run_pipeline"

    # Every log line must include run_id and duration_ms.
    # Step-level events must additionally include step=.
    for line in lines:
        assert "run_id=" in line, (
            f"Log line missing run_id: {line}"
        )
        assert "duration_ms=" in line, (
            f"Log line missing duration_ms: {line}"
        )

    # At least one line should have step= (pipeline_start/dry_run may not,
    # but step-level events do). For a dry_run, the pipeline_dry_run event
    # is pipeline-level. Run a real minimal pipeline to get step lines.
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        orchestrator.run_pipeline(
            options={
                "steps": "5_ml_prediction",
                "restaurant_id": "test_rid",
                "location_id": "test_lid",
            },
        )

    output2 = buf2.getvalue()
    step_lines = [l.strip() for l in output2.splitlines() if "step=" in l]

    assert step_lines, (
        "Expected at least one step-level log line with step="
    )

    for line in step_lines:
        assert "run_id=" in line, f"Step log line missing run_id: {line}"
        assert "step=" in line, f"Step log line missing step: {line}"
        assert "duration_ms=" in line, f"Step log line missing duration_ms: {line}"

    print("PASS test_structured_logs_include_run_id_step_duration_ms")


if __name__ == "__main__":
    test_ingestion_missing_file_returns_error()
    print("PASS test_ingestion_missing_file_returns_error")

    test_last_run_missing_tenant_keys_returns_error()
    print("PASS test_last_run_missing_tenant_keys_returns_error")

    test_forecast_under_7_rows_returns_error()
    print("PASS test_forecast_under_7_rows_returns_error")

    test_structured_logs_include_run_id_step_duration_ms()

    print("\nAll smoke tests passed.")
