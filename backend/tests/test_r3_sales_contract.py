"""PHASE R3 tests — Sales data contract.

Tests:
1. Missing required field returns DATA_INGESTION_MISSING_REQUIRED_FIELD.
   Covers each of: date, net_sales, restaurant_id, location_id.
2. demo=false does not load sample sales (returns FILE_NOT_FOUND, not demo data).
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from restaurant_ai_platform import data_ingestion  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: str, rows: list[dict]) -> None:
    """Write a list of dicts to a CSV file."""
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_valid_row(**overrides) -> dict:
    """Return a row dict with all required fields present."""
    row = {
        "date": "2024-01-01",
        "net_sales": "1500.00",
        "restaurant_id": "rest_r3",
        "location_id": "loc_r3",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Test 1 — Missing required field returns DATA_INGESTION_MISSING_REQUIRED_FIELD
# ---------------------------------------------------------------------------

def _assert_missing_field_error(field_name: str, bad_row: dict) -> None:
    """Write a CSV with bad_row, call data_ingestion.run(), assert error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "rest_r3__loc_r3__sales.csv")
        _write_csv(csv_path, [bad_row])

        original_dir = data_ingestion.BASE_DATA_DIR
        data_ingestion.BASE_DATA_DIR = tmpdir
        try:
            result = data_ingestion.run({
                "restaurant_id": "rest_r3",
                "location_id": "loc_r3",
                "demo": False,
            })
        finally:
            data_ingestion.BASE_DATA_DIR = original_dir

    assert result.get("status") == "error", (
        f"field={field_name}: expected status=error, got: {result}"
    )
    assert result.get("error_code") == data_ingestion.DATA_INGESTION_ERROR_MISSING_REQUIRED_FIELD, (
        f"field={field_name}: expected DATA_INGESTION_MISSING_REQUIRED_FIELD, got: {result.get('error_code')}"
    )
    assert result.get("missing_field") == field_name, (
        f"expected missing_field={field_name!r}, got: {result.get('missing_field')!r}"
    )
    assert "sales" not in result or result.get("sales") is None, (
        f"field={field_name}: sales must not be returned on error"
    )


def test_missing_date_returns_error():
    row = _make_valid_row(date="")
    _assert_missing_field_error("date", row)


def test_missing_net_sales_returns_error():
    row = _make_valid_row(net_sales="")
    _assert_missing_field_error("net_sales", row)


def test_missing_restaurant_id_returns_error():
    row = _make_valid_row(restaurant_id="")
    _assert_missing_field_error("restaurant_id", row)


def test_missing_location_id_returns_error():
    row = _make_valid_row(location_id="")
    _assert_missing_field_error("location_id", row)


def test_invalid_net_sales_non_numeric_returns_error():
    row = _make_valid_row(net_sales="not-a-number")
    _assert_missing_field_error("net_sales", row)


# ---------------------------------------------------------------------------
# Test 2 — demo=false does not load sample sales
# ---------------------------------------------------------------------------

def test_demo_false_does_not_load_sample_sales():
    """When demo=False and no tenant file exists, must return FILE_NOT_FOUND.
    Must not return demo/sample data."""
    result = data_ingestion.run({
        "restaurant_id": "no_such_tenant_r3",
        "location_id": "no_such_loc_r3",
        "demo": False,
    })

    assert result.get("status") == "error", (
        f"Expected status=error (no file), got: {result}"
    )
    assert result.get("error_code") == data_ingestion.DATA_INGESTION_ERROR_FILE_NOT_FOUND, (
        f"Expected FILE_NOT_FOUND, got: {result.get('error_code')}"
    )
    # Must not contain demo or sample data.
    assert result.get("source_type") != "DEMO", "demo=false must not return DEMO source_type"
    assert result.get("sales_source") != "demo", "demo=false must not return sales_source=demo"


if __name__ == "__main__":
    test_missing_date_returns_error()
    print("PASS test_missing_date_returns_error")

    test_missing_net_sales_returns_error()
    print("PASS test_missing_net_sales_returns_error")

    test_missing_restaurant_id_returns_error()
    print("PASS test_missing_restaurant_id_returns_error")

    test_missing_location_id_returns_error()
    print("PASS test_missing_location_id_returns_error")

    test_invalid_net_sales_non_numeric_returns_error()
    print("PASS test_invalid_net_sales_non_numeric_returns_error")

    test_demo_false_does_not_load_sample_sales()
    print("PASS test_demo_false_does_not_load_sample_sales")

    print("\nAll PHASE R3 tests passed.")
