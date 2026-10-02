"""PHASE R6 tests — Measured waste events.

Tests:
1. No waste_events does not invent kg (waste_total is null).
2. waste_events sum to a numeric total (method=measured).
3. method=measured only when events are present.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from restaurant_ai_platform import dashboard_update  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_RECIPE = [
    {
        "name": "Pasta",
        "expected_meals": 10,
        "ingredients": [
            {"name": "flour", "unit": "kg", "quantity_per_meal": 0.2},
        ],
    }
]

_VALID_SALES = [
    {"net_sales": 1500.0, "restaurant_id": "rest_r6", "location_id": "loc_r6",
     "date": f"2024-01-0{i+1}"}
    for i in range(7)
]

_VALID_ON_HAND = {"flour": 0.0}

_VALID_PREDICTION = {
    "status": "ok",
    "forecast": [{"predicted_sales": 1500.0, "date": "2024-01-08"}],
    "labor_recommendation_allowed": True,
}


def _make_context(**overrides) -> dict:
    ctx: dict = {
        "restaurant_id": "rest_r6",
        "location_id": "loc_r6",
        "sales": _VALID_SALES,
        "recipes": _VALID_RECIPE,
        "on_hand": _VALID_ON_HAND,
        "5_ml_prediction": _VALID_PREDICTION,
        # waste_events intentionally absent unless overridden
    }
    ctx.update(overrides)
    return ctx


# ---------------------------------------------------------------------------
# Test 1 — No waste_events does not invent kg
# ---------------------------------------------------------------------------

def test_no_events_does_not_invent_kg():
    """When waste_events is absent, waste_total must be null — not a ratio-derived number."""
    result = dashboard_update.run(_make_context())

    assert result.get("dashboard_update_status") == "ok", (
        f"Expected dashboard_update_status=ok, got: {result}"
    )

    assert result.get("waste_total") is None, (
        f"waste_total must be null when no events provided, got: {result.get('waste_total')}"
    )
    assert result.get("waste_method") is None, (
        f"waste_method must be null when no events provided, got: {result.get('waste_method')!r}"
    )
    assert result.get("waste_error_code") == dashboard_update.WASTE_ERROR_NO_EVENTS, (
        f"Expected WASTE_NO_EVENTS_PROVIDED, got: {result.get('waste_error_code')}"
    )


def test_empty_events_list_does_not_invent_kg():
    """An empty waste_events list must also produce null, not a ratio number."""
    result = dashboard_update.run(_make_context(waste_events=[]))

    assert result.get("waste_total") is None, (
        f"waste_total must be null for empty events list, got: {result.get('waste_total')}"
    )
    assert result.get("waste_method") is None, (
        f"waste_method must be null for empty events list"
    )


# ---------------------------------------------------------------------------
# Test 2 — waste_events sum to a numeric total
# ---------------------------------------------------------------------------

def test_events_sum_to_numeric_total():
    events = [
        {"item": "chicken", "qty": 2.5, "reason": "spoiled", "ts": "2024-01-01T10:00:00"},
        {"item": "rice",    "qty": 1.0, "reason": "overcooked", "ts": "2024-01-01T12:00:00"},
        {"item": "oil",     "qty": 0.3, "reason": "expired", "ts": "2024-01-01T14:00:00"},
    ]
    result = dashboard_update.run(_make_context(waste_events=events))

    assert result.get("dashboard_update_status") == "ok", (
        f"Expected dashboard_update_status=ok, got: {result}"
    )

    waste_total = result.get("waste_total")
    assert isinstance(waste_total, float), (
        f"waste_total must be a float, got: {type(waste_total)}"
    )
    assert waste_total == 3.8, (
        f"Expected waste_total=3.8 (2.5+1.0+0.3), got: {waste_total}"
    )
    assert result.get("waste_method") == "measured", (
        f"waste_method must be 'measured' when events exist, got: {result.get('waste_method')!r}"
    )
    assert result.get("waste_error_code") is None, (
        f"waste_error_code must be None when events are valid, got: {result.get('waste_error_code')}"
    )


# ---------------------------------------------------------------------------
# Test 3 — _process_waste_events unit tests
# ---------------------------------------------------------------------------

def test_process_waste_events_none_returns_null():
    r = dashboard_update._process_waste_events(None)
    assert r["waste_total"] is None
    assert r["method"] is None
    assert r["error_code"] == dashboard_update.WASTE_ERROR_NO_EVENTS


def test_process_waste_events_valid_returns_measured():
    events = [
        {"item": "bread", "qty": 1.5, "reason": "stale", "ts": "2024-01-01T08:00:00"},
        {"item": "milk",  "qty": 0.5, "reason": "expired", "ts": "2024-01-01T09:00:00"},
    ]
    r = dashboard_update._process_waste_events(events)
    assert r["method"] == "measured"
    assert r["waste_total"] == 2.0
    assert r["error_code"] is None
    assert len(r["waste_events"]) == 2


if __name__ == "__main__":
    test_no_events_does_not_invent_kg()
    print("PASS test_no_events_does_not_invent_kg")

    test_empty_events_list_does_not_invent_kg()
    print("PASS test_empty_events_list_does_not_invent_kg")

    test_events_sum_to_numeric_total()
    print("PASS test_events_sum_to_numeric_total")

    test_process_waste_events_none_returns_null()
    print("PASS test_process_waste_events_none_returns_null")

    test_process_waste_events_valid_returns_measured()
    print("PASS test_process_waste_events_valid_returns_measured")

    print("\nAll PHASE R6 tests passed.")
